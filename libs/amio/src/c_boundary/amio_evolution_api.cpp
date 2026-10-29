#define AMIO_BUILDING_LIBRARY 1

#include "amio/amio_evolution.h"

#include <cstddef>
#include <cstdint>
#include <bit>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <new>
#include <random>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "c_boundary/handle_table.hpp"
#include "catalog/posix_catalog.hpp"
#include "catalog/catalog_format.hpp"
#include "catalog/collection_reader.hpp"
#include "execution/context.hpp"
#include "c_boundary/amio_core.hpp"
#include "schema/schema.hpp"
#include "staging/resource_budget.hpp"
#include "steps/step_coordinator.hpp"
#include "drivers/netcdf/netcdf_driver.hpp"
#include "drivers/grib2/grib2_driver.hpp"
#include "staging/staging_pool.hpp"
#include "drivers/zarr/zarr_steps.hpp"
#include "drivers/grib2/grib2_steps.hpp"
#include "prefetch/step_read_queue.hpp"

#ifdef AMIO_HAS_NETCDF
#include <netcdf.h>
#endif

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif

namespace {
using amio::detail::HandleKind;
using amio::detail::HandleTable;
using amio::detail::process_handle_table;

#ifdef AMIO_HAS_TENSORSTORE
constexpr bool kTensorStoreAvailable = true;
#else
constexpr bool kTensorStoreAvailable = false;
#endif

struct ContextPayload {
    std::unique_ptr<amio::detail::ExecutionContext> context;
    std::string canonical_configuration;
    std::atomic<std::uint64_t> open_datasets{0};
    std::atomic<std::uint64_t> open_step_handles{0};
};

struct EvolutionDatasetPayload {
    static constexpr std::size_t kStepReadCacheBytes = 16u * 1024u * 1024u;
    ContextPayload *context = nullptr;
    std::string location;
    amio_evolution_format_t format = AMIO_FORMAT_NETCDF4;
    bool distributed_netcdf = false;
    bool distributed_zarr = false;
    std::uint64_t payload_budget_bytes = 0;
    amio::detail::PosixSyncLevel requested_sync = amio::detail::PosixSyncLevel::Visibility;
    std::unique_ptr<amio::detail::GRIB2_Driver> grib2_driver;
    std::string uuid;
    std::unique_ptr<amio::detail::ResourceBudget> budget;
    std::unique_ptr<amio::detail::StepCoordinator> steps;
    std::unique_ptr<amio::detail::PosixCatalog> catalog;
    std::map<std::string, std::shared_ptr<const amio::detail::Schema>, std::less<>> schemas;
    std::map<std::string, bool, std::less<>> global_present_fields;
    amio::detail::ZarrShardShapes zarr_shard_shapes;
    amio::detail::StepReadQueue read_cache{kStepReadCacheBytes};
    std::atomic<std::uint64_t> open_step_handles{0};
    std::atomic<std::uint64_t> open_step_leases{0};
    std::string previous_commit;
    std::string pending_commit;
    std::mutex mutex;
};

bool is_distributed_output(const EvolutionDatasetPayload &dataset) noexcept {
    return dataset.distributed_netcdf || dataset.distributed_zarr;
}

struct SchemaHandlePayload {
    std::shared_ptr<const amio::detail::Schema> schema;
};

struct ProducerStepPayload {
    EvolutionDatasetPayload *dataset = nullptr;
    std::uint64_t step_id = 0;
    bool partitioned = false;
    std::map<std::string, amio::detail::RegionSelection, std::less<>> partitions;
    bool ended = false;
};

struct OperationHandlePayload {
    std::shared_ptr<amio::detail::OperationRecord> operation;
};

struct ConsumerLeasePayload {
    EvolutionDatasetPayload *dataset = nullptr;
    std::uint64_t step_id = 0;
};

std::size_t dtype_size(amio_dtype_t dtype) noexcept {
    switch (dtype) {
        case AMIO_DTYPE_I8: case AMIO_DTYPE_U8: return 1;
        case AMIO_DTYPE_I16: case AMIO_DTYPE_U16: return 2;
        case AMIO_DTYPE_I32: case AMIO_DTYPE_U32: case AMIO_DTYPE_F32: return 4;
        case AMIO_DTYPE_I64: case AMIO_DTYPE_U64: case AMIO_DTYPE_F64: return 8;
        default: return 0;
    }
}

amio::detail::Grib2Settings grib2_settings_for(const ContextPayload &context,
                                                const amio::detail::Variable &variable) {
    auto config = conf::Config::from_string(context.canonical_configuration);
    auto settings = amio::detail::GRIB2_Driver::read_settings(config);
    for (const auto &[name, attribute] : variable.attributes) {
        constexpr std::string_view prefix = "grib2.";
        if (!name.starts_with(prefix)) continue;
        const auto key = std::string_view(name).substr(prefix.size());
        std::int64_t value = 0;
        if (const auto *signed_value = std::get_if<std::int64_t>(&attribute.value)) value = *signed_value;
        else if (const auto *unsigned_value = std::get_if<std::uint64_t>(&attribute.value)) {
            if (*unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::invalid_argument("GRIB2 variable attribute is outside the signed metadata range");
            value = static_cast<std::int64_t>(*unsigned_value);
        } else throw std::invalid_argument("GRIB2 variable attributes must be signed or unsigned integers");
#define AMIO_GRIB2_SETTING(n, m) if (key == n) settings.m = value
        AMIO_GRIB2_SETTING("discipline", discipline);
        else AMIO_GRIB2_SETTING("center", center);
        else AMIO_GRIB2_SETTING("subcenter", subcenter);
        else AMIO_GRIB2_SETTING("master_table_version", master_table_version);
        else AMIO_GRIB2_SETTING("local_table_version", local_table_version);
        else AMIO_GRIB2_SETTING("significance_of_ref_time", significance_of_ref_time);
        else AMIO_GRIB2_SETTING("production_status", production_status);
        else AMIO_GRIB2_SETTING("type_of_data", type_of_data);
        else AMIO_GRIB2_SETTING("gdt_number", gdt_number);
        else AMIO_GRIB2_SETTING("lat_first", lat_first);
        else AMIO_GRIB2_SETTING("lon_first", lon_first);
        else AMIO_GRIB2_SETTING("lat_last", lat_last);
        else AMIO_GRIB2_SETTING("lon_last", lon_last);
        else AMIO_GRIB2_SETTING("pdt_number", pdt_number);
        else AMIO_GRIB2_SETTING("parameter_category", parameter_category);
        else AMIO_GRIB2_SETTING("parameter_number", parameter_number);
        else AMIO_GRIB2_SETTING("type_of_first_fixed_surface", type_of_first_fixed_surface);
        else AMIO_GRIB2_SETTING("scale_factor_first_surface", scale_factor_first_surface);
        else AMIO_GRIB2_SETTING("scaled_value_first_surface", scaled_value_first_surface);
        else AMIO_GRIB2_SETTING("forecast_time", forecast_time);
        else AMIO_GRIB2_SETTING("indicator_of_unit_of_time", indicator_of_unit_of_time);
        else AMIO_GRIB2_SETTING("decimal_scale_factor", decimal_scale_factor);
        else AMIO_GRIB2_SETTING("chemical_constituent_type", chemical_constituent_type);
        else AMIO_GRIB2_SETTING("aerosol_type", aerosol_type);
        else AMIO_GRIB2_SETTING("size_dist_param_first", size_dist_param_first);
        else AMIO_GRIB2_SETTING("size_dist_param_second", size_dist_param_second);
        else AMIO_GRIB2_SETTING("optical_property_type", optical_property_type);
        else AMIO_GRIB2_SETTING("wavelength_first_nm", wavelength_first_nm);
        else AMIO_GRIB2_SETTING("wavelength_last_nm", wavelength_last_nm);
        else AMIO_GRIB2_SETTING("ensemble_perturbation_number", ensemble_perturbation_number);
        else AMIO_GRIB2_SETTING("statistical_process", statistical_process);
        else AMIO_GRIB2_SETTING("time_range_unit", time_range_unit);
        else AMIO_GRIB2_SETTING("time_range_length", time_range_length);
        else AMIO_GRIB2_SETTING("number_of_time_range_specs", number_of_time_range_specs);
        else AMIO_GRIB2_SETTING("total_missing_from_statistical_process", total_missing_from_statistical_process);
        else AMIO_GRIB2_SETTING("n_parallel", n_parallel);
        else throw std::invalid_argument("unsupported GRIB2 variable metadata attribute: " + name);
#undef AMIO_GRIB2_SETTING
    }
    return settings;
}

std::string grib2_identity(const amio::detail::Grib2Settings &settings) {
    return amio::detail::GRIB2_Driver::field_identity_name(
        settings.discipline, settings.parameter_category, settings.parameter_number,
        settings.type_of_first_fixed_surface, settings.scaled_value_first_surface, settings.pdt_number,
        settings.chemical_constituent_type, settings.aerosol_type, settings.optical_property_type,
        settings.wavelength_first_nm, settings.wavelength_last_nm, settings.ensemble_perturbation_number,
        settings.statistical_process);
}

bool variable_payload_size(const amio::detail::Variable &variable, std::size_t &bytes) noexcept {
    const auto width = dtype_size(variable.source_dtype);
    if (width == 0) return false;
    std::size_t elements = 1;
    for (const auto &dimension : variable.dimensions) {
        if (dimension.extent == 0 || dimension.extent > std::numeric_limits<std::size_t>::max() / elements) return false;
        elements *= static_cast<std::size_t>(dimension.extent);
    }
    if (width > std::numeric_limits<std::size_t>::max() / elements) return false;
    bytes = width * elements;
    return true;
}

std::string step_payload_name(std::uint64_t step_id) {
    auto digits = std::to_string(step_id);
    std::string name = "payloads/steps/";
    if (digits.size() < 20) name.append(20 - digits.size(), '0');
    name += digits;
    name += ".nc";
    return name;
}

bool valid_request(const amio_validation_request_t *request) noexcept {
    if (request == nullptr || request->struct_size < offsetof(amio_validation_request_t, version) + sizeof(request->version)) {
        return false;
    }
    if (request->version != AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 ||
        request->struct_size < offsetof(amio_validation_request_t, reserved) + sizeof(request->reserved)) return false;
    if (request->reserved32 != 0 || request->reserved[0] != 0 || request->reserved[1] != 0 || request->max_results == 0) {
        return false;
    }
    constexpr std::uint64_t known = (UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1)) |
                                    (UINT64_C(1) << (AMIO_VALIDATE_REQUIRED_FIELDS - 1)) |
                                    (UINT64_C(1) << (AMIO_VALIDATE_PAYLOAD_PRESENCE - 1)) |
                                    (UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1)) |
                                    (UINT64_C(1) << (AMIO_VALIDATE_ORPHAN_OBJECTS - 1));
    return request->check_mask != 0 && (request->check_mask & ~known) == 0;
}

template <typename Handle>
amio_status_t release_typed(Handle handle, HandleKind kind) noexcept {
    if (handle == nullptr) return AMIO_ERR_NULL_HANDLE;
    return process_handle_table().release(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(handle))), kind);
}

template <typename Handle>
bool token_payload(Handle handle, HandleKind kind, void **payload) noexcept {
    if (handle == nullptr) return false;
    return process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(handle))), kind, payload) == AMIO_OK;
}

template <typename Descriptor>
bool valid_descriptor_header(const Descriptor *descriptor, std::size_t minimum_size) noexcept {
    return descriptor != nullptr && descriptor->struct_size >= minimum_size &&
           descriptor->version == AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
}

bool copy_utf8(const char *data, std::uint64_t bytes, std::string &out, bool allow_empty = false) {
    if (bytes > amio::detail::Schema::kMaxStringBytes || bytes > std::numeric_limits<std::size_t>::max() ||
        (bytes != 0 && data == nullptr) || (!allow_empty && bytes == 0)) return false;
    out.assign(data == nullptr ? "" : data, static_cast<std::size_t>(bytes));
    return out.find('\0') == std::string::npos;
}

bool copy_fixed_text(const char *data, std::size_t capacity, std::string &out, bool allow_empty = true) {
    if (data == nullptr || capacity == 0) return false;
    const auto *end = static_cast<const char *>(std::memchr(data, '\0', capacity));
    if (end == nullptr) return false;
    out.assign(data, static_cast<std::size_t>(end - data));
    return allow_empty || !out.empty();
}

bool fill_step_summary(const amio::detail::CatalogStep &step, amio_step_summary_t *out) {
    if (out == nullptr || step.completion.schema_digest.size() > sizeof(out->schema_digest) - 1) return false;
    *out = {};
    out->struct_size = sizeof(*out);
    out->version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    out->step_id = step.completion.step_id;
    out->present_field_count = static_cast<std::uint32_t>(step.completion.present_fields.size());
    out->required_field_count = static_cast<std::uint32_t>(step.completion.required_fields.size());
    std::memcpy(out->schema_digest, step.completion.schema_digest.data(), step.completion.schema_digest.size());
    return true;
}

std::string random_identity(std::string_view prefix) {
    static constexpr char hex[] = "0123456789abcdef";
    std::random_device source;
    std::string result(prefix);
    result.reserve(prefix.size() + 32);
    for (int i = 0; i < 16; ++i) {
        const auto value = static_cast<unsigned int>(source()) & 0xffu;
        result.push_back(hex[value >> 4]);
        result.push_back(hex[value & 0x0fu]);
    }
    return result;
}

bool convert_attribute(const amio_attribute_descriptor_t &descriptor, amio::detail::Attribute &out) {
    constexpr auto minimum = offsetof(amio_attribute_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    if (!valid_descriptor_header(&descriptor, minimum) || descriptor.reserved32 != 0 || descriptor.reserved[0] != 0 ||
        descriptor.reserved[1] != 0 || descriptor.element_count == 0 || descriptor.element_count > 1'000'000 ||
        descriptor.value_bytes > amio::detail::Schema::kMaxStringBytes || descriptor.value_bytes > std::numeric_limits<std::size_t>::max() ||
        descriptor.value == nullptr) return false;

    auto copy_values = [&]<typename T>(std::vector<T> &values) {
        if (descriptor.element_count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
            descriptor.value_bytes != descriptor.element_count * sizeof(T)) return false;
        values.resize(static_cast<std::size_t>(descriptor.element_count));
        std::memcpy(values.data(), descriptor.value, static_cast<std::size_t>(descriptor.value_bytes));
        return true;
    };
    switch (descriptor.type) {
        case AMIO_ATTRIBUTE_BOOL: {
            if (descriptor.element_count != 1 || descriptor.value_bytes != 1) return false;
            const auto value = *static_cast<const std::uint8_t *>(descriptor.value);
            if (value > 1) return false;
            out.value = value != 0;
            return true;
        }
        case AMIO_ATTRIBUTE_I64: {
            std::vector<std::int64_t> values;
            if (!copy_values(values)) return false;
            if (values.size() == 1) out.value = values.front(); else out.value = std::move(values);
            return true;
        }
        case AMIO_ATTRIBUTE_U64: {
            std::vector<std::uint64_t> values;
            if (!copy_values(values)) return false;
            if (values.size() == 1) out.value = values.front(); else out.value = std::move(values);
            return true;
        }
        case AMIO_ATTRIBUTE_F64: {
            std::vector<double> values;
            if (!copy_values(values)) return false;
            if (values.size() == 1) out.value = values.front(); else out.value = std::move(values);
            return true;
        }
        case AMIO_ATTRIBUTE_UTF8: {
            if (descriptor.element_count != 1) return false;
            std::string value;
            if (!copy_utf8(static_cast<const char *>(descriptor.value), descriptor.value_bytes, value, true)) return false;
            out.value = std::move(value);
            return true;
        }
        default: return false;
    }
}

amio_status_t compare_schema_collectively(const amio::detail::ExecutionContext &context, const std::string &canonical) {
#ifdef AMIO_HAS_MPI
    const MPI_Comm comm = context.communicator();
    unsigned long long root_bytes = context.rank() == 0 ? static_cast<unsigned long long>(canonical.size()) : 0;
    if (MPI_Bcast(&root_bytes, 1, MPI_UNSIGNED_LONG_LONG, 0, comm) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    if (root_bytes > amio::detail::Schema::kMaxCanonicalBytes) return AMIO_ERR_INVALID_INPUT;
    std::string root_canonical(static_cast<std::size_t>(root_bytes), '\0');
    if (context.rank() == 0) root_canonical = canonical;
    constexpr std::size_t chunk_bytes = 1u << 20;
    for (std::size_t offset = 0; offset < root_canonical.size(); offset += chunk_bytes) {
        const auto count = std::min(chunk_bytes, root_canonical.size() - offset);
        if (MPI_Bcast(root_canonical.data() + offset, static_cast<int>(count), MPI_CHAR, 0, comm) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    const int mismatch_local = canonical != root_canonical ? 1 : 0;
    int mismatch_any = 0;
    if (MPI_Allreduce(&mismatch_local, &mismatch_any, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    return mismatch_any ? AMIO_ERR_SCHEMA_MISMATCH : AMIO_OK;
#else
    (void)context;
    (void)canonical;
    return AMIO_OK;
#endif
}

amio_status_t collective_status(const amio::detail::ExecutionContext &context, amio_status_t local_status) {
#ifdef AMIO_HAS_MPI
    const MPI_Comm comm = context.communicator();
    int rank = 0;
    if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    const int candidate = local_status == AMIO_OK ? std::numeric_limits<int>::max() : rank;
    int first_failure_rank = std::numeric_limits<int>::max();
    if (MPI_Allreduce(&candidate, &first_failure_rank, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (first_failure_rank == std::numeric_limits<int>::max()) return AMIO_OK;
    int agreed_status = rank == first_failure_rank ? local_status : AMIO_OK;
    if (MPI_Bcast(&agreed_status, 1, MPI_INT, first_failure_rank, comm) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    return static_cast<amio_status_t>(agreed_status);
#else
    (void)context;
    return local_status;
#endif
}

amio_status_t validate_partition_descriptors(const amio::detail::Schema &schema,
                                             const amio_field_partition_descriptor_t *partitions,
                                             size_t partition_count) {
    if (partitions == nullptr || partition_count != schema.variables().size()) return AMIO_ERR_INVALID_INPUT;
    std::set<std::string, std::less<>> seen;
    for (size_t i = 0; i < partition_count; ++i) {
        const auto &part = partitions[i];
        constexpr std::size_t minimum = offsetof(amio_field_partition_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
        if (!valid_descriptor_header(&part, minimum) || part.reserved32 != 0 || part.reserved[0] != 0 || part.reserved[1] != 0)
            return AMIO_ERR_INVALID_INPUT;
        std::string name;
        if (!copy_utf8(part.field, part.field_bytes, name) || !seen.insert(name).second) return AMIO_ERR_INVALID_INPUT;
        const auto found = schema.variables().find(name);
        if (found == schema.variables().end() || part.rank != found->second.dimensions.size() || part.rank > AMIO_MAX_RANK)
            return AMIO_ERR_INVALID_INPUT;
        for (std::size_t d = 0; d < part.rank; ++d) {
            if (part.offsets[d] < 0 || part.extents[d] < 0) return AMIO_ERR_INVALID_INPUT;
            const auto offset = static_cast<std::uint64_t>(part.offsets[d]);
            const auto extent = static_cast<std::uint64_t>(part.extents[d]);
            const auto global = found->second.dimensions[d].extent;
            if (offset > global || extent > global - offset) return AMIO_ERR_INVALID_INPUT;
        }
        for (std::size_t d = part.rank; d < AMIO_MAX_RANK; ++d)
            if (part.offsets[d] != 0 || part.extents[d] != 0) return AMIO_ERR_INVALID_INPUT;
    }
    return seen.size() == schema.variables().size() ? AMIO_OK : AMIO_ERR_INVALID_INPUT;
}

struct PartitionWire {
    std::uint32_t present = 0;
    std::uint32_t rank = 0;
    std::uint64_t offsets[AMIO_MAX_RANK]{};
    std::uint64_t extents[AMIO_MAX_RANK]{};
};

amio_status_t validate_distributed_partitions(
    EvolutionDatasetPayload &dataset, const ProducerStepPayload &step, const amio::detail::Schema &schema,
    std::map<std::string, bool, std::less<>> &present_fields) {
#ifdef AMIO_HAS_MPI
    const auto &context = *dataset.context->context;
    const MPI_Comm comm = context.communicator();
    const int rank_count = static_cast<int>(context.size());
    if (rank_count <= 1) return AMIO_ERR_INVALID_POLICY;
    for (const auto &[name, variable] : schema.variables()) {
        const auto planned = step.partitions.find(name);
        if (planned == step.partitions.end()) return AMIO_ERR_INVALID_POLICY;
        PartitionWire local{};
        local.present = dataset.steps->contributions().contains(name) ? 1u : 0u;
        local.rank = static_cast<std::uint32_t>(variable.dimensions.size());
        for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
            local.offsets[d] = planned->second.offsets[d];
            local.extents[d] = planned->second.extents[d];
        }
        std::vector<PartitionWire> gathered(static_cast<std::size_t>(rank_count));
        if (MPI_Allgather(&local, static_cast<int>(sizeof(local)), MPI_BYTE, gathered.data(),
                          static_cast<int>(sizeof(local)), MPI_BYTE, comm) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;

        std::vector<const PartitionWire *> active;
        for (const auto &partition : gathered) if (partition.present != 0) active.push_back(&partition);
        if (active.empty()) {
            if (variable.required) return AMIO_ERR_FIELD_NOT_PRESENT;
            present_fields[name] = false;
            continue;
        }

        std::uint64_t global_volume = 1;
        for (const auto &dimension : variable.dimensions) {
            if (dimension.extent == 0 || global_volume > std::numeric_limits<std::uint64_t>::max() / dimension.extent)
                return AMIO_ERR_INVALID_INPUT;
            global_volume *= dimension.extent;
        }
        std::uint64_t volume_sum = 0;
        for (std::size_t i = 0; i < active.size(); ++i) {
            const auto &a = *active[i];
            if (a.rank != variable.dimensions.size()) return AMIO_ERR_SCHEMA_MISMATCH;
            std::uint64_t volume = 1;
            for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
                const auto global = variable.dimensions[d].extent;
                const auto offset = a.offsets[d], extent = a.extents[d];
                if (extent == 0 || offset > global || extent > global - offset ||
                    volume > std::numeric_limits<std::uint64_t>::max() / extent)
                    return AMIO_ERR_INVALID_INPUT;
                volume *= extent;
            }
            if (volume > std::numeric_limits<std::uint64_t>::max() - volume_sum) return AMIO_ERR_INVALID_INPUT;
            volume_sum += volume;
            for (std::size_t j = 0; j < i; ++j) {
                const auto &b = *active[j];
                bool intersects = true;
                for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
                    const auto a_end = a.offsets[d] + a.extents[d];
                    const auto b_end = b.offsets[d] + b.extents[d];
                    if (a_end <= b.offsets[d] || b_end <= a.offsets[d]) { intersects = false; break; }
                }
                if (intersects) return AMIO_ERR_INVALID_POLICY;
            }
        }
        if (volume_sum != global_volume) return AMIO_ERR_INVALID_POLICY;
        present_fields[name] = true;
    }
    return AMIO_OK;
#else
    (void)dataset; (void)step; (void)schema; (void)present_fields;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t derive_distributed_zarr_shards(
    EvolutionDatasetPayload &dataset, const ProducerStepPayload &step, const amio::detail::Schema &schema,
    amio::detail::ZarrShardShapes &shard_shapes) {
#if defined(AMIO_HAS_MPI) && defined(AMIO_HAS_TENSORSTORE)
    const auto &context = *dataset.context->context;
    const int rank_count = static_cast<int>(context.size());
    std::map<std::string, std::vector<amio::detail::ZarrRegionView>, std::less<>> all_regions;
    for (const auto &[name, variable] : schema.variables()) {
        const auto planned = step.partitions.find(name);
        if (planned == step.partitions.end()) return AMIO_ERR_INVALID_POLICY;
        PartitionWire local{};
        local.present = planned->second.empty ? 0u : 1u;
        local.rank = static_cast<std::uint32_t>(variable.dimensions.size());
        for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
            local.offsets[d] = planned->second.offsets[d];
            local.extents[d] = planned->second.extents[d];
        }
        std::vector<PartitionWire> gathered(static_cast<std::size_t>(rank_count));
        if (MPI_Allgather(&local, static_cast<int>(sizeof(local)), MPI_BYTE, gathered.data(),
                          static_cast<int>(sizeof(local)), MPI_BYTE, context.communicator()) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        auto &regions = all_regions[name];
        for (const auto &part : gathered) {
            if (part.present == 0) continue;
            if (part.rank != variable.dimensions.size()) return AMIO_ERR_SCHEMA_MISMATCH;
            amio::detail::ZarrRegionView region;
            for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
                region.offsets.push_back(part.offsets[d]);
                region.extents.push_back(part.extents[d]);
            }
            regions.push_back(std::move(region));
        }
    }
    const auto budget = static_cast<std::size_t>(std::min<std::uint64_t>(dataset.payload_budget_bytes, 64u * 1024u * 1024u));
    std::string error;
    return amio::detail::derive_zarr_shard_shapes(schema, all_regions, budget, shard_shapes, error)
               ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
#else
    (void)dataset; (void)step; (void)schema; (void)shard_shapes;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t broadcast_publication(const amio::detail::ExecutionContext &context,
                                    amio::detail::PublicationResult &result) {
#ifdef AMIO_HAS_MPI
    const MPI_Comm comm = context.communicator();
    int wire[2] = {static_cast<int>(result.status), static_cast<int>(result.state)};
    unsigned long long path_bytes = static_cast<unsigned long long>(result.commit_path.size());
    if (MPI_Bcast(wire, 2, MPI_INT, 0, comm) != MPI_SUCCESS ||
        MPI_Bcast(&path_bytes, 1, MPI_UNSIGNED_LONG_LONG, 0, comm) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (path_bytes > 1024 * 1024) return AMIO_ERR_BACKEND_FAILURE;
    if (context.rank() != 0) result.commit_path.resize(static_cast<std::size_t>(path_bytes));
    constexpr std::size_t chunk = 1u << 20;
    for (std::size_t offset = 0; offset < result.commit_path.size(); offset += chunk) {
        const auto count = std::min(chunk, result.commit_path.size() - offset);
        if (MPI_Bcast(result.commit_path.data() + offset, static_cast<int>(count), MPI_CHAR, 0, comm) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    result.status = static_cast<amio_status_t>(wire[0]);
    result.state = static_cast<amio::detail::PublicationState>(wire[1]);
#else
    (void)context; (void)result;
#endif
    return AMIO_OK;
}
} // namespace

extern "C" {

AMIO_API amio_status_t amio_init_ex(const amio_context_descriptor_t *descriptor, amio_context_handle *out_context) {
    if (out_context == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_context = nullptr;
    constexpr std::size_t minimum_size = offsetof(amio_context_descriptor_t, reserved) + 2 * sizeof(uint64_t);
    if (descriptor == nullptr) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->struct_size < offsetof(amio_context_descriptor_t, version) + sizeof(descriptor->version)) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->version != AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 || descriptor->struct_size < minimum_size) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->rank_count == 0 || descriptor->rank_count > 1'000'000 || descriptor->configuration_bytes == 0 ||
        descriptor->configuration_bytes > 1024 * 1024 || descriptor->canonical_configuration == nullptr ||
        descriptor->producer_count > descriptor->rank_count || descriptor->writer_count > descriptor->rank_count ||
        descriptor->consumer_count > descriptor->rank_count ||
        (descriptor->producer_count != 0 && descriptor->producers == nullptr) ||
        (descriptor->writer_count != 0 && descriptor->writers == nullptr) ||
        (descriptor->consumer_count != 0 && descriptor->consumers == nullptr)) return AMIO_ERR_INVALID_INPUT;

    try {
        amio::detail::RoleConfiguration roles;
        roles.rank_count = descriptor->rank_count;
        switch (descriptor->execution_mode) {
            case AMIO_EXECUTION_PERSISTENT_DIRECT: roles.mode = amio::detail::ExecutionMode::Direct; break;
            case AMIO_EXECUTION_PERSISTENT_AGGREGATED: roles.mode = amio::detail::ExecutionMode::Aggregated; break;
            case AMIO_EXECUTION_MPI_STREAM: roles.mode = amio::detail::ExecutionMode::Stream; break;
            default: return AMIO_ERR_INVALID_POLICY;
        }
        amio::detail::ProgressPolicy progress;
        if (descriptor->progress_mode == AMIO_PROGRESS_CALLER) progress = amio::detail::ProgressPolicy::Caller;
        else if (descriptor->progress_mode == AMIO_PROGRESS_BACKGROUND) progress = amio::detail::ProgressPolicy::Background;
        else return AMIO_ERR_INVALID_POLICY;
        if (descriptor->producer_count) roles.producers.assign(descriptor->producers, descriptor->producers + descriptor->producer_count);
        if (descriptor->writer_count) roles.writers.assign(descriptor->writers, descriptor->writers + descriptor->writer_count);
        if (descriptor->consumer_count) roles.consumers.assign(descriptor->consumers, descriptor->consumers + descriptor->consumer_count);
        const std::string canonical(descriptor->canonical_configuration, static_cast<std::size_t>(descriptor->configuration_bytes));
        std::unique_ptr<amio::detail::ExecutionContext> context;
#ifdef AMIO_HAS_MPI
        if (descriptor->fortran_communicator < std::numeric_limits<MPI_Fint>::min() ||
            descriptor->fortran_communicator > std::numeric_limits<MPI_Fint>::max()) return AMIO_ERR_INVALID_INPUT;
        const MPI_Comm caller = MPI_Comm_f2c(static_cast<MPI_Fint>(descriptor->fortran_communicator));
        const auto status = amio::detail::ExecutionContext::create(roles, progress, canonical, caller, context);
#else
        if (descriptor->fortran_communicator != 0) return AMIO_ERR_UNSUPPORTED;
        const auto status = amio::detail::ExecutionContext::create(roles, progress, canonical, 0, context);
#endif
        if (status != AMIO_OK) return status;
        auto payload = std::make_unique<ContextPayload>();
        payload->context = std::move(context);
        payload->canonical_configuration = canonical;
        const auto token = process_handle_table().insert(HandleKind::Context, payload.get());
        if (token == 0) return AMIO_ERR_BACKEND_FAILURE;
        *out_context = reinterpret_cast<amio_context_handle>(HandleTable::to_ptr(token));
        payload.release();
        return AMIO_OK;
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
}

AMIO_API amio_status_t amio_query_capabilities(amio_context_handle context, char *buffer, size_t buffer_capacity,
                                               size_t *out_required_bytes) {
    if (out_required_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_required_bytes = 0;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) {
        return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    }
    (void)payload;
    static constexpr char report[] =
        "{\"api_version\":1,\"capability_levels\":[\"unsupported\",\"compiled\",\"environment_validated\",\"destination_certified\"],"
        "\"features\":{\"immutable_schema\":\"compiled\",\"borrowed_buffer\":\"unsupported\","
        "\"device_memory\":\"unsupported\",\"destination_visibility\":\"compiled\",\"storage_sync\":\"compiled\"}}";
    constexpr std::size_t required = sizeof(report);
    *out_required_bytes = required;
    if (buffer == nullptr && buffer_capacity == 0) return AMIO_OK;
    if (buffer == nullptr || buffer_capacity < required) return AMIO_ERR_BUFFER_TOO_SMALL;
    std::memcpy(buffer, report, required);
    return AMIO_OK;
}

AMIO_API amio_status_t amio_release_context(amio_context_handle context) {
    if (context == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(context))), HandleKind::Context, &payload);
    if (lookup != AMIO_OK) return lookup;
    if (static_cast<ContextPayload *>(payload)->open_datasets.load() != 0) return AMIO_ERR_INVALID_POLICY;
    const auto status = process_handle_table().release(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(context))), HandleKind::Context, &payload);
    if (status != AMIO_OK) return status;
    delete static_cast<ContextPayload *>(payload);
    return AMIO_OK;
}

AMIO_API amio_status_t amio_open_dataset_ex(amio_context_handle context, const amio_dataset_descriptor_t *descriptor,
                                            amio_dataset_handle *out_dataset) {
    if (out_dataset == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_dataset = nullptr;
    void *context_payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &context_payload))
        return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(context_payload);
    constexpr std::size_t minimum = offsetof(amio_dataset_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    amio_status_t local_status = AMIO_OK;
    if (owner->context->roles().mode != amio::detail::ExecutionMode::Direct)
        local_status = AMIO_ERR_UNSUPPORTED;
    if (!valid_descriptor_header(descriptor, minimum)) local_status = AMIO_ERR_INVALID_INPUT;
    else if (descriptor->reserved32 != 0 || descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0 ||
             descriptor->location_bytes == 0 || descriptor->location_bytes > 1024 * 1024 || descriptor->location == nullptr ||
             descriptor->budget_bytes == 0 || descriptor->budget_descriptors == 0 ||
             descriptor->control_budget_bytes > descriptor->budget_bytes ||
             descriptor->control_budget_descriptors > descriptor->budget_descriptors ||
             descriptor->budget_bytes > std::numeric_limits<std::size_t>::max() ||
             descriptor->budget_descriptors > std::numeric_limits<std::size_t>::max() ||
             descriptor->control_budget_bytes > std::numeric_limits<std::size_t>::max() ||
             descriptor->control_budget_descriptors > std::numeric_limits<std::size_t>::max()) local_status = AMIO_ERR_INVALID_INPUT;
    else if (descriptor->format != AMIO_FORMAT_NETCDF4 && descriptor->format != AMIO_FORMAT_ZARR3 &&
             descriptor->format != AMIO_FORMAT_GRIB2)
        local_status = AMIO_ERR_UNSUPPORTED;
    else if (owner->context->size() > 1 && descriptor->format != AMIO_FORMAT_NETCDF4 &&
             !(descriptor->format == AMIO_FORMAT_ZARR3 && kTensorStoreAvailable))
        local_status = AMIO_ERR_UNSUPPORTED;
#if !defined(AMIO_HAS_MPI) || !defined(AMIO_HAS_PARALLEL_NETCDF)
    else if (owner->context->size() > 1 && descriptor->format == AMIO_FORMAT_NETCDF4)
        local_status = AMIO_ERR_UNSUPPORTED;
#endif
#ifndef AMIO_HAS_TENSORSTORE
    else if (descriptor->format == AMIO_FORMAT_ZARR3) local_status = AMIO_ERR_UNSUPPORTED;
#endif
#ifndef AMIO_HAS_G2C
    else if (descriptor->format == AMIO_FORMAT_GRIB2) local_status = AMIO_ERR_UNSUPPORTED;
#endif
    auto agreed_status = collective_status(*owner->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

    std::unique_ptr<EvolutionDatasetPayload> dataset;
    local_status = AMIO_OK;
    try {
        dataset = std::make_unique<EvolutionDatasetPayload>();
        dataset->context = owner;
        dataset->format = static_cast<amio_evolution_format_t>(descriptor->format);
        dataset->distributed_netcdf = owner->context->size() > 1 && descriptor->format == AMIO_FORMAT_NETCDF4;
        dataset->distributed_zarr = owner->context->size() > 1 && descriptor->format == AMIO_FORMAT_ZARR3;
        dataset->payload_budget_bytes = descriptor->budget_bytes;
        const auto storage_config = conf::Config::from_string(owner->canonical_configuration);
        if (storage_config.has("completion.level")) {
            const auto level = storage_config.get_string("completion.level");
            if (level == "visibility") dataset->requested_sync = amio::detail::PosixSyncLevel::Visibility;
            else if (level == "file_sync") dataset->requested_sync = amio::detail::PosixSyncLevel::FileSync;
            else if (level == "directory_sync") dataset->requested_sync = amio::detail::PosixSyncLevel::DirectorySync;
            else local_status = AMIO_ERR_INVALID_POLICY;
        }
        if (dataset->format == AMIO_FORMAT_ZARR3 && dataset->requested_sync != amio::detail::PosixSyncLevel::Visibility)
            local_status = AMIO_ERR_UNSUPPORTED;
        if (!copy_utf8(descriptor->location, descriptor->location_bytes, dataset->location) ||
            dataset->location.find('\n') != std::string::npos || dataset->location.find('\r') != std::string::npos)
            local_status = AMIO_ERR_INVALID_INPUT;
#ifdef AMIO_HAS_G2C
        if (local_status == AMIO_OK && descriptor->format == AMIO_FORMAT_GRIB2) {
            dataset->grib2_driver = std::make_unique<amio::detail::GRIB2_Driver>();
            dataset->grib2_driver->open_step_encoder(conf::Config::from_string(owner->canonical_configuration));
        }
#endif
    } catch (const std::bad_alloc &) {
        local_status = AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        local_status = AMIO_ERR_INVALID_INPUT;
    }
    agreed_status = collective_status(*owner->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

#ifdef AMIO_HAS_MPI
    const MPI_Comm comm = owner->context->communicator();
    const std::uint64_t local_values[] = {descriptor->format, descriptor->budget_bytes, descriptor->budget_descriptors,
                                          descriptor->control_budget_bytes, descriptor->control_budget_descriptors};
    std::uint64_t min_values[5] = {}, max_values[5] = {};
    if (MPI_Allreduce(local_values, min_values, 5, MPI_UINT64_T, MPI_MIN, comm) != MPI_SUCCESS ||
        MPI_Allreduce(local_values, max_values, 5, MPI_UINT64_T, MPI_MAX, comm) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    for (std::size_t i = 0; i < 5; ++i) if (min_values[i] != max_values[i]) return AMIO_ERR_INVALID_POLICY;
#endif
    agreed_status = compare_schema_collectively(*owner->context, dataset->location);
    if (agreed_status != AMIO_OK) return agreed_status;

    int root_status = AMIO_OK;
    if (owner->context->rank() == 0) {
        try {
            dataset->uuid = random_identity("dataset-");
        } catch (...) {
            root_status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
#ifdef AMIO_HAS_MPI
    if (MPI_Bcast(&root_status, 1, MPI_INT, 0, owner->context->communicator()) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    constexpr std::size_t uuid_capacity = 64;
    char uuid_buffer[uuid_capacity] = {};
    if (owner->context->rank() == 0 && root_status == AMIO_OK) {
        if (dataset->uuid.size() >= uuid_capacity) root_status = AMIO_ERR_BACKEND_FAILURE;
        else std::memcpy(uuid_buffer, dataset->uuid.data(), dataset->uuid.size());
    }
    if (MPI_Bcast(&root_status, 1, MPI_INT, 0, owner->context->communicator()) != MPI_SUCCESS ||
        MPI_Bcast(uuid_buffer, static_cast<int>(uuid_capacity), MPI_CHAR, 0, owner->context->communicator()) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (root_status != AMIO_OK) return static_cast<amio_status_t>(root_status);
    dataset->uuid.assign(uuid_buffer);
#endif

    local_status = AMIO_OK;
    try {
        dataset->budget = std::make_unique<amio::detail::ResourceBudget>(amio::detail::ResourceBudget::Limits{
            static_cast<std::size_t>(descriptor->budget_bytes), static_cast<std::size_t>(descriptor->budget_descriptors),
            static_cast<std::size_t>(descriptor->control_budget_bytes), static_cast<std::size_t>(descriptor->control_budget_descriptors)});
        dataset->steps = std::make_unique<amio::detail::StepCoordinator>(dataset->uuid, *dataset->budget);
    } catch (const std::bad_alloc &) {
        local_status = AMIO_ERR_BACKEND_FAILURE;
    } catch (const std::invalid_argument &) {
        local_status = AMIO_ERR_INVALID_INPUT;
    } catch (...) {
        local_status = AMIO_ERR_BACKEND_FAILURE;
    }
    agreed_status = collective_status(*owner->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

    if (owner->context->rank() == 0) {
        try {
            const auto owner_token = random_identity("owner-");
            dataset->catalog = std::make_unique<amio::detail::PosixCatalog>(dataset->location, dataset->uuid, owner_token);
            root_status = dataset->catalog->create_exclusive();
        } catch (...) {
            root_status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
#ifdef AMIO_HAS_MPI
    if (MPI_Bcast(&root_status, 1, MPI_INT, 0, owner->context->communicator()) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
#endif
    if (root_status != AMIO_OK) return static_cast<amio_status_t>(root_status);

    const auto token = process_handle_table().insert(HandleKind::EvolutionDataset, dataset.get());
    local_status = token == 0 ? AMIO_ERR_BACKEND_FAILURE : AMIO_OK;
    agreed_status = collective_status(*owner->context, local_status);
    if (agreed_status != AMIO_OK) {
        if (token != 0) {
            void *unused = nullptr;
            (void)process_handle_table().release(token, HandleKind::EvolutionDataset, &unused);
        }
        return agreed_status;
    }
    owner->open_datasets.fetch_add(1);
    *out_dataset = reinterpret_cast<amio_dataset_handle>(HandleTable::to_ptr(token));
    dataset.release();
    return AMIO_OK;
}

AMIO_API amio_status_t amio_close_dataset_ex(amio_dataset_handle dataset) {
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *evolution_dataset = static_cast<EvolutionDatasetPayload *>(payload);
    std::unique_lock<std::mutex> lock(evolution_dataset->mutex);
    if (evolution_dataset->open_step_handles.load() != 0) return AMIO_ERR_INVALID_POLICY;
    if (evolution_dataset->open_step_leases.load() != 0) return AMIO_ERR_INVALID_POLICY;
    const auto state = evolution_dataset->steps->state();
    if (state == amio::detail::StepState::Open || state == amio::detail::StepState::Sealed ||
        state == amio::detail::StepState::Transferring || state == amio::detail::StepState::PayloadReady ||
        state == amio::detail::StepState::Publishing || state == amio::detail::StepState::Reconciling)
        return AMIO_ERR_INVALID_POLICY;
    void *released = nullptr;
    const auto status = process_handle_table().release(token, HandleKind::EvolutionDataset, &released);
    if (status != AMIO_OK) return status;
    auto *closed = static_cast<EvolutionDatasetPayload *>(released);
    lock.unlock();
    closed->context->open_datasets.fetch_sub(1);
    delete closed;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_define_schema(amio_dataset_handle dataset, const amio_schema_descriptor_t *descriptor,
                                          amio_schema_handle *out_schema) {
    if (out_schema == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_schema = nullptr;
    void *dataset_payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))), HandleKind::EvolutionDataset, &dataset_payload);
    if (lookup != AMIO_OK) return dataset == nullptr ? AMIO_ERR_NULL_HANDLE : lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(dataset_payload);
    std::lock_guard<std::mutex> dataset_lock(owner->mutex);

    constexpr std::size_t minimum = offsetof(amio_schema_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    amio_status_t local_status = AMIO_OK;
    if (!valid_descriptor_header(descriptor, minimum)) local_status = AMIO_ERR_INVALID_INPUT;
    else if (descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0 || descriptor->variable_count == 0 ||
             descriptor->variable_count > amio::detail::Schema::kMaxVariables || descriptor->variables == nullptr)
        local_status = AMIO_ERR_INVALID_INPUT;
    auto agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

    std::shared_ptr<amio::detail::Schema> schema;
    auto build_schema = [&]() -> amio_status_t {
      try {
        schema = std::make_shared<amio::detail::Schema>();
        std::set<std::string, std::less<>> grib_identities;
        for (std::uint64_t i = 0; i < descriptor->variable_count; ++i) {
            const auto &input = descriptor->variables[i];
            constexpr std::size_t var_minimum = offsetof(amio_variable_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
            if (!valid_descriptor_header(&input, var_minimum) || input.reserved32 != 0 || input.reserved[0] != 0 ||
                input.reserved[1] != 0 || input.required > 1 || input.dimension_count == 0 ||
                input.dimension_count > AMIO_MAX_RANK || input.dimensions == nullptr || input.attribute_count >
                amio::detail::Schema::kMaxAttributesPerVariable || (input.attribute_count != 0 && input.attributes == nullptr))
                return AMIO_ERR_INVALID_INPUT;
            amio::detail::Variable variable;
            if (!copy_utf8(input.name, input.name_bytes, variable.name)) return AMIO_ERR_INVALID_INPUT;
            if (input.source_dtype > AMIO_DTYPE_U64 || input.stored_dtype > AMIO_DTYPE_U64) return AMIO_ERR_INVALID_INPUT;
            variable.source_dtype = static_cast<amio_dtype_t>(input.source_dtype);
            variable.stored_dtype = static_cast<amio_dtype_t>(input.stored_dtype);
            variable.required = input.required != 0;
            variable.dimensions.reserve(input.dimension_count);
            for (std::uint32_t d = 0; d < input.dimension_count; ++d) {
                const auto &dim = input.dimensions[d];
                constexpr std::size_t dim_minimum = offsetof(amio_dimension_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
                if (!valid_descriptor_header(&dim, dim_minimum) || dim.reserved[0] != 0 || dim.reserved[1] != 0)
                    return AMIO_ERR_INVALID_INPUT;
                amio::detail::Dimension dimension;
                if (!copy_utf8(dim.name, dim.name_bytes, dimension.name)) return AMIO_ERR_INVALID_INPUT;
                dimension.extent = dim.extent;
                if (dim.coordinate_variable_bytes != 0) {
                    std::string coordinate;
                    if (!copy_utf8(dim.coordinate_variable, dim.coordinate_variable_bytes, coordinate)) return AMIO_ERR_INVALID_INPUT;
                    dimension.coordinate_variable = std::move(coordinate);
                } else if (dim.coordinate_variable != nullptr) return AMIO_ERR_INVALID_INPUT;
                variable.dimensions.push_back(std::move(dimension));
            }
            for (std::uint32_t a = 0; a < input.attribute_count; ++a) {
                const auto &attr_input = input.attributes[a];
                amio::detail::Attribute attribute;
                std::string name;
                if (!copy_utf8(attr_input.name, attr_input.name_bytes, name) || !convert_attribute(attr_input, attribute) ||
                    !variable.attributes.emplace(std::move(name), std::move(attribute)).second) return AMIO_ERR_INVALID_INPUT;
            }
            if (owner->format == AMIO_FORMAT_GRIB2) {
                if (variable.source_dtype != AMIO_DTYPE_F32 || variable.stored_dtype != AMIO_DTYPE_F32 ||
                    variable.dimensions.empty() || variable.dimensions.size() > 2) return AMIO_ERR_UNSUPPORTED;
                const auto settings = grib2_settings_for(*owner->context, variable);
                const auto ni = static_cast<std::int64_t>(variable.dimensions.back().extent);
                const auto nj = variable.dimensions.size() > 1
                                    ? static_cast<std::int64_t>(variable.dimensions[variable.dimensions.size() - 2].extent)
                                    : 1;
                amio::detail::validate_grib2_representability(settings, ni, nj);
                if (!grib_identities.insert(grib2_identity(settings)).second) return AMIO_ERR_INVALID_INPUT;
            }
            if (!schema->add_variable(std::move(variable)).ok()) return AMIO_ERR_INVALID_INPUT;
        }
        if (!schema->seal().ok()) return AMIO_ERR_INVALID_INPUT;
        return AMIO_OK;
      } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
      } catch (...) {
        return AMIO_ERR_INVALID_INPUT;
      }
    };
    local_status = build_schema();
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;
    agreed_status = compare_schema_collectively(*owner->context->context, schema->canonical());
    if (agreed_status != AMIO_OK) return agreed_status;
    const bool schema_already_registered = owner->schemas.contains(schema->digest());

    local_status = owner->steps->register_schema(schema);
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

    int schema_status = AMIO_OK;
    if (owner->context->context->rank() == 0) {
        try {
            amio::detail::SchemaDocument document;
            document.dataset_uuid = owner->uuid;
            document.digest = schema->digest();
            document.canonical_schema = schema->canonical();
            for (const auto &[name, variable] : schema->variables()) if (variable.required) document.required_fields.push_back(name);
            std::string encoded;
            schema_status = amio::detail::encode_schema_document(document, encoded);
            if (schema_status == AMIO_OK) {
                const auto path = std::string("payloads/schemas/") + schema->digest() + ".json";
                schema_status = schema_already_registered ? AMIO_OK : owner->catalog->write_payload(
                    path, encoded.data(), encoded.size(), owner->requested_sync);
            }
        } catch (...) { schema_status = AMIO_ERR_BACKEND_FAILURE; }
    }
#ifdef AMIO_HAS_MPI
    if (MPI_Bcast(&schema_status, 1, MPI_INT, 0, owner->context->context->communicator()) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
#endif
    if (schema_status != AMIO_OK) return static_cast<amio_status_t>(schema_status);

    std::unique_ptr<SchemaHandlePayload> handle_payload;
    local_status = AMIO_OK;
    try {
        handle_payload = std::make_unique<SchemaHandlePayload>();
        handle_payload->schema = schema;
        owner->schemas[schema->digest()] = schema;
    } catch (...) {
        local_status = AMIO_ERR_BACKEND_FAILURE;
    }
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

    const auto schema_token = process_handle_table().insert(HandleKind::Schema, handle_payload.get());
    local_status = schema_token == 0 ? AMIO_ERR_BACKEND_FAILURE : AMIO_OK;
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) {
        if (schema_token != 0) {
            void *unused = nullptr;
            (void)process_handle_table().release(schema_token, HandleKind::Schema, &unused);
        }
        return agreed_status;
    }
    *out_schema = reinterpret_cast<amio_schema_handle>(HandleTable::to_ptr(schema_token));
    handle_payload.release();
    return AMIO_OK;
}

AMIO_API amio_status_t amio_release_schema(amio_schema_handle schema) {
    if (schema == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto status = process_handle_table().release(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(schema))), HandleKind::Schema, &payload);
    if (status != AMIO_OK) return status;
    delete static_cast<SchemaHandlePayload *>(payload);
    return AMIO_OK;
}

static amio_status_t begin_step_impl(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor,
                                    const amio_field_partition_descriptor_t *partitions, size_t partition_count,
                                    bool partitioned, amio_step_handle *out_step) {
    if (out_step == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_step = nullptr;
    void *dataset_payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))), HandleKind::EvolutionDataset, &dataset_payload);
    if (lookup != AMIO_OK) return dataset == nullptr ? AMIO_ERR_NULL_HANDLE : lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(dataset_payload);
    if (partitioned && (!is_distributed_output(*owner) || partitions == nullptr || partition_count == 0))
        return AMIO_ERR_UNSUPPORTED;
    if (!partitioned && is_distributed_output(*owner)) return AMIO_ERR_INVALID_POLICY;
    constexpr std::size_t minimum = offsetof(amio_step_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    if (!valid_descriptor_header(descriptor, minimum) || descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0 ||
        descriptor->schema == nullptr) return AMIO_ERR_INVALID_INPUT;

    void *schema_payload = nullptr;
    const auto schema_lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(descriptor->schema))), HandleKind::Schema, &schema_payload);
    if (schema_lookup != AMIO_OK) return schema_lookup;
    const auto schema = static_cast<SchemaHandlePayload *>(schema_payload)->schema;
    if (!schema || !owner->schemas.contains(schema->digest())) return AMIO_ERR_INVALID_POLICY;
    if (partitioned) {
        std::string signature = schema->digest();
        signature.append(reinterpret_cast<const char *>(&descriptor->step_id), sizeof(descriptor->step_id));
        const auto model_time_bits = std::bit_cast<std::uint64_t>(descriptor->model_time);
        signature.append(reinterpret_cast<const char *>(&model_time_bits), sizeof(model_time_bits));
        const auto signature_status = compare_schema_collectively(*owner->context->context, signature);
        if (signature_status != AMIO_OK) return signature_status;
        const auto local_validation = validate_partition_descriptors(*schema, partitions, partition_count);
        const auto validation_status = collective_status(*owner->context->context, local_validation);
        if (validation_status != AMIO_OK) return validation_status;
    }

    auto step_payload = std::make_unique<ProducerStepPayload>();
    step_payload->dataset = owner;
    step_payload->step_id = descriptor->step_id;
    step_payload->partitioned = partitioned;
    std::lock_guard<std::mutex> lock(owner->mutex);
    amio_status_t local_begin_status = owner->open_step_handles.load() != 0 ? AMIO_ERR_INVALID_POLICY : AMIO_OK;
    if (partitioned) local_begin_status = collective_status(*owner->context->context, local_begin_status);
    if (local_begin_status != AMIO_OK) return local_begin_status;
    amio::detail::StepBeginRequest request;
    request.step_id = descriptor->step_id;
    request.model_time = descriptor->model_time;
    request.schema_digest = schema->digest();
    request.snapshot_descriptors = schema->variables().size();
    request.defer_required_field_validation = partitioned;
    request.contributions.reserve(schema->variables().size());
    for (const auto &[name, variable] : schema->variables()) {
        if (variable.source_dtype != variable.stored_dtype) return AMIO_ERR_UNSUPPORTED;
        amio::detail::ContributionDeclaration declaration;
        declaration.variable = name;
        if (!partitioned) {
            std::size_t bytes = 0;
            if (!variable_payload_size(variable, bytes) || bytes > std::numeric_limits<std::size_t>::max() - request.snapshot_bytes)
                return AMIO_ERR_INVALID_INPUT;
            request.snapshot_bytes += bytes;
            declaration.snapshot_bytes = bytes;
            for (const auto &dimension : variable.dimensions) {
                declaration.region.offsets.push_back(0);
                declaration.region.extents.push_back(dimension.extent);
            }
        } else {
            const amio_field_partition_descriptor_t *input = nullptr;
            for (size_t i = 0; i < partition_count; ++i) {
                const auto &candidate = partitions[i];
                constexpr std::size_t part_minimum = offsetof(amio_field_partition_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
                if (!valid_descriptor_header(&candidate, part_minimum) || candidate.reserved32 != 0 ||
                    candidate.reserved[0] != 0 || candidate.reserved[1] != 0)
                    return AMIO_ERR_INVALID_INPUT;
                std::string candidate_name;
                if (!copy_utf8(candidate.field, candidate.field_bytes, candidate_name)) return AMIO_ERR_INVALID_INPUT;
                if (candidate_name == name) {
                    if (input != nullptr) return AMIO_ERR_INVALID_INPUT;
                    input = &candidate;
                }
            }
            if (input == nullptr || input->rank != variable.dimensions.size() || input->rank > AMIO_MAX_RANK)
                return AMIO_ERR_INVALID_INPUT;
            bool empty = false;
            for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
                if (input->offsets[d] < 0 || input->extents[d] < 0) return AMIO_ERR_INVALID_INPUT;
                const auto offset = static_cast<std::uint64_t>(input->offsets[d]);
                const auto extent = static_cast<std::uint64_t>(input->extents[d]);
                const auto global = variable.dimensions[d].extent;
                if (offset > global || extent > global - offset) return AMIO_ERR_INVALID_INPUT;
                declaration.region.offsets.push_back(offset);
                declaration.region.extents.push_back(extent);
                empty = empty || extent == 0;
            }
            for (std::size_t d = input->rank; d < AMIO_MAX_RANK; ++d)
                if (input->offsets[d] != 0 || input->extents[d] != 0) return AMIO_ERR_INVALID_INPUT;
            declaration.region.empty = empty;
            declaration.absent = empty;
            if (!empty) {
                std::uint64_t elements = 1;
                for (const auto extent : declaration.region.extents) {
                    if (extent == 0 || extent > std::numeric_limits<std::size_t>::max() / elements)
                        return AMIO_ERR_INVALID_INPUT;
                    elements *= extent;
                }
                const auto width = dtype_size(variable.source_dtype);
                if (width == 0 || width > std::numeric_limits<std::size_t>::max() / elements)
                    return AMIO_ERR_INVALID_INPUT;
                declaration.snapshot_bytes = static_cast<std::size_t>(elements * width);
                if (declaration.snapshot_bytes > std::numeric_limits<std::size_t>::max() - request.snapshot_bytes)
                    return AMIO_ERR_INVALID_INPUT;
                request.snapshot_bytes += declaration.snapshot_bytes;
            }
            step_payload->partitions.emplace(name, declaration.region);
        }
        request.contributions.push_back(std::move(declaration));
    }
    if (partitioned && partition_count != schema->variables().size()) return AMIO_ERR_INVALID_INPUT;
    if (owner->distributed_zarr) {
        const auto shard_status = derive_distributed_zarr_shards(*owner, *step_payload, *schema,
                                                                 owner->zarr_shard_shapes);
        const auto agreed_shard_status = collective_status(*owner->context->context, shard_status);
        if (agreed_shard_status != AMIO_OK) return agreed_shard_status;
    }
    const auto local_coordinator_status = owner->steps->begin_step(request);
    const auto begin_status = partitioned ? collective_status(*owner->context->context, local_coordinator_status)
                                          : local_coordinator_status;
    if (begin_status != AMIO_OK && local_coordinator_status == AMIO_OK) (void)owner->steps->cancel_step();
    if (begin_status != AMIO_OK) return begin_status;
    const auto token = process_handle_table().insert(HandleKind::ProducerStep, step_payload.get());
    amio_status_t token_status = token == 0 ? AMIO_ERR_BACKEND_FAILURE : AMIO_OK;
    if (partitioned) token_status = collective_status(*owner->context->context, token_status);
    if (token_status != AMIO_OK) {
        if (token != 0) {
            void *unused = nullptr;
            (void)process_handle_table().release(token, HandleKind::ProducerStep, &unused);
        }
        (void)owner->steps->cancel_step();
        return token_status;
    }
    owner->context->open_step_handles.fetch_add(1);
    owner->open_step_handles.fetch_add(1);
    *out_step = reinterpret_cast<amio_step_handle>(HandleTable::to_ptr(token));
    step_payload.release();
    return AMIO_OK;
}

AMIO_API amio_status_t amio_begin_step(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor,
                                       amio_step_handle *out_step) {
    return begin_step_impl(dataset, descriptor, nullptr, 0, false, out_step);
}

AMIO_API amio_status_t amio_begin_step_partitioned(amio_dataset_handle dataset,
                                                   const amio_step_descriptor_t *descriptor,
                                                   const amio_field_partition_descriptor_t *partitions,
                                                   size_t partition_count, amio_step_handle *out_step) {
    return begin_step_impl(dataset, descriptor, partitions, partition_count, true, out_step);
}

AMIO_API amio_status_t amio_put_field(amio_step_handle step, const char *field, size_t field_bytes,
                                      const void *data, size_t data_bytes, amio_operation_handle *out_operation) {
    if (out_operation == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_operation = nullptr;
    if (step == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *step_payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(step)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::ProducerStep, &step_payload);
    if (lookup != AMIO_OK) return lookup;
    auto *producer_step = static_cast<ProducerStepPayload *>(step_payload);
    if (producer_step->ended) return AMIO_ERR_INVALID_POLICY;
    std::string field_name;
    if (!copy_utf8(field, field_bytes, field_name)) return AMIO_ERR_INVALID_INPUT;
    auto *owner = producer_step->dataset;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const auto result = owner->steps->put(field_name, data, data_bytes);
    if (result.status != AMIO_OK) return result.status;
    auto operation = owner->steps->operation_record(result.handle);
    if (!operation) {
        (void)owner->steps->poison_dataset(AMIO_ERR_BACKEND_FAILURE);
        return AMIO_ERR_BACKEND_FAILURE;
    }
    auto operation_payload = std::make_unique<OperationHandlePayload>();
    operation_payload->operation = std::move(operation);
    const auto operation_token = process_handle_table().insert(HandleKind::Operation, operation_payload.get());
    if (operation_token == 0) {
        (void)owner->steps->poison_dataset(AMIO_ERR_BACKEND_FAILURE);
        return AMIO_ERR_BACKEND_FAILURE;
    }
    *out_operation = reinterpret_cast<amio_operation_handle>(HandleTable::to_ptr(operation_token));
    operation_payload.release();
    return AMIO_OK;
}

AMIO_API amio_status_t amio_cancel_step(amio_step_handle step) {
    if (step == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *step_payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(step)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::ProducerStep, &step_payload);
    if (lookup != AMIO_OK) return lookup;
    auto *producer_step = static_cast<ProducerStepPayload *>(step_payload);
    auto *owner = producer_step->dataset;
    std::lock_guard<std::mutex> lock(owner->mutex);
#ifdef AMIO_HAS_MPI
    if (is_distributed_output(*owner)) {
        const int locally_cancellable = !producer_step->ended && owner->steps->state() == amio::detail::StepState::Open ? 1 : 0;
        int all_cancellable = 0;
        if (MPI_Allreduce(&locally_cancellable, &all_cancellable, 1, MPI_INT, MPI_MIN,
                          owner->context->context->communicator()) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        if (!all_cancellable) return AMIO_ERR_INVALID_POLICY;
    } else
#endif
    if (producer_step->ended) return AMIO_ERR_INVALID_POLICY;
    const auto status = owner->steps->cancel_step();
    const auto agreed = is_distributed_output(*owner) ? collective_status(*owner->context->context, status) : status;
    if (agreed == AMIO_OK) producer_step->ended = true;
    return agreed;
}

AMIO_API amio_status_t amio_end_step(amio_step_handle step) {
    if (step == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *step_payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(step)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::ProducerStep, &step_payload);
    if (lookup != AMIO_OK) return lookup;
    auto *producer_step = static_cast<ProducerStepPayload *>(step_payload);
    auto *owner = producer_step->dataset;
    std::lock_guard<std::mutex> lock(owner->mutex);
    auto &coordinator = *owner->steps;
#ifdef AMIO_HAS_MPI
    if (producer_step->partitioned) {
        const auto state = coordinator.state();
        const int locally_valid = !producer_step->ended &&
                                          (state == amio::detail::StepState::Open || state == amio::detail::StepState::Reconciling)
                                      ? 1
                                      : 0;
        int all_valid = 0;
        if (MPI_Allreduce(&locally_valid, &all_valid, 1, MPI_INT, MPI_MIN,
                          owner->context->context->communicator()) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        if (!all_valid) return AMIO_ERR_INVALID_POLICY;
    } else
#endif
    if (producer_step->ended) return AMIO_ERR_INVALID_POLICY;

    if (coordinator.state() == amio::detail::StepState::Reconciling) {
        amio::detail::PublicationResult reconciled;
        if (owner->context->context->rank() == 0)
            reconciled = owner->catalog->reconcile(owner->pending_commit);
        const auto broadcast_status = broadcast_publication(*owner->context->context, reconciled);
        if (broadcast_status != AMIO_OK) return broadcast_status;
        if (reconciled.state != amio::detail::PublicationState::Complete || reconciled.status != AMIO_OK)
            return AMIO_ERR_PUBLICATION_PENDING;
        const auto status = coordinator.mark_publication_visible(owner->requested_sync != amio::detail::PosixSyncLevel::Visibility);
        if (status != AMIO_OK) return status;
        owner->previous_commit = owner->pending_commit;
        owner->pending_commit.clear();
        producer_step->ended = true;
        return AMIO_OK;
    }

    auto status = coordinator.end_step();
    if (producer_step->partitioned) {
        status = collective_status(*owner->context->context, status);
        if (status != AMIO_OK) {
            // A local end-step validation failure must become terminal on
            // every rank. Some participants may already have sealed their
            // local coordinator before the collective error vote arrives.
            (void)coordinator.poison_dataset(status);
            return status;
        }
        const auto schema_it_for_coverage = owner->schemas.find(coordinator.active_schema_digest());
        if (schema_it_for_coverage == owner->schemas.end()) return AMIO_ERR_SCHEMA_MISMATCH;
        std::map<std::string, bool, std::less<>> globally_present;
        status = validate_distributed_partitions(*owner, *producer_step, *schema_it_for_coverage->second, globally_present);
        status = collective_status(*owner->context->context, status);
        if (status != AMIO_OK) {
            (void)coordinator.poison_dataset(status);
            return status;
        }
        owner->global_present_fields = std::move(globally_present);
    }
    if (status != AMIO_OK) return status;
    const auto schema_it = owner->schemas.find(coordinator.active_schema_digest());
    if (schema_it == owner->schemas.end()) {
        (void)coordinator.poison_dataset(AMIO_ERR_SCHEMA_MISMATCH);
        return AMIO_ERR_SCHEMA_MISMATCH;
    }
    const auto relative_payload = step_payload_name(coordinator.active_step_id());
    const auto payload_path = (std::filesystem::path(owner->location) / relative_payload).string();
    std::map<std::string, std::string, std::less<>> zarr_payload_paths;
    amio::detail::Grib2StepPayload grib_payload;
    amio_status_t payload_status = AMIO_OK;
    try {
        if (owner->format == AMIO_FORMAT_NETCDF4) {
            if (producer_step->partitioned) {
                amio_status_t directory_status = AMIO_OK;
                if (owner->context->context->rank() == 0) {
                    try { std::filesystem::create_directories(std::filesystem::path(payload_path).parent_path()); }
                    catch (...) { directory_status = AMIO_ERR_BACKEND_FAILURE; }
                }
                payload_status = collective_status(*owner->context->context, directory_status);
                if (payload_status != AMIO_OK) throw std::runtime_error("failed to agree on NetCDF payload directory creation");
            } else std::filesystem::create_directories(std::filesystem::path(payload_path).parent_path());
            std::map<std::string, amio::detail::NetCDFStepFieldView, std::less<>> fields;
            for (const auto &[name, contribution] : coordinator.contributions())
                fields.emplace(name, amio::detail::NetCDFStepFieldView{contribution.region, contribution.bytes.data()});
            std::int64_t fortran_communicator = 0;
#ifdef AMIO_HAS_MPI
            fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(owner->context->context->communicator()));
#endif
            amio::detail::write_netcdf_step_payload(payload_path, *schema_it->second, coordinator.active_model_time(),
                                                    fields, fortran_communicator);
        } else if (owner->format == AMIO_FORMAT_ZARR3) {
#ifdef AMIO_HAS_TENSORSTORE
            std::string error;
            if (producer_step->partitioned) {
                if (owner->context->context->rank() == 0 &&
                    !amio::detail::create_zarr_step(owner->location, owner->uuid, coordinator.active_step_id(),
                                                    *schema_it->second, owner->zarr_shard_shapes, error))
                    payload_status = AMIO_ERR_BACKEND_FAILURE;
                payload_status = collective_status(*owner->context->context, payload_status);
                if (payload_status != AMIO_OK) throw std::runtime_error(error.empty() ? "TensorStore Zarr metadata creation failed" : error);
                std::map<std::string, amio::detail::ZarrRegionView, std::less<>> regions;
                for (const auto &[name, contribution] : coordinator.contributions()) {
                    amio::detail::ZarrRegionView region;
                    region.offsets = contribution.region.offsets;
                    region.extents = contribution.region.extents;
                    region.data = std::span<const std::byte>(contribution.bytes.data(), contribution.bytes.size());
                    regions.emplace(name, std::move(region));
                }
                if (!amio::detail::write_zarr_step_regions(owner->location, owner->uuid, coordinator.active_step_id(),
                                                           *schema_it->second, owner->zarr_shard_shapes, regions, error))
                    throw std::runtime_error(error.empty() ? "TensorStore Zarr region write failed" : error);
            } else {
                std::map<std::string, std::span<const std::byte>, std::less<>> fields;
                for (const auto &[name, contribution] : coordinator.contributions())
                    fields.emplace(name, std::span<const std::byte>(contribution.bytes.data(), contribution.bytes.size()));
                if (!amio::detail::write_zarr_step(owner->location, owner->uuid, coordinator.active_step_id(),
                                                   *schema_it->second, fields, error))
                    throw std::runtime_error(error.empty() ? "TensorStore Zarr step write failed" : error);
            }
            amio::detail::ZarrStepLayout layout;
            if (!amio::detail::make_zarr_step_layout(owner->location, coordinator.active_step_id(),
                                                     *schema_it->second, layout, error))
                throw std::runtime_error(error.empty() ? "TensorStore Zarr step path creation failed" : error);
            std::size_t path_index = 0;
            for (const auto &[name, variable] : schema_it->second->variables()) {
                (void)variable;
                zarr_payload_paths.emplace(name,
                    std::filesystem::path(layout.array_paths.at(path_index++)).lexically_relative(owner->location).generic_string());
            }
#else
            throw std::runtime_error("TensorStore Zarr step I/O is not enabled in this build");
#endif
        } else if (owner->format == AMIO_FORMAT_GRIB2) {
#ifdef AMIO_HAS_G2C
            for (const auto &[name, contribution] : coordinator.contributions()) {
                const auto variable_it = schema_it->second->variables().find(name);
                if (variable_it == schema_it->second->variables().end()) throw std::runtime_error("GRIB2 step field is absent from its schema");
                const auto settings = grib2_settings_for(*owner->context, variable_it->second);
                amio_shape_t shape{};
                shape.rank = static_cast<std::int32_t>(variable_it->second.dimensions.size());
                for (std::size_t d = 0; d < variable_it->second.dimensions.size(); ++d) {
                    shape.extents[d] = static_cast<std::int64_t>(variable_it->second.dimensions[d].extent);
                }
                amio::detail::VarMeta meta{};
                meta.name = name;
                meta.dtype = variable_it->second.source_dtype;
                meta.shape = shape;
                amio::detail::StagingBuffer source{};
                source.data = const_cast<std::byte *>(contribution.bytes.data());
                source.capacity_bytes = contribution.bytes.size();
                source.used_bytes = contribution.bytes.size();
                const auto message = owner->grib2_driver->encode_step_message(
                    source, meta, static_cast<std::size_t>(owner->payload_budget_bytes),
                    static_cast<std::size_t>(owner->payload_budget_bytes), settings);
                const std::vector<std::vector<unsigned char>> messages{message};
                amio::detail::append_grib2_field_messages(grib_payload, name,
                    static_cast<std::uint32_t>(owner->context->context->rank()), messages,
                    owner->payload_budget_bytes, true);
            }
            std::filesystem::create_directories(std::filesystem::path(payload_path).parent_path());
            std::string error;
            if (!amio::detail::write_grib2_step_payload(payload_path, grib_payload, error))
                throw std::runtime_error(error.empty() ? "GRIB2 step payload write failed" : error);
#else
            throw std::runtime_error("GRIB2 support is not enabled in this build");
#endif
        } else {
            throw std::runtime_error("Requested evolution format is not supported by the direct writer");
        }
    } catch (...) { payload_status = AMIO_ERR_BACKEND_FAILURE; }
    if (producer_step->partitioned) payload_status = collective_status(*owner->context->context, payload_status);
    if (payload_status != AMIO_OK) {
        (void)coordinator.poison_dataset(payload_status);
        return payload_status;
    }
    status = coordinator.mark_payload_complete();
    if (status != AMIO_OK) return status;

    amio::detail::CompletionRecord record;
    record.dataset_uuid = owner->uuid;
    record.step_id = coordinator.active_step_id();
    record.schema_digest = coordinator.active_schema_digest();
    record.previous_commit = owner->previous_commit;
    std::uint64_t file_bytes = 0;
    if (owner->format == AMIO_FORMAT_NETCDF4) {
        if (!producer_step->partitioned || owner->context->context->rank() == 0) {
            try { file_bytes = std::filesystem::file_size(payload_path); }
            catch (...) { status = AMIO_ERR_BACKEND_FAILURE; }
        }
        if (producer_step->partitioned) {
            status = collective_status(*owner->context->context, status);
            if (status != AMIO_OK) {
                (void)coordinator.poison_dataset(status);
                return status;
            }
#ifdef AMIO_HAS_MPI
            if (MPI_Bcast(&file_bytes, 1, MPI_UINT64_T, 0, owner->context->context->communicator()) != MPI_SUCCESS)
                return AMIO_ERR_COMM_SPLIT_FAILED;
#endif
        }
    }
    if (owner->format == AMIO_FORMAT_GRIB2) {
        for (const auto &[name, variable] : schema_it->second->variables())
            if (variable.required) record.required_fields.push_back(name);
        status = amio::detail::append_grib2_catalog_payloads(grib_payload, relative_payload, record);
        if (status != AMIO_OK) {
            (void)coordinator.poison_dataset(status);
            return status;
        }
    } else {
        for (const auto &[name, variable] : schema_it->second->variables()) {
            if (variable.required) record.required_fields.push_back(name);
            const auto contribution = coordinator.contributions().find(name);
            const bool is_present = producer_step->partitioned
                                        ? owner->global_present_fields[name]
                                        : contribution != coordinator.contributions().end();
            if (!is_present) continue;
            record.present_fields.push_back(name);
            amio::detail::PayloadDescriptor payload;
            payload.field = name;
            payload.relative_path = owner->format == AMIO_FORMAT_NETCDF4 ? relative_payload : zarr_payload_paths.at(name);
            if (owner->format == AMIO_FORMAT_NETCDF4) payload.bytes = file_bytes;
            else if (producer_step->partitioned) {
                std::size_t full_field_bytes = 0;
                if (!variable_payload_size(variable, full_field_bytes)) {
                    (void)coordinator.poison_dataset(AMIO_ERR_INVALID_INPUT);
                    return AMIO_ERR_INVALID_INPUT;
                }
                payload.bytes = full_field_bytes;
            } else payload.bytes = contribution->second.bytes.size();
            payload.array_container = owner->format == AMIO_FORMAT_ZARR3;
            payload.producer = 0;
            if (producer_step->partitioned) {
                payload.offsets.assign(variable.dimensions.size(), 0);
                for (const auto &dimension : variable.dimensions) payload.extents.push_back(dimension.extent);
            } else {
                payload.offsets = contribution->second.region.offsets;
                payload.extents = contribution->second.region.extents;
            }
            record.payloads.push_back(std::move(payload));
        }
    }
    status = coordinator.mark_publishing();
    if (status != AMIO_OK) return status;
    amio::detail::PublicationResult publication;
    if (owner->context->context->rank() == 0)
        publication = owner->catalog->publish(std::move(record), owner->requested_sync);
    status = broadcast_publication(*owner->context->context, publication);
    if (status != AMIO_OK) return status;
    if (publication.state == amio::detail::PublicationState::Complete && publication.status == AMIO_OK) {
        status = coordinator.mark_publication_visible(owner->requested_sync != amio::detail::PosixSyncLevel::Visibility);
        if (status != AMIO_OK) return status;
        owner->previous_commit = publication.commit_path;
        producer_step->ended = true;
        return AMIO_OK;
    }
    if (publication.state == amio::detail::PublicationState::Reconciling) {
        status = coordinator.mark_reconciling();
        if (status != AMIO_OK) return status;
        owner->pending_commit = publication.commit_path;
        return AMIO_ERR_PUBLICATION_PENDING;
    }
    (void)coordinator.fail_publication(publication.status);
    return publication.status == AMIO_OK ? AMIO_ERR_BACKEND_FAILURE : publication.status;
}

AMIO_API amio_status_t amio_query_operation(amio_operation_handle operation, amio_operation_status_t *out_status) {
    if (out_status == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_status = {};
    if (operation == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(operation))), HandleKind::Operation, &payload);
    if (lookup != AMIO_OK) return lookup;
    const auto snapshot = static_cast<OperationHandlePayload *>(payload)->operation->snapshot();
    out_status->struct_size = sizeof(*out_status);
    out_status->version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    out_status->outcome = static_cast<int32_t>(snapshot.outcome);
    out_status->terminal_status = snapshot.error;
    if (snapshot.source_released) out_status->milestones |= AMIO_OPERATION_SOURCE_RELEASED;
    if (snapshot.payload_complete) out_status->milestones |= AMIO_OPERATION_BACKEND_COMPLETE;
    if (snapshot.reader_visible) out_status->milestones |= AMIO_OPERATION_READER_VISIBLE;
    if (snapshot.storage_synced) out_status->milestones |= AMIO_OPERATION_DURABLE;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_release_operation(amio_operation_handle operation) {
    if (operation == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto status = process_handle_table().release(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(operation))), HandleKind::Operation, &payload);
    if (status != AMIO_OK) return status;
    auto *handle = static_cast<OperationHandlePayload *>(payload);
    (void)handle->operation->release_observer();
    delete handle;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_wait_operation(amio_operation_handle operation, uint32_t milestone, uint64_t timeout_ms) {
    if (operation == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(operation))), HandleKind::Operation, &payload);
    if (lookup != AMIO_OK) return lookup;
    amio::detail::OperationMilestone target;
    switch (milestone) {
        case AMIO_OPERATION_SOURCE_RELEASED: target = amio::detail::OperationMilestone::SourceReleased; break;
        case AMIO_OPERATION_BACKEND_COMPLETE: target = amio::detail::OperationMilestone::PayloadComplete; break;
        case AMIO_OPERATION_READER_VISIBLE: target = amio::detail::OperationMilestone::ReaderVisible; break;
        case AMIO_OPERATION_DURABLE: target = amio::detail::OperationMilestone::StorageSynced; break;
        default: return AMIO_ERR_INVALID_INPUT;
    }
    if (timeout_ms > static_cast<std::uint64_t>(std::numeric_limits<std::chrono::milliseconds::rep>::max()))
        return AMIO_ERR_INVALID_INPUT;
    const auto handle = static_cast<OperationHandlePayload *>(payload);
    const auto result = handle->operation->wait_for(target, std::chrono::milliseconds(timeout_ms));
    if (result == amio::detail::WaitOutcome::Reached) return AMIO_OK;
    if (result == amio::detail::WaitOutcome::Timeout) return AMIO_ERR_TIMEOUT;
    const auto snapshot = handle->operation->snapshot();
    if (snapshot.outcome == amio::detail::OperationOutcome::Succeeded) return AMIO_ERR_UNSUPPORTED;
    return snapshot.error == AMIO_OK ? AMIO_ERR_STEP_FAILED : snapshot.error;
}

AMIO_API amio_status_t amio_list_steps(amio_dataset_handle dataset, const amio_catalog_cursor_t *cursor,
                                       uint32_t max_results, amio_step_summary_t *results, size_t result_capacity,
                                       size_t *out_result_count, amio_catalog_cursor_t *out_next_cursor, int *out_has_more) {
    if (out_result_count != nullptr) *out_result_count = 0;
    if (out_has_more != nullptr) *out_has_more = 0;
    if (out_next_cursor != nullptr) {
        *out_next_cursor = {};
        out_next_cursor->struct_size = sizeof(*out_next_cursor);
        out_next_cursor->version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    }
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    if (out_result_count == nullptr || out_next_cursor == nullptr || out_has_more == nullptr || max_results == 0 ||
        max_results > 10000 || (results == nullptr && result_capacity != 0)) return AMIO_ERR_INVALID_INPUT;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))), HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    amio::detail::CatalogCursor internal;
    if (cursor != nullptr) {
        constexpr std::size_t minimum = offsetof(amio_catalog_cursor_t, reserved) + 2 * sizeof(std::uint64_t);
        if (!valid_descriptor_header(cursor, minimum) || cursor->reserved[0] != 0 || cursor->reserved[1] != 0 ||
            !copy_fixed_text(cursor->dataset_uuid, sizeof(cursor->dataset_uuid), internal.dataset_uuid) ||
            !copy_fixed_text(cursor->generation, sizeof(cursor->generation), internal.generation) ||
            !copy_fixed_text(cursor->next_commit, sizeof(cursor->next_commit), internal.next_commit)) return AMIO_ERR_INVALID_INPUT;
        if (!internal.dataset_uuid.empty() && internal.dataset_uuid != owner->uuid) return AMIO_ERR_INVALID_INPUT;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    std::error_code head_error;
    if (!std::filesystem::exists(std::filesystem::path(owner->location) / "head.json", head_error) && !head_error) {
        if (!internal.next_commit.empty() || !internal.generation.empty()) return AMIO_ERR_INVALID_INPUT;
        if (owner->uuid.size() >= sizeof(out_next_cursor->dataset_uuid)) return AMIO_ERR_BUFFER_TOO_SMALL;
        std::memcpy(out_next_cursor->dataset_uuid, owner->uuid.data(), owner->uuid.size());
        return AMIO_OK;
    }
    if (head_error) return AMIO_ERR_BACKEND_FAILURE;
    amio::detail::CollectionReader reader(owner->location, owner->uuid);
    auto page = reader.list_steps(internal, max_results);
    if (page.status != AMIO_OK) return page.status;
    *out_result_count = page.steps.size();
    *out_has_more = page.has_more ? 1 : 0;
    if (results != nullptr && result_capacity < page.steps.size()) {
        *out_result_count = 0;
        return AMIO_ERR_BUFFER_TOO_SMALL;
    }
    if (results != nullptr) {
        for (std::size_t i = 0; i < page.steps.size(); ++i)
            if (!fill_step_summary(page.steps[i], &results[i])) return AMIO_ERR_BACKEND_FAILURE;
    }
    const auto &next = page.next;
    if (next.dataset_uuid.size() >= sizeof(out_next_cursor->dataset_uuid) ||
        next.generation.size() >= sizeof(out_next_cursor->generation) ||
        next.next_commit.size() >= sizeof(out_next_cursor->next_commit)) return AMIO_ERR_BUFFER_TOO_SMALL;
    std::memcpy(out_next_cursor->dataset_uuid, next.dataset_uuid.data(), next.dataset_uuid.size());
    std::memcpy(out_next_cursor->generation, next.generation.data(), next.generation.size());
    std::memcpy(out_next_cursor->next_commit, next.next_commit.data(), next.next_commit.size());
    return AMIO_OK;
}

AMIO_API amio_status_t amio_describe_step(amio_dataset_handle dataset, uint64_t step_id, amio_step_summary_t *out_summary) {
    if (out_summary == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_summary = {};
    out_summary->struct_size = sizeof(*out_summary);
    out_summary->version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))), HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    std::lock_guard<std::mutex> lock(owner->mutex);
    amio::detail::CollectionReader reader(owner->location, owner->uuid);
    amio::detail::CatalogStep step;
    const auto status = reader.describe_step(step_id, step);
    if (status != AMIO_OK) return status;
    return fill_step_summary(step, out_summary) ? AMIO_OK : AMIO_ERR_BACKEND_FAILURE;
}

AMIO_API amio_status_t amio_read_step_field(amio_dataset_handle dataset, uint64_t step_id, const char *field,
                                            size_t field_bytes, const amio_step_selection_t *selection,
                                            void *data, size_t data_capacity, size_t *out_required_bytes) {
    if (out_required_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_required_bytes = 0;
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    std::string field_name;
    if (!copy_utf8(field, field_bytes, field_name) || (data == nullptr && data_capacity != 0)) return AMIO_ERR_INVALID_INPUT;
    if (selection != nullptr) {
        constexpr std::size_t minimum = offsetof(amio_step_selection_t, reserved) + 2 * sizeof(std::uint64_t);
        if (!valid_descriptor_header(selection, minimum) || selection->reserved32 != 0 ||
            selection->reserved[0] != 0 || selection->reserved[1] != 0 || selection->rank == 0 || selection->rank > AMIO_MAX_RANK)
            return AMIO_ERR_INVALID_INPUT;
    }
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))), HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    std::lock_guard<std::mutex> lock(owner->mutex);
    amio::detail::CollectionReader reader(owner->location, owner->uuid);
    amio::detail::CatalogStep step;
    auto status = reader.describe_step(step_id, step);
    if (status != AMIO_OK) return status;
    const auto schema_it = owner->schemas.find(step.completion.schema_digest);
    if (schema_it == owner->schemas.end()) return AMIO_ERR_SCHEMA_MISMATCH;
    const auto variable_it = schema_it->second->variables().find(field_name);
    if (variable_it == schema_it->second->variables().end()) return AMIO_ERR_INVALID_INPUT;
    if (std::find(step.completion.present_fields.begin(), step.completion.present_fields.end(), field_name) ==
        step.completion.present_fields.end()) return AMIO_ERR_FIELD_NOT_PRESENT;
    const auto payload_it = std::find_if(step.completion.payloads.begin(), step.completion.payloads.end(),
                                         [&](const auto &entry) { return entry.field == field_name; });
    if (payload_it == step.completion.payloads.end()) return AMIO_ERR_BACKEND_FAILURE;
    amio::detail::RegionSelection region;
    std::size_t elements = 1;
    for (std::size_t d = 0; d < variable_it->second.dimensions.size(); ++d) {
        const auto global = variable_it->second.dimensions[d].extent;
        std::uint64_t offset = 0, extent = global;
        if (selection != nullptr) {
            if (selection->rank != variable_it->second.dimensions.size()) return AMIO_ERR_INVALID_INPUT;
            if (selection->offsets[d] < 0 || selection->extents[d] <= 0 || selection->strides[d] <= 0) return AMIO_ERR_INVALID_INPUT;
            if (selection->strides[d] != 1) return AMIO_ERR_UNSUPPORTED;
            offset = static_cast<std::uint64_t>(selection->offsets[d]);
            extent = static_cast<std::uint64_t>(selection->extents[d]);
            if (offset > global || extent > global - offset) return AMIO_ERR_INVALID_INPUT;
        }
        if (extent > std::numeric_limits<std::size_t>::max() / elements) return AMIO_ERR_INVALID_INPUT;
        elements *= static_cast<std::size_t>(extent);
        region.offsets.push_back(offset);
        region.extents.push_back(extent);
    }
    if (selection != nullptr) {
        for (std::size_t d = selection->rank; d < AMIO_MAX_RANK; ++d)
            if (selection->offsets[d] != 0 || selection->extents[d] != 0 || selection->strides[d] != 0) return AMIO_ERR_INVALID_INPUT;
    }
    const auto width = dtype_size(variable_it->second.stored_dtype);
    if (width == 0 || width > std::numeric_limits<std::size_t>::max() / elements) return AMIO_ERR_INVALID_INPUT;
    const auto required = width * elements;
    *out_required_bytes = required;
    if (data == nullptr && data_capacity == 0) return AMIO_OK;
    if (data_capacity < required) return AMIO_ERR_BUFFER_TOO_SMALL;

    amio::detail::StepReadKey cache_key;
    cache_key.dataset_uuid = owner->uuid;
    cache_key.step_id = step_id;
    cache_key.schema_digest = step.completion.schema_digest;
    cache_key.field = field_name;
    if (selection != nullptr) {
        amio_bbox_t bbox{};
        bbox.rank = static_cast<std::int32_t>(selection->rank);
        for (std::size_t d = 0; d < selection->rank; ++d) {
            bbox.offsets[d] = selection->offsets[d];
            bbox.extents[d] = selection->extents[d];
            bbox.strides[d] = selection->strides[d];
        }
        cache_key.selection = bbox;
    }
    amio::detail::StepReadMetadata cache_metadata;
    cache_metadata.dataset_uuid = owner->uuid;
    cache_metadata.step_id = step_id;
    cache_metadata.schema_digest = step.completion.schema_digest;
    cache_metadata.field = field_name;
    cache_metadata.field_present = true;
    cache_metadata.payload_bytes = required;
    for (const auto &dimension : variable_it->second.dimensions)
        cache_metadata.global_extents.push_back(dimension.extent);
    std::vector<std::byte> cached;
    const auto cache_status = owner->read_cache.load(cache_key, cache_metadata, cached);
    if (cache_status == AMIO_OK) {
        if (cached.size() != required) return AMIO_ERR_SCHEMA_MISMATCH;
        std::memcpy(data, cached.data(), required);
        return AMIO_OK;
    }
    if (cache_status != AMIO_ERR_BACKEND_FAILURE) return cache_status;

    const auto payload_path = (std::filesystem::path(owner->location) / payload_it->relative_path).string();
    try {
        if (owner->format == AMIO_FORMAT_NETCDF4) {
            amio::detail::NetCDF_Driver::read_step_region(payload_path, field_name, region, data);
        } else if (owner->format == AMIO_FORMAT_ZARR3) {
#ifdef AMIO_HAS_TENSORSTORE
            std::vector<std::byte> selected_field;
            std::string error;
            if (!amio::detail::read_zarr_step_field(owner->location, owner->uuid, step_id, *schema_it->second,
                                                    field_name, region.offsets, region.extents, selected_field, error))
                return AMIO_ERR_BACKEND_FAILURE;
            if (selected_field.size() != required) return AMIO_ERR_SCHEMA_MISMATCH;
            std::memcpy(data, selected_field.data(), required);
#else
            return AMIO_ERR_UNSUPPORTED;
#endif
        } else if (owner->format == AMIO_FORMAT_GRIB2) {
#ifdef AMIO_HAS_G2C
            const auto settings = grib2_settings_for(*owner->context, variable_it->second);
            const auto identity = grib2_identity(settings);
            amio::detail::GRIB2_Driver reader_driver;
            reader_driver.open_read(conf::Config::from_string("path: " + payload_path + "\n"));
            amio::detail::VarMeta meta{};
            meta.name = identity;
            meta.dtype = AMIO_DTYPE_F32;
            meta.shape.rank = static_cast<std::int32_t>(variable_it->second.dimensions.size());
            for (std::size_t d = 0; d < variable_it->second.dimensions.size(); ++d) {
                meta.shape.extents[d] = static_cast<std::int64_t>(variable_it->second.dimensions[d].extent);
            }
            std::optional<amio::detail::BoundingBox> bbox;
            if (selection != nullptr) {
                amio::detail::BoundingBox selected{};
                selected.rank = static_cast<std::int32_t>(selection->rank);
                for (std::size_t d = 0; d < selection->rank; ++d) {
                    selected.offsets[d] = selection->offsets[d];
                    selected.extents[d] = selection->extents[d];
                    selected.strides[d] = selection->strides[d];
                }
                bbox = selected;
            }
            amio::detail::StagingBuffer destination{};
            destination.data = static_cast<std::byte *>(data);
            destination.capacity_bytes = data_capacity;
            destination.used_bytes = 0;
            reader_driver.read(destination, meta, 0, bbox);
            reader_driver.close();
#else
            return AMIO_ERR_UNSUPPORTED;
#endif
        } else {
            return AMIO_ERR_UNSUPPORTED;
        }
    } catch (...) { return AMIO_ERR_BACKEND_FAILURE; }
    // Cache admission is optional: oversized reads still succeed, while the
    // queue enforces its byte ceiling before allocating a cache copy.
    (void)owner->read_cache.store(cache_key, cache_metadata, data, required);
    return AMIO_OK;
}

AMIO_API amio_status_t amio_validate_dataset(amio_dataset_handle dataset, const amio_validation_request_t *request,
                                              amio_validation_result_t *results, size_t result_capacity,
                                              size_t *out_result_count, uint64_t *out_next_cursor) {
    if (out_result_count != nullptr) *out_result_count = 0;
    if (out_next_cursor != nullptr) *out_next_cursor = 0;
    if (!valid_request(request) || out_result_count == nullptr || out_next_cursor == nullptr ||
        (results == nullptr && result_capacity != 0) || (results != nullptr && result_capacity == 0)) return AMIO_ERR_INVALID_INPUT;
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;

    void *payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset)));
    auto lookup = process_handle_table().lookup(token, HandleKind::Dataset, &payload);
    bool evolution_dataset = false;
    if (lookup != AMIO_OK) {
        lookup = process_handle_table().lookup(token, HandleKind::EvolutionDataset, &payload);
        evolution_dataset = lookup == AMIO_OK;
    }
    if (lookup != AMIO_OK) return lookup;

    const auto total = static_cast<std::size_t>(std::popcount(request->check_mask));
    if (request->cursor > total) return AMIO_ERR_INVALID_INPUT;
    const auto requested = std::min<std::size_t>(request->max_results, total - static_cast<std::size_t>(request->cursor));
    *out_result_count = requested;
    *out_next_cursor = request->cursor + requested < total ? request->cursor + requested : 0;
    if (results == nullptr && result_capacity == 0) return AMIO_OK;
    if (result_capacity < requested) {
        *out_result_count = 0;
        return AMIO_ERR_BUFFER_TOO_SMALL;
    }
    bool structural_pass = false;
    amio_status_t structural_error = AMIO_ERR_UNSUPPORTED;
    if (evolution_dataset) {
        auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
        std::lock_guard<std::mutex> lock(owner->mutex);
        amio::detail::CollectionReader reader(owner->location, owner->uuid);
        amio::detail::CatalogCursor cursor;
        structural_pass = true;
        bool any_step = false;
        for (;;) {
            auto page = reader.list_steps(cursor, 64);
            if (page.status == AMIO_ERR_STEP_FAILED && !any_step) {
                std::error_code ec;
                if (!std::filesystem::exists(std::filesystem::path(owner->location) / "head.json", ec) && !ec) break;
            }
            if (page.status != AMIO_OK) {
                structural_pass = false;
                structural_error = page.status;
                break;
            }
            any_step = any_step || !page.steps.empty();
            if (!page.has_more) break;
            cursor = std::move(page.next);
        }
        structural_error = structural_pass ? AMIO_OK : structural_error;
    }
    std::size_t index = 0;
    std::size_t skipped = 0;
    for (std::int32_t check = AMIO_VALIDATE_COMPLETION_CHAIN; check <= AMIO_VALIDATE_ORPHAN_OBJECTS; ++check) {
        if ((request->check_mask & (UINT64_C(1) << (check - 1))) == 0) continue;
        if (skipped++ < request->cursor) continue;
        if (index == requested) break;
        results[index] = {};
        results[index].struct_size = sizeof(amio_validation_result_t);
        results[index].version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        results[index].check = check;
        if (!evolution_dataset || check == AMIO_VALIDATE_INTEGRITY || check == AMIO_VALIDATE_ORPHAN_OBJECTS) {
            results[index].status = AMIO_VALIDATION_UNSUPPORTED;
        } else {
            results[index].status = structural_pass ? AMIO_VALIDATION_PASS : AMIO_VALIDATION_FAIL;
            results[index].detail_code = structural_pass ? 0 : structural_error;
        }
        ++index;
    }
    return AMIO_OK;
}

AMIO_API amio_status_t amio_release_step_handle(amio_step_handle step) {
    if (step == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(step)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::ProducerStep, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *producer_step = static_cast<ProducerStepPayload *>(payload);
    auto *owner = producer_step->dataset;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const bool failed_terminal = owner->steps->state() == amio::detail::StepState::Failed &&
                                 owner->steps->active_step_id() == producer_step->step_id;
    if (!producer_step->ended && !failed_terminal) {
        // The synchronous producer handle owns the only route to seal this
        // step. Reject an early release while work is active; failed terminal
        // steps can be released after their accepted operations are failed.
        return AMIO_ERR_INVALID_POLICY;
    }
    const auto status = process_handle_table().release(token, HandleKind::ProducerStep, &payload);
    if (status != AMIO_OK) return status;
    producer_step = static_cast<ProducerStepPayload *>(payload);
    producer_step->dataset->context->open_step_handles.fetch_sub(1);
    producer_step->dataset->open_step_handles.fetch_sub(1);
    delete producer_step;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_release_step_lease(amio_step_lease_handle lease) {
    if (lease == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(lease)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::ConsumerLease, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *consumer_lease = static_cast<ConsumerLeasePayload *>(payload);
    const auto status = process_handle_table().release(token, HandleKind::ConsumerLease, &payload);
    if (status != AMIO_OK) return status;
    consumer_lease = static_cast<ConsumerLeasePayload *>(payload);
    consumer_lease->dataset->open_step_leases.fetch_sub(1);
    delete consumer_lease;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_acquire_step_lease(amio_dataset_handle dataset, uint64_t step_id,
                                               amio_step_lease_handle *out_lease) {
    if (out_lease == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_lease = nullptr;
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(
        HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
        HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    std::lock_guard<std::mutex> lock(owner->mutex);
    amio::detail::CollectionReader reader(owner->location, owner->uuid);
    amio::detail::CatalogStep published_step;
    const auto status = reader.describe_step(step_id, published_step);
    if (status != AMIO_OK) return status;

    auto lease = std::make_unique<ConsumerLeasePayload>();
    lease->dataset = owner;
    lease->step_id = step_id;
    const auto token = process_handle_table().insert(HandleKind::ConsumerLease, lease.get());
    if (token == 0) return AMIO_ERR_BACKEND_FAILURE;
    owner->open_step_leases.fetch_add(1);
    *out_lease = reinterpret_cast<amio_step_lease_handle>(HandleTable::to_ptr(token));
    lease.release();
    return AMIO_OK;
}

} // extern "C"
