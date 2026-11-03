#define AMIO_BUILDING_LIBRARY 1

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "amio/amio_evolution.h"
#include "c_boundary/amio_core.hpp"
#include "c_boundary/amio_tool_api.hpp"
#include "c_boundary/handle_table.hpp"
#include "catalog/catalog_format.hpp"
#include "catalog/collection_reader.hpp"
#include "catalog/posix_catalog.hpp"
#include "diagnostics/diagnostic_store.hpp"
#include "drivers/grib2/grib2_driver.hpp"
#include "drivers/grib2/grib2_steps.hpp"
#include "drivers/netcdf/netcdf_driver.hpp"
#include "drivers/zarr/zarr_steps.hpp"
#include "execution/context.hpp"
#include "execution/credit_window.hpp"
#include "execution/mpi_transport.hpp"
#include "execution/progress_engine.hpp"
#include "execution/role_map.hpp"
#include "execution/stream_execution.hpp"
#include "execution/stream_retention.hpp"
#include "memory/memory_descriptor.hpp"
#include "operators/integrity_reader.hpp"
#include "operators/operator_executor.hpp"
#include "operators/payload_integrity.hpp"
#include "prefetch/step_read_queue.hpp"
#include "prefetch/stream_read_queue.hpp"
#include "schema/schema.hpp"
#include "staging/resource_budget.hpp"
#include "staging/staging_pool.hpp"
#include "steps/step_coordinator.hpp"
#include "workers/worker_pool.hpp"

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
    std::unique_ptr<amio::detail::CreditWindow> stream_credits;
    std::unique_ptr<amio::detail::MpiTransport> stream_transport;
    std::unique_ptr<amio::detail::StreamRetention> stream_retention;
    std::unique_ptr<amio::detail::StreamExecution> stream_execution;
    std::unique_ptr<amio::detail::StreamReadQueue> stream_reads;
    std::array<std::uint8_t, amio::detail::kDatasetUuidBytes> stream_uuid{};
    std::uint64_t stream_epoch = 1;
    std::string canonical_configuration;
    std::unique_ptr<amio::detail::DiagnosticStore> diagnostics;
    // Declared after context so workers are joined before the communicator is freed.
    std::unique_ptr<amio::detail::WorkerPool> writer_workers;
    std::unique_ptr<amio::detail::ProgressEngine> progress_engine;
    std::atomic<std::uint64_t> open_datasets{0};
    std::atomic<std::uint64_t> open_step_handles{0};
};

struct ScopedBudgetReservation {
    amio::detail::ResourceBudget *budget = nullptr;
    std::optional<amio::detail::ResourceBudget::Reservation> reservation;
    ~ScopedBudgetReservation() {
        if (budget && reservation) (void)budget->release(*reservation);
    }
};

struct StreamLeasePayload {
    ContextPayload *context = nullptr;
    amio::detail::StreamLease lease;
};

struct EvolutionDatasetPayload {
    static constexpr std::size_t kStepReadCacheBytes = 16u * 1024u * 1024u;
    ContextPayload *context = nullptr;
    std::string location;
    amio_evolution_format_t format = AMIO_FORMAT_NETCDF4;
    bool distributed_netcdf = false;
    bool distributed_zarr = false;
    bool read_only = false;
    std::uint64_t payload_budget_bytes = 0;
    amio::detail::PosixSyncLevel requested_sync = amio::detail::PosixSyncLevel::Visibility;
    std::unique_ptr<amio::detail::GRIB2_Driver> grib2_driver;
    std::string uuid;
    std::unique_ptr<amio::detail::ResourceBudget> budget;
    std::unique_ptr<amio::detail::StepCoordinator> steps;
    std::map<std::string, amio::detail::ResourceBudget::Reservation, std::less<>> schema_reservations;
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
    /** Field identifiers retained until publication so staging gauges can return to zero. */
    std::map<std::string, std::uint32_t, std::less<>> diagnostic_field_ids;
    bool ended = false;
};

void record_step_publication(ContextPayload &context, const EvolutionDatasetPayload &dataset, const ProducerStepPayload &step, std::uint64_t bytes,
                             std::uint64_t elapsed_ns) {
    if (!context.diagnostics) return;
    amio::detail::DiagnosticScope scope;
    scope.context_epoch = 1;
    scope.dataset_id = std::hash<std::string>{}(dataset.uuid);
    scope.step_id = step.step_id;
    scope.role = amio::detail::DiagnosticRole::Producer;
    scope.participant = context.context->rank();
    context.diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::Completed);
    context.diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::BytesWritten, bytes);
    context.diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::BytesPublished, bytes);
    context.diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::PublicationNs, elapsed_ns);
    context.diagnostics->record_event(scope, amio::detail::DiagnosticEventKind::Publication, "step became reader-visible");
    for (const auto &[field, field_id] : step.diagnostic_field_ids) {
        (void)field;
        scope.field_id = field_id;
        context.diagnostics->set_resource(scope, amio::detail::DiagnosticResource::StagingRetainedBytes, 0, dataset.payload_budget_bytes);
        context.diagnostics->set_resource(scope, amio::detail::DiagnosticResource::ActiveOperations, 0,
                                          dataset.budget ? dataset.budget->limits().descriptors : 0);
    }
}

void record_stream_counter(ContextPayload &context, std::uint64_t step_id, amio::detail::DiagnosticRole role, amio::detail::DiagnosticCounter counter,
                           std::uint64_t amount = 1) {
    if (!context.diagnostics) return;
    amio::detail::DiagnosticScope scope;
    scope.context_epoch = context.stream_epoch;
    scope.dataset_id = std::hash<std::string>{}(context.canonical_configuration);
    scope.step_id = step_id;
    scope.role = role;
    scope.participant = static_cast<std::uint32_t>(context.context->rank());
    context.diagnostics->add_counter(scope, counter, amount);
}

struct OperationHandlePayload {
    std::shared_ptr<amio::detail::OperationRecord> operation;
};

struct ConsumerLeasePayload {
    EvolutionDatasetPayload *dataset = nullptr;
    std::uint64_t step_id = 0;
};

void record_step_read_cache_occupancy(EvolutionDatasetPayload &dataset, std::uint64_t step_id, std::string_view field) {
    auto *context = dataset.context;
    if (context == nullptr || !context->diagnostics) return;
    amio::detail::DiagnosticScope scope;
    scope.context_epoch = 1;
    scope.dataset_id = std::hash<std::string>{}(dataset.uuid);
    scope.step_id = step_id;
    scope.field_id = static_cast<std::uint32_t>(std::hash<std::string_view>{}(field));
    scope.role = amio::detail::DiagnosticRole::Consumer;
    scope.participant = context->context->rank();
    context->diagnostics->set_resource(scope, amio::detail::DiagnosticResource::PrefetchBytes, dataset.read_cache.bytes_used(),
                                       EvolutionDatasetPayload::kStepReadCacheBytes);
}

std::size_t dtype_size(amio_dtype_t dtype) noexcept {
    switch (dtype) {
        case AMIO_DTYPE_I8:
        case AMIO_DTYPE_U8:
            return 1;
        case AMIO_DTYPE_I16:
        case AMIO_DTYPE_U16:
            return 2;
        case AMIO_DTYPE_I32:
        case AMIO_DTYPE_U32:
        case AMIO_DTYPE_F32:
            return 4;
        case AMIO_DTYPE_I64:
        case AMIO_DTYPE_U64:
        case AMIO_DTYPE_F64:
            return 8;
        default:
            return 0;
    }
}

amio::detail::OperatorDestination operator_destination(amio_evolution_format_t format) noexcept {
    switch (format) {
        case AMIO_FORMAT_NETCDF4:
            return amio::detail::OperatorDestination::NetCDF4;
        case AMIO_FORMAT_ZARR3:
            return amio::detail::OperatorDestination::Zarr3;
        case AMIO_FORMAT_GRIB2:
            return amio::detail::OperatorDestination::GRIB2;
        default:
            return amio::detail::OperatorDestination::Unknown;
    }
}

amio_status_t collective_status(const amio::detail::ExecutionContext &context, amio_status_t local_status);

using IntegrityRegionMap = std::map<std::string, std::vector<amio::detail::PayloadDescriptor::IntegrityRegion>, std::less<>>;

struct IntegrityRegionWire {
    std::uint32_t present = 0;
    std::uint32_t rank = 0;
    std::uint64_t offsets[AMIO_MAX_RANK]{};
    std::uint64_t extents[AMIO_MAX_RANK]{};
    char digest[83]{};
};

amio_status_t gather_integrity_regions(EvolutionDatasetPayload &owner, const ProducerStepPayload &producer_step, const amio::detail::Schema &schema,
                                       const std::map<std::string, amio::detail::ExecutedField, std::less<>> &operated, IntegrityRegionMap &out,
                                       ScopedBudgetReservation &reservation) {
#ifdef AMIO_HAS_MPI
    const auto field_count = schema.variables().size();
    if (field_count > static_cast<std::size_t>(INT_MAX) / sizeof(IntegrityRegionWire)) return AMIO_ERR_BUDGET_EXCEEDED;
    const auto per_rank_bytes = field_count * sizeof(IntegrityRegionWire);
    if (per_rank_bytes > static_cast<std::size_t>(INT_MAX)) return AMIO_ERR_BUDGET_EXCEEDED;
    std::vector<IntegrityRegionWire> local;
    amio_status_t status = AMIO_OK;
    try {
        local.resize(field_count);
    } catch (...) {
        status = AMIO_ERR_BACKEND_FAILURE;
    }
    std::size_t index = 0;
    for (const auto &[name, variable] : schema.variables()) {
        (void)variable;
        const auto contribution = owner.steps->contributions().find(name);
        const auto field = operated.find(name);
        if (contribution != owner.steps->contributions().end() && field != operated.end()) {
            auto &wire = local[index];
            if (contribution->second.region.offsets.size() > AMIO_MAX_RANK ||
                contribution->second.region.offsets.size() != contribution->second.region.extents.size() ||
                field->second.integrity_digest.size() >= sizeof(wire.digest)) {
                status = AMIO_ERR_SCHEMA_MISMATCH;
                break;
            }
            wire.present = 1;
            wire.rank = static_cast<std::uint32_t>(contribution->second.region.offsets.size());
            for (std::size_t d = 0; d < wire.rank; ++d) {
                wire.offsets[d] = contribution->second.region.offsets[d];
                wire.extents[d] = contribution->second.region.extents[d];
            }
            std::memcpy(wire.digest, field->second.integrity_digest.data(), field->second.integrity_digest.size());
        }
        ++index;
    }
    status = collective_status(*owner.context->context, status);
    if (status != AMIO_OK) return status;
    const int root = static_cast<int>(owner.context->context->storage_root_rank());
    const int size = owner.context->context->size();
    if (root < 0 || root >= size || size <= 0 || per_rank_bytes > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(size))
        return AMIO_ERR_INVALID_INPUT;
    const auto receive_bytes = per_rank_bytes * static_cast<std::size_t>(size);
    if (field_count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(size)) return AMIO_ERR_BUDGET_EXCEEDED;
    const auto reserve_bytes = owner.context->context->rank() == root ? receive_bytes : per_rank_bytes;
    const auto reserve_descriptors = owner.context->context->rank() == root ? field_count * static_cast<std::size_t>(size) : field_count;
    reservation.reservation = owner.budget->reserve(reserve_bytes, reserve_descriptors, amio::detail::ResourceBudget::Class::Control);
    status = reservation.reservation ? AMIO_OK : AMIO_ERR_BUDGET_EXCEEDED;
    status = collective_status(*owner.context->context, status);
    if (status != AMIO_OK) return status;
    std::vector<IntegrityRegionWire> received;
    if (owner.context->context->rank() == root) {
        try {
            received.resize(field_count * static_cast<std::size_t>(size));
        } catch (...) {
            status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
    status = collective_status(*owner.context->context, status);
    if (status != AMIO_OK) return status;
    if (MPI_Gather(local.data(), static_cast<int>(per_rank_bytes), MPI_BYTE, owner.context->context->rank() == root ? received.data() : nullptr,
                   static_cast<int>(per_rank_bytes), MPI_BYTE, root, owner.context->context->communicator()) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (owner.context->context->rank() != root) return AMIO_OK;
    try {
        index = 0;
        for (const auto &[name, variable] : schema.variables()) {
            (void)variable;
            auto &regions = out[name];
            for (int rank = 0; rank < size; ++rank) {
                const auto &wire = received[static_cast<std::size_t>(rank) * field_count + index];
                if (wire.present == 0) continue;
                if (wire.present != 1 || wire.rank > AMIO_MAX_RANK || wire.digest[0] == '\0') return AMIO_ERR_SCHEMA_MISMATCH;
                amio::detail::PayloadDescriptor::IntegrityRegion region;
                region.offsets.assign(wire.offsets, wire.offsets + wire.rank);
                region.extents.assign(wire.extents, wire.extents + wire.rank);
                region.digest.assign(wire.digest);
                regions.push_back(std::move(region));
            }
            ++index;
        }
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    return AMIO_OK;
#else
    (void)owner;
    (void)producer_step;
    (void)schema;
    (void)operated;
    (void)out;
    (void)reservation;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio::detail::Grib2Settings grib2_settings_for(const ContextPayload &context, const amio::detail::Variable &variable) {
    auto config = conf::Config::from_string(context.canonical_configuration);
    auto settings = amio::detail::GRIB2_Driver::read_settings(config);
    for (const auto &[name, attribute] : variable.attributes) {
        constexpr std::string_view prefix = "grib2.";
        if (!name.starts_with(prefix)) continue;
        const auto key = std::string_view(name).substr(prefix.size());
        std::int64_t value = 0;
        if (const auto *signed_value = std::get_if<std::int64_t>(&attribute.value))
            value = *signed_value;
        else if (const auto *unsigned_value = std::get_if<std::uint64_t>(&attribute.value)) {
            if (*unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::invalid_argument("GRIB2 variable attribute is outside the signed metadata range");
            value = static_cast<std::int64_t>(*unsigned_value);
        } else
            throw std::invalid_argument("GRIB2 variable attributes must be signed or unsigned integers");
#define AMIO_GRIB2_SETTING(n, m) \
    if (key == n) settings.m = value
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
    return amio::detail::GRIB2_Driver::field_identity_name(settings.discipline, settings.parameter_category, settings.parameter_number,
                                                           settings.type_of_first_fixed_surface, settings.scaled_value_first_surface,
                                                           settings.pdt_number, settings.chemical_constituent_type, settings.aerosol_type,
                                                           settings.optical_property_type, settings.wavelength_first_nm, settings.wavelength_last_nm,
                                                           settings.ensemble_perturbation_number, settings.statistical_process);
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

/** Estimate retained schema metadata from its canonical bytes and owned record/container objects. */
bool schema_metadata_bytes(const amio::detail::Schema &schema, std::size_t &bytes) noexcept {
    bytes = sizeof(amio::detail::Schema);
    const auto add = [&bytes](std::size_t amount) {
        if (amount > std::numeric_limits<std::size_t>::max() - bytes) return false;
        bytes += amount;
        return true;
    };
    // Charge retained capacity, including the duplicate key and variable name
    // owned by the schema map. Small-string capacity is charged conservatively.
    const auto add_string = [&add](const std::string &value) {
        return value.capacity() != std::numeric_limits<std::size_t>::max() && add(value.capacity() + 1);
    };
    constexpr std::size_t node_links = 4 * sizeof(void *);
    if (!add_string(schema.canonical()) || !add_string(schema.digest())) return false;
    for (const auto &[name, variable] : schema.variables()) {
        if (!add(sizeof(std::string) + sizeof(amio::detail::Variable) + node_links) || !add_string(name) || !add_string(variable.name) ||
            variable.dimensions.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(amio::detail::Dimension) ||
            !add(variable.dimensions.capacity() * sizeof(amio::detail::Dimension)) ||
            variable.attributes.size() > std::numeric_limits<std::size_t>::max() / (sizeof(amio::detail::Attribute) + sizeof(std::string)))
            return false;
        for (const auto &dimension : variable.dimensions) {
            if (!add_string(dimension.name) || (dimension.coordinate_variable && !add_string(*dimension.coordinate_variable))) return false;
        }
        for (const auto &[attribute_name, attribute] : variable.attributes) {
            if (!add(sizeof(amio::detail::Attribute) + sizeof(std::string) + node_links) || !add_string(attribute_name)) return false;
            const bool values_fit = std::visit(
                [&add, &add_string](const auto &value) {
                    using Value = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<Value, std::string>) {
                        return add_string(value);
                    } else if constexpr (requires {
                                             typename Value::value_type;
                                             value.size();
                                         }) {
                        using Element = typename Value::value_type;
                        if (value.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(Element) || !add(value.capacity() * sizeof(Element)))
                            return false;
                        if constexpr (std::is_same_v<Element, std::string>) {
                            for (const auto &text : value)
                                if (!add_string(text)) return false;
                        }
                        return true;
                    } else {
                        return true;
                    }
                },
                attribute.value);
            if (!values_fit) return false;
        }
    }
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
        request->struct_size < offsetof(amio_validation_request_t, reserved) + sizeof(request->reserved))
        return false;
    if (request->reserved32 != 0 || request->reserved[0] != 0 || request->reserved[1] != 0 || request->max_results == 0) {
        return false;
    }
    constexpr std::uint64_t known = (UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1)) | (UINT64_C(1) << (AMIO_VALIDATE_REQUIRED_FIELDS - 1)) |
                                    (UINT64_C(1) << (AMIO_VALIDATE_PAYLOAD_PRESENCE - 1)) | (UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1)) |
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
    return descriptor != nullptr && descriptor->struct_size >= minimum_size && descriptor->version == AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
}

bool copy_utf8(const char *data, std::uint64_t bytes, std::string &out, bool allow_empty = false) {
    if (bytes > amio::detail::Schema::kMaxStringBytes || bytes > std::numeric_limits<std::size_t>::max() || (bytes != 0 && data == nullptr) ||
        (!allow_empty && bytes == 0))
        return false;
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
    if (!valid_descriptor_header(&descriptor, minimum) || descriptor.reserved32 != 0 || descriptor.reserved[0] != 0 || descriptor.reserved[1] != 0 ||
        descriptor.element_count == 0 || descriptor.element_count > 1'000'000 || descriptor.value_bytes > amio::detail::Schema::kMaxStringBytes ||
        descriptor.value_bytes > std::numeric_limits<std::size_t>::max() || descriptor.value == nullptr)
        return false;

    auto copy_values = [&]<typename T>(std::vector<T> &values) {
        if (descriptor.element_count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
            descriptor.value_bytes != descriptor.element_count * sizeof(T))
            return false;
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
            if (values.size() == 1)
                out.value = values.front();
            else
                out.value = std::move(values);
            return true;
        }
        case AMIO_ATTRIBUTE_U64: {
            std::vector<std::uint64_t> values;
            if (!copy_values(values)) return false;
            if (values.size() == 1)
                out.value = values.front();
            else
                out.value = std::move(values);
            return true;
        }
        case AMIO_ATTRIBUTE_F64: {
            std::vector<double> values;
            if (!copy_values(values)) return false;
            if (values.size() == 1)
                out.value = values.front();
            else
                out.value = std::move(values);
            return true;
        }
        case AMIO_ATTRIBUTE_UTF8: {
            if (descriptor.element_count != 1) return false;
            std::string value;
            if (!copy_utf8(static_cast<const char *>(descriptor.value), descriptor.value_bytes, value, true)) return false;
            out.value = std::move(value);
            return true;
        }
        default:
            return false;
    }
}

amio_status_t compare_schema_collectively(const amio::detail::ExecutionContext &context, const std::string &canonical) {
    return amio::detail::compare_context_metadata(context, canonical, amio::detail::Schema::kMaxCanonicalBytes);
}

amio_status_t collective_status(const amio::detail::ExecutionContext &context, amio_status_t local_status) {
#ifdef AMIO_HAS_MPI
    const MPI_Comm comm = context.communicator();
    int rank = 0;
    if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    const int candidate = local_status == AMIO_OK ? std::numeric_limits<int>::max() : rank;
    int first_failure_rank = std::numeric_limits<int>::max();
    if (MPI_Allreduce(&candidate, &first_failure_rank, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    if (first_failure_rank == std::numeric_limits<int>::max()) return AMIO_OK;
    int agreed_status = rank == first_failure_rank ? local_status : AMIO_OK;
    if (MPI_Bcast(&agreed_status, 1, MPI_INT, first_failure_rank, comm) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    return static_cast<amio_status_t>(agreed_status);
#else
    (void)context;
    return local_status;
#endif
}

amio_status_t validate_partition_descriptors(const amio::detail::Schema &schema, const amio_field_partition_descriptor_t *partitions,
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

amio_status_t validate_distributed_partitions(EvolutionDatasetPayload &dataset, const ProducerStepPayload &step, const amio::detail::Schema &schema,
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
        if (MPI_Allgather(&local, static_cast<int>(sizeof(local)), MPI_BYTE, gathered.data(), static_cast<int>(sizeof(local)), MPI_BYTE, comm) !=
            MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;

        std::vector<const PartitionWire *> active;
        for (const auto &partition : gathered)
            if (partition.present != 0) active.push_back(&partition);
        if (active.empty()) {
            if (variable.required) return AMIO_ERR_FIELD_NOT_PRESENT;
            present_fields[name] = false;
            continue;
        }

        std::uint64_t global_volume = 1;
        for (const auto &dimension : variable.dimensions) {
            if (dimension.extent == 0 || global_volume > std::numeric_limits<std::uint64_t>::max() / dimension.extent) return AMIO_ERR_INVALID_INPUT;
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
                if (extent == 0 || offset > global || extent > global - offset || volume > std::numeric_limits<std::uint64_t>::max() / extent)
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
                    if (a_end <= b.offsets[d] || b_end <= a.offsets[d]) {
                        intersects = false;
                        break;
                    }
                }
                if (intersects) return AMIO_ERR_INVALID_POLICY;
            }
        }
        if (volume_sum != global_volume) return AMIO_ERR_INVALID_POLICY;
        present_fields[name] = true;
    }
    return AMIO_OK;
#else
    (void)dataset;
    (void)step;
    (void)schema;
    (void)present_fields;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t derive_distributed_zarr_shards(EvolutionDatasetPayload &dataset, const ProducerStepPayload &step, const amio::detail::Schema &schema,
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
        if (MPI_Allgather(&local, static_cast<int>(sizeof(local)), MPI_BYTE, gathered.data(), static_cast<int>(sizeof(local)), MPI_BYTE,
                          context.communicator()) != MPI_SUCCESS)
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
    return amio::detail::derive_zarr_shard_shapes(schema, all_regions, budget, shard_shapes, error) ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
#else
    (void)dataset;
    (void)step;
    (void)schema;
    (void)shard_shapes;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t broadcast_publication(const amio::detail::ExecutionContext &context, amio::detail::PublicationResult &result) {
#ifdef AMIO_HAS_MPI
    const MPI_Comm comm = context.communicator();
    const int root = static_cast<int>(context.storage_root_rank());
    int wire[2] = {static_cast<int>(result.status), static_cast<int>(result.state)};
    unsigned long long path_bytes = static_cast<unsigned long long>(result.commit_path.size());
    if (MPI_Bcast(wire, 2, MPI_INT, root, comm) != MPI_SUCCESS || MPI_Bcast(&path_bytes, 1, MPI_UNSIGNED_LONG_LONG, root, comm) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (path_bytes > 1024 * 1024) return AMIO_ERR_BACKEND_FAILURE;
    if (context.rank() != static_cast<int>(context.storage_root_rank())) result.commit_path.resize(static_cast<std::size_t>(path_bytes));
    constexpr std::size_t chunk = 1u << 20;
    for (std::size_t offset = 0; offset < result.commit_path.size(); offset += chunk) {
        const auto count = std::min(chunk, result.commit_path.size() - offset);
        if (MPI_Bcast(result.commit_path.data() + offset, static_cast<int>(count), MPI_CHAR, root, comm) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    result.status = static_cast<amio_status_t>(wire[0]);
    result.state = static_cast<amio::detail::PublicationState>(wire[1]);
#else
    (void)context;
    (void)result;
#endif
    return AMIO_OK;
}
}  // namespace

namespace {
std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 2);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\b':
                output += "\\b";
                break;
            case '\f':
                output += "\\f";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    output += "\\u00";
                    output += hex[(ch >> 4) & 0xf];
                    output += hex[ch & 0xf];
                } else
                    output += static_cast<char>(ch);
        }
    }
    return output;
}

std::string diagnostics_json(const amio::detail::DiagnosticSnapshot &snapshot, const amio::detail::ExecutionContext &context) {
    static constexpr const char *counter_names[] = {"accepted",
                                                    "rejected",
                                                    "completed",
                                                    "failed",
                                                    "bytes_accepted",
                                                    "bytes_written",
                                                    "bytes_published",
                                                    "source_released_bytes",
                                                    "transport_received_bytes",
                                                    "transport_published_bytes",
                                                    "view_bytes",
                                                    "stream_skips",
                                                    "stream_delivered_bytes",
                                                    "stream_released_bytes",
                                                    "stream_disconnects",
                                                    "operator_input_bytes",
                                                    "operator_output_bytes"};
    static constexpr const char *histogram_names[] = {"admission_ns",   "snapshot_ns",   "operator_ns", "transport_ns",     "backend_ns",
                                                      "publication_ns", "end_to_end_ns", "wait_ns",     "source_release_ns"};
    static constexpr const char *resource_names[] = {"queue_descriptors",      "staging_retained_bytes",  "staging_descriptors", "transport_bytes",
                                                     "transport_descriptors",  "metadata_bytes",          "prefetch_bytes",      "view_bytes",
                                                     "operator_scratch_bytes", "exception_history_bytes", "event_history_bytes", "active_operations"};
    const char *execution_mode = "unknown";
    switch (context.roles().mode) {
        case amio::detail::ExecutionMode::Direct:
            execution_mode = "persistent_direct";
            break;
        case amio::detail::ExecutionMode::Aggregated:
            execution_mode = "persistent_aggregated";
            break;
        case amio::detail::ExecutionMode::Stream:
            execution_mode = "mpi_stream";
            break;
    }
    const char *progress_mode = context.progress_policy() == amio::detail::ProgressPolicy::Caller ? "caller" : "background";
    std::ostringstream out;
    const auto &roles = context.role_map();
    out << "{\"api_version\":2,\"topology\":{\"execution_mode\":\"" << execution_mode << "\",\"progress_mode\":\"" << progress_mode
        << "\",\"rank\":" << context.rank() << ",\"rank_count\":" << context.size() << ",\"producers\":[";
    const auto write_ranks = [&out](const std::vector<int> &ranks) {
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            if (i != 0) out << ',';
            out << ranks[i];
        }
    };
    write_ranks(roles.producers());
    out << "],\"writers\":[";
    write_ranks(roles.writers());
    out << "],\"consumers\":[";
    write_ranks(context.roles().consumers);
    out << "]},\"scope_count\":" << snapshot.scope_count << ",\"evicted_scope_count\":" << snapshot.evicted_scope_count
        << ",\"dropped_sample_count\":" << snapshot.dropped_sample_count << ",\"cumulative_counters\":{";
    for (std::size_t i = 0; i < snapshot.cumulative.size(); ++i) {
        if (i != 0) out << ',';
        out << '\"' << counter_names[i] << "\":" << snapshot.cumulative[i];
    }
    out << "},\"counter_names\":[";
    for (std::size_t i = 0; i < snapshot.cumulative.size(); ++i) {
        if (i != 0) out << ',';
        out << '\"' << counter_names[i] << '\"';
    }
    out << "],\"histogram_names\":[";
    for (std::size_t i = 0; i < snapshot.cumulative_histograms.size(); ++i) {
        if (i != 0) out << ',';
        out << '\"' << histogram_names[i] << '\"';
    }
    out << "],\"resource_names\":[";
    for (std::size_t i = 0; i < snapshot.cumulative_resource_high_water.size(); ++i) {
        if (i != 0) out << ',';
        out << '\"' << resource_names[i] << '\"';
    }
    out << "],\"cumulative_histograms\":[";
    for (std::size_t i = 0; i < snapshot.cumulative_histograms.size(); ++i) {
        if (i != 0) out << ',';
        out << '[';
        for (std::size_t bin = 0; bin < snapshot.cumulative_histograms[i].size(); ++bin) {
            if (bin != 0) out << ',';
            out << snapshot.cumulative_histograms[i][bin];
        }
        out << ']';
    }
    out << "],\"cumulative_resource_high_water\":[";
    for (std::size_t i = 0; i < snapshot.cumulative_resource_high_water.size(); ++i) {
        if (i != 0) out << ',';
        out << snapshot.cumulative_resource_high_water[i];
    }
    out << "],\"scopes\":[";
    bool first_scope = true;
    for (const auto &[scope, value] : snapshot.scopes) {
        if (!first_scope) out << ',';
        first_scope = false;
        out << "{\"context_epoch\":" << scope.context_epoch << ",\"dataset_id\":" << scope.dataset_id << ",\"step_id\":" << scope.step_id
            << ",\"field_id\":" << scope.field_id << ",\"role\":" << static_cast<unsigned>(scope.role) << ",\"participant\":" << scope.participant
            << ",\"counters\":[";
        for (std::size_t i = 0; i < value.counters.size(); ++i) {
            if (i != 0) out << ',';
            out << value.counters[i];
        }
        out << "],\"histograms\":[";
        for (std::size_t i = 0; i < value.histograms.size(); ++i) {
            if (i != 0) out << ',';
            out << '[';
            for (std::size_t bin = 0; bin < value.histograms[i].size(); ++bin) {
                if (bin != 0) out << ',';
                out << value.histograms[i][bin];
            }
            out << ']';
        }
        out << "],\"resource_current\":[";
        for (std::size_t i = 0; i < value.resource_current.size(); ++i) {
            if (i != 0) out << ',';
            out << value.resource_current[i];
        }
        out << "],\"resource_high_water\":[";
        for (std::size_t i = 0; i < value.resource_high_water.size(); ++i) {
            if (i != 0) out << ',';
            out << value.resource_high_water[i];
        }
        out << "],\"resource_ceilings\":[";
        for (std::size_t i = 0; i < value.resource_ceilings.size(); ++i) {
            if (i != 0) out << ',';
            out << value.resource_ceilings[i];
        }
        out << "],\"first_error\":{\"status\":" << value.first_error.status << ",\"message\":\"" << json_escape(value.first_error.message) << "\"}}";
    }
    out << "],\"first_error\":{" << "\"status\":" << snapshot.first_error.status << ",\"message\":\"" << json_escape(snapshot.first_error.message)
        << "\"},\"events\":[";
    bool first = true;
    for (const auto &event : snapshot.events) {
        if (!first) out << ',';
        first = false;
        out << "{\"kind\":" << static_cast<unsigned>(event.kind) << ",\"context_epoch\":" << event.scope.context_epoch
            << ",\"dataset_id\":" << event.scope.dataset_id << ",\"step_id\":" << event.scope.step_id << ",\"field_id\":" << event.scope.field_id
            << ",\"role\":" << static_cast<unsigned>(event.scope.role) << ",\"participant\":" << event.scope.participant << ",\"message\":\""
            << json_escape(event.message) << "\"}";
    }
    out << "]}";
    return out.str();
}

std::string reduced_diagnostics_json(const std::array<std::uint64_t, amio::detail::kDiagnosticCounterCount> &counters) {
    static constexpr const char *names[] = {"accepted",
                                            "rejected",
                                            "completed",
                                            "failed",
                                            "bytes_accepted",
                                            "bytes_written",
                                            "bytes_published",
                                            "source_released_bytes",
                                            "transport_received_bytes",
                                            "transport_published_bytes",
                                            "view_bytes",
                                            "stream_skips",
                                            "stream_delivered_bytes",
                                            "stream_released_bytes",
                                            "stream_disconnects"};
    std::ostringstream out;
    out << "{\"api_version\":2,\"scope\":\"context\",\"cumulative_counters\":{";
    for (std::size_t i = 0; i < counters.size(); ++i) {
        if (i != 0) out << ',';
        auto digits = std::to_string(counters[i]);
        out << '\"' << names[i] << "\":\"" << std::string(20 - digits.size(), '0') << digits << '\"';
    }
    out << "}}";
    return out.str();
}
}  // namespace

extern "C" {

AMIO_API amio_status_t amio_init_ex(const amio_context_descriptor_t *descriptor, amio_context_handle *out_context) {
    if (out_context == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_context = nullptr;
    constexpr std::size_t minimum_size = offsetof(amio_context_descriptor_t, reserved) + 2 * sizeof(uint64_t);
    if (descriptor == nullptr) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->struct_size < offsetof(amio_context_descriptor_t, version) + sizeof(descriptor->version)) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->struct_size < minimum_size) return AMIO_ERR_INVALID_INPUT;
#ifdef AMIO_HAS_MPI
    // Once the descriptor supplies a usable communicator, agree on all local
    // validation that would otherwise return before collective context setup.
    // A null/short descriptor or unusable communicator cannot identify a
    // common ordering domain and remains a caller-side precondition.
    int initialized = 0, finalized = 0;
    if (MPI_Initialized(&initialized) == MPI_SUCCESS && MPI_Finalized(&finalized) == MPI_SUCCESS && initialized && !finalized &&
        descriptor->fortran_communicator >= std::numeric_limits<MPI_Fint>::min() &&
        descriptor->fortran_communicator <= std::numeric_limits<MPI_Fint>::max()) {
        const auto caller = MPI_Comm_f2c(static_cast<MPI_Fint>(descriptor->fortran_communicator));
        if (caller != MPI_COMM_NULL) {
            int local_status = AMIO_OK;
            if (descriptor->version != AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 || descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0 ||
                descriptor->rank_count == 0 || descriptor->rank_count > 1'000'000 || descriptor->configuration_bytes == 0 ||
                descriptor->configuration_bytes > 1024 * 1024 || descriptor->canonical_configuration == nullptr ||
                descriptor->producer_count > descriptor->rank_count || descriptor->writer_count > descriptor->rank_count ||
                descriptor->consumer_count > descriptor->rank_count || (descriptor->producer_count != 0 && descriptor->producers == nullptr) ||
                (descriptor->writer_count != 0 && descriptor->writers == nullptr) ||
                (descriptor->consumer_count != 0 && descriptor->consumers == nullptr))
                local_status = AMIO_ERR_INVALID_INPUT;
            else if ((descriptor->execution_mode != AMIO_EXECUTION_PERSISTENT_DIRECT &&
                      descriptor->execution_mode != AMIO_EXECUTION_PERSISTENT_AGGREGATED &&
                      descriptor->execution_mode != AMIO_EXECUTION_MPI_STREAM) ||
                     (descriptor->progress_mode != AMIO_PROGRESS_CALLER && descriptor->progress_mode != AMIO_PROGRESS_BACKGROUND))
                local_status = AMIO_ERR_INVALID_POLICY;
            int agreed_status = AMIO_OK;
            if (MPI_Allreduce(&local_status, &agreed_status, 1, MPI_INT, MPI_MAX, caller) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
            if (agreed_status != AMIO_OK) return agreed_status;
        }
    }
#endif
    if (descriptor->version != AMIO_EVOLUTION_DESCRIPTOR_VERSION_1) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0) return AMIO_ERR_INVALID_INPUT;
    if (descriptor->rank_count == 0 || descriptor->rank_count > 1'000'000 || descriptor->configuration_bytes == 0 ||
        descriptor->configuration_bytes > 1024 * 1024 || descriptor->canonical_configuration == nullptr ||
        descriptor->producer_count > descriptor->rank_count || descriptor->writer_count > descriptor->rank_count ||
        descriptor->consumer_count > descriptor->rank_count || (descriptor->producer_count != 0 && descriptor->producers == nullptr) ||
        (descriptor->writer_count != 0 && descriptor->writers == nullptr) || (descriptor->consumer_count != 0 && descriptor->consumers == nullptr))
        return AMIO_ERR_INVALID_INPUT;

    try {
        amio::detail::RoleConfiguration roles;
        roles.rank_count = descriptor->rank_count;
        switch (descriptor->execution_mode) {
            case AMIO_EXECUTION_PERSISTENT_DIRECT:
                roles.mode = amio::detail::ExecutionMode::Direct;
                break;
            case AMIO_EXECUTION_PERSISTENT_AGGREGATED:
                roles.mode = amio::detail::ExecutionMode::Aggregated;
                break;
            case AMIO_EXECUTION_MPI_STREAM:
                roles.mode = amio::detail::ExecutionMode::Stream;
                break;
            default:
                return AMIO_ERR_INVALID_POLICY;
        }
        amio::detail::ProgressPolicy progress;
        if (descriptor->progress_mode == AMIO_PROGRESS_CALLER)
            progress = amio::detail::ProgressPolicy::Caller;
        else if (descriptor->progress_mode == AMIO_PROGRESS_BACKGROUND)
            progress = amio::detail::ProgressPolicy::Background;
        else
            return AMIO_ERR_INVALID_POLICY;
        std::string canonical;
        amio_status_t allocation_status = AMIO_OK;
        try {
            if (descriptor->producer_count) roles.producers.assign(descriptor->producers, descriptor->producers + descriptor->producer_count);
            if (descriptor->writer_count) roles.writers.assign(descriptor->writers, descriptor->writers + descriptor->writer_count);
            if (descriptor->consumer_count) roles.consumers.assign(descriptor->consumers, descriptor->consumers + descriptor->consumer_count);
            canonical.assign(descriptor->canonical_configuration, static_cast<std::size_t>(descriptor->configuration_bytes));
        } catch (...) {
            allocation_status = AMIO_ERR_BACKEND_FAILURE;
        }
        std::unique_ptr<amio::detail::ExecutionContext> context;
#ifdef AMIO_HAS_MPI
        if (descriptor->fortran_communicator < std::numeric_limits<MPI_Fint>::min() ||
            descriptor->fortran_communicator > std::numeric_limits<MPI_Fint>::max())
            return AMIO_ERR_INVALID_INPUT;
        const MPI_Comm caller = initialized && !finalized ? MPI_Comm_f2c(static_cast<MPI_Fint>(descriptor->fortran_communicator)) : MPI_COMM_NULL;
        if (caller != MPI_COMM_NULL) {
            int local_allocation = allocation_status, agreed_allocation = AMIO_OK;
            if (MPI_Allreduce(&local_allocation, &agreed_allocation, 1, MPI_INT, MPI_MAX, caller) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
            if (agreed_allocation != AMIO_OK) return agreed_allocation;
        }
        auto status = amio::detail::ExecutionContext::create(roles, progress, canonical, caller, context);
        int local_creation = status, agreed_creation = AMIO_OK;
        if (caller != MPI_COMM_NULL && MPI_Allreduce(&local_creation, &agreed_creation, 1, MPI_INT, MPI_MAX, caller) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        if (caller != MPI_COMM_NULL) status = agreed_creation;
#else
        if (allocation_status != AMIO_OK) return allocation_status;
        if (descriptor->fortran_communicator != 0) return AMIO_ERR_UNSUPPORTED;
        const auto status = amio::detail::ExecutionContext::create(roles, progress, canonical, 0, context);
#endif
        if (status != AMIO_OK) return status;
        std::unique_ptr<ContextPayload> payload;
        HandleTable::Token token = 0;
        amio_status_t payload_status = AMIO_OK;
        try {
            payload = std::make_unique<ContextPayload>();
            payload->context = std::move(context);
            payload->canonical_configuration = canonical;
            const auto diagnostic_settings = conf::Config::from_string(canonical);
            const bool diagnostics_enabled = !diagnostic_settings.has("diagnostics.enabled") || diagnostic_settings.get_bool("diagnostics.enabled");
            if (diagnostics_enabled) {
                payload->diagnostics =
                    std::make_unique<amio::detail::DiagnosticStore>(amio::detail::DiagnosticStore::Limits{256, 1024u * 1024u, 1024, 1024});
            }
            if (roles.mode == amio::detail::ExecutionMode::Aggregated && progress == amio::detail::ProgressPolicy::Background &&
                payload->context->can_open_storage()) {
                const auto settings = conf::Config::from_string(canonical);
                amio::detail::WorkerPoolConfig workers;
                if (settings.has("worker_pool.threads")) {
                    const int count = settings.get_int("worker_pool.threads");
                    if (count <= 0 || count > 256)
                        payload_status = AMIO_ERR_INVALID_POLICY;
                    else
                        workers.thread_count = static_cast<std::size_t>(count);
                }
                amio::detail::ThreadConfig affinity;
                if (settings.has("worker_pool.cpu_cores")) affinity.cpu_cores = settings.get_int_list("worker_pool.cpu_cores");
                if (settings.has("worker_pool.numa_domain")) affinity.numa_domain = settings.get_int("worker_pool.numa_domain");
                workers.thread_configs.assign(workers.thread_count, affinity);
                if (payload_status == AMIO_OK) payload_status = amio::detail::bind_writer_worker_config(*payload->context, workers);
                if (payload_status == AMIO_OK) payload->writer_workers = std::make_unique<amio::detail::WorkerPool>(workers);
            }
            if (roles.mode == amio::detail::ExecutionMode::Aggregated &&
                (progress == amio::detail::ProgressPolicy::Caller || payload->context->can_open_storage())) {
                int thread_level = 0;
#ifdef AMIO_HAS_MPI
                if (MPI_Query_thread(&thread_level) != MPI_SUCCESS) payload_status = AMIO_ERR_COMM_SPLIT_FAILED;
#endif
                if (payload_status == AMIO_OK)
                    payload->progress_engine = std::make_unique<amio::detail::ProgressEngine>(progress, 1024, thread_level);
            }
#ifdef AMIO_HAS_MPI
            if (roles.mode == amio::detail::ExecutionMode::Stream) {
                const auto stream_settings = conf::Config::from_string(canonical);
                auto retention_policy = amio::detail::StreamRetentionPolicy::Reliable;
                if (stream_settings.has("stream.retention_policy")) {
                    const auto selected = stream_settings.get_string("stream.retention_policy");
                    if (selected == "latest")
                        retention_policy = amio::detail::StreamRetentionPolicy::Latest;
                    else if (selected != "reliable")
                        payload_status = AMIO_ERR_INVALID_POLICY;
                }
                const auto uuid_seed = static_cast<std::uint64_t>(std::hash<std::string>{}(canonical));
                for (std::size_t i = 0; i < payload->stream_uuid.size(); ++i)
                    payload->stream_uuid[i] = static_cast<std::uint8_t>(uuid_seed >> ((i % 8) * 8));
                const auto peers = static_cast<std::size_t>(payload->context->size());
                payload->stream_credits = std::make_unique<amio::detail::CreditWindow>(
                    amio::detail::CreditWindow::Limits{peers, 16u * 1024u * 1024u, 16, 64u * 1024u * 1024u, 256, 1u * 1024u * 1024u, 64});
                payload->stream_transport =
                    std::make_unique<amio::detail::MpiTransport>(*payload->context, *payload->stream_credits, 1u * 1024u * 1024u);
                payload->stream_retention = std::make_unique<amio::detail::StreamRetention>(retention_policy, 1, 8, 64u * 1024u * 1024u, 4096);
                payload->stream_execution =
                    std::make_unique<amio::detail::StreamExecution>(*payload->context, *payload->stream_transport, *payload->stream_retention,
                                                                    payload->stream_epoch, payload->stream_uuid, 64u * 1024u * 1024u);
                payload->stream_reads = std::make_unique<amio::detail::StreamReadQueue>(*payload->stream_retention, 0);
            }
#else
            if (roles.mode == amio::detail::ExecutionMode::Stream) payload_status = AMIO_ERR_UNSUPPORTED;
#endif
            if (payload_status != AMIO_OK) throw std::runtime_error("invalid writer worker configuration");
            token = process_handle_table().insert(HandleKind::Context, payload.get());
            if (token == 0) payload_status = AMIO_ERR_BACKEND_FAILURE;
        } catch (...) {
            if (payload_status == AMIO_OK) payload_status = AMIO_ERR_BACKEND_FAILURE;
        }
#ifdef AMIO_HAS_MPI
        int local_payload = payload_status, agreed_payload = AMIO_OK;
        if (MPI_Allreduce(&local_payload, &agreed_payload, 1, MPI_INT, MPI_MAX, caller) != MPI_SUCCESS) agreed_payload = AMIO_ERR_COMM_SPLIT_FAILED;
        payload_status = agreed_payload;
#endif
        if (payload_status != AMIO_OK) {
            if (token != 0) {
                void *unused = nullptr;
                (void)process_handle_table().release(token, HandleKind::Context, &unused);
            }
            return payload_status;
        }
        *out_context = reinterpret_cast<amio_context_handle>(HandleTable::to_ptr(token));
        payload.release();
        return AMIO_OK;
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
}

AMIO_API amio_status_t amio_query_capabilities(amio_context_handle context, char *buffer, size_t buffer_capacity, size_t *out_required_bytes) {
    if (out_required_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_required_bytes = 0;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) {
        return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    }
    (void)payload;
#ifdef AMIO_HAS_KOKKOS_DEVICE
    static constexpr char report[] =
        "{\"api_version\":1,\"capability_levels\":[\"unsupported\",\"compiled\",\"environment_validated\",\"destination_certified\"],"
        "\"features\":{\"immutable_schema\":\"compiled\",\"borrowed_buffer\":\"unsupported\","
        "\"device_memory\":\"compiled\",\"f64_f32_precision\":\"compiled\",\"logical_integrity\":\"compiled\","
        "\"native_lossless_codec_selection\":\"destination-negotiated\",\"native_lossy_codec\":\"unsupported\","
        "\"destination_visibility\":\"compiled\",\"storage_sync\":\"compiled\"}}";
#else
    static constexpr char report[] =
        "{\"api_version\":1,\"capability_levels\":[\"unsupported\",\"compiled\",\"environment_validated\",\"destination_certified\"],"
        "\"features\":{\"immutable_schema\":\"compiled\",\"borrowed_buffer\":\"unsupported\","
        "\"device_memory\":\"unsupported\",\"f64_f32_precision\":\"compiled\",\"logical_integrity\":\"compiled\","
        "\"native_lossless_codec_selection\":\"destination-negotiated\",\"native_lossy_codec\":\"unsupported\","
        "\"destination_visibility\":\"compiled\",\"storage_sync\":\"compiled\"}}";
#endif
    constexpr std::size_t required = sizeof(report);
    *out_required_bytes = required;
    if (buffer == nullptr && buffer_capacity == 0) return AMIO_OK;
    if (buffer == nullptr || buffer_capacity < required) return AMIO_ERR_BUFFER_TOO_SMALL;
    std::memcpy(buffer, report, required);
    return AMIO_OK;
}

/** Return this rank's effective role and role-group rank without entering MPI collectives. */
AMIO_API amio_status_t amio_query_context_role(amio_context_handle context, uint32_t *out_role, uint32_t *out_group_rank) {
    if (out_role == nullptr || out_group_rank == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_role = AMIO_ROLE_INACTIVE;
    *out_group_rank = 0;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    const auto &execution = *static_cast<ContextPayload *>(payload)->context;
    const auto role = execution.role_map().role(execution.rank());
    switch (role) {
        case amio::detail::RankRole::Producer:
            *out_role = AMIO_ROLE_PRODUCER;
            break;
        case amio::detail::RankRole::Writer:
            *out_role = AMIO_ROLE_WRITER;
            break;
        case amio::detail::RankRole::Consumer:
            *out_role = AMIO_ROLE_CONSUMER;
            break;
        case amio::detail::RankRole::Inactive:
            return AMIO_ERR_INVALID_POLICY;
    }
#ifdef AMIO_HAS_MPI
    MPI_Comm group = MPI_COMM_NULL;
    switch (role) {
        case amio::detail::RankRole::Producer:
            group = execution.producer_communicator();
            break;
        case amio::detail::RankRole::Writer:
            group = execution.writer_communicator();
            break;
        case amio::detail::RankRole::Consumer:
            group = execution.consumer_communicator();
            break;
        case amio::detail::RankRole::Inactive:
            break;
    }
    if (group != MPI_COMM_NULL) {
        int group_rank = 0;
        if (MPI_Comm_rank(group, &group_rank) != MPI_SUCCESS || group_rank < 0) return AMIO_ERR_COMM_SPLIT_FAILED;
        *out_group_rank = static_cast<uint32_t>(group_rank);
    }
#endif
    return AMIO_OK;
}

/** Return a bounded local diagnostic snapshot as versioned JSON; no collective is entered. */
AMIO_API amio_status_t amio_query_diagnostics(amio_context_handle context, char *buffer, size_t buffer_capacity, size_t *out_required_bytes) {
    if (out_required_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_required_bytes = 0;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    if (!owner->diagnostics) return AMIO_ERR_UNSUPPORTED;
    const auto json = diagnostics_json(owner->diagnostics->snapshot(), *owner->context);
    const auto required = json.size() + 1;
    *out_required_bytes = required;
    if (buffer == nullptr && buffer_capacity == 0) return AMIO_OK;
    if (buffer == nullptr || buffer_capacity < required) return AMIO_ERR_BUFFER_TOO_SMALL;
    std::memcpy(buffer, json.c_str(), required);
    return AMIO_OK;
}

/** Collectively sum cumulative context counters; size discovery remains local. */
AMIO_API amio_status_t amio_reduce_diagnostics(amio_context_handle context, char *buffer, size_t buffer_capacity, size_t *out_required_bytes) {
    if (out_required_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_required_bytes = 0;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    if (!owner->diagnostics) return AMIO_ERR_UNSUPPORTED;
    constexpr std::size_t required = sizeof(
        "{\"api_version\":1,\"scope\":\"context\",\"cumulative_counters\":{\"accepted\":\"00000000000000000000\",\"rejected\":"
        "\"00000000000000000000\",\"completed\":\"00000000000000000000\",\"failed\":\"00000000000000000000\",\"bytes_accepted\":"
        "\"00000000000000000000\",\"bytes_written\":\"00000000000000000000\",\"bytes_published\":\"00000000000000000000\",\"source_released_bytes\":"
        "\"00000000000000000000\",\"transport_received_bytes\":\"00000000000000000000\",\"transport_published_bytes\":\"00000000000000000000\","
        "\"view_bytes\":\"00000000000000000000\",\"stream_skips\":\"00000000000000000000\",\"stream_delivered_bytes\":\"00000000000000000000\","
        "\"stream_released_bytes\":\"00000000000000000000\",\"stream_disconnects\":\"00000000000000000000\"}}");
    *out_required_bytes = required;
    if (buffer == nullptr && buffer_capacity == 0) return AMIO_OK;
    const auto local_status = (buffer == nullptr || buffer_capacity < required) ? AMIO_ERR_BUFFER_TOO_SMALL : AMIO_OK;
    const auto agreed_status = collective_status(*owner->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;

    const auto local = owner->diagnostics->snapshot().cumulative;
    auto reduced = local;
#ifdef AMIO_HAS_MPI
    std::array<std::uint64_t, amio::detail::kDiagnosticCounterCount> low{}, low_sum{};
    std::array<std::uint64_t, amio::detail::kDiagnosticCounterCount> high{}, high_sum{};
    for (std::size_t i = 0; i < local.size(); ++i) {
        low[i] = static_cast<std::uint32_t>(local[i]);
        high[i] = local[i] >> 32;
    }
    if (MPI_Allreduce(low.data(), low_sum.data(), static_cast<int>(low.size()), MPI_UINT64_T, MPI_SUM, owner->context->communicator()) !=
            MPI_SUCCESS ||
        MPI_Allreduce(high.data(), high_sum.data(), static_cast<int>(high.size()), MPI_UINT64_T, MPI_SUM, owner->context->communicator()) !=
            MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    for (std::size_t i = 0; i < reduced.size(); ++i) {
        const auto upper = high_sum[i] + (static_cast<std::uint64_t>(low_sum[i]) >> 32);
        reduced[i] = upper > std::numeric_limits<std::uint32_t>::max() ? std::numeric_limits<std::uint64_t>::max()
                                                                       : (upper << 32) | static_cast<std::uint32_t>(low_sum[i]);
    }
#endif
    const auto json = reduced_diagnostics_json(reduced);
    std::memcpy(buffer, json.c_str(), json.size() + 1);
    return AMIO_OK;
}

AMIO_API amio_status_t amio_release_context(amio_context_handle context) {
    if (context == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(context))),
                                                      HandleKind::Context, &payload);
    if (lookup != AMIO_OK) return lookup;
    const auto *context_payload = static_cast<ContextPayload *>(payload);
    if (context_payload->open_datasets.load() != 0 || context_payload->open_step_handles.load() != 0) return AMIO_ERR_INVALID_POLICY;
    if (context_payload->progress_engine && context_payload->progress_engine->pending() != 0) return AMIO_ERR_OPERATION_PENDING;
    const auto status = process_handle_table().release(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(context))),
                                                       HandleKind::Context, &payload);
    if (status != AMIO_OK) return status;
    delete static_cast<ContextPayload *>(payload);
    return AMIO_OK;
}

/** Run bounded local progress work only under the explicit caller-driven policy. */
AMIO_API amio_status_t amio_progress(amio_context_handle context, uint32_t max_callbacks, uint32_t *out_completed) {
    if (out_completed == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_completed = 0;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    if (!owner->progress_engine) return AMIO_ERR_INVALID_POLICY;
    std::size_t completed = 0;
    const auto status = owner->progress_engine->progress(max_callbacks, completed);
    *out_completed = static_cast<uint32_t>(completed);
    return status;
}

/** Validate service-role use before reporting that persistent service dispatch is unavailable. */
AMIO_API amio_status_t amio_run_service(amio_context_handle context) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    const auto *owner = static_cast<const ContextPayload *>(payload);
    if (owner->context->roles().mode != amio::detail::ExecutionMode::Aggregated ||
        owner->context->role_map().role(owner->context->rank()) != amio::detail::RankRole::Writer)
        return AMIO_ERR_INVALID_POLICY;
    // The transport's bounded EOS/control loop is available internally, but
    // producer dataset operations are not yet wired to it. Returning a clear
    // capability error prevents a service rank from entering a loop that no
    // public producer operation can currently drive.
    return AMIO_ERR_UNSUPPORTED;
}

AMIO_API amio_status_t amio_stream_submit(amio_context_handle context, const amio_stream_contribution_t *contribution, const void *data,
                                          size_t data_bytes) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    constexpr std::size_t minimum = offsetof(amio_stream_contribution_t, reserved) + 2 * sizeof(std::uint64_t);
    if (!valid_descriptor_header(contribution, minimum) || contribution->reserved[0] || contribution->reserved[1] ||
        contribution->rank > AMIO_MAX_RANK || contribution->sequence == 0 || contribution->element_bytes == 0 ||
        contribution->element_bytes > std::numeric_limits<std::size_t>::max() || data == nullptr || data_bytes == 0 ||
        data_bytes > std::numeric_limits<std::size_t>::max())
        return AMIO_ERR_INVALID_INPUT;
    if (!owner->stream_execution) return AMIO_ERR_INVALID_POLICY;
    amio::detail::TransportEnvelope frame;
    frame.context_epoch = owner->stream_epoch;
    frame.dataset_uuid = owner->stream_uuid;
    frame.kind = amio::detail::MessageKind::Chunk;
    frame.producer_rank = static_cast<std::uint32_t>(owner->context->rank());
    frame.sequence = contribution->sequence;
    frame.step_id = contribution->step_id;
    frame.field_id = contribution->field_id;
    frame.schema_digest = {};
    std::copy(std::begin(contribution->schema_digest), std::end(contribution->schema_digest), frame.schema_digest.begin());
    frame.dimension_count = contribution->rank;
    for (std::size_t i = 0; i < contribution->rank; ++i) {
        frame.offsets[i] = contribution->offsets[i];
        frame.extents[i] = contribution->extents[i];
    }
    const auto status =
        owner->stream_execution->submit(frame, std::span<const std::byte>(static_cast<const std::byte *>(data), static_cast<std::size_t>(data_bytes)),
                                        static_cast<std::size_t>(contribution->element_bytes));
    if (status == AMIO_OK) {
        record_stream_counter(*owner, contribution->step_id, amio::detail::DiagnosticRole::Producer,
                              amio::detail::DiagnosticCounter::TransportPublishedBytes, data_bytes);
        record_stream_counter(*owner, contribution->step_id, amio::detail::DiagnosticRole::Producer,
                              amio::detail::DiagnosticCounter::SourceReleasedBytes, data_bytes);
    }
    return status;
}

AMIO_API amio_status_t amio_stream_wait_release(amio_context_handle context, uint64_t step_id) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    return owner->stream_execution ? owner->stream_execution->wait_for_release(step_id) : AMIO_ERR_INVALID_POLICY;
}

AMIO_API amio_status_t amio_stream_try_wait_release(amio_context_handle context, uint64_t step_id, uint64_t timeout_milliseconds) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    if (!owner->stream_execution || timeout_milliseconds > static_cast<uint64_t>(std::chrono::milliseconds::max().count()))
        return AMIO_ERR_INVALID_POLICY;
    return owner->stream_execution->wait_for_release(step_id, std::chrono::milliseconds(timeout_milliseconds));
}

AMIO_API amio_status_t amio_stream_receive_step(amio_context_handle context, const amio_stream_receive_descriptor_t *descriptor) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    constexpr std::size_t minimum = offsetof(amio_stream_receive_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    if (!valid_descriptor_header(descriptor, minimum) || descriptor->fields_per_producer == 0 || descriptor->reserved[0] || descriptor->reserved[1] ||
        (descriptor->absent_optional_count != 0 &&
         (descriptor->absent_optional_names == nullptr || descriptor->absent_optional_name_bytes == nullptr)) ||
        descriptor->absent_optional_count > 65536)
        return AMIO_ERR_INVALID_INPUT;
    if (!owner->stream_execution) return AMIO_ERR_INVALID_POLICY;
    std::vector<std::string> absent;
    try {
        absent.reserve(descriptor->absent_optional_count);
        for (std::uint32_t i = 0; i < descriptor->absent_optional_count; ++i) {
            std::string name;
            if (!copy_utf8(descriptor->absent_optional_names[i], descriptor->absent_optional_name_bytes[i], name)) return AMIO_ERR_INVALID_INPUT;
            absent.push_back(std::move(name));
        }
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    std::array<std::uint8_t, amio::detail::kSchemaDigestBytes> digest{};
    std::copy(std::begin(descriptor->schema_digest), std::end(descriptor->schema_digest), digest.begin());
    std::uint64_t received_bytes = 0;
    const auto status = owner->stream_execution->receive_step(descriptor->step_id, digest, absent, descriptor->fields_per_producer, &received_bytes);
    if (status == AMIO_OK) {
        record_stream_counter(*owner, descriptor->step_id, amio::detail::DiagnosticRole::Consumer,
                              amio::detail::DiagnosticCounter::TransportReceivedBytes, received_bytes);
    }
    return status;
}

AMIO_API amio_status_t amio_stream_acquire(amio_context_handle context, uint64_t timeout_milliseconds, amio_stream_acquire_result_t *out_result) {
    if (out_result == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_result = {};
    out_result->struct_size = sizeof(*out_result);
    out_result->version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    if (!owner->stream_reads || timeout_milliseconds > static_cast<uint64_t>(std::chrono::milliseconds::max().count()))
        return AMIO_ERR_INVALID_POLICY;
    const auto acquired = owner->stream_reads->acquire(std::chrono::milliseconds(timeout_milliseconds));
    out_result->state = static_cast<int32_t>(acquired.state);
    out_result->failure_status = acquired.failure;
    if (acquired.skipped) {
        out_result->skipped_first = acquired.skipped->first;
        out_result->skipped_last = acquired.skipped->last;
        const auto count = acquired.skipped->last - acquired.skipped->first + 1;
        record_stream_counter(*owner, acquired.skipped->last, amio::detail::DiagnosticRole::Consumer, amio::detail::DiagnosticCounter::StreamSkips,
                              count);
    }
    if (!acquired.lease) return AMIO_OK;
    record_stream_counter(*owner, acquired.lease->step->step_id, amio::detail::DiagnosticRole::Consumer,
                          amio::detail::DiagnosticCounter::StreamDeliveredBytes, acquired.lease->step->payload_bytes);
    try {
        auto lease = std::make_unique<StreamLeasePayload>();
        lease->context = owner;
        lease->lease = *acquired.lease;
        const auto token = process_handle_table().insert(HandleKind::ConsumerLease, lease.get());
        if (token == 0) {
            (void)owner->stream_execution->release_lease(lease->lease.lease_id);
            return AMIO_ERR_BACKEND_FAILURE;
        }
        ++owner->open_step_handles;
        out_result->lease = reinterpret_cast<amio_step_lease_handle>(HandleTable::to_ptr(token));
        lease.release();
        return AMIO_OK;
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
}

AMIO_API amio_status_t amio_stream_read(amio_step_lease_handle lease, const amio_stream_read_descriptor_t *descriptor, void *buffer,
                                        size_t buffer_capacity, size_t *out_bytes) {
    if (out_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_bytes = 0;
    void *payload = nullptr;
    if (!token_payload(lease, HandleKind::ConsumerLease, &payload)) return lease == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<StreamLeasePayload *>(payload);
    constexpr std::size_t minimum = offsetof(amio_stream_read_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    if (!valid_descriptor_header(descriptor, minimum) || descriptor->reserved32 || descriptor->reserved[0] || descriptor->reserved[1] ||
        descriptor->rank > AMIO_MAX_RANK || descriptor->element_bytes == 0 || descriptor->element_bytes > std::numeric_limits<std::size_t>::max() ||
        (descriptor->logical_field_name_bytes != 0 && descriptor->logical_field_name == nullptr))
        return AMIO_ERR_INVALID_INPUT;
    if (buffer == nullptr && buffer_capacity != 0) return AMIO_ERR_INVALID_INPUT;
    amio::detail::RegionSelection selection;
    selection.offsets.reserve(descriptor->rank);
    selection.extents.reserve(descriptor->rank);
    for (std::size_t i = 0; i < descriptor->rank; ++i) {
        selection.offsets.push_back(descriptor->offsets[i]);
        selection.extents.push_back(descriptor->extents[i]);
    }
    const auto field = "field_" + std::to_string(descriptor->field_id) + "_producer_" + std::to_string(descriptor->producer_rank);
    std::string logical_field_name;
    if (descriptor->logical_field_name_bytes != 0 &&
        !copy_utf8(descriptor->logical_field_name, descriptor->logical_field_name_bytes, logical_field_name))
        return AMIO_ERR_INVALID_INPUT;
    const auto status =
        owner->context->stream_reads->read(owner->lease, field, selection, static_cast<std::size_t>(descriptor->element_bytes),
                                           std::span<std::byte>(static_cast<std::byte *>(buffer), buffer_capacity), *out_bytes, logical_field_name);
    if (status == AMIO_OK)
        record_stream_counter(*owner->context, owner->lease.step->step_id, amio::detail::DiagnosticRole::Consumer,
                              amio::detail::DiagnosticCounter::ViewBytes, *out_bytes);
    return status;
}

AMIO_API amio_status_t amio_stream_release(amio_step_lease_handle lease) {
    void *payload = nullptr;
    if (!token_payload(lease, HandleKind::ConsumerLease, &payload)) return lease == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<StreamLeasePayload *>(payload);
    const auto step_id = owner->lease.step->step_id;
    const auto bytes = owner->lease.step->payload_bytes;
    const auto status = owner->context->stream_execution->release_lease(owner->lease.lease_id);
    if (status != AMIO_OK) return status;
    record_stream_counter(*owner->context, step_id, amio::detail::DiagnosticRole::Consumer, amio::detail::DiagnosticCounter::StreamReleasedBytes,
                          bytes);
    void *released = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(lease)));
    const auto release_status = process_handle_table().release(token, HandleKind::ConsumerLease, &released);
    if (release_status != AMIO_OK) return release_status;
    --owner->context->open_step_handles;
    delete static_cast<StreamLeasePayload *>(released);
    return AMIO_OK;
}

AMIO_API amio_status_t amio_stream_finish(amio_context_handle context, int32_t graceful) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    if (!owner->stream_execution || (graceful != 0 && graceful != 1)) return AMIO_ERR_INVALID_POLICY;
    if (owner->context->role_map().role(owner->context->rank()) == amio::detail::RankRole::Consumer) {
        const auto status = owner->stream_execution->finish_consumer();
        if (status == AMIO_ERR_STREAM_DISCONNECTED || status == AMIO_ERR_STEP_FAILED)
            record_stream_counter(*owner, 0, amio::detail::DiagnosticRole::Consumer, amio::detail::DiagnosticCounter::StreamDisconnects);
        return status;
    }
    return owner->stream_execution->finish_producer(graceful != 0);
}

AMIO_API amio_status_t amio_stream_cancel(amio_context_handle context) {
    void *payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(payload);
    return owner->stream_execution ? owner->stream_execution->cancel_producer() : AMIO_ERR_INVALID_POLICY;
}

AMIO_API amio_status_t amio_open_dataset_reader(amio_context_handle context, const char *location, size_t location_bytes,
                                                amio_dataset_handle *out_dataset) {
    if (out_dataset == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_dataset = nullptr;
    void *context_payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &context_payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *context_owner = static_cast<ContextPayload *>(context_payload);
    if (location == nullptr || location_bytes == 0 || location_bytes > 1024 * 1024) return AMIO_ERR_INVALID_INPUT;
    std::string path_text;
    if (!copy_utf8(location, location_bytes, path_text) || path_text.find('\n') != std::string::npos || path_text.find('\r') != std::string::npos)
        return AMIO_ERR_INVALID_INPUT;
    auto bounded_read = [](const std::filesystem::path &path, std::size_t limit, std::string &out) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return false;
        struct stat st{};
        bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) <= limit;
        if (ok) {
            out.assign(static_cast<std::size_t>(st.st_size), '\0');
            std::size_t at = 0;
            while (at < out.size()) {
                const auto n = ::read(fd, out.data() + at, out.size() - at);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    ok = false;
                    break;
                }
                if (n == 0) {
                    ok = false;
                    break;
                }
                at += static_cast<std::size_t>(n);
            }
        }
        return ::close(fd) == 0 && ok;
    };
    const std::filesystem::path root(path_text);
    std::string identity;
    if (!bounded_read(root / "amio.json", 4096, identity)) return AMIO_ERR_MANIFEST_NOT_FOUND;
    constexpr std::string_view marker = "\"dataset_uuid\":\"";
    const auto uuid_at = identity.find(marker);
    if (uuid_at == std::string::npos) return AMIO_ERR_MANIFEST_INVALID;
    const auto uuid_begin = uuid_at + marker.size();
    const auto uuid_end = identity.find('"', uuid_begin);
    if (uuid_end == std::string::npos || uuid_end == uuid_begin) return AMIO_ERR_MANIFEST_INVALID;
    const std::string uuid = identity.substr(uuid_begin, uuid_end - uuid_begin);
    auto dataset = std::make_unique<EvolutionDatasetPayload>();
    dataset->context = context_owner;
    dataset->location = path_text;
    dataset->uuid = uuid;
    dataset->read_only = true;
    dataset->payload_budget_bytes = 64u * 1024u * 1024u;
    try {
        dataset->budget =
            std::make_unique<amio::detail::ResourceBudget>(amio::detail::ResourceBudget::Limits{64u * 1024u * 1024u, 4096, 1u * 1024u * 1024u, 1024});
        amio::detail::CollectionReader reader(root, uuid);
        amio::detail::CatalogCursor cursor;
        bool any_step = false;
        bool format_known = false;
        for (;;) {
            auto page = reader.list_steps(cursor, 64);
            if (page.status != AMIO_OK) return AMIO_ERR_BACKEND_FAILURE;
            for (const auto &step : page.steps) {
                any_step = true;
                for (const auto &payload : step.completion.payloads) {
                    const auto inferred =
                        payload.array_container ? AMIO_FORMAT_ZARR3 : (!payload.message_ranges.empty() ? AMIO_FORMAT_GRIB2 : AMIO_FORMAT_NETCDF4);
                    if (format_known && dataset->format != inferred) return AMIO_ERR_MANIFEST_INVALID;
                    dataset->format = static_cast<amio_evolution_format_t>(inferred);
                    format_known = true;
                }
                if (!dataset->schemas.contains(step.completion.schema_digest)) {
                    std::string encoded;
                    const auto schema_path = root / "payloads" / "schemas" / (step.completion.schema_digest + ".json");
                    if (!bounded_read(schema_path, amio::detail::Schema::kMaxCanonicalBytes + 1024, encoded)) return AMIO_ERR_SCHEMA_MISMATCH;
                    amio::detail::SchemaDocument document;
                    if (amio::detail::parse_schema_document(encoded, document) != AMIO_OK || document.dataset_uuid != uuid ||
                        document.digest != step.completion.schema_digest)
                        return AMIO_ERR_SCHEMA_MISMATCH;
                    auto schema = std::make_shared<amio::detail::Schema>();
                    if (amio::detail::Schema::from_canonical(document.canonical_schema, amio::detail::Schema::kMaxCanonicalBytes, *schema).ok() ==
                        false)
                        return AMIO_ERR_SCHEMA_MISMATCH;
                    dataset->schemas.emplace(step.completion.schema_digest, std::move(schema));
                }
            }
            if (!page.has_more) break;
            cursor = std::move(page.next);
        }
        if (!any_step || !format_known) return AMIO_ERR_UNSUPPORTED;
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    const auto token = process_handle_table().insert(HandleKind::EvolutionDataset, dataset.get());
    if (token == 0) return AMIO_ERR_BACKEND_FAILURE;
    context_owner->open_datasets.fetch_add(1);
    *out_dataset = reinterpret_cast<amio_dataset_handle>(HandleTable::to_ptr(token));
    dataset.release();
    return AMIO_OK;
}

AMIO_API amio_status_t amio_get_step_model_time(amio_dataset_handle dataset, uint64_t step_id, double *out_model_time) {
    if (out_model_time == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_model_time = 0.0;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
                                                      HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return dataset == nullptr ? AMIO_ERR_NULL_HANDLE : lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    if (owner->format != AMIO_FORMAT_NETCDF4) return AMIO_ERR_UNSUPPORTED;
    amio::detail::CollectionReader reader(owner->location, owner->uuid);
    amio::detail::CatalogStep step;
    auto status = reader.describe_step(step_id, step);
    if (status != AMIO_OK) return status;
    if (step.completion.payloads.empty()) return AMIO_ERR_STEP_FAILED;
    const auto path = (std::filesystem::path(owner->location) / step.completion.payloads.front().relative_path).string();
    return amio::detail::NetCDF_Driver::read_step_model_time(path, *out_model_time) ? AMIO_OK : AMIO_ERR_UNSUPPORTED;
}

AMIO_API amio_status_t amio_open_dataset_ex(amio_context_handle context, const amio_dataset_descriptor_t *descriptor,
                                            amio_dataset_handle *out_dataset) {
    if (out_dataset == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_dataset = nullptr;
    void *context_payload = nullptr;
    if (!token_payload(context, HandleKind::Context, &context_payload)) return context == nullptr ? AMIO_ERR_NULL_HANDLE : AMIO_ERR_INVALID_HANDLE;
    auto *owner = static_cast<ContextPayload *>(context_payload);
    constexpr std::size_t minimum = offsetof(amio_dataset_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    amio_status_t local_status = AMIO_OK;
    if (!owner->context->can_open_storage()) local_status = AMIO_ERR_INVALID_POLICY;
    if (owner->context->roles().mode != amio::detail::ExecutionMode::Direct) local_status = AMIO_ERR_UNSUPPORTED;
    if (!valid_descriptor_header(descriptor, minimum))
        local_status = AMIO_ERR_INVALID_INPUT;
    else if (descriptor->reserved32 != 0 || descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0 || descriptor->location_bytes == 0 ||
             descriptor->location_bytes > 1024 * 1024 || descriptor->location == nullptr || descriptor->budget_bytes == 0 ||
             descriptor->budget_descriptors == 0 || descriptor->control_budget_bytes > descriptor->budget_bytes ||
             descriptor->control_budget_descriptors > descriptor->budget_descriptors ||
             descriptor->budget_bytes > std::numeric_limits<std::size_t>::max() ||
             descriptor->budget_descriptors > std::numeric_limits<std::size_t>::max() ||
             descriptor->control_budget_bytes > std::numeric_limits<std::size_t>::max() ||
             descriptor->control_budget_descriptors > std::numeric_limits<std::size_t>::max())
        local_status = AMIO_ERR_INVALID_INPUT;
    else if (descriptor->format != AMIO_FORMAT_NETCDF4 && descriptor->format != AMIO_FORMAT_ZARR3 && descriptor->format != AMIO_FORMAT_GRIB2)
        local_status = AMIO_ERR_UNSUPPORTED;
    else if (owner->context->size() > 1 && descriptor->format != AMIO_FORMAT_NETCDF4 &&
             !(descriptor->format == AMIO_FORMAT_ZARR3 && kTensorStoreAvailable))
        local_status = AMIO_ERR_UNSUPPORTED;
#if !defined(AMIO_HAS_MPI) || !defined(AMIO_HAS_PARALLEL_NETCDF)
    else if (owner->context->size() > 1 && descriptor->format == AMIO_FORMAT_NETCDF4)
        local_status = AMIO_ERR_UNSUPPORTED;
#endif
#ifndef AMIO_HAS_TENSORSTORE
    else if (descriptor->format == AMIO_FORMAT_ZARR3)
        local_status = AMIO_ERR_UNSUPPORTED;
#endif
#ifndef AMIO_HAS_G2C
    else if (descriptor->format == AMIO_FORMAT_GRIB2)
        local_status = AMIO_ERR_UNSUPPORTED;
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
            if (level == "visibility")
                dataset->requested_sync = amio::detail::PosixSyncLevel::Visibility;
            else if (level == "file_sync")
                dataset->requested_sync = amio::detail::PosixSyncLevel::FileSync;
            else if (level == "directory_sync")
                dataset->requested_sync = amio::detail::PosixSyncLevel::DirectorySync;
            else
                local_status = AMIO_ERR_INVALID_POLICY;
        }
        if (dataset->format == AMIO_FORMAT_ZARR3 && dataset->requested_sync != amio::detail::PosixSyncLevel::Visibility)
            local_status = AMIO_ERR_UNSUPPORTED;
        if (!copy_utf8(descriptor->location, descriptor->location_bytes, dataset->location) || dataset->location.find('\n') != std::string::npos ||
            dataset->location.find('\r') != std::string::npos)
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
    for (std::size_t i = 0; i < 5; ++i)
        if (min_values[i] != max_values[i]) return AMIO_ERR_INVALID_POLICY;
#endif
    agreed_status = compare_schema_collectively(*owner->context, dataset->location);
    if (agreed_status != AMIO_OK) return agreed_status;

    int root_status = AMIO_OK;
    if (owner->context->rank() == owner->context->storage_root_rank()) {
        try {
            dataset->uuid = random_identity("dataset-");
        } catch (...) {
            root_status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
#ifdef AMIO_HAS_MPI
    const int storage_root = static_cast<int>(owner->context->storage_root_rank());
    if (MPI_Bcast(&root_status, 1, MPI_INT, storage_root, owner->context->communicator()) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
    constexpr std::size_t uuid_capacity = 64;
    char uuid_buffer[uuid_capacity] = {};
    if (owner->context->rank() == owner->context->storage_root_rank() && root_status == AMIO_OK) {
        if (dataset->uuid.size() >= uuid_capacity)
            root_status = AMIO_ERR_BACKEND_FAILURE;
        else
            std::memcpy(uuid_buffer, dataset->uuid.data(), dataset->uuid.size());
    }
    if (MPI_Bcast(&root_status, 1, MPI_INT, storage_root, owner->context->communicator()) != MPI_SUCCESS ||
        MPI_Bcast(uuid_buffer, static_cast<int>(uuid_capacity), MPI_CHAR, storage_root, owner->context->communicator()) != MPI_SUCCESS)
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

    if (owner->context->rank() == owner->context->storage_root_rank()) {
        try {
            const auto owner_token = random_identity("owner-");
            dataset->catalog = std::make_unique<amio::detail::PosixCatalog>(dataset->location, dataset->uuid, owner_token);
            root_status = dataset->catalog->create_exclusive();
        } catch (...) {
            root_status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
#ifdef AMIO_HAS_MPI
    if (MPI_Bcast(&root_status, 1, MPI_INT, storage_root, owner->context->communicator()) != MPI_SUCCESS) return AMIO_ERR_COMM_SPLIT_FAILED;
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
    if (!evolution_dataset->read_only) {
        const auto state = evolution_dataset->steps->state();
        if (state == amio::detail::StepState::Open || state == amio::detail::StepState::Sealed || state == amio::detail::StepState::Transferring ||
            state == amio::detail::StepState::PayloadReady || state == amio::detail::StepState::Publishing ||
            state == amio::detail::StepState::Reconciling)
            return AMIO_ERR_INVALID_POLICY;
    }
    void *released = nullptr;
    const auto status = process_handle_table().release(token, HandleKind::EvolutionDataset, &released);
    if (status != AMIO_OK) return status;
    auto *closed = static_cast<EvolutionDatasetPayload *>(released);
    lock.unlock();
    closed->context->open_datasets.fetch_sub(1);
    delete closed;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_define_schema(amio_dataset_handle dataset, const amio_schema_descriptor_t *descriptor, amio_schema_handle *out_schema) {
    if (out_schema == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_schema = nullptr;
    void *dataset_payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
                                                      HandleKind::EvolutionDataset, &dataset_payload);
    if (lookup != AMIO_OK) return dataset == nullptr ? AMIO_ERR_NULL_HANDLE : lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(dataset_payload);
    std::lock_guard<std::mutex> dataset_lock(owner->mutex);
    if (owner->read_only) return AMIO_ERR_INVALID_POLICY;

    constexpr std::size_t minimum = offsetof(amio_schema_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    amio_status_t local_status = AMIO_OK;
    if (!valid_descriptor_header(descriptor, minimum))
        local_status = AMIO_ERR_INVALID_INPUT;
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
            std::size_t input_bytes = 0;
            const auto charge_input = [&](std::uint64_t bytes) {
                const auto limit = owner->budget->limits().control_bytes;
                if (bytes > limit - input_bytes) return false;
                input_bytes += static_cast<std::size_t>(bytes);
                return true;
            };
            for (std::uint64_t i = 0; i < descriptor->variable_count; ++i) {
                const auto &input = descriptor->variables[i];
                constexpr std::size_t var_minimum = offsetof(amio_variable_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
                if (!valid_descriptor_header(&input, var_minimum) || input.reserved32 != 0 || input.reserved[0] != 0 || input.reserved[1] != 0 ||
                    input.required > 1 || input.dimension_count == 0 || input.dimension_count > AMIO_MAX_RANK || input.dimensions == nullptr ||
                    input.attribute_count > amio::detail::Schema::kMaxAttributesPerVariable ||
                    (input.attribute_count != 0 && input.attributes == nullptr))
                    return AMIO_ERR_INVALID_INPUT;
                amio::detail::Variable variable;
                if (!charge_input(input.name_bytes)) return AMIO_ERR_BUDGET_EXCEEDED;
                if (!copy_utf8(input.name, input.name_bytes, variable.name)) return AMIO_ERR_INVALID_INPUT;
                if (input.source_dtype > AMIO_DTYPE_U64 || input.stored_dtype > AMIO_DTYPE_U64) return AMIO_ERR_INVALID_INPUT;
                variable.source_dtype = static_cast<amio_dtype_t>(input.source_dtype);
                variable.stored_dtype = static_cast<amio_dtype_t>(input.stored_dtype);
                variable.required = input.required != 0;
                variable.dimensions.reserve(input.dimension_count);
                for (std::uint32_t d = 0; d < input.dimension_count; ++d) {
                    const auto &dim = input.dimensions[d];
                    constexpr std::size_t dim_minimum = offsetof(amio_dimension_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
                    if (!valid_descriptor_header(&dim, dim_minimum) || dim.reserved[0] != 0 || dim.reserved[1] != 0) return AMIO_ERR_INVALID_INPUT;
                    amio::detail::Dimension dimension;
                    if (!charge_input(dim.name_bytes) || !charge_input(dim.coordinate_variable_bytes)) return AMIO_ERR_BUDGET_EXCEEDED;
                    if (!copy_utf8(dim.name, dim.name_bytes, dimension.name)) return AMIO_ERR_INVALID_INPUT;
                    dimension.extent = dim.extent;
                    if (dim.coordinate_variable_bytes != 0) {
                        std::string coordinate;
                        if (!copy_utf8(dim.coordinate_variable, dim.coordinate_variable_bytes, coordinate)) return AMIO_ERR_INVALID_INPUT;
                        dimension.coordinate_variable = std::move(coordinate);
                    } else if (dim.coordinate_variable != nullptr)
                        return AMIO_ERR_INVALID_INPUT;
                    variable.dimensions.push_back(std::move(dimension));
                }
                for (std::uint32_t a = 0; a < input.attribute_count; ++a) {
                    const auto &attr_input = input.attributes[a];
                    amio::detail::Attribute attribute;
                    std::string name;
                    if (!charge_input(attr_input.name_bytes) || !charge_input(attr_input.value_bytes)) return AMIO_ERR_BUDGET_EXCEEDED;
                    if (!copy_utf8(attr_input.name, attr_input.name_bytes, name) || !convert_attribute(attr_input, attribute) ||
                        !variable.attributes.emplace(std::move(name), std::move(attribute)).second)
                        return AMIO_ERR_INVALID_INPUT;
                }
                if (owner->format == AMIO_FORMAT_GRIB2) {
                    if (variable.source_dtype != AMIO_DTYPE_F32 || variable.stored_dtype != AMIO_DTYPE_F32 || variable.dimensions.empty() ||
                        variable.dimensions.size() > 2)
                        return AMIO_ERR_UNSUPPORTED;
                    const auto settings = grib2_settings_for(*owner->context, variable);
                    const auto ni = static_cast<std::int64_t>(variable.dimensions.back().extent);
                    const auto nj =
                        variable.dimensions.size() > 1 ? static_cast<std::int64_t>(variable.dimensions[variable.dimensions.size() - 2].extent) : 1;
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
    const auto existing_schema = owner->schemas.find(schema->digest());
    const bool schema_already_registered = existing_schema != owner->schemas.end();
    if (schema_already_registered && !existing_schema->second->equivalent(*schema)) local_status = AMIO_ERR_SCHEMA_MISMATCH;
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) return agreed_status;
    std::shared_ptr<const amio::detail::Schema> registered_schema = schema_already_registered ? existing_schema->second : schema;
    // Existing handles and the dataset share one charged immutable schema.
    if (schema_already_registered) schema.reset();

    std::optional<amio::detail::ResourceBudget::Reservation> metadata_reservation;
    if (!schema_already_registered) {
        std::size_t retained_bytes = 0;
        if (!schema_metadata_bytes(*registered_schema, retained_bytes)) {
            local_status = AMIO_ERR_BUDGET_EXCEEDED;
        } else {
            metadata_reservation = owner->budget->reserve(retained_bytes, 1, amio::detail::ResourceBudget::Class::Control);
            if (!metadata_reservation) local_status = AMIO_ERR_BUDGET_EXCEEDED;
        }
    }
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) {
        if (metadata_reservation) (void)owner->budget->release(*metadata_reservation);
        return agreed_status;
    }

    int schema_status = AMIO_OK;
    if (owner->context->context->rank() == owner->context->context->storage_root_rank()) {
        try {
            amio::detail::SchemaDocument document;
            document.dataset_uuid = owner->uuid;
            document.digest = registered_schema->digest();
            document.canonical_schema = registered_schema->canonical();
            for (const auto &[name, variable] : registered_schema->variables())
                if (variable.required) document.required_fields.push_back(name);
            std::string encoded;
            schema_status = amio::detail::encode_schema_document(document, encoded);
            if (schema_status == AMIO_OK) {
                const auto path = std::string("payloads/schemas/") + registered_schema->digest() + ".json";
                schema_status =
                    schema_already_registered ? AMIO_OK : owner->catalog->write_payload(path, encoded.data(), encoded.size(), owner->requested_sync);
            }
        } catch (...) {
            schema_status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
#ifdef AMIO_HAS_MPI
    if (MPI_Bcast(&schema_status, 1, MPI_INT, static_cast<int>(owner->context->context->storage_root_rank()),
                  owner->context->context->communicator()) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
#endif
    agreed_status = collective_status(*owner->context->context, static_cast<amio_status_t>(schema_status));
    if (agreed_status != AMIO_OK) {
        if (metadata_reservation) (void)owner->budget->release(*metadata_reservation);
        return agreed_status;
    }

    local_status = owner->steps->register_schema(registered_schema);
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) {
        if (metadata_reservation) (void)owner->budget->release(*metadata_reservation);
        return agreed_status;
    }

    std::unique_ptr<SchemaHandlePayload> handle_payload;
    local_status = AMIO_OK;
    try {
        handle_payload = std::make_unique<SchemaHandlePayload>();
        handle_payload->schema = registered_schema;
        owner->schemas[registered_schema->digest()] = registered_schema;
        if (metadata_reservation) owner->schema_reservations.emplace(registered_schema->digest(), *metadata_reservation);
    } catch (...) {
        local_status = AMIO_ERR_BACKEND_FAILURE;
    }
    agreed_status = collective_status(*owner->context->context, local_status);
    if (agreed_status != AMIO_OK) {
        if (metadata_reservation) {
            owner->schemas.erase(registered_schema->digest());
            owner->schema_reservations.erase(registered_schema->digest());
            (void)owner->budget->release(*metadata_reservation);
        }
        return agreed_status;
    }

    if (owner->context->diagnostics) {
        std::size_t metadata_bytes = 0;
        for (const auto &[digest, reservation] : owner->schema_reservations) {
            (void)digest;
            metadata_bytes = reservation.bytes > std::numeric_limits<std::size_t>::max() - metadata_bytes ? std::numeric_limits<std::size_t>::max()
                                                                                                          : metadata_bytes + reservation.bytes;
        }
        amio::detail::DiagnosticScope scope;
        scope.context_epoch = 1;
        scope.dataset_id = std::hash<std::string>{}(owner->uuid);
        scope.role = owner->context->context->role_map().role(owner->context->context->rank()) == amio::detail::RankRole::Writer
                         ? amio::detail::DiagnosticRole::Writer
                         : amio::detail::DiagnosticRole::Producer;
        scope.participant = owner->context->context->rank();
        owner->context->diagnostics->set_resource(scope, amio::detail::DiagnosticResource::MetadataBytes, metadata_bytes,
                                                  owner->budget->limits().control_bytes);
    }

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
    const auto status = process_handle_table().release(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(schema))),
                                                       HandleKind::Schema, &payload);
    if (status != AMIO_OK) return status;
    delete static_cast<SchemaHandlePayload *>(payload);
    return AMIO_OK;
}

static amio_status_t begin_step_impl(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor,
                                     const amio_field_partition_descriptor_t *partitions, size_t partition_count, bool partitioned,
                                     amio_step_handle *out_step) {
    if (out_step == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_step = nullptr;
    void *dataset_payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
                                                      HandleKind::EvolutionDataset, &dataset_payload);
    if (lookup != AMIO_OK) return dataset == nullptr ? AMIO_ERR_NULL_HANDLE : lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(dataset_payload);
    if (owner->read_only) return AMIO_ERR_INVALID_POLICY;
    if (partitioned && (!is_distributed_output(*owner) || partitions == nullptr || partition_count == 0)) return AMIO_ERR_UNSUPPORTED;
    if (!partitioned && is_distributed_output(*owner)) return AMIO_ERR_INVALID_POLICY;
    constexpr std::size_t minimum = offsetof(amio_step_descriptor_t, reserved) + 2 * sizeof(std::uint64_t);
    if (!valid_descriptor_header(descriptor, minimum) || descriptor->reserved[0] != 0 || descriptor->reserved[1] != 0 ||
        descriptor->schema == nullptr)
        return AMIO_ERR_INVALID_INPUT;

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
        amio::detail::CompiledFieldPipeline pipeline;
        const auto pipeline_status = amio::detail::compile_variable_pipeline(variable, pipeline, operator_destination(owner->format));
        if (pipeline_status != AMIO_OK) return pipeline_status;
        if (pipeline.request.codec == amio::detail::NativeCodec::Grib2Aec || pipeline.request.codec == amio::detail::NativeCodec::Grib2Jpeg2000) {
            if (variable.stored_dtype != AMIO_DTYPE_F32) return AMIO_ERR_UNSUPPORTED;
            try {
                if (grib2_settings_for(*owner->context, variable).decimal_scale_factor != 0) return AMIO_ERR_INVALID_POLICY;
            } catch (...) {
                return AMIO_ERR_INVALID_INPUT;
            }
        }
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
                if (!valid_descriptor_header(&candidate, part_minimum) || candidate.reserved32 != 0 || candidate.reserved[0] != 0 ||
                    candidate.reserved[1] != 0)
                    return AMIO_ERR_INVALID_INPUT;
                std::string candidate_name;
                if (!copy_utf8(candidate.field, candidate.field_bytes, candidate_name)) return AMIO_ERR_INVALID_INPUT;
                if (candidate_name == name) {
                    if (input != nullptr) return AMIO_ERR_INVALID_INPUT;
                    input = &candidate;
                }
            }
            if (input == nullptr || input->rank != variable.dimensions.size() || input->rank > AMIO_MAX_RANK) return AMIO_ERR_INVALID_INPUT;
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
                    if (extent == 0 || extent > std::numeric_limits<std::size_t>::max() / elements) return AMIO_ERR_INVALID_INPUT;
                    elements *= extent;
                }
                const auto width = dtype_size(variable.source_dtype);
                if (width == 0 || width > std::numeric_limits<std::size_t>::max() / elements) return AMIO_ERR_INVALID_INPUT;
                declaration.snapshot_bytes = static_cast<std::size_t>(elements * width);
                if (declaration.snapshot_bytes > std::numeric_limits<std::size_t>::max() - request.snapshot_bytes) return AMIO_ERR_INVALID_INPUT;
                request.snapshot_bytes += declaration.snapshot_bytes;
            }
            step_payload->partitions.emplace(name, declaration.region);
        }
        request.contributions.push_back(std::move(declaration));
    }
    if (partitioned && partition_count != schema->variables().size()) return AMIO_ERR_INVALID_INPUT;
    if (owner->distributed_zarr) {
        const auto shard_status = derive_distributed_zarr_shards(*owner, *step_payload, *schema, owner->zarr_shard_shapes);
        const auto agreed_shard_status = collective_status(*owner->context->context, shard_status);
        if (agreed_shard_status != AMIO_OK) return agreed_shard_status;
    }
    const auto local_coordinator_status = owner->steps->begin_step(request);
    const auto begin_status = partitioned ? collective_status(*owner->context->context, local_coordinator_status) : local_coordinator_status;
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

AMIO_API amio_status_t amio_begin_step(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor, amio_step_handle *out_step) {
    return begin_step_impl(dataset, descriptor, nullptr, 0, false, out_step);
}

AMIO_API amio_status_t amio_begin_step_partitioned(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor,
                                                   const amio_field_partition_descriptor_t *partitions, size_t partition_count,
                                                   amio_step_handle *out_step) {
    return begin_step_impl(dataset, descriptor, partitions, partition_count, true, out_step);
}

AMIO_API amio_status_t amio_put_field(amio_step_handle step, const char *field, size_t field_bytes, const void *data, size_t data_bytes,
                                      amio_operation_handle *out_operation) {
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
    const auto started = std::chrono::steady_clock::now();
    const auto diagnostic_field_id = static_cast<std::uint32_t>(std::hash<std::string>{}(field_name));
    if (owner->context->diagnostics) {
        try {
            producer_step->diagnostic_field_ids.try_emplace(field_name, diagnostic_field_id);
        } catch (...) {
            return AMIO_ERR_BACKEND_FAILURE;
        }
    }
    const auto result = owner->steps->put(field_name, data, data_bytes);
    if (owner->context->diagnostics) {
        amio::detail::DiagnosticScope scope;
        scope.context_epoch = 1;
        scope.dataset_id = std::hash<std::string>{}(owner->uuid);
        scope.step_id = producer_step->step_id;
        scope.field_id = diagnostic_field_id;
        scope.role = amio::detail::DiagnosticRole::Producer;
        scope.participant = owner->context->context->rank();
        const auto elapsed = std::chrono::steady_clock::now() - started;
        const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        owner->context->diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::AdmissionNs,
                                                      elapsed_ns < 0 ? 0 : static_cast<std::uint64_t>(elapsed_ns));
        if (result.status == AMIO_OK) {
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::Accepted);
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::BytesAccepted, data_bytes);
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::SourceReleasedBytes, data_bytes);
            // put() returns only after the caller's bytes have been copied into
            // AMIO-owned storage, which is the source-release boundary. Keep
            // this separate from end_step publication latency so diagnostics
            // report when the model may safely reuse its input buffer.
            owner->context->diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::SnapshotNs,
                                                          elapsed_ns < 0 ? 0 : static_cast<std::uint64_t>(elapsed_ns));
            owner->context->diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::SourceReleaseNs,
                                                          elapsed_ns < 0 ? 0 : static_cast<std::uint64_t>(elapsed_ns));
            std::uint64_t retained_bytes = 0;
            for (const auto &[name, contribution] : owner->steps->contributions()) {
                (void)name;
                const auto bytes = static_cast<std::uint64_t>(contribution.bytes.size());
                retained_bytes = bytes > std::numeric_limits<std::uint64_t>::max() - retained_bytes ? std::numeric_limits<std::uint64_t>::max()
                                                                                                    : retained_bytes + bytes;
            }
            owner->context->diagnostics->set_resource(scope, amio::detail::DiagnosticResource::StagingRetainedBytes, retained_bytes,
                                                      owner->payload_budget_bytes);
            owner->context->diagnostics->set_resource(scope, amio::detail::DiagnosticResource::ActiveOperations, owner->steps->contributions().size(),
                                                      owner->budget ? owner->budget->limits().descriptors : 0);
            owner->context->diagnostics->record_event(scope, amio::detail::DiagnosticEventKind::Admission, "field snapshot admitted");
        } else {
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::Rejected);
            owner->context->diagnostics->record_error(scope, result.status, "field snapshot admission failed");
        }
    }
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

AMIO_API amio_status_t amio_put_field_memory(amio_step_handle step, const char *field, size_t field_bytes, const amio_memory_descriptor_t *memory,
                                             const void *data, size_t data_bytes, amio_operation_handle *out_operation) {
    if (out_operation == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_operation = nullptr;
    if (memory == nullptr) return AMIO_ERR_INVALID_INPUT;
    if (memory->struct_size != sizeof(amio_memory_descriptor_t) || memory->version != 1 || memory->reserved32 != 0 || memory->reserved[0] != 0 ||
        memory->reserved[1] != 0 || memory->rank == 0 || memory->rank > AMIO_MAX_RANK || memory->producer_writes_synchronized > 1)
        return AMIO_ERR_INVALID_INPUT;
    amio::detail::MemoryDescriptor descriptor;
    descriptor.kind = static_cast<amio::detail::MemoryKind>(memory->memory_kind);
    descriptor.ownership = static_cast<amio::detail::BufferOwnership>(memory->ownership);
    descriptor.device_id = memory->device_id;
    descriptor.producer_writes_synchronized = memory->producer_writes_synchronized != 0;
    descriptor.rank = memory->rank;
    descriptor.element_bytes = memory->element_bytes;
    try {
        descriptor.extents.assign(memory->extents, memory->extents + memory->rank);
        descriptor.strides.assign(memory->strides, memory->strides + memory->rank);
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    const auto valid = amio::detail::validate_memory_descriptor(descriptor, data_bytes);
    if (valid != AMIO_OK) return valid;
#ifndef AMIO_HAS_KOKKOS_DEVICE
    if (descriptor.kind == amio::detail::MemoryKind::Device) return AMIO_ERR_UNSUPPORTED;
#endif
    if (step == nullptr) return AMIO_ERR_NULL_HANDLE;
    std::string field_name;
    if (!copy_utf8(field, field_bytes, field_name)) return AMIO_ERR_INVALID_INPUT;
    void *step_payload = nullptr;
    const auto token = HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(step)));
    const auto lookup = process_handle_table().lookup(token, HandleKind::ProducerStep, &step_payload);
    if (lookup != AMIO_OK) return lookup;
    auto *producer_step = static_cast<ProducerStepPayload *>(step_payload);
    if (producer_step->ended) return AMIO_ERR_INVALID_POLICY;
    auto *owner = producer_step->dataset;
    std::lock_guard<std::mutex> lock(owner->mutex);
    const auto started = std::chrono::steady_clock::now();
    const auto field_id = static_cast<std::uint32_t>(std::hash<std::string>{}(field_name));
    const auto result = owner->steps->put_memory(field_name, descriptor, data, data_bytes);
    if (owner->context->diagnostics) {
        amio::detail::DiagnosticScope scope;
        scope.context_epoch = 1;
        scope.dataset_id = std::hash<std::string>{}(owner->uuid);
        scope.step_id = producer_step->step_id;
        scope.field_id = field_id;
        scope.role = amio::detail::DiagnosticRole::Producer;
        scope.participant = owner->context->context->rank();
        if (result.status == AMIO_OK) {
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::Accepted);
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::BytesAccepted, data_bytes);
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::SourceReleasedBytes, data_bytes);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
            const auto duration = elapsed_ns < 0 ? 0 : static_cast<std::uint64_t>(elapsed_ns);
            owner->context->diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::SnapshotNs, duration);
            owner->context->diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::SourceReleaseNs, duration);
            if (result.used_fallback)
                owner->context->diagnostics->record_event(scope, amio::detail::DiagnosticEventKind::Snapshot,
                                                          "device payload staged into owned host memory via Kokkos fallback");
        } else {
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::Rejected);
            owner->context->diagnostics->record_error(scope, result.status, "memory field snapshot admission failed");
        }
    }
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
        if (MPI_Allreduce(&locally_cancellable, &all_cancellable, 1, MPI_INT, MPI_MIN, owner->context->context->communicator()) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        if (!all_cancellable) return AMIO_ERR_INVALID_POLICY;
    } else
#endif
        if (producer_step->ended)
        return AMIO_ERR_INVALID_POLICY;
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
    std::uint64_t contribution_bytes = 0;
    for (const auto &[name, contribution] : owner->steps->contributions()) {
        (void)name;
        const auto amount = static_cast<std::uint64_t>(contribution.bytes.size());
        contribution_bytes = amount > std::numeric_limits<std::uint64_t>::max() - contribution_bytes ? std::numeric_limits<std::uint64_t>::max()
                                                                                                     : contribution_bytes + amount;
    }
    const auto publication_started = std::chrono::steady_clock::now();
    auto record_publication = [&]() {
        const auto elapsed = std::chrono::steady_clock::now() - publication_started;
        const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        record_step_publication(*owner->context, *owner, *producer_step, contribution_bytes,
                                elapsed_ns < 0 ? 0 : static_cast<std::uint64_t>(elapsed_ns));
    };
    auto &coordinator = *owner->steps;
#ifdef AMIO_HAS_MPI
    if (producer_step->partitioned) {
        const auto state = coordinator.state();
        const int locally_valid =
            !producer_step->ended && (state == amio::detail::StepState::Open || state == amio::detail::StepState::Reconciling) ? 1 : 0;
        int all_valid = 0;
        if (MPI_Allreduce(&locally_valid, &all_valid, 1, MPI_INT, MPI_MIN, owner->context->context->communicator()) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        if (!all_valid) return AMIO_ERR_INVALID_POLICY;
    } else
#endif
        if (producer_step->ended)
        return AMIO_ERR_INVALID_POLICY;

    if (coordinator.state() == amio::detail::StepState::Reconciling) {
        amio::detail::PublicationResult reconciled;
        if (owner->context->context->rank() == owner->context->context->storage_root_rank())
            reconciled = owner->catalog->reconcile(owner->pending_commit);
        const auto broadcast_status = broadcast_publication(*owner->context->context, reconciled);
        if (broadcast_status != AMIO_OK) return broadcast_status;
        if (reconciled.state != amio::detail::PublicationState::Complete || reconciled.status != AMIO_OK) return AMIO_ERR_PUBLICATION_PENDING;
        const auto status = coordinator.mark_publication_visible(owner->requested_sync != amio::detail::PosixSyncLevel::Visibility);
        if (status != AMIO_OK) return status;
        owner->previous_commit = owner->pending_commit;
        owner->pending_commit.clear();
        producer_step->ended = true;
        record_publication();
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
    ScopedBudgetReservation operator_reservation{owner->budget.get(), std::nullopt};
    std::map<std::string, amio::detail::ExecutedField, std::less<>> operated_fields;
    amio_status_t operator_status = AMIO_OK;
    const auto operator_started = std::chrono::steady_clock::now();
    std::size_t requested_operator_scratch = 0;
    std::size_t operator_scratch_descriptors = 0;
    std::string failed_operator_field;
    for (const auto &[name, contribution] : coordinator.contributions()) {
        const auto &variable = schema_it->second->variables().at(name);
        if (variable.source_dtype == variable.stored_dtype) continue;
        if (contribution.bytes.size() > std::numeric_limits<std::size_t>::max() / 2 ||
            contribution.bytes.size() * 2 > std::numeric_limits<std::size_t>::max() - requested_operator_scratch) {
            operator_status = AMIO_ERR_BUDGET_EXCEEDED;
            break;
        }
        requested_operator_scratch += contribution.bytes.size() * 2;
        ++operator_scratch_descriptors;
    }
    if (operator_status == AMIO_OK && requested_operator_scratch != 0) {
        operator_reservation.reservation = owner->budget->reserve(requested_operator_scratch, operator_scratch_descriptors);
        if (!operator_reservation.reservation) operator_status = AMIO_ERR_BUDGET_EXCEEDED;
    }
    if (producer_step->partitioned) operator_status = collective_status(*owner->context->context, operator_status);
    try {
        for (const auto &[name, contribution] : coordinator.contributions()) {
            if (operator_status != AMIO_OK) break;
            const auto variable = schema_it->second->variables().find(name);
            if (variable == schema_it->second->variables().end()) {
                operator_status = AMIO_ERR_SCHEMA_MISMATCH;
                failed_operator_field = name;
                break;
            }
            amio::detail::ExecutedField result;
            operator_status = amio::detail::execute_field_pipeline(variable->second, contribution.bytes, contribution.region.offsets,
                                                                   contribution.region.extents, static_cast<std::size_t>(owner->payload_budget_bytes),
                                                                   result, operator_destination(owner->format));
            if (operator_status != AMIO_OK) {
                failed_operator_field = name;
                break;
            }
            operated_fields.emplace(name, std::move(result));
        }
    } catch (...) {
        operator_status = AMIO_ERR_BACKEND_FAILURE;
    }
    if (producer_step->partitioned) operator_status = collective_status(*owner->context->context, operator_status);
    if (operator_status != AMIO_OK) {
        if (owner->context->diagnostics) {
            amio::detail::DiagnosticScope scope;
            scope.context_epoch = 1;
            scope.dataset_id = std::hash<std::string>{}(owner->uuid);
            scope.step_id = coordinator.active_step_id();
            scope.field_id = failed_operator_field.empty() ? 0 : static_cast<std::uint32_t>(std::hash<std::string>{}(failed_operator_field));
            scope.role = amio::detail::DiagnosticRole::Producer;
            scope.participant = owner->context->context->rank();
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::Failed);
            owner->context->diagnostics->record_error(scope, operator_status,
                                                      failed_operator_field.empty() ? "operator preflight or scratch reservation failed"
                                                                                    : "field operator accuracy or integrity processing failed");
        }
        (void)coordinator.poison_dataset(operator_status);
        return operator_status;
    }
    IntegrityRegionMap partition_integrity_regions;
    ScopedBudgetReservation integrity_metadata_reservation{owner->budget.get(), std::nullopt};
    if (producer_step->partitioned) {
        auto integrity_status = gather_integrity_regions(*owner, *producer_step, *schema_it->second, operated_fields, partition_integrity_regions,
                                                         integrity_metadata_reservation);
        integrity_status = collective_status(*owner->context->context, integrity_status);
        if (integrity_status != AMIO_OK) {
            (void)coordinator.poison_dataset(integrity_status);
            return integrity_status;
        }
    }
    if (owner->context->diagnostics) {
        const auto elapsed = std::chrono::steady_clock::now() - operator_started;
        const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        for (const auto &[name, executed] : operated_fields) {
            amio::detail::DiagnosticScope scope;
            scope.context_epoch = 1;
            scope.dataset_id = std::hash<std::string>{}(owner->uuid);
            scope.step_id = coordinator.active_step_id();
            scope.field_id = static_cast<std::uint32_t>(std::hash<std::string>{}(name));
            scope.role = amio::detail::DiagnosticRole::Producer;
            scope.participant = owner->context->context->rank();
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::OperatorInputBytes, executed.input_bytes);
            owner->context->diagnostics->add_counter(scope, amio::detail::DiagnosticCounter::OperatorOutputBytes, executed.output_bytes);
            owner->context->diagnostics->observe_duration(scope, amio::detail::DiagnosticHistogram::OperatorNs,
                                                          elapsed_ns < 0 ? 0 : static_cast<std::uint64_t>(elapsed_ns));
            owner->context->diagnostics->set_resource(scope, amio::detail::DiagnosticResource::OperatorScratchBytes, executed.scratch_bytes,
                                                      owner->payload_budget_bytes);
        }
    }
    const auto operated_data = [&](const std::string &name, const amio::detail::StepContributionSnapshot &source) -> const void * {
        const auto found = operated_fields.find(name);
        return found != operated_fields.end() && !found->second.bytes.empty() ? static_cast<const void *>(found->second.bytes.data())
                                                                              : static_cast<const void *>(source.bytes.data());
    };
    amio::detail::Grib2StepPayload grib_payload;
    amio_status_t payload_status = AMIO_OK;
    try {
        if (owner->format == AMIO_FORMAT_NETCDF4) {
            if (producer_step->partitioned) {
                amio_status_t directory_status = AMIO_OK;
                if (owner->context->context->rank() == owner->context->context->storage_root_rank()) {
                    try {
                        std::filesystem::create_directories(std::filesystem::path(payload_path).parent_path());
                    } catch (...) {
                        directory_status = AMIO_ERR_BACKEND_FAILURE;
                    }
                }
                payload_status = collective_status(*owner->context->context, directory_status);
                if (payload_status != AMIO_OK) throw std::runtime_error("failed to agree on NetCDF payload directory creation");
            } else
                std::filesystem::create_directories(std::filesystem::path(payload_path).parent_path());
            std::map<std::string, amio::detail::NetCDFStepFieldView, std::less<>> fields;
            for (const auto &[name, contribution] : coordinator.contributions())
                fields.emplace(name, amio::detail::NetCDFStepFieldView{contribution.region, operated_data(name, contribution)});
            std::int64_t fortran_communicator = 0;
#ifdef AMIO_HAS_MPI
            fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(owner->context->context->communicator()));
#endif
            amio::detail::write_netcdf_step_payload(payload_path, *schema_it->second, coordinator.active_model_time(), fields, fortran_communicator);
        } else if (owner->format == AMIO_FORMAT_ZARR3) {
#ifdef AMIO_HAS_TENSORSTORE
            std::string error;
            if (producer_step->partitioned) {
                if (owner->context->context->rank() == owner->context->context->storage_root_rank() &&
                    !amio::detail::create_zarr_step(owner->location, owner->uuid, coordinator.active_step_id(), *schema_it->second,
                                                    owner->zarr_shard_shapes, error))
                    payload_status = AMIO_ERR_BACKEND_FAILURE;
                payload_status = collective_status(*owner->context->context, payload_status);
                if (payload_status != AMIO_OK) throw std::runtime_error(error.empty() ? "TensorStore Zarr metadata creation failed" : error);
                std::map<std::string, amio::detail::ZarrRegionView, std::less<>> regions;
                for (const auto &[name, contribution] : coordinator.contributions()) {
                    amio::detail::ZarrRegionView region;
                    region.offsets = contribution.region.offsets;
                    region.extents = contribution.region.extents;
                    const auto &operated = operated_fields.at(name);
                    region.data = operated.bytes.empty() ? std::span<const std::byte>(contribution.bytes.data(), contribution.bytes.size())
                                                         : std::span<const std::byte>(operated.bytes.data(), operated.bytes.size());
                    regions.emplace(name, std::move(region));
                }
                if (!amio::detail::write_zarr_step_regions(owner->location, owner->uuid, coordinator.active_step_id(), *schema_it->second,
                                                           owner->zarr_shard_shapes, regions, error))
                    throw std::runtime_error(error.empty() ? "TensorStore Zarr region write failed" : error);
            } else {
                std::map<std::string, std::span<const std::byte>, std::less<>> fields;
                for (const auto &[name, contribution] : coordinator.contributions()) {
                    const auto &operated = operated_fields.at(name);
                    fields.emplace(name, operated.bytes.empty() ? std::span<const std::byte>(contribution.bytes.data(), contribution.bytes.size())
                                                                : std::span<const std::byte>(operated.bytes.data(), operated.bytes.size()));
                }
                if (!amio::detail::write_zarr_step(owner->location, owner->uuid, coordinator.active_step_id(), *schema_it->second, fields, error))
                    throw std::runtime_error(error.empty() ? "TensorStore Zarr step write failed" : error);
            }
            amio::detail::ZarrStepLayout layout;
            if (!amio::detail::make_zarr_step_layout(owner->location, coordinator.active_step_id(), *schema_it->second, layout, error))
                throw std::runtime_error(error.empty() ? "TensorStore Zarr step path creation failed" : error);
            std::size_t path_index = 0;
            for (const auto &[name, variable] : schema_it->second->variables()) {
                (void)variable;
                zarr_payload_paths.emplace(
                    name, std::filesystem::path(layout.array_paths.at(path_index++)).lexically_relative(owner->location).generic_string());
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
                meta.dtype = variable_it->second.stored_dtype;
                meta.shape = shape;
                amio::detail::StagingBuffer source{};
                const auto &operated = operated_fields.at(name);
                source.data = const_cast<std::byte *>(static_cast<const std::byte *>(operated_data(name, contribution)));
                source.capacity_bytes = operated.bytes.empty() ? contribution.bytes.size() : operated.bytes.size();
                source.used_bytes = source.capacity_bytes;
                std::vector<unsigned char> message;
                const auto codec_attribute = variable_it->second.attributes.find("amio.codec");
                if (codec_attribute != variable_it->second.attributes.end()) {
                    const auto *codec = std::get_if<std::string>(&codec_attribute->second.value);
                    if (codec == nullptr || settings.decimal_scale_factor != 0)
                        throw std::runtime_error("GRIB2 lossless codec requires UTF-8 selection and zero decimal scaling");
                    amio::detail::GRIB2_DRT drt;
                    if (*codec == "libaec" || *codec == "grib2_aec")
                        drt = amio::detail::GRIB2_DRT::AdaptiveEntropyCoding;
                    else if (*codec == "lossless_jpeg2000" || *codec == "grib2_lossless")
                        drt = amio::detail::GRIB2_DRT::LosslessJPEG2000;
                    else
                        throw std::runtime_error("Unsupported GRIB2 native codec selection");
                    message = owner->grib2_driver->encode_step_message(source, meta, static_cast<std::size_t>(owner->payload_budget_bytes),
                                                                       static_cast<std::size_t>(owner->payload_budget_bytes), settings, drt);
                } else {
                    message = owner->grib2_driver->encode_step_message(source, meta, static_cast<std::size_t>(owner->payload_budget_bytes),
                                                                       static_cast<std::size_t>(owner->payload_budget_bytes), settings);
                }
                const std::vector<std::vector<unsigned char>> messages{message};
                amio::detail::append_grib2_field_messages(grib_payload, name, static_cast<std::uint32_t>(owner->context->context->rank()), messages,
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
    } catch (...) {
        payload_status = AMIO_ERR_BACKEND_FAILURE;
    }
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
        if (!producer_step->partitioned || owner->context->context->rank() == owner->context->context->storage_root_rank()) {
            try {
                file_bytes = std::filesystem::file_size(payload_path);
            } catch (...) {
                status = AMIO_ERR_BACKEND_FAILURE;
            }
        }
        if (producer_step->partitioned) {
            status = collective_status(*owner->context->context, status);
            if (status != AMIO_OK) {
                (void)coordinator.poison_dataset(status);
                return status;
            }
#ifdef AMIO_HAS_MPI
            if (MPI_Bcast(&file_bytes, 1, MPI_UINT64_T, static_cast<int>(owner->context->context->storage_root_rank()),
                          owner->context->context->communicator()) != MPI_SUCCESS)
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
            const bool is_present =
                producer_step->partitioned ? owner->global_present_fields[name] : contribution != coordinator.contributions().end();
            if (!is_present) continue;
            record.present_fields.push_back(name);
            amio::detail::PayloadDescriptor payload;
            payload.field = name;
            payload.relative_path = owner->format == AMIO_FORMAT_NETCDF4 ? relative_payload : zarr_payload_paths.at(name);
            if (owner->format == AMIO_FORMAT_NETCDF4)
                payload.bytes = file_bytes;
            else if (producer_step->partitioned) {
                std::size_t full_field_bytes = 0;
                if (!variable_payload_size(variable, full_field_bytes)) {
                    (void)coordinator.poison_dataset(AMIO_ERR_INVALID_INPUT);
                    return AMIO_ERR_INVALID_INPUT;
                }
                payload.bytes = full_field_bytes;
            } else
                payload.bytes = operated_fields.at(name).output_bytes;
            payload.array_container = owner->format == AMIO_FORMAT_ZARR3;
            payload.producer = 0;
            amio::detail::CompiledFieldPipeline compiled_pipeline;
            const auto pipeline_status = amio::detail::compile_variable_pipeline(variable, compiled_pipeline, operator_destination(owner->format));
            if (pipeline_status != AMIO_OK) {
                (void)coordinator.poison_dataset(pipeline_status);
                return pipeline_status;
            }
            payload.operator_pipeline = std::move(compiled_pipeline.identity);
            if (producer_step->partitioned) {
                payload.offsets.assign(variable.dimensions.size(), 0);
                for (const auto &dimension : variable.dimensions) payload.extents.push_back(dimension.extent);
            } else {
                payload.offsets = contribution->second.region.offsets;
                payload.extents = contribution->second.region.extents;
            }
            if (producer_step->partitioned) {
                const auto regions = partition_integrity_regions.find(name);
                if (regions != partition_integrity_regions.end()) payload.integrity_regions = regions->second;
            } else {
                payload.integrity_sha256 = operated_fields.at(name).integrity_digest;
            }
            record.payloads.push_back(std::move(payload));
        }
    }
    status = coordinator.mark_publishing();
    if (status != AMIO_OK) return status;
    amio::detail::PublicationResult publication;
    if (owner->context->context->rank() == owner->context->context->storage_root_rank())
        publication = owner->catalog->publish(std::move(record), owner->requested_sync);
    status = broadcast_publication(*owner->context->context, publication);
    if (status != AMIO_OK) return status;
    if (publication.state == amio::detail::PublicationState::Complete && publication.status == AMIO_OK) {
        status = coordinator.mark_publication_visible(owner->requested_sync != amio::detail::PosixSyncLevel::Visibility);
        if (status != AMIO_OK) return status;
        owner->previous_commit = publication.commit_path;
        producer_step->ended = true;
        record_publication();
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
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(operation))),
                                                      HandleKind::Operation, &payload);
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
    const auto status = process_handle_table().release(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(operation))),
                                                       HandleKind::Operation, &payload);
    if (status != AMIO_OK) return status;
    auto *handle = static_cast<OperationHandlePayload *>(payload);
    (void)handle->operation->release_observer();
    delete handle;
    return AMIO_OK;
}

AMIO_API amio_status_t amio_wait_operation(amio_operation_handle operation, uint32_t milestone, uint64_t timeout_ms) {
    if (operation == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(operation))),
                                                      HandleKind::Operation, &payload);
    if (lookup != AMIO_OK) return lookup;
    amio::detail::OperationMilestone target;
    switch (milestone) {
        case AMIO_OPERATION_SOURCE_RELEASED:
            target = amio::detail::OperationMilestone::SourceReleased;
            break;
        case AMIO_OPERATION_BACKEND_COMPLETE:
            target = amio::detail::OperationMilestone::PayloadComplete;
            break;
        case AMIO_OPERATION_READER_VISIBLE:
            target = amio::detail::OperationMilestone::ReaderVisible;
            break;
        case AMIO_OPERATION_DURABLE:
            target = amio::detail::OperationMilestone::StorageSynced;
            break;
        default:
            return AMIO_ERR_INVALID_INPUT;
    }
    if (timeout_ms > static_cast<std::uint64_t>(std::numeric_limits<std::chrono::milliseconds::rep>::max())) return AMIO_ERR_INVALID_INPUT;
    const auto handle = static_cast<OperationHandlePayload *>(payload);
    const auto result = handle->operation->wait_for(target, std::chrono::milliseconds(timeout_ms));
    if (result == amio::detail::WaitOutcome::Reached) return AMIO_OK;
    if (result == amio::detail::WaitOutcome::Timeout) return AMIO_ERR_TIMEOUT;
    const auto snapshot = handle->operation->snapshot();
    if (snapshot.outcome == amio::detail::OperationOutcome::Succeeded) return AMIO_ERR_UNSUPPORTED;
    return snapshot.error == AMIO_OK ? AMIO_ERR_STEP_FAILED : snapshot.error;
}

AMIO_API amio_status_t amio_list_steps(amio_dataset_handle dataset, const amio_catalog_cursor_t *cursor, uint32_t max_results,
                                       amio_step_summary_t *results, size_t result_capacity, size_t *out_result_count,
                                       amio_catalog_cursor_t *out_next_cursor, int *out_has_more) {
    if (out_result_count != nullptr) *out_result_count = 0;
    if (out_has_more != nullptr) *out_has_more = 0;
    if (out_next_cursor != nullptr) {
        *out_next_cursor = {};
        out_next_cursor->struct_size = sizeof(*out_next_cursor);
        out_next_cursor->version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    }
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    if (out_result_count == nullptr || out_next_cursor == nullptr || out_has_more == nullptr || max_results == 0 || max_results > 10000 ||
        (results == nullptr && result_capacity != 0))
        return AMIO_ERR_INVALID_INPUT;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
                                                      HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    amio::detail::CatalogCursor internal;
    if (cursor != nullptr) {
        constexpr std::size_t minimum = offsetof(amio_catalog_cursor_t, reserved) + 2 * sizeof(std::uint64_t);
        if (!valid_descriptor_header(cursor, minimum) || cursor->reserved[0] != 0 || cursor->reserved[1] != 0 ||
            !copy_fixed_text(cursor->dataset_uuid, sizeof(cursor->dataset_uuid), internal.dataset_uuid) ||
            !copy_fixed_text(cursor->generation, sizeof(cursor->generation), internal.generation) ||
            !copy_fixed_text(cursor->next_commit, sizeof(cursor->next_commit), internal.next_commit))
            return AMIO_ERR_INVALID_INPUT;
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
    if (next.dataset_uuid.size() >= sizeof(out_next_cursor->dataset_uuid) || next.generation.size() >= sizeof(out_next_cursor->generation) ||
        next.next_commit.size() >= sizeof(out_next_cursor->next_commit))
        return AMIO_ERR_BUFFER_TOO_SMALL;
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
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
                                                      HandleKind::EvolutionDataset, &payload);
    if (lookup != AMIO_OK) return lookup;
    auto *owner = static_cast<EvolutionDatasetPayload *>(payload);
    std::lock_guard<std::mutex> lock(owner->mutex);
    amio::detail::CollectionReader reader(owner->location, owner->uuid);
    amio::detail::CatalogStep step;
    const auto status = reader.describe_step(step_id, step);
    if (status != AMIO_OK) return status;
    return fill_step_summary(step, out_summary) ? AMIO_OK : AMIO_ERR_BACKEND_FAILURE;
}

AMIO_API amio_status_t amio_read_step_field(amio_dataset_handle dataset, uint64_t step_id, const char *field, size_t field_bytes,
                                            const amio_step_selection_t *selection, void *data, size_t data_capacity, size_t *out_required_bytes) {
    if (out_required_bytes == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_required_bytes = 0;
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    std::string field_name;
    if (!copy_utf8(field, field_bytes, field_name) || (data == nullptr && data_capacity != 0)) return AMIO_ERR_INVALID_INPUT;
    if (selection != nullptr) {
        constexpr std::size_t minimum = offsetof(amio_step_selection_t, reserved) + 2 * sizeof(std::uint64_t);
        if (!valid_descriptor_header(selection, minimum) || selection->reserved32 != 0 || selection->reserved[0] != 0 ||
            selection->reserved[1] != 0 || selection->rank == 0 || selection->rank > AMIO_MAX_RANK)
            return AMIO_ERR_INVALID_INPUT;
    }
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
                                                      HandleKind::EvolutionDataset, &payload);
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
    if (std::find(step.completion.present_fields.begin(), step.completion.present_fields.end(), field_name) == step.completion.present_fields.end())
        return AMIO_ERR_FIELD_NOT_PRESENT;
    const auto payload_it =
        std::find_if(step.completion.payloads.begin(), step.completion.payloads.end(), [&](const auto &entry) { return entry.field == field_name; });
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
    for (const auto &dimension : variable_it->second.dimensions) cache_metadata.global_extents.push_back(dimension.extent);
    std::vector<std::byte> cached;
    const auto cache_status = owner->read_cache.load(cache_key, cache_metadata, cached);
    if (cache_status == AMIO_OK) {
        if (cached.size() != required) return AMIO_ERR_SCHEMA_MISMATCH;
        std::memcpy(data, cached.data(), required);
        record_step_read_cache_occupancy(*owner, step_id, field_name);
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
            if (!amio::detail::read_zarr_step_field(owner->location, owner->uuid, step_id, *schema_it->second, field_name, region.offsets,
                                                    region.extents, selected_field, error))
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
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    // Cache admission is optional: oversized reads still succeed, while the
    // queue enforces its byte ceiling before allocating a cache copy.
    (void)owner->read_cache.store(cache_key, cache_metadata, data, required);
    record_step_read_cache_occupancy(*owner, step_id, field_name);
    return AMIO_OK;
}

namespace {

std::string integrity_dtype_name(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_I8:
            return "i8";
        case AMIO_DTYPE_U8:
            return "u8";
        case AMIO_DTYPE_I16:
            return "i16";
        case AMIO_DTYPE_U16:
            return "u16";
        case AMIO_DTYPE_I32:
            return "i32";
        case AMIO_DTYPE_U32:
            return "u32";
        case AMIO_DTYPE_I64:
            return "i64";
        case AMIO_DTYPE_U64:
            return "u64";
        case AMIO_DTYPE_F32:
            return "f32";
        case AMIO_DTYPE_F64:
            return "f64";
        default:
            return {};
    }
}

amio_status_t verify_committed_payload(EvolutionDatasetPayload &owner, const amio::detail::CompletionRecord &completion,
                                       const amio::detail::PayloadDescriptor &payload) {
    const auto schema_it = owner.schemas.find(completion.schema_digest);
    if (schema_it == owner.schemas.end()) return AMIO_ERR_SCHEMA_MISMATCH;
    const auto variable_it = schema_it->second->variables().find(payload.field);
    if (variable_it == schema_it->second->variables().end()) return AMIO_ERR_SCHEMA_MISMATCH;
    const auto dtype = variable_it->second.stored_dtype;
    std::vector<amio::detail::IntegrityRegion> regions;
    if (!payload.integrity_sha256.empty()) {
        regions.push_back({payload.field, integrity_dtype_name(dtype), payload.offsets, payload.extents, payload.integrity_sha256});
    } else {
        for (const auto &region : payload.integrity_regions)
            regions.push_back({payload.field, integrity_dtype_name(dtype), region.offsets, region.extents, region.digest});
    }
    if (regions.empty()) return AMIO_ERR_UNSUPPORTED;
    std::vector<std::uint64_t> selection_offsets, selection_extents;
    for (const auto &dimension : variable_it->second.dimensions) {
        selection_offsets.push_back(0);
        selection_extents.push_back(dimension.extent);
    }
    if (regions.size() == 1 && regions.front().offsets.size() != selection_extents.size()) return AMIO_ERR_SCHEMA_MISMATCH;
    const auto payload_path = (std::filesystem::path(owner.location) / payload.relative_path).string();
    amio::detail::IntegrityRegionReader region_reader = [&](const amio::detail::IntegrityRegion &region, std::size_t max_bytes,
                                                            std::vector<std::byte> &data) -> amio_status_t {
        if (region.offsets.size() != selection_extents.size() || region.extents.size() != selection_extents.size()) return AMIO_ERR_SCHEMA_MISMATCH;
        data.resize(max_bytes);
        try {
            if (owner.format == AMIO_FORMAT_NETCDF4) {
                amio::detail::RegionSelection selected;
                selected.offsets = region.offsets;
                selected.extents = region.extents;
                amio::detail::NetCDF_Driver::read_step_region(payload_path, payload.field, selected, data.data());
            } else if (owner.format == AMIO_FORMAT_ZARR3) {
#ifdef AMIO_HAS_TENSORSTORE
                std::string error;
                if (!amio::detail::read_zarr_step_field(owner.location, owner.uuid, completion.step_id, *schema_it->second, payload.field,
                                                        region.offsets, region.extents, data, error))
                    return AMIO_ERR_BACKEND_FAILURE;
#else
                return AMIO_ERR_UNSUPPORTED;
#endif
            } else
                return AMIO_ERR_UNSUPPORTED;
        } catch (...) {
            return AMIO_ERR_BACKEND_FAILURE;
        }
        return data.size() == max_bytes ? AMIO_OK : AMIO_ERR_SCHEMA_MISMATCH;
    };
    std::size_t verified = 0;
    return amio::detail::verify_intersecting_integrity_regions(regions, selection_offsets, selection_extents, owner.payload_budget_bytes,
                                                               region_reader, verified);
}

}  // namespace

AMIO_API amio_status_t amio_validate_dataset(amio_dataset_handle dataset, const amio_validation_request_t *request, amio_validation_result_t *results,
                                             size_t result_capacity, size_t *out_result_count, uint64_t *out_next_cursor) {
    if (out_result_count != nullptr) *out_result_count = 0;
    if (out_next_cursor != nullptr) *out_next_cursor = 0;
    if (!valid_request(request) || out_result_count == nullptr || out_next_cursor == nullptr || (results == nullptr && result_capacity != 0) ||
        (results != nullptr && result_capacity == 0))
        return AMIO_ERR_INVALID_INPUT;
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
    bool integrity_supported = true;
    bool integrity_pass = true;
    std::size_t integrity_payload_count = 0;
    amio_status_t integrity_error = AMIO_OK;
    const bool verify_integrity = (request->check_mask & (UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1))) != 0;
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
            if (verify_integrity) {
                for (const auto &step : page.steps) {
                    for (const auto &payload_descriptor : step.completion.payloads) {
                        ++integrity_payload_count;
                        const auto verify_status = verify_committed_payload(*owner, step.completion, payload_descriptor);
                        if (verify_status == AMIO_ERR_UNSUPPORTED)
                            integrity_supported = false;
                        else if (verify_status != AMIO_OK) {
                            integrity_pass = false;
                            integrity_error = verify_status;
                        }
                    }
                }
            }
            if (!page.has_more) break;
            cursor = std::move(page.next);
        }
        structural_error = structural_pass ? AMIO_OK : structural_error;
        if (verify_integrity && integrity_payload_count == 0) integrity_supported = false;
        if (verify_integrity && !structural_pass) {
            integrity_supported = false;
            integrity_pass = false;
            integrity_error = structural_error;
        }
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
        if (!evolution_dataset || check == AMIO_VALIDATE_ORPHAN_OBJECTS) {
            results[index].status = AMIO_VALIDATION_UNSUPPORTED;
        } else if (check == AMIO_VALIDATE_INTEGRITY) {
            results[index].status =
                !integrity_pass ? AMIO_VALIDATION_FAIL : (integrity_supported ? AMIO_VALIDATION_PASS : AMIO_VALIDATION_UNSUPPORTED);
            results[index].detail_code = results[index].status == AMIO_VALIDATION_FAIL ? integrity_error : 0;
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
    const bool failed_terminal = owner->steps->state() == amio::detail::StepState::Failed && owner->steps->active_step_id() == producer_step->step_id;
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

AMIO_API amio_status_t amio_acquire_step_lease(amio_dataset_handle dataset, uint64_t step_id, amio_step_lease_handle *out_lease) {
    if (out_lease == nullptr) return AMIO_ERR_INVALID_INPUT;
    *out_lease = nullptr;
    if (dataset == nullptr) return AMIO_ERR_NULL_HANDLE;
    void *payload = nullptr;
    const auto lookup = process_handle_table().lookup(HandleTable::from_ptr(const_cast<void *>(reinterpret_cast<const void *>(dataset))),
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

}  // extern "C"
