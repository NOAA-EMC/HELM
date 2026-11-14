#include <algorithm>
#include <cstring>
#include <set>
#include <type_traits>
#include <variant>

#include "tool_common.hpp"

namespace amio::tools {
namespace {
constexpr std::size_t kCopyBudget = 64u * 1024u * 1024u;
struct AttributeWire {
    std::string name;
    std::variant<std::uint8_t, std::int64_t, std::uint64_t, double, std::string, std::vector<std::int64_t>, std::vector<std::uint64_t>,
                 std::vector<double>>
        value;
    amio_attribute_descriptor_t descriptor{};
};
struct VariableWire {
    std::vector<amio_dimension_descriptor_t> dimensions;
    std::vector<AttributeWire> attributes;
    std::vector<amio_attribute_descriptor_t> attribute_descriptors;
    amio_variable_descriptor_t descriptor{};
};

bool make_variable_wire(const std::string &name, const amio::detail::Variable &variable, VariableWire &wire) {
    if (variable.source_dtype != variable.stored_dtype || variable.dimensions.size() > AMIO_MAX_RANK) return false;
    wire.dimensions.reserve(variable.dimensions.size());
    for (const auto &d : variable.dimensions) {
        amio_dimension_descriptor_t out{};
        out.struct_size = sizeof(out);
        out.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        out.name = d.name.data();
        out.name_bytes = d.name.size();
        out.extent = d.extent;
        if (d.coordinate_variable) {
            out.coordinate_variable = d.coordinate_variable->data();
            out.coordinate_variable_bytes = d.coordinate_variable->size();
        }
        wire.dimensions.push_back(out);
    }
    wire.attributes.reserve(variable.attributes.size());
    for (const auto &[attr_name, attr] : variable.attributes) {
        AttributeWire out;
        out.name = attr_name;
        bool supported = true;
        std::visit(
            [&](const auto &value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, bool>)
                    out.value = static_cast<std::uint8_t>(value);
                else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t> || std::is_same_v<T, double> ||
                                   std::is_same_v<T, std::string> || std::is_same_v<T, std::vector<std::int64_t>> ||
                                   std::is_same_v<T, std::vector<std::uint64_t>> || std::is_same_v<T, std::vector<double>>)
                    out.value = value;
                else
                    supported = false;
            },
            attr.value);
        if (!supported) return false;
        wire.attributes.push_back(std::move(out));
    }
    wire.descriptor.struct_size = sizeof(wire.descriptor);
    wire.descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    wire.descriptor.name = name.data();
    wire.descriptor.name_bytes = name.size();
    wire.descriptor.source_dtype = variable.stored_dtype;
    wire.descriptor.stored_dtype = variable.stored_dtype;
    wire.descriptor.required = variable.required ? 1u : 0u;
    wire.descriptor.dimension_count = static_cast<std::uint32_t>(wire.dimensions.size());
    wire.descriptor.dimensions = wire.dimensions.data();
    wire.descriptor.attribute_count = static_cast<std::uint32_t>(wire.attributes.size());
    for (auto &a : wire.attributes) {
        auto &d = a.descriptor;
        d.struct_size = sizeof(d);
        d.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        d.name = a.name.data();
        d.name_bytes = a.name.size();
        std::visit(
            [&](auto &value) {
                using T = std::decay_t<decltype(value)>;
                d.value = &value;
                if constexpr (std::is_same_v<T, std::uint8_t>) {
                    d.type = AMIO_ATTRIBUTE_BOOL;
                    d.element_count = 1;
                    d.value_bytes = 1;
                } else if constexpr (std::is_same_v<T, std::int64_t>) {
                    d.type = AMIO_ATTRIBUTE_I64;
                    d.element_count = 1;
                    d.value_bytes = sizeof(value);
                } else if constexpr (std::is_same_v<T, std::uint64_t>) {
                    d.type = AMIO_ATTRIBUTE_U64;
                    d.element_count = 1;
                    d.value_bytes = sizeof(value);
                } else if constexpr (std::is_same_v<T, double>) {
                    d.type = AMIO_ATTRIBUTE_F64;
                    d.element_count = 1;
                    d.value_bytes = sizeof(value);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    d.type = AMIO_ATTRIBUTE_UTF8;
                    d.element_count = 1;
                    d.value_bytes = value.size();
                    d.value = value.data();
                } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                    d.type = AMIO_ATTRIBUTE_I64;
                    d.element_count = value.size();
                    d.value_bytes = value.size() * sizeof(value[0]);
                    d.value = value.data();
                } else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>>) {
                    d.type = AMIO_ATTRIBUTE_U64;
                    d.element_count = value.size();
                    d.value_bytes = value.size() * sizeof(value[0]);
                    d.value = value.data();
                } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                    d.type = AMIO_ATTRIBUTE_F64;
                    d.element_count = value.size();
                    d.value_bytes = value.size() * sizeof(value[0]);
                    d.value = value.data();
                }
            },
            a.value);
    }
    wire.attribute_descriptors.reserve(wire.attributes.size());
    for (const auto &a : wire.attributes) wire.attribute_descriptors.push_back(a.descriptor);
    wire.descriptor.attributes = wire.attribute_descriptors.data();
    return true;
}

CliResult invalid(int code, std::string message) {
    CliResult r{code, "copy", "copy preflight failed"};
    r.errors.push_back(std::move(message));
    return r;
}
}  // namespace

CliResult run_copy(const Options &options) {
    if (options.positional.size() != 2) return invalid(2, "usage: amio copy SOURCE DEST --config FILE --steps FIRST:LAST --fields LIST [--json]");
    const auto &source_path = options.positional[0];
    const auto &destination_path = options.positional[1];
    if (!destination_is_available(destination_path)) return invalid(2, "destination already exists or cannot be inspected");
    auto range = options.values.find("--steps"), field_arg = options.values.find("--fields");
    if (!options.values.contains("--config")) return invalid(2, "--config is required");
    if (range == options.values.end() || field_arg == options.values.end()) return invalid(2, "--steps and --fields are required");
    std::uint64_t first = 0, last = 0;
    if (!parse_inclusive_step_range(range->second, first, last)) return invalid(2, "--steps must be an inclusive FIRST:LAST range");
    if (last - first >= 10000) return invalid(3, "requested range exceeds the 10,000-step bounded copy limit");
    std::vector<std::string> requested;
    if (!parse_field_list(field_arg->second, requested)) return invalid(2, "--fields must be a nonempty comma-separated unique list");
    if (options.values.contains("--allow-omit"))
        return invalid(3, "metadata omission cannot be authorized because this copy preserves schema attributes exactly");

    CollectionInfo source(source_path, "");
    auto status = open_collection(source_path, source);
    if (status != AMIO_OK) return invalid(status == AMIO_ERR_INVALID_INPUT ? 2 : 4, "source is not a readable AMIO collection");
    auto page = source.reader.list_steps({}, 128);
    if (page.status != AMIO_OK) return invalid(4, "source catalog is corrupt or incomplete");
    std::vector<amio::detail::CatalogStep> steps;
    while (true) {
        for (const auto &step : page.steps)
            if (step.completion.step_id >= first && step.completion.step_id <= last) steps.push_back(step);
        if (!page.has_more) break;
        page = source.reader.list_steps(page.next, 128);
        if (page.status != AMIO_OK) return invalid(4, "source catalog pagination failed");
    }
    if (steps.size() != static_cast<std::size_t>(last - first + 1)) return invalid(2, "inclusive step range contains missing step IDs");
    std::sort(steps.begin(), steps.end(), [](const auto &a, const auto &b) { return a.completion.step_id < b.completion.step_id; });
    for (std::size_t i = 0; i < steps.size(); ++i)
        if (steps[i].completion.step_id != first + i) return invalid(2, "inclusive step range contains missing step IDs");
    for (const auto &step : steps)
        for (const auto &payload : step.completion.payloads)
            if (payload.array_container || !payload.message_ranges.empty())
                return invalid(3, "bounded copy currently supports native NetCDF collections only");
    const auto schema_digest = steps.front().completion.schema_digest;
    for (const auto &step : steps)
        if (step.completion.schema_digest != schema_digest)
            return invalid(3, "schema changes within the selected range require an explicit conversion policy");
    amio::detail::SchemaDocument doc;
    amio::detail::Schema schema;
    if (!load_schema(source, schema_digest, doc, schema)) return invalid(4, "source schema document is unreadable");
    std::set<std::string> closure(requested.begin(), requested.end());
    for (const auto &name : requested) {
        auto it = schema.variables().find(name);
        if (it == schema.variables().end()) return invalid(2, "selected field is not declared: " + name);
        for (const auto &dimension : it->second.dimensions)
            if (dimension.coordinate_variable) closure.insert(*dimension.coordinate_variable);
    }
    for (const auto &name : closure) {
        auto it = schema.variables().find(name);
        if (it == schema.variables().end()) return invalid(3, "required coordinate dependency is unavailable: " + name);
        if (it->second.source_dtype != it->second.stored_dtype)
            return invalid(3, "field precision conversion requires an explicit operator policy: " + name);
    }
    std::map<std::uint64_t, double> model_times;
    std::map<std::uint64_t, std::map<std::string, std::size_t, std::less<>>> byte_sizes;
    CliRuntime runtime;
    status = open_cli_dataset(source_path, options, runtime);
    if (status != AMIO_OK) return invalid(5, "could not open source dataset reader: " + std::to_string(status));
    for (const auto &step : steps) {
        std::vector<std::byte> no_data;
        for (const auto &name : closure) {
            const bool present =
                std::find(step.completion.present_fields.begin(), step.completion.present_fields.end(), name) != step.completion.present_fields.end();
            const auto &variable = schema.variables().at(name);
            if (!present) {
                if (variable.required)
                    return invalid(4, "required field is missing in step " + std::to_string(step.completion.step_id) + ": " + name);
                continue;
            }
            auto payload =
                std::find_if(step.completion.payloads.begin(), step.completion.payloads.end(), [&](const auto &p) { return p.field == name; });
            if (payload == step.completion.payloads.end()) return invalid(4, "field payload reference is missing: " + name);
            std::size_t bytes = 0;
            status = amio_read_step_field(runtime.dataset, step.completion.step_id, name.data(), name.size(), nullptr, nullptr, 0, &bytes);
            if (status != AMIO_OK) return invalid(status == AMIO_ERR_UNSUPPORTED ? 3 : 4, "could not preflight source field bytes: " + name);
            if (bytes > kCopyBudget) return invalid(3, "a selected field exceeds the 64 MiB local copy budget: " + name);
            byte_sizes[step.completion.step_id][name] = bytes;
        }
        std::size_t total = 0;
        for (const auto &[name, bytes] : byte_sizes[step.completion.step_id]) {
            (void)name;
            if (bytes > kCopyBudget - total) return invalid(3, "selected step exceeds the 64 MiB local copy budget");
            total += bytes;
        }
        double time = 0;
        status = amio_get_step_model_time(runtime.dataset, step.completion.step_id, &time);
        if (status != AMIO_OK)
            return invalid(status == AMIO_ERR_UNSUPPORTED ? 3 : 4,
                           "source model time cannot be preserved for step " + std::to_string(step.completion.step_id));
        model_times[step.completion.step_id] = time;
    }

    std::vector<VariableWire> wire_variables;
    wire_variables.reserve(closure.size());
    std::vector<amio_variable_descriptor_t> variable_descriptors;
    variable_descriptors.reserve(closure.size());
    for (const auto &name : closure) {
        wire_variables.emplace_back();
        if (!make_variable_wire(name, schema.variables().at(name), wire_variables.back()))
            return invalid(3, "schema attribute type is not representable by the public destination API: " + name);
    }
    for (const auto &wire : wire_variables) variable_descriptors.push_back(wire.descriptor);
    amio_schema_descriptor_t schema_descriptor{};
    schema_descriptor.struct_size = sizeof(schema_descriptor);
    schema_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_descriptor.variable_count = variable_descriptors.size();
    schema_descriptor.variables = variable_descriptors.data();
    const auto dest_text = destination_path;
    amio_dataset_descriptor_t dataset_descriptor{};
    dataset_descriptor.struct_size = sizeof(dataset_descriptor);
    dataset_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_descriptor.location = dest_text.data();
    dataset_descriptor.location_bytes = dest_text.size();
    dataset_descriptor.format = AMIO_FORMAT_NETCDF4;
    dataset_descriptor.budget_bytes = kCopyBudget;
    dataset_descriptor.budget_descriptors = 4096;
    dataset_descriptor.control_budget_bytes = 16u * 1024u * 1024u;
    dataset_descriptor.control_budget_descriptors = 1024;
    amio_dataset_handle destination = nullptr;
    status = amio_open_dataset_ex(runtime.context, &dataset_descriptor, &destination);
    if (status != AMIO_OK)
        return invalid(status == AMIO_ERR_INVALID_INPUT ? 2 : 5, "could not create destination collection: " + std::to_string(status));
    amio_schema_handle output_schema = nullptr;
    status = amio_define_schema(destination, &schema_descriptor, &output_schema);
    if (status != AMIO_OK) {
        (void)amio_close_dataset_ex(destination);
        return invalid(5, "could not register selected destination schema: " + std::to_string(status));
    }
    std::size_t completed = 0;
    std::uint64_t unfinished = first;
    for (const auto &source_step : steps) {
        unfinished = source_step.completion.step_id;
        amio_step_descriptor_t step_descriptor{};
        step_descriptor.struct_size = sizeof(step_descriptor);
        step_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        step_descriptor.step_id = source_step.completion.step_id;
        step_descriptor.model_time = model_times.at(step_descriptor.step_id);
        step_descriptor.schema = output_schema;
        amio_step_handle output_step = nullptr;
        status = amio_begin_step(destination, &step_descriptor, &output_step);
        if (status != AMIO_OK) break;
        std::vector<amio_operation_handle> operations;
        bool failed = false;
        for (const auto &name : closure) {
            auto size_it = byte_sizes[step_descriptor.step_id].find(name);
            if (size_it == byte_sizes[step_descriptor.step_id].end()) continue;
            std::vector<std::byte> data(size_it->second);
            std::size_t required = 0;
            status = amio_read_step_field(runtime.dataset, step_descriptor.step_id, name.data(), name.size(), nullptr, nullptr, 0, &required);
            if (status != AMIO_OK) {
                failed = true;
                break;
            }
            status = amio_read_step_field(runtime.dataset, step_descriptor.step_id, name.data(), name.size(), nullptr, data.data(), data.size(),
                                          &required);
            if (status != AMIO_OK || required != data.size()) {
                failed = true;
                if (status == AMIO_OK) status = AMIO_ERR_SCHEMA_MISMATCH;
                break;
            }
            amio_operation_handle operation = nullptr;
            status = amio_put_field(output_step, name.data(), name.size(), data.data(), data.size(), &operation);
            if (status != AMIO_OK) {
                failed = true;
                break;
            }
            operations.push_back(operation);
        }
        if (!failed)
            status = amio_end_step(output_step);
        else
            (void)amio_cancel_step(output_step);
        for (auto operation : operations) (void)amio_release_operation(operation);
        (void)amio_release_step_handle(output_step);
        if (failed || status != AMIO_OK) break;
        ++completed;
    }
    (void)amio_release_schema(output_schema);
    const auto close_status = amio_close_dataset_ex(destination);
    if (status != AMIO_OK || completed != steps.size() || close_status != AMIO_OK) {
        CliResult result{status == AMIO_ERR_TIMEOUT ? 6 : 5, "copy", "copy stopped with unfinished work"};
        result.errors.push_back("completed_steps=" + std::to_string(completed) + ", unfinished_step=" + std::to_string(unfinished) +
                                ", status=" + std::to_string(status == AMIO_OK ? close_status : status));
        result.summary_json = "{\"completed_steps\":" + std::to_string(completed) + ",\"unfinished_step\":" + std::to_string(unfinished) + "}";
        return result;
    }
    CliResult result{0, "copy", "selected steps copied to a new NetCDF collection"};
    result.summary_json = "{\"completed_steps\":" + std::to_string(completed) + ",\"first_step\":" + std::to_string(first) +
                          ",\"last_step\":" + std::to_string(last) + ",\"fields\":" + std::to_string(closure.size()) +
                          ",\"destination_created\":true}";
    return result;
}
}  // namespace amio::tools
