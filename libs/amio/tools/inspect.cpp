#include <charconv>
#include <cstring>

#include "tool_common.hpp"
#ifdef AMIO_TOOLS_HAS_MPI
#include <mpi.h>
#endif

namespace amio::tools {
namespace {
bool parse_u64(const std::string &text, std::uint64_t &value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}
}  // namespace

CliResult run_capabilities(const Options &options) {
    CliResult result{0, "capabilities", "effective compiled capabilities"};
    CliRuntime runtime;
    const auto initialized = open_cli_context(options, runtime);
    if (initialized != AMIO_OK) {
        result.exit_code = initialized == AMIO_ERR_INVALID_INPUT || initialized == AMIO_ERR_MANIFEST_INVALID ? 2 : 5;
        result.summary = "could not initialize capability query";
        result.errors.push_back("AMIO context initialization failed with status " + std::to_string(initialized));
        return result;
    }
    std::size_t required = 0;
    auto status = amio_query_capabilities(runtime.context, nullptr, 0, &required);
    if (status == AMIO_OK && required > 0 && required <= 65536) {
        std::string json(required, '\0');
        status = amio_query_capabilities(runtime.context, json.data(), json.size(), &required);
        if (status == AMIO_OK && !json.empty()) result.results_json.push_back("{\"capabilities\":" + json.substr(0, json.find('\0')) + "}");
    } else if (status == AMIO_OK)
        status = AMIO_ERR_BUFFER_TOO_SMALL;
    if (status != AMIO_OK) {
        result.exit_code = 5;
        result.summary = "capability query failed";
        result.errors.push_back("AMIO capability query failed with status " + std::to_string(status));
    }
    return result;
}

CliResult run_inspect(const Options &options) {
    CliResult result{0, "inspect", "collection metadata"};
    if (options.positional.size() != 1) {
        result.exit_code = 2;
        result.errors.push_back("usage: amio inspect URI [--step ID] [--field NAME] [--limit N] [--cursor UUID|GENERATION|COMMIT] [--json]");
        return result;
    }
    CollectionInfo collection(options.positional[0], "");
    auto status = open_collection(options.positional[0], collection);
    if (status != AMIO_OK) {
        result.exit_code = status == AMIO_ERR_INVALID_INPUT ? 2 : 4;
        result.summary = "not a readable AMIO step collection";
        result.errors.push_back("collection identity is missing or malformed");
        return result;
    }
    CliRuntime runtime;
    status = open_cli_dataset(options.positional[0], options, runtime);
    if (status != AMIO_OK) {
        result.exit_code = status == AMIO_ERR_INVALID_INPUT ? 2 : 5;
        result.errors.push_back("could not open read-only dataset handle: " + std::to_string(status));
        return result;
    }
    std::uint64_t step_id = 0;
    const auto selected_step = options.values.find("--step");
    if (selected_step != options.values.end() && !parse_u64(selected_step->second, step_id)) {
        result.exit_code = 2;
        result.errors.push_back("--step must be an unsigned integer");
        return result;
    }
    std::uint64_t limit = 100;
    if (const auto it = options.values.find("--limit"); it != options.values.end() && (!parse_u64(it->second, limit) || limit == 0 || limit > 256)) {
        result.exit_code = 2;
        result.errors.push_back("--limit must be between 1 and 256");
        return result;
    }
    amio::detail::CatalogCursor cursor;
    if (const auto it = options.values.find("--cursor"); it != options.values.end()) {
        const auto first = it->second.find('|');
        const auto second = first == std::string::npos ? first : it->second.find('|', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            result.exit_code = 2;
            result.errors.push_back("cursor must contain UUID|GENERATION|NEXT_COMMIT");
            return result;
        }
        cursor.dataset_uuid = it->second.substr(0, first);
        cursor.generation = it->second.substr(first + 1, second - first - 1);
        cursor.next_commit = it->second.substr(second + 1);
        if (cursor.dataset_uuid != collection.uuid) {
            result.exit_code = 2;
            result.errors.push_back("cursor belongs to another collection UUID");
            return result;
        }
    }
    const std::string field = options.values.contains("--field") ? options.values.at("--field") : std::string{};
    if (selected_step != options.values.end()) {
        amio_step_summary_t public_step{};
        status = amio_describe_step(runtime.dataset, step_id, &public_step);
        if (status != AMIO_OK) {
            result.exit_code = status == AMIO_ERR_STEP_FAILED ? 4 : 5;
            result.errors.push_back("public step inspection failed with status " + std::to_string(status));
            return result;
        }
    } else {
        amio_catalog_cursor_t public_cursor{};
        public_cursor.struct_size = sizeof(public_cursor);
        public_cursor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        auto copy_cursor = [](char *dst, std::size_t capacity, const std::string &src) {
            if (src.size() >= capacity) return false;
            std::memcpy(dst, src.data(), src.size());
            return true;
        };
        if (!cursor.dataset_uuid.empty() && (!copy_cursor(public_cursor.dataset_uuid, sizeof(public_cursor.dataset_uuid), cursor.dataset_uuid) ||
                                             !copy_cursor(public_cursor.generation, sizeof(public_cursor.generation), cursor.generation) ||
                                             !copy_cursor(public_cursor.next_commit, sizeof(public_cursor.next_commit), cursor.next_commit))) {
            result.exit_code = 2;
            result.errors.push_back("cursor exceeds public API bounds");
            return result;
        }
        amio_catalog_cursor_t public_next{};
        std::size_t public_count = 0;
        int public_more = 0;
        status = amio_list_steps(runtime.dataset, cursor.dataset_uuid.empty() ? nullptr : &public_cursor, static_cast<std::uint32_t>(limit), nullptr,
                                 0, &public_count, &public_next, &public_more);
        if (status != AMIO_OK) {
            result.exit_code = status == AMIO_ERR_INVALID_INPUT ? 2 : 4;
            result.errors.push_back("public catalog inspection failed with status " + std::to_string(status));
            return result;
        }
    }
    amio::detail::CatalogPage page;
    if (selected_step != options.values.end()) {
        amio::detail::CatalogStep step;
        status = collection.reader.describe_step(step_id, step);
        if (status != AMIO_OK) {
            result.exit_code = status == AMIO_ERR_STEP_FAILED ? 4 : 5;
            result.errors.push_back("requested step could not be read");
            return result;
        }
        page.steps.push_back(std::move(step));
    } else {
        page = collection.reader.list_steps(cursor, static_cast<std::size_t>(limit));
        if (page.status != AMIO_OK) {
            result.exit_code = page.status == AMIO_ERR_INVALID_INPUT ? 2 : 4;
            result.errors.push_back("catalog traversal failed with status " + std::to_string(page.status));
            return result;
        }
    }
    std::string schema_digest = page.steps.empty() ? std::string{} : page.steps.front().completion.schema_digest;
    std::string format = "netcdf4";
    if (!page.steps.empty() && !page.steps.front().completion.payloads.empty()) {
        const auto &payload = page.steps.front().completion.payloads.front();
        if (payload.array_container)
            format = "zarr3";
        else if (!payload.message_ranges.empty())
            format = "grib2";
    }
    std::string model_time_json = "null";
    if (!page.steps.empty()) {
        double model_time = 0;
        if (amio_get_step_model_time(runtime.dataset, page.steps.front().completion.step_id, &model_time) == AMIO_OK)
            model_time_json = std::to_string(model_time);
        else
            result.warnings.push_back("model time is unavailable for the first inspected step");
    }
    amio::detail::SchemaDocument schema_document;
    amio::detail::Schema schema;
    if (!schema_digest.empty()) {
        if (!load_schema(collection, schema_digest, schema_document, schema)) {
            result.exit_code = 4;
            result.errors.push_back("schema document is missing, malformed or inconsistent");
            return result;
        }
        if (!field.empty() && !schema.variables().contains(field)) {
            result.exit_code = 2;
            result.errors.push_back("selected field is not declared by the page schema");
            return result;
        }
    }
    result.summary = "collection " + collection.uuid + ", " + std::to_string(page.steps.size()) + " step(s) in page";
    result.summary_json =
        "{\"dataset_uuid\":" + ResultWriter::quote(collection.uuid) + ",\"generation\":" + ResultWriter::quote(page.next.generation) +
        ",\"schema_digest\":" + ResultWriter::quote(schema_digest) +
        ",\"schema_fields\":" + (schema_digest.empty() ? "[]" : schema_fields_json(schema, field)) + ",\"format\":" + ResultWriter::quote(format) +
        ",\"model_time\":" + model_time_json + ",\"has_more\":" + (page.has_more ? "true" : "false") + ",\"next_cursor\":";
    if (page.has_more) {
        const auto &next = page.next;
        result.summary_json += ResultWriter::quote(next.dataset_uuid + "|" + next.generation + "|" + next.next_commit);
    } else
        result.summary_json += "null";
    result.summary_json += "}";
    for (const auto &step : page.steps) {
        auto json = step_json(step, field);
        if (!json.empty() && json.back() == '}') {
            json.pop_back();
            double model_time = 0;
            if (amio_get_step_model_time(runtime.dataset, step.completion.step_id, &model_time) == AMIO_OK)
                json += ",\"model_time\":" + std::to_string(model_time) + "}";
            else
                json += ",\"model_time\":null}";
        }
        result.results_json.push_back(std::move(json));
    }
    return result;
}
}  // namespace amio::tools
