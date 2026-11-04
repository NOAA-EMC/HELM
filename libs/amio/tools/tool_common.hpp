#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "amio/amio_evolution.h"
#include "c_boundary/amio_tool_api.hpp"
#include "catalog/catalog_format.hpp"
#include "catalog/collection_reader.hpp"
#include "result_writer.hpp"
#include "schema/schema.hpp"

namespace amio::tools {

struct Options {
    std::string command;
    std::vector<std::string> positional;
    std::map<std::string, std::string, std::less<>> values;
    std::vector<std::string> flags;
    bool json = false;
};

struct CollectionInfo {
    std::string uuid;
    std::filesystem::path root;
    amio::detail::CollectionReader reader;
    explicit CollectionInfo(std::filesystem::path path, std::string id) : uuid(std::move(id)), root(std::move(path)), reader(root, uuid) {}
};

struct CliRuntime {
    amio_context_handle context = nullptr;
    amio_dataset_handle dataset = nullptr;
    bool finalize_mpi = false;
    ~CliRuntime();
};

[[nodiscard]] bool parse_options(int argc, char **argv, Options &options, std::string &error);
[[nodiscard]] bool read_bounded_file(const std::filesystem::path &path, std::size_t limit, std::string &out);
[[nodiscard]] amio_status_t open_collection(const std::filesystem::path &path, CollectionInfo &out);
[[nodiscard]] bool load_schema(const CollectionInfo &collection, const std::string &digest, amio::detail::SchemaDocument &document,
                               amio::detail::Schema &schema);
[[nodiscard]] amio_status_t open_cli_dataset(const std::filesystem::path &path, const Options &options, CliRuntime &runtime);
[[nodiscard]] amio_status_t open_cli_context(const Options &options, CliRuntime &runtime);
void finalize_cli_mpi();
[[nodiscard]] std::string dtype_name(amio_dtype_t dtype);
[[nodiscard]] std::string schema_fields_json(const amio::detail::Schema &schema, const std::string &filter = {});
[[nodiscard]] std::string step_json(const amio::detail::CatalogStep &step, const std::string &field_filter = {});
[[nodiscard]] CliResult run_capabilities(const Options &options);
[[nodiscard]] CliResult run_inspect(const Options &options);
[[nodiscard]] CliResult run_validate(const Options &options);
[[nodiscard]] CliResult run_copy(const Options &options);
[[nodiscard]] CliResult run_diagnostics(const Options &options);
[[nodiscard]] bool parse_inclusive_step_range(const std::string &text, std::uint64_t &first, std::uint64_t &last);
[[nodiscard]] bool parse_field_list(const std::string &text, std::vector<std::string> &fields);
[[nodiscard]] bool destination_is_available(const std::filesystem::path &path);

}  // namespace amio::tools
