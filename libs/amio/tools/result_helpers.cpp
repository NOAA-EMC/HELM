#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <type_traits>
#include <variant>

#include "tool_common.hpp"
#ifdef AMIO_TOOLS_HAS_MPI
#include <mpi.h>
#endif

namespace amio::tools {
namespace {
#ifdef AMIO_TOOLS_HAS_MPI
bool cli_owns_mpi = false;
#endif
bool extract_string(const std::string &json, const std::string &key, std::string &value) {
    const auto marker = ResultWriter::quote(key) + ":\"";
    auto at = json.find(marker);
    if (at == std::string::npos) return false;
    at += marker.size();
    const auto end = json.find('"', at);
    if (end == std::string::npos) return false;
    value.assign(json, at, end - at);
    return !value.empty();
}

std::string attribute_type(const amio::detail::AttributeValue &value) {
    return std::visit(
        [](const auto &v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>)
                return "bool";
            else if constexpr (std::is_same_v<T, std::int64_t>)
                return "i64";
            else if constexpr (std::is_same_v<T, std::uint64_t>)
                return "u64";
            else if constexpr (std::is_same_v<T, double>)
                return "f64";
            else if constexpr (std::is_same_v<T, std::string>)
                return "utf8";
            else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
                return "i64[]";
            else if constexpr (std::is_same_v<T, std::vector<std::uint64_t>>)
                return "u64[]";
            else if constexpr (std::is_same_v<T, std::vector<double>>)
                return "f64[]";
            else
                return "utf8[]";
        },
        value);
}
}  // namespace

CliRuntime::~CliRuntime() {
    if (dataset) (void)amio_close_dataset_ex(dataset);
    if (context) (void)amio_release_context(context);
}

void finalize_cli_mpi() {
#ifdef AMIO_TOOLS_HAS_MPI
    if (!cli_owns_mpi) return;
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Finalize();
    cli_owns_mpi = false;
#endif
}

amio_status_t open_cli_dataset(const std::filesystem::path &path, const Options &options, CliRuntime &runtime) {
    auto status = open_cli_context(options, runtime);
    if (status != AMIO_OK) return status;
    const auto path_text = path.string();
    return amio_open_dataset_reader(runtime.context, path_text.data(), path_text.size(), &runtime.dataset);
}

amio_status_t open_cli_context(const Options &options, CliRuntime &runtime) {
    std::string configuration = "{}";
    if (const auto it = options.values.find("--config"); it != options.values.end() && !read_bounded_file(it->second, 1024 * 1024, configuration))
        return AMIO_ERR_INVALID_INPUT;
    amio_context_descriptor_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    descriptor.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    descriptor.progress_mode = AMIO_PROGRESS_CALLER;
#ifdef AMIO_TOOLS_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        if (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS || provided < MPI_THREAD_MULTIPLE)
            return AMIO_ERR_THREADING_UNSUPPORTED;
        runtime.finalize_mpi = true;
        cli_owns_mpi = true;
    }
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    descriptor.rank_count = static_cast<std::uint32_t>(size);
    descriptor.fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
#else
    descriptor.rank_count = 1;
#endif
    descriptor.canonical_configuration = configuration.data();
    descriptor.configuration_bytes = configuration.size();
    return amio_init_ex(&descriptor, &runtime.context);
}

bool parse_options(int argc, char **argv, Options &options, std::string &error) {
    if (argc < 2) {
        error = "missing command";
        return false;
    }
    options.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string token = argv[i];
        if (token == "--json") {
            options.json = true;
            continue;
        }
        if (token.starts_with("--")) {
            if (i + 1 < argc && !std::string(argv[i + 1]).starts_with("--"))
                options.values[token] = argv[++i];
            else
                options.flags.push_back(token);
        } else
            options.positional.push_back(std::move(token));
    }
    return true;
}

bool read_bounded_file(const std::filesystem::path &path, std::size_t limit, std::string &out) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st{};
    bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) <= limit;
    if (ok) {
        out.assign(static_cast<std::size_t>(st.st_size), '\0');
        std::size_t offset = 0;
        while (offset < out.size()) {
            const auto n = ::read(fd, out.data() + offset, out.size() - offset);
            if (n < 0) {
                if (errno == EINTR) continue;
                ok = false;
                break;
            }
            if (n == 0) {
                ok = false;
                break;
            }
            offset += static_cast<std::size_t>(n);
        }
    }
    return ::close(fd) == 0 && ok;
}

amio_status_t open_collection(const std::filesystem::path &path, CollectionInfo &out) {
    std::string identity;
    if (!read_bounded_file(path / "amio.json", 4096, identity)) return AMIO_ERR_INVALID_INPUT;
    std::string uuid;
    if (!extract_string(identity, "dataset_uuid", uuid)) return AMIO_ERR_BACKEND_FAILURE;
    out = CollectionInfo(path, std::move(uuid));
    return AMIO_OK;
}

bool load_schema(const CollectionInfo &collection, const std::string &digest, amio::detail::SchemaDocument &document, amio::detail::Schema &schema) {
    std::string encoded;
    if (!read_bounded_file(collection.root / "payloads" / "schemas" / (digest + ".json"), amio::detail::Schema::kMaxCanonicalBytes + 1024, encoded) ||
        amio::detail::parse_schema_document(encoded, document) != AMIO_OK || document.dataset_uuid != collection.uuid || document.digest != digest)
        return false;
    return amio::detail::Schema::from_canonical(document.canonical_schema, amio::detail::Schema::kMaxCanonicalBytes, schema).ok();
}

std::string dtype_name(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return "f32";
        case AMIO_DTYPE_F64:
            return "f64";
        case AMIO_DTYPE_I8:
            return "i8";
        case AMIO_DTYPE_I16:
            return "i16";
        case AMIO_DTYPE_I32:
            return "i32";
        case AMIO_DTYPE_I64:
            return "i64";
        case AMIO_DTYPE_U8:
            return "u8";
        case AMIO_DTYPE_U16:
            return "u16";
        case AMIO_DTYPE_U32:
            return "u32";
        case AMIO_DTYPE_U64:
            return "u64";
        default:
            return "unknown";
    }
}

std::string schema_fields_json(const amio::detail::Schema &schema, const std::string &filter) {
    std::string out = "[";
    bool first = true;
    for (const auto &[name, variable] : schema.variables()) {
        if (!filter.empty() && filter != name) continue;
        if (!first) out += ',';
        first = false;
        out += "{\"name\":" + ResultWriter::quote(name) + ",\"source_dtype\":" + ResultWriter::quote(dtype_name(variable.source_dtype)) +
               ",\"stored_dtype\":" + ResultWriter::quote(dtype_name(variable.stored_dtype)) +
               ",\"required\":" + (variable.required ? "true" : "false") + ",\"dimensions\":[";
        for (std::size_t i = 0; i < variable.dimensions.size(); ++i) {
            if (i) out += ',';
            const auto &dimension = variable.dimensions[i];
            out += "{\"name\":" + ResultWriter::quote(dimension.name) + ",\"extent\":" + std::to_string(dimension.extent) +
                   ",\"coordinate\":" + (dimension.coordinate_variable ? ResultWriter::quote(*dimension.coordinate_variable) : "null") + "}";
        }
        out += "],\"attributes\":[";
        bool first_attribute = true;
        for (const auto &[attribute_name, attribute] : variable.attributes) {
            if (!first_attribute) out += ',';
            first_attribute = false;
            out += "{\"name\":" + ResultWriter::quote(attribute_name) + ",\"type\":" + ResultWriter::quote(attribute_type(attribute.value)) + "}";
        }
        out += "]}";
    }
    out += ']';
    return out;
}

std::string step_json(const amio::detail::CatalogStep &step, const std::string &field_filter) {
    const auto &record = step.completion;
    std::string out = "{\"step_id\":" + std::to_string(record.step_id) + ",\"schema_digest\":" + ResultWriter::quote(record.schema_digest) +
                      ",\"achieved_guarantee\":" + ResultWriter::quote(record.achieved_guarantee) +
                      ",\"legacy_unverified\":" + (step.legacy_unverified ? "true" : "false") + ",\"required_fields\":[";
    for (std::size_t i = 0; i < record.required_fields.size(); ++i) {
        if (i) out += ',';
        out += ResultWriter::quote(record.required_fields[i]);
    }
    out += "],\"present_fields\":[";
    bool first = true;
    for (const auto &field : record.present_fields) {
        if (!field_filter.empty() && field_filter != field) continue;
        if (!first) out += ',';
        first = false;
        out += ResultWriter::quote(field);
    }
    out += "]}";
    return out;
}
}  // namespace amio::tools
