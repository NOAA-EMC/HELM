#include "catalog/catalog_format.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <variant>

namespace amio::detail {
namespace {

struct Json {
    enum class Kind { Null, Boolean, Number, String, Array, Object } kind = Kind::Null;
    std::string scalar;
    std::vector<Json> array;
    std::map<std::string, Json, std::less<>> object;
};

void append_utf8(std::string &out, std::uint32_t code) {
    if (code <= 0x7f)
        out.push_back(static_cast<char>(code));
    else if (code <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else if (code <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (code >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (code >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
}

bool hex_value(char c, unsigned &value) {
    if (c >= '0' && c <= '9')
        value = static_cast<unsigned>(c - '0');
    else if (c >= 'a' && c <= 'f')
        value = static_cast<unsigned>(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F')
        value = static_cast<unsigned>(c - 'A' + 10);
    else
        return false;
    return true;
}

class Parser {
   public:
    Parser(std::string_view text, const CatalogLimits &limits) : text_(text), limits_(limits) {}
    bool parse(Json &out) {
        skip_ws();
        return parse_value(out, 0) && (skip_ws(), position_ == text_.size());
    }

   private:
    void skip_ws() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' || text_[position_] == '\r' || text_[position_] == '\t'))
            ++position_;
    }
    bool parse_value(Json &out, std::size_t depth) {
        if (depth > 32 || position_ >= text_.size()) return false;
        const char c = text_[position_];
        if (c == '"') {
            out.kind = Json::Kind::String;
            return parse_string(out.scalar);
        }
        if (c == '{') return parse_object(out, depth + 1);
        if (c == '[') return parse_array(out, depth + 1);
        if (c == 't' && consume("true")) {
            out.kind = Json::Kind::Boolean;
            out.scalar = "true";
            return true;
        }
        if (c == 'f' && consume("false")) {
            out.kind = Json::Kind::Boolean;
            out.scalar = "false";
            return true;
        }
        if (c == 'n' && consume("null")) {
            out.kind = Json::Kind::Null;
            return true;
        }
        return parse_number(out);
    }
    bool consume(std::string_view token) {
        if (text_.substr(position_, token.size()) != token) return false;
        position_ += token.size();
        return true;
    }
    bool parse_string(std::string &out) {
        if (text_[position_++] != '"') return false;
        out.clear();
        while (position_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') return out.size() <= limits_.string_bytes;
            if (c < 0x20) return false;
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
            } else {
                if (position_ >= text_.size()) return false;
                const char esc = text_[position_++];
                switch (esc) {
                    case '"':
                        out.push_back('"');
                        break;
                    case '\\':
                        out.push_back('\\');
                        break;
                    case '/':
                        out.push_back('/');
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u': {
                        std::uint32_t code = 0;
                        if (!read_hex4(code)) return false;
                        if (code >= 0xd800 && code <= 0xdbff) {
                            if (position_ + 2 > text_.size() || text_[position_++] != '\\' || text_[position_++] != 'u') return false;
                            std::uint32_t low = 0;
                            if (!read_hex4(low) || low < 0xdc00 || low > 0xdfff) return false;
                            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                        } else if (code >= 0xdc00 && code <= 0xdfff)
                            return false;
                        if (code == 0) return false;
                        append_utf8(out, code);
                        break;
                    }
                    default:
                        return false;
                }
            }
            // The complete document has its own hard cap; this also permits a
            // bounded canonical schema string larger than ordinary field names.
            if (out.size() > limits_.document_bytes) return false;
        }
        return false;
    }
    bool read_hex4(std::uint32_t &value) {
        if (position_ + 4 > text_.size()) return false;
        value = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned digit = 0;
            if (!hex_value(text_[position_++], digit)) return false;
            value = (value << 4) | digit;
        }
        return true;
    }
    bool parse_number(Json &out) {
        const auto start = position_;
        if (position_ < text_.size() && text_[position_] == '-') ++position_;
        if (position_ >= text_.size()) return false;
        if (text_[position_] == '0')
            ++position_;
        else {
            if (text_[position_] < '1' || text_[position_] > '9') return false;
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const auto fraction = position_;
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
            if (fraction == position_) return false;
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            const auto exp = position_;
            while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
            if (exp == position_) return false;
        }
        out.kind = Json::Kind::Number;
        out.scalar = std::string(text_.substr(start, position_ - start));
        return true;
    }
    bool parse_array(Json &out, std::size_t depth) {
        ++position_;
        out.kind = Json::Kind::Array;
        skip_ws();
        if (position_ < text_.size() && text_[position_] == ']') {
            ++position_;
            return true;
        }
        while (out.array.size() < limits_.payloads + limits_.fields + 64) {
            Json item;
            skip_ws();
            if (!parse_value(item, depth)) return false;
            out.array.push_back(std::move(item));
            skip_ws();
            if (position_ >= text_.size()) return false;
            if (text_[position_] == ']') {
                ++position_;
                return true;
            }
            if (text_[position_++] != ',') return false;
        }
        return false;
    }
    bool parse_object(Json &out, std::size_t depth) {
        ++position_;
        out.kind = Json::Kind::Object;
        skip_ws();
        if (position_ < text_.size() && text_[position_] == '}') {
            ++position_;
            return true;
        }
        while (out.object.size() < limits_.payloads + limits_.fields + 64) {
            skip_ws();
            if (position_ >= text_.size() || text_[position_] != '"') return false;
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (position_ >= text_.size() || text_[position_++] != ':') return false;
            Json value;
            skip_ws();
            if (!parse_value(value, depth) || !out.object.emplace(std::move(key), std::move(value)).second) return false;
            skip_ws();
            if (position_ >= text_.size()) return false;
            if (text_[position_] == '}') {
                ++position_;
                return true;
            }
            if (text_[position_++] != ',') return false;
        }
        return false;
    }
    std::string_view text_;
    const CatalogLimits &limits_;
    std::size_t position_ = 0;
};

void quote(std::string_view value, std::string &out) {
    static constexpr char hex[] = "0123456789ABCDEF";
    out.push_back('"');
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0xf]);
                } else
                    out.push_back(static_cast<char>(c));
        }
    }
    out.push_back('"');
}

void key(std::string_view name, std::string &out, bool &first) {
    if (!first) out.push_back(',');
    first = false;
    quote(name, out);
    out.push_back(':');
}
const Json *member(const Json &value, std::string_view name) {
    if (value.kind != Json::Kind::Object) return nullptr;
    const auto it = value.object.find(name);
    return it == value.object.end() ? nullptr : &it->second;
}
bool string_value(const Json *v, std::string &out) {
    if (!v || v->kind != Json::Kind::String) return false;
    out = v->scalar;
    return true;
}
bool uint_value(const Json *v, std::uint64_t &out) {
    if (!v || v->kind != Json::Kind::Number || v->scalar.empty() || v->scalar[0] == '-' || v->scalar.find_first_of(".eE") != std::string::npos)
        return false;
    const auto result = std::from_chars(v->scalar.data(), v->scalar.data() + v->scalar.size(), out);
    return result.ec == std::errc{} && result.ptr == v->scalar.data() + v->scalar.size();
}
bool array_strings(const Json *v, std::vector<std::string> &out, std::size_t limit) {
    if (!v || v->kind != Json::Kind::Array || v->array.size() > limit) return false;
    out.clear();
    std::set<std::string> unique;
    for (const auto &item : v->array) {
        if (item.kind != Json::Kind::String || !unique.insert(item.scalar).second) return false;
        out.push_back(item.scalar);
    }
    return true;
}
bool array_u64(const Json *v, std::vector<std::uint64_t> &out, std::size_t limit) {
    if (!v || v->kind != Json::Kind::Array || v->array.size() > limit) return false;
    out.clear();
    for (const auto &item : v->array) {
        std::uint64_t x = 0;
        if (!uint_value(&item, x)) return false;
        out.push_back(x);
    }
    return true;
}
bool array_message_ranges(const Json *v, std::vector<PayloadDescriptor::MessageRange> &out, std::size_t limit, std::uint64_t payload_bytes) {
    if (!v || v->kind != Json::Kind::Array || v->array.size() > limit) return false;
    out.clear();
    std::uint64_t previous_end = 0;
    for (const auto &item : v->array) {
        if (item.kind != Json::Kind::Array || item.array.size() != 2) return false;
        PayloadDescriptor::MessageRange range;
        if (!uint_value(&item.array[0], range.offset) || !uint_value(&item.array[1], range.length) || range.length == 0 ||
            range.offset < previous_end || range.offset > payload_bytes || range.length > payload_bytes - range.offset)
            return false;
        previous_end = range.offset + range.length;
        out.push_back(range);
    }
    return true;
}
bool valid_fields(const std::vector<std::string> &fields, const CatalogLimits &limits) {
    if (fields.size() > limits.fields) return false;
    for (const auto &field : fields)
        if (field.empty() || field.size() > limits.string_bytes || field.find('\0') != std::string::npos) return false;
    return true;
}
void encode_strings(const std::vector<std::string> &values, std::string &out) {
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out.push_back(',');
        quote(values[i], out);
    }
    out.push_back(']');
}
void encode_u64s(const std::vector<std::uint64_t> &values, std::string &out) {
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out.push_back(',');
        out += std::to_string(values[i]);
    }
    out.push_back(']');
}
void encode_message_ranges(const std::vector<PayloadDescriptor::MessageRange> &values, std::string &out) {
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out.push_back(',');
        out.push_back('[');
        out += std::to_string(values[i].offset);
        out.push_back(',');
        out += std::to_string(values[i].length);
        out.push_back(']');
    }
    out.push_back(']');
}
bool parse_document(std::string_view json, const CatalogLimits &limits, Json &value) {
    return json.size() <= limits.document_bytes && Parser(json, limits).parse(value);
}

}  // namespace

bool catalog_path_is_safe(std::string_view path) {
    if (path.empty() || path.front() == '/' || path.find('\\') != std::string_view::npos || path.find('\0') != std::string_view::npos) return false;
    std::size_t start = 0;
    while (start <= path.size()) {
        const auto end = path.find('/', start);
        const auto part = path.substr(start, end == std::string_view::npos ? path.size() - start : end - start);
        if (part.empty() || part == "." || part == "..") return false;
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return true;
}

bool read_bounded_catalog_file(const std::filesystem::path &path, std::size_t max_bytes, std::string &bytes, bool &missing) {
    missing = false;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        missing = errno == ENOENT;
        return false;
    }
    struct stat st{};
    bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) <= max_bytes;
    if (ok) {
        bytes.assign(static_cast<std::size_t>(st.st_size), '\0');
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto count = ::read(fd, bytes.data() + offset, bytes.size() - offset);
            if (count < 0) {
                if (errno == EINTR) continue;
                ok = false;
                break;
            }
            if (count == 0) {
                ok = false;
                break;
            }
            offset += static_cast<std::size_t>(count);
        }
    }
    return ::close(fd) == 0 && ok;
}

bool parse_catalog_head(std::string_view text, std::string_view uuid, std::uint64_t &step, std::string &commit) {
    const auto prefix = std::string("{\"format_version\":1,\"dataset_uuid\":\"") + std::string(uuid) + "\",\"step_id\":";
    constexpr std::string_view marker = ",\"commit\":\"";
    if (!text.starts_with(prefix)) return false;
    const auto marker_at = text.find(marker, prefix.size());
    if (marker_at == std::string_view::npos || marker_at == prefix.size() || text.size() < marker_at + marker.size() + 2 ||
        text.substr(text.size() - 2) != "\"}")
        return false;
    const auto parsed = std::from_chars(text.data() + prefix.size(), text.data() + marker_at, step);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + marker_at) return false;
    commit.assign(text.substr(marker_at + marker.size(), text.size() - marker_at - marker.size() - 2));
    return catalog_path_is_safe(commit) && commit.starts_with("commits/");
}

std::string encode_field_path(std::string_view field) {
    static constexpr char hex[] = "0123456789ABCDEF";
    if (field.empty() || field.size() > 1024 * 1024 || field.find('\0') != std::string_view::npos) return {};
    std::string out;
    out.reserve(field.size());
    for (const unsigned char c : field) {
        const bool unreserved =
            (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '~' || c == '.';
        if (unreserved && !(field == "." || field == ".."))
            out.push_back(static_cast<char>(c));
        else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0xf]);
        }
    }
    return out;
}

bool decode_field_path(std::string_view component, std::string &field, std::size_t max_bytes) {
    std::string decoded;
    decoded.reserve(component.size());
    for (std::size_t i = 0; i < component.size(); ++i) {
        if (component[i] != '%')
            decoded.push_back(component[i]);
        else {
            unsigned hi = 0, lo = 0;
            if (i + 2 >= component.size() || !hex_value(component[i + 1], hi) || !hex_value(component[i + 2], lo)) return false;
            decoded.push_back(static_cast<char>((hi << 4) | lo));
            i += 2;
        }
        if (decoded.size() > max_bytes) return false;
    }
    if (decoded.empty() || decoded.find('\0') != std::string::npos || encode_field_path(decoded) != component) return false;
    field = std::move(decoded);
    return true;
}

std::string encode_typed_double(double value) {
    if (std::isfinite(value)) {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return out.str();
    }
    const char *tag = std::isnan(value) ? "nan" : (std::signbit(value) ? "-infinity" : "infinity");
    return std::string("{\"$amio\":\"float64\",\"value\":\"") + tag + "\"}";
}

amio_status_t encode_schema_document(const SchemaDocument &schema, std::string &json, const CatalogLimits &limits) {
    if (schema.format_version != 1 || schema.dataset_uuid.empty() || schema.digest.empty() || schema.canonical_schema.empty() ||
        !valid_fields(schema.required_fields, limits) || schema.dataset_uuid.size() > limits.string_bytes ||
        schema.digest.size() > limits.string_bytes || schema.canonical_schema.size() > limits.document_bytes)
        return AMIO_ERR_INVALID_INPUT;
    std::string out = "{";
    bool first = true;
    key("format_version", out, first);
    out += "1";
    key("dataset_uuid", out, first);
    quote(schema.dataset_uuid, out);
    key("digest", out, first);
    quote(schema.digest, out);
    key("canonical_schema", out, first);
    quote(schema.canonical_schema, out);
    key("required_fields", out, first);
    encode_strings(schema.required_fields, out);
    out.push_back('}');
    if (out.size() > limits.document_bytes) return AMIO_ERR_BUDGET_EXCEEDED;
    json = std::move(out);
    return AMIO_OK;
}

amio_status_t parse_schema_document(std::string_view json, SchemaDocument &schema, const CatalogLimits &limits) {
    Json root;
    if (!parse_document(json, limits, root) || root.kind != Json::Kind::Object || root.object.size() != 5) return AMIO_ERR_INVALID_INPUT;
    std::uint64_t version = 0;
    SchemaDocument parsed;
    if (!uint_value(member(root, "format_version"), version) || version != 1 || !string_value(member(root, "dataset_uuid"), parsed.dataset_uuid) ||
        !string_value(member(root, "digest"), parsed.digest) || !string_value(member(root, "canonical_schema"), parsed.canonical_schema) ||
        !array_strings(member(root, "required_fields"), parsed.required_fields, limits.fields) || parsed.dataset_uuid.empty() ||
        parsed.digest.empty() || parsed.canonical_schema.empty() || !valid_fields(parsed.required_fields, limits))
        return AMIO_ERR_INVALID_INPUT;
    parsed.format_version = 1;
    schema = std::move(parsed);
    return AMIO_OK;
}

amio_status_t encode_commit_document(const CompletionRecord &record, std::string &json, const CatalogLimits &limits) {
    if (record.format_version != 1 || record.dataset_uuid.empty() || record.schema_digest.empty() || record.achieved_guarantee.empty() ||
        record.dataset_uuid.size() > limits.string_bytes || record.schema_digest.size() > limits.string_bytes ||
        record.achieved_guarantee.size() > limits.string_bytes || record.previous_commit.size() > limits.string_bytes ||
        (record.achieved_guarantee != "visibility" && record.achieved_guarantee != "file_sync" && record.achieved_guarantee != "directory_sync") ||
        (!record.previous_commit.empty() && !catalog_path_is_safe(record.previous_commit)) || !valid_fields(record.required_fields, limits) ||
        !valid_fields(record.present_fields, limits) || record.payloads.size() > limits.payloads)
        return AMIO_ERR_INVALID_INPUT;
    std::set<std::string> present(record.present_fields.begin(), record.present_fields.end());
    for (const auto &field : record.required_fields)
        if (!present.contains(field)) return AMIO_ERR_FIELD_NOT_PRESENT;
    std::string out = "{";
    bool first = true;
    key("format_version", out, first);
    out += "1";
    key("dataset_uuid", out, first);
    quote(record.dataset_uuid, out);
    key("step_id", out, first);
    out += std::to_string(record.step_id);
    key("schema_digest", out, first);
    quote(record.schema_digest, out);
    key("required_fields", out, first);
    encode_strings(record.required_fields, out);
    key("present_fields", out, first);
    encode_strings(record.present_fields, out);
    key("achieved_guarantee", out, first);
    quote(record.achieved_guarantee, out);
    key("previous_commit", out, first);
    quote(record.previous_commit, out);
    key("payloads", out, first);
    out.push_back('[');
    for (std::size_t i = 0; i < record.payloads.size(); ++i) {
        const auto &p = record.payloads[i];
        if (p.field.empty() || p.field.size() > limits.string_bytes || !catalog_path_is_safe(p.relative_path) ||
            p.relative_path.size() > limits.string_bytes || p.offsets.size() != p.extents.size() || p.offsets.size() > limits.dimensions ||
            p.message_ranges.size() > limits.payloads || !present.contains(p.field))
            return AMIO_ERR_INVALID_INPUT;
        std::uint64_t previous_end = 0;
        for (const auto &range : p.message_ranges) {
            if (range.length == 0 || range.offset < previous_end || range.offset > p.bytes || range.length > p.bytes - range.offset)
                return AMIO_ERR_INVALID_INPUT;
            previous_end = range.offset + range.length;
        }
        if (i) out.push_back(',');
        out.push_back('{');
        bool pf = true;
        key("field", out, pf);
        quote(p.field, out);
        key("path", out, pf);
        quote(p.relative_path, out);
        key("bytes", out, pf);
        out += std::to_string(p.bytes);
        key("producer", out, pf);
        out += std::to_string(p.producer);
        key("offsets", out, pf);
        encode_u64s(p.offsets, out);
        key("extents", out, pf);
        encode_u64s(p.extents, out);
        if (!p.message_ranges.empty()) {
            key("message_ranges", out, pf);
            encode_message_ranges(p.message_ranges, out);
        }
        if (p.array_container) {
            key("array_container", out, pf);
            out += "true";
        }
        if (!p.integrity_sha256.empty()) {
            if (!p.integrity_sha256.starts_with("sha256-logical-v1:") || p.integrity_sha256.size() != 82) return AMIO_ERR_INVALID_INPUT;
            key("integrity_sha256", out, pf);
            quote(p.integrity_sha256, out);
        }
        if (!p.operator_pipeline.empty()) {
            if (p.operator_pipeline.size() > limits.string_bytes) return AMIO_ERR_INVALID_INPUT;
            key("operator_pipeline", out, pf);
            quote(p.operator_pipeline, out);
        }
        if (!p.integrity_regions.empty()) {
            if (p.integrity_regions.size() > limits.payloads) return AMIO_ERR_BUDGET_EXCEEDED;
            key("integrity_regions", out, pf);
            out.push_back('[');
            for (std::size_t region_index = 0; region_index < p.integrity_regions.size(); ++region_index) {
                const auto &region = p.integrity_regions[region_index];
                if (region.offsets.size() != region.extents.size() || region.offsets.size() > limits.dimensions || region.digest.size() != 82 ||
                    !region.digest.starts_with("sha256-logical-v1:"))
                    return AMIO_ERR_INVALID_INPUT;
                if (region_index) out.push_back(',');
                out.push_back('{');
                bool rf = true;
                key("offsets", out, rf);
                encode_u64s(region.offsets, out);
                key("extents", out, rf);
                encode_u64s(region.extents, out);
                key("digest", out, rf);
                quote(region.digest, out);
                out.push_back('}');
            }
            out.push_back(']');
        }
        out.push_back('}');
    }
    out += "]}";
    if (out.size() > limits.document_bytes) return AMIO_ERR_BUDGET_EXCEEDED;
    json = std::move(out);
    return AMIO_OK;
}

amio_status_t parse_commit_document(std::string_view json, CompletionRecord &record, const CatalogLimits &limits) {
    Json root;
    if (!parse_document(json, limits, root) || root.kind != Json::Kind::Object || root.object.size() != 9) return AMIO_ERR_INVALID_INPUT;
    std::uint64_t version = 0;
    CompletionRecord parsed;
    if (!uint_value(member(root, "format_version"), version) || version != 1 || !string_value(member(root, "dataset_uuid"), parsed.dataset_uuid) ||
        !uint_value(member(root, "step_id"), parsed.step_id) || !string_value(member(root, "schema_digest"), parsed.schema_digest) ||
        !array_strings(member(root, "required_fields"), parsed.required_fields, limits.fields) ||
        !array_strings(member(root, "present_fields"), parsed.present_fields, limits.fields) ||
        !string_value(member(root, "achieved_guarantee"), parsed.achieved_guarantee) ||
        !string_value(member(root, "previous_commit"), parsed.previous_commit) || parsed.dataset_uuid.empty() || parsed.schema_digest.empty() ||
        (parsed.achieved_guarantee != "visibility" && parsed.achieved_guarantee != "file_sync" && parsed.achieved_guarantee != "directory_sync") ||
        !valid_fields(parsed.required_fields, limits) || !valid_fields(parsed.present_fields, limits) ||
        parsed.dataset_uuid.size() > limits.string_bytes || parsed.schema_digest.size() > limits.string_bytes ||
        parsed.achieved_guarantee.size() > limits.string_bytes || parsed.previous_commit.size() > limits.string_bytes ||
        (!parsed.previous_commit.empty() && !catalog_path_is_safe(parsed.previous_commit)))
        return AMIO_ERR_INVALID_INPUT;
    const auto *payloads = member(root, "payloads");
    if (!payloads || payloads->kind != Json::Kind::Array || payloads->array.size() > limits.payloads) return AMIO_ERR_INVALID_INPUT;
    std::set<std::string> present(parsed.present_fields.begin(), parsed.present_fields.end());
    for (const auto &field : parsed.required_fields)
        if (!present.contains(field)) return AMIO_ERR_FIELD_NOT_PRESENT;
    for (const auto &item : payloads->array) {
        if (item.kind != Json::Kind::Object || item.object.size() < 6 || item.object.size() > 11) return AMIO_ERR_INVALID_INPUT;
        PayloadDescriptor p;
        std::uint64_t producer = 0;
        if (!string_value(member(item, "field"), p.field) || !string_value(member(item, "path"), p.relative_path) ||
            !uint_value(member(item, "bytes"), p.bytes) || !uint_value(member(item, "producer"), producer) || producer > UINT32_MAX ||
            !array_u64(member(item, "offsets"), p.offsets, limits.dimensions) || !array_u64(member(item, "extents"), p.extents, limits.dimensions) ||
            p.offsets.size() != p.extents.size() || p.field.empty() || p.field.size() > limits.string_bytes ||
            p.relative_path.size() > limits.string_bytes || !present.contains(p.field) || !catalog_path_is_safe(p.relative_path))
            return AMIO_ERR_INVALID_INPUT;
        const auto *ranges = member(item, "message_ranges");
        const auto *array_container = member(item, "array_container");
        const auto *integrity = member(item, "integrity_sha256");
        const auto *integrity_regions = member(item, "integrity_regions");
        const auto *operator_pipeline = member(item, "operator_pipeline");
        if ((ranges != nullptr) != (item.object.contains("message_ranges")) ||
            (array_container != nullptr && array_container->kind != Json::Kind::Boolean) ||
            (integrity != nullptr && (!string_value(integrity, p.integrity_sha256) || !p.integrity_sha256.starts_with("sha256-logical-v1:") ||
                                      p.integrity_sha256.size() != 82)) ||
            item.object.size() != 6 + (ranges != nullptr ? 1 : 0) + (array_container != nullptr ? 1 : 0) + (integrity != nullptr ? 1 : 0) +
                                      (integrity_regions != nullptr ? 1 : 0) + (operator_pipeline != nullptr ? 1 : 0))
            return AMIO_ERR_INVALID_INPUT;
        if (operator_pipeline != nullptr &&
            (!string_value(operator_pipeline, p.operator_pipeline) || p.operator_pipeline.size() > limits.string_bytes))
            return AMIO_ERR_INVALID_INPUT;
        if (ranges != nullptr) {
            if (!array_message_ranges(ranges, p.message_ranges, limits.payloads, p.bytes)) return AMIO_ERR_INVALID_INPUT;
        }
        if (integrity_regions != nullptr) {
            if (integrity_regions->kind != Json::Kind::Array || integrity_regions->array.empty() || integrity_regions->array.size() > limits.payloads)
                return AMIO_ERR_INVALID_INPUT;
            for (const auto &region_json : integrity_regions->array) {
                if (region_json.kind != Json::Kind::Object || region_json.object.size() != 3) return AMIO_ERR_INVALID_INPUT;
                PayloadDescriptor::IntegrityRegion region;
                if (!array_u64(member(region_json, "offsets"), region.offsets, limits.dimensions) ||
                    !array_u64(member(region_json, "extents"), region.extents, limits.dimensions) ||
                    !string_value(member(region_json, "digest"), region.digest) || region.offsets.size() != region.extents.size() ||
                    region.digest.size() != 82 || !region.digest.starts_with("sha256-logical-v1:"))
                    return AMIO_ERR_INVALID_INPUT;
                p.integrity_regions.push_back(std::move(region));
            }
        }
        p.array_container = array_container != nullptr && array_container->scalar == "true";
        p.producer = static_cast<std::uint32_t>(producer);
        parsed.payloads.push_back(std::move(p));
    }
    parsed.format_version = 1;
    record = std::move(parsed);
    return AMIO_OK;
}

}  // namespace amio::detail
