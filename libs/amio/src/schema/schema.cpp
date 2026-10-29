#include "schema/schema.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <type_traits>
#include <utility>

namespace amio::detail {
namespace {

constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

bool valid_utf8(std::string_view text) {
    if (text.size() > Schema::kMaxStringBytes) {
        return false;
    }
    const auto *bytes = reinterpret_cast<const unsigned char *>(text.data());
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = bytes[i];
        if (lead == 0) {
            return false;
        }
        if (lead <= 0x7f) {
            ++i;
            continue;
        }
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        if ((lead & 0xe0) == 0xc0) {
            length = 2;
            codepoint = lead & 0x1f;
        } else if ((lead & 0xf0) == 0xe0) {
            length = 3;
            codepoint = lead & 0x0f;
        } else if ((lead & 0xf8) == 0xf0) {
            length = 4;
            codepoint = lead & 0x07;
        } else {
            return false;
        }
        if (i + length > text.size()) {
            return false;
        }
        for (std::size_t j = 1; j < length; ++j) {
            const unsigned char continuation = bytes[i + j];
            if ((continuation & 0xc0) != 0x80) {
                return false;
            }
            codepoint = (codepoint << 6) | (continuation & 0x3f);
        }
        if ((length == 2 && codepoint < 0x80) || (length == 3 && codepoint < 0x800) || (length == 4 && codepoint < 0x10000) ||
            codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
        i += length;
    }
    return true;
}

bool valid_dtype(amio_dtype_t dtype) {
    const auto value = static_cast<int>(dtype);
    return value >= static_cast<int>(AMIO_DTYPE_F32) && value <= static_cast<int>(AMIO_DTYPE_U64);
}

void append_text(std::ostringstream &out, std::string_view text) {
    out << text.size() << ':' << text << ';';
}

void append_double(std::ostringstream &out, double value) {
    if (std::isnan(value)) {
        out << "nan;";
    } else if (std::isinf(value)) {
        out << (std::signbit(value) ? "-inf;" : "inf;");
    } else {
        out << std::hexfloat << value << std::defaultfloat << ';';
    }
}

void append_attribute(std::ostringstream &out, const Attribute &attribute) {
    out << attribute.value.index() << ':' << attribute.count() << ':';
    std::visit(
        [&out](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, bool>) {
                out << (value ? '1' : '0') << ';';
            } else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>) {
                out << value << ';';
            } else if constexpr (std::is_same_v<T, double>) {
                append_double(out, value);
            } else if constexpr (std::is_same_v<T, std::string>) {
                append_text(out, value);
            } else {
                for (const auto &item : value) {
                    if constexpr (std::is_same_v<typename T::value_type, double>) {
                        append_double(out, item);
                    } else if constexpr (std::is_same_v<typename T::value_type, std::string>) {
                        append_text(out, item);
                    } else {
                        out << item << ';';
                    }
                }
            }
        },
        attribute.value);
}

bool valid_attribute(const Attribute &attribute) {
    return std::visit(
        [](const auto &value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return valid_utf8(value);
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                std::size_t total_bytes = 0;
                for (const auto &item : value) {
                    if (!valid_utf8(item) || item.size() > Schema::kMaxCanonicalBytes - total_bytes) {
                        return false;
                    }
                    total_bytes += item.size();
                }
                return value.size() <= Schema::kMaxCanonicalBytes / sizeof(std::uint64_t);
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>> || std::is_same_v<T, std::vector<std::uint64_t>> ||
                                 std::is_same_v<T, std::vector<double>>) {
                return value.size() <= Schema::kMaxCanonicalBytes / sizeof(typename T::value_type);
            } else {
                return true;
            }
        },
        attribute.value);
}

std::string canonicalize(const std::map<std::string, Variable, std::less<>> &variables) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "amio-schema-v" << Schema::kVersion << ";digest=fnv1a64-v1;variables=" << variables.size() << ';';
    for (const auto &[name, variable] : variables) {
        append_text(out, name);
        out << static_cast<int>(variable.source_dtype) << ';' << static_cast<int>(variable.stored_dtype) << ';' << (variable.required ? '1' : '0')
            << ';' << variable.dimensions.size() << ';';
        for (const auto &dimension : variable.dimensions) {
            append_text(out, dimension.name);
            out << dimension.extent << ';';
            out << (dimension.coordinate_variable ? '1' : '0') << ';';
            if (dimension.coordinate_variable) {
                append_text(out, *dimension.coordinate_variable);
            }
        }
        out << variable.attributes.size() << ';';
        for (const auto &[attribute_name, attribute] : variable.attributes) {
            append_text(out, attribute_name);
            append_attribute(out, attribute);
        }
    }
    return out.str();
}

bool account_bytes(std::size_t &total, std::size_t bytes) {
    if (bytes > Schema::kMaxCanonicalBytes - total) {
        return false;
    }
    total += bytes;
    return true;
}

std::size_t estimate_attribute_bytes(const Attribute &attribute) {
    return std::visit(
        [](const auto &value) -> std::size_t {
            using T = std::decay_t<decltype(value)>;
            constexpr auto too_large = Schema::kMaxCanonicalBytes + 1;
            if constexpr (std::is_same_v<T, std::string>) {
                return value.size() + 32;
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>> || std::is_same_v<T, std::vector<std::uint64_t>> ||
                                 std::is_same_v<T, std::vector<double>>) {
                if (value.size() > Schema::kMaxCanonicalBytes / 32) {
                    return too_large;
                }
                return value.size() * 32 + 32;
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                std::size_t total = 32;
                for (const auto &item : value) {
                    if (total > Schema::kMaxCanonicalBytes - 32 || item.size() > Schema::kMaxCanonicalBytes - total - 32) {
                        return too_large;
                    }
                    total += item.size() + 32;
                }
                return total;
            } else {
                return 64;
            }
        },
        attribute.value);
}

} // namespace

std::size_t Attribute::count() const noexcept {
    return std::visit(
        [](const auto &value) -> std::size_t {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::vector<std::int64_t>> || std::is_same_v<T, std::vector<std::uint64_t>> ||
                          std::is_same_v<T, std::vector<double>> || std::is_same_v<T, std::vector<std::string>>) {
                return value.size();
            }
            return 1;
        },
        value);
}

SchemaStatus Schema::add_variable(Variable variable) {
    if (sealed_) {
        return SchemaStatus::failure("schema is immutable after seal");
    }
    if (variables_.size() >= kMaxVariables) {
        return SchemaStatus::failure("schema variable limit exceeded");
    }
    if (variable.name.empty() || !valid_utf8(variable.name) || variable.name.size() > 255) {
        return SchemaStatus::failure("variable name must be nonempty UTF-8 no longer than 255 bytes");
    }
    if (!valid_dtype(variable.source_dtype) || !valid_dtype(variable.stored_dtype)) {
        return SchemaStatus::failure("source and stored dtypes must be supported AMIO types");
    }
    if (variable.dimensions.empty() || variable.dimensions.size() > AMIO_MAX_RANK) {
        return SchemaStatus::failure("variable rank must be in [1, AMIO_MAX_RANK]");
    }
    std::set<std::string, std::less<>> dimension_names;
    for (const auto &dimension : variable.dimensions) {
        if (dimension.name.empty() || !valid_utf8(dimension.name) || dimension.name.size() > 255 || dimension.extent == 0 ||
            dimension.extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return SchemaStatus::failure("dimension name or extent is invalid");
        }
        if (!dimension_names.insert(dimension.name).second) {
            return SchemaStatus::failure("dimension names must be unique within a variable");
        }
        if (dimension.coordinate_variable &&
            (dimension.coordinate_variable->empty() || !valid_utf8(*dimension.coordinate_variable) || dimension.coordinate_variable->size() > 255)) {
            return SchemaStatus::failure("coordinate variable reference must be nonempty UTF-8");
        }
    }
    if (variable.attributes.size() > kMaxAttributesPerVariable) {
        return SchemaStatus::failure("attribute count exceeds the per-variable limit");
    }
    for (const auto &[name, attribute] : variable.attributes) {
        if (name.empty() || !valid_utf8(name) || name.size() > 255 || !valid_attribute(attribute)) {
            return SchemaStatus::failure("attribute name or value is invalid or exceeds its bound");
        }
    }
    if (variables_.contains(variable.name)) {
        return SchemaStatus::failure("variable names must be unique within a schema");
    }
    variables_.emplace(variable.name, std::move(variable));
    return {};
}

SchemaStatus Schema::seal() {
    if (sealed_) {
        return {};
    }
    if (variables_.empty()) {
        return SchemaStatus::failure("schema must contain at least one variable");
    }
    std::size_t estimated_bytes = 64;
    for (const auto &[name, variable] : variables_) {
        if (!account_bytes(estimated_bytes, name.size() + 64)) {
            return SchemaStatus::failure("schema metadata exceeds its byte limit");
        }
        for (const auto &dimension : variable.dimensions) {
            if (!account_bytes(estimated_bytes, dimension.name.size() + 48) ||
                (dimension.coordinate_variable && !account_bytes(estimated_bytes, dimension.coordinate_variable->size() + 16))) {
                return SchemaStatus::failure("schema metadata exceeds its byte limit");
            }
            if (dimension.coordinate_variable && !variables_.contains(*dimension.coordinate_variable)) {
                return SchemaStatus::failure("coordinate variable reference does not resolve in this schema");
            }
        }
        for (const auto &[attribute_name, attribute] : variable.attributes) {
            if (!account_bytes(estimated_bytes, attribute_name.size() + estimate_attribute_bytes(attribute))) {
                return SchemaStatus::failure("schema metadata exceeds its byte limit");
            }
        }
    }
    canonical_ = canonicalize(variables_);
    if (canonical_.size() > kMaxCanonicalBytes) {
        canonical_.clear();
        return SchemaStatus::failure("canonical schema exceeds its byte limit");
    }
    digest_ = schema_digest(canonical_);
    sealed_ = true;
    return {};
}

bool Schema::equivalent(const Schema &other) const noexcept {
    return sealed_ && other.sealed_ && digest_ == other.digest_ && canonical_ == other.canonical_;
}

std::string schema_digest(std::string_view canonical) {
    std::uint64_t hash = kFnvOffset;
    for (const unsigned char byte : canonical) {
        hash ^= byte;
        hash *= kFnvPrime;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

} // namespace amio::detail
