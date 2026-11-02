#ifndef AMIO_SRC_SCHEMA_SCHEMA_HPP
#define AMIO_SRC_SCHEMA_SCHEMA_HPP

#include "amio/amio_types.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <utility>

namespace amio::detail {

struct SchemaStatus {
    bool success = true;
    std::string message;

    [[nodiscard]] bool ok() const noexcept { return success; }
    static SchemaStatus failure(std::string message) { return {false, std::move(message)}; }
};

using AttributeValue = std::variant<bool, std::int64_t, std::uint64_t, double, std::string, std::vector<std::int64_t>,
                                    std::vector<std::uint64_t>, std::vector<double>, std::vector<std::string>>;

struct Attribute {
    AttributeValue value;

    [[nodiscard]] std::size_t count() const noexcept;
};

struct Dimension {
    std::string name;
    std::uint64_t extent = 0;
    std::optional<std::string> coordinate_variable;
};

struct Variable {
    std::string name;
    amio_dtype_t source_dtype = AMIO_DTYPE_F32;
    amio_dtype_t stored_dtype = AMIO_DTYPE_F32;
    std::vector<Dimension> dimensions;
    bool required = true;
    std::map<std::string, Attribute, std::less<>> attributes;
};

class Schema {
   public:
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::size_t kMaxVariables = 10000;
    static constexpr std::size_t kMaxAttributesPerVariable = 256;
    static constexpr std::size_t kMaxStringBytes = 1024 * 1024;
    static constexpr std::size_t kMaxCanonicalBytes = 16 * 1024 * 1024;

    SchemaStatus add_variable(Variable variable);
    SchemaStatus seal();

    /** Decode a canonical schema from a bounded wire representation.
     * The destination remains unchanged on failure. The caller must account for
     * both the wire representation and the resulting schema in its metadata budget.
     */
    static SchemaStatus from_canonical(std::string_view wire, std::size_t max_wire_bytes, Schema &out);

    [[nodiscard]] bool sealed() const noexcept { return sealed_; }
    [[nodiscard]] const std::map<std::string, Variable, std::less<>> &variables() const noexcept { return variables_; }
    [[nodiscard]] const std::string &canonical() const noexcept { return canonical_; }
    [[nodiscard]] const std::string &digest() const noexcept { return digest_; }
    [[nodiscard]] bool equivalent(const Schema &other) const noexcept;

   private:
    std::map<std::string, Variable, std::less<>> variables_;
    std::string canonical_;
    std::string digest_;
    bool sealed_ = false;
};

/** FNV-1a 64-bit digest, rendered as 16 lowercase hex digits. */
[[nodiscard]] std::string schema_digest(std::string_view canonical);

} // namespace amio::detail

#endif // AMIO_SRC_SCHEMA_SCHEMA_HPP
