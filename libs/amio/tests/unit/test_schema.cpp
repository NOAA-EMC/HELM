#include "schema/schema.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace {

amio::detail::Variable scalar(std::string name, amio_dtype_t dtype) {
    amio::detail::Variable variable;
    variable.name = std::move(name);
    variable.source_dtype = dtype;
    variable.stored_dtype = dtype;
    variable.required = true;
    variable.dimensions.push_back({"cell", 8, std::nullopt});
    return variable;
}

void test_fnv_known_answer() {
    assert(amio::detail::schema_digest("hello") == "a430d84680aabd0b");
}

void test_canonical_order_and_typed_attributes() {
    amio::detail::Schema first;
    auto a = scalar("temperature", AMIO_DTYPE_F64);
    a.attributes.emplace("units", amio::detail::Attribute{std::string("K")});
    a.attributes.emplace("levels", amio::detail::Attribute{std::vector<std::int64_t>{100, 200}});
    auto b = scalar("pressure", AMIO_DTYPE_F32);
    assert(first.add_variable(a).ok());
    assert(first.add_variable(b).ok());
    assert(first.seal().ok());

    amio::detail::Schema second;
    assert(second.add_variable(b).ok());
    assert(second.add_variable(a).ok());
    assert(second.seal().ok());
    assert(first.canonical() == second.canonical());
    assert(first.digest() == second.digest());
    assert(first.variables().at("temperature").attributes.at("levels").count() == 2);
}

void test_rejects_bad_names_and_coordinate_references() {
    amio::detail::Schema schema;
    auto bad_utf8 = scalar(std::string("bad\xC0\xAF", 5), AMIO_DTYPE_F32);
    assert(!schema.add_variable(bad_utf8).ok());

    amio::detail::Schema unresolved;
    auto field = scalar("temperature", AMIO_DTYPE_F32);
    field.dimensions.front().coordinate_variable = "latitude";
    assert(unresolved.add_variable(field).ok());
    assert(!unresolved.seal().ok());
}

void test_rejects_duplicate_names_and_nonfinite_extents() {
    amio::detail::Schema schema;
    assert(schema.add_variable(scalar("temperature", AMIO_DTYPE_F32)).ok());
    assert(!schema.add_variable(scalar("temperature", AMIO_DTYPE_F32)).ok());

    amio::detail::Schema invalid_extent;
    auto variable = scalar("temperature", AMIO_DTYPE_F32);
    variable.dimensions.front().extent = 0;
    assert(!invalid_extent.add_variable(variable).ok());
}

void test_bounded_canonical_decode() {
    using namespace amio::detail;
    Schema original;
    auto field = scalar("field", AMIO_DTYPE_F64);
    field.attributes.emplace("bool", Attribute{true});
    field.attributes.emplace("signed", Attribute{std::int64_t{-4}});
    field.attributes.emplace("unsigned", Attribute{std::uint64_t{8}});
    field.attributes.emplace("real", Attribute{-1.25});
    field.attributes.emplace("text", Attribute{std::string("x:;y")});
    field.attributes.emplace("ints", Attribute{std::vector<std::int64_t>{-1, 2}});
    field.attributes.emplace("uints", Attribute{std::vector<std::uint64_t>{0, 2}});
    field.attributes.emplace("reals", Attribute{std::vector<double>{0.0, -0.0, std::numeric_limits<double>::denorm_min(),
                                                                   std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(),
                                                                   std::numeric_limits<double>::quiet_NaN()}});
    field.attributes.emplace("texts", Attribute{std::vector<std::string>{"", "ab"}});
    assert(original.add_variable(std::move(field)).ok());
    assert(original.seal().ok());
    Schema decoded;
    assert(Schema::from_canonical(original.canonical(), original.canonical().size(), decoded).ok());
    assert(original.equivalent(decoded));
    const auto digest = decoded.digest();
    assert(!Schema::from_canonical(original.canonical(), original.canonical().size() - 1, decoded).ok());
    assert(decoded.digest() == digest);
    auto malformed = original.canonical();
    malformed.pop_back();
    assert(!Schema::from_canonical(malformed, malformed.size(), decoded).ok());
    assert(decoded.digest() == digest);
    malformed = original.canonical() + "junk";
    assert(!Schema::from_canonical(malformed, malformed.size(), decoded).ok());
    assert(decoded.digest() == digest);
    const std::string excessive_count =
        "amio-schema-v1;digest=fnv1a64-v1;variables=1;1:x;0;0;1;1;1:d;1;0;1;1:a;5:18446744073709551615:";
    assert(!Schema::from_canonical(excessive_count, excessive_count.size(), decoded).ok());
    assert(decoded.digest() == digest);
    const std::string noncanonical =
        "amio-schema-v1;digest=fnv1a64-v1;variables=01;1:x;0;0;1;1;1:d;1;0;0;";
    assert(!Schema::from_canonical(noncanonical, noncanonical.size(), decoded).ok());
    assert(decoded.digest() == digest);
}

} // namespace

int main() {
    test_fnv_known_answer();
    test_canonical_order_and_typed_attributes();
    test_rejects_bad_names_and_coordinate_references();
    test_rejects_duplicate_names_and_nonfinite_extents();
    test_bounded_canonical_decode();
    return 0;
}
