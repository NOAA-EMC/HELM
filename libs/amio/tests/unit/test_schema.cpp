#include "schema/schema.hpp"

#include <cassert>
#include <cstdint>
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

} // namespace

int main() {
    test_fnv_known_answer();
    test_canonical_order_and_typed_attributes();
    test_rejects_bad_names_and_coordinate_references();
    test_rejects_duplicate_names_and_nonfinite_extents();
    return 0;
}
