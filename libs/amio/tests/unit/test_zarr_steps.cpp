#include "drivers/zarr/zarr_steps.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace amio::detail;

int main() {
    Schema schema;
    Variable temperature;
    temperature.name = "temperature";
    temperature.source_dtype = AMIO_DTYPE_F32;
    temperature.stored_dtype = AMIO_DTYPE_F32;
    temperature.required = true;
    temperature.dimensions.push_back({"y", 2, std::nullopt});
    temperature.dimensions.push_back({"x", 3, std::nullopt});
    assert(schema.add_variable(std::move(temperature)).ok());

    Variable optional;
    optional.name = "optional tendency";
    optional.source_dtype = AMIO_DTYPE_I16;
    optional.stored_dtype = AMIO_DTYPE_I16;
    optional.required = false;
    optional.dimensions.push_back({"cell", 3, std::nullopt});
    assert(schema.add_variable(std::move(optional)).ok());
    assert(schema.seal().ok());

    std::string error;
    ZarrStepLayout layout;
    assert(make_zarr_step_layout("/tmp/collection", 12, schema, layout, error));
    assert(layout.step_root.ends_with("/steps/00000000000000000012"));
    assert(layout.array_paths.size() == 2);
    assert(layout.array_paths[0].find("/fields/") != std::string::npos);
    assert(layout.array_paths[0] != layout.array_paths[1]); // opaque encoded names cannot collide
    assert(layout.array_paths[1].find("..") == std::string::npos);

    std::map<std::string, std::vector<std::byte>, std::less<>> payloads;
    payloads.emplace("temperature", std::vector<std::byte>(2 * 3 * sizeof(float)));
    assert(validate_zarr_step_payloads(schema, payloads, error)); // optional omission is valid

    payloads.clear();
    payloads.emplace("optional tendency", std::vector<std::byte>(3 * sizeof(std::int16_t)));
    assert(!validate_zarr_step_payloads(schema, payloads, error)); // required omission

    payloads.clear();
    payloads.emplace("temperature", std::vector<std::byte>(sizeof(float)));
    assert(!validate_zarr_step_payloads(schema, payloads, error)); // byte/shape mismatch

    payloads["temperature"].resize(2 * 3 * sizeof(float));
    payloads.emplace("undeclared", std::vector<std::byte>(sizeof(float)));
    assert(!validate_zarr_step_payloads(schema, payloads, error)); // undeclared field

#ifdef AMIO_HAS_TENSORSTORE
    payloads.clear();
    const std::array<float, 6> expected{1.25f, 2.5f, 3.75f, 5.0f, 6.25f, 7.5f};
    std::vector<std::byte> field_bytes(sizeof(expected));
    std::memcpy(field_bytes.data(), expected.data(), sizeof(expected));
    payloads.emplace("temperature", std::move(field_bytes));
    const auto root = (std::filesystem::temp_directory_path() / "amio-zarr-step-native-test").string();
    std::filesystem::remove_all(root);
    if (!write_zarr_step(root, "dataset-zarr-001", 12, schema, payloads, error)) {
        std::fprintf(stderr, "TensorStore write failed: %s\n", error.c_str());
        return 1;
    }
    std::vector<std::byte> readback;
    assert(read_zarr_step_field(root, "dataset-zarr-001", 12, schema, "temperature", readback, error));
    assert(readback.size() == sizeof(expected));
    assert(std::memcmp(readback.data(), expected.data(), sizeof(expected)) == 0);
    assert(!read_zarr_step_field(root, "dataset-zarr-001", 12, schema, "optional tendency", readback, error));
    assert(!error.empty());
    assert(!write_zarr_step(root, "dataset-zarr-001", 12, schema, payloads, error)); // immutable step collision
    std::filesystem::remove_all(root);
#endif
}
