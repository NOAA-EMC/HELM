#ifndef AMIO_TESTS_QUALIFICATION_NATIVE_READERS_HPP
#define AMIO_TESTS_QUALIFICATION_NATIVE_READERS_HPP

#include <cstdint>
#include <filesystem>
#include <string>

namespace amio::qualification {

[[nodiscard]] std::string step_payload_path(const std::filesystem::path &root, std::uint64_t step);
[[nodiscard]] std::string field_name(std::uint64_t field);
void verify_native_netcdf_field(const std::filesystem::path &root, std::uint64_t step, std::uint64_t cells,
                                std::uint64_t field, double expected_model_time,
                                const std::string &expected_schema_digest);

} // namespace amio::qualification

#endif
