#ifndef AMIO_SRC_OPERATORS_OPERATOR_EXECUTOR_HPP
#define AMIO_SRC_OPERATORS_OPERATOR_EXECUTOR_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "operators/operator_policy.hpp"
#include "schema/schema.hpp"

namespace amio::detail {

struct ExecutedField {
    std::vector<std::byte> bytes;
    std::string integrity_digest;
    double maximum_absolute_error = 0.0;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t scratch_bytes = 0;
    std::string pipeline_identity;
};

[[nodiscard]] amio_status_t compile_variable_pipeline(const Variable &variable, CompiledFieldPipeline &pipeline,
                                                      OperatorDestination destination = OperatorDestination::NetCDF4);

[[nodiscard]] amio_status_t execute_field_pipeline(const Variable &variable, std::span<const std::byte> source,
                                                   std::span<const std::uint64_t> offsets, std::span<const std::uint64_t> extents,
                                                   std::size_t scratch_limit, ExecutedField &result,
                                                   OperatorDestination destination = OperatorDestination::NetCDF4);

}  // namespace amio::detail

#endif
