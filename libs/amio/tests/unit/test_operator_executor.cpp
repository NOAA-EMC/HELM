#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

#include "operators/operator_executor.hpp"

using namespace amio::detail;

int main() {
    Variable variable;
    variable.name = "temperature";
    variable.source_dtype = AMIO_DTYPE_F64;
    variable.stored_dtype = AMIO_DTYPE_F32;
    variable.attributes.emplace("amio.precision", Attribute{std::string("f64_to_f32_absolute")});
    variable.attributes.emplace("amio.tolerance", Attribute{0.001});
    variable.attributes.emplace("amio.authorize_field_loss", Attribute{true});
    variable.attributes.emplace("amio.authorize_lossy_restart", Attribute{true});
    CompiledFieldPipeline pipeline;
    assert(compile_variable_pipeline(variable, pipeline) == AMIO_OK);
    assert(pipeline.request.precision == PrecisionOperator::F64ToF32Absolute);

    const std::array<double, 3> source{1.0, 2.0001, -0.0};
    const std::array<std::uint64_t, 1> offsets{0}, extents{3};
    ExecutedField output;
    assert(execute_field_pipeline(variable, std::span<const std::byte>(reinterpret_cast<const std::byte *>(source.data()), sizeof(source)), offsets,
                                  extents, 1024, output) == AMIO_OK);
    assert(output.bytes.size() == 3 * sizeof(float));
    assert(output.input_bytes == sizeof(source) && output.output_bytes == output.bytes.size());
    assert(output.maximum_absolute_error <= 0.001 && !output.integrity_digest.empty());
    std::array<float, 3> decoded{};
    std::memcpy(decoded.data(), output.bytes.data(), output.bytes.size());
    assert(decoded[0] == 1.0f && decoded[1] == 2.0001f && std::signbit(decoded[2]));

    variable.attributes["amio.authorize_lossy_restart"] = Attribute{false};
    assert(compile_variable_pipeline(variable, pipeline) == AMIO_ERR_INVALID_INPUT);
    variable.attributes["amio.authorize_lossy_restart"] = Attribute{true};
    variable.attributes["amio.codec"] = Attribute{std::string("deflate")};
    assert(compile_variable_pipeline(variable, pipeline, OperatorDestination::NetCDF4) == AMIO_OK);
    assert(pipeline.request.codec == NativeCodec::Deflate);
    variable.attributes["amio.codec"] = Attribute{std::string("zstandard")};
    assert(compile_variable_pipeline(variable, pipeline, OperatorDestination::NetCDF4) == AMIO_ERR_UNSUPPORTED);
    assert(compile_variable_pipeline(variable, pipeline, OperatorDestination::Zarr3) == AMIO_OK);

    Variable grib;
    grib.name = "forecast";
    grib.source_dtype = AMIO_DTYPE_F32;
    grib.stored_dtype = AMIO_DTYPE_F32;
    grib.attributes.emplace("amio.codec", Attribute{std::string("lossless_jpeg2000")});
    assert(compile_variable_pipeline(grib, pipeline, OperatorDestination::GRIB2) == AMIO_OK);
    const std::array<float, 2> exceptional{1.0f, std::numeric_limits<float>::infinity()};
    const std::array<std::uint64_t, 1> grib_offsets{0}, grib_extents{2};
    assert(execute_field_pipeline(grib, std::span<const std::byte>(reinterpret_cast<const std::byte *>(exceptional.data()), sizeof(exceptional)),
                                  grib_offsets, grib_extents, 1024, output, OperatorDestination::GRIB2) == AMIO_ERR_UNSUPPORTED);
}
