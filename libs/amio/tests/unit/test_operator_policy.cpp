#include "operators/operator_policy.hpp"

#include <cassert>

using namespace amio::detail;

int main() {
    CompiledFieldPipeline pipeline;
    FieldOperatorRequest request;
    assert(compile_field_pipeline(request, pipeline) == AMIO_OK);
    assert((pipeline.order == std::vector<std::string>{"integrity"}));
    assert(!pipeline.identity.empty());

    request.precision = PrecisionOperator::F64ToF32Absolute;
    request.tolerance = 0.01;
    assert(compile_field_pipeline(request, pipeline) == AMIO_ERR_INVALID_INPUT);
    request.authorize_field_loss = true;
    assert(compile_field_pipeline(request, pipeline) == AMIO_ERR_INVALID_INPUT);
    request.authorize_lossy_restart = true;
    assert(compile_field_pipeline(request, pipeline) == AMIO_OK);
    assert((pipeline.order == std::vector<std::string>{"precision", "integrity"}));

    request.codec = NativeCodec::Deflate;
    assert(compile_field_pipeline(request, pipeline) == AMIO_ERR_UNSUPPORTED);
    request.destination = OperatorDestination::NetCDF4;
    assert(compile_field_pipeline(request, pipeline) == AMIO_OK);
    assert(pipeline.order.back() == "native-codec");
    request.codec = NativeCodec::Zstd;
    assert(compile_field_pipeline(request, pipeline) == AMIO_ERR_UNSUPPORTED);
    request.destination = OperatorDestination::Zarr3;
    assert(compile_field_pipeline(request, pipeline) == AMIO_OK);
    request.destination = OperatorDestination::GRIB2;
    request.codec = NativeCodec::Grib2Jpeg2000;
    assert(compile_field_pipeline(request, pipeline) == AMIO_OK);
    request.codec = static_cast<NativeCodec>(999);
    assert(compile_field_pipeline(request, pipeline) == AMIO_ERR_UNSUPPORTED);
    request.codec = NativeCodec::None;
    request.tolerance = -1.0;
    assert(compile_field_pipeline(request, pipeline) == AMIO_ERR_INVALID_INPUT);
}
