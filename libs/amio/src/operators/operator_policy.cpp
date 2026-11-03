#include "operators/operator_policy.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace amio::detail {

bool native_codec_supported(OperatorDestination destination, NativeCodec codec) noexcept {
    switch (destination) {
        case OperatorDestination::NetCDF4: return codec == NativeCodec::Deflate;
        case OperatorDestination::Zarr3: return codec == NativeCodec::Zstd;
        case OperatorDestination::GRIB2: return codec == NativeCodec::Grib2Aec || codec == NativeCodec::Grib2Jpeg2000;
        case OperatorDestination::Unknown: return false;
    }
    return false;
}

amio_status_t compile_field_pipeline(const FieldOperatorRequest &request, CompiledFieldPipeline &out,
                                     std::string *reason) {
    auto fail = [&](amio_status_t status, const char *message) {
        if (reason) *reason = message;
        return status;
    };
    if (request.precision != PrecisionOperator::None && request.precision != PrecisionOperator::F64ToF32Absolute &&
        request.precision != PrecisionOperator::F64ToF32Relative) return fail(AMIO_ERR_UNSUPPORTED, "unsupported precision operator");
    if (request.integrity != IntegrityOperator::None && request.integrity != IntegrityOperator::Sha256LogicalV1)
        return fail(AMIO_ERR_UNSUPPORTED, "unsupported integrity operator");
    if (request.codec != NativeCodec::None && !native_codec_supported(request.destination, request.codec))
        return fail(AMIO_ERR_UNSUPPORTED, "codec is not supported by the selected native backend");
    if (!std::isfinite(request.tolerance) || request.tolerance < 0.0) return fail(AMIO_ERR_INVALID_INPUT, "tolerance must be finite and nonnegative");
    if (request.precision != PrecisionOperator::None && (!request.authorize_field_loss || !request.authorize_lossy_restart))
        return fail(AMIO_ERR_INVALID_INPUT, "precision loss requires field accuracy and restart authorization");
    if (request.precision == PrecisionOperator::None && request.tolerance != 0.0)
        return fail(AMIO_ERR_INVALID_INPUT, "tolerance has no meaning without a precision operator");

    CompiledFieldPipeline compiled;
    compiled.request = request;
    if (request.precision != PrecisionOperator::None) compiled.order.emplace_back("precision");
    if (request.integrity != IntegrityOperator::None) compiled.order.emplace_back("integrity");
    if (request.codec != NativeCodec::None) compiled.order.emplace_back("native-codec");
    std::ostringstream identity;
    identity << "pipeline-v1;precision=" << static_cast<unsigned>(request.precision)
             << ";integrity=" << static_cast<unsigned>(request.integrity)
             << ";codec=" << static_cast<unsigned>(request.codec)
             << ";destination=" << static_cast<unsigned>(request.destination)
             << ";tolerance=" << std::setprecision(17) << request.tolerance
             << ";field-loss=" << request.authorize_field_loss
             << ";restart-loss=" << request.authorize_lossy_restart;
    compiled.identity = identity.str();
    out = std::move(compiled);
    if (reason) reason->clear();
    return AMIO_OK;
}

} // namespace amio::detail
