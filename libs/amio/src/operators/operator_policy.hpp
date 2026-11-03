#ifndef AMIO_SRC_OPERATORS_OPERATOR_POLICY_HPP
#define AMIO_SRC_OPERATORS_OPERATOR_POLICY_HPP

#include <string>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

enum class PrecisionOperator { None, F64ToF32Absolute, F64ToF32Relative };
enum class IntegrityOperator { None, Sha256LogicalV1 };
enum class OperatorDestination { Unknown, NetCDF4, Zarr3, GRIB2 };
enum class NativeCodec { None, Deflate, Zstd, Grib2Aec, Grib2Jpeg2000 };

struct FieldOperatorRequest {
    PrecisionOperator precision = PrecisionOperator::None;
    IntegrityOperator integrity = IntegrityOperator::Sha256LogicalV1;
    NativeCodec codec = NativeCodec::None;
    OperatorDestination destination = OperatorDestination::Unknown;
    double tolerance = 0.0;
    bool authorize_field_loss = false;
    bool authorize_lossy_restart = false;
};

struct CompiledFieldPipeline {
    FieldOperatorRequest request;
    std::vector<std::string> order;
    std::string identity;
};

[[nodiscard]] amio_status_t compile_field_pipeline(const FieldOperatorRequest &request,
                                                    CompiledFieldPipeline &out,
                                                    std::string *reason = nullptr);
[[nodiscard]] bool native_codec_supported(OperatorDestination destination, NativeCodec codec) noexcept;

} // namespace amio::detail

#endif
