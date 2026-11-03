#include "operators/operator_executor.hpp"

#include "operators/payload_integrity.hpp"
#include "operators/precision_conversion.hpp"

#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

namespace amio::detail {
namespace {

const AttributeValue *attribute(const Variable &variable, std::string_view name) {
    const auto found = variable.attributes.find(name);
    return found == variable.attributes.end() ? nullptr : &found->second.value;
}

bool get_bool(const Variable &variable, std::string_view name, bool &value) {
    const auto *v = attribute(variable, name);
    if (!v) { value = false; return true; }
    if (const auto *b = std::get_if<bool>(v)) { value = *b; return true; }
    return false;
}

std::string_view dtype_name(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_I8: return "i8"; case AMIO_DTYPE_U8: return "u8";
        case AMIO_DTYPE_I16: return "i16"; case AMIO_DTYPE_U16: return "u16";
        case AMIO_DTYPE_I32: return "i32"; case AMIO_DTYPE_U32: return "u32";
        case AMIO_DTYPE_I64: return "i64"; case AMIO_DTYPE_U64: return "u64";
        case AMIO_DTYPE_F32: return "f32"; case AMIO_DTYPE_F64: return "f64";
        default: return {};
    }
}

} // namespace

amio_status_t compile_variable_pipeline(const Variable &variable, CompiledFieldPipeline &pipeline,
                                       OperatorDestination destination) {
    FieldOperatorRequest request;
    request.destination = destination;
    request.integrity = IntegrityOperator::Sha256LogicalV1;
    if (const auto *codec_value = attribute(variable, "amio.codec")) {
        const auto *codec = std::get_if<std::string>(codec_value);
        if (codec == nullptr) return AMIO_ERR_INVALID_INPUT;
        if (*codec == "none") request.codec = NativeCodec::None;
        else if (*codec == "deflate") request.codec = NativeCodec::Deflate;
        else if (*codec == "zstandard" || *codec == "zstd") request.codec = NativeCodec::Zstd;
        else if (*codec == "grib2_aec" || *codec == "libaec") request.codec = NativeCodec::Grib2Aec;
        else if (*codec == "lossless_jpeg2000" || *codec == "grib2_lossless") request.codec = NativeCodec::Grib2Jpeg2000;
        else return AMIO_ERR_UNSUPPORTED;
    }
    if (variable.source_dtype != variable.stored_dtype) {
        if (variable.source_dtype != AMIO_DTYPE_F64 || variable.stored_dtype != AMIO_DTYPE_F32)
            return AMIO_ERR_UNSUPPORTED;
        const auto *precision_value = attribute(variable, "amio.precision");
        const auto *precision = precision_value ? std::get_if<std::string>(precision_value) : nullptr;
        const auto *tolerance_value = attribute(variable, "amio.tolerance");
        const auto *tolerance = tolerance_value ? std::get_if<double>(tolerance_value) : nullptr;
        bool field_loss = false, restart_loss = false;
        if (!precision || !tolerance || !get_bool(variable, "amio.authorize_field_loss", field_loss) ||
            !get_bool(variable, "amio.authorize_lossy_restart", restart_loss)) return AMIO_ERR_INVALID_POLICY;
        if (*precision == "f64_to_f32_absolute") request.precision = PrecisionOperator::F64ToF32Absolute;
        else if (*precision == "f64_to_f32_relative") request.precision = PrecisionOperator::F64ToF32Relative;
        else return AMIO_ERR_UNSUPPORTED;
        request.tolerance = *tolerance;
        request.authorize_field_loss = field_loss;
        request.authorize_lossy_restart = restart_loss;
    }
    return compile_field_pipeline(request, pipeline);
}

amio_status_t execute_field_pipeline(const Variable &variable, std::span<const std::byte> source,
                                     std::span<const std::uint64_t> offsets,
                                     std::span<const std::uint64_t> extents,
                                     std::size_t scratch_limit, ExecutedField &result,
                                     OperatorDestination destination) {
    CompiledFieldPipeline compiled;
    const auto policy_status = compile_variable_pipeline(variable, compiled, destination);
    if (policy_status != AMIO_OK) return policy_status;
    const auto request = compiled.request;
    ExecutedField executed;
    executed.input_bytes = source.size();
    executed.pipeline_identity = compiled.identity;
    if (request.precision == PrecisionOperator::None) {
        executed.output_bytes = source.size();
        if ((request.codec == NativeCodec::Grib2Aec || request.codec == NativeCodec::Grib2Jpeg2000) &&
            variable.stored_dtype == AMIO_DTYPE_F32) {
            if (source.size() % sizeof(float) != 0) return AMIO_ERR_INVALID_INPUT;
            for (std::size_t i=0; i<source.size()/sizeof(float); ++i) {
                float value = 0.0f;
                std::memcpy(&value, source.data()+i*sizeof(float), sizeof(float));
                if (!std::isfinite(value)) return AMIO_ERR_UNSUPPORTED;
            }
        }
        const auto digest_status = sha256_logical_payload(dtype_name(variable.stored_dtype), offsets, extents,
                                                          source.data(), source.size(), executed.integrity_digest);
        if (digest_status != AMIO_OK) return digest_status;
        result = std::move(executed);
        return AMIO_OK;
    } else {
        if (source.size() % sizeof(double) != 0 || source.size() > std::numeric_limits<std::size_t>::max() / 2 ||
            source.size() * 2 > scratch_limit) return AMIO_ERR_BUDGET_EXCEEDED;
        executed.scratch_bytes = source.size() * 2;
        std::vector<double> values(source.size() / sizeof(double));
        if (!source.empty()) std::memcpy(values.data(), source.data(), source.size());
        PrecisionConversionPolicy conversion;
        if (request.precision == PrecisionOperator::F64ToF32Absolute) conversion.absolute_tolerance = request.tolerance;
        else conversion.relative_tolerance = request.tolerance;
        if (const auto *sentinel_value = attribute(variable, "_FillValue"))
            if (const auto *sentinel = std::get_if<double>(sentinel_value)) conversion.missing_sentinel = *sentinel;
        bool canonical_nan = false;
        if (!get_bool(variable, "amio.canonical_nan", canonical_nan)) return AMIO_ERR_INVALID_INPUT;
        conversion.canonicalize_nan = canonical_nan;
        PrecisionConversionResult converted;
        const auto conversion_status = convert_f64_to_f32(values, conversion, converted);
        if (conversion_status != AMIO_OK) return conversion_status;
        executed.maximum_absolute_error = converted.maximum_absolute_error;
        executed.bytes.resize(converted.values.size() * sizeof(float));
        if (!executed.bytes.empty()) std::memcpy(executed.bytes.data(), converted.values.data(), executed.bytes.size());
    }
    if ((request.codec == NativeCodec::Grib2Aec || request.codec == NativeCodec::Grib2Jpeg2000) &&
        variable.stored_dtype == AMIO_DTYPE_F32) {
        const auto &logical = executed.bytes.empty() ? source : std::span<const std::byte>(executed.bytes);
        if (logical.size() % sizeof(float) != 0) return AMIO_ERR_INVALID_INPUT;
        for (std::size_t i=0; i<logical.size()/sizeof(float); ++i) {
            float value = 0.0f;
            std::memcpy(&value, logical.data()+i*sizeof(float), sizeof(float));
            if (!std::isfinite(value)) return AMIO_ERR_UNSUPPORTED;
        }
    }
    executed.output_bytes = executed.bytes.size();
    const auto digest_status = sha256_logical_payload(dtype_name(variable.stored_dtype), offsets, extents,
                                                      executed.bytes.data(), executed.bytes.size(), executed.integrity_digest);
    if (digest_status != AMIO_OK) return digest_status;
    result = std::move(executed);
    return AMIO_OK;
}

} // namespace amio::detail
