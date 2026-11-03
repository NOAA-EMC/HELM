#include "tool_common.hpp"

#include <charconv>
#include <algorithm>

namespace amio::tools {
CliResult run_validate(const Options &options) {
    CliResult result{0, "validate", "structural collection validation"};
    if (options.positional.size() != 1) {
        result.exit_code = 2; result.errors.push_back("usage: amio validate URI [--integrity] [--json]"); return result;
    }
    CollectionInfo collection(options.positional.front(), "");
    auto status = open_collection(options.positional.front(), collection);
    if (status != AMIO_OK) {
        result.exit_code = status == AMIO_ERR_INVALID_INPUT ? 2 : 4;
        result.errors.push_back("collection identity is missing or malformed"); return result;
    }
    CliRuntime runtime;
    status = open_cli_dataset(options.positional.front(), options, runtime);
    if (status != AMIO_OK) { result.exit_code = status == AMIO_ERR_INVALID_INPUT ? 2 : 5; result.errors.push_back("could not open read-only dataset handle: " + std::to_string(status)); return result; }
    const bool integrity_requested = options.values.contains("--integrity") ||
        std::find(options.flags.begin(), options.flags.end(), "--integrity") != options.flags.end();
    std::uint64_t mask = (UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1)) |
                         (UINT64_C(1) << (AMIO_VALIDATE_REQUIRED_FIELDS - 1)) |
                         (UINT64_C(1) << (AMIO_VALIDATE_PAYLOAD_PRESENCE - 1)) |
                         (UINT64_C(1) << (AMIO_VALIDATE_ORPHAN_OBJECTS - 1));
    if (integrity_requested) mask |= UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1);
    std::uint64_t cursor = 0;
    std::vector<std::string> report;
    bool unsupported_completeness = false;
    do {
        amio_validation_request_t request{};
        request.struct_size = sizeof(request); request.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        request.check_mask = mask; request.cursor = cursor; request.max_results = 2;
        amio_validation_result_t page[2]{}; std::size_t count = 0; std::uint64_t next = 0;
        status = amio_validate_dataset(runtime.dataset, &request, page, 2, &count, &next);
        if (status != AMIO_OK) { result.exit_code = status == AMIO_ERR_INVALID_INPUT ? 2 : 5; result.errors.push_back("validation API failed with status " + std::to_string(status)); return result; }
        for (std::size_t i=0;i<count;++i) {
            const auto &entry=page[i];
            report.push_back("{\"check\":"+std::to_string(entry.check)+",\"status\":"+std::to_string(entry.status)+",\"detail_code\":"+std::to_string(entry.detail_code)+"}");
            if (entry.status == AMIO_VALIDATION_FAIL) { result.exit_code=4; result.errors.push_back("validation check "+std::to_string(entry.check)+" failed with detail "+std::to_string(entry.detail_code)); }
            else if (entry.status == AMIO_VALIDATION_UNSUPPORTED && entry.check != AMIO_VALIDATE_ORPHAN_OBJECTS &&
                     !(entry.check == AMIO_VALIDATE_INTEGRITY && !integrity_requested)) {
                result.exit_code=3; result.errors.push_back("validation check "+std::to_string(entry.check)+" is unsupported");
            }
            if (entry.status == AMIO_VALIDATION_UNSUPPORTED && (entry.check == AMIO_VALIDATE_ORPHAN_OBJECTS || entry.check == AMIO_VALIDATE_INTEGRITY))
                result.warnings.push_back("validation check "+std::to_string(entry.check)+" is unsupported");
            if (entry.status == AMIO_VALIDATION_UNSUPPORTED) unsupported_completeness = true;
        }
        cursor=next;
    } while(cursor!=0);
    result.summary = "structural validation report for collection " + collection.uuid;
    result.summary_json = "{\"dataset_uuid\":" + ResultWriter::quote(collection.uuid) + ",\"complete\":"+
        (result.exit_code==0 && !unsupported_completeness ? "true":"false")+",\"checks\":[";
    for(std::size_t i=0;i<report.size();++i){if(i)result.summary_json+=',';result.summary_json+=report[i];}
    result.summary_json += "]}";
    return result;
}
} // namespace amio::tools
