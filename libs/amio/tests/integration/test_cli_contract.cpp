#include <cassert>
#include <sstream>

#include "result_writer.hpp"
int main() {
    for (int status = 0; status <= 6; ++status) {
        amio::tools::CliResult r;
        r.exit_code = status;
        r.command = "validate";
        r.summary = "unsupported";
        r.summary_json = "{\"check\":\"integrity\"}";
        r.results_json = {"{\"step_id\":7}"};
        r.warnings.push_back("bounded warning");
        r.errors.push_back("validation status");
        std::ostringstream out, err;
        const int code = amio::tools::ResultWriter(true, out, err).write(r);
        assert(code == status);
        const auto json = out.str();
        for (const char *key : {"schema_version", "command", "status", "summary", "results", "warnings", "errors"})
            assert(json.find(key) != std::string::npos);
        assert(json.find("\"status\":" + std::to_string(status)) != std::string::npos);
        assert(json.find("validation status") != std::string::npos);
    }
    const auto escaped = amio::tools::ResultWriter::quote("a\n\"b");
    assert(escaped == "\"a\\n\\\"b\"");
    amio::tools::CliResult bounded{4, "validate", "bounded errors"};
    for (int i = 0; i < 100; ++i) bounded.errors.push_back("sample-" + std::to_string(i));
    std::ostringstream bounded_out, bounded_err;
    amio::tools::ResultWriter(true, bounded_out, bounded_err).write(bounded);
    assert(bounded_out.str().find("sample-31") != std::string::npos);
    assert(bounded_out.str().find("sample-32") == std::string::npos);
    assert(bounded_out.str().find("additional errors omitted") != std::string::npos);
    return 0;
}
