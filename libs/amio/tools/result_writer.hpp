#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace amio::tools {

struct CliResult {
    int exit_code = 0;
    std::string command;
    std::string summary;
    std::string summary_json = "{}";
    std::vector<std::string> results_json;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

class ResultWriter {
public:
    explicit ResultWriter(bool json, std::ostream &out, std::ostream &err);
    int write(const CliResult &result) const;
    [[nodiscard]] static std::string quote(const std::string &value);
private:
    bool json_;
    std::ostream &out_;
    std::ostream &err_;
};

} // namespace amio::tools
