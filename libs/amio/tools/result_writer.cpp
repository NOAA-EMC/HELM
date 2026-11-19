#include "result_writer.hpp"

#include <algorithm>
#include <ostream>

namespace amio::tools {
namespace {
constexpr std::size_t kMaxMessages = 32;
constexpr std::size_t kMaxMessageBytes = 2048;
std::string bounded_message(const std::string &message) {
    if (message.size() <= kMaxMessageBytes) return message;
    return message.substr(0, kMaxMessageBytes) + " [truncated]";
}
}  // namespace

ResultWriter::ResultWriter(bool json, std::ostream &out, std::ostream &err) : json_(json), out_(out), err_(err) {}

std::string ResultWriter::quote(const std::string &value) {
    std::string out = "\"";
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    static constexpr char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[ch >> 4];
                    out += hex[ch & 0xf];
                } else
                    out += static_cast<char>(ch);
        }
    }
    out += '"';
    return out;
}

int ResultWriter::write(const CliResult &result) const {
    if (json_) {
        out_ << "{\"schema_version\":1,\"command\":" << quote(result.command) << ",\"status\":" << result.exit_code
             << ",\"summary\":" << result.summary_json << ",\"results\":[";
        for (std::size_t i = 0; i < result.results_json.size(); ++i) {
            if (i) out_ << ',';
            out_ << result.results_json[i];
        }
        out_ << "],\"warnings\":[";
        const auto warning_count = std::min(result.warnings.size(), kMaxMessages);
        for (std::size_t i = 0; i < warning_count; ++i) {
            if (i) out_ << ',';
            out_ << quote(bounded_message(result.warnings[i]));
        }
        out_ << "],\"errors\":[";
        const auto error_count = std::min(result.errors.size(), kMaxMessages);
        for (std::size_t i = 0; i < error_count; ++i) {
            if (i) out_ << ',';
            out_ << quote(bounded_message(result.errors[i]));
        }
        if (result.errors.size() > error_count) {
            if (error_count) out_ << ',';
            out_ << quote("additional errors omitted");
        }
        out_ << "]}\n";
        return result.exit_code;
    }
    out_ << result.command << ": " << result.summary << '\n';
    for (const auto &line : result.results_json) out_ << "  " << line << '\n';
    for (std::size_t i = 0; i < std::min(result.warnings.size(), kMaxMessages); ++i)
        err_ << "warning: " << bounded_message(result.warnings[i]) << '\n';
    for (std::size_t i = 0; i < std::min(result.errors.size(), kMaxMessages); ++i) err_ << "error: " << bounded_message(result.errors[i]) << '\n';
    if (result.errors.size() > kMaxMessages) err_ << "error: additional errors omitted\n";
    return result.exit_code;
}

}  // namespace amio::tools
