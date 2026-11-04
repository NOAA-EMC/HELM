#include <charconv>
#include <set>

#include "tool_common.hpp"
namespace amio::tools {
namespace {
bool parse_id(const std::string &s, std::uint64_t &v) {
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size();
}
}  // namespace
bool parse_inclusive_step_range(const std::string &text, std::uint64_t &first, std::uint64_t &last) {
    const auto sep = text.find(':');
    return sep != std::string::npos && parse_id(text.substr(0, sep), first) && parse_id(text.substr(sep + 1), last) && last >= first;
}
bool parse_field_list(const std::string &text, std::vector<std::string> &fields) {
    fields.clear();
    std::set<std::string> unique;
    std::size_t at = 0;
    while (at <= text.size()) {
        auto end = text.find(',', at);
        if (end == std::string::npos) end = text.size();
        auto name = text.substr(at, end - at);
        if (name.empty() || !unique.insert(name).second) {
            fields.clear();
            return false;
        }
        fields.push_back(std::move(name));
        if (end == text.size()) break;
        at = end + 1;
    }
    return !fields.empty();
}
bool destination_is_available(const std::filesystem::path &path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    return !error && !exists;
}
}  // namespace amio::tools
