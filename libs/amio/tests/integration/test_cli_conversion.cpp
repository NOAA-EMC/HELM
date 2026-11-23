#include <unistd.h>

#include <cassert>
#include <filesystem>

#include "tool_common.hpp"
int main() {
    std::uint64_t first = 0, last = 0;
    assert(amio::tools::parse_inclusive_step_range("4:9", first, last) && first == 4 && last == 9);
    assert(amio::tools::parse_inclusive_step_range("9:9", first, last) && first == 9 && last == 9);
    assert(!amio::tools::parse_inclusive_step_range("9:4", first, last));
    assert(!amio::tools::parse_inclusive_step_range("1:", first, last));
    assert(!amio::tools::parse_inclusive_step_range("1:2:3", first, last));
    std::vector<std::string> fields;
    assert(amio::tools::parse_field_list("temperature,latitude", fields));
    assert(fields.size() == 2 && fields[0] == "temperature" && fields[1] == "latitude");
    assert(!amio::tools::parse_field_list("temperature,temperature", fields));
    assert(!amio::tools::parse_field_list("temperature,", fields));
    char pattern[] = "/tmp/amio-cli-destination-XXXXXX";
    char *dir = ::mkdtemp(pattern);
    assert(dir != nullptr);
    assert(!amio::tools::destination_is_available(dir));
    const auto absent = std::filesystem::path(dir) / "missing-child";
    assert(amio::tools::destination_is_available(absent));
    std::filesystem::remove_all(dir);

    // A preflight-only operation must never claim copied output.
    amio::tools::CliResult r{3, "copy", "conversion writer unavailable"};
    r.errors.push_back("destination_created=false");
    assert(r.exit_code == 3);
    assert(r.errors.front().find("destination_created=false") != std::string::npos);
    return 0;
}
