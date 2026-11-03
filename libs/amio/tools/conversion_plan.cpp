#include "tool_common.hpp"

namespace amio::tools {
CliResult run_conversion_plan(const Options &options) {
    // Keep the preflight entry point behavior identical to the copy command;
    // a second, stale preflight path could otherwise approve a request that
    // the bounded writer must reject (or report obsolete unsupported claims).
    return run_copy(options);
}
}  // namespace amio::tools
