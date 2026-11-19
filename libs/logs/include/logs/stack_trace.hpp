#ifndef LOGS_STACK_TRACE_HPP
#define LOGS_STACK_TRACE_HPP

/// @file stack_trace.hpp
/// @brief Stack_Frame data type and format_stack_trace() pure formatter.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace logs {

/// Caller-captured frame data. LOGS never captures or symbolizes frames itself;
/// it only formats supplied data.
struct Stack_Frame {
    std::optional<std::string> function;
    std::optional<std::filesystem::path> file;
    std::optional<int> line;
};

/// Pure formatting transformation over a supplied frame sequence.
[[nodiscard]] std::string format_stack_trace(std::span<const Stack_Frame> frames);

/// Cap on rendered frames before truncation.
inline constexpr std::size_t MAX_TRACE_FRAMES = 256;

}  // namespace logs

#endif  // LOGS_STACK_TRACE_HPP
