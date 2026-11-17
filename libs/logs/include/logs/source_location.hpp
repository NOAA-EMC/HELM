#ifndef LOGS_SOURCE_LOCATION_HPP
#define LOGS_SOURCE_LOCATION_HPP

/// @file source_location.hpp
/// @brief Source_Location optional annotation type.

#include <filesystem>
#include <string>

namespace logs {

/// Caller-supplied source-location annotation. When absent, the Log_Record
/// stores std::nullopt rather than fabricating field values.
struct Source_Location {
    std::filesystem::path file;  ///< source file name, verbatim
    int line{0};                 ///< line number, guaranteed >= 1 when present
    std::string function;        ///< function name, verbatim
};

}  // namespace logs

#endif  // LOGS_SOURCE_LOCATION_HPP
