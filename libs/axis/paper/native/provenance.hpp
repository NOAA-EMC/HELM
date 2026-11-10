#pragma once
#include <netcdf.h>
#include <sys/utsname.h>

#include <thread>

#include "build_info.hpp"
#include "result_writer.hpp"
namespace paper {
inline std::string build_provenance() {
    utsname runtime{};
    const bool platform_known = uname(&runtime) == 0;
    return "{\"source_commit\":" + quote(PAPER_COMMIT) + ",\"compiler\":" + quote(PAPER_COMPILER) +
           ",\"flags\":" + quote(PAPER_FLAGS) + ",\"netcdf_version\":" + quote(nc_inq_libvers()) +
           ",\"runtime_os\":" + (platform_known ? quote(runtime.sysname) : "null") +
           ",\"runtime_release\":" + (platform_known ? quote(runtime.release) : "null") +
           ",\"runtime_machine\":" + (platform_known ? quote(runtime.machine) : "null") +
           ",\"logical_cpus\":" + std::to_string(std::thread::hardware_concurrency()) +
           ",\"precision\":\"float64\",\"ranks\":1}";
}
}  // namespace paper
