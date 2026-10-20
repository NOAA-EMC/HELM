#ifndef AMIO_NETCDF_MUTEX_HPP
#define AMIO_NETCDF_MUTEX_HPP
#include <mutex>
namespace amio::detail {
/** @brief Process-wide guard shared by NetCDF and NCZarr operations.
 * @return Mutex protecting the non-thread-safe netCDF-C runtime.
 * @note Serialize library calls even when separate dataset handles are used.
 * MPI ranks hold independent guards and can still perform parallel MPI-IO.
 */
inline std::mutex &netcdf_mutex() {
    static std::mutex mutex;
    return mutex;
}
}  // namespace amio::detail
#endif
