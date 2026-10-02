// mpi_threading.cpp -- AMIO MPI threading level validation implementation.
//
// Implements the MPI_THREAD_MULTIPLE gate described in design.md §3
// (Worker_Pool / MPI_THREAD_MULTIPLE handling).
//
// Key design decisions:
//
//   * If the manifest requests background MPI-IO (any backend that
//     issues MPI calls from Worker_Pool threads), the host MUST have
//     initialized MPI with MPI_THREAD_MULTIPLE.  If not, amio_init
//     returns AMIO_ERR_THREADING_UNSUPPORTED before creating the
//     Worker_Pool (R3.8).
//
//   * If background MPI-IO is not requested (e.g., file-based I/O
//     without MPI collectives, or single-rank operation), no
//     threading constraint is imposed.
//
//   * The per-(dataset,variable) ordering mutex is dropped before
//     any MPI-IO collective (R3.7).  This is enforced by the
//     WorkerPool callback contract (the callback itself drops the
//     lock before MPI calls), not by this module.
//
// Validates: R3.7, R3.8

#include "workers/mpi_threading.hpp"

#ifdef AMIO_HAS_MPI
#include <halo/environment.hpp>
#endif

namespace amio::detail {

amio_err_t validate_mpi_threading(const MpiThreadingConfig &config) {
    // If no background MPI-IO is requested, any thread level is fine.
    if (!config.requires_background_mpi_io) {
        return AMIO_OK;
    }

    // Background MPI-IO requires MPI_THREAD_MULTIPLE.
    if (config.mpi_thread_level_provided < kMpiThreadMultiple) {
        return AMIO_ERR_THREADING_UNSUPPORTED;
    }

    return AMIO_OK;
}

int query_mpi_thread_level() {
#ifdef AMIO_HAS_MPI
    // Delegate to HALO's Environment singleton which queries
    // MPI_Query_thread during halo::Environment::initialize().
    // Returns -1 if initialize() has not been called yet.
    return halo::Environment::thread_support_level();
#else
    // Standalone build (no MPI linked) -- cannot query thread level.
    // Return -1 to indicate "unknown / not available".
    return -1;
#endif
}

}  // namespace amio::detail
