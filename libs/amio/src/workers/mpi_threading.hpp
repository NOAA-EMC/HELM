// mpi_threading.hpp -- AMIO MPI threading level validation.
//
// This header is PRIVATE to the AMIO_Core build (`src/workers/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// Provides a function to validate that the host application has
// initialized MPI with MPI_THREAD_MULTIPLE when the manifest
// requests background MPI-IO.  If the host has not initialized at
// MPI_THREAD_MULTIPLE and background MPI-IO is requested, the
// function returns AMIO_ERR_THREADING_UNSUPPORTED so that
// amio_init can reject the configuration before creating the
// Worker_Pool (R3.8).
//
// The per-(dataset,variable) ordering mutex is dropped before any
// MPI-IO collective (R3.7) -- this is enforced by the WorkerPool
// callback contract, not by this module.  This module only gates
// initialization.
//
// Thread safety
// -------------
// validate_mpi_threading() should be called once during
// initialization (before Worker_Pool threads are spawned).
//
// Validates: R3.7, R3.8

#ifndef AMIO_SRC_WORKERS_MPI_THREADING_HPP
#define AMIO_SRC_WORKERS_MPI_THREADING_HPP

#include "amio/amio_errors.h"

namespace amio::detail {

// MpiThreadingConfig -- configuration for MPI threading validation.
//
// Fields:
//   requires_background_mpi_io - true if the manifest requests
//       background MPI-IO (e.g., NetCDF_Driver with parallel
//       writes, or any backend that issues MPI calls from worker
//       threads).
//   mpi_thread_level_provided - the MPI thread level that was
//       provided by MPI_Init_thread (or equivalent).  Values
//       follow the MPI standard:
//         0 = MPI_THREAD_SINGLE
//         1 = MPI_THREAD_FUNNELED
//         2 = MPI_THREAD_SERIALIZED
//         3 = MPI_THREAD_MULTIPLE
//       A value of -1 means "MPI not initialized" or "unknown".
struct MpiThreadingConfig {
    bool requires_background_mpi_io = false;
    int mpi_thread_level_provided = -1;
};

// Thread level constants matching MPI standard values.
static constexpr int kMpiThreadSingle = 0;
static constexpr int kMpiThreadFunneled = 1;
static constexpr int kMpiThreadSerialized = 2;
static constexpr int kMpiThreadMultiple = 3;

// validate_mpi_threading -- check that the MPI threading level is
// sufficient for the requested configuration.
//
// Behavior:
//   * If config.requires_background_mpi_io is false, the function
//     returns AMIO_OK regardless of the thread level (no background
//     MPI-IO means no threading constraint).
//
//   * If config.requires_background_mpi_io is true AND
//     config.mpi_thread_level_provided < MPI_THREAD_MULTIPLE (3),
//     returns AMIO_ERR_THREADING_UNSUPPORTED.
//
//   * If config.requires_background_mpi_io is true AND
//     config.mpi_thread_level_provided >= MPI_THREAD_MULTIPLE (3),
//     returns AMIO_OK.
//
// Parameters:
//   config - the MPI threading configuration to validate
//
// Returns:
//   AMIO_OK if the threading level is sufficient.
//   AMIO_ERR_THREADING_UNSUPPORTED if background MPI-IO is
//   requested but the host did not initialize at
//   MPI_THREAD_MULTIPLE.
amio_err_t validate_mpi_threading(const MpiThreadingConfig &config);

// query_mpi_thread_level -- query the MPI thread level that was
// provided by MPI_Init_thread.
//
// When MPI is available, delegates to
// halo::Environment::thread_support_level() which internally
// queries the MPI thread level detected during HALO initialization.
// When MPI is not available (standalone build), returns -1.
//
// This is a convenience function for use during amio_init to
// populate MpiThreadingConfig::mpi_thread_level_provided.
int query_mpi_thread_level();

}  // namespace amio::detail

#endif  // AMIO_SRC_WORKERS_MPI_THREADING_HPP
