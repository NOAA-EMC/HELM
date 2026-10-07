// comm_split.cpp -- AMIO MPI communicator split implementation.
//
// When MPI is available (AMIO_HAS_MPI), delegates to
// halo::Communicator::split() for RAII-managed communicator splitting.
// When MPI is not available, the functions still compile: default configs
// succeed, non-default configs return AMIO_ERR_COMM_SPLIT_FAILED.
//
// Validates: R3.5, R3.6

#include "workers/comm_split.hpp"

#include <algorithm>
#include <set>

#ifdef AMIO_HAS_MPI
MPI_Comm g_amio_parent_comm = MPI_COMM_WORLD;
#else
int g_amio_parent_comm = 0;
#endif

extern "C" {
AMIO_API void amio_set_parent_communicator(int comm_f) {
#ifdef AMIO_HAS_MPI
    g_amio_parent_comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_f));
#else
    g_amio_parent_comm = comm_f;
#endif
}
}

// MPI availability detection.
// In a full build, AMIO_HAS_MPI would be set by CMake when MPI is found.
// For standalone compilation without MPI, the functions degrade gracefully.
#ifdef AMIO_HAS_MPI
#define AMIO_CAN_SPLIT 1
#else
#define AMIO_CAN_SPLIT 0
#endif

namespace amio::detail {

amio_err_t validate_comm_config(const CommConfig &config) {
    if (config.is_default()) {
        return AMIO_OK;
    }

    // Must have a known world size to validate ranks.
    if (config.world_size <= 0) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }

    // io_ranks must not be empty (already checked by is_default).
    // io_ranks must be a proper subset: not all ranks can be I/O.
    if (static_cast<int>(config.io_ranks.size()) >= config.world_size) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }

    // Check for duplicates and out-of-range values.
    std::set<int> seen;
    for (int rank : config.io_ranks) {
        if (rank < 0 || rank >= config.world_size) {
            return AMIO_ERR_COMM_SPLIT_FAILED;
        }
        if (!seen.insert(rank).second) {
            // Duplicate rank.
            return AMIO_ERR_COMM_SPLIT_FAILED;
        }
    }

    return AMIO_OK;
}

amio_err_t split_communicator(const CommConfig &config, int my_rank, IOCommunicator &result) {
    // Default: no split requested.  All ranks participate in I/O.
    if (config.is_default()) {
        result.valid = true;
        result.is_io_rank = true;
        // io_comm remains nullopt -- accessors return MPI_COMM_WORLD / 0 / 1.
        return AMIO_OK;
    }

    // Validate the configuration.
    amio_err_t validation = validate_comm_config(config);
    if (validation != AMIO_OK) {
        return validation;
    }

    // Validate my_rank is within world_size.
    if (my_rank < 0 || my_rank >= config.world_size) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }

    // Determine if this rank is in the I/O set.
    bool is_io = std::find(config.io_ranks.begin(), config.io_ranks.end(), my_rank) != config.io_ranks.end();

#if AMIO_CAN_SPLIT
    // Use HALO's RAII Communicator for the split.
    // Color: 0 = I/O ranks, 1 = compute ranks.
    int color = is_io ? 0 : 1;
    try {
        // Wrap parent communicator safely (duplicating to prevent RAII destruction of externally owned custom communicator)
        MPI_Comm comm_to_wrap = g_amio_parent_comm;
        if (g_amio_parent_comm != MPI_COMM_NULL && g_amio_parent_comm != MPI_COMM_WORLD && g_amio_parent_comm != MPI_COMM_SELF) {
            MPI_Comm_dup(g_amio_parent_comm, &comm_to_wrap);
        }
        halo::Communicator world(comm_to_wrap);
        // split() returns a new RAII-owned Communicator.
        halo::Communicator split_comm = world.split(color, my_rank);

        result.valid = true;
        result.is_io_rank = is_io;
        result.io_comm = std::move(split_comm);
        return AMIO_OK;
    } catch (const std::runtime_error &) {
        // HALO split failed -- RAII guarantees no handle leak.
        return AMIO_ERR_COMM_SPLIT_FAILED;
    } catch (...) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }

#else  // !AMIO_CAN_SPLIT
    // MPI is not available.  Non-default configs cannot be split.
    // Return error, leaving world communicator unmodified (R3.6).
    (void)is_io;
    return AMIO_ERR_COMM_SPLIT_FAILED;
#endif
}

}  // namespace amio::detail
