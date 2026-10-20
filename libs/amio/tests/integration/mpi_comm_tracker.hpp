// Test-only MPI profiling interposition: observe duplicate ownership without
// waiting for communicator-context exhaustion after thousands of outputs.
#ifndef AMIO_TEST_MPI_COMM_TRACKER_HPP
#define AMIO_TEST_MPI_COMM_TRACKER_HPP
#include <mpi.h>

#include <mutex>
#include <set>

namespace amio_test {
inline std::mutex comm_mutex;
inline std::set<MPI_Comm> duplicated_comms;
inline std::size_t live_duplicates() {
    std::lock_guard<std::mutex> lock(comm_mutex);
    return duplicated_comms.size();
}
}  // namespace amio_test

// Each test executable includes this header in exactly one translation unit.
extern "C" int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *duplicate) {
    const int rc = PMPI_Comm_dup(comm, duplicate);
    if (rc == MPI_SUCCESS) {
        std::lock_guard<std::mutex> lock(amio_test::comm_mutex);
        amio_test::duplicated_comms.insert(*duplicate);
    }
    return rc;
}
extern "C" int MPI_Comm_free(MPI_Comm *comm) {
    const MPI_Comm old = *comm;
    const int rc = PMPI_Comm_free(comm);
    if (rc == MPI_SUCCESS) {
        std::lock_guard<std::mutex> lock(amio_test::comm_mutex);
        amio_test::duplicated_comms.erase(old);
    }
    return rc;
}
#endif
