#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include "execution/context.hpp"

using namespace amio::detail;

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size >= 2);

    RoleConfiguration roles;
    roles.mode = ExecutionMode::Direct;
    roles.rank_count = static_cast<std::uint32_t>(size);
    if (rank == 0)
        roles.producers = {0};
    else {
        for (int producer = 0; producer < size; ++producer) roles.producers.push_back(producer);
    }
    std::unique_ptr<ExecutionContext> context;
    const auto status = ExecutionContext::create(roles, ProgressPolicy::Caller, "same-configuration", MPI_COMM_WORLD, context);
    assert(status == AMIO_ERR_INVALID_POLICY);
    assert(!context);

    assert(validate_progress_support(ProgressPolicy::Background, 2) == AMIO_ERR_THREADING_UNSUPPORTED);
    assert(validate_progress_support(ProgressPolicy::Caller, 2) == AMIO_OK);
    MPI_Finalize();
    std::cout << "test_mpi_step_failure: passed\n";
}
