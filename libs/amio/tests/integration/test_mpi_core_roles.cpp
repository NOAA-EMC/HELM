#include "amio/amio.h"

#include <mpi.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) return 1;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) MPI_Abort(MPI_COMM_WORLD, 2);

    const char *manifest = R"yaml(
staging_pool:
  buffer_count: 2
  buffer_capacity_bytes: 4096
  max_buffer_count: 2
worker_pool:
  threads: 1
prefetch:
  depth: 1
  read_timeout_s: 1
staging_timeout_ms: 1000
backpressure:
  queue_capacity: 8
backend: netcdf4
io_ranks:
  - 1
)yaml";

    amio_core_handle core = nullptr;
    const auto init_status = amio_init_from_string(manifest, "yaml", &core);
    int local_status = init_status == AMIO_OK && core != nullptr ? 0 : 1;
    int global_status = 1;
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_status == 0) {
        if (rank == 0) {
            amio_dataset_handle dataset = reinterpret_cast<amio_dataset_handle>(static_cast<std::uintptr_t>(1));
            const auto open_status = amio_open_dataset_from_string(core, manifest, "yaml", AMIO_MODE_WRITE, &dataset);
            if (open_status != AMIO_ERR_INVALID_POLICY || dataset != nullptr) local_status = 1;
        } else {
            const std::string dataset_manifest = std::string(manifest) +
                "\npath: /tmp/amio-role-owned-output.nc\noutput_path: /tmp/amio-role-owned-output.nc\ndata_model: classic\n";
            amio_dataset_handle dataset = nullptr;
            const auto open_status = amio_open_dataset_from_string(core, dataset_manifest.c_str(), "yaml", AMIO_MODE_WRITE, &dataset);
            if (open_status != AMIO_OK || dataset == nullptr) local_status = 1;
            else if (amio_close_dataset(dataset) != AMIO_OK) local_status = 1;
            std::remove("/tmp/amio-role-owned-output.nc");
        }
        MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    if (global_status == 0) {
        const auto finalize_status = amio_finalize(core);
        local_status = finalize_status == AMIO_OK ? 0 : 1;
        MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    // Legacy initialization is rank-local.  A bad manifest on one rank must
    // not strand another rank in an evolution-specific collective.
    if (global_status == 0) {
        const char *evolution_manifest = R"yaml(
amio_version: 1
execution:
  mode: persistent_aggregated
  progress: caller
destination:
  backend: netcdf4
  path: /tmp/amio-evolution-role-output.nc
  create_policy: exclusive
membership:
  producers: [0]
  writers: [1]
budgets:
  total_bytes: 1000000
  staging_bytes: 200000
  metadata_bytes: 100000
  transport_bytes: 200000
  view_bytes: 100000
  control_reserve_bytes: 10000
  total_descriptors: 256
  control_reserve_descriptors: 8
transfer:
  chunk_bytes: 4096
  credits_per_peer: 4
)yaml";
        std::string input(evolution_manifest);
        if (rank == 1) input += "\nunknown_key: true\n";
        core = reinterpret_cast<amio_core_handle>(static_cast<std::uintptr_t>(1));
        const auto expected = rank == 0 ? AMIO_ERR_UNSUPPORTED : AMIO_ERR_MANIFEST_INVALID;
        if (amio_init_from_string(input.c_str(), "yaml", &core) != expected || core != nullptr) local_status = 1;
        MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    if (global_status != 0 && rank == 0) std::fprintf(stderr, "configured I/O-rank initialization/finalization failed\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return global_status;
}
