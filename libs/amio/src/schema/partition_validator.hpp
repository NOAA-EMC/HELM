#ifndef AMIO_SRC_SCHEMA_PARTITION_VALIDATOR_HPP
#define AMIO_SRC_SCHEMA_PARTITION_VALIDATOR_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "amio/amio_errors.h"

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif

namespace amio::detail {

struct PartitionBox {
    int producer_rank = 0;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> extents;
    bool empty = false;
};

struct PartitionResult {
    amio_status_t status = AMIO_OK;
    bool global_checks_performed = false;
    bool complete = false;
    std::size_t comparisons = 0;
};

class PartitionValidator {
   public:
    [[nodiscard]] static PartitionResult validate(const std::vector<std::uint64_t> &global_extents, const std::vector<PartitionBox> &boxes,
                                                  bool strict, std::size_t work_limit, std::size_t descriptor_limit);
#ifdef AMIO_HAS_MPI
    [[nodiscard]] static PartitionResult validate_distributed(MPI_Comm communicator, const std::vector<std::uint64_t> &global_extents,
                                                              const std::vector<PartitionBox> &local_boxes, bool strict, std::size_t work_limit,
                                                              std::size_t descriptor_limit, std::size_t metadata_byte_limit = 16 * 1024 * 1024);
#endif
};

}  // namespace amio::detail

#endif  // AMIO_SRC_SCHEMA_PARTITION_VALIDATOR_HPP
