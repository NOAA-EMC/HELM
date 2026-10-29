#ifndef AMIO_SRC_STEPS_COLLECTIVE_PREFLIGHT_HPP
#define AMIO_SRC_STEPS_COLLECTIVE_PREFLIGHT_HPP

#include "amio/amio_errors.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif

namespace amio::detail {

struct ContributionRegion {
    std::string variable;
    std::vector<std::uint64_t> global_extents;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> extents;
    bool empty = false;
};

struct PreflightRequest {
    std::string schema_digest;
    std::vector<std::string> schema_variables;
    std::vector<std::string> required_variables;
    std::vector<ContributionRegion> contributions;
    amio_status_t local_admission_status = AMIO_OK;
    std::size_t descriptor_limit = 100000;
    std::size_t metadata_byte_limit = 16 * 1024 * 1024;
};

struct PreflightRound {
    std::string variable;
    std::size_t fragment_index = 0;
    std::vector<int> participants;
    std::optional<ContributionRegion> local_region;
};

struct PreflightResult {
    amio_status_t status = AMIO_OK;
    std::vector<PreflightRound> rounds;
};

/** Validate bounded metadata and produce identical transfer-round order on all participants. */
class CollectivePreflight {
   public:
#ifdef AMIO_HAS_MPI
    [[nodiscard]] static PreflightResult run(MPI_Comm communicator, const PreflightRequest &request);
#else
    [[nodiscard]] static PreflightResult run(int communicator, const PreflightRequest &request);
#endif
};

} // namespace amio::detail

#endif // AMIO_SRC_STEPS_COLLECTIVE_PREFLIGHT_HPP
