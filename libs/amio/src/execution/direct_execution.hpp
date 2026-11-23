#ifndef AMIO_SRC_EXECUTION_DIRECT_EXECUTION_HPP
#define AMIO_SRC_EXECUTION_DIRECT_EXECUTION_HPP

#include <functional>
#include <map>
#include <string>

#include "catalog/posix_catalog.hpp"
#include "steps/step_coordinator.hpp"

namespace amio::detail {

/**
 * Synchronous direct-writer orchestration for one sealed producer step.
 *
 * The payload callback writes every native field payload and fills the record's
 * payload descriptors. The vote callback must return the communicator-wide
 * result (for a one-rank direct writer it can return its argument unchanged).
 * The completion record and authoritative head are published only after that
 * vote succeeds.
 */
class DirectExecution {
   public:
    using PayloadWriter = std::function<amio_status_t(const std::map<std::string, StepContributionSnapshot, std::less<>> &, CompletionRecord &)>;
    using WriterSuccessVote = std::function<amio_status_t(amio_status_t)>;

    DirectExecution(PosixCatalog &catalog, PosixSyncLevel guarantee = PosixSyncLevel::Visibility) : catalog_(&catalog), guarantee_(guarantee) {}

    [[nodiscard]] PublicationResult publish(StepCoordinator &step, CompletionRecord record, const PayloadWriter &write_payloads,
                                            const WriterSuccessVote &writer_success_vote);
    [[nodiscard]] PublicationResult reconcile(StepCoordinator &step);

   private:
    [[nodiscard]] PublicationResult fail(StepCoordinator &step, amio_status_t status);

    PosixCatalog *catalog_;
    PosixSyncLevel guarantee_;
    std::string pending_commit_;
    std::string previous_commit_;
};

}  // namespace amio::detail

#endif
