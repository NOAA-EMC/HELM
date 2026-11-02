#include "execution/context.hpp"
#include "steps/collective_preflight.hpp"
#include "schema/partition_validator.hpp"
#include "staging/resource_budget.hpp"

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

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
    roles.mode = ExecutionMode::Aggregated;
    roles.rank_count = static_cast<std::uint32_t>(size);
    roles.producers = {0};
    for (int writer = 1; writer < size; ++writer) roles.writers.push_back(writer);
    std::unique_ptr<ExecutionContext> context;
    const auto status = ExecutionContext::create(roles, ProgressPolicy::Caller, "schema=consistent;step=1", MPI_COMM_WORLD, context);
    assert(status == AMIO_OK);
    assert(context->control_communicator() != MPI_COMM_NULL);
    assert(context->data_communicator() != MPI_COMM_NULL);
    assert(context->control_communicator() != context->data_communicator());
    assert(context && context->rank() == static_cast<std::uint32_t>(rank));
    assert(context->size() == static_cast<std::uint32_t>(size));
    std::unique_ptr<ExecutionContext> second_output;
    assert(ExecutionContext::create(roles, ProgressPolicy::Caller, "schema=second-output", MPI_COMM_WORLD, second_output) == AMIO_OK);
    assert(second_output && second_output->communicator() != context->communicator());

    // A schema digest disagreement is a collective preflight failure on every rank.
    const auto local_schema = std::string("schema=") + (rank == 0 ? "digest-a" : "digest-b");
    std::unique_ptr<ExecutionContext> mismatched;
    const auto mismatch_status = ExecutionContext::create(roles, ProgressPolicy::Caller, local_schema, MPI_COMM_WORLD, mismatched);
    assert(mismatch_status == AMIO_ERR_INVALID_POLICY);
    assert(!mismatched);

    PreflightRequest preflight;
    preflight.schema_digest = "schema-v1";
    preflight.schema_variables = {"temperature"};
    preflight.required_variables = {"temperature"};
    if (rank == 0) preflight.contributions.push_back({"temperature", {4, 4}, {0, 0}, {2, 4}, false});
    else if (rank == 1) preflight.contributions.push_back({"temperature", {4, 4}, {2, 0}, {2, 4}, false});
    else preflight.contributions.push_back({"temperature", {4, 4}, {0, 0}, {0, 0}, true});
    auto rounds = CollectivePreflight::run(MPI_COMM_WORLD, preflight);
    assert(rounds.status == AMIO_OK && rounds.rounds.size() == 1);
    assert(rounds.rounds[0].participants.size() == static_cast<std::size_t>(size));
    assert(rounds.rounds[0].local_region.has_value() == (rank < 2));

    preflight.schema_digest = std::string("schema-") + std::to_string(rank);
    assert(CollectivePreflight::run(MPI_COMM_WORLD, preflight).status == AMIO_ERR_SCHEMA_MISMATCH);
    preflight.schema_digest = "schema-v1";
    preflight.local_admission_status = rank == 0 ? AMIO_ERR_BUDGET_EXCEEDED : AMIO_OK;
    assert(CollectivePreflight::run(MPI_COMM_WORLD, preflight).status == AMIO_ERR_BUDGET_EXCEEDED);
    preflight.local_admission_status = AMIO_OK;
    preflight.descriptor_limit = 1;
    assert(CollectivePreflight::run(MPI_COMM_WORLD, preflight).status == AMIO_ERR_BUDGET_EXCEEDED);

    // Empty contributors participate in control agreement without contributing payload volume.
    ResourceBudget budget({64, 8, 16, 2});
    const auto reservation = budget.reserve(rank == 0 ? 128 : 16, 1);
    int local_admission = reservation ? AMIO_OK : AMIO_ERR_BUDGET_EXCEEDED;
    int collective_admission = AMIO_OK;
    MPI_Allreduce(&local_admission, &collective_admission, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    assert(collective_admission == AMIO_ERR_BUDGET_EXCEEDED);
    if (reservation) budget.release(*reservation);

    PartitionValidator validator;
    std::vector<PartitionBox> boxes;
    if (rank == 0) boxes.push_back({0, {0, 0}, {2, 4}, false});
    else if (rank == 1) boxes.push_back({1, {2, 0}, {2, 4}, false});
    else boxes.push_back({rank, {0, 0}, {0, 0}, true});
    auto partition = validator.validate_distributed(MPI_COMM_WORLD, {4, 4}, boxes, true, 128, 32, 4096);
    assert(partition.status == AMIO_OK && partition.complete);
    if (rank == 1) boxes[0].offsets[0] = 1;
    partition = validator.validate_distributed(MPI_COMM_WORLD, {4, 4}, boxes, true, 128, 32, 4096);
    assert(partition.status == AMIO_ERR_PARTITION_INVALID);
    if (rank == 1) boxes[0] = {1, {0, 0}, {0, 0}, true};
    partition = validator.validate_distributed(MPI_COMM_WORLD, {4, 4}, boxes, true, 128, 32, 4096);
    assert(partition.status == AMIO_ERR_PARTITION_INVALID);
    boxes.clear();
    if (rank == 0) {
        boxes.push_back({0, {0, 0}, {1, 4}, false});
        boxes.push_back({0, {1, 0}, {1, 4}, false});
    } else if (rank == 1) boxes.push_back({1, {2, 0}, {2, 4}, false});
    else boxes.push_back({rank, {0, 0}, {0, 0}, true});
    partition = validator.validate_distributed(MPI_COMM_WORLD, {4, 4}, boxes, true, 1, 32, 4096);
    assert(partition.status == AMIO_ERR_BUDGET_EXCEEDED);

    MPI_Finalize();
    std::cout << "test_mpi_steps: passed\n";
}
