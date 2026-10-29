#include "execution/context.hpp"

#include <cassert>

namespace {

using amio::detail::ExecutionMode;
using amio::detail::ProgressPolicy;
using amio::detail::RoleConfiguration;

void test_valid_role_partitions() {
    RoleConfiguration aggregated;
    aggregated.mode = ExecutionMode::Aggregated;
    aggregated.rank_count = 4;
    aggregated.producers = {0, 1, 2};
    aggregated.writers = {3};
    assert(amio::detail::validate_roles(aggregated) == AMIO_OK);

    RoleConfiguration stream;
    stream.mode = ExecutionMode::Stream;
    stream.rank_count = 2;
    stream.producers = {0};
    stream.consumers = {1};
    assert(amio::detail::validate_roles(stream) == AMIO_OK);
}

void test_invalid_or_overlapping_roles_are_rejected() {
    RoleConfiguration one_rank_aggregation;
    one_rank_aggregation.mode = ExecutionMode::Aggregated;
    one_rank_aggregation.rank_count = 1;
    one_rank_aggregation.producers = {0};
    assert(amio::detail::validate_roles(one_rank_aggregation) == AMIO_ERR_INVALID_POLICY);

    RoleConfiguration overlap;
    overlap.mode = ExecutionMode::Aggregated;
    overlap.rank_count = 2;
    overlap.producers = {0, 1};
    overlap.writers = {1};
    assert(amio::detail::validate_roles(overlap) == AMIO_ERR_INVALID_POLICY);

    RoleConfiguration missing_rank;
    missing_rank.mode = ExecutionMode::Stream;
    missing_rank.rank_count = 3;
    missing_rank.producers = {0};
    missing_rank.consumers = {1};
    assert(amio::detail::validate_roles(missing_rank) == AMIO_ERR_INVALID_POLICY);
}

void test_progress_mode_preflight() {
    assert(amio::detail::validate_progress_support(ProgressPolicy::Background, 3) == AMIO_OK);
    assert(amio::detail::validate_progress_support(ProgressPolicy::Background, 2) == AMIO_ERR_THREADING_UNSUPPORTED);
    assert(amio::detail::validate_progress_support(ProgressPolicy::Caller, 1) == AMIO_OK);
    assert(amio::detail::validate_progress_support(ProgressPolicy::Caller, 0) == AMIO_ERR_THREADING_UNSUPPORTED);
}

} // namespace

int main() {
    test_valid_role_partitions();
    test_invalid_or_overlapping_roles_are_rejected();
    test_progress_mode_preflight();
    return 0;
}
