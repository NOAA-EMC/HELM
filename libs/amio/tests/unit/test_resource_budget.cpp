#include "staging/resource_budget.hpp"
#include "schema/schema_wire_buffer.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

void test_data_capacity_respects_control_reserve() {
    using amio::detail::ResourceBudget;
    const ResourceBudget::Limits limits{1000, 10, 100, 2};
    ResourceBudget budget(limits);

    auto retained_capacity = budget.reserve(700, 1);
    assert(retained_capacity.has_value());
    assert(!budget.reserve(201, 1).has_value());
    auto control = budget.reserve(100, 2, ResourceBudget::Class::Control);
    assert(control.has_value());
    assert(!budget.reserve(1, 1, ResourceBudget::Class::Control).has_value());

    const auto snapshot = budget.snapshot();
    assert(snapshot.data_bytes == 700);
    assert(snapshot.control_bytes == 100);
    assert(snapshot.data_descriptors == 1);
    assert(snapshot.control_descriptors == 2);
}

void test_descriptor_and_size_overflow_reject_before_accounting() {
    using amio::detail::ResourceBudget;
    ResourceBudget budget(ResourceBudget::Limits{64, 4, 8, 1});
    assert(!budget.reserve(std::numeric_limits<std::size_t>::max(), 1).has_value());
    assert(!budget.reserve(1, std::numeric_limits<std::size_t>::max()).has_value());
    assert(!budget.reserve(1, 0).has_value());
    const auto snapshot = budget.snapshot();
    assert(snapshot.data_bytes == 0);
    assert(snapshot.data_descriptors == 0);
}

void test_reservation_release_is_exactly_once() {
    using amio::detail::ResourceBudget;
    ResourceBudget budget(ResourceBudget::Limits{32, 3, 4, 1});
    auto reservation = budget.reserve(16, 2);
    assert(reservation.has_value());
    assert(budget.release(*reservation));
    assert(!budget.release(*reservation));
    const auto snapshot = budget.snapshot();
    assert(snapshot.data_bytes == 0);
    assert(snapshot.data_descriptors == 0);
    assert(snapshot.reservations == 0);
}

void test_invalid_reserve_configuration_is_rejected() {
    using amio::detail::ResourceBudget;
    bool threw = false;
    try {
        ResourceBudget invalid(ResourceBudget::Limits{64, 4, 65, 1});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

void test_schema_wire_reserves_before_allocation_and_releases() {
    using namespace amio::detail;
    ResourceBudget budget({64, 4, 8, 2});
    {
        SchemaWireBuffer buffer(budget, 16);
        assert(buffer.begin(9) == AMIO_ERR_BUDGET_EXCEEDED);
        assert(budget.snapshot().control_bytes == 0);
        assert(buffer.begin(4) == AMIO_OK);
        assert(buffer.begin(4) == AMIO_ERR_INVALID_INPUT);
        const std::array<std::byte, 2> bytes{std::byte{'a'}, std::byte{'b'}};
        assert(buffer.append(1, bytes) == AMIO_ERR_INVALID_INPUT);
        assert(buffer.append(0, bytes) == AMIO_OK);
        assert(buffer.complete().empty());
        assert(buffer.append(2, bytes) == AMIO_OK);
        assert(buffer.complete() == "abab");
        assert(buffer.append(4, bytes) == AMIO_ERR_INVALID_INPUT);
        assert(budget.snapshot().control_bytes == 4);
    }
    assert(budget.snapshot().control_bytes == 0 && budget.snapshot().control_descriptors == 0);
    const auto peak = budget.high_water();
    assert(peak.control_bytes == 4 && peak.control_descriptors == 1 && peak.reservations == 1);
}

} // namespace

int main() {
    test_data_capacity_respects_control_reserve();
    test_descriptor_and_size_overflow_reject_before_accounting();
    test_reservation_release_is_exactly_once();
    test_invalid_reserve_configuration_is_rejected();
    test_schema_wire_reserves_before_allocation_and_releases();
    return 0;
}
