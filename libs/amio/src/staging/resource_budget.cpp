#include "staging/resource_budget.hpp"

#include <stdexcept>
#include <utility>

namespace amio::detail {

ResourceBudget::ResourceBudget(Limits limits) : limits_(limits) {
    if (limits_.bytes == 0 || limits_.descriptors == 0 || limits_.control_bytes > limits_.bytes ||
        limits_.control_descriptors > limits_.descriptors) {
        throw std::invalid_argument("resource budget limits must be finite and control reserves must fit the total limits");
    }
}

std::optional<ResourceBudget::Reservation> ResourceBudget::reserve(std::size_t bytes, std::size_t descriptors, Class resource_class) {
    if (descriptors == 0 || (resource_class != Class::Data && resource_class != Class::Control)) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (next_id_ == 0) {
        return std::nullopt;
    }

    if (resource_class == Class::Data) {
        const std::size_t byte_limit = limits_.bytes - limits_.control_bytes;
        const std::size_t descriptor_limit = limits_.descriptors - limits_.control_descriptors;
        if (usage_.data_bytes > byte_limit || bytes > byte_limit - usage_.data_bytes || usage_.data_descriptors > descriptor_limit ||
            descriptors > descriptor_limit - usage_.data_descriptors) {
            return std::nullopt;
        }
    } else if (usage_.control_bytes > limits_.control_bytes || bytes > limits_.control_bytes - usage_.control_bytes ||
               usage_.control_descriptors > limits_.control_descriptors || descriptors > limits_.control_descriptors - usage_.control_descriptors) {
        return std::nullopt;
    }

    const std::uint64_t id = next_id_++;
    try {
        live_.emplace(id, Record{bytes, descriptors, resource_class});
    } catch (...) {
        return std::nullopt;
    }
    if (resource_class == Class::Data) {
        usage_.data_bytes += bytes;
        usage_.data_descriptors += descriptors;
    } else {
        usage_.control_bytes += bytes;
        usage_.control_descriptors += descriptors;
    }
    ++usage_.reservations;
    return Reservation{id, bytes, descriptors, resource_class};
}

bool ResourceBudget::release(const Reservation &reservation) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = live_.find(reservation.id);
    if (found == live_.end()) {
        return false;
    }
    const Record record = found->second;
    live_.erase(found);
    if (record.resource_class == Class::Data) {
        usage_.data_bytes -= record.bytes;
        usage_.data_descriptors -= record.descriptors;
    } else {
        usage_.control_bytes -= record.bytes;
        usage_.control_descriptors -= record.descriptors;
    }
    --usage_.reservations;
    return true;
}

ResourceBudget::Snapshot ResourceBudget::snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return usage_;
}

} // namespace amio::detail
