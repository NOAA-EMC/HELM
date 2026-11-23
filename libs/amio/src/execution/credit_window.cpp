#include "execution/credit_window.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace amio::detail {

CreditWindow::CreditWindow(Limits limits) : limits_(limits), peers_(limits.peer_count) {
    if (limits_.peer_count == 0 || limits_.per_peer_bytes == 0 || limits_.per_peer_descriptors == 0 || limits_.total_data_bytes == 0 ||
        limits_.total_data_descriptors == 0 || limits_.control_bytes == 0 || limits_.control_descriptors == 0) {
        throw std::invalid_argument("credit windows require finite positive peer, data, and control limits");
    }
}

std::optional<CreditWindow::Ticket> CreditWindow::acquire(std::size_t peer, CreditOwner owner, std::size_t bytes, std::size_t descriptors) {
    if (peer >= peers_.size() || (owner != CreditOwner::Sender && owner != CreditOwner::Receiver) || descriptors == 0) return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_id_ == 0 || usage_.data_bytes > limits_.total_data_bytes || bytes > limits_.total_data_bytes - usage_.data_bytes ||
        usage_.data_descriptors > limits_.total_data_descriptors || descriptors > limits_.total_data_descriptors - usage_.data_descriptors) {
        return std::nullopt;
    }
    auto &use = peers_[peer];
    const auto peer_bytes = owner == CreditOwner::Sender ? use.sender_bytes : use.receiver_bytes;
    const auto peer_descriptors = owner == CreditOwner::Sender ? use.sender_descriptors : use.receiver_descriptors;
    if (peer_bytes > limits_.per_peer_bytes || bytes > limits_.per_peer_bytes - peer_bytes || peer_descriptors > limits_.per_peer_descriptors ||
        descriptors > limits_.per_peer_descriptors - peer_descriptors)
        return std::nullopt;
    const auto id = next_id_++;
    try {
        live_.emplace(id, Record{peer, bytes, descriptors, owner});
    } catch (...) {
        return std::nullopt;
    }
    if (owner == CreditOwner::Sender) {
        use.sender_bytes += bytes;
        use.sender_descriptors += descriptors;
        usage_.sender_bytes += bytes;
        usage_.sender_descriptors += descriptors;
    } else {
        use.receiver_bytes += bytes;
        use.receiver_descriptors += descriptors;
        usage_.receiver_bytes += bytes;
        usage_.receiver_descriptors += descriptors;
    }
    usage_.data_bytes += bytes;
    usage_.data_descriptors += descriptors;
    high_water_.sender_bytes = std::max(high_water_.sender_bytes, usage_.sender_bytes);
    high_water_.receiver_bytes = std::max(high_water_.receiver_bytes, usage_.receiver_bytes);
    high_water_.data_bytes = std::max(high_water_.data_bytes, usage_.data_bytes);
    high_water_.sender_descriptors = std::max(high_water_.sender_descriptors, usage_.sender_descriptors);
    high_water_.receiver_descriptors = std::max(high_water_.receiver_descriptors, usage_.receiver_descriptors);
    high_water_.data_descriptors = std::max(high_water_.data_descriptors, usage_.data_descriptors);
    return Ticket{id, peer, owner, bytes, descriptors};
}

std::optional<CreditWindow::Ticket> CreditWindow::acquire_control(std::size_t bytes, std::size_t descriptors) {
    if (descriptors == 0) return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_id_ == 0 || usage_.control_bytes > limits_.control_bytes || bytes > limits_.control_bytes - usage_.control_bytes ||
        usage_.control_descriptors > limits_.control_descriptors || descriptors > limits_.control_descriptors - usage_.control_descriptors) {
        return std::nullopt;
    }
    const auto id = next_id_++;
    try {
        live_.emplace(id, Record{0, bytes, descriptors, CreditOwner::Control});
    } catch (...) {
        return std::nullopt;
    }
    usage_.control_bytes += bytes;
    usage_.control_descriptors += descriptors;
    high_water_.control_bytes = std::max(high_water_.control_bytes, usage_.control_bytes);
    high_water_.control_descriptors = std::max(high_water_.control_descriptors, usage_.control_descriptors);
    return Ticket{id, 0, CreditOwner::Control, bytes, descriptors};
}

bool CreditWindow::release(const Ticket &ticket) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = live_.find(ticket.id);
    if (found == live_.end()) return false;
    const auto record = found->second;
    if (ticket.peer != record.peer || ticket.owner != record.owner || ticket.bytes != record.bytes || ticket.descriptors != record.descriptors)
        return false;
    live_.erase(found);
    if (record.owner == CreditOwner::Control) {
        usage_.control_bytes -= record.bytes;
        usage_.control_descriptors -= record.descriptors;
    } else {
        auto &use = peers_[record.peer];
        if (record.owner == CreditOwner::Sender) {
            use.sender_bytes -= record.bytes;
            use.sender_descriptors -= record.descriptors;
            usage_.sender_bytes -= record.bytes;
            usage_.sender_descriptors -= record.descriptors;
        } else {
            use.receiver_bytes -= record.bytes;
            use.receiver_descriptors -= record.descriptors;
            usage_.receiver_bytes -= record.bytes;
            usage_.receiver_descriptors -= record.descriptors;
        }
        usage_.data_bytes -= record.bytes;
        usage_.data_descriptors -= record.descriptors;
    }
    return true;
}

CreditWindow::Snapshot CreditWindow::snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return usage_;
}

CreditWindow::Snapshot CreditWindow::high_water() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return high_water_;
}

}  // namespace amio::detail
