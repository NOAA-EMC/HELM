#ifndef AMIO_SRC_SCHEMA_SCHEMA_WIRE_BUFFER_HPP
#define AMIO_SRC_SCHEMA_SCHEMA_WIRE_BUFFER_HPP

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "amio/amio_errors.h"
#include "staging/resource_budget.hpp"

namespace amio::detail {

/** Own a single schema wire representation under a control-budget reservation.
 * The budget must outlive this object. Fragments must arrive contiguously.
 * The reservation covers wire bytes; a decoded schema needs its own budget.
 */
class SchemaWireBuffer {
   public:
    explicit SchemaWireBuffer(ResourceBudget &budget, std::size_t max_bytes) noexcept : budget_(&budget), max_bytes_(max_bytes) {}
    SchemaWireBuffer(const SchemaWireBuffer &) = delete;
    SchemaWireBuffer &operator=(const SchemaWireBuffer &) = delete;
    ~SchemaWireBuffer() {
        reset();
    }

    /** Reserve before allocating; failed admission leaves the buffer empty. */
    [[nodiscard]] amio_status_t begin(std::size_t bytes) noexcept {
        if (reservation_ || bytes == 0 || bytes > max_bytes_) return AMIO_ERR_INVALID_INPUT;
        try {
            auto reservation = budget_->reserve(bytes, 1, ResourceBudget::Class::Control);
            if (!reservation) return AMIO_ERR_BUDGET_EXCEEDED;
            reservation_ = reservation;
            try {
                bytes_ = std::make_unique<char[]>(bytes);
            } catch (...) {
                reset();
                return AMIO_ERR_BACKEND_FAILURE;
            }
            capacity_ = bytes;
            return AMIO_OK;
        } catch (...) {
            return AMIO_ERR_BACKEND_FAILURE;
        }
    }

    /** Copy the next fragment without growing the reserved allocation. */
    [[nodiscard]] amio_status_t append(std::size_t offset, std::span<const std::byte> fragment) noexcept {
        if (!reservation_ || offset != size_ || fragment.size() > capacity_ - size_) return AMIO_ERR_INVALID_INPUT;
        if (!fragment.empty()) std::memcpy(bytes_.get() + size_, fragment.data(), fragment.size());
        size_ += fragment.size();
        return AMIO_OK;
    }

    /** Expose only a fully received representation. */
    [[nodiscard]] std::string_view complete() const noexcept {
        return reservation_ && size_ == capacity_ ? std::string_view(bytes_.get(), size_) : std::string_view{};
    }

    /** Free storage before returning its control reservation. */
    void reset() noexcept {
        bytes_.reset();
        capacity_ = size_ = 0;
        if (reservation_) {
            (void)budget_->release(*reservation_);
            reservation_.reset();
        }
    }

   private:
    ResourceBudget *budget_;
    std::size_t max_bytes_;
    std::unique_ptr<char[]> bytes_;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
    std::optional<ResourceBudget::Reservation> reservation_;
};

}  // namespace amio::detail

#endif
