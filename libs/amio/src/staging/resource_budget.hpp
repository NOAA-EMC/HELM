#ifndef AMIO_SRC_STAGING_RESOURCE_BUDGET_HPP
#define AMIO_SRC_STAGING_RESOURCE_BUDGET_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace amio::detail {

/** Thread-safe accounting for retained capacity and operation descriptors. */
class ResourceBudget {
   public:
    enum class Class : std::uint8_t { Data, Control };

    struct Limits {
        std::size_t bytes;
        std::size_t descriptors;
        std::size_t control_bytes;
        std::size_t control_descriptors;
    };

    struct Reservation {
        std::uint64_t id;
        std::size_t bytes;
        std::size_t descriptors;
        Class resource_class;
    };

    struct Snapshot {
        std::size_t data_bytes = 0;
        std::size_t control_bytes = 0;
        std::size_t data_descriptors = 0;
        std::size_t control_descriptors = 0;
        std::size_t reservations = 0;
    };

    explicit ResourceBudget(Limits limits);
    ResourceBudget(const ResourceBudget &) = delete;
    ResourceBudget &operator=(const ResourceBudget &) = delete;

    /** Reserve retained capacity, not merely currently active payload bytes. */
    [[nodiscard]] std::optional<Reservation> reserve(std::size_t bytes, std::size_t descriptors, Class resource_class = Class::Data);
    /** Release a reservation once; returns false for an unknown/already released ID. */
    bool release(const Reservation &reservation) noexcept;
    [[nodiscard]] Snapshot snapshot() const noexcept;
    [[nodiscard]] const Limits &limits() const noexcept { return limits_; }

   private:
    struct Record {
        std::size_t bytes;
        std::size_t descriptors;
        Class resource_class;
    };

    Limits limits_;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, Record> live_;
    std::uint64_t next_id_ = 1;
    Snapshot usage_;
};

} // namespace amio::detail

#endif // AMIO_SRC_STAGING_RESOURCE_BUDGET_HPP
