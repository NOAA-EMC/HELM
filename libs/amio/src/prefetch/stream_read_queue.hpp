#ifndef AMIO_SRC_PREFETCH_STREAM_READ_QUEUE_HPP
#define AMIO_SRC_PREFETCH_STREAM_READ_QUEUE_HPP

#include <chrono>
#include <cstddef>
#include <span>
#include <string_view>

#include "execution/stream_retention.hpp"
#include "steps/contribution_plan.hpp"

namespace amio::detail {

/** Consumer-side lease and bounded region reads over completed stream steps. */
class StreamReadQueue {
   public:
    StreamReadQueue(StreamRetention &retention, std::uint32_t consumer) noexcept : retention_(&retention), consumer_(consumer) {}

    [[nodiscard]] StreamAcquireResult acquire(std::chrono::milliseconds timeout) {
        return retention_->acquire_next(consumer_, timeout);
    }
    [[nodiscard]] amio_status_t read(const StreamLease &lease, std::string_view field, const RegionSelection &selection, std::size_t element_bytes,
                                     std::span<std::byte> destination, std::size_t &written, std::string_view logical_field_name = {}) const noexcept;
    [[nodiscard]] amio_status_t release(const StreamLease &lease) noexcept;

   private:
    StreamRetention *retention_;
    std::uint32_t consumer_;
};

}  // namespace amio::detail

#endif
