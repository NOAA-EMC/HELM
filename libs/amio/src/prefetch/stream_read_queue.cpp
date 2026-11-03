#include "prefetch/stream_read_queue.hpp"

#include <cstring>
#include <limits>

namespace amio::detail {

amio_status_t StreamReadQueue::read(const StreamLease &lease, std::string_view field, const RegionSelection &selection, std::size_t element_bytes,
                                    std::span<std::byte> destination, std::size_t &written, std::string_view logical_field_name) const noexcept {
    written = 0;
    if (!lease.step || lease.consumer != consumer_ || element_bytes == 0 ||
        retention_->lease_step_id(lease.lease_id, consumer_) != lease.step->step_id)
        return AMIO_ERR_INVALID_HANDLE;
    const auto payload = lease.step->fields.find(field);
    const auto shape = lease.step->field_shapes.find(field);
    const auto origin = lease.step->field_offsets.find(field);
    if (payload == lease.step->fields.end() || shape == lease.step->field_shapes.end() || origin == lease.step->field_offsets.end())
        return lease.step->absent_optional_fields.contains(field) ||
                       (!logical_field_name.empty() && lease.step->absent_optional_fields.contains(logical_field_name))
                   ? AMIO_ERR_FIELD_NOT_PRESENT
                   : AMIO_ERR_INVALID_INPUT;
    const auto rank = shape->second.size();
    if (rank == 0 || origin->second.size() != rank || selection.offsets.size() != rank || selection.extents.size() != rank ||
        payload->second.size() % element_bytes != 0)
        return AMIO_ERR_INVALID_INPUT;
    std::vector<std::uint64_t> local_offsets(rank), strides(rank, 1);
    std::size_t source_elements = 1, output_elements = 1;
    for (std::size_t i = rank; i-- > 0;) {
        if (selection.offsets[i] < origin->second[i]) return AMIO_ERR_PARTITION_INVALID;
        local_offsets[i] = selection.offsets[i] - origin->second[i];
        if (local_offsets[i] > shape->second[i] || selection.extents[i] > shape->second[i] - local_offsets[i]) return AMIO_ERR_PARTITION_INVALID;
        if (i + 1 < rank) {
            if (shape->second[i + 1] != 0 && strides[i + 1] > std::numeric_limits<std::size_t>::max() / shape->second[i + 1])
                return AMIO_ERR_BUDGET_EXCEEDED;
            strides[i] = strides[i + 1] * static_cast<std::size_t>(shape->second[i + 1]);
        }
        if (shape->second[i] != 0 && source_elements > std::numeric_limits<std::size_t>::max() / shape->second[i]) return AMIO_ERR_BUDGET_EXCEEDED;
        source_elements *= static_cast<std::size_t>(shape->second[i]);
        if (selection.extents[i] != 0 && output_elements > std::numeric_limits<std::size_t>::max() / selection.extents[i])
            return AMIO_ERR_BUDGET_EXCEEDED;
        output_elements *= static_cast<std::size_t>(selection.extents[i]);
    }
    if (source_elements != payload->second.size() / element_bytes || output_elements > std::numeric_limits<std::size_t>::max() / element_bytes)
        return AMIO_ERR_SCHEMA_MISMATCH;
    written = output_elements * element_bytes;
    if (destination.size() < written) {
        written = 0;
        return AMIO_ERR_BUFFER_TOO_SMALL;
    }
    for (std::size_t linear = 0; linear < output_elements; ++linear) {
        auto remainder = linear;
        std::size_t source_index = 0;
        for (std::size_t i = rank; i-- > 0;) {
            const auto extent = static_cast<std::size_t>(selection.extents[i]);
            const auto coordinate = extent ? remainder % extent : 0;
            if (extent) remainder /= extent;
            source_index += (static_cast<std::size_t>(local_offsets[i]) + coordinate) * strides[i];
        }
        std::memcpy(destination.data() + linear * element_bytes, payload->second.data() + source_index * element_bytes, element_bytes);
    }
    return AMIO_OK;
}

amio_status_t StreamReadQueue::release(const StreamLease &lease) noexcept {
    if (!lease.step || lease.consumer != consumer_ || retention_->lease_step_id(lease.lease_id, consumer_) != lease.step->step_id)
        return AMIO_ERR_INVALID_HANDLE;
    return retention_->release(lease.lease_id);
}

}  // namespace amio::detail
