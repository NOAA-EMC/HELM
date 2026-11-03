#include "schema/partition_validator.hpp"

#include <algorithm>
#include <limits>

namespace amio::detail {
namespace {

bool checked_product(const std::vector<std::uint64_t> &values, std::uint64_t &product) noexcept {
    product = 1;
    for (const auto value : values) {
        if (value == 0 || value > std::numeric_limits<std::uint64_t>::max() / product) return false;
        product *= value;
    }
    return true;
}

PartitionResult invalid(amio_status_t status) noexcept {
    return {status, false, false, 0};
}

}  // namespace

PartitionResult PartitionValidator::validate(const std::vector<std::uint64_t> &global_extents, const std::vector<PartitionBox> &boxes, bool strict,
                                             std::size_t work_limit, std::size_t descriptor_limit) {
    if (global_extents.empty() || global_extents.size() > 7 || descriptor_limit == 0 || (strict && work_limit == 0)) {
        return invalid(AMIO_ERR_INVALID_INPUT);
    }
    for (const auto extent : global_extents)
        if (extent == 0) return invalid(AMIO_ERR_INVALID_INPUT);
    if (boxes.size() > descriptor_limit) return invalid(AMIO_ERR_BUDGET_EXCEEDED);

    for (const auto &box : boxes) {
        if (box.producer_rank < 0 || box.offsets.size() != global_extents.size() || box.extents.size() != global_extents.size()) {
            return invalid(AMIO_ERR_PARTITION_INVALID);
        }
        bool has_zero = false;
        for (std::size_t dimension = 0; dimension < global_extents.size(); ++dimension) {
            if (box.offsets[dimension] > global_extents[dimension] || box.extents[dimension] > global_extents[dimension] - box.offsets[dimension]) {
                return invalid(AMIO_ERR_PARTITION_INVALID);
            }
            has_zero = has_zero || box.extents[dimension] == 0;
        }
        if (box.empty != has_zero) return invalid(AMIO_ERR_PARTITION_INVALID);
    }

    if (!strict) return {AMIO_OK, false, false, 0};

    std::size_t comparisons = 0;
    std::vector<const PartitionBox *> nonempty;
    nonempty.reserve(boxes.size());
    for (const auto &box : boxes)
        if (!box.empty) nonempty.push_back(&box);
    for (std::size_t left = 0; left < nonempty.size(); ++left) {
        for (std::size_t right = left + 1; right < nonempty.size(); ++right) {
            if (comparisons == work_limit) return invalid(AMIO_ERR_BUDGET_EXCEEDED);
            ++comparisons;
            bool intersects = true;
            for (std::size_t dimension = 0; dimension < global_extents.size(); ++dimension) {
                const auto left_end = nonempty[left]->offsets[dimension] + nonempty[left]->extents[dimension];
                const auto right_end = nonempty[right]->offsets[dimension] + nonempty[right]->extents[dimension];
                if (nonempty[left]->offsets[dimension] >= right_end || nonempty[right]->offsets[dimension] >= left_end) {
                    intersects = false;
                    break;
                }
            }
            if (intersects) return {AMIO_ERR_PARTITION_INVALID, true, false, comparisons};
        }
    }

    std::uint64_t expected_volume = 0;
    if (!checked_product(global_extents, expected_volume)) return invalid(AMIO_ERR_INVALID_INPUT);
    std::uint64_t covered_volume = 0;
    for (const auto *box : nonempty) {
        std::uint64_t volume = 0;
        if (!checked_product(box->extents, volume) || volume > std::numeric_limits<std::uint64_t>::max() - covered_volume) {
            return invalid(AMIO_ERR_INVALID_INPUT);
        }
        covered_volume += volume;
    }
    if (covered_volume != expected_volume) return {AMIO_ERR_PARTITION_INVALID, true, false, comparisons};
    return {AMIO_OK, true, true, comparisons};
}

#ifdef AMIO_HAS_MPI
PartitionResult PartitionValidator::validate_distributed(MPI_Comm communicator, const std::vector<std::uint64_t> &global_extents,
                                                         const std::vector<PartitionBox> &local_boxes, bool strict, std::size_t work_limit,
                                                         std::size_t descriptor_limit, std::size_t metadata_byte_limit) {
    int rank = 0;
    int size = 0;
    if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS || MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
        return invalid(AMIO_ERR_COMM_SPLIT_FAILED);
    }
    int local_status = AMIO_OK;
    if (global_extents.empty() || global_extents.size() > 7 || local_boxes.size() > descriptor_limit || descriptor_limit == 0 ||
        metadata_byte_limit == 0 || (strict && work_limit == 0))
        local_status = AMIO_ERR_INVALID_INPUT;
    if (global_extents.size() > 7) local_status = AMIO_ERR_INVALID_INPUT;
    for (const auto &box : local_boxes) {
        if (box.producer_rank != rank) local_status = AMIO_ERR_PARTITION_INVALID;
    }
    auto local_bounds = validate(global_extents, local_boxes, false, 1, std::max<std::size_t>(1, descriptor_limit));
    if (local_status == AMIO_OK && local_bounds.status != AMIO_OK) local_status = local_bounds.status;
    int common_status = AMIO_OK;
    if (MPI_Allreduce(&local_status, &common_status, 1, MPI_INT, MPI_MAX, communicator) != MPI_SUCCESS) return invalid(AMIO_ERR_COMM_SPLIT_FAILED);
    if (common_status != AMIO_OK) return invalid(static_cast<amio_status_t>(common_status));

    const auto dimensions = static_cast<int>(global_extents.size());
    std::vector<unsigned long long> local_shape(global_extents.begin(), global_extents.end());
    std::vector<unsigned long long> min_shape(local_shape.size()), max_shape(local_shape.size());
    if (MPI_Allreduce(local_shape.data(), min_shape.data(), dimensions, MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator) != MPI_SUCCESS ||
        MPI_Allreduce(local_shape.data(), max_shape.data(), dimensions, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator) != MPI_SUCCESS) {
        return invalid(AMIO_ERR_COMM_SPLIT_FAILED);
    }
    if (min_shape != max_shape) return invalid(AMIO_ERR_INVALID_POLICY);

    unsigned long long local_limits[3] = {static_cast<unsigned long long>(work_limit), static_cast<unsigned long long>(descriptor_limit),
                                          static_cast<unsigned long long>(metadata_byte_limit)};
    unsigned long long common_limits[3] = {};
    if (MPI_Allreduce(local_limits, common_limits, 3, MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator) != MPI_SUCCESS) {
        return invalid(AMIO_ERR_COMM_SPLIT_FAILED);
    }

    const std::size_t words_per_box = 2 + global_extents.size() * 2;
    std::vector<unsigned long long> local_data;
    if (local_boxes.size() > std::numeric_limits<std::size_t>::max() / words_per_box) return invalid(AMIO_ERR_BUDGET_EXCEEDED);
    local_data.reserve(local_boxes.size() * words_per_box);
    for (const auto &box : local_boxes) {
        local_data.push_back(static_cast<unsigned long long>(box.producer_rank));
        local_data.push_back(box.empty ? 1ull : 0ull);
        for (const auto value : box.offsets) local_data.push_back(static_cast<unsigned long long>(value));
        for (const auto value : box.extents) local_data.push_back(static_cast<unsigned long long>(value));
    }
    if (local_data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return invalid(AMIO_ERR_BUDGET_EXCEEDED);
    const int local_word_count = static_cast<int>(local_data.size());
    std::vector<int> word_counts(static_cast<std::size_t>(size));
    if (MPI_Allgather(&local_word_count, 1, MPI_INT, word_counts.data(), 1, MPI_INT, communicator) != MPI_SUCCESS) {
        return invalid(AMIO_ERR_COMM_SPLIT_FAILED);
    }
    std::vector<int> displacements(static_cast<std::size_t>(size));
    std::size_t total_words = 0;
    for (int member = 0; member < size; ++member) {
        if (word_counts[member] < 0 || static_cast<std::size_t>(word_counts[member]) % words_per_box != 0 ||
            total_words > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return invalid(AMIO_ERR_INVALID_INPUT);
        displacements[member] = static_cast<int>(total_words);
        total_words += static_cast<std::size_t>(word_counts[member]);
    }
    const auto global_descriptors = total_words / words_per_box;
    if (global_descriptors > common_limits[1] || total_words > common_limits[2] / sizeof(unsigned long long) ||
        total_words > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return invalid(AMIO_ERR_BUDGET_EXCEEDED);

    std::vector<unsigned long long> gathered(total_words);
    if (MPI_Allgatherv(local_data.data(), local_word_count, MPI_UNSIGNED_LONG_LONG, gathered.data(), word_counts.data(), displacements.data(),
                       MPI_UNSIGNED_LONG_LONG, communicator) != MPI_SUCCESS)
        return invalid(AMIO_ERR_COMM_SPLIT_FAILED);

    std::vector<std::uint64_t> shape(min_shape.begin(), min_shape.end());
    std::vector<PartitionBox> boxes;
    boxes.reserve(global_descriptors);
    for (int member = 0; member < size; ++member) {
        const auto start = static_cast<std::size_t>(displacements[member]);
        const auto count = static_cast<std::size_t>(word_counts[member]);
        for (std::size_t offset = 0; offset < count; offset += words_per_box) {
            PartitionBox box;
            const auto index = start + offset;
            if (gathered[index] != static_cast<unsigned long long>(member) || gathered[index + 1] > 1) return invalid(AMIO_ERR_PARTITION_INVALID);
            box.producer_rank = member;
            box.empty = gathered[index + 1] != 0;
            box.offsets.resize(global_extents.size());
            box.extents.resize(global_extents.size());
            for (std::size_t dimension = 0; dimension < global_extents.size(); ++dimension) {
                box.offsets[dimension] = static_cast<std::uint64_t>(gathered[index + 2 + dimension]);
                box.extents[dimension] = static_cast<std::uint64_t>(gathered[index + 2 + global_extents.size() + dimension]);
            }
            boxes.push_back(std::move(box));
        }
    }
    return validate(shape, boxes, strict, static_cast<std::size_t>(common_limits[0]), static_cast<std::size_t>(common_limits[1]));
}
#endif

}  // namespace amio::detail
