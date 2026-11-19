#include "steps/contribution_plan.hpp"

#include <limits>

namespace amio::detail {
namespace {
std::size_t element_size(amio_dtype_t dtype) noexcept {
    switch (dtype) {
        case AMIO_DTYPE_F32:
        case AMIO_DTYPE_I32:
        case AMIO_DTYPE_U32:
            return 4;
        case AMIO_DTYPE_F64:
        case AMIO_DTYPE_I64:
        case AMIO_DTYPE_U64:
            return 8;
        case AMIO_DTYPE_I16:
        case AMIO_DTYPE_U16:
            return 2;
        case AMIO_DTYPE_I8:
        case AMIO_DTYPE_U8:
            return 1;
        default:
            return 0;
    }
}

bool region_size(const Variable &variable, const RegionSelection &region, std::size_t &bytes) noexcept {
    if (region.empty || region.offsets.size() != variable.dimensions.size() || region.extents.size() != variable.dimensions.size()) return false;
    std::size_t elements = 1;
    for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
        const auto offset = region.offsets[d];
        const auto extent = region.extents[d];
        const auto global = variable.dimensions[d].extent;
        if (extent == 0 || offset > global || extent > global - offset || extent > std::numeric_limits<std::size_t>::max() / elements) return false;
        elements *= static_cast<std::size_t>(extent);
    }
    const auto size = element_size(variable.source_dtype);
    if (size == 0 || size > std::numeric_limits<std::size_t>::max() / elements) return false;
    bytes = elements * size;
    return true;
}
}  // namespace

amio_status_t ContributionPlan::configure(const Schema &schema, const std::vector<ContributionDeclaration> &declarations, std::size_t reserved_bytes,
                                          bool empty_contributor, bool defer_required_field_validation) {
    entries_.clear();
    empty_contributor_ = empty_contributor;
    defer_required_field_validation_ = defer_required_field_validation;
    std::size_t required_bytes = 0;
    for (const auto &declaration : declarations) {
        const auto variable = schema.variables().find(declaration.variable);
        if (variable == schema.variables().end() || entries_.find(declaration.variable) != entries_.end()) return AMIO_ERR_INVALID_POLICY;
        if (declaration.absent) {
            if ((variable->second.required && !empty_contributor && !defer_required_field_validation) || declaration.snapshot_bytes != 0)
                return AMIO_ERR_INVALID_POLICY;
        } else {
            if (empty_contributor) return AMIO_ERR_INVALID_POLICY;
            std::size_t declared_region_size = 0;
            if (!region_size(variable->second, declaration.region, declared_region_size)) return AMIO_ERR_INVALID_INPUT;
            if (declaration.snapshot_bytes != declared_region_size || required_bytes > reserved_bytes ||
                declaration.snapshot_bytes > reserved_bytes - required_bytes) {
                return AMIO_ERR_BUDGET_EXCEEDED;
            }
            required_bytes += declaration.snapshot_bytes;
        }
        entries_.emplace(declaration.variable,
                         Entry{declaration.snapshot_bytes, declaration.absent, false, variable->second.required, declaration.region});
    }
    if (entries_.size() != schema.variables().size()) return AMIO_ERR_INVALID_POLICY;
    return AMIO_OK;
}

const RegionSelection *ContributionPlan::region(const std::string &variable) const noexcept {
    const auto found = entries_.find(variable);
    return found == entries_.end() ? nullptr : &found->second.region;
}

std::optional<std::size_t> ContributionPlan::expected_bytes(const std::string &variable) const noexcept {
    const auto found = entries_.find(variable);
    if (found == entries_.end() || found->second.absent) return std::nullopt;
    return found->second.bytes;
}

bool ContributionPlan::explicitly_absent(const std::string &variable) const noexcept {
    const auto found = entries_.find(variable);
    return found != entries_.end() && found->second.absent;
}

bool ContributionPlan::mark_present(const std::string &variable) noexcept {
    const auto found = entries_.find(variable);
    if (found == entries_.end() || found->second.absent || found->second.present) return false;
    found->second.present = true;
    return true;
}

bool ContributionPlan::required_fields_present() const noexcept {
    if (empty_contributor_ || defer_required_field_validation_) return true;
    for (const auto &[name, entry] : entries_) {
        (void)name;
        if (entry.required && !entry.present) return false;
    }
    return true;
}

void ContributionPlan::reset() noexcept {
    entries_.clear();
    empty_contributor_ = false;
    defer_required_field_validation_ = false;
}

}  // namespace amio::detail
