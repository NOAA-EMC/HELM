#ifndef AMIO_SRC_STEPS_CONTRIBUTION_PLAN_HPP
#define AMIO_SRC_STEPS_CONTRIBUTION_PLAN_HPP

#include "amio/amio_errors.h"
#include "schema/schema.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace amio::detail {

struct RegionSelection {
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> extents;
    bool empty = false;
};

struct ContributionDeclaration {
    std::string variable;
    std::size_t snapshot_bytes = 0;
    bool absent = false;
    RegionSelection region;
};

/** Immutable per-step admission plan, built before any source buffer is read. */
class ContributionPlan {
   public:
    [[nodiscard]] amio_status_t configure(const Schema &schema, const std::vector<ContributionDeclaration> &declarations,
                                          std::size_t reserved_bytes, bool empty_contributor,
                                          bool defer_required_field_validation = false);
    [[nodiscard]] std::optional<std::size_t> expected_bytes(const std::string &variable) const noexcept;
    [[nodiscard]] const RegionSelection *region(const std::string &variable) const noexcept;
    [[nodiscard]] bool explicitly_absent(const std::string &variable) const noexcept;
    [[nodiscard]] bool mark_present(const std::string &variable) noexcept;
    [[nodiscard]] bool required_fields_present() const noexcept;
    void reset() noexcept;

   private:
    struct Entry {
        std::size_t bytes = 0;
        bool absent = false;
        bool present = false;
        bool required = false;
        RegionSelection region;
    };
    std::map<std::string, Entry, std::less<>> entries_;
    bool empty_contributor_ = false;
    bool defer_required_field_validation_ = false;
};

} // namespace amio::detail

#endif // AMIO_SRC_STEPS_CONTRIBUTION_PLAN_HPP
