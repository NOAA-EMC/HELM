// var_attributes.hpp -- CF / UGRID convention metadata and per-variable
// attribute model shared by the netCDF-c-based backend drivers.
//
// This header is PRIVATE to the AMIO_Core build (`src/drivers/common/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// AMIO writes self-describing, convention-compliant datasets:
//
//   * NetCDF-4 and Zarr (NCZarr / TensorStore) output carries a global
//     `Conventions` attribute (default "CF-1.10"; "CF-1.10 UGRID-1.0"
//     when an unstructured mesh is described).
//   * Each variable may carry CF attributes (units, long_name,
//     standard_name, _FillValue, coordinates, ...) and, for
//     unstructured grids, UGRID attributes (cf_role, mesh,
//     location, topology_dimension, ...).
//
// Attributes are declared in the dataset manifest under:
//
//   conventions: "CF-1.10"                 # optional global override
//   global_attributes:                     # optional extra globals
//     title: "..."
//     institution: "..."
//   variables:
//     <var_name>:
//       attributes:
//         units: "K"
//         standard_name: "air_temperature"
//         _FillValue: -9999.0
//
// The drivers translate this model into native attribute calls
// (nc_put_att_* for netCDF/NCZarr; JSON metadata for TensorStore).
//
// Typing
// ------
// Attribute values are captured both as their original string form and
// as a parsed numeric form when the string parses as a number.  This
// lets a driver emit `_FillValue` / `scale_factor` / `add_offset` as a
// numeric netCDF attribute (matching the variable's type where it
// matters) while still supporting free-form text attributes such as
// `units` or `standard_name`.

#ifndef AMIO_SRC_DRIVERS_COMMON_VAR_ATTRIBUTES_HPP
#define AMIO_SRC_DRIVERS_COMMON_VAR_ATTRIBUTES_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace conf {
class Config;
}  // namespace conf

namespace amio::detail {

// Default convention strings.
inline constexpr const char *kDefaultCFConventions = "CF-1.10";
inline constexpr const char *kDefaultCFUGRIDConventions = "CF-1.10 UGRID-1.0";

// AttrValue -- a single attribute value.
//
// `text` always holds the original string.  When the string parsed as
// a number, `is_numeric` is true and `number` holds the value and
// `is_integer` records whether it was an integral literal (no '.',
// 'e', or 'E').  Drivers may use the numeric form to emit a typed
// attribute, or fall back to text.
struct AttrValue {
    std::string text;
    double number = 0.0;
    bool is_numeric = false;
    bool is_integer = false;
};

// VarAttributes -- ordered attribute set for one variable.
//
// Insertion order is preserved so emitted files have stable, readable
// attribute ordering (CF attributes appear in manifest order).
struct VarAttributes {
    std::vector<std::pair<std::string, AttrValue>> items;

    void set(const std::string &key, const AttrValue &value) {
        for (auto &kv : items) {
            if (kv.first == key) {
                kv.second = value;
                return;
            }
        }
        items.emplace_back(key, value);
    }

    bool empty() const noexcept {
        return items.empty();
    }
};

// DatasetAttributes -- the full attribute model parsed from a manifest.
struct DatasetAttributes {
    // Global `Conventions` string (defaults applied by the parser).
    std::string conventions = kDefaultCFConventions;

    // Extra global attributes (title, institution, source, ...).
    VarAttributes global;

    // Per-variable attribute sets, keyed by variable name.
    std::map<std::string, VarAttributes> per_variable;

    // True when any variable declares a UGRID role (cf_role /
    // topology_dimension), which upgrades the default Conventions to
    // include "UGRID-1.0" unless explicitly overridden.
    bool uses_ugrid = false;

    // Look up a variable's attributes; returns nullptr when none were
    // declared for `var_name`.
    const VarAttributes *find(const std::string &var_name) const {
        auto it = per_variable.find(var_name);
        return it == per_variable.end() ? nullptr : &it->second;
    }
};

// Parse a single scalar string into an AttrValue, detecting numeric
// literals.  Always succeeds; non-numeric strings become text-only.
AttrValue parse_attr_value(const std::string &raw);

// Parse the dataset attribute model from a conf::Config.
//
// Reads `conventions`, `global_attributes.*`, and
// `variables.<name>.attributes.*`.  Applies the CF (or CF+UGRID)
// default to `conventions` when the manifest does not set it
// explicitly.
DatasetAttributes parse_dataset_attributes(const conf::Config &config);

}  // namespace amio::detail

#endif  // AMIO_SRC_DRIVERS_COMMON_VAR_ATTRIBUTES_HPP
