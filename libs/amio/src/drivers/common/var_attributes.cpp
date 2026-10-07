// var_attributes.cpp -- implementation of the CF/UGRID attribute model.
//
// See var_attributes.hpp for the contract.  The manifest parsing uses
// HELM::CONF's typed accessors and dotted-path API to walk the
// configuration tree.

#include "drivers/common/var_attributes.hpp"

#include <cctype>
#include <conf/config.hpp>
#include <conf/value.hpp>
#include <cstdlib>
#include <string>

namespace amio::detail {

// ---------------------------------------------------------------
// parse_attr_value -- detect numeric literals; keep original text.
// ---------------------------------------------------------------

AttrValue parse_attr_value(const std::string &raw) {
    AttrValue v;
    v.text = raw;
    v.is_numeric = false;
    v.is_integer = false;

    if (raw.empty()) {
        return v;
    }

    // Attempt a full-string numeric parse via std::strtod.  Only treat
    // the value as numeric when the entire (trimmed) string is consumed.
    const char *begin = raw.c_str();
    char *end = nullptr;
    errno = 0;
    double d = std::strtod(begin, &end);

    // Skip trailing whitespace after the parsed number.
    while (end != nullptr && *end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }

    if (end != nullptr && *end == '\0' && end != begin) {
        v.is_numeric = true;
        v.number = d;
        // Integer iff no decimal point / exponent appears in the text.
        v.is_integer = raw.find('.') == std::string::npos && raw.find('e') == std::string::npos && raw.find('E') == std::string::npos;
    }

    return v;
}

namespace {

// Detect a UGRID role in a variable's attribute set.
bool has_ugrid_role(const VarAttributes &attrs) {
    for (const auto &kv : attrs.items) {
        if (kv.first == "cf_role" || kv.first == "topology_dimension" || kv.first == "mesh_topology") {
            return true;
        }
    }
    return false;
}

}  // namespace

DatasetAttributes parse_dataset_attributes(const conf::Config &config) {
    DatasetAttributes out;

    // Global extra attributes: read known attribute keys under
    // "global_attributes.*".  CONF doesn't expose keys() iteration
    // over a sub-map directly, so we check for commonly used global
    // attribute keys.  If the manifest uses a flat structure, we can
    // read them by dotted path.
    //
    // For a fully dynamic approach, the attribute parsing would need
    // CONF's Value/node iteration API.  For now, we attempt to read
    // common CF global attributes via try_string.
    if (config.has("global_attributes")) {
        // Read known global attribute keys that CF mandates or commonly uses.
        static const char *known_global_keys[] = {
            "title", "institution", "source", "history", "references", "comment", "Conventions", "contact", "project",
            // ACDD global attributes
            "keywords", "summary", "acknowledgement", "id", "license", "creator_name", "creator_url", "creator_email", "publisher_name",
            "publisher_url", "publisher_email", "geospatial_bounds", "geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min",
            "geospatial_lon_max", "geospatial_vertical_min", "geospatial_vertical_max", "geospatial_vertical_positive", "geospatial_bounds_crs",
            "geospatial_bounds_vertical_crs", "naming_authority", "project", "processing_level", "standard_name_vocabulary", "time_coverage_start",
            "time_coverage_end", "time_coverage_duration", "time_coverage_resolution", "date_created", "date_modified", "date_metadata_modified",
            // NCEI template attributes
            "cdm_data_type", "featureType", "ncei_template_version", "uuid",
            // Used by CECE's standalone writer
            "gridspec_file"};
        for (const char *key : known_global_keys) {
            std::string dotted = std::string("global_attributes.") + key;
            auto val = config.try_string(dotted);
            if (val.has_value()) {
                out.global.set(key, parse_attr_value(*val));
            }
        }
    }

    // Per-variable attributes: enumerate the `variables` map directly. The
    // manifest is the source of truth for which variables exist, so its keys
    // are the variable names. A `variable_names` string list is also honoured
    // when present, so a manifest that cannot be introspected as a map still
    // works.
    std::vector<std::string> var_names;
    if (config.is_map("variables")) {
        var_names = config.at("variables").keys();
    } else if (config.has("variable_names")) {
        try {
            var_names = config.get_string_list("variable_names");
        } catch (...) {
            // Malformed block -- ignore; per-variable map stays empty.
        }
    }

    for (const std::string &var_name : var_names) {
        std::string prefix = "variables." + var_name + ".attributes";
        if (!config.has(prefix)) {
            continue;
        }
        VarAttributes attrs;
        // Read known CF/UGRID per-variable attribute keys.
        static const char *known_var_keys[] = {
            "units",    "long_name",          "standard_name",        "_FillValue", "coordinates", "cell_methods", "cf_role",     "mesh",
            "location", "topology_dimension", "scale_factor",         "add_offset", "valid_min",   "valid_max",    "valid_range", "bounds",
            "axis",     "grid_mapping",       "coverage_content_type"};
        for (const char *key : known_var_keys) {
            std::string dotted = prefix + "." + key;
            auto val = config.try_string(dotted);
            if (val.has_value()) {
                if (std::string(key) == "units") {
                    // CF units are text, including the dimensionless unit "1".
                    AttrValue units;
                    units.text = *val;
                    units.is_numeric = false;
                    units.is_integer = false;
                    attrs.set(key, units);
                } else {
                    attrs.set(key, parse_attr_value(*val));
                }
            }
        }
        if (has_ugrid_role(attrs)) {
            out.uses_ugrid = true;
        }
        if (!attrs.empty()) {
            out.per_variable.emplace(var_name, std::move(attrs));
        }
    }

    // Resolve the Conventions string: explicit manifest value wins;
    // otherwise default to CF, upgraded to CF+UGRID when a mesh role
    // was declared.
    if (config.has("conventions")) {
        out.conventions = config.get_or<std::string>("conventions", kDefaultCFConventions);
    } else {
        out.conventions = out.uses_ugrid ? kDefaultCFUGRIDConventions : kDefaultCFConventions;
    }

    return out;
}

}  // namespace amio::detail
