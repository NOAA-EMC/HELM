// grib2_driver.cpp -- AMIO GRIB2_Driver implementation.
//
// Implements the GRIB2 backend driver using nceplibs-g2c for encoding
// and decoding GRIB2 records.  Registers with BackendFactory under
// key "grib2" at static initialization time.
//
// Table sourcing
// --------------
// All GRIB2 code/template tables come from NCEPLIBS-g2c itself.  The
// encoding path calls the legacy g2c packing API:
//
//   g2_create()  -> Sections 0 & 1 (indicator + identification)
//   g2_addgrid() -> Section 3 (grid definition); g2c looks up the
//                   Grid Definition Template layout in its NCEP table
//                   via getgridtemplate()
//   g2_addfield()-> Sections 4/5/6/7 (product, data representation,
//                   bitmap, data); g2c looks up the Product- and Data-
//                   Representation-Template layouts via getpdstemplate()
//                   / getdrstemplate()
//   g2_gribend() -> Section 8 (end)
//
// AMIO supplies only the *numeric* code-table values (discipline,
// parameter category/number, template numbers, fixed-surface
// descriptors) from the manifest.  Those values ARE the WMO/NCEP
// table entries and are forwarded to g2c verbatim -- there is no
// external string->code map anymore.
//
// Conditional compilation:
//   - AMIO_HAS_G2C: when defined, uses the real g2c API.
//   - When g2c is not defined, the driver compiles but throws on
//     construction indicating g2c is not available.
//
// Validates: R9.1, R9.2, R9.3, R9.4, R9.5, R9.6, R9.7, R9.8

#include "drivers/grib2/grib2_driver.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"

#ifdef AMIO_HAS_G2C
// g2c's grib2.h has no extern "C" guard, so wrap it.  Pull in its
// standard C header dependencies first (outside the extern "C" block)
// so their include guards keep them from being processed under C
// linkage.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
extern "C" {
#include <grib2.h>  // nceplibs-g2c public header (g2int, g2_create, ...)
}
#endif

#include <conf/config.hpp>
#include <conf/error.hpp>

namespace amio::detail {

// ---------------------------------------------------------------
// Static factory registration
// ---------------------------------------------------------------

namespace {
// Register GRIB2_Driver with BackendFactory under key "grib2" at
// static initialization time (R4.2, R4.5).
BackendRegistrar<GRIB2_Driver> grib2_registrar("grib2");

// Read a single int64 manifest field, falling back to `def` when the
// key is absent.  Centralizes the integer getter pattern for the CONF
// configuration interface.
std::int64_t cfg_int(const conf::Config &config, const std::string &key, std::int64_t def) {
    if (!config.has(key)) {
        return def;
    }
    return static_cast<std::int64_t>(config.get_or<int>(key, static_cast<int>(def)));
}
}  // anonymous namespace

// ---------------------------------------------------------------
// DRT name parsing
// ---------------------------------------------------------------

GRIB2_DRT parse_drt_name(const std::string &name) {
    // Normalize to lowercase for comparison.
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    if (lower == "adaptive_entropy_coding" || lower == "libaec" || lower == "adaptive entropy coding via libaec" ||
        lower == "adaptive entropy coding") {
        return GRIB2_DRT::AdaptiveEntropyCoding;
    }
    if (lower == "lossless_jpeg2000" || lower == "jpeg2000" || lower == "lossless jpeg2000") {
        return GRIB2_DRT::LosslessJPEG2000;
    }

    // Not in the allowed set -- this is an unrecognized DRT (R9.6, R9.7).
    throw std::runtime_error("GRIB2_Driver: unrecognized Data Representation Template: '" + name +
                             "'. DRT name is not recognized. "
                             "Allowed values: {Adaptive Entropy Coding via libaec, "
                             "Lossless JPEG2000}. Zero record bytes emitted.");
}

// ---------------------------------------------------------------
// GRIB2_Driver construction / destruction
// ---------------------------------------------------------------

GRIB2_Driver::GRIB2_Driver() {
#ifndef AMIO_HAS_G2C
    // g2c is not available in this build.  Throw on construction
    // so the factory knows this driver cannot be used.
    throw std::runtime_error(
        "GRIB2_Driver: nceplibs-g2c is not available in this build. "
        "Rebuild AMIO with AMIO_HAS_G2C=ON to use the GRIB2 backend.");
#endif
}

GRIB2_Driver::~GRIB2_Driver() {
    if (initialized_) {
        // Best-effort cleanup; close() should have been called.
        try {
            close();
        } catch (...) {
            // Suppress exceptions in destructor.
        }
    }
}

// ---------------------------------------------------------------
// read_settings -- pull the numeric GRIB2 identifiers from manifest
// ---------------------------------------------------------------

Grib2Settings GRIB2_Driver::read_settings(const conf::Config &config) {
    Grib2Settings s{};  // start from NCEP-flavored defaults

    s.discipline = cfg_int(config, "grib2.discipline", s.discipline);
    s.center = cfg_int(config, "grib2.center", s.center);
    s.subcenter = cfg_int(config, "grib2.subcenter", s.subcenter);
    s.master_table_version = cfg_int(config, "grib2.master_table_version", s.master_table_version);
    s.local_table_version = cfg_int(config, "grib2.local_table_version", s.local_table_version);
    s.significance_of_ref_time = cfg_int(config, "grib2.significance_of_ref_time", s.significance_of_ref_time);
    s.production_status = cfg_int(config, "grib2.production_status", s.production_status);
    s.type_of_data = cfg_int(config, "grib2.type_of_data", s.type_of_data);

    s.gdt_number = cfg_int(config, "grib2.gdt_number", s.gdt_number);
    s.lat_first = cfg_int(config, "grib2.lat_first", s.lat_first);
    s.lon_first = cfg_int(config, "grib2.lon_first", s.lon_first);
    s.lat_last = cfg_int(config, "grib2.lat_last", s.lat_last);
    s.lon_last = cfg_int(config, "grib2.lon_last", s.lon_last);

    s.pdt_number = cfg_int(config, "grib2.pdt_number", s.pdt_number);
    // Accept both the precise names and the short aliases used in the
    // example manifests ("category"/"parameter").
    s.parameter_category = cfg_int(config, "grib2.parameter_category", s.parameter_category);
    s.parameter_category = cfg_int(config, "grib2.category", s.parameter_category);
    s.parameter_number = cfg_int(config, "grib2.parameter_number", s.parameter_number);
    s.parameter_number = cfg_int(config, "grib2.parameter", s.parameter_number);
    s.type_of_first_fixed_surface = cfg_int(config, "grib2.type_of_first_fixed_surface", s.type_of_first_fixed_surface);
    s.scale_factor_first_surface = cfg_int(config, "grib2.scale_factor_first_surface", s.scale_factor_first_surface);
    s.scaled_value_first_surface = cfg_int(config, "grib2.scaled_value_first_surface", s.scaled_value_first_surface);
    s.forecast_time = cfg_int(config, "grib2.forecast_time", s.forecast_time);
    s.indicator_of_unit_of_time = cfg_int(config, "grib2.indicator_of_unit_of_time", s.indicator_of_unit_of_time);

    s.decimal_scale_factor = cfg_int(config, "grib2.decimal_scale_factor", s.decimal_scale_factor);

    // Composition-specific metadata (PDTs 4.8, 4.40, 4.44–4.49, GDT 3.40).
    // All default to zero when absent (Req 1.9).
    s.chemical_constituent_type = cfg_int(config, "grib2.chemical_constituent_type", 0);
    s.aerosol_type = cfg_int(config, "grib2.aerosol_type", 0);
    s.size_dist_param_first = cfg_int(config, "grib2.size_dist_param_first", 0);
    s.size_dist_param_second = cfg_int(config, "grib2.size_dist_param_second", 0);
    s.optical_property_type = cfg_int(config, "grib2.optical_property_type", 0);
    s.wavelength_first_nm = cfg_int(config, "grib2.wavelength_first_nm", 0);
    s.wavelength_last_nm = cfg_int(config, "grib2.wavelength_last_nm", 0);
    s.ensemble_perturbation_number = cfg_int(config, "grib2.ensemble_perturbation_number", 0);
    s.statistical_process = cfg_int(config, "grib2.statistical_process", 0);
    s.time_range_unit = cfg_int(config, "grib2.time_range_unit", 0);
    s.time_range_length = cfg_int(config, "grib2.time_range_length", 0);
    s.number_of_time_range_specs = cfg_int(config, "grib2.number_of_time_range_specs", 0);
    s.total_missing_from_statistical_process = cfg_int(config, "grib2.total_missing_from_statistical_process", 0);
    s.n_parallel = cfg_int(config, "grib2.n_parallel", 0);

    return s;
}

// ---------------------------------------------------------------
// open_write / open_read
// ---------------------------------------------------------------

void GRIB2_Driver::open_write(const conf::Config &config) {
    active_drt_ = validate_drt(config);
    initialize(config);
}

void GRIB2_Driver::open_read(const conf::Config &config) {
#ifndef AMIO_HAS_G2C
    // g2c is not available in this build.  A read open must fail so the
    // C-boundary cordon translates the throw to AMIO_ERR_BACKEND_FAILURE
    // (Req 13.7).
    (void)config;
    throw std::runtime_error(
        "GRIB2_Driver::open_read: nceplibs-g2c is not available in this build. "
        "Rebuild AMIO with AMIO_HAS_G2C=ON to use the GRIB2 read path.");
#else
    if (initialized_) {
        return;  // Already initialized.
    }

    // Default numeric product identifiers from the manifest.  These act
    // as fallbacks and document the encode settings; the record index is
    // built from the descriptors carried in the file itself.
    settings_ = read_settings(config);

    // Resolve the input path.  Accept the same key aliases the rest of
    // the toolchain emits (path / input_path / output_path).
    if (config.has("path")) {
        input_path_ = config.get_string("path");
    } else if (config.has("input_path")) {
        input_path_ = config.get_string("input_path");
    } else if (config.has("output_path")) {
        input_path_ = config.get_string("output_path");
    }
    if (input_path_.empty()) {
        throw std::runtime_error("GRIB2_Driver::open_read: 'path' (or 'input_path') field is required.");
    }

    // Scan the file and build the in-memory record index (Req 13.1).
    build_record_index(config);

    read_mode_ = true;
    initialized_ = true;
#endif  // AMIO_HAS_G2C
}

// ---------------------------------------------------------------
// field_identity_name -- variable-name <-> field-identity convention
//
// GRIB2 records carry no variable names, only WMO numeric descriptors.
// AMIO synthesizes a stable name from the same identifiers the encode
// path writes (discipline + PDT parameter category/number + first fixed
// surface type/value), so a write-then-read round trip resolves to the
// same key.  See the header for the full rationale.
// ---------------------------------------------------------------

std::string GRIB2_Driver::field_identity_name(std::int64_t discipline, std::int64_t parameter_category, std::int64_t parameter_number,
                                              std::int64_t surface_type, std::int64_t surface_value) {
    return "d" + std::to_string(discipline) + "_c" + std::to_string(parameter_category) + "_n" + std::to_string(parameter_number) + "_s" +
           std::to_string(surface_type) + "_l" + std::to_string(surface_value);
}

// ---------------------------------------------------------------
// field_identity_name (extended) -- composition-aware overload
//
// Builds the base name then appends PDT-specific composition suffixes.
// Uses pdt_number to determine which segments to include, ensuring
// PDT 4.0 records never accidentally gain segments (Req 12.7, 15.1–15.4).
// ---------------------------------------------------------------

std::string GRIB2_Driver::field_identity_name(std::int64_t discipline, std::int64_t parameter_category, std::int64_t parameter_number,
                                              std::int64_t surface_type, std::int64_t surface_value, std::int64_t pdt_number,
                                              std::int64_t chemical_constituent_type, std::int64_t aerosol_type, std::int64_t optical_property_type,
                                              std::int64_t wavelength_first_nm, std::int64_t wavelength_last_nm,
                                              std::int64_t ensemble_perturbation_number, std::int64_t statistical_process) {
    // Base name: "d{d}_c{c}_n{n}_s{s}_l{l}" — identical to the original overload.
    std::string name = field_identity_name(discipline, parameter_category, parameter_number, surface_type, surface_value);

    // Append composition suffixes based on PDT number.
    switch (pdt_number) {
        case 40:
            // PDT 4.40: chemical constituent
            name += "_ct" + std::to_string(chemical_constituent_type);
            break;
        case 44:
            // PDT 4.44: aerosol at a point in time
            name += "_at" + std::to_string(aerosol_type);
            break;
        case 45:
            // PDT 4.45: ensemble aerosol forecast
            name += "_at" + std::to_string(aerosol_type);
            name += "_ep" + std::to_string(ensemble_perturbation_number);
            break;
        case 46:
            // PDT 4.46: statistically processed aerosol
            name += "_at" + std::to_string(aerosol_type);
            name += "_sp" + std::to_string(statistical_process);
            break;
        case 48:
            // PDT 4.48: aerosol optical properties at a wavelength
            name += "_at" + std::to_string(aerosol_type);
            name += "_op" + std::to_string(optical_property_type);
            name += "_wl" + std::to_string(wavelength_first_nm) + "_" + std::to_string(wavelength_last_nm);
            break;
        case 49:
            // PDT 4.49: ensemble aerosol optical properties
            name += "_at" + std::to_string(aerosol_type);
            name += "_op" + std::to_string(optical_property_type);
            name += "_wl" + std::to_string(wavelength_first_nm) + "_" + std::to_string(wavelength_last_nm);
            name += "_ep" + std::to_string(ensemble_perturbation_number);
            break;
        case 8:
            // PDT 4.8: statistically processed at a level
            name += "_sp" + std::to_string(statistical_process);
            break;
        case 0:
        default:
            // PDT 4.0 or unknown: no suffix (backward compatible).
            break;
    }

    return name;
}

// ---------------------------------------------------------------
// extract_composition_metadata -- extract composition-specific
// metadata from a PDT template array based on PDT number.
//
// Uses known PDT-specific index positions (WMO template layouts) to
// extract the relevant fields.  For unsupported or PDT 4.0 records,
// all composition fields remain zero — producing the backward-
// compatible identity.  (Req 13.1, 13.2, 13.3, 13.4, 13.5, 13.6)
// ---------------------------------------------------------------

CompositionMetadata GRIB2_Driver::extract_composition_metadata(std::int64_t pdt_number, const std::vector<std::int64_t> &ipdtmpl) {
    CompositionMetadata meta{};
    meta.pdt_number = pdt_number;

    // Helper to safely access an element, returning 0 if out of range.
    auto safe_get = [&](std::size_t idx) -> std::int64_t { return idx < ipdtmpl.size() ? ipdtmpl[idx] : 0; };

    switch (pdt_number) {
        case 40:
            // PDT 4.40: chemical_constituent_type at index 2
            meta.chemical_constituent_type = safe_get(2);
            break;
        case 44:
            // PDT 4.44: aerosol_type at index 2
            meta.aerosol_type = safe_get(2);
            break;
        case 45:
            // PDT 4.45: aerosol_type at index 2,
            //            ensemble_perturbation_number at index 16
            meta.aerosol_type = safe_get(2);
            meta.ensemble_perturbation_number = safe_get(16);
            break;
        case 46:
            // PDT 4.46: aerosol_type at index 2,
            //            statistical_process at index 29
            meta.aerosol_type = safe_get(2);
            meta.statistical_process = safe_get(29);
            break;
        case 48:
            // PDT 4.48: aerosol_type at index 2,
            //            optical_property_type at index 8,
            //            wavelength_first_nm at index 10,
            //            wavelength_last_nm at index 12
            meta.aerosol_type = safe_get(2);
            meta.optical_property_type = safe_get(8);
            meta.wavelength_first_nm = safe_get(10);
            meta.wavelength_last_nm = safe_get(12);
            break;
        case 49:
            // PDT 4.49: aerosol_type at index 2,
            //            optical_property_type at index 8,
            //            wavelength_first_nm at index 10,
            //            wavelength_last_nm at index 12,
            //            ensemble_perturbation_number at index 21
            meta.aerosol_type = safe_get(2);
            meta.optical_property_type = safe_get(8);
            meta.wavelength_first_nm = safe_get(10);
            meta.wavelength_last_nm = safe_get(12);
            meta.ensemble_perturbation_number = safe_get(21);
            break;
        case 8:
            // PDT 4.8: statistical_process at index 23
            meta.statistical_process = safe_get(23);
            break;
        case 0:
        default:
            // PDT 4.0 or unsupported: all-zero metadata (no composition
            // segments — backward compatible identity).
            break;
    }

    return meta;
}

// ---------------------------------------------------------------
// build_record_index -- scan the GRIB2 file and index every field
// (Req 13.1)
//
// Walks the file message-by-message with seekgb(), reads each message
// into memory, enumerates its fields with g2_info(), and for each field
// decodes the metadata-only header with g2_getfld(unpack=0).  From the
// decoded gribfield it derives the field-identity name (discipline +
// PDT category/number + first fixed surface type/value) and the grid
// geometry (Ni/Nj from the GDT, ngrdpts), appending a record location
// keyed by that identity.  Records for the same identity are ordered by
// file position, so records[t] is the field's timestep t.
// ---------------------------------------------------------------

#ifdef AMIO_HAS_G2C
namespace {

// PDT-relative offsets of the descriptors that make up a field identity.
// Category and number are always at indices 0 and 1 for all supported PDTs.
// Surface type/value positions vary by PDT (the composition PDTs insert
// extra fields between the common header and the fixed-surface section).
constexpr int kPdtCategoryIdx = 0;
constexpr int kPdtNumberIdx = 1;

// GDT-relative offsets of Ni / Nj for the templates whose layout matches
// 3.0 (regular lat/lon) -- Ni at index 7, Nj at index 8.  Used as a
// best-effort grid-shape probe; falls back to ngrdpts as a single row.
constexpr int kGdtNiIdx = 7;
constexpr int kGdtNjIdx = 8;

std::int64_t pdt_value(const gribfield *gfld, int idx) {
    if (gfld->ipdtmpl != nullptr && idx < gfld->ipdtlen) {
        return static_cast<std::int64_t>(gfld->ipdtmpl[idx]);
    }
    return 0;
}

// Return the PDT-specific index of the first fixed surface type field.
// Each composition PDT inserts different fields before the surface section.
int pdt_surface_type_index(std::int64_t pdt_number) {
    switch (pdt_number) {
        case 0:
            return 9;  // PDT 4.0:  [9]
        case 8:
            return 9;  // PDT 4.8:  [9]
        case 40:
            return 10;  // PDT 4.40: [10]
        case 44:
            return 15;  // PDT 4.44: [15]
        case 45:
            return 18;  // PDT 4.45: [18]
        case 46:
            return 15;  // PDT 4.46: [15]
        case 48:
            return 20;  // PDT 4.48: [20]
        case 49:
            return 23;  // PDT 4.49: [23]
        default:
            return 9;  // Fallback to PDT 4.0 layout
    }
}

// Return the PDT-specific index of the first fixed surface scaled value.
// Always 2 positions after the surface type index (type, scale_factor, value).
int pdt_surface_value_index(std::int64_t pdt_number) {
    return pdt_surface_type_index(pdt_number) + 2;
}

}  // namespace
#endif  // AMIO_HAS_G2C

void GRIB2_Driver::build_record_index(const conf::Config &config) {
#ifndef AMIO_HAS_G2C
    (void)config;
    throw std::runtime_error("GRIB2_Driver::build_record_index: nceplibs-g2c not available");
#else
    (void)config;
    records_.clear();

    std::FILE *fp = std::fopen(input_path_.c_str(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("GRIB2_Driver::open_read: failed to open input file '" + input_path_ + "' for reading.");
    }

    // RAII-ish guard: ensure the scan handle is closed on every path.
    struct FileGuard {
        std::FILE *f;
        ~FileGuard() {
            if (f != nullptr) std::fclose(f);
        }
    } guard{fp};

    g2int seek_from = 0;
    const g2int mseek = 32000;  // bytes to scan ahead per seekgb call

    while (true) {
        g2int lskip = 0;  // byte offset of the located message
        g2int lgrib = 0;  // length of the located message
        seekgb(fp, seek_from, mseek, &lskip, &lgrib);
        if (lgrib == 0) {
            break;  // no further GRIB2 message found
        }

        // Read the full message into memory.
        std::vector<unsigned char> cgrib(static_cast<std::size_t>(lgrib));
        if (std::fseek(fp, static_cast<long>(lskip), SEEK_SET) != 0) {
            throw std::runtime_error("GRIB2_Driver::open_read: seek failed while indexing '" + input_path_ + "'.");
        }
        std::size_t got = std::fread(cgrib.data(), 1, static_cast<std::size_t>(lgrib), fp);
        if (got != static_cast<std::size_t>(lgrib)) {
            throw std::runtime_error("GRIB2_Driver::open_read: short read while indexing '" + input_path_ + "'.");
        }

        // Enumerate the fields in this message.
        g2int listsec0[3] = {0, 0, 0};
        g2int listsec1[13] = {0};
        g2int numfields = 0;
        g2int numlocal = 0;
        g2int ret = g2_info(cgrib.data(), listsec0, listsec1, &numfields, &numlocal);
        if (ret != 0) {
            throw std::runtime_error("GRIB2_Driver::open_read: g2_info failed (code " + std::to_string(static_cast<long long>(ret)) +
                                     ") while indexing '" + input_path_ + "'.");
        }

        const std::int64_t discipline = static_cast<std::int64_t>(listsec0[0]);

        for (g2int n = 1; n <= numfields; ++n) {
            gribfield *gfld = nullptr;
            // unpack=0, expand=0: metadata only -- fast, no data decode.
            ret = g2_getfld(cgrib.data(), n, 0, 0, &gfld);
            if (ret != 0 || gfld == nullptr) {
                if (gfld != nullptr) {
                    g2_free(gfld);
                }
                throw std::runtime_error("GRIB2_Driver::open_read: g2_getfld (metadata) failed (code " + std::to_string(static_cast<long long>(ret)) +
                                         ") while indexing '" + input_path_ + "'.");
            }

            const std::int64_t category = pdt_value(gfld, kPdtCategoryIdx);
            const std::int64_t number = pdt_value(gfld, kPdtNumberIdx);

            // Get PDT number to determine the correct surface type/value indices.
            const std::int64_t pdt_num = static_cast<std::int64_t>(gfld->ipdtnum);
            const std::int64_t surf_type = pdt_value(gfld, pdt_surface_type_index(pdt_num));
            const std::int64_t surf_value = pdt_value(gfld, pdt_surface_value_index(pdt_num));

            // Convert the PDT template array to std::vector for
            // composition metadata extraction (Req 13.1–13.5).
            std::vector<std::int64_t> ipdtmpl_vec;
            if (gfld->ipdtmpl != nullptr && gfld->ipdtlen > 0) {
                ipdtmpl_vec.reserve(static_cast<std::size_t>(gfld->ipdtlen));
                for (int i = 0; i < gfld->ipdtlen; ++i) {
                    ipdtmpl_vec.push_back(static_cast<std::int64_t>(gfld->ipdtmpl[i]));
                }
            }

            // Extract composition-specific metadata from the PDT template.
            const CompositionMetadata meta = extract_composition_metadata(pdt_num, ipdtmpl_vec);

            // Build the field identity key using the extended overload
            // that includes composition suffixes (Req 12.8, 13.1–13.5).
            const std::string key = field_identity_name(discipline, category, number, surf_type, surf_value, pdt_num, meta.chemical_constituent_type,
                                                        meta.aerosol_type, meta.optical_property_type, meta.wavelength_first_nm,
                                                        meta.wavelength_last_nm, meta.ensemble_perturbation_number, meta.statistical_process);

            // Derive grid geometry.  Ni/Nj come from the GDT for the
            // common lat/lon-style layouts; fall back to a single row of
            // ngrdpts points when the template does not expose them.
            std::int64_t ngrdpts = static_cast<std::int64_t>(gfld->ngrdpts);
            std::int64_t ni = 0;
            std::int64_t nj = 0;
            if (gfld->igdtmpl != nullptr && gfld->igdtlen > kGdtNjIdx) {
                ni = static_cast<std::int64_t>(gfld->igdtmpl[kGdtNiIdx]);
                nj = static_cast<std::int64_t>(gfld->igdtmpl[kGdtNjIdx]);
            }
            if (ni <= 0 || nj <= 0 || (ni * nj) != ngrdpts) {
                // Template did not yield a usable 2D shape; treat the
                // field as a single row of ngrdpts points.
                ni = ngrdpts;
                nj = 1;
            }

            GribFieldIndex &entry = records_[key];
            if (entry.records.empty()) {
                entry.ni = ni;
                entry.nj = nj;
                entry.ngrdpts = ngrdpts;
            }
            entry.records.push_back(
                GribRecordLocation{static_cast<std::int64_t>(lskip), static_cast<std::int64_t>(lgrib), static_cast<std::int64_t>(n)});

            g2_free(gfld);
        }

        // Advance past this message for the next seekgb scan.
        seek_from = lskip + lgrib;
    }
#endif  // AMIO_HAS_G2C
}

// ---------------------------------------------------------------
// describe_variable -- report a field's dtype/shape/timestep count
// from the record index (Req 4.1, 4.2, 4.5, 13.x)
// ---------------------------------------------------------------

VariableInfo GRIB2_Driver::describe_variable(const std::string &name) {
    VariableInfo info{};  // found == false by default.

#ifdef AMIO_HAS_G2C
    if (!initialized_ || !read_mode_) {
        return info;
    }

    auto it = records_.find(name);
    if (it == records_.end() || it->second.records.empty()) {
        return info;  // Unknown variable.
    }

    const GribFieldIndex &entry = it->second;

    // GRIB2 grid-point fields are delivered as F32 (Req 13.3).
    info.dtype = AMIO_DTYPE_F32;

    // Shape: Nj (points along a meridian -> slowest) x Ni (points along a
    // parallel -> fastest), matching the encode path's row-major layout.
    info.shape = amio_shape_t{};
    if (entry.nj > 1) {
        info.shape.rank = 2;
        info.shape.extents[0] = entry.nj;
        info.shape.extents[1] = entry.ni;
    } else {
        info.shape.rank = 1;
        info.shape.extents[0] = entry.ni;
    }

    // One record per timestep for this field identity (Req 4.5).
    info.total_timesteps = static_cast<std::int64_t>(entry.records.size());
    info.found = true;
#else
    (void)name;
#endif  // AMIO_HAS_G2C

    return info;
}

// ---------------------------------------------------------------
// initialize -- read product identifiers + open output file
// ---------------------------------------------------------------

void GRIB2_Driver::initialize(const conf::Config &config) {
    if (initialized_) {
        return;  // Already initialized.
    }

#ifndef AMIO_HAS_G2C
    (void)config;
    throw std::runtime_error("GRIB2_Driver: nceplibs-g2c is not available in this build.");
#else
    // The GRIB2 tables are provided by g2c itself; AMIO only needs the
    // numeric product identifiers from the manifest.
    settings_ = read_settings(config);

    // Resolve the output path (write mode only).
    if (config.has("path")) {
        output_path_ = config.get_string("path");
    } else if (config.has("output_path")) {
        output_path_ = config.get_string("output_path");
    }

    if (!output_path_.empty()) {
        out_file_ = std::fopen(output_path_.c_str(), "wb");
        if (out_file_ == nullptr) {
            throw std::runtime_error("GRIB2_Driver: failed to open output file '" + output_path_ + "' for writing.");
        }
    }

    initialized_ = true;
#endif  // AMIO_HAS_G2C
}

// ---------------------------------------------------------------
// NCEP template builders
//
// These populate the g2c template-value arrays.  g2c consults its
// own NCEP template tables (getgridtemplate / getpdstemplate /
// getdrstemplate) for the matching layout; we only supply values.
// ---------------------------------------------------------------

std::vector<std::int64_t> GRIB2_Driver::build_gdt_3_0(const Grib2Settings &s, std::int64_t ni, std::int64_t nj) {
    // Grid Definition Template 3.0 (regular lat/lon), 19 entries.
    // Layout per WMO GRIB2 Template 3.0 / NCEP getgridtemplate(0).
    std::vector<std::int64_t> t(19, 0);
    t[0] = 6;             // shape of earth (6 = sphere, radius 6,371,229 m)
    t[1] = 0;             // scale factor of radius of spherical earth
    t[2] = 0;             // scaled value of radius of spherical earth
    t[3] = 0;             // scale factor of major axis
    t[4] = 0;             // scaled value of major axis
    t[5] = 0;             // scale factor of minor axis
    t[6] = 0;             // scaled value of minor axis
    t[7] = ni;            // Ni - number of points along a parallel
    t[8] = nj;            // Nj - number of points along a meridian
    t[9] = 0;             // basic angle of initial production domain
    t[10] = 0;            // subdivisions of basic angle (missing = all 1s -> 0 here)
    t[11] = s.lat_first;  // La1 (1e-6 deg)
    t[12] = s.lon_first;  // Lo1 (1e-6 deg)
    t[13] = 48;           // resolution and component flags (0x30: i,j dir incr given)
    t[14] = s.lat_last;   // La2 (1e-6 deg)
    t[15] = s.lon_last;   // Lo2 (1e-6 deg)
    // Di / Dj direction increments (1e-6 deg), derived from the span.
    std::int64_t di = (ni > 1) ? ((s.lon_last - s.lon_first) / (ni - 1)) : 0;
    std::int64_t dj = (nj > 1) ? ((s.lat_first - s.lat_last) / (nj - 1)) : 0;
    t[16] = di < 0 ? -di : di;  // Di must be positive
    t[17] = dj < 0 ? -dj : dj;  // Dj must be positive
    t[18] = 0;                  // scanning mode (0 = +i, -j, row-major from NW corner)
    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_gdt_3_40(const Grib2Settings &s, std::int64_t ni, std::int64_t nj) {
    // Grid Definition Template 3.40 (Gaussian latitude/longitude), 19 entries.
    // Layout per WMO GRIB2 Template 3.40 / NCEP getgridtemplate(40).
    // Identical to GDT 3.0 except index 17 carries N (number of parallels
    // between equator and pole) instead of Dj.
    std::vector<std::int64_t> t(19, 0);
    t[0] = 6;             // shape of earth (6 = sphere, radius 6,371,229 m)
    t[1] = 0;             // scale factor of radius of spherical earth
    t[2] = 0;             // scaled value of radius of spherical earth
    t[3] = 0;             // scale factor of major axis
    t[4] = 0;             // scaled value of major axis
    t[5] = 0;             // scale factor of minor axis
    t[6] = 0;             // scaled value of minor axis
    t[7] = ni;            // Ni - number of points along a parallel
    t[8] = nj;            // Nj - number of points along a meridian
    t[9] = 0;             // basic angle of initial production domain
    t[10] = 0;            // subdivisions of basic angle
    t[11] = s.lat_first;  // La1 (latitude of first grid point, 1e-6 deg)
    t[12] = s.lon_first;  // Lo1 (longitude of first grid point, 1e-6 deg)
    t[13] = 48;           // resolution and component flags (0x30: i,j dir incr given)
    t[14] = s.lat_last;   // La2 (latitude of last grid point, 1e-6 deg)
    t[15] = s.lon_last;   // Lo2 (longitude of last grid point, 1e-6 deg)
    // Di (i-direction increment, 1e-6 deg), derived from the span.
    std::int64_t di = (ni > 1) ? ((s.lon_last - s.lon_first) / (ni - 1)) : 0;
    t[16] = di < 0 ? -di : di;  // Di must be positive
    t[17] = s.n_parallel;       // N - number of parallels between equator and pole
    t[18] = 0;                  // scanning mode (0 = +i, -j, row-major from NW corner)
    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_0(const Grib2Settings &s) {
    // Product Definition Template 4.0, 15 entries.
    std::vector<std::int64_t> t(15, 0);
    t[0] = s.parameter_category;           // Table 4.1
    t[1] = s.parameter_number;             // Table 4.2
    t[2] = 2;                              // generating process type (2 = forecast)
    t[3] = 0;                              // background generating process id
    t[4] = 0;                              // analysis/forecast generating process id
    t[5] = 0;                              // hours after reference time (data cutoff)
    t[6] = 0;                              // minutes after reference time (data cutoff)
    t[7] = s.indicator_of_unit_of_time;    // Table 4.4
    t[8] = s.forecast_time;                // forecast time in above units
    t[9] = s.type_of_first_fixed_surface;  // Table 4.5
    t[10] = s.scale_factor_first_surface;  // scale factor of first fixed surface
    t[11] = s.scaled_value_first_surface;  // scaled value of first fixed surface
    t[12] = 255;                           // type of second fixed surface (255 = missing)
    t[13] = 0;                             // scale factor of second fixed surface
    t[14] = 0;                             // scaled value of second fixed surface
    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_8(const Grib2Settings &s) {
    // Product Definition Template 4.8 (statistically processed at a
    // horizontal level), 29 entries.
    // Indices 0–14 share the PDT 4.0 layout; indices 15–28 carry
    // end-of-overall-time-interval and statistical processing fields.
    std::vector<std::int64_t> t(29, 0);

    // --- Indices 0–14: same as PDT 4.0 ---
    t[0] = s.parameter_category;           // Table 4.1
    t[1] = s.parameter_number;             // Table 4.2
    t[2] = 2;                              // generating process type (2 = forecast)
    t[3] = 0;                              // background generating process id
    t[4] = 0;                              // analysis/forecast generating process id
    t[5] = 0;                              // hours after reference time (data cutoff)
    t[6] = 0;                              // minutes after reference time (data cutoff)
    t[7] = s.indicator_of_unit_of_time;    // Table 4.4
    t[8] = s.forecast_time;                // forecast time in above units
    t[9] = s.type_of_first_fixed_surface;  // Table 4.5
    t[10] = s.scale_factor_first_surface;  // scale factor of first fixed surface
    t[11] = s.scaled_value_first_surface;  // scaled value of first fixed surface
    t[12] = 255;                           // type of second fixed surface (255 = missing)
    t[13] = 0;                             // scale factor of second fixed surface
    t[14] = 0;                             // scaled value of second fixed surface

    // --- Indices 15–20: end of overall time interval ---
    // These are typically set by the caller or left at zero.
    t[15] = 0;  // year of end of overall time interval
    t[16] = 0;  // month of end
    t[17] = 0;  // day of end
    t[18] = 0;  // hour of end
    t[19] = 0;  // minute of end
    t[20] = 0;  // second of end

    // --- Indices 21–28: statistical processing ---
    t[21] = s.number_of_time_range_specs;              // n
    t[22] = s.total_missing_from_statistical_process;  // total missing
    t[23] = s.statistical_process;                     // Table 4.10
    t[24] = 2;                                         // type of time increment (2 = successive times, same forecast time)
    t[25] = s.time_range_unit;                         // Table 4.4 indicator of unit of time for range
    t[26] = s.time_range_length;                       // time range length
    t[27] = s.indicator_of_unit_of_time;               // indicator of unit of successive fields
    t[28] = 0;                                         // time increment (0 = continuous/contiguous)

    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_40(const Grib2Settings &s) {
    // Product Definition Template 4.40 (chemical constituent at a
    // horizontal level at a point in time), 16 entries.
    // Layout per WMO GRIB2 Template 4.40 / NCEP getpdstemplate(40).
    std::vector<std::int64_t> t(16, 0);
    t[0] = s.parameter_category;            // Table 4.1
    t[1] = s.parameter_number;              // Table 4.2
    t[2] = s.chemical_constituent_type;     // Table 4.230
    t[3] = 2;                               // type of generating process (2 = forecast)
    t[4] = 0;                               // background generating process id
    t[5] = 0;                               // analysis/forecast generating process id
    t[6] = 0;                               // hours after reference time (data cutoff)
    t[7] = 0;                               // minutes after reference time (data cutoff)
    t[8] = s.indicator_of_unit_of_time;     // Table 4.4
    t[9] = s.forecast_time;                 // forecast time in above units
    t[10] = s.type_of_first_fixed_surface;  // Table 4.5
    t[11] = s.scale_factor_first_surface;   // scale factor of first fixed surface
    t[12] = s.scaled_value_first_surface;   // scaled value of first fixed surface
    t[13] = 255;                            // type of second fixed surface (255 = missing)
    t[14] = 0;                              // scale factor of second fixed surface
    t[15] = 0;                              // scaled value of second fixed surface
    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_44(const Grib2Settings &s) {
    // Product Definition Template 4.44 (aerosol at a horizontal level
    // at a point in time), 21 entries.
    // Layout per WMO GRIB2 Template 4.44 / NCEP getpdstemplate(44).
    std::vector<std::int64_t> t(21, 0);
    t[0] = s.parameter_category;            // Table 4.1
    t[1] = s.parameter_number;              // Table 4.2
    t[2] = s.aerosol_type;                  // Table 4.233
    t[3] = 0;                               // type of interval for size distribution (not in Grib2Settings)
    t[4] = 0;                               // scale factor of first size (not in Grib2Settings)
    t[5] = s.size_dist_param_first;         // scaled value of first size
    t[6] = 0;                               // scale factor of second size (not in Grib2Settings)
    t[7] = s.size_dist_param_second;        // scaled value of second size
    t[8] = 2;                               // type of generating process (2 = forecast)
    t[9] = 0;                               // background generating process id
    t[10] = 0;                              // analysis/forecast generating process id
    t[11] = 0;                              // hours after reference time (data cutoff)
    t[12] = 0;                              // minutes after reference time (data cutoff)
    t[13] = s.indicator_of_unit_of_time;    // Table 4.4
    t[14] = s.forecast_time;                // forecast time in above units
    t[15] = s.type_of_first_fixed_surface;  // Table 4.5
    t[16] = s.scale_factor_first_surface;   // scale factor of first fixed surface
    t[17] = s.scaled_value_first_surface;   // scaled value of first fixed surface
    t[18] = 255;                            // type of second fixed surface (255 = missing)
    t[19] = 0;                              // scale factor of second fixed surface
    t[20] = 0;                              // scaled value of second fixed surface
    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_45(const Grib2Settings &s) {
    // Product Definition Template 4.45 (individual ensemble forecast
    // for aerosol), 24 entries.
    // Layout per WMO GRIB2 Template 4.45 / NCEP getpdstemplate(45).
    std::vector<std::int64_t> t(24, 0);

    // --- Indices 0–7: parameter + aerosol / size distribution ---
    t[0] = s.parameter_category;      // Table 4.1
    t[1] = s.parameter_number;        // Table 4.2
    t[2] = s.aerosol_type;            // Table 4.233
    t[3] = 0;                         // type of interval for first size distribution (not in settings)
    t[4] = 0;                         // scale factor of first size
    t[5] = s.size_dist_param_first;   // scaled value of first size
    t[6] = 0;                         // scale factor of second size
    t[7] = s.size_dist_param_second;  // scaled value of second size

    // --- Indices 8–14: generating process + time ---
    t[8] = 2;                             // type of generating process (2 = forecast)
    t[9] = 0;                             // background generating process id
    t[10] = 0;                            // analysis/forecast generating process id
    t[11] = 0;                            // hours after reference time (data cutoff)
    t[12] = 0;                            // minutes after reference time (data cutoff)
    t[13] = s.indicator_of_unit_of_time;  // Table 4.4
    t[14] = s.forecast_time;              // forecast time in above units

    // --- Indices 15–17: ensemble ---
    t[15] = 0;                               // type of ensemble forecast (not in settings)
    t[16] = s.ensemble_perturbation_number;  // perturbation number
    t[17] = 0;                               // number of forecasts in ensemble (not in settings)

    // --- Indices 18–23: fixed surfaces ---
    t[18] = s.type_of_first_fixed_surface;  // Table 4.5
    t[19] = s.scale_factor_first_surface;   // scale factor of first fixed surface
    t[20] = s.scaled_value_first_surface;   // scaled value of first fixed surface
    t[21] = 255;                            // type of second fixed surface (255 = missing)
    t[22] = 0;                              // scale factor of second fixed surface
    t[23] = 0;                              // scaled value of second fixed surface

    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_46(const Grib2Settings &s) {
    // Product Definition Template 4.46 (statistically processed aerosol),
    // 35 entries.
    // Aerosol-specific fields at indices 2-7, common forecast fields at
    // indices 8-14, fixed surfaces at 15-20, end-of-overall-time-interval
    // at 21-26, statistical processing at 27-34.
    std::vector<std::int64_t> t(35, 0);

    // --- Indices 0-1: parameter identification ---
    t[0] = s.parameter_category;  // Table 4.1
    t[1] = s.parameter_number;    // Table 4.2

    // --- Indices 2-7: aerosol-specific fields ---
    t[2] = s.aerosol_type;            // Table 4.233
    t[3] = 0;                         // type of interval for size distribution
    t[4] = 0;                         // scale factor first size
    t[5] = s.size_dist_param_first;   // scaled value first size
    t[6] = 0;                         // scale factor second size
    t[7] = s.size_dist_param_second;  // scaled value second size

    // --- Indices 8-14: generating process + forecast time ---
    t[8] = 2;                             // type of generating process (2 = forecast)
    t[9] = 0;                             // background generating process id
    t[10] = 0;                            // analysis/forecast generating process id
    t[11] = 0;                            // hours after reference time (data cutoff)
    t[12] = 0;                            // minutes after reference time (data cutoff)
    t[13] = s.indicator_of_unit_of_time;  // Table 4.4
    t[14] = s.forecast_time;              // forecast time in above units

    // --- Indices 15-20: fixed surfaces ---
    t[15] = s.type_of_first_fixed_surface;  // Table 4.5
    t[16] = s.scale_factor_first_surface;   // scale factor of first fixed surface
    t[17] = s.scaled_value_first_surface;   // scaled value of first fixed surface
    t[18] = 255;                            // type of second fixed surface (255 = missing)
    t[19] = 0;                              // scale factor of second fixed surface
    t[20] = 0;                              // scaled value of second fixed surface

    // --- Indices 21-26: end of overall time interval ---
    t[21] = 0;  // year of end of overall time interval
    t[22] = 0;  // month of end
    t[23] = 0;  // day of end
    t[24] = 0;  // hour of end
    t[25] = 0;  // minute of end
    t[26] = 0;  // second of end

    // --- Indices 27-34: statistical processing ---
    t[27] = s.number_of_time_range_specs;              // n
    t[28] = s.total_missing_from_statistical_process;  // total missing
    t[29] = s.statistical_process;                     // Table 4.10
    t[30] = 0;                                         // type of time increment
    t[31] = s.time_range_unit;                         // Table 4.4 indicator of unit of time for range
    t[32] = s.time_range_length;                       // time range length
    t[33] = 0;                                         // indicator of unit of successive fields
    t[34] = 0;                                         // time increment

    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_48(const Grib2Settings &s) {
    // Product Definition Template 4.48 (aerosol optical properties at
    // a wavelength), 26 entries.
    // Layout per WMO GRIB2 Template 4.48 / NCEP getpdstemplate(48).
    std::vector<std::int64_t> t(26, 0);

    // --- Indices 0–7: parameter + aerosol / size distribution ---
    t[0] = s.parameter_category;  // Table 4.1
    t[1] = s.parameter_number;    // Table 4.2
    t[2] = s.aerosol_type;        // Table 4.233
    t[3] = 0;                     // type of interval for size distribution
    t[4] = 0;                     // scale factor of first size
    t[5] = 0;                     // scaled value of first size
    t[6] = 0;                     // scale factor of second size
    t[7] = 0;                     // scaled value of second size

    // --- Indices 8–12: optical property + wavelength ---
    t[8] = s.optical_property_type;  // optical property type
    t[9] = 0;                        // scale factor of first wavelength
    t[10] = s.wavelength_first_nm;   // scaled value of first wavelength (wavelength_first_nm)
    t[11] = 0;                       // scale factor of second wavelength
    t[12] = s.wavelength_last_nm;    // scaled value of second wavelength (wavelength_last_nm)

    // --- Indices 13–19: generating process + time ---
    t[13] = 2;                            // type of generating process (2 = forecast)
    t[14] = 0;                            // background generating process id
    t[15] = 0;                            // analysis/forecast generating process id
    t[16] = 0;                            // hours after reference time (data cutoff)
    t[17] = 0;                            // minutes after reference time (data cutoff)
    t[18] = s.indicator_of_unit_of_time;  // Table 4.4
    t[19] = s.forecast_time;              // forecast time in above units

    // --- Indices 20–25: fixed surfaces ---
    t[20] = s.type_of_first_fixed_surface;  // Table 4.5
    t[21] = s.scale_factor_first_surface;   // scale factor of first fixed surface
    t[22] = s.scaled_value_first_surface;   // scaled value of first fixed surface
    t[23] = 255;                            // type of second fixed surface (255 = missing)
    t[24] = 0;                              // scale factor of second fixed surface
    t[25] = 0;                              // scaled value of second fixed surface

    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_pdt_4_49(const Grib2Settings &s) {
    // Product Definition Template 4.49 (individual ensemble forecast
    // for aerosol optical properties), 29 entries.
    // Layout per WMO GRIB2 Template 4.49 / NCEP getpdstemplate(49).
    std::vector<std::int64_t> t(29, 0);

    // --- Indices 0–7: parameter + aerosol / size distribution ---
    t[0] = s.parameter_category;  // Table 4.1
    t[1] = s.parameter_number;    // Table 4.2
    t[2] = s.aerosol_type;        // Table 4.233
    t[3] = 0;                     // type of interval for size distribution
    t[4] = 0;                     // scale factor of first size
    t[5] = 0;                     // scaled value of first size
    t[6] = 0;                     // scale factor of second size
    t[7] = 0;                     // scaled value of second size

    // --- Indices 8–12: optical property + wavelength ---
    t[8] = s.optical_property_type;  // optical property type
    t[9] = 0;                        // scale factor of first wavelength
    t[10] = s.wavelength_first_nm;   // scaled value of first wavelength (wavelength_first_nm)
    t[11] = 0;                       // scale factor of second wavelength
    t[12] = s.wavelength_last_nm;    // scaled value of second wavelength (wavelength_last_nm)

    // --- Indices 13–19: generating process + time ---
    t[13] = 2;                            // type of generating process (2 = forecast)
    t[14] = 0;                            // background generating process id
    t[15] = 0;                            // analysis/forecast generating process id
    t[16] = 0;                            // hours after reference time (data cutoff)
    t[17] = 0;                            // minutes after reference time (data cutoff)
    t[18] = s.indicator_of_unit_of_time;  // Table 4.4
    t[19] = s.forecast_time;              // forecast time in above units

    // --- Indices 20–22: ensemble ---
    t[20] = 0;                               // type of ensemble forecast
    t[21] = s.ensemble_perturbation_number;  // perturbation number
    t[22] = 0;                               // number of forecasts in ensemble

    // --- Indices 23–28: fixed surfaces ---
    t[23] = s.type_of_first_fixed_surface;  // Table 4.5
    t[24] = s.scale_factor_first_surface;   // scale factor of first fixed surface
    t[25] = s.scaled_value_first_surface;   // scaled value of first fixed surface
    t[26] = 255;                            // type of second fixed surface (255 = missing)
    t[27] = 0;                              // scale factor of second fixed surface
    t[28] = 0;                              // scaled value of second fixed surface

    return t;
}

std::vector<std::int64_t> GRIB2_Driver::build_drs_template(GRIB2_DRT drt, const Grib2Settings &s) {
    if (drt == GRIB2_DRT::LosslessJPEG2000) {
        // Data Representation Template 5.40 (JPEG2000), 7 entries.
        // Entry 0 (reference value) is filled by g2c during packing.
        std::vector<std::int64_t> t(7, 0);
        t[1] = 0;                       // binary scale factor
        t[2] = s.decimal_scale_factor;  // decimal scale factor
        t[3] = 0;                       // number of bits (0 = g2c chooses)
        t[4] = 0;                       // original field type (0 = floating point)
        t[5] = 0;                       // compression type (0 = lossless)
        t[6] = 255;                     // target compression ratio (255 = lossless)
        return t;
    }
    // Data Representation Template 5.42 (CCSDS / AEC via libaec), 8 entries.
    std::vector<std::int64_t> t(8, 0);
    t[1] = 0;                       // binary scale factor
    t[2] = s.decimal_scale_factor;  // decimal scale factor
    t[3] = 0;                       // number of bits (0 = g2c chooses)
    t[4] = 0;                       // original field type (0 = floating point)
    t[5] = 0;                       // CCSDS compression options mask
    t[6] = 0;                       // CCSDS block size
    t[7] = 0;                       // CCSDS reference sample interval
    return t;
}

// ---------------------------------------------------------------
// write -- encode a GRIB2 record (R9.3, R9.4, R9.5, R9.6, R9.7, R9.8)
// ---------------------------------------------------------------

void GRIB2_Driver::write(const StagingBuffer &src, const VarMeta &meta) {
    if (!initialized_) {
        throw std::runtime_error("GRIB2_Driver::write called before successful initialization");
    }

    // GRIB2 grid-point encoding operates on float fields.  Only F32
    // payloads are supported; anything else is rejected before any
    // bytes are emitted (R9.8).
    if (meta.dtype != AMIO_DTYPE_F32) {
        throw std::runtime_error(
            "GRIB2_Driver::write: only AMIO_DTYPE_F32 fields can be encoded "
            "to GRIB2 grid-point data. Zero record bytes emitted.");
    }

    const std::size_t elem_size = dtype_size(meta.dtype);
    const std::size_t num_elements = total_elements(meta.shape);
    if (num_elements == 0 || elem_size == 0) {
        throw std::runtime_error("GRIB2_Driver::write: empty or invalid field shape. Zero record bytes emitted.");
    }

    // Contiguity check gates fast vs slow path (R9.4, R9.5).
    const std::byte *encode_ptr = nullptr;
    std::vector<std::byte> packed_buffer;
    if (is_contiguous_row_major(meta.shape)) {
        // Fast path: pass the staging buffer through directly.
        encode_ptr = src.data;
    } else {
        // Slow path: pack into a contiguous row-major buffer.
        packed_buffer = pack_row_major(src.data, meta.shape, elem_size);
        encode_ptr = packed_buffer.data();
    }

#ifdef AMIO_HAS_G2C
    // Derive the grid dimensions.  For rank >= 2 the last dimension is
    // longitudes (Ni, fastest varying) and the second-to-last is
    // latitudes (Nj).  For rank 1 we treat the field as a single row.
    std::int64_t ni = 0;
    std::int64_t nj = 0;
    if (meta.shape.rank >= 2) {
        nj = meta.shape.extents[meta.shape.rank - 2];
        ni = meta.shape.extents[meta.shape.rank - 1];
    } else {
        nj = 1;
        ni = static_cast<std::int64_t>(num_elements);
    }

    // Section 0 (indicator): [discipline, edition].
    g2int listsec0[2];
    listsec0[0] = static_cast<g2int>(settings_.discipline);
    listsec0[1] = 2;  // GRIB edition 2

    // Section 1 (identification): 13 entries.
    g2int listsec1[13];
    listsec1[0] = static_cast<g2int>(settings_.center);
    listsec1[1] = static_cast<g2int>(settings_.subcenter);
    listsec1[2] = static_cast<g2int>(settings_.master_table_version);
    listsec1[3] = static_cast<g2int>(settings_.local_table_version);
    listsec1[4] = static_cast<g2int>(settings_.significance_of_ref_time);
    listsec1[5] = 1970;  // year   (reference time; AMIO does not yet thread real dates)
    listsec1[6] = 1;     // month
    listsec1[7] = 1;     // day
    listsec1[8] = 0;     // hour
    listsec1[9] = 0;     // minute
    listsec1[10] = 0;    // second
    listsec1[11] = static_cast<g2int>(settings_.production_status);
    listsec1[12] = static_cast<g2int>(settings_.type_of_data);

    // Generously size the GRIB message buffer: header + raw field +
    // slack for section overhead.  g2c writes into this buffer.
    const std::size_t field_bytes = num_elements * sizeof(float);
    std::vector<unsigned char> cgrib(field_bytes + 8192 + num_elements);

    g2int ierr = g2_create(cgrib.data(), listsec0, listsec1);
    if (ierr <= 0) {
        throw std::runtime_error("GRIB2_Driver::write: g2_create failed (Section 0/1). Zero record bytes emitted.");
    }

    // Section 3 (grid definition).  igds describes the source of the
    // grid definition and the number of data points.
    // GDT dispatch: select the builder based on the configured GDT number
    // (Req 11.1, 11.2, 11.3).
    std::vector<std::int64_t> gdt;
    switch (settings_.gdt_number) {
        case 0:
            gdt = build_gdt_3_0(settings_, ni, nj);
            break;
        case 40:
            gdt = build_gdt_3_40(settings_, ni, nj);
            break;
        default:
            throw std::runtime_error("GRIB2_Driver::write: unsupported GDT number " + std::to_string(settings_.gdt_number) +
                                     ". Zero record bytes emitted.");
    }
    std::vector<g2int> igdstmpl(gdt.begin(), gdt.end());
    g2int igds[5];
    igds[0] = 0;                                         // grid defined by template (Table 3.0)
    igds[1] = static_cast<g2int>(ni * nj);               // number of data points
    igds[2] = 0;                                         // octets for optional list of numbers
    igds[3] = 0;                                         // interpretation of optional list
    igds[4] = static_cast<g2int>(settings_.gdt_number);  // Grid Definition Template number

    ierr = g2_addgrid(cgrib.data(), igds, igdstmpl.data(), nullptr, 0);
    if (ierr <= 0) {
        throw std::runtime_error("GRIB2_Driver::write: g2_addgrid failed (Section 3). Zero record bytes emitted.");
    }

    // Sections 4/5/6/7 (product, data representation, bitmap, data).
    // PDT dispatch: select the correct template builder based on pdt_number.
    std::vector<std::int64_t> pdt;
    switch (settings_.pdt_number) {
        case 0:
            pdt = build_pdt_4_0(settings_);
            break;
        case 8:
            pdt = build_pdt_4_8(settings_);
            break;
        case 40:
            pdt = build_pdt_4_40(settings_);
            break;
        case 44:
            pdt = build_pdt_4_44(settings_);
            break;
        case 45:
            pdt = build_pdt_4_45(settings_);
            break;
        case 46:
            pdt = build_pdt_4_46(settings_);
            break;
        case 48:
            pdt = build_pdt_4_48(settings_);
            break;
        case 49:
            pdt = build_pdt_4_49(settings_);
            break;
        default:
            throw std::runtime_error("GRIB2_Driver::write: unsupported PDT number " + std::to_string(settings_.pdt_number) +
                                     ". Zero record bytes emitted.");
    }
    std::vector<g2int> ipdstmpl(pdt.begin(), pdt.end());

    // Note (Req 12.8, 14.3): The encode-path field identity is implicitly
    // consistent with the decode-path identity.  Both paths use the same
    // settings fields → template indices → extraction logic:
    //   - Encode: settings_.X → build_pdt_P()[idx]
    //   - Decode: ipdtmpl[idx] → extract_composition_metadata() → field_identity_name()
    // This round-trip equivalence is formally verified by Property 3
    // (test_p31_identity_roundtrip).  No explicit field_identity_name call
    // is needed here since the write path does not log or return the identity.

    auto drs = build_drs_template(active_drt_, settings_);
    std::vector<g2int> idrstmpl(drs.begin(), drs.end());

    // Copy the (already row-major) payload into a float field array.
    std::vector<float> fld(num_elements);
    std::memcpy(fld.data(), encode_ptr, num_elements * sizeof(float));

    g2int idrsnum = static_cast<g2int>(active_drt_);  // 40 or 42
    ierr = g2_addfield(cgrib.data(),
                       static_cast<g2int>(settings_.pdt_number),  // Product Definition Template number
                       ipdstmpl.data(),
                       nullptr,  // coordlist
                       0,        // numcoord
                       idrsnum, idrstmpl.data(), fld.data(), static_cast<g2int>(num_elements),
                       255,       // ibmap = 255 -> bitmap does not apply (no bmap); 0/254 would dereference bmap
                       nullptr);  // bmap (none -- no missing-value grid points)
    if (ierr <= 0) {
        throw std::runtime_error("GRIB2_Driver::write: g2_addfield failed (Section 4/5/7). Zero record bytes emitted.");
    }

    g2int msglen = g2_gribend(cgrib.data());
    if (msglen <= 0) {
        throw std::runtime_error("GRIB2_Driver::write: g2_gribend failed (Section 8). Zero record bytes emitted.");
    }

    // Commit the complete GRIB2 message to the output file.
    if (out_file_ != nullptr) {
        std::size_t written = std::fwrite(cgrib.data(), 1, static_cast<std::size_t>(msglen), out_file_);
        if (written != static_cast<std::size_t>(msglen)) {
            throw std::runtime_error("GRIB2_Driver::write: short write committing GRIB2 message to disk.");
        }
    }
#else
    (void)encode_ptr;
    throw std::runtime_error("GRIB2_Driver::write: nceplibs-g2c not available");
#endif
}

// ---------------------------------------------------------------
// read -- decode a GRIB2 record into StagingBuffer
// ---------------------------------------------------------------

void GRIB2_Driver::read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) {
    if (!initialized_) {
        throw std::runtime_error("GRIB2_Driver::read called before successful initialization");
    }

#ifdef AMIO_HAS_G2C
    if (!read_mode_) {
        throw std::runtime_error("GRIB2_Driver::read: driver was not opened for reading.");
    }

    // ---- Locate the indexed record for (variable, timestep) ----
    // meta.name is the synthetic field-identity string (see
    // field_identity_name); the read coordinator threads the caller's
    // variable name through unchanged.  A missing variable or an
    // out-of-range timestep is a hard failure -- no partial success
    // (Req 13.6).
    auto it = records_.find(meta.name);
    if (it == records_.end() || it->second.records.empty()) {
        throw std::runtime_error("GRIB2_Driver::read: field '" + meta.name + "' not found in the GRIB2 record index.");
    }
    const GribFieldIndex &entry = it->second;

    if (timestep < 0 || timestep >= static_cast<std::int64_t>(entry.records.size())) {
        throw std::runtime_error("GRIB2_Driver::read: timestep " + std::to_string(static_cast<long long>(timestep)) + " out of range for field '" +
                                 meta.name + "' (" + std::to_string(entry.records.size()) + " records).");
    }
    const GribRecordLocation &loc = entry.records[static_cast<std::size_t>(timestep)];
    if (loc.length <= 0) {
        throw std::runtime_error("GRIB2_Driver::read: invalid record length for field '" + meta.name + "'.");
    }

    // ---- Read the GRIB2 message bytes from the source file ----
    std::FILE *fp = std::fopen(input_path_.c_str(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("GRIB2_Driver::read: failed to open input file '" + input_path_ + "' for reading.");
    }
    struct FileGuard {
        std::FILE *f;
        ~FileGuard() {
            if (f != nullptr) std::fclose(f);
        }
    } guard{fp};

    std::vector<unsigned char> cgrib(static_cast<std::size_t>(loc.length));
    if (std::fseek(fp, static_cast<long>(loc.offset), SEEK_SET) != 0) {
        throw std::runtime_error("GRIB2_Driver::read: seek failed decoding field '" + meta.name + "'.");
    }
    std::size_t got = std::fread(cgrib.data(), 1, static_cast<std::size_t>(loc.length), fp);
    if (got != static_cast<std::size_t>(loc.length)) {
        throw std::runtime_error("GRIB2_Driver::read: short read decoding field '" + meta.name + "'.");
    }

    // ---- Decode the requested record (unpack=1, expand=1) ----
    // expand=1 fills bit-mapped-out grid points so fld matches the grid,
    // keeping the decoded layout aligned with the encode path.
    gribfield *gfld = nullptr;
    g2int ret = g2_getfld(cgrib.data(), static_cast<g2int>(loc.field_number), /*unpack=*/1, /*expand=*/1, &gfld);
    if (ret != 0 || gfld == nullptr) {
        if (gfld != nullptr) {
            g2_free(gfld);
        }
        throw std::runtime_error("GRIB2_Driver::read: g2_getfld failed (code " + std::to_string(static_cast<long long>(ret)) + ") decoding field '" +
                                 meta.name + "'.");
    }

    // gfld->fld is a g2float* (== float*, confirmed in grib2.h) of
    // ngrdpts unpacked grid-point values.  Copy into a contiguous
    // full-grid float buffer; GRIB2 fields are delivered as F32
    // consistent with the encode path (Req 13.3).
    const std::int64_t ngrdpts = static_cast<std::int64_t>(gfld->ngrdpts);
    if (ngrdpts <= 0 || gfld->fld == nullptr) {
        g2_free(gfld);
        throw std::runtime_error("GRIB2_Driver::read: decoded field '" + meta.name + "' has no grid-point data.");
    }

    std::vector<float> full(static_cast<std::size_t>(ngrdpts));
    std::memcpy(full.data(), gfld->fld, static_cast<std::size_t>(ngrdpts) * sizeof(float));

    // Release the gribfield on every subsequent path (no leaks); all the
    // data we need now lives in `full`.
    g2_free(gfld);
    gfld = nullptr;

    // Decoded grid geometry (row-major: Nj rows of Ni), matching the
    // shape describe_variable reports and the encode path's layout.
    std::int64_t grid_extents[AMIO_MAX_RANK] = {};
    std::int32_t grid_rank = 0;
    if (entry.nj > 1) {
        grid_rank = 2;
        grid_extents[0] = entry.nj;
        grid_extents[1] = entry.ni;
    } else {
        grid_rank = 1;
        grid_extents[0] = ngrdpts;
    }

    // ---- Deliver the full grid or the requested sub-region (Req 13.5) ----
    // GRIB2 has no random intra-record access, so the full record is
    // decoded above and the bounding box selects from the decoded grid
    // (decode-then-subset).  The extraction mirrors pack_row_major: it
    // walks the destination in row-major order over the box extents and
    // gathers each element from its strided position in the full grid.
    std::size_t payload_bytes = 0;
    if (!bbox.has_value()) {
        payload_bytes = static_cast<std::size_t>(ngrdpts) * sizeof(float);
        if (payload_bytes > dst.capacity_bytes) {
            throw std::runtime_error("GRIB2_Driver::read: decoded payload exceeds staging buffer capacity.");
        }
        std::memcpy(dst.data, full.data(), payload_bytes);
    } else {
        const BoundingBox &b = *bbox;
        if (b.rank != grid_rank) {
            throw std::runtime_error("GRIB2_Driver::read: bounding-box rank does not match field rank.");
        }

        // Validate the box against the decoded grid and count elements.
        std::size_t sub_elems = 1;
        for (std::int32_t d = 0; d < grid_rank; ++d) {
            if (b.extents[d] < 1 || b.strides[d] < 1 || b.offsets[d] < 0) {
                throw std::runtime_error("GRIB2_Driver::read: invalid bounding box for field '" + meta.name + "'.");
            }
            const std::int64_t last = b.offsets[d] + (b.extents[d] - 1) * b.strides[d];
            if (last >= grid_extents[d]) {
                throw std::runtime_error("GRIB2_Driver::read: bounding box exceeds field extents for '" + meta.name + "'.");
            }
            sub_elems *= static_cast<std::size_t>(b.extents[d]);
        }

        payload_bytes = sub_elems * sizeof(float);
        if (payload_bytes > dst.capacity_bytes) {
            throw std::runtime_error("GRIB2_Driver::read: sub-region exceeds staging buffer capacity.");
        }

        // Full-grid row-major strides (in elements).
        std::int64_t full_strides[AMIO_MAX_RANK] = {};
        std::int64_t stride_acc = 1;
        for (std::int32_t d = grid_rank - 1; d >= 0; --d) {
            full_strides[d] = stride_acc;
            stride_acc *= grid_extents[d];
        }

        // Gather the sub-region into the destination buffer.
        float *out = reinterpret_cast<float *>(dst.data);
        std::int64_t idx[AMIO_MAX_RANK] = {};
        for (std::size_t e = 0; e < sub_elems; ++e) {
            std::int64_t src_off = 0;
            for (std::int32_t d = 0; d < grid_rank; ++d) {
                src_off += (b.offsets[d] + idx[d] * b.strides[d]) * full_strides[d];
            }
            out[e] = full[static_cast<std::size_t>(src_off)];

            // Increment the multi-dimensional index (row-major: last dim first).
            for (std::int32_t d = grid_rank - 1; d >= 0; --d) {
                if (++idx[d] < b.extents[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    dst.used_bytes = payload_bytes;
#else
    (void)dst;
    (void)meta;
    (void)timestep;
    (void)bbox;
    throw std::runtime_error("GRIB2_Driver::read: nceplibs-g2c not available");
#endif
}

// ---------------------------------------------------------------
// flush -- flush the output file buffer
// ---------------------------------------------------------------

void GRIB2_Driver::flush() {
#ifdef AMIO_HAS_G2C
    if (out_file_ != nullptr) {
        std::fflush(out_file_);
    }
#endif
}

// ---------------------------------------------------------------
// close -- release file handle and g2c resources
// ---------------------------------------------------------------

void GRIB2_Driver::close() {
    if (!initialized_) {
        return;
    }

#ifdef AMIO_HAS_G2C
    if (out_file_ != nullptr) {
        std::fclose(out_file_);
        out_file_ = nullptr;
    }
#endif

    // Release read-side state (the scan handle is already closed by
    // build_record_index; only the in-memory index persists).
    records_.clear();
    read_mode_ = false;
    input_path_.clear();

    initialized_ = false;
}

// ---------------------------------------------------------------
// validate_drt -- check DRT field is present and in allowed set
// (R9.6, R9.7)
// ---------------------------------------------------------------

GRIB2_DRT GRIB2_Driver::validate_drt(const conf::Config &config) const {
    // Check if DRT field is present (R9.7 - "missing" case).
    if (!config.has("data_representation_template") && !config.has("drt")) {
        throw std::runtime_error(
            "GRIB2_Driver: Data Representation Template field is missing "
            "from configuration. Required field: 'data_representation_"
            "template' or 'drt'. The field was missing. "
            "Zero record bytes emitted.");
    }

    // Get the DRT name.
    std::string drt_name;
    if (config.has("data_representation_template")) {
        drt_name = config.get_string("data_representation_template");
    } else {
        drt_name = config.get_string("drt");
    }

    if (drt_name.empty()) {
        throw std::runtime_error(
            "GRIB2_Driver: Data Representation Template field is present "
            "but empty. The field value is missing. "
            "Required: one of {Adaptive Entropy Coding via libaec, "
            "Lossless JPEG2000}. Zero record bytes emitted.");
    }

    // parse_drt_name throws for unrecognized names (R9.6, R9.7 -
    // "unrecognized" case).
    return parse_drt_name(drt_name);
}

// ---------------------------------------------------------------
// is_contiguous_row_major -- check if shape describes contiguous
// row-major layout (R9.4)
// ---------------------------------------------------------------

bool GRIB2_Driver::is_contiguous_row_major(const amio_shape_t &shape) {
    if (shape.rank <= 0 || shape.rank > AMIO_MAX_RANK) {
        return false;
    }

    // If all strides are 0, the layout is contiguous row-major by
    // convention (the shape descriptor asks AMIO to derive strides
    // from extents).  This is equivalent to is_always_contiguous()
    // returning true with row-major layout.
    bool all_strides_zero = true;
    for (std::int32_t d = 0; d < shape.rank; ++d) {
        if (shape.strides[d] != 0) {
            all_strides_zero = false;
            break;
        }
    }
    if (all_strides_zero) {
        return true;
    }

    // Verify explicit strides match row-major contiguous layout.
    // Row-major: stride[rank-1] = 1, stride[d] = product(extents[d+1..rank-1])
    std::int64_t expected_stride = 1;
    for (std::int32_t d = shape.rank - 1; d >= 0; --d) {
        if (shape.strides[d] != 0 && shape.strides[d] != expected_stride) {
            return false;
        }
        expected_stride *= shape.extents[d];
    }

    return true;
}

// ---------------------------------------------------------------
// pack_row_major -- pack non-contiguous data into contiguous buffer
// (R9.5)
// ---------------------------------------------------------------

std::vector<std::byte> GRIB2_Driver::pack_row_major(const std::byte *src_data, const amio_shape_t &shape, std::size_t element_size) {
    const std::size_t num_elements = total_elements(shape);
    std::vector<std::byte> packed(num_elements * element_size);

    if (num_elements == 0 || element_size == 0) {
        return packed;
    }

    // Compute actual strides (resolve zeros to row-major defaults).
    std::int64_t actual_strides[AMIO_MAX_RANK] = {};
    std::int64_t row_major_stride = 1;
    for (std::int32_t d = shape.rank - 1; d >= 0; --d) {
        if (shape.strides[d] == 0) {
            actual_strides[d] = row_major_stride;
        } else {
            actual_strides[d] = shape.strides[d];
        }
        row_major_stride *= shape.extents[d];
    }

    // Iterate over all elements in row-major order and copy each
    // element from its strided position to the packed buffer.
    std::int64_t indices[AMIO_MAX_RANK] = {};
    std::size_t dst_offset = 0;

    for (std::size_t elem = 0; elem < num_elements; ++elem) {
        // Compute source offset from indices and strides (in elements).
        std::size_t src_elem_offset = 0;
        for (std::int32_t d = 0; d < shape.rank; ++d) {
            src_elem_offset += static_cast<std::size_t>(indices[d]) * static_cast<std::size_t>(actual_strides[d]);
        }
        std::size_t src_byte_offset = src_elem_offset * element_size;

        // Copy one element.
        std::memcpy(packed.data() + dst_offset, src_data + src_byte_offset, element_size);
        dst_offset += element_size;

        // Increment multi-dimensional index (row-major: last dim first).
        for (std::int32_t d = shape.rank - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape.extents[d]) {
                break;
            }
            indices[d] = 0;
        }
    }

    return packed;
}

// ---------------------------------------------------------------
// total_elements -- compute total element count from shape
// ---------------------------------------------------------------

std::size_t GRIB2_Driver::total_elements(const amio_shape_t &shape) {
    if (shape.rank <= 0 || shape.rank > AMIO_MAX_RANK) {
        return 0;
    }
    std::size_t count = 1;
    for (std::int32_t d = 0; d < shape.rank; ++d) {
        if (shape.extents[d] <= 0) {
            return 0;
        }
        count *= static_cast<std::size_t>(shape.extents[d]);
    }
    return count;
}

// ---------------------------------------------------------------
// dtype_size -- element size in bytes from dtype tag
// ---------------------------------------------------------------

std::size_t GRIB2_Driver::dtype_size(amio_dtype_t dtype) {
    // Delegate to the shared element_size helper (backend_driver.hpp); it
    // already returns 0 as the unknown-dtype sentinel that this driver relies
    // on (see read()'s elem_size == 0 guard).
    return element_size(dtype);
}

}  // namespace amio::detail

extern "C" void amio_register_grib2_driver() {}
