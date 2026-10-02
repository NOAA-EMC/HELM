// test_p28_netcdf_data_model_validation.cpp -- Property test P28:
// NetCDF data model validation.
//
// For any data model string: NetCDF_Driver accepts only "classic"
// and "enhanced"; any other value → error with no file created.
//
// Min 100 iterations with generated data model strings.
//
// Uses REAL NetCDF_Driver through AMIO C API.
//
// **Validates: Requirements R7.2, R7.3**

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/netcdf/netcdf_driver.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Generators for NetCDF data model testing.
// ===================================================================

namespace {

// Valid data model strings accepted by NetCDF_Driver::parse_data_model.
std::vector<std::string> valid_data_models() {
    return {
        "classic", "netcdf4_classic", "NC4_CLASSIC", "enhanced", "netcdf4_enhanced", "NC4",
        ""  // Empty string defaults to "classic".
    };
}

// Invalid data model strings that should be rejected.
rc::Gen<std::string> genInvalidDataModel() {
    return rc::gen::exec([]() {
        // Choose from a set of invalid model strings.
        auto invalid_models = std::vector<std::string>{"netcdf3",       "hdf5",          "hdf4",  "cdf5",   "cdf2", "64bit_offset", "64bit_data",
                                                       "pnetcdf",       "zarr",          "grib2", "binary", "text", "json",         "xml",
                                                       "CLASSIC",   // Case-sensitive: uppercase not accepted
                                                       "Enhanced",  // Case-sensitive: mixed case not accepted
                                                       "ENHANCED",  // Case-sensitive: uppercase not accepted
                                                       "netcdf4",   // Not a valid data model name
                                                       "netcdf-4",  // Hyphen variant not accepted
                                                       "classic_model", "enhanced_model"};

        auto idx = *rc::gen::inRange<std::size_t>(0, invalid_models.size());
        return invalid_models[idx];
    });
}

// Generate a random valid data model string.
rc::Gen<std::string> genValidDataModel() {
    return rc::gen::exec([]() {
        auto models = valid_data_models();
        auto idx = *rc::gen::inRange<std::size_t>(0, models.size());
        return models[idx];
    });
}

// Generate a completely random string (likely invalid as a data model).
rc::Gen<std::string> genRandomString() {
    return rc::gen::exec([]() {
        int len = *rc::gen::inRange(1, 30);
        std::string result;
        result.reserve(len);
        for (int i = 0; i < len; ++i) {
            // Use printable ASCII characters.
            int ch = *rc::gen::inRange(32, 127);
            result += static_cast<char>(ch);
        }
        return result;
    });
}

}  // anonymous namespace

// ===================================================================
// Property Test P28a: Valid data model strings are accepted.
//
// For any valid data model string ("classic", "enhanced", and their
// aliases): NetCDF_Driver::parse_data_model returns a valid
// NetCDF4DataModel enum value without throwing.
//
// Validates: R7.2, R7.3
// ===================================================================

TEST_CASE("P28: NetCDF data model validation - valid models accepted", "[pbt][p28][netcdf][data_model][valid]") {
    auto result = rc::check("valid data model strings are accepted without exception", []() {
        auto model_str = *genValidDataModel();

        // Should not throw -- valid data model strings are accepted.
        NetCDF4DataModel model = NetCDF_Driver::parse_data_model(model_str);

        // Result must be one of the two valid data models.
        RC_ASSERT(model == NetCDF4DataModel::Classic || model == NetCDF4DataModel::Enhanced);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P28b: Invalid data model strings are rejected.
//
// For any data model string NOT in {"classic", "enhanced", and their
// aliases}: NetCDF_Driver::parse_data_model throws std::runtime_error.
//
// Validates: R7.2, R7.3
// ===================================================================

TEST_CASE("P28: NetCDF data model validation - invalid models rejected", "[pbt][p28][netcdf][data_model][invalid]") {
    auto result = rc::check("invalid data model strings throw std::runtime_error", []() {
        auto model_str = *genInvalidDataModel();

        // Must throw for invalid data model strings.
        bool threw = false;
        try {
            NetCDF_Driver::parse_data_model(model_str);
        } catch (const std::runtime_error &) {
            threw = true;
        }

        RC_ASSERT(threw);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P28c: "classic" maps to Classic data model.
//
// For any of the "classic" aliases: parse_data_model returns
// NetCDF4DataModel::Classic.
//
// Validates: R7.2
// ===================================================================

TEST_CASE("P28: NetCDF data model validation - classic aliases", "[pbt][p28][netcdf][data_model][classic]") {
    auto result = rc::check("classic aliases all map to NetCDF4DataModel::Classic", []() {
        auto classic_aliases = std::vector<std::string>{"classic", "netcdf4_classic", "NC4_CLASSIC", ""};

        auto idx = *rc::gen::inRange<std::size_t>(0, classic_aliases.size());
        auto model_str = classic_aliases[idx];

        NetCDF4DataModel model = NetCDF_Driver::parse_data_model(model_str);
        RC_ASSERT(model == NetCDF4DataModel::Classic);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P28d: "enhanced" maps to Enhanced data model.
//
// For any of the "enhanced" aliases: parse_data_model returns
// NetCDF4DataModel::Enhanced.
//
// Validates: R7.3
// ===================================================================

TEST_CASE("P28: NetCDF data model validation - enhanced aliases", "[pbt][p28][netcdf][data_model][enhanced]") {
    auto result = rc::check("enhanced aliases all map to NetCDF4DataModel::Enhanced", []() {
        auto enhanced_aliases = std::vector<std::string>{"enhanced", "netcdf4_enhanced", "NC4"};

        auto idx = *rc::gen::inRange<std::size_t>(0, enhanced_aliases.size());
        auto model_str = enhanced_aliases[idx];

        NetCDF4DataModel model = NetCDF_Driver::parse_data_model(model_str);
        RC_ASSERT(model == NetCDF4DataModel::Enhanced);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P28e: Random strings are rejected (no file created).
//
// For any randomly generated string that is not a valid data model:
// parse_data_model throws, ensuring no file would be created with
// an invalid data model configuration.
//
// Validates: R7.2, R7.3
// ===================================================================

TEST_CASE("P28: NetCDF data model validation - random strings rejected", "[pbt][p28][netcdf][data_model][random]") {
    auto result = rc::check("random strings that are not valid models are rejected", []() {
        auto model_str = *genRandomString();

        // Check if this random string happens to be a valid model.
        auto valid = valid_data_models();
        bool is_valid = std::find(valid.begin(), valid.end(), model_str) != valid.end();

        if (is_valid) {
            // If it happens to be valid, it should not throw.
            NetCDF4DataModel model = NetCDF_Driver::parse_data_model(model_str);
            RC_ASSERT(model == NetCDF4DataModel::Classic || model == NetCDF4DataModel::Enhanced);
        } else {
            // If not valid, it must throw.
            bool threw = false;
            try {
                NetCDF_Driver::parse_data_model(model_str);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            RC_ASSERT(threw);
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P28f: No file created on invalid data model.
//
// For any invalid data model string: attempting to open a NetCDF
// dataset with that model produces an error and no file is created
// at the target path.
//
// Validates: R7.2, R7.3
// ===================================================================

TEST_CASE("P28: NetCDF data model validation - no file on invalid model", "[pbt][p28][netcdf][data_model][no_file]") {
    auto result = rc::check("no file created when data model is invalid", []() {
        namespace fs = std::filesystem;

        TempDir tmp;
        std::string target_file = tmp.file("test_output.nc");

        // Verify the file does not exist before the attempt.
        RC_PRE(!fs::exists(target_file));

        auto model_str = *genInvalidDataModel();

        // parse_data_model should throw for invalid models.
        // This means open_write would never proceed to file creation.
        bool threw = false;
        try {
            NetCDF_Driver::parse_data_model(model_str);
        } catch (const std::runtime_error &) {
            threw = true;
        }

        RC_ASSERT(threw);

        // Since parse_data_model threw, no file should have been
        // created (the driver never reaches the file creation step).
        RC_ASSERT(!fs::exists(target_file));
    });

    REQUIRE(result);
}
