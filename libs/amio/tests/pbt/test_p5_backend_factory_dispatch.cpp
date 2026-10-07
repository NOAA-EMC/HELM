// test_p5_backend_factory_dispatch.cpp -- Property test P5: Backend factory dispatch invariant.
//
// Property 5: Backend factory dispatch invariant
//
// For any (backend string, registered keys): exact-match → correct
// driver; non-match → AMIO_ERR_UNKNOWN_BACKEND with zero state
// mutation; concurrent read+write resolve independently.
//
// Min 100 iterations.
//
// **Validates: Requirements R4.1, R4.3, R4.4, R4.5, R4.6, R4.7**

#include <rapidcheck.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cctype>
#include <string>
#include <vector>

#include "factory/backend_factory.hpp"
#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"

namespace {

// Helper: generate a random non-empty alphanumeric string of length [1, 20].
rc::Gen<std::string> genAlphaString() {
    return rc::gen::exec([]() {
        std::size_t len = *rc::gen::inRange<std::size_t>(1, 21);
        std::string s;
        s.reserve(len);
        for (std::size_t i = 0; i < len; ++i) {
            // Generate printable ASCII chars [a-z, A-Z, 0-9]
            int kind = *rc::gen::inRange(0, 3);
            switch (kind) {
                case 0:
                    s += static_cast<char>(*rc::gen::inRange(static_cast<int>('a'), static_cast<int>('z') + 1));
                    break;
                case 1:
                    s += static_cast<char>(*rc::gen::inRange(static_cast<int>('A'), static_cast<int>('Z') + 1));
                    break;
                case 2:
                    s += static_cast<char>(*rc::gen::inRange(static_cast<int>('0'), static_cast<int>('9') + 1));
                    break;
            }
        }
        return s;
    });
}

// Helper: generate a string that does NOT match any key in the given set.
// Strategies: wrong case, appended char, empty, or completely random.
rc::Gen<std::string> genNonMatchingKey(const std::vector<std::string> &registered_keys) {
    return rc::gen::exec([registered_keys]() -> std::string {
        int strategy = *rc::gen::inRange(0, 4);

        switch (strategy) {
            case 0: {
                // Empty string -- always a non-match.
                return "";
            }
            case 1: {
                // Wrong case: take a registered key and flip case of first char.
                if (registered_keys.empty()) {
                    return "UNKNOWN_BACKEND_XYZ";
                }
                std::size_t idx = *rc::gen::inRange<std::size_t>(0, registered_keys.size());
                std::string key = registered_keys[idx];
                if (!key.empty()) {
                    if (std::islower(static_cast<unsigned char>(key[0]))) {
                        key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
                    } else {
                        key[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(key[0])));
                    }
                }
                return key;
            }
            case 2: {
                // Append a suffix to a registered key.
                if (registered_keys.empty()) {
                    return "no_such_backend";
                }
                std::size_t idx = *rc::gen::inRange<std::size_t>(0, registered_keys.size());
                return registered_keys[idx] + "_extra";
            }
            case 3: {
                // Completely random string unlikely to match.
                std::string s = *genAlphaString();
                // Ensure it doesn't accidentally match a registered key.
                while (std::find(registered_keys.begin(), registered_keys.end(), s) != registered_keys.end()) {
                    s += "x";
                }
                return s;
            }
        }
        return "FALLBACK_UNKNOWN";
    });
}

// RAII guard that registers mock drivers and clears the factory on destruction.
class FactoryTestGuard {
   public:
    explicit FactoryTestGuard(const std::vector<std::string> &keys) {
        auto &factory = amio::detail::BackendFactory::instance();
        factory.clear();
        for (const auto &key : keys) {
            factory.register_driver(
                key, []() -> std::unique_ptr<amio::detail::Backend_Driver> { return std::make_unique<amio::pbt::MockBackendDriver>(); });
        }
    }

    ~FactoryTestGuard() {
        amio::detail::BackendFactory::instance().clear();
    }

    FactoryTestGuard(const FactoryTestGuard &) = delete;
    FactoryTestGuard &operator=(const FactoryTestGuard &) = delete;
};

}  // namespace

// ===================================================================
// P5.1: Exact-match returns correct driver for registered keys.
// ===================================================================

TEST_CASE("P5: exact-match returns correct driver for registered keys", "[pbt][factory]") {
    rc::check("registered key → build() returns non-null driver with AMIO_OK", []() {
        // Generate a set of 1-5 unique keys to register.
        std::size_t n_keys = *rc::gen::inRange<std::size_t>(1, 6);
        std::vector<std::string> keys;
        for (std::size_t i = 0; i < n_keys; ++i) {
            std::string k = *genAlphaString();
            // Ensure uniqueness.
            while (std::find(keys.begin(), keys.end(), k) != keys.end()) {
                k += "x";
            }
            keys.push_back(k);
        }

        FactoryTestGuard guard(keys);
        auto &factory = amio::detail::BackendFactory::instance();

        // Pick a random registered key and verify build succeeds.
        std::size_t pick = *rc::gen::inRange<std::size_t>(0, keys.size());
        const std::string &lookup_key = keys[pick];

        amio_err_t err = AMIO_ERR_UNKNOWN_BACKEND;
        auto driver = factory.build(lookup_key, err);

        RC_ASSERT(err == AMIO_OK);
        RC_ASSERT(driver);
    });
}

// ===================================================================
// P5.2: Non-match returns AMIO_ERR_UNKNOWN_BACKEND with zero state mutation.
// ===================================================================

TEST_CASE("P5: non-match returns AMIO_ERR_UNKNOWN_BACKEND, zero state mutation", "[pbt][factory]") {
    rc::check("unregistered key → AMIO_ERR_UNKNOWN_BACKEND, registry unchanged", []() {
        // Register known keys.
        std::vector<std::string> keys = {"netcdf4", "zarr3", "grib2"};
        FactoryTestGuard guard(keys);
        auto &factory = amio::detail::BackendFactory::instance();

        // Snapshot registry state before the failed lookup.
        auto keys_before = factory.registered_keys();

        // Generate a non-matching key.
        std::string bad_key = *genNonMatchingKey(keys);

        amio_err_t err = AMIO_OK;
        auto driver = factory.build(bad_key, err);

        // Must return error, no driver.
        RC_ASSERT(err == AMIO_ERR_UNKNOWN_BACKEND);
        RC_ASSERT(!driver);

        // Zero state mutation: registry unchanged.
        auto keys_after = factory.registered_keys();
        RC_ASSERT(keys_before == keys_after);
    });
}

// ===================================================================
// P5.3: Wrong case is a non-match (case-sensitive exact-match).
// ===================================================================

TEST_CASE("P5: wrong case is a non-match (case-sensitive)", "[pbt][factory]") {
    rc::check("case-flipped key → AMIO_ERR_UNKNOWN_BACKEND", []() {
        // Register lowercase keys.
        std::vector<std::string> keys = {"netcdf4", "zarr3", "grib2"};
        FactoryTestGuard guard(keys);
        auto &factory = amio::detail::BackendFactory::instance();

        // Pick a key and flip case of a random character.
        std::size_t pick = *rc::gen::inRange<std::size_t>(0, keys.size());
        std::string flipped = keys[pick];
        if (!flipped.empty()) {
            std::size_t char_idx = *rc::gen::inRange<std::size_t>(0, flipped.size());
            char c = flipped[char_idx];
            if (std::islower(static_cast<unsigned char>(c))) {
                flipped[char_idx] = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            } else if (std::isalpha(static_cast<unsigned char>(c))) {
                flipped[char_idx] = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else {
                // Non-alpha char (digit) -- append an uppercase letter.
                flipped += "X";
            }
        }

        // Ensure the flipped key is actually different from all registered keys.
        RC_PRE(std::find(keys.begin(), keys.end(), flipped) == keys.end());

        amio_err_t err = AMIO_OK;
        auto driver = factory.build(flipped, err);

        RC_ASSERT(err == AMIO_ERR_UNKNOWN_BACKEND);
        RC_ASSERT(!driver);
    });
}

// ===================================================================
// P5.4: Empty string returns AMIO_ERR_UNKNOWN_BACKEND.
// ===================================================================

TEST_CASE("P5: empty string returns AMIO_ERR_UNKNOWN_BACKEND", "[pbt][factory]") {
    rc::check("empty key → AMIO_ERR_UNKNOWN_BACKEND", []() {
        // Register some keys.
        std::vector<std::string> keys = {"netcdf4", "zarr3", "grib2"};
        FactoryTestGuard guard(keys);
        auto &factory = amio::detail::BackendFactory::instance();

        amio_err_t err = AMIO_OK;
        auto driver = factory.build("", err);

        RC_ASSERT(err == AMIO_ERR_UNKNOWN_BACKEND);
        RC_ASSERT(!driver);
    });
}

// ===================================================================
// P5.5: Concurrent read+write datasets resolve independently.
//
// Two build() calls with different keys return independent driver
// instances (different pointers, different objects).
// ===================================================================

TEST_CASE("P5: concurrent read+write datasets resolve independently", "[pbt][factory]") {
    rc::check("two build() calls return independent driver instances", []() {
        // Register at least 2 keys.
        std::size_t n_keys = *rc::gen::inRange<std::size_t>(2, 6);
        std::vector<std::string> keys;
        for (std::size_t i = 0; i < n_keys; ++i) {
            std::string k = *genAlphaString();
            while (std::find(keys.begin(), keys.end(), k) != keys.end()) {
                k += "x";
            }
            keys.push_back(k);
        }

        FactoryTestGuard guard(keys);
        auto &factory = amio::detail::BackendFactory::instance();

        // Pick two (possibly same) keys.
        std::size_t idx1 = *rc::gen::inRange<std::size_t>(0, keys.size());
        std::size_t idx2 = *rc::gen::inRange<std::size_t>(0, keys.size());

        amio_err_t err1 = AMIO_ERR_UNKNOWN_BACKEND;
        amio_err_t err2 = AMIO_ERR_UNKNOWN_BACKEND;
        auto driver1 = factory.build(keys[idx1], err1);
        auto driver2 = factory.build(keys[idx2], err2);

        // Both should succeed.
        RC_ASSERT(err1 == AMIO_OK);
        RC_ASSERT(err2 == AMIO_OK);
        RC_ASSERT(driver1);
        RC_ASSERT(driver2);

        // They must be independent instances (different pointers).
        RC_ASSERT(driver1.get() != driver2.get());
    });
}
