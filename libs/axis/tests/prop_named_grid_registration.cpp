// ─── Property-Based Tests: Named-Grid Family Registration Round-Trip ─────────
// Feature: helm-axis-microlibrary, Property 5: Named-Grid Family Registration
//          Round-Trip
//
// Uses RapidCheck to verify that:
//   1. Valid one-letter name strings (O/F/N/R + positive integer) → parse
//      succeeds, is_registered returns true, family and number match
//   2. Invalid name strings (unknown prefix, non-positive numbers, empty string,
//      non-numeric suffix) → parse throws std::invalid_argument,
//      is_registered returns false
//   3. registered_families() returns exactly {'C','F','G','N','O','R'} (sorted)
//   4. G-family names use the separate grid<num> syntax and are covered below
//
// No Kokkos needed for parse/is_registered tests — these are pure string logic.
//
// **Validates: Requirements 6.3, 6.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <axis/topology/named_grid_registry.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using axis::topology::NamedGridRegistry;

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a valid one-letter family prefix character: one of 'C', 'O', 'F', 'N', 'R'.
/// G is intentionally excluded here because it uses the grid<num> syntax.
rc::Gen<char> genValidFamily() {
    return rc::gen::element('C', 'O', 'F', 'N', 'R');
}

/// Generate a positive integer suitable as a grid number.
/// Range [1, 500] — small enough for fast tests but covers typical grid sizes.
rc::Gen<int> genPositiveNumber() {
    return rc::gen::inRange(1, 501);
}

/// Generate a valid grid name string (e.g. "O1280", "F128", "N320").
rc::Gen<std::string> genValidName() {
    return rc::gen::apply([](char family, int number) { return std::string(1, family) + std::to_string(number); }, genValidFamily(),
                          genPositiveNumber());
}

/// Generate an invalid family prefix character: NOT one of 'C', 'O', 'F', 'N', 'R',
/// 'c', 'o', 'f', 'n', 'r' — actually parse uppercases, so any char NOT in {C, O, F, N, R}
/// after toupper will be invalid. We pick from characters that are definitely
/// not valid families.
rc::Gen<char> genInvalidFamily() {
    return rc::gen::suchThat(rc::gen::inRange<char>('A', '['),  // 'A'..'Z'
                             [](char c) { return c != 'C' && c != 'O' && c != 'F' && c != 'N' && c != 'R'; });
}

/// Generate a non-positive number (0 or negative).
rc::Gen<int> genNonPositiveNumber() {
    return rc::gen::inRange(-100, 1);  // [-100, 0]
}

/// Generate a non-numeric suffix string (contains at least one non-digit).
/// We generate a length, then build the string manually via rc::gen::exec.
rc::Gen<std::string> genNonNumericSuffix() {
    return rc::gen::exec([]() {
        // Generate length in [1, 5]
        int len = *rc::gen::inRange(1, 6);
        std::string result;
        result.reserve(static_cast<std::size_t>(len));
        for (int i = 0; i < len; ++i) {
            // All branches produce non-digit characters (a-z, A-Z, '.')
            char c = *rc::gen::oneOf(rc::gen::inRange<char>('a', '{'),  // 'a'..'z'
                                     rc::gen::inRange<char>('A', '['),  // 'A'..'Z'
                                     rc::gen::just<char>('.'));
            result.push_back(c);
        }
        return result;
    });
}

// ─── Property 5a: Valid names parse successfully with correct family/number ──
// Generate valid one-letter grid name strings (O/F/N/R + positive integer),
// verify parse succeeds, returned family matches, returned number matches, and
// is_registered returns true.
//
// **Validates: Requirements 6.3**

RC_GTEST_PROP(PropNamedGridRegistration, ValidNameParsesCorrectly, ()) {
    const char family = *genValidFamily();
    const int number = *genPositiveNumber();
    const std::string name = std::string(1, family) + std::to_string(number);

    // parse must not throw
    NamedGridRegistry::ParsedName parsed = NamedGridRegistry::parse(name);

    // Family and number must match
    RC_ASSERT(parsed.family == family);
    RC_ASSERT(parsed.number == number);

    // is_registered must return true
    RC_ASSERT(NamedGridRegistry::is_registered(name));
}

// ─── Property 5b: Unknown family prefix throws std::invalid_argument ─────────
// Generate name strings with an unregistered family prefix (e.g. 'X', 'Z',
// 'A', etc.) followed by a positive integer, verify parse throws
// std::invalid_argument and is_registered returns false.
//
// **Validates: Requirements 6.4**

RC_GTEST_PROP(PropNamedGridRegistration, UnknownFamilyThrows, ()) {
    const char bad_family = *genInvalidFamily();
    const int number = *genPositiveNumber();
    const std::string name = std::string(1, bad_family) + std::to_string(number);

    // parse must throw std::invalid_argument
    RC_ASSERT_THROWS_AS((void)NamedGridRegistry::parse(name), std::invalid_argument);

    // is_registered must return false
    RC_ASSERT(!NamedGridRegistry::is_registered(name));
}

// ─── Property 5c: Non-positive number throws std::invalid_argument ───────────
// Generate name strings with a valid family prefix but a non-positive number
// (0 or negative), verify parse throws std::invalid_argument and
// is_registered returns false.
//
// **Validates: Requirements 6.4**

RC_GTEST_PROP(PropNamedGridRegistration, NonPositiveNumberThrows, ()) {
    const char family = *genValidFamily();
    const int number = *genNonPositiveNumber();
    const std::string name = std::string(1, family) + std::to_string(number);

    // parse must throw std::invalid_argument
    RC_ASSERT_THROWS_AS((void)NamedGridRegistry::parse(name), std::invalid_argument);

    // is_registered must return false
    RC_ASSERT(!NamedGridRegistry::is_registered(name));
}

// ─── Property 5d: Empty string throws std::invalid_argument ──────────────────
// The empty string is always invalid — parse throws and is_registered returns
// false. This is a single-example property (no generation needed) but included
// for completeness with RC_GTEST_PROP style.
//
// **Validates: Requirements 6.4**

RC_GTEST_PROP(PropNamedGridRegistration, EmptyStringThrows, ()) {
    const std::string name;  // empty

    RC_ASSERT_THROWS_AS((void)NamedGridRegistry::parse(name), std::invalid_argument);
    RC_ASSERT(!NamedGridRegistry::is_registered(name));
}

// ─── Property 5e: Non-numeric suffix throws std::invalid_argument ────────────
// Generate name strings with a valid family prefix but a non-numeric suffix
// (e.g. "Oabc", "F12x"), verify parse throws std::invalid_argument and
// is_registered returns false.
//
// **Validates: Requirements 6.4**

RC_GTEST_PROP(PropNamedGridRegistration, NonNumericSuffixThrows, ()) {
    const char family = *genValidFamily();
    const std::string suffix = *genNonNumericSuffix();
    const std::string name = std::string(1, family) + suffix;

    // parse must throw std::invalid_argument
    RC_ASSERT_THROWS_AS((void)NamedGridRegistry::parse(name), std::invalid_argument);

    // is_registered must return false
    RC_ASSERT(!NamedGridRegistry::is_registered(name));
}

// ─── Property 5f: Family-only string (no number) throws ──────────────────────
// A single character that is a valid one-letter family ('O', 'F', 'N', 'R')
// with no number following is malformed.
//
// **Validates: Requirements 6.4**

RC_GTEST_PROP(PropNamedGridRegistration, FamilyOnlyThrows, ()) {
    const char family = *genValidFamily();
    const std::string name(1, family);  // e.g. "O", "F", "N", "R"

    RC_ASSERT_THROWS_AS((void)NamedGridRegistry::parse(name), std::invalid_argument);
    RC_ASSERT(!NamedGridRegistry::is_registered(name));
}

// ─── Property 5g: registered_families() returns exactly {'C','F','G','N','O','R'} ─
// Verify the sorted set of registered families is always the same regardless
// of how many times it is called.
//
// **Validates: Requirements 6.3**

RC_GTEST_PROP(PropNamedGridRegistration, RegisteredFamiliesAreCFGNOR, ()) {
    auto families = NamedGridRegistry::registered_families();

    // Must be sorted
    RC_ASSERT(std::is_sorted(families.begin(), families.end()));

    // Must contain exactly C, F, G, N, O, R
    const std::vector<char> expected = {'C', 'F', 'G', 'N', 'O', 'R'};
    RC_ASSERT(families == expected);
}

// ─── Property 5h: Registered GRIB grid numbers parse successfully ────────────
// Verify that all registered NOAA GRIB grid numbers (case-insensitive) parse
// successfully as family 'G' with the correct number.
RC_GTEST_PROP(PropNamedGridRegistration, RegisteredGribGridsParseCorrectly, ()) {
    const std::vector<int> registered_numbers = {3,   4,   87,  88,  90,  91,  92,  130, 132, 138, 139, 140, 145, 146, 147, 148,
                                                 150, 151, 160, 163, 179, 184, 187, 188, 189, 197, 198, 200, 201, 202, 203, 205,
                                                 206, 207, 209, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                                                 224, 226, 227, 236, 237, 240, 241, 242, 245, 246, 247, 249, 252};
    const int number = *rc::gen::elementOf(registered_numbers);

    // Test case-insensitivity: randomly choose "grid", "GRID", "Grid"
    const std::string prefix = *rc::gen::element<std::string>("grid", "GRID", "Grid");
    const std::string name = prefix + std::to_string(number);

    NamedGridRegistry::ParsedName parsed = NamedGridRegistry::parse(name);
    RC_ASSERT(parsed.family == 'G');
    RC_ASSERT(parsed.number == number);
    RC_ASSERT(NamedGridRegistry::is_registered(name));
}

// ─── Property 5i: Unregistered GRIB grid numbers throw std::invalid_argument ─
// Verify that any grid numbers not in the registered GRIB list throw.
RC_GTEST_PROP(PropNamedGridRegistration, UnregisteredGribGridsThrow, ()) {
    const std::vector<int> registered_numbers = {3,   4,   87,  88,  90,  91,  92,  130, 132, 138, 139, 140, 145, 146, 147, 148,
                                                 150, 151, 160, 163, 179, 184, 187, 188, 189, 197, 198, 200, 201, 202, 203, 205,
                                                 206, 207, 209, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                                                 224, 226, 227, 236, 237, 240, 241, 242, 245, 246, 247, 249, 252};
    const int number = *rc::gen::suchThat(rc::gen::inRange(1, 1000), [&](int n) {
        return std::find(registered_numbers.begin(), registered_numbers.end(), n) == registered_numbers.end();
    });

    const std::string name = "grid" + std::to_string(number);

    RC_ASSERT_THROWS_AS((void)NamedGridRegistry::parse(name), std::invalid_argument);
    RC_ASSERT(!NamedGridRegistry::is_registered(name));
}

}  // namespace
