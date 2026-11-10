#pragma once

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace axis::test {

class KokkosEnvironment final : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }

    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};

// An inline variable gives the test executable one shared registration even
// when this header is included by many test translation units.
inline auto *const kokkos_environment = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

}  // namespace axis::test
