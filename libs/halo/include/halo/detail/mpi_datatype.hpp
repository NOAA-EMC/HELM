#ifndef HALO_DETAIL_MPI_DATATYPE_HPP
#define HALO_DETAIL_MPI_DATATYPE_HPP

/// @file halo/detail/mpi_datatype.hpp
/// @brief Compile-time MPI_Datatype resolution for C++ value types.
///
/// Provides a single canonical helper for mapping C++ arithmetic types to
/// their corresponding MPI_Datatype. Used by exchange functions, persistent
/// handles, and structured exchange templates.

#include <mpi.h>

#include <type_traits>

namespace halo::detail {

template <typename T>
inline constexpr bool has_native_mpi_type_v =
    std::is_same_v<T, double> || std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned long long> || std::is_same_v<T, char>;

/// @brief Resolve the MPI_Datatype for a given C++ value type.
///
/// Maps common arithmetic types to their MPI equivalents. Falls back to
/// MPI_BYTE for unknown types (caller must scale the element count by
/// sizeof(T) when using the BYTE fallback).
///
/// @tparam T The C++ arithmetic type.
/// @return The corresponding MPI_Datatype.
template <typename T>
inline MPI_Datatype mpi_datatype_for() noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, int>) {
        return MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        return MPI_LONG;
    } else if constexpr (std::is_same_v<T, long long>) {
        return MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<T, char>) {
        return MPI_CHAR;
    } else {
        return MPI_BYTE;
    }
}

}  // namespace halo::detail

#endif  // HALO_DETAIL_MPI_DATATYPE_HPP
