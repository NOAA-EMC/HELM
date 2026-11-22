#ifndef HALO_COMMUNICATOR_HPP
#define HALO_COMMUNICATOR_HPP

/// @file communicator.hpp
/// @brief RAII wrapper for MPI_Comm handles.
///
/// The Communicator class takes exclusive ownership of an MPI_Comm handle and
/// guarantees cleanup via MPI_Comm_free on all exit paths (including exceptions).
///
/// @note The handle() accessor returns the raw MPI_Comm by value without
/// transferring ownership. Callers must ensure the Communicator outlives any
/// use of the returned handle.

#include <mpi.h>

#include <stdexcept>

namespace halo {

/// @brief RAII wrapper around an MPI_Comm handle.
///
/// Manages the lifetime of an MPI communicator. Predefined communicators
/// (MPI_COMM_WORLD, MPI_COMM_SELF) and MPI_COMM_NULL are stored but never freed.
/// Move-only semantics enforce unique ownership.
///
/// @note HALO-owned communicators (non-predefined, non-null) have their error
/// handler set to MPI_ERRORS_RETURN so that MPI errors are returned as error
/// codes rather than invoking the default abort behavior. Predefined
/// communicators (WORLD, SELF) are NOT modified to avoid side effects on the
/// application's global error handling strategy.
class Communicator {
   public:
    /// @brief Construct from a raw MPI_Comm handle. Takes exclusive ownership.
    /// @param comm The MPI communicator handle to own.
    explicit Communicator(MPI_Comm comm) noexcept;

    /// @brief Destructor. Calls MPI_Comm_free if the handle is not NULL,
    /// not predefined (WORLD/SELF), and MPI has not been finalized.
    ~Communicator();

    /// @brief Move constructor. Transfers ownership; source becomes MPI_COMM_NULL.
    Communicator(Communicator &&other) noexcept;

    /// @brief Move assignment. Transfers ownership; source becomes MPI_COMM_NULL.
    Communicator &operator=(Communicator &&other) noexcept;

    /// @brief Copy construction is deleted (unique ownership).
    Communicator(const Communicator &) = delete;

    /// @brief Copy assignment is deleted (unique ownership).
    Communicator &operator=(const Communicator &) = delete;

    /// @brief Access the raw MPI_Comm handle without transferring ownership.
    /// @return The owned MPI_Comm value.
    /// @note Does not extend the lifetime of this Communicator. Callers must
    /// ensure this object outlives any use of the returned handle.
    [[nodiscard]] MPI_Comm handle() const noexcept;

    /// @brief Query the rank of the calling process within this communicator.
    /// @return The integer rank.
    /// @throws std::runtime_error if MPI_Comm_rank fails.
    [[nodiscard]] int rank() const;

    /// @brief Query the total number of processes in this communicator.
    /// @return The integer size.
    /// @throws std::runtime_error if MPI_Comm_size fails.
    [[nodiscard]] int size() const;

    /// @brief Split this communicator into a sub-communicator.
    /// @param color Control of subset assignment (MPI_UNDEFINED yields NULL comm).
    /// @param key Control of rank assignment within the new communicator.
    /// @return A new Communicator owning the split result.
    /// @throws std::runtime_error if MPI_Comm_split fails.
    [[nodiscard]] Communicator split(int color, int key) const;

    /// @brief Duplicate this communicator.
    /// @return A new Communicator owning the duplicated handle.
    /// @throws std::runtime_error if MPI_Comm_dup fails.
    [[nodiscard]] Communicator duplicate() const;

    /// @brief Duplicate a borrowed raw communicator without adopting it.
    /// @param comm Any live MPI communicator handle owned by the caller.
    /// @return A new Communicator owning the duplicated handle.
    /// @throws std::runtime_error if MPI_Comm_dup fails.
    [[nodiscard]] static Communicator duplicate_of(MPI_Comm comm);

   private:
    MPI_Comm comm_{MPI_COMM_NULL};

    /// @brief Check if the held communicator is predefined (WORLD or SELF).
    [[nodiscard]] bool is_predefined() const noexcept;
};

}  // namespace halo

#endif  // HALO_COMMUNICATOR_HPP
