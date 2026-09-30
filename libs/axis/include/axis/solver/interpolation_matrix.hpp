// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_INTERPOLATION_MATRIX_HPP
#define AXIS_SOLVER_INTERPOLATION_MATRIX_HPP

/// @file axis/solver/interpolation_matrix.hpp
/// @brief Sparse interpolation operator in COO form (factorList + factorIndexList).
///
/// InterpolationMatrix stores the sparse weights and associated conservation
/// bookkeeping arrays (frac_a, frac_b, area_a, area_b) produced by
/// WeightGenerator. This is the exact data layout that ESMF weight files
/// express: factorList (S), factorIndexList col/row (src/dst), plus per-cell
/// fraction and area arrays for conservative normalization.
///
/// The matrix also supports conversion to CSR (Compressed Sparse Row) format
/// via to_csr(), enabling row-parallel SpMV without atomics.
///
/// Templated on a Kokkos MemorySpace (HELM Law #2: explicit placement, no UVM).
/// Light inline accessors live in this header; the .cpp provides explicit
/// template instantiations for common memory spaces.

#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <algorithm>
#include <axis/types.hpp>
#include <cstddef>

namespace axis::solver {

/// A (destination, source) index pair representing one nonzero entry in the
/// sparse interpolation matrix. Matches ESMF factorIndexList semantics where
/// col = source cell index and row = destination cell index.
struct IndexPair {
    index_t row;  ///< destination cell index
    index_t col;  ///< source cell index
};

/// Sparse interpolation operator dst = S · src in COO form.
///
/// For each nonzero k:
///   factor_list[k] = S(k)         (weight of the k-th nonzero)
///   factor_row[k]  = dst_index    (destination cell)
///   factor_col[k]  = src_index    (source cell)
///
/// Conservation bookkeeping arrays (populated for conservative methods):
///   frac_a[i] = fraction of source cell i covered by destination cells
///   frac_b[j] = fraction of destination cell j covered by source cells
///   area_a[i] = area of source cell i
///   area_b[j] = area of destination cell j
///
/// Optionally converts to CSR format via to_csr() for row-parallel apply:
///   row_ptr[n_dst+1] — row pointer offsets
///   col_idx[nnz]     — sorted column indices
///   csr_vals[nnz]    — values sorted to match col_idx
///
/// @tparam MemorySpace Kokkos memory space for internal array storage
///         (Kokkos::HostSpace, CudaSpace, HIPSpace, etc.)
template <class MemorySpace = Kokkos::HostSpace>
class InterpolationMatrix {
   public:
    using memory_space = MemorySpace;

    /// Default-construct an empty matrix.
    InterpolationMatrix() = default;

    /// Construct by adopting pre-built Kokkos arrays (moved in, no copy).
    ///
    /// @param factor_list  Interpolation weights [nnz]
    /// @param factor_row   Destination (row) indices [nnz]
    /// @param factor_col   Source (col) indices [nnz]
    /// @param frac_a       Source fractions [n_src]
    /// @param frac_b       Destination fractions [n_dst]
    /// @param area_a       Source cell areas [n_src]
    /// @param area_b       Destination cell areas [n_dst]
    /// @param n_src        Number of source cells (n_a in ESMF terms)
    /// @param n_dst        Number of destination cells (n_b in ESMF terms)
    InterpolationMatrix(Kokkos::View<double *, MemorySpace> factor_list, Kokkos::View<index_t *, MemorySpace> factor_row,
                        Kokkos::View<index_t *, MemorySpace> factor_col, Kokkos::View<double *, MemorySpace> frac_a,
                        Kokkos::View<double *, MemorySpace> frac_b, Kokkos::View<double *, MemorySpace> area_a,
                        Kokkos::View<double *, MemorySpace> area_b, std::size_t n_src, std::size_t n_dst)
        : factor_list_(std::move(factor_list)),
          factor_row_(std::move(factor_row)),
          factor_col_(std::move(factor_col)),
          frac_a_(std::move(frac_a)),
          frac_b_(std::move(frac_b)),
          area_a_(std::move(area_a)),
          area_b_(std::move(area_b)),
          n_src_(n_src),
          n_dst_(n_dst) {}

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Number of nonzero entries in the sparse matrix (n_s in ESMF terms).
    [[nodiscard]] std::size_t nnz() const noexcept {
        return factor_list_.extent(0);
    }

    /// Number of source cells (n_a in ESMF terms).
    [[nodiscard]] std::size_t n_src() const noexcept {
        return n_src_;
    }

    /// Number of destination cells (n_b in ESMF terms).
    [[nodiscard]] std::size_t n_dst() const noexcept {
        return n_dst_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Non-owning field_view accessors (zero-copy, layout_left)
    //
    // These return std::mdspan<const T, dextents, layout_left> directly
    // wrapping the internal Kokkos::View data pointers. No allocation or copy.
    // ─────────────────────────────────────────────────────────────────────────

    /// Interpolation weights S: [nnz].
    [[nodiscard]] field_view<const double, 1> factor_list() const noexcept {
        return field_view<const double, 1>{factor_list_.data(), factor_list_.extent(0)};
    }

    /// Source (column) indices: [nnz]. Each entry is the source cell index
    /// for the corresponding weight in factor_list.
    [[nodiscard]] field_view<const index_t, 1> factor_col() const noexcept {
        return field_view<const index_t, 1>{factor_col_.data(), factor_col_.extent(0)};
    }

    /// Destination (row) indices: [nnz]. Each entry is the destination cell
    /// index for the corresponding weight in factor_list.
    [[nodiscard]] field_view<const index_t, 1> factor_row() const noexcept {
        return field_view<const index_t, 1>{factor_row_.data(), factor_row_.extent(0)};
    }

    /// Combined (row, col) index pair accessor for the k-th nonzero.
    /// Provided for ESMF factorIndexList semantics compatibility.
    [[nodiscard]] IndexPair factor_index(std::size_t k) const noexcept {
        return IndexPair{factor_row_.data()[k], factor_col_.data()[k]};
    }

    /// Source fractions: [n_src]. frac_a[i] is the fraction of source cell i
    /// covered by the destination mesh.
    [[nodiscard]] field_view<const double, 1> frac_a() const noexcept {
        return field_view<const double, 1>{frac_a_.data(), frac_a_.extent(0)};
    }

    /// Destination fractions: [n_dst]. frac_b[j] is the fraction of
    /// destination cell j covered by the source mesh.
    [[nodiscard]] field_view<const double, 1> frac_b() const noexcept {
        return field_view<const double, 1>{frac_b_.data(), frac_b_.extent(0)};
    }

    /// Source cell areas: [n_src].
    [[nodiscard]] field_view<const double, 1> area_a() const noexcept {
        return field_view<const double, 1>{area_a_.data(), area_a_.extent(0)};
    }

    /// Destination cell areas: [n_dst].
    [[nodiscard]] field_view<const double, 1> area_b() const noexcept {
        return field_view<const double, 1>{area_b_.data(), area_b_.extent(0)};
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CSR (Compressed Sparse Row) interface
    //
    // Call to_csr() to convert from COO to CSR format. Once converted,
    // solver::apply can dispatch to the row-parallel path (no atomics).
    // The COO data remains accessible after conversion.
    // ─────────────────────────────────────────────────────────────────────────

    /// Convert the internal COO representation to CSR format.
    ///
    /// Sorts entries by (row, column) and builds the row_ptr, col_idx,
    /// and csr_vals arrays using Kokkos parallel patterns on the same
    /// MemorySpace as the matrix data.
    ///
    /// If already in CSR format (is_csr() == true), this is a no-op.
    void to_csr() {
        if (has_csr_) return;

        const auto n_nz = nnz();
        const auto n_rows = n_dst_;

        if (n_nz == 0) {
            // Empty matrix: just allocate zero-sized CSR arrays
            row_ptr_ = Kokkos::View<index_t *, MemorySpace>("csr_row_ptr", n_rows + 1);
            col_idx_ = Kokkos::View<index_t *, MemorySpace>("csr_col_idx", 0);
            csr_vals_ = Kokkos::View<double *, MemorySpace>("csr_vals", 0);
            // row_ptr is already zero-initialized by Kokkos
            has_csr_ = true;
            return;
        }

        // --- Step 1: Build permutation array sorted by (row, col) ---
        // We sort on the host side using a mirror for portability, then
        // apply the permutation to build CSR arrays on the target space.

        // Create host mirrors of row and col indices
        auto h_row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, factor_row_);
        auto h_col = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, factor_col_);
        auto h_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, factor_list_);

        // Build permutation indices on host
        Kokkos::View<index_t *, Kokkos::HostSpace> h_perm("perm", n_nz);
        Kokkos::parallel_for(
            "init_perm", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_nz),
            KOKKOS_LAMBDA(const std::size_t i) { h_perm(i) = static_cast<index_t>(i); });
        Kokkos::fence();

        // Sort permutation by (row, col) using std::sort on the host
        // This gives us the sorted order without moving the original COO data
        auto h_perm_ptr = h_perm.data();
        auto h_row_ptr = h_row.data();
        auto h_col_ptr = h_col.data();
        std::sort(h_perm_ptr, h_perm_ptr + n_nz, [h_row_ptr, h_col_ptr](index_t a, index_t b) {
            if (h_row_ptr[a] != h_row_ptr[b]) return h_row_ptr[a] < h_row_ptr[b];
            return h_col_ptr[a] < h_col_ptr[b];
        });

        // --- Step 2: Build CSR arrays on host ---
        Kokkos::View<index_t *, Kokkos::HostSpace> h_row_ptr_csr("h_csr_row_ptr", n_rows + 1);
        Kokkos::View<index_t *, Kokkos::HostSpace> h_col_idx("h_csr_col_idx", n_nz);
        Kokkos::View<double *, Kokkos::HostSpace> h_csr_vals("h_csr_vals", n_nz);

        // Count entries per row
        for (std::size_t k = 0; k < n_nz; ++k) {
            const auto row_idx = h_row.data()[h_perm.data()[k]];
            h_row_ptr_csr.data()[row_idx + 1]++;
        }

        // Exclusive prefix sum to get row_ptr
        for (std::size_t i = 0; i < n_rows; ++i) {
            h_row_ptr_csr.data()[i + 1] += h_row_ptr_csr.data()[i];
        }

        // Fill sorted col_idx and csr_vals using the permutation
        for (std::size_t k = 0; k < n_nz; ++k) {
            const auto orig_idx = h_perm.data()[k];
            h_col_idx.data()[k] = h_col.data()[orig_idx];
            h_csr_vals.data()[k] = h_vals.data()[orig_idx];
        }

        // --- Step 3: Copy CSR arrays to target MemorySpace ---
        row_ptr_ = Kokkos::View<index_t *, MemorySpace>("csr_row_ptr", n_rows + 1);
        col_idx_ = Kokkos::View<index_t *, MemorySpace>("csr_col_idx", n_nz);
        csr_vals_ = Kokkos::View<double *, MemorySpace>("csr_vals", n_nz);

        Kokkos::deep_copy(row_ptr_, h_row_ptr_csr);
        Kokkos::deep_copy(col_idx_, h_col_idx);
        Kokkos::deep_copy(csr_vals_, h_csr_vals);

        has_csr_ = true;
    }

    /// Returns true if the CSR representation has been built via to_csr().
    [[nodiscard]] bool is_csr() const noexcept {
        return has_csr_;
    }

    /// CSR row pointer array: [n_dst+1].
    /// row_ptr[j] to row_ptr[j+1] spans the entries for destination cell j.
    /// @pre is_csr() == true
    [[nodiscard]] Kokkos::View<const index_t *, MemorySpace> row_ptr() const noexcept {
        return row_ptr_;
    }

    /// CSR column index array: [nnz]. Sorted by column within each row.
    /// @pre is_csr() == true
    [[nodiscard]] Kokkos::View<const index_t *, MemorySpace> col_idx() const noexcept {
        return col_idx_;
    }

    /// CSR values array: [nnz]. Entries correspond to col_idx positions.
    /// @pre is_csr() == true
    [[nodiscard]] Kokkos::View<const double *, MemorySpace> csr_values() const noexcept {
        return csr_vals_;
    }

    /// Lazily-built, cached KokkosKernels CrsMatrix wrapping the CSR arrays.
    ///
    /// The interpolation weights never change after to_csr(), so the derived
    /// KokkosSparse::CrsMatrix operator (row_map/entries/vals + graph) can be
    /// built once and reused across every apply() call. On first invocation
    /// this deep-copies the CSR structure into non-const backing Views (held as
    /// mutable members so they outlive the returned matrix, whose graph does
    /// not own the row_map/entries) and constructs the CrsMatrix; subsequent
    /// calls return the cached matrix unchanged.
    ///
    /// KokkosKernels is a required dependency of AXIS (enforced in
    /// libs/axis/CMakeLists.txt), so this accessor is always available.
    ///
    /// Uses the same index_t/device types as axis::solver::apply.
    ///
    /// @pre is_csr() == true
    using kk_device_t = Kokkos::Device<typename MemorySpace::execution_space, MemorySpace>;
    using kk_crs_matrix_t = KokkosSparse::CrsMatrix<double, index_t, kk_device_t, void, index_t>;

    [[nodiscard]] const kk_crs_matrix_t &kk_crs_matrix() const {
        if (!kk_csr_built_) {
            using graph_t = typename kk_crs_matrix_t::staticcrsgraph_type;

            kk_row_map_ = Kokkos::View<index_t *, MemorySpace>("kk_row_map", row_ptr_.extent(0));
            Kokkos::deep_copy(kk_row_map_, row_ptr_);
            kk_entries_ = Kokkos::View<index_t *, MemorySpace>("kk_entries", col_idx_.extent(0));
            Kokkos::deep_copy(kk_entries_, col_idx_);
            kk_vals_ = Kokkos::View<double *, MemorySpace>("kk_vals", csr_vals_.extent(0));
            Kokkos::deep_copy(kk_vals_, csr_vals_);

            graph_t graph(kk_entries_, kk_row_map_);
            kk_crs_matrix_ = kk_crs_matrix_t("axis_spmv", static_cast<index_t>(n_src_), kk_vals_, graph);
            kk_csr_built_ = true;
        }
        return kk_crs_matrix_;
    }

    /// @brief Attach a per-destination unmapped mask (0 = mapped, 1 = unmapped).
    ///
    /// Populated by WeightGenerator when RegridConfig::unmapped ==
    /// UnmappedAction::Mask. Callers can then distinguish "no data" rows
    /// (zero weights + mask==1) from genuine zeros (mask==0).
    void set_unmapped_mask(Kokkos::View<int *, MemorySpace> mask) {
        if (mask.extent(0) != n_dst_) {
            throw std::invalid_argument("InterpolationMatrix::set_unmapped_mask: mask.extent(0) (" + std::to_string(mask.extent(0)) + ") != n_dst (" +
                                        std::to_string(n_dst_) + ")");
        }
        unmapped_mask_ = std::move(mask);
    }

    /// @brief True if an unmapped mask has been attached (UnmappedAction::Mask).
    [[nodiscard]] bool has_unmapped_mask() const noexcept {
        return unmapped_mask_.extent(0) > 0;
    }

    /// Per-destination unmapped mask: [n_dst]. 1 = destination cell has no
    /// source coverage, 0 = covered. Empty view when no mask was attached.
    /// @see has_unmapped_mask()
    [[nodiscard]] field_view<const int, 1> unmapped_mask() const noexcept {
        return field_view<const int, 1>{unmapped_mask_.data(), unmapped_mask_.extent(0)};
    }

    /// Internal Kokkos::View accessor for the unmapped mask (serialization).
    [[nodiscard]] const auto &unmapped_mask_view() const noexcept {
        return unmapped_mask_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal Kokkos::View accessors (for WeightGenerator, apply, and
    // conservation accounting that need direct View access)
    // ─────────────────────────────────────────────────────────────────────────

    [[nodiscard]] const auto &factor_list_view() const noexcept {
        return factor_list_;
    }
    [[nodiscard]] const auto &factor_row_view() const noexcept {
        return factor_row_;
    }
    [[nodiscard]] const auto &factor_col_view() const noexcept {
        return factor_col_;
    }
    [[nodiscard]] const auto &frac_a_view() const noexcept {
        return frac_a_;
    }
    [[nodiscard]] const auto &frac_b_view() const noexcept {
        return frac_b_;
    }
    [[nodiscard]] const auto &area_a_view() const noexcept {
        return area_a_;
    }
    [[nodiscard]] const auto &area_b_view() const noexcept {
        return area_b_;
    }

   private:
    // COO storage
    Kokkos::View<double *, MemorySpace> factor_list_;  ///< weights [nnz]
    Kokkos::View<index_t *, MemorySpace> factor_row_;  ///< destination indices [nnz]
    Kokkos::View<index_t *, MemorySpace> factor_col_;  ///< source indices [nnz]
    Kokkos::View<double *, MemorySpace> frac_a_;       ///< source fractions [n_src]
    Kokkos::View<double *, MemorySpace> frac_b_;       ///< destination fractions [n_dst]
    Kokkos::View<double *, MemorySpace> area_a_;       ///< source cell areas [n_src]
    Kokkos::View<double *, MemorySpace> area_b_;       ///< destination cell areas [n_dst]
    std::size_t n_src_{0};
    std::size_t n_dst_{0};

    // Optional per-destination unmapped mask (populated for UnmappedAction::Mask):
    // 1 = destination cell has no source coverage, 0 = covered. Empty when unset.
    Kokkos::View<int *, MemorySpace> unmapped_mask_;

    // CSR storage (populated by to_csr())
    Kokkos::View<index_t *, MemorySpace> row_ptr_;  ///< row pointers [n_dst+1]
    Kokkos::View<index_t *, MemorySpace> col_idx_;  ///< column indices [nnz]
    Kokkos::View<double *, MemorySpace> csr_vals_;  ///< values [nnz]
    bool has_csr_{false};                           ///< CSR representation available

    // Lazily-built cache of the derived KokkosKernels CrsMatrix. Because apply()
    // takes a const InterpolationMatrix&, these are mutable. The row_map/entries
    // backing Views must persist as members: the CrsMatrix graph references them
    // without owning copies.
    mutable Kokkos::View<index_t *, MemorySpace> kk_row_map_;  ///< non-const row_map backing the cached graph
    mutable Kokkos::View<index_t *, MemorySpace> kk_entries_;  ///< non-const entries backing the cached graph
    mutable Kokkos::View<double *, MemorySpace> kk_vals_;      ///< non-const values backing the cached matrix
    mutable kk_crs_matrix_t kk_crs_matrix_;                    ///< cached KokkosKernels CrsMatrix
    mutable bool kk_csr_built_{false};                         ///< true once the cached matrix is built
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_INTERPOLATION_MATRIX_HPP
