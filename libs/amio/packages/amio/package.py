# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
####################################################################
# AMIO Spack recipe (task 1.3)
#
# This recipe is vendored alongside the source so that NOAA spack-stack
# integrators can install AMIO by adding `packages/` to their Spack
# repository configuration:
#
#     spack repo add /path/to/AMIO/packages
#     spack install amio
#     spack install amio~tensorstore        # air-gapped / NCZarr fallback
#
# The variants and conflicts declared here implement requirements
# R13.1 (CMake-package consumability) and R13.2 (no Rust / Cargo / Go
# in the dependency closure).  They feed the same configure-time flow
# documented in design.md (TensorStore discovery vs NCZarr fallback)
# by wiring `~tensorstore` to `-DAMIO_FORCE_NCZARR=ON`.
####################################################################

from spack.package import *


class Amio(CMakePackage):
    """Asynchronous Multidimensional I/O for NOAA NWS Earth system models.

    AMIO decouples scientific compute loops from physical storage
    architectures by exposing a flat C99 FFI surface plus a Fortran
    2003 iso_c_binding wrapper.  Internally it owns a staging memory
    pool, a look-ahead prefetch queue, and a native-thread worker
    pool that dispatches to one of three pluggable backend drivers:

      * NetCDF-4 (Parallel HDF5 + MPI-IO)
      * Zarr v3  (Google TensorStore; netCDF-c NCZarr fallback for
                  air-gapped sites)
      * GRIB2    (nceplibs-g2c)

    Public headers are C99-only and never transitively expose eckit,
    TensorStore, netCDF-cxx4, or nceplibs-g2c headers to the host
    application's compile path.
    """

    homepage = "https://github.com/noaa-emc/AMIO"
    url = "https://github.com/noaa-emc/AMIO/archive/refs/tags/v0.1.0.tar.gz"
    git = "https://github.com/noaa-emc/spack.git"

    maintainers("NOAA-EMC")

    license("Apache-2.0")

    version("main", branch="main")
    version(
        "0.1.0",
        sha256="0000000000000000000000000000000000000000000000000000000000000000",
    )

    ####################################################################
    # Variants
    #
    # `+tensorstore` (default ON)  -> build with TensorStore Zarr v3
    # `~tensorstore`               -> NCZarr fallback only (air-gapped)
    # `+mpi`         (default ON)  -> Parallel HDF5 / MPI communicator splits
    # `+shared`      (default ON)  -> build libamio.so (vs static)
    ####################################################################

    variant(
        "tensorstore",
        default=True,
        description=(
            "Use Google TensorStore for Zarr v3 (Bazel toolchain required "
            "at build time). Disable for air-gapped sites; the build "
            "falls back to netCDF-c NCZarr (no sharding, no cloud "
            "object-store transports)."
        ),
    )
    variant(
        "mpi",
        default=True,
        description=(
            "Build with MPI support, required for Parallel HDF5 writes and MPI communicator splits onto dedicated I/O ranks."
        ),
    )
    variant(
        "shared",
        default=True,
        description="Build libamio as a shared library.",
    )

    ####################################################################
    # Dependencies
    #
    # All third-party libraries are PRIVATE-linked into libamio so they
    # do not appear on the consumer's compile path (R13.6).  The Spack
    # recipe still declares the full closure so spack-stack solvers can
    # build a consistent stack.
    ####################################################################

    depends_on("cmake@3.20:", type="build")

    # NO-ECKIT (Requirement 12.1): AMIO neither links nor exports eckit. The
    # core library never calls find_package(eckit) and libamio.so carries zero
    # eckit symbols, so the recipe MUST NOT declare an eckit dependency. AMIO
    # has no optional eckit integration: there is no AMIO_HAS_ECKIT build
    # option and no eckit code path anywhere in src/ or the test suite.
    depends_on("kokkos-mdspan")  # header-only Memory_View
    depends_on("netcdf-cxx4")
    depends_on("netcdf-c +nczarr +blosc +zstd")  # required for NCZarr path
    depends_on("nceplibs-g2c")
    depends_on("mpi", when="+mpi")
    depends_on("tensorstore", when="+tensorstore")

    # netcdf-cxx4 must itself be MPI-aware whenever AMIO is, otherwise
    # the NetCDF_Driver cannot open Parallel-HDF5 datasets (R7.1).
    depends_on("netcdf-c +mpi", when="+mpi")
    depends_on("netcdf-cxx4 +mpi", when="+mpi")

    ####################################################################
    # Hard prohibitions (Requirement R13.2)
    #
    # AMIO must never pull a Rust, Cargo, or Go toolchain into its
    # dependency closure -- transitively or otherwise.  The CI check in
    # task 1.4 verifies this via `spack spec amio | grep -E 'rust|cargo|go@'`
    # which must produce zero lines.  The `conflicts` directives below
    # cause the Spack solver to refuse a concretization that would
    # introduce any of those toolchains, surfacing the violation at
    # `spack install` time rather than at runtime.
    ####################################################################

    conflicts(
        "^rust",
        msg="AMIO disallows Rust toolchains in its dependency closure "
        "(R13.2). Re-concretize with --reuse or remove the spec that "
        "introduced ^rust.",
    )
    conflicts(
        "^cargo",
        msg="AMIO disallows Cargo in its dependency closure (R13.2).",
    )
    conflicts(
        "^go",
        msg="AMIO disallows Go in its dependency closure (R13.2).",
    )

    ####################################################################
    # Configure
    ####################################################################

    def cmake_args(self):
        args = [
            self.define_from_variant("AMIO_WITH_MPI", "mpi"),
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
        ]

        # Wire `~tensorstore` -> `-DAMIO_FORCE_NCZARR=ON` so air-gapped
        # installs short-circuit TensorStore discovery and route directly
        # to the NCZarr branch of the configure flowchart documented in
        # design.md (R13.3, R13.5).  Conversely, an explicit `+tensorstore`
        # spec must NOT force NCZarr -- the build will hard-fail at
        # configure time if TensorStore is unavailable, which is the
        # intended behaviour for stacks that requested it.
        if self.spec.satisfies("~tensorstore"):
            args.append(self.define("AMIO_FORCE_NCZARR", True))
        else:
            args.append(self.define("AMIO_FORCE_NCZARR", False))

        return args
