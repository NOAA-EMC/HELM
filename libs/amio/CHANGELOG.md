# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-28

### Added
- Initial Phase 1 implementation of AMIO
- C99 public API with Fortran 2003 iso_c_binding wrappers
- NetCDF-4 backend driver (Parallel HDF5 + MPI-IO)
- Zarr v3 backend driver (NCZarr fallback mode)
- GRIB2 backend driver (nceplibs-g2c)
- Staging pool with configurable buffer management
- Worker pool with per-(dataset, variable) ordering
- Prefetch queue for read-ahead optimization
- eckit::Factory-based backend dispatcher
- Property-based test suite (28 properties, 100 iterations each)
- Doxygen + Moxygen + MkDocs documentation pipeline
- Docker development container with full dependency stack
- CMake build system with Spack recipe
- Header isolation guarantees (C99-only public surface)
