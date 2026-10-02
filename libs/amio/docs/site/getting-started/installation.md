# Installation

AMIO can be built from source using CMake, installed via Spack, or run inside a
Docker development container.

## Prerequisites

- **CMake** ≥ 3.20
- **C++ compiler** with C++20 support (GCC ≥ 11, Clang ≥ 14)
- **Fortran compiler** with Fortran 2003 support (gfortran ≥ 11)
- **MPI** (optional, for parallel I/O)

### Backend Dependencies

| Backend | Required Library | Notes |
|---------|-----------------|-------|
| NetCDF-4 | netCDF-cxx4 + Parallel HDF5 | Required for `.nc` output |
| Zarr v3 | TensorStore **or** netCDF-c with NCZarr | Falls back to NCZarr if TensorStore unavailable |
| GRIB2 | nceplibs-g2c | Required for GRIB2 output |

## CMake Build

```bash
# Clone the repository
git clone https://github.com/NOAA-EMC/AMIO.git
cd AMIO

# Configure
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMIO_BUILD_TESTING=ON \
  -DAMIO_BUILD_EXAMPLES=ON \
  -DAMIO_BUILD_DOCS=ON

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Install
make install
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `AMIO_BUILD_TESTING` | `OFF` | Build unit and property-based tests |
| `AMIO_BUILD_DOCS` | `OFF` | Build documentation targets |
| `AMIO_BUILD_EXAMPLES` | `OFF` | Build usage examples |
| `AMIO_FORCE_NCZARR` | `OFF` | Skip TensorStore discovery; use NCZarr fallback |

## Spack

AMIO provides a Spack recipe for integration with [spack-stack](https://github.com/JCSDA/spack-stack):

```bash
# Add the AMIO package repository
spack repo add packages/

# Install with default settings (TensorStore + MPI)
spack install amio

# Install without TensorStore (air-gapped / NCZarr fallback)
spack install amio~tensorstore

# Install with MPI support
spack install amio+mpi
```

### Spack Variants

| Variant | Default | Description |
|---------|---------|-------------|
| `+tensorstore` | ON | Use TensorStore for Zarr v3 (requires network) |
| `~tensorstore` | — | Force NCZarr fallback mode |
| `+mpi` | ON | Enable MPI parallel I/O |

## Docker (Development Container)

A devcontainer configuration is provided for VS Code / GitHub Codespaces:

```bash
# Build the development image
docker build -t amio-dev .devcontainer/

# Run interactively
docker run -it --rm -v $(pwd):/workspace amio-dev

# Inside the container, all dependencies are pre-installed
cd /workspace
mkdir build && cd build
cmake .. -DAMIO_BUILD_TESTING=ON -DAMIO_BUILD_EXAMPLES=ON
make -j$(nproc)
```

## Using AMIO in Your Project

After installation, use `find_package` in your CMakeLists.txt:

```cmake
find_package(AMIO REQUIRED)

# For C/C++ applications
target_link_libraries(my_app PRIVATE AMIO::amio_core)

# For Fortran applications
target_link_libraries(my_fortran_app PRIVATE AMIO::amio_fortran)
```

The exported targets automatically set up include paths and link dependencies.
No manual `-I` or `-L` flags are needed.
