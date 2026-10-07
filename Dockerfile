# Base image with deep native C++20 support
FROM ubuntu:24.04

# Prevent interactive tzdata prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Core HPC & C++20 Toolchain
RUN apt-get update && apt-get install -y \
    build-essential \
    clang-format \
    clang-tidy \
    g++-13 \
    gcc-13 \
    gfortran-13 \
    libomp-dev \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    openmpi-bin \
    libopenmpi-dev \
    libgtest-dev \
    libproj-dev \
    libnetcdf-dev \
    libnetcdf-mpi-dev \
    libnetcdf-c++4-dev \
    catch2 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# AMIO (in-tree since the 002 absorption) needs the parallel netCDF headers
# (netcdf_par.h, from libnetcdf-mpi-dev) and the netCDF-C++4 API
# (libnetcdf-c++4-dev) for its NetCDF_Driver, and Catch2 v3 for its property-
# based test suite. RapidCheck is built from source in step 4 below; AMIO's
# PBT gate skips gracefully if Catch2 or RapidCheck is absent.

# CDO backs the AXIS-vs-CDO comparison tests (Stage 6). It is packaged for
# x86_64 Ubuntu (CI) but not on arm64 ports (Apple Silicon dev boxes), where
# the conda-forge build in Dockerfile-Benchmarks provides it instead.
RUN apt-get update \
    && (apt-get install -y --no-install-recommends cdo \
        || echo "cdo unavailable for $(uname -m): run benchmarks via Dockerfile-Benchmarks") \
    && rm -rf /var/lib/apt/lists/*

# Set GCC-13 as the default compiler (C, C++, and Fortran)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-13 100

# 2. Compile and Install Google Test globally
RUN cd /usr/src/gtest \
    && cmake CMakeLists.txt \
    && make \
    && cp lib/*.a /usr/lib/ \
    && mkdir -p /usr/local/lib/gtest/ \
    && ln -s /usr/lib/libgtest.a /usr/local/lib/gtest/libgtest.a \
    && ln -s /usr/lib/libgtest_main.a /usr/local/lib/gtest/libgtest_main.a

# 3. Clone and install Kokkos 5.1 (C++20 minimum, OpenMP backend for local CPU testing)
# We install this globally so CMake's find_package(Kokkos) works out of the box
RUN git clone -b 5.1.1 https://github.com/kokkos/kokkos.git /tmp/kokkos \
    && cd /tmp/kokkos \
    && cmake -B build \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build \
    && rm -rf /tmp/kokkos

# 4. Clone and install RapidCheck (property-based testing) globally
# Includes the GTest integration extra so CMake's find_package(rapidcheck)
# exposes both the rapidcheck and rapidcheck_gtest targets out of the box.
RUN git clone https://github.com/emil-e/rapidcheck.git /tmp/rapidcheck \
    && cd /tmp/rapidcheck \
    && cmake -B build \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DRC_ENABLE_GTEST=ON \
    -DRC_INSTALL_ALL_EXTRAS=ON \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build \
    && rm -rf /tmp/rapidcheck

# 5. Clone and install KokkosKernels (sparse linear algebra — spmv, sort)
# Provides hardware-tuned SpMV (cuSPARSE on GPU, MKL on CPU) for AXIS CSR apply.
# Only the Sparse component is enabled — no BLAS/LAPACK/Batched/Graph/ODE.
RUN git clone --depth 1 https://github.com/kokkos/kokkos-kernels.git /tmp/kokkos-kernels \
    && cd /tmp/kokkos-kernels \
    && cmake -B build \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DKokkosKernels_ENABLE_ALL_COMPONENTS=OFF \
    -DKokkosKernels_ENABLE_COMPONENT_SPARSE=ON \
    -DKokkosKernels_ENABLE_COMPONENT_BLAS=OFF \
    -DKokkosKernels_ENABLE_COMPONENT_GRAPH=OFF \
    -DKokkosKernels_ENABLE_COMPONENT_BATCHED=OFF \
    -DKokkosKernels_ENABLE_COMPONENT_LAPACK=OFF \
    -DKokkosKernels_ENABLE_COMPONENT_ODE=OFF \
    -DKokkosKernels_ADD_DEFAULT_ETI=OFF \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build \
    && rm -rf /tmp/kokkos-kernels

# 6. Workspace Setup
WORKDIR /workspace/helm-project

# Provide uv for Python env management
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# Default command to keep the container alive if run detached
CMD ["/bin/bash"]
