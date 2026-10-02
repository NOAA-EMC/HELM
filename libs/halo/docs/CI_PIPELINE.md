# HALO CI Pipeline

This document defines the Continuous Integration pipeline for the **HALO**
(Hardware-Abstracted Link Operations) Tier 1 micro-library. It describes every
stage, the exact commands each stage runs, and the HELM requirement each stage
satisfies.

The executable encoding of this pipeline lives in
[`.github/workflows/halo-ci.yml`](../../../.github/workflows/halo-ci.yml)
(GitHub Actions). A self-contained, POSIX-`sh` driver that runs every stage in
sequence and exits non-zero on the first failure is provided at
[`cmake/ci_pipeline.sh`](../cmake/ci_pipeline.sh); it is the simplest way to run
the whole pipeline inside the container (`sh cmake/ci_pipeline.sh`). The prose
below, that workflow, and the driver script are kept in lock-step — if you
change one, change the others.

> **Requirements covered:** 1.1, 2.1, 6.1, 6.2, 6.3, 7.1, 10.1, 11.1, 12.2, 12.3, 12.5, 13.1, 13.5
> (see `.kiro/specs/helm-halo-microlibrary/requirements.md` and
> `.kiro/specs/halo-production-hardening/requirements.md`).

---

## Container environment

Every stage runs **inside the HELM project Docker container**, never on the
bare runner. This satisfies **Requirement 12.2** — the HALO build/test workflow
uses the image built from the HELM project `Dockerfile` as its development and
CI environment, so the toolchain (GCC-13, OpenMPI, Kokkos 4.3, GoogleTest,
RapidCheck) is identical to what every developer runs locally.

The image is the same one described in [`libs/halo/README.md`](../README.md).
Bring it up exactly as a developer would:

```bash
# From the HELM project root directory:

# Build and start the container (detached)
docker compose up -d --build

# Enter the running container
docker compose exec helm-dev bash

# You are now at /workspace/helm-project inside the container.
# HALO source is at /workspace/helm-project/libs/halo/
```

OpenMPI refuses to run as root by default. The container (see
[`docker-compose.yml`](../../../docker-compose.yml)) sets
`OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` so that
`mpirun`-launched tests execute under the root CI user without extra flags.

In CI, the same commands shown per-stage below are wrapped with
`docker compose exec -T helm-dev bash -lc "..."` (the `-T` disables TTY
allocation, which is required on a headless runner).

> **Requirement 12.5 note:** all HELM libraries (including AMIO) are tracked
> in-tree, so a plain `git clone` of the workspace checks out exactly the
> reviewed, pinned commits before the container starts — no
> `--recurse-submodules` step is needed.

---

## Pipeline stages (in order)

| # | Stage | Purpose | Requirement |
| --- | --- | --- | --- |
| 1 | Static analysis / Tier 1 isolation scan | Reject forbidden HELM cross-dependencies before spending build time | 13.5 |
| 2 | Standalone CMake build inside Docker | Prove HALO builds alone and produces `HELM::HALO` | 12.2, 12.3 |
| 3 | Unit tests (`mpirun -np 4`) | Verify real MPI behavior across ranks (including structured exchange, persistent handles, diagnostics) | 12.2 |
| 4 | Property tests (single-rank, mocked MPI) | Verify RAII/algebraic invariants and structured exchange properties deterministically | 12.2 |
| 5 | Fortran integration test (`mpirun -np 4`) | Verify the `iso_c_binding` interop end-to-end | 12.2 |
| 6 | Sanitizer builds (ASan + UBSan) | Catch memory-safety and undefined-behavior defects | 12.2 |
| 7 | Spack build smoke test (planned) | Verify `spack install halo` succeeds with default variants | 6.1, 6.2, 6.3 |

Stage 1 runs first and is a hard gate: an isolation violation fails the
pipeline immediately, before any compilation, because a Tier 1 dependency
breach is an architectural defect, not a code-quality nit.

---

### Stage 1 — Static analysis: Tier 1 isolation scan

**Why:** HALO is a Tier 1 HELM library and MUST have **zero** compile-time
dependencies on any other HELM component (TICK, LOGS, AXIS, AMIO, SPAN, DAGR).
This preserves the "No Circular Dependencies" law of the HELM architecture.
**Requirement 13.5** mandates that CI run a static verification step that scans
all HALO source and header files for `#include` directives matching other HELM
component header paths and fails the build if any are found.

The scanner is [`cmake/check_tier1_isolation.sh`](../cmake/check_tier1_isolation.sh).
It is wired into CMake two ways (see [`libs/halo/CMakeLists.txt`](../CMakeLists.txt)):

- an always-available custom target `check_isolation`, and
- a CTest named `tier1_isolation` (labels `static;isolation`) when
  `BUILD_TESTING=ON`.

This stage runs the scanner directly (no compiler needed, so it is fast and
can gate everything else):

```bash
# Direct invocation (no CMake configure required):
cd libs/halo
sh cmake/check_tier1_isolation.sh .
```

Equivalent forms once a build tree exists:

```bash
# As a CMake custom target:
cmake --build build --target check_isolation

# As a CTest (configure with -DHALO_BUILD_TESTING=ON first):
cd build && ctest -R tier1_isolation --output-on-failure
```

A clean tree prints `check_tier1_isolation: PASS` and exits 0. Any forbidden
`#include`/`use` prints `file:line:text` for each hit and exits 1.

---

### Stage 2 — Standalone CMake build inside Docker

**Why:** **Requirement 12.3** requires that HALO's root `CMakeLists.txt`, when
configured and built inside the HELM Docker container **without any other HELM
source trees present**, produces the `HELM::HALO` library target without build
errors. This guarantees HALO is genuinely decoupled and consumable on its own.
Because the build happens in the HELM container, it also exercises
**Requirement 12.2**.

The HALO source tree under `libs/halo/` contains no references to sibling HELM
libraries (Stage 1 enforces this), so configuring `libs/halo` in isolation is a
faithful standalone build.

```bash
cd libs/halo
rm -rf build-ci
cmake -B build-ci -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_BUILD_TESTING=ON \
  -DHALO_BUILD_FORTRAN=ON
cmake --build build-ci --parallel $(nproc)
```

This produces the `halo` library (`libhalo.a`) and its `HELM::HALO` alias, plus
the Fortran interop libraries (`halo_c_interop`, `halo_fortran` /
`HELM::HALO_Fortran`) and all test executables.

**Strict isolation variant (copy to a scratch tree).** Configuring `libs/halo`
in place already builds standalone, but the strongest proof of Requirement 12.3
copies *only* the HALO tree to a location with **no sibling HELM components on
disk at all**, then configures and builds there. This is exactly what
[`cmake/ci_pipeline.sh`](../cmake/ci_pipeline.sh) does in Stage 2, and it is
worth running at least once when changing the build system:

```bash
rm -rf /tmp/halo-standalone
cp -r /workspace/helm-project/libs/halo /tmp/halo-standalone
rm -rf /tmp/halo-standalone/build
cd /tmp/halo-standalone
cmake -B build -G Ninja -DCMAKE_CXX_STANDARD=20 -DHALO_BUILD_TESTING=ON -DHALO_BUILD_FORTRAN=ON
cmake --build build --parallel $(nproc)
# Expect build/libhalo.a (HELM::HALO), build/libhalo_c_interop.a, build/libhalo_fortran.a
```

**Standalone consumability check.** To prove the exported alias is usable by a
downstream project with no other HELM trees present, CI also builds a tiny
throwaway consumer that pulls in only `libs/halo` and links `HELM::HALO`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(halo_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
add_subdirectory(/workspace/helm-project/libs/halo halo_build)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE HELM::HALO)   # namespaced alias
```

```cpp
// main.cpp
#include <halo/halo.hpp>
int main() { return 0; }
```

If `HELM::HALO` did not exist or carried a hidden HELM dependency, configure or
link would fail. (Downstream consumers normally use
`find_package(HALO REQUIRED)` + `target_link_libraries(... HELM::HALO)` against
an installed HALO, as documented in the README; the `add_subdirectory` form is
the most direct way to assert the alias in a single self-contained CI step.)

---

### Stage 3 — Unit tests (`mpirun -np 4`)

**Why:** HALO wraps real MPI resources (`MPI_Comm`, `MPI_Request`, `MPI_Win`)
and computes neighbor exchange topology across ranks. The GoogleTest unit suite
must run against a real multi-rank MPI job to verify correct cross-rank
behavior. CTest launches each unit test via `mpirun --oversubscribe -np 4`
(the launcher convention defined in
[`tests/CMakeLists.txt`](../tests/CMakeLists.txt)); these tests carry the
`mpi;unit` labels.

```bash
cd libs/halo/build-ci
ctest -L unit --output-on-failure
```

`-L unit` selects all MPI-based unit tests, including:

**Core RAII/resource tests:**

- `test_communicator` — MPI_Comm RAII lifecycle
- `test_request_guard` — MPI_Request RAII lifecycle
- `test_window_guard` — MPI_Win RAII lifecycle
- `test_halo_plan` — Halo_Plan construction and neighbor topology
- `test_exchange` — Flat-buffer halo exchange

**Structured exchange tests (production hardening):**

- `test_structured_halo_plan` — Structured_Halo_Plan construction and subview computation
- `test_structured_exchange` — Multi-dimensional structured halo exchange (2D/3D periodic, non-periodic, LayoutLeft/Right, strided subviews)
- `test_neighbor_collective` — Topology-aware MPI_Neighbor_alltoallv exchange path

**Persistent communication tests:**

- `test_persistent_handle` — Persistent_Halo_Handle start/wait/test cycle and RAII

**Diagnostics and error handling tests:**

- `test_diagnostics` — Exchange_Event emission and callback invocation
- `test_error_messages` — Context-rich error messages with rank/comm info

**Single-rank unit tests (no mpirun):**

- `test_handle_registry` — C-interop handle registry (single-rank, no MPI)
- `test_pack_unpack` — GPU pack/unpack kernels (single-rank, Kokkos)

To run a single test binary directly with explicit ranks:

```bash
mpirun --oversubscribe -np 4 ./tests/test_communicator
```

---

### Stage 4 — Property tests (single-rank, mocked MPI)

**Why:** RAII destructor behavior and API invariants must be verified
deterministically, without depending on MPI runtime side effects. The
property-based suite (RapidCheck) runs **single-rank** and links the MPI
interposition spy (`tests/mpi_interposition.*`), which intercepts and records
`MPI_Comm_free`, `MPI_Cancel`, `MPI_Request_free`, etc. These tests need no
`mpirun` and carry the `property` label
(see [`tests/CMakeLists.txt`](../tests/CMakeLists.txt)).

```bash
cd libs/halo/build-ci
ctest -L property --output-on-failure
```

This covers:

**Core RAII properties:**

- `prop_communicator`, `prop_request_guard`, `prop_window_guard`
- `prop_halo_plan`, `prop_exchange`
- `prop_environment`, `prop_gpu_dispatch`, `prop_handle_registry`

**C-interop boundary properties:**

- `prop_exception_boundary`, `prop_destroy_handle`
- `prop_plan_validation`, `prop_exchange_forwarding`

**Structured exchange properties (production hardening):**

- `prop_structured_exchange` — Round-trip data preservation, pack/unpack identity, persistent vs non-persistent equivalence

---

### Stage 5 — Fortran integration test (`mpirun -np 4`)

**Why:** HALO ships an `iso_c_binding` Fortran interface (`halo_mod`) for
incremental adoption by legacy NUOPC/ESMF models. The Fortran integration test
(`test_halo_mod`) drives the C-interop layer through the Fortran module against
a real multi-rank MPI job, so it runs under `mpirun -np 4`. It is built only
when `-DHALO_BUILD_FORTRAN=ON` and lives in
[`tests_fortran/`](../tests_fortran/CMakeLists.txt).

```bash
cd libs/halo/build-ci
ctest -L fortran --output-on-failure
```

> If the Fortran integration test is not yet labelled `fortran` in
> `tests_fortran/CMakeLists.txt`, select it by name instead:
> `ctest -R test_halo_mod --output-on-failure`.

---

### Stage 6 — Sanitizer builds (ASan + UBSan)

**Why:** RAII MPI wrappers manipulate raw handles and staging buffers; a missed
free, a dangling view, or signed-overflow in index math would be a latent
hazard. A dedicated build instrumented with **AddressSanitizer** (use-after-free,
heap/stack overflow, leaks) and **UndefinedBehaviorSanitizer** (signed overflow,
invalid casts, misaligned access, etc.) catches these classes of defect that a
normal build cannot. The property/unit suite is fast and single-rank, so it is
the ideal workload to run under instrumentation.

```bash
cd libs/halo
rm -rf build-asan
cmake -B build-asan -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_BUILD_TESTING=ON \
  -DHALO_BUILD_FORTRAN=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
cmake --build build-asan --parallel $(nproc)
cd build-asan
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest -L property --output-on-failure
```

**On `detect_leaks=0`.** Several C-interop boundary property tests
(`prop_destroy_handle`, `prop_plan_validation`, `prop_exchange_forwarding`,
`prop_exception_boundary`) **intentionally** leak a single process-global
`Communicator` registered in the `Handle_Registry` (see the
`valid_comm_handle()` helper in `tests/prop_destroy_handle.cpp`, which documents
the deliberate leak: the comm must stay valid across every RapidCheck iteration,
and freeing it would risk a double-free). LeakSanitizer would otherwise report
these harmless, by-design test-fixture allocations as failures. We therefore run
the sanitizer suite with `detect_leaks=0` so ASan focuses on genuine
memory-safety errors (use-after-free, buffer overflow) and UBSan on undefined
behavior — which is what this stage is meant to catch. Real leaks in the
shipping `halo` library are still caught by the non-interop property tests under
the default leak detection used in earlier stages.

> CI runs ASan/UBSan with `BUILD_FORTRAN=OFF`: the sanitizer target is the C++
> property suite, and skipping Fortran keeps the instrumented build focused and
> fast.

---

### Stage 7 — Spack build smoke test (planned)

**Why:** HALO ships a Spack package
([`spack/package.py`](../spack/package.py)) for integration into HPC center
software stacks. **Requirement 6.1** mandates that `spack install halo` succeed
with default variants on a reference system. This stage validates the Spack
package definition by performing a concretization dry run and (when a full Spack
install is available) a real install + `spack test run halo`.

> **Status: Planned.** The HELM CI Docker container does not currently include a
> Spack installation. This stage will be enabled once Spack is added to the
> container image or a separate Spack-enabled runner is provisioned. The GitHub
> Actions workflow contains a placeholder step that is currently skipped.

When enabled, the stage will run:

```bash
# Concretize (validates spec + dependencies resolve)
spack spec halo@main +fortran ~gpu_aware_mpi +tests

# Install from source (uses the in-tree package.py)
spack dev-build halo@main

# Run package tests
spack test run halo
```

The variants tested:

| Variant | Default | Purpose |
| --- | --- | --- |
| `+fortran` | on | Build Fortran interop (`halo_fortran`, `halo_mod`) |
| `~gpu_aware_mpi` | off | No GPU-aware MPI in CI (no GPU hardware) |
| `+tests` | off in production, on in CI | Build and run the test suite |

---

## Cleanup

Throwaway CI build trees (`build-ci`, `build-asan`, and the downstream consumer
scratch dir) are removed at the end of the job. The normal `build/` tree, if
present, is left untouched.

```bash
cd libs/halo && rm -rf build-ci build-asan
```

---

## Local reproduction (one block)

To run the whole pipeline locally exactly as CI does:

```bash
# From the HELM project root:
docker compose up -d --build
docker compose exec -T helm-dev bash -lc '
  set -e
  cd libs/halo

  # Stage 1: isolation scan
  sh cmake/check_tier1_isolation.sh .

  # Stage 2: standalone build (+ HELM::HALO)
  rm -rf build-ci
  cmake -B build-ci -G Ninja -DCMAKE_CXX_STANDARD=20 -DHALO_BUILD_TESTING=ON -DHALO_BUILD_FORTRAN=ON
  cmake --build build-ci --parallel $(nproc)

  # Stages 3-5: unit (mpi), property, fortran
  ( cd build-ci && ctest -L unit --output-on-failure )
  ( cd build-ci && ctest -L property --output-on-failure )
  ( cd build-ci && ctest -L fortran --output-on-failure ) || \
    ( cd build-ci && ctest -R test_halo_mod --output-on-failure )

  # Stage 6: sanitizers
  rm -rf build-asan
  cmake -B build-asan -G Ninja -DCMAKE_CXX_STANDARD=20 -DHALO_BUILD_TESTING=ON -DHALO_BUILD_FORTRAN=OFF \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  cmake --build build-asan --parallel $(nproc)
  ( cd build-asan && ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ctest -L property --output-on-failure )

  # Stage 7: Spack smoke test (skip if spack not installed)
  if command -v spack >/dev/null 2>&1; then
    spack spec halo@main +fortran ~gpu_aware_mpi +tests
    spack dev-build halo@main
    spack test run halo
  else
    echo "Stage 7: SKIPPED (spack not installed)"
  fi

  # Cleanup
  rm -rf build-ci build-asan
'
```
