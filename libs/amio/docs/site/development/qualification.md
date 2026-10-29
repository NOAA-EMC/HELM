# AMIO Qualification Baseline

This baseline records the existing public C ABI and the qualification paths
before the scalable step-engine work. It is intentionally separate from
runtime capability claims: source-level optional backends do not prove that a
particular build or destination can use them.

## Current baseline

- ABI version is 0.2.0. Preserve existing public function signatures, enum
  numbers, and struct layouts when adding the step API.
- Error/dtype values and public struct layouts are captured in
  [`legacy_abi.json`](../../../tests/fixtures/legacy_abi.json). Struct sizes
  are currently unavailable because the C compiler driver cannot find the
  configured Xcode Command Line Tools; do not assume or record a measured
  value until a functioning compiler is available.
- Exported symbols from a built AMIO library and configured backend
  capabilities are unavailable at this baseline because there is no configured
  build tree or AMIO library in the workspace. Do not interpret this as a
  passing symbol or backend check.
- Existing tests are registered from `tests/header_isolation`,
  `tests/cmake_discovery`, `tests/unit`, optional `tests/pbt`, and
  `tests/integration`. PBT registration depends on Catch2 and RapidCheck.
- CMake options default to host library-only behavior: testing, PBT-only,
  documentation, examples, and forced NCZarr are off unless configured.

## Qualification rule

Record a check as passing only after it has run against the stated build,
backend, MPI rank count, storage system, and hardware. Mark absent tools or
infrastructure as unavailable and list the exact claim that remains
unqualified. Keep routine correctness checks separate from large-scale,
credentialed object-store, and accelerator release gates.
