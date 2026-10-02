# HELM: Heterogeneous Execution & Load-balancing Manager

HELM is a modern, composable C++20 toolkit designed to modernize Earth System modeling. It replaces monolithic legacy frameworks (like ESMF) with a lightweight, decentralized architecture emphasizing zero-copy memory mapping and native Kokkos GPU hardware portability.

📖 **[Read the Full Project Charter & Architectural Vision here](docs/architecture/HELM_Project_Charter.md)**

## The HELM Ecosystem
HELM is split into independent micro-libraries:
* **DAGR:** (Tier 3) Directed Acyclic Graph Router (The Orchestrator)
* **SPAN:** (Tier 2) Shared Pointer & Array Network (The C++/Fortran Interface)
* **AXIS:** (Tier 1) Arbitrary eXgrid Interpolation Solver (The Math Engine)
* **HALO:** (Tier 1) Hardware-Abstracted Link Operations (The MPI Engine)
* **TICK:** (Tier 1) Time Integration & Chronology Kernel (The Time Manager)
* **LOGS:** (Tier 1) Logging and State Syncronization (The Log Manager)
* **AMIO:** (Tier 1b) Asyncronous Multidimensional Input Output (The IO Engine)

---

## Quickstart (Local Docker Development)
To build and test the ecosystem locally, we use a containerized Ubuntu 24.04 environment with GCC-13, OpenMPI, and Kokkos pre-configured.

```bash
# 1. Spin up the development container
docker compose up -d --build

# 2. Enter the container
docker compose exec helm-dev bash

# 3. Build a specific component (e.g., TICK)
cd libs/tick
mkdir build && cd build
cmake .. -DTICK_BUILD_TESTING=ON
make -j4
ctest --output-on-failure
```

## License

This project is part of NOAA-EMC Ecosystem.

See LICENSE and DISCLAIMER for details.
