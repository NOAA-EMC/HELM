#ifndef HALO_HALO_HPP
#define HALO_HALO_HPP

/// @file halo.hpp
/// @brief Umbrella header for the HALO (Hardware-Abstracted Link Operations) library.
///
/// Including this single header provides access to the complete public API of
/// the HALO Tier 1 micro-library: RAII MPI wrappers, precomputed halo exchange
/// plans, blocking and asynchronous exchange functions, and the environment
/// initialization singleton.
///
/// @note HALO is a Tier 1 component of the HELM ecosystem. It depends only on
/// the C++ standard library, MPI, and Kokkos. It has zero compile-time
/// dependencies on other HELM libraries (TICK, LOGS, AXIS, AMIO, SPAN, DAGR).

#include <halo/bounded_peer_exchange.hpp>
#include <halo/collectives.hpp>
#include <halo/communicator.hpp>
#include <halo/diagnostics.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <halo/exchange.hpp>
#include <halo/exchange_structured.hpp>
#include <halo/gather_replicated.hpp>
#include <halo/halo_handle.hpp>
#include <halo/halo_plan.hpp>
#include <halo/indexed_gather_plan.hpp>
#include <halo/persistent_halo_handle.hpp>
#include <halo/replicated_gather_plan.hpp>
#include <halo/request_guard.hpp>
#include <halo/structured_halo_plan.hpp>
#include <halo/version.hpp>
#include <halo/window_guard.hpp>

#endif  // HALO_HALO_HPP
