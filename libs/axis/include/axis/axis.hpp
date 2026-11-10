// SPDX-License-Identifier: BSD-3-Clause
// Copyright NOAA / HELM Project

/// @file axis.hpp
/// @brief Umbrella header for the AXIS micro-library.
///
/// Including this single header brings in the entire AXIS public API:
///   - Plain-data ingest/egress contract (GridDescriptor, WeightEgress, MeshEgress)
///   - Topology data structures and builders (meshes, grids, factories, generators)
///   - Solver engine (weight generation, sparse apply, conservation accounting)
///   - Internal detail utilities (memory traits, mdspan interop, RAII handles)
///
/// AXIS is a Tier 1 HELM utility — it includes NO headers from HALO, AMIO,
/// TICK, LOGS, SPAN, DAGR, eckit, or domain-science models.

#ifndef AXIS_AXIS_HPP
#define AXIS_AXIS_HPP

// ─── Types ───────────────────────────────────────────────────────────────────
#include <axis/types.hpp>

// ─── Ingest / Egress Contract ────────────────────────────────────────────────
#include <axis/ingest/egress.hpp>
#include <axis/ingest/grid_descriptor.hpp>

// ─── Topology ────────────────────────────────────────────────────────────────
#include <axis/topology/enums.hpp>
#include <axis/topology/gmsh_writer.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/multi_face_grid.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/topology/reduced_gaussian_grid.hpp>
#include <axis/topology/rule_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>

// ─── Solver ──────────────────────────────────────────────────────────────────
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>
#include <axis/solver/halo_pattern.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>

// ─── Detail (internal utilities, exposed for advanced use) ───────────────────
#include <axis/detail/mdspan_interop.hpp>
#include <axis/detail/memory_traits.hpp>
#include <axis/detail/raii_handles.hpp>

#endif  // AXIS_AXIS_HPP
