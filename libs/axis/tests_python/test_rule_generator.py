# SPDX-License-Identifier: Apache-2.0

import axis


def test_rule_generator_regular():
    config = {
        "kind": "RegularLatLon",
        "bbox_min_x": 0.0,
        "bbox_max_x": 360.0,
        "bbox_min_y": -90.0,
        "bbox_max_y": 90.0,
        "r_x": 36.0,
        "r_y": 36.0,
    }
    geom = axis.grid.RuleGeometry(config)
    mesh = geom.to_mesh()
    assert mesh.n_cells == 50
    assert mesh.n_nodes == 66
