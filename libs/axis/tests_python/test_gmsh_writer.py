import os

from axis import axis_py


def test_gmsh_writer_export(tmp_path):
    mesh = axis_py.make_regular_mesh(5, 5, 0.0, -90.0, 72.0, 36.0)
    filepath = os.path.join(tmp_path, "test.msh")

    # Verify high-level wrapping
    axis_py.write_gmsh(filepath, mesh)
    assert os.path.exists(filepath)
    assert os.path.getsize(filepath) > 0
