"""AXIS Python bindings — spatial interpolation for Earth-system fields"""

import enum
from typing import Annotated, Any, overload

import numpy
from numpy.typing import NDArray

HAVE_PROJ: bool = True

HAVE_NETCDF: bool = True

class Method(enum.Enum):
    Bilinear = 0

    NearestNeighbor = 1

    Bicubic = 2

    Patch = 3

    Conservative = 4

    Conservative1stOrder = 4

    Conservative2ndOrder = 5

class NormType(enum.Enum):
    DstArea = 0

    FracArea = 1

class UnmappedAction(enum.Enum):
    Error = 0

    Ignore = 1

    Mask = 2

class ExtrapolationAction(enum.Enum):
    NoExtrap = 0

    NearestWet = 1

class LineType(enum.Enum):
    Cartesian = 0

    GreatCircle = 1

class Mesh:
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_cells(self) -> int: ...

class Matrix:
    @property
    def nnz(self) -> int: ...
    @property
    def n_src(self) -> int: ...
    @property
    def n_dst(self) -> int: ...
    @property
    def is_csr(self) -> bool: ...
    @property
    def has_unmapped_mask(self) -> bool: ...
    def unmapped_mask(self) -> object:
        """Per-destination unmapped mask (int array) or None"""

    def to_csr(self) -> None:
        """Convert internal COO representation to CSR format"""

    def to_bytes(self) -> bytes:
        """Serialize the matrix to a binary blob (WeightCache format)"""

    @staticmethod
    def from_bytes(data: bytes) -> Matrix:
        """Deserialize a binary blob (WeightCache format) into a Matrix"""

def make_regular_mesh(ni: int, nj: int, lon_start: float, lat_start: float, dlon: float, dlat: float) -> Mesh:
    """Create a regular lat-lon UnstructuredMesh"""

def make_projected_mesh(
    ni: int,
    nj: int,
    proj_string: str,
    center_x: Annotated[NDArray[numpy.float64], dict(shape=(None,))],
    center_y: Annotated[NDArray[numpy.float64], dict(shape=(None,))],
) -> Mesh:
    """Create a projected UnstructuredMesh using PROJ"""

def make_ugrid_mesh(
    node_coords: Annotated[NDArray[numpy.float64], dict(shape=(None, None))],
    conn_offsets: Annotated[NDArray[numpy.int64], dict(shape=(None,))],
    conn_indices: Annotated[NDArray[numpy.int64], dict(shape=(None,))],
    cell_mask: object | None = None,
) -> Mesh:
    """
    Create an unstructured UGRID UnstructuredMesh. cell_mask (optional int array,
    nonzero = active source cell) enables wet renormalization + nearest-wet extrapolation.
    """

def make_named_mesh(name: str) -> Mesh:
    """Generate a named grid (e.g., 'O32', 'F64')"""

def list_named_grid_families() -> str:
    """Return the registered named-grid family letters (e.g. 'CFGNOR')"""

def named_grid_layout(name: str) -> dict[str, Any]:
    """
    Rectangular cell-order layout of a named grid (ni/nj/n_tiles, structured flag)
    """

def named_grid_cell_centers(name: str) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """Cell centers as interleaved (lon, lat) degrees in engine cell order"""

@overload
def generate_weights(src_mesh: Mesh, dst_mesh: Mesh, method: Method) -> Matrix:
    """Generate interpolation weights between src and dst meshes"""

@overload
def generate_weights(src_mesh: Mesh, dst_mesh: Mesh, config: dict[str, Any]) -> Matrix:
    """Generate interpolation weights with a RegridConfig dictionary"""

@overload
def apply_weights(matrix: Matrix, src: Annotated[NDArray[numpy.float64], dict(shape=(None,))]) -> NDArray[numpy.float64]:  # type: ignore[overload-overlap]
    """Apply interpolation matrix to source field, return destination array"""

@overload
def apply_weights(matrix: Matrix, src: Annotated[NDArray[numpy.float32], dict(shape=(None,))]) -> NDArray[numpy.float32]:
    """
    Apply interpolation matrix to a float32 source field (CSR required, double accumulation)
    """

@overload
def batch_apply(matrix: Matrix, src: Annotated[NDArray[numpy.float64], dict(shape=(None, None))]) -> NDArray[numpy.float64]:  # type: ignore[overload-overlap]
    """
    Apply interpolation matrix to multiple fields (cells × variables).
    Accepts both C-order and Fortran-order 2-D arrays.
    """

@overload
def batch_apply(matrix: Matrix, src: Annotated[NDArray[numpy.float32], dict(shape=(None, None))]) -> NDArray[numpy.float32]:
    """
    Apply interpolation matrix to multiple float32 fields (CSR required, double accumulation).
    """

def check_conservation(
    matrix: Matrix,
    src: Annotated[NDArray[numpy.float64], dict(shape=(None,))],
    dst: Annotated[NDArray[numpy.float64], dict(shape=(None,))],
) -> dict[str, Any]:
    """Check conservation between source and destination fields"""

def detect_tripolar_grid(mesh: Mesh, ni: int, nj: int) -> dict[str, Any]:
    """Detect whether an unstructured mesh represents a folded tripolar grid"""

def compute_rotation_angles(mesh: Mesh) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """Compute per-cell vector rotation angles (radians) from mesh geometry"""

def generate_vector_weights(
    src_mesh: Mesh, dst_mesh: Mesh, src_alpha: object | None = None, dst_alpha: object | None = None, config: dict[str, Any] = {}
) -> tuple[Matrix, Matrix]:
    """
    Generate coupled vector interpolation weights for U and V wind components.
    Pass None for src_alpha/dst_alpha to use engine-computed grid orientation.
    """

def write_esmf(filepath: str, matrix: Matrix) -> None:
    """Write the interpolation weights matrix to an ESMF netCDF file"""

def read_esmf(filepath: str) -> Matrix:
    """Read the interpolation weights matrix from an ESMF netCDF file"""

def interpolate_vertical(
    src_field: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)],
    src_levels: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)],
    dst_levels: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)],
    tension: float = 0.0,
) -> NDArray[numpy.float64]:
    """Interpolate vertical 2D profiles using 1D uniform coordinates"""

def interpolate_vertical_varying(
    src_field: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)],
    src_levels: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)],
    dst_levels: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)],
    tension: float = 0.0,
) -> NDArray[numpy.float64]:
    """
    Interpolate vertical 2D profiles using 2D spatially-varying coordinates
    """
