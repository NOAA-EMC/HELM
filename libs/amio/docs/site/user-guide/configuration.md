# Configuration (Manifests)

AMIO is configured through YAML manifest files that specify the backend driver,
staging pool parameters, worker pool settings, and output paths.

## Manifest Structure

```yaml
amio:
  version: "0.1.0"

  staging:
    buffer_count: 8              # Number of staging buffers [1, 4096]
    buffer_size_bytes: 52428800  # Per-buffer capacity in bytes [1, 1 GiB]

  workers:
    thread_count: 2              # Worker threads [1, 256]
    pin_threads: false           # CPU/NUMA pinning

  backend:
    name: "netcdf4"              # One of: netcdf4, zarr3, grib2
    options:
      # Backend-specific options (see below)

  output:
    path: "output.nc"
    overwrite: true
```

## Staging Pool

The staging pool is a bounded set of pre-allocated buffers used for
synchronous data snapshots during writes.

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `buffer_count` | integer | [1, 4096] | 8 | Number of staging buffers |
| `buffer_size_bytes` | integer | [1, 1 GiB] | 50 MiB | Per-buffer capacity |

!!! tip "Sizing the Staging Pool"
    Set `buffer_size_bytes` to at least the size of your largest single
    variable write. Set `buffer_count` to at least 2× the number of
    concurrent writes you expect to overlap with computation.

## Worker Pool

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `thread_count` | integer | [1, 256] | 1 | Number of background I/O threads |
| `pin_threads` | boolean | — | false | Pin threads to CPU cores |

## Backend: NetCDF-4

```yaml
backend:
  name: "netcdf4"
  options:
    parallel_io: false           # Use MPI parallel I/O
    compression:
      algorithm: "zlib"          # Lossless codec
      level: 4                   # Compression level [1-9]
    chunking:
      auto: true                 # Auto-determine chunk sizes
```

### NetCDF-4 Options

| Field | Values | Description |
|-------|--------|-------------|
| `parallel_io` | `true` / `false` | Enable MPI-IO collective writes |
| `compression.algorithm` | `zlib`, `szip` | Lossless compression codec |
| `compression.level` | 1–9 | Compression effort |
| `chunking.auto` | `true` / `false` | Auto-compute chunk dimensions |

## Backend: Zarr v3

```yaml
backend:
  name: "zarr3"
  options:
    mode: "nczarr"               # "tensorstore" or "nczarr"
    compression:
      algorithm: "zstd"          # One of: blosc, zstd
      level: 3
    chunking:
      auto: true
```

### Zarr v3 Options

| Field | Values | Description |
|-------|--------|-------------|
| `mode` | `tensorstore`, `nczarr` | Backend implementation |
| `compression.algorithm` | `blosc`, `zstd` | Lossless codec |
| `compression.level` | 1–22 (zstd) | Compression effort |

!!! note "TensorStore vs NCZarr"
    TensorStore mode supports cloud storage (S3, GCS, HTTPS) and Zarr v3
    sharding. NCZarr fallback mode uses the system netCDF-c library and
    supports only local file output without sharding.

## Backend: GRIB2

```yaml
backend:
  name: "grib2"
  options:
    packing: "complex_second_order"
    discipline: 0                # WMO discipline code
    compression:
      algorithm: "jpeg2000"      # One of: jpeg2000, aec
      target_ratio: 20
```

### GRIB2 Options

| Field | Values | Description |
|-------|--------|-------------|
| `packing` | `complex_second_order` | Data representation template |
| `discipline` | 0–255 | WMO GRIB2 discipline code |
| `compression.algorithm` | `jpeg2000`, `aec` | Lossless DRT |

### GRIB2 Product Definition Templates (PDTs)

The GRIB2 driver supports multiple Product Definition Templates for
atmospheric composition variables. Set `pdt_number` in the `grib2:` block
to select the template:

| `pdt_number` | Template | Use Case |
|--------------|----------|----------|
| 0 | PDT 4.0 | Analysis/forecast at a level (default) |
| 8 | PDT 4.8 | Statistically processed (average, accumulation) |
| 40 | PDT 4.40 | Chemical constituent (trace gases) |
| 44 | PDT 4.44 | Aerosol at a point in time |
| 45 | PDT 4.45 | Individual ensemble forecast for aerosol |
| 46 | PDT 4.46 | Statistically processed aerosol |
| 48 | PDT 4.48 | Aerosol optical properties at a wavelength |
| 49 | PDT 4.49 | Ensemble aerosol optical properties |

### GRIB2 Grid Definition Templates (GDTs)

| `gdt_number` | Template | Use Case |
|--------------|----------|----------|
| 0 | GDT 3.0 | Regular latitude/longitude (default) |
| 40 | GDT 3.40 | Gaussian latitude/longitude |

### GRIB2 Composition Fields

These optional manifest fields carry WMO code-table integers for composition
PDTs. All default to 0 (neutral/missing) when absent.

| Field | WMO Table | Used By |
|-------|-----------|---------|
| `chemical_constituent_type` | 4.230 | PDT 4.40 |
| `aerosol_type` | 4.233 | PDT 4.44, 4.45, 4.46, 4.48, 4.49 |
| `size_dist_param_first` | — | PDT 4.44, 4.45, 4.46 |
| `size_dist_param_second` | — | PDT 4.44, 4.45, 4.46 |
| `optical_property_type` | — | PDT 4.48, 4.49 |
| `wavelength_first_nm` | — | PDT 4.48, 4.49 |
| `wavelength_last_nm` | — | PDT 4.48, 4.49 |
| `ensemble_perturbation_number` | — | PDT 4.45, 4.49 |
| `statistical_process` | 4.10 | PDT 4.8, 4.46 |
| `time_range_unit` | 4.4 | PDT 4.8, 4.46 |
| `time_range_length` | — | PDT 4.8, 4.46 |
| `number_of_time_range_specs` | — | PDT 4.8, 4.46 |
| `total_missing_from_statistical_process` | — | PDT 4.8, 4.46 |
| `n_parallel` | — | GDT 3.40 |

### GRIB2 Field Identity Naming

On read, the GRIB2 driver synthesizes a unique field name from WMO
descriptors:

```text
d{discipline}_c{category}_n{number}_s{surface_type}_l{surface_value}
```

Composition PDTs append additional suffixes to distinguish species:

| PDT | Suffix Example |
|-----|----------------|
| 4.40 | `_ct7` (chemical constituent type 7) |
| 4.44 | `_at5` (aerosol type 5) |
| 4.45 | `_at5_ep3` (aerosol + ensemble member 3) |
| 4.46 | `_at5_sp2` (aerosol + statistical process 2) |
| 4.48 | `_at5_op1_wl550_600` (aerosol + optical + wavelength) |
| 4.49 | `_at5_op1_wl550_600_ep3` (+ ensemble) |
| 4.8 | `_sp2` (statistical process 2) |
| 4.0 | (no suffix) |

### GRIB2 Composition Example

```yaml
backend: grib2
path: aerosol_od.grib2
drt: jpeg2000

grib2:
  discipline: 0
  center: 7
  parameter_category: 3
  parameter_number: 5
  type_of_first_fixed_surface: 100
  scaled_value_first_surface: 50000
  pdt_number: 48
  gdt_number: 40
  aerosol_type: 5
  optical_property_type: 1
  wavelength_first_nm: 550
  wavelength_last_nm: 600
  n_parallel: 48
```

## Validation Rules

AMIO validates the manifest on `amio_init`. Invalid configurations return
`AMIO_ERR_MANIFEST_INVALID` with a diagnostic message naming the failing field.

| Field | Constraint |
|-------|-----------|
| `staging.buffer_count` | ∈ [1, 4096] |
| `staging.buffer_size_bytes` | ∈ [1, 1,073,741,824] |
| `workers.thread_count` | ∈ [1, 256] |
| `backend.name` | One of `netcdf4`, `zarr3`, `grib2` |
| Compression codec | Must be on the lossless allow-list |

## Example Manifests

Complete example manifests are provided in `examples/manifests/`:

- [`netcdf4_manifest.yaml`](https://github.com/NOAA-EMC/AMIO/blob/main/examples/manifests/netcdf4_manifest.yaml)
- [`zarr3_manifest.yaml`](https://github.com/NOAA-EMC/AMIO/blob/main/examples/manifests/zarr3_manifest.yaml)
- [`grib2_manifest.yaml`](https://github.com/NOAA-EMC/AMIO/blob/main/examples/manifests/grib2_manifest.yaml)
