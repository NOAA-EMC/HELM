# Fortran API

The AMIO Fortran API is provided through the `amio_mod` module, which wraps
the C99 API using Fortran 2003 `iso_c_binding`. It provides both raw interface
blocks (matching the C signatures exactly) and convenience subroutines that
hide the C pointer mechanics behind Fortran assumed-shape arrays.

## Module Usage

```fortran
use amio_mod
```

This imports:

- Constants: `AMIO_MODE_WRITE`, `AMIO_MODE_READ`, all `AMIO_ERR_*` codes, all `AMIO_DTYPE_*` tags
- Derived types: `amio_shape_t`, `amio_bbox_t`
- Interface blocks: `amio_init`, `amio_finalize`, `amio_open_dataset`, etc.
- Generic interface: `amio_write_array` (resolves type and rank automatically)

## Convenience Subroutines

The `amio_write_array` generic interface is the recommended way to write data
from Fortran. It automatically:

1. Determines the data type (`AMIO_DTYPE_F32` for `real(c_float)`, etc.)
2. Computes the rank from the array's intrinsic rank
3. Extracts extents from `size(array, dim=d)`
4. Obtains the data pointer via `c_loc()`
5. Calls the raw `amio_write` C function

### Supported Type/Rank Combinations

| Type | 1D | 2D | 3D |
|------|----|----|-----|
| `real(c_float)` | ✓ | ✓ | ✓ |
| `real(c_double)` | ✓ | ✓ | ✓ |
| `integer(c_int32_t)` | ✓ | ✓ | ✓ |

### Example

```fortran
use amio_mod
use, intrinsic :: iso_c_binding

type(c_ptr) :: dataset, io_handle
integer(c_int32_t) :: rc
real(c_float), allocatable, target :: temperature(:,:,:)

allocate(temperature(100, 50, 25))
temperature = 288.15  ! Fill with data

! Write — type and shape are detected automatically
call amio_write_array(dataset, "temperature" // c_null_char, &
                      temperature, io_handle, rc)

if (rc /= AMIO_OK) then
    write(*,*) 'Write failed with code:', rc
    error stop
end if
```

## Raw Interface Blocks

For cases where the convenience subroutines don't cover your needs (e.g.,
higher ranks, unsigned types), use the raw `amio_write` interface directly:

```fortran
use amio_mod
use, intrinsic :: iso_c_binding

type(c_ptr) :: dataset, io_handle, data_ptr
integer(c_int32_t) :: rc
type(amio_shape_t) :: shp
real(c_double), allocatable, target :: field(:,:,:,:)

allocate(field(72, 36, 10, 5))
field = 0.0_c_double

! Build shape manually
shp%rank = 4_c_int32_t
shp%extents = 0_c_int64_t
shp%strides = 0_c_int64_t
shp%extents(1) = 72_c_int64_t
shp%extents(2) = 36_c_int64_t
shp%extents(3) = 10_c_int64_t
shp%extents(4) = 5_c_int64_t

data_ptr = c_loc(field(1, 1, 1, 1))
rc = amio_write(dataset, "field4d" // c_null_char, data_ptr, &
                AMIO_DTYPE_F64, shp, io_handle)
```

## String Handling

Fortran strings passed to AMIO must be null-terminated using `c_null_char`:

```fortran
rc = amio_init("path/to/manifest.yaml" // c_null_char, core)
```

!!! warning "Null Termination"
    Forgetting `// c_null_char` will cause undefined behavior. The C API
    expects NUL-terminated strings.

## Error Handling Pattern

```fortran
subroutine check_amio(status, context)
    integer(c_int32_t), intent(in) :: status
    character(len=*), intent(in) :: context

    if (status /= AMIO_OK) then
        write(*,'(A,A,A,I0)') 'AMIO error in ', context, &
            ', code=', status
        error stop 'AMIO operation failed'
    end if
end subroutine check_amio
```

## Complete Lifecycle

```fortran
program amio_example
    use, intrinsic :: iso_c_binding
    use amio_mod
    implicit none

    type(c_ptr) :: core, dataset, io_handle
    integer(c_int32_t) :: rc
    real(c_float), allocatable, target :: data(:,:)

    ! 1. Initialize
    rc = amio_init("manifest.yaml" // c_null_char, core)

    ! 2. Open dataset
    rc = amio_open_dataset(core, "manifest.yaml" // c_null_char, &
                           AMIO_MODE_WRITE, dataset)

    ! 3. Write data
    allocate(data(360, 180))
    data = 300.0_c_float
    call amio_write_array(dataset, "sst" // c_null_char, &
                          data, io_handle, rc)

    ! 4. Flush
    rc = amio_flush(dataset, 0_c_int64_t)

    ! 5. Close
    rc = amio_close_dataset(dataset)

    ! 6. Finalize
    rc = amio_finalize(core)

    deallocate(data)
end program amio_example
```

## Constants Reference

### Error Codes

| Constant | Value | Meaning |
|----------|-------|---------|
| `AMIO_OK` | 0 | Success |
| `AMIO_ERR_NULL_HANDLE` | 1 | NULL handle passed |
| `AMIO_ERR_INVALID_HANDLE` | 2 | Stale or wrong-kind handle |
| `AMIO_ERR_MANIFEST_NOT_FOUND` | 3 | Manifest file not found |
| `AMIO_ERR_MANIFEST_INVALID` | 4 | Manifest parse/validation error |
| `AMIO_ERR_STAGING_BACKPRESSURE` | 7 | Staging pool exhausted |
| `AMIO_ERR_UNKNOWN_BACKEND` | 11 | Backend name not in registry |
| `AMIO_ERR_BACKEND_FAILURE` | 16 | Backend driver error |
| `AMIO_ERR_INVALID_INPUT` | 17 | Invalid argument |

### Data Type Tags

| Constant | Fortran Kind | C Type |
|----------|-------------|--------|
| `AMIO_DTYPE_F32` | `c_float` | `float` |
| `AMIO_DTYPE_F64` | `c_double` | `double` |
| `AMIO_DTYPE_I32` | `c_int32_t` | `int32_t` |
| `AMIO_DTYPE_I64` | `c_int64_t` | `int64_t` |
