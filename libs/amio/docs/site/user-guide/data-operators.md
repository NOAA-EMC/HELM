# Field precision and integrity

AMIO compiles each field's operator policy when a step begins. The supported precision path is host `F64` input converted to native `F32` storage. The conversion runs on AMIO-owned snapshots before the backend write. A data-dependent accuracy failure poisons the step and prevents its completion record from being published.

The operation order is precision conversion, logical integrity digest, then the destination's native storage codec. Per-field `amio.codec` selects only a lossless codec implemented by that destination: `deflate` for NetCDF-4, `zstandard` for Zarr v3 when TensorStore is enabled, and `libaec` or `lossless_jpeg2000` for GRIB2 when g2c is enabled. An unknown codec or a codec used with the wrong destination is rejected before transfer. GRIB2 codec selection requires finite F32 values and zero decimal scaling. General lossy codecs remain unsupported.

## Authorizing F64 to F32

The schema stores policy as typed variable attributes. `amio.precision` is either `f64_to_f32_absolute` or `f64_to_f32_relative`; `amio.tolerance` is a finite, nonnegative F64 value. Both `amio.authorize_field_loss` and `amio.authorize_lossy_restart` must be true. These authorizations are separate: satisfying an output accuracy bound does not authorize lossy restart data.

For each finite value `x`, AMIO rounds to F32 and checks either `abs(decoded - x) <= tolerance` for absolute mode or `abs(decoded - x) <= tolerance * abs(x)` for relative mode. Relative-only conversion therefore requires exact zero when `x` is zero. Overflow, underflow that violates the bound, and an inexact missing-value sentinel fail before publication. Signed zero and infinity sign are preserved. NaN input is rejected unless `amio.canonical_nan` is true, in which case AMIO stores the canonical quiet-NaN representation.

Example C metadata descriptors:

```c
/* Attach these typed attributes to an F64-source, F32-stored variable. */
amio_attribute_descriptor_t attrs[4] = {
    { .struct_size = sizeof(amio_attribute_descriptor_t), .version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
      .name = "amio.precision", .name_bytes = 14, .type = AMIO_ATTRIBUTE_UTF8,
      .element_count = 1, .value = "f64_to_f32_absolute", .value_bytes = 19 },
    { .struct_size = sizeof(amio_attribute_descriptor_t), .version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
      .name = "amio.tolerance", .name_bytes = 14, .type = AMIO_ATTRIBUTE_F64,
      .element_count = 1, .value = &tolerance, .value_bytes = sizeof(tolerance) },
    { .struct_size = sizeof(amio_attribute_descriptor_t), .version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
      .name = "amio.authorize_field_loss", .name_bytes = 25, .type = AMIO_ATTRIBUTE_BOOL,
      .element_count = 1, .value = &yes, .value_bytes = sizeof(yes) },
    { .struct_size = sizeof(amio_attribute_descriptor_t), .version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
      .name = "amio.authorize_lossy_restart", .name_bytes = 28, .type = AMIO_ATTRIBUTE_BOOL,
      .element_count = 1, .value = &yes, .value_bytes = sizeof(yes) }
};
```

To select a native codec, attach one additional UTF-8 attribute to the variable; for example, set `amio.codec` to `deflate` for NetCDF-4. The schema API copies these attributes, so the caller can release or reuse the value after schema definition.

The Fortran module uses the same `amio_attribute_descriptor_t` and `AMIO_ATTRIBUTE_UTF8`, `AMIO_ATTRIBUTE_F64`, and `AMIO_ATTRIBUTE_BOOL` constants. Pass C-interoperable, contiguous values and exact byte lengths; the schema call copies the attributes before returning. See the [Fortran API guide](fortran-api.md) for descriptor lifetime and `c_loc` requirements.

## Integrity representation

For supported persistent NetCDF and TensorStore Zarr steps, the completion record stores a versioned SHA-256 digest for each present field region, plus the compiled pipeline identity. The digest covers the `AMIO-LOGICAL-SHA256-v1` domain tag, dtype, rank, region offsets/extents, and stored logical values in little-endian byte order. It is calculated after precision conversion and before storage encoding, so storage-container metadata does not affect the logical checksum. Partitioned NetCDF steps gather each producer's region digest into the completion record. A partial-region verifier reads and checks every complete checksum region that intersects a requested selection, subject to a caller-supplied byte ceiling.

`amio_validate_dataset` can report integrity `PASS`, `FAIL`, or `UNSUPPORTED`. A missing digest (including older collections), unsupported format, unavailable reader, or a step without checksummed regions is not reported as a pass. C validation checks whole-field regions for non-partitioned NetCDF and supported TensorStore Zarr, and each gathered partition region for distributed NetCDF. GRIB2 does not yet persist logical integrity regions and returns `UNSUPPORTED` for integrity validation. Corrupt stored values return `FAIL` while the catalog chain remains unchanged.

## Resource accounting and limits

Conversion scratch is reserved against the dataset's byte and descriptor budget before conversion starts. Diagnostics report `operator_input_bytes`, `operator_output_bytes`, `operator_ns`, and `operator_scratch_bytes`; failed accuracy checks record their first status and field scope. A reservation or scratch-limit failure rejects the step before payload writing. Successful `amio_put_field` still means the caller's source bytes were copied and may be reused; publication occurs only when `amio_end_step` succeeds.

The capability query reports F64-to-F32 conversion and logical integrity as compiled features, and native lossless codec selection as destination-negotiated. It reports borrowed buffers, device memory, and per-field lossy codecs as unsupported. Codec availability still depends on the destination backend being built; consult the outcomes above and the platform-specific build configuration.
