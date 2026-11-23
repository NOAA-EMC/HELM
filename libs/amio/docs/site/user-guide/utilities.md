# Command-line utilities

When `AMIO_BUILD_TOOLS=ON`, AMIO builds and installs the `amio` executable.
The tool emits human-readable output by default. Add `--json` for a versioned
JSON object containing `schema_version`, `command`, `status`, `summary`,
`results`, `warnings`, and `errors`.

```sh
amio capabilities --json
amio inspect /path/to/collection --limit 100 --json
amio validate /path/to/collection --json
amio diagnostics --input snapshot.json --json
```

Exit status `0` means the requested operation completed, `2` indicates invalid
input, `3` an unsupported capability, `4` incomplete or corrupt collection
data, and `5` a runtime failure. Exit status `6` is reserved for a timed-out
operation with potentially active work.

The current utility build reads AMIO's POSIX step collections through a
read-only dataset handle. Inspection uses UUID and generation-bound cursors
and reports schema, physical format, step completion, and available model time
without loading full field payloads. Structural validation checks committed
steps, required-field coverage, and schema references without repairing data.
Orphan enumeration remains unsupported when the destination cannot list
objects; the JSON report marks validation incomplete and includes a warning.
`validate --integrity` verifies payload digests when the collection records
them. Legacy collections without integrity records remain unverified.

`copy SOURCE DEST --config FILE --steps FIRST:LAST --fields LIST` performs a
bounded exact-schema NetCDF copy. It validates the whole inclusive range,
required fields and coordinate closure, model times, schema attributes, and a
64 MiB per-step read budget before creating the destination. It preserves the
selected schema and publishes completed output steps. Zarr and GRIB input,
format-changing conversions, dtype/operator transformations, and named
metadata omissions are unsupported and fail before destination creation.

`diagnostics --input` accepts and decodes an API-v2 JSON snapshot no larger
than 1 MiB. It reports bounded topology, counters, budgets, scopes, and event
metadata while excluding message text. `diagnostics --export PATH` writes an
explicit snapshot to a new file with owner-only permissions and does not print
the snapshot contents.
