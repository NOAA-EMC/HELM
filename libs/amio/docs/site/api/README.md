# API Reference

This directory contains auto-generated API documentation produced by
[Moxygen](https://github.com/nickhutchinson/moxygen) from Doxygen XML output.

## Generating API Docs

The files in this directory (except this README) are generated and should not
be edited by hand. To regenerate:

```bash
# From the project root:
./docs/generate-api.sh

# Or via CMake:
cmake --build build --target docs
```

## Pipeline

1. **Doxygen** parses the C99 headers and Fortran module → XML output in `docs/xml/`
2. **Moxygen** converts the XML → Markdown files in `docs/site/api/`
3. **MkDocs** assembles the full documentation site from `docs/site/`

## Contents (when generated)

- `api-amio_8h.md` — amio.h umbrella header
- `api-amio__errors_8h.md` — Error codes and amio_strerror
- `api-amio__types_8h.md` — Data types, shapes, handles
- `api-amio__export_8h.md` — AMIO_API visibility macro
- `api-amio__mod.md` — Fortran module (amio_mod)
