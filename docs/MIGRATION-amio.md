# AMIO Migration: submodule → in-tree library

AMIO (`libs/amio/`) was absorbed into this repository on branch
`feature/amio_desubmodule`. It was previously HELM's only git submodule,
pointing at the standalone repository
[`bbakernoaa/amio`](https://github.com/bbakernoaa/amio) (branch `develop`).
From the absorption commit onward, AMIO is ordinary tracked content: a plain
`git clone` of HELM delivers it, it builds with the rest of the ecosystem, and
it is reviewed and CI-tested like every other library.

## Provenance (the anchor)

| Field | Value |
|---|---|
| Snapshot source commit | `1979fd2253267a9c11bd07830dc0e7ad2b82ea12` |
| Snapshot date / subject | 2026-10-02 — `Merge pull request #8 from DWesl/patch-1` (live `develop` tip at absorption time) |
| Previously pinned gitlink | `bbdc39b9d52c2f7f0cde3c88db740b483295a2a2` |
| Contributor checkout at move time | `dacabeae49add69e7bcffea1fe99a8e292630275` (`v0.1.0-27-gdacabea`) |
| Ancestry | `bbdc39b9` and `dacabea` are both ancestors of `1979fd2` (linear, verified) |
| Source repository | `https://github.com/bbakernoaa/amio` |
| Latest standalone tag | `v0.1.0` |
| Absorption commit | trailer `Absorbed-from: https://github.com/bbakernoaa/amio@1979fd2253267a9c11bd07830dc0e7ad2b82ea12` |

The snapshot is the **live `origin/develop` HEAD** at absorption time, not the
older pinned gitlink. This was both the directive and the technically correct
choice: the pinned/older checkouts fail to build AMIO's own unit tests in the
HELM container (missing `CONF` include on `test_backend_factory`), a fix that
landed only in the `develop` tip's PR #8 merge commits.

### Looking up pre-move history

Full pre-move history (authors, commit messages, PR discussion) lives in the
standalone repository, which retains it permanently. One documented step from
any current file:

```text
libs/amio/<file>
   → this page (Provenance)
   → standalone repo @ 1979fd2253267a9c11bd07830dc0e7ad2b82ea12
   → git log -- <file>   (answers "why is this behavior here?" for pre-move code)
```

```sh
git clone https://github.com/bbakernoaa/amio amio-history
git -C amio-history log --follow -- src/workers/worker_pool.cpp
```

## Contributor migration (one-time, after pulling the absorption)

If you had AMIO checked out as a submodule, git keeps the old submodule
registration locally. Clean it out once:

1. **Commit any local AMIO work first.** If you have uncommitted changes under
   the old submodule checkout, save them to a branch/patch **before** the steps
   below — `deinit`/`rm -rf` will discard them.
2. `git submodule deinit -f libs/amio`
3. `git rm -f libs/amio` (if git still tracks it as a gitlink locally)
4. `rm -rf .git/modules/libs/amio` (drops the cached submodule clone)
5. `git pull` — `libs/amio/` arrives as ordinary tracked content.
6. `git config --get-regexp submodule` should return nothing for `libs/amio`.

On case-insensitive filesystems (macOS/Windows), if a stale `libs/AMIO`
directory lingers after the pull, remove it: `rm -rf libs/AMIO` (the tracked
path is lowercase `libs/amio`; case-sensitive Linux CI requires lowercase).

## Versioning policy

Standalone release automation (release-please, tags like `v0.1.0`) is retired
with the move. From the absorption forward, **AMIO versions with HELM**: it
follows HELM's repository versioning, changelog discipline, and release
process — there is no separate AMIO release train. The `libs/amio/VERSION`
file is retained as historical content; the authoritative version is HELM's.

## NO-ECKIT invariant (carried over, now enforced in HELM CI)

AMIO must never depend on, link, or export `eckit` — a long-standing AMIO
architectural invariant, now enforced by four gates wired into
`.github/workflows/amio-ci.yml` Stage 6:

| Gate | Checks |
|---|---|
| `libs/amio/tests/ci/check_no_eckit_includes.sh` | zero `#include <eckit/...>` in `src/`, `include/`, `fortran/`, **and `tests/`** |
| `ctest` `ci.no_eckit_symbols` → `check_no_eckit_symbols.sh` | `nm -D libamio.so` demangles to zero `eckit::` symbols |
| `check_build_no_eckit.sh` | core reconfigures successfully with eckit stripped from `CMAKE_PREFIX_PATH` |
| `ctest` `ci.no_eckit_recipe` → `check_no_eckit_recipe.sh` | no eckit dependency, build edge, or `eckit::` symbol anywhere — the Spack recipe, **every** CMake file (`find_package(eckit)`, an eckit link, an `if(eckit_FOUND)` guard, `AMIO_HAS_ECKIT`), and all C++ sources under `src/`, `include/`, `tests/` |

**There is no optional eckit integration.** Earlier revisions carried dormant
`#ifdef AMIO_HAS_ECKIT` code paths in `src/workers/` and test targets gated on
`if(eckit_FOUND ...)`; those have been removed. Manifests are parsed with
`conf::Config` (HELM's own YAML configuration library, already a dependency),
so no test source includes an eckit header either. `libamio.so` has no eckit
symbols and no eckit library in its link closure.

## External consumers and in-flight work (dispositions)

At absorption time the standalone repository had **no open pull requests**.
Its **open issues** and known consumers are dispositioned as follows:

| Item | Disposition |
|---|---|
| Issue #12 — adopt `[[nodiscard]]`/`[[maybe_unused]]` consistently | Recreate as a HELM issue; contribute against `libs/amio/` here |
| Issue #13 — evaluate `restrict` on public C API pointer params | Recreate as a HELM issue; contribute against `libs/amio/` here |
| Issue #14 — retrieve non-scalar dataset/var attributes | Recreate as a HELM issue; contribute against `libs/amio/` here |
| Issue #3 — GHA workflows | **Closed by this migration** — AMIO CI now lives in HELM (`.github/workflows/amio-ci.yml`) |
| Standalone `Dockerfile` (cloned HELM to build conf/halo/logs) | Moot — retired in-tree during absorption; the HELM `Dockerfile` + `amio-ci.yml` replace it |
| Spack recipe `libs/amio/packages/` | Still available in-tree at `libs/amio/packages` — `spack repo add /path/to/HELM-Project/libs/amio/packages` |

## Standalone repository status

Per the migration decision, `bbakernoaa/amio` stays **public** (read-only),
never deleted:

- Repository description: `MOVED → HELM-Project libs/amio (in-tree)`
- README banner: contributions now go through a PR against
  `bbakernoaa/HELM-Project`, path `libs/amio/`
- GitHub **archive** applied — after the absorption PR merges (ordering rule:
  never leave both locations broken at once)
