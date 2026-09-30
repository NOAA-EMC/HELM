# SPDX-License-Identifier: Apache-2.0
"""Worker-local weight provisioning for distributed execution (FR-031…FR-033).

The compiled weight matrix is generated once on the driver and reaches each
Dask worker **at most once** — never shipped per chunk or per task (FR-032).
The mechanisms:

* ``weights_fingerprint(blob)`` — content hash of the serialized matrix; the
  worker cache key is ``(regridder_uid, fingerprint)`` so a stale entry can
  never serve a different regridder's weights ("stale worker caches" edge case).
* ``provision_worker(client, key, blob)`` — driver-side ``client.run`` of
  ``install_weights`` guarded by ``_DRIVER_CACHE``: at most one sync per
  ``(client_id, key)``.
* ``install_weights(key, blob)`` — worker-side: deserialize the bytes once
  into ``axis.core._WORKER_CACHE`` and bump ``_WORKER_SYNC_COUNTS[key]``
  (the SC-005 observability counter).
* Local schedulers (synchronous / threads / processes) need no cluster: the
  applier deserializes lazily in-process and the same counters apply (FR-033).

Optional extension (not wired by default): a ``distributed.WorkerPlugin`` that
pre-warms the cache at worker start; ``client.run`` provisioning already
guarantees at-most-once, so the plugin only shifts latency, not traffic.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import dask.distributed

logger = logging.getLogger("axis")


def weights_fingerprint(blob: bytes) -> str:
    """Short content hash of a serialized weight blob (cache-key component)."""
    return hashlib.sha256(blob).hexdigest()[:16]


def install_weights(key: str, blob: bytes) -> int:
    """Worker-side: deserialize ``blob`` into the local cache under ``key``.

    Idempotent: a second call with the same key does not re-parse; it returns
    the existing sync count. Returns the number of syncs recorded for ``key``.
    """
    from . import _core
    from .core import _WORKER_CACHE, _WORKER_SYNC_COUNTS

    if key not in _WORKER_CACHE:
        _WORKER_CACHE[key] = _core.Matrix.from_bytes(blob)
        _WORKER_SYNC_COUNTS[key] = _WORKER_SYNC_COUNTS.get(key, 0) + 1
    return int(_WORKER_SYNC_COUNTS.get(key, 0))


def sync_counts() -> dict[str, int]:
    """Worker-side observability hook: per-key sync counters (SC-005)."""
    from .core import _WORKER_SYNC_COUNTS

    return dict(_WORKER_SYNC_COUNTS)


# Driver-side guard: (client_id, cache_key) -> provisioned once (FR-032).
_DRIVER_CACHE: set[tuple[str, str]] = set()


def provision_worker(client: dask.distributed.Client, key: str, blob: bytes) -> dict[str, Any]:
    """Push ``blob`` to every worker of ``client`` at most once per key.

    Returns the per-worker sync counts observed after provisioning (or the
    cached result if this client already provisioned ``key``).
    """
    client_id = str(client.id)
    if (client_id, key) in _DRIVER_CACHE:
        logger.debug("AXIS: weights %s already provisioned to client %s — skipping sync", key, client_id)
        return {}
    futures = client.run(install_weights, key, blob)
    _DRIVER_CACHE.add((client_id, key))
    logger.info("AXIS: provisioned weights %s to %d worker(s) (one-time sync)", key, len(futures))
    return dict(futures)


def worker_key(uid: str, blob: bytes) -> str:
    """The collision-proof worker cache key for a fitted regridder."""
    return f"weights_{uid}_{weights_fingerprint(blob)}"
