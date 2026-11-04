# Step Completion and Storage Guarantees

AMIO publishes a step in this order: write immutable payloads, write the
completion record, then atomically replace `head.json`. Readers begin at the
published head and follow immutable commit links. A visible head is the
completion boundary; a payload file by itself is not a completed step.

## POSIX collection adapter

The POSIX adapter writes temporary files in the destination directory and
publishes immutable payloads and commits with create-exclusive hard links. It
replaces `head.json` with a same-directory rename. A reader that opens the
collection after a successful publication sees the new head and can traverse
the referenced commit chain. If publication acknowledgement is ambiguous or
the head cannot be read, AMIO returns `AMIO_ERR_PUBLICATION_PENDING` and keeps
the step in reconciliation; it does not report the step as failed or complete
based only on a timeout.

The adapter supports three requested levels:

| Request | Adapter actions | Limit |
| --- | --- | --- |
| `visibility` | Close payload and metadata files before publication. | Does not request `fsync`. |
| `file_sync` | `fsync` payload, commit, and head files before reporting publication. | Does not sync containing directories. |
| `directory_sync` | Performs file syncs and `fsync` on payload, commit, and collection directories after publication. | The result reflects successful POSIX calls; it cannot certify filesystem, controller, or device power-loss behavior. |

The public evolution API defaults to `visibility`. Set
`completion.level: file_sync` or `completion.level: directory_sync` in the
context configuration to request the stronger POSIX levels. Successful
publication records the achieved level, and operation handles expose the
durable milestone only after the requested sync calls complete. `directory_sync`
syncs each payload and metadata file, their containing directory chain through
the collection root, and the parent directory containing the collection.

The adapter does not claim that `amio_flush` or a successful `directory_sync`
request guarantees stable-media durability on every filesystem or storage
stack. Network filesystems and object-store mounts require deployment-specific
qualification. The current POSIX calls are the reported guarantee; they do not
certify filesystem, controller, or device power-loss behavior. Stronger sync
levels are rejected for TensorStore Zarr because the adapter has no recursive
stable-media sync contract for array directories.

## TensorStore and other key-value stores

The generic KvStore catalog requires explicit certification of atomic
single-key replacement and fresh reads after writes. It also requires
atomic create-if-absent or a caller-declared external exclusive-owner
precondition. Ambiguous writes are reconciled by a fresh read. Generic
TensorStore APIs do not establish backend-specific atomicity, freshness, or
stable-media behavior, and generic `sync` is unsupported. The opt-in fetched
TensorStore v0.1.86 build exercises the file KvStore adapter and the public
Zarr v3 collection path, including two- and four-rank writes where each
producer owns complete, disjoint shards. Local file Zarr arrays are catalogued
as array directories and published at visibility after every writer's futures
complete. The POSIX catalog rejects file-sync and directory-sync requests for
these arrays because it has no recursive stable-media sync contract. Cloud
destinations and arbitrary concurrent partial updates to one shard are not
certified. A destination must not be advertised until its driver has passed
the store-specific guarantee tests.

## Qualification scope

`integration.catalog_guarantees` exercises the local POSIX adapter at
visibility, file-sync, and directory-sync request levels, checks reader
visibility through the published chain while 12 head generations are
replaced, and injects an unreadable head to verify that an already-written
commit remains in reconciliation. It does not measure crash consistency or
power-loss behavior. KvStore catalog unit tests
use an in-memory mock to exercise capability rejection and ambiguous
acknowledgement reconciliation. `unit.tensorstore_kvstore` checks local file
driver create-only writes, including a concurrent two-writer race where exactly
one create succeeds, replacement, fresh reads, size, missing keys, and default
capability rejection. `unit.evolution_zarr` checks the single-rank public Zarr
write/catalog/read path including selected reads. The MPI Zarr region tests
exercise shard-aligned multi-writer output and public readback on the local
file driver. These tests do not certify cloud stores, concurrent partial
writes to one shard, crash consistency, or power-loss behavior.
