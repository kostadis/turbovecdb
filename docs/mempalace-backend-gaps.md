# turbovecdb → MemPalace backend: gap list

Gaps turbovecdb must close to fully satisfy what MemPalace requires of a storage
backend, derived from MemPalace's
[`backend-requirements.md`](../../mempalace/docs/design/backend-requirements.md)
(Tier A contract + Tier B implementation-specific assumptions).

**Scope & method.** This compares two `ARCHITECTURE.md` documents only — not
source. Verdicts marked *(confirm in code)* are inferences from the architecture
doc that should be checked against `collection.py` / `index.py` before acting.

**Reading the verdicts:**

- **GAP** — a real gap turbovecdb must close.
- **PARTIAL** — partly satisfied; a specific piece is missing.
- **DISSOLVED** — the requirement exists only because of ChromaDB's design; it
  does not apply to turbovecdb, usually because *SQLite is the source of truth
  and `.tvim` is a rebuildable cache*. Listed so the absence is a deliberate,
  recorded decision rather than an oversight.
- **CALLER** — the gap is on the MemPalace side (it reaches into ChromaDB
  internals); turbovecdb has nothing to close, but it can only be exercised once
  MemPalace refactors the caller.

The single fact that dissolves most of the Tier B gaps: turbovecdb's vector
index **cannot reach an unloadable/segfaulting state** — a bad or stale `.tvim`
is rebuilt from `docs.vector` on next open. ChromaDB's entire repair/quarantine/
divergence apparatus exists to survive an index that *can* corrupt on disk and
crash the process. turbovecdb removes the failure mode rather than recovering
from it.

---

## Real gaps to close

### GAP-1 — Embedder identity guard (`EmbedderIdentityMismatchError`)
**Requirement (B.7):** MemPalace expects the backend to reject reads/writes when
the embedding model differs from the one a collection was built with; the
contract names `EmbedderIdentityMismatchError`. A silent model swap is the worst
failure class — plausible-but-wrong results, no error.

**turbovecdb status: GAP (acknowledged in turbovecdb's own docs).** The
architecture doc states: *"Single embedding model per collection — mixing models
silently corrupts the vector space. A stored-model-name guard is on the backlog
(cf. MemPalace's `EmbedderIdentityMismatchError`, which this library does not yet
enforce)."* Dimensionality **is** guarded (`_commit_dim` → `DimensionMismatchError`),
but same-dim/different-model is not.

**To close:** persist an embedder identity string in the `meta` table on first
write (alongside `dim`); on subsequent writes/queries that supply an embedder,
compare and raise. turbovec ships no model, so the identity must come from the
caller-supplied `embedder` (e.g. a `name`/`model_id` attribute) — this needs a
small contract addition on the `embedder` callable.

---

### GAP-2 — `supports_contains_fast` / lexical candidate selection
**Requirement (A.4 + B.2):** `$contains` is a required operator, and ChromaDB
advertises `supports_contains_fast` because its `$contains` is FTS-backed.

**turbovecdb status: PARTIAL.** turbovecdb supports `where_document $contains`
but compiles it to `document LIKE ? ESCAPE '\'` — a full table scan, no
full-text index. Correct, but not *fast*, and there is no BM25/FTS ranking
surface at all.

**To close (only if MemPalace needs it):**
- Do **not** advertise the `supports_contains_fast` capability — that would be an
  optimistic lie. Advertise plain `$contains` support only.
- If/when MemPalace wants store-side lexical *candidate selection* (not just
  substring filtering), turbovecdb would need an FTS index over `docs.document`.
  **Likely not required:** MemPalace computes BM25 in Python over the candidate
  set returned by vector search; it only needs store-side full-text for the
  ChromaDB *corruption fallback*, which is DISSOLVED here (see GAP-7). So this is
  a capability-flag honesty fix, not a feature build, unless MemPalace adds a
  store-side lexical path.

---

### GAP-3 — Atomic metadata-only update (`update` / `supports_update`)
**Requirement (A.1):** the contract has an `update` method; backends that can do
an atomic single-round-trip update advertise `supports_update`. MemPalace's
`update_drawer` changes metadata without re-chunking/re-embedding.

**turbovecdb status: PARTIAL *(confirm in code)*.** The doc shows `add` /
`upsert` / `delete` but no `update`. `upsert` reuses the `uid` for an existing
`str_id` (remove + re-add), so metadata-only update is *expressible* as
get(include vectors) → upsert(vectors=stored) — but that is two round-trips and
re-inserts into the index. There is no atomic in-place metadata write.

**To close (optional):** add an `update` that writes `metadata`/`document` to the
`docs` row without touching the vector or the `.tvim` (metadata lives only in
SQLite, so this need not bump the index at all — potentially cheaper than
ChromaDB). If not added, MemPalace falls back to the contract's default
get+merge+upsert, which works but is non-atomic and dirties the index.

---

### GAP-4 — Version / format-skew handling
**Requirement (B.9):** MemPalace expects a story for on-disk format evolution
(ChromaDB needed pre-open BLOB→INTEGER migration etc.).

**turbovecdb status: PARTIAL.** Two sub-cases:
- **`.tvim` format skew across turbovec versions** — **CLOSED (2026-09-05),
  and exercised for real by the turbovec 0.9 → 1.0 upgrade.** `Collection::
  reload_index` treats *any* load failure as a cache miss and falls through
  to rebuild-from-SQLite; it never propagates. The 1.0 upgrade was exactly
  the feared event — 1.0 reads only format v7 and refuses the v3 files 0.9
  wrote — and no palace was bricked, because the SQLite vectors are the
  source of truth and the rebuild is automatic. The load failure is now
  logged rather than silently swallowed, so the one-time rebuild is
  explained instead of looking like a stall. Regression test:
  `tests/test_collection.py::test_rebuilds_when_tvim_is_legacy_format`.
- **SQLite schema evolution** (`docs`/`meta` columns) — no migration framework
  is mentioned. v0.1.0 is early; a documented schema-version key in `meta` and a
  migration-on-open hook would match MemPalace's expectation.

**To close:** (a) guarantee `load_index` failure degrades to rebuild, never
raises to the caller; (b) add a `schema_version` to `meta` and a forward-only
migration step on open.

---

### GAP-5 — Health / integrity surface
**Requirement (B.4 + A.2 `health`):** MemPalace's `repair status` compares
rows-in-index vs rows-in-store and runs SQLite `quick_check`. The contract's
`health()` defaults to healthy.

**turbovecdb status: PARTIAL.** Divergence as ChromaDB knows it is DISSOLVED
(the index self-heals), so most of `repair.py` is N/A. But two residual integrity
concerns remain because the store is still SQLite:
- SQLite-level corruption (`store.sqlite3` quick_check) — turbovecdb is as
  exposed to this as ChromaDB; it should expose it via `health()`.
- A way to assert "the index covers every row" after a rebuild, for callers that
  want a positive integrity signal.

**To close:** implement `health(palace)` to run a cheap `PRAGMA quick_check`
and report `store_gen`/`tvim_gen` coherence. Low effort, high diagnostic value.

---

## Gaps dissolved by turbovecdb's architecture (record, don't build)

### GAP-7 — Vector-independent recall fallback *(DISSOLVED)*
**Requirement (B.2):** MemPalace's 100%-recall promise relies on
`_bm25_only_via_sqlite()` because ChromaDB's HNSW can become unloadable and
segfault, leaving vector search dead.

**Why dissolved:** turbovecdb's vector path is always recoverable — a missing or
stale `.tvim` rebuilds from `docs.vector` sub-second. The condition the fallback
exists to survive (vector search permanently unavailable) does not arise. The
normal hybrid path keeps working, so no degraded lexical-only mode is needed.
**Caveat:** if `store.sqlite3` *itself* corrupts, both paths are gone — but so is
the source of truth, which no fallback can recover. (See GAP-5 for surfacing it.)

### GAP-8 — Index divergence probe / capacity check *(DISSOLVED)*
**Requirement (B.4):** ChromaDB needs `hnsw_capacity_status` (SQLite count vs
HNSW pickle `id_to_label`) and a `2 × sync_threshold` flush-lag tolerance.

**Why dissolved:** `store_gen`/`tvim_gen` make divergence a normal, self-resolving
state — a stale cache is reloaded on the next read, not a corruption to be
measured. There is no asynchronous flush-lag window to tolerate; writes commit to
SQLite synchronously and the index is reconciled per-query.

### GAP-9 — Physical-file corruption heuristics *(DISSOLVED)*
**Requirement (B.4):** ChromaDB sniffs `link_lists.bin`/`data_level0.bin` size
ratios and byte-sniffs `index_metadata.pickle` to avoid SIGSEGV on open.

**Why dissolved:** there are no analogous bloat-prone, segfault-on-open segment
files. `.tvim` is a single rebuildable artifact; a bad one is discarded and
rebuilt, never opened defensively.

### GAP-10 — Index-bloat tuning (`sync_threshold`/`batch_size`) *(DISSOLVED)*
**Requirement (B.6):** ChromaDB needs large batch/sync thresholds to defeat
`link_lists.bin` sparse-file bloat.

**Why dissolved:** 4-bit TurboQuant has no resize+persist feedback loop; `.tvim`
is written atomically on `flush()`/`close()`, once, not incrementally per batch.

### GAP-11 — Single-writer requirement & write serialization *(DISSOLVED / better)*
**Requirement (B.5):** ChromaDB's HNSW is not thread-safe on insert, forcing
MemPalace's one-consumer pipeline + `mine_palace_lock`.

**Why dissolved:** turbovecdb is multi-process safe by construction — file-lock-
serialized writers, lock-free readers coherent via `store_gen`, no shared mutable
index. MemPalace's serialization stays *correct* (serialized writes are always
safe) but becomes an unnecessary throughput ceiling. This is an **opportunity for
MemPalace**, not a gap for turbovecdb.

### GAP-12 — EF non-persistence / `name()` spoof *(DISSOLVED)*
**Requirement (B.7, second half):** ChromaDB 1.5 persists EF identity and rejects
a differently-named EF, forcing MemPalace's `name()="default"` spoof.

**Why dissolved:** turbovecdb ships no embedder and does not validate EF *name*
on read — the spoof is unnecessary. (Note the *positive* identity guard is still
needed — that is GAP-1, a different concern.)

### GAP-13 — External-rebuild cache invalidation *(DISSOLVED / better)*
**Requirement (B.8):** ChromaDB needs an inode+mtime freshness check to notice a
palace rebuilt on disk by another process.

**Why dissolved:** readers poll `store_gen` every query and reload automatically,
so another process's writes (or a full rebuild) are picked up on the next call
without an external freshness heuristic. **Confirm:** behaviour when the entire
`store.sqlite3` is *replaced* (inode swap) under a long-lived open connection —
WAL/connection caching may need an explicit reopen path. Minor edge.

### GAP-14 — Version-migration pre-open repair ordering *(DISSOLVED)*
**Requirement (B.9, ordering side-effects):** ChromaDB requires a specific
pre-open repair sequence because opening certain client states leaves WAL state
that crashes the next open.

**Why dissolved:** no pre-open repair pass exists or is needed; `connect()` does
no I/O and `collection()` opens cleanly. (The *format-evolution* portion of B.9
is the real residual — tracked as GAP-4.)

---

## Caller-side items (MemPalace must move, turbovecdb has nothing to close)

### CALLER-1 — Direct `chroma.sqlite3` schema reads (B.3)
MemPalace's `searcher.py` and `repair.py` read ChromaDB's private tables
(`segments`, `embedding_metadata`, `embedding_fulltext_search`, `max_seq_id`,
the `chroma:document` key) directly. On a turbovec palace these code paths are
inapplicable — turbovecdb's schema is `docs`/`meta`. turbovecdb need not mimic
ChromaDB's schema; MemPalace must route these through contract methods
(`get`/`query`/`count`) before the turbovec backend can serve them. The data
they need (document text, metadata, counts) **is** available through
turbovecdb's contract surface.

### CALLER-2 — Cosine `1 − distance` ranking math (B.1)
**Status: NO GAP on turbovecdb's side.** turbovecdb returns exact cosine
`distance = 1 − dot(q, v) ∈ [0, 2]` from its re-rank stage — precisely the range
MemPalace's `vec_sim = max(0, 1 − distance)` assumes. turbovecdb arguably
satisfies this *better* than ChromaDB (exact float32 re-rank vs raw ANN score).
The only residual is MemPalace's hard-coded cosine assumption in `_hybrid_rank`,
which is a MemPalace-side neutralization, not a turbovecdb gap.

---

## Summary table

| ID | Requirement (origin) | Verdict | Action owner |
|---|---|---|---|
| GAP-1 | Embedder identity guard (B.7) | **GAP** | turbovecdb |
| GAP-2 | `$contains` fast / lexical selection (A.4, B.2) | **PARTIAL** | turbovecdb (flag honesty) |
| GAP-3 | Atomic metadata update (A.1) | **PARTIAL** | turbovecdb (optional) |
| GAP-4 | Format/version skew (B.9) | **PARTIAL** *(confirm)* | turbovecdb |
| GAP-5 | Health / integrity surface (B.4, A.2) | **PARTIAL** | turbovecdb |
| GAP-7 | Vector-independent recall fallback (B.2) | DISSOLVED | — |
| GAP-8 | Divergence/capacity probe (B.4) | DISSOLVED | — |
| GAP-9 | Physical-file corruption heuristics (B.4) | DISSOLVED | — |
| GAP-10 | Index-bloat tuning (B.6) | DISSOLVED | — |
| GAP-11 | Single-writer serialization (B.5) | DISSOLVED (MemPalace opportunity) | — |
| GAP-12 | EF non-persistence / name spoof (B.7) | DISSOLVED | — |
| GAP-13 | External-rebuild invalidation (B.8) | DISSOLVED *(confirm inode-swap)* | — |
| GAP-14 | Pre-open repair ordering (B.9) | DISSOLVED | — |
| CALLER-1 | Direct `chroma.sqlite3` reads (B.3) | NO turbovecdb gap | MemPalace |
| CALLER-2 | Cosine `1 − distance` math (B.1) | NO gap (met, arguably better) | MemPalace |

**Net:** one clear gap (GAP-1, embedder identity), three partials worth closing
(GAP-2 flag honesty, GAP-4 format skew, GAP-5 health), one optional (GAP-3). The
bulk of MemPalace's backend complexity is ChromaDB corruption-recovery that
turbovecdb's source-of-truth-plus-rebuildable-cache design makes unnecessary —
those should be recorded as deliberately-absent, not built.
