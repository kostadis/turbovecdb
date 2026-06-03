# MemPalace TurboVec Migration Handoff (Claude Opus 4.6)

Use this document as a copy/paste handoff for an engineer who just cloned
`mempalace` and wants Claude Opus 4.6 to perform the migration work.

---

## Goal

Modify MemPalace so TurboVecDB is the default storage backend (replacing
ChromaDB as default), while preserving migration safety and existing user data
access paths where possible.

The repository already contains a TurboVec backend adapter under
`mempalace/backends/turbovec.py`, plus backend registry plumbing. The work is to
finish the product-level transition cleanly.

---

## Operator Workflow

1. Clone repository.
2. Open this file.
3. Copy the prompt below into Claude Opus 4.6.
4. Let Claude implement, test, and summarize.

---

## Copy/Paste Prompt For Claude Opus 4.6

You are editing the MemPalace repository.

Task: make TurboVecDB the default backend for MemPalace, replacing ChromaDB as
the default path, while keeping the codebase stable and migration-safe.

### Requirements

1. Backend defaults
   - Change default backend resolution from `chroma` to `turbovec`.
   - Keep explicit override behavior:
     - CLI/explicit backend arg (if present)
     - per-palace config override
     - `MEMPALACE_BACKEND` env var
     - auto-detect on-disk artifacts
   - Preserve ability to select Chroma explicitly for legacy palaces.

2. Packaging/dependencies
   - Ensure `turbovecdb` is in core runtime dependencies (not optional-only).
   - Keep Chroma support available as optional/legacy path if feasible.
   - Update entry points and install docs so a normal install can run TurboVec
     without extra manual steps.

3. Chroma-coupled code paths
   - Audit and fix runtime paths that assume Chroma internals, including:
     - MCP server imports and caches tied directly to Chroma classes
     - search fallback paths that query `chroma.sqlite3` directly
   - Refactor these paths behind backend-aware behavior.
   - If a fallback is inherently Chroma-specific, gate it so TurboVec runs do
     not break and surface clear behavior.

4. Data layout and migration UX
   - Confirm/standardize TurboVec on-disk layout under palace paths.
   - Ensure a user can regenerate a palace using TurboVec with documented steps.
   - Add migration notes for users moving from Chroma to TurboVec.

5. Tests
   - Update/add tests for:
     - backend default resolution
     - TurboVec default flow in mine/search/status
     - explicit Chroma selection still working
   - Keep/adjust existing tests so CI remains green.

6. Documentation
   - Update README and docs to describe TurboVec as default.
   - Add a concise migration section: existing Chroma users must regenerate or
     explicitly opt into Chroma for legacy access.

### Constraints

- Do not remove Chroma support entirely unless absolutely required.
- Do not use destructive git commands.
- Keep changes focused and reviewable.
- Run relevant tests and include commands/results in your summary.

### Deliverables

1. Code changes implementing TurboVec default backend behavior.
2. Updated docs and dependency metadata.
3. Test updates and passing test evidence.
4. A short migration note for operators.

After implementation, provide:
- changed files list
- rationale for each major change
- test commands run and outcomes
- any residual risks or follow-ups

---

## Acceptance Criteria (Human Review)

- Fresh install can run MemPalace with TurboVec without extra backend setup.
- `mine/search/status` work using TurboVec by default.
- Setting `MEMPALACE_BACKEND=chroma` still works for legacy users.
- Chroma-specific fallback logic does not crash TurboVec runs.
- Docs clearly state default backend and migration expectations.

