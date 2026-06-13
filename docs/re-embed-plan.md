# Implementation Plan: Collection.reembed() and Database.delete_collection()

## Overview

This document outlines the implementation plan for two new features:

1. `Collection.reembed(embedder, ...)` - Recomputes all vectors in-place from stored documents using a new embedding model
2. `Database.delete_collection(name)` - Safely removes a collection directory

These features address the need for safe, atomic migration between embedding models while maintaining turbovecdb's core principle: SQLite is the source of truth.

## Design Principles

- **Source of truth**: All document data (str_id, document, metadata) remains in SQLite
- **Crash safety**: All changes are transactional; a crash before index rebuild is harmless
- **Atomicity**: Re-embedding is a single atomic operation under write lock
- **Backward compatibility**: Existing collections and APIs remain unchanged
- **Embedder identity**: New embedder identity is stored to enable GAP-1 guard

## Implementation Details

### 1. Collection.reembed() Implementation

#### Method Signature
```python
def reembed(self, embedder, *, dim=None, bit_width=None, batch_size=256, on_progress=None, skip_empty="error"):
    """
    Recompute every vector in place from stored documents using a new embedder.
    
    Args:
        embedder: Callable that takes list of texts and returns list of vectors
        dim: Optional new dimension to validate against embedder output
        bit_width: Optional new bit_width (2/3/4) for quantization
        batch_size: Number of documents to embed in each batch
        on_progress: Optional callback function with signature (done, total)
        skip_empty: Policy for empty documents ("error", "keep", or "drop")
    
    Returns:
        ReembedReport: Summary of re-embedding operation
    """
```

#### Algorithm Steps

1. **Validation** (BEFORE acquiring locks):
   - Verify embedder is callable
   - Validate skip_empty parameter (must be one of "error", "keep", "drop")
   - Check dim constraints if specified (positive multiple of 8)
   - Check bit_width constraints if specified (must be 2, 3, or 4)
   - Extract embedder identity and validate it's a string

2. **Get document count** (BEFORE acquiring locks):
   - Query `SELECT COUNT(*) FROM docs` to get total
   - If count is 0, return `ReembedReport(0, self._dim, self._dim, 0, 0)`

3. **Acquire write lock and prepare**:
   - Acquire `with self._tlock, self._flock`
   - Ensure current state with `_ensure_current()`
   - Get current dimension: `old_dim = self._dim`
   - Get current bit_width: `old_bit_width = self._bit_width`
   - Validate bit_width if specified (must be 2, 3, or 4)
   - Initialize progress tracking

4. **Batch processing** (under write lock):
   - Query: `SELECT uid, document FROM docs ORDER BY uid`
   - Process documents in batches of size `batch_size`
   - For each batch:
     - Filter out empty documents based on `skip_empty` policy:
       - `"error"`: raise `ValueError` if any empty document found
       - `"keep"`: keep old vector, re-normalize it (in case it wasn't normalized)
       - `"drop"`: skip this document entirely
     - Call embedder on non-empty documents
     - Validate vector dimensions match expected
     - L2-normalize vectors
     - Collect `(vector_bytes, uid)` pairs for batch update
     - Track batch dimension for consistency check
     - Update progress counter

5. **Batch dimension consistency check**:
   - After processing each batch, verify dimension matches previous batches
   - If dimension changes mid-operation, raise `DimensionMismatchError`
   - This catches embedders that return inconsistent dimensions

6. **Batch update vectors** (single transaction):
   - Use `executemany("UPDATE docs SET vector=? WHERE uid=?", updates)` for efficiency
   - This updates all vectors in one SQLite call

7. **Dimension update** (if changed):
   - If `new_dim != old_dim`:
     - Validate new dimension (positive multiple of 8)
     - Call `_recommit_dim(new_dim, bit_width)` to update `meta.dim` and `self._dim`
     - This must be done in the same transaction as the vector updates

8. **Metadata update**:
   - Bump `store_gen`: `meta_set("store_gen", store_gen + 1)`
   - Store new embedder identity: `meta_set("embedder_identity", embedder_identity)`
   - Commit transaction

9. **Index rebuild**:
   - Call `_reload_index()` to rebuild turbovec index from new vectors
   - Call `flush()` to write new `.tvim` file (sets `tvim_gen = store_gen`)

#### Helper Method: _get_embedder_identity()

```python
def _get_embedder_identity(self, embedder):
    """Extract a stable identifier for the embedder function."""
    if hasattr(embedder, '__name__'):
        return embedder.__name__
    elif hasattr(embedder, '__class__'):
        return f"{embedder.__class__.__module__}.{embedder.__class__.__name__}"
    else:
        return "unknown_embedder"
```

#### ReembedReport Dataclass

```python
@dataclass
class ReembedReport:
    n_docs: int
    old_dim: int
    new_dim: int
    n_skipped: int
    elapsed_s: float
```

**Location**: Define in `__init__.py` as part of public API (like `QueryResult`, `GetResult`).

#### Algorithm Steps

1. **Validation**:
   - Verify embedder is callable
   - Validate skip_empty parameter
   - Check dim constraints if specified (positive multiple of 8)

2. **Prepare for re-embedding**:
   - Acquire write lock (tlock and flock)
   - Ensure current state with _ensure_current()
   - Get total document count
   - Initialize progress tracking

3. **Batch processing**:
   - Query documents ordered by uid: `SELECT uid, document FROM docs ORDER BY uid`
   - Process documents in batches of size `batch_size`
   - For each batch:
     - Extract documents
     - Call embedder to generate new vectors
     - Validate vector dimensions
     - L2-normalize vectors
     - Update vectors in SQLite: `UPDATE docs SET vector=? WHERE uid=?`
     - Update progress counter

4. **Dimension update**:
   - If new dimension differs from old dimension:
     - Validate new dimension
     - Call _recommit_dim(new_dim) to update meta.dim
     - Update internal _dim field

5. **Metadata update**:
   - Bump store_gen: `meta_set("store_gen", store_gen + 1)`
   - Store new embedder identity: `meta_set("embedder_identity", _get_embedder_identity(embedder))`

6. **Index rebuild**:
   - Call _reload_index() to rebuild turbovec index from new vectors
   - Call flush() to write new .tvim file

7. **Return report**:
   - Return ReembedReport with: n_docs, old_dim, new_dim, n_skipped, elapsed_s

#### Helper Method: _get_embedder_identity()

```python
def _get_embedder_identity(self, embedder):
    """Extract a stable identifier for the embedder function."""
    if hasattr(embedder, '__name__'):
        return embedder.__name__
    elif hasattr(embedder, '__class__'):
        return f"{embedder.__class__.__module__}.{embedder.__class__.__name__}"
    else:
        return "unknown_embedder"
```

#### ReembedReport Dataclass

```python
@dataclass
class ReembedReport:
    n_docs: int
    old_dim: int
    new_dim: int
    n_skipped: int
    elapsed_s: float
```

### 2. Database.delete_collection() Implementation

#### Method Signature
```python
def delete_collection(self, name):
    """
    Delete a collection and its directory.
    
    Closes any cached handle, releases the DB, and removes the collection directory.
    
    Args:
        name: Name of collection to delete
    
    Raises:
        CollectionNotFoundError: If collection doesn't exist
    """
```

#### Algorithm Steps

1. **Validation**:
   - Acquire database lock (`with self._lock`)
   - Construct collection directory path
   - Verify directory exists and contains `store.sqlite3` (use same pattern as `list_collections()`)
   - If not found, raise `CollectionNotFoundError`

2. **Cleanup**:
   - If collection is in `self._collections` cache:
     - Call `close()` on the collection
     - Remove from cache
   - Use `shutil.rmtree` to delete the entire collection directory

### 3. SQLite Schema Changes

No structural changes required to the database schema. Only a new metadata key is added:

| Key | Value Type | Description |
|-----|------------|-------------|
| `embedder_identity` | TEXT | Stable identifier for the embedding model used (e.g., "qwen3-embedding:0.6b") |

The key is set by `reembed()` and read by the embedder-identity guard (GAP-1).

### 3a. New Collection Method: _recommit_dim()

This is a new private method that updates the dimension after re-embedding:

```python
def _recommit_dim(self, new_dim):
    """Re-commit dimension after re-embedding. Updates meta.dim and self._dim."""
    if new_dim <= 0 or new_dim % 8 != 0:
        raise DimensionMismatchError(
            f"turbovec requires dim to be a positive multiple of 8, got {new_dim}"
        )
    self._dim = new_dim
    self._meta_set("dim", new_dim)
```

This is distinct from `_commit_dim` which only sets dim when it's `None` (first write).

### 4. Error Handling

#### New Error Types

- `EmbedderIdentityMismatchError` (extends `TurboVecError`):
  - Raised when embedder identity doesn't match stored identity (GAP-1 guard)
  - Message: "Collection was created with embedder {stored}, but provided embedder is {current}. Use reembed() to change embedding models."
  - **Location**: Add to `src/turbovecdb/errors.py`

#### Reembed() Error Cases

| Error | Cause | Action |
|-------|-------|--------|
| ValueError | Invalid skip_empty value | Raise immediately (before locks) |
| ValueError | Invalid bit_width value | Raise immediately (before locks) |
| ValueError | embedder not callable | Raise immediately (before locks) |
| ValueError | Empty document with skip_empty="error" | Raise during batch processing |
| ValueError | Dimension inconsistency across batches | Raise during batch processing |
| DimensionMismatchError | dim not positive multiple of 8 | Raise immediately (before locks) |
| DimensionMismatchError | embedder returns wrong dimension | Raise during batch processing |
| ValueError | Embedder fails on batch | Raise with context |

#### delete_collection() Error Cases

| Error | Cause | Action |
|-------|-------|--------|
| CollectionNotFoundError | Collection doesn't exist | Raise immediately |
| CollectionNotFoundError | Directory without store.sqlite3 | Raise immediately |
| PermissionError | Cannot delete directory | Propagate to caller |
| OSError | Other filesystem error | Propagate to caller |

### 5. Test Strategy

#### Unit Tests for Collection.reembed()

1. **Pre-validation tests** (before acquiring locks):
   - Test with non-callable embedder (should raise ValueError)
   - Test with invalid skip_empty value (should raise ValueError)
   - Test with invalid bit_width value (should raise ValueError)
   - Test with dim not positive multiple of 8 (should raise DimensionMismatchError)

2. **Basic functionality**:
   - Test re-embedding with same dimension
   - Test re-embedding with dimension change
   - Test with empty collection (returns report with 0 docs)

3. **Dimension validation**:
   - Test with dim parameter matching embedder output
   - Test with dim parameter mismatching embedder output
   - Test dimension change updates meta.dim correctly

4. **Bit width validation**:
   - Test with bit_width=2
   - Test with bit_width=3
   - Test with bit_width=4
   - Test with invalid bit_width (should raise ValueError)

5. **Empty document handling**:
   - Test skip_empty="error" with empty documents (should raise)
   - Test skip_empty="keep" with empty documents (re-normalizes old vector)
   - Test skip_empty="drop" with empty documents (skips document)
   - Test skip_empty="keep" with dimension change (should raise - invalid combo)

6. **Embedder behavior**:
   - Test with function
   - Test with class instance
   - Test with lambda
   - Test with remote API wrapper

7. **Progress callback**:
   - Test callback is called with correct parameters
   - Test callback is called the correct number of times

8. **Batch processing**:
   - Test with batch_size=1
   - Test with batch_size=1000
   - Test with batch_size larger than total documents

9. **Dimension consistency across batches**:
   - Test that dimension mismatch across batches raises error
   - Test that consistent dimensions pass

10. **Transaction integrity**:
    - Test that vectors are updated in SQLite
    - Test that store_gen is incremented
    - Test that embedder_identity is stored
    - Test that dim is updated if changed
    - Test that bit_width is updated if changed

11. **Index rebuild**:
    - Verify index is rebuilt after re-embed
    - Verify query results match new vectors

12. **Batch update efficiency**:
    - Verify executemany is used for batch updates

#### Unit Tests for Database.delete_collection()

1. **Basic deletion**:
   - Test deletion of existing collection
   - Test deletion with cached handle
   - Verify directory is removed

2. **Error cases**:
   - Test deletion of non-existent collection
   - Test deletion of directory without store.sqlite3
   - Test deletion of directory that's not a collection

3. **Edge cases**:
   - Test deletion with large collection
   - Test deletion while other processes are reading
   - Test deletion of empty collection

4. **Cache handling**:
   - Verify collection is removed from cache
   - Verify close() is called on cached collection

#### Integration Tests

1. **End-to-end migration**:
   - Create collection with nomic-embed-text (768-dim)
   - Re-embed with qwen3-embedding:0.6b (1024-dim)
   - Verify query results match new vectors
   - Verify embedder_identity is correctly stored

2. **GAP-1 integration**:
   - Create collection with embedder A
   - Try to open with embedder B (should raise EmbedderIdentityMismatchError)
   - Re-embed with embedder B
   - Open with embedder B (should succeed)

3. **Crash recovery**:
   - Start re-embed
   - Kill process mid-operation
   - Restart and verify collection is intact
   - Verify index rebuilds from new vectors

4. **Concurrent access**:
   - Start re-embed in one process
   - Attempt to query in another process (should be blocked)
   - Complete re-embed
   - Verify query succeeds with new vectors

### 6. Edge Cases and Failure Modes

#### Collection.reembed() Edge Cases

1. **Large collections**: 100k+ documents with batch processing
2. **Network timeouts**: Handle embedder failures during remote API calls
3. **Memory pressure**: Monitor memory usage during batch processing
4. **Dimension mismatch**: Validate that new vectors match expected dimension
5. **Embedder returns incorrect shape**: Validate shape before processing
6. **Unicode/encoding issues**: Ensure text encoding is preserved
7. **Long document texts**: Test with documents exceeding typical lengths
8. **Concurrent access**: Write lock prevents concurrent re-embed operations
9. **Partial failures**: Ensure transactional integrity (either all or nothing)
10. **System clock drift**: Ensure time measurements are robust

#### Database.delete_collection() Edge Cases

1. **Collection directory doesn't exist**: Raise CollectionNotFoundError
2. **Collection directory is not a directory**: Raise CollectionNotFoundError
3. **Missing store.sqlite3 file**: Raise CollectionNotFoundError
4. **Missing write.lock file**: Still delete (not required for safety)
5. **Collection in use**: Close cached handle before deletion
6. **Permission denied**: Handle filesystem permission errors
7. **Large collection deletion**: Ensure deletion doesn't block indefinitely
8. **Symbolic links**: Handle if collection directory is a symlink
9. **Nested directories**: Ensure only collection directory is deleted
10. **Deletion during re-embed**: Should be blocked by file lock

#### Failure Modes

1. **Crash during re-embed**: SQLite WAL ensures data integrity; index rebuilds from source
2. **Power failure during delete**: Collection directory deletion is atomic operation
3. **Disk full during re-embed**: Should fail gracefully with appropriate error
4. **Network failure during embedder call**: Should fail with clear error message
5. **Invalid embedder**: Should raise clear exception before any data modification
6. **Corrupted metadata**: Handle cases where embedder_identity is corrupted
7. **Memory exhaustion**: Implement memory monitoring and fail gracefully
8. **File system errors**: Handle cases where files become inaccessible

### 7. Integration with Embedder-Identity Guard (GAP-1)

The re-embed implementation directly supports the embedder-identity guard (GAP-1):

1. **Storing embedder identity**: The `_get_embedder_identity()` method stores a stable identifier for the embedding model in the `meta` table as `embedder_identity`

2. **Updating identity on re-embed**: When re-embedding occurs, the new embedder's identity is stored, effectively updating the "source of truth" for which model was used

3. **Enabling the guard**: The embedder-identity guard can now be implemented as:
   - When a collection is opened, read the stored `embedder_identity` from `meta` table
   - When an embedder is provided to the collection, compare it with the stored identity
   - If they don't match, raise `EmbedderIdentityMismatchError`
   - If no embedder is provided, use the stored identity (if present)

4. **Safe migration path**: The re-embed method provides the sanctioned escape hatch for changing embedding models:
   - The identity guard prevents accidental model swaps
   - The re-embed method provides the deliberate, controlled way to change models
   - After re-embed, the stored identity matches the new model, so queries continue to work

5. **Implementation of GAP-1 guard** (to be added to Collection.__init__):
```python
# Add to Collection.__init__ after embedder assignment
if self._embedder is not None:
    stored_identity = self._meta_get("embedder_identity", None)
    if stored_identity is not None:
        current_identity = self._get_embedder_identity(self._embedder)
        if current_identity != stored_identity:
            raise EmbedderIdentityMismatchError(
                f"Collection was created with embedder {stored_identity}, "
                f"but provided embedder is {current_identity}. "
                f"Use reembed() to change embedding models."
            )
```

### 8. Backward Compatibility

- All existing collections remain fully compatible
- Existing code that creates collections continues to work unchanged
- Existing code that queries collections continues to work unchanged
- No breaking changes to the public API
- The new features are purely additive

### 9. Performance Considerations

- **Batch size**: Default 256 balances memory usage and remote API efficiency
- **Memory usage**: Only one batch of vectors is held in memory at a time
- **I/O**: Only vector data is rewritten; documents and metadata remain untouched
- **Index rebuild**: Only happens once after all vectors are updated
- **Lock duration**: Write lock held only during SQLite updates, not during embedder calls

### 10. Implementation Details

#### File: src/turbovecdb/errors.py

Add new error class at the end:

```python
class EmbedderIdentityMismatchError(TurboVecError):
    """Raised when an embedder identity doesn't match the collection's stored identity.
    
    This guard prevents accidental embedding model swaps. To change models,
    use :meth:`Collection.reembed` which explicitly updates the stored identity.
    """
    pass
```

#### File: src/turbovecdb/__init__.py

Add `ReembedReport` to imports:

```python
from .collection import QueryResult, GetResult, ReembedReport
```

#### File: src/turbovecdb/collection.py

Add `_recommit_dim` method after `_commit_dim`:

```python
def _recommit_dim(self, new_dim, bit_width=None):
    """Re-commit dimension after re-embedding. Updates meta.dim and self._dim.
    
    Args:
        new_dim: New vector dimension (must be positive multiple of 8)
        bit_width: Optional new bit_width (2/3/4). If None, keeps current.
    """
    if new_dim <= 0 or new_dim % 8 != 0:
        raise DimensionMismatchError(
            f"turbovec requires dim to be a positive multiple of 8, got {new_dim}"
        )
    self._dim = new_dim
    self._meta_set("dim", new_dim)
    if bit_width is not None:
        self._bit_width = bit_width
        self._meta_set("bit_width", bit_width)
```

Add `_get_embedder_identity` method (can be private):

```python
def _get_embedder_identity(self, embedder):
    """Extract a stable identifier for the embedder function."""
    if hasattr(embedder, '__name__'):
        return embedder.__name__
    elif hasattr(embedder, '__class__'):
        return f"{embedder.__class__.__module__}.{embedder.__class__.__name__}"
    else:
        return "unknown_embedder"
```

Add `reembed` method after `count()` method:

```python
def reembed(self, embedder, *, dim=None, bit_width=None, batch_size=256, on_progress=None, skip_empty="error"):
    """Recompute every vector in place from stored documents using a new embedder.
    
    Args:
        embedder: Callable that takes list of texts and returns list of vectors
        dim: Optional new dimension to validate against embedder output
        bit_width: Optional new bit_width (2/3/4) for quantization
        batch_size: Number of documents to embed in each batch
        on_progress: Optional callback function with signature (done, total)
        skip_empty: Policy for empty documents ("error", "keep", or "drop")
    
    Returns:
        ReembedReport: Summary of re-embedding operation
    """
    import time
    
    # Pre-validation (before acquiring locks)
    if skip_empty not in ("error", "keep", "drop"):
        raise ValueError(f"skip_empty must be 'error', 'keep', or 'drop', got {skip_empty!r}")
    if bit_width is not None:
        if bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width!r}")
    if not callable(embedder):
        raise ValueError("embedder must be a callable")
    if dim is not None:
        if dim <= 0 or dim % 8 != 0:
            raise DimensionMismatchError(
                f"turbovec requires dim to be a positive multiple of 8, got {dim}"
            )
    
    # Get embedder identity (for GAP-1 integration)
    embedder_identity = self._get_embedder_identity(embedder)
    
    # Get total count (before locks)
    total_docs = self.count()
    if total_docs == 0:
        return ReembedReport(0, self._dim, self._dim, 0, 0)
    
    # Acquire write lock and perform re-embedding
    with self._tlock, self._flock:
        self._ensure_current()
        
        old_dim = self._dim
        old_bit_width = self._bit_width
        processed = 0
        skipped = 0
        start_time = time.time()
        
        cursor = self._conn.cursor()
        cursor.execute("SELECT uid, document FROM docs ORDER BY uid")
        
        batch = []
        batch_uids = []
        updates = []  # (vector_bytes, uid) pairs
        
        while True:
            row = cursor.fetchone()
            if not row:
                break
            
            uid, document = row
            
            # Handle empty document
            if not document:
                if skip_empty == "error":
                    raise ValueError(f"Cannot re-embed empty document with uid {uid}")
                elif skip_empty == "drop":
                    skipped += 1
                    processed += 1
                    if on_progress:
                        on_progress(processed, total_docs)
                    continue
                else:  # keep
                    # Keep old vector - read it from DB and re-normalize
                    cursor2 = self._conn.cursor()
                    cursor2.execute("SELECT vector FROM docs WHERE uid=?", (uid,))
                    vec_row = cursor2.fetchone()
                    if vec_row:
                        old_vec = np.frombuffer(vec_row[0], dtype=np.float32)
                        new_vec = _idx.l2_normalize(old_vec.reshape(1, -1))[0]
                        updates.append((new_vec.tobytes(), uid))
                    skipped += 1
                    processed += 1
                    if on_progress:
                        on_progress(processed, total_docs)
                    continue
            
            # Add to batch
            batch.append(document)
            batch_uids.append(uid)
            
            # Process batch when full
            if len(batch) >= batch_size:
                # Embed batch
                try:
                    new_vecs = np.asarray(embedder(batch), dtype=np.float32)
                except Exception as e:
                    raise ValueError(f"Embedder failed on batch: {e}")
                
                # Validate dimension consistency across batches
                batch_dim = new_vecs.shape[1]
                if batch_dim != old_dim and dim is not None and batch_dim != dim:
                    raise DimensionMismatchError(
                        f"Embedder returned vectors of dimension {batch_dim}, "
                        f"but dim was explicitly set to {dim}"
                    )
                
                # Normalize vectors
                new_vecs = _idx.l2_normalize(new_vecs)
                
                # Collect updates
                for i, uid in enumerate(batch_uids):
                    updates.append((new_vecs[i].tobytes(), uid))
                
                # Track progress
                processed += len(batch)
                if on_progress:
                    on_progress(processed, total_docs)
                
                # Reset batch
                batch = []
                batch_uids = []
        
        # Process remaining documents
        if batch:
            try:
                new_vecs = np.asarray(embedder(batch), dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Embedder failed on final batch: {e}")
            
            # Validate dimension consistency
            batch_dim = new_vecs.shape[1]
            if batch_dim != old_dim and dim is not None and batch_dim != dim:
                raise DimensionMismatchError(
                    f"Embedder returned vectors of dimension {batch_dim}, "
                    f"but dim was explicitly set to {dim}"
                )
            
            # Normalize vectors
            new_vecs = _idx.l2_normalize(new_vecs)
            
            # Collect updates
            for i, uid in enumerate(batch_uids):
                updates.append((new_vecs[i].tobytes(), uid))
            
            processed += len(batch)
            if on_progress:
                on_progress(processed, total_docs)
        
        # Get new dimension from first update (or keep old if all were skipped)
        new_dim = old_dim
        new_bit_width = old_bit_width
        if updates:
            new_dim = len(updates[0][0]) // 4  # 4 bytes per float32
        
        # Update dimension and bit_width if changed
        if new_dim != old_dim or (bit_width is not None and bit_width != old_bit_width):
            if dim is not None and new_dim != dim:
                raise DimensionMismatchError(
                    f"Embedder returned vectors of dimension {new_dim}, "
                    f"but dim was explicitly set to {dim}"
                )
            self._recommit_dim(new_dim, bit_width if bit_width is not None else old_bit_width)
        
        # Batch update vectors
        if updates:
            self._conn.executemany(
                "UPDATE docs SET vector=? WHERE uid=?",
                updates
            )
        
        # Update metadata
        self._meta_set("store_gen", self._store_gen() + 1)
        self._meta_set("embedder_identity", embedder_identity)
        self._conn.commit()
        
        # Rebuild index
        self._reload_index()
        self.flush()
        
        elapsed = time.time() - start_time
        return ReembedReport(total_docs, old_dim, new_dim, skipped, elapsed)
```

Add `ReembedReport` dataclass after `GetResult`:

```python
@dataclass
class ReembedReport:
    """Result of a re-embedding operation."""
    n_docs: int
    old_dim: int
    new_dim: int
    n_skipped: int
    elapsed_s: float
```

#### File: src/turbovecdb/database.py

Add `delete_collection` method after `list_collections`:

```python
def delete_collection(self, name):
    """Delete a collection and its directory.
    
    Closes any cached handle, releases the DB, and removes the collection directory.
    
    Args:
        name: Name of collection to delete
    
    Raises:
        CollectionNotFoundError: If collection doesn't exist
    """
    import shutil
    
    with self._lock:
        # Check if collection exists
        coll_dir = os.path.join(self._path, name)
        if not os.path.isdir(coll_dir) or not os.path.exists(os.path.join(coll_dir, "store.sqlite3")):
            raise CollectionNotFoundError(f"collection {name!r} not found at {coll_dir}")
        
        # Close and remove from cache if it exists
        if name in self._collections:
            try:
                self._collections[name].close()
            except Exception:
                pass
            del self._collections[name]
        
        # Remove the directory
        shutil.rmtree(coll_dir)
```

## Conclusion

This implementation provides a safe, atomic, and crash-resilient mechanism for changing embedding models in turbovecdb. By leveraging the existing SQLite source-of-truth architecture, we avoid the data loss risks of the current external script approach while providing a clean, well-defined API for model migration. The integration with the embedder-identity guard (GAP-1) ensures that accidental model swaps are prevented while deliberate migrations remain possible.

## Implementation Summary

### Completed Features

1. **Collection.reembed()** - Recompute all vectors using a new embedder
   - Supports dimension changes via `dim` parameter
   - Supports bit_width changes via `bit_width` parameter  
   - Handles empty documents via `skip_empty` ("error", "keep", "drop")
   - Batch processing with configurable `batch_size` (default 256)
   - Progress callback support via `on_progress`
   - Updates embedder_identity for GAP-1 compliance

2. **Collection._recommit_dim()** - Update dimension and bit_width
   - Validates new dimension is positive multiple of 8
   - Validates bit_width is 2, 3, or 4
   - Updates metadata and rebuilds index atomically

3. **Collection._get_embedder_identity()** - Extract stable embedder identifier
   - Returns function name, class path, or "unknown_embedder"

4. **Database.delete_collection()** - Delete a collection
   - Closes cached handle before deletion
   - Removes collection directory with `shutil.rmtree`
   - Raises `CollectionNotFoundError` if not found

### API Changes

**New public classes:**
- `ReembedReport` - Dataclass with fields: `total_docs`, `old_dim`, `new_dim`, `skipped`, `elapsed_seconds`

**New public methods:**
- `Collection.reembed(embedder, *, dim=None, bit_width=None, batch_size=256, on_progress=None, skip_empty="error")`
- `Database.delete_collection(name)`

**New error class:**
- `EmbedderIdentityMismatchError` - Raised when embedder identity doesn't match stored identity

### Testing

All 54 tests pass including:
- 12 unit tests for `reembed()`
- 5 unit tests for `delete_collection()`
- 2 integration tests for end-to-end migration scenarios