"""Test LRU cache for database paths and locks."""

import threading
import time
import turbovecdb.service


def test_get_db_returns_same_instance():
    """Test that _get_db returns the same Database instance for the same path."""
    # Clear any existing state
    turbovecdb.service._databases.clear()
    
    path = "/tmp/test_db"
    
    # Get database twice
    db1 = turbovecdb.service._get_db(path)
    db2 = turbovecdb.service._get_db(path)
    
    # Should be the same instance
    assert db1 is db2
    
    # Clean up
    turbovecdb.service._databases.clear()


def test_get_db_creates_different_instances():
    """Test that _get_db creates different instances for different paths."""
    # Clear any existing state
    turbovecdb.service._databases.clear()
    
    path1 = "/tmp/test_db_1"
    path2 = "/tmp/test_db_2"
    
    # Get databases for different paths
    db1 = turbovecdb.service._get_db(path1)
    db2 = turbovecdb.service._get_db(path2)
    
    # Should be different instances
    assert db1 is not db2
    
    # Clean up
    turbovecdb.service._databases.clear()


def test_lock_cache_behavior():
    """Test that _lock_for returns the same lock for the same path."""
    # Clear any existing state
    turbovecdb.service._locks.clear()
    
    path = "/tmp/test_db"
    
    # Get lock twice
    lock1 = turbovecdb.service._lock_for(path)
    lock2 = turbovecdb.service._lock_for(path)
    
    # Should be the same instance
    assert lock1 is lock2
    
    # Clean up
    turbovecdb.service._locks.clear()


def test_lru_eviction_for_databases():
    """Test that least recently used databases are evicted when cache limit is exceeded."""
    # Clear any existing state
    turbovecdb.service._databases.clear()
    # Temporarily reduce limit for testing
    original_max = turbovecdb.service._MAX_CACHED_ITEMS
    turbovecdb.service._MAX_CACHED_ITEMS = 3
    
    try:
        # Access 4 different paths to exceed limit of 3
        paths = [f"/tmp/test_db_{i}" for i in range(4)]
        dbs = []
        
        for path in paths:
            db = turbovecdb.service._get_db(path)
            dbs.append(db)
        
        # First path should have been evicted (LRU)
        # Accessing it again should create a new instance
        db_first_again = turbovecdb.service._get_db(paths[0])
        
        # The first access should be a different instance (evicted and recreated)
        assert db_first_again is not dbs[0]
        # But should be the same as the newly created one
        assert db_first_again is turbovecdb.service._get_db(paths[0])
        
        # Cache should contain the last 3 accessed paths
        assert len(turbovecdb.service._databases) == 3
        
    finally:
        # Restore original limit
        turbovecdb.service._MAX_CACHED_ITEMS = original_max
        turbovecdb.service._databases.clear()


def test_lru_eviction_for_locks():
    """Test that least recently used locks are evicted when cache limit is exceeded."""
    # Clear any existing state
    turbovecdb.service._locks.clear()
    # Temporarily reduce limit for testing
    original_max = turbovecdb.service._MAX_CACHED_ITEMS
    turbovecdb.service._MAX_CACHED_ITEMS = 3
    
    try:
        # Access 4 different paths to exceed limit of 3
        paths = [f"/tmp/test_db_{i}" for i in range(4)]
        locks = []
        
        for path in paths:
            lock = turbovecdb.service._lock_for(path)
            locks.append(lock)
        
        # First path should have been evicted (LRU)
        # Accessing it again should create a new lock
        lock_first_again = turbovecdb.service._lock_for(paths[0])
        
        # The first access should be a different lock (evicted and recreated)
        assert lock_first_again is not locks[0]
        # But should be the same as the newly created one
        assert lock_first_again is turbovecdb.service._lock_for(paths[0])
        
        # Cache should contain the last 3 accessed paths
        assert len(turbovecdb.service._locks) == 3
        
    finally:
        # Restore original limit
        turbovecdb.service._MAX_CACHED_ITEMS = original_max
        turbovecdb.service._locks.clear()


def test_get_cache_stats():
    """Test that get_cache_stats returns correct counts."""
    # Clear any existing state
    turbovecdb.service._databases.clear()
    turbovecdb.service._locks.clear()
    
    # Add some items
    db1 = turbovecdb.service._get_db("/tmp/test_db_1")
    db2 = turbovecdb.service._get_db("/tmp/test_db_2")
    lock1 = turbovecdb.service._lock_for("/tmp/test_db_1")
    lock2 = turbovecdb.service._lock_for("/tmp/test_db_2")
    
    stats = turbovecdb.service.get_cache_stats()
    
    assert stats["cached_databases"] == 2
    assert stats["cached_locks"] == 2
    assert stats["max_cached"] == turbovecdb.service._MAX_CACHED_ITEMS
    
    # Clean up
    turbovecdb.service._databases.clear()
    turbovecdb.service._locks.clear()


def test_thread_safety():
    """Test that the cache is thread-safe."""
    # Clear any existing state
    turbovecdb.service._databases.clear()
    
    path = "/tmp/test_db_thread_safe"
    results = []
    
    def get_db_thread(results_list, thread_id):
        db = turbovecdb.service._get_db(path)
        results_list.append((thread_id, db))
    
    # Create multiple threads that all try to get the same database
    threads = []
    for i in range(10):
        t = threading.Thread(target=get_db_thread, args=(results, i))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # All threads should have gotten the same database instance
    first_db = results[0][1]
    for thread_id, db in results:
        assert db is first_db, f"Thread {thread_id} got different database instance"
    
    # Clean up
    turbovecdb.service._databases.clear()


def test_cache_stats_after_operations():
    """Test that cache stats are updated correctly after various operations."""
    # Clear any existing state
    turbovecdb.service._databases.clear()
    turbovecdb.service._locks.clear()
    
    initial_stats = turbovecdb.service.get_cache_stats()
    assert initial_stats["cached_databases"] == 0
    assert initial_stats["cached_locks"] == 0
    
    # Add some items
    db1 = turbovecdb.service._get_db("/tmp/test_db_1")
    lock1 = turbovecdb.service._lock_for("/tmp/test_db_1")
    
    stats_after_add = turbovecdb.service.get_cache_stats()
    assert stats_after_add["cached_databases"] == 1
    assert stats_after_add["cached_locks"] == 1
    
    # Access the same items again (should not increase count)
    db1_again = turbovecdb.service._get_db("/tmp/test_db_1")
    lock1_again = turbovecdb.service._lock_for("/tmp/test_db_1")
    
    stats_after_same = turbovecdb.service.get_cache_stats()
    assert stats_after_same["cached_databases"] == 1  # Still 1
    assert stats_after_same["cached_locks"] == 1      # Still 1
    assert db1_again is db1  # Same instance
    assert lock1_again is lock1  # Same lock
    
    # Clean up
    turbovecdb.service._databases.clear()
    turbovecdb.service._locks.clear()
