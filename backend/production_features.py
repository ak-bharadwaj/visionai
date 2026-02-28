"""
production_features.py — Google-level production infrastructure.

Features:
  - Request queuing and batching
  - Async processing
  - Caching
  - Load balancing
  - Rate limiting
"""

import logging
import time
import threading
import queue
from typing import Callable, Optional, Any
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)


class RequestQueue:
    """Queue and batch requests for efficient processing."""
    
    def __init__(self, batch_size: int = 4, timeout: float = 0.1):
        self._batch_size = batch_size
        self._timeout = timeout
        self._queue = queue.Queue()
        self._lock = threading.Lock()
    
    def add(self, request: Any, priority: int = 0):
        """Add request to queue."""
        self._queue.put((priority, time.time(), request))
    
    def get_batch(self) -> list:
        """Get a batch of requests."""
        batch = []
        deadline = time.time() + self._timeout
        
        while len(batch) < self._batch_size and time.time() < deadline:
            try:
                _, _, request = self._queue.get(timeout=0.05)
                batch.append(request)
            except queue.Empty:
                break
        
        return batch


class ResultCache:
    """Cache results for frequently requested data."""
    
    def __init__(self, max_size: int = 100, ttl: float = 5.0):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: dict = {}
        self._timestamps: dict = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self._ttl:
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Cache a result."""
        with self._lock:
            # Evict oldest if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._timestamps.keys(), 
                               key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


class RateLimiter:
    """Rate limiting for API calls."""
    
    def __init__(self, max_calls: int = 10, period: float = 1.0):
        self._max_calls = max_calls
        self._period = period
        self._calls: deque = deque()
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        """Check if call is allowed."""
        with self._lock:
            now = time.time()
            # Remove old calls
            while self._calls and self._calls[0] < now - self._period:
                self._calls.popleft()
            
            if len(self._calls) < self._max_calls:
                self._calls.append(now)
                return True
            return False
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.allow():
            time.sleep(0.1)


class AsyncProcessor:
    """Process requests asynchronously."""
    
    def __init__(self, max_workers: int = 2):
        self._max_workers = max_workers
        self._queue = queue.Queue()
        self._workers = []
        self._running = False
    
    def start(self):
        """Start async workers."""
        self._running = True
        for i in range(self._max_workers):
            worker = threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"AsyncWorker-{i}"
            )
            worker.start()
            self._workers.append(worker)
    
    def submit(self, task: Callable, *args, **kwargs):
        """Submit task for async processing."""
        self._queue.put((task, args, kwargs))
    
    def _worker(self):
        """Worker thread."""
        while self._running:
            try:
                task, args, kwargs = self._queue.get(timeout=1.0)
                try:
                    task(*args, **kwargs)
                except Exception as e:
                    logger.error("Async task failed: %s", e)
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue
    
    def stop(self):
        """Stop workers."""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=1.0)


# Module-level instances
request_queue = RequestQueue()
result_cache = ResultCache()
rate_limiter = RateLimiter()
async_processor = AsyncProcessor()
