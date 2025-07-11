# Performance Improvements

## 1. Database Performance

### Current Issues

- **No Connection Pooling**: Database connections not pooled efficiently
- **Missing Indexes**: No proper database indexing strategy
- **N+1 Query Problems**: Multiple database queries for single operations
- **No Query Optimization**: No query performance monitoring
- **No Database Caching**: No application-level database caching

### Proposed Solutions

#### Connection Pooling Implementation

```python
# arklex/core/database/pool.py
import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import asynccontextmanager

class DatabasePool:
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        """
        Initialize database connection pool.
        
        Args:
            database_url: Database connection string
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum number of connections that can be created beyond pool_size
        """
        self.engine = create_async_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False  # Set to True for SQL debugging
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session from the pool."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close all database connections."""
        await self.engine.dispose()

# Global database pool instance
db_pool: Optional[DatabasePool] = None

def get_db_pool() -> DatabasePool:
    """Get the global database pool instance."""
    global db_pool
    if db_pool is None:
        from arklex.config.settings import settings
        db_pool = DatabasePool(
            database_url=settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow
        )
    return db_pool
```

#### Database Indexing Strategy

```sql
-- arklex/migrations/add_performance_indexes.sql

-- Message table indexes for conversation queries
CREATE INDEX CONCURRENTLY idx_messages_user_id_created_at 
ON messages(user_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_messages_agent_id 
ON messages(agent_id);

CREATE INDEX CONCURRENTLY idx_messages_conversation_id 
ON messages(conversation_id);

-- Agent table indexes
CREATE INDEX CONCURRENTLY idx_agents_status 
ON agents(status);

CREATE INDEX CONCURRENTLY idx_agents_type 
ON agents(agent_type);

-- Tool usage indexes
CREATE INDEX CONCURRENTLY idx_tool_usage_agent_id 
ON tool_usage(agent_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_tool_usage_tool_name 
ON tool_usage(tool_name);

-- Vector search indexes (if using PostgreSQL with pgvector)
CREATE INDEX CONCURRENTLY idx_documents_embedding 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY idx_messages_user_agent_time 
ON messages(user_id, agent_id, created_at DESC);

-- Partial indexes for active records
CREATE INDEX CONCURRENTLY idx_agents_active 
ON agents(id) WHERE status = 'active';

-- Text search indexes
CREATE INDEX CONCURRENTLY idx_documents_content_gin 
ON documents USING gin(to_tsvector('english', content));
```

#### Query Optimization

```python
# arklex/core/database/optimized_queries.py
from typing import List, Optional, Dict, Any
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload, joinedload
from arklex.models import Message, Agent, Conversation, ToolUsage

class OptimizedQueryManager:
    def __init__(self, db_pool: DatabasePool):
        self.db_pool = db_pool
    
    async def get_conversation_history(
        self, 
        user_id: str, 
        limit: int = 50,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history with optimized query.
        
        Uses eager loading to avoid N+1 queries and includes
        only necessary fields for better performance.
        """
        async with self.db_pool.get_session() as session:
            # Build optimized query
            query = (
                select(Message)
                .where(Message.user_id == user_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            
            if include_metadata:
                # Eager load related data to avoid N+1 queries
                query = query.options(
                    selectinload(Message.agent),
                    selectinload(Message.conversation),
                    selectinload(Message.tool_usages)
                )
            
            result = await session.execute(query)
            messages = result.scalars().all()
            
            return [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "response": msg.response,
                    "created_at": msg.created_at.isoformat(),
                    "agent_name": msg.agent.name if msg.agent else None,
                    "tool_usages": [
                        {
                            "tool_name": tu.tool_name,
                            "input": tu.input_data,
                            "output": tu.output_data,
                            "duration": tu.duration
                        }
                        for tu in msg.tool_usages
                    ] if include_metadata else []
                }
                for msg in messages
            ]
    
    async def get_agent_performance_stats(
        self, 
        agent_id: str, 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get agent performance statistics with optimized aggregation.
        """
        async with self.db_pool.get_session() as session:
            # Use database-level aggregation for better performance
            query = (
                select(
                    func.count(Message.id).label("total_messages"),
                    func.avg(Message.response_time).label("avg_response_time"),
                    func.count(ToolUsage.id).label("total_tool_uses"),
                    func.avg(ToolUsage.duration).label("avg_tool_duration")
                )
                .select_from(Message)
                .outerjoin(ToolUsage, Message.id == ToolUsage.message_id)
                .where(
                    and_(
                        Message.agent_id == agent_id,
                        Message.created_at >= func.date_sub(func.now(), days)
                    )
                )
            )
            
            result = await session.execute(query)
            stats = result.first()
            
            return {
                "total_messages": stats.total_messages or 0,
                "avg_response_time": float(stats.avg_response_time or 0),
                "total_tool_uses": stats.total_tool_uses or 0,
                "avg_tool_duration": float(stats.avg_tool_duration or 0)
            }
    
    async def search_messages(
        self, 
        user_id: str, 
        search_term: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search messages with full-text search optimization.
        """
        async with self.db_pool.get_session() as session:
            # Use PostgreSQL full-text search for better performance
            query = (
                select(Message)
                .where(
                    and_(
                        Message.user_id == user_id,
                        Message.content.ilike(f"%{search_term}%")
                    )
                )
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            
            result = await session.execute(query)
            messages = result.scalars().all()
            
            return [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "relevance_score": self._calculate_relevance(msg.content, search_term)
                }
                for msg in messages
            ]
    
    def _calculate_relevance(self, content: str, search_term: str) -> float:
        """Calculate relevance score for search results."""
        # Simple relevance calculation - could be enhanced with TF-IDF
        content_lower = content.lower()
        term_lower = search_term.lower()
        
        if term_lower in content_lower:
            return 1.0
        elif any(word in content_lower for word in term_lower.split()):
            return 0.5
        else:
            return 0.1
```

## 2. Caching Strategy

### Current Issues

- **No Application Caching**: No Redis or in-memory caching
- **No Query Result Caching**: Database query results not cached
- **No API Response Caching**: API responses not cached
- **No Session Caching**: User sessions not cached
- **No Tool Result Caching**: Tool execution results not cached

### Proposed Solutions

#### Redis Caching Implementation

```python
# arklex/core/cache/redis_cache.py
import json
import hashlib
from typing import Any, Optional, Union, Dict
import aioredis
from datetime import timedelta
import pickle

class RedisCache:
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        """
        Initialize Redis cache manager.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
        """
        self.redis = aioredis.from_url(redis_url)
        self.default_ttl = default_ttl
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [prefix] + [str(arg) for arg in args]
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend([f"{k}:{v}" for k, v in sorted_kwargs])
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            serialized_value = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            return await self.redis.set(key, serialized_value, ex=ttl)
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(await self.redis.delete(key))
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache clear pattern error: {e}")
            return 0

# Global cache instance
cache: Optional[RedisCache] = None

def get_cache() -> RedisCache:
    """Get the global cache instance."""
    global cache
    if cache is None:
        from arklex.config.settings import settings
        cache = RedisCache(
            redis_url=settings.redis_url,
            default_ttl=settings.cache_ttl
        )
    return cache
```

#### Caching Decorators

```python
# arklex/core/cache/decorators.py
import functools
import asyncio
from typing import Callable, Any, Optional
from arklex.core.cache.redis_cache import get_cache

def cache_result(
    prefix: str,
    ttl: Optional[int] = None,
    key_generator: Optional[Callable] = None
):
    """
    Decorator to cache function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        key_generator: Custom key generation function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_instance = get_cache()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(prefix, *args, **kwargs)
            else:
                cache_key = cache_instance._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_instance = get_cache()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(prefix, *args, **kwargs)
            else:
                cache_key = cache_instance._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = asyncio.run(cache_instance.get(cache_key))
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            asyncio.run(cache_instance.set(cache_key, result, ttl))
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def invalidate_cache(pattern: str):
    """
    Decorator to invalidate cache after function execution.
    
    Args:
        pattern: Cache key pattern to invalidate
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            cache_instance = get_cache()
            await cache_instance.clear_pattern(pattern)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            cache_instance = get_cache()
            asyncio.run(cache_instance.clear_pattern(pattern))
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

#### Application-Level Caching

```python
# arklex/core/cache/application_cache.py
from typing import Dict, Any, Optional
from arklex.core.cache.redis_cache import get_cache
from arklex.core.cache.decorators import cache_result, invalidate_cache

class ApplicationCache:
    def __init__(self):
        self.cache = get_cache()
    
    @cache_result("agent_config", ttl=3600)
    async def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Cache agent configuration."""
        # This would normally fetch from database
        # For now, return mock data
        return {
            "id": agent_id,
            "name": f"Agent {agent_id}",
            "model": "gpt-4",
            "temperature": 0.7
        }
    
    @cache_result("user_session", ttl=1800)
    async def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """Cache user session data."""
        return {
            "user_id": user_id,
            "preferences": {},
            "conversation_history": []
        }
    
    @cache_result("rag_results", ttl=1800)
    async def get_rag_results(self, query: str, limit: int = 10) -> list:
        """Cache RAG search results."""
        # This would normally perform vector search
        return [
            {"id": f"doc_{i}", "content": f"Document {i}", "score": 0.9 - i * 0.1}
            for i in range(limit)
        ]
    
    @invalidate_cache("agent_config:*")
    async def update_agent_config(self, agent_id: str, config: Dict[str, Any]):
        """Update agent config and invalidate cache."""
        # Update database
        # Cache will be invalidated automatically
        pass
    
    @invalidate_cache("user_session:*")
    async def update_user_session(self, user_id: str, session_data: Dict[str, Any]):
        """Update user session and invalidate cache."""
        # Update database
        # Cache will be invalidated automatically
        pass
```

## 3. Async Performance Optimization

### Current Issues

- **Blocking Operations**: Synchronous operations blocking event loop
- **No Connection Pooling**: No async connection pooling
- **No Concurrent Processing**: No parallel task execution
- **No Resource Management**: No proper async resource management
- **No Performance Monitoring**: No async performance metrics

### Proposed Solutions

#### Async Connection Pooling

```python
# arklex/core/async_pool.py
import asyncio
from typing import TypeVar, Callable, Awaitable, Optional
from contextlib import asynccontextmanager
import aiohttp
import httpx

T = TypeVar('T')

class AsyncConnectionPool:
    def __init__(self, max_connections: int = 100):
        """
        Initialize async connection pool.
        
        Args:
            max_connections: Maximum number of concurrent connections
        """
        self.semaphore = asyncio.Semaphore(max_connections)
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.httpx_client: Optional[httpx.AsyncClient] = None
    
    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.http_session is None or self.http_session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.semaphore._value,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            self.http_session = aiohttp.ClientSession(connector=connector)
        return self.http_session
    
    async def get_httpx_client(self) -> httpx.AsyncClient:
        """Get or create HTTPX client."""
        if self.httpx_client is None:
            limits = httpx.Limits(
                max_connections=self.semaphore._value,
                max_keepalive_connections=20
            )
            self.httpx_client = httpx.AsyncClient(limits=limits)
        return self.httpx_client
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        async with self.semaphore:
            yield
    
    async def execute_with_pool(
        self, 
        func: Callable[..., Awaitable[T]], 
        *args, 
        **kwargs
    ) -> T:
        """Execute function with connection pool management."""
        async with self.acquire():
            return await func(*args, **kwargs)
    
    async def close(self):
        """Close all connections."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        if self.httpx_client:
            await self.httpx_client.aclose()

# Global connection pool
connection_pool = AsyncConnectionPool()
```

#### Concurrent Task Execution

```python
# arklex/core/concurrent_executor.py
import asyncio
from typing import List, TypeVar, Callable, Awaitable, Any
from concurrent.futures import ThreadPoolExecutor
import functools

T = TypeVar('T')

class ConcurrentExecutor:
    def __init__(self, max_workers: int = 10):
        """
        Initialize concurrent executor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loop = asyncio.get_event_loop()
    
    async def run_in_thread(
        self, 
        func: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """Run synchronous function in thread pool."""
        return await self.loop.run_in_executor(
            self.executor,
            functools.partial(func, *args, **kwargs)
        )
    
    async def gather_with_semaphore(
        self, 
        tasks: List[Awaitable[T]], 
        semaphore_limit: int = 10
    ) -> List[T]:
        """Execute tasks with concurrency limit."""
        semaphore = asyncio.Semaphore(semaphore_limit)
        
        async def limited_task(task: Awaitable[T]) -> T:
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        return await asyncio.gather(*limited_tasks)
    
    async def execute_batch(
        self, 
        func: Callable[..., Awaitable[T]], 
        items: List[Any], 
        batch_size: int = 10
    ) -> List[T]:
        """Execute function on items in batches."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [func(item) for item in batch]
            batch_results = await self.gather_with_semaphore(batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)

# Global executor
executor = ConcurrentExecutor()
```

## 4. Memory Optimization

### Current Issues

- **Memory Leaks**: No memory leak detection
- **Large Object Retention**: Large objects not properly garbage collected
- **No Memory Monitoring**: No memory usage tracking
- **Inefficient Data Structures**: No optimized data structures
- **No Resource Cleanup**: No proper resource cleanup

### Proposed Solutions

#### Memory Monitoring

```python
# arklex/core/memory_monitor.py
import psutil
import gc
import asyncio
from typing import Dict, Any
import tracemalloc

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.baseline_memory = self.process.memory_info().rss
        tracemalloc.start()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Get top memory allocations
        current, peak = tracemalloc.get_traced_memory()
        top_stats = tracemalloc.get_statistics()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": memory_percent,
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024,
            "top_allocations": [
                {
                    "file": stat.traceback.format()[-1],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                }
                for stat in top_stats[:5]
            ]
        }
    
    def check_memory_leak(self) -> bool:
        """Check for potential memory leak."""
        if self.baseline_memory is None:
            return False
        
        current_memory = self.process.memory_info().rss
        memory_increase = current_memory - self.baseline_memory
        
        return memory_increase > self.memory_threshold
    
    def force_garbage_collection(self):
        """Force garbage collection."""
        collected = gc.collect()
        return collected
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        memory_usage = self.get_memory_usage()
        memory_usage["has_leak"] = self.check_memory_leak()
        memory_usage["gc_stats"] = gc.get_stats()
        
        return memory_usage

# Global memory monitor
memory_monitor = MemoryMonitor()
```

#### Optimized Data Structures

```python
# arklex/core/optimized_structures.py
from typing import Dict, List, Any, Optional
from collections import OrderedDict
import weakref
import heapq

class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[key] = value

class MemoryEfficientList:
    """Memory-efficient list implementation."""
    
    def __init__(self, initial_capacity: int = 1000):
        self._data = []
        self._capacity = initial_capacity
        self._size = 0
    
    def append(self, item: Any):
        """Append item to list."""
        if self._size >= self._capacity:
            # Double capacity
            self._capacity *= 2
            # Trim unused space
            self._data = self._data[:self._size]
        
        self._data.append(item)
        self._size += 1
    
    def __getitem__(self, index: int) -> Any:
        return self._data[index]
    
    def __len__(self) -> int:
        return self._size
    
    def clear(self):
        """Clear list and free memory."""
        self._data.clear()
        self._size = 0

class WeakReferenceCache:
    """Cache using weak references to allow garbage collection."""
    
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        self._cache[key] = value
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
```

## 5. Performance Monitoring

### Current Issues

- **No Performance Metrics**: No application performance metrics
- **No Response Time Tracking**: No response time monitoring
- **No Throughput Monitoring**: No request throughput tracking
- **No Resource Monitoring**: No CPU/memory monitoring
- **No Alerting**: No performance alerting

### Proposed Solutions

#### Performance Metrics Collection

```python
# arklex/core/metrics/collector.py
import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import threading

@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]

class MetricsCollector:
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.lock = threading.Lock()
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        with self.lock:
            self.counters[name] += value
        
        self.metrics.append(PerformanceMetric(
            name=name,
            value=float(value),
            timestamp=time.time(),
            tags=tags or {}
        ))
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        with self.lock:
            self.timers[name].append(duration)
        
        self.metrics.append(PerformanceMetric(
            name=f"{name}_duration",
            value=duration,
            timestamp=time.time(),
            tags=tags or {}
        ))
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self.metrics.append(PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        ))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self.lock:
            summary = {
                "counters": dict(self.counters),
                "timers": {
                    name: {
                        "count": len(values),
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "avg": sum(values) / len(values) if values else 0,
                        "p95": self._percentile(values, 95) if values else 0,
                        "p99": self._percentile(values, 99) if values else 0
                    }
                    for name, values in self.timers.items()
                }
            }
        
        return summary
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def clear_metrics(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.timers.clear()

# Global metrics collector
metrics_collector = MetricsCollector()
```

#### Performance Decorators

```python
# arklex/core/metrics/decorators.py
import time
import functools
import asyncio
from typing import Callable, Any
from arklex.core.metrics.collector import metrics_collector

def track_performance(operation_name: str):
    """
    Decorator to track function performance.
    
    Args:
        operation_name: Name of the operation for metrics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metrics
                metrics_collector.record_timer(f"{operation_name}_success", duration)
                metrics_collector.record_counter(f"{operation_name}_success_count")
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metrics
                metrics_collector.record_timer(f"{operation_name}_error", duration)
                metrics_collector.record_counter(f"{operation_name}_error_count")
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metrics
                metrics_collector.record_timer(f"{operation_name}_success", duration)
                metrics_collector.record_counter(f"{operation_name}_success_count")
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metrics
                metrics_collector.record_timer(f"{operation_name}_error", duration)
                metrics_collector.record_counter(f"{operation_name}_error_count")
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def monitor_memory(func_name: str):
    """
    Decorator to monitor memory usage.
    
    Args:
        func_name: Name of the function for memory tracking
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            from arklex.core.memory_monitor import memory_monitor
            
            # Record memory before
            memory_before = memory_monitor.get_memory_usage()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record memory after
                memory_after = memory_monitor.get_memory_usage()
                memory_diff = memory_after["rss_mb"] - memory_before["rss_mb"]
                
                metrics_collector.record_gauge(f"{func_name}_memory_usage", memory_after["rss_mb"])
                metrics_collector.record_gauge(f"{func_name}_memory_increase", memory_diff)
                
                return result
            
            except Exception as e:
                # Record memory on error
                memory_after = memory_monitor.get_memory_usage()
                metrics_collector.record_gauge(f"{func_name}_memory_error", memory_after["rss_mb"])
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            from arklex.core.memory_monitor import memory_monitor
            
            # Record memory before
            memory_before = memory_monitor.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # Record memory after
                memory_after = memory_monitor.get_memory_usage()
                memory_diff = memory_after["rss_mb"] - memory_before["rss_mb"]
                
                metrics_collector.record_gauge(f"{func_name}_memory_usage", memory_after["rss_mb"])
                metrics_collector.record_gauge(f"{func_name}_memory_increase", memory_diff)
                
                return result
            
            except Exception as e:
                # Record memory on error
                memory_after = memory_monitor.get_memory_usage()
                metrics_collector.record_gauge(f"{func_name}_memory_error", memory_after["rss_mb"])
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

## Success Metrics

### Performance Metrics

- [ ] <100ms response time for 95th percentile
- [ ] <500ms database query response time
- [ ] <1GB memory usage under normal load
- [ ] <50% CPU usage under normal load
- [ ] 1000+ concurrent requests handled

### Caching Metrics

- [ ] 80%+ cache hit rate for frequently accessed data
- [ ] <10ms cache access time
- [ ] Zero cache-related errors
- [ ] Automatic cache invalidation working
- [ ] Memory-efficient cache implementation

### Database Metrics

- [ ] Connection pool utilization <80%
- [ ] Query response time <100ms for 95th percentile
- [ ] Zero connection leaks
- [ ] Proper indexing strategy implemented
- [ ] Query optimization completed

### Monitoring Metrics

- [ ] Real-time performance monitoring
- [ ] Automated alerting for performance issues
- [ ] Comprehensive metrics collection
- [ ] Performance dashboards implemented
- [ ] Memory leak detection active

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Implement database connection pooling
- [ ] Add Redis caching infrastructure
- [ ] Set up basic performance monitoring
- [ ] Implement memory monitoring

### Phase 2: Optimization (Week 3-4)
- [ ] Optimize database queries
- [ ] Implement caching strategies
- [ ] Add async performance optimizations
- [ ] Optimize data structures

### Phase 3: Advanced Features (Week 5-6)
- [ ] Implement advanced caching
- [ ] Add performance decorators
- [ ] Set up comprehensive monitoring
- [ ] Implement alerting system

### Phase 4: Production Ready (Week 7-8)
- [ ] Load testing and optimization
- [ ] Performance tuning
- [ ] Production monitoring setup
- [ ] Performance documentation 