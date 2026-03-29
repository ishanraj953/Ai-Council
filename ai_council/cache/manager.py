"""Main CacheManager for Orchestration Layer."""

import time
from typing import Optional, Dict, Any, List
from ai_council.core.logger import get_logger
from ai_council.core.models import Task, FinalResponse, Subtask, AgentResponse
from .redis_backend import RedisCacheBackend
from .semantic import SemanticCacheHelper
from .batching import BatchOptimizer

logger = get_logger(__name__)


class CacheManager:
    """Manages exact and semantic caching for requests and subtasks."""
    
    def __init__(self, backend_url: str = "redis://localhost:6379/0"):
        self.backend = RedisCacheBackend(url=backend_url)
        self.semantic = SemanticCacheHelper()
        self.batch_optimizer = BatchOptimizer()
        self.metrics = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "latency_savings_ms": 0.0,
            "cost_savings": 0.0
        }
        logger.info("CacheManager initialized")

    async def check_cache(self, task: Task) -> Optional[FinalResponse]:
        """Check cache for the given task."""
        # Check exact cache
        task_hash = self.backend.hash_task(task.content)
        start_time = time.time()
        
        cached_data = await self.backend.get(task_hash)
        if cached_data:
            self._record_hit("exact", 0.0) # We'll estimate cost savings if encoded in data
            return self._deserialize_response(cached_data)
        
        # Check semantic cache
        embedding = self.semantic.encode(task.content)
        semantic_match_id = await self.semantic.find_similar(embedding, self.backend)
        if semantic_match_id:
            cached_data = await self.backend.get(semantic_match_id)
            if cached_data:
                self._record_hit("semantic", 0.0)
                return self._deserialize_response(cached_data)
        
        self.metrics["misses"] += 1
        return None

    async def store_response(self, task: Task, response: FinalResponse):
        """Store response in cache."""
        task_hash = self.backend.hash_task(task.content)
        embedding = self.semantic.encode(task.content)
        
        # Determine TTL based on confidence
        ttl = self._calculate_ttl(response.overall_confidence)
        
        serialized = self._serialize_response(response)
        await self.backend.set(task_hash, serialized, ttl=ttl)
        
        # Store embedding for semantic cache
        await self.semantic.store_embedding(task_hash, embedding, self.backend, ttl=ttl)
        
    def _calculate_ttl(self, confidence: float) -> int:
        """Calculate TTL in seconds based on response confidence."""
        base_ttl = 86400  # 24 hours
        if confidence < 0.5:
            return 3600  # 1 hour for low confidence
        elif confidence < 0.8:
            return 43200  # 12 hours
        return base_ttl

    def _record_hit(self, hit_type: str, cost_savings: float):
        self.metrics[f"{hit_type}_hits"] += 1
        self.metrics["cost_savings"] += cost_savings
        
    def _serialize_response(self, response: FinalResponse) -> Dict[str, Any]:
        return {
            "content": response.content,
            "overall_confidence": response.overall_confidence,
            "models_used": response.models_used,
            "success": response.success,
            "error_message": response.error_message
        }
        
    def _deserialize_response(self, data: Dict[str, Any]) -> FinalResponse:
        from ai_council.core.models import FinalResponse, ExecutionMetadata, CostBreakdown
        return FinalResponse(
            content=data.get("content", ""),
            overall_confidence=data.get("overall_confidence", 0.5),
            execution_metadata=ExecutionMetadata(),
            success=data.get("success", True),
            error_message=data.get("error_message", None),
            cost_breakdown=CostBreakdown(execution_time=0.01), # very fast
            models_used=data.get("models_used", ["cache"])
        )
