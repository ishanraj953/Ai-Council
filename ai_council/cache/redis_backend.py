"""Redis Backend for exact and semantic cache storage."""

import json
import hashlib
from typing import Optional, Dict, Any
from redis.asyncio import Redis, from_url
from ai_council.core.logger import get_logger

logger = get_logger(__name__)

class RedisCacheBackend:
    """Handles communication with Redis cache."""
    
    def __init__(self, url: str):
        self.url = url
        self.redis: Optional[Redis] = None
        
    async def connect(self):
        """Connect to Redis asynchronously."""
        if not self.redis:
            try:
                self.redis = await from_url(self.url)
                logger.info("Connected to Redis cache", extra={"url": self.url})
            except Exception as e:
                logger.error("Failed to connect to Redis cache", extra={"error": str(e)})

    def hash_task(self, text: str) -> str:
        """Create a deterministic hash for an exact cache match."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a key from Redis."""
        if not self.redis:
            await self.connect()
        try:
            if self.redis:
                data = await self.redis.get(key)
                if data:
                    return json.loads(data)
        except Exception as e:
            logger.error("Redis get error", extra={"error": str(e), "key": key})
        return None

    async def set(self, key: str, value: Dict[str, Any], ttl: int = 86400):
        """Set a key in Redis with TTL."""
        if not self.redis:
            await self.connect()
        try:
            if self.redis:
                # Store stringified JSON
                await self.redis.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.error("Redis set error", extra={"error": str(e), "key": key})
