"""Semantic caching using sentence-transformers."""

import numpy as np
import json
from ai_council.core.logger import get_logger
from .redis_backend import RedisCacheBackend
from typing import Optional, List, Any

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = get_logger(__name__)

class SemanticCacheHelper:
    """Helper to compute embeddings and find semantic matches."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.95):
        self.threshold = threshold
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
        else:
            logger.warning("sentence-transformers not available. Semantic caching disabled.")
            
        # In a real Redisearch implementation, we'd use index. 
        # For simplicity here, we assume a "semantic_index" list of keys and embeddings.
        
    def encode(self, text: str) -> Optional[List[float]]:
        """Encode text to embedding vector."""
        if not self.model:
            return None
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        a = np.array(v1)
        b = np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def find_similar(self, query_embedding: Optional[List[float]], backend: RedisCacheBackend) -> Optional[str]:
        """Find a similar semantic match in the cache."""
        if not query_embedding or not backend.redis:
            return None
            
        # VERY basic implementation for index scanning (in production this would use RediSearch)
        # We store mappings as a list or hash set in Redis
        try:
            # Get all encoded query ids from a dedicated set
            keys = await backend.redis.smembers("semantic_keys")
            
            best_match_id = None
            highest_score = 0.0
            
            for key_bytes in keys:
                key = key_bytes.decode('utf-8')
                embedding_data = await backend.redis.get(f"emb:{key}")
                if embedding_data:
                    cached_emb = json.loads(embedding_data)
                    score = self._cosine_similarity(query_embedding, cached_emb)
                    if score > self.threshold and score > highest_score:
                        highest_score = score
                        best_match_id = key
                        
            if best_match_id:
                logger.info(f"Semantic match found (score: {highest_score:.2f})")
                return best_match_id
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            
        return None

    async def store_embedding(self, original_hash: str, embedding: Optional[List[float]], backend: RedisCacheBackend, ttl: int):
        """Store the embedding associated with the result."""
        if not embedding or not backend.redis:
            return
            
        try:
            await backend.redis.set(f"emb:{original_hash}", json.dumps(embedding), ex=ttl)
            await backend.redis.sadd("semantic_keys", original_hash)
            # In a real system, we'd want TTL on set elements or use Redisearch.
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
