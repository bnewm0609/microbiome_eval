"""
Caching module for LLM API calls.
Provides persistent, thread-safe caching using SQLite.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class LLMCache:
    """
    Thread-safe cache for LLM API responses using SQLite.

    Cache keys are generated from hash of (model, prompt, generation_params).
    """

    def __init__(self, cache_dir: str = ".llm_cache", enabled: bool = True):
        """
        Initialize the LLM cache.

        Args:
            cache_dir: Directory to store cache database
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

        if self.enabled:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.cache_dir / "llm_cache.db"

            # Initialize database
            self._init_db()

    def _init_db(self):
        """Initialize SQLite database with cache table."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                params TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create index for faster lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_key ON cache(key)
        """
        )

        conn.commit()
        conn.close()

    def _generate_key(
        self, model: str, prompt: str, generation_params: Dict[str, Any]
    ) -> str:
        """
        Generate cache key from model, prompt, and parameters.

        Args:
            model: Model name
            prompt: Input prompt
            generation_params: Generation parameters (temperature, top_p, max_tokens)

        Returns:
            SHA256 hash as cache key
        """
        # Create deterministic string representation
        key_data = {
            "model": model,
            "prompt": prompt,
            "params": {
                **generation_params
                # "temperature": generation_params.get("temperature"),
                # "top_p": generation_params.get("top_p"),
                # "max_tokens": generation_params.get("max_tokens"),
            },
        }

        # Convert to JSON with sorted keys for consistency
        key_string = json.dumps(key_data, sort_keys=True)

        # Generate hash
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self, model: str, prompt: str, generation_params: Dict[str, Any]
    ) -> Optional[str]:
        """
        Retrieve cached response if available.

        Args:
            model: Model name
            prompt: Input prompt
            generation_params: Generation parameters

        Returns:
            Cached response or None if not found
        """
        if not self.enabled:
            return None

        key = self._generate_key(model, prompt, generation_params)

        with self.lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()

            cursor.execute("SELECT response FROM cache WHERE key = ?", (key,))
            result = cursor.fetchone()

            conn.close()

            if result:
                self.hits += 1
                return result[0]
            else:
                self.misses += 1
                return None

    def set(
        self,
        model: str,
        prompt: str,
        generation_params: Dict[str, Any],
        response: str,
    ):
        """
        Store response in cache.

        Args:
            model: Model name
            prompt: Input prompt
            generation_params: Generation parameters
            response: Model response to cache
        """
        if not self.enabled:
            return

        key = self._generate_key(model, prompt, generation_params)

        with self.lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()

            # Use INSERT OR REPLACE to handle duplicates
            cursor.execute(
                """
                INSERT OR REPLACE INTO cache (key, model, prompt, params, response)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    key,
                    model,
                    prompt,
                    json.dumps(generation_params, sort_keys=True),
                    response,
                ),
            )

            conn.commit()
            conn.close()

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, and hit rate
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0

        stats = {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2f}%",
        }

        if self.enabled:
            stats["cache_dir"] = str(self.cache_dir)
            stats["cache_size"] = self._get_cache_size()

        return stats

    def _get_cache_size(self) -> int:
        """Get number of entries in cache."""
        if not self.enabled:
            return 0

        with self.lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]

            conn.close()

            return count

    def clear(self):
        """Clear all cache entries."""
        if not self.enabled:
            return

        with self.lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM cache")

            conn.commit()
            conn.close()

        print(f"Cache cleared: {self.db_path}")