# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time


class RateLimiter:
    """Rate limiter using token bucket algorithm"""

    def __init__(self, rate_limit: int):
        """
        Initialize rate limiter
        
        Args:
            rate_limit: Maximum requests per second
        """
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        """Acquire rate limit token"""
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                
                # Refill tokens based on elapsed time
                self.tokens = min(
                    self.rate_limit, 
                    self.tokens + elapsed * self.rate_limit
                )
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return self
                
                # Wait until next token is available
                wait_time = (1 - self.tokens) / self.rate_limit
                await asyncio.sleep(wait_time)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        return False
