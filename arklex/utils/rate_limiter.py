import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, tokens_per_minute: int = 9000, safety_margin: float = 0.9):
        """
        Initialize rate limiter
        :param tokens_per_minute: Maximum tokens per minute (default 9000 to stay under 10000 limit)
        :param safety_margin: Safety margin to stay under the limit (default 0.9 = 90%)
        """
        self.tokens_per_minute = int(tokens_per_minute * safety_margin)
        self.tokens_used = 0
        self.last_reset = time.time()
        self.chunk_size = 8000  # Maximum chunk size to stay under limit

    def estimate_tokens(self, text: str) -> int:
        """
        Roughly estimate the number of tokens in a text
        :param text: The text to estimate tokens for
        :return: Estimated number of tokens
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Wait if we're approaching the rate limit
        :param estimated_tokens: Number of tokens that will be used
        """
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.last_reset >= 60:
            self.tokens_used = 0
            self.last_reset = current_time
        
        # If the request is too large, break it into chunks
        if estimated_tokens > self.chunk_size:
            chunks = (estimated_tokens + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Large request detected ({estimated_tokens} tokens). Breaking into {chunks} chunks.")
            
            for i in range(chunks):
                chunk_tokens = min(self.chunk_size, estimated_tokens - (i * self.chunk_size))
                
                # If we're approaching the limit, wait
                if self.tokens_used + chunk_tokens > self.tokens_per_minute:
                    wait_time = 60 - (current_time - self.last_reset)
                    if wait_time > 0:
                        logger.info(f"Rate limit approaching. Waiting {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        self.tokens_used = 0
                        self.last_reset = time.time()
                
                self.tokens_used += chunk_tokens
                if i < chunks - 1:  # Don't wait after the last chunk
                    time.sleep(1)  # Small delay between chunks
        else:
            # For smaller requests, use the original logic
            if self.tokens_used + estimated_tokens > self.tokens_per_minute:
                wait_time = 60 - (current_time - self.last_reset)
                if wait_time > 0:
                    logger.info(f"Rate limit approaching. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    self.tokens_used = 0
                    self.last_reset = time.time()
            
            self.tokens_used += estimated_tokens 