# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging

logger = logging.getLogger(__name__)


class RequestQueue:
    """Queue for managing concurrent requests"""

    def __init__(self, max_concurrent: int, queue_size: int):
        """
        Initialize request queue
        
        Args:
            max_concurrent: Maximum number of concurrent requests
            queue_size: Maximum number of requests in queue
        """
        self.max_concurrent = max_concurrent
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()

    async def enqueue(self, task):
        """
        Add task to queue
        
        Args:
            task: Async task to enqueue
            
        Returns:
            True if task was enqueued, False if queue is full
        """
        try:
            self.queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            logger.warning("Request queue is full")
            return False

    async def process(self):
        """Process tasks from queue with concurrency control"""
        while True:
            # Get task from queue
            task = await self.queue.get()
            
            # Wait for available semaphore slot
            await self.semaphore.acquire()
            
            # Track active task
            self.active_tasks.add(task)
            
            # Release semaphore when task completes
            task.add_done_callback(
                lambda t: (
                    self.semaphore.release(),
                    self.active_tasks.discard(t),
                    self.queue.task_done()
                )
            )
