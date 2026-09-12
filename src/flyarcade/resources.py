"""Cooperative guards: checked before allocations and at simulation boundaries."""

import shutil
import time
from pathlib import Path

import psutil

from flyarcade.config import Limits


class ResourceLimit(RuntimeError):
    """A run stopped safely before exceeding its resource budget."""


class ResourceGuard:
    def __init__(self, limits: Limits = Limits(), directory: Path = Path(".")):
        self.limits = limits
        self.directory = directory
        self.started = time.monotonic()

    def snapshot(self):
        return {
            "rss_mb": psutil.Process().memory_info().rss / 2**20,
            "free_disk_mb": shutil.disk_usage(self.directory).free / 2**20,
            "elapsed_seconds": time.monotonic() - self.started,
        }

    def check(self):
        stats = self.snapshot()
        if stats["rss_mb"] > self.limits.max_rss_mb:
            raise ResourceLimit("resident-memory guard exceeded")
        if stats["free_disk_mb"] < self.limits.min_free_disk_mb:
            raise ResourceLimit("free-disk guard exceeded")
        if stats["elapsed_seconds"] > self.limits.max_seconds:
            raise ResourceLimit("runtime guard exceeded")
        return stats
