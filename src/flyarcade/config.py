"""Conservative, explicit run limits."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Limits:
    max_neurons: int = 4096
    max_edges: int = 250_000
    max_rss_mb: int = 2048
    min_free_disk_mb: int = 1024
    max_seconds: float = 120
    max_download_mb: int = 128

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if name != "max_seconds" and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")

    def check_graph(self, neurons: int, edges: int):
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (neurons, edges)):
            raise ValueError("graph sizes must be integers")
        if not 0 < neurons <= self.max_neurons or not 0 <= edges <= self.max_edges:
            raise ValueError("graph exceeds configured resource limits")
