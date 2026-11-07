"""Timing helpers for the Week 7 benchmark lab.

Always report p50 AND p95. The mean hides the tail, and your users live in the tail.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class BenchResult:
    """Every field here is mandatory. A number without its setup is noise."""

    name: str
    hardware: str
    runtime: str
    model: str
    quantisation: str
    context_len: int
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    latencies_ms: list[float] = field(default_factory=list)
    seed: int | None = None
    notes: str = ""

    @property
    def p50_ms(self) -> float | None:
        return statistics.median(self.latencies_ms) if self.latencies_ms else None

    @property
    def p95_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        s = sorted(self.latencies_ms)
        return s[min(int(0.95 * len(s)), len(s) - 1)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["p50_ms"], d["p95_ms"] = self.p50_ms, self.p95_ms
        return d


@contextmanager
def timer(label: str = "") -> Iterator[Callable[[], float]]:
    """with timer() as elapsed: ...  -> elapsed() gives milliseconds."""
    start = time.perf_counter()
    end: float | None = None

    def elapsed() -> float:
        return ((end or time.perf_counter()) - start) * 1000.0

    try:
        yield elapsed
    finally:
        end = time.perf_counter()
        if label:
            print(f"{label}: {elapsed():.1f} ms")
