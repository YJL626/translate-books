from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def local_now() -> datetime:
    return datetime.now().astimezone()


class Progress:
    """Estimate remaining translation time from recent uncached work, weighted by text size."""

    def __init__(
        self,
        total_items: int,
        total_work: int,
        *,
        workers: int = 1,
        started_at: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = local_now,
    ):
        self.total_items = total_items
        self.total_work = total_work
        self.workers = workers
        self.clock = clock
        self.wall_clock = wall_clock
        self.started_at = clock() if started_at is None else started_at
        self.completed_items = 0
        self.completed_work = 0
        self.samples: deque[tuple[int, float]] = deque(maxlen=50)

    @staticmethod
    def weight(text: str) -> int:
        # A small fixed cost accounts for request overhead and very short headings/captions.
        return len(text) + 100

    def complete(self, work: int, seconds: float, *, cached: bool) -> None:
        self.completed_items += 1
        self.completed_work += work
        if not cached and seconds > 0:
            self.samples.append((work, seconds))

    @property
    def remaining_seconds(self) -> float | None:
        remaining = max(0, self.total_work - self.completed_work)
        if self.completed_items >= self.total_items or remaining == 0:
            return 0
        if len(self.samples) < 3:
            return None
        work = sum(work for work, _ in self.samples)
        seconds = sum(seconds for _, seconds in self.samples)
        active_workers = min(self.workers, self.total_items - self.completed_items)
        return remaining * seconds / work / active_workers

    def render(self, label: str) -> str:
        now = self.wall_clock()
        elapsed = duration(self.clock() - self.started_at)
        remaining = self.remaining_seconds
        if remaining is None:
            estimate = "预计剩余 估算中"
        else:
            estimate = f"预计剩余 {duration(math.ceil(remaining))}"
            if remaining:
                finish = now + timedelta(seconds=math.ceil(remaining))
                estimate += f" · 预计完成 {finish:%m-%d %H:%M:%S}"
        percentage = 100 * self.completed_items / self.total_items if self.total_items else 100
        return (
            f"[当前 {now:%Y-%m-%d %H:%M:%S %z}] {label} | "
            f"进度 {self.completed_items}/{self.total_items} ({percentage:.1f}%) | "
            f"并发 {self.workers} | "
            f"本次已用 {elapsed} | {estimate}"
        )
