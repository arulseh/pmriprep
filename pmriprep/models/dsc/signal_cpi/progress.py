"""Bounded progress reporting for independent and spatial CPI fitting passes."""

import logging
import math
import time

import numpy as np

logger = logging.getLogger("nipype.interface")

_PROGRESS_FRACTION = 0.05
_PROGRESS_INTERVAL_SECONDS = 300.0


def _format_duration(seconds: float) -> str:
    """Format a nonnegative duration compactly for progress logs."""
    value = max(0, round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class _CPIProgress:
    """Emit bounded controller-side progress while voxel fits complete."""

    def __init__(self, total: int, *, label: str = "CPI fitting") -> None:
        self.total = total
        self.label = label
        self.completed = 0
        self.step = max(1, math.ceil(total * _PROGRESS_FRACTION))
        self.next_count = self.step
        self.started = time.monotonic()
        self.last_report = self.started

    def advance(self) -> None:
        """Record one completed voxel and log when count or time is due."""
        if self.total <= 0:
            return
        self.completed += 1
        now = time.monotonic()
        if (
            self.completed < self.total
            and self.completed < self.next_count
            and now - self.last_report < _PROGRESS_INTERVAL_SECONDS
        ):
            return
        elapsed = max(now - self.started, np.finfo(float).eps)
        remaining = (self.total - self.completed) * elapsed / self.completed
        logger.info(
            "%s progress: %d/%d voxels (%.1f%%); elapsed %s; "
            "estimated remaining %s",
            self.label,
            self.completed,
            self.total,
            100 * self.completed / self.total,
            _format_duration(elapsed),
            _format_duration(remaining),
        )
        self.last_report = now
        while self.next_count <= self.completed:
            self.next_count += self.step
