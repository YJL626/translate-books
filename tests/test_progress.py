from datetime import datetime, timedelta, timezone

import pytest

from translate_books.progress import Progress, duration


def test_remaining_uses_weighted_work_and_ignores_cache_speed():
    clock = [0.0]
    now = datetime(2026, 9, 5, 10, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    progress = Progress(10, 1000, clock=lambda: clock[0], wall_clock=lambda: now)
    for _ in range(6):
        progress.complete(100, 0.001, cached=True)
    assert progress.remaining_seconds is None
    for work, seconds in [(50, 5), (100, 10), (150, 15)]:
        progress.complete(work, seconds, cached=False)
    assert progress.remaining_seconds == pytest.approx(10)
    clock[0] = 30
    output = progress.render("翻译正文")
    assert "当前 2026-09-05 10:00:00 +0800" in output
    assert "进度 9/10 (90.0%)" in output
    assert "本次已用 00:00:30" in output
    assert "预计剩余 00:00:10" in output
    assert "预计完成 09-05 10:00:10" in output


def test_estimate_warmup_and_completion_with_all_cached_results():
    progress = Progress(3, 300)
    assert "估算中" in progress.render("翻译")
    for _ in range(3):
        progress.complete(100, 0.001, cached=True)
    assert progress.remaining_seconds == 0
    assert "预计剩余 00:00:00" in progress.render("完成")
    assert "100.0%" in progress.render("完成")
    assert not progress.samples


def test_recent_speed_replaces_cold_start_and_finish_rolls_over_midnight():
    now = datetime(2026, 9, 5, 23, 59, 59, tzinfo=timezone(timedelta(hours=8)))
    progress = Progress(52, 5200, wall_clock=lambda: now)
    progress.complete(100, 1000, cached=False)
    for _ in range(50):
        progress.complete(100, 2, cached=False)
    assert progress.remaining_seconds == 2
    assert "预计完成 09-06 00:00:01" in progress.render("翻译")


def test_elapsed_time_uses_monotonic_clock_and_includes_summary_phase():
    progress = Progress(1, 100, started_at=10, clock=lambda: 100)
    assert "本次已用 00:01:30" in progress.render("翻译")
    assert duration(90061) == "25:01:01"
    assert duration(-1) == "00:00:00"
