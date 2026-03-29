"""Тесты аналитики задержек."""

from __future__ import annotations

from cleaning_detection.analytics import (
    delays_empty_to_approach,
    events_to_dataframe,
    mean_delay_seconds,
)
from cleaning_detection.events import EventKind, FrameEvent


def test_events_to_dataframe_empty() -> None:
    """Пустой список событий — пустой DataFrame с нужными колонками."""
    df = events_to_dataframe([])
    assert list(df.columns) == ['frame', 'time_sec', 'event']
    assert len(df) == 0


def test_delays_empty_to_approach_pair() -> None:
    """Одна пара EMPTY→APPROACH даёт одну задержку."""
    events = [
        FrameEvent(0, 0.0, EventKind.EMPTY),
        FrameEvent(10, 5.0, EventKind.APPROACH),
    ]
    d = delays_empty_to_approach(events)
    assert d == [5.0]


def test_delays_multiple_cycles() -> None:
    """Несколько циклов пусто→подход накапливают несколько интервалов."""
    events = [
        FrameEvent(0, 0.0, EventKind.EMPTY),
        FrameEvent(1, 2.0, EventKind.APPROACH),
        FrameEvent(2, 10.0, EventKind.EMPTY),
        FrameEvent(3, 13.0, EventKind.APPROACH),
    ]
    d = delays_empty_to_approach(events)
    assert d == [2.0, 3.0]


def test_mean_delay_seconds() -> None:
    """Среднее по списку; пустой список — None."""
    assert mean_delay_seconds([]) is None
    assert mean_delay_seconds([2.0, 4.0]) == 3.0


def test_approach_without_prior_empty_ignored() -> None:
    """APPROACH без предшествующего EMPTY не даёт интервала."""
    events = [FrameEvent(0, 1.0, EventKind.APPROACH)]
    assert delays_empty_to_approach(events) == []
