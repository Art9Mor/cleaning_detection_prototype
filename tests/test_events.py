from __future__ import annotations

from cleaning_detection.events import EventKind, collect_frame_events


def test_first_frame_empty() -> None:
    """
    Первый кадр без людей в зоне даёт событие EMPTY.
    """

    ev, prev = collect_frame_events(None, False, 0, 0.0)
    assert prev is False
    assert len(ev) == 1
    assert ev[0].kind == EventKind.EMPTY


def test_first_frame_occupied() -> None:
    """
    Первый кадр с человеком в зоне даёт OCCUPIED без APPROACH.
    """

    ev, prev = collect_frame_events(None, True, 0, 0.0)
    assert prev is True
    assert len(ev) == 1
    assert ev[0].kind == EventKind.OCCUPIED


def test_empty_to_occupied_emits_approach_and_occupied() -> None:
    """
    Переход пусто→занято порождает APPROACH и OCCUPIED.
    """

    ev, prev = collect_frame_events(False, True, 10, 1.0)
    assert prev is True
    kinds = [e.kind for e in ev]
    assert kinds == [EventKind.APPROACH, EventKind.OCCUPIED]


def test_occupied_to_empty() -> None:
    """
    Переход занято→пусто даёт EMPTY.
    """

    ev, prev = collect_frame_events(True, False, 20, 2.0)
    assert prev is False
    assert len(ev) == 1
    assert ev[0].kind == EventKind.EMPTY


def test_no_change_no_events() -> None:
    """
    Без смены состояния новых событий нет.
    """

    ev, prev = collect_frame_events(True, True, 5, 0.5)
    assert prev is True
    assert ev == []
