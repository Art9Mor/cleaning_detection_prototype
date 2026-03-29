from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from cleaning_detection.logging_setup import logger


class EventKind(str, Enum):
    """
    Типы событий для таблицы и аналитики.
    """

    EMPTY = 'empty'
    OCCUPIED = 'occupied'
    APPROACH = 'approach'


@dataclass(frozen=True)
class FrameEvent:
    """
    Одно зафиксированное событие на кадре (номер кадра, время, тип).
    """

    frame_index: int
    time_sec: float
    kind: EventKind


def collect_frame_events(
    prev_occupied: Optional[bool],
    occupied: bool,
    frame_index: int,
    time_sec: float,
) -> Tuple[List[FrameEvent], bool]:
    """
    По смене флага «в зоне есть человек» возвращает новые события и обновлённое состояние.

    - Первый кадр: фиксируем начальное empty/occupied.
    - Пусто -> занято: approach и occupied (подход после периода без людей).
    - Занято -> пусто: empty.
    - Занято после старта без предшествующего empty: только occupied (без approach).
    """

    events: List[FrameEvent] = []

    if prev_occupied is None:
        kind = EventKind.OCCUPIED if occupied else EventKind.EMPTY
        events.append(FrameEvent(frame_index, time_sec, kind))
        logger.trace(
            'Начальное состояние кадра {}: {}',
            frame_index,
            kind.value,
        )
        return events, occupied

    if prev_occupied == occupied:
        return events, occupied

    if occupied:
        # Было пусто -> появился человек
        events.append(FrameEvent(frame_index, time_sec, EventKind.APPROACH))
        events.append(FrameEvent(frame_index, time_sec, EventKind.OCCUPIED))
        logger.trace(
            'Кадр {} t={:.3f}s: подход и занятие (approach, occupied)',
            frame_index,
            time_sec,
        )
    else:
        events.append(FrameEvent(frame_index, time_sec, EventKind.EMPTY))
        logger.trace(
            'Кадр {} t={:.3f}s: стол пуст (empty)',
            frame_index,
            time_sec,
        )

    return events, occupied
