from __future__ import annotations

from typing import List, Optional

import pandas as pd

from cleaning_detection.events import EventKind, FrameEvent
from cleaning_detection.logging_setup import logger


def events_to_dataframe(events: List[FrameEvent]) -> pd.DataFrame:
    """
    Сбор списка событий в таблицу с колонками frame, time_sec, event.
    """

    if not events:
        logger.trace('Таблица событий: пустой список')
        return pd.DataFrame(
            columns=['frame', 'time_sec', 'event'],
        )
    rows = [
        {
            'frame': e.frame_index,
            'time_sec': e.time_sec,
            'event': e.kind.value,
        }
        for e in events
    ]
    logger.trace('Таблица событий: {} строк', len(rows))
    return pd.DataFrame(rows)


def delays_empty_to_approach(events: List[FrameEvent]) -> List[float]:
    """
    Для каждого момента, когда стол стал пустым, время (сек) до следующего подхода.

    Подход — событие APPROACH; пустота — EMPTY. Пары идут в хронологическом порядке.
    """

    delays: List[float] = []
    last_empty_time: Optional[float] = None

    for e in sorted(events, key=lambda x: (x.time_sec, x.frame_index)):
        if e.kind == EventKind.EMPTY:
            last_empty_time = e.time_sec
        elif e.kind == EventKind.APPROACH and last_empty_time is not None:
            delays.append(e.time_sec - last_empty_time)
            last_empty_time = None

    logger.trace('Интервалов empty→approach: {}', len(delays))
    return delays


def mean_delay_seconds(delays: List[float]) -> Optional[float]:
    """
    Среднее арифметическое задержек (сек); для пустого списка — None.
    """

    if not delays:
        logger.trace('Средняя задержка: нет данных')
        return None
    mean_val = sum(delays) / len(delays)
    logger.trace('Средняя задержка empty→approach: {:.4f} с (n={})', mean_val, len(delays))
    return mean_val
