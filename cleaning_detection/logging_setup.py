"""Настройка Loguru: консоль и файлы в logs/, ротация и хранение 7 дней, уровни из .env."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Final

from dotenv import load_dotenv
from loguru import logger

# Цифра 7 — максимум (все уровни, в терминах Loguru это TRACE).
_NUMERIC_TO_LEVEL: Final[dict[int, str]] = {
    1: 'CRITICAL',
    2: 'ERROR',
    3: 'WARNING',
    4: 'INFO',
    5: 'SUCCESS',
    6: 'DEBUG',
    7: 'TRACE',
}

_VALID_LEVELS: Final[frozenset[str]] = frozenset(
    {
        'TRACE',
        'DEBUG',
        'INFO',
        'SUCCESS',
        'WARNING',
        'ERROR',
        'CRITICAL',
    },
)


def _parse_level(value: str) -> str:
    """
    Уровень из .env: цифра 1–7 или имя Loguru (TRACE … CRITICAL).

    Шкала 1–7 (чем больше, тем подробнее): 1 — только критичные, 7 — все уровни.
    """
    s = value.strip()
    if not s:
        raise ValueError('Пустой уровень логирования')
    if s.isdigit():
        n = int(s)
        if n not in _NUMERIC_TO_LEVEL:
            raise ValueError(
                f'Уровень по числу: целое от 1 до 7, получено {n!r}. '
                f'7 — отображать все уровни (TRACE).',
            )
        return _NUMERIC_TO_LEVEL[n]
    su = s.upper()
    if su in _VALID_LEVELS:
        return su
    raise ValueError(
        f'Недопустимый уровень: {value!r}. '
        f'Укажите цифру 1–7 (7 — все уровни) или имя: {", ".join(sorted(_VALID_LEVELS))}.',
    )


def _level_from_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return _parse_level(default)
    return _parse_level(str(raw).strip())


def configure_logging(project_root: Path) -> None:
    """
    Загружает `.env` из корня проекта, создаёт `logs/`, настраивает консоль и файлы.

    Переменные окружения (после load_dotenv):
    - LOG_LEVEL — общий уровень по умолчанию (цифра 1–7 или имя); по умолчанию **4** (INFO).
    - LOG_CONSOLE_LEVEL — только stderr; пусто = как LOG_LEVEL.
    - LOG_FILE_LEVEL — только файлы в logs/; пусто = как LOG_LEVEL.

    Шкала **1–7**: чем больше число, тем больше сообщений; **7** — все уровни (TRACE).
    """
    load_dotenv(project_root / '.env')

    default = _level_from_env('LOG_LEVEL', '4')
    console_raw = os.getenv('LOG_CONSOLE_LEVEL')
    file_raw = os.getenv('LOG_FILE_LEVEL')
    if console_raw is None or not str(console_raw).strip():
        console_level = default
    else:
        console_level = _parse_level(str(console_raw).strip())
    if file_raw is None or not str(file_raw).strip():
        file_level = default
    else:
        file_level = _parse_level(str(file_raw).strip())

    logs_dir = project_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    fmt_console = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>'
    )
    fmt_file = '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}'

    logger.add(
        sys.stderr,
        level=console_level,
        format=fmt_console,
        colorize=True,
    )

    logger.add(
        str(logs_dir / 'detection_{time:YYYY-MM-DD}.log'),
        level=file_level,
        format=fmt_file,
        encoding='utf-8',
        rotation='00:00',
        retention='7 days',
        compression=None,
        enqueue=True,
    )

    logger.debug(
        'Логирование: default={!r}, console={!r}, file={!r}, logs_dir={}',
        default,
        console_level,
        file_level,
        logs_dir,
    )


__all__ = ['configure_logging', 'logger']
