"""Тесты настройки логирования."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pytest

from cleaning_detection.logging_setup import _parse_level, configure_logging


@pytest.fixture(autouse=True)
def reset_loguru_handlers() -> None:
    """Между тестами сбрасываем sinks loguru, чтобы не накапливались."""
    logger.remove()
    yield
    logger.remove()


def test_configure_logging_creates_logs_dir(tmp_path: Path) -> None:
    """После configure_logging в корне появляется каталог logs."""
    configure_logging(tmp_path)
    assert (tmp_path / 'logs').is_dir()


def test_configure_logging_invalid_level_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неверный уровень в окружении даёт понятную ошибку."""
    monkeypatch.setenv('LOG_LEVEL', 'NOT_A_LEVEL')
    with pytest.raises(ValueError, match='Недопустимый уровень'):
        configure_logging(tmp_path)


def test_parse_level_numeric_7_is_trace() -> None:
    """7 — все уровни (TRACE), максимум шкалы."""
    assert _parse_level('7') == 'TRACE'


def test_parse_level_numeric_4_is_info() -> None:
    """4 — INFO (значение по умолчанию)."""
    assert _parse_level('4') == 'INFO'


def test_parse_level_invalid_numeric_raises() -> None:
    """Допустимы только 1–7."""
    with pytest.raises(ValueError, match='1 до 7'):
        _parse_level('0')
    with pytest.raises(ValueError, match='1 до 7'):
        _parse_level('8')
    with pytest.raises(ValueError, match='1 до 7'):
        _parse_level('9')


def test_parse_level_name_backward_compat() -> None:
    """Имена уровней Loguru по-прежнему допустимы."""
    assert _parse_level('warning') == 'WARNING'
