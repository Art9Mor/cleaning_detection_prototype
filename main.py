#!/usr/bin/env python3
"""
Прототип: один столик на видео, YOLOv8n (люди), события пусто/занято/подход,
разметка ROI через cv2.selectROI, вывод в outputs/ и отчёт в консоль.

Запуск: python main.py --video path/to/file.mp4
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

from cleaning_detection.analytics import (
    delays_empty_to_approach,
    events_to_dataframe,
    mean_delay_seconds,
)
from cleaning_detection.events import FrameEvent, collect_frame_events
from cleaning_detection.geometry import person_in_table_zone
from cleaning_detection.logging_setup import configure_logging, logger

# Корень проекта (рядом с main.py): .env, каталоги logs/, outputs/
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / 'outputs' / 'output.mp4'
DEFAULT_REPORT_FILE = PROJECT_ROOT / 'outputs' / 'report.txt'

# Цвета BGR для ROI: зелёный — пусто, красный — занято
COLOR_EMPTY = (0, 200, 0)
COLOR_OCCUPIED = (0, 0, 220)


def parse_args() -> argparse.Namespace:
    """
    Парсинг аргументов CLI и возврат namespace с путями и параметрами детекции.
    """

    p = argparse.ArgumentParser(
        description='Детекция событий у выбранного столика по видео (YOLO + ROI).',
    )
    p.add_argument(
        '--video',
        type=Path,
        required=True,
        help='Путь к входному видео (например videos/видео 1.mp4).',
    )
    p.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_VIDEO,
        help='Куда сохранить размеченное видео (по умолчанию outputs/output.mp4).',
    )
    p.add_argument(
        '--report',
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help='Текстовый отчёт со средней задержкой (по умолчанию outputs/report.txt).',
    )
    p.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Веса YOLO (Ultralytics), по умолчанию yolov8n.pt.',
    )
    p.add_argument(
        '--conf',
        type=float,
        default=0.35,
        help='Порог уверенности детекции человека.',
    )
    p.add_argument(
        '--skip-roi',
        action='store_true',
        help='Не показывать окно ROI: взять весь кадр как зону (для тестов без GUI).',
    )
    return p.parse_args()


def select_table_roi(first_frame: np.ndarray, *, skip_roi: bool) -> tuple[int, int, int, int]:
    """
    Ручной выбор прямоугольника столика на первом кадре.
    Возвращает (x, y, w, h) в пикселях, как в cv2.selectROI.
    """
    if skip_roi:
        h, w = first_frame.shape[:2]
        logger.info('ROI: весь кадр {}x{} (--skip-roi)', w, h)
        return 0, 0, w, h
    logger.info(
        'Окно с первым кадром видео: мышью тяните прямоугольник вокруг одного столика, '
        'затем нажмите Enter или Пробел. Отмена — клавиша C (пустой прямоугольник — ошибка).',
    )
    # Без namedWindow: на Wayland/Qt часто ломается cv2.namedWindow + selectROI(name, img).
    try:
        r = cv2.selectROI(
            first_frame,
            showCrosshair=True,
            fromCenter=False,
            printNotice=False,
        )
    except cv2.error as e:
        logger.exception('Ошибка OpenCV при выборе ROI: {}', e)
        raise SystemExit(
            'Не удалось открыть окно выбора ROI (OpenCV HighGUI / Wayland / Qt).\n'
            'Попробуйте: QT_QPA_PLATFORM=xcb uv run python main.py --video ...\n'
            'или без GUI: --skip-roi (зона = весь кадр).\n'
            f'Исходная ошибка: {e}',
        ) from e
    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        logger.error('ROI отменён или нулевой размер')
        raise SystemExit('ROI не выбран: ширина/высота должны быть > 0.')
    logger.info('ROI столика (x, y, w, h): ({}, {}, {}, {})', x, y, w, h)
    return x, y, w, h


def frame_has_person_in_roi(
    model: Any,
    frame_bgr: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    conf: float,
) -> bool:
    """
    Проверка, есть ли в ROI хотя бы один человек (YOLO, класс person) выше порога conf.
    """

    results = model.predict(frame_bgr, conf=conf, classes=[0], verbose=False)
    if not results:
        return False
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False
    raw = boxes.xyxy
    xyxy = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    for row in xyxy:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        if person_in_table_zone((x1, y1, x2, y2), roi_xywh):
            return True
    return False


def draw_table_roi(
    frame: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    occupied: bool,
) -> None:
    """
    Отрисовка на кадре ROI столика и подпись состояния (цвет: пусто / занято).
    """

    x, y, w, h = roi_xywh
    color = COLOR_OCCUPIED if occupied else COLOR_EMPTY
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = 'occupied' if occupied else 'empty'
    cv2.putText(
        frame,
        label,
        (x, max(y - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def run_pipeline(
    video_path: Path,
    output_path: Path,
    report_path: Path,
    model_name: str,
    conf: float,
    skip_roi: bool,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Полный цикл: ROI, YOLO по кадрам, события, отчёт, запись размеченного видео.
    """

    logger.info(
        'Старт пайплайна: video={} output={} report={} model={} conf={} skip_roi={}',
        video_path,
        output_path,
        report_path,
        model_name,
        conf,
        skip_roi,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error('Не удалось открыть видео: {}', video_path)
        raise SystemExit(f'Не удалось открыть видео: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info('Видео: {:.2f} FPS, размер {}x{}', fps, width, height)

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        logger.error('Пустое видео или ошибка чтения первого кадра')
        raise SystemExit('Пустое видео или ошибка чтения первого кадра.')

    roi = select_table_roi(first, skip_roi=skip_roi)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        logger.error('Не удалось создать выходное видео: {}', output_path)
        raise SystemExit(f'Не удалось создать выходное видео: {output_path}')

    logger.info('Выходное видео открыто для записи: {}', output_path)

    logger.info(
        'Загрузка модели YOLO: {}. Первый запуск: скачивание весов (~6 МБ) — '
        'дождитесь окончания, не прерывайте Ctrl+C.',
        model_name,
    )
    model = YOLO(model_name)
    logger.info(
        'Модель готова (conf={}). Покадровая обработка всего ролика может занять много времени; '
        'прогресс в логе каждые 500 кадров (уровень DEBUG).',
        conf,
    )

    all_events: List[FrameEvent] = []
    prev_occupied: Optional[bool] = None
    frame_index = 0
    frame: Optional[np.ndarray] = first

    log_every = 500
    while frame is not None:
        time_sec = frame_index / fps
        if frame_index > 0 and frame_index % log_every == 0:
            logger.debug('Обработка кадра {} (~{:.1f} с)', frame_index, time_sec)
        occupied = frame_has_person_in_roi(model, frame, roi, conf)
        new_ev, prev_occupied = collect_frame_events(
            prev_occupied,
            occupied,
            frame_index,
            time_sec,
        )
        all_events.extend(new_ev)

        draw_table_roi(frame, roi, occupied)
        writer.write(frame)

        frame_index += 1
        _ok, frame = cap.read()
        if not _ok:
            frame = None

    cap.release()
    writer.release()
    logger.info(
        'Кадры обработаны, выходной файл закрыт ({} кадров)',
        frame_index,
    )

    df = events_to_dataframe(all_events)
    delays = delays_empty_to_approach(all_events)
    mean_d = mean_delay_seconds(delays)

    by_kind = Counter(e.kind.value for e in all_events)
    logger.info('События по типам: {}', dict(by_kind))

    lines = [
        f'Видео: {video_path}',
        f'ROI столика (x, y, w, h): {roi}',
        f'Кадров обработано: {frame_index}',
        f'Событий: {len(all_events)}',
        f'Интервалов «пусто -> подход»: {len(delays)}',
    ]
    if mean_d is not None:
        msg = f'Среднее время между уходом (пустой стол) и подходом следующего: {mean_d:.3f} с'
        lines.append(msg)
    else:
        lines.append('Среднее время: нет пар empty -> approach (недостаточно событий).')

    report_text = '\n'.join(lines) + '\n'
    report_path.write_text(report_text, encoding='utf-8')
    logger.info('Отчёт записан в {}', report_path)
    logger.info('Итог:\n{}', report_text.rstrip('\n'))
    print(report_text)

    logger.success(
        'Пайплайн завершён: кадров={}, событий={}, средняя задержка={}',
        frame_index,
        len(all_events),
        f'{mean_d:.3f} с' if mean_d is not None else '—',
    )

    return df, mean_d


def main() -> None:
    """
    Точка входа: валидация пути к видео и запуск пайплайна.
    """

    configure_logging(PROJECT_ROOT)
    args = parse_args()
    logger.debug(
        'CLI: video={} output={} report={} model={} conf={} skip_roi={}',
        args.video,
        args.output,
        args.report,
        args.model,
        args.conf,
        args.skip_roi,
    )
    if not args.video.is_file():
        logger.error('Файл не найден: {}', args.video)
        print(f'Файл не найден: {args.video}', file=sys.stderr)
        sys.exit(1)
    run_pipeline(
        args.video,
        args.output,
        args.report,
        args.model,
        args.conf,
        args.skip_roi,
    )


if __name__ == '__main__':
    main()
