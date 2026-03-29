"""Геометрия: пересечение бокса детекции с ROI столика."""

from __future__ import annotations


def intersection_area(
    box_xyxy: tuple[float, float, float, float],
    roi_xywh: tuple[int, int, int, int],
) -> float:
    """
    Площадь пересечения axis-aligned бокса человека и прямоугольника ROI.
    """

    x1, y1, x2, y2 = box_xyxy
    rx, ry, rw, rh = roi_xywh
    rx2, ry2 = rx + rw, ry + rh
    ix1 = max(x1, float(rx))
    iy1 = max(y1, float(ry))
    ix2 = min(x2, float(rx2))
    iy2 = min(y2, float(ry2))
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def person_box_area(box_xyxy: tuple[float, float, float, float]) -> float:
    """Площадь axis-aligned бокса детекции (xyxy)."""
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def person_in_table_zone(
    box_xyxy: tuple[float, float, float, float],
    roi_xywh: tuple[int, int, int, int],
    *,
    min_overlap_ratio: float = 0.08,
) -> bool:
    """
    Человек «у столика», если пересечение его бокса с ROI достаточно велико
    (устойчиво к частичному попаданию и сидячей позе).
    """
    area = person_box_area(box_xyxy)
    if area <= 0:
        return False
    inter = intersection_area(box_xyxy, roi_xywh)
    return (inter / area) >= min_overlap_ratio
