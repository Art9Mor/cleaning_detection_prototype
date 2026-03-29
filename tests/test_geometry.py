"""Тесты геометрии ROI и боксов."""

from __future__ import annotations

import pytest

from cleaning_detection.geometry import (
    intersection_area,
    person_box_area,
    person_in_table_zone,
)


def test_intersection_area_overlap() -> None:
    """Пересечение двух перекрывающихся прямоугольников даёт ожидаемую площадь."""
    box = (10.0, 10.0, 50.0, 50.0)
    roi = (20, 20, 40, 40)
    assert intersection_area(box, roi) == 30.0 * 30.0


def test_intersection_area_no_overlap() -> None:
    """Непересекающиеся боксы дают нулевую площадь пересечения."""
    box = (0.0, 0.0, 10.0, 10.0)
    roi = (20, 20, 10, 10)
    assert intersection_area(box, roi) == 0.0


def test_person_box_area() -> None:
    """Площадь бокса по координатам xyxy."""
    assert person_box_area((0.0, 0.0, 10.0, 20.0)) == 200.0


def test_person_in_table_zone_by_overlap() -> None:
    """Большое пересечение бокса с ROI проходит порог overlap."""
    roi = (100, 100, 100, 100)
    # Большая часть бокса внутри ROI
    box = (110.0, 110.0, 190.0, 190.0)
    assert person_in_table_zone(box, roi, min_overlap_ratio=0.08) is True


def test_person_in_table_zone_outside() -> None:
    """Бокс вне ROI не считается «у столика»."""
    roi = (100, 100, 100, 100)
    box = (0.0, 0.0, 10.0, 10.0)
    assert person_in_table_zone(box, roi) is False


def test_person_in_table_zone_overlap_ratio_quarter() -> None:
    """При доле пересечения 0.25 порог 0.08 проходит, 0.5 — нет."""
    roi = (0, 0, 100, 100)
    box = (80.0, 80.0, 120.0, 120.0)
    inter = intersection_area(box, roi)
    area = person_box_area(box)
    assert inter / area == pytest.approx(0.25, rel=0.01)
    assert person_in_table_zone(box, roi, min_overlap_ratio=0.08) is True
    assert person_in_table_zone(box, roi, min_overlap_ratio=0.5) is False
