"""Mark renderer — draws Mark objects onto frames using cv2.

This module converts declarative Mark data into actual pixels.
All rendering is done on a copy of the input frame.

Example:
    >>> from vpx.viz.renderer import render_marks
    >>> from vpx.sdk.marks import BBoxMark
    >>> marks = [BBoxMark(x=0.1, y=0.2, w=0.3, h=0.4, color=(0, 255, 0))]
    >>> output = render_marks(frame, marks)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from vpx.sdk.marks import (
    Mark,
    BBoxMark,
    KeypointsMark,
    BarMark,
    LabelMark,
    DrawStyle,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_marks(
    frame: np.ndarray,
    marks: list[Mark],
    style: DrawStyle | None = None,
) -> np.ndarray:
    """Render a list of marks onto a frame.

    Args:
        frame: BGR image (H, W, 3). A copy is made internally.
        marks: Mark objects from module.annotate().
        style: Optional style overrides.

    Returns:
        Annotated frame (copy).
    """
    if not marks:
        return frame

    output = frame.copy()
    h, w = output.shape[:2]

    for mark in marks:
        if isinstance(mark, BBoxMark):
            _render_bbox(output, mark, w, h, style)
        elif isinstance(mark, KeypointsMark):
            _render_keypoints(output, mark, w, h, style)
        elif isinstance(mark, BarMark):
            _render_bar(output, mark, w, h, style)
        elif isinstance(mark, LabelMark):
            _render_label(output, mark, w, h, style)

    return output


def _resolve_color(
    mark_color: tuple[int, int, int] | None,
    style: DrawStyle | None,
    default: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    """Resolve color: style override > mark color > default."""
    if style and style.color is not None:
        return style.color
    if mark_color is not None:
        return mark_color
    return default


def _render_bbox(
    image: np.ndarray,
    mark: BBoxMark,
    w: int,
    h: int,
    style: DrawStyle | None,
) -> None:
    x1 = int(mark.x * w)
    y1 = int(mark.y * h)
    x2 = int((mark.x + mark.w) * w)
    y2 = int((mark.y + mark.h) * h)

    color = _resolve_color(mark.color, style, (0, 255, 0))
    thickness = style.thickness if style and style.thickness is not None else mark.thickness

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label
    show_labels = style.show_labels if style else True
    if mark.label and show_labels:
        font_scale = style.font_scale if style and style.font_scale is not None else 0.45
        label = mark.label
        show_conf = style.show_confidence if style else True
        if show_conf and mark.confidence < 1.0:
            label = f"{label} {mark.confidence:.2f}"

        label_size = cv2.getTextSize(label, FONT, font_scale, 1)[0]
        label_y = y1 - 5 if y1 > 25 else y2 + 15

        # Background
        cv2.rectangle(
            image,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0] + 4, label_y + 2),
            color,
            -1,
        )
        # Text (dark on colored background)
        cv2.putText(image, label, (x1 + 2, label_y - 2), FONT, font_scale, (20, 20, 20), 1)


def _render_keypoints(
    image: np.ndarray,
    mark: KeypointsMark,
    w: int,
    h: int,
    style: DrawStyle | None,
) -> None:
    line_color = _resolve_color(None, style, mark.line_color)
    min_conf = mark.min_confidence

    # Draw connections
    for idx1, idx2 in mark.connections:
        if idx1 >= len(mark.points) or idx2 >= len(mark.points):
            continue
        p1 = mark.points[idx1]
        p2 = mark.points[idx2]
        if p1[2] < min_conf or p2[2] < min_conf:
            continue
        if mark.normalized:
            pt1 = (int(p1[0] * w), int(p1[1] * h))
            pt2 = (int(p2[0] * w), int(p2[1] * h))
        else:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(image, pt1, pt2, line_color, 2)

    # Draw points
    default_point_color = _resolve_color(None, style, mark.line_color)
    for i, p in enumerate(mark.points):
        if p[2] < min_conf:
            continue
        if mark.normalized:
            pt = (int(p[0] * w), int(p[1] * h))
        else:
            pt = (int(p[0]), int(p[1]))
        pc = (
            mark.point_colors.get(i, default_point_color)
            if mark.point_colors
            else default_point_color
        )
        radius = mark.point_radius
        cv2.circle(image, pt, radius, pc, -1)
        cv2.circle(image, pt, radius, (20, 20, 20), 1)


def _render_bar(
    image: np.ndarray,
    mark: BarMark,
    w: int,
    h: int,
    style: DrawStyle | None,
) -> None:
    x = int(mark.x * w)
    y = int(mark.y * h)
    bar_w = int(mark.w * w)
    bar_h = mark.height_px
    color = _resolve_color(mark.color, style, (0, 255, 255))
    fill = min(1.0, max(0.0, mark.value))

    # Background
    cv2.rectangle(image, (x, y), (x + bar_w, y + bar_h), (20, 20, 20), -1)
    # Fill
    if fill > 0:
        cv2.rectangle(image, (x, y), (x + int(bar_w * fill), y + bar_h), color, -1)


def _render_label(
    image: np.ndarray,
    mark: LabelMark,
    w: int,
    h: int,
    style: DrawStyle | None,
) -> None:
    if mark.normalized:
        x = int(mark.x * w)
        y = int(mark.y * h)
    else:
        x = int(mark.x)
        y = int(mark.y)

    font_scale = style.font_scale if style and style.font_scale is not None else mark.font_scale
    color = _resolve_color(mark.color, style, (255, 255, 255))

    if mark.background is not None:
        label_size = cv2.getTextSize(mark.text, FONT, font_scale, 1)[0]
        cv2.rectangle(
            image,
            (x - 2, y - label_size[1] - 4),
            (x + label_size[0] + 4, y + 4),
            mark.background,
            -1,
        )

    cv2.putText(image, mark.text, (x, y), FONT, font_scale, color, 1)
