from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

A4_LANDSCAPE_M = (0.297, 0.210)
MARKER_MARGIN_M = 0.025
MARKER_SIZE_M = 0.032
MARKER_IDS = (0, 1, 2, 3)


def marker_world_points() -> dict[int, tuple[float, float]]:
    """Coordinates of marker centres in the calibrated board frame."""
    width = A4_LANDSCAPE_M[0] - 2 * MARKER_MARGIN_M
    height = A4_LANDSCAPE_M[1] - 2 * MARKER_MARGIN_M
    return {0: (0.0, 0.0), 1: (width, 0.0), 2: (width, height), 3: (0.0, height)}


def _aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def make_capture_board(
    output: str | Path,
    *,
    dpi: int = 300,
    title: str = "TRACE2TOUCH CAPTURE BOARD",
) -> Path:
    """Create an A4 landscape calibration board that must be printed at 100%."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width_px = round(A4_LANDSCAPE_M[0] / 0.0254 * dpi)
    height_px = round(A4_LANDSCAPE_M[1] / 0.0254 * dpi)
    canvas = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    px_per_m = width_px / A4_LANDSCAPE_M[0]
    marker_px = round(MARKER_SIZE_M * px_per_m)
    margin_x = round(MARKER_MARGIN_M * px_per_m)
    margin_y = round(MARKER_MARGIN_M * height_px / A4_LANDSCAPE_M[1])
    centres = {
        0: (margin_x, margin_y),
        1: (width_px - margin_x, margin_y),
        2: (width_px - margin_x, height_px - margin_y),
        3: (margin_x, height_px - margin_y),
    }

    dictionary = _aruco_dictionary()
    for marker_id, (cx, cy) in centres.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        x0, y0 = cx - marker_px // 2, cy - marker_px // 2
        canvas[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker

    # Keep a clean white quiet zone around each marker. Lines touching the black
    # border look harmless to people but make square-marker decoding unreliable.
    inset = round(marker_px * 0.75)
    cv2.rectangle(
        canvas,
        (centres[0][0] + inset, centres[0][1] + inset),
        (centres[2][0] - inset, centres[2][1] - inset),
        color=(205, 205, 205),
        thickness=max(2, dpi // 80),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        title,
        (round(width_px * 0.30), round(height_px * 0.08)),
        font,
        1.1 * dpi / 300,
        (20, 20, 20),
        max(2, dpi // 120),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Print A4 landscape at 100%  |  Keep all four markers visible",
        (round(width_px * 0.25), round(height_px * 0.94)),
        font,
        0.65 * dpi / 300,
        (50, 50, 50),
        max(1, dpi // 150),
        cv2.LINE_AA,
    )

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    if output.suffix.lower() == ".pdf":
        image.save(output, "PDF", resolution=dpi)
    else:
        image.save(output, dpi=(dpi, dpi))

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "page_size_m": A4_LANDSCAPE_M,
                "marker_size_m": MARKER_SIZE_M,
                "marker_centres_m": marker_world_points(),
                "dictionary": "DICT_4X4_50",
                "print_scale": "100%",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def render_board_array(width_px: int = 1188) -> np.ndarray:
    """Return a lower-resolution board image for deterministic synthetic videos."""
    height_px = round(width_px * A4_LANDSCAPE_M[1] / A4_LANDSCAPE_M[0])
    canvas = np.full((height_px, width_px, 3), 250, dtype=np.uint8)
    px_per_m = width_px / A4_LANDSCAPE_M[0]
    marker_px = round(MARKER_SIZE_M * px_per_m)
    margin_x = round(MARKER_MARGIN_M * px_per_m)
    margin_y = round(MARKER_MARGIN_M * height_px / A4_LANDSCAPE_M[1])
    centres = {
        0: (margin_x, margin_y),
        1: (width_px - margin_x, margin_y),
        2: (width_px - margin_x, height_px - margin_y),
        3: (margin_x, height_px - margin_y),
    }
    dictionary = _aruco_dictionary()
    for marker_id, (cx, cy) in centres.items():
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        x0, y0 = cx - marker_px // 2, cy - marker_px // 2
        canvas[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker
    inset = round(marker_px * 0.75)
    cv2.rectangle(
        canvas,
        (centres[0][0] + inset, centres[0][1] + inset),
        (centres[2][0] - inset, centres[2][1] - inset),
        (190, 190, 190),
        3,
    )
    return canvas
