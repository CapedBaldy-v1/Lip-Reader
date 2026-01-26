from __future__ import annotations

"""
Artifact helpers for Swin-VALLR.
Stores images, plots, matrices, and metadata in structured folders.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:
    np = None

from logging_utils import setup_logging


LOGGER = setup_logging("artifacts")
_RUN_IDS: Dict[str, str] = {}


def _run_id(component: str) -> str:
    run_id = _RUN_IDS.get(component)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        _RUN_IDS[component] = run_id
    return run_id


def _date_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_artifact_root(component: str, base_dir: Optional[Path] = None) -> Path:
    base_dir = base_dir or Path(__file__).resolve().parent
    root = base_dir / "artifacts" / component / _date_folder() / _run_id(component)
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_useful_dir(component: str, name: str, base_dir: Optional[Path] = None) -> Path:
    root = get_artifact_root(component, base_dir)
    useful = root / f"useful_{name}"
    useful.mkdir(parents=True, exist_ok=True)
    return useful


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    LOGGER.debug("Saved text artifact: %s", path)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    LOGGER.debug("Saved json artifact: %s", path)


def save_csv(path: Path, rows: Iterable[Sequence], headers: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
    LOGGER.debug("Saved csv artifact: %s", path)


def save_numpy(path: Path, array: np.ndarray) -> None:
    if np is None:
        LOGGER.warning("numpy not available; skipping numpy save for %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    LOGGER.debug("Saved numpy artifact: %s", path)


def _import_cv2():
    try:
        import cv2
        return cv2
    except Exception as exc:
        LOGGER.warning("cv2 not available; skipping image save. %s", exc)
        return None


def save_image(path: Path, image: np.ndarray) -> None:
    cv2 = _import_cv2()
    if cv2 is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    LOGGER.debug("Saved image artifact: %s", path)


def make_montage(frames: List[np.ndarray], cols: int = 10, pad: int = 2) -> Optional[np.ndarray]:
    if np is None:
        LOGGER.warning("numpy not available; skipping montage creation.")
        return None
    if not frames:
        return None
    cv2 = _import_cv2()
    if cv2 is None:
        return None

    normalized_frames = []
    rows = int(np.ceil(len(frames) / cols))
    h, w = frames[0].shape[:2]
    channels = 1 if frames[0].ndim == 2 else frames[0].shape[2]

    montage_h = rows * h + (rows - 1) * pad
    montage_w = cols * w + (cols - 1) * pad
    if channels == 1:
        canvas = np.zeros((montage_h, montage_w), dtype=frames[0].dtype)
    else:
        canvas = np.zeros((montage_h, montage_w, channels), dtype=frames[0].dtype)

    for frame in frames:
        resized = frame
        if frame.shape[:2] != (h, w):
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        if channels == 1 and resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        elif channels != 1 and resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        normalized_frames.append(resized)

    for idx, frame in enumerate(normalized_frames):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y:y + h, x:x + w] = frame

    return canvas


def save_montage(path: Path, frames: List[np.ndarray], cols: int = 10, pad: int = 2) -> None:
    montage = make_montage(frames, cols=cols, pad=pad)
    if montage is None:
        return
    save_image(path, montage)


def save_line_plot(
    path: Path,
    series: Dict[str, List[float]],
    title: str = "",
    width: int = 900,
    height: int = 500
) -> None:
    if np is None:
        LOGGER.warning("numpy not available; skipping plot save.")
        return
    cv2 = _import_cv2()
    if cv2 is None:
        return

    if not series:
        return

    all_values = [v for values in series.values() for v in values]
    if not all_values:
        return

    min_val = float(min(all_values))
    max_val = float(max(all_values))
    if min_val == max_val:
        max_val += 1.0

    margin = 50
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (0, 0, 0), 1)

    colors = [
        (0, 120, 255),
        (0, 200, 0),
        (200, 0, 0),
        (120, 0, 200),
    ]

    for idx, (label, values) in enumerate(series.items()):
        if len(values) < 2:
            continue
        color = colors[idx % len(colors)]
        points = []
        for i, value in enumerate(values):
            x = margin + int((width - 2 * margin) * (i / (len(values) - 1)))
            y = margin + int((height - 2 * margin) * (1 - (value - min_val) / (max_val - min_val)))
            points.append((x, y))
        for i in range(1, len(points)):
            cv2.line(canvas, points[i - 1], points[i], color, 2)

        cv2.putText(
            canvas,
            label,
            (margin + 5, margin + 20 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    if title:
        cv2.putText(
            canvas,
            title,
            (margin, margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    save_image(path, canvas)
