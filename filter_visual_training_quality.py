#!/usr/bin/env python3
"""
Filter videos for visual lip-reading training quality.

This script is intended to run after filter_american_english.py. It samples
frames from each video, detects face landmarks with MediaPipe Face Mesh, and
keeps videos that contain at least one high-quality usable section where a face
is visible, the mouth is large and clear enough, and the visible mouth appears
to be the active speaker. Bad sections elsewhere in the video are recorded but
do not automatically reject the whole file.

Output layout:

    <output>/trainable
    <output>/rejected

The default action is hardlink, so the original videos are not destroyed and
the dataset is not duplicated on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_INPUT = Path("/media/agam/Local Disk/the_code/roman/majorproject/american_english_filtered/american")
DEFAULT_OUTPUT = Path("/media/agam/Local Disk/the_code/roman/majorproject/american_visual_filtered")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")

OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIP_POINTS = sorted(set(OUTER_LIPS + INNER_LIPS))

FACE_LEFT = 234
FACE_RIGHT = 454
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

_FACE_MESH = None
_WORKER_CONFIG: Optional["VisualConfig"] = None


@dataclass
class VisualConfig:
    model_path: str
    sample_frames: int
    candidate_frames: int
    active_speaker_check: bool
    segment_acceptance: bool
    sample_windows: int
    window_seconds: float
    frame_stride: int
    max_motion_frames: int
    resize_width: int
    max_faces: int
    min_detection_confidence: float
    min_tracking_confidence: float
    min_face_rate: float
    min_good_frame_rate: float
    min_usable_segments: int
    min_segment_sampled_frames: int
    min_segment_face_rate: float
    min_segment_good_frame_rate: float
    min_face_height_ratio: float
    min_face_width_ratio: float
    min_mouth_width_ratio: float
    min_mouth_width_px: float
    min_mouth_height_px: float
    min_mouth_area_px: float
    max_yaw_proxy: float
    max_roll_degrees: float
    max_ambiguous_face_rate: float
    second_face_area_ratio: float
    edge_margin_ratio: float
    track_match_distance_ratio: float
    min_active_track_rate: float
    min_active_good_frame_rate: float
    min_primary_active_rate: float
    min_mouth_motion: float
    min_speech_open_ratio: float
    min_speech_open_rate: float
    active_competitor_ratio: float
    min_competitor_track_rate: float


class JsonState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": None,
            "files": {},
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update(loaded)
                self.data.setdefault("files", {})

    def is_done(self, video: Path, signature: Dict[str, Any], reprocess: bool) -> bool:
        if reprocess:
            return False
        entry = self.data.get("files", {}).get(str(video))
        return bool(entry and entry.get("signature") == signature)

    def put(self, video: Path, signature: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.data.setdefault("files", {})[str(video)] = {
            "signature": signature,
            "result": result,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        tmp.replace(self.path)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("visual_training_filter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "visual_training_filter.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def ensure_face_landmarker_model(model_path: Path, logger: logging.Logger) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    logger.info("Downloading MediaPipe face landmarker model to %s", model_path)
    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(model_path)
    return model_path


def cap_vram(device: str, limit_gb: float, logger: logging.Logger) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available() or limit_gb <= 0:
        return
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    fraction = min(1.0, max(0.05, limit_gb / total_gb))
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        logger.info(
            "GPU: %s, %.2f GB total, capped to %.2f GB (%.1f%%)",
            torch.cuda.get_device_name(0),
            total_gb,
            total_gb * fraction,
            fraction * 100.0,
        )
    except Exception as exc:
        logger.warning("Could not set PyTorch VRAM cap: %s", exc)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested in {"cuda", "cuda:0"} and not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA was requested, but torch.cuda.is_available() is false.")
    return "cuda:0" if requested == "cuda" else requested


def video_signature(video: Path) -> Dict[str, Any]:
    stat = video.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def config_hash(config: VisualConfig) -> str:
    payload = asdict(config)
    payload["model_path"] = Path(config.model_path).name
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def processing_signature(video: Path, config: VisualConfig) -> Dict[str, Any]:
    signature = video_signature(video)
    signature["config_hash"] = config_hash(config)
    return signature


def iter_videos(input_dir: Path, extensions: Sequence[str]) -> List[Path]:
    normalized = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions)
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in normalized)


def init_worker(config: VisualConfig) -> None:
    global _FACE_MESH, _WORKER_CONFIG
    import mediapipe as mp

    _WORKER_CONFIG = config
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=config.max_faces,
        min_face_detection_confidence=config.min_detection_confidence,
        min_face_presence_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _FACE_MESH = FaceLandmarker.create_from_options(options)


def frame_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0:
        return []
    if sample_count <= 1:
        return [frame_count // 2]
    start = 0.08
    end = 0.92
    return sorted(set(int(round((frame_count - 1) * f)) for f in np.linspace(start, end, sample_count)))


def sampling_windows(frame_count: int, fps: float, cfg: VisualConfig) -> List[Dict[str, Any]]:
    if frame_count <= 0:
        return []

    fps = fps if fps and fps > 0 else 30.0
    stride = max(1, int(cfg.frame_stride))
    half_window = max(stride, int(round(fps * max(0.1, cfg.window_seconds) * 0.5)))
    centers = frame_indices(frame_count, max(1, cfg.sample_windows))

    windows: List[Dict[str, Any]] = []
    for window_index, center in enumerate(centers):
        start = max(0, center - half_window)
        end = min(frame_count - 1, center + half_window)
        windows.append(
            {
                "window_index": window_index,
                "center_frame": int(center),
                "start_frame": int(start),
                "end_frame": int(end),
                "start_second": round(start / fps, 3),
                "end_second": round(end / fps, 3),
                "center_second": round(center / fps, 3),
            }
        )
    return windows


def motion_frame_indices(frame_count: int, fps: float, cfg: VisualConfig) -> List[int]:
    if frame_count <= 0:
        return []
    if not cfg.active_speaker_check and not cfg.segment_acceptance:
        return frame_indices(frame_count, max(cfg.sample_frames, cfg.candidate_frames))

    stride = max(1, int(cfg.frame_stride))

    indices = set()
    for window in sampling_windows(frame_count, fps, cfg):
        start = int(window["start_frame"])
        end = int(window["end_frame"])
        for idx in range(start, end + 1, stride):
            indices.add(idx)
        indices.add(int(window["center_frame"]))

    sorted_indices = sorted(indices)
    if cfg.max_motion_frames > 0 and len(sorted_indices) > cfg.max_motion_frames:
        keep_positions = np.linspace(0, len(sorted_indices) - 1, cfg.max_motion_frames)
        sorted_indices = [sorted_indices[int(round(pos))] for pos in keep_positions]
        sorted_indices = sorted(set(sorted_indices))
    return sorted_indices


def resize_for_detection(frame: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame
    scale = resize_width / float(width)
    return cv2.resize(frame, (resize_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)


def bbox_from_points(points: np.ndarray) -> Tuple[float, float, float, float]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])


def landmark_xy(landmarks: Sequence[Any], indices: Sequence[int], width: int, height: int) -> np.ndarray:
    return np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in indices], dtype=np.float32)


def face_area(landmarks: Sequence[Any], width: int, height: int) -> float:
    coords = landmark_xy(landmarks, range(len(landmarks)), width, height)
    x1, y1, x2, y2 = bbox_from_points(coords)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def analyze_face(landmarks: Sequence[Any], width: int, height: int, cfg: VisualConfig) -> Dict[str, Any]:
    all_points = landmark_xy(landmarks, range(len(landmarks)), width, height)
    mouth_points = landmark_xy(landmarks, LIP_POINTS, width, height)
    face_x1, face_y1, face_x2, face_y2 = bbox_from_points(all_points)
    mouth_x1, mouth_y1, mouth_x2, mouth_y2 = bbox_from_points(mouth_points)

    face_width = max(1.0, face_x2 - face_x1)
    face_height = max(1.0, face_y2 - face_y1)
    mouth_width = max(0.0, mouth_x2 - mouth_x1)
    mouth_height = max(0.0, mouth_y2 - mouth_y1)
    mouth_area = mouth_width * mouth_height
    mouth_open_px = abs((landmarks[14].y - landmarks[13].y) * height)
    mouth_open_ratio = mouth_open_px / max(mouth_width, 1.0)
    face_center = [(face_x1 + face_x2) * 0.5, (face_y1 + face_y2) * 0.5]
    mouth_center = [(mouth_x1 + mouth_x2) * 0.5, (mouth_y1 + mouth_y2) * 0.5]

    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    roll_degrees = abs(math.degrees(math.atan2((right.y - left.y) * height, (right.x - left.x) * width)))

    nose_x = landmarks[NOSE_TIP].x * width
    face_center_x = (face_x1 + face_x2) * 0.5
    yaw_proxy = abs(nose_x - face_center_x) / max(face_width * 0.5, 1.0)

    edge_margin_x = cfg.edge_margin_ratio * width
    edge_margin_y = cfg.edge_margin_ratio * height
    mouth_in_frame = (
        mouth_x1 >= edge_margin_x
        and mouth_y1 >= edge_margin_y
        and mouth_x2 <= width - edge_margin_x
        and mouth_y2 <= height - edge_margin_y
    )

    good = (
        face_height / height >= cfg.min_face_height_ratio
        and face_width / width >= cfg.min_face_width_ratio
        and mouth_width / width >= cfg.min_mouth_width_ratio
        and mouth_width >= cfg.min_mouth_width_px
        and mouth_height >= cfg.min_mouth_height_px
        and mouth_area >= cfg.min_mouth_area_px
        and yaw_proxy <= cfg.max_yaw_proxy
        and roll_degrees <= cfg.max_roll_degrees
        and mouth_in_frame
    )

    return {
        "good": good,
        "face_width_ratio": face_width / width,
        "face_height_ratio": face_height / height,
        "face_area_ratio": (face_width * face_height) / max(1.0, width * height),
        "mouth_width_ratio": mouth_width / width,
        "mouth_width_px": mouth_width,
        "mouth_height_px": mouth_height,
        "mouth_area_px": mouth_area,
        "mouth_open_px": mouth_open_px,
        "mouth_open_ratio": mouth_open_ratio,
        "yaw_proxy": yaw_proxy,
        "roll_degrees": roll_degrees,
        "mouth_in_frame": mouth_in_frame,
        "face_center": face_center,
        "mouth_center": mouth_center,
        "face_bbox": [face_x1, face_y1, face_x2, face_y2],
        "mouth_bbox": [mouth_x1, mouth_y1, mouth_x2, mouth_y2],
    }


def choose_primary_face(multi_face_landmarks: Sequence[Sequence[Any]], width: int, height: int) -> Tuple[Sequence[Any], List[float]]:
    areas = [face_area(face, width, height) for face in multi_face_landmarks]
    order = np.argsort(np.array(areas))[::-1]
    primary_index = int(order[0])
    return multi_face_landmarks[primary_index], [float(a) for a in sorted(areas, reverse=True)]


def read_sampled_frames(video: Path, cfg: VisualConfig) -> Tuple[List[Tuple[int, np.ndarray]], Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

    windows = sampling_windows(frame_count, fps, cfg) if cfg.segment_acceptance else []
    indices = motion_frame_indices(frame_count, fps, cfg)
    frames: List[Tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append((idx, resize_for_detection(frame, cfg.resize_width)))
    cap.release()

    meta = {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "sampled_frames": len(frames),
        "requested_frames": len(indices),
        "frame_stride": cfg.frame_stride if cfg.active_speaker_check or cfg.segment_acceptance else None,
        "sampling_windows": windows,
    }
    return frames, meta


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def assign_track(
    tracks: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    frame_number: int,
    cfg: VisualConfig,
    width: int,
    height: int,
) -> int:
    center = np.array(metrics["face_center"], dtype=np.float32)
    max_distance = cfg.track_match_distance_ratio * max(width, height)
    best_index = -1
    best_distance = float("inf")

    for idx, track in enumerate(tracks):
        if track["last_frame"] == frame_number:
            continue
        distance = float(np.linalg.norm(center - np.array(track["last_center"], dtype=np.float32)))
        if distance < best_distance and distance <= max_distance:
            best_index = idx
            best_distance = distance

    if best_index < 0:
        track_id = len(tracks)
        tracks.append(
            {
                "id": track_id,
                "last_center": metrics["face_center"],
                "last_frame": frame_number,
                "frames": [],
                "good": [],
                "primary": [],
                "mouth_open_ratio": [],
                "mouth_open_px": [],
                "mouth_width_px": [],
                "face_area_ratio": [],
            }
        )
        best_index = track_id

    track = tracks[best_index]
    track["last_center"] = metrics["face_center"]
    track["last_frame"] = frame_number
    track["frames"].append(frame_number)
    track["good"].append(bool(metrics.get("good")))
    track["primary"].append(False)
    track["mouth_open_ratio"].append(float(metrics.get("mouth_open_ratio", 0.0)))
    track["mouth_open_px"].append(float(metrics.get("mouth_open_px", 0.0)))
    track["mouth_width_px"].append(float(metrics.get("mouth_width_px", 0.0)))
    track["face_area_ratio"].append(float(metrics.get("face_area_ratio", 0.0)))
    return int(track["id"])


def summarize_active_speaker(
    tracks: List[Dict[str, Any]],
    sampled_frames: int,
    cfg: VisualConfig,
    frame_numbers: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    selected_frames = set(frame_numbers) if frame_numbers is not None else None
    if not tracks or sampled_frames <= 0:
        return {
            "enabled": cfg.active_speaker_check,
            "tracks": [],
            "active_track_id": None,
            "active_score": 0.0,
            "competitor_score": 0.0,
            "competitor_ratio": 0.0,
            "status": "no_tracks",
        }

    track_summaries: List[Dict[str, Any]] = []
    for track in tracks:
        selected_positions = [
            idx for idx, frame_number in enumerate(track["frames"]) if selected_frames is None or frame_number in selected_frames
        ]
        if not selected_positions:
            continue

        opens = np.array([track["mouth_open_ratio"][idx] for idx in selected_positions], dtype=np.float32)
        if opens.size >= 2:
            motion = float(np.percentile(opens, 90) - np.percentile(opens, 10))
            open_std = float(opens.std())
        else:
            motion = 0.0
            open_std = 0.0
        mean_open = float(opens.mean()) if opens.size else 0.0
        open_rate = float(np.mean(opens >= cfg.min_speech_open_ratio)) if opens.size else 0.0
        visible_rate = len(selected_positions) / max(1, sampled_frames)
        good_rate = sum(1 for idx in selected_positions if track["good"][idx]) / max(1, sampled_frames)
        primary_rate = sum(1 for idx in selected_positions if track["primary"][idx]) / max(1, len(selected_positions))
        face_area_values = [track["face_area_ratio"][idx] for idx in selected_positions]
        face_area = float(np.mean(face_area_values)) if face_area_values else 0.0

        activity_score = motion + 0.5 * open_std + 0.05 * open_rate + 0.05 * min(mean_open, 0.2)
        track_summaries.append(
            {
                "track_id": int(track["id"]),
                "frames": len(selected_positions),
                "visible_rate": visible_rate,
                "good_rate": good_rate,
                "primary_rate": primary_rate,
                "face_area_ratio_mean": face_area,
                "mouth_open_ratio_mean": mean_open,
                "mouth_motion": motion,
                "mouth_open_std": open_std,
                "speech_open_rate": open_rate,
                "activity_score": float(activity_score),
                "mouth_width_px": summarize([float(track["mouth_width_px"][idx]) for idx in selected_positions]),
            }
        )

    if not track_summaries:
        return {
            "enabled": cfg.active_speaker_check,
            "tracks": [],
            "active_track_id": None,
            "active_score": 0.0,
            "competitor_score": 0.0,
            "competitor_ratio": 0.0,
            "status": "no_tracks",
        }

    track_summaries.sort(key=lambda item: item["activity_score"], reverse=True)
    active = track_summaries[0]
    competitor = next(
        (item for item in track_summaries[1:] if item["visible_rate"] >= cfg.min_competitor_track_rate),
        None,
    )
    competitor_score = float(competitor["activity_score"]) if competitor else 0.0
    competitor_ratio = competitor_score / max(float(active["activity_score"]), 1e-6)

    status = "ok"
    if active["visible_rate"] < cfg.min_active_track_rate:
        status = "active_speaker_not_visible_enough"
    elif active["good_rate"] < cfg.min_active_good_frame_rate:
        status = "active_speaker_mouth_not_trainable"
    elif active["primary_rate"] < cfg.min_primary_active_rate:
        status = "active_speaker_not_primary_face"
    elif active["mouth_motion"] < cfg.min_mouth_motion:
        status = "mouth_not_moving_enough"
    elif active["speech_open_rate"] < cfg.min_speech_open_rate:
        status = "mouth_mostly_closed"
    elif competitor and competitor_ratio >= cfg.active_competitor_ratio:
        status = "competing_active_speaker"

    return {
        "enabled": cfg.active_speaker_check,
        "tracks": track_summaries,
        "active_track_id": active["track_id"],
        "active_score": float(active["activity_score"]),
        "competitor_score": competitor_score,
        "competitor_ratio": float(competitor_ratio),
        "status": status,
    }


def decide_full_video_visual(metrics: Dict[str, Any], cfg: VisualConfig) -> Tuple[bool, str]:
    if metrics["sampled_frames"] <= 0:
        return False, "no_decodable_frames"
    if metrics["face_rate"] < cfg.min_face_rate:
        return False, "low_face_detection_rate"
    if metrics["good_frame_rate"] < cfg.min_good_frame_rate:
        return False, "mouth_or_face_too_small_or_unclear"
    if metrics["ambiguous_face_rate"] > cfg.max_ambiguous_face_rate and not cfg.active_speaker_check:
        return False, "ambiguous_multiple_faces"
    if metrics["mouth_width_px"]["median"] < cfg.min_mouth_width_px:
        return False, "mouth_too_small"
    if metrics["face_height_ratio"]["median"] < cfg.min_face_height_ratio:
        return False, "face_too_far"
    if metrics["yaw_proxy"]["median"] > cfg.max_yaw_proxy:
        return False, "face_too_profile"
    if cfg.active_speaker_check:
        speaker = metrics.get("active_speaker", {})
        if speaker.get("status") != "ok":
            return False, str(speaker.get("status") or "active_speaker_failed")
    return True, "trainable_visual_quality"


def decide_segment_visual(segment: Dict[str, Any], cfg: VisualConfig) -> Tuple[bool, str]:
    if segment["sampled_frames"] < cfg.min_segment_sampled_frames:
        return False, "segment_too_few_sampled_frames"
    if segment["face_rate"] < cfg.min_segment_face_rate:
        return False, "segment_low_face_detection_rate"
    if segment["good_frame_rate"] < cfg.min_segment_good_frame_rate:
        return False, "segment_mouth_or_face_too_small_or_unclear"
    if segment["ambiguous_face_rate"] > cfg.max_ambiguous_face_rate and not cfg.active_speaker_check:
        return False, "segment_ambiguous_multiple_faces"
    if segment["mouth_width_px"]["median"] < cfg.min_mouth_width_px:
        return False, "segment_mouth_too_small"
    if segment["face_height_ratio"]["median"] < cfg.min_face_height_ratio:
        return False, "segment_face_too_far"
    if segment["yaw_proxy"]["median"] > cfg.max_yaw_proxy:
        return False, "segment_face_too_profile"
    if cfg.active_speaker_check:
        speaker = segment.get("active_speaker", {})
        if speaker.get("status") != "ok":
            return False, f"segment_{speaker.get('status') or 'active_speaker_failed'}"
    return True, "usable_visual_segment"


def summarize_segment(
    window: Dict[str, Any],
    frame_records: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    cfg: VisualConfig,
) -> Dict[str, Any]:
    start_frame = int(window["start_frame"])
    end_frame = int(window["end_frame"])
    records = [
        record
        for record in frame_records
        if start_frame <= int(record.get("source_frame_index", -1)) <= end_frame
    ]
    detected_records = [record for record in records if record.get("detected")]
    ambiguous = sum(1 for record in detected_records if record.get("ambiguous_faces"))
    frame_numbers = [int(record["frame_index"]) for record in records]

    segment = {
        **window,
        "sampled_frames": len(records),
        "detected_frames": len(detected_records),
        "good_frames": sum(1 for record in detected_records if record.get("good")),
        "face_rate": len(detected_records) / max(1, len(records)),
        "good_frame_rate": sum(1 for record in detected_records if record.get("good")) / max(1, len(records)),
        "ambiguous_face_rate": ambiguous / max(1, len(detected_records)),
        "face_height_ratio": summarize([float(record["face_height_ratio"]) for record in detected_records]),
        "face_width_ratio": summarize([float(record["face_width_ratio"]) for record in detected_records]),
        "mouth_width_ratio": summarize([float(record["mouth_width_ratio"]) for record in detected_records]),
        "mouth_width_px": summarize([float(record["mouth_width_px"]) for record in detected_records]),
        "mouth_height_px": summarize([float(record["mouth_height_px"]) for record in detected_records]),
        "mouth_area_px": summarize([float(record["mouth_area_px"]) for record in detected_records]),
        "mouth_open_px": summarize([float(record["mouth_open_px"]) for record in detected_records]),
        "mouth_open_ratio": summarize([float(record["mouth_open_ratio"]) for record in detected_records]),
        "yaw_proxy": summarize([float(record["yaw_proxy"]) for record in detected_records]),
        "roll_degrees": summarize([float(record["roll_degrees"]) for record in detected_records]),
        "active_speaker": summarize_active_speaker(tracks, len(records), cfg, frame_numbers),
    }
    trainable, reason = decide_segment_visual(segment, cfg)
    segment["trainable"] = trainable
    segment["reason"] = reason
    return segment


def summarize_segments(
    windows: List[Dict[str, Any]],
    frame_records: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    cfg: VisualConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    segments = [summarize_segment(window, frame_records, tracks, cfg) for window in windows]
    usable_segments = [segment for segment in segments if segment.get("trainable")]
    reason_counts: Dict[str, int] = {}
    for segment in segments:
        reason = str(segment.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return segments, usable_segments, reason_counts


def compact_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    speaker = segment.get("active_speaker", {})
    return {
        "window_index": segment.get("window_index"),
        "start_second": segment.get("start_second"),
        "end_second": segment.get("end_second"),
        "sampled_frames": segment.get("sampled_frames"),
        "detected_frames": segment.get("detected_frames"),
        "good_frames": segment.get("good_frames"),
        "face_rate": segment.get("face_rate"),
        "good_frame_rate": segment.get("good_frame_rate"),
        "active_track_id": speaker.get("active_track_id"),
        "active_score": speaker.get("active_score"),
        "competitor_ratio": speaker.get("competitor_ratio"),
    }


def decide_visual(metrics: Dict[str, Any], cfg: VisualConfig) -> Tuple[bool, str]:
    full_video_trainable, full_video_reason = decide_full_video_visual(metrics, cfg)
    if cfg.segment_acceptance:
        usable_segments = metrics.get("usable_segments", [])
        if len(usable_segments) >= cfg.min_usable_segments:
            return True, "usable_visual_segment"
        if full_video_trainable:
            return True, full_video_reason
        segment_reasons = metrics.get("segment_failure_reasons", {})
        if segment_reasons:
            top_reason = max(segment_reasons.items(), key=lambda item: item[1])[0]
            return False, f"no_usable_visual_segment:{top_reason}"
    return full_video_trainable, full_video_reason


def process_video_worker(video_str: str) -> Dict[str, Any]:
    global _FACE_MESH, _WORKER_CONFIG
    if _FACE_MESH is None or _WORKER_CONFIG is None:
        raise RuntimeError("worker was not initialized")

    cfg = _WORKER_CONFIG
    video = Path(video_str)
    started = time.time()
    frame_records: List[Dict[str, Any]] = []

    try:
        frames, meta = read_sampled_frames(video, cfg)
        detected = 0
        good = 0
        ambiguous = 0
        tracks: List[Dict[str, Any]] = []

        for frame_index, (source_frame_index, frame) in enumerate(frames):
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = _FACE_MESH.detect(mp_image)
            faces = result.face_landmarks or []
            if not faces:
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "source_frame_index": source_frame_index,
                        "detected": False,
                    }
                )
                continue

            detected += 1
            face_metrics_list = []
            for face in faces:
                metrics = analyze_face(face, width, height, cfg)
                face_metrics_list.append(metrics)
            face_metrics_list.sort(key=lambda item: item["face_area_ratio"], reverse=True)
            face_metrics = face_metrics_list[0]

            track_ids = []
            for metrics in face_metrics_list:
                track_ids.append(assign_track(tracks, metrics, frame_index, cfg, width, height))
            primary_track_id = track_ids[0]
            for track in tracks:
                if track["id"] == primary_track_id and track["primary"]:
                    track["primary"][-1] = True
                    break

            if face_metrics["good"]:
                good += 1
            face_areas = [float(item["face_area_ratio"]) for item in face_metrics_list]
            if len(face_areas) > 1 and face_areas[1] >= face_areas[0] * cfg.second_face_area_ratio:
                ambiguous += 1

            frame_records.append(
                {
                    "frame_index": frame_index,
                    "detected": True,
                    "face_count": len(faces),
                    "source_frame_index": source_frame_index,
                    "primary_track_id": primary_track_id,
                    "track_ids": track_ids,
                    "ambiguous_faces": len(face_areas) > 1
                    and face_areas[1] >= face_areas[0] * cfg.second_face_area_ratio,
                    **face_metrics,
                }
            )

        detected_records = [r for r in frame_records if r.get("detected")]
        metrics = {
            **meta,
            "detected_frames": detected,
            "good_frames": good,
            "face_rate": detected / max(1, meta["sampled_frames"]),
            "good_frame_rate": good / max(1, meta["sampled_frames"]),
            "ambiguous_face_rate": ambiguous / max(1, detected),
            "face_height_ratio": summarize([float(r["face_height_ratio"]) for r in detected_records]),
            "face_width_ratio": summarize([float(r["face_width_ratio"]) for r in detected_records]),
            "mouth_width_ratio": summarize([float(r["mouth_width_ratio"]) for r in detected_records]),
            "mouth_width_px": summarize([float(r["mouth_width_px"]) for r in detected_records]),
            "mouth_height_px": summarize([float(r["mouth_height_px"]) for r in detected_records]),
            "mouth_area_px": summarize([float(r["mouth_area_px"]) for r in detected_records]),
            "mouth_open_px": summarize([float(r["mouth_open_px"]) for r in detected_records]),
            "mouth_open_ratio": summarize([float(r["mouth_open_ratio"]) for r in detected_records]),
            "yaw_proxy": summarize([float(r["yaw_proxy"]) for r in detected_records]),
            "roll_degrees": summarize([float(r["roll_degrees"]) for r in detected_records]),
        }
        metrics["active_speaker"] = summarize_active_speaker(tracks, meta["sampled_frames"], cfg)
        if cfg.segment_acceptance:
            segments, usable_segments, segment_failure_reasons = summarize_segments(
                meta.get("sampling_windows", []),
                frame_records,
                tracks,
                cfg,
            )
            usable_segments = sorted(
                usable_segments,
                key=lambda segment: (
                    float(segment.get("good_frame_rate", 0.0)),
                    float(segment.get("face_rate", 0.0)),
                    float(segment.get("active_speaker", {}).get("active_score", 0.0)),
                ),
                reverse=True,
            )
            metrics["segment_acceptance"] = {
                "enabled": True,
                "min_usable_segments": cfg.min_usable_segments,
                "usable_segment_count": len(usable_segments),
            }
            metrics["segments"] = segments
            metrics["usable_segments"] = [compact_segment(segment) for segment in usable_segments]
            metrics["best_usable_segment"] = compact_segment(usable_segments[0]) if usable_segments else None
            metrics["segment_failure_reasons"] = segment_failure_reasons
        else:
            metrics["segment_acceptance"] = {"enabled": False}
            metrics["segments"] = []
            metrics["usable_segments"] = []
            metrics["best_usable_segment"] = None
            metrics["segment_failure_reasons"] = {}
        trainable, reason = decide_visual(metrics, cfg)
        bucket = "trainable" if trainable else "rejected"

        return {
            "video": str(video),
            "file_name": video.name,
            "bucket": bucket,
            "reason": reason,
            "metrics": metrics,
            "frames": frame_records,
            "error": None,
            "seconds": round(time.time() - started, 3),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        return {
            "video": str(video),
            "file_name": video.name,
            "bucket": "rejected",
            "reason": "processing_error",
            "metrics": {},
            "frames": frame_records,
            "error": str(exc),
            "seconds": round(time.time() - started, 3),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }


def unique_destination(dest: Path, source: Path) -> Path:
    if not dest.exists():
        return dest
    try:
        if dest.stat().st_size == source.stat().st_size:
            return dest
    except OSError:
        pass
    for idx in range(1, 10_000):
        candidate = dest.with_name(f"{dest.stem}__{idx}{dest.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find free destination for {dest}")


def place_file(source: Path, dest_dir: Path, action: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(dest_dir / source.name, source)
    if dest.exists():
        return dest
    if action == "hardlink":
        os.link(source, dest)
    elif action == "symlink":
        os.symlink(source, dest)
    elif action == "copy":
        shutil.copy2(source, dest)
    elif action == "move":
        shutil.move(str(source), str(dest))
    else:
        raise ValueError(f"unknown action: {action}")
    return dest


def append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def summarize_state(state: JsonState) -> Dict[str, int]:
    counts = {"trainable": 0, "rejected": 0, "errors": 0}
    for entry in state.data.get("files", {}).values():
        result = entry.get("result", {})
        bucket = result.get("bucket")
        if bucket in counts:
            counts[bucket] += 1
        if result.get("error"):
            counts["errors"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter videos by face and mouth visibility for lip-reading training.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input folder, usually american_english_filtered/american.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder.")
    parser.add_argument(
        "--action",
        choices=("hardlink", "symlink", "copy", "move"),
        default="hardlink",
        help="How to place files into trainable/rejected folders.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu. Used for ROCm availability/cap.")
    parser.add_argument("--vram-limit-gb", type=float, default=13.0, help="PyTorch per-process GPU memory cap.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel MediaPipe worker processes.")
    parser.add_argument("--sample-frames", type=int, default=12, help="Frames used for final quality decision.")
    parser.add_argument("--candidate-frames", type=int, default=12, help="Candidate sampled frames per video.")
    parser.add_argument(
        "--active-speaker-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track mouth motion over short windows and require the primary visible face to be the active speaker.",
    )
    parser.add_argument(
        "--segment-acceptance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep a video when at least one sampled temporal segment is trainable, even if other parts are bad.",
    )
    parser.add_argument("--sample-windows", type=int, default=6, help="Temporal windows sampled across the video.")
    parser.add_argument("--window-seconds", type=float, default=1.5, help="Seconds around each sampled window center.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Inspect every Nth frame inside sampled windows.")
    parser.add_argument("--max-motion-frames", type=int, default=96, help="Cap total frames inspected per video.")
    parser.add_argument("--resize-width", type=int, default=640, help="Downscale frames to this width before detection.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional local MediaPipe face_landmarker.task path.")
    parser.add_argument("--max-faces", type=int, default=3, help="Faces to detect per frame.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.55)
    parser.add_argument("--min-face-rate", type=float, default=0.65)
    parser.add_argument("--min-good-frame-rate", type=float, default=0.45)
    parser.add_argument("--min-usable-segments", type=int, default=1)
    parser.add_argument("--min-segment-sampled-frames", type=int, default=5)
    parser.add_argument("--min-segment-face-rate", type=float, default=0.65)
    parser.add_argument("--min-segment-good-frame-rate", type=float, default=0.45)
    parser.add_argument("--min-face-height-ratio", type=float, default=0.16)
    parser.add_argument("--min-face-width-ratio", type=float, default=0.10)
    parser.add_argument("--min-mouth-width-ratio", type=float, default=0.035)
    parser.add_argument("--min-mouth-width-px", type=float, default=28.0)
    parser.add_argument("--min-mouth-height-px", type=float, default=6.0)
    parser.add_argument("--min-mouth-area-px", type=float, default=180.0)
    parser.add_argument("--max-yaw-proxy", type=float, default=0.46)
    parser.add_argument("--max-roll-degrees", type=float, default=35.0)
    parser.add_argument("--max-ambiguous-face-rate", type=float, default=0.35)
    parser.add_argument("--second-face-area-ratio", type=float, default=0.55)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01)
    parser.add_argument("--track-match-distance-ratio", type=float, default=0.18)
    parser.add_argument("--min-active-track-rate", type=float, default=0.45)
    parser.add_argument("--min-active-good-frame-rate", type=float, default=0.30)
    parser.add_argument("--min-primary-active-rate", type=float, default=0.70)
    parser.add_argument("--min-mouth-motion", type=float, default=0.020)
    parser.add_argument("--min-speech-open-ratio", type=float, default=0.035)
    parser.add_argument("--min-speech-open-rate", type=float, default=0.12)
    parser.add_argument("--active-competitor-ratio", type=float, default=0.70)
    parser.add_argument("--min-competitor-track-rate", type=float, default=0.20)
    parser.add_argument("--limit", type=int, default=0, help="Process only this many unprocessed videos.")
    parser.add_argument("--reprocess", action="store_true", help="Ignore existing state and classify again.")
    parser.add_argument("--dry-run", action="store_true", help="Classify without linking/copying/moving files.")
    parser.add_argument("--extensions", nargs="+", default=list(VIDEO_EXTENSIONS), help="Video extensions to scan.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VisualConfig:
    if args.sample_frames < 1 or args.candidate_frames < 1:
        raise ValueError("--sample-frames and --candidate-frames must be at least 1.")
    if args.candidate_frames < args.sample_frames:
        args.candidate_frames = args.sample_frames
    return VisualConfig(
        model_path=str(args.model_path),
        sample_frames=args.sample_frames,
        candidate_frames=args.candidate_frames,
        active_speaker_check=bool(args.active_speaker_check),
        segment_acceptance=bool(args.segment_acceptance),
        sample_windows=max(1, args.sample_windows),
        window_seconds=max(0.1, args.window_seconds),
        frame_stride=max(1, args.frame_stride),
        max_motion_frames=max(0, args.max_motion_frames),
        resize_width=args.resize_width,
        max_faces=max(1, args.max_faces),
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_face_rate=args.min_face_rate,
        min_good_frame_rate=args.min_good_frame_rate,
        min_usable_segments=max(1, args.min_usable_segments),
        min_segment_sampled_frames=max(1, args.min_segment_sampled_frames),
        min_segment_face_rate=args.min_segment_face_rate,
        min_segment_good_frame_rate=args.min_segment_good_frame_rate,
        min_face_height_ratio=args.min_face_height_ratio,
        min_face_width_ratio=args.min_face_width_ratio,
        min_mouth_width_ratio=args.min_mouth_width_ratio,
        min_mouth_width_px=args.min_mouth_width_px,
        min_mouth_height_px=args.min_mouth_height_px,
        min_mouth_area_px=args.min_mouth_area_px,
        max_yaw_proxy=args.max_yaw_proxy,
        max_roll_degrees=args.max_roll_degrees,
        max_ambiguous_face_rate=args.max_ambiguous_face_rate,
        second_face_area_ratio=args.second_face_area_ratio,
        edge_margin_ratio=args.edge_margin_ratio,
        track_match_distance_ratio=args.track_match_distance_ratio,
        min_active_track_rate=args.min_active_track_rate,
        min_active_good_frame_rate=args.min_active_good_frame_rate,
        min_primary_active_rate=args.min_primary_active_rate,
        min_mouth_motion=args.min_mouth_motion,
        min_speech_open_ratio=args.min_speech_open_ratio,
        min_speech_open_rate=args.min_speech_open_rate,
        active_competitor_ratio=args.active_competitor_ratio,
        min_competitor_track_rate=args.min_competitor_track_rate,
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    logger = setup_logging(output_dir)
    started = time.time()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.exists():
        logger.error("Input folder does not exist: %s", input_dir)
        return 2

    device = resolve_device(args.device)
    cap_vram(device, args.vram_limit_gb, logger)
    if args.model_path is None:
        args.model_path = output_dir / "model_cache" / "face_landmarker.task"
    args.model_path = ensure_face_landmarker_model(args.model_path.expanduser().resolve(), logger)
    config = build_config(args)

    (output_dir / "trainable").mkdir(parents=True, exist_ok=True)
    (output_dir / "rejected").mkdir(parents=True, exist_ok=True)

    logger.info("Input: %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Action: %s%s", args.action, " (dry run)" if args.dry_run else "")
    logger.info(
        "Workers: %d, active speaker check: %s, segment acceptance: %s, windows: %d, frame stride: %d, max frames/video: %d",
        args.workers,
        config.active_speaker_check,
        config.segment_acceptance,
        config.sample_windows,
        config.frame_stride,
        config.max_motion_frames,
    )
    logger.info("Resize width: %d", config.resize_width)
    logger.info(
        "Thresholds: face_rate>=%.2f, good_frame_rate>=%.2f, face_height>=%.2f, mouth_width>=%.0fpx, mouth_motion>=%.3f",
        config.min_face_rate,
        config.min_good_frame_rate,
        config.min_face_height_ratio,
        config.min_mouth_width_px,
        config.min_mouth_motion,
    )
    if config.segment_acceptance:
        logger.info(
            "Segment thresholds: usable_segments>=%d, segment_frames>=%d, segment_face_rate>=%.2f, segment_good_frame_rate>=%.2f",
            config.min_usable_segments,
            config.min_segment_sampled_frames,
            config.min_segment_face_rate,
            config.min_segment_good_frame_rate,
        )

    state = JsonState(output_dir / "state.json")
    result_log = output_dir / ("dry_run_results.jsonl" if args.dry_run else "results.jsonl")

    videos = iter_videos(input_dir, args.extensions)
    todo: List[Path] = []
    for video in videos:
        signature = processing_signature(video, config)
        if not state.is_done(video, signature, args.reprocess):
            todo.append(video)
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    logger.info("Found %d videos, %d pending.", len(videos), len(todo))
    if not todo:
        logger.info("Nothing to do.")
        return 0

    counts = {"trainable": 0, "rejected": 0}
    worker_count = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=worker_count, initializer=init_worker, initargs=(config,)) as pool:
        futures = {pool.submit(process_video_worker, str(video)): video for video in todo}
        progress = tqdm(total=len(todo), unit="video", desc="Visual filter")
        try:
            for future in as_completed(futures):
                video = futures[future]
                result = future.result()
                signature = processing_signature(video, config)
                result["signature"] = signature
                dest_path: Optional[Path] = None
                if not args.dry_run:
                    dest_path = place_file(video, output_dir / result["bucket"], args.action)
                    result["destination"] = str(dest_path)
                    result["action"] = args.action
                    state.put(video, signature, result)
                    state.save()
                else:
                    result["destination"] = None
                    result["action"] = "dry_run"
                append_jsonl(result_log, result)

                counts[result["bucket"]] = counts.get(result["bucket"], 0) + 1
                progress.update(1)
                progress.set_postfix(trainable=counts.get("trainable", 0), rejected=counts.get("rejected", 0))
        finally:
            progress.close()

    final_counts = counts if args.dry_run else summarize_state(state)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "action": args.action,
        "device": device,
        "total_found": len(videos),
        "processed_this_run": sum(counts.values()),
        "counts": final_counts,
        "config": asdict(config),
        "seconds": round(time.time() - started, 2),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = output_dir / ("dry_run_summary.json" if args.dry_run else "summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(
        "Done. Trainable: %d, rejected: %d",
        final_counts.get("trainable", 0),
        final_counts.get("rejected", 0),
    )
    logger.info("Results: %s", result_log)
    logger.info("Summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
