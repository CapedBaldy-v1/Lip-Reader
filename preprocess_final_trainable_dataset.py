#!/usr/bin/env python3
"""
Build a training-ready lip-reading dataset from final_trainable_videos_heavy.

This script is intentionally stricter than the older preprocessors:
- it creates or reuses Whisper word-level transcripts for every source video;
- it makes labelled clips from word windows, not whole-video labels;
- it tracks multiple faces and keeps the most likely active speaker;
- it never pads a clip with blank/no-face frames.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import math
import os
import shutil
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

warnings.filterwarnings(
    "ignore",
    message=r"Failed to launch Triton kernels.*",
    category=UserWarning,
    module=r"whisper\.timing",
)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
BOTCHED_WORD_RE = re.compile(r"[^a-zA-Z0-9' -]+")

cv2 = None
np = None
tqdm = None
MediaPipeProcessor = None
MEDIAPIPE_AVAILABLE = False


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Detection:
    frame_index: int
    time_s: float
    frame: np.ndarray
    metrics: Any
    landmarks: Any
    track_id: int = -1


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "preprocess_final_dataset.log"

    logger = logging.getLogger("final_preprocessor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def clean_word(text: str) -> str:
    text = BOTCHED_WORD_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text


def clean_label(words: Sequence[Word]) -> str:
    label = " ".join(clean_word(word.word) for word in words)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "video"


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def media_files(input_dir: Path, limit: Optional[int]) -> List[Path]:
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if limit is not None:
        files = files[:limit]
    return files


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    payload = {"time": datetime.now().isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def configure_project_runtime_storage(output_dir: Path) -> None:
    runtime_dir = output_dir / "runtime"
    tmp_dir = runtime_dir / "tmp"
    cache_dir = runtime_dir / "cache"
    torch_dir = cache_dir / "torch"
    whisper_dir = cache_dir / "whisper"
    for path in (tmp_dir, cache_dir, torch_dir, whisper_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TORCH_HOME", str(torch_dir))
    os.environ.setdefault("WHISPER_CACHE_DIR", str(whisper_dir))


def source_id_from_sample_id(sample_id: str) -> str:
    marker = "_clip_"
    if marker in sample_id:
        return sample_id.split(marker, 1)[0]
    return sample_id


def preprocessing_config_signature(args: argparse.Namespace) -> str:
    relevant = {
        "whisper_model": args.whisper_model,
        "language": args.language,
        "clip_seconds": args.clip_seconds,
        "stride_seconds": args.stride_seconds,
        "context_seconds": args.context_seconds,
        "num_frames": args.num_frames,
        "target_fps": args.target_fps,
        "roi_size": args.roi_size,
        "max_faces": args.max_faces,
        "max_track_distance": args.max_track_distance,
        "candidate_multiplier": args.candidate_multiplier,
        "min_face_frame_rate": args.min_face_frame_rate,
        "min_lip_size": args.min_lip_size,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_source_duration_seconds": args.max_source_duration_seconds,
        "visual_scan_fps": args.visual_scan_fps,
        "visual_scan_min_step": args.visual_scan_min_step,
        "visual_scan_max_samples": args.visual_scan_max_samples,
        "visual_merge_gap": args.visual_merge_gap,
        "visual_padding": args.visual_padding,
        "min_visual_overlap": args.min_visual_overlap,
        "min_transcribe_face_sample_rate": args.min_transcribe_face_sample_rate,
        "min_transcribe_face_samples": args.min_transcribe_face_samples,
        "skip_visual_prefilter": args.skip_visual_prefilter,
        "max_clips_per_video": args.max_clips_per_video,
        "av1_transcode": args.av1_transcode,
        "av1_transcode_crf": args.av1_transcode_crf,
        "av1_transcode_preset": args.av1_transcode_preset,
        "compat_transcode_codecs": args.compat_transcode_codecs,
    }
    raw = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def ffprobe_video_codec(video_path: Path) -> Optional[str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
    except Exception:
        return None

    codec = result.stdout.strip().splitlines()
    return codec[0].strip().lower() if codec else None


def ensure_visual_compatible_video(
    video_path: Path,
    transcode_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Path:
    mode = args.av1_transcode
    if mode == "never":
        return video_path

    codec = ffprobe_video_codec(video_path)
    compat_codecs = {
        item.strip().lower()
        for item in args.compat_transcode_codecs.split(",")
        if item.strip()
    }
    should_transcode = mode == "always" or (codec is not None and codec in compat_codecs)
    if not should_transcode:
        return video_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("Cannot transcode %s: ffmpeg is not available on PATH", video_path.name)
        return video_path

    transcode_dir.mkdir(parents=True, exist_ok=True)
    output_path = transcode_dir / f"{safe_stem(video_path)}__h264.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Using cached H.264 visual-compatible copy for %s", video_path.name)
        return output_path

    tmp_path = output_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("Transcoding %s from %s to clean H.264 for OpenCV/MediaPipe compatibility", video_path.name, codec or "unknown")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        args.av1_transcode_preset,
        "-crf",
        str(args.av1_transcode_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        str(tmp_path),
    ]
    try:
        started = time.time()
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        tmp_path.replace(output_path)
        logger.info(
            "Finished H.264 compatibility transcode for %s in %.1fs: %s",
            video_path.name,
            time.time() - started,
            output_path,
        )
        return output_path
    except subprocess.CalledProcessError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        stderr = (exc.stderr or "").strip()
        logger.warning("H.264 transcode failed for %s: %s", video_path.name, stderr[-1200:])
        return video_path


def transcript_paths(transcripts_dir: Path, video_path: Path) -> Dict[str, Path]:
    stem = safe_stem(video_path)
    return {
        "json": transcripts_dir / f"{stem}.json",
        "txt": transcripts_dir / f"{stem}.txt",
        "words": transcripts_dir / f"{stem}_words.txt",
        "srt": transcripts_dir / f"{stem}.srt",
    }


def words_from_transcript(payload: Dict[str, Any]) -> List[Word]:
    words: List[Word] = []
    for item in payload.get("words", []):
        word = clean_word(str(item.get("word", "")))
        if not word:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end <= start:
            end = start + 0.05
        words.append(
            Word(
                word=word,
                start=start,
                end=end,
                probability=float(item.get("probability", 1.0) or 1.0),
            )
        )
    return words


def write_transcript_sidecars(paths: Dict[str, Path], payload: Dict[str, Any]) -> None:
    words = words_from_transcript(payload)
    paths["txt"].write_text(str(payload.get("transcript", "")).strip() + "\n", encoding="utf-8")

    readable_lines = [
        f"Video: {payload.get('video_id', '')}",
        f"Words: {len(words)}",
        "",
    ]
    for word in words:
        readable_lines.append(f"{word.start:9.3f} - {word.end:9.3f} | {word.word}")
    paths["words"].write_text("\n".join(readable_lines) + "\n", encoding="utf-8")

    srt_lines: List[str] = []
    for idx, word in enumerate(words, 1):
        srt_lines.extend([
            str(idx),
            f"{format_srt_time(word.start)} --> {format_srt_time(word.end)}",
            word.word,
            "",
        ])
    paths["srt"].write_text("\n".join(srt_lines), encoding="utf-8")


def load_or_create_transcript(
    video_path: Path,
    transcripts_dir: Path,
    model: Any,
    device: str,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    paths = transcript_paths(transcripts_dir, video_path)
    if paths["json"].exists() and not args.retranscribe:
        payload = load_json(paths["json"], None)
        if payload and words_from_transcript(payload):
            logger.info("Using cached transcript for %s (%s words)", video_path.name, payload.get("word_count", "?"))
            return payload

    logger.info("Transcribing %s with Whisper %s", video_path.name, args.whisper_model)
    log_gpu_memory(logger, device, f"Before Whisper transcription {video_path.name}")
    try:
        result = model.transcribe(
            str(video_path),
            language=args.language,
            fp16=(device == "cuda"),
            verbose=False,
            word_timestamps=True,
        )
    except Exception as exc:
        logger.exception("Whisper failed for %s: %s", video_path, exc)
        return None
    log_gpu_memory(logger, device, f"After Whisper transcription {video_path.name}")

    segments: List[Dict[str, Any]] = []
    flat_words: List[Dict[str, Any]] = []
    for segment in result.get("segments", []):
        segment_words: List[Dict[str, Any]] = []
        for raw_word in segment.get("words", []):
            word = clean_word(str(raw_word.get("word", "")))
            if not word:
                continue
            start = float(raw_word.get("start", segment.get("start", 0.0)))
            end = float(raw_word.get("end", start + 0.05))
            item = {
                "word": word,
                "start": start,
                "end": max(end, start + 0.05),
                "probability": float(raw_word.get("probability", 1.0) or 1.0),
            }
            segment_words.append(item)
            flat_words.append(item)
        segments.append(
            {
                "id": segment.get("id", len(segments)),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": str(segment.get("text", "")).strip(),
                "words": segment_words,
            }
        )

    payload = {
        "video_id": safe_stem(video_path),
        "video_path": str(video_path.resolve()),
        "transcript": str(result.get("text", "")).strip(),
        "segments": segments,
        "words": flat_words,
        "word_count": len(flat_words),
        "segment_count": len(segments),
        "language": result.get("language", args.language),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(paths["json"], payload)
    write_transcript_sidecars(paths, payload)
    return payload


def build_word_windows(words: Sequence[Word], args: argparse.Namespace) -> List[Tuple[float, float, List[Word]]]:
    if not words:
        return []

    clip_seconds = float(args.clip_seconds)
    stride = float(args.stride_seconds or args.clip_seconds)
    first_start = max(0.0, words[0].start - args.context_seconds)
    last_end = words[-1].end + args.context_seconds

    windows: List[Tuple[float, float, List[Word]]] = []
    start = first_start
    while start < last_end:
        end = start + clip_seconds
        selected = [
            word for word in words
            if start <= ((word.start + word.end) / 2.0) < end
        ]
        if len(selected) >= args.min_words:
            if args.max_words and len(selected) > args.max_words:
                selected = selected[:args.max_words]
            if clean_label(selected):
                windows.append((start, end, selected))
        start += stride
    return windows


def get_video_info(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": 0.0, "frames": 0.0, "duration": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"fps": fps, "frames": frames, "duration": duration}


def merge_intervals(
    intervals: Sequence[Tuple[float, float]],
    gap_seconds: float,
    padding_seconds: float,
    duration: float,
) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    padded = [
        (max(0.0, start - padding_seconds), min(duration, end + padding_seconds))
        for start, end in sorted(intervals)
    ]
    merged: List[Tuple[float, float]] = []
    for start, end in padded:
        if not merged or start - merged[-1][1] > gap_seconds:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def scan_visual_segments(
    video_path: Path,
    processor: MediaPipeProcessor,
    visual_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    visual_dir.mkdir(parents=True, exist_ok=True)
    cache_path = visual_dir / f"{safe_stem(video_path)}.json"
    if cache_path.exists() and not args.rescan_visual:
        cached = load_json(cache_path, None)
        if cached:
            return cached

    info = get_video_info(video_path)
    duration = float(info["duration"])
    fps = float(info["fps"])
    if duration <= 0 or fps <= 0:
        payload = {
            "video_id": safe_stem(video_path),
            "duration": duration,
            "fps": fps,
            "sample_fps": args.visual_scan_fps,
            "intervals": [],
            "face_samples": 0,
            "total_samples": 0,
            "error": "invalid_video_timing",
        }
        save_json(cache_path, payload)
        return payload

    sample_step = max(1.0 / max(args.visual_scan_fps, 0.01), args.visual_scan_min_step)
    sample_times_list = np.arange(0.0, duration, sample_step, dtype=np.float32)
    max_samples = int(args.visual_scan_max_samples)
    if max_samples > 0 and len(sample_times_list) > max_samples:
        sample_times_list = np.linspace(0.0, duration, max_samples, endpoint=False, dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    raw_intervals: List[Tuple[float, float]] = []
    face_samples = 0
    best_openness = 0.0

    logger.info(
        "Visual scan %s: %.1fs, %s samples",
        video_path.name,
        duration,
        len(sample_times_list),
    )
    scan_iter = make_progress(
        sample_times_list,
        desc=f"Visual scan {safe_stem(video_path)}",
        unit="sample",
        leave=False,
        disable=bool(args.no_progress),
    )
    for time_s in scan_iter:
        frame_index = min(int(info["frames"]) - 1, max(0, int(round(float(time_s) * fps))))
        frame = read_frame_at(cap, frame_index)
        if frame is None:
            continue
        observations = processor.extract_face_observations(frame)
        usable = [
            metrics for metrics, _landmarks in observations
            if metrics.size >= args.min_lip_size
        ]
        if usable:
            face_samples += 1
            best_openness = max(best_openness, max(float(metrics.openness) for metrics in usable))
            raw_intervals.append((float(time_s), min(duration, float(time_s) + sample_step)))
        if face_samples:
            scan_iter.set_postfix(face=f"{face_samples}/{len(sample_times_list)}", refresh=False)

    cap.release()
    intervals = merge_intervals(
        raw_intervals,
        gap_seconds=args.visual_merge_gap,
        padding_seconds=args.visual_padding,
        duration=duration,
    )
    payload = {
        "video_id": safe_stem(video_path),
        "duration": duration,
        "fps": fps,
        "frames": int(info["frames"]),
        "sample_fps": args.visual_scan_fps,
        "sample_step": sample_step,
        "total_samples": int(len(sample_times_list)),
        "face_samples": face_samples,
        "face_sample_rate": face_samples / max(1, int(len(sample_times_list))),
        "best_mouth_openness": best_openness,
        "intervals": [[round(start, 3), round(end, 3)] for start, end in intervals],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(cache_path, payload)
    return payload


def window_overlaps_intervals(
    start: float,
    end: float,
    intervals: Sequence[Sequence[float]],
    min_overlap: float,
) -> bool:
    midpoint = (start + end) / 2.0
    for interval_start, interval_end in intervals:
        if interval_start <= midpoint <= interval_end:
            return True
        overlap = min(end, float(interval_end)) - max(start, float(interval_start))
        if overlap >= min_overlap:
            return True
    return False


def filter_windows_by_visual(
    windows: Sequence[Tuple[float, float, List[Word]]],
    intervals: Sequence[Sequence[float]],
    args: argparse.Namespace,
) -> List[Tuple[float, float, List[Word]]]:
    if args.skip_visual_prefilter:
        return list(windows)
    if not intervals:
        return []
    return [
        window for window in windows
        if window_overlaps_intervals(window[0], window[1], intervals, args.min_visual_overlap)
    ]


def limit_windows_evenly(
    windows: Sequence[Tuple[float, float, List[Word]]],
    max_count: int,
) -> List[Tuple[float, float, List[Word]]]:
    if max_count <= 0 or len(windows) <= max_count:
        return list(windows)
    indices = np.linspace(0, len(windows) - 1, max_count).round().astype(int)
    deduped: List[int] = []
    seen = set()
    for idx in indices:
        int_idx = int(idx)
        if int_idx not in seen:
            deduped.append(int_idx)
            seen.add(int_idx)
    cursor = 0
    while len(deduped) < max_count and cursor < len(windows):
        if cursor not in seen:
            deduped.append(cursor)
            seen.add(cursor)
        cursor += 1
    return [windows[idx] for idx in sorted(deduped[:max_count])]


def make_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def sample_times(start: float, end: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.array([(start + end) / 2.0], dtype=np.float32)
    return np.linspace(start, end, count, endpoint=False, dtype=np.float32) + ((end - start) / count / 2.0)


def assign_tracks(
    detections_by_frame: Sequence[List[Detection]],
    max_track_distance: float,
) -> Dict[int, List[Detection]]:
    tracks: Dict[int, List[Detection]] = {}
    next_track_id = 0

    for detections in detections_by_frame:
        taken: set[int] = set()
        for detection in detections:
            center = np.array(detection.metrics.center, dtype=np.float32)
            best_track = -1
            best_distance = float("inf")

            for track_id, track_detections in tracks.items():
                if track_id in taken or not track_detections:
                    continue
                prev = np.array(track_detections[-1].metrics.center, dtype=np.float32)
                distance = float(np.linalg.norm(center - prev))
                if distance < best_distance and distance <= max_track_distance:
                    best_distance = distance
                    best_track = track_id

            if best_track < 0:
                best_track = next_track_id
                next_track_id += 1
                tracks[best_track] = []

            detection.track_id = best_track
            tracks[best_track].append(detection)
            taken.add(best_track)

    return tracks


def score_track(track: Sequence[Detection], total_candidates: int) -> float:
    openness = np.array([det.metrics.openness for det in track], dtype=np.float32)
    sizes = np.array([det.metrics.size for det in track], dtype=np.float32)
    if openness.size == 0:
        return 0.0
    motion = float(np.mean(np.abs(np.diff(openness)))) if openness.size > 1 else 0.0
    detection_rate = len(track) / max(1, total_candidates)
    return (
        float(np.mean(openness))
        + 1.75 * float(np.std(openness))
        + 0.75 * motion
        + 0.02 * float(np.mean(sizes))
        + 5.0 * detection_rate
    )


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def extract_speaker_clip(
    video_path: Path,
    start: float,
    end: float,
    processor: MediaPipeProcessor,
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"skip_reason": "video_open_failed"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or args.target_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None, {"skip_reason": "invalid_video_timing"}

    candidate_count = max(args.num_frames, int(math.ceil(args.num_frames * args.candidate_multiplier)))
    candidate_times = sample_times(max(0.0, start), max(start + 0.1, end), candidate_count)
    detections_by_frame: List[List[Detection]] = []
    frames_read = 0

    for time_s in candidate_times:
        frame_index = min(total_frames - 1, max(0, int(round(float(time_s) * fps))))
        frame = read_frame_at(cap, frame_index)
        if frame is None:
            detections_by_frame.append([])
            continue
        frames_read += 1
        observations = processor.extract_face_observations(frame)
        frame_detections = [
            Detection(
                frame_index=frame_index,
                time_s=float(time_s),
                frame=frame,
                metrics=metrics,
                landmarks=landmarks,
            )
            for metrics, landmarks in observations
            if metrics.size >= args.min_lip_size
        ]
        detections_by_frame.append(frame_detections)

    cap.release()

    tracks = assign_tracks(detections_by_frame, args.max_track_distance)
    if not tracks:
        return None, {
            "skip_reason": "no_face_detected",
            "candidate_frames": candidate_count,
            "frames_read": frames_read,
        }

    min_valid = int(math.ceil(args.num_frames * args.min_face_frame_rate))
    usable_tracks = {
        track_id: track for track_id, track in tracks.items()
        if len(track) >= min_valid
    }
    if not usable_tracks:
        return None, {
            "skip_reason": "not_enough_face_frames",
            "candidate_frames": candidate_count,
            "best_face_frames": max((len(track) for track in tracks.values()), default=0),
            "required_face_frames": min_valid,
        }

    best_id, best_track = max(
        usable_tracks.items(),
        key=lambda item: score_track(item[1], candidate_count),
    )
    best_track = sorted(best_track, key=lambda det: det.time_s)

    if len(best_track) < args.num_frames:
        return None, {
            "skip_reason": "not_enough_selected_speaker_frames",
            "selected_face_frames": len(best_track),
            "required_face_frames": args.num_frames,
        }

    chosen_indices = np.linspace(0, len(best_track) - 1, args.num_frames).round().astype(int)
    rois: List[np.ndarray] = []
    for idx in chosen_indices:
        det = best_track[int(idx)]
        roi = processor.extract_lip_roi_from_landmarks(det.frame, det.landmarks)
        if roi is None:
            return None, {"skip_reason": "roi_crop_failed"}
        rois.append(roi)

    openness = [float(det.metrics.openness) for det in best_track]
    clip = np.stack(rois).astype(np.uint8)
    return clip, {
        "selected_track_id": best_id,
        "candidate_frames": candidate_count,
        "selected_face_frames": len(best_track),
        "face_frame_rate": len(best_track) / max(1, candidate_count),
        "mean_mouth_openness": float(np.mean(openness)) if openness else 0.0,
        "mouth_motion": float(np.mean(np.abs(np.diff(openness)))) if len(openness) > 1 else 0.0,
        "track_count": len(tracks),
    }


def resolve_torch_device(args: argparse.Namespace, logger: logging.Logger) -> Tuple[Any, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing PyTorch. Install ROCm PyTorch before GPU preprocessing."
        ) from exc

    requested = args.device
    cuda_available = torch.cuda.is_available()
    resolved = "cuda" if requested == "auto" and cuda_available else requested
    if requested == "auto" and not cuda_available:
        resolved = "cpu"
    if resolved == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA/ROCm device was requested, but torch.cuda.is_available() is false. "
            "Whisper would run on CPU; fix ROCm PyTorch first or pass --device cpu."
        )

    logger.info("PyTorch: %s", torch.__version__)
    logger.info("ROCm/HIP: %s", getattr(torch.version, "hip", None))
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        logger.info("GPU: %s, %.2f GB total", torch.cuda.get_device_name(0), total_gb)
        if resolved == "cuda" and args.vram_limit_gb > 0:
            fraction = min(1.0, max(0.05, args.vram_limit_gb / total_gb))
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                logger.info(
                    "PyTorch GPU memory cap: %.2f GB (%.1f%%)",
                    total_gb * fraction,
                    fraction * 100.0,
                )
            except Exception as exc:
                logger.warning("Could not set PyTorch GPU memory cap: %s", exc)
    else:
        logger.warning("No ROCm/CUDA GPU visible to PyTorch.")

    return torch, resolved


def log_gpu_memory(logger: logging.Logger, device: str, label: str) -> None:
    if device != "cuda":
        return

    try:
        import torch

        if not torch.cuda.is_available():
            return
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
        logger.info(
            "%s GPU memory: allocated=%.2f GB reserved=%.2f GB peak=%.2f GB",
            label,
            allocated_gb,
            reserved_gb,
            max_allocated_gb,
        )
    except Exception as exc:
        logger.debug("Could not read GPU memory for %s: %s", label, exc)


def load_whisper(args: argparse.Namespace, logger: logging.Logger) -> Tuple[Any, str]:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Missing Whisper dependencies. Install them with: "
            "py -m pip install -U openai-whisper"
        ) from exc

    _torch, device = resolve_torch_device(args, logger)
    configure_whisper_ffmpeg(whisper, logger)
    logger.info("Loading Whisper model '%s' on %s", args.whisper_model, device)
    model = whisper.load_model(args.whisper_model, device=device)
    log_gpu_memory(logger, device, "After loading Whisper")
    return model, device


def configure_whisper_ffmpeg(whisper_module: Any, logger: logging.Logger) -> None:
    if shutil.which("ffmpeg"):
        logger.info("Using ffmpeg from PATH")
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Whisper needs ffmpeg to read audio, but ffmpeg is not on PATH. "
            "Install the bundled fallback with: py -m pip install imageio-ffmpeg"
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe or not Path(ffmpeg_exe).exists():
        raise RuntimeError(
            "Could not find a usable ffmpeg executable. Install one with: "
            "py -m pip install imageio-ffmpeg"
        )

    import whisper.audio as whisper_audio

    def load_audio_with_bundled_ffmpeg(file: str, sr: int = 16000) -> np.ndarray:
        cmd = [
            ffmpeg_exe,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg could not read audio from {file}: {stderr[:800]}") from exc
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    whisper_audio.load_audio = load_audio_with_bundled_ffmpeg
    logger.info("Using bundled imageio-ffmpeg executable: %s", ffmpeg_exe)


def import_runtime_dependencies() -> None:
    global MEDIAPIPE_AVAILABLE, MediaPipeProcessor, cv2, np, tqdm

    missing: List[str] = []
    try:
        import cv2 as cv2_module
    except ImportError:
        missing.append("opencv-python")
    else:
        cv2 = cv2_module

    try:
        import numpy as numpy_module
    except ImportError:
        missing.append("numpy")
    else:
        np = numpy_module

    try:
        from tqdm import tqdm as tqdm_func
    except ImportError:
        def tqdm_func(iterable, *args, **kwargs):
            return iterable
    else:
        tqdm = tqdm_func
    tqdm = tqdm_func

    if missing:
        raise RuntimeError(
            "Missing preprocessing dependencies: "
            + ", ".join(missing)
            + ". Install them with: py -m pip install -r requirements.txt"
        )

    try:
        from data_pipeline import MEDIAPIPE_AVAILABLE as mp_available
        from data_pipeline import MediaPipeProcessor as mp_processor
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the project MediaPipe processor. "
            "Install requirements with: py -m pip install -r requirements.txt. "
            f"Original import error: {exc}"
        ) from exc

    MEDIAPIPE_AVAILABLE = mp_available
    MediaPipeProcessor = mp_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess final_trainable_videos_heavy into trainer-ready word-aligned lip ROI clips."
    )
    parser.add_argument("--input", default="final_trainable_videos_heavy", help="Folder containing raw final videos.")
    parser.add_argument("--output", default="final_preprocessed_dataset", help="Output dataset folder.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model: tiny, base, small, medium, large.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Whisper device.")
    parser.add_argument("--vram-limit-gb", type=float, default=13.0, help="PyTorch ROCm/CUDA memory cap for Whisper; 0 disables.")
    parser.add_argument("--language", default="en", help="Whisper language code.")
    parser.add_argument("--clip-seconds", type=float, default=2.0, help="Length of each generated training clip.")
    parser.add_argument("--stride-seconds", type=float, default=1.0, help="Step between clip windows.")
    parser.add_argument("--context-seconds", type=float, default=0.05, help="Small timing padding around transcript words.")
    parser.add_argument("--num-frames", type=int, default=50, help="Frames saved per .npy clip.")
    parser.add_argument("--target-fps", type=float, default=25.0, help="Training FPS used for metadata.")
    parser.add_argument("--roi-size", type=int, default=96, help="Lip ROI width/height.")
    parser.add_argument("--max-faces", type=int, default=4, help="Faces to detect per frame.")
    parser.add_argument("--max-track-distance", type=float, default=140.0, help="Max mouth-center pixel jump for a face track.")
    parser.add_argument("--candidate-multiplier", type=float, default=2.0, help="How many candidate frames to inspect per output frame.")
    parser.add_argument("--min-face-frame-rate", type=float, default=0.90, help="Minimum selected-speaker face coverage before saving a clip.")
    parser.add_argument("--min-lip-size", type=float, default=8.0, help="Reject very tiny mouth detections.")
    parser.add_argument("--min-words", type=int, default=1, help="Minimum transcript words per clip.")
    parser.add_argument("--max-words", type=int, default=16, help="Maximum transcript words per clip label; 0 disables.")
    parser.add_argument(
        "--max-source-duration-seconds",
        type=float,
        default=0.0,
        help="Skip source videos longer than this before transcode/Whisper; 0 disables.",
    )
    parser.add_argument("--visual-scan-fps", type=float, default=0.25, help="Cheap pre-scan FPS used to find face-present regions.")
    parser.add_argument("--visual-scan-min-step", type=float, default=2.0, help="Minimum seconds between visual scan samples.")
    parser.add_argument("--visual-scan-max-samples", type=int, default=1500, help="Cap visual pre-scan samples per source video; 0 disables.")
    parser.add_argument("--visual-merge-gap", type=float, default=8.0, help="Merge face intervals separated by this many seconds.")
    parser.add_argument("--visual-padding", type=float, default=4.0, help="Expand face intervals by this many seconds.")
    parser.add_argument("--min-visual-overlap", type=float, default=0.25, help="Minimum overlap between a word window and face interval.")
    parser.add_argument(
        "--min-transcribe-face-sample-rate",
        type=float,
        default=0.0,
        help=(
            "Skip Whisper transcription when visual pre-scan face_sample_rate is below this value. "
            "Use 0 to disable. This is a speed/quality tradeoff for deadline runs."
        ),
    )
    parser.add_argument(
        "--min-transcribe-face-samples",
        type=int,
        default=0,
        help=(
            "Skip Whisper transcription when visual pre-scan has fewer face-positive samples than this. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument("--skip-visual-prefilter", action="store_true", help="Disable face-timeline prefiltering.")
    parser.add_argument("--rescan-visual", action="store_true", help="Ignore cached visual scans.")
    parser.add_argument("--max-clips-per-video", type=int, default=0, help="Debug cap after filtering; 0 means no cap.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos for testing.")
    parser.add_argument("--retranscribe", action="store_true", help="Ignore cached transcripts and run Whisper again.")
    parser.add_argument("--reprocess", action="store_true", help="Overwrite existing clip .npy/.txt files.")
    parser.add_argument(
        "--trust-existing-processed",
        action="store_true",
        help="Skip videos already marked processed even if current speed/config options differ.",
    )
    parser.add_argument(
        "--av1-transcode",
        choices=["auto", "always", "never"],
        default="auto",
        help="Transcode AV1 videos to cached H.264 copies before OpenCV/MediaPipe visual processing.",
    )
    parser.add_argument("--av1-transcode-crf", type=int, default=18, help="H.264 CRF for AV1 compatibility transcodes.")
    parser.add_argument(
        "--av1-transcode-preset",
        default="veryfast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="libx264 preset for compatibility transcodes.",
    )
    parser.add_argument(
        "--compat-transcode-codecs",
        default="av1",
        help="Comma-separated codecs to normalize to clean H.264 before OpenCV/MediaPipe, for example av1,h264.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    videos_dir = output_dir / "videos"
    labels_dir = output_dir / "labels"
    transcripts_dir = output_dir / "transcripts"
    visual_dir = output_dir / "visual_scans"
    transcode_dir = output_dir / "transcoded_h264"
    for path in (videos_dir, labels_dir, transcripts_dir, visual_dir, transcode_dir):
        path.mkdir(parents=True, exist_ok=True)
    configure_project_runtime_storage(output_dir)

    logger = configure_logging(output_dir)
    try:
        import_runtime_dependencies()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 2

    if not input_dir.exists():
        logger.error("Input folder does not exist: %s", input_dir)
        return 2
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe is not installed, so face/lip preprocessing cannot run.")
        return 2

    files = media_files(input_dir, args.limit)
    if not files:
        logger.error("No media files found in %s", input_dir)
        return 2

    logger.info("Input: %s", input_dir.resolve())
    logger.info("Output: %s", output_dir.resolve())
    logger.info("Videos to process: %s", len(files))

    try:
        whisper_model, whisper_device = load_whisper(args, logger)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 2

    processor = MediaPipeProcessor(
        roi_size=args.roi_size,
        stabilize=False,
        crop_mode="adaptive",
        max_faces=args.max_faces,
        align_mouth=True,
        profile_aware=True,
    )

    metadata_path = output_dir / "metadata.json"
    state_path = output_dir / "preprocessing_state.json"
    existing_metadata = load_json(metadata_path, [])
    metadata_by_id = {
        str(item.get("id")): item for item in existing_metadata
        if isinstance(item, dict) and item.get("id")
    }
    for sample_id, item in metadata_by_id.items():
        item.setdefault("source_video_id", source_id_from_sample_id(sample_id))
        item.setdefault("label", f"labels/{sample_id}.txt")
    state = load_json(state_path, {"processed_videos": {}, "failed_videos": {}})
    state.setdefault("processed_videos", {})
    state.setdefault("failed_videos", {})
    events_path = output_dir / "preprocess_events.jsonl"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    append_jsonl(events_path, {
        "event": "run_start",
        "run_id": run_id,
        "input": str(input_dir.resolve()),
        "output": str(output_dir.resolve()),
        "video_count": len(files),
        "args": vars(args),
    })

    stats = {
        "videos_seen": 0,
        "videos_skipped_completed": 0,
        "videos_transcribed": 0,
        "clips_created": 0,
        "clips_existing": 0,
        "clips_skipped": 0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_signature = preprocessing_config_signature(args)
    start_time = time.time()

    try:
        outer_progress = make_progress(
            files,
            desc="Source videos",
            unit="video",
            disable=bool(args.no_progress),
        )
        for video_path in outer_progress:
            stats["videos_seen"] += 1
            video_id = safe_stem(video_path)
            if hasattr(outer_progress, "set_postfix"):
                outer_progress.set_postfix(
                    video=video_path.name[:28],
                    done=stats["videos_skipped_completed"],
                    made=stats["clips_created"],
                    reused=stats["clips_existing"],
                    skipped=stats["clips_skipped"],
                    refresh=False,
                )
            logger.info("Processing source video: %s", video_path.name)
            video_started_at = time.time()
            processed_entry = state["processed_videos"].get(video_id)
            already_done = (
                processed_entry
                and not args.reprocess
                and (
                    processed_entry.get("config_signature") == config_signature
                    or args.trust_existing_processed
                )
            )
            if already_done:
                stats["videos_skipped_completed"] += 1
                if processed_entry.get("config_signature") == config_signature:
                    logger.info("Skipping %s: already completed for this preprocessing config", video_path.name)
                else:
                    logger.info("Skipping %s: already processed; trusting existing output despite changed speed/config options", video_path.name)
                append_jsonl(events_path, {
                    "event": "video_skipped_completed",
                    "run_id": run_id,
                    "video_id": video_id,
                    "source": str(video_path.resolve()),
                    "config_signature": config_signature,
                })
                continue
            append_jsonl(events_path, {
                "event": "video_start",
                "run_id": run_id,
                "video_id": video_id,
                "source": str(video_path.resolve()),
            })
            if args.max_source_duration_seconds > 0:
                source_info = get_video_info(video_path)
                source_duration = float(source_info.get("duration", 0.0) or 0.0)
                if source_duration > args.max_source_duration_seconds:
                    state["failed_videos"][video_id] = "source_duration_above_limit"
                    save_json(state_path, state)
                    logger.info(
                        "Skipping %s: duration %.1fs exceeds max-source-duration-seconds %.1fs",
                        video_path.name,
                        source_duration,
                        args.max_source_duration_seconds,
                    )
                    append_jsonl(events_path, {
                        "event": "video_failed",
                        "run_id": run_id,
                        "video_id": video_id,
                        "reason": "source_duration_above_limit",
                        "duration": source_duration,
                        "max_source_duration_seconds": args.max_source_duration_seconds,
                    })
                    continue

            visual_video_path = ensure_visual_compatible_video(video_path, transcode_dir, args, logger)
            visual_scan = scan_visual_segments(visual_video_path, processor, visual_dir, args, logger)
            visual_intervals = visual_scan.get("intervals", [])
            logger.info(
                "Visual scan result for %s: intervals=%s face_samples=%s/%s face_rate=%.3f",
                video_path.name,
                len(visual_intervals),
                visual_scan.get("face_samples", 0),
                visual_scan.get("total_samples", 0),
                float(visual_scan.get("face_sample_rate", 0.0) or 0.0),
            )
            append_jsonl(events_path, {
                "event": "visual_scan",
                "run_id": run_id,
                "video_id": video_id,
                "intervals": len(visual_intervals),
                "face_samples": visual_scan.get("face_samples", 0),
                "total_samples": visual_scan.get("total_samples", 0),
                "face_sample_rate": visual_scan.get("face_sample_rate", 0.0),
                "duration": visual_scan.get("duration", 0.0),
                "visual_source": str(visual_video_path.resolve()),
            })
            if not args.skip_visual_prefilter and not visual_intervals:
                state["failed_videos"][video_id] = "no_face_regions"
                save_json(state_path, state)
                logger.info("Skipping %s: no face regions found in visual pre-scan", video_path.name)
                continue
            face_sample_rate = float(visual_scan.get("face_sample_rate", 0.0) or 0.0)
            face_samples = int(visual_scan.get("face_samples", 0) or 0)
            if (
                not args.skip_visual_prefilter
                and (
                    (args.min_transcribe_face_sample_rate > 0 and face_sample_rate < args.min_transcribe_face_sample_rate)
                    or (args.min_transcribe_face_samples > 0 and face_samples < args.min_transcribe_face_samples)
                )
            ):
                state["failed_videos"][video_id] = "visual_scan_below_transcription_threshold"
                save_json(state_path, state)
                logger.info(
                    "Skipping %s before Whisper: visual face rate %.3f, face samples %s below threshold",
                    video_path.name,
                    face_sample_rate,
                    face_samples,
                )
                append_jsonl(events_path, {
                    "event": "video_failed",
                    "run_id": run_id,
                    "video_id": video_id,
                    "reason": "visual_scan_below_transcription_threshold",
                    "face_sample_rate": face_sample_rate,
                    "face_samples": face_samples,
                    "min_transcribe_face_sample_rate": args.min_transcribe_face_sample_rate,
                    "min_transcribe_face_samples": args.min_transcribe_face_samples,
                })
                continue

            transcript = load_or_create_transcript(
                video_path,
                transcripts_dir,
                whisper_model,
                whisper_device,
                args,
                logger,
            )
            if not transcript:
                state["failed_videos"][video_id] = "transcription_failed"
                stats["clips_skipped"] += 1
                save_json(state_path, state)
                append_jsonl(events_path, {
                    "event": "video_failed",
                    "run_id": run_id,
                    "video_id": video_id,
                    "reason": "transcription_failed",
                })
                continue
            stats["videos_transcribed"] += 1
            append_jsonl(events_path, {
                "event": "transcript_ready",
                "run_id": run_id,
                "video_id": video_id,
                "word_count": transcript.get("word_count", 0),
                "segment_count": transcript.get("segment_count", 0),
            })

            words = words_from_transcript(transcript)
            windows = build_word_windows(words, args)
            transcript_windows = len(windows)
            windows = filter_windows_by_visual(windows, visual_intervals, args)
            uncapped_windows = len(windows)
            windows = limit_windows_evenly(windows, args.max_clips_per_video)
            if not windows:
                state["failed_videos"][video_id] = "no_word_windows_after_visual_filter"
                save_json(state_path, state)
                logger.info(
                    "Skipping %s: 0 usable word windows after visual filter (transcript windows=%s, face intervals=%s)",
                    video_path.name,
                    transcript_windows,
                    len(visual_intervals),
                )
                append_jsonl(events_path, {
                    "event": "video_failed",
                    "run_id": run_id,
                    "video_id": video_id,
                    "reason": "no_word_windows_after_visual_filter",
                    "transcript_windows": transcript_windows,
                    "visual_intervals": len(visual_intervals),
                })
                continue
            logger.info(
                "Candidate windows for %s: %s/%s kept after visual prefilter; processing=%s",
                video_path.name,
                uncapped_windows,
                transcript_windows,
                len(windows),
            )
            append_jsonl(events_path, {
                "event": "windows_ready",
                "run_id": run_id,
                "video_id": video_id,
                "windows": len(windows),
                "uncapped_windows": uncapped_windows,
                "transcript_windows": transcript_windows,
                "visual_intervals": len(visual_intervals),
                "max_clips_per_video": args.max_clips_per_video,
            })

            created_for_video = 0
            existing_for_video = 0
            skipped_for_video = 0
            skip_reasons: Counter[str] = Counter()
            clip_iter = make_progress(
                list(enumerate(windows)),
                desc=f"Clips {video_id}",
                unit="clip",
                leave=False,
                disable=bool(args.no_progress),
            )
            for clip_index, (clip_start, clip_end, clip_words) in clip_iter:
                sample_id = f"{video_id}_clip_{clip_index:05d}"
                npy_path = videos_dir / f"{sample_id}.npy"
                label_path = labels_dir / f"{sample_id}.txt"
                label = clean_label(clip_words)
                if not label:
                    skipped_for_video += 1
                    skip_reasons["empty_label"] += 1
                    continue

                if npy_path.exists() and label_path.exists() and not args.reprocess:
                    existing_for_video += 1
                    stats["clips_existing"] += 1
                    if hasattr(clip_iter, "set_postfix"):
                        clip_iter.set_postfix(
                            created=created_for_video,
                            existing=existing_for_video,
                            skipped=skipped_for_video,
                            refresh=False,
                        )
                    if sample_id not in metadata_by_id:
                        metadata_by_id[sample_id] = {
                            "id": sample_id,
                            "video": f"videos/{sample_id}.npy",
                            "label": f"labels/{sample_id}.txt",
                            "text": label,
                            "word_count": len(label.split()),
                            "language": args.language,
                            "source_video": str(video_path.resolve()),
                            "processing_video": str(visual_video_path.resolve()),
                            "source_video_id": video_id,
                            "clip_start": clip_start,
                            "clip_end": clip_end,
                            "num_frames": args.num_frames,
                            "roi_size": args.roi_size,
                            "resumed_existing": True,
                        }
                    else:
                        metadata_by_id[sample_id]["source_video_id"] = video_id
                        metadata_by_id[sample_id]["source_video"] = str(video_path.resolve())
                        metadata_by_id[sample_id]["processing_video"] = str(visual_video_path.resolve())
                    continue

                clip_array, clip_info = extract_speaker_clip(
                    visual_video_path,
                    clip_start,
                    clip_end,
                    processor,
                    args,
                )
                if clip_array is None:
                    skipped_for_video += 1
                    skip_reasons[str(clip_info.get("skip_reason", "unknown"))] += 1
                    if hasattr(clip_iter, "set_postfix"):
                        clip_iter.set_postfix(
                            created=created_for_video,
                            existing=existing_for_video,
                            skipped=skipped_for_video,
                            refresh=False,
                        )
                    continue

                np.save(npy_path, clip_array)
                label_path.write_text(label + "\n", encoding="utf-8")

                metadata_by_id[sample_id] = {
                    "id": sample_id,
                    "video": f"videos/{sample_id}.npy",
                    "label": f"labels/{sample_id}.txt",
                    "text": label,
                    "word_count": len(label.split()),
                    "language": args.language,
                    "source_video": str(video_path.resolve()),
                    "processing_video": str(visual_video_path.resolve()),
                    "source_video_id": video_id,
                    "clip_start": round(float(clip_start), 3),
                    "clip_end": round(float(clip_end), 3),
                    "words": [
                        {
                            "word": word.word,
                            "start": round(word.start - clip_start, 3),
                            "end": round(word.end - clip_start, 3),
                            "absolute_start": round(word.start, 3),
                            "absolute_end": round(word.end, 3),
                            "probability": round(word.probability, 4),
                        }
                        for word in clip_words
                    ],
                    "num_frames": args.num_frames,
                    "target_fps": args.target_fps,
                    "roi_size": args.roi_size,
                    "speaker_selection": clip_info,
                }
                created_for_video += 1
                stats["clips_created"] += 1
                if hasattr(clip_iter, "set_postfix"):
                    clip_iter.set_postfix(
                        created=created_for_video,
                        existing=existing_for_video,
                        skipped=skipped_for_video,
                        refresh=False,
                    )

            stats["clips_skipped"] += skipped_for_video
            state["failed_videos"].pop(video_id, None)
            state["processed_videos"][video_id] = {
                "source": str(video_path.resolve()),
                "processing_source": str(visual_video_path.resolve()),
                "config_signature": config_signature,
                "windows": len(windows),
                "transcript_windows": transcript_windows,
                "visual_intervals": len(visual_intervals),
                "created": created_for_video,
                "existing": existing_for_video,
                "skipped": skipped_for_video,
                "skip_reasons": dict(skip_reasons),
                "elapsed_seconds": round(time.time() - video_started_at, 2),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_json(metadata_path, list(metadata_by_id.values()))
            save_json(state_path, state)
            logger.info(
                "Finished %s: created=%s existing=%s skipped=%s windows=%s",
                video_path.name,
                created_for_video,
                existing_for_video,
                skipped_for_video,
                len(windows),
            )
            if skip_reasons:
                logger.info("Skip summary for %s: %s", video_path.name, dict(skip_reasons))
            if hasattr(outer_progress, "set_postfix"):
                outer_progress.set_postfix(
                    made=stats["clips_created"],
                    reused=stats["clips_existing"],
                    skipped=stats["clips_skipped"],
                    refresh=False,
                )
            append_jsonl(events_path, {
                "event": "video_done",
                "run_id": run_id,
                "video_id": video_id,
                "created": created_for_video,
                "existing": existing_for_video,
                "skipped": skipped_for_video,
                "skip_reasons": dict(skip_reasons),
                "elapsed_seconds": round(time.time() - video_started_at, 2),
            })
    finally:
        if hasattr(processor, "close"):
            processor.close()

    summary = {
        **stats,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_samples": len(metadata_by_id),
        "output": str(output_dir.resolve()),
    }
    save_json(metadata_path, list(metadata_by_id.values()))
    save_json(output_dir / "preprocessing_summary.json", summary)

    logger.info("")
    logger.info("Done.")
    logger.info("Created clips this run: %s", stats["clips_created"])
    logger.info("Existing clips reused this run: %s", stats["clips_existing"])
    logger.info("Total dataset samples: %s", len(metadata_by_id))
    logger.info("Metadata: %s", metadata_path)
    logger.info("Log: %s", output_dir / "preprocess_final_dataset.log")
    logger.info("Events: %s", events_path)
    append_jsonl(events_path, {
        "event": "run_done",
        "run_id": run_id,
        **summary,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
