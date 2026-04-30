#!/usr/bin/env python3
"""
Fast American-English video sorter for the raw YouTube download folder.

The sorter samples short audio windows from each video, classifies English
accent with a SpeechBrain CommonAccent model, verifies English with Whisper
language detection, and places each video into:

    <output>/american
    <output>/non_american

By default it uses hard links, so sorting 169 GB of videos does not duplicate
the dataset and does not destroy the original raw folder. Use --action move
only after auditing a dry run or the generated results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_INPUT = Path("/media/agam/Local Disk/the_code/roman/majorproject/youtube_raw_downloads")
DEFAULT_OUTPUT = Path("/media/agam/Local Disk/the_code/roman/majorproject/american_english_filtered")
DEFAULT_ACCENT_MODEL = "Jzuluaga/accent-id-commonaccent_ecapa"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")
SAMPLE_RATE = 16_000
US_LABELS = {"us", "usa", "united_states", "united states", "american", "en_us"}


@dataclass
class ClipAudio:
    video: Path
    clip_index: int
    start: float
    samples: np.ndarray


@dataclass
class VideoAudio:
    video: Path
    duration: float
    clips: List[ClipAudio]
    rms_db: float
    active_ratio: float
    error: Optional[str] = None


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
    logger = logging.getLogger("american_english_filter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(output_dir / "american_english_filter.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested in {"cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm was requested, but torch.cuda.is_available() is false.")
        return "cuda:0"
    return requested


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


def require_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([binary, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as exc:
            raise RuntimeError(f"{binary} is required but was not found on PATH.") from exc


def video_signature(video: Path) -> Dict[str, Any]:
    stat = video.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def iter_videos(input_dir: Path, extensions: Sequence[str]) -> List[Path]:
    normalized = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions)
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in normalized)


def ffprobe_duration(video: Path, timeout: int = 20) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid duration: {proc.stdout!r}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"bad duration: {duration}")
    return duration


def clip_starts(duration: float, clips_per_video: int, clip_seconds: float) -> List[float]:
    if clips_per_video <= 1 or duration <= clip_seconds + 1.0:
        return [0.0]
    usable = max(0.0, duration - clip_seconds)
    fractions = np.linspace(0.18, 0.82, clips_per_video)
    starts = []
    for frac in fractions:
        start = float(usable * frac)
        if usable > 8.0:
            start = max(2.0, start)
        starts.append(min(start, usable))
    return starts


def extract_audio_clip(video: Path, start: float, seconds: float, timeout: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{seconds:.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or "ffmpeg failed")
    audio_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
    if audio_i16.size < SAMPLE_RATE // 2:
        raise RuntimeError("too little decoded audio")
    return audio_i16.astype(np.float32) / 32768.0


def audio_stats_for_array(audio: np.ndarray, active_db_threshold: float) -> Tuple[float, float]:
    if audio.size == 0:
        return -120.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-8))

    frame = max(1, int(SAMPLE_RATE * 0.05))
    usable = (audio.size // frame) * frame
    if usable == 0:
        return rms_db, 0.0
    framed = audio[:usable].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(np.square(framed), axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))
    return rms_db, float(np.mean(frame_db >= active_db_threshold))


def audio_stats(clips: Sequence[ClipAudio], active_db_threshold: float) -> Tuple[float, float]:
    if not clips:
        return -120.0, 0.0
    audio = np.concatenate([c.samples for c in clips])
    return audio_stats_for_array(audio, active_db_threshold)


def extract_video_audio(video: Path, args: argparse.Namespace) -> VideoAudio:
    try:
        duration = ffprobe_duration(video)
        candidate_clips: List[ClipAudio] = []
        timeout = max(20, int(args.clip_seconds * 8))
        candidate_count = max(args.clips_per_video, args.candidate_clips)
        for idx, start in enumerate(clip_starts(duration, candidate_count, args.clip_seconds)):
            samples = extract_audio_clip(video, start, args.clip_seconds, timeout=timeout)
            candidate_clips.append(ClipAudio(video=video, clip_index=idx, start=start, samples=samples))

        ranked: List[Tuple[float, float, ClipAudio]] = []
        for clip in candidate_clips:
            rms_db, active_ratio = audio_stats_for_array(clip.samples, args.active_db_threshold)
            ranked.append((active_ratio, rms_db, clip))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        clips = [clip for _active, _rms, clip in ranked[: args.clips_per_video]]
        clips.sort(key=lambda clip: clip.start)

        rms_db, active_ratio = audio_stats(clips, args.active_db_threshold)
        return VideoAudio(video=video, duration=duration, clips=clips, rms_db=rms_db, active_ratio=active_ratio)
    except Exception as exc:
        return VideoAudio(video=video, duration=0.0, clips=[], rms_db=-120.0, active_ratio=0.0, error=str(exc))


class SpeechBrainAccentClassifier:
    def __init__(self, model_name: str, device: str, cache_dir: Path, logger: logging.Logger) -> None:
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except Exception as exc:
            raise RuntimeError(
                "speechbrain is required for the accent model. Install it or use the project environment."
            ) from exc

        self.device = torch.device(device)
        savedir = cache_dir / model_name.replace("/", "__")
        logger.info("Loading accent model: %s", model_name)
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=str(savedir),
            run_opts={"device": device},
        )
        self.classifier.eval()
        label_encoder = getattr(self.classifier.hparams, "label_encoder", None)
        labels = getattr(label_encoder, "ind2lab", None)
        if isinstance(labels, dict):
            self.labels = {int(k): str(v) for k, v in labels.items()}
        else:
            raise RuntimeError("Accent model did not expose label_encoder.ind2lab.")
        self.us_indices = [idx for idx, label in self.labels.items() if normalize_label(label) in US_LABELS]
        if not self.us_indices:
            raise RuntimeError(f"Accent model labels do not contain a US class: {self.labels}")
        logger.info("Accent labels: %s", ", ".join(self.labels[i] for i in sorted(self.labels)))

    def classify(self, clips: Sequence[ClipAudio], batch_size: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for start in range(0, len(clips), batch_size):
            chunk = clips[start : start + batch_size]
            max_len = max(c.samples.size for c in chunk)
            wavs = torch.zeros((len(chunk), max_len), dtype=torch.float32)
            wav_lens = torch.zeros((len(chunk),), dtype=torch.float32)
            for row, clip in enumerate(chunk):
                samples = torch.from_numpy(clip.samples.astype(np.float32, copy=False))
                wavs[row, : samples.numel()] = samples
                wav_lens[row] = samples.numel() / max_len

            with torch.inference_mode():
                out_prob, _score, _index, _text_lab = self.classifier.classify_batch(
                    wavs.to(self.device),
                    wav_lens.to(self.device),
                )
                out_prob = out_prob.detach().float().cpu()
                # SpeechBrain classifiers commonly return per-label posterior
                # scores in [0, 1], not necessarily a distribution summing to 1.
                # Use those raw calibrated scores per row when present. Fall
                # back to softmax only for logit/log-probability style rows.
                rows = []
                for row in out_prob:
                    if row.min().item() >= 0.0 and row.max().item() <= 1.0:
                        rows.append(row)
                    else:
                        rows.append(torch.softmax(row, dim=-1))
                probs = torch.stack(rows, dim=0)

            for clip, prob in zip(chunk, probs):
                scores = {self.labels[i]: float(prob[i].item()) for i in range(prob.numel())}
                top_index = int(torch.argmax(prob).item())
                us_score = max(float(prob[i].item()) for i in self.us_indices)
                results.append(
                    {
                        "video": str(clip.video),
                        "clip_index": clip.clip_index,
                        "start": clip.start,
                        "top_label": self.labels[top_index],
                        "top_score": float(prob[top_index].item()),
                        "us_score": us_score,
                        "scores": scores,
                    }
                )
        return results


class WhisperEnglishDetector:
    def __init__(self, model_name: str, device: str, logger: logging.Logger) -> None:
        import whisper

        self.whisper = whisper
        logger.info("Loading Whisper language detector: %s", model_name)
        self.model = whisper.load_model(model_name, device=device)
        self.model.eval()
        self.device = device

    def english_probability(self, clips: Sequence[ClipAudio]) -> float:
        if not clips:
            return 0.0
        audio = np.concatenate([clip.samples for clip in clips]).astype(np.float32, copy=False)
        audio_tensor = torch.from_numpy(audio)
        audio_tensor = self.whisper.pad_or_trim(audio_tensor)
        mel = self.whisper.log_mel_spectrogram(audio_tensor).to(self.model.device)
        with torch.inference_mode():
            _tokens, probs = self.model.detect_language(mel)
        return float(probs.get("en", 0.0))


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_")


def aggregate_accent(
    clip_results: Sequence[Dict[str, Any]],
    labels: Dict[int, str],
) -> Dict[str, Any]:
    if not clip_results:
        return {
            "us_score": 0.0,
            "us_margin": 0.0,
            "us_vote_fraction": 0.0,
            "top_label": "none",
            "label_scores": {},
            "clip_results": [],
        }

    label_scores: Dict[str, float] = {}
    for label in labels.values():
        values = [float(result["scores"].get(label, 0.0)) for result in clip_results]
        label_scores[label] = float(np.mean(values)) if values else 0.0

    us_score = max(score for label, score in label_scores.items() if normalize_label(label) in US_LABELS)
    non_us_scores = [score for label, score in label_scores.items() if normalize_label(label) not in US_LABELS]
    next_best = max(non_us_scores) if non_us_scores else 0.0
    top_label = max(label_scores.items(), key=lambda item: item[1])[0]
    us_votes = sum(1 for result in clip_results if normalize_label(str(result["top_label"])) in US_LABELS)

    return {
        "us_score": us_score,
        "us_margin": us_score - next_best,
        "us_vote_fraction": us_votes / max(1, len(clip_results)),
        "top_label": top_label,
        "label_scores": dict(sorted(label_scores.items(), key=lambda item: item[1], reverse=True)),
        "clip_results": list(clip_results),
    }


def decide(video_audio: VideoAudio, accent: Dict[str, Any], english_prob: float, args: argparse.Namespace) -> Tuple[bool, str]:
    if video_audio.error:
        return False, "decode_error"
    if not video_audio.clips:
        return False, "no_audio"
    if video_audio.rms_db < args.min_rms_db:
        return False, "too_quiet"
    if video_audio.active_ratio < args.min_active_ratio:
        return False, "not_enough_active_speech"
    if english_prob < args.min_english_prob:
        return False, "non_english_or_uncertain_language"
    if accent["us_score"] < args.min_us_score:
        return False, "low_us_accent_score"
    if accent["us_margin"] < args.min_us_margin:
        return False, "ambiguous_us_accent_margin"
    if accent["us_vote_fraction"] < args.min_us_vote_fraction:
        return False, "not_enough_us_clip_votes"
    return True, "american_english"


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
    counts = {"american": 0, "non_american": 0, "errors": 0}
    for entry in state.data.get("files", {}).values():
        result = entry.get("result", {})
        bucket = result.get("bucket")
        if bucket in counts:
            counts[bucket] += 1
        if result.get("reason") == "decode_error":
            counts["errors"] += 1
    return counts


def process_batch(
    videos: Sequence[Path],
    args: argparse.Namespace,
    accent_model: SpeechBrainAccentClassifier,
    english_detector: Optional[WhisperEnglishDetector],
    output_dir: Path,
    logger: logging.Logger,
) -> List[Tuple[Path, Dict[str, Any]]]:
    extracted: List[VideoAudio] = []
    with ThreadPoolExecutor(max_workers=args.ffmpeg_workers) as pool:
        futures = {pool.submit(extract_video_audio, video, args): video for video in videos}
        for future in as_completed(futures):
            extracted.append(future.result())

    clips = [clip for item in extracted for clip in item.clips]
    accent_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if clips:
        for result in accent_model.classify(clips, args.accent_batch_size):
            accent_by_key[(result["video"], int(result["clip_index"]))] = result

    results: List[Tuple[Path, Dict[str, Any]]] = []
    for item in sorted(extracted, key=lambda x: str(x.video)):
        clip_results = [
            accent_by_key[(str(clip.video), clip.clip_index)]
            for clip in item.clips
            if (str(clip.video), clip.clip_index) in accent_by_key
        ]
        accent = aggregate_accent(clip_results, accent_model.labels)
        english_prob = 1.0
        if english_detector is not None and item.clips:
            try:
                english_prob = english_detector.english_probability(item.clips)
            except Exception as exc:
                logger.warning("Whisper language detection failed for %s: %s", item.video.name, exc)
                english_prob = 0.0

        is_american, reason = decide(item, accent, english_prob, args)
        bucket = "american" if is_american else "non_american"
        dest_path: Optional[Path] = None
        signature = video_signature(item.video)
        if not args.dry_run:
            dest_path = place_file(item.video, output_dir / bucket, args.action)

        result = {
            "video": str(item.video),
            "file_name": item.video.name,
            "signature": signature,
            "bucket": bucket,
            "reason": reason,
            "destination": str(dest_path) if dest_path else None,
            "action": "dry_run" if args.dry_run else args.action,
            "duration": item.duration,
            "rms_db": item.rms_db,
            "active_ratio": item.active_ratio,
            "english_probability": english_prob,
            "accent": {
                "top_label": accent["top_label"],
                "us_score": accent["us_score"],
                "us_margin": accent["us_margin"],
                "us_vote_fraction": accent["us_vote_fraction"],
                "label_scores": accent["label_scores"],
            },
            "clip_count": len(item.clips),
            "error": item.error,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
        results.append((item.video, result))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort raw videos into American-English and non-American folders using ROCm-capable models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Folder containing raw scraped videos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder to create.")
    parser.add_argument(
        "--action",
        choices=("hardlink", "symlink", "copy", "move"),
        default="hardlink",
        help="How to place files in the output buckets. hardlink is fast and non-duplicating.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--vram-limit-gb", type=float, default=13.0, help="PyTorch per-process GPU memory cap.")
    parser.add_argument("--accent-model", default=DEFAULT_ACCENT_MODEL, help="SpeechBrain accent model.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model for language detection.")
    parser.add_argument(
        "--language-check",
        choices=("whisper", "off"),
        default="whisper",
        help="Use Whisper language detection or skip language detection.",
    )
    parser.add_argument("--clips-per-video", type=int, default=3, help="Number of audio windows sampled per video.")
    parser.add_argument(
        "--candidate-clips",
        type=int,
        default=5,
        help="Candidate windows decoded per video before selecting the most speech-like clips.",
    )
    parser.add_argument("--clip-seconds", type=float, default=6.0, help="Seconds per sampled audio window.")
    parser.add_argument("--batch-videos", type=int, default=16, help="Videos to decode/classify per batch.")
    parser.add_argument("--ffmpeg-workers", type=int, default=6, help="Parallel ffmpeg audio decoders.")
    parser.add_argument("--accent-batch-size", type=int, default=48, help="Audio clips per GPU accent batch.")
    parser.add_argument("--min-us-score", type=float, default=0.62, help="Minimum averaged US accent score.")
    parser.add_argument("--min-us-margin", type=float, default=0.10, help="US score margin over next accent.")
    parser.add_argument("--min-us-vote-fraction", type=float, default=0.50, help="Fraction of clips whose top accent is US.")
    parser.add_argument("--min-english-prob", type=float, default=0.80, help="Minimum Whisper English probability.")
    parser.add_argument("--min-rms-db", type=float, default=-45.0, help="Reject audio quieter than this dBFS RMS.")
    parser.add_argument(
        "--min-active-ratio",
        type=float,
        default=0.08,
        help="Reject clips with too little active audio above --active-db-threshold.",
    )
    parser.add_argument("--active-db-threshold", type=float, default=-45.0, help="Frame threshold for active audio.")
    parser.add_argument("--limit", type=int, default=0, help="Process only this many unprocessed videos.")
    parser.add_argument("--reprocess", action="store_true", help="Ignore prior state and classify again.")
    parser.add_argument("--dry-run", action="store_true", help="Classify without linking/copying/moving files.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(VIDEO_EXTENSIONS),
        help="Video extensions to scan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clips_per_video < 1 or args.candidate_clips < 1:
        raise ValueError("--clips-per-video and --candidate-clips must both be at least 1.")
    if args.candidate_clips < args.clips_per_video:
        args.candidate_clips = args.clips_per_video

    output_dir = args.output.expanduser().resolve()
    logger = setup_logging(output_dir)
    start_time = time.time()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.exists():
        logger.error("Input folder does not exist: %s", input_dir)
        return 2
    require_ffmpeg()

    (output_dir / "american").mkdir(parents=True, exist_ok=True)
    (output_dir / "non_american").mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cap_vram(device, args.vram_limit_gb, logger)
    logger.info("Input: %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Action: %s%s", args.action, " (dry run)" if args.dry_run else "")
    logger.info(
        "Thresholds: us_score>=%.2f, margin>=%.2f, us_votes>=%.2f, english>=%.2f",
        args.min_us_score,
        args.min_us_margin,
        args.min_us_vote_fraction,
        args.min_english_prob,
    )

    state = JsonState(output_dir / "state.json")
    result_log = output_dir / ("dry_run_results.jsonl" if args.dry_run else "results.jsonl")

    videos = iter_videos(input_dir, args.extensions)
    todo: List[Path] = []
    for video in videos:
        sig = video_signature(video)
        if not state.is_done(video, sig, args.reprocess):
            todo.append(video)
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    logger.info("Found %d videos, %d pending.", len(videos), len(todo))
    if not todo:
        logger.info("Nothing to do.")
        return 0

    cache_dir = output_dir / "model_cache"
    accent_model = SpeechBrainAccentClassifier(args.accent_model, device, cache_dir, logger)
    english_detector = None
    if args.language_check == "whisper":
        english_detector = WhisperEnglishDetector(args.whisper_model, device, logger)

    counts = {"american": 0, "non_american": 0}
    progress = tqdm(total=len(todo), unit="video", desc="Filtering")
    try:
        for batch_start in range(0, len(todo), args.batch_videos):
            batch = todo[batch_start : batch_start + args.batch_videos]
            batch_results = process_batch(batch, args, accent_model, english_detector, output_dir, logger)
            for video, result in batch_results:
                counts[result["bucket"]] = counts.get(result["bucket"], 0) + 1
                append_jsonl(result_log, result)
                if not args.dry_run:
                    state.put(video, result["signature"], result)
            if not args.dry_run:
                state.save()
            progress.update(len(batch_results))
            progress.set_postfix(american=counts.get("american", 0), non_american=counts.get("non_american", 0))
    finally:
        progress.close()

    final_counts = counts if args.dry_run else summarize_state(state)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "action": args.action,
        "device": device,
        "accent_model": args.accent_model,
        "whisper_model": args.whisper_model if args.language_check == "whisper" else None,
        "total_found": len(videos),
        "processed_this_run": sum(counts.values()),
        "counts": final_counts,
        "thresholds": {
            "min_us_score": args.min_us_score,
            "min_us_margin": args.min_us_margin,
            "min_us_vote_fraction": args.min_us_vote_fraction,
            "min_english_prob": args.min_english_prob,
            "min_rms_db": args.min_rms_db,
            "min_active_ratio": args.min_active_ratio,
        },
        "seconds": round(time.time() - start_time, 2),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = output_dir / ("dry_run_summary.json" if args.dry_run else "summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info("Done. American: %d, non-American: %d", final_counts.get("american", 0), final_counts.get("non_american", 0))
    logger.info("Results: %s", result_log)
    logger.info("Summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
