#!/usr/bin/env python3
"""
Fast, Windows-safe YouTube CC scraper for Swin-VALLR.

This scraper intentionally avoids the YouTube Data API. It uses yt-dlp against
YouTube's Creative Commons search filter, then verifies every candidate with
full metadata before downloading:

- license must be Creative Commons
- American-English heuristic score must pass the configured threshold
- duration must be long enough to be useful for lip-reading

Speed comes from bounded parallel metadata checks and a small download pool.
Safety comes from jittered pacing, conservative defaults, and cool-downs when
yt-dlp reports rate-limit, bot-check, or block-like errors.

Windows:
    python scrape_youtube_no_api.py status
    python scrape_youtube_no_api.py --test
    python scrape_youtube_no_api.py run --target 2000
    python scrape_youtube_no_api.py continue

Linux/macOS:
    python3 scrape_youtube_no_api.py status
    python3 scrape_youtube_no_api.py --test
    python3 scrape_youtube_no_api.py run --target 2000
    python3 scrape_youtube_no_api.py continue
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote


TARGET = 2000
OUTPUT_DIR = Path("youtube_raw_downloads")
STATE_FILE = Path("no_api_state.json")
LOG_FILE = Path("no_api_scraper.log")
EVENT_LOG_FILE = Path("no_api_scraper_events.jsonl")
STATE_SCHEMA_VERSION = 2

# YouTube CC filter URL parameter.
# sp=EgIwAQ%3D%3D filters search results to Creative Commons videos.
CC_FILTER = "EgIwAQ%3D%3D"

# Conservative defaults. Increase from the CLI only if your connection and
# YouTube responses stay stable.
RESULTS_PER_QUERY = 30
DEFAULT_METADATA_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = 2
DEFAULT_MAX_DOWNLOADS_PER_QUERY = 8

AMERICAN_THRESHOLD = 0
MIN_DURATION_SECONDS = 60
MIN_VIDEO_BYTES = 5_000_000
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm"}

DELAY_BETWEEN_QUERIES = 10.0
DELAY_BATCH = 120.0
METADATA_DELAY_RANGE = (0.35, 1.4)
DOWNLOAD_DELAY_RANGE = (2.0, 5.0)
RATE_LIMIT_COOLDOWN = 15 * 60

SEARCH_TIMEOUT = 75
METADATA_TIMEOUT = 45
DOWNLOAD_TIMEOUT = 900


QUERIES = [
    # American-specific searches.
    "american speech", "american english speaking", "american accent",
    "US college lecture", "american university lecture",
    "american history lecture", "american politics speech",
    "american business presentation", "american motivational speech",

    # TED and public talks.
    "TED talk", "TEDx talk", "TED conference", "TED ideas",

    # Known educational channels and institutions.
    "khan academy", "crash course", "MIT lecture", "Harvard lecture",
    "Stanford lecture", "Yale lecture", "PBS documentary",

    # Professional content.
    "sales training", "marketing presentation", "startup pitch",
    "silicon valley talk", "wall street presentation",
    "american corporate training", "american leadership",

    # American news and media.
    "CNN interview", "NBC news", "ABC news report",
    "american press conference", "white house briefing",

    # Culture and direct-to-camera content.
    "NFL speech", "NBA interview", "american sports commentary",
    "american cooking show", "american travel vlog",

    # Generic but still CC-filtered.
    "lecture", "tutorial", "how to", "documentary",
    "presentation", "training video", "explainer",
    "science lecture", "technology talk", "history documentary",
    "health education", "finance explained",
]


BLOCK_PATTERNS = (
    "too many requests",
    "http error 429",
    "429:",
    "captcha",
    "sign in to confirm",
    "unusual traffic",
    "temporarily blocked",
    "not a bot",
    "confirm you are not a bot",
    "precondition check failed",
)


NON_AMERICAN_COUNTRIES = {
    "GB", "UK", "IN", "PK", "BD", "LK", "NP", "AU", "NZ", "CA",
    "IE", "ZA", "NG", "KE", "GH", "PH", "SG", "MY",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("no_api_scraper")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


log = setup_logger()


@dataclass
class ScraperConfig:
    target: int = TARGET
    output_dir: Path = OUTPUT_DIR
    state_file: Path = STATE_FILE
    metadata_workers: int = DEFAULT_METADATA_WORKERS
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS
    max_downloads_per_query: int = DEFAULT_MAX_DOWNLOADS_PER_QUERY
    results_per_query: int = RESULTS_PER_QUERY
    american_threshold: int = AMERICAN_THRESHOLD
    min_duration: int = MIN_DURATION_SECONDS
    query_delay: float = DELAY_BETWEEN_QUERIES
    batch_delay: float = DELAY_BATCH
    metadata_delay_min: float = METADATA_DELAY_RANGE[0]
    metadata_delay_max: float = METADATA_DELAY_RANGE[1]
    download_delay_min: float = DOWNLOAD_DELAY_RANGE[0]
    download_delay_max: float = DOWNLOAD_DELAY_RANGE[1]
    rate_limit_cooldown: float = RATE_LIMIT_COOLDOWN
    search_timeout: int = SEARCH_TIMEOUT
    metadata_timeout: int = METADATA_TIMEOUT
    download_timeout: int = DOWNLOAD_TIMEOUT
    yt_dlp_bin: Optional[str] = None
    once: bool = False
    event_log_file: Path = EVENT_LOG_FILE


@dataclass
class MetadataResult:
    video_id: str
    meta: Optional[Dict[str, Any]]
    error: str = ""
    rate_limited: bool = False


@dataclass
class DownloadResult:
    video_id: str
    success: bool
    size_mb: float = 0.0
    error: str = ""
    rate_limited: bool = False


_YTDLP_CMD_LOCK = threading.Lock()
_YTDLP_CMD: Optional[List[str]] = None


def _clean_state_list(values: Iterable[Any]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for value in values or []:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"


def config_to_dict(config: ScraperConfig) -> Dict[str, Any]:
    data = asdict(config)
    for key in ("output_dir", "state_file", "event_log_file"):
        data[key] = str(data[key])
    return data


def config_from_saved(data: Dict[str, Any]) -> ScraperConfig:
    allowed = set(ScraperConfig.__dataclass_fields__.keys())
    cleaned = {key: value for key, value in (data or {}).items() if key in allowed}
    for key in ("output_dir", "state_file", "event_log_file"):
        if key in cleaned:
            cleaned[key] = Path(cleaned[key])
    return ScraperConfig(**cleaned)


def merge_resume_config(saved: Dict[str, Any], args: argparse.Namespace) -> ScraperConfig:
    """Load the previous run config, allowing only explicit path overrides."""
    if not saved:
        log.warning(
            "No saved scraper command was found in this state file. "
            "Continuing with the current CLI/default settings."
        )
        config = config_from_args(args)
    else:
        config = config_from_saved(saved)
    config.state_file = args.state_file
    if args.yt_dlp_bin:
        config.yt_dlp_bin = args.yt_dlp_bin
    return config


def load_state(path: Path = STATE_FILE) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}

    state.setdefault("schema_version", STATE_SCHEMA_VERSION)
    state.setdefault("run_id", None)
    state.setdefault("run_status", "idle")
    state.setdefault("current_phase", "idle")
    state.setdefault("last_command_line", "")
    state.setdefault("resume_config", {})
    state.setdefault("total_american", 0)
    state.setdefault("total_checked", 0)
    state.setdefault("cc_found", 0)
    state.setdefault("non_cc", 0)
    state.setdefault("metadata_failures", 0)
    state.setdefault("download_failures", 0)
    state.setdefault("downloaded_ids", [])
    state.setdefault("checked_ids", [])
    state.setdefault("pending_download_ids", [])
    state.setdefault("failed_ids", [])
    state.setdefault("in_progress_downloads", {})
    state.setdefault("current_download_batch", [])
    state.setdefault("current_query", None)
    state.setdefault("current_query_index", None)
    state.setdefault("current_pass_number", None)
    state.setdefault("last_checked_video", None)
    state.setdefault("last_download_started", None)
    state.setdefault("last_download_finished", None)
    state.setdefault("last_error", None)
    state.setdefault("errors", [])
    state.setdefault("query_index", 0)
    state.setdefault("pass_number", 0)
    state.setdefault("start_time", datetime.now().isoformat())
    state.setdefault("last_update", datetime.now().isoformat())
    state.setdefault("last_rate_limit", None)

    state["downloaded_ids"] = _clean_state_list(state.get("downloaded_ids", []))
    state["checked_ids"] = _clean_state_list(state.get("checked_ids", []))
    state["pending_download_ids"] = _clean_state_list(state.get("pending_download_ids", []))
    state["failed_ids"] = _clean_state_list(state.get("failed_ids", []))
    if not isinstance(state.get("in_progress_downloads"), dict):
        state["in_progress_downloads"] = {}
    if not isinstance(state.get("errors"), list):
        state["errors"] = []

    # Keep counters consistent with the durable lists.
    state["total_american"] = max(int(state.get("total_american", 0)), len(state["downloaded_ids"]))
    state["total_checked"] = max(int(state.get("total_checked", 0)), len(state["checked_ids"]))
    state["schema_version"] = STATE_SCHEMA_VERSION
    return state


def save_state(state: dict, path: Path = STATE_FILE) -> None:
    state["last_update"] = datetime.now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(path)


def log_event(event: str, state: Optional[dict] = None, **fields: Any) -> None:
    """Append a structured JSON event for crash/restart analysis."""
    event_path = EVENT_LOG_FILE
    if state:
        saved_config = state.get("resume_config") or {}
        if saved_config.get("event_log_file"):
            event_path = Path(saved_config["event_log_file"])
    payload = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "run_id": state.get("run_id") if state else None,
        **fields,
    }
    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        log.debug("Failed to write event log", exc_info=True)


def remember_error(
    state: dict,
    phase: str,
    message: str,
    video_id: Optional[str] = None,
    rate_limited: bool = False,
) -> None:
    record = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "video_id": video_id,
        "message": str(message)[:1000],
        "rate_limited": bool(rate_limited),
    }
    state["last_error"] = record
    errors = state.setdefault("errors", [])
    errors.append(record)
    if len(errors) > 250:
        del errors[:-250]
    log_event("error", state, **record)


def set_phase(state: dict, phase: str, **fields: Any) -> None:
    state["current_phase"] = phase
    for key, value in fields.items():
        state[key] = value
    log_event("phase", state, phase=phase, **fields)


def begin_or_resume_run(state: dict, config: ScraperConfig, continue_mode: bool) -> None:
    if not continue_mode or not state.get("run_id"):
        state["run_id"] = _new_run_id()
        state["run_started_at"] = datetime.now().isoformat()
    state["run_status"] = "running"
    state["resume_config"] = config_to_dict(config)
    state["last_command_line"] = " ".join(sys.argv)
    state["last_pid"] = os.getpid()
    state["last_host_os"] = os.name
    set_phase(state, "starting")
    log_event(
        "run_start",
        state,
        continue_mode=continue_mode,
        command=state["last_command_line"],
        config=state["resume_config"],
    )


def complete_run(state: dict, status: str = "completed") -> None:
    state["run_status"] = status
    state["run_finished_at"] = datetime.now().isoformat()
    set_phase(state, status)
    log_event("run_end", state, status=status)


def _run_command(
    cmd: Sequence[str],
    timeout: int,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _candidate_yt_dlp_commands(explicit_bin: Optional[str]) -> List[List[str]]:
    candidates: List[List[str]] = []
    if explicit_bin:
        candidates.append([explicit_bin])

    env_bin = os.getenv("YT_DLP_BIN")
    if env_bin:
        candidates.append([env_bin])

    # This works well on Windows when yt-dlp is installed into the active venv
    # but the console script is not on PATH.
    candidates.append([sys.executable, "-m", "yt_dlp"])

    path_bin = shutil.which("yt-dlp")
    if path_bin:
        candidates.append([path_bin])

    # Windows launcher fallback.
    if os.name == "nt" and shutil.which("py"):
        candidates.append(["py", "-m", "yt_dlp"])

    deduped: List[List[str]] = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def yt_dlp_cmd(explicit_bin: Optional[str] = None) -> List[str]:
    global _YTDLP_CMD
    with _YTDLP_CMD_LOCK:
        if _YTDLP_CMD is not None:
            return list(_YTDLP_CMD)

        for candidate in _candidate_yt_dlp_commands(explicit_bin):
            try:
                result = _run_command(candidate + ["--version"], timeout=20)
            except Exception:
                continue
            if result.returncode == 0 and result.stdout.strip():
                _YTDLP_CMD = candidate
                log.info("Using yt-dlp command: %s", " ".join(candidate))
                return list(candidate)

    raise RuntimeError(
        "yt-dlp was not found. Install it with: python -m pip install yt-dlp"
    )


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def preflight(config: ScraperConfig) -> None:
    yt_dlp_cmd(config.yt_dlp_bin)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not has_ffmpeg():
        log.warning(
            "ffmpeg is not on PATH. yt-dlp can still download some combined MP4s, "
            "but merging best video+audio may fail. Install ffmpeg for best results."
        )


def jitter_sleep(min_seconds: float, max_seconds: float, reason: str = "") -> None:
    min_seconds = max(0.0, min_seconds)
    max_seconds = max(min_seconds, max_seconds)
    delay = random.uniform(min_seconds, max_seconds)
    if reason and delay >= 1.0:
        log.info("Sleeping %.1fs (%s)", delay, reason)
    time.sleep(delay)


def cooldown(seconds: float, reason: str) -> None:
    seconds = max(0.0, seconds)
    if seconds <= 0:
        return
    log.warning("Cooling down for %.1f minutes: %s", seconds / 60.0, reason)
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        sleep_for = min(60.0, remaining)
        time.sleep(sleep_for)
        remaining = end - time.monotonic()
        if remaining > 0:
            log.info("Cooldown remaining: %.1f minutes", remaining / 60.0)


def looks_rate_limited(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in BLOCK_PATTERNS)


def cc_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote(query)}&sp={CC_FILTER}"


def search_cc_video_ids(query: str, n: int, config: ScraperConfig) -> Tuple[List[str], bool]:
    url = cc_search_url(query)
    cmd = yt_dlp_cmd(config.yt_dlp_bin) + [
        url,
        "--get-id",
        "--no-warnings",
        "--ignore-errors",
        "--flat-playlist",
        "--playlist-end",
        str(max(1, min(n, 100))),
    ]

    try:
        result = _run_command(cmd, timeout=config.search_timeout)
    except subprocess.TimeoutExpired:
        log.warning("Search timed out for query: %s", query)
        return [], False
    except Exception as exc:
        log.warning("Search error for query %r: %s", query, exc)
        return [], False

    combined = f"{result.stdout}\n{result.stderr}"
    if looks_rate_limited(combined):
        log.warning("Search hit a rate-limit/block signal for query: %s", query)
        return [], True

    ids = []
    seen = set()
    for line in result.stdout.splitlines():
        video_id = line.strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        ids.append(video_id)

    log.info("CC search returned %d video IDs", len(ids))
    return ids, False


def get_video_metadata(video_id: str, config: ScraperConfig) -> MetadataResult:
    jitter_sleep(config.metadata_delay_min, config.metadata_delay_max)
    cmd = yt_dlp_cmd(config.yt_dlp_bin) + [
        f"https://youtube.com/watch?v={video_id}",
        "--skip-download",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
    ]

    last_error = ""
    for attempt in range(1, 3):
        try:
            result = _run_command(cmd, timeout=config.metadata_timeout)
        except subprocess.TimeoutExpired:
            last_error = "metadata timeout"
        except Exception as exc:
            last_error = str(exc)
        else:
            combined = f"{result.stdout}\n{result.stderr}"
            if looks_rate_limited(combined):
                return MetadataResult(video_id, None, "rate limited", rate_limited=True)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return MetadataResult(video_id, json.loads(result.stdout.strip()))
                except json.JSONDecodeError as exc:
                    last_error = f"json parse error: {exc}"
            else:
                last_error = (result.stderr or "no metadata").strip()[:300]

        if attempt < 2:
            jitter_sleep(2.0, 5.0, "metadata retry")

    return MetadataResult(video_id, None, last_error)


def fetch_metadata_batch(
    video_ids: List[str],
    config: ScraperConfig
) -> List[MetadataResult]:
    if not video_ids:
        return []

    workers = max(1, min(config.metadata_workers, len(video_ids)))
    log.info("Fetching metadata for %d videos with %d workers", len(video_ids), workers)
    results: List[MetadataResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(get_video_metadata, video_id, config): video_id
            for video_id in video_ids
        }
        for future in concurrent.futures.as_completed(future_map):
            video_id = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(MetadataResult(video_id, None, str(exc)))

    order = {video_id: idx for idx, video_id in enumerate(video_ids)}
    results.sort(key=lambda item: order.get(item.video_id, 0))
    return results


def is_cc(meta: Dict[str, Any]) -> bool:
    license_text = (meta.get("license") or "").lower()
    return "creative commons" in license_text or "creativecommon" in license_text


def _metadata_text(meta: Dict[str, Any]) -> str:
    tags = meta.get("tags") or []
    if isinstance(tags, list):
        tags_text = " ".join(str(tag) for tag in tags)
    else:
        tags_text = str(tags)
    return " ".join([
        str(meta.get("title") or ""),
        str(meta.get("description") or ""),
        str(meta.get("channel") or ""),
        str(meta.get("uploader") or ""),
        str(meta.get("uploader_id") or ""),
        tags_text,
    ]).lower()


def american_score(meta: Dict[str, Any]) -> int:
    score = 0
    text = _metadata_text(meta)

    country = (
        meta.get("channel_country")
        or meta.get("uploader_country")
        or meta.get("location")
        or ""
    )
    country = str(country).strip().upper()
    if country == "US":
        score += 3
    elif country in NON_AMERICAN_COUNTRIES:
        score -= 3

    for keyword in ("american", "usa", "united states", "u.s.", "u.s.a"):
        if keyword in text:
            score += 3
            break

    american_channels = [
        "ted", "tedx", "harvard", "mit", "stanford", "yale", "columbia",
        "khan academy", "crash course", "national geographic", "pbs",
        "cnn", "nbc", "abc news", "cbs", "fox news", "msnbc",
        "andy elliott", "7 figure", "marines", "nasa", "fbi", "cia",
        "us army", "u.s. army", "white house", "smithsonian",
    ]
    for keyword in american_channels:
        if keyword in text:
            score += 2
            break

    american_topics = [
        "college", "university lecture", "high school", "community college",
        "american history", "us history", "constitution", "congress",
        "silicon valley", "wall street", "new york", "los angeles",
        "chicago", "texas", "california", "florida", "washington dc",
    ]
    for keyword in american_topics:
        if keyword in text:
            score += 1
            break

    non_american = [
        "sadhguru", "bollywood", "modi", "nehru", "jaishankar",
        "hindi", "urdu", "ielts", "british accent", "uk accent",
        "australian accent", "indian accent", "nigerian", "pakistani",
        "priyanka chopra", "shah rukh", "srk", "virat kohli",
        "sachin", "alia bhatt", "kangana", "vicky kaushal",
        "indira gandhi", "sundar pichai", "osho", "muniba mazari",
        "welltalk", "magneq", "bishal sarkar", "happiness institute",
        "anu tv", "polyu", "englishing", "jitendra english",
        "rahul gandhi", "narendra modi", "prime minister of canada",
        "prime minister of india", "prime minister of australia",
        "prime minister of uk", "prime minister of britain",
        "bharat", "yatra", "india gate", "new delhi", "afroman",
    ]
    for keyword in non_american:
        if keyword in text:
            score -= 5
            break

    mild_negative = [
        "ielts", "toefl", "esl", "english as second language",
        "english learner", "learn english", "english practice",
    ]
    for keyword in mild_negative:
        if keyword in text:
            score -= 1
            break

    return score


def _existing_download(video_id: str, output_dir: Path) -> Optional[Path]:
    for path in output_dir.glob(f"{video_id}.*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.stat().st_size > MIN_VIDEO_BYTES:
            return path
    return None


def _looks_like_youtube_id(value: str) -> bool:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    return len(value) == 11 and all(char in allowed for char in value)


def sync_existing_downloads(state: dict, config: ScraperConfig) -> None:
    """Record completed files so resumes do not redownload already-saved videos."""
    if not config.output_dir.exists():
        return

    discovered: List[str] = []
    for path in config.output_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if path.stat().st_size <= MIN_VIDEO_BYTES:
            continue
        video_id = path.stem
        if not _looks_like_youtube_id(video_id):
            continue
        if _add_unique(state, "downloaded_ids", video_id):
            discovered.append(video_id)
        _remove_value(state, "pending_download_ids", video_id)
        _remove_value(state, "failed_ids", video_id)
        state.setdefault("in_progress_downloads", {}).pop(video_id, None)

    if discovered:
        state["total_american"] = max(
            int(state.get("total_american", 0)),
            len(state.get("downloaded_ids", [])),
        )
        log.info("Recorded %d completed file(s) already on disk", len(discovered))
        log_event("sync_existing_downloads", state, count=len(discovered), video_ids=discovered[:50])


def _remove_partial_outputs(video_id: str, output_dir: Path) -> None:
    for path in output_dir.glob(f"{video_id}*"):
        if path.suffix.lower() in {".part", ".ytdl"} or ".part" in path.name:
            try:
                path.unlink()
            except OSError:
                pass


def download_video(video_id: str, config: ScraperConfig) -> DownloadResult:
    existing = _existing_download(video_id, config.output_dir)
    if existing is not None:
        size_mb = existing.stat().st_size / 1_048_576
        return DownloadResult(video_id, True, size_mb=size_mb)

    jitter_sleep(config.download_delay_min, config.download_delay_max, "download pacing")
    output_template = str(config.output_dir / f"{video_id}.%(ext)s")
    cmd = yt_dlp_cmd(config.yt_dlp_bin) + [
        f"https://youtube.com/watch?v={video_id}",
        "-f",
        (
            "best[height<=720][ext=mp4]/"
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=720]+bestaudio/"
            "best[height<=720]/best"
        ),
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        "--no-warnings",
        "--quiet",
        "--ignore-errors",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--sleep-requests",
        "0.75",
    ]

    try:
        result = _run_command(cmd, timeout=config.download_timeout)
    except subprocess.TimeoutExpired:
        _remove_partial_outputs(video_id, config.output_dir)
        return DownloadResult(video_id, False, error="download timeout")
    except Exception as exc:
        return DownloadResult(video_id, False, error=str(exc))

    combined = f"{result.stdout}\n{result.stderr}"
    if looks_rate_limited(combined):
        return DownloadResult(video_id, False, error="rate limited", rate_limited=True)

    existing = _existing_download(video_id, config.output_dir)
    if result.returncode == 0 and existing is not None:
        size_mb = existing.stat().st_size / 1_048_576
        return DownloadResult(video_id, True, size_mb=size_mb)

    _remove_partial_outputs(video_id, config.output_dir)
    return DownloadResult(
        video_id,
        False,
        error=(result.stderr or "download failed").strip()[:300],
    )


def download_batch(
    video_ids: List[str],
    config: ScraperConfig
) -> List[DownloadResult]:
    if not video_ids:
        return []

    workers = max(1, min(config.download_workers, len(video_ids)))
    log.info("Downloading %d videos with %d worker(s)", len(video_ids), workers)
    results: List[DownloadResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(download_video, video_id, config): video_id
            for video_id in video_ids
        }
        for future in concurrent.futures.as_completed(future_map):
            video_id = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(DownloadResult(video_id, False, error=str(exc)))

    order = {video_id: idx for idx, video_id in enumerate(video_ids)}
    results.sort(key=lambda item: order.get(item.video_id, 0))
    return results


def _add_unique(state: dict, key: str, video_id: str) -> bool:
    values = state.setdefault(key, [])
    if video_id in values:
        return False
    values.append(video_id)
    return True


def _remove_value(state: dict, key: str, video_id: str) -> None:
    values = state.setdefault(key, [])
    state[key] = [item for item in values if item != video_id]


def add_pending_download(state: dict, video_id: str) -> bool:
    if video_id in state.get("downloaded_ids", []):
        return False
    if video_id in state.get("failed_ids", []):
        return False
    return _add_unique(state, "pending_download_ids", video_id)


def recover_in_progress_downloads(state: dict) -> None:
    in_progress = state.get("in_progress_downloads") or {}
    if not in_progress:
        return
    recovered = []
    for video_id in in_progress.keys():
        if video_id not in state.get("downloaded_ids", []):
            add_pending_download(state, video_id)
            recovered.append(video_id)
    state["last_recovered_in_progress"] = {
        "timestamp": datetime.now().isoformat(),
        "videos": recovered,
        "previous": in_progress,
    }
    state["in_progress_downloads"] = {}
    state["current_download_batch"] = []
    if recovered:
        log.warning(
            "Recovered %d interrupted download(s) into pending queue: %s",
            len(recovered),
            ", ".join(recovered),
        )
        log_event("recover_in_progress", state, videos=recovered, previous=in_progress)


def _query_plan(state: dict, config: ScraperConfig) -> Tuple[int, int, str, int]:
    query_idx = int(state.get("query_index", 0))
    pass_number = int(state.get("pass_number", 0))

    if query_idx >= len(QUERIES):
        pass_number += 1
        query_idx = 0
        state["pass_number"] = pass_number
        log.info("Starting pass %d with deeper query results", pass_number + 1)

    query = QUERIES[query_idx]
    results_this_pass = min(config.results_per_query + pass_number * 10, 100)
    state["query_index"] = query_idx + 1
    return query_idx, pass_number, query, results_this_pass


def evaluate_metadata(
    result: MetadataResult,
    state: dict,
    config: ScraperConfig
) -> Optional[str]:
    video_id = result.video_id

    if result.rate_limited:
        return None

    state["last_checked_video"] = video_id
    if _add_unique(state, "checked_ids", video_id):
        state["total_checked"] = int(state.get("total_checked", 0)) + 1

    if not result.meta:
        state["metadata_failures"] = int(state.get("metadata_failures", 0)) + 1
        log.info("[%s] metadata unavailable: %s", video_id, result.error or "unknown")
        remember_error(state, "metadata", result.error or "metadata unavailable", video_id)
        return None

    meta = result.meta
    title = str(meta.get("title") or "")[:70]
    channel = str(meta.get("channel") or meta.get("uploader") or "Unknown")
    license_text = str(meta.get("license") or "None")
    country = str(meta.get("channel_country") or meta.get("uploader_country") or "unknown").upper()
    duration = int(meta.get("duration") or 0)

    log.info("[%s] %s", video_id, title)
    log.info("  channel=%s country=%s license=%s", channel[:60], country, license_text)

    if not is_cc(meta):
        state["non_cc"] = int(state.get("non_cc", 0)) + 1
        log.info("  skip: metadata license is not Creative Commons")
        log_event("candidate_skip", state, video_id=video_id, reason="not_cc", license=license_text)
        return None

    state["cc_found"] = int(state.get("cc_found", 0)) + 1
    log_event("candidate_cc", state, video_id=video_id, license=license_text)

    score = american_score(meta)
    if score < config.american_threshold:
        log.info(
            "  skip: American score %s < threshold %s",
            score,
            config.american_threshold,
        )
        log_event(
            "candidate_skip",
            state,
            video_id=video_id,
            reason="american_score",
            score=score,
            threshold=config.american_threshold,
        )
        return None

    if duration < config.min_duration:
        log.info("  skip: too short (%ss)", duration)
        log_event(
            "candidate_skip",
            state,
            video_id=video_id,
            reason="too_short",
            duration=duration,
            min_duration=config.min_duration,
        )
        return None

    if video_id in state.get("downloaded_ids", []):
        log.info("  already recorded as downloaded")
        log_event("candidate_skip", state, video_id=video_id, reason="already_downloaded")
        return None

    existing = _existing_download(video_id, config.output_dir)
    if existing is not None:
        _add_unique(state, "downloaded_ids", video_id)
        state["total_american"] = int(state.get("total_american", 0)) + 1
        log.info("  already on disk: %s", existing.name)
        log_event("download_already_on_disk", state, video_id=video_id, path=str(existing))
        return None

    log.info("  eligible: CC + American score %s + duration %ss", score, duration)
    add_pending_download(state, video_id)
    log_event(
        "candidate_eligible",
        state,
        video_id=video_id,
        score=score,
        duration=duration,
        title=title,
    )
    return video_id


def drain_pending_downloads(state: dict, config: ScraperConfig) -> bool:
    """Download queued eligible videos, preserving the queue across crashes."""
    rate_limited = False

    while int(state["total_american"]) < config.target:
        pending = [
            video_id
            for video_id in state.get("pending_download_ids", [])
            if video_id not in state.get("downloaded_ids", [])
            and video_id not in state.get("failed_ids", [])
        ]
        state["pending_download_ids"] = pending
        if not pending:
            state["current_download_batch"] = []
            state["in_progress_downloads"] = {}
            save_state(state, config.state_file)
            return rate_limited

        remaining = config.target - int(state["total_american"])
        batch = pending[:min(config.max_downloads_per_query, remaining)]
        now = datetime.now().isoformat()
        in_progress = state.setdefault("in_progress_downloads", {})
        for video_id in batch:
            in_progress[video_id] = {
                "started_at": now,
                "output_dir": str(config.output_dir),
            }
            state["last_download_started"] = {
                "timestamp": now,
                "video_id": video_id,
            }
            log_event("download_start", state, video_id=video_id)

        set_phase(
            state,
            "download",
            current_download_batch=batch,
        )
        save_state(state, config.state_file)

        results = download_batch(batch, config)
        batch_rate_limited = any(item.rate_limited for item in results)

        for result in results:
            state.setdefault("in_progress_downloads", {}).pop(result.video_id, None)
            if result.success:
                _remove_value(state, "pending_download_ids", result.video_id)
                if _add_unique(state, "downloaded_ids", result.video_id):
                    state["total_american"] = int(state["total_american"]) + 1
                state["last_download_finished"] = {
                    "timestamp": datetime.now().isoformat(),
                    "video_id": result.video_id,
                    "size_mb": result.size_mb,
                }
                log.info(
                    "[%s] downloaded %.1f MB. Total=%d/%d",
                    result.video_id,
                    result.size_mb,
                    int(state["total_american"]),
                    config.target,
                )
                log_event(
                    "download_success",
                    state,
                    video_id=result.video_id,
                    size_mb=result.size_mb,
                    total_american=int(state["total_american"]),
                )
            elif result.rate_limited:
                # Keep the video in pending so `continue` can retry it later.
                state["last_rate_limit"] = datetime.now().isoformat()
                remember_error(
                    state,
                    "download",
                    result.error or "rate limited",
                    result.video_id,
                    rate_limited=True,
                )
                log.warning("[%s] download rate-limited: %s", result.video_id, result.error)
            else:
                _remove_value(state, "pending_download_ids", result.video_id)
                _add_unique(state, "failed_ids", result.video_id)
                state["download_failures"] = int(state.get("download_failures", 0)) + 1
                remember_error(
                    state,
                    "download",
                    result.error or "download failed",
                    result.video_id,
                )
                log.warning("[%s] download failed: %s", result.video_id, result.error)
            save_state(state, config.state_file)

        if batch_rate_limited:
            rate_limited = True
            save_state(state, config.state_file)
            cooldown(config.rate_limit_cooldown, "download rate-limit/block signal")
            return rate_limited

        # Continue draining only if this was a resume backlog. During normal
        # scraping, the caller queues at most max_downloads_per_query at a time.
        if len(batch) < config.max_downloads_per_query:
            return rate_limited

    return rate_limited


def run(config: ScraperConfig, continue_mode: bool = False) -> None:
    state = load_state(config.state_file)
    begin_or_resume_run(state, config, continue_mode=continue_mode)
    recover_in_progress_downloads(state)
    save_state(state, config.state_file)

    try:
        set_phase(state, "preflight")
        save_state(state, config.state_file)
        preflight(config)
        sync_existing_downloads(state, config)
        save_state(state, config.state_file)
    except Exception as exc:
        remember_error(state, "preflight", str(exc))
        complete_run(state, status="error")
        save_state(state, config.state_file)
        raise

    checked_set = set(state.get("checked_ids", []))
    downloaded_set = set(state.get("downloaded_ids", []))

    log.info("=" * 72)
    log.info("YOUTUBE CC SCRAPER - FAST SAFE MODE")
    log.info("Target: %d American-English Creative Commons videos", config.target)
    log.info(
        "Workers: metadata=%d downloads=%d max_downloads_per_query=%d",
        config.metadata_workers,
        config.download_workers,
        config.max_downloads_per_query,
    )
    log.info("Progress: %d/%d", int(state["total_american"]), config.target)
    log.info(
        "Checked=%d CC=%d Non-CC=%d Files=%d",
        int(state["total_checked"]),
        int(state["cc_found"]),
        int(state["non_cc"]),
        len(list(config.output_dir.glob("*.mp4"))),
    )
    log.info("=" * 72)

    batch_count = 0

    while int(state["total_american"]) < config.target:
        if state.get("pending_download_ids"):
            log.info(
                "Pending queue has %d video(s); downloading these before new search.",
                len(state["pending_download_ids"]),
            )
            rate_limited = drain_pending_downloads(state, config)
            checked_set = set(state.get("checked_ids", []))
            downloaded_set = set(state.get("downloaded_ids", []))
            if int(state["total_american"]) >= config.target:
                break
            if rate_limited:
                continue

        query_idx, pass_number, query, result_count = _query_plan(state, config)
        batch_count += 1
        set_phase(
            state,
            "search",
            current_query=query,
            current_query_index=query_idx,
            current_pass_number=pass_number,
        )
        save_state(state, config.state_file)

        log.info("")
        log.info("=" * 72)
        log.info(
            "QUERY [%d/%d] pass %d: %s",
            query_idx + 1,
            len(QUERIES),
            pass_number + 1,
            query,
        )
        log.info(
            "Progress: %d/%d (%.1f%%)",
            int(state["total_american"]),
            config.target,
            int(state["total_american"]) / max(config.target, 1) * 100.0,
        )
        log.info("=" * 72)

        log_event(
            "search_start",
            state,
            query=query,
            query_index=query_idx,
            pass_number=pass_number,
            result_count=result_count,
        )
        video_ids, rate_limited = search_cc_video_ids(query, result_count, config)
        log_event("search_result", state, query=query, count=len(video_ids), rate_limited=rate_limited)
        save_state(state, config.state_file)
        if rate_limited:
            state["last_rate_limit"] = datetime.now().isoformat()
            remember_error(state, "search", f"rate-limit/block signal while searching {query}", rate_limited=True)
            save_state(state, config.state_file)
            cooldown(config.rate_limit_cooldown, "search rate-limit/block signal")
            continue

        new_candidates = [
            video_id
            for video_id in video_ids
            if video_id not in checked_set and video_id not in downloaded_set
        ]
        if not new_candidates:
            log.info("No new candidates in this query")
            jitter_sleep(config.query_delay * 0.8, config.query_delay * 1.25, "next query")
            if config.once:
                break
            continue

        set_phase(
            state,
            "metadata",
            current_query=query,
            current_query_index=query_idx,
            current_pass_number=pass_number,
        )
        save_state(state, config.state_file)
        log_event("metadata_batch_start", state, query=query, count=len(new_candidates))
        metadata_results = fetch_metadata_batch(new_candidates, config)
        if any(item.rate_limited for item in metadata_results):
            state["last_rate_limit"] = datetime.now().isoformat()
            remember_error(state, "metadata", f"rate-limit/block signal in query {query}", rate_limited=True)
            save_state(state, config.state_file)
            cooldown(config.rate_limit_cooldown, "metadata rate-limit/block signal")
            continue

        eligible: List[str] = []
        for result in metadata_results:
            video_id = evaluate_metadata(result, state, config)
            checked_set = set(state.get("checked_ids", []))
            downloaded_set = set(state.get("downloaded_ids", []))
            save_state(state, config.state_file)

            if video_id is not None:
                eligible.append(video_id)
            remaining = config.target - int(state["total_american"])
            if len(eligible) >= min(config.max_downloads_per_query, remaining):
                break

        if eligible:
            log_event("eligible_batch", state, query=query, video_ids=eligible)
            drain_pending_downloads(state, config)

        log.info("")
        log.info(
            "Query done. Progress=%d/%d Checked=%d CC=%d",
            int(state["total_american"]),
            config.target,
            int(state["total_checked"]),
            int(state["cc_found"]),
        )
        save_state(state, config.state_file)

        if int(state["total_american"]) >= config.target or config.once:
            break

        if batch_count % 10 == 0:
            cooldown(config.batch_delay, "batch pacing")
        else:
            jitter_sleep(config.query_delay * 0.8, config.query_delay * 1.25, "next query")

    log.info("")
    log.info("=" * 72)
    log.info("DONE")
    log.info("American-English CC videos: %d/%d", int(state["total_american"]), config.target)
    log.info("Total checked: %d", int(state["total_checked"]))
    cc_rate = int(state["cc_found"]) / max(int(state["total_checked"]), 1) * 100.0
    log.info("CC rate: %.1f%%", cc_rate)
    log.info("=" * 72)
    complete_run(state, status="completed")
    save_state(state, config.state_file)


def show_status(config: ScraperConfig) -> None:
    if not config.state_file.exists():
        print("No scraping session found. Run: python scrape_youtube_no_api.py run --test")
        return

    state = load_state(config.state_file)
    if state.get("resume_config"):
        saved_config = config_from_saved(state["resume_config"])
        # Keep the state file selected by this status command, but show the
        # target/output/event paths from the run that created the state.
        saved_config.state_file = config.state_file
        config = saved_config

    american = int(state.get("total_american", 0))
    checked = int(state.get("total_checked", 0))
    cc_count = int(state.get("cc_found", 0))
    failed = len(state.get("failed_ids", []))
    pending = len(state.get("pending_download_ids", []))
    in_progress = len(state.get("in_progress_downloads", {}))
    pct = american / max(config.target, 1) * 100.0
    cc_rate = cc_count / max(checked, 1) * 100.0
    filled = int(40 * american / max(config.target, 1))
    bar = "#" * filled + "." * (40 - filled)

    print("")
    print("NO-API SCRAPER STATUS")
    print("=" * 50)
    print(f"Run:       {state.get('run_id') or 'none'} ({state.get('run_status')})")
    print(f"Phase:     {state.get('current_phase')}")
    print(f"Target:    {config.target}")
    print(f"American:  {american} ({pct:.1f}%)")
    print(f"Checked:   {checked}")
    print(f"CC found:  {cc_count} ({cc_rate:.0f}% CC rate)")
    print(f"Pending:   {pending}")
    print(f"In flight: {in_progress}")
    print(f"Failed:    {failed}")
    print(f"Files:     {len(list(config.output_dir.glob('*.mp4')))}")
    print(f"State:     {config.state_file}")
    print(f"Output:    {config.output_dir}")
    print(f"Events:    {config.event_log_file}")
    if state.get("last_command_line"):
        print(f"Command:   {state['last_command_line']}")
    if state.get("current_query"):
        print(f"Query:     {state['current_query']}")
    if state.get("last_checked_video"):
        print(f"Last checked:    {state['last_checked_video']}")
    if state.get("last_download_started"):
        last_started = state["last_download_started"]
        print(f"Last download started:  {last_started.get('video_id')} at {last_started.get('timestamp')}")
    if state.get("last_download_finished"):
        last_finished = state["last_download_finished"]
        print(f"Last download finished: {last_finished.get('video_id')} at {last_finished.get('timestamp')}")
    if state.get("last_rate_limit"):
        print(f"Last rate-limit/block signal: {state['last_rate_limit']}")
    if state.get("last_error"):
        last_error = state["last_error"]
        print(
            "Last error: "
            f"{last_error.get('phase')} {last_error.get('video_id') or ''} "
            f"{last_error.get('message')}"
        )
    print("")
    print(f"[{bar}] {pct:.1f}%")
    print("")
    print("Continue: python scrape_youtube_no_api.py continue")
    print(f"Log:      {LOG_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast, Windows-safe YouTube Creative Commons scraper."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "continue", "status"),
        default="run",
        help="run a new scrape, continue the saved scrape, or show status",
    )
    parser.add_argument("--test", action="store_true", help="Stop after 10 total videos")
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    parser.add_argument("--once", action="store_true", help="Process one query and exit")
    parser.add_argument("--target", type=int, default=TARGET, help="Target download count")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--event-log-file", type=Path, default=EVENT_LOG_FILE)
    parser.add_argument("--yt-dlp", dest="yt_dlp_bin", help="Path to yt-dlp executable")

    parser.add_argument("--metadata-workers", type=int, default=DEFAULT_METADATA_WORKERS)
    parser.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    parser.add_argument("--max-downloads-per-query", type=int, default=DEFAULT_MAX_DOWNLOADS_PER_QUERY)
    parser.add_argument("--results-per-query", type=int, default=RESULTS_PER_QUERY)

    parser.add_argument("--american-threshold", type=int, default=AMERICAN_THRESHOLD)
    parser.add_argument("--min-duration", type=int, default=MIN_DURATION_SECONDS)

    parser.add_argument("--query-delay", type=float, default=DELAY_BETWEEN_QUERIES)
    parser.add_argument("--batch-delay", type=float, default=DELAY_BATCH)
    parser.add_argument("--metadata-delay-min", type=float, default=METADATA_DELAY_RANGE[0])
    parser.add_argument("--metadata-delay-max", type=float, default=METADATA_DELAY_RANGE[1])
    parser.add_argument("--download-delay-min", type=float, default=DOWNLOAD_DELAY_RANGE[0])
    parser.add_argument("--download-delay-max", type=float, default=DOWNLOAD_DELAY_RANGE[1])
    parser.add_argument("--rate-limit-cooldown", type=float, default=RATE_LIMIT_COOLDOWN)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ScraperConfig:
    target = 10 if args.test else args.target
    return ScraperConfig(
        target=max(1, target),
        output_dir=args.output_dir,
        state_file=args.state_file,
        metadata_workers=max(1, args.metadata_workers),
        download_workers=max(1, args.download_workers),
        max_downloads_per_query=max(1, args.max_downloads_per_query),
        results_per_query=max(1, args.results_per_query),
        american_threshold=args.american_threshold,
        min_duration=max(0, args.min_duration),
        query_delay=max(0.0, args.query_delay),
        batch_delay=max(0.0, args.batch_delay),
        metadata_delay_min=max(0.0, args.metadata_delay_min),
        metadata_delay_max=max(args.metadata_delay_min, args.metadata_delay_max),
        download_delay_min=max(0.0, args.download_delay_min),
        download_delay_max=max(args.download_delay_min, args.download_delay_max),
        rate_limit_cooldown=max(0.0, args.rate_limit_cooldown),
        yt_dlp_bin=args.yt_dlp_bin,
        once=bool(args.once),
        event_log_file=args.event_log_file,
    )


def main() -> int:
    args = parse_args()
    command = "status" if args.status else args.command
    config: Optional[ScraperConfig] = None

    if command == "status":
        config = config_from_args(args)
        show_status(config)
        return 0

    try:
        if command == "continue":
            state = load_state(args.state_file)
            config = merge_resume_config(state.get("resume_config", {}), args)
            run(config, continue_mode=True)
        else:
            config = config_from_args(args)
            run(config)
        return 0
    except KeyboardInterrupt:
        if config is not None:
            state = load_state(config.state_file)
            complete_run(state, status="interrupted")
            save_state(state, config.state_file)
        log.warning("Interrupted. State has been saved after the last completed step.")
        return 130
    except Exception as exc:
        if config is not None:
            try:
                state = load_state(config.state_file)
                if state.get("run_status") != "error":
                    remember_error(state, state.get("current_phase") or "main", str(exc))
                    complete_run(state, status="error")
                save_state(state, config.state_file)
            except Exception:
                log.debug("Failed to persist fatal error state", exc_info=True)
        log.exception("Fatal scraper error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
