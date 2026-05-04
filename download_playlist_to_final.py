#!/usr/bin/env python3
"""
Download a known-good YouTube playlist directly into the final trainable folder.

The script is intentionally separate from the broad scraper. It trusts the
playlist selection, writes files as <video_id>.mp4 where possible, and uses a
yt-dlp download archive so reruns continue without duplicating videos.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_PLAYLIST_URL = (
    "https://www.youtube.com/watch?v=tF4ytvvIUJc"
    "&list=PLrLY2VO2cgPV9T2IpKBdUEp8aVOs0OkDx&index=1"
)
DEFAULT_OUTPUT_DIR = Path("final_trainable_videos_heavy")
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm"}
BLOCK_PATTERNS = (
    "sign in to confirm",
    "not a bot",
    "captcha",
    "unusual traffic",
    "temporarily blocked",
    "too many requests",
    "http error 429",
    "429:",
)


def candidate_yt_dlp_commands(explicit_bin: Optional[str]) -> List[List[str]]:
    candidates: List[List[str]] = []
    if explicit_bin:
        candidates.append([explicit_bin])

    env_bin = os.getenv("YT_DLP_BIN")
    if env_bin:
        candidates.append([env_bin])

    candidates.append([sys.executable, "-m", "yt_dlp"])

    path_bin = shutil.which("yt-dlp")
    if path_bin:
        candidates.append([path_bin])

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
    for candidate in candidate_yt_dlp_commands(explicit_bin):
        try:
            result = subprocess.run(
                candidate + ["--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            return candidate

    raise RuntimeError(
        "yt-dlp was not found. Install it first with: python -m pip install -U yt-dlp"
    )


def media_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS
    ]


def folder_size_gb(path: Path) -> float:
    total = 0
    for item in media_files(path):
        total += item.stat().st_size
    return total / 1_073_741_824


def archive_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def looks_blocked(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in BLOCK_PATTERNS)


def available_js_runtime() -> Optional[str]:
    for runtime in ("node", "deno"):
        if shutil.which(runtime):
            return runtime
    return None


def cooldown(seconds: float, reason: str) -> None:
    seconds = max(0.0, seconds)
    if seconds <= 0:
        return

    print("")
    print(f"BLOCK/COOLDOWN: sleeping for {seconds / 60:.1f} minutes ({reason})")
    print("If this keeps happening: sign into YouTube in your browser, or turn on Proton VPN, then let this retry.")
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        sleep_for = min(60.0, remaining)
        time.sleep(sleep_for)
        remaining = end - time.monotonic()
        if remaining > 0:
            print(f"Cooldown remaining: {remaining / 60:.1f} minutes")


def build_command(args: argparse.Namespace) -> List[str]:
    output_dir = args.output_dir
    output_template = str(output_dir / "%(id)s.%(ext)s")
    archive_file = args.archive_file or output_dir / "_playlist_download_archive.txt"

    cmd = yt_dlp_cmd(args.yt_dlp) + [
        args.url,
        "--yes-playlist",
        "--ignore-errors",
        "--continue",
        "--no-overwrites",
        "--download-archive",
        str(archive_file),
        "-o",
        output_template,
        "-f",
        (
            "best[height<=720][ext=mp4]/"
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo[height<=720]+bestaudio/"
            "best[height<=720]/best"
        ),
        "--merge-output-format",
        "mp4",
        "--retries",
        str(args.retries),
        "--fragment-retries",
        str(args.fragment_retries),
        "--sleep-requests",
        str(args.sleep_requests),
        "--sleep-interval",
        str(args.sleep_interval),
        "--max-sleep-interval",
        str(args.max_sleep_interval),
        "--concurrent-fragments",
        str(args.concurrent_fragments),
        "--newline",
    ]

    if args.playlist_start:
        cmd += ["--playlist-start", str(args.playlist_start)]
    if args.playlist_end:
        cmd += ["--playlist-end", str(args.playlist_end)]
    if args.verify_creative_commons:
        cmd += ["--match-filter", "license~='(?i)creative\\s*commons|creativecommon'"]
    if args.cookies_file:
        cmd += ["--cookies", str(args.cookies_file)]
    elif args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    if args.js_runtime == "auto":
        runtime = available_js_runtime()
        if runtime:
            cmd += ["--js-runtimes", runtime]
    elif args.js_runtime:
        cmd += ["--js-runtimes", args.js_runtime]
    if args.dry_run:
        cmd += ["--simulate", "--print", "%(playlist_index)s %(id)s %(title)s"]

    return cmd


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def stream_command(cmd: Sequence[str], log_file: Path) -> Tuple[int, bool]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(f"{datetime.now().isoformat()} command: {' '.join(cmd)}\n")
        log.flush()

        process = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        blocked = False
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
            if looks_blocked(line):
                blocked = True
                message = (
                    f"{datetime.now().isoformat()} detected YouTube block/bot-check; "
                    "stopping yt-dlp so it does not burn through the rest of the playlist.\n"
                )
                print(message, end="")
                log.write(message)
                log.flush()
                stop_process(process)
                break
        return process.wait(), blocked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the curated playlist into final_trainable_videos_heavy."
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_PLAYLIST_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive-file", type=Path)
    parser.add_argument("--log-file", type=Path, default=Path("playlist_final_downloader.log"))
    parser.add_argument("--yt-dlp", help="Path to yt-dlp executable")
    parser.add_argument("--playlist-start", type=int)
    parser.add_argument("--playlist-end", type=int)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--fragment-retries", type=int, default=10)
    parser.add_argument("--sleep-requests", type=float, default=0.75)
    parser.add_argument("--sleep-interval", type=float, default=2.0)
    parser.add_argument("--max-sleep-interval", type=float, default=6.0)
    parser.add_argument("--concurrent-fragments", type=int, default=3)
    parser.add_argument("--block-cooldown", type=float, default=30 * 60)
    parser.add_argument("--block-retries", type=int, default=8)
    parser.add_argument(
        "--js-runtime",
        default="auto",
        help="JavaScript runtime for yt-dlp signature extraction: auto, node, deno, or empty string to disable.",
    )
    parser.add_argument(
        "--verify-creative-commons",
        action="store_true",
        help="Only download videos whose yt-dlp license metadata says Creative Commons.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        help="Optional yt-dlp browser cookie source, e.g. chrome, edge, or firefox.",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="Optional Netscape-format cookies.txt file exported from your browser.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List playlist items without downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before_count = len(list(media_files(args.output_dir)))
    before_gb = folder_size_gb(args.output_dir)
    archive_file = args.archive_file or args.output_dir / "_playlist_download_archive.txt"
    print(f"Output folder: {args.output_dir.resolve()}")
    print(f"Before: {before_count} media file(s), {before_gb:.2f} GB")
    print(f"Archive already has {archive_count(archive_file)} completed playlist item(s)")

    try:
        cmd = build_command(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    exit_code = 1
    for attempt in range(1, max(1, args.block_retries) + 1):
        print("")
        print(f"Attempt {attempt}/{max(1, args.block_retries)}")
        print(f"Completed playlist archive entries before attempt: {archive_count(archive_file)}")
        exit_code, blocked = stream_command(cmd, args.log_file)
        if not blocked:
            break
        if attempt >= max(1, args.block_retries):
            print("")
            print("Stopped after repeated YouTube bot-checks.")
            print("Best next move: sign into YouTube in Chrome/Edge and rerun with --cookies-from-browser chrome or edge.")
            print("If your home IP is temporarily flagged, turn on Proton VPN before rerunning.")
            break
        cooldown(args.block_cooldown, "YouTube bot-check/block signal")

    after_count = len(list(media_files(args.output_dir)))
    after_gb = folder_size_gb(args.output_dir)
    print("")
    print(f"After:  {after_count} media file(s), {after_gb:.2f} GB")
    print(f"Added:  {max(0, after_count - before_count)} media file(s)")
    print(f"Log:    {args.log_file}")
    print(f"Archive:{archive_file}")
    print(f"Archive completed entries: {archive_count(archive_file)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
