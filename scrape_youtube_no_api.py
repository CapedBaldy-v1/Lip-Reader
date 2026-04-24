#!/usr/bin/env python3
"""
YouTube Scraper WITHOUT API — No quota limits, no bullshit.

Uses YouTube's own CC filter URL (sp=EgIwAQ%3D%3D) scraped via yt-dlp.
This gives ~100% CC hit rate instead of ~0% from generic search.

USAGE:
    python3 scrape_youtube_no_api.py          # full run (2000 videos)
    python3 scrape_youtube_no_api.py --test   # test run (10 videos)
    python3 scrape_youtube_no_api.py --status # show progress
"""

import sys
import json
import time
import random
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('no_api_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET         = 2000
OUTPUT_DIR     = Path('youtube_raw_downloads')
STATE_FILE     = Path('no_api_state.json')

# YouTube CC filter URL parameter — this is the key!
# sp=EgIwAQ%3D%3D  →  filters search results to Creative Commons only
CC_FILTER = "EgIwAQ%3D%3D"

# Delays (seconds)
DELAY_BETWEEN_VIDEOS  = 3    # between downloads
DELAY_BETWEEN_QUERIES = 15   # between search queries
DELAY_BATCH           = 90   # every 10 queries

# Results per query (YouTube returns max ~20 per page, we paginate)
RESULTS_PER_QUERY = 20

# American English score threshold
AMERICAN_THRESHOLD = 0   # block known non-American, pass everything else

# ── Search queries ─────────────────────────────────────────────────────────────
# These are searched WITH the CC filter, so ~100% of results will be CC
QUERIES = [
    # American-specific searches (highest yield)
    "american speech", "american english speaking", "american accent",
    "US college lecture", "american university lecture",
    "american history lecture", "american politics speech",
    "american business presentation", "american motivational speech",

    # TED (mostly American speakers)
    "TED talk", "TEDx talk", "TED conference", "TED ideas",

    # Known American educational channels
    "khan academy", "crash course", "MIT lecture", "Harvard lecture",
    "Stanford lecture", "Yale lecture", "PBS documentary",

    # American professional content
    "sales training", "marketing presentation", "startup pitch",
    "silicon valley talk", "wall street presentation",
    "american corporate training", "american leadership",

    # American news / media
    "CNN interview", "NBC news", "ABC news report",
    "american press conference", "white house briefing",

    # American sports / culture
    "NFL speech", "NBA interview", "american sports commentary",
    "american cooking show", "american travel vlog",

    # Generic but filtered by CC (still gets diverse content)
    "lecture", "tutorial", "how to", "documentary",
    "presentation", "training video", "explainer",
    "science lecture", "technology talk", "history documentary",
    "health education", "finance explained",
]


# ── State ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'total_american': 0,
        'total_checked': 0,
        'cc_found': 0,
        'non_cc': 0,
        'downloaded_ids': [],
        'checked_ids': [],
        'query_index': 0,
        'pass_number': 0,
        'start_time': datetime.now().isoformat(),
    }

def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# ── Core helpers ──────────────────────────────────────────────────────────────
def cc_search_url(query: str, page_token: str = '') -> str:
    """Build YouTube CC-filtered search URL."""
    q = quote(query)
    url = f"https://www.youtube.com/results?search_query={q}&sp={CC_FILTER}"
    return url


def search_cc_video_ids(query: str, n: int = RESULTS_PER_QUERY) -> list:
    """
    Search YouTube with CC filter and return video IDs.
    Uses yt-dlp to scrape the search results page directly.
    """
    url = cc_search_url(query)
    cmd = [
        'yt-dlp',
        url,
        '--get-id',
        '--no-warnings',
        '--ignore-errors',
        '--flat-playlist',
        f'--playlist-end', str(n),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        ids = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
        log.info(f"  CC search returned {len(ids)} video IDs")
        return ids
    except Exception as e:
        log.error(f"  Search error: {e}")
        return []


def get_video_metadata(video_id: str) -> dict | None:
    """Fetch full metadata for one video."""
    cmd = [
        'yt-dlp',
        f'https://youtube.com/watch?v={video_id}',
        '--skip-download',
        '--dump-json',
        '--no-warnings',
        '--ignore-errors',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout.strip())
    except Exception as e:
        log.debug(f"  Metadata error {video_id}: {e}")
    return None


def is_cc(meta: dict) -> bool:
    lic = (meta.get('license') or '').lower()
    return 'creative commons' in lic or 'creativecommon' in lic


def american_score(meta: dict) -> int:
    score = 0

    text = ' '.join([
        meta.get('title') or '',
        meta.get('description') or '',
        meta.get('channel') or '',
        meta.get('uploader') or '',
        meta.get('uploader_id') or '',
        meta.get('tags') and ' '.join(meta['tags']) or '',
    ]).lower()

    # Strong positive signals
    for kw in ('american', 'usa', 'united states', 'u.s.', 'u.s.a'):
        if kw in text:
            score += 3
            break

    # Positive signals — American channels/content
    american_channels = [
        'ted', 'tedx', 'harvard', 'mit', 'stanford', 'yale', 'columbia',
        'khan academy', 'crash course', 'national geographic', 'pbs',
        'cnn', 'nbc', 'abc news', 'cbs', 'fox news', 'msnbc',
        'andy elliott', '7 figure', 'marines', 'nasa', 'fbi', 'cia',
    ]
    for kw in american_channels:
        if kw in text:
            score += 2
            break

    # Neutral but likely American topics
    american_topics = [
        'college', 'university lecture', 'high school', 'community college',
        'american history', 'us history', 'constitution', 'congress',
        'silicon valley', 'wall street', 'new york', 'los angeles',
        'chicago', 'texas', 'california', 'florida',
    ]
    for kw in american_topics:
        if kw in text:
            score += 1
            break

    # Strong negative signals — non-American speakers
    non_american = [
        'sadhguru', 'bollywood', 'modi', 'nehru', 'jaishankar',
        'hindi', 'urdu', 'ielts', 'british accent', 'uk accent',
        'australian accent', 'indian accent', 'nigerian', 'pakistani',
        'priyanka chopra', 'shah rukh', 'srk', 'virat kohli',
        'sachin', 'alia bhatt', 'kangana', 'vicky kaushal',
        'indira gandhi', 'sundar pichai', 'osho', 'muniba mazari',
        'welltalk', 'magneq', 'bishal sarkar', 'happiness institute',
        'anu tv', 'polyu', 'englishing', 'jitendra english',
        'rahul gandhi', 'narendra modi', 'prime minister of canada',
        'prime minister of india', 'prime minister of australia',
        'prime minister of uk', 'prime minister of britain',
        'bharat', 'yatra', 'india gate', 'new delhi',
        'afroman',  # not a speech/lecture
    ]
    for kw in non_american:
        if kw in text:
            score -= 5
            break

    # Mild negative — likely non-American
    mild_negative = [
        'ielts', 'toefl', 'esl', 'english as second language',
        'english learner', 'learn english', 'english practice',
    ]
    for kw in mild_negative:
        if kw in text:
            score -= 1
            break

    return score


def download_video(video_id: str) -> bool:
    out = OUTPUT_DIR / f'{video_id}.mp4'
    if out.exists() and out.stat().st_size > 5_000_000:
        log.info(f"  ✓ Already on disk: {video_id}")
        return True

    cmd = [
        'yt-dlp',
        f'https://youtube.com/watch?v={video_id}',
        '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        '--merge-output-format', 'mp4',
        '-o', str(out),
        '--no-warnings',
        '--quiet',
        '--progress',
        '--ignore-errors',
    ]
    try:
        r = subprocess.run(cmd, timeout=300)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 5_000_000:
            mb = out.stat().st_size / 1_048_576
            log.info(f"  ✅ Downloaded {video_id} ({mb:.1f} MB)")
            return True
        log.error(f"  ✗ Download failed: {video_id}")
        if out.exists():
            out.unlink()
        return False
    except subprocess.TimeoutExpired:
        log.error(f"  ✗ Timeout: {video_id}")
        if out.exists():
            out.unlink()
        return False
    except Exception as e:
        log.error(f"  ✗ Error: {e}")
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(target: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    log.info("=" * 70)
    log.info("YOUTUBE CC SCRAPER — NO API, NO QUOTA LIMITS")
    log.info(f"Target: {target} American English CC videos")
    log.info(f"Method: YouTube CC filter URL (100% CC hit rate)")
    log.info("=" * 70)
    log.info(f"Progress: {state['total_american']}/{target}")
    log.info(f"Checked: {state['total_checked']}  |  CC: {state['cc_found']}  |  Non-CC: {state['non_cc']}")

    if state['total_american'] >= target:
        log.info("✅ Target already reached!")
        return

    query_idx   = state['query_index']
    pass_number = state['pass_number']
    batch_count = 0

    while state['total_american'] < target:

        # ── pick next query ──────────────────────────────────────────────────
        if query_idx >= len(QUERIES):
            pass_number += 1
            query_idx = 0
            state['pass_number'] = pass_number
            # On subsequent passes, get more results per query
            log.info(f"\n🔄 Pass #{pass_number + 1} — getting more results per query")

        query = QUERIES[query_idx]
        results_this_pass = min(RESULTS_PER_QUERY + pass_number * 10, 50)
        query_idx += 1
        state['query_index'] = query_idx
        batch_count += 1

        log.info(f"\n{'='*70}")
        log.info(f"QUERY [{query_idx}/{len(QUERIES)}] pass {pass_number+1}: '{query}'")
        log.info(f"Progress: {state['total_american']}/{target}  "
                 f"({state['total_american']/target*100:.1f}%)")
        log.info("=" * 70)

        # ── search with CC filter ────────────────────────────────────────────
        video_ids = search_cc_video_ids(query, results_this_pass)
        if not video_ids:
            log.warning("  No results — skipping")
            time.sleep(DELAY_BETWEEN_QUERIES)
            save_state(state)
            continue

        # ── process each video ───────────────────────────────────────────────
        new_this_query = 0
        for i, vid in enumerate(video_ids, 1):

            if state['total_american'] >= target:
                break

            if vid in state['checked_ids']:
                log.debug(f"  [{i}] {vid} already checked")
                continue

            log.info(f"\n  [{i}/{len(video_ids)}] {vid}")

            meta = get_video_metadata(vid)
            state['checked_ids'].append(vid)
            state['total_checked'] += 1

            if not meta:
                log.warning("  ⚠️  No metadata")
                save_state(state)
                continue

            title   = (meta.get('title') or '')[:60]
            channel = (meta.get('channel') or 'Unknown')
            lic     = meta.get('license') or 'None'
            country = (meta.get('channel_country') or
                       meta.get('uploader_country') or 'unknown').upper()

            log.info(f"  Title:   {title}")
            log.info(f"  Channel: {channel}  [{country}]")
            log.info(f"  License: {lic}")

            # ── CC check ─────────────────────────────────────────────────────
            if not is_cc(meta):
                log.info("  ✗ Not CC — skip")
                state['non_cc'] += 1
                save_state(state)
                time.sleep(1)
                continue

            state['cc_found'] += 1
            log.info("  ✓ Creative Commons!")

            # ── American English check ────────────────────────────────────────
            score = american_score(meta)
            log.info(f"  American score: {score}")

            if score < AMERICAN_THRESHOLD:
                log.info(f"  ✗ Score {score} < {AMERICAN_THRESHOLD} — not American English")
                save_state(state)
                time.sleep(1)
                continue

            # ── Duration check — skip shorts/clips under 60 seconds ───────────
            duration = meta.get('duration') or 0
            if duration < 60:
                log.info(f"  ✗ Too short ({duration}s) — skipping shorts/clips")
                save_state(state)
                time.sleep(1)
                continue

            log.info(f"  ✓ American English! Duration: {duration//60}m{duration%60}s")

            if vid in state['downloaded_ids']:
                log.info("  ✓ Already downloaded")
                continue

            # ── Download ──────────────────────────────────────────────────────
            log.info("  ⬇️  Downloading...")
            ok = download_video(vid)

            if ok:
                state['downloaded_ids'].append(vid)
                state['total_american'] += 1
                new_this_query += 1
                log.info(f"  🎉 Total: {state['total_american']}/{target}")

            save_state(state)
            time.sleep(DELAY_BETWEEN_VIDEOS)

        log.info(f"\n✅ Query done — new videos: {new_this_query}")
        log.info(f"📊 {state['total_american']}/{target} "
                 f"({state['total_american']/target*100:.1f}%)")
        log.info(f"   CC rate: {state['cc_found']}/{state['total_checked']} "
                 f"({state['cc_found']/max(state['total_checked'],1)*100:.0f}%)")

        save_state(state)

        if state['total_american'] >= target:
            break

        # ── delay ─────────────────────────────────────────────────────────────
        if batch_count % 10 == 0:
            log.info(f"\n⏸️  Batch cooldown ({DELAY_BATCH}s)...")
            time.sleep(DELAY_BATCH)
        else:
            log.info(f"\n⏸️  Next query in {DELAY_BETWEEN_QUERIES}s...")
            time.sleep(DELAY_BETWEEN_QUERIES)

    # ── Done ──────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("🎉 DONE!")
    log.info(f"American CC videos: {state['total_american']}/{target}")
    log.info(f"Total checked: {state['total_checked']}")
    cc_rate = state['cc_found'] / max(state['total_checked'], 1) * 100
    log.info(f"CC rate: {cc_rate:.1f}%")
    log.info("=" * 70)


def show_status():
    if not STATE_FILE.exists():
        print("No scraping session found. Run: python3 scrape_youtube_no_api.py --test")
        return
    with open(STATE_FILE) as f:
        s = json.load(f)
    american = s['total_american']
    checked  = s['total_checked']
    cc       = s['cc_found']
    pct      = american / TARGET * 100
    cc_rate  = cc / max(checked, 1) * 100
    bar      = '█' * int(40 * american / TARGET) + '░' * (40 - int(40 * american / TARGET))
    print(f"\n📊 NO-API SCRAPER STATUS")
    print(f"{'='*50}")
    print(f"🎯 Target:   {TARGET}")
    print(f"✅ American: {american} ({pct:.1f}%)")
    print(f"📋 Checked:  {checked}")
    print(f"📄 CC found: {cc} ({cc_rate:.0f}% CC rate)")
    print(f"📁 Files:    {len(list(OUTPUT_DIR.glob('*.mp4')))}")
    print(f"\n[{bar}] {pct:.1f}%")
    print(f"\nResume: python3 scrape_youtube_no_api.py")
    print(f"Log:    tail -f no_api_scraper.log")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',   action='store_true', help='Stop after 10 videos')
    parser.add_argument('--status', action='store_true', help='Show progress and exit')
    args = parser.parse_args()

    if args.status:
        show_status()
        sys.exit(0)

    target = 10 if args.test else TARGET

    try:
        run(target)
    except KeyboardInterrupt:
        log.warning("\n⚠️  Interrupted — state saved. Run again to resume.")
        sys.exit(0)
    except Exception as e:
        log.error(f"\n❌ FATAL: {e}")
        log.exception("Traceback:")
        sys.exit(1)
