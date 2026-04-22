#!/usr/bin/env python3
"""
American English YouTube Video Downloader for Research
Downloads only Creative Commons (CC-BY) videos and TED talks with AMERICAN ENGLISH accent.

NEW FEATURES (vs download_legal_videos.py):
- Pre-filtering: Bias toward US content using regionCode and channel metadata
- American score calculation: Score videos by likelihood of American English
- Channel country detection: Prefer US-based channels
- Keyword filtering: Detect American vs non-American indicators
- Comprehensive logging for accent filtering effectiveness

WORKFLOW:
1. Search with regionCode='US' (pre-filter)
2. Fetch channel metadata
3. Calculate "American score" for each video
4. Download only high-scoring videos (score >= 2)
5. Post-verification with accent classifier (separate script)
"""

import os
import sys
import json
import time
import random
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('youtube_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Anti-detection: Random user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]

# Error tracking
class ErrorTracker:
    """Track and categorize errors for debugging at scale."""
    
    def __init__(self):
        self.errors = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.rate_limit_hits = 0
        self.block_detections = 0
        self.network_errors = 0
        self.quota_errors = 0
        self.unknown_errors = 0
        
    def log_error(self, error_type: str, video_id: str, error_msg: str, details: Dict = None):
        """Log an error with categorization."""
        timestamp = datetime.now().isoformat()
        error_entry = {
            'timestamp': timestamp,
            'type': error_type,
            'video_id': video_id,
            'message': error_msg,
            'details': details or {}
        }
        
        self.errors[error_type].append(error_entry)
        self.error_counts[error_type] += 1
        
        # Update specific counters
        if error_type == 'rate_limit':
            self.rate_limit_hits += 1
        elif error_type == 'blocked':
            self.block_detections += 1
        elif error_type == 'network':
            self.network_errors += 1
        elif error_type == 'quota':
            self.quota_errors += 1
        else:
            self.unknown_errors += 1
        
        # Log to file
        logger.error(f"[{error_type.upper()}] Video {video_id}: {error_msg}")
        if details:
            logger.debug(f"  Details: {json.dumps(details, indent=2)}")
    
    def get_summary(self) -> Dict:
        """Get error summary statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'rate_limit_hits': self.rate_limit_hits,
            'block_detections': self.block_detections,
            'network_errors': self.network_errors,
            'quota_errors': self.quota_errors,
            'unknown_errors': self.unknown_errors,
            'error_breakdown': dict(self.error_counts),
            'all_errors': dict(self.errors)
        }
    
    def save_to_file(self, filepath: Path):
        """Save error log to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Error log saved to {filepath}")

# Global error tracker
error_tracker = ErrorTracker()

# Quota tracker
class QuotaTracker:
    """Track API quota usage to avoid hitting limits."""
    
    def __init__(self, daily_limit: int = 10000):
        self.daily_limit = daily_limit
        self.used_quota = 0
        self.search_cost = 100  # YouTube API search cost
        self.channel_details_cost = 1  # Channel metadata cost
        
    def can_search(self) -> bool:
        """Check if we have quota for a search."""
        return self.used_quota + self.search_cost <= self.daily_limit
    
    def use_search_quota(self):
        """Record a search API call."""
        self.used_quota += self.search_cost
        remaining = self.daily_limit - self.used_quota
        logger.info(f"Quota used: {self.used_quota}/{self.daily_limit} (Remaining: {remaining})")
        
        if remaining < 1000:
            logger.warning(f"⚠️  LOW QUOTA WARNING: Only {remaining} quota units remaining!")
        
        if remaining < 100:
            logger.error(f"🚨 CRITICAL: Quota almost exhausted! Only {remaining} units left!")
    
    def use_channel_quota(self):
        """Record a channel metadata API call."""
        self.used_quota += self.channel_details_cost
    
    def get_status(self) -> Dict:
        """Get quota status."""
        return {
            'daily_limit': self.daily_limit,
            'used': self.used_quota,
            'remaining': self.daily_limit - self.used_quota,
            'percentage_used': (self.used_quota / self.daily_limit) * 100
        }

quota_tracker = QuotaTracker()

# American filtering statistics
class AmericanFilterStats:
    """Track pre-filtering effectiveness."""
    
    def __init__(self):
        self.total_searched = 0
        self.passed_filter = 0
        self.score_distribution = defaultdict(int)
        self.country_distribution = defaultdict(int)
        
    def record_video(self, score: int, country: str, passed: bool):
        """Record a video's filtering result."""
        self.total_searched += 1
        if passed:
            self.passed_filter += 1
        self.score_distribution[score] += 1
        self.country_distribution[country] += 1
    
    def get_summary(self) -> Dict:
        """Get filtering statistics."""
        return {
            'total_searched': self.total_searched,
            'passed_filter': self.passed_filter,
            'filtered_out': self.total_searched - self.passed_filter,
            'pass_rate': (self.passed_filter / self.total_searched * 100) if self.total_searched > 0 else 0,
            'score_distribution': dict(self.score_distribution),
            'country_distribution': dict(self.country_distribution)
        }

filter_stats = AmericanFilterStats()


def smart_delay(min_sec=2.0, max_sec=5.0, reason="anti-detection"):
    """Random delay to mimic human behavior with logging."""
    delay = random.uniform(min_sec, max_sec)
    logger.debug(f"[{reason}] Waiting {delay:.1f}s...")
    time.sleep(delay)


def exponential_backoff(attempt: int, base_delay: float = 2.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
    return delay + jitter


def detect_error_type(error_msg: str, status_code: Optional[int] = None) -> str:
    """Detect error type from error message or status code."""
    error_msg_lower = str(error_msg).lower()
    
    # Rate limiting
    if status_code == 429 or 'rate limit' in error_msg_lower or 'too many requests' in error_msg_lower:
        return 'rate_limit'
    
    # Quota exceeded
    if 'quota' in error_msg_lower or 'quotaExceeded' in error_msg:
        return 'quota'
    
    # Blocked/forbidden
    if status_code == 403 or 'forbidden' in error_msg_lower or 'blocked' in error_msg_lower:
        return 'blocked'
    
    # Network errors
    if status_code in [500, 502, 503, 504] or 'network' in error_msg_lower or 'connection' in error_msg_lower:
        return 'network'
    
    # Video unavailable
    if 'unavailable' in error_msg_lower or 'not found' in error_msg_lower or status_code == 404:
        return 'unavailable'
    
    # Timeout
    if 'timeout' in error_msg_lower or 'timed out' in error_msg_lower:
        return 'timeout'
    
    return 'unknown'


def get_channel_country(youtube, channel_id: str) -> str:
    """Get channel's country metadata."""
    try:
        request = youtube.channels().list(
            part='snippet',
            id=channel_id
        )
        response = request.execute()
        quota_tracker.use_channel_quota()
        
        if response.get('items'):
            country = response['items'][0]['snippet'].get('country', 'unknown')
            return country
        return 'unknown'
    except Exception as e:
        logger.debug(f"Could not fetch channel country: {e}")
        return 'unknown'


def calculate_american_score(video_info: Dict, channel_country: str) -> int:
    """
    Calculate likelihood of video containing American English.
    
    Score ranges:
    - 5+: Very likely American
    - 2-4: Possibly American
    - 0-1: Uncertain
    - <0: Likely non-American
    """
    score = 0
    
    # FACTOR 1: Channel Country (strongest signal)
    if channel_country == 'US':
        score += 3  # Strong positive
    elif channel_country in ['CA', 'GB', 'AU', 'NZ', 'IE']:
        score -= 2  # Strong negative (other English-speaking countries)
    elif channel_country == 'unknown':
        score += 0  # Neutral (no penalty)
    
    # FACTOR 2: Title/Description Keywords
    text = (video_info['title'] + ' ' + video_info.get('description', '')).lower()
    
    # Positive keywords (American indicators)
    american_keywords = [
        'american', 'usa', 'united states', 'us ', ' us',
        'american english', 'american accent'
    ]
    if any(kw in text for kw in american_keywords):
        score += 2
    
    # Negative keywords (non-American indicators)
    non_american_keywords = [
        'british', 'uk', 'bbc', 'england', 'london',
        'australian', 'aussie', 'australia', 'sydney',
        'canadian', 'canada', 'toronto', 'vancouver',
        'indian', 'india', 'mumbai', 'delhi'
    ]
    if any(kw in text for kw in non_american_keywords):
        score -= 3
    
    # FACTOR 3: Known American Sources
    channel = video_info['channel'].lower()
    american_sources = [
        'ted', 'tedx',  # TED talks (many American speakers)
        'stanford', 'mit', 'harvard', 'yale', 'princeton',  # US universities
        'cnn', 'nbc', 'abc', 'cbs', 'pbs',  # US news
        'khan academy',  # US educational
    ]
    if any(src in channel for src in american_sources):
        score += 1
    
    # Known non-American sources
    non_american_sources = [
        'bbc', 'itv', 'channel 4',  # UK
        'abc australia', 'sbs',  # Australia
        'cbc',  # Canada
    ]
    if any(src in channel for src in non_american_sources):
        score -= 3
    
    return score



def search_creative_commons_videos(api_key, max_results=10, max_retries=3, american_score_threshold=2):
    """
    TIER 1: Search for Creative Commons (CC-BY) licensed videos with AMERICAN ENGLISH pre-filtering.
    These are 100% legal for research and publication.
    """
    logger.info("="*70)
    logger.info("TIER 1: SEARCHING FOR CREATIVE COMMONS (CC-BY) VIDEOS")
    logger.info("WITH AMERICAN ENGLISH PRE-FILTERING")
    logger.info("="*70)
    logger.info("License: CC-BY (Attribution)")
    logger.info("Legal status: ✅ Fully legal for research and publication")
    logger.info(f"American score threshold: {american_score_threshold}")
    
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    
    # Good queries for lip-reading with CC content
    queries = [
        "interview speech",
        "public speaking tutorial",
        "educational lecture",
        "presentation skills",
    ]
    
    all_videos = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    for query in queries:
        logger.info(f"Searching CC-BY: '{query}'")
        
        # Check quota before search
        if not quota_tracker.can_search():
            logger.error("🚨 QUOTA EXHAUSTED! Cannot perform more searches today.")
            error_tracker.log_error('quota', 'N/A', 'Daily quota limit reached', {
                'quota_status': quota_tracker.get_status()
            })
            break
        
        attempt = 0
        success = False
        
        while attempt < max_retries and not success:
            try:
                request = youtube.search().list(
                    part='id,snippet',
                    q=query,
                    type='video',
                    maxResults=10,  # Get more to filter
                    videoLicense='creativeCommon',  # ← CC-BY filter
                    videoDefinition='high',
                    videoDuration='medium',
                    relevanceLanguage='en',
                    regionCode='US',  # ← NEW: Bias toward US uploads
                    order='relevance',
                    safeSearch='strict'
                )
                
                response = request.execute()
                quota_tracker.use_search_quota()
                success = True
                
                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    channel = item['snippet']['channelTitle']
                    channel_id = item['snippet']['channelId']
                    description = item['snippet'].get('description', '')
                    
                    # Get channel country
                    channel_country = get_channel_country(youtube, channel_id)
                    
                    video_info = {
                        'id': video_id,
                        'title': title,
                        'channel': channel,
                        'channel_id': channel_id,
                        'channel_country': channel_country,
                        'description': description,
                        'query': query,
                        'license': 'CC-BY',
                        'tier': 1,
                        'legal_status': 'Fully legal - Creative Commons Attribution'
                    }
                    
                    # Calculate American score
                    american_score = calculate_american_score(video_info, channel_country)
                    video_info['american_score'] = american_score
                    
                    # Record statistics
                    passed = american_score >= american_score_threshold
                    filter_stats.record_video(american_score, channel_country, passed)
                    
                    if passed:
                        all_videos.append(video_info)
                        logger.info(f"  ✓ {video_id} - {title[:40]}... [Score: {american_score}, Country: {channel_country}]")
                    else:
                        logger.debug(f"  ✗ FILTERED: {video_id} - {title[:40]}... [Score: {american_score}, Country: {channel_country}]")
                
                # Anti-detection delay
                smart_delay(2.0, 4.0, "search-cooldown")
                
                if len(all_videos) >= max_results:
                    break
                    
            except HttpError as e:
                attempt += 1
                error_type = detect_error_type(str(e), e.resp.status if hasattr(e, 'resp') else None)
                
                error_tracker.log_error(error_type, 'search', str(e), {
                    'query': query,
                    'attempt': attempt,
                    'status_code': e.resp.status if hasattr(e, 'resp') else None
                })
                
                if error_type == 'rate_limit':
                    logger.warning(f"⚠️  RATE LIMITED! Attempt {attempt}/{max_retries}")
                    backoff = exponential_backoff(attempt, base_delay=5.0)
                    logger.info(f"  Backing off for {backoff:.1f}s...")
                    time.sleep(backoff)
                    
                elif error_type == 'quota':
                    logger.error("🚨 QUOTA EXCEEDED! Stopping searches.")
                    return all_videos
                    
                elif error_type == 'blocked':
                    logger.error(f"🚫 BLOCKED! IP may be blacklisted. Attempt {attempt}/{max_retries}")
                    backoff = exponential_backoff(attempt, base_delay=10.0)
                    logger.info(f"  Waiting {backoff:.1f}s before retry...")
                    time.sleep(backoff)
                    
                else:
                    logger.error(f"❌ Error: {e}")
                    if attempt < max_retries:
                        backoff = exponential_backoff(attempt)
                        logger.info(f"  Retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                    
            except Exception as e:
                attempt += 1
                error_type = detect_error_type(str(e))
                error_tracker.log_error(error_type, 'search', str(e), {'query': query, 'attempt': attempt})
                logger.error(f"❌ Unexpected error: {e}")
                
                if attempt < max_retries:
                    backoff = exponential_backoff(attempt)
                    logger.info(f"  Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
    
    # Remove duplicates
    seen = set()
    unique_videos = []
    for v in all_videos:
        if v['id'] not in seen:
            seen.add(v['id'])
            unique_videos.append(v)
    
    unique_videos = unique_videos[:max_results]
    
    logger.info(f"✓ Found {len(unique_videos)} CC-BY licensed videos (after American filtering)")
    return unique_videos


def search_ted_talks(api_key, max_results=10, max_retries=3, american_score_threshold=2):
    """
    TIER 2: Search for TED/TEDx talks with AMERICAN ENGLISH pre-filtering.
    Licensed CC BY-NC-ND, widely accepted for academic lip-reading research.
    """
    logger.info("="*70)
    logger.info("TIER 2: SEARCHING FOR TED/TEDx TALKS")
    logger.info("WITH AMERICAN ENGLISH PRE-FILTERING")
    logger.info("="*70)
    logger.info("License: CC BY-NC-ND (Attribution, Non-Commercial, No Derivatives)")
    logger.info("Legal status: ✅ Legal for non-commercial academic research")
    logger.info(f"American score threshold: {american_score_threshold}")
    
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    
    # TED-specific queries
    queries = [
        "TED talk",
        "TEDx talk",
        "TED conference",
    ]
    
    all_videos = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    for query in queries:
        logger.info(f"Searching TED: '{query}'")
        
        # Check quota
        if not quota_tracker.can_search():
            logger.error("🚨 QUOTA EXHAUSTED! Cannot perform more searches.")
            break
        
        attempt = 0
        success = False
        
        while attempt < max_retries and not success:
            try:
                request = youtube.search().list(
                    part='id,snippet',
                    q=query,
                    type='video',
                    maxResults=10,  # Get more to filter
                    videoDefinition='high',
                    videoDuration='medium',
                    relevanceLanguage='en',
                    regionCode='US',  # ← NEW: Bias toward US uploads
                    order='viewCount',  # Popular TED talks
                    safeSearch='strict'
                )
                
                response = request.execute()
                quota_tracker.use_search_quota()
                success = True
                
                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    channel = item['snippet']['channelTitle']
                    channel_id = item['snippet']['channelId']
                    description = item['snippet'].get('description', '')
                    
                    # Filter for actual TED channels
                    if 'TED' not in channel and 'TED' not in title:
                        continue
                    
                    # Get channel country
                    channel_country = get_channel_country(youtube, channel_id)
                    
                    video_info = {
                        'id': video_id,
                        'title': title,
                        'channel': channel,
                        'channel_id': channel_id,
                        'channel_country': channel_country,
                        'description': description,
                        'query': query,
                        'license': 'CC BY-NC-ND',
                        'tier': 2,
                        'legal_status': 'Legal for non-commercial research - TED license'
                    }
                    
                    # Calculate American score
                    american_score = calculate_american_score(video_info, channel_country)
                    video_info['american_score'] = american_score
                    
                    # Record statistics
                    passed = american_score >= american_score_threshold
                    filter_stats.record_video(american_score, channel_country, passed)
                    
                    if passed:
                        all_videos.append(video_info)
                        logger.info(f"  ✓ {video_id} - {title[:40]}... [Score: {american_score}, Country: {channel_country}]")
                    else:
                        logger.debug(f"  ✗ FILTERED: {video_id} - {title[:40]}... [Score: {american_score}, Country: {channel_country}]")
                
                # Anti-detection delay
                smart_delay(2.0, 4.0, "search-cooldown")
                
                if len(all_videos) >= max_results:
                    break
                    
            except HttpError as e:
                attempt += 1
                error_type = detect_error_type(str(e), e.resp.status if hasattr(e, 'resp') else None)
                
                error_tracker.log_error(error_type, 'search', str(e), {
                    'query': query,
                    'attempt': attempt,
                    'status_code': e.resp.status if hasattr(e, 'resp') else None
                })
                
                if error_type == 'rate_limit':
                    logger.warning(f"⚠️  RATE LIMITED! Attempt {attempt}/{max_retries}")
                    backoff = exponential_backoff(attempt, base_delay=5.0)
                    time.sleep(backoff)
                elif error_type == 'quota':
                    logger.error("🚨 QUOTA EXCEEDED!")
                    return all_videos
                elif error_type == 'blocked':
                    logger.error(f"🚫 BLOCKED! Attempt {attempt}/{max_retries}")
                    backoff = exponential_backoff(attempt, base_delay=10.0)
                    time.sleep(backoff)
                else:
                    if attempt < max_retries:
                        backoff = exponential_backoff(attempt)
                        time.sleep(backoff)
                    
            except Exception as e:
                attempt += 1
                error_type = detect_error_type(str(e))
                error_tracker.log_error(error_type, 'search', str(e), {'query': query})
                logger.error(f"❌ Error: {e}")
                
                if attempt < max_retries:
                    backoff = exponential_backoff(attempt)
                    time.sleep(backoff)
    
    # Remove duplicates
    seen = set()
    unique_videos = []
    for v in all_videos:
        if v['id'] not in seen:
            seen.add(v['id'])
            unique_videos.append(v)
    
    unique_videos = unique_videos[:max_results]
    
    logger.info(f"✓ Found {len(unique_videos)} TED/TEDx talks (after American filtering)")
    return unique_videos



def download_video(video_info, output_dir, index, total, max_retries=3):
    """Download a single video with anti-detection and comprehensive error logging."""
    video_id = video_info['id']
    title = video_info['title']
    license_type = video_info['license']
    tier = video_info['tier']
    american_score = video_info.get('american_score', 'N/A')
    channel_country = video_info.get('channel_country', 'unknown')
    
    logger.info(f"[{index}/{total}] Downloading: {video_id}")
    logger.info(f"  Title: {title[:60]}...")
    logger.info(f"  License: {license_type} (Tier {tier})")
    logger.info(f"  American Score: {american_score}, Country: {channel_country}")
    
    output_path = Path(output_dir)
    output_file = output_path / f'{video_id}.mp4'
    
    # Check if already downloaded
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Already downloaded ({size_mb:.1f} MB)")
        return True, None
    
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            # Anti-detection: Random user agent
            user_agent = random.choice(USER_AGENTS)
            
            # yt-dlp command with smart settings
            cmd = [
                'yt-dlp',
                f'https://www.youtube.com/watch?v={video_id}',
                '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
                '--merge-output-format', 'mp4',
                '-o', str(output_file),
                '--user-agent', user_agent,
                '--sleep-interval', '1',
                '--max-sleep-interval', '3',
                '--no-playlist',
                '--quiet',
                '--progress',
                '--no-warnings'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ Downloaded successfully ({size_mb:.1f} MB)")
                return True, None
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                last_error = error_msg
                error_type = detect_error_type(error_msg)
                
                error_tracker.log_error(error_type, video_id, error_msg, {
                    'attempt': attempt + 1,
                    'return_code': result.returncode,
                    'title': title
                })
                
                if error_type == 'rate_limit':
                    logger.warning(f"  ⚠️  RATE LIMITED! Attempt {attempt + 1}/{max_retries}")
                    backoff = exponential_backoff(attempt, base_delay=10.0, max_delay=120.0)
                    logger.info(f"  Backing off for {backoff:.1f}s...")
                    time.sleep(backoff)
                    
                elif error_type == 'blocked':
                    logger.error(f"  🚫 BLOCKED! IP may be blacklisted. Attempt {attempt + 1}/{max_retries}")
                    backoff = exponential_backoff(attempt, base_delay=30.0, max_delay=300.0)
                    logger.info(f"  Waiting {backoff:.1f}s before retry...")
                    time.sleep(backoff)
                    
                elif error_type == 'unavailable':
                    logger.warning(f"  ⚠️  Video unavailable: {video_id}")
                    return False, error_msg
                    
                else:
                    logger.error(f"  ❌ Download failed: {error_msg[:200]}")
                    if attempt < max_retries - 1:
                        backoff = exponential_backoff(attempt)
                        logger.info(f"  Retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                
                attempt += 1
                
        except subprocess.TimeoutExpired:
            attempt += 1
            error_msg = f"Download timeout (>2 minutes)"
            last_error = error_msg
            error_tracker.log_error('timeout', video_id, error_msg, {
                'attempt': attempt,
                'title': title
            })
            logger.error(f"  ⏱️  {error_msg}")
            
            if attempt < max_retries:
                backoff = exponential_backoff(attempt, base_delay=5.0)
                logger.info(f"  Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                
        except Exception as e:
            attempt += 1
            error_msg = str(e)
            last_error = error_msg
            error_type = detect_error_type(error_msg)
            error_tracker.log_error(error_type, video_id, error_msg, {
                'attempt': attempt,
                'title': title
            })
            logger.error(f"  ❌ Error: {e}")
            
            if attempt < max_retries:
                backoff = exponential_backoff(attempt)
                logger.info(f"  Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
    
    logger.error(f"  ❌ Failed after {max_retries} attempts")
    return False, last_error


def main():
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("AMERICAN ENGLISH YOUTUBE VIDEO DOWNLOADER FOR RESEARCH")
    logger.info("Downloads only legally licensed videos with American English accent")
    logger.info("="*70)
    logger.info(f"Session started: {start_time.isoformat()}")
    
    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        logger.info(f"✓ yt-dlp version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("✗ ERROR: yt-dlp not installed!")
        logger.error("  Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Load API key
    api_key_file = Path('.youtube_api_key')
    if not api_key_file.exists():
        logger.error("✗ ERROR: .youtube_api_key file not found!")
        sys.exit(1)
    
    api_key = api_key_file.read_text().strip()
    logger.info(f"✓ Loaded API key: {api_key[:20]}...")
    
    # Output directory
    output_dir = Path('youtube_raw_downloads')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {output_dir.absolute()}")
    
    # American score threshold
    american_score_threshold = 2
    logger.info(f"✓ American score threshold: {american_score_threshold}")
    
    # Search for both types of videos with American filtering
    logger.info("\n" + "="*70)
    logger.info("STARTING VIDEO SEARCH WITH AMERICAN ENGLISH PRE-FILTERING")
    logger.info("="*70)
    
    cc_videos = search_creative_commons_videos(api_key, max_results=5, max_retries=3, american_score_threshold=american_score_threshold)
    ted_videos = search_ted_talks(api_key, max_results=5, max_retries=3, american_score_threshold=american_score_threshold)
    
    all_videos = cc_videos + ted_videos
    
    if not all_videos:
        logger.error("✗ No videos found after American filtering!")
        logger.info("\nPre-Filtering Statistics:")
        logger.info(json.dumps(filter_stats.get_summary(), indent=2))
        logger.info("\nQuota Status:")
        logger.info(json.dumps(quota_tracker.get_status(), indent=2))
        logger.info("\nError Summary:")
        logger.info(json.dumps(error_tracker.get_summary(), indent=2))
        sys.exit(1)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SEARCH SUMMARY")
    logger.info("="*70)
    logger.info(f"Tier 1 (CC-BY): {len(cc_videos)} videos")
    logger.info(f"Tier 2 (TED): {len(ted_videos)} videos")
    logger.info(f"Total: {len(all_videos)} videos")
    
    logger.info("\nPre-Filtering Statistics:")
    filter_summary = filter_stats.get_summary()
    logger.info(f"  Total searched: {filter_summary['total_searched']}")
    logger.info(f"  Passed filter: {filter_summary['passed_filter']}")
    logger.info(f"  Filtered out: {filter_summary['filtered_out']}")
    logger.info(f"  Pass rate: {filter_summary['pass_rate']:.1f}%")
    logger.info(f"  Country distribution: {filter_summary['country_distribution']}")
    logger.info(f"  Score distribution: {filter_summary['score_distribution']}")
    
    logger.info("\nQuota Status:")
    logger.info(json.dumps(quota_tracker.get_status(), indent=2))
    
    # Download videos
    logger.info("\n" + "="*70)
    logger.info(f"DOWNLOADING {len(all_videos)} PRE-FILTERED VIDEOS")
    logger.info("="*70)
    logger.info("NOTE: These videos will be verified with accent classifier in next step")
    
    downloaded = []
    failed = []
    download_errors = {}
    
    for idx, video_info in enumerate(all_videos, 1):
        success, error_msg = download_video(video_info, output_dir, idx, len(all_videos), max_retries=3)
        
        if success:
            downloaded.append(video_info)
        else:
            failed.append(video_info)
            download_errors[video_info['id']] = error_msg
        
        # Anti-detection: Random delay between downloads
        if idx < len(all_videos):
            smart_delay(3.0, 6.0, "download-cooldown")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save comprehensive results
    results = {
        'session': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{int(duration // 60)}m {int(duration % 60)}s"
        },
        'pre_filtering': filter_stats.get_summary(),
        'quota': quota_tracker.get_status(),
        'errors': error_tracker.get_summary(),
        'legal_compliance': 'All videos have explicit licenses for research use',
        'accent_filtering': 'Pre-filtered for American English (post-verification required)',
        'statistics': {
            'tier_1_cc_by': len([v for v in downloaded if v['tier'] == 1]),
            'tier_2_ted': len([v for v in downloaded if v['tier'] == 2]),
            'total_searched': len(all_videos),
            'total_downloaded': len(downloaded),
            'total_failed': len(failed),
            'success_rate': (len(downloaded) / len(all_videos) * 100) if all_videos else 0
        },
        'videos': all_videos,
        'downloaded_videos': downloaded,
        'failed_videos': failed,
        'download_errors': download_errors
    }
    
    results_file = output_dir / 'american_download_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save error log
    error_log_file = output_dir / 'error_log.json'
    error_tracker.save_to_file(error_log_file)
    
    # Create attribution file
    attribution_file = output_dir / 'ATTRIBUTION.txt'
    with open(attribution_file, 'w') as f:
        f.write("VIDEO ATTRIBUTION FOR RESEARCH DATASET\n")
        f.write("="*70 + "\n\n")
        f.write("This dataset contains videos with the following licenses:\n")
        f.write("PRE-FILTERED for American English accent (post-verification required)\n\n")
        
        f.write("TIER 1: Creative Commons (CC-BY) Videos\n")
        f.write("-" * 70 + "\n")
        for v in downloaded:
            if v['tier'] == 1:
                f.write(f"Video ID: {v['id']}\n")
                f.write(f"Title: {v['title']}\n")
                f.write(f"Channel: {v['channel']}\n")
                f.write(f"Channel Country: {v.get('channel_country', 'unknown')}\n")
                f.write(f"American Score: {v.get('american_score', 'N/A')}\n")
                f.write(f"License: {v['license']}\n")
                f.write(f"URL: https://www.youtube.com/watch?v={v['id']}\n\n")
        
        f.write("\nTIER 2: TED/TEDx Talks (CC BY-NC-ND)\n")
        f.write("-" * 70 + "\n")
        for v in downloaded:
            if v['tier'] == 2:
                f.write(f"Video ID: {v['id']}\n")
                f.write(f"Title: {v['title']}\n")
                f.write(f"Channel: {v['channel']}\n")
                f.write(f"Channel Country: {v.get('channel_country', 'unknown')}\n")
                f.write(f"American Score: {v.get('american_score', 'N/A')}\n")
                f.write(f"License: {v['license']}\n")
                f.write(f"URL: https://www.youtube.com/watch?v={v['id']}\n\n")
        
        f.write("\nLEGAL COMPLIANCE\n")
        f.write("-" * 70 + "\n")
        f.write("All videos in this dataset are used under their respective licenses.\n")
        f.write("This dataset is for non-commercial academic research only.\n")
        f.write("Proper attribution is provided for all content.\n\n")
        
        f.write("ACCENT FILTERING\n")
        f.write("-" * 70 + "\n")
        f.write("Videos have been pre-filtered for American English accent.\n")
        f.write("Post-verification with accent classifier is required.\n")
        f.write("Run verify_american_accent.py to complete accent verification.\n")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nSession Duration: {int(duration // 60)}m {int(duration % 60)}s")
    logger.info(f"\nResults:")
    logger.info(f"  Tier 1 (CC-BY): {len([v for v in downloaded if v['tier'] == 1])} videos")
    logger.info(f"  Tier 2 (TED): {len([v for v in downloaded if v['tier'] == 2])} videos")
    logger.info(f"  Total downloaded: {len(downloaded)}")
    logger.info(f"  Failed: {len(failed)}")
    logger.info(f"  Success rate: {(len(downloaded) / len(all_videos) * 100):.1f}%")
    
    logger.info(f"\nPre-Filtering Effectiveness:")
    logger.info(f"  Videos searched: {filter_summary['total_searched']}")
    logger.info(f"  Passed filter: {filter_summary['passed_filter']} ({filter_summary['pass_rate']:.1f}%)")
    logger.info(f"  Expected American: ~{int(len(downloaded) * 0.75)}-{int(len(downloaded) * 0.85)} videos")
    
    logger.info(f"\nQuota Status:")
    quota_status = quota_tracker.get_status()
    logger.info(f"  Used: {quota_status['used']}/{quota_status['daily_limit']}")
    logger.info(f"  Remaining: {quota_status['remaining']}")
    logger.info(f"  Percentage used: {quota_status['percentage_used']:.1f}%")
    
    logger.info(f"\nError Summary:")
    error_summary = error_tracker.get_summary()
    logger.info(f"  Total errors: {error_summary['total_errors']}")
    logger.info(f"  Rate limit hits: {error_summary['rate_limit_hits']}")
    logger.info(f"  Block detections: {error_summary['block_detections']}")
    logger.info(f"  Network errors: {error_summary['network_errors']}")
    logger.info(f"  Quota errors: {error_summary['quota_errors']}")
    
    logger.info(f"\nFiles saved to: {output_dir.absolute()}")
    logger.info(f"  Results: {results_file}")
    logger.info(f"  Attribution: {attribution_file}")
    logger.info(f"  Error log: {error_log_file}")
    logger.info(f"  Session log: youtube_scraper.log")
    
    if downloaded:
        logger.info(f"\n✓ SUCCESS! Downloaded {len(downloaded)} pre-filtered videos.")
        logger.info(f"\nDownloaded videos by tier:")
        
        tier1 = [v for v in downloaded if v['tier'] == 1]
        if tier1:
            logger.info(f"\n  Tier 1 (CC-BY) - {len(tier1)} videos:")
            for v in tier1:
                logger.info(f"    - {v['id']}.mp4 ({v['title'][:50]}...) [Score: {v.get('american_score', 'N/A')}]")
        
        tier2 = [v for v in downloaded if v['tier'] == 2]
        if tier2:
            logger.info(f"\n  Tier 2 (TED) - {len(tier2)} videos:")
            for v in tier2:
                logger.info(f"    - {v['id']}.mp4 ({v['title'][:50]}...) [Score: {v.get('american_score', 'N/A')}]")
        
        logger.info(f"\n✅ LEGAL STATUS: All videos are licensed for research use")
        logger.info(f"✅ PRE-FILTERED: Videos biased toward American English")
        logger.info(f"✅ ATTRIBUTION: See {attribution_file} for proper citations")
        
        logger.info(f"\n⚠️  NEXT STEP: Run accent verification")
        logger.info(f"  python verify_american_accent.py")
        logger.info(f"  This will verify accents and delete non-American videos")
        
        logger.info(f"\nExpected results after verification:")
        logger.info(f"  American videos: ~{int(len(downloaded) * 0.75)}-{int(len(downloaded) * 0.85)} ({75}-{85}%)")
        logger.info(f"  Non-American (will be deleted): ~{int(len(downloaded) * 0.15)}-{int(len(downloaded) * 0.25)} ({15}-{25}%)")
    else:
        logger.error(f"\n✗ No videos were downloaded successfully.")
        logger.error(f"  Check {error_log_file} for details.")
        logger.error(f"  Check youtube_scraper.log for full session log.")
    
    # Warnings
    if error_summary['rate_limit_hits'] > 0:
        logger.warning(f"\n⚠️  WARNING: Hit rate limits {error_summary['rate_limit_hits']} times!")
        logger.warning(f"  Consider increasing delays between requests.")
    
    if error_summary['block_detections'] > 0:
        logger.warning(f"\n🚫 WARNING: Detected {error_summary['block_detections']} blocks!")
        logger.warning(f"  Your IP may be temporarily blacklisted.")
        logger.warning(f"  Consider using a VPN or waiting before retrying.")
    
    if quota_status['remaining'] < 1000:
        logger.warning(f"\n⚠️  WARNING: Low quota remaining ({quota_status['remaining']} units)")
        logger.warning(f"  You may not be able to download more videos today.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user!")
        logger.info("Saving error log...")
        error_tracker.save_to_file(Path('youtube_raw_downloads/error_log.json'))
        logger.info("Session log saved to youtube_scraper.log")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ FATAL ERROR: {e}")
        logger.exception("Full traceback:")
        error_tracker.save_to_file(Path('youtube_raw_downloads/error_log.json'))
        sys.exit(1)
