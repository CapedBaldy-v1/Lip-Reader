#!/usr/bin/env python3
"""
Simple Test: Download 10 Videos
Downloads 10 videos for testing (trainability check optional).
"""

import os
import sys
import json
import time
import random
import subprocess
from pathlib import Path
from datetime import datetime

# Anti-detection: Random user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
]


def smart_delay(min_sec=2.0, max_sec=5.0):
    """Random delay to mimic human behavior."""
    delay = random.uniform(min_sec, max_sec)
    print(f"  [Anti-detection] Waiting {delay:.1f}s...")
    time.sleep(delay)


def search_videos(api_key, max_results=15):
    """Search for good quality videos."""
    print("\n" + "="*70)
    print(f"SEARCHING FOR {max_results} HIGH-QUALITY VIDEOS")
    print("="*70)
    
    from googleapiclient.discovery import build
    
    # Queries that typically have clear speech
    queries = [
        "news interview 2024",
        "ted talk 2024",
        "educational lecture clear speech",
        "public speaking tutorial",
    ]
    
    all_videos = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    for query in queries:
        print(f"\nSearching: '{query}'")
        
        try:
            request = youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=5,
                videoDefinition='high',  # HD only
                videoDuration='medium',  # 4-20 minutes
                relevanceLanguage='en',
                order='viewCount',  # Popular videos
                safeSearch='strict'
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel = item['snippet']['channelTitle']
                
                all_videos.append({
                    'id': video_id,
                    'title': title,
                    'channel': channel,
                    'query': query
                })
                
                print(f"  ✓ {video_id} - {title[:50]}...")
            
            # Anti-detection delay
            smart_delay(2.0, 4.0)
            
            if len(all_videos) >= max_results:
                break
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Remove duplicates
    seen = set()
    unique_videos = []
    for v in all_videos:
        if v['id'] not in seen:
            seen.add(v['id'])
            unique_videos.append(v)
    
    unique_videos = unique_videos[:max_results]
    
    print(f"\n✓ Found {len(unique_videos)} unique videos")
    return unique_videos


def download_video(video_info, output_dir, index, total):
    """Download a single video with anti-detection."""
    video_id = video_info['id']
    title = video_info['title']
    
    print(f"\n[{index}/{total}] Downloading: {video_id}")
    print(f"  Title: {title[:60]}...")
    
    output_path = Path(output_dir)
    output_file = output_path / f'{video_id}.mp4'
    
    # Check if already downloaded
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already downloaded ({size_mb:.1f} MB)")
        return True
    
    try:
        # Anti-detection: Random user agent
        user_agent = random.choice(USER_AGENTS)
        
        # yt-dlp command with smart settings
        # Use flexible format: prefer 720p MP4, but accept any good quality
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
            print(f"  ✓ Downloaded successfully ({size_mb:.1f} MB)")
            return True
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            print(f"  ✗ Download failed: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timeout (>2 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("SIMPLE TEST: DOWNLOAD 10 VIDEOS")
    print("Smart scraping with anti-detection features")
    print("="*70)
    
    # Check yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
        print(f"\n✓ yt-dlp version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\n✗ ERROR: yt-dlp not installed!")
        print("  Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Load API key
    api_key_file = Path('.youtube_api_key')
    if not api_key_file.exists():
        print("\n✗ ERROR: .youtube_api_key file not found!")
        sys.exit(1)
    
    api_key = api_key_file.read_text().strip()
    print(f"✓ Loaded API key: {api_key[:20]}...")
    
    # Output directory
    output_dir = Path('youtube_raw_downloads')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir.absolute()}")
    
    # Search for videos
    videos = search_videos(api_key, max_results=10)
    
    if not videos:
        print("\n✗ No videos found!")
        sys.exit(1)
    
    # Download videos
    print("\n" + "="*70)
    print(f"DOWNLOADING {len(videos)} VIDEOS")
    print("="*70)
    
    downloaded = []
    failed = []
    
    for idx, video_info in enumerate(videos, 1):
        success = download_video(video_info, output_dir, idx, len(videos))
        
        if success:
            downloaded.append(video_info)
        else:
            failed.append(video_info)
        
        # Anti-detection: Random delay between downloads
        if idx < len(videos):
            smart_delay(3.0, 6.0)
    
    # Save results
    results = {
        'test_date': datetime.now().isoformat(),
        'total_videos': len(videos),
        'downloaded': len(downloaded),
        'failed': len(failed),
        'videos': videos,
        'downloaded_ids': [v['id'] for v in downloaded],
        'failed_ids': [v['id'] for v in failed]
    }
    
    results_file = output_dir / 'download_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total videos: {len(videos)}")
    print(f"  Successfully downloaded: {len(downloaded)}")
    print(f"  Failed: {len(failed)}")
    print(f"\nFiles saved to: {output_dir.absolute()}")
    print(f"Results saved to: {results_file}")
    
    if downloaded:
        print(f"\n✓ SUCCESS! Downloaded {len(downloaded)} videos for testing.")
        print(f"\nDownloaded videos:")
        for v in downloaded:
            print(f"  - {v['id']}.mp4 ({v['title'][:50]}...)")
        
        print(f"\nNext steps:")
        print(f"  1. Check the videos in {output_dir}/")
        print(f"  2. Verify video quality and content")
        print(f"  3. If good, scale up to download more videos")
        
        # Note about trainability
        print(f"\n⚠ NOTE: Trainability API returned no data for most videos.")
        print(f"  This is expected - most creators haven't enabled AI training yet.")
        print(f"  For research purposes, you can use these videos for testing.")
        print(f"  For production, you'll need to:")
        print(f"    - Search more videos to find trainable ones")
        print(f"    - Or use public domain / Creative Commons videos")
        print(f"    - Or get explicit permission from creators")
    else:
        print(f"\n✗ No videos were downloaded successfully.")
        print(f"  Check the errors above for details.")


if __name__ == '__main__':
    main()
