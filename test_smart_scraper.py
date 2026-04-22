#!/usr/bin/env python3
"""
Smart YouTube Test Scraper
Tests API and downloads 10 videos with anti-detection features.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

# Anti-detection: Random user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
]


def smart_delay(min_sec=1.0, max_sec=3.0):
    """Random delay to mimic human behavior."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)


def get_random_user_agent():
    """Get random user agent."""
    return random.choice(USER_AGENTS)


def test_api_key(api_key):
    """Test if API key works."""
    print("\n" + "="*70)
    print("STEP 1: TESTING API KEY")
    print("="*70)
    
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        
        print("Building YouTube API client...")
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        print("Testing API with simple search...")
        request = youtube.search().list(
            part='id,snippet',
            q='test',
            type='video',
            maxResults=1
        )
        
        response = request.execute()
        
        if 'items' in response:
            print("✓ API key is VALID and working!")
            return True
        else:
            print("✗ API returned unexpected response")
            return False
            
    except HttpError as e:
        print(f"✗ API Error: {e}")
        if e.resp.status == 403:
            print("  → API key is invalid or quota exceeded")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def search_candidate_videos(api_key, max_results=20):
    """Search for candidate videos with anti-detection."""
    print("\n" + "="*70)
    print(f"STEP 2: SEARCHING FOR {max_results} CANDIDATE VIDEOS")
    print("="*70)
    
    from googleapiclient.discovery import build
    
    # Good queries for lip-reading
    queries = [
        "news anchor interview",
        "ted talk",
        "educational lecture",
    ]
    
    all_video_ids = []
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    for query in queries:
        print(f"\nSearching: '{query}'")
        
        try:
            request = youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=10,
                videoDefinition='high',
                videoDuration='medium',
                relevanceLanguage='en',
                order='relevance',
                safeSearch='strict'
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                all_video_ids.append(video_id)
                print(f"  Found: {video_id} - {title[:50]}...")
            
            # Anti-detection: Random delay between queries
            smart_delay(2.0, 4.0)
            
            if len(all_video_ids) >= max_results:
                break
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Remove duplicates
    unique_ids = list(dict.fromkeys(all_video_ids))[:max_results]
    
    print(f"\n✓ Found {len(unique_ids)} unique candidate videos")
    return unique_ids


def check_trainability(api_key, video_ids):
    """Check which videos are trainable with anti-detection."""
    print("\n" + "="*70)
    print(f"STEP 3: CHECKING TRAINABILITY FOR {len(video_ids)} VIDEOS")
    print("="*70)
    
    import requests
    
    trainable_ids = []
    
    for idx, video_id in enumerate(video_ids, 1):
        print(f"\n[{idx}/{len(video_ids)}] Checking {video_id}...")
        
        try:
            url = f"https://youtube.googleapis.com/youtube/v3/videoTrainability?id={video_id}&key={api_key}"
            
            # Anti-detection: Random user agent
            headers = {'User-Agent': get_random_user_agent()}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'items' in data and len(data['items']) > 0:
                    permitted = data['items'][0].get('permitted', 'none')
                    
                    if permitted == 'all':
                        print(f"  ✓ TRAINABLE (permitted: {permitted})")
                        trainable_ids.append(video_id)
                    else:
                        print(f"  ✗ Not trainable (permitted: {permitted})")
                else:
                    print(f"  ✗ No trainability data")
            else:
                print(f"  ✗ HTTP {response.status_code}")
            
            # Anti-detection: Random delay between checks
            smart_delay(0.5, 1.5)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n✓ Found {len(trainable_ids)} trainable videos out of {len(video_ids)}")
    print(f"  Trainability rate: {len(trainable_ids)/len(video_ids)*100:.1f}%")
    
    return trainable_ids


def download_videos(video_ids, output_dir, max_downloads=10):
    """Download videos with anti-detection using yt-dlp."""
    print("\n" + "="*70)
    print(f"STEP 4: DOWNLOADING {min(len(video_ids), max_downloads)} VIDEOS")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path.absolute()}")
    
    # Limit downloads
    video_ids = video_ids[:max_downloads]
    
    downloaded = []
    failed = []
    
    for idx, video_id in enumerate(video_ids, 1):
        print(f"\n[{idx}/{len(video_ids)}] Downloading {video_id}...")
        
        try:
            # Anti-detection: Random user agent
            user_agent = get_random_user_agent()
            
            # yt-dlp command with anti-detection
            cmd = [
                'yt-dlp',
                f'https://www.youtube.com/watch?v={video_id}',
                '-f', 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]',
                '-o', str(output_path / f'{video_id}.mp4'),
                '--user-agent', user_agent,
                '--sleep-interval', '2',  # Sleep 2 seconds between downloads
                '--max-sleep-interval', '5',  # Max 5 seconds
                '--no-playlist',
                '--no-warnings',
                '--quiet',
                '--progress'
            ]
            
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                video_file = output_path / f'{video_id}.mp4'
                if video_file.exists():
                    size_mb = video_file.stat().st_size / (1024 * 1024)
                    print(f"  ✓ Downloaded successfully ({size_mb:.1f} MB)")
                    downloaded.append(video_id)
                else:
                    print(f"  ✗ Download failed (file not found)")
                    failed.append(video_id)
            else:
                print(f"  ✗ Download failed: {result.stderr[:100]}")
                failed.append(video_id)
            
            # Anti-detection: Random delay between downloads
            if idx < len(video_ids):
                delay = random.uniform(3.0, 6.0)
                print(f"  Waiting {delay:.1f}s before next download...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append(video_id)
            continue
    
    print(f"\n✓ Download complete!")
    print(f"  Successfully downloaded: {len(downloaded)}/{len(video_ids)}")
    print(f"  Failed: {len(failed)}")
    
    return downloaded, failed


def save_results(output_dir, candidates, trainable, downloaded, failed):
    """Save test results."""
    print("\n" + "="*70)
    print("STEP 5: SAVING RESULTS")
    print("="*70)
    
    output_path = Path(output_dir)
    
    results = {
        'test_date': datetime.now().isoformat(),
        'total_candidates': len(candidates),
        'trainable_count': len(trainable),
        'trainable_rate': f"{len(trainable)/len(candidates)*100:.1f}%" if candidates else "0%",
        'downloaded_count': len(downloaded),
        'failed_count': len(failed),
        'candidate_video_ids': candidates,
        'trainable_video_ids': trainable,
        'downloaded_video_ids': downloaded,
        'failed_video_ids': failed
    }
    
    # Save JSON report
    report_file = output_path / 'test_results.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {report_file}")
    
    # Save trainable IDs
    if trainable:
        trainable_file = output_path / 'trainable_video_ids.txt'
        with open(trainable_file, 'w') as f:
            for vid_id in trainable:
                f.write(f"{vid_id}\n")
        print(f"✓ Trainable IDs saved to: {trainable_file}")
    
    # Save downloaded IDs
    if downloaded:
        downloaded_file = output_path / 'downloaded_video_ids.txt'
        with open(downloaded_file, 'w') as f:
            for vid_id in downloaded:
                f.write(f"{vid_id}\n")
        print(f"✓ Downloaded IDs saved to: {downloaded_file}")


def main():
    print("\n" + "="*70)
    print("SMART YOUTUBE TEST SCRAPER")
    print("Testing with 10 videos + Anti-Detection Features")
    print("="*70)
    
    # Load API key
    api_key_file = Path('.youtube_api_key')
    if not api_key_file.exists():
        print("\n✗ ERROR: .youtube_api_key file not found!")
        print("  Create it with your API key first.")
        sys.exit(1)
    
    api_key = api_key_file.read_text().strip()
    print(f"\n✓ Loaded API key: {api_key[:20]}...")
    
    # Output directory
    output_dir = 'youtube_raw_downloads'
    
    # Step 1: Test API key
    if not test_api_key(api_key):
        print("\n✗ API key test failed. Please check your key.")
        sys.exit(1)
    
    # Step 2: Search for candidates (20 videos to get ~2-3 trainable)
    candidates = search_candidate_videos(api_key, max_results=20)
    
    if not candidates:
        print("\n✗ No candidate videos found.")
        sys.exit(1)
    
    # Step 3: Check trainability
    trainable = check_trainability(api_key, candidates)
    
    if not trainable:
        print("\n⚠ WARNING: No trainable videos found!")
        print("  This is expected - most videos are not trainable by default.")
        print("  You may need to search more videos to find trainable ones.")
        
        # Save results anyway
        save_results(output_dir, candidates, trainable, [], [])
        sys.exit(0)
    
    # Step 4: Download trainable videos (max 10)
    downloaded, failed = download_videos(trainable, output_dir, max_downloads=10)
    
    # Step 5: Save results
    save_results(output_dir, candidates, trainable, downloaded, failed)
    
    # Final summary
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Candidates searched: {len(candidates)}")
    print(f"  Trainable found: {len(trainable)} ({len(trainable)/len(candidates)*100:.1f}%)")
    print(f"  Videos downloaded: {len(downloaded)}")
    print(f"  Download failures: {len(failed)}")
    print(f"\nVideos saved to: {Path(output_dir).absolute()}")
    print(f"Results saved to: {Path(output_dir).absolute() / 'test_results.json'}")
    
    if downloaded:
        print(f"\n✓ SUCCESS! Downloaded {len(downloaded)} trainable videos for testing.")
        print(f"\nNext steps:")
        print(f"  1. Check the videos in {output_dir}/")
        print(f"  2. If everything looks good, scale up to download more videos")
        print(f"  3. Process videos later with the full pipeline")
    else:
        print(f"\n⚠ No videos were downloaded.")
        print(f"  This might be due to:")
        print(f"    - No trainable videos found (expected ~5-15% trainability rate)")
        print(f"    - yt-dlp not installed (install with: pip install yt-dlp)")
        print(f"    - Network issues")


if __name__ == '__main__':
    main()
