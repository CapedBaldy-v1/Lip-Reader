#!/usr/bin/env python3
"""
Download trainable YouTube videos without processing.
Just search, check trainability, and download raw videos.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from youtube_scraper.youtube_video_finder import YouTubeVideoFinder
from youtube_scraper.youtube_trainability_checker import YouTubeTrainabilityChecker

def main():
    API_KEY = "AIzaSyDf86HCHZefQyA8qXFAFszM-Xv44d0LDmc"
    OUTPUT_DIR = "youtube_raw_downloads"
    
    print("\n" + "="*70)
    print("YOUTUBE VIDEO DOWNLOADER - RAW VIDEOS ONLY")
    print("="*70 + "\n")
    
    # Step 1: Search for videos
    print("STEP 1: Searching for candidate videos...")
    print("-" * 70)
    
    finder = YouTubeVideoFinder(API_KEY, "youtube_scraper/data")
    candidates = finder.search_multiple_queries(
        queries=None,  # Use defaults
        max_results_per_query=50,
        enrich=True
    )
    
    print(f"\n✓ Found {len(candidates)} unique videos")
    
    # Step 2: Check trainability
    print("\n" + "="*70)
    print("STEP 2: Checking trainability...")
    print("-" * 70)
    
    checker = YouTubeTrainabilityChecker(
        API_KEY,
        cache_file="youtube_scraper/data/trainability_cache.json"
    )
    
    video_ids = [c.video_id for c in candidates]
    trainable_ids = checker.filter_trainable(video_ids)
    
    print(f"\n✓ Found {len(trainable_ids)} trainable videos")
    
    # Step 3: Download videos
    print("\n" + "="*70)
    print("STEP 3: Downloading trainable videos...")
    print("-" * 70)
    
    import yt_dlp
    
    os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/captions", exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    for idx, video_id in enumerate(trainable_ids, 1):
        print(f"\n[{idx}/{len(trainable_ids)}] Downloading: {video_id}")
        
        try:
            # Download video
            ydl_opts = {
                'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
                'outtmpl': f'{OUTPUT_DIR}/videos/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
            # Download captions
            ydl_opts_captions = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'outtmpl': f'{OUTPUT_DIR}/captions/%(id)s',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts_captions) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
            downloaded += 1
            print(f"  ✓ Downloaded successfully")
            
        except Exception as e:
            failed += 1
            print(f"  ✗ Failed: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"Total trainable videos: {len(trainable_ids)}")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"\nVideos saved to: {OUTPUT_DIR}/videos/")
    print(f"Captions saved to: {OUTPUT_DIR}/captions/")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
