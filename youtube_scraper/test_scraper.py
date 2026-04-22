#!/usr/bin/env python3
"""
Test script for YouTube Dataset Scraper
Tests each component individually.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import yt_dlp
        print("  ✓ yt-dlp")
    except ImportError:
        print("  ✗ yt-dlp (install with: pip install yt-dlp)")
        return False
    
    try:
        from googleapiclient.discovery import build
        print("  ✓ google-api-python-client")
    except ImportError:
        print("  ✗ google-api-python-client (install with: pip install google-api-python-client)")
        return False
    
    try:
        import cv2
        print("  ✓ opencv-python")
    except ImportError:
        print("  ✗ opencv-python")
        return False
    
    try:
        import mediapipe
        print("  ✓ mediapipe")
    except ImportError:
        print("  ✗ mediapipe")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy")
        return False
    
    try:
        from youtube_scraper import (
            YouTubeVideoFinder,
            YouTubeTrainabilityChecker,
            YouTubeDatasetBuilder
        )
        print("  ✓ youtube_scraper modules")
    except ImportError as e:
        print(f"  ✗ youtube_scraper modules: {e}")
        return False
    
    return True


def test_trainability_checker(api_key: str):
    """Test trainability checker with a known video."""
    print("\nTesting trainability checker...")
    
    from youtube_scraper import YouTubeTrainabilityChecker
    
    # Test with a known video ID (example from Google's documentation)
    test_video_id = "jNQXAC9IVRw"
    
    checker = YouTubeTrainabilityChecker(api_key)
    status = checker.check_video(test_video_id)
    
    print(f"  Video ID: {test_video_id}")
    print(f"  Permitted: {status.permitted}")
    print(f"  Trainable: {status.is_trainable}")
    print(f"  Error: {status.error}")
    
    if status.error:
        print("  ✗ Trainability check failed")
        return False
    else:
        print("  ✓ Trainability check successful")
        return True


def test_video_finder(api_key: str):
    """Test video finder with a simple search."""
    print("\nTesting video finder...")
    
    from youtube_scraper import YouTubeVideoFinder
    
    finder = YouTubeVideoFinder(api_key, "youtube_scraper/data/test")
    
    # Search for a small number of videos
    candidates = finder.search_videos(
        query="news interview",
        max_results=5
    )
    
    print(f"  Found {len(candidates)} videos")
    
    if candidates:
        print(f"  Sample video: {candidates[0].title}")
        print("  ✓ Video search successful")
        return True
    else:
        print("  ✗ Video search failed")
        return False


def test_mediapipe():
    """Test MediaPipe face detection."""
    print("\nTesting MediaPipe...")
    
    try:
        from data_pipeline import MediaPipeProcessor
        import numpy as np
        
        processor = MediaPipeProcessor(roi_size=96)
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Try to extract ROI (will fail on black frame, but tests initialization)
        roi = processor.extract_lip_roi(dummy_frame)
        
        processor.close()
        
        print("  ✓ MediaPipe initialized successfully")
        return True
    
    except Exception as e:
        print(f"  ✗ MediaPipe test failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YouTube Dataset Scraper')
    parser.add_argument('--api_key', help='YouTube API key (optional, for API tests)')
    parser.add_argument('--full', action='store_true', help='Run full tests including API calls')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YouTube Dataset Scraper - Component Tests")
    print("="*60)
    
    # Test imports (always run)
    if not test_imports():
        print("\n✗ Import test failed. Install missing dependencies.")
        sys.exit(1)
    
    # Test MediaPipe
    if not test_mediapipe():
        print("\n✗ MediaPipe test failed.")
        sys.exit(1)
    
    # API tests (only if API key provided)
    if args.full and args.api_key:
        if not test_trainability_checker(args.api_key):
            print("\n✗ Trainability checker test failed.")
            sys.exit(1)
        
        if not test_video_finder(args.api_key):
            print("\n✗ Video finder test failed.")
            sys.exit(1)
    elif args.full:
        print("\n⚠ Skipping API tests (no API key provided)")
        print("  Run with: python test_scraper.py --api_key YOUR_KEY --full")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nYou're ready to build your dataset!")
    print("\nQuick start:")
    print("  python youtube_scraper/build_youtube_dataset.py \\")
    print("    --api_key YOUR_API_KEY \\")
    print("    --output_dir /path/to/dataset \\")
    print("    --full_pipeline")


if __name__ == '__main__':
    main()
