#!/usr/bin/env python3
"""
YouTube Trainability Checker
Checks if YouTube videos are permitted for AI training.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils import setup_logging, log_exception

LOGGER = setup_logging("youtube_trainability_checker")


@dataclass
class TrainabilityStatus:
    """Trainability status for a video."""
    video_id: str
    permitted: str  # "all", "none", or list of companies
    checked_at: str
    is_trainable: bool
    error: Optional[str] = None


class YouTubeTrainabilityChecker:
    """
    Check YouTube video trainability status using the Video Trainability API.
    """
    
    API_ENDPOINT = "https://youtube.googleapis.com/youtube/v3/videoTrainability"
    
    def __init__(self, api_key: str, cache_file: Optional[str] = None):
        """
        Initialize the trainability checker.
        
        Args:
            api_key: YouTube Data API v3 key
            cache_file: Optional path to cache trainability results
        """
        self.api_key = api_key
        self.cache_file = cache_file
        self.cache: Dict[str, TrainabilityStatus] = {}
        
        if cache_file and Path(cache_file).exists():
            self._load_cache()
        
        LOGGER.info("YouTubeTrainabilityChecker initialized")
    
    def _load_cache(self):
        """Load cached trainability results."""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    status = TrainabilityStatus(**item)
                    self.cache[status.video_id] = status
            LOGGER.info(f"Loaded {len(self.cache)} cached trainability results")
        except Exception as e:
            log_exception(LOGGER, e, "Failed to load cache")
    
    def _save_cache(self):
        """Save trainability results to cache."""
        if not self.cache_file:
            return
        
        try:
            cache_data = [asdict(status) for status in self.cache.values()]
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            LOGGER.info(f"Saved {len(self.cache)} trainability results to cache")
        except Exception as e:
            log_exception(LOGGER, e, "Failed to save cache")
    
    def check_video(self, video_id: str, use_cache: bool = True) -> TrainabilityStatus:
        """
        Check trainability status for a single video.
        
        Args:
            video_id: YouTube video ID
            use_cache: Whether to use cached results
        
        Returns:
            TrainabilityStatus object
        """
        # Check cache first
        if use_cache and video_id in self.cache:
            LOGGER.debug(f"Using cached result for {video_id}")
            return self.cache[video_id]
        
        # Make API request
        try:
            url = f"{self.API_ENDPOINT}?id={video_id}&key={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse response
                if 'items' in data and len(data['items']) > 0:
                    item = data['items'][0]
                    permitted = item.get('permitted', 'none')
                    
                    # Determine if trainable
                    is_trainable = (permitted == 'all')
                    
                    status = TrainabilityStatus(
                        video_id=video_id,
                        permitted=str(permitted),
                        checked_at=datetime.now().isoformat(),
                        is_trainable=is_trainable
                    )
                else:
                    # Video not found or no trainability info
                    status = TrainabilityStatus(
                        video_id=video_id,
                        permitted='none',
                        checked_at=datetime.now().isoformat(),
                        is_trainable=False,
                        error='Video not found or no trainability data'
                    )
                
                LOGGER.info(f"Video {video_id}: permitted={status.permitted}, trainable={status.is_trainable}")
            
            elif response.status_code == 403:
                # API key issue
                status = TrainabilityStatus(
                    video_id=video_id,
                    permitted='none',
                    checked_at=datetime.now().isoformat(),
                    is_trainable=False,
                    error='API key invalid or quota exceeded'
                )
                LOGGER.error(f"API key error for {video_id}: {response.text}")
            
            elif response.status_code == 404:
                # Video doesn't exist
                status = TrainabilityStatus(
                    video_id=video_id,
                    permitted='none',
                    checked_at=datetime.now().isoformat(),
                    is_trainable=False,
                    error='Video not found'
                )
                LOGGER.warning(f"Video {video_id} not found")
            
            else:
                # Other error
                status = TrainabilityStatus(
                    video_id=video_id,
                    permitted='none',
                    checked_at=datetime.now().isoformat(),
                    is_trainable=False,
                    error=f'HTTP {response.status_code}: {response.text[:100]}'
                )
                LOGGER.error(f"Error checking {video_id}: {response.status_code}")
        
        except Exception as e:
            log_exception(LOGGER, e, f"Exception checking video {video_id}")
            status = TrainabilityStatus(
                video_id=video_id,
                permitted='none',
                checked_at=datetime.now().isoformat(),
                is_trainable=False,
                error=str(e)
            )
        
        # Cache result
        self.cache[video_id] = status
        return status
    
    def check_videos_batch(
        self,
        video_ids: List[str],
        delay: float = 0.1,
        save_every: int = 100
    ) -> List[TrainabilityStatus]:
        """
        Check trainability for multiple videos.
        
        Args:
            video_ids: List of YouTube video IDs
            delay: Delay between API calls (seconds)
            save_every: Save cache every N videos
        
        Returns:
            List of TrainabilityStatus objects
        """
        results = []
        
        LOGGER.info(f"Checking trainability for {len(video_ids)} videos")
        
        for idx, video_id in enumerate(video_ids, 1):
            status = self.check_video(video_id)
            results.append(status)
            
            # Progress logging
            if idx % 10 == 0:
                trainable_count = sum(1 for s in results if s.is_trainable)
                LOGGER.info(f"Progress: {idx}/{len(video_ids)} checked, {trainable_count} trainable")
            
            # Save cache periodically
            if idx % save_every == 0:
                self._save_cache()
            
            # Rate limiting
            if delay > 0:
                time.sleep(delay)
        
        # Final save
        self._save_cache()
        
        # Summary
        trainable_count = sum(1 for s in results if s.is_trainable)
        error_count = sum(1 for s in results if s.error)
        
        LOGGER.info(f"Trainability check complete:")
        LOGGER.info(f"  Total: {len(results)}")
        LOGGER.info(f"  Trainable: {trainable_count} ({trainable_count/len(results)*100:.1f}%)")
        LOGGER.info(f"  Not trainable: {len(results) - trainable_count}")
        LOGGER.info(f"  Errors: {error_count}")
        
        return results
    
    def filter_trainable(self, video_ids: List[str]) -> List[str]:
        """
        Filter list to only trainable videos.
        
        Args:
            video_ids: List of YouTube video IDs
        
        Returns:
            List of trainable video IDs
        """
        results = self.check_videos_batch(video_ids)
        trainable = [s.video_id for s in results if s.is_trainable]
        
        LOGGER.info(f"Filtered {len(video_ids)} videos to {len(trainable)} trainable")
        return trainable
    
    def get_trainability_report(self) -> Dict:
        """
        Generate a summary report of trainability checks.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.cache)
        trainable = sum(1 for s in self.cache.values() if s.is_trainable)
        errors = sum(1 for s in self.cache.values() if s.error)
        
        # Count permitted types
        permitted_counts = {}
        for status in self.cache.values():
            permitted_counts[status.permitted] = permitted_counts.get(status.permitted, 0) + 1
        
        return {
            'total_checked': total,
            'trainable': trainable,
            'not_trainable': total - trainable,
            'errors': errors,
            'trainable_percentage': (trainable / total * 100) if total > 0 else 0,
            'permitted_breakdown': permitted_counts,
            'last_updated': datetime.now().isoformat()
        }


def main():
    """CLI interface for trainability checker."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check YouTube video trainability status'
    )
    parser.add_argument(
        '--api_key',
        required=True,
        help='YouTube Data API v3 key'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input file with video IDs (one per line)'
    )
    parser.add_argument(
        '--output',
        default='trainable_videos.txt',
        help='Output file for trainable video IDs'
    )
    parser.add_argument(
        '--cache',
        default='trainability_cache.json',
        help='Cache file for trainability results'
    )
    parser.add_argument(
        '--report',
        default='trainability_report.json',
        help='Output file for summary report'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between API calls (seconds)'
    )
    
    args = parser.parse_args()
    
    # Load video IDs
    print(f"Loading video IDs from {args.input}...")
    with open(args.input, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(video_ids)} video IDs")
    
    # Check trainability
    checker = YouTubeTrainabilityChecker(args.api_key, args.cache)
    trainable_ids = checker.filter_trainable(video_ids)
    
    # Save trainable IDs
    with open(args.output, 'w') as f:
        for vid_id in trainable_ids:
            f.write(f"{vid_id}\n")
    
    print(f"\nSaved {len(trainable_ids)} trainable video IDs to {args.output}")
    
    # Generate report
    report = checker.get_trainability_report()
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved trainability report to {args.report}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINABILITY SUMMARY")
    print("="*60)
    print(f"Total videos checked: {report['total_checked']}")
    print(f"Trainable: {report['trainable']} ({report['trainable_percentage']:.1f}%)")
    print(f"Not trainable: {report['not_trainable']}")
    print(f"Errors: {report['errors']}")
    print("="*60)


if __name__ == '__main__':
    main()
