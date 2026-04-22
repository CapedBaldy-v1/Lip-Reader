#!/usr/bin/env python3
"""
YouTube Video Finder
Search for candidate videos suitable for lip-reading dataset.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("Warning: google-api-python-client not installed")
    print("Install with: pip install google-api-python-client")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils import setup_logging, log_exception

LOGGER = setup_logging("youtube_video_finder")


@dataclass
class VideoCandidate:
    """Candidate video information."""
    video_id: str
    title: str
    channel_title: str
    published_at: str
    duration: Optional[str] = None
    view_count: Optional[int] = None
    definition: Optional[str] = None  # 'hd' or 'sd'
    caption: Optional[str] = None  # 'true' or 'false'
    found_at: str = None
    
    def __post_init__(self):
        if self.found_at is None:
            self.found_at = datetime.now().isoformat()


class YouTubeVideoFinder:
    """
    Find YouTube videos suitable for lip-reading dataset.
    Focuses on videos with clear speech and frontal faces.
    """
    
    # Search queries optimized for lip-reading
    DEFAULT_QUERIES = [
        # News and interviews
        "news anchor interview",
        "news broadcast",
        "television interview",
        "talk show interview",
        
        # Educational content
        "educational lecture frontal",
        "tutorial talking head",
        "online course lecture",
        "presentation speech",
        
        # Public speaking
        "public speech",
        "ted talk style",
        "conference presentation",
        "keynote speech",
        
        # Vlogs and direct-to-camera
        "vlog talking to camera",
        "youtube creator talking",
        "direct to camera speech",
        
        # Language learning (often has clear speech)
        "english speaking practice",
        "pronunciation tutorial",
        "speaking english clearly",
    ]
    
    def __init__(self, api_key: str, output_dir: str = "youtube_scraper/data"):
        """
        Initialize the video finder.
        
        Args:
            api_key: YouTube Data API v3 key
            output_dir: Directory to save results
        """
        if not YOUTUBE_API_AVAILABLE:
            raise ImportError("google-api-python-client is required")
        
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.candidates: Dict[str, VideoCandidate] = {}
        
        LOGGER.info("YouTubeVideoFinder initialized")
    
    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        video_duration: str = 'medium',  # short (<4min), medium (4-20min), long (>20min)
        video_definition: str = 'high',  # 'high' for HD, 'any' for all
        relevance_language: str = 'en',
        order: str = 'relevance'  # relevance, date, rating, viewCount
    ) -> List[VideoCandidate]:
        """
        Search for videos matching query.
        
        Args:
            query: Search query
            max_results: Maximum number of results (max 50 per request)
            video_duration: Duration filter
            video_definition: Quality filter
            relevance_language: Language code
            order: Sort order
        
        Returns:
            List of VideoCandidate objects
        """
        LOGGER.info(f"Searching for: '{query}' (max {max_results} results)")
        
        candidates = []
        
        try:
            # Search request
            request = self.youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=min(max_results, 50),
                videoDefinition=video_definition,
                videoDuration=video_duration,
                relevanceLanguage=relevance_language,
                order=order,
                videoCaption='any',  # We'll filter for captions later
                safeSearch='strict'  # Family-friendly content only
            )
            
            response = request.execute()
            
            # Parse results
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                candidate = VideoCandidate(
                    video_id=video_id,
                    title=snippet['title'],
                    channel_title=snippet['channelTitle'],
                    published_at=snippet['publishedAt']
                )
                
                candidates.append(candidate)
                self.candidates[video_id] = candidate
            
            LOGGER.info(f"Found {len(candidates)} videos for query: '{query}'")
        
        except HttpError as e:
            log_exception(LOGGER, e, f"HTTP error searching for '{query}'")
        except Exception as e:
            log_exception(LOGGER, e, f"Error searching for '{query}'")
        
        return candidates
    
    def enrich_video_details(self, video_ids: List[str]) -> None:
        """
        Fetch additional details for videos (duration, views, captions).
        
        Args:
            video_ids: List of video IDs to enrich
        """
        LOGGER.info(f"Enriching details for {len(video_ids)} videos")
        
        # YouTube API allows up to 50 IDs per request
        batch_size = 50
        
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i+batch_size]
            
            try:
                request = self.youtube.videos().list(
                    part='contentDetails,statistics,status',
                    id=','.join(batch)
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    video_id = item['id']
                    
                    if video_id in self.candidates:
                        candidate = self.candidates[video_id]
                        
                        # Duration
                        candidate.duration = item['contentDetails'].get('duration')
                        
                        # View count
                        stats = item.get('statistics', {})
                        if 'viewCount' in stats:
                            candidate.view_count = int(stats['viewCount'])
                        
                        # Definition (HD/SD)
                        candidate.definition = item['contentDetails'].get('definition', 'sd')
                        
                        # Caption availability
                        candidate.caption = item['contentDetails'].get('caption', 'false')
                
                LOGGER.info(f"Enriched batch {i//batch_size + 1}/{(len(video_ids)-1)//batch_size + 1}")
            
            except HttpError as e:
                log_exception(LOGGER, e, f"HTTP error enriching batch {i//batch_size + 1}")
            except Exception as e:
                log_exception(LOGGER, e, f"Error enriching batch {i//batch_size + 1}")
            
            # Rate limiting
            time.sleep(0.1)
    
    def search_multiple_queries(
        self,
        queries: Optional[List[str]] = None,
        max_results_per_query: int = 50,
        enrich: bool = True
    ) -> List[VideoCandidate]:
        """
        Search for videos using multiple queries.
        
        Args:
            queries: List of search queries (uses defaults if None)
            max_results_per_query: Max results per query
            enrich: Whether to fetch additional video details
        
        Returns:
            List of unique VideoCandidate objects
        """
        if queries is None:
            queries = self.DEFAULT_QUERIES
        
        LOGGER.info(f"Searching with {len(queries)} queries")
        
        all_candidates = []
        
        for idx, query in enumerate(queries, 1):
            LOGGER.info(f"Query {idx}/{len(queries)}: {query}")
            
            candidates = self.search_videos(
                query=query,
                max_results=max_results_per_query
            )
            
            all_candidates.extend(candidates)
            
            # Rate limiting between queries
            time.sleep(1.0)
        
        # Remove duplicates (keep first occurrence)
        unique_ids = list(dict.fromkeys([c.video_id for c in all_candidates]))
        unique_candidates = [self.candidates[vid_id] for vid_id in unique_ids]
        
        LOGGER.info(f"Found {len(unique_candidates)} unique videos from {len(queries)} queries")
        
        # Enrich with additional details
        if enrich:
            self.enrich_video_details(unique_ids)
        
        return unique_candidates
    
    def filter_candidates(
        self,
        min_views: Optional[int] = None,
        require_hd: bool = True,
        require_captions: bool = False
    ) -> List[VideoCandidate]:
        """
        Filter candidates based on quality criteria.
        
        Args:
            min_views: Minimum view count (None = no filter)
            require_hd: Require HD quality
            require_captions: Require captions available
        
        Returns:
            Filtered list of VideoCandidate objects
        """
        filtered = []
        
        for candidate in self.candidates.values():
            # View count filter
            if min_views and (candidate.view_count is None or candidate.view_count < min_views):
                continue
            
            # HD filter
            if require_hd and candidate.definition != 'hd':
                continue
            
            # Caption filter
            if require_captions and candidate.caption != 'true':
                continue
            
            filtered.append(candidate)
        
        LOGGER.info(f"Filtered {len(self.candidates)} candidates to {len(filtered)}")
        return filtered
    
    def save_candidates(self, filename: str = "video_candidates.json") -> None:
        """Save candidates to JSON file."""
        filepath = self.output_dir / filename
        
        data = [asdict(c) for c in self.candidates.values()]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        LOGGER.info(f"Saved {len(data)} candidates to {filepath}")
    
    def save_video_ids(self, filename: str = "video_ids.txt") -> None:
        """Save video IDs to text file (one per line)."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for video_id in self.candidates.keys():
                f.write(f"{video_id}\n")
        
        LOGGER.info(f"Saved {len(self.candidates)} video IDs to {filepath}")
    
    def load_candidates(self, filename: str = "video_candidates.json") -> None:
        """Load candidates from JSON file."""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            LOGGER.warning(f"Candidate file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for item in data:
            candidate = VideoCandidate(**item)
            self.candidates[candidate.video_id] = candidate
        
        LOGGER.info(f"Loaded {len(self.candidates)} candidates from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about found candidates."""
        total = len(self.candidates)
        
        if total == 0:
            return {'total': 0}
        
        hd_count = sum(1 for c in self.candidates.values() if c.definition == 'hd')
        caption_count = sum(1 for c in self.candidates.values() if c.caption == 'true')
        
        view_counts = [c.view_count for c in self.candidates.values() if c.view_count]
        avg_views = sum(view_counts) / len(view_counts) if view_counts else 0
        
        return {
            'total': total,
            'hd_videos': hd_count,
            'hd_percentage': (hd_count / total * 100) if total > 0 else 0,
            'with_captions': caption_count,
            'caption_percentage': (caption_count / total * 100) if total > 0 else 0,
            'average_views': int(avg_views),
            'total_views': sum(view_counts) if view_counts else 0
        }


def main():
    """CLI interface for video finder."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find YouTube videos for lip-reading dataset'
    )
    parser.add_argument(
        '--api_key',
        required=True,
        help='YouTube Data API v3 key'
    )
    parser.add_argument(
        '--queries',
        nargs='+',
        help='Custom search queries (uses defaults if not specified)'
    )
    parser.add_argument(
        '--max_per_query',
        type=int,
        default=50,
        help='Maximum results per query (default: 50)'
    )
    parser.add_argument(
        '--output_dir',
        default='youtube_scraper/data',
        help='Output directory for results'
    )
    parser.add_argument(
        '--min_views',
        type=int,
        help='Minimum view count filter'
    )
    parser.add_argument(
        '--require_hd',
        action='store_true',
        help='Require HD quality'
    )
    parser.add_argument(
        '--require_captions',
        action='store_true',
        help='Require captions available'
    )
    
    args = parser.parse_args()
    
    # Initialize finder
    print("Initializing YouTube Video Finder...")
    finder = YouTubeVideoFinder(args.api_key, args.output_dir)
    
    # Search for videos
    print(f"\nSearching for videos...")
    candidates = finder.search_multiple_queries(
        queries=args.queries,
        max_results_per_query=args.max_per_query,
        enrich=True
    )
    
    print(f"Found {len(candidates)} unique videos")
    
    # Apply filters
    if args.min_views or args.require_hd or args.require_captions:
        print("\nApplying filters...")
        filtered = finder.filter_candidates(
            min_views=args.min_views,
            require_hd=args.require_hd,
            require_captions=args.require_captions
        )
        print(f"After filtering: {len(filtered)} videos")
    
    # Save results
    print("\nSaving results...")
    finder.save_candidates()
    finder.save_video_ids()
    
    # Print statistics
    stats = finder.get_statistics()
    print("\n" + "="*60)
    print("VIDEO SEARCH STATISTICS")
    print("="*60)
    print(f"Total videos found: {stats['total']}")
    print(f"HD videos: {stats['hd_videos']} ({stats['hd_percentage']:.1f}%)")
    print(f"With captions: {stats['with_captions']} ({stats['caption_percentage']:.1f}%)")
    print(f"Average views: {stats['average_views']:,}")
    print("="*60)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - video_candidates.json (detailed info)")
    print(f"  - video_ids.txt (IDs only)")


if __name__ == '__main__':
    main()
