#!/usr/bin/env python3
"""
Complete YouTube Dataset Pipeline
Main script to orchestrate the entire dataset creation process.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from youtube_scraper.youtube_video_finder import YouTubeVideoFinder, YOUTUBE_API_AVAILABLE
from youtube_scraper.youtube_trainability_checker import YouTubeTrainabilityChecker
from youtube_scraper.youtube_dataset_builder import YouTubeDatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline to build lip-reading dataset from YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (search, check trainability, build dataset)
  python build_youtube_dataset.py --api_key YOUR_KEY --output_dir /path/to/dataset --full_pipeline
  
  # Just search for videos
  python build_youtube_dataset.py --api_key YOUR_KEY --search_only
  
  # Check trainability of existing video list
  python build_youtube_dataset.py --api_key YOUR_KEY --video_ids videos.txt --check_only
  
  # Build dataset from trainable videos
  python build_youtube_dataset.py --video_ids trainable_videos.txt --output_dir /path/to/dataset --build_only
        """
    )
    
    # API key
    parser.add_argument(
        '--api_key',
        help='YouTube Data API v3 key (required for search and trainability check)'
    )
    
    # Pipeline mode
    parser.add_argument(
        '--full_pipeline',
        action='store_true',
        help='Run complete pipeline: search → check trainability → build dataset'
    )
    parser.add_argument(
        '--search_only',
        action='store_true',
        help='Only search for candidate videos'
    )
    parser.add_argument(
        '--check_only',
        action='store_true',
        help='Only check trainability of videos'
    )
    parser.add_argument(
        '--build_only',
        action='store_true',
        help='Only build dataset (requires trainable video IDs)'
    )
    
    # Input/output
    parser.add_argument(
        '--video_ids',
        help='Input file with video IDs (one per line)'
    )
    parser.add_argument(
        '--output_dir',
        default='/media/agam/Local Disk/the_code/roman/majorproject/youtube_dataset',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--work_dir',
        default='youtube_scraper/data',
        help='Working directory for intermediate files'
    )
    
    # Search parameters
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
        '--min_views',
        type=int,
        help='Minimum view count filter'
    )
    parser.add_argument(
        '--require_hd',
        action='store_true',
        help='Require HD quality'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--roi_size',
        type=int,
        default=96,
        help='Size of lip ROI (default: 96)'
    )
    parser.add_argument(
        '--target_fps',
        type=int,
        default=25,
        help='Target FPS (default: 25)'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=50,
        help='Frames per clip (default: 50)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if not any([args.full_pipeline, args.search_only, args.check_only, args.build_only]):
        parser.error("Must specify one of: --full_pipeline, --search_only, --check_only, --build_only")
    
    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("YOUTUBE LIP-READING DATASET BUILDER")
    print("="*70 + "\n")
    
    # =========================================================================
    # PHASE 1: SEARCH FOR VIDEOS
    # =========================================================================
    
    if args.full_pipeline or args.search_only:
        if not args.api_key:
            parser.error("--api_key is required for video search")
        
        if not YOUTUBE_API_AVAILABLE:
            print("ERROR: google-api-python-client is not installed")
            print("Install with: pip install google-api-python-client")
            sys.exit(1)
        
        print("PHASE 1: SEARCHING FOR CANDIDATE VIDEOS")
        print("-" * 70)
        
        finder = YouTubeVideoFinder(args.api_key, str(work_dir))
        
        # Search for videos
        candidates = finder.search_multiple_queries(
            queries=args.queries,
            max_results_per_query=args.max_per_query,
            enrich=True
        )
        
        print(f"Found {len(candidates)} unique videos")
        
        # Apply filters
        if args.min_views or args.require_hd:
            print("\nApplying quality filters...")
            filtered = finder.filter_candidates(
                min_views=args.min_views,
                require_hd=args.require_hd
            )
            print(f"After filtering: {len(filtered)} videos")
        
        # Save results
        finder.save_candidates('video_candidates.json')
        finder.save_video_ids('candidate_video_ids.txt')
        
        stats = finder.get_statistics()
        print(f"\nSearch Statistics:")
        print(f"  Total videos: {stats['total']}")
        print(f"  HD videos: {stats['hd_videos']} ({stats['hd_percentage']:.1f}%)")
        print(f"  With captions: {stats['with_captions']} ({stats['caption_percentage']:.1f}%)")
        
        candidate_file = work_dir / 'candidate_video_ids.txt'
        
        if args.search_only:
            print(f"\nCandidate video IDs saved to: {candidate_file}")
            print("\nNext step: Check trainability with:")
            print(f"  python build_youtube_dataset.py --api_key YOUR_KEY --video_ids {candidate_file} --check_only")
            return
    
    # =========================================================================
    # PHASE 2: CHECK TRAINABILITY
    # =========================================================================
    
    if args.full_pipeline or args.check_only:
        if not args.api_key:
            parser.error("--api_key is required for trainability check")
        
        print("\n" + "="*70)
        print("PHASE 2: CHECKING VIDEO TRAINABILITY")
        print("-" * 70)
        
        # Determine input file
        if args.video_ids:
            input_file = args.video_ids
        elif args.full_pipeline:
            input_file = work_dir / 'candidate_video_ids.txt'
        else:
            parser.error("--video_ids is required for --check_only mode")
        
        # Load video IDs
        print(f"Loading video IDs from {input_file}...")
        with open(input_file, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Checking trainability for {len(video_ids)} videos...")
        
        # Check trainability
        checker = YouTubeTrainabilityChecker(
            args.api_key,
            cache_file=str(work_dir / 'trainability_cache.json')
        )
        
        trainable_ids = checker.filter_trainable(video_ids)
        
        # Save trainable IDs
        trainable_file = work_dir / 'trainable_video_ids.txt'
        with open(trainable_file, 'w') as f:
            for vid_id in trainable_ids:
                f.write(f"{vid_id}\n")
        
        # Generate report
        report = checker.get_trainability_report()
        report_file = work_dir / 'trainability_report.json'
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTrainability Results:")
        print(f"  Total checked: {report['total_checked']}")
        print(f"  Trainable: {report['trainable']} ({report['trainable_percentage']:.1f}%)")
        print(f"  Not trainable: {report['not_trainable']}")
        print(f"  Errors: {report['errors']}")
        
        print(f"\nTrainable video IDs saved to: {trainable_file}")
        print(f"Trainability report saved to: {report_file}")
        
        if args.check_only:
            print("\nNext step: Build dataset with:")
            print(f"  python build_youtube_dataset.py --video_ids {trainable_file} --output_dir /path/to/dataset --build_only")
            return
    
    # =========================================================================
    # PHASE 3: BUILD DATASET
    # =========================================================================
    
    if args.full_pipeline or args.build_only:
        print("\n" + "="*70)
        print("PHASE 3: BUILDING DATASET")
        print("-" * 70)
        
        # Determine input file
        if args.video_ids:
            input_file = args.video_ids
        elif args.full_pipeline:
            input_file = work_dir / 'trainable_video_ids.txt'
        else:
            parser.error("--video_ids is required for --build_only mode")
        
        # Load video IDs
        print(f"Loading trainable video IDs from {input_file}...")
        with open(input_file, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Building dataset from {len(video_ids)} trainable videos...")
        print(f"Output directory: {args.output_dir}")
        
        # Build dataset
        builder = YouTubeDatasetBuilder(
            output_dir=args.output_dir,
            roi_size=args.roi_size,
            target_fps=args.target_fps,
            num_frames=args.num_frames,
            max_workers=args.max_workers
        )
        
        summary = builder.build_dataset(video_ids, parallel=True)
        
        print("\n" + "="*70)
        print("DATASET BUILD COMPLETE!")
        print("="*70)
        print(f"\nDataset location: {args.output_dir}")
        print(f"  - Videos: processed/videos/ ({summary['total_clips']} clips)")
        print(f"  - Labels: processed/labels/ ({summary['total_clips']} transcripts)")
        print(f"  - Metadata: metadata/")
        
        print(f"\nTo train the model, use:")
        print(f"  python train_swin_vallr.py --data_dir {args.output_dir}/processed")


if __name__ == '__main__':
    main()
