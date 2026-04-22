#!/usr/bin/env python3
"""
Example usage of YouTube Dataset Scraper
Shows how to use the scraper programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from youtube_scraper import (
    YouTubeVideoFinder,
    YouTubeTrainabilityChecker,
    YouTubeDatasetBuilder
)


def example_1_search_videos(api_key: str):
    """
    Example 1: Search for candidate videos
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Search for Candidate Videos")
    print("="*60 + "\n")
    
    # Initialize finder
    finder = YouTubeVideoFinder(api_key, output_dir="youtube_scraper/data")
    
    # Search with custom queries
    custom_queries = [
        "news anchor interview",
        "educational lecture",
        "public speech"
    ]
    
    candidates = finder.search_multiple_queries(
        queries=custom_queries,
        max_results_per_query=10,  # Small number for testing
        enrich=True
    )
    
    print(f"Found {len(candidates)} unique videos")
    
    # Filter for quality
    filtered = finder.filter_candidates(
        min_views=1000,
        require_hd=True,
        require_captions=False
    )
    
    print(f"After filtering: {len(filtered)} videos")
    
    # Save results
    finder.save_candidates('example_candidates.json')
    finder.save_video_ids('example_video_ids.txt')
    
    print("\nSaved to:")
    print("  - youtube_scraper/data/example_candidates.json")
    print("  - youtube_scraper/data/example_video_ids.txt")
    
    return finder


def example_2_check_trainability(api_key: str, video_ids: list):
    """
    Example 2: Check trainability status
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Check Trainability Status")
    print("="*60 + "\n")
    
    # Initialize checker
    checker = YouTubeTrainabilityChecker(
        api_key,
        cache_file="youtube_scraper/data/example_trainability_cache.json"
    )
    
    # Check trainability
    print(f"Checking {len(video_ids)} videos...")
    trainable_ids = checker.filter_trainable(video_ids)
    
    print(f"\nResults:")
    print(f"  Trainable: {len(trainable_ids)}")
    print(f"  Not trainable: {len(video_ids) - len(trainable_ids)}")
    
    # Get detailed report
    report = checker.get_trainability_report()
    print(f"\nTrainability rate: {report['trainable_percentage']:.1f}%")
    
    # Save trainable IDs
    output_file = Path("youtube_scraper/data/example_trainable_ids.txt")
    with open(output_file, 'w') as f:
        for vid_id in trainable_ids:
            f.write(f"{vid_id}\n")
    
    print(f"\nSaved trainable IDs to: {output_file}")
    
    return trainable_ids


def example_3_build_dataset(video_ids: list):
    """
    Example 3: Build dataset from trainable videos
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Build Dataset")
    print("="*60 + "\n")
    
    # Initialize builder
    builder = YouTubeDatasetBuilder(
        output_dir="youtube_scraper/data/example_dataset",
        roi_size=96,
        target_fps=25,
        num_frames=50,
        max_workers=2  # Use 2 workers for example
    )
    
    # Build dataset
    print(f"Building dataset from {len(video_ids)} videos...")
    print("This may take a while...\n")
    
    summary = builder.build_dataset(
        video_ids=video_ids[:3],  # Only process first 3 for example
        parallel=True
    )
    
    print("\nDataset built successfully!")
    print(f"Total clips: {summary['total_clips']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    return summary


def example_4_single_video_processing(video_id: str):
    """
    Example 4: Process a single video
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Process Single Video")
    print("="*60 + "\n")
    
    # Initialize builder
    builder = YouTubeDatasetBuilder(
        output_dir="youtube_scraper/data/example_single_video",
        roi_size=96,
        target_fps=25,
        num_frames=50,
        max_workers=1
    )
    
    # Process single video
    print(f"Processing video: {video_id}")
    result = builder.process_video(video_id)
    
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Clips created: {result.clips_created}")
    print(f"  Quality score: {result.quality_score:.2f}")
    print(f"  Processing time: {result.processing_time:.1f}s")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    return result


def example_5_custom_quality_filters():
    """
    Example 5: Custom quality filtering
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Quality Filters")
    print("="*60 + "\n")
    
    from data_pipeline import VisualFilterConfig
    
    # Create custom filter config
    custom_config = VisualFilterConfig(
        enabled=True,
        sample_frames=50,  # Sample fewer frames for speed
        min_face_rate=0.8,  # Stricter face detection (80%)
        min_lip_size_ratio=0.04,  # Larger lips required
        min_openness_std=0.005,  # More mouth movement
        min_motion=0.005,
        max_motion=0.06,
        min_sharpness=15.0  # Sharper videos
    )
    
    print("Custom quality thresholds:")
    print(f"  Face detection rate: ≥{custom_config.min_face_rate*100}%")
    print(f"  Lip size ratio: ≥{custom_config.min_lip_size_ratio*100}%")
    print(f"  Mouth movement: ≥{custom_config.min_openness_std}")
    print(f"  Sharpness: ≥{custom_config.min_sharpness}")
    
    print("\nNote: Modify YouTubeDatasetBuilder.filter_config to use custom filters")
    
    return custom_config


def main():
    """
    Run all examples
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='YouTube Scraper Examples')
    parser.add_argument('--api_key', required=True, help='YouTube API key')
    parser.add_argument('--example', type=int, choices=[1,2,3,4,5], help='Run specific example')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("YouTube Dataset Scraper - Usage Examples")
    print("="*60)
    
    if args.all or args.example == 1:
        # Example 1: Search
        finder = example_1_search_videos(args.api_key)
        
        # Get video IDs for next examples
        video_ids = list(finder.candidates.keys())
    else:
        # Load existing video IDs
        video_ids_file = Path("youtube_scraper/data/example_video_ids.txt")
        if video_ids_file.exists():
            with open(video_ids_file, 'r') as f:
                video_ids = [line.strip() for line in f if line.strip()]
        else:
            print("\nNo video IDs found. Run with --example 1 first.")
            return
    
    if args.all or args.example == 2:
        # Example 2: Check trainability
        trainable_ids = example_2_check_trainability(args.api_key, video_ids)
    else:
        # Load existing trainable IDs
        trainable_file = Path("youtube_scraper/data/example_trainable_ids.txt")
        if trainable_file.exists():
            with open(trainable_file, 'r') as f:
                trainable_ids = [line.strip() for line in f if line.strip()]
        else:
            trainable_ids = video_ids[:3]  # Use first 3 as fallback
    
    if args.all or args.example == 3:
        # Example 3: Build dataset
        if trainable_ids:
            example_3_build_dataset(trainable_ids)
        else:
            print("\nNo trainable videos found. Skipping dataset build.")
    
    if args.all or args.example == 4:
        # Example 4: Single video
        if trainable_ids:
            example_4_single_video_processing(trainable_ids[0])
        else:
            print("\nNo trainable videos found. Skipping single video example.")
    
    if args.all or args.example == 5:
        # Example 5: Custom filters
        example_5_custom_quality_filters()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)
    print("\nFor production use, see:")
    print("  python youtube_scraper/build_youtube_dataset.py --help")


if __name__ == '__main__':
    main()
