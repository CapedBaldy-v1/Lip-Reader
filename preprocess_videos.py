#!/usr/bin/env python3
"""
Preprocess videos using existing data_pipeline.py MediaPipe code.
Extracts lip ROIs from all downloaded videos.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

# Import your existing MediaPipe processor
from data_pipeline import MediaPipeProcessor, MEDIAPIPE_AVAILABLE

def main():
    print("\n" + "="*70)
    print("VIDEO PREPROCESSING - LIP ROI EXTRACTION")
    print("Using existing MediaPipeProcessor from data_pipeline.py")
    print("="*70)
    
    if not MEDIAPIPE_AVAILABLE:
        print("\n✗ MediaPipe is not available!")
        print("  Your data_pipeline.py detected MediaPipe is missing.")
        sys.exit(1)
    
    # Directories
    video_dir = Path('youtube_raw_downloads')
    output_dir = Path('preprocessed_data')
    output_dir.mkdir(exist_ok=True)
    
    # Find videos
    video_files = sorted(video_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"\n✗ No videos found in {video_dir}/")
        sys.exit(1)
    
    print(f"\n✓ Found {len(video_files)} videos")
    print(f"✓ MediaPipe available")
    
    # Initialize MediaPipe processor
    print(f"\nInitializing MediaPipe processor...")
    processor = MediaPipeProcessor(
        roi_size=96,
        stabilize=True,
        crop_mode="adaptive",
        align_mouth=True,
        profile_aware=True
    )
    print(f"✓ Processor initialized")
    
    # Process each video
    print("\n" + "="*70)
    print(f"PROCESSING {len(video_files)} VIDEOS")
    print("="*70)
    
    results = []
    successful = 0
    failed = 0
    
    for idx, video_path in enumerate(video_files, 1):
        video_id = video_path.stem
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_id}")
        
        try:
            # Extract lip ROIs (50 frames at 25 FPS)
            lip_rois = processor.process_video(
                str(video_path),
                target_fps=25,
                num_frames=50
            )
            
            if lip_rois is not None:
                # Save ROIs as numpy array
                import numpy as np
                output_file = output_dir / f"{video_id}_rois.npy"
                np.save(output_file, lip_rois)
                
                print(f"  ✓ Extracted {lip_rois.shape[0]} ROIs ({lip_rois.shape})")
                print(f"  Saved: {output_file.name}")
                
                results.append({
                    'video_id': video_id,
                    'video_path': str(video_path),
                    'output_file': str(output_file),
                    'num_frames': lip_rois.shape[0],
                    'roi_shape': list(lip_rois.shape)
                })
                successful += 1
            else:
                print(f"  ✗ Failed to extract ROIs")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    # Close processor
    processor.close()
    
    # Save metadata
    metadata_file = output_dir / 'preprocessing_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'total_videos': len(video_files),
            'successful': successful,
            'failed': failed,
            'videos': results
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - Lip ROIs: {successful} .npy files")
    print(f"  - Metadata: preprocessing_metadata.json")
    
    if successful > 0:
        print(f"\n✓ SUCCESS! Preprocessed {successful} videos")
        print(f"\nNext steps:")
        print(f"  1. Combine with transcripts from transcripts/")
        print(f"  2. Create training dataset")
        print(f"  3. Train Swin-VALLR model")
        print(f"\nYou now have:")
        print(f"  - {successful} videos with lip ROIs (preprocessed_data/)")
        print(f"  - {len(list(Path('transcripts').glob('*.txt')))} transcripts (transcripts/)")
        print(f"  - Ready for training!")


if __name__ == '__main__':
    main()
