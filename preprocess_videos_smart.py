#!/usr/bin/env python3
"""
SMART Video Preprocessing - Only extracts frames with detected faces and speech.
Filters out silence, pauses, and frames without faces.
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

from data_pipeline import MediaPipeProcessor, MEDIAPIPE_AVAILABLE


def load_word_timestamps(video_id: str):
    """Load word-level timestamps to know when speech occurs."""
    transcript_path = Path(f'transcripts_word_level/{video_id}.json')
    if not transcript_path.exists():
        return None
    
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    return data.get('words', [])


def is_speaking_at_time(timestamp: float, words: list, buffer: float = 0.1) -> bool:
    """Check if person is speaking at given timestamp."""
    if not words:
        return True  # If no word data, assume always speaking
    
    for word in words:
        # Add buffer before/after word for natural speech
        if word['start'] - buffer <= timestamp <= word['end'] + buffer:
            return True
    
    return False


def extract_speaking_frames(video_path: str, processor: MediaPipeProcessor, 
                            words: list = None, max_frames: int = 100):
    """
    Extract frames only when:
    1. Face is detected
    2. Person is speaking (based on word timestamps)
    3. Lips are visible
    
    Args:
        video_path: Path to video
        processor: MediaPipe processor
        words: Word timestamp data
        max_frames: Maximum frames to extract
    
    Returns:
        frames: List of frame data
        timestamps: List of timestamps for each frame
        lip_metrics: List of lip detection metrics
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frames = []
    timestamps = []
    lip_metrics = []
    
    frame_idx = 0
    
    print(f"  Scanning {total_frames} frames ({duration:.1f}s)...")
    
    with tqdm(total=min(total_frames, max_frames * 10), desc="  Scanning", leave=False) as pbar:
        while len(frames) < max_frames and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # Check if speaking at this time
            if words and not is_speaking_at_time(timestamp, words):
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Try to detect face/lips
            metrics = processor.extract_lip_metrics(frame)
            
            if metrics is not None:
                # Face detected! Save this frame
                frames.append(frame)
                timestamps.append(timestamp)
                lip_metrics.append({
                    'center': metrics.center,
                    'size': metrics.size,
                    'openness': metrics.openness,
                    'bbox': metrics.bbox
                })
            
            frame_idx += 1
            pbar.update(1)
            
            # Stop if we have enough frames
            if len(frames) >= max_frames:
                break
    
    cap.release()
    
    return frames, timestamps, lip_metrics


def extract_rois_from_frames(frames: list, lip_metrics: list, 
                             processor: MediaPipeProcessor, 
                             target_frames: int = 50):
    """
    Extract lip ROIs from selected frames.
    
    Args:
        frames: List of frames with detected faces
        lip_metrics: Lip detection metrics for each frame
        processor: MediaPipe processor
        target_frames: Number of frames to output
    
    Returns:
        ROI array (target_frames, 96, 96, 3)
    """
    if not frames:
        # No frames with faces - return empty
        return np.zeros((target_frames, 96, 96, 3), dtype=np.uint8)
    
    # Sample frames uniformly if we have more than needed
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
        lip_metrics = [lip_metrics[i] for i in indices]
    
    # Extract ROIs
    rois = []
    for frame in frames:
        roi = processor.extract_lip_roi(frame)
        if roi is not None:
            rois.append(roi)
        elif rois:
            # Use last valid ROI
            rois.append(rois[-1].copy())
        else:
            # No valid ROI yet, use zeros
            rois.append(np.zeros((96, 96, 3), dtype=np.uint8))
    
    # Pad if needed
    while len(rois) < target_frames:
        if rois:
            rois.append(rois[-1].copy())
        else:
            rois.append(np.zeros((96, 96, 3), dtype=np.uint8))
    
    return np.stack(rois[:target_frames], axis=0)


def main():
    print("\n" + "="*70)
    print("SMART VIDEO PREPROCESSING")
    print("Only extracts frames with detected faces and speech")
    print("="*70)
    
    if not MEDIAPIPE_AVAILABLE:
        print("\n✗ MediaPipe is not available!")
        sys.exit(1)
    
    # Directories
    video_dir = Path('youtube_raw_downloads')
    output_dir = Path('preprocessed_data_smart')
    output_dir.mkdir(exist_ok=True)
    
    # Find videos
    video_files = sorted(video_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"\n✗ No videos found in {video_dir}/")
        sys.exit(1)
    
    print(f"\n✓ Found {len(video_files)} videos")
    
    # Check for word-level transcripts
    has_word_transcripts = Path('transcripts_word_level').exists()
    if has_word_transcripts:
        print(f"✓ Using word-level timestamps for speech detection")
    else:
        print(f"⚠ No word-level transcripts - will use all frames with faces")
    
    # Initialize processor
    print(f"\nInitializing MediaPipe processor...")
    processor = MediaPipeProcessor(
        roi_size=96,
        stabilize=True,
        crop_mode="adaptive",
        align_mouth=True,
        profile_aware=True
    )
    print(f"✓ Processor initialized")
    
    # Process videos
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
            # Load word timestamps if available
            words = load_word_timestamps(video_id) if has_word_transcripts else None
            
            if words:
                print(f"  ✓ Loaded {len(words)} word timestamps")
            
            # Extract frames with faces and speech
            frames, timestamps, lip_metrics = extract_speaking_frames(
                str(video_path),
                processor,
                words=words,
                max_frames=100  # Scan up to 100 frames
            )
            
            if frames is None or len(frames) == 0:
                print(f"  ✗ No frames with detected faces")
                failed += 1
                continue
            
            print(f"  ✓ Found {len(frames)} frames with faces and speech")
            
            # Extract ROIs from selected frames
            lip_rois = extract_rois_from_frames(
                frames,
                lip_metrics,
                processor,
                target_frames=50
            )
            
            # Calculate quality metrics
            avg_lip_size = np.mean([m['size'] for m in lip_metrics])
            avg_openness = np.mean([m['openness'] for m in lip_metrics])
            
            # Save ROIs
            output_file = output_dir / f"{video_id}_rois.npy"
            np.save(output_file, lip_rois)
            
            print(f"  ✓ Extracted {lip_rois.shape[0]} ROIs ({lip_rois.shape})")
            print(f"  ✓ Avg lip size: {avg_lip_size:.1f}px, openness: {avg_openness:.1f}px")
            print(f"  Saved: {output_file.name}")
            
            results.append({
                'video_id': video_id,
                'video_path': str(video_path),
                'output_file': str(output_file),
                'num_frames': lip_rois.shape[0],
                'roi_shape': list(lip_rois.shape),
                'frames_with_faces': len(frames),
                'avg_lip_size': float(avg_lip_size),
                'avg_openness': float(avg_openness),
                'used_word_timestamps': words is not None
            })
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
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
            'used_word_timestamps': has_word_transcripts,
            'videos': results
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SMART PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    
    if successful > 0:
        total_frames_with_faces = sum(r['frames_with_faces'] for r in results)
        avg_frames_per_video = total_frames_with_faces / successful
        print(f"\nQuality metrics:")
        print(f"  Total frames with faces: {total_frames_with_faces}")
        print(f"  Avg frames per video: {avg_frames_per_video:.1f}")
        print(f"  Used word timestamps: {'Yes' if has_word_transcripts else 'No'}")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - Lip ROIs: {successful} .npy files")
    print(f"  - Metadata: preprocessing_metadata.json")
    
    if successful > 0:
        print(f"\n✓ SUCCESS! Preprocessed {successful} videos")
        print(f"\nAdvantages of smart preprocessing:")
        print(f"  ✓ Only frames with detected faces")
        print(f"  ✓ Only frames during speech (if word timestamps available)")
        print(f"  ✓ No wasted training on silence or missing faces")
        print(f"  ✓ Higher quality training data")
        print(f"\nNext steps:")
        print(f"  1. Use preprocessed_data_smart/ instead of preprocessed_data/")
        print(f"  2. Prepare training dataset")
        print(f"  3. Train with higher quality data!")


if __name__ == '__main__':
    main()
