#!/usr/bin/env python3
"""
Extract lip ROIs from videos using MediaPipe Face Mesh.
Preprocessing step 2: Extract 96x96 lip regions from each frame.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    mp_face_mesh = solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: MediaPipe not properly installed: {e}")
    MEDIAPIPE_AVAILABLE = False

# Lip landmarks indices (MediaPipe Face Mesh)
# Outer lip: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88
# Inner lip: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]


def extract_lip_roi(frame, face_landmarks, roi_size=96, margin=0.3):
    """
    Extract lip ROI from a frame using face landmarks.
    
    Args:
        frame: Input frame (BGR)
        face_landmarks: MediaPipe face landmarks
        roi_size: Output ROI size (default 96x96)
        margin: Margin around lips (default 30%)
    
    Returns:
        lip_roi: Cropped and resized lip region (96x96)
        bbox: Bounding box coordinates (x, y, w, h)
    """
    h, w, _ = frame.shape
    
    # Get lip landmark coordinates
    lip_points = []
    for idx in MOUTH_OUTLINE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        lip_points.append((x, y))
    
    lip_points = np.array(lip_points)
    
    # Calculate bounding box
    x_min, y_min = lip_points.min(axis=0)
    x_max, y_max = lip_points.max(axis=0)
    
    # Add margin
    lip_width = x_max - x_min
    lip_height = y_max - y_min
    
    x_margin = int(lip_width * margin)
    y_margin = int(lip_height * margin)
    
    x_min = max(0, x_min - x_margin)
    y_min = max(0, y_min - y_margin)
    x_max = min(w, x_max + x_margin)
    y_max = min(h, y_max + y_margin)
    
    # Crop lip region
    lip_roi = frame[y_min:y_max, x_min:x_max]
    
    # Resize to target size
    if lip_roi.size > 0:
        lip_roi = cv2.resize(lip_roi, (roi_size, roi_size))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return lip_roi, bbox
    else:
        return None, None


def process_video(video_path, output_dir, roi_size=96, target_fps=25):
    """
    Process a single video and extract lip ROIs.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for ROIs
        roi_size: Size of lip ROI (default 96x96)
        target_fps: Target FPS for extraction (default 25)
    
    Returns:
        dict: Processing results
    """
    video_id = video_path.stem
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip for target FPS
    frame_skip = max(1, int(original_fps / target_fps))
    
    print(f"\n[Processing] {video_id}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Target FPS: {target_fps}")
    print(f"  Frame skip: {frame_skip}")
    print(f"  Total frames: {total_frames}")
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        frame_idx = 0
        extracted_frames = []
        failed_frames = 0
        
        pbar = tqdm(total=total_frames // frame_skip, desc=f"  Extracting ROIs")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to match target FPS
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face landmarks
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract lip ROI
                lip_roi, bbox = extract_lip_roi(frame, face_landmarks, roi_size)
                
                if lip_roi is not None:
                    # Save ROI
                    roi_filename = f"frame_{frame_idx:06d}.jpg"
                    roi_path = video_output_dir / roi_filename
                    cv2.imwrite(str(roi_path), lip_roi)
                    
                    extracted_frames.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / original_fps,
                        'bbox': bbox,
                        'roi_file': roi_filename
                    })
                else:
                    failed_frames += 1
            else:
                failed_frames += 1
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
    
    # Save metadata
    metadata = {
        'video_id': video_id,
        'video_path': str(video_path),
        'original_fps': original_fps,
        'target_fps': target_fps,
        'frame_skip': frame_skip,
        'total_frames': total_frames,
        'extracted_frames': len(extracted_frames),
        'failed_frames': failed_frames,
        'roi_size': roi_size,
        'frames': extracted_frames
    }
    
    metadata_file = video_output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Extracted: {len(extracted_frames)} ROIs")
    print(f"  ✗ Failed: {failed_frames} frames")
    
    return metadata


def main():
    print("\n" + "="*70)
    print("LIP ROI EXTRACTION (MediaPipe Face Mesh)")
    print("="*70)
    
    if not MEDIAPIPE_AVAILABLE:
        print("\n✗ MediaPipe is not available!")
        print("  Install with: pip install mediapipe")
        return
    
    # Directories
    video_dir = Path('youtube_raw_downloads')
    output_dir = Path('lip_rois')
    output_dir.mkdir(exist_ok=True)
    
    # Find videos
    video_files = sorted(video_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"\n✗ No videos found in {video_dir}/")
        return
    
    print(f"\n✓ Found {len(video_files)} videos")
    
    # Process each video
    print("\n" + "="*70)
    print(f"PROCESSING {len(video_files)} VIDEOS")
    print("="*70)
    
    results = []
    total_rois = 0
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}]", end=" ")
        
        try:
            metadata = process_video(video_path, output_dir)
            results.append(metadata)
            total_rois += metadata['extracted_frames']
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save all results
    all_results_file = output_dir / 'all_metadata.json'
    with open(all_results_file, 'w') as f:
        json.dump({
            'total_videos': len(video_files),
            'processed_videos': len(results),
            'total_rois': total_rois,
            'videos': results
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Processed: {len(results)}")
    print(f"  Total lip ROIs extracted: {total_rois:,}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - ROIs organized by video ID")
    print(f"  - Metadata: all_metadata.json")
    
    if total_rois > 0:
        print(f"\n✓ SUCCESS! Extracted {total_rois:,} lip ROIs")
        print(f"\nNext steps:")
        print(f"  1. Review ROIs in {output_dir}/")
        print(f"  2. Align ROIs with transcripts")
        print(f"  3. Create training dataset")
        print(f"  4. Train Swin-VALLR model")


if __name__ == '__main__':
    main()
