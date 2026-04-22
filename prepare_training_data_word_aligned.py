#!/usr/bin/env python3
"""
Prepare training dataset with WORD-LEVEL alignment.
Uses word timestamps to create precise frame-to-text mappings.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random

def load_preprocessing_metadata() -> Dict:
    """Load preprocessing metadata."""
    with open('preprocessed_data/preprocessing_metadata.json', 'r') as f:
        return json.load(f)


def load_word_level_transcript(video_id: str) -> Dict:
    """Load word-level transcript with timestamps."""
    transcript_path = f'transcripts_word_level/{video_id}.json'
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def align_frames_to_words(
    video_duration: float,
    num_frames: int,
    words: List[Dict]
) -> List[Dict]:
    """
    Align video frames to words based on timestamps.
    
    Args:
        video_duration: Total video duration in seconds
        num_frames: Number of extracted frames (e.g., 50)
        words: List of word dictionaries with 'start', 'end', 'word'
    
    Returns:
        List of frame alignments with word information
    """
    # Calculate time per frame
    time_per_frame = video_duration / num_frames
    
    frame_alignments = []
    
    for frame_idx in range(num_frames):
        # Calculate frame time (center of frame)
        frame_start = frame_idx * time_per_frame
        frame_end = (frame_idx + 1) * time_per_frame
        frame_center = (frame_start + frame_end) / 2
        
        # Find words that overlap with this frame
        frame_words = []
        for word_info in words:
            word_start = word_info['start']
            word_end = word_info['end']
            
            # Check if word overlaps with frame
            if word_start <= frame_end and word_end >= frame_start:
                frame_words.append({
                    'word': word_info['word'],
                    'start': word_start,
                    'end': word_end,
                    'overlap': min(frame_end, word_end) - max(frame_start, word_start)
                })
        
        # Sort by overlap (most overlapping word first)
        frame_words.sort(key=lambda x: x['overlap'], reverse=True)
        
        frame_alignments.append({
            'frame_idx': frame_idx,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'frame_center': frame_center,
            'words': frame_words,
            'primary_word': frame_words[0]['word'] if frame_words else ''
        })
    
    return frame_alignments


def create_frame_level_labels(frame_alignments: List[Dict]) -> str:
    """
    Create frame-level text labels from alignments.
    
    Args:
        frame_alignments: List of frame alignment dictionaries
    
    Returns:
        Space-separated text of words aligned to frames
    """
    words = []
    prev_word = None
    
    for alignment in frame_alignments:
        word = alignment['primary_word']
        # Avoid duplicates
        if word and word != prev_word:
            words.append(word)
            prev_word = word
    
    return ' '.join(words)


def create_dataset_structure(output_dir: str = 'dataset_word_aligned'):
    """Create dataset directory structure."""
    dirs = [
        f'{output_dir}/train/videos',
        f'{output_dir}/train/labels',
        f'{output_dir}/train/alignments',
        f'{output_dir}/val/videos',
        f'{output_dir}/val/labels',
        f'{output_dir}/val/alignments',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Created dataset structure in {output_dir}/")


def split_data(video_ids: List[str], train_ratio: float = 0.7) -> Tuple[List[str], List[str]]:
    """Split video IDs into train and validation sets."""
    random.seed(42)  # For reproducibility
    shuffled = video_ids.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_ids = shuffled[:split_idx]
    val_ids = shuffled[split_idx:]
    
    return train_ids, val_ids


def prepare_sample(video_id: str, split: str, output_dir: str) -> Dict:
    """
    Prepare a single training sample with word-level alignment.
    
    Args:
        video_id: Video ID
        split: 'train' or 'val'
        output_dir: Output directory
    
    Returns:
        Sample statistics
    """
    # Load ROI data
    roi_path = f'preprocessed_data/{video_id}_rois.npy'
    rois = np.load(roi_path)  # Shape: (50, 96, 96, 3)
    num_frames = rois.shape[0]
    
    # Load word-level transcript
    transcript_data = load_word_level_transcript(video_id)
    words = transcript_data['words']
    duration = transcript_data['duration']
    full_transcript = transcript_data['transcript']
    
    # Align frames to words
    frame_alignments = align_frames_to_words(duration, num_frames, words)
    
    # Create frame-level labels
    frame_labels = create_frame_level_labels(frame_alignments)
    
    # Save video data (ROIs)
    video_output = f'{output_dir}/{split}/videos/{video_id}.npy'
    np.save(video_output, rois)
    
    # Save frame-level labels (for training)
    label_output = f'{output_dir}/{split}/labels/{video_id}.txt'
    with open(label_output, 'w', encoding='utf-8') as f:
        f.write(frame_labels)
    
    # Save alignment information (for analysis)
    alignment_output = f'{output_dir}/{split}/alignments/{video_id}.json'
    with open(alignment_output, 'w', encoding='utf-8') as f:
        json.dump({
            'video_id': video_id,
            'duration': duration,
            'num_frames': num_frames,
            'num_words': len(words),
            'full_transcript': full_transcript,
            'frame_labels': frame_labels,
            'frame_alignments': frame_alignments
        }, f, indent=2)
    
    return {
        'video_id': video_id,
        'num_frames': num_frames,
        'num_words': len(words),
        'frame_label_length': len(frame_labels),
        'full_transcript_length': len(full_transcript)
    }


def create_manifest(video_ids: List[str], split: str, output_dir: str):
    """Create manifest file listing all samples."""
    manifest = []
    for video_id in video_ids:
        manifest.append({
            'video_id': video_id,
            'video_path': f'{split}/videos/{video_id}.npy',
            'label_path': f'{split}/labels/{video_id}.txt',
            'alignment_path': f'{split}/alignments/{video_id}.json'
        })
    
    manifest_path = f'{output_dir}/{split}/manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ Created {split} manifest with {len(manifest)} samples")


def main():
    print("="*70)
    print("PREPARING WORD-ALIGNED TRAINING DATASET")
    print("="*70)
    
    # Load metadata
    print("\n[1/6] Loading metadata...")
    metadata = load_preprocessing_metadata()
    video_ids = [v['video_id'] for v in metadata['videos']]
    print(f"  ✓ Found {len(video_ids)} preprocessed videos")
    
    # Create dataset structure
    print("\n[2/6] Creating dataset structure...")
    output_dir = 'dataset_word_aligned'
    create_dataset_structure(output_dir)
    
    # Split data
    print("\n[3/6] Splitting data...")
    train_ids, val_ids = split_data(video_ids, train_ratio=0.7)
    print(f"  ✓ Train: {len(train_ids)} videos ({len(train_ids)/len(video_ids)*100:.0f}%)")
    print(f"  ✓ Val:   {len(val_ids)} videos ({len(val_ids)/len(video_ids)*100:.0f}%)")
    print(f"\n  Train videos: {train_ids}")
    print(f"  Val videos:   {val_ids}")
    
    # Prepare training samples
    print("\n[4/6] Preparing training samples with word alignment...")
    train_stats = []
    for video_id in train_ids:
        stats = prepare_sample(video_id, 'train', output_dir)
        train_stats.append(stats)
        print(f"  ✓ {video_id}: {stats['num_frames']} frames, {stats['num_words']} words")
    
    # Prepare validation samples
    print("\n[5/6] Preparing validation samples with word alignment...")
    val_stats = []
    for video_id in val_ids:
        stats = prepare_sample(video_id, 'val', output_dir)
        val_stats.append(stats)
        print(f"  ✓ {video_id}: {stats['num_frames']} frames, {stats['num_words']} words")
    
    # Create manifests
    print("\n[6/6] Creating manifests...")
    create_manifest(train_ids, 'train', output_dir)
    create_manifest(val_ids, 'val', output_dir)
    
    # Calculate statistics
    total_train_words = sum(s['num_words'] for s in train_stats)
    total_val_words = sum(s['num_words'] for s in val_stats)
    total_train_frames = sum(s['num_frames'] for s in train_stats)
    total_val_frames = sum(s['num_frames'] for s in val_stats)
    
    # Save split info
    split_info = {
        'total_videos': len(video_ids),
        'train_videos': len(train_ids),
        'val_videos': len(val_ids),
        'train_ratio': len(train_ids) / len(video_ids),
        'train_ids': train_ids,
        'val_ids': val_ids,
        'train_words': total_train_words,
        'val_words': total_val_words,
        'train_frames': total_train_frames,
        'val_frames': total_val_frames,
        'train_stats': train_stats,
        'val_stats': val_stats
    }
    
    with open(f'{output_dir}/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("WORD-ALIGNED DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nDataset location: {output_dir}/")
    print(f"\nSplit:")
    print(f"  Train: {len(train_ids)} videos, {total_train_words:,} words, {total_train_frames} frames")
    print(f"  Val:   {len(val_ids)} videos, {total_val_words:,} words, {total_val_frames} frames")
    
    print(f"\nAlignment Quality:")
    avg_words_per_frame_train = total_train_words / total_train_frames
    avg_words_per_frame_val = total_val_words / total_val_frames
    print(f"  Train: {avg_words_per_frame_train:.2f} words per frame")
    print(f"  Val:   {avg_words_per_frame_val:.2f} words per frame")
    
    print(f"\nFiles per sample:")
    print(f"  - videos/{'{video_id}'}.npy       : Video frames (50, 96, 96, 3)")
    print(f"  - labels/{'{video_id}'}.txt       : Frame-aligned text labels")
    print(f"  - alignments/{'{video_id}'}.json  : Detailed alignment info")
    
    print(f"\nNext steps:")
    print(f"  1. Review alignments in {output_dir}/train/alignments/")
    print(f"  2. Run training: python3 train_word_aligned.py")
    print()


if __name__ == '__main__':
    main()
