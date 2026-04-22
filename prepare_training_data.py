#!/usr/bin/env python3
"""
Prepare training dataset from preprocessed videos and transcripts.
Creates train/val split and organizes data for Swin-VALLR training.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random

def load_metadata() -> Dict:
    """Load preprocessing metadata."""
    with open('preprocessed_data/preprocessing_metadata.json', 'r') as f:
        return json.load(f)

def load_transcript(video_id: str) -> str:
    """Load transcript text for a video."""
    transcript_path = f'transcripts/{video_id}.txt'
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def create_dataset_structure(output_dir: str = 'dataset'):
    """Create dataset directory structure."""
    dirs = [
        f'{output_dir}/train/videos',
        f'{output_dir}/train/labels',
        f'{output_dir}/val/videos',
        f'{output_dir}/val/labels',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Created dataset structure in {output_dir}/")

def split_data(video_ids: List[str], train_ratio: float = 0.7) -> Tuple[List[str], List[str]]:
    """
    Split video IDs into train and validation sets.
    
    Args:
        video_ids: List of video IDs
        train_ratio: Ratio of training data (0.7 = 70% train, 30% val)
    
    Returns:
        train_ids, val_ids
    """
    # Shuffle for random split
    random.seed(42)  # For reproducibility
    shuffled = video_ids.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_ids = shuffled[:split_idx]
    val_ids = shuffled[split_idx:]
    
    return train_ids, val_ids

def prepare_sample(video_id: str, split: str, output_dir: str):
    """
    Prepare a single training sample.
    
    Args:
        video_id: Video ID
        split: 'train' or 'val'
        output_dir: Output directory
    """
    # Load ROI data
    roi_path = f'preprocessed_data/{video_id}_rois.npy'
    rois = np.load(roi_path)  # Shape: (50, 96, 96, 3)
    
    # Load transcript
    transcript = load_transcript(video_id)
    
    # Save video data
    video_output = f'{output_dir}/{split}/videos/{video_id}.npy'
    np.save(video_output, rois)
    
    # Save label (transcript)
    label_output = f'{output_dir}/{split}/labels/{video_id}.txt'
    with open(label_output, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    return len(transcript)

def create_manifest(video_ids: List[str], split: str, output_dir: str):
    """Create manifest file listing all samples."""
    manifest = []
    for video_id in video_ids:
        manifest.append({
            'video_id': video_id,
            'video_path': f'{split}/videos/{video_id}.npy',
            'label_path': f'{split}/labels/{video_id}.txt'
        })
    
    manifest_path = f'{output_dir}/{split}/manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✓ Created {split} manifest with {len(manifest)} samples")

def main():
    print("="*70)
    print("PREPARING TRAINING DATASET")
    print("="*70)
    
    # Load metadata
    print("\n[1/5] Loading metadata...")
    metadata = load_metadata()
    video_ids = [v['video_id'] for v in metadata['videos']]
    print(f"  ✓ Found {len(video_ids)} preprocessed videos")
    
    # Create dataset structure
    print("\n[2/5] Creating dataset structure...")
    output_dir = 'dataset'
    create_dataset_structure(output_dir)
    
    # Split data
    print("\n[3/5] Splitting data...")
    train_ids, val_ids = split_data(video_ids, train_ratio=0.7)
    print(f"  ✓ Train: {len(train_ids)} videos ({len(train_ids)/len(video_ids)*100:.0f}%)")
    print(f"  ✓ Val:   {len(val_ids)} videos ({len(val_ids)/len(video_ids)*100:.0f}%)")
    print(f"\n  Train videos: {train_ids}")
    print(f"  Val videos:   {val_ids}")
    
    # Prepare training samples
    print("\n[4/5] Preparing training samples...")
    train_chars = 0
    for video_id in train_ids:
        chars = prepare_sample(video_id, 'train', output_dir)
        train_chars += chars
    print(f"  ✓ Prepared {len(train_ids)} training samples ({train_chars:,} characters)")
    
    # Prepare validation samples
    print("\n[5/5] Preparing validation samples...")
    val_chars = 0
    for video_id in val_ids:
        chars = prepare_sample(video_id, 'val', output_dir)
        val_chars += chars
    print(f"  ✓ Prepared {len(val_ids)} validation samples ({val_chars:,} characters)")
    
    # Create manifests
    print("\n[6/6] Creating manifests...")
    create_manifest(train_ids, 'train', output_dir)
    create_manifest(val_ids, 'val', output_dir)
    
    # Save split info
    split_info = {
        'total_videos': len(video_ids),
        'train_videos': len(train_ids),
        'val_videos': len(val_ids),
        'train_ratio': len(train_ids) / len(video_ids),
        'train_ids': train_ids,
        'val_ids': val_ids,
        'train_characters': train_chars,
        'val_characters': val_chars
    }
    
    with open(f'{output_dir}/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nDataset location: {output_dir}/")
    print(f"\nSplit:")
    print(f"  Train: {len(train_ids)} videos, {train_chars:,} characters")
    print(f"  Val:   {len(val_ids)} videos, {val_chars:,} characters")
    print(f"\nNext steps:")
    print(f"  1. Review dataset in {output_dir}/")
    print(f"  2. Run training: python3 train_small_dataset.py")
    print()

if __name__ == '__main__':
    main()
