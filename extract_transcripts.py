#!/usr/bin/env python3
"""
Extract transcripts from videos using Whisper on AMD GPU (ROCm).
"""

import os
import sys
import json
import torch
import whisper
from pathlib import Path
from datetime import datetime

def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
        print(f"✓ PyTorch version: {torch.__version__}")
        return True
    else:
        print("⚠ No GPU detected, will use CPU (slower)")
        return False


def extract_transcript(video_path, model, device):
    """Extract transcript from a single video."""
    video_id = video_path.stem
    print(f"\n[Processing] {video_id}")
    print(f"  Video: {video_path.name}")
    
    try:
        # Transcribe with Whisper
        print(f"  Transcribing with Whisper...")
        result = model.transcribe(
            str(video_path),
            language='en',  # English only
            fp16=(device == 'cuda'),  # Use FP16 on GPU
            verbose=False
        )
        
        # Extract text
        transcript_text = result['text'].strip()
        
        # Extract segments with timestamps
        segments = []
        for seg in result['segments']:
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            })
        
        print(f"  ✓ Transcribed: {len(transcript_text)} characters, {len(segments)} segments")
        
        return {
            'video_id': video_id,
            'video_path': str(video_path),
            'transcript': transcript_text,
            'segments': segments,
            'language': result['language'],
            'duration': result['segments'][-1]['end'] if result['segments'] else 0
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("WHISPER TRANSCRIPT EXTRACTION (AMD GPU / ROCm)")
    print("="*70)
    
    # Check GPU
    has_gpu = check_gpu()
    device = 'cuda' if has_gpu else 'cpu'
    
    # Input/output directories
    video_dir = Path('youtube_raw_downloads')
    output_dir = Path('transcripts')
    output_dir.mkdir(exist_ok=True)
    
    # Find all videos
    video_files = sorted(video_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"\n✗ No videos found in {video_dir}/")
        sys.exit(1)
    
    print(f"\n✓ Found {len(video_files)} videos")
    
    # Load Whisper model
    print(f"\nLoading Whisper model...")
    print(f"  Model: base (fastest, good accuracy)")
    print(f"  Device: {device}")
    
    model = whisper.load_model("base", device=device)
    print(f"  ✓ Model loaded")
    
    # Process each video
    print("\n" + "="*70)
    print(f"PROCESSING {len(video_files)} VIDEOS")
    print("="*70)
    
    results = []
    successful = 0
    failed = 0
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}]", end=" ")
        
        result = extract_transcript(video_path, model, device)
        
        if result:
            results.append(result)
            successful += 1
            
            # Save individual transcript
            transcript_file = output_dir / f"{result['video_id']}.txt"
            with open(transcript_file, 'w') as f:
                f.write(result['transcript'])
            
            # Save detailed JSON
            json_file = output_dir / f"{result['video_id']}.json"
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  Saved: {transcript_file.name}")
        else:
            failed += 1
    
    # Save all results
    all_results_file = output_dir / 'all_transcripts.json'
    with open(all_results_file, 'w') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'model': 'whisper-base',
            'device': device,
            'total_videos': len(video_files),
            'successful': successful,
            'failed': failed,
            'transcripts': results
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully transcribed: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - Individual transcripts: {successful} .txt files")
    print(f"  - Detailed JSON: {successful} .json files")
    print(f"  - All results: all_transcripts.json")
    
    if successful > 0:
        print(f"\n✓ SUCCESS! Transcribed {successful} videos")
        print(f"\nNext steps:")
        print(f"  1. Review transcripts in {output_dir}/")
        print(f"  2. Extract lip ROIs from videos")
        print(f"  3. Align transcripts with video frames")
        print(f"  4. Prepare dataset for training")
    else:
        print(f"\n✗ No videos were transcribed successfully")


if __name__ == '__main__':
    main()
