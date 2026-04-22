#!/usr/bin/env python3
"""
Extract transcripts from videos using Whisper with WORD-LEVEL timestamps.
Uses AMD GPU (ROCm) for acceleration.
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


def extract_transcript_with_words(video_path, model, device):
    """Extract transcript with word-level timestamps from a single video."""
    video_id = video_path.stem
    print(f"\n[Processing] {video_id}")
    print(f"  Video: {video_path.name}")
    
    try:
        # Transcribe with Whisper - word_timestamps=True enables word-level timing
        print(f"  Transcribing with Whisper (word-level timestamps)...")
        result = model.transcribe(
            str(video_path),
            language='en',  # English only
            fp16=(device == 'cuda'),  # Use FP16 on GPU
            verbose=False,
            word_timestamps=True  # ← Enable word-level timestamps
        )
        
        # Extract full text
        transcript_text = result['text'].strip()
        
        # Extract segments with word-level timestamps
        segments = []
        all_words = []
        word_count = 0
        
        for seg in result['segments']:
            segment_data = {
                'id': seg['id'],
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip(),
                'words': []
            }
            
            # Extract word-level timestamps if available
            if 'words' in seg:
                for word_info in seg['words']:
                    word_data = {
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'probability': word_info.get('probability', 1.0)
                    }
                    segment_data['words'].append(word_data)
                    all_words.append(word_data)
                    word_count += 1
            
            segments.append(segment_data)
        
        print(f"  ✓ Transcribed: {len(transcript_text)} characters")
        print(f"  ✓ Segments: {len(segments)}")
        print(f"  ✓ Words: {word_count} (with timestamps)")
        
        return {
            'video_id': video_id,
            'video_path': str(video_path),
            'transcript': transcript_text,
            'segments': segments,
            'words': all_words,  # All words in one flat list
            'word_count': word_count,
            'segment_count': len(segments),
            'language': result['language'],
            'duration': result['segments'][-1]['end'] if result['segments'] else 0
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def create_word_level_transcript(result):
    """Create a human-readable word-level transcript."""
    lines = []
    lines.append(f"Video: {result['video_id']}")
    lines.append(f"Duration: {result['duration']:.2f} seconds")
    lines.append(f"Words: {result['word_count']}")
    lines.append("="*70)
    lines.append("")
    
    for word_info in result['words']:
        timestamp = format_timestamp(word_info['start'])
        word = word_info['word']
        prob = word_info['probability']
        lines.append(f"[{timestamp}] {word} (confidence: {prob:.2%})")
    
    return "\n".join(lines)


def create_srt_subtitle(result):
    """Create SRT subtitle format with word-level timing."""
    lines = []
    
    for idx, word_info in enumerate(result['words'], 1):
        start_time = format_timestamp(word_info['start']).replace('.', ',')
        end_time = format_timestamp(word_info['end']).replace('.', ',')
        word = word_info['word']
        
        lines.append(str(idx))
        lines.append(f"{start_time} --> {end_time}")
        lines.append(word)
        lines.append("")
    
    return "\n".join(lines)


def main():
    print("\n" + "="*70)
    print("WHISPER WORD-LEVEL TRANSCRIPT EXTRACTION (AMD GPU / ROCm)")
    print("="*70)
    
    # Check GPU
    has_gpu = check_gpu()
    device = 'cuda' if has_gpu else 'cpu'
    
    # Input/output directories
    video_dir = Path('youtube_raw_downloads')
    output_dir = Path('transcripts_word_level')
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
    print(f"  Word timestamps: ENABLED")
    
    model = whisper.load_model("base", device=device)
    print(f"  ✓ Model loaded")
    
    # Process each video
    print("\n" + "="*70)
    print(f"PROCESSING {len(video_files)} VIDEOS")
    print("="*70)
    
    results = []
    successful = 0
    failed = 0
    total_words = 0
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}]", end=" ")
        
        result = extract_transcript_with_words(video_path, model, device)
        
        if result:
            results.append(result)
            successful += 1
            total_words += result['word_count']
            
            video_id = result['video_id']
            
            # 1. Save plain text transcript
            transcript_file = output_dir / f"{video_id}.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result['transcript'])
            
            # 2. Save detailed JSON with word timestamps
            json_file = output_dir / f"{video_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # 3. Save word-level readable transcript
            word_transcript_file = output_dir / f"{video_id}_words.txt"
            with open(word_transcript_file, 'w', encoding='utf-8') as f:
                f.write(create_word_level_transcript(result))
            
            # 4. Save SRT subtitle file
            srt_file = output_dir / f"{video_id}.srt"
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(create_srt_subtitle(result))
            
            print(f"  Saved: {video_id}.txt, .json, _words.txt, .srt")
        else:
            failed += 1
    
    # Save all results
    all_results_file = output_dir / 'all_transcripts.json'
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'model': 'whisper-base',
            'device': device,
            'word_timestamps': True,
            'total_videos': len(video_files),
            'successful': successful,
            'failed': failed,
            'total_words': total_words,
            'transcripts': results
        }, f, indent=2, ensure_ascii=False)
    
    # Create summary statistics
    summary_file = output_dir / 'SUMMARY.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("WORD-LEVEL TRANSCRIPT EXTRACTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: whisper-base\n")
        f.write(f"Device: {device}\n")
        f.write(f"Word Timestamps: ENABLED\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total videos: {len(video_files)}\n")
        f.write(f"  Successfully transcribed: {successful}\n")
        f.write(f"  Failed: {failed}\n")
        f.write(f"  Total words extracted: {total_words:,}\n")
        f.write(f"  Average words per video: {total_words/successful if successful > 0 else 0:.0f}\n\n")
        
        f.write("Files generated per video:\n")
        f.write("  - {video_id}.txt         : Plain text transcript\n")
        f.write("  - {video_id}.json        : Full data with word timestamps\n")
        f.write("  - {video_id}_words.txt   : Word-by-word with timestamps\n")
        f.write("  - {video_id}.srt         : Subtitle file (SRT format)\n\n")
        
        if successful > 0:
            f.write("Video Details:\n")
            f.write("-" * 70 + "\n")
            for result in results:
                f.write(f"\n{result['video_id']}:\n")
                f.write(f"  Duration: {result['duration']:.2f}s\n")
                f.write(f"  Words: {result['word_count']}\n")
                f.write(f"  Segments: {result['segment_count']}\n")
                f.write(f"  Characters: {len(result['transcript'])}\n")
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully transcribed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total words extracted: {total_words:,}")
    print(f"  Average words per video: {total_words/successful if successful > 0 else 0:.0f}")
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - Plain text: {successful} .txt files")
    print(f"  - Detailed JSON: {successful} .json files")
    print(f"  - Word-level transcripts: {successful} _words.txt files")
    print(f"  - SRT subtitles: {successful} .srt files")
    print(f"  - Summary: SUMMARY.txt")
    print(f"  - All results: all_transcripts.json")
    
    if successful > 0:
        print(f"\n✓ SUCCESS! Transcribed {successful} videos with word-level timestamps")
        print(f"\nExample files for first video:")
        if results:
            first_id = results[0]['video_id']
            print(f"  - {output_dir}/{first_id}.txt")
            print(f"  - {output_dir}/{first_id}.json")
            print(f"  - {output_dir}/{first_id}_words.txt")
            print(f"  - {output_dir}/{first_id}.srt")
        
        print(f"\nNext steps:")
        print(f"  1. Review word-level transcripts in {output_dir}/")
        print(f"  2. Use .json files for frame-level alignment")
        print(f"  3. Use .srt files for video playback with subtitles")
    else:
        print(f"\n✗ No videos were transcribed successfully")


if __name__ == '__main__':
    main()
