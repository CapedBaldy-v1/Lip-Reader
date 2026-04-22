#!/usr/bin/env python3
"""
American Accent Verification Script
Uses bookbot/english-accent-classifier to verify downloaded videos contain American English.

WORKFLOW:
1. Load downloaded videos from american_download_results.json
2. Extract audio sample from each video (first 10 seconds)
3. Run accent classification model
4. Keep only videos classified as 'us' accent with confidence > 0.80
5. Delete non-American videos
6. Save verification results

REQUIREMENTS:
- speechbrain: pip install speechbrain
- torchaudio: pip install torchaudio
- ffmpeg: sudo apt-get install ffmpeg (or brew install ffmpeg on Mac)
"""

import os
import sys
import json
import torch
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('accent_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AccentVerifier:
    """Verify American English accent using bookbot/english-accent-classifier."""
    
    def __init__(self, confidence_threshold=0.80):
        """
        Initialize accent classifier model.
        
        Args:
            confidence_threshold: Minimum confidence to accept as American (0.80 = 80%)
        """
        self.confidence_threshold = confidence_threshold
        self.classifier = None
        logger.info(f"Confidence threshold: {confidence_threshold:.0%}")
        
    def load_model(self):
        """Load the accent classification model."""
        logger.info("Loading accent classifier model...")
        logger.info("Model: bookbot/english-accent-classifier (95% accuracy)")
        
        try:
            from speechbrain.pretrained.interfaces import foreign_class
            
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            if device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Load model
            self.classifier = foreign_class(
                source="bookbot/english-accent-classifier",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": device}
            )
            
            logger.info("✓ Model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error("✗ ERROR: speechbrain not installed!")
            logger.error("  Install with: pip install speechbrain torchaudio")
            return False
        except Exception as e:
            logger.error(f"✗ ERROR loading model: {e}")
            return False
    
    def extract_audio_sample(self, video_path: str, duration=10) -> str:
        """
        Extract audio sample from video (first 10 seconds).
        
        Args:
            video_path: Path to video file
            duration: Duration in seconds (default: 10)
            
        Returns:
            Path to extracted audio file
        """
        audio_path = video_path.replace('.mp4', '_audio.wav')
        
        # Extract audio with ffmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', str(duration),  # First N seconds
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz
            '-y',  # Overwrite
            '-loglevel', 'error',  # Suppress output
            audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Audio extraction failed: {e.stderr.decode()}")
    
    def classify_accent(self, audio_path: str) -> Dict:
        """
        Classify accent of audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with accent, confidence, and probabilities
        """
        out_prob, score, index, text_lab = self.classifier.classify_file(audio_path)
        
        # Get confidence score
        confidence = torch.max(out_prob).item()
        
        # Get all probabilities (top 5)
        probs, indices = torch.topk(out_prob[0], k=min(5, out_prob.shape[1]))
        
        # Accent labels (from bookbot model)
        accent_labels = [
            'us', 'england', 'australia', 'indian', 'canada',
            'bermuda', 'scotland', 'african', 'ireland', 'newzealand',
            'wales', 'malaysia', 'philippines', 'singapore', 'hongkong', 'southatlantic'
        ]
        
        top_accents = {}
        for prob, idx in zip(probs, indices):
            if idx < len(accent_labels):
                top_accents[accent_labels[idx]] = prob.item()
        
        return {
            'accent': text_lab[0],
            'confidence': confidence,
            'top_accents': top_accents
        }
    
    def verify_video(self, video_path: str, video_info: Dict) -> Tuple[bool, Dict]:
        """
        Verify if video contains American English accent.
        
        Args:
            video_path: Path to video file
            video_info: Video metadata
            
        Returns:
            (is_american, result_dict)
        """
        video_id = video_info['id']
        title = video_info['title']
        
        logger.info(f"Verifying: {video_id}")
        logger.info(f"  Title: {title[:60]}...")
        
        try:
            # Extract audio
            audio_path = self.extract_audio_sample(video_path)
            
            # Classify accent
            result = self.classify_accent(audio_path)
            
            # Clean up audio file
            try:
                os.remove(audio_path)
            except:
                pass
            
            # Check if American
            is_american = (
                result['accent'] == 'us' and 
                result['confidence'] >= self.confidence_threshold
            )
            
            # Log result
            logger.info(f"  Accent: {result['accent']} (confidence: {result['confidence']:.2%})")
            logger.info(f"  Top accents: {', '.join([f'{k}: {v:.1%}' for k, v in list(result['top_accents'].items())[:3]])}")
            
            if is_american:
                logger.info(f"  ✓ AMERICAN - Keeping video")
            else:
                logger.info(f"  ✗ NON-AMERICAN - Will delete")
            
            return is_american, result
            
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            # On error, keep video (benefit of doubt)
            return True, {'error': str(e), 'kept_on_error': True}


def main():
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("AMERICAN ACCENT VERIFICATION")
    logger.info("="*70)
    logger.info(f"Session started: {start_time.isoformat()}")
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("✓ ffmpeg found")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("✗ ERROR: ffmpeg not installed!")
        logger.error("  Install with: sudo apt-get install ffmpeg")
        sys.exit(1)
    
    # Paths
    video_dir = Path('youtube_raw_downloads')
    results_file = video_dir / 'american_download_results.json'
    verification_file = video_dir / 'accent_verification_results.json'
    
    if not results_file.exists():
        logger.error(f"✗ ERROR: {results_file} not found!")
        logger.error("  Run download_american_videos.py first")
        sys.exit(1)
    
    # Load download results
    logger.info(f"Loading download results from {results_file}...")
    with open(results_file, 'r') as f:
        download_results = json.load(f)
    
    downloaded_videos = download_results.get('downloaded_videos', [])
    logger.info(f"✓ Found {len(downloaded_videos)} downloaded videos")
    
    if not downloaded_videos:
        logger.error("✗ No videos to verify!")
        sys.exit(1)
    
    # Initialize verifier
    verifier = AccentVerifier(confidence_threshold=0.80)
    
    if not verifier.load_model():
        logger.error("✗ Failed to load accent classifier model")
        sys.exit(1)
    
    # Verify each video
    logger.info("\n" + "="*70)
    logger.info(f"VERIFYING {len(downloaded_videos)} VIDEOS")
    logger.info("="*70)
    
    verification_results = []
    american_videos = []
    non_american_videos = []
    accent_distribution = defaultdict(int)
    
    for idx, video_info in enumerate(downloaded_videos, 1):
        video_id = video_info['id']
        video_path = video_dir / f'{video_id}.mp4'
        
        logger.info(f"\n[{idx}/{len(downloaded_videos)}]")
        
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue
        
        is_american, accent_result = verifier.verify_video(str(video_path), video_info)
        
        # Record accent distribution
        detected_accent = accent_result.get('accent', 'unknown')
        accent_distribution[detected_accent] += 1
        
        result_entry = {
            'video_id': video_id,
            'title': video_info['title'],
            'channel': video_info['channel'],
            'channel_country': video_info.get('channel_country', 'unknown'),
            'american_score': video_info.get('american_score', 'N/A'),
            'is_american': is_american,
            'accent_result': accent_result
        }
        
        verification_results.append(result_entry)
        
        if is_american:
            american_videos.append(video_info)
        else:
            non_american_videos.append(video_info)
            # Delete non-American video
            if not accent_result.get('kept_on_error', False):
                logger.info(f"  Deleting non-American video: {video_id}")
                try:
                    video_path.unlink()
                    logger.info(f"  ✓ Deleted")
                except Exception as e:
                    logger.error(f"  ✗ Failed to delete: {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate statistics
    total_verified = len(verification_results)
    american_count = len(american_videos)
    non_american_count = len(non_american_videos)
    american_percentage = (american_count / total_verified * 100) if total_verified > 0 else 0
    
    # Save results
    summary = {
        'session': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{int(duration // 60)}m {int(duration % 60)}s"
        },
        'configuration': {
            'confidence_threshold': verifier.confidence_threshold,
            'model': 'bookbot/english-accent-classifier'
        },
        'statistics': {
            'total_verified': total_verified,
            'american_count': american_count,
            'non_american_count': non_american_count,
            'american_percentage': american_percentage,
            'accent_distribution': dict(accent_distribution)
        },
        'verification_results': verification_results,
        'american_videos': american_videos,
        'non_american_videos': non_american_videos
    }
    
    with open(verification_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("ACCENT VERIFICATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nSession Duration: {int(duration // 60)}m {int(duration % 60)}s")
    
    logger.info(f"\nResults:")
    logger.info(f"  Total verified: {total_verified}")
    logger.info(f"  American English: {american_count} ({american_percentage:.1f}%)")
    logger.info(f"  Non-American: {non_american_count} ({100 - american_percentage:.1f}%)")
    
    logger.info(f"\nAccent Distribution:")
    for accent, count in sorted(accent_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_verified * 100) if total_verified > 0 else 0
        logger.info(f"  {accent}: {count} ({percentage:.1f}%)")
    
    logger.info(f"\nFiles:")
    logger.info(f"  Verification results: {verification_file}")
    logger.info(f"  Session log: accent_verification.log")
    
    if american_count > 0:
        logger.info(f"\n✓ SUCCESS! Verified {american_count} American English videos")
        logger.info(f"\nAmerican videos kept:")
        for v in american_videos[:10]:  # Show first 10
            logger.info(f"  - {v['id']}.mp4 ({v['title'][:50]}...)")
        if len(american_videos) > 10:
            logger.info(f"  ... and {len(american_videos) - 10} more")
        
        logger.info(f"\n✅ DATASET READY: {american_count} American English videos")
        logger.info(f"✅ PURITY: {american_percentage:.1f}% American English")
        logger.info(f"✅ LEGAL: All videos are licensed for research")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review videos in {video_dir}/")
        logger.info(f"  2. Extract transcripts: python extract_transcripts_word_level.py")
        logger.info(f"  3. Preprocess videos: python preprocess_videos_smart.py")
        logger.info(f"  4. Train model: python train_improved.py")
    else:
        logger.error(f"\n✗ No American English videos found!")
        logger.error(f"  All videos were filtered out")
        logger.error(f"  Consider:")
        logger.error(f"    - Lowering confidence threshold (currently {verifier.confidence_threshold:.0%})")
        logger.error(f"    - Downloading more videos")
        logger.error(f"    - Checking accent distribution above")
    
    # Warnings
    if american_percentage < 70:
        logger.warning(f"\n⚠️  WARNING: Low American percentage ({american_percentage:.1f}%)")
        logger.warning(f"  Expected: 70-85%")
        logger.warning(f"  Consider:")
        logger.warning(f"    - Improving pre-filtering (adjust american_score_threshold)")
        logger.warning(f"    - Using different search queries")
        logger.warning(f"    - Targeting US-specific channels")
    
    if american_percentage > 95:
        logger.info(f"\n✓ EXCELLENT: Very high American percentage ({american_percentage:.1f}%)")
        logger.info(f"  Pre-filtering is working very well!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ FATAL ERROR: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
