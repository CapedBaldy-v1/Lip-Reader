#!/usr/bin/env python3
"""
YouTube Dataset Builder
Complete pipeline to build lip-reading dataset from trainable YouTube videos.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_utils import setup_logging, log_exception
from data_pipeline import MediaPipeProcessor, VisualFilterConfig, MEDIAPIPE_AVAILABLE

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not installed")
    print("Install with: pip install yt-dlp")

LOGGER = setup_logging("youtube_dataset_builder")


@dataclass
class VideoProcessingResult:
    """Result of processing a single video."""
    video_id: str
    success: bool
    clips_created: int = 0
    duration: float = 0.0
    quality_score: float = 0.0
    error: Optional[str] = None
    processing_time: float = 0.0


class YouTubeDatasetBuilder:
    """
    Build a complete lip-reading dataset from trainable YouTube videos.
    
    Pipeline:
    1. Download video and captions
    2. Check visual quality
    3. Extract lip ROI clips
    4. Align transcripts
    5. Save in training format
    """
    
    def __init__(
        self,
        output_dir: str,
        roi_size: int = 96,
        target_fps: int = 25,
        clip_duration: float = 2.0,
        num_frames: int = 50,
        max_workers: int = 4
    ):
        """
        Initialize the dataset builder.
        
        Args:
            output_dir: Root directory for dataset
            roi_size: Size of lip ROI (pixels)
            target_fps: Target frames per second
            clip_duration: Duration of each clip (seconds)
            num_frames: Number of frames per clip
            max_workers: Number of parallel workers
        """
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for ROI extraction")
        
        self.output_dir = Path(output_dir)
        self.roi_size = roi_size
        self.target_fps = target_fps
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.max_workers = max_workers
        
        # Setup directory structure
        self.setup_directories()
        
        # Initialize MediaPipe processor
        self.processor = MediaPipeProcessor(
            roi_size=roi_size,
            stabilize=True,
            crop_mode="adaptive",
            align_mouth=True,
            profile_aware=True
        )
        
        # Visual quality filter config
        self.filter_config = VisualFilterConfig(
            enabled=True,
            sample_frames=100,
            min_face_rate=0.7,
            min_lip_size_ratio=0.03,
            min_openness_std=0.004,
            min_motion=0.003,
            max_motion=0.08,
            min_sharpness=10.0
        )
        
        LOGGER.info("YouTubeDatasetBuilder initialized")
        LOGGER.info(f"Output directory: {self.output_dir}")
        LOGGER.info(f"ROI size: {roi_size}, FPS: {target_fps}, Frames: {num_frames}")
    
    def setup_directories(self):
        """Create directory structure."""
        dirs = [
            'raw_videos',
            'captions',
            'processed/videos',
            'processed/labels',
            'logs',
            'metadata'
        ]
        
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
        
        LOGGER.info("Directory structure created")
    
    def download_video(self, video_id: str) -> Optional[str]:
        """
        Download YouTube video.
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            Path to downloaded video file, or None if failed
        """
        output_path = self.output_dir / 'raw_videos' / f'{video_id}.mp4'
        
        if output_path.exists():
            LOGGER.info(f"Video {video_id} already downloaded")
            return str(output_path)
        
        try:
            ydl_opts = {
                'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
                'outtmpl': str(self.output_dir / 'raw_videos' / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
            # Find the downloaded file (extension might vary)
            for ext in ['.mp4', '.webm', '.mkv']:
                potential_path = self.output_dir / 'raw_videos' / f'{video_id}{ext}'
                if potential_path.exists():
                    # Convert to mp4 if needed
                    if ext != '.mp4':
                        LOGGER.info(f"Converting {ext} to mp4 for {video_id}")
                        self._convert_to_mp4(str(potential_path), str(output_path))
                        potential_path.unlink()
                    return str(output_path)
            
            LOGGER.warning(f"Downloaded video not found for {video_id}")
            return None
        
        except Exception as e:
            log_exception(LOGGER, e, f"Failed to download video {video_id}")
            return None
    
    def _convert_to_mp4(self, input_path: str, output_path: str):
        """Convert video to mp4 using ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac', '-y', output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            log_exception(LOGGER, e, f"Failed to convert {input_path}")
    
    def download_captions(self, video_id: str) -> Optional[str]:
        """
        Download video captions/subtitles.
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            Path to caption file, or None if not available
        """
        caption_path = self.output_dir / 'captions' / f'{video_id}.en.vtt'
        
        if caption_path.exists():
            LOGGER.info(f"Captions for {video_id} already downloaded")
            return str(caption_path)
        
        try:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'outtmpl': str(self.output_dir / 'captions' / '%(id)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
            # Check if caption file was created
            if caption_path.exists():
                return str(caption_path)
            
            LOGGER.warning(f"No captions available for {video_id}")
            return None
        
        except Exception as e:
            log_exception(LOGGER, e, f"Failed to download captions for {video_id}")
            return None
    
    def check_video_quality(self, video_path: str) -> Tuple[bool, float]:
        """
        Check if video meets quality requirements.
        
        Args:
            video_path: Path to video file
        
        Returns:
            (passes_quality_check, quality_score)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, 0.0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0 or fps == 0:
                cap.release()
                return False, 0.0
            
            # Sample frames for quality check
            sample_count = min(self.filter_config.sample_frames, total_frames)
            sample_interval = max(1, total_frames // sample_count)
            
            face_detections = 0
            lip_sizes = []
            openness_values = []
            sharpness_values = []
            
            for i in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Check for face and lip metrics
                metrics = self.processor.extract_lip_metrics(frame)
                
                if metrics:
                    face_detections += 1
                    lip_sizes.append(metrics.size)
                    openness_values.append(metrics.openness)
                    
                    # Calculate sharpness (Laplacian variance)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharpness_values.append(sharpness)
            
            cap.release()
            
            # Calculate metrics
            if face_detections == 0:
                return False, 0.0
            
            face_rate = face_detections / sample_count
            avg_lip_size = np.mean(lip_sizes) if lip_sizes else 0
            lip_size_ratio = avg_lip_size / frame.shape[0]  # Relative to frame height
            openness_std = np.std(openness_values) if openness_values else 0
            avg_sharpness = np.mean(sharpness_values) if sharpness_values else 0
            
            # Check thresholds
            passes = (
                face_rate >= self.filter_config.min_face_rate and
                lip_size_ratio >= self.filter_config.min_lip_size_ratio and
                openness_std >= self.filter_config.min_openness_std and
                avg_sharpness >= self.filter_config.min_sharpness
            )
            
            # Calculate quality score (0-1)
            quality_score = (
                face_rate * 0.4 +
                min(lip_size_ratio / 0.05, 1.0) * 0.2 +
                min(openness_std / 0.01, 1.0) * 0.2 +
                min(avg_sharpness / 20.0, 1.0) * 0.2
            )
            
            LOGGER.info(f"Quality check: face_rate={face_rate:.2f}, "
                       f"lip_size_ratio={lip_size_ratio:.3f}, "
                       f"openness_std={openness_std:.4f}, "
                       f"sharpness={avg_sharpness:.1f}, "
                       f"score={quality_score:.2f}, passes={passes}")
            
            return passes, quality_score
        
        except Exception as e:
            log_exception(LOGGER, e, f"Error checking quality for {video_path}")
            return False, 0.0
    
    def parse_vtt_captions(self, caption_path: str) -> List[Dict]:
        """
        Parse VTT caption file.
        
        Args:
            caption_path: Path to VTT file
        
        Returns:
            List of caption segments with timestamps
        """
        captions = []
        
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for timestamp line (format: 00:00:00.000 --> 00:00:02.000)
                if '-->' in line:
                    parts = line.split('-->')
                    start_time = self._parse_timestamp(parts[0].strip())
                    end_time = self._parse_timestamp(parts[1].strip())
                    
                    # Next line(s) contain the text
                    text_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    text = ' '.join(text_lines)
                    
                    # Remove HTML tags and formatting
                    text = self._clean_caption_text(text)
                    
                    if text:
                        captions.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })
                
                i += 1
            
            LOGGER.info(f"Parsed {len(captions)} caption segments from {caption_path}")
        
        except Exception as e:
            log_exception(LOGGER, e, f"Error parsing captions from {caption_path}")
        
        return captions
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert VTT timestamp to seconds."""
        # Format: HH:MM:SS.mmm or MM:SS.mmm
        parts = timestamp.split(':')
        
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            return float(parts[0])
    
    def _clean_caption_text(self, text: str) -> str:
        """Clean caption text (remove HTML, formatting, etc.)."""
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove speaker labels like [Music], [Applause]
        text = re.sub(r'\[[^\]]+\]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_clips(
        self,
        video_id: str,
        video_path: str,
        captions: List[Dict]
    ) -> int:
        """
        Extract lip ROI clips from video with aligned transcripts.
        
        Args:
            video_id: YouTube video ID
            video_path: Path to video file
            captions: List of caption segments
        
        Returns:
            Number of clips created
        """
        clips_created = 0
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return 0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Generate clip timestamps (every 2 seconds)
            clip_starts = np.arange(0, duration - self.clip_duration, self.clip_duration)
            
            for clip_idx, start_time in enumerate(clip_starts):
                end_time = start_time + self.clip_duration
                
                # Find matching caption
                caption_text = self._find_caption_for_clip(captions, start_time, end_time)
                
                if not caption_text:
                    continue  # Skip clips without captions
                
                # Extract frames for this clip
                start_frame = int(start_time * fps)
                frames_to_extract = []
                
                for frame_offset in range(self.num_frames):
                    frame_idx = start_frame + int(frame_offset * fps / self.target_fps)
                    
                    if frame_idx >= total_frames:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frames_to_extract.append(frame)
                
                if len(frames_to_extract) < self.num_frames:
                    continue  # Skip incomplete clips
                
                # Extract lip ROIs
                rois = []
                for frame in frames_to_extract:
                    roi = self.processor.extract_lip_roi(frame)
                    
                    if roi is not None:
                        rois.append(roi)
                    elif rois:
                        rois.append(rois[-1])  # Repeat last ROI
                    else:
                        rois.append(np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8))
                
                if len(rois) != self.num_frames:
                    continue
                
                # Save clip
                clip_name = f"{video_id}_clip_{clip_idx:04d}"
                
                # Save video ROIs
                video_output = self.output_dir / 'processed' / 'videos' / f'{clip_name}.npy'
                np.save(video_output, np.array(rois))
                
                # Save transcript
                label_output = self.output_dir / 'processed' / 'labels' / f'{clip_name}.txt'
                with open(label_output, 'w', encoding='utf-8') as f:
                    f.write(caption_text)
                
                clips_created += 1
            
            cap.release()
            
            LOGGER.info(f"Created {clips_created} clips from {video_id}")
        
        except Exception as e:
            log_exception(LOGGER, e, f"Error extracting clips from {video_id}")
        
        return clips_created
    
    def _find_caption_for_clip(
        self,
        captions: List[Dict],
        start_time: float,
        end_time: float
    ) -> Optional[str]:
        """Find caption text that overlaps with clip time range."""
        matching_texts = []
        
        for caption in captions:
            # Check if caption overlaps with clip
            if caption['start'] < end_time and caption['end'] > start_time:
                matching_texts.append(caption['text'])
        
        if matching_texts:
            return ' '.join(matching_texts)
        
        return None
    
    def process_video(self, video_id: str) -> VideoProcessingResult:
        """
        Process a single video through the complete pipeline.
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            VideoProcessingResult object
        """
        start_time = datetime.now()
        
        LOGGER.info(f"Processing video: {video_id}")
        
        try:
            # Step 1: Download video
            video_path = self.download_video(video_id)
            if not video_path:
                return VideoProcessingResult(
                    video_id=video_id,
                    success=False,
                    error="Failed to download video"
                )
            
            # Step 2: Download captions
            caption_path = self.download_captions(video_id)
            if not caption_path:
                return VideoProcessingResult(
                    video_id=video_id,
                    success=False,
                    error="No captions available"
                )
            
            # Step 3: Check quality
            passes_quality, quality_score = self.check_video_quality(video_path)
            if not passes_quality:
                return VideoProcessingResult(
                    video_id=video_id,
                    success=False,
                    quality_score=quality_score,
                    error="Failed quality check"
                )
            
            # Step 4: Parse captions
            captions = self.parse_vtt_captions(caption_path)
            if not captions:
                return VideoProcessingResult(
                    video_id=video_id,
                    success=False,
                    error="Failed to parse captions"
                )
            
            # Step 5: Extract clips
            clips_created = self.extract_clips(video_id, video_path, captions)
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VideoProcessingResult(
                video_id=video_id,
                success=True,
                clips_created=clips_created,
                duration=duration,
                quality_score=quality_score,
                processing_time=processing_time
            )
        
        except Exception as e:
            log_exception(LOGGER, e, f"Error processing video {video_id}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VideoProcessingResult(
                video_id=video_id,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def build_dataset(
        self,
        video_ids: List[str],
        parallel: bool = True
    ) -> Dict:
        """
        Build complete dataset from list of video IDs.
        
        Args:
            video_ids: List of YouTube video IDs
            parallel: Whether to process videos in parallel
        
        Returns:
            Summary statistics dictionary
        """
        LOGGER.info(f"Starting dataset build with {len(video_ids)} videos")
        print(f"\n{'='*60}")
        print(f"BUILDING DATASET FROM {len(video_ids)} VIDEOS")
        print(f"{'='*60}\n")
        
        results = []
        
        if parallel and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_video, vid_id): vid_id 
                          for vid_id in video_ids}
                
                with tqdm(total=len(video_ids), desc="Processing videos") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                        
                        if result.success:
                            pbar.set_postfix({'clips': result.clips_created})
        else:
            # Sequential processing
            for vid_id in tqdm(video_ids, desc="Processing videos"):
                result = self.process_video(vid_id)
                results.append(result)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Save metadata
        self._save_metadata(results, summary)
        
        # Print summary
        self._print_summary(summary)
        
        LOGGER.info("Dataset build complete!")
        
        return summary
    
    def _generate_summary(self, results: List[VideoProcessingResult]) -> Dict:
        """Generate summary statistics from processing results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_clips = sum(r.clips_created for r in successful)
        total_duration = sum(r.duration for r in successful)
        avg_quality = np.mean([r.quality_score for r in successful]) if successful else 0
        total_time = sum(r.processing_time for r in results)
        
        # Error breakdown
        error_counts = {}
        for r in failed:
            error = r.error or "Unknown error"
            error_counts[error] = error_counts.get(error, 0) + 1
        
        return {
            'total_videos': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'total_clips': total_clips,
            'total_duration_hours': total_duration / 3600,
            'average_quality_score': avg_quality,
            'total_processing_time_hours': total_time / 3600,
            'error_breakdown': error_counts,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_metadata(self, results: List[VideoProcessingResult], summary: Dict):
        """Save processing results and summary to metadata files."""
        # Save detailed results
        results_file = self.output_dir / 'metadata' / 'processing_results.json'
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / 'metadata' / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        LOGGER.info(f"Metadata saved to {self.output_dir / 'metadata'}")
    
    def _print_summary(self, summary: Dict):
        """Print dataset build summary."""
        print(f"\n{'='*60}")
        print("DATASET BUILD SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos processed: {summary['total_videos']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"\nTotal clips created: {summary['total_clips']}")
        print(f"Total video duration: {summary['total_duration_hours']:.2f} hours")
        print(f"Average quality score: {summary['average_quality_score']:.2f}")
        print(f"Processing time: {summary['total_processing_time_hours']:.2f} hours")
        
        if summary['error_breakdown']:
            print(f"\nError breakdown:")
            for error, count in summary['error_breakdown'].items():
                print(f"  - {error}: {count}")
        
        print(f"{'='*60}\n")


def main():
    """CLI interface for dataset builder."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build lip-reading dataset from YouTube videos'
    )
    parser.add_argument(
        '--video_ids',
        required=True,
        help='File with video IDs (one per line)'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for dataset'
    )
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
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Process videos sequentially (no parallelization)'
    )
    
    args = parser.parse_args()
    
    # Load video IDs
    print(f"Loading video IDs from {args.video_ids}...")
    with open(args.video_ids, 'r') as f:
        video_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(video_ids)} video IDs")
    
    # Initialize builder
    builder = YouTubeDatasetBuilder(
        output_dir=args.output_dir,
        roi_size=args.roi_size,
        target_fps=args.target_fps,
        num_frames=args.num_frames,
        max_workers=args.max_workers
    )
    
    # Build dataset
    summary = builder.build_dataset(
        video_ids=video_ids,
        parallel=not args.sequential
    )
    
    print(f"\nDataset saved to: {args.output_dir}")
    print(f"  - processed/videos/ ({summary['total_clips']} .npy files)")
    print(f"  - processed/labels/ ({summary['total_clips']} .txt files)")
    print(f"  - metadata/ (processing results and summary)")


if __name__ == '__main__':
    main()
