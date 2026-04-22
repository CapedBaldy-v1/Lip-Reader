# YouTube Dataset Scraper for Swin-VALLR

Complete pipeline to build a lip-reading dataset from trainable YouTube videos.

## Overview

This scraper system:
1. **Searches** for candidate videos on YouTube
2. **Checks** trainability status via YouTube's Video Trainability API
3. **Downloads** videos and captions
4. **Filters** for visual quality (face detection, lip visibility, etc.)
5. **Extracts** lip ROI sequences (96x96 pixels, 50 frames @ 25fps)
6. **Aligns** transcripts with video clips
7. **Saves** in training-ready format for Swin-VALLR

## Legal & Ethical

✅ **Only uses videos where creators have explicitly enabled AI training** (`permitted: "all"`)  
✅ **Respects YouTube's Terms of Service**  
✅ **Stores trainability status with metadata**  

## Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install yt-dlp google-api-python-client

# Already installed (from main project)
# - opencv-python
# - mediapipe
# - numpy
# - torch
```

### 2. Get YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable **YouTube Data API v3**
4. Create credentials → API key
5. Copy the API key

## Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete pipeline in one command:

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_YOUTUBE_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset \
  --full_pipeline \
  --max_per_query 50 \
  --require_hd \
  --max_workers 4
```

This will:
- Search for ~800 candidate videos (16 queries × 50 results)
- Check trainability for all candidates
- Download and process trainable videos
- Create dataset in `youtube_dataset/`

### Option 2: Step-by-Step

#### Step 1: Search for Videos

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --search_only \
  --max_per_query 50 \
  --require_hd
```

Output: `youtube_scraper/data/candidate_video_ids.txt`

#### Step 2: Check Trainability

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --video_ids youtube_scraper/data/candidate_video_ids.txt \
  --check_only
```

Output: `youtube_scraper/data/trainable_video_ids.txt`

#### Step 3: Build Dataset

```bash
python youtube_scraper/build_youtube_dataset.py \
  --video_ids youtube_scraper/data/trainable_video_ids.txt \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset \
  --build_only \
  --max_workers 4
```

Output: Complete dataset in `youtube_dataset/`

## Output Structure

```
youtube_dataset/
├── raw_videos/              # Downloaded videos
│   ├── VIDEO_ID_1.mp4
│   ├── VIDEO_ID_2.mp4
│   └── ...
├── captions/                # Downloaded captions
│   ├── VIDEO_ID_1.en.vtt
│   ├── VIDEO_ID_2.en.vtt
│   └── ...
├── processed/
│   ├── videos/              # Lip ROI sequences (.npy)
│   │   ├── VIDEO_ID_1_clip_0000.npy
│   │   ├── VIDEO_ID_1_clip_0001.npy
│   │   └── ...
│   └── labels/              # Transcripts (.txt)
│       ├── VIDEO_ID_1_clip_0000.txt
│       ├── VIDEO_ID_1_clip_0001.txt
│       └── ...
├── metadata/
│   ├── processing_results.json
│   └── dataset_summary.json
└── logs/
```

## Configuration

### Search Parameters

- `--queries`: Custom search queries (uses 16 defaults if not specified)
- `--max_per_query`: Results per query (default: 50, max: 50)
- `--min_views`: Minimum view count filter
- `--require_hd`: Only HD videos

### Dataset Parameters

- `--roi_size`: Lip ROI size in pixels (default: 96)
- `--target_fps`: Target frames per second (default: 25)
- `--num_frames`: Frames per clip (default: 50)
- `--max_workers`: Parallel workers (default: 4)

### Quality Filters (Automatic)

- Face detection rate: ≥70%
- Lip size ratio: ≥3% of frame
- Mouth openness std: ≥0.004 (speech activity)
- Motion range: 0.003 to 0.08
- Sharpness: ≥10.0

## Expected Results

### Trainability Rate

Based on YouTube's current settings:
- **Expected trainable rate: 5-15%** of searched videos
- Most creators have training **disabled by default**
- Higher rates in educational/news content

### Dataset Size Estimates

**For 1,000 candidate videos:**
- Trainable: ~50-150 videos (5-15%)
- Passing quality: ~30-100 videos (60-70% of trainable)
- Total clips: ~1,500-5,000 clips
- Storage: ~1-3 GB

**For 10,000 candidate videos:**
- Trainable: ~500-1,500 videos
- Passing quality: ~300-1,000 videos
- Total clips: ~15,000-50,000 clips
- Storage: ~10-30 GB

### Processing Time

**Per video (5 minutes):**
- Download: 30 seconds
- Quality check: 10 seconds
- ROI extraction: 2-3 minutes
- Total: ~4 minutes

**For 100 videos with 4 workers:**
- Sequential: ~6.7 hours
- Parallel (4 workers): ~1.7 hours

## Troubleshooting

### API Quota Exceeded

YouTube Data API has daily quotas:
- **10,000 units/day** (free tier)
- Search: 100 units per request
- Video details: 1 unit per request
- Trainability check: 1 unit per request

**Solution:** Spread searches across multiple days or request quota increase.

### No Trainable Videos Found

**Possible causes:**
- Most creators have training disabled
- Search queries not targeting right content
- API key issues

**Solutions:**
- Try different search queries (educational, news, tutorials)
- Increase `--max_per_query` to search more videos
- Check API key permissions

### MediaPipe Not Detecting Faces

**Possible causes:**
- Poor video quality
- Profile/side views
- Occlusions (masks, hands)

**Solutions:**
- Enable `--require_hd` for better quality
- Adjust quality thresholds in code
- Use different search queries for frontal faces

### Download Failures

**Possible causes:**
- Video removed/private
- Geographic restrictions
- Network issues

**Solutions:**
- Script automatically skips failed downloads
- Check logs in `youtube_dataset/logs/`
- Retry with `--video_ids` of failed videos

## Advanced Usage

### Custom Search Queries

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --queries "news anchor" "ted talk" "online lecture" \
  --search_only
```

### Resume Failed Processing

If processing was interrupted:

```bash
# Get list of already processed videos
ls youtube_dataset/raw_videos/ | sed 's/.mp4//' > processed.txt

# Filter out processed videos from trainable list
comm -23 <(sort trainable_video_ids.txt) <(sort processed.txt) > remaining.txt

# Resume with remaining videos
python youtube_scraper/build_youtube_dataset.py \
  --video_ids remaining.txt \
  --output_dir youtube_dataset \
  --build_only
```

### Batch Processing

For very large datasets, process in batches:

```bash
# Split trainable videos into batches of 100
split -l 100 trainable_video_ids.txt batch_

# Process each batch
for batch in batch_*; do
  python youtube_scraper/build_youtube_dataset.py \
    --video_ids $batch \
    --output_dir youtube_dataset \
    --build_only
done
```

## Integration with Training

After building the dataset:

```bash
# Train Swin-VALLR model
python train_swin_vallr.py \
  --data_dir youtube_dataset/processed \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0001
```

## Performance Tips

1. **Use SSD for output directory** (faster I/O)
2. **Adjust `--max_workers`** based on CPU cores
3. **Enable `--require_hd`** to reduce low-quality videos
4. **Use `--min_views`** to filter for popular (likely better quality) videos
5. **Process in batches** for very large datasets

## Monitoring

### Check Progress

```bash
# Count processed clips
ls youtube_dataset/processed/videos/*.npy | wc -l

# Check dataset summary
cat youtube_dataset/metadata/dataset_summary.json | jq
```

### View Logs

```bash
# Latest log
tail -f logs/youtube_dataset_builder/$(ls -t logs/youtube_dataset_builder/ | head -1)
```

## Citation

If you use this scraper in your research, please cite:

```bibtex
@software{youtube_lipreading_scraper,
  title={YouTube Dataset Scraper for Visual Speech Recognition},
  author={Your Name},
  year={2026},
  url={https://github.com/CapedBaldy-v1/Lip-Reader}
}
```

## License

MIT License - See main project LICENSE file.

## Support

For issues or questions:
1. Check this README
2. Review logs in `youtube_dataset/logs/`
3. Open an issue on GitHub
