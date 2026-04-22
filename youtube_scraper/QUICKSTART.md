# YouTube Dataset Scraper - Quick Start Guide

Get your lip-reading dataset up and running in 15 minutes!

## Prerequisites

- Python 3.9+
- YouTube Data API v3 key
- ~50 GB free disk space (for 100K clips)
- Internet connection

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd /media/agam/Local\ Disk/the_code/roman/majorproject

# Install scraper dependencies
pip install -r youtube_scraper/requirements.txt

# Install ffmpeg (if not already installed)
sudo apt install ffmpeg  # Ubuntu/Debian
```

## Step 2: Get YouTube API Key (5 minutes)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Name it "Lip Reading Dataset" → Create
4. In the search bar, type "YouTube Data API v3" → Enable it
5. Go to "Credentials" → "Create Credentials" → "API Key"
6. Copy your API key (looks like: `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`)

**Important:** Keep your API key secret!

## Step 3: Test the Setup (2 minutes)

```bash
# Test without API calls
python youtube_scraper/test_scraper.py

# Test with API calls (replace YOUR_API_KEY)
python youtube_scraper/test_scraper.py --api_key YOUR_API_KEY --full
```

Expected output: `✓ All tests passed!`

## Step 4: Build Your Dataset (varies)

### Option A: Quick Test (10 videos, ~40 minutes)

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test \
  --full_pipeline \
  --max_per_query 10 \
  --require_hd \
  --max_workers 4
```

Expected result: ~5-50 clips (depending on trainability rate)

### Option B: Small Dataset (100 videos, ~7 hours)

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset \
  --full_pipeline \
  --max_per_query 20 \
  --require_hd \
  --max_workers 4
```

Expected result: ~500-2,000 clips (~1-3 GB)

### Option C: Large Dataset (1000+ videos, ~3 days)

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset \
  --full_pipeline \
  --max_per_query 50 \
  --require_hd \
  --max_workers 4
```

Expected result: ~5,000-20,000 clips (~10-40 GB)

## Step 5: Monitor Progress

### Check how many clips have been created:

```bash
# Count video clips
ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/videos/*.npy | wc -l

# Count label files
ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/labels/*.txt | wc -l
```

### View the summary:

```bash
cat /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/metadata/dataset_summary.json
```

### Watch the logs:

```bash
tail -f logs/youtube_dataset_builder/$(ls -t logs/youtube_dataset_builder/ | head -1)
```

## Step 6: Train Your Model

Once you have enough clips (recommended: 5,000+):

```bash
python train_swin_vallr.py \
  --data_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0001
```

## Troubleshooting

### "API quota exceeded"

**Problem:** YouTube API has daily limits (10,000 units/day)

**Solution:** 
- Wait 24 hours for quota reset
- Or request quota increase in Google Cloud Console
- Or split into multiple days

### "No trainable videos found"

**Problem:** Most creators have training disabled by default

**Solution:**
- This is normal! Expected rate: 5-15%
- Increase `--max_per_query` to search more videos
- Try different search queries (educational content has higher rates)

### "MediaPipe not detecting faces"

**Problem:** Video quality too low or faces not visible

**Solution:**
- Already handled! Script automatically filters low-quality videos
- Use `--require_hd` flag (already in examples above)

### "Download failed"

**Problem:** Video removed, private, or geo-restricted

**Solution:**
- Script automatically skips failed downloads
- Check logs for details
- This is normal, ~10-20% of videos may fail

## Understanding the Output

### Directory Structure:

```
youtube_dataset/
├── raw_videos/           # Original downloaded videos
├── captions/             # Downloaded captions/subtitles
├── processed/
│   ├── videos/          # 👈 USE THIS for training (lip ROI .npy files)
│   └── labels/          # 👈 USE THIS for training (transcript .txt files)
├── metadata/            # Processing statistics
└── logs/                # Detailed logs
```

### File Format:

**Video files** (`VIDEO_ID_clip_0000.npy`):
- Shape: (50, 96, 96, 3)
- 50 frames at 25 fps = 2 seconds
- 96x96 pixels = lip region
- 3 channels = RGB

**Label files** (`VIDEO_ID_clip_0000.txt`):
- Plain text transcript
- Aligned with video clip
- Example: "Hello, welcome to this tutorial"

## Tips for Best Results

### 1. Start Small
- Test with 10 videos first
- Verify output quality
- Then scale up

### 2. Use Good Search Queries
Best results from:
- News broadcasts
- Educational lectures
- Interviews
- Public speeches

Avoid:
- Music videos
- Gaming content
- Vlogs with lots of cuts

### 3. Monitor Quality
Check a few random clips:
```bash
# View a random clip
python -c "import numpy as np; import cv2; frames = np.load('youtube_dataset/processed/videos/VIDEO_ID_clip_0000.npy'); [cv2.imshow('frame', frame) and cv2.waitKey(40) for frame in frames]; cv2.destroyAllWindows()"
```

### 4. Optimize for Your Hardware
- **4 CPU cores:** `--max_workers 2`
- **8 CPU cores:** `--max_workers 4`
- **16+ CPU cores:** `--max_workers 8`

### 5. Save API Quota
If you already have video IDs from another source:
```bash
# Skip search, just check trainability and build
echo "VIDEO_ID_1" > my_videos.txt
echo "VIDEO_ID_2" >> my_videos.txt

python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --video_ids my_videos.txt \
  --output_dir youtube_dataset \
  --check_only

# Then build from trainable ones
python youtube_scraper/build_youtube_dataset.py \
  --video_ids youtube_scraper/data/trainable_video_ids.txt \
  --output_dir youtube_dataset \
  --build_only
```

## Expected Timeline

| Dataset Size | Videos | Clips | Storage | Time (4 workers) |
|--------------|--------|-------|---------|------------------|
| Test | 10 | 50-200 | 100 MB | 40 min |
| Small | 100 | 500-2K | 1-3 GB | 7 hours |
| Medium | 500 | 2.5K-10K | 5-20 GB | 35 hours |
| Large | 1000+ | 5K-20K | 10-40 GB | 70 hours |

**Note:** Times include search, trainability check, download, and processing.

## Next Steps

After building your dataset:

1. **Verify quality:** Check random samples
2. **Train model:** Use `train_swin_vallr.py`
3. **Evaluate:** Test on held-out videos
4. **Iterate:** Add more data if needed

## Getting Help

1. Check `youtube_scraper/README.md` for detailed docs
2. Review logs in `youtube_dataset/logs/`
3. Open an issue on GitHub with:
   - Error message
   - Log file
   - Command you ran

## Legal Reminder

✅ This scraper only uses videos where creators have **explicitly enabled AI training**  
✅ Respects YouTube's Terms of Service  
✅ Stores trainability status with metadata  

**Do not:**
- Redistribute original videos
- Use for commercial purposes without checking licenses
- Ignore creator preferences

---

**You're all set! Start building your dataset now! 🚀**
