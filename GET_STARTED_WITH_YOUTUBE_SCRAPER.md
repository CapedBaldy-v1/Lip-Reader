# Get Started with YouTube Dataset Scraper

**Complete guide to build your lip-reading dataset in under 30 minutes!**

---

## 📋 What You Need

- ✅ Python 3.9+ (you have this)
- ✅ Your existing Swin-VALLR project (you have this)
- ⏳ YouTube Data API v3 key (we'll get this)
- ⏳ 50-100 GB free disk space
- ⏳ Internet connection

---

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies (2 minutes)

```bash
# You're already in the project directory
cd /media/agam/Local\ Disk/the_code/roman/majorproject

# Install scraper dependencies
pip install -r youtube_scraper/requirements.txt

# Install ffmpeg (if not already installed)
sudo apt install ffmpeg

# Verify installation
python youtube_scraper/test_scraper.py
```

**Expected output:** `✓ All tests passed!` (except API tests)

---

### Step 2: Get YouTube API Key (5 minutes)

#### 2.1 Go to Google Cloud Console
Open: https://console.cloud.google.com/

#### 2.2 Create a Project
1. Click "Select a project" (top bar)
2. Click "NEW PROJECT"
3. Name: `Lip Reading Dataset`
4. Click "CREATE"
5. Wait ~30 seconds for project creation

#### 2.3 Enable YouTube Data API
1. In search bar, type: `YouTube Data API v3`
2. Click on "YouTube Data API v3"
3. Click "ENABLE"
4. Wait ~10 seconds

#### 2.4 Create API Key
1. Click "Credentials" (left sidebar)
2. Click "CREATE CREDENTIALS" → "API key"
3. Copy your API key (looks like: `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`)
4. Click "RESTRICT KEY" (recommended)
5. Under "API restrictions", select "Restrict key"
6. Check "YouTube Data API v3"
7. Click "SAVE"

**⚠️ IMPORTANT:** Keep your API key secret! Don't commit it to GitHub!

---

### Step 3: Test with Your API Key (2 minutes)

```bash
# Replace YOUR_API_KEY with your actual key
python youtube_scraper/test_scraper.py --api_key YOUR_API_KEY --full
```

**Expected output:**
```
✓ yt-dlp
✓ google-api-python-client
✓ opencv-python
✓ mediapipe
✓ numpy
✓ youtube_scraper modules
✓ MediaPipe initialized successfully
✓ Trainability check successful
✓ Video search successful
✓ All tests passed!
```

If you see this, **you're ready to go!** 🎉

---

## 🎬 Build Your First Dataset

### Option A: Quick Test (10 videos, ~40 minutes)

Perfect for testing the pipeline:

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test \
  --full_pipeline \
  --max_per_query 10 \
  --require_hd \
  --max_workers 4
```

**What this does:**
- Searches for 160 videos (16 queries × 10 results)
- Checks trainability (~8-24 trainable)
- Downloads and processes trainable videos
- Creates ~50-200 training clips
- Takes ~40 minutes

**Output location:**
```
/media/agam/Local Disk/the_code/roman/majorproject/youtube_dataset_test/
  processed/
    videos/  ← Use this for training
    labels/  ← Use this for training
```

---

### Option B: Production Dataset (1000+ videos, ~3 days)

For serious training:

```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset \
  --full_pipeline \
  --max_per_query 50 \
  --require_hd \
  --max_workers 4
```

**What this does:**
- Searches for 800 videos (16 queries × 50 results)
- Checks trainability (~40-120 trainable)
- Downloads and processes trainable videos
- Creates ~2,000-6,000 training clips
- Takes ~1-3 days (can run overnight)

**Output location:**
```
/media/agam/Local Disk/the_code/roman/majorproject/youtube_dataset/
  processed/
    videos/  ← 2,000-6,000 .npy files
    labels/  ← 2,000-6,000 .txt files
```

---

## 📊 Monitor Progress

### Check Clip Count
```bash
# Count video clips
ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/videos/*.npy | wc -l

# Count labels
ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/labels/*.txt | wc -l
```

### View Summary
```bash
cat /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/metadata/dataset_summary.json
```

### Watch Live Progress
```bash
# Watch the latest log file
tail -f logs/youtube_dataset_builder/$(ls -t logs/youtube_dataset_builder/ | head -1)
```

---

## 🎓 Train Your Model

Once you have 1,000+ clips:

```bash
python train_swin_vallr.py \
  --data_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0001
```

---

## 🔧 Troubleshooting

### Problem: "API quota exceeded"

**Cause:** YouTube API has daily limits (10,000 units/day)

**Solution:**
- Wait 24 hours for quota reset
- Or split your search across multiple days
- Or request quota increase in Google Cloud Console

### Problem: "No trainable videos found"

**Cause:** Most creators have training disabled (this is normal!)

**Solution:**
- Expected rate: 5-15% trainable
- Increase `--max_per_query` to search more videos
- Try different search queries

### Problem: "yt-dlp not found" or "google-api-python-client not found"

**Cause:** Dependencies not installed

**Solution:**
```bash
pip install -r youtube_scraper/requirements.txt
```

### Problem: "MediaPipe not detecting faces"

**Cause:** Video quality too low

**Solution:**
- Already handled! Script filters low-quality videos automatically
- Use `--require_hd` flag (already in examples)

### Problem: "Download failed"

**Cause:** Video removed, private, or geo-restricted

**Solution:**
- Script automatically skips failed downloads
- This is normal, ~10-20% may fail
- Check logs for details

---

## 📁 Understanding the Output

### Directory Structure
```
youtube_dataset/
├── raw_videos/              # Downloaded MP4 files (can delete after processing)
├── captions/                # Downloaded captions (can delete after processing)
├── processed/
│   ├── videos/             # 👈 USE THIS for training
│   │   ├── VIDEO_ID_clip_0000.npy
│   │   ├── VIDEO_ID_clip_0001.npy
│   │   └── ...
│   └── labels/             # 👈 USE THIS for training
│       ├── VIDEO_ID_clip_0000.txt
│       ├── VIDEO_ID_clip_0001.txt
│       └── ...
├── metadata/
│   ├── processing_results.json  # Detailed results
│   └── dataset_summary.json     # Summary statistics
└── logs/                   # Execution logs
```

### File Formats

**Video files** (`.npy`):
- NumPy array: shape (50, 96, 96, 3)
- 50 frames at 25 fps = 2 seconds
- 96×96 pixels = lip region
- 3 channels = RGB
- Data type: uint8 (0-255)

**Label files** (`.txt`):
- Plain text transcript
- Aligned with video clip
- Example: "Hello, welcome to this tutorial"

---

## 💡 Pro Tips

### 1. Start Small
Test with 10 videos first to verify everything works, then scale up.

### 2. Run Overnight
Large datasets take time. Start before bed, check in the morning.

### 3. Use Your Data Partition
Your root partition has only 21 GB free. Always use:
```
/media/agam/Local Disk/the_code/roman/majorproject/youtube_dataset
```

### 4. Save Your API Key
Create a file to store it:
```bash
echo "YOUR_API_KEY" > ~/.youtube_api_key
chmod 600 ~/.youtube_api_key

# Then use it:
API_KEY=$(cat ~/.youtube_api_key)
python youtube_scraper/build_youtube_dataset.py --api_key $API_KEY ...
```

### 5. Resume if Interrupted
If the script stops, just run it again with the same output directory. It will:
- Skip already downloaded videos
- Use cached trainability results
- Continue where it left off

---

## 📚 Additional Resources

- **Full Documentation:** `youtube_scraper/README.md`
- **Quick Start:** `youtube_scraper/QUICKSTART.md`
- **Pipeline Diagram:** `youtube_scraper/PIPELINE.md`
- **Examples:** `youtube_scraper/example_usage.py`
- **Summary:** `youtube_scraper/SUMMARY.md`

---

## 🎯 Your Action Plan

### Today (30 minutes):
1. ✅ Install dependencies
2. ✅ Get API key
3. ✅ Run test
4. ✅ Start quick test (10 videos)

### This Week:
1. ✅ Verify test dataset quality
2. ✅ Start production dataset (1000 videos)
3. ✅ Let it run for 2-3 days

### Next Week:
1. ✅ Check dataset (should have 2,000-6,000 clips)
2. ✅ Train model with your new dataset
3. ✅ Evaluate results

---

## 🆘 Need Help?

1. **Check logs:** `youtube_dataset/logs/`
2. **Read docs:** `youtube_scraper/README.md`
3. **Run test:** `python youtube_scraper/test_scraper.py --api_key YOUR_KEY --full`
4. **Open issue:** GitHub with error message and log file

---

## ✅ Quick Command Reference

```bash
# Test setup
python youtube_scraper/test_scraper.py --api_key YOUR_KEY --full

# Quick test (10 videos)
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_KEY \
  --output_dir youtube_dataset_test \
  --full_pipeline \
  --max_per_query 10 \
  --require_hd

# Production dataset (1000 videos)
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_KEY \
  --output_dir youtube_dataset \
  --full_pipeline \
  --max_per_query 50 \
  --require_hd

# Check progress
ls youtube_dataset/processed/videos/*.npy | wc -l

# Train model
python train_swin_vallr.py \
  --data_dir youtube_dataset/processed \
  --batch_size 16 \
  --epochs 50
```

---

**You're all set! Start building your dataset now! 🚀**

```bash
# Copy this command and replace YOUR_API_KEY
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test \
  --full_pipeline \
  --max_per_query 10 \
  --require_hd \
  --max_workers 4
```
