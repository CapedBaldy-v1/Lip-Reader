# YouTube Dataset Scraper - Setup Checklist

Use this checklist to get your dataset scraper up and running!

---

## ☐ Phase 1: Installation (5 minutes)

### ☐ 1.1 Install Python Dependencies
```bash
cd /media/agam/Local\ Disk/the_code/roman/majorproject
pip install -r youtube_scraper/requirements.txt
```

**Expected output:** Successfully installed yt-dlp, google-api-python-client, etc.

### ☐ 1.2 Install ffmpeg
```bash
sudo apt install ffmpeg
```

**Verify:**
```bash
ffmpeg -version
```

### ☐ 1.3 Test Imports
```bash
python youtube_scraper/test_scraper.py
```

**Expected:** `✓ All tests passed!` (except API tests)

---

## ☐ Phase 2: Get API Key (5 minutes)

### ☐ 2.1 Go to Google Cloud Console
- Open: https://console.cloud.google.com/
- Sign in with your Google account

### ☐ 2.2 Create Project
- Click "Select a project" → "NEW PROJECT"
- Name: `Lip Reading Dataset`
- Click "CREATE"
- Wait ~30 seconds

### ☐ 2.3 Enable YouTube Data API v3
- Search: `YouTube Data API v3`
- Click "ENABLE"
- Wait ~10 seconds

### ☐ 2.4 Create API Key
- Go to "Credentials"
- Click "CREATE CREDENTIALS" → "API key"
- Copy your API key: `AIzaSy...`
- Click "RESTRICT KEY"
- Select "YouTube Data API v3"
- Click "SAVE"

### ☐ 2.5 Save API Key Securely
```bash
echo "YOUR_API_KEY" > ~/.youtube_api_key
chmod 600 ~/.youtube_api_key
```

**Verify:**
```bash
cat ~/.youtube_api_key
```

---

## ☐ Phase 3: Test with API Key (2 minutes)

### ☐ 3.1 Run Full Test
```bash
API_KEY=$(cat ~/.youtube_api_key)
python youtube_scraper/test_scraper.py --api_key $API_KEY --full
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

### ☐ 3.2 If Tests Pass
**You're ready to build your dataset! 🎉**

### ☐ 3.3 If Tests Fail
- Check error message
- Verify API key is correct
- Check internet connection
- Review `youtube_scraper/README.md` troubleshooting section

---

## ☐ Phase 4: Quick Test Dataset (40 minutes)

### ☐ 4.1 Start Quick Test
```bash
API_KEY=$(cat ~/.youtube_api_key)

python youtube_scraper/build_youtube_dataset.py \
  --api_key $API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test \
  --full_pipeline \
  --max_per_query 10 \
  --require_hd \
  --max_workers 4
```

### ☐ 4.2 Monitor Progress
Open a new terminal:
```bash
# Watch clip count
watch -n 10 'ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test/processed/videos/*.npy 2>/dev/null | wc -l'
```

### ☐ 4.3 Wait for Completion
**Expected time:** ~40 minutes  
**Expected clips:** 50-200

### ☐ 4.4 Check Results
```bash
# Count clips
ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test/processed/videos/*.npy | wc -l

# View summary
cat /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset_test/metadata/dataset_summary.json
```

### ☐ 4.5 Verify Quality
```bash
# View a random clip
python -c "
import numpy as np
import cv2
import glob
import random

clips = glob.glob('/media/agam/Local Disk/the_code/roman/majorproject/youtube_dataset_test/processed/videos/*.npy')
if clips:
    sample = random.choice(clips)
    print(f'Viewing: {sample}')
    frames = np.load(sample)
    for frame in frames:
        cv2.imshow('Lip ROI', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
else:
    print('No clips found')
"
```

**Expected:** Should see 50 frames of lip movements

---

## ☐ Phase 5: Production Dataset (2-3 days)

### ☐ 5.1 Start Production Build
```bash
API_KEY=$(cat ~/.youtube_api_key)

python youtube_scraper/build_youtube_dataset.py \
  --api_key $API_KEY \
  --output_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset \
  --full_pipeline \
  --max_per_query 50 \
  --require_hd \
  --max_workers 4
```

### ☐ 5.2 Let It Run
- Can run overnight
- Can run in background with `nohup`
- Will take 2-3 days for 1000 videos

### ☐ 5.3 Monitor Progress (Daily)
```bash
# Check clip count
ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/videos/*.npy | wc -l

# View summary
cat /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/metadata/dataset_summary.json | jq
```

### ☐ 5.4 Target Metrics
- ☐ Minimum: 2,000 clips
- ☐ Recommended: 5,000 clips
- ☐ Ideal: 10,000+ clips

---

## ☐ Phase 6: Train Model

### ☐ 6.1 Verify Dataset
```bash
# Count clips
VIDEO_COUNT=$(ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/videos/*.npy | wc -l)
LABEL_COUNT=$(ls /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed/labels/*.txt | wc -l)

echo "Videos: $VIDEO_COUNT"
echo "Labels: $LABEL_COUNT"

# Should be equal and ≥2000
```

### ☐ 6.2 Start Training
```bash
python train_swin_vallr.py \
  --data_dir /media/agam/Local\ Disk/the_code/roman/majorproject/youtube_dataset/processed \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0001
```

### ☐ 6.3 Monitor Training
- Check loss curves
- Validate on held-out set
- Save checkpoints

---

## ☐ Troubleshooting Checklist

### ☐ If "API quota exceeded"
- ☐ Wait 24 hours for reset
- ☐ Or request quota increase in Google Cloud Console
- ☐ Or split search across multiple days

### ☐ If "No trainable videos found"
- ☐ This is normal! Expected rate: 5-15%
- ☐ Increase `--max_per_query` to search more
- ☐ Try different search queries

### ☐ If "Download failed"
- ☐ Check internet connection
- ☐ Check logs: `youtube_dataset/logs/`
- ☐ Script automatically skips failed downloads

### ☐ If "MediaPipe error"
- ☐ Check if MediaPipe is installed: `pip install mediapipe`
- ☐ Check logs for details
- ☐ Script automatically skips problematic videos

### ☐ If "Low quality videos"
- ☐ This is normal! ~60-70% pass quality checks
- ☐ Script automatically filters low-quality videos
- ☐ Adjust thresholds if needed (see README.md)

---

## ☐ Optional: Advanced Setup

### ☐ Run in Background
```bash
nohup python youtube_scraper/build_youtube_dataset.py \
  --api_key $(cat ~/.youtube_api_key) \
  --output_dir youtube_dataset \
  --full_pipeline \
  --max_per_query 50 \
  --require_hd \
  --max_workers 4 \
  > youtube_scraper.log 2>&1 &

# Get process ID
echo $! > youtube_scraper.pid

# Check progress
tail -f youtube_scraper.log

# Stop if needed
kill $(cat youtube_scraper.pid)
```

### ☐ Schedule with Cron
```bash
# Edit crontab
crontab -e

# Add line to run daily at 2 AM
0 2 * * * cd /media/agam/Local\ Disk/the_code/roman/majorproject && python youtube_scraper/build_youtube_dataset.py --api_key $(cat ~/.youtube_api_key) --output_dir youtube_dataset --full_pipeline --max_per_query 50 --require_hd
```

### ☐ Monitor with Email Notifications
```bash
# Install mail utility
sudo apt install mailutils

# Modify script to send email on completion
python youtube_scraper/build_youtube_dataset.py ... && \
  echo "Dataset build complete!" | mail -s "YouTube Scraper Done" your@email.com
```

---

## ☐ Completion Checklist

### ☐ Setup Complete
- ☐ Dependencies installed
- ☐ API key obtained and saved
- ☐ Tests passed
- ☐ Quick test completed successfully

### ☐ Dataset Built
- ☐ Production dataset started
- ☐ ≥2,000 clips created
- ☐ Quality verified
- ☐ Metadata saved

### ☐ Training Started
- ☐ Model training initiated
- ☐ Checkpoints saving
- ☐ Validation running

### ☐ Documentation Read
- ☐ GET_STARTED_WITH_YOUTUBE_SCRAPER.md
- ☐ QUICKSTART.md
- ☐ README.md (at least troubleshooting section)

---

## 📊 Progress Tracker

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Installation | ☐ | | |
| API Key | ☐ | | |
| Testing | ☐ | | |
| Quick Test | ☐ | | Clips: ___ |
| Production | ☐ | | Clips: ___ |
| Training | ☐ | | Epoch: ___ |

---

## 🎯 Success Criteria

- ✅ All tests pass
- ✅ Quick test produces 50-200 clips
- ✅ Production dataset has ≥2,000 clips
- ✅ Quality score ≥0.75
- ✅ Training starts successfully

---

## 📞 Need Help?

- **Documentation:** `youtube_scraper/README.md`
- **Quick Start:** `youtube_scraper/QUICKSTART.md`
- **Examples:** `youtube_scraper/example_usage.py`
- **Logs:** `youtube_dataset/logs/`

---

**Print this checklist and check off items as you complete them! ✓**
