# YouTube Dataset Scraper - Pipeline Flow

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUTUBE DATASET SCRAPER PIPELINE                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: VIDEO DISCOVERY                                            │
│ (youtube_video_finder.py)                                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  YouTube Search API     │
                    │  - 16 default queries   │
                    │  - Custom queries       │
                    │  - HD filter            │
                    │  - Caption filter       │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Candidate Videos       │
                    │  ~800-1000 videos       │
                    │  (50 per query)         │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Enrich with Details    │
                    │  - Duration             │
                    │  - View count           │
                    │  - HD/SD                │
                    │  - Captions available   │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Save Candidates        │
                    │  video_ids.txt          │
                    └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TRAINABILITY CHECK                                         │
│ (youtube_trainability_checker.py)                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  For each video ID:     │
                    │  Check Trainability API │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Parse Response         │
                    │  - permitted: "all"  ✓  │
                    │  - permitted: "none" ✗  │
                    │  - permitted: [list] ?  │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Filter Trainable       │
                    │  ~5-15% pass            │
                    │  (50-150 from 1000)     │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Save Trainable IDs     │
                    │  trainable_ids.txt      │
                    │  + cache results        │
                    └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: DATASET BUILDING                                           │
│ (youtube_dataset_builder.py)                                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────┐
        │  For each trainable video (parallel):       │
        └─────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │  Download Video   │       │  Download Captions│
        │  (yt-dlp)         │       │  (yt-dlp)         │
        │  - 720p max       │       │  - VTT format     │
        │  - MP4 format     │       │  - English        │
        └───────────────────┘       └───────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │  Quality Check          │
                    │  (MediaPipe + OpenCV)   │
                    └─────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │  Visual Metrics   │       │  Thresholds       │
        │  - Face rate      │       │  - ≥70% faces     │
        │  - Lip size       │       │  - ≥3% lip size   │
        │  - Mouth movement │       │  - ≥0.004 motion  │
        │  - Sharpness      │       │  - ≥10.0 sharp    │
        └───────────────────┘       └───────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │  Pass Quality? (60-70%) │
                    └─────────────────────────┘
                            │           │
                        PASS│           │FAIL
                            ▼           ▼
                ┌───────────────┐   ┌──────────┐
                │  Continue     │   │  Skip    │
                └───────────────┘   └──────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │  Parse Captions           │
                │  - Extract timestamps     │
                │  - Clean text             │
                │  - Remove formatting      │
                └───────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │  Segment into 2s Clips    │
                │  - Every 2 seconds        │
                │  - 50 frames @ 25fps      │
                │  - Align with captions    │
                └───────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │  For each clip:           │
                └───────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌───────────────────┐   ┌───────────────────┐
    │  Extract Frames   │   │  Find Caption     │
    │  - 50 frames      │   │  - Match time     │
    │  - Resample 25fps │   │  - Get text       │
    └───────────────────┘   └───────────────────┘
                │                       │
                ▼                       │
    ┌───────────────────┐               │
    │  MediaPipe ROI    │               │
    │  - Detect face    │               │
    │  - Find lips      │               │
    │  - Stabilize      │               │
    │  - Align mouth    │               │
    │  - Crop 96x96     │               │
    └───────────────────┘               │
                │                       │
                ▼                       ▼
    ┌───────────────────┐   ┌───────────────────┐
    │  Save Video ROI   │   │  Save Transcript  │
    │  .npy (50,96,96,3)│   │  .txt (plain text)│
    └───────────────────┘   └───────────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                ┌───────────────────────────┐
                │  Clip Complete!           │
                │  video_id_clip_0000.npy   │
                │  video_id_clip_0000.txt   │
                └───────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT                                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Dataset Structure      │
                    │                         │
                    │  processed/             │
                    │    videos/              │
                    │      *.npy (ROIs)       │
                    │    labels/              │
                    │      *.txt (text)       │
                    │                         │
                    │  metadata/              │
                    │    summary.json         │
                    │    results.json         │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Ready for Training!    │
                    │  train_swin_vallr.py    │
                    └─────────────────────────┘
```

## Data Flow Summary

### Input
- YouTube API key
- Search queries (optional)
- Quality thresholds (optional)

### Processing
1. **Search:** 800-1000 candidate videos
2. **Filter:** 50-150 trainable videos (5-15%)
3. **Quality:** 30-100 passing videos (60-70%)
4. **Extract:** 50-150 clips per video
5. **Output:** 1,500-15,000 training clips

### Output
- Video ROIs: `.npy` files (50, 96, 96, 3)
- Transcripts: `.txt` files (plain text)
- Metadata: Processing statistics
- Logs: Detailed execution logs

## Parallel Processing

```
┌─────────────────────────────────────────────────────────────────────┐
│ PARALLEL EXECUTION (--max_workers 4)                                │
└─────────────────────────────────────────────────────────────────────┘

Worker 1: [Video 1] ──► [Download] ──► [Quality] ──► [Extract] ──► [Save]
Worker 2: [Video 2] ──► [Download] ──► [Quality] ──► [Extract] ──► [Save]
Worker 3: [Video 3] ──► [Download] ──► [Quality] ──► [Extract] ──► [Save]
Worker 4: [Video 4] ──► [Download] ──► [Quality] ──► [Extract] ──► [Save]
    │         │             │             │             │             │
    └─────────┴─────────────┴─────────────┴─────────────┴─────────────┘
                                  │
                                  ▼
                        [Progress Bar: 4/100]
                                  │
                                  ▼
                        [Save Every 100 Videos]
```

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ ERROR HANDLING                                                       │
└─────────────────────────────────────────────────────────────────────┘

                    [Process Video]
                          │
                          ▼
                    ┌─────────┐
                    │ Error?  │
                    └─────────┘
                      │     │
                  YES │     │ NO
                      ▼     ▼
            ┌──────────┐  ┌──────────┐
            │  Log     │  │ Continue │
            │  Skip    │  └──────────┘
            │  Continue│
            └──────────┘
                  │
                  ▼
        [Next Video in Queue]
                  │
                  ▼
        [Final Summary Report]
            - Successful: X
            - Failed: Y
            - Error breakdown
```

## Quality Filter Decision Tree

```
                    [Video Downloaded]
                          │
                          ▼
                ┌──────────────────┐
                │ Face Rate ≥70%?  │
                └──────────────────┘
                    │         │
                 YES│         │NO → REJECT
                    ▼         ▼
        ┌──────────────────┐
        │ Lip Size ≥3%?    │
        └──────────────────┘
            │         │
         YES│         │NO → REJECT
            ▼         ▼
┌──────────────────────┐
│ Mouth Movement OK?   │
└──────────────────────┘
    │         │
 YES│         │NO → REJECT
    ▼         ▼
┌──────────────────────┐
│ Sharpness ≥10?       │
└──────────────────────┘
    │         │
 YES│         │NO → REJECT
    ▼         ▼
[ACCEPT]   [REJECT]
    │
    ▼
[Extract Clips]
```

## Time Breakdown (Per Video)

```
Total: ~4 minutes per video

┌────────────────────────────────────────┐
│ Download Video        │ 30s  │ ████   │
│ Download Captions     │ 5s   │ █      │
│ Quality Check         │ 10s  │ ██     │
│ Parse Captions        │ 5s   │ █      │
│ Extract ROIs          │ 120s │ ████████│
│ Save Files            │ 10s  │ ██     │
└────────────────────────────────────────┘
```

## Storage Breakdown (Per 1000 Clips)

```
Total: ~2-4 GB per 1000 clips

┌────────────────────────────────────────┐
│ Video ROIs (.npy)     │ 2-3 GB │ ████████│
│ Transcripts (.txt)    │ 1 MB   │         │
│ Metadata (.json)      │ 1 MB   │         │
│ Raw Videos (temp)     │ 5-10GB │ (deleted)│
└────────────────────────────────────────┘
```

---

**This pipeline is fully automated and production-ready!**

Run with:
```bash
python youtube_scraper/build_youtube_dataset.py \
  --api_key YOUR_KEY \
  --output_dir /path/to/dataset \
  --full_pipeline
```
