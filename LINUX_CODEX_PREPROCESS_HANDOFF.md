# Linux Codex Preprocessing Handoff

Last updated: 2026-05-04

This handoff is for a fresh Linux Codex session that does not have the chat
history. Ask Linux Codex to read this file first:

```text
LINUX_CODEX_PREPROCESS_HANDOFF.md
```

## Current Goal

The immediate goal is to preprocess the final YouTube/video dataset into a
training-ready lip-reading dataset quickly enough to fit the user's time budget.

The user originally tried a full uncapped run over about 614 source videos. That
run was technically working, but it was far too slow. It was estimated to take
many days because long videos can create thousands of two-second clip windows.

The practical goal now is:

- Use Linux ROCm GPU where possible.
- Keep preprocessing to a few hours, not days.
- Skip untrainable/no-face/no-active-speaker clips.
- Save word-level transcripts/captions.
- Produce `videos/*.npy`, `labels/*.txt`, and `metadata.json` for training.
- Prefer a capped fast dataset, sampled evenly across each source video.

## Important User Preferences

- Do not use `gemma_local` unless the user explicitly asks. In this thread the
  user explicitly said not to use it earlier.
- The dataset should remain English/CC/US-accent focused as much as the existing
  scraper/filtering flow already made it.
- The user has frequent power cuts. Commands must be resumable.
- The user wants commands they can copy and paste.
- The user is comfortable stopping a long run with `Ctrl+C` if the work is
  safely resumable.

## Repository Path

Windows path:

```text
D:\the_code\roman\majorproject
```

WSL/Linux path used in this Codex session:

```text
/mnt/d/the_code/roman/majorproject
```

If booted into native Linux, the NTFS drive may be mounted somewhere like:

```text
/media/agam/Local Disk/the_code/roman/majorproject
```

Linux Codex should first run:

```bash
pwd
ls
```

and confirm it is in the repo root containing `data_pipeline.py`,
`preprocess_final_trainable_dataset.py`, and `final_trainable_videos_heavy/`.

## Dataset Paths

Raw final source videos:

```text
final_trainable_videos_heavy/
```

Current known stats:

- About 614 media files.
- About 55 GB.
- Contains the manually verified videos imported from:

```text
D:\the_code\roman\majorproject\drive download\verified
```

Old/full uncapped preprocessing output:

```text
final_preprocessed_dataset/
```

Known state at handoff time:

- `videos/*.npy`: 8383 files
- `labels/*.txt`: 8383 files
- `metadata.json`: 7291 committed samples
- `preprocessing_state.json`: 7 completed source videos, 0 failed
- size: about 11 GB

Important: the uncapped run was interrupted while processing a source video, so
there may be some `.npy`/label files not yet committed to metadata. This folder
is useful as a cache/reference but should not be treated as the clean final fast
dataset unless Linux Codex audits it.

Fast capped preprocessing output:

```text
final_preprocessed_dataset_fast/
```

Known state at handoff time:

- `videos/*.npy`: 912 files
- `labels/*.txt`: 912 files
- `metadata.json`: 869 committed samples
- `preprocessing_state.json`: 24 completed source videos, 0 failed

This fast folder is the better continuation target if the user wants results in
hours. It was created after the time-budget issue was discovered.

## Key Files Changed/Added In This Chat

### `preprocess_final_trainable_dataset.py`

New main preprocessing script. It was added during this chat.

Purpose:

- Input: raw videos from `final_trainable_videos_heavy/`.
- Output: training-ready dataset folder.
- Generates/reuses Whisper word-level transcripts.
- Extracts lip ROI clips with MediaPipe.
- Tracks multiple faces and chooses likely active speaker based on mouth
  openness and mouth-motion scoring.
- Skips frames/clips without usable faces instead of padding blank/no-face
  frames.
- Saves:
  - `videos/<sample_id>.npy`
  - `labels/<sample_id>.txt`
  - `metadata.json`
  - `transcripts/<video_id>.json`
  - `transcripts/<video_id>.txt`
  - `transcripts/<video_id>_words.txt`
  - `transcripts/<video_id>.srt`
  - `visual_scans/<video_id>.json`
  - `preprocess_final_dataset.log`
  - `preprocess_events.jsonl`
  - `preprocessing_state.json`

Important CLI options:

```text
--input                         source video folder
--output                        output dataset folder
--whisper-model                 Whisper model, default base
--device auto|cpu|cuda          PyTorch device; ROCm uses cuda
--clip-seconds                  default 2.0
--stride-seconds                default 1.0
--num-frames                    default 50
--target-fps                    default 25
--roi-size                      default 96
--max-faces                     default 4
--candidate-multiplier          candidate face frames per output frame
--min-face-frame-rate           default 0.90
--visual-scan-fps               cheap visual pre-scan FPS
--visual-scan-min-step          minimum seconds between visual scan samples
--visual-scan-max-samples       cap visual pre-scan samples
--max-clips-per-video           crucial fast-mode cap; 0 means uncapped
--retranscribe                  redo Whisper transcripts
--reprocess                     overwrite existing clips
--rescan-visual                 redo visual face-region scan
--no-progress                   disable tqdm bars
```

Critical change: `--max-clips-per-video` now samples windows evenly across the
whole source video instead of just taking the first N windows. This makes capped
fast preprocessing much more useful.

### `requirements.txt`

Updated in this chat:

- `mediapipe==0.10.11` was relaxed because Windows Python 3.13 had no matching
  wheel:

```text
mediapipe>=0.10.30,<0.11
```

- `numpy>=1.24.0,<2.0` was relaxed to:

```text
numpy>=1.24.0
```

- Added:

```text
openai-whisper>=20231117
imageio-ffmpeg>=0.6.0
```

`imageio-ffmpeg` is used as a Windows fallback because Whisper requires ffmpeg
to read audio.

### `.gitignore`

Updated to ignore generated preprocessing output:

```text
final_preprocessed_dataset/
final_training_dataset/
```

Also contains scraper-related ignore entries from previous work:

```text
playlist_final_downloader.log
_playlist_download_archive.txt
playlist_download_archive.txt
youtube_cookies*.txt
cookies*.txt
```

### `download_playlist_to_final.py`

Added earlier in the chat for downloading a specific verified YouTube playlist
into the final training folder. It has bot-block handling and archive support.

Important context:

- User had tried downloading 2000 videos but only got about 378 usable.
- We added a playlist download for:

```text
https://www.youtube.com/watch?v=tF4ytvvIUJc&list=PLrLY2VO2cgPV9T2IpKBdUEp8aVOs0OkDx&index=1
```

- User hit YouTube bot checks after about 60 playlist files.
- Script was updated to detect bot-block messages and cooldown/retry.
- Cookies from Chrome/Edge failed on Windows because of DPAPI/cookie DB issues.
- User manually verified/downloaded extra files and those were copied into
  `final_trainable_videos_heavy/`.

The scraper/downloader is not the immediate task now. The immediate task is
preprocessing.

## Current Preprocessing Behavior

The script works in phases per source video:

1. Load or create visual scan:
   - Saves face-present intervals in `visual_scans/`.
   - Used to avoid spending time on transcript windows with no face.

2. Load or create Whisper transcript:
   - Saves word-level JSON/TXT/SRT sidecars in `transcripts/`.
   - Reused by default on rerun.

3. Build word windows:
   - Default two-second windows, one-second stride.
   - Label is the words inside that window.

4. Filter windows by visual face intervals:
   - Skips areas with no face-present region.

5. Apply optional cap:
   - `--max-clips-per-video 50` means at most 50 windows per source video,
     evenly sampled from that source video.

6. Extract lip ROI clips:
   - Uses MediaPipe multi-face observations.
   - Chooses active speaker via mouth openness/motion score.
   - Saves only fixed `(50, 96, 96, 3)` uint8 arrays.
   - Skips bad clips instead of padding no-face frames.

7. Save metadata/state/events:
   - `metadata.json` is flushed after each completed source video.
   - `preprocessing_state.json` tracks completed/failed videos.
   - `preprocess_events.jsonl` has structured event logs.

## Why The Uncapped Run Was Too Slow

The user ran:

```powershell
py preprocess_final_trainable_dataset.py --input final_trainable_videos_heavy --output final_preprocessed_dataset --whisper-model base
```

No cap means long videos produce thousands of candidate clip windows. Actual
completed timings from the uncapped run:

```text
-2TKRPIQJyI:    1720 windows, 52 min
-5rWzXKyiZg:    2865 windows, 95 min
-B2aj51Fa7g:    1573 windows, 47 min
-hHPx1hPQBA:    1910 windows, 53 min
-pj01zova0c:     407 windows, 10 min
-sfiReUu3o0:     268 windows, 8 min
-ttDfEguRpw:     478 windows, 27 min
```

At that rate, all 614 videos could take many days or weeks. The earlier
54-minute ETA came from the small capped test and was not representative.

## Recommended Fast Command

If the user wants a dataset in a few hours, run capped fast mode. Recommended
starting point:

```bash
python3 preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cuda \
  --max-clips-per-video 50
```

If `torch.cuda.is_available()` is false on Linux, use:

```bash
python3 preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cpu \
  --max-clips-per-video 50
```

If the user has overnight time and wants more data:

```bash
python3 preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cuda \
  --max-clips-per-video 100
```

Important: do not use the uncapped command unless the user explicitly accepts a
multi-day run.

## Resume Behavior

The script is resumable.

If power cuts happen, rerun the exact same command. It will:

- Reuse completed transcripts.
- Reuse visual scans.
- Reuse existing clips.
- Continue adding new completed videos.

Do not pass `--reprocess` unless the user wants to overwrite existing `.npy`
clips.

Do not pass `--retranscribe` unless the user wants to redo Whisper transcripts.

Do not pass `--rescan-visual` unless the user wants to redo visual scans.

## Linux ROCm / GPU Notes

ROCm helps mainly with Whisper transcription. MediaPipe face/lip extraction is
still mostly CPU-bound. OpenCV reading and `.npy` writing are disk/CPU-bound.

PyTorch ROCm exposes AMD GPUs through the `cuda` API name, so the script should
use:

```text
--device cuda
```

Linux Codex should first check:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("hip:", getattr(torch.version, "hip", None))
PY
```

Expected for ROCm:

```text
cuda available: True
hip: non-empty version string
```

If false, Whisper will run on CPU. Linux Codex should not blindly reinstall
PyTorch if the user already has ROCm working. If PyTorch ROCm is missing, use the
official PyTorch selector for the installed ROCm version, then re-run the check.

Whisper model choice:

- `base`: current default; good balance.
- `tiny`: faster but worse captions.
- `small`: better captions but slower.

For speed, keep `base` unless the user explicitly asks for better captions.

## Setup Commands For Native Linux

Recommended:

```bash
cd /path/to/majorproject
python3 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

If ROCm PyTorch is already installed globally or in another venv, do not replace
it without asking. Otherwise install ROCm PyTorch appropriate to the machine, then:

```bash
python -m pip install -r requirements.txt
```

Verify dependencies:

```bash
python - <<'PY'
import cv2, mediapipe, whisper, imageio_ffmpeg, torch
print("cv2 ok")
print("mediapipe", mediapipe.__version__)
print("whisper ok")
print("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("hip", getattr(torch.version, "hip", None))
PY
```

Then run a tiny smoke test:

```bash
python preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cuda \
  --limit 2 \
  --max-clips-per-video 20
```

If that works, run the full fast command with no `--limit`:

```bash
python preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cuda \
  --max-clips-per-video 50
```

## Useful Monitoring Commands

Dataset counts:

```bash
find final_preprocessed_dataset_fast/videos -name '*.npy' | wc -l
find final_preprocessed_dataset_fast/labels -name '*.txt' | wc -l
python - <<'PY'
import json, pathlib
root = pathlib.Path("final_preprocessed_dataset_fast")
meta = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else []
print("metadata samples:", len(meta))
print("source videos:", len(set(item.get("source_video_id") for item in meta)))
PY
```

Tail logs:

```bash
tail -f final_preprocessed_dataset_fast/preprocess_final_dataset.log
```

Structured events:

```bash
tail -n 40 final_preprocessed_dataset_fast/preprocess_events.jsonl
```

State summary:

```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("final_preprocessed_dataset_fast")
state = json.loads((root / "preprocessing_state.json").read_text())
processed = state.get("processed_videos", {})
failed = state.get("failed_videos", {})
print("processed:", len(processed))
print("failed:", len(failed))
for video_id, info in list(processed.items())[-10:]:
    print(video_id, "created=", info.get("created"), "existing=", info.get("existing"), "skipped=", info.get("skipped"), "sec=", info.get("elapsed_seconds"))
PY
```

GPU monitor:

```bash
watch -n 1 rocm-smi
```

## Interpreting Logs

Harmless MediaPipe noise:

```text
portable_clearcut_uploader.cc:90 Failed to send to clearcut
```

This is MediaPipe telemetry upload failing. It does not mean preprocessing
failed. It does pollute tqdm output. It can be ignored.

Healthy lines:

```text
Using cached transcript for VIDEO.mp4
Visual scan result for VIDEO.mp4: intervals=...
Candidate windows for VIDEO.mp4: kept after visual prefilter; processing=...
Finished VIDEO.mp4: created=... existing=... skipped=...
```

Skip reasons:

```text
no_face_detected
not_enough_face_frames
not_enough_selected_speaker_frames
roi_crop_failed
```

These are clip-level skips, not necessarily full-video failures. They are
expected. The script is intentionally rejecting bad training samples.

## Known Issue: Metadata vs NPY Count During Interrupted Runs

`metadata.json` is saved after a source video completes. If the user presses
`Ctrl+C` during a source video, some `.npy` and label files from the in-progress
video may exist without metadata entries.

That is why at one point:

```text
final_preprocessed_dataset/videos/*.npy: 8383
metadata.json samples: 7291
```

For clean training, prefer a fresh or completed output folder:

```text
final_preprocessed_dataset_fast/
```

If Linux Codex needs to clean orphan files later, do it carefully by comparing
metadata sample IDs with files on disk. Do not delete user data blindly.

## Training Data Contract

The trainer expects:

```text
<data_dir>/videos/*.npy
<data_dir>/labels/*.txt
<data_dir>/metadata.json
```

`data_pipeline.py` `LipReadingDataset` loads `metadata.json`, then each sample's
relative `video` path and `text`.

Preprocessor output samples are shaped:

```text
(50, 96, 96, 3)
```

Labels are plain text files and are also included in `metadata.json` as `text`.

## Training Command After Preprocessing

Once `final_preprocessed_dataset_fast/metadata.json` has enough samples, likely
train with:

```bash
python train_swin_vallr.py --data_dir final_preprocessed_dataset_fast
```

Before running a long train, Linux Codex should inspect `train_swin_vallr.py
--help` because exact training flags may matter for backend, batch size, epochs,
checkpoint path, and GPU memory.

## Prior Chat History Summary

Major events:

1. User asked to inspect the whole project, logs, and every file.
2. User explicitly said not to use `gemma_local`.
3. User asked to delete junk one-time `.md`/`.py` files. Some cleanup happened
   earlier, but verify with git status before assuming.
4. User asked to fix `.gitignore`. It was updated for generated data/logs.
5. User asked to make the YouTube scraper faster and Windows-compatible while
   avoiding YouTube blocking/rate-limits and keeping CC/American-English focus.
6. User wanted a resumable 2000-video command with logging. Downloader/archive
   logic was improved.
7. User asked ETAs and where videos were saved. The final source videos are in
   `final_trainable_videos_heavy/`.
8. User discovered only about 378 usable videos from an original 2000 download.
9. User asked to download a known CC/free playlist into the final folder.
10. YouTube bot checks occurred after about 60 playlist files. Script was updated
    for block detection/cooldown and cookie support, but browser cookie extraction
    failed on Windows.
11. User manually verified/downloaded additional videos into:
    `drive download/verified`.
12. Those were imported into `final_trainable_videos_heavy/`, resulting in about
    614 media files and about 55 GB.
13. User asked if project has preprocessing scripts. Existing old scripts were
    identified but were not adequate for the new dataset.
14. New `preprocess_final_trainable_dataset.py` was created.
15. Windows dependency fixes were made:
    - OpenCV installed.
    - MediaPipe pin relaxed.
    - `imageio-ffmpeg` added because Whisper could not find ffmpeg.
16. Initial capped test over 5 videos succeeded and produced 107 samples.
17. Full uncapped run was attempted and proved too slow.
18. Fast capped mode was recommended and `--max-clips-per-video` behavior was
    changed to sample evenly across videos.

## Immediate Recommendation For Linux Codex

Do not continue the uncapped folder unless the user explicitly asks.

Use:

```bash
python preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cuda \
  --max-clips-per-video 50
```

If the user wants more data and has overnight time:

```bash
python preprocess_final_trainable_dataset.py \
  --input final_trainable_videos_heavy \
  --output final_preprocessed_dataset_fast \
  --whisper-model base \
  --device cuda \
  --max-clips-per-video 100
```

If ROCm/GPU is not detected, explain that Whisper will be CPU-bound, but
MediaPipe extraction is CPU-bound either way.

