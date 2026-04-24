Handoff Document - Swin-VALLR (Linux Session)
==============================================

Purpose
-------
This document is a complete handoff for reopening this project on Linux in a fresh
Codex session with no memory. It explains what the project is, the goal, detailed
pipeline flow, how to run each file (including every CLI flag), key environment
variables, and recommended end-to-end commands for testing. It also includes a
summary of the chat history from the current Windows session.

Project Summary
---------------
Swin-VALLR is a visual-only lip-reading system. It takes video (no audio),
extracts mouth ROIs, and predicts phoneme sequences with a Swin Transformer-based
encoder. A compact LLM (Qwen2-0.5B) refines the phoneme sequence into readable text.
The demo app adds live captions, translation, and post-session summary using small
local transformer models.

Goal
----
Build a high-quality visual speech recognition pipeline that:
- Works without audio.
- Runs on AMD DirectML (Windows), NVIDIA CUDA (Windows/Linux), and AMD ROCm (Linux).
- Handles accents (English), multi-angle video, and speaker separation (by mouth motion).
- Provides a lightweight, local translation and summarization path.
- Logs and saves artifacts for research and debugging.
- Keeps training time reasonable while improving robustness.

Non-Goals
---------
- Audio input is not used.
- The system is not a multi-modal (audio+video) model. It is visual-only.

High-Level Pipeline Flow
------------------------
1) Setup:
   - Detect OS and GPU vendor.
   - Install backend-specific PyTorch (ROCm/CUDA/DirectML).
   - Install requirements and verify with env_test.py.

2) Data preparation:
   - Use data_pipeline.py interactive setup to prepare datasets (LRS2/LRS3/custom).
   - MediaPipe extracts mouth ROIs from video and stores .npy sequences.
   - Metadata is generated: id, video path, text, word_count, language.
   - Optional visual-only filtering and dataset cleaning for label quality.

3) Training:
   - train_swin_vallr.py loads preprocessed ROI sequences + labels.
   - Curriculum learning: single words -> short sentences -> full dataset.
   - CTC loss for phoneme prediction.
   - Optional LoRA fine-tuning for Qwen2 refiner.
   - Logs and artifacts saved for each run.

4) Inference / Demo:
   - app.py (Streamlit) captures webcam or file upload.
   - MediaPipe extracts ROIs (multi-speaker and profile-aware options).
   - Model predicts phonemes and refined text.
   - Optional live translation and final summary.
   - YouTube-style captions and transcript panel with speaker tags.

Detailed Data Flow (Model)
--------------------------
Input: (B, C, T, H, W) video frames (lip ROI)
1) Tubelet embedding (Conv3D)
2) Swin Transformer blocks (DirectML-safe ops)
3) Temporal adapter (1D conv, stride=1 to preserve CTC resolution)
4) Optional temporal multi-scale fusion
5) Optional temporal attention (Conformer-style)
6) Temporal feature normalization
7) Phoneme head (CTC log-softmax)
8) CTC decoding (beam search and alternatives)
9) Linguistic refiner (Qwen2-0.5B) to produce text

Demo App Flow (Streamlit)
-------------------------
1) Load model + MediaPipe processor (sidebar).
2) Capture webcam (WebRTC) or upload a video file.
3) Extract ROIs:
   - Single speaker: first detected face.
   - Multi-speaker: select most active mouth based on openness + motion.
   - Profile-aware crop expansion and optional mouth alignment rotation.
4) Run model inference.
5) Optional translation and summary.
6) Display phonemes, refined text, translation, captions, transcript history.

Key Files and Roles
-------------------
- app.py: Streamlit demo UI and inference loop.
- backend_manager.py: backend detection (CUDA/ROCm/DirectML/CPU) and safe ops.
- data_pipeline.py: dataset creation, MediaPipe ROI extraction, augmentations.
- dataset_cleaning.py: label cleaning and optional n-gram correction.
- env_test.py: backend validation and smoke test.
- logging_utils.py: structured logging (per-component logs).
- artifact_utils.py: save plots/images/json/text in structured folders.
- model_architecture.py: Swin-VALLR model definition + refiner.
- nlp_utils.py: translation and summary (local transformer models).
- refiner_finetune.py: LoRA fine-tuning for refiner.
- setup.py: OS/GPU-aware dependency setup with env detection.
- train_swin_vallr.py: full training loop.
- other research papers/: curated PDFs (currently 9 files: 1,2,3,4,7,8,10,11,12).

Notes:
- There are no paper download/rename scripts in this repo (no download_specific_papers.py,
  rename_papers.py, or filter_papers_v2.py).

Backend and OS Support
----------------------
- Linux + AMD GPU: ROCm. setup.py installs ROCm system packages (no venv).
- Windows + AMD GPU: DirectML in venv.
- Windows/Linux + NVIDIA GPU: CUDA in venv (Windows) or system python (Linux).
- CPU: fallback, slower.

Logging and Artifacts
---------------------
Logs:
- logs/<component>/<YYYY-MM-DD>/<timestamp>_<pid>.txt

Artifacts:
- artifacts/<component>/<YYYY-MM-DD>/<run_id>/
- artifacts/<component>/<YYYY-MM-DD>/<run_id>/useful_<name>/

Notable artifact categories:
- training:
  - useful_loss_curves/
  - useful_batch_previews/
  - useful_logit_matrices/
  - useful_sample_predictions/
- app:
  - useful_inference/<timestamp>/ (frames.png, rois.png, result.json, text.txt)
- setup:
  - useful_env_summary/
- data_pipeline:
  - useful_roi_previews/
  - useful_visual_filter/ (previews, summaries)
  - useful_dataset_summaries/

Environment Variables (Key)
---------------------------
App / Inference:
- SWIN_VALLR_REFINER_ADAPTER: optional LoRA adapter path.
- SWIN_VALLR_LANGUAGE_HINT: optional text language hint.
- SWIN_VALLR_REFINER_MODE: phoneme | roman | hybrid
- SWIN_VALLR_DUAL_HYP: enable dual hypothesis decoding (true/false)
- SWIN_VALLR_DUAL_HYP_SMOOTH_KERNEL: int
- SWIN_VALLR_DUAL_HYP_CHUNK_FRAMES: int
- SWIN_VALLR_UNCERTAINTY_PROMPT: true/false
- SWIN_VALLR_ALT_TOPK: int
- SWIN_VALLR_ALT_MAX: int
- SWIN_VALLR_ACCENT_AWARE: true/false
- SWIN_VALLR_ACCENT_MAX: int
- SWIN_VALLR_MULTI_SPEAKER: true/false
- SWIN_VALLR_MAX_FACES: int
- SWIN_VALLR_ALIGN_MOUTH: true/false
- SWIN_VALLR_PROFILE_AWARE: true/false
- SWIN_VALLR_PROFILE_EXPAND: float
- SWIN_VALLR_PROFILE_YAW_SCALE: float

NLP:
- SWIN_VALLR_NLP_DEVICE: cpu|cuda|rocm
- SWIN_VALLR_SUMMARY_MODEL: default google/flan-t5-small

Data pipeline:
- SWIN_VALLR_VISUAL_FILTER: true/false
- SWIN_VALLR_VISUAL_FILTER_FRAMES: int
- SWIN_VALLR_VISUAL_FILTER_STRIDE: int
- SWIN_VALLR_VISUAL_FILTER_MIN_FACE_RATE: float
- SWIN_VALLR_VISUAL_FILTER_MIN_LIP_RATIO: float
- SWIN_VALLR_VISUAL_FILTER_MIN_OPENNESS: float
- SWIN_VALLR_VISUAL_FILTER_MIN_MOTION: float
- SWIN_VALLR_VISUAL_FILTER_MAX_MOTION: float
- SWIN_VALLR_VISUAL_FILTER_MIN_SHARPNESS: float
- SWIN_VALLR_VISUAL_FILTER_PREVIEW_LIMIT: int
- SWIN_VALLR_VISUAL_FILTER_PREVIEW_FRAMES: int
- SWIN_VALLR_VISUAL_FILTER_PREVIEW_SIZE: int
- SWIN_VALLR_LANGUAGE_BALANCE: true/false
- SWIN_VALLR_LANGUAGE_BALANCE_POWER: float

Setup:
- ROCM_VERSION: ROCm version for apt install (default 5.6).
- CUDA_INDEX_URL: override CUDA PyTorch wheel index URL.

How to Run Each File (All CLI flags)
------------------------------------
1) setup.py
   Command:
   - python setup.py
   Flags: none
   Notes: Detects OS/GPU, installs backend packages, runs env_test.py, saves summary.

2) env_test.py
   Command:
   - python env_test.py
   Flags: none
   Notes: Validates backend and prints ENV_TEST_RESULT JSON line.

3) backend_manager.py
   Command:
   - python backend_manager.py
   Flags: none
   Notes: Prints backend info and runs safe_roll tests.

4) data_pipeline.py
   Command:
   - python data_pipeline.py
   Flags: none (interactive)
   Notes: Prompts for dataset selection and creates preprocessed data.

5) dataset_cleaning.py
   Command:
   - python dataset_cleaning.py --labels-dir PATH [options]
   Required:
   - --labels-dir: path to folder of .txt labels
   Output behavior (choose one):
   - --output-dir: write cleaned labels to a new directory
   - --in-place: overwrite labels
   - --dry-run: show changes only
   Language options:
   - --language: auto|arabic|chinese|english|french|greek|spanish
   - --keep-case: preserve case (default lowercases)
   Text normalization:
   - --strip-bracketed
   - --strip-speaker-tags
   - --remove-urls
   - --strip-diacritics
   N-gram correction:
   - --ngram-model: load existing model
   - --save-ngram-model: save new model
   - --corpus: corpus files/dirs (repeatable)
   - --use-labels-corpus
   - --ngram-order: 1|2|3
   - --smoothing: float
   - --max-edit-distance: int
   - --max-candidates: int
   - --distance-penalty: float
   - --min-score-gain: float
   - --replace-known
   - --use-phonetic
   Quality filtering:
   - --apply-quality-filter
   - --drop-duplicates
   - --save-quality-csv
   - --min-tokens, --max-tokens
   - --min-chars, --max-chars
   - --max-token-length
   - --min-alpha-ratio
   - --max-digit-ratio
   - --max-non-alnum-ratio
   - --max-repeat-char
   - --min-unique-token-ratio
   - --min-ngram-log-prob
   - --quality-samples

6) train_swin_vallr.py
   Command:
   - python train_swin_vallr.py --data_dir PATH [options]
   Required:
   - --data_dir: path to preprocessed dataset
   Training options:
   - --batch_size: int
   - --epochs: int
   - --lr: float
   - --num_workers: int
   - --load_refiner: load Qwen2 refiner for joint training
   - --checkpoint_dir: path for checkpoints
   - --resume: checkpoint path to resume
   Loss weights:
   - --decorrelation_weight: float
   - --decorrelation_anchor: mean|first
   - --temporal_smoothness_weight: float
   - --attention_entropy_weight: float
   Temporal attention settings:
   - --temporal_attention_layers: int
   - --temporal_attention_heads: int
   - --temporal_attention_kernel: int
   - --temporal_attention_dropout: float
   - --temporal_attention_streaming: disable MHSA
   - --disable_temporal_multiscale: turn off multiscale fusion

7) refiner_finetune.py
   Command:
   - python refiner_finetune.py --output_dir PATH [options]
   Required:
   - --output_dir: where to save the LoRA adapter
   Inputs:
   - --corpus: text file/dir (repeatable)
   - --labels_dir: labels directory (optional)
   Mode:
   - --mode: phoneme|roman
   - --language_hint: label for roman mode
   Base model:
   - --model_name: default Qwen/Qwen2-0.5B-Instruct
   Training params:
   - --max_samples
   - --min_words
   - --max_length
   - --batch_size
   - --epochs
   - --lr
   - --warmup_steps
   - --grad_accum
   - --seed
   - --lora_r
   - --lora_alpha
   - --lora_dropout
   - --fp16 (CUDA/ROCm only)

8) app.py
   Command:
   - streamlit run app.py
   Notes:
   - Use sidebar to load model and configure vision/decoding options.
   - File upload fallback if streamlit-webrtc is unavailable.

9) model_architecture.py
   Command:
   - python model_architecture.py
   Flags: none
   Notes: Runs self-test with dummy input.

10) nlp_utils.py
   No CLI. Imported by app.py for translation and summary.

Recommended End-to-End Linux Commands (Copy/Paste)
--------------------------------------------------
1) Setup and backend verification (Linux, AMD ROCm example):
   python setup.py
   python env_test.py

2) Prepare dataset (interactive):
   python data_pipeline.py

3) Optional label cleanup (example):
   python dataset_cleaning.py \
     --labels-dir ./data/labels \
     --output-dir ./data/labels_clean \
     --language english \
     --strip-bracketed --strip-speaker-tags --remove-urls --strip-diacritics \
     --apply-quality-filter --drop-duplicates --save-quality-csv

4) Train (example safe defaults for a test run):
   python train_swin_vallr.py \
     --data_dir ./data \
     --batch_size 8 \
     --epochs 1 \
     --lr 1e-4 \
     --num_workers 2 \
     --temporal_attention_layers 1 \
     --temporal_attention_heads 8 \
     --temporal_attention_kernel 3 \
     --temporal_attention_dropout 0.1

5) Run demo:
   streamlit run app.py

6) Optional refiner fine-tune (example):
   python refiner_finetune.py \
     --mode phoneme \
     --corpus ./data/labels \
     --output_dir ./adapters/refiner_lora \
     --epochs 1 \
     --batch_size 4 \
     --max_length 256

End-of-Project Final Commands (Full Pipeline)
---------------------------------------------
Use these as final copy/paste commands on Linux once data is ready.

Setup:
python setup.py
python env_test.py

Dataset prep:
python data_pipeline.py

Training (full run example):
python train_swin_vallr.py \
  --data_dir ./data \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --num_workers 4 \
  --load_refiner \
  --temporal_attention_layers 1 \
  --temporal_attention_heads 16 \
  --temporal_attention_kernel 3 \
  --temporal_attention_dropout 0.1

Demo (with optional env tuning):
export SWIN_VALLR_MULTI_SPEAKER=true
export SWIN_VALLR_MAX_FACES=2
export SWIN_VALLR_ALIGN_MOUTH=true
export SWIN_VALLR_PROFILE_AWARE=true
export SWIN_VALLR_PROFILE_EXPAND=1.4
export SWIN_VALLR_PROFILE_YAW_SCALE=2.5
export SWIN_VALLR_ACCENT_AWARE=true
export SWIN_VALLR_REFINER_MODE=hybrid
streamlit run app.py

Known Edge Cases and Fixes Applied
----------------------------------
- DirectML checkpoint loading now maps to CPU to avoid torch.load device errors.
- safe_roll handles zero-length tensors.
- data_pipeline handles invalid metadata.json, missing video/labels, empty labels.
- dataset __getitem__ handles corrupt/mis-shaped ROI arrays.
- CTC loss handles empty phoneme targets.
- setup.py handles PowerShell failures and permission errors while scanning venvs.

Chat History Summary (Current Windows Session)
----------------------------------------------
This is a high-level chronological summary of what happened in this session:
1) User requested a full project understanding and file rewrites for clarity.
2) Multiple modules were updated to be more readable and to add features:
   - app.py: captions, translation, summary, multi-speaker, multi-angle UI controls.
   - data_pipeline.py: profile-aware ROI, alignment, augmentations (beard/black bar).
   - model_architecture.py: hybrid refiner, accent-aware decoding.
   - nlp_utils.py: translation and summary support.
3) Setup enhancements were added to detect OS/GPU and install ROCm/DirectML/CUDA.
4) Added logging and artifact saving (images, plots, matrices, json/text).
5) Dataset cleaning script was added for future dataset builds.
6) Multiple "research papers" were discussed; audio-related ideas were rejected.
7) Research papers were reviewed and a curated folder was maintained (non-contiguous numbering).
8) Edge-case hardening pass was done across app, data, training, setup.
9) The project was confirmed as visual-only (not multi-modal).

Important Notes for Linux Session
---------------------------------
- The setup script uses system Python on Linux (no venv).
- ROCm install uses sudo apt on Ubuntu/Debian; other distros require manual install.
- The demo uses no audio, only video.
- MediaPipe is optional; without it, center-crop fallback is used.
- If files are moved across OSs, recheck UI strings for encoding regressions.

If You Need Help Later
----------------------
Ask the new Linux session to:
1) Read this HANDOFF.md.
2) Run the "Recommended End-to-End Linux Commands."
3) Report any failures with logs from logs/<component>.
