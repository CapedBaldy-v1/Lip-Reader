# Swin-VALLR

Swin-VALLR is a visual-only lip reading system. It takes video (no audio),
extracts mouth regions, predicts phoneme sequences with a Swin Transformer-based
encoder, and refines them into text using a compact LLM (Qwen2-0.5B). The demo
adds live captions, translation, and a post-session summary.

This project is not multimodal. It is video-only.

## What This Project Does

- Visual speech recognition from video-only input.
- Robust ROI extraction (multi-angle, profile-aware, mouth alignment).
- Optional multi-speaker selection based on mouth motion.
- Optional live translation and summary using small local models.
- Runs on CUDA (NVIDIA), ROCm (AMD Linux), DirectML (AMD Windows), or CPU.

## Features

- Swin-Tiny visual encoder with DirectML-safe ops.
- Temporal adapter (stride 1 for CTC resolution).
- CTC decoding with uncertainty alternatives and accent-aware variants.
- Qwen2-based refiner (phoneme, roman, or hybrid prompt modes).
- Streamlit demo with YouTube-style captions and transcript history.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Optional GPU: CUDA (NVIDIA) or ROCm (AMD Linux) or DirectML (AMD Windows)
- MediaPipe (for ROI extraction)

## Installation

Recommended (auto-detect backend):

```bash
python setup.py
```

Manual (if you want to control it yourself):

```bash
pip install -r requirements.txt

# AMD Windows (DirectML)
pip install torch-directml

# NVIDIA CUDA or AMD ROCm
# Install PyTorch from the official index for your backend.
```

## Quick Start (Demo)

```bash
streamlit run app.py
```

Open the UI in your browser, load a checkpoint from the sidebar, and run
inference from webcam or uploaded video.

## Data Preparation

Interactive dataset setup:

```bash
python data_pipeline.py
```

This prepares:

```
data/
  videos/     # .npy ROI sequences
  labels/     # .txt transcripts
  metadata.json
```

## Training

Basic training:

```bash
python train_swin_vallr.py --data_dir path/to/data
```

Common options:

- `--batch_size` (int)
- `--epochs` (int)
- `--lr` (float)
- `--load_refiner` (fine-tune Qwen2 refiner with LoRA)
- `--resume` (checkpoint path)

## Refiner Fine-Tuning (Optional)

```bash
python refiner_finetune.py \
  --mode phoneme \
  --corpus ./data/labels \
  --output_dir ./adapters/refiner_lora
```

## Configuration (Environment Variables)

Inference / Demo:
- `SWIN_VALLR_REFINER_MODE=phoneme|roman|hybrid`
- `SWIN_VALLR_ACCENT_AWARE=true|false`
- `SWIN_VALLR_MULTI_SPEAKER=true|false`
- `SWIN_VALLR_MAX_FACES=2`
- `SWIN_VALLR_ALIGN_MOUTH=true|false`
- `SWIN_VALLR_PROFILE_AWARE=true|false`

Data pipeline:
- `SWIN_VALLR_VISUAL_FILTER=true|false`
- `SWIN_VALLR_LANGUAGE_BALANCE=true|false`

NLP utilities:
- `SWIN_VALLR_NLP_DEVICE=cpu|cuda|rocm`
- `SWIN_VALLR_SUMMARY_MODEL=google/flan-t5-small`

## Hardware Support

| Platform | Backend | Notes |
|----------|---------|-------|
| NVIDIA (Win/Linux) | CUDA | Full support |
| AMD (Linux) | ROCm | Requires ROCm install |
| AMD (Windows) | DirectML | Uses safe ops |
| Any | CPU | Slow, for testing |

## Logs and Artifacts

Logs:
- `logs/<component>/<YYYY-MM-DD>/<timestamp>_<pid>.txt`

Artifacts:
- `artifacts/<component>/<YYYY-MM-DD>/<run_id>/`

## Project Structure

```
majorproject/
  app.py
  backend_manager.py
  data_pipeline.py
  dataset_cleaning.py
  env_test.py
  logging_utils.py
  artifact_utils.py
  model_architecture.py
  nlp_utils.py
  refiner_finetune.py
  setup.py
  train_swin_vallr.py
  requirements.txt
  README.md
```

## License

MIT License.

## Contributing

PRs and issues are welcome.
