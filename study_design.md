# Study Design and Methods (Codebase-Aligned)

This write-up reflects what is actually implemented in the Swin-VALLR codebase in
`d:\the_code\roman\majorproject`. It maps directly to sections 3.4-3.12 and
uses the project's real data flow (PyTorch, LRS2/LRS3/custom datasets, MediaPipe
ROI extraction, Swin Transformer + CTC, Streamlit demo).

## 3.4 Study Design

**Initial Phase (Data Collection & Preprocessing)**
- The pipeline supports LRS2, LRS3, or a custom dataset (plus a tiny placeholder
  dataset for quick testing). LRS2/LRS3 are downloaded manually due to licensing.
- Raw videos are processed with MediaPipe Face Mesh to extract a mouth ROI.
  Frames are resized to a fixed ROI (default 96x96), sampled to a target FPS
  (default 25), and clipped to a fixed number of frames (default 50).
- Cropping can be fixed, smoothed, or adaptive, and optional mouth alignment is
  applied. A Kalman filter stabilizes center/size across frames.
- Videos are stored as `.npy` arrays; labels are stored as `.txt` transcripts.
  Normalization uses ImageNet mean/std after scaling to [0,1].
- Optional "visual-only" filtering computes face rate, lip size ratio, motion,
  and sharpness to reject low-quality samples.

**Data Pipeline Creation**
- Data loading is handled with a PyTorch `DataLoader` + a custom dataset class
  (`LipReadingDataset`), not `tf.data.Dataset`.
- Training uses a curriculum sampler (single words -> short sentences -> full
  dataset), plus optional language-balanced sampling for multilingual data.
- A custom collate function batches fixed-length video tensors and variable
  length text strings.

**Model Design & Training**
- The model is Swin Transformer-based, built around:
  - 3D tubelet embedding + Swin-Tiny visual encoder
  - Temporal adapter (1D conv) and optional temporal attention/multi-scale fusion
  - CTC phoneme head
  - Optional Qwen2-0.5B refiner for text cleanup
- Training is in PyTorch with AdamW, cosine LR scheduling, AMP (when available),
  gradient clipping, and checkpointing.
- The core loss is CTC. Auxiliary losses include decorrelation, temporal
  smoothness, and attention-entropy regularization.
- LoRA-based fine-tuning for the refiner is supported when enabled.

**Deployment & User Interaction**
- The application is hosted via Streamlit (`app.py`). Users can:
  - Run webcam inference (via `streamlit-webrtc`) or upload a video file
  - See live captions, transcript history, and optional translations/summaries
  - Enable multi-speaker selection (based on mouth motion)
- Inference uses CTC decoding (greedy or beam search) plus optional refiner
  post-processing.

**Testing & Validation**
- There is no dedicated unit-test suite or formal WER/CER evaluation script.
- Validation in practice relies on training loss curves, qualitative inference
  outputs in the Streamlit demo, and optional dataset cleaning/visual filtering
  tools for data quality control.

## 3.5 Study Population

Training and evaluation data come from LRS2/LRS3 (BBC/TED) or a custom dataset
provided by the user. The system expects videos with visible lip motion and
paired text transcripts. Language detection is applied at the sample level, and
optional balancing is available if multilingual data is present.

## 3.6 Sample Size Estimation

There is no fixed sample size hard-coded in the project. The training set size
is determined by whatever is present in the preprocessed dataset directory
(`videos/`, `labels/`, `metadata.json`). The placeholder sample dataset contains
5 examples for quick pipeline testing. If users want separate train/val/test
splits, they must organize them externally or use separate data directories.

## 3.7 Sampling Techniques

- Samples are paired by ID: `videos/<id>.npy` and `labels/<id>.txt` (or via
  `metadata.json`).
- Training uses curriculum learning phases (single-word -> short-sentence -> full
  dataset) with epoch-seeded shuffling.
- An optional language-balanced sampler weights examples by inverse frequency.
- Custom dataset ingestion uses MediaPipe ROI extraction and optional visual
  prefiltering before saving `.npy`/`.txt` artifacts.

## 3.8 Study Variables / Operational Definitions

- **Inputs:** sequences of mouth ROI frames (shape: TxHxWx3).
- **Primary outputs:** phoneme logits (CTC) and decoded phoneme sequences.
- **Final text output:** refined transcription from the optional Qwen2 refiner.
- **Primary optimization variable:** CTC loss on phoneme predictions.
- **Auxiliary optimization variables:** decorrelation loss, temporal smoothness
  loss, and attention-entropy loss (when enabled).
- **Data quality variables (optional):** face detection rate, lip size ratio,
  motion, and sharpness from the visual filter.

## 3.9 Study Tools, Techniques, and Interventions

- **Deep Learning:** PyTorch + Swin Transformer video encoder; CTC decoding.
- **NLP / Refinement:** Qwen2-0.5B refiner (LoRA-fine-tunable), optional
  translation and summarization via HuggingFace models.
- **Computer Vision:** OpenCV for video I/O and MediaPipe Face Mesh for lip ROI.
- **Data Handling:** NumPy for array storage (`.npy`), HuggingFace Hub for sample
  dataset download, TensorBoard for training metrics.
- **Deployment:** Streamlit + streamlit-webrtc for interactive inference.

## 3.10 Data Collection Methods

- LRS2/LRS3 datasets are downloaded manually and processed with
  `data_pipeline.py`, which extracts lip ROIs and writes standardized arrays and
  transcript files.
- Custom datasets are ingested from a user-supplied video directory, processed
  by MediaPipe, and stored in the same `.npy`/`.txt` + `metadata.json` format.
- A small placeholder dataset is generated when no download is available.

## 3.11 Data Management

- Preprocessed data lives in a structured folder:
  - `videos/` -> `.npy` lip ROI sequences
  - `labels/` -> `.txt` transcripts
  - `metadata.json` -> sample metadata
- Logs are stored under `logs/` and artifacts (plots, previews, metrics) under
  `artifacts/` with timestamped subfolders.
- Training and inference load data into memory in batches via PyTorch's
  `DataLoader`.

## 3.12 Data Analysis

- Model performance is tracked through CTC loss and auxiliary losses logged to
  TensorBoard and CSV artifacts.
- During inference, CTC decoding (greedy or beam search) produces phoneme
  sequences, which can be refined into readable text.
- The project does not implement formal WER/CER scoring in the current codebase;
  evaluation is primarily loss-driven and qualitative via the Streamlit demo.
