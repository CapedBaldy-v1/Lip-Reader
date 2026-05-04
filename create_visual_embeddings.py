#!/usr/bin/env python3
"""Create cached Swin-VALLR temporal embeddings for a preprocessed dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from data_pipeline import DataConfig
from model_architecture import SwinConfig, SwinVALLR


def load_metadata(data_dir: Path) -> List[Dict[str, Any]]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"metadata.json must contain a list, got {type(data).__name__}")
    return data


def resolve_video_path(data_dir: Path, sample: Dict[str, Any]) -> Path:
    video = sample.get("video")
    if not video:
        raise ValueError(f"Sample {sample.get('id', '<unknown>')} has no video path")
    path = Path(str(video))
    if not path.is_absolute():
        path = data_dir / path
    return path


def normalize_clip(clip: np.ndarray, config: DataConfig) -> np.ndarray:
    if clip.ndim == 3:
        clip = clip[..., None]
    if clip.ndim != 4:
        raise ValueError(f"Expected clip shape (T,H,W,C), got {clip.shape}")
    if clip.shape[-1] == 1:
        clip = np.repeat(clip, 3, axis=-1)
    elif clip.shape[-1] > 3:
        clip = clip[..., :3]

    if clip.shape[0] != config.num_frames:
        raise ValueError(f"Expected {config.num_frames} frames, got {clip.shape[0]}")
    if clip.shape[1] != config.roi_size or clip.shape[2] != config.roi_size:
        raise ValueError(f"Expected ROI {config.roi_size}x{config.roi_size}, got {clip.shape[1:3]}")

    clip = clip.astype(np.float32) / 255.0
    mean = np.array(config.mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(config.std, dtype=np.float32).reshape(1, 1, 1, 3)
    return (clip - mean) / std


def load_checkpoint(model: SwinVALLR, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model")
        if state is None:
            state = checkpoint
    else:
        state = checkpoint

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] checkpoint missing {len(missing)} keys")
    if unexpected:
        print(f"[warn] checkpoint has {len(unexpected)} unexpected keys")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda/ROCm, but torch.cuda.is_available() is false")
    return torch.device(requested)


def save_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cached temporal visual embeddings from Swin-VALLR preprocessed clips.")
    parser.add_argument("--data-dir", required=True, help="Preprocessed dataset directory with metadata.json/videos/labels.")
    parser.add_argument("--output", required=True, help="Output embedding cache directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint to load before embedding.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Embedding device.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num-frames", type=int, default=50, help="Expected frames per preprocessed clip.")
    parser.add_argument("--roi-size", type=int, default=96, help="Expected ROI size.")
    parser.add_argument("--vram-limit-gb", type=float, default=13.0, help="CUDA/ROCm memory cap; 0 disables.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing embedding files.")
    parser.add_argument(
        "--stage",
        choices=["visual_encoder", "ctc_features"],
        default="visual_encoder",
        help=(
            "visual_encoder caches frozen Swin visual features for fast downstream training; "
            "ctc_features also runs temporal adapter/attention/norm and is mainly for inference/analysis."
        ),
    )
    parser.add_argument("--pool", choices=["none", "mean"], default="none", help="Save full temporal features or a temporal mean.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N metadata samples; 0 means all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_dir)
    if args.limit > 0:
        metadata = metadata[: args.limit]

    device = resolve_device(args.device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        if args.vram_limit_gb > 0:
            fraction = min(1.0, max(0.05, args.vram_limit_gb / total_gb))
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.2f} GB)")
    print(f"Device: {device}")

    data_config = DataConfig(num_frames=args.num_frames, roi_size=args.roi_size)
    model = SwinVALLR(SwinConfig(), load_refiner=False)
    if args.checkpoint:
        load_checkpoint(model, Path(args.checkpoint))
        checkpoint_name: Optional[str] = str(Path(args.checkpoint).resolve())
    else:
        checkpoint_name = None
        print("[warn] no checkpoint supplied; embeddings will come from randomly initialized Swin-VALLR weights")
    model.to(device)
    model.eval()

    manifest: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    batch_clips: List[torch.Tensor] = []
    batch_samples: List[Dict[str, Any]] = []

    def flush() -> None:
        if not batch_clips:
            return
        video = torch.stack(batch_clips, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            if args.stage == "visual_encoder":
                features = model.visual_encoder(video)
            else:
                features = model.extract_features(video)
            features = features.detach().float().cpu().numpy()
        for sample, embedding in zip(batch_samples, features):
            sample_id = str(sample["id"])
            if args.pool == "mean":
                stored = embedding.mean(axis=0).astype(np.float32)
            else:
                stored = embedding.astype(np.float32)
            out_path = embeddings_dir / f"{sample_id}.npy"
            np.save(out_path, stored)
            manifest.append({
                "id": sample_id,
                "embedding": str(out_path.relative_to(output_dir)),
                "source_video": sample.get("video"),
                "label": sample.get("label"),
                "text": sample.get("text", ""),
                "word_count": sample.get("word_count"),
                "shape": list(stored.shape),
                "stage": args.stage,
                "pooled": args.pool == "mean",
            })
        batch_clips.clear()
        batch_samples.clear()

    for sample in tqdm(metadata, desc="Embeddings", unit="sample"):
        sample_id = str(sample.get("id") or "")
        if not sample_id:
            errors.append({"id": "<missing>", "error": "missing sample id"})
            continue
        out_path = embeddings_dir / f"{sample_id}.npy"
        if out_path.exists() and not args.overwrite:
            manifest.append({
                "id": sample_id,
                "embedding": str(out_path.relative_to(output_dir)),
                "source_video": sample.get("video"),
                "label": sample.get("label"),
                "text": sample.get("text", ""),
                "word_count": sample.get("word_count"),
                "shape": list(np.load(out_path, mmap_mode="r").shape),
                "stage": args.stage,
                "pooled": args.pool == "mean",
                "reused": True,
            })
            continue
        try:
            clip = np.load(resolve_video_path(data_dir, sample))
            clip = normalize_clip(clip, data_config)
            tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float()
        except Exception as exc:
            errors.append({"id": sample_id, "error": str(exc)})
            continue
        batch_clips.append(tensor)
        batch_samples.append(sample)
        if len(batch_clips) >= args.batch_size:
            flush()
    flush()

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir.resolve()),
        "output": str(output_dir.resolve()),
        "checkpoint": checkpoint_name,
        "device": str(device),
        "samples_in_metadata": len(metadata),
        "embeddings": len(manifest),
        "errors": len(errors),
        "pool": args.pool,
        "stage": args.stage,
        "num_frames": args.num_frames,
        "roi_size": args.roi_size,
    }
    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "errors.json", errors)
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
