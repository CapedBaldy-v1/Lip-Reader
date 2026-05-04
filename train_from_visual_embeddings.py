#!/usr/bin/env python3
"""Train Swin-VALLR temporal/CTC layers from cached visual embeddings.

This is the fast training path for deadline runs. It assumes embeddings were
created with:

    create_visual_embeddings.py --stage visual_encoder --pool none

That freezes the Swin visual encoder and trains the downstream temporal adapter,
optional temporal modules, feature norm, and phoneme CTC head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from data_pipeline import get_phoneme_vocab, text_to_phonemes
from model_architecture import SwinConfig, SwinVALLR


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir: Path):
        self.embedding_dir = embedding_dir
        manifest_path = embedding_dir / "manifest.json"
        summary_path = embedding_dir / "summary.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing embedding manifest: {manifest_path}")
        self.summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        self.samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples = [
            sample for sample in self.samples
            if sample.get("embedding") and sample.get("text") and not sample.get("pooled", False)
        ]
        if not self.samples:
            raise RuntimeError("No usable non-pooled embedding samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        emb_path = Path(sample["embedding"])
        if not emb_path.is_absolute():
            emb_path = self.embedding_dir / emb_path
        embedding = np.load(emb_path).astype(np.float32)
        if embedding.ndim != 2:
            raise ValueError(f"Expected temporal embedding (T,D), got {embedding.shape} for {emb_path}")
        return {
            "id": sample["id"],
            "embedding": torch.from_numpy(embedding),
            "text": str(sample.get("text", "")),
        }


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    embeddings = torch.stack([item["embedding"] for item in batch], dim=0)
    return {
        "embedding": embeddings,
        "text": [item["text"] for item in batch],
        "id": [item["id"] for item in batch],
    }


class CTCTextLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = get_phoneme_vocab()
        self.to_idx = {phoneme: idx for idx, phoneme in enumerate(self.vocab)}
        self.blank_idx = len(self.vocab) - 1
        self.loss = nn.CTCLoss(blank=self.blank_idx, reduction="mean", zero_infinity=True)

    def targets(self, texts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        all_targets: List[int] = []
        lengths: List[int] = []
        for text in texts:
            phonemes = text_to_phonemes(text)
            indices = [self.to_idx.get(phoneme, self.blank_idx) for phoneme in phonemes]
            if not indices:
                indices = [self.blank_idx]
            all_targets.extend(indices)
            lengths.append(len(indices))
        return (
            torch.tensor(all_targets, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device),
        )

    def forward(self, log_probs: torch.Tensor, texts: Sequence[str]) -> torch.Tensor:
        batch, steps, _classes = log_probs.shape
        targets, target_lengths = self.targets(texts, log_probs.device)
        input_lengths = torch.full((batch,), steps, dtype=torch.long, device=log_probs.device)
        return self.loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)


class EmbeddingCTCModel(nn.Module):
    def __init__(self, config: SwinConfig):
        super().__init__()
        full = SwinVALLR(config, load_refiner=False)
        self.temporal_adapter = full.temporal_adapter
        self.temporal_multiscale = full.temporal_multiscale
        self.temporal_attention = full.temporal_attention
        self.feature_norm = full.feature_norm
        self.phoneme_head = full.phoneme_head

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        x = self.temporal_adapter(visual_features)
        x = self.temporal_multiscale(x)
        x = self.temporal_attention(x)
        x = self.feature_norm(x)
        return self.phoneme_head(x)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda/ROCm, but torch.cuda.is_available() is false")
    return torch.device(requested)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CTCTextLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="train" if training else "val", unit="batch", leave=False)
    for idx, batch in enumerate(pbar):
        embeddings = batch["embedding"].to(device, non_blocking=True)
        texts = batch["text"]
        with torch.set_grad_enabled(training):
            log_probs = model(embeddings)
            loss = criterion(log_probs, texts)
            if training:
                (loss / max(1, accumulation_steps)).backward()
                if (idx + 1) % max(1, accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        value = float(loss.detach().cpu())
        total += value
        count += 1
        pbar.set_postfix(loss=f"{value:.4f}")
    if training and count % max(1, accumulation_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(1, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal/CTC layers from cached Swin-VALLR visual embeddings.")
    parser.add_argument("--embedding-dir", required=True, help="Embedding directory with manifest.json and embeddings/*.npy.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/embedding_fast_run", help="Output checkpoint directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vram-limit-gb", type=float, default=13.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        if args.vram_limit_gb > 0:
            torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.05, args.vram_limit_gb / total_gb)), 0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.2f} GB)")
    print(f"Device: {device}")

    dataset = EmbeddingDataset(Path(args.embedding_dir))
    val_size = max(1, int(round(len(dataset) * args.val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate,
    )

    model = EmbeddingCTCModel(SwinConfig()).to(device)
    criterion = CTCTextLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "dataset_samples": len(dataset),
                "train_samples": train_size,
                "val_samples": val_size,
                "model": "EmbeddingCTCModel",
                "frozen_visual_encoder": True,
                "swin_config": asdict(SwinConfig()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best = math.inf
    history: List[Dict[str, float]] = []
    print(f"Samples: train={train_size}, val={val_size}, total={len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            accumulation_steps=args.accumulation_steps,
            grad_clip=args.grad_clip,
        )
        val_loss = run_epoch(model, val_loader, criterion, device)
        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_loss < best:
            best = val_loss
            save_checkpoint(checkpoint_dir / "best_embedding_model.pt", model, optimizer, epoch, val_loss, args)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer, epoch, val_loss, args)
    save_checkpoint(checkpoint_dir / "final_embedding_model.pt", model, optimizer, args.epochs, history[-1]["val_loss"], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
