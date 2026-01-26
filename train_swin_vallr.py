"""
Swin-VALLR Training Script
===========================
Training script optimized for AMD 6800XT with mixed precision support.

Features:
- Automatic Mixed Precision (AMP) for ROCm/CUDA
- CTC Loss for phoneme prediction
- LoRA fine-tuning for Qwen2-0.5B
- Curriculum learning phases
- Checkpoint saving/resuming
- TensorBoard logging
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend_manager import (
    get_device, get_backend, is_mixed_precision_available,
    get_autocast_context, get_grad_scaler, print_device_info
)
from dataclasses import asdict

from model_architecture import SwinVALLR, SwinConfig, create_model, count_parameters
from data_pipeline import (
    create_dataloader, DataConfig, get_phoneme_vocab, text_to_phonemes
)
from logging_utils import setup_logging, log_system_info, log_exception
from artifact_utils import (
    get_artifact_root,
    get_useful_dir,
    save_json,
    save_csv,
    save_line_plot,
    save_montage,
    save_text,
    save_numpy,
    save_image
)


LOGGER = setup_logging("training")


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Training hyperparameters."""

    # Basic training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Gradient settings
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Curriculum learning
    curriculum_phases: Tuple[int, int] = (5, 20)

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5

    # Logging
    log_every: int = 10
    eval_every: int = 1

    # LoRA settings for LLM fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Decorrelation loss (paper-4 inspired stabilization)
    decorrelation_weight: float = 0.03
    decorrelation_anchor: str = "mean"  # "mean" or "first"

    # CTC alignment smoothing (paper-2+ inspired)
    temporal_smoothness_weight: float = 0.02

    # Attention sink mitigation (paper-4+ inspired)
    attention_entropy_weight: float = 0.01


# =============================================================================
# CTC Loss with Label Preparation
# =============================================================================

class CTCLossWrapper(nn.Module):
    """
    CTC Loss wrapper that handles label preparation.
    """

    def __init__(self, blank_idx: int = 39):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        self.phoneme_vocab = get_phoneme_vocab()
        self.phoneme_to_idx = {p: i for i, p in enumerate(self.phoneme_vocab)}
        self.blank_idx = blank_idx

    def text_to_targets(self, texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text strings to phoneme targets.

        Args:
            texts: List of text strings

        Returns:
            targets: Flattened target tensor
            target_lengths: Length of each target sequence
        """
        all_targets = []
        target_lengths = []

        for text in texts:
            phonemes = text_to_phonemes(text)
            indices = [self.phoneme_to_idx.get(p, self.blank_idx) for p in phonemes]
            if not indices:
                LOGGER.warning("Empty phoneme target for text: %s", text)
                indices = [self.blank_idx]

            all_targets.extend(indices)
            target_lengths.append(len(indices))

        targets = torch.tensor(all_targets, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        LOGGER.debug("Prepared CTC targets: %d sequences", len(target_lengths))
        return targets, target_lengths

    def forward(
        self,
        log_probs: torch.Tensor,
        texts: list,
        input_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            log_probs: Log probabilities from model (B, T, C)
            texts: List of target text strings
            input_lengths: Length of each input sequence

        Returns:
            CTC loss value
        """
        batch_size, time_steps, num_classes = log_probs.shape

        targets, target_lengths = self.text_to_targets(texts)
        targets = targets.to(log_probs.device)
        target_lengths = target_lengths.to(log_probs.device)

        if input_lengths is None:
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=log_probs.device)

        log_probs = log_probs.permute(1, 0, 2)

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        LOGGER.debug("CTC loss computed: %f", loss.item())
        return loss


# =============================================================================
# LoRA Setup for LLM Fine-tuning
# =============================================================================

def setup_lora(model: SwinVALLR, config: TrainingConfig) -> Optional[nn.Module]:
    """
    Set up LoRA for LLM fine-tuning.

    Args:
        model: SwinVALLR model with loaded refiner
        config: Training configuration

    Returns:
        LoRA-enabled model or None if refiner not loaded
    """
    if model.refiner is None or model.refiner.model is None:
        print("[LoRA] Skipping LoRA setup - refiner not loaded")
        LOGGER.warning("LoRA setup skipped - refiner not loaded.")
        return None

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model.refiner.model = get_peft_model(model.refiner.model, lora_config)

        trainable_params = sum(p.numel() for p in model.refiner.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.refiner.model.parameters())

        print(f"[LoRA] Enabled with r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"[LoRA] Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        LOGGER.info(
            "LoRA enabled (r=%s, alpha=%s, dropout=%s). Trainable=%s Total=%s",
            config.lora_r, config.lora_alpha, config.lora_dropout, trainable_params, total_params
        )

        return model.refiner.model

    except ImportError:
        print("[LoRA] PEFT not available - skipping LoRA setup")
        LOGGER.warning("PEFT not available; skipping LoRA setup.")
        return None


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """
    Training manager for Swin-VALLR.
    """

    def __init__(
        self,
        model: SwinVALLR,
        train_loader,
        val_loader,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.ctc_loss = CTCLossWrapper()

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )

        self.use_amp = is_mixed_precision_available()
        self.scaler = get_grad_scaler() if self.use_amp else None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.checkpoint_dir / "logs")

        self.artifact_root = get_artifact_root("training")
        self.loss_artifacts = get_useful_dir("training", "loss_curves")
        self.preview_artifacts = get_useful_dir("training", "batch_previews")

        self.step_metrics: List[Tuple[int, float, float, float, float, float]] = []
        self.epoch_metrics: List[Tuple[int, float, float, float, float, float]] = []
        self.val_losses: List[Tuple[int, float]] = []
        self._preview_epochs = set()
        self._logit_epochs = set()
        self._prediction_epochs = set()

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        print(f"[Trainer] Initialized with AMP: {self.use_amp}")
        LOGGER.info("Trainer initialized (AMP=%s).", self.use_amp)
        self._save_training_metadata()

    def _save_training_metadata(self) -> None:
        """Persist training configuration and environment details."""
        try:
            summary = {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "gradient_clip": self.config.gradient_clip,
                "accumulation_steps": self.config.accumulation_steps,
                "curriculum_phases": self.config.curriculum_phases,
                "decorrelation_weight": self.config.decorrelation_weight,
                "decorrelation_anchor": self.config.decorrelation_anchor,
                "temporal_smoothness_weight": self.config.temporal_smoothness_weight,
                "attention_entropy_weight": self.config.attention_entropy_weight
            }
            save_json(self.artifact_root / "training_config.json", summary)
            save_text(self.artifact_root / "notes.txt", "Training artifacts and metrics.")
        except Exception:
            log_exception(LOGGER, "Failed to save training metadata.")

    def _decorrelation_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Penalize cosine similarity between an anchor token and other tokens."""
        if self.config.decorrelation_weight <= 0:
            return torch.tensor(0.0, device=features.device)

        if features.ndim != 3 or features.shape[1] < 2:
            return torch.tensor(0.0, device=features.device)

        if self.config.decorrelation_anchor == "first":
            anchor = features[:, 0, :]
            tokens = features[:, 1:, :]
        else:
            anchor = features.mean(dim=1)
            tokens = features

        anchor = torch.nn.functional.normalize(anchor, dim=-1)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)

        cosine = (tokens * anchor.unsqueeze(1)).sum(dim=-1)
        return (cosine ** 2).mean()

    def _temporal_smoothness_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Encourage smooth frame-to-frame CTC logits."""
        if self.config.temporal_smoothness_weight <= 0:
            return torch.tensor(0.0, device=log_probs.device)
        if log_probs.ndim != 3 or log_probs.shape[1] < 2:
            return torch.tensor(0.0, device=log_probs.device)
        diffs = log_probs[:, 1:, :] - log_probs[:, :-1, :]
        return (diffs ** 2).mean()

    def _attention_entropy_loss(self, attn_weights: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Penalize low-entropy attention (attention sinks)."""
        if self.config.attention_entropy_weight <= 0:
            return torch.tensor(0.0, device=device)
        if not attn_weights:
            return torch.tensor(0.0, device=device)

        penalties: List[torch.Tensor] = []
        for weights in attn_weights:
            if weights is None:
                continue
            if weights.ndim == 3:
                weights = weights.unsqueeze(1)
            if weights.ndim != 4:
                continue
            p = weights.clamp(min=1e-8)
            entropy = -(p * p.log()).sum(dim=-1)
            denom = math.log(p.shape[-1]) if p.shape[-1] > 1 else 1.0
            entropy_norm = entropy / denom
            penalties.append(1.0 - entropy_norm.mean())

        if not penalties:
            return torch.tensor(0.0, device=device)
        return torch.stack(penalties).mean()

    def _save_loss_artifacts(self) -> None:
        """Save loss curves and raw loss data."""
        try:
            if self.step_metrics:
                save_csv(
                    self.loss_artifacts / "train_steps.csv",
                    [
                        (step, total_loss, ctc_loss, decor_loss, smooth_loss, attn_loss)
                        for step, total_loss, ctc_loss, decor_loss, smooth_loss, attn_loss in self.step_metrics
                    ],
                    headers=[
                        "step",
                        "total_loss",
                        "ctc_loss",
                        "decorrelation_loss",
                        "smoothness_loss",
                        "attention_entropy_loss"
                    ]
                )
                save_line_plot(
                    self.loss_artifacts / "train_step_loss.png",
                    {
                        "train_step_loss": [loss for _, loss, _, _, _, _ in self.step_metrics],
                        "train_step_ctc_loss": [loss for _, _, loss, _, _, _ in self.step_metrics],
                        "train_step_decor_loss": [loss for _, _, _, loss, _, _ in self.step_metrics],
                        "train_step_smooth_loss": [loss for _, _, _, _, loss, _ in self.step_metrics],
                        "train_step_attn_loss": [loss for _, _, _, _, _, loss in self.step_metrics]
                    },
                    title="Train Step Loss"
                )

            if self.epoch_metrics:
                save_csv(
                    self.loss_artifacts / "epoch_metrics.csv",
                    [
                        (
                            epoch,
                            train_loss,
                            train_ctc_loss,
                            train_decorr_loss,
                            train_smooth_loss,
                            train_attn_loss,
                            self._val_loss_for_epoch(epoch)
                        )
                        for epoch, train_loss, train_ctc_loss, train_decorr_loss, train_smooth_loss, train_attn_loss in self.epoch_metrics
                    ],
                    headers=[
                        "epoch",
                        "train_loss",
                        "train_ctc_loss",
                        "train_decorr_loss",
                        "train_smooth_loss",
                        "train_attn_loss",
                        "val_loss"
                    ]
                )

                series = {
                    "train_epoch_loss": [loss for _, loss, _, _, _, _ in self.epoch_metrics],
                    "train_epoch_ctc_loss": [loss for _, _, loss, _, _, _ in self.epoch_metrics],
                    "train_epoch_smooth_loss": [loss for _, _, _, _, loss, _ in self.epoch_metrics],
                    "train_epoch_attn_loss": [loss for _, _, _, _, _, loss in self.epoch_metrics]
                }
                if self.config.decorrelation_weight > 0:
                    series["train_epoch_decor_loss"] = [
                        loss for _, _, _, loss, _, _ in self.epoch_metrics
                    ]
                if self.val_losses:
                    series["val_epoch_loss"] = [loss for _, loss in self.val_losses]
                save_line_plot(
                    self.loss_artifacts / "epoch_loss.png",
                    series,
                    title="Epoch Loss"
                )
        except Exception:
            log_exception(LOGGER, "Failed to save loss artifacts.")

    def _val_loss_for_epoch(self, epoch: int) -> Optional[float]:
        for ep, loss in self.val_losses:
            if ep == epoch:
                return loss
        return None

    def _save_batch_preview(self, video: torch.Tensor, epoch: int) -> None:
        """Save a montage preview for the first batch in an epoch."""
        if epoch in self._preview_epochs:
            return
        try:
            sample = video[0].detach().cpu().float()
            sample = sample.permute(1, 2, 3, 0).numpy()

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
            sample = (sample * std + mean) * 255.0
            sample = np.clip(sample, 0, 255).astype(np.uint8)

            frames = [frame for frame in sample[:min(sample.shape[0], 20)]]
            save_montage(self.preview_artifacts / f"epoch_{epoch+1}.png", frames, cols=10, pad=2)
            self._preview_epochs.add(epoch)
        except Exception:
            log_exception(LOGGER, "Failed to save batch preview.")

    def _save_logit_matrix(self, logits: torch.Tensor, epoch: int) -> None:
        """Save logit matrix and heatmap for the first batch in an epoch."""
        if epoch in self._logit_epochs:
            return
        try:
            out_dir = get_useful_dir("training", "logit_matrices")
            matrix = logits[0].detach().cpu().float().numpy()
            save_numpy(out_dir / f"epoch_{epoch+1}.npy", matrix)

            # Create a heatmap image for quick inspection.
            min_val = float(matrix.min())
            max_val = float(matrix.max())
            if max_val == min_val:
                max_val += 1.0
            normalized = (matrix - min_val) / (max_val - min_val)
            heatmap = (normalized * 255).astype(np.uint8)
            try:
                import cv2
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
            except Exception:
                heatmap = np.stack([heatmap] * 3, axis=-1)
            save_image(out_dir / f"epoch_{epoch+1}.png", heatmap)

            self._logit_epochs.add(epoch)
        except Exception:
            log_exception(LOGGER, "Failed to save logit matrix.")

    def _save_sample_predictions(self, logits: torch.Tensor, texts: List[str], epoch: int) -> None:
        """Save a small set of decoded predictions for inspection."""
        if epoch in self._prediction_epochs:
            return
        try:
            out_dir = get_useful_dir("training", "sample_predictions")
            decoded = self.model.decode_ctc(logits.detach())
            samples = []
            for i in range(min(3, len(decoded))):
                samples.append({
                    "target_text": texts[i] if i < len(texts) else "",
                    "predicted_phonemes": decoded[i]
                })
            save_json(out_dir / f"epoch_{epoch+1}.json", {"samples": samples})
            self._prediction_epochs.add(epoch)
        except Exception:
            log_exception(LOGGER, "Failed to save sample predictions.")

    def _run_backward(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Handle backward pass with or without AMP."""
        loss = loss / self.config.accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

    def _log_training_step(
        self,
        total_loss: float,
        ctc_loss: float,
        decor_loss: float,
        smooth_loss: float,
        attn_loss: float
    ) -> None:
        """Write training metrics to TensorBoard."""
        self.writer.add_scalar('train/loss', total_loss, self.global_step)
        self.writer.add_scalar('train/ctc_loss', ctc_loss, self.global_step)
        if self.config.decorrelation_weight > 0:
            self.writer.add_scalar('train/decorrelation_loss', decor_loss, self.global_step)
        if self.config.temporal_smoothness_weight > 0:
            self.writer.add_scalar('train/smoothness_loss', smooth_loss, self.global_step)
        if self.config.attention_entropy_weight > 0:
            self.writer.add_scalar('train/attention_entropy_loss', attn_loss, self.global_step)
        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ctc_loss = 0.0
        total_decor_loss = 0.0
        total_smooth_loss = 0.0
        total_attn_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            video = batch['video'].to(self.device)
            texts = batch['text']

            if batch_idx == 0:
                self._save_batch_preview(video, self.current_epoch)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if self.config.attention_entropy_weight > 0:
                        features, attn_weights = self.model.extract_features_with_attention(video)
                    else:
                        features = self.model.extract_features(video)
                        attn_weights = []
                    logits = self.model.phoneme_head(features)
                    ctc_loss = self.ctc_loss(logits, texts)
                    decor_loss = self._decorrelation_loss(features)
                    smooth_loss = self._temporal_smoothness_loss(logits)
                    attn_loss = self._attention_entropy_loss(attn_weights, device=logits.device)
                    loss = (
                        ctc_loss
                        + self.config.decorrelation_weight * decor_loss
                        + self.config.temporal_smoothness_weight * smooth_loss
                        + self.config.attention_entropy_weight * attn_loss
                    )
            else:
                if self.config.attention_entropy_weight > 0:
                    features, attn_weights = self.model.extract_features_with_attention(video)
                else:
                    features = self.model.extract_features(video)
                    attn_weights = []
                logits = self.model.phoneme_head(features)
                ctc_loss = self.ctc_loss(logits, texts)
                decor_loss = self._decorrelation_loss(features)
                smooth_loss = self._temporal_smoothness_loss(logits)
                attn_loss = self._attention_entropy_loss(attn_weights, device=logits.device)
                loss = (
                    ctc_loss
                    + self.config.decorrelation_weight * decor_loss
                    + self.config.temporal_smoothness_weight * smooth_loss
                    + self.config.attention_entropy_weight * attn_loss
                )

            if batch_idx == 0:
                self._save_logit_matrix(logits, self.current_epoch)
                self._save_sample_predictions(logits, texts, self.current_epoch)

            self._run_backward(loss, batch_idx)

            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_decor_loss += decor_loss.item()
            total_smooth_loss += smooth_loss.item()
            total_attn_loss += attn_loss.item()
            num_batches += 1
            self.global_step += 1
            LOGGER.debug(
                "Train step=%s total_loss=%f ctc_loss=%f decor_loss=%f smooth_loss=%f attn_loss=%f",
                self.global_step,
                loss.item(),
                ctc_loss.item(),
                decor_loss.item(),
                smooth_loss.item(),
                attn_loss.item()
            )
            self.step_metrics.append(
                (
                    self.global_step,
                    loss.item(),
                    ctc_loss.item(),
                    decor_loss.item(),
                    smooth_loss.item(),
                    attn_loss.item()
                )
            )

            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'ctc': total_ctc_loss / num_batches
            })

            if self.global_step % self.config.log_every == 0:
                self._log_training_step(
                    loss.item(),
                    ctc_loss.item(),
                    decor_loss.item(),
                    smooth_loss.item(),
                    attn_loss.item()
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            video = batch['video'].to(self.device)
            texts = batch['text']

            logits = self.model(video)
            loss = self.ctc_loss(logits, texts)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        self.writer.add_scalar('val/loss', avg_loss, self.current_epoch)
        LOGGER.info("Validation complete (epoch=%s, loss=%f).", self.current_epoch, avg_loss)
        self.val_losses.append((self.current_epoch + 1, avg_loss))

        return avg_loss

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
            'model_config': asdict(self.model.config)
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"[Trainer] Saved checkpoint: {filename}")
        LOGGER.info("Checkpoint saved: %s", self.checkpoint_dir / filename)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"[Trainer] Loaded checkpoint from epoch {self.current_epoch}")
        LOGGER.info("Checkpoint loaded: %s (epoch=%s)", checkpoint_path, self.current_epoch)

    def _describe_curriculum_phase(self, epoch: int) -> str:
        """Return the curriculum phase string for a given epoch."""
        if epoch < self.config.curriculum_phases[0]:
            return "Single Words"
        if epoch < self.config.curriculum_phases[1]:
            return "Short Sentences"
        return "Full Dataset"

    def train(self):
        """Full training loop."""
        print(f"\n[Trainer] Starting training for {self.config.epochs} epochs")
        print(f"[Trainer] Batch size: {self.config.batch_size}")
        print(f"[Trainer] Learning rate: {self.config.learning_rate}")
        print(f"[Trainer] Curriculum phases: {self.config.curriculum_phases}")
        LOGGER.info(
            "Training started (epochs=%s, batch_size=%s, lr=%s, curriculum=%s).",
            self.config.epochs, self.config.batch_size, self.config.learning_rate, self.config.curriculum_phases
        )

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            phase = self._describe_curriculum_phase(epoch)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Curriculum Phase: {phase}")
            print(f"{'='*60}")

            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            LOGGER.info("Epoch %s train loss: %f", epoch + 1, train_loss)
            avg_ctc = total_ctc = 0.0
            avg_decor = total_decor = 0.0
            avg_smooth = total_smooth = 0.0
            avg_attn = total_attn = 0.0
            if self.step_metrics:
                last_steps = self.step_metrics[-len(self.train_loader):]
                total_ctc = sum(item[2] for item in last_steps)
                total_decor = sum(item[3] for item in last_steps)
                total_smooth = sum(item[4] for item in last_steps)
                total_attn = sum(item[5] for item in last_steps)
                avg_ctc = total_ctc / max(len(last_steps), 1)
                avg_decor = total_decor / max(len(last_steps), 1)
                avg_smooth = total_smooth / max(len(last_steps), 1)
                avg_attn = total_attn / max(len(last_steps), 1)
            self.epoch_metrics.append((epoch + 1, train_loss, avg_ctc, avg_decor, avg_smooth, avg_attn))
            self._save_loss_artifacts()

            self.scheduler.step()

            if (epoch + 1) % self.config.eval_every == 0:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")
                LOGGER.info("Epoch %s val loss: %f", epoch + 1, val_loss)
                self._save_loss_artifacts()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        self.save_checkpoint("final_model.pt")
        print("\n[Trainer] Training complete!")
        LOGGER.info("Training complete.")

        self.writer.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Swin-VALLR model")

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to preprocessed dataset')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')

    parser.add_argument('--load_refiner', action='store_true',
                        help='Load Qwen2 refiner for joint training')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    parser.add_argument('--decorrelation_weight', type=float, default=0.03,
                        help='Weight for decorrelation loss (default: 0.03)')
    parser.add_argument('--decorrelation_anchor', type=str, default='mean',
                        choices=['mean', 'first'],
                        help='Anchor choice for decorrelation loss (mean or first)')
    parser.add_argument('--temporal_smoothness_weight', type=float, default=0.02,
                        help='Weight for temporal smoothness loss (default: 0.02)')
    parser.add_argument('--attention_entropy_weight', type=float, default=0.01,
                        help='Weight for attention entropy loss (default: 0.01)')

    parser.add_argument('--temporal_attention_layers', type=int, default=1,
                        help='Number of temporal attention layers (default: 1)')
    parser.add_argument('--temporal_attention_heads', type=int, default=16,
                        help='Heads in temporal attention (default: 16)')
    parser.add_argument('--temporal_attention_kernel', type=int, default=3,
                        help='Kernel size for temporal conv (default: 3)')
    parser.add_argument('--temporal_attention_dropout', type=float, default=0.1,
                        help='Dropout for temporal attention (default: 0.1)')
    parser.add_argument('--temporal_attention_streaming', action='store_true',
                        help='Disable MHSA in temporal attention for streaming')
    parser.add_argument('--disable_temporal_multiscale', action='store_true',
                        help='Disable multi-scale temporal fusion (enabled by default)')

    return parser.parse_args()


def _build_training_config(args) -> TrainingConfig:
    """Create TrainingConfig from CLI args."""
    config = TrainingConfig()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.checkpoint_dir = args.checkpoint_dir
    config.decorrelation_weight = args.decorrelation_weight
    config.decorrelation_anchor = args.decorrelation_anchor
    config.temporal_smoothness_weight = args.temporal_smoothness_weight
    config.attention_entropy_weight = args.attention_entropy_weight
    return config


def _build_model_config(args) -> SwinConfig:
    """Create SwinConfig from CLI args."""
    config = SwinConfig()
    config.temporal_multiscale = not args.disable_temporal_multiscale
    config.temporal_attention_layers = args.temporal_attention_layers
    config.temporal_attention_heads = args.temporal_attention_heads
    config.temporal_attention_kernel = args.temporal_attention_kernel
    config.temporal_attention_dropout = args.temporal_attention_dropout
    config.temporal_attention_use_mhsa = not args.temporal_attention_streaming
    return config


def main():
    """Main training function."""
    args = parse_args()
    log_system_info(LOGGER)
    LOGGER.info("Training args: %s", args)

    print_device_info()
    device = get_device()

    config = _build_training_config(args)
    LOGGER.info("Training config: %s", vars(config))

    print("\n[Main] Creating model...")
    model_config = _build_model_config(args)
    model = create_model(load_refiner=args.load_refiner, config=model_config)
    LOGGER.info("Model created. Refiner loaded: %s", args.load_refiner)

    if args.load_refiner:
        setup_lora(model, config)

    print("\n[Main] Creating dataloaders...")
    train_loader = create_dataloader(
        args.data_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        epoch=0,
        training=True
    )

    val_loader = create_dataloader(
        args.data_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        epoch=0,
        training=False
    )

    LOGGER.info("Train loader batches: %s", len(train_loader))
    LOGGER.info("Val loader batches: %s", len(val_loader))
    if len(train_loader) == 0:
        raise RuntimeError("No training samples found. Check data_dir and preprocessing outputs.")
    if len(val_loader) == 0:
        LOGGER.warning("Validation loader is empty; validation will be skipped.")

    trainer = Trainer(model, train_loader, val_loader, config, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_exception(LOGGER, "Training failed with an unhandled exception.")
        raise
