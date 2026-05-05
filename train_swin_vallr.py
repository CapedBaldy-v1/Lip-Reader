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

# AF_UNIX sockets have a 108-char path limit on Linux. Force TMPDIR to a short
# path before any imports so multiprocessing worker processes inherit it.
import os as _os
_train_tmp = _os.environ.get("TMPDIR") or "/tmp/agam-train-tmp"
if len(_train_tmp) > 80:
    _train_tmp = "/tmp/agam-train-tmp"
_os.makedirs(_train_tmp, exist_ok=True)
for _v in ("TMPDIR", "TEMP", "TMP"):
    _os.environ[_v] = _train_tmp

import os
import sys
import argparse
import time
import math
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime

import numpy as np
import torch
import gc  # For memory cleanup
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
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

    # Basic training - OPTIMIZED FOR SMALL DATASETS
    batch_size: int = 4  # REDUCED from 32 for small datasets
    epochs: int = 300  # INCREASED to 300 for better convergence (early stopping will handle it)
    learning_rate: float = 3e-5  # REDUCED from 1e-4 for better convergence
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Gradient settings
    gradient_clip: float = 0.5  # REDUCED from 1.0 to prevent gradient explosion
    accumulation_steps: int = 2  # REDUCED to 2 for batch_size=8 (effective batch = 8*2=16)

    # Curriculum learning
    curriculum_phases: Tuple[int, int] = (10, 50)  # Adjusted for 300 epochs: single words (1-10), short sentences (11-50), full dataset (51+)
    
    # NaN detection and handling
    detect_nan: bool = True  # Stop training if NaN detected
    log_gradient_norms: bool = True  # Log gradient norms to detect explosion early

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save checkpoint every 5 epochs

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
    
    # NEW: Label smoothing for better generalization
    label_smoothing: float = 0.0
    
    # NEW: Use beam search by default
    use_beam_search: bool = True
    beam_width: int = 10
    
    # OPTIMIZATION: Early stopping for 20-hour deadline
    early_stopping_patience: int = 100  # INCREASED to 100 for slow learning rate (was 20)
    early_stopping_min_delta: float = 0.0001  # REDUCED to capture smaller improvements (was 0.001)
    
    # OPTIMIZATION: Adaptive validation (validate less frequently early on)
    adaptive_validation: bool = True
    adaptive_val_start_every: int = 2  # REDUCED: Validate every 2 epochs initially (see progress faster)
    adaptive_val_end_every: int = 1     # REDUCED: Validate every epoch after warmup (see all progress)
    
    # OPTIMIZATION: Reproducibility
    seed: int = 42
    deterministic: bool = False  # Set to True for full reproducibility (slower)
    
    # OPTIMIZATION: Smoke test mode
    smoke_test: bool = False  # Run 2 epochs with 10 batches each to verify setup
    
    # OPTIMIZATION: Time budget (in hours, 0 = no limit)
    time_budget_hours: float = 0.0  # Set to 19.5 to leave buffer for 20-hour deadline

    # Fast fine-tuning controls
    freeze_visual_encoder: bool = False
    progress_every: int = 100
    save_last_every: int = 3


# =============================================================================
# CTC Loss with Label Preparation
# =============================================================================

class CTCLossWrapper(nn.Module):
    """
    CTC Loss wrapper that handles label preparation with optional label smoothing.
    """

    def __init__(self, blank_idx: int = 39, label_smoothing: float = 0.0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        self.phoneme_vocab = get_phoneme_vocab()
        self.phoneme_to_idx = {p: i for i, p in enumerate(self.phoneme_vocab)}
        self.blank_idx = blank_idx
        self.label_smoothing = label_smoothing

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
            phonemes = _target_phoneme_sequence(text, self.phoneme_vocab)
            indices = [self.phoneme_to_idx[p] for p in phonemes if p in self.phoneme_to_idx]
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
        Compute CTC loss with optional label smoothing.

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

        log_probs_ctc = log_probs.permute(1, 0, 2)

        # Standard CTC loss
        ctc_loss = self.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)
        
        # Label smoothing: encourage uniform distribution
        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean()
            loss = (1 - self.label_smoothing) * ctc_loss + self.label_smoothing * smooth_loss
        else:
            loss = ctc_loss

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
# Helpers
# =============================================================================

def _char_error_rate(pred: str, target: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    if not target:
        return 0.0 if not pred else 1.0
    m, n = len(pred), len(target)
    # single-row DP
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n] / n


def _word_error_rate(pred: str, target: str) -> float:
    """Word Error Rate via Levenshtein distance on word tokens."""
    p_words = pred.split()
    t_words = target.split()
    if not t_words:
        return 0.0 if not p_words else 1.0
    m, n = len(p_words), len(t_words)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if p_words[i - 1] == t_words[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n] / n


def _target_phoneme_sequence(text: str, vocab: Optional[List[str]] = None) -> List[str]:
    """Convert text to trainable phoneme tokens, excluding unknowns and blanks."""
    allowed = set(vocab or get_phoneme_vocab())
    allowed.discard('<blank>')
    return [p for p in text_to_phonemes(text) if p in allowed]


def _update_confusion(
    confusion: "np.ndarray",
    ph_correct: "np.ndarray",
    ph_total: "np.ndarray",
    true_phonemes: List[str],
    pred_phonemes: List[str],
    ph_to_idx: Dict[str, int],
) -> None:
    """
    Align true and predicted phoneme sequences with a simple greedy alignment
    and update the confusion matrix and per-phoneme accuracy arrays in-place.
    Uses edit-distance alignment so insertions/deletions are handled gracefully.
    """
    n_ph = confusion.shape[0]
    tp = [ph_to_idx[p] for p in true_phonemes if p in ph_to_idx]
    pp = [ph_to_idx[p] for p in pred_phonemes if p in ph_to_idx]
    if not tp:
        return

    # DP alignment (same as Levenshtein but we trace back)
    m, n = len(tp), len(pp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tp[i - 1] == pp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Traceback
    i, j = m, n
    while i > 0:
        t_idx = tp[i - 1]
        if t_idx >= n_ph:
            i -= 1
            continue
        ph_total[t_idx] += 1
        if j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if tp[i-1] == pp[j-1] else 1):
            p_idx = pp[j - 1] if j > 0 else -1
            if 0 <= p_idx < n_ph:
                confusion[t_idx, p_idx] += 1
                if t_idx == p_idx:
                    ph_correct[t_idx] += 1
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1  # insertion in pred
        else:
            # deletion — true phoneme has no prediction
            confusion[t_idx, t_idx] = max(0, confusion[t_idx, t_idx])
            i -= 1


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
        
        # OPTIMIZATION: Set random seeds for reproducibility
        self._set_seed(config.seed)

        self.ctc_loss = CTCLossWrapper(label_smoothing=config.label_smoothing)

        if config.freeze_visual_encoder:
            for param in model.visual_encoder.parameters():
                param.requires_grad = False
            model.visual_encoder.eval()
            LOGGER.info("Visual encoder frozen for fast fine-tuning.")

        # IMPROVED: Differential learning rates for different components.
        param_groups = []

        def add_param_group(module: nn.Module, lr: float, name: str) -> None:
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                param_groups.append({'params': params, 'lr': lr, 'name': name})

        if not config.freeze_visual_encoder:
            add_param_group(model.visual_encoder, config.learning_rate * 0.3, 'visual_encoder')
        add_param_group(model.temporal_adapter, config.learning_rate * 0.5, 'temporal_adapter')
        add_param_group(model.phoneme_head, config.learning_rate, 'phoneme_head')
        
        # Add temporal attention if it exists
        if hasattr(model, 'temporal_attention') and model.temporal_attention is not None:
            add_param_group(model.temporal_attention, config.learning_rate * 0.5, 'temporal_attention')
        
        # Add temporal multiscale if it exists
        if hasattr(model, 'temporal_multiscale') and model.temporal_multiscale is not None:
            add_param_group(model.temporal_multiscale, config.learning_rate * 0.5, 'temporal_multiscale')
        if hasattr(model, 'feature_norm') and model.feature_norm is not None:
            add_param_group(model.feature_norm, config.learning_rate * 0.7, 'feature_norm')

        if not param_groups:
            raise RuntimeError("No trainable parameters selected. Check freeze/training configuration.")

        self.optimizer = AdamW(
            param_groups,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),  # IMPROVED: Better betas for transformers
            eps=1e-6
        )

        # Use constant LR scheduler (no warmup/decay) to avoid NaN issues
        from torch.optim.lr_scheduler import LambdaLR
        self.scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0)  # Constant LR

        self.use_amp = False  # DISABLED AMP - causes NaN
        self.scaler = None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.checkpoint_dir / "best"
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.checkpoint_dir / "logs")

        self.artifact_root = get_artifact_root("training")
        self.loss_artifacts = get_useful_dir("training", "loss_curves")
        self.preview_artifacts = get_useful_dir("training", "batch_previews")

        self.step_metrics: List[Tuple[int, float, float, float, float, float]] = []
        self.epoch_metrics: List[Tuple[int, float, float, float, float, float]] = []
        self.val_losses: List[Tuple[int, float]] = []
        
        # MEMORY LEAK FIX: Limit list sizes to prevent unbounded growth
        self.max_step_metrics = 10000  # Keep only last 10k steps
        self.max_epoch_metrics = 1000  # Keep only last 1000 epochs
        self.max_val_losses = 1000     # Keep only last 1000 validations
        
        self._preview_epochs = set()
        self._logit_epochs = set()
        self._prediction_epochs = set()

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_cer  = float('inf')
        self.best_metrics_history: List[Dict] = []  # one entry per time best is beaten
        
        # OPTIMIZATION: Early stopping tracking
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        # OPTIMIZATION: Training time tracking
        self.training_start_time = None
        self.epoch_times: List[float] = []

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
                "attention_entropy_weight": self.config.attention_entropy_weight,
                "seed": self.config.seed,
                "early_stopping_patience": self.config.early_stopping_patience,
                "time_budget_hours": self.config.time_budget_hours,
                "label_smoothing": self.config.label_smoothing,
                "freeze_visual_encoder": self.config.freeze_visual_encoder,
                "save_last_every": self.config.save_last_every,
            }
            save_json(self.artifact_root / "training_config.json", summary)
            save_text(self.artifact_root / "notes.txt", "Training artifacts and metrics.")
        except Exception:
            log_exception(LOGGER, "Failed to save training metadata.")
    
    def _cleanup_memory(self) -> None:
        """Force garbage collection and clear GPU cache to prevent memory leaks."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.debug("Memory cleanup: garbage collected and GPU cache cleared")
    
    def _trim_metrics_lists(self) -> None:
        """Trim metrics lists to prevent unbounded memory growth."""
        if len(self.step_metrics) > self.max_step_metrics:
            self.step_metrics = self.step_metrics[-self.max_step_metrics:]
            LOGGER.debug("Trimmed step_metrics to %d entries", len(self.step_metrics))
        
        if len(self.epoch_metrics) > self.max_epoch_metrics:
            self.epoch_metrics = self.epoch_metrics[-self.max_epoch_metrics:]
            LOGGER.debug("Trimmed epoch_metrics to %d entries", len(self.epoch_metrics))
        
        if len(self.val_losses) > self.max_val_losses:
            self.val_losses = self.val_losses[-self.max_val_losses:]
            LOGGER.debug("Trimmed val_losses to %d entries", len(self.val_losses))
    
    def _flush_tensorboard(self) -> None:
        """Flush TensorBoard writer to disk to prevent memory accumulation."""
        try:
            self.writer.flush()
            LOGGER.debug("TensorBoard writer flushed")
        except Exception:
            log_exception(LOGGER, "Failed to flush TensorBoard writer")
    
    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"[Trainer] Deterministic mode enabled (slower but reproducible)")
            LOGGER.info("Deterministic mode enabled for full reproducibility")
        else:
            torch.backends.cudnn.benchmark = True
            LOGGER.info("CuDNN benchmark mode enabled for performance")
        
        print(f"[Trainer] Random seed set to {seed}")
        LOGGER.info("Random seed set to %d (deterministic=%s)", seed, self.config.deterministic)
    
    def _estimate_time_remaining(self) -> str:
        """Estimate remaining training time based on epoch times."""
        if not self.epoch_times:
            return "Unknown"
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.config.epochs - self.current_epoch
        remaining_seconds = avg_epoch_time * remaining_epochs
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        LOGGER.debug("Time estimation: avg_epoch_time=%.1fs, remaining_epochs=%d, ETA=%dh %dm",
                    avg_epoch_time, remaining_epochs, hours, minutes)
        
        return f"{hours}h {minutes}m"
    
    def _check_time_budget(self) -> bool:
        """Check if we're within time budget. Returns True if we should stop."""
        if self.config.time_budget_hours <= 0:
            return False
        
        if self.training_start_time is None:
            return False
        
        elapsed_hours = (time.time() - self.training_start_time) / 3600
        remaining_hours = self.config.time_budget_hours - elapsed_hours
        
        LOGGER.debug("Time budget check: elapsed=%.2fh, budget=%.2fh, remaining=%.2fh",
                    elapsed_hours, self.config.time_budget_hours, remaining_hours)
        
        if elapsed_hours >= self.config.time_budget_hours:
            LOGGER.warning("Time budget exceeded: elapsed=%.2fh >= budget=%.2fh",
                          elapsed_hours, self.config.time_budget_hours)
            return True
        
        # Warn if less than 1 hour remaining
        if remaining_hours < 1.0 and remaining_hours > 0:
            LOGGER.warning("Time budget warning: only %.1fh remaining", remaining_hours)
        
        return False
    
    def _should_validate_this_epoch(self, epoch: int) -> bool:
        """Determine if we should validate this epoch (adaptive validation)."""
        if not self.config.adaptive_validation:
            should_validate = (epoch + 1) % self.config.eval_every == 0
            LOGGER.debug("Validation check (non-adaptive): epoch=%d, should_validate=%s",
                        epoch + 1, should_validate)
            return should_validate
        
        # Validate less frequently early on, more frequently after warmup
        if epoch < self.config.warmup_epochs:
            should_validate = (epoch + 1) % self.config.adaptive_val_start_every == 0
            LOGGER.debug("Validation check (warmup phase): epoch=%d, should_validate=%s, interval=%d",
                        epoch + 1, should_validate, self.config.adaptive_val_start_every)
        else:
            should_validate = (epoch + 1) % self.config.adaptive_val_end_every == 0
            LOGGER.debug("Validation check (post-warmup): epoch=%d, should_validate=%s, interval=%d",
                        epoch + 1, should_validate, self.config.adaptive_val_end_every)
        
        return should_validate
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if we should stop early. Returns True if we should stop."""
        improvement = self.best_val_loss - val_loss
        
        if val_loss < (self.best_val_loss - self.config.early_stopping_min_delta):
            # Significant improvement - reset counter and update best epoch
            self.epochs_without_improvement = 0
            self.best_epoch = self.current_epoch + 1
            LOGGER.info("Early stopping: significant improvement detected (delta=%.4f, new_best=%.4f, best_epoch=%d)",
                       improvement, val_loss, self.best_epoch)
            return False
        else:
            # No significant improvement
            self.epochs_without_improvement += 1
            LOGGER.info("Early stopping: no significant improvement (delta=%.4f < min_delta=%.4f, patience=%d/%d, best_epoch=%d)",
                       improvement, self.config.early_stopping_min_delta,
                       self.epochs_without_improvement, self.config.early_stopping_patience,
                       self.best_epoch)
            
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {self.config.early_stopping_patience} epochs")
                print(f"[Early Stopping] Best epoch was {self.best_epoch} with val_loss={self.best_val_loss:.4f}")
                LOGGER.warning("Early stopping triggered: no improvement for %d epochs (best_epoch=%d, best_val_loss=%.4f)",
                              self.config.early_stopping_patience, self.best_epoch, self.best_val_loss)
                return True
            
            return False

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

    def _extract_training_features(self, video: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract features for training, optionally freezing the visual encoder."""
        if not self.config.freeze_visual_encoder:
            if self.config.attention_entropy_weight > 0:
                return self.model.extract_features_with_attention(video)
            return self.model.extract_features(video), []

        self.model.visual_encoder.eval()
        with torch.no_grad():
            features = self.model.visual_encoder(video)
        features = self.model.temporal_adapter(features)
        features = self.model.temporal_multiscale(features)
        if self.config.attention_entropy_weight > 0:
            features, attn_weights = self.model.temporal_attention(features, return_attn=True)
        else:
            features = self.model.temporal_attention(features)
            attn_weights = []
        features = self.model.feature_norm(features)
        return features, attn_weights

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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                # Log gradient norm to detect explosion early
                if self.config.log_gradient_norms and self.global_step % 10 == 0:
                    LOGGER.debug("Step %d gradient norm: %.4f", self.global_step, grad_norm.item())
                    self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Step scheduler BEFORE zero_grad (after optimizer.step)
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                # Log gradient norm to detect explosion early
                if self.config.log_gradient_norms and self.global_step % 10 == 0:
                    LOGGER.debug("Step %d gradient norm: %.4f", self.global_step, grad_norm.item())
                    self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
                
                self.optimizer.step()
                
                # Step scheduler BEFORE zero_grad (after optimizer.step)
                self.scheduler.step()
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

        # CRITICAL FIX: Don't use tqdm with num_workers=0, it can cause issues
        # Use manual progress updates instead
        use_tqdm = self.train_loader.num_workers > 0
        
        # SMOKE TEST: Limit to 10 batches
        max_batches = 10 if self.config.smoke_test else len(self.train_loader)
        
        if self.config.smoke_test:
            LOGGER.info("Smoke test mode: limiting to %d batches", max_batches)
        
        if use_tqdm:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
            iterator = enumerate(pbar)
        else:
            # Manual progress for single-threaded loading
            print(f"Epoch {self.current_epoch+1}/{self.config.epochs}: ", end='', flush=True)
            iterator = enumerate(self.train_loader)

        for batch_idx, batch in iterator:
            if batch_idx >= max_batches:
                LOGGER.debug("Reached max_batches limit (%d) for smoke test", max_batches)
                break
            
            # Manual progress update for single-threaded
            if not use_tqdm and batch_idx % max(1, self.config.progress_every) == 0:
                print(f"{batch_idx}/{max_batches}...", end='', flush=True)
                
            video = batch['video'].to(self.device)
            texts = batch['text']

            if batch_idx == 0:
                self._save_batch_preview(video, self.current_epoch)

            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    features, attn_weights = self._extract_training_features(video)
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
                features, attn_weights = self._extract_training_features(video)
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

            # NaN detection - stop training immediately if NaN detected
            if self.config.detect_nan and (torch.isnan(loss) or torch.isinf(loss)):
                LOGGER.error("NaN or Inf loss detected at epoch %d, batch %d! Stopping training.", 
                           self.current_epoch + 1, batch_idx)
                LOGGER.error("Loss components: ctc=%.4f, decor=%.4f, smooth=%.4f, attn=%.4f",
                           ctc_loss.item(), decor_loss.item(), smooth_loss.item(), attn_loss.item())
                raise RuntimeError(f"NaN/Inf loss detected at epoch {self.current_epoch + 1}, batch {batch_idx}")

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
            # MEMORY LEAK FIX: Properly detach tensors before storing
            self.step_metrics.append(
                (
                    self.global_step,
                    float(loss.detach().cpu().item()),
                    float(ctc_loss.detach().cpu().item()),
                    float(decor_loss.detach().cpu().item()),
                    float(smooth_loss.detach().cpu().item()),
                    float(attn_loss.detach().cpu().item())
                )
            )

            if use_tqdm:
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
            
            # MEMORY LEAK FIX: Explicitly delete tensors to free memory
            del video, logits, features, loss, ctc_loss, decor_loss, smooth_loss, attn_loss
            if 'attn_weights' in locals():
                del attn_weights
        
        # Close manual progress line
        if not use_tqdm:
            avg_loss = total_loss / max(num_batches, 1)
            print(f" Done! Avg loss: {avg_loss:.4f}")

        # MEMORY LEAK FIX: Cleanup at end of epoch
        self._trim_metrics_lists()
        self._flush_tensorboard()
        self._cleanup_memory()
        
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Run validation. Returns (val_loss, val_cer).
        Also stores per-sample predictions in self._last_val_results for figure generation."""
        LOGGER.info("Starting validation for epoch %d", self.current_epoch + 1)
        self.model.eval()

        total_loss  = 0.0
        total_cer   = 0.0
        total_wer   = 0.0
        num_batches = 0
        num_samples = 0

        # Phoneme-level confusion: true_phoneme -> {pred_phoneme: count}
        phoneme_vocab = get_phoneme_vocab()
        ph_to_idx = {p: i for i, p in enumerate(phoneme_vocab)}
        n_ph = len(phoneme_vocab)
        confusion = np.zeros((n_ph, n_ph), dtype=np.int64)

        # Per-phoneme correct / total
        ph_correct = np.zeros(n_ph, dtype=np.int64)
        ph_total   = np.zeros(n_ph, dtype=np.int64)

        # Store sample-level results for qualitative table
        # MEMORY LEAK FIX: Limit number of samples stored
        sample_results: List[Dict] = []
        max_samples_to_store = 100  # Only store first 100 samples

        use_tqdm = self.val_loader.num_workers > 0
        if use_tqdm:
            iterator = enumerate(tqdm(self.val_loader, desc="Validation"))
        else:
            print("Validation: ", end="", flush=True)
            iterator = enumerate(self.val_loader)

        for batch_idx, batch in iterator:
            if not use_tqdm and batch_idx % 10 == 0:
                print(f"{batch_idx}/{len(self.val_loader)}...", end="", flush=True)

            video = batch['video'].to(self.device)
            texts = batch['text']

            logits = self.model(video)
            loss   = self.ctc_loss(logits, texts)
            total_loss  += loss.item()
            num_batches += 1

            try:
                preds = self.model.decode_ctc(logits)
                for pred_phonemes, target_text in zip(preds, texts):
                    pred_str    = ' '.join(pred_phonemes).lower().strip()
                    tgt_phonemes = _target_phoneme_sequence(target_text, phoneme_vocab)
                    target_str  = ' '.join(tgt_phonemes).lower().strip()

                    # CER is now measured on the phoneme string. WER is a
                    # phoneme-token error rate proxy, not English word WER.
                    cer = _char_error_rate(pred_str, target_str) if target_str else 0.0
                    total_cer  += cer
                    num_samples += 1

                    wer = _word_error_rate(pred_str, target_str)
                    total_wer += wer

                    # Phoneme confusion: align predicted phonemes vs target phonemes
                    _update_confusion(confusion, ph_correct, ph_total,
                                      tgt_phonemes, pred_phonemes, ph_to_idx)

                    # MEMORY LEAK FIX: Only store limited number of samples
                    if len(sample_results) < max_samples_to_store:
                        sample_results.append({
                            "target":    target_text,
                            "target_phonemes": target_str,
                            "predicted": pred_str,
                            "cer":       round(cer, 4),
                            "wer":       round(wer, 4),
                        })
            except Exception as e:
                LOGGER.warning("Error during validation decoding: %s", str(e))
                pass

            del video, logits, loss

        if not use_tqdm:
            print("Done!", flush=True)

        avg_loss = total_loss / max(num_batches, 1)
        avg_cer  = total_cer  / max(num_samples, 1)
        avg_wer  = total_wer  / max(num_samples, 1)

        self.writer.add_scalar('val/loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('val/cer',  avg_cer,  self.current_epoch)
        self.writer.add_scalar('val/wer',  avg_wer,  self.current_epoch)
        
        LOGGER.info("Validation complete: epoch=%d, batches=%d, samples=%d, loss=%.4f, cer=%.4f, wer=%.4f",
                    self.current_epoch + 1, num_batches, num_samples, avg_loss, avg_cer, avg_wer)
        
        self.val_losses.append((self.current_epoch + 1, avg_loss))

        # Stash for figure generation
        self._last_val_results = {
            "val_loss":      avg_loss,
            "val_cer":       avg_cer,
            "val_wer":       avg_wer,
            "confusion":     confusion,
            "ph_correct":    ph_correct,
            "ph_total":      ph_total,
            "phoneme_vocab": phoneme_vocab,
            "samples":       sample_results,
        }
        
        LOGGER.debug("Validation results stored for figure generation: %d sample predictions", len(sample_results))

        # MEMORY LEAK FIX: Cleanup after validation
        self._cleanup_memory()
        
        return avg_loss, avg_cer

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save_best(self, val_loss: float, val_cer: float,
                   train_loss: float, epoch: int) -> None:
        """
        Save the best model and its metrics into  <checkpoint_dir>/best/

        Files written every time a new best is found:
          best/best_model.pt          – full checkpoint (weights + optimizer + scheduler)
          best/metrics.json           – metrics for this best epoch
          best/metrics_history.json   – all epochs that ever beat the previous best
          best/figures/               – all paper-quality plots and confusion matrix
        """
        LOGGER.info("Saving new best model: epoch=%d, val_loss=%.4f, val_cer=%.4f, train_loss=%.4f",
                   epoch, val_loss, val_cer, train_loss)
        
        self.save_checkpoint_to(self.best_dir / "best_model.pt")

        metrics = {
            "epoch":      epoch,
            "val_loss":   round(val_loss,  6),
            "val_cer":    round(val_cer,   6),
            "val_wer":    round(getattr(self, '_last_val_results', {}).get('val_wer', 0.0), 6),
            "train_loss": round(train_loss, 6),
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.best_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        LOGGER.debug("Saved best model metrics: %s", metrics)

        self.best_metrics_history.append(metrics)
        with open(self.best_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(self.best_metrics_history, f, indent=2)
        
        LOGGER.debug("Updated best metrics history: %d entries", len(self.best_metrics_history))

        # Generate all paper-quality figures
        LOGGER.info("Generating publication-ready figures for best model")
        self._generate_best_figures(epoch)

        print(f"[Best]  epoch={epoch}  val_loss={val_loss:.4f}  "
              f"val_cer={val_cer:.4f}  → saved to {self.best_dir}")
        LOGGER.info("Best model saved successfully to %s", self.best_dir)

    def _generate_best_figures(self, epoch: int) -> None:
        """
        Generate and save all paper-quality figures into best/figures/.

        Figures produced:
          01_loss_curves.png          – train loss + val loss over all epochs
          02_cer_curve.png            – val CER over epochs
          03_wer_curve.png            – val WER over epochs
          04_cer_wer_combined.png     – CER + WER on one axes (for paper)
          05_loss_components.png      – CTC / decorr / smooth / attn losses
          06_confusion_matrix.png     – phoneme confusion matrix (normalised)
          07_confusion_matrix_raw.png – raw counts version
          08_per_phoneme_accuracy.png – per-phoneme accuracy bar chart
          09_top_confusions.png       – top-20 most confused phoneme pairs
          10_sample_predictions.txt   – qualitative prediction examples
          11_training_summary.json    – all numeric results in one file
        """
        LOGGER.info("Generating publication-ready figures for epoch %d", epoch)
        
        try:
            import matplotlib
            matplotlib.use("Agg")          # no display needed
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            LOGGER.error("matplotlib not available — cannot generate figures")
            print("[Warning] matplotlib not available — skipping figure generation.")
            return

        fig_dir = self.best_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Figure directory: %s", fig_dir)

        epochs_x = [m[0] for m in self.epoch_metrics]
        train_losses = [m[1] for m in self.epoch_metrics]
        val_epochs   = [ep for ep, _ in self.val_losses]
        val_loss_y   = [l  for _, l  in self.val_losses]
        val_cer_y    = [m["val_cer"] for m in self.best_metrics_history]
        val_wer_y    = [m.get("val_wer", 0.0) for m in self.best_metrics_history]
        best_epochs  = [m["epoch"] for m in self.best_metrics_history]

        # ── 01  Train + Val loss ──────────────────────────────────────────────
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs_x, train_losses, label="Train Loss", linewidth=2)
            if val_loss_y:
                ax.plot(val_epochs, val_loss_y, label="Val Loss",
                        linewidth=2, linestyle="--", marker="o", markersize=4)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / "01_loss_curves.png", dpi=150)
            plt.close(fig)
        except Exception:
            log_exception(LOGGER, "Failed to save loss curves figure.")

        # ── 02  Val CER curve ─────────────────────────────────────────────────
        try:
            if val_cer_y:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(best_epochs, val_cer_y, color="tab:orange",
                        linewidth=2, marker="o", markersize=5)
                ax.set_xlabel("Epoch"); ax.set_ylabel("CER")
                ax.set_title("Validation Character Error Rate (CER)")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_dir / "02_cer_curve.png", dpi=150)
                plt.close(fig)
        except Exception:
            log_exception(LOGGER, "Failed to save CER curve.")

        # ── 03  Val WER curve ─────────────────────────────────────────────────
        try:
            if val_wer_y:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(best_epochs, val_wer_y, color="tab:red",
                        linewidth=2, marker="s", markersize=5)
                ax.set_xlabel("Epoch"); ax.set_ylabel("WER")
                ax.set_title("Validation Word Error Rate (WER)")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_dir / "03_wer_curve.png", dpi=150)
                plt.close(fig)
        except Exception:
            log_exception(LOGGER, "Failed to save WER curve.")

        # ── 04  CER + WER combined (paper-ready) ──────────────────────────────
        try:
            if val_cer_y and val_wer_y:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(best_epochs, val_cer_y, label="CER", color="tab:orange",
                        linewidth=2, marker="o", markersize=5)
                ax.plot(best_epochs, val_wer_y, label="WER", color="tab:red",
                        linewidth=2, marker="s", markersize=5, linestyle="--")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Error Rate")
                ax.set_title("Validation CER and WER")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_dir / "04_cer_wer_combined.png", dpi=150)
                plt.close(fig)
        except Exception:
            log_exception(LOGGER, "Failed to save CER+WER figure.")

        # ── 05  Loss components ───────────────────────────────────────────────
        try:
            if self.epoch_metrics:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(epochs_x, [m[2] for m in self.epoch_metrics],
                        label="CTC",          linewidth=1.5)
                ax.plot(epochs_x, [m[3] for m in self.epoch_metrics],
                        label="Decorrelation", linewidth=1.5, linestyle="--")
                ax.plot(epochs_x, [m[4] for m in self.epoch_metrics],
                        label="Smoothness",    linewidth=1.5, linestyle="-.")
                ax.plot(epochs_x, [m[5] for m in self.epoch_metrics],
                        label="Attn Entropy",  linewidth=1.5, linestyle=":")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title("Loss Component Breakdown")
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_dir / "05_loss_components.png", dpi=150)
                plt.close(fig)
        except Exception:
            log_exception(LOGGER, "Failed to save loss components figure.")

        # ── 06 + 07  Phoneme confusion matrix ─────────────────────────────────
        vr = getattr(self, '_last_val_results', None)
        if vr is not None:
            confusion     = vr["confusion"]
            ph_correct    = vr["ph_correct"]
            ph_total      = vr["ph_total"]
            phoneme_vocab = vr["phoneme_vocab"]
            samples       = vr["samples"]

            # Only keep phonemes that actually appeared
            active = np.where(ph_total > 0)[0]
            if len(active) > 1:
                sub_conf  = confusion[np.ix_(active, active)]
                sub_vocab = [phoneme_vocab[i] for i in active]

                # Normalised (row = true, col = predicted)
                row_sums = sub_conf.sum(axis=1, keepdims=True).clip(min=1)
                norm_conf = sub_conf.astype(float) / row_sums

                for tag, matrix, fmt, cmap in [
                    ("06_confusion_matrix",     norm_conf, ".2f", "Blues"),
                    ("07_confusion_matrix_raw", sub_conf,  "d",   "Greens"),
                ]:
                    try:
                        n = len(sub_vocab)
                        cell = max(0.25, min(0.55, 12.0 / n))
                        fig, ax = plt.subplots(figsize=(n * cell + 2, n * cell + 1.5))
                        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                                       vmin=0, vmax=(1.0 if "norm" in tag else None))
                        ax.set_xticks(range(n)); ax.set_xticklabels(sub_vocab,
                            rotation=90, fontsize=max(4, min(8, 120 // n)))
                        ax.set_yticks(range(n)); ax.set_yticklabels(sub_vocab,
                            fontsize=max(4, min(8, 120 // n)))
                        ax.set_xlabel("Predicted Phoneme")
                        ax.set_ylabel("True Phoneme")
                        title = ("Phoneme Confusion Matrix (Normalised)"
                                 if "norm" in tag else
                                 "Phoneme Confusion Matrix (Raw Counts)")
                        ax.set_title(title)
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        fig.tight_layout()
                        fig.savefig(fig_dir / f"{tag}.png", dpi=150)
                        plt.close(fig)
                    except Exception:
                        log_exception(LOGGER, f"Failed to save {tag}.")

                # ── 08  Per-phoneme accuracy bar chart ────────────────────────
                try:
                    ph_acc = np.where(ph_total[active] > 0,
                                      ph_correct[active] / ph_total[active], 0.0)
                    order  = np.argsort(ph_acc)
                    sorted_vocab = [sub_vocab[i] for i in order]
                    sorted_acc   = ph_acc[order]

                    fig, ax = plt.subplots(figsize=(max(10, len(active) * 0.35), 5))
                    bars = ax.bar(range(len(sorted_vocab)), sorted_acc,
                                  color=plt.cm.RdYlGn(sorted_acc))
                    ax.set_xticks(range(len(sorted_vocab)))
                    ax.set_xticklabels(sorted_vocab, rotation=90,
                                       fontsize=max(5, min(9, 200 // len(sorted_vocab))))
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Per-Phoneme Recognition Accuracy")
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                    ax.axhline(ph_acc.mean(), color="navy", linestyle="--",
                               linewidth=1.5, label=f"Mean {ph_acc.mean():.1%}")
                    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
                    fig.tight_layout()
                    fig.savefig(fig_dir / "08_per_phoneme_accuracy.png", dpi=150)
                    plt.close(fig)
                except Exception:
                    log_exception(LOGGER, "Failed to save per-phoneme accuracy.")

                # ── 09  Top-20 most confused pairs ────────────────────────────
                try:
                    # Zero the diagonal (correct predictions)
                    off_diag = sub_conf.copy().astype(float)
                    np.fill_diagonal(off_diag, 0)
                    flat = off_diag.flatten()
                    top_k = min(20, int((flat > 0).sum()))
                    if top_k > 0:
                        top_idx = np.argsort(flat)[::-1][:top_k]
                        rows, cols = np.unravel_index(top_idx, off_diag.shape)
                        labels  = [f"{sub_vocab[r]}→{sub_vocab[c]}"
                                   for r, c in zip(rows, cols)]
                        counts  = [off_diag[r, c] for r, c in zip(rows, cols)]

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh(range(top_k), counts[::-1],
                                color="tab:red", alpha=0.75)
                        ax.set_yticks(range(top_k))
                        ax.set_yticklabels(labels[::-1], fontsize=9)
                        ax.set_xlabel("Count")
                        ax.set_title(f"Top {top_k} Most Confused Phoneme Pairs\n"
                                     "(true → predicted)")
                        ax.grid(True, alpha=0.3, axis="x")
                        fig.tight_layout()
                        fig.savefig(fig_dir / "09_top_confusions.png", dpi=150)
                        plt.close(fig)
                except Exception:
                    log_exception(LOGGER, "Failed to save top confusions.")

            # ── 10  Qualitative prediction samples ────────────────────────────
            try:
                lines = [f"Best model — epoch {epoch}",
                         f"Val CER: {vr['val_cer']:.4f}  "
                         f"Val WER: {vr['val_wer']:.4f}",
                         "=" * 60]
                for i, s in enumerate(samples[:20], 1):
                    lines += [
                        f"\n[{i}]",
                        f"  Target:    {s['target']}",
                        f"  Target PH: {s.get('target_phonemes', '')}",
                        f"  Predicted: {s['predicted']}",
                        f"  CER: {s['cer']:.4f}   WER: {s['wer']:.4f}",
                    ]
                (fig_dir / "10_sample_predictions.txt").write_text(
                    "\n".join(lines), encoding="utf-8")
            except Exception:
                log_exception(LOGGER, "Failed to save sample predictions.")

        # ── 11  Numeric summary JSON ──────────────────────────────────────────
        try:
            summary = {
                "best_epoch":          epoch,
                "val_loss":            round(self.best_val_loss, 6),
                "val_cer":             round(self.best_val_cer,  6),
                "val_wer":             round(vr["val_wer"] if vr else 0.0, 6),
                "train_loss_at_best":  round(train_loss if 'train_loss' in dir() else 0.0, 6),
                "total_epochs_so_far": epoch,
                "improvement_history": self.best_metrics_history,
            }
            with open(fig_dir / "11_training_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            log_exception(LOGGER, "Failed to save training summary JSON.")

        LOGGER.info("All publication-ready figures generated successfully: 11 files saved to %s", fig_dir)
        
        # MEMORY LEAK FIX: Clear validation results after figure generation
        self._last_val_results = None
        gc.collect()
        LOGGER.debug("Cleared validation results and collected garbage after figure generation")

    def _atomic_torch_save(self, checkpoint: Dict, path: Path) -> None:
        """Save a checkpoint through a per-process temp file, then replace.

        Reusing a fixed temp filename can race when an old/stale training process is
        still writing to the same checkpoint directory, and NTFS mounts are less
        forgiving about that pattern.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                LOGGER.warning("Could not remove temporary checkpoint file: %s", temp_path)

    def save_checkpoint_to(self, path: Path) -> None:
        """Save a full checkpoint to an explicit path."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_cer':  self.best_val_cer,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': vars(self.config),
            'model_config': asdict(self.model.config),
            'epoch_metrics': self.epoch_metrics,
            'step_metrics': self.step_metrics,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        self._atomic_torch_save(checkpoint, path)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint with all training state."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_cer': self.best_val_cer,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': vars(self.config),
            'model_config': asdict(self.model.config),
            'epoch_metrics': self.epoch_metrics,
            'step_metrics': self.step_metrics,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        
        self._atomic_torch_save(checkpoint, checkpoint_path)
        
        print(f"[Trainer] Saved checkpoint: {filename}")
        LOGGER.info("Checkpoint saved: %s (epoch=%d, global_step=%d, best_val_loss=%.4f)",
                   filename, self.current_epoch, self.global_step, self.best_val_loss)

    def load_checkpoint(self, checkpoint_path: str, weights_only: bool = False):
        """Load model checkpoint and restore all training state."""
        LOGGER.info("Loading checkpoint from: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if weights_only:
            self.current_epoch = 0
            self.global_step = 0
            loaded_best_loss = checkpoint.get('best_val_loss', float('inf'))
            loaded_best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', -1) + 1)
            self.best_val_loss = float('inf')
            self.best_val_cer = float('inf')
            self.best_epoch = 0
            self.epochs_without_improvement = 0
            print(f"[Trainer] Loaded model weights only from {checkpoint_path}")
            print(f"[Trainer] Previous source best was val_loss={loaded_best_loss:.4f} (epoch {loaded_best_epoch})")
            print("[Trainer] Reset optimizer and best tracking for the new fine-tune run")
            LOGGER.info(
                "Loaded checkpoint weights only: source=%s, source_best_val_loss=%.4f, source_best_epoch=%d",
                checkpoint_path, loaded_best_loss, loaded_best_epoch
            )
            return

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # resume from NEXT epoch
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_cer  = checkpoint.get('best_val_cer',  float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        self.epoch_metrics = checkpoint.get('epoch_metrics', [])
        self.step_metrics  = checkpoint.get('step_metrics', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.epoch_times = checkpoint.get('epoch_times', [])

        # Reload best metrics history from disk if it exists
        history_path = self.best_dir / "metrics_history.json"
        if history_path.exists():
            with open(history_path, encoding="utf-8") as f:
                self.best_metrics_history = json.load(f)
            LOGGER.info("Loaded best metrics history: %d entries", len(self.best_metrics_history))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            LOGGER.debug("Loaded gradient scaler state")

        print(f"[Trainer] ✅ Resumed from epoch {checkpoint['epoch']} "
              f"(continuing with epoch {self.current_epoch})")
        print(f"[Trainer] Best val_loss so far: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        print(f"[Trainer] Early stopping: {self.epochs_without_improvement}/{self.config.early_stopping_patience} epochs without improvement")
        
        LOGGER.info("Checkpoint loaded successfully: resuming from epoch %d (next: %d), global_step=%d, best_val_loss=%.4f",
                    checkpoint['epoch'], self.current_epoch, self.global_step, self.best_val_loss)

    def _describe_curriculum_phase(self, epoch: int) -> str:
        """Return the curriculum phase string for a given epoch."""
        if epoch < self.config.curriculum_phases[0]:
            return "Single Words"
        if epoch < self.config.curriculum_phases[1]:
            return "Short Sentences"
        return "Full Dataset"

    def train(self):
        """Full training loop with optimizations for 20-hour deadline."""
        print(f"\n[Trainer] Starting training for {self.config.epochs} epochs")
        print(f"[Trainer] Batch size: {self.config.batch_size} (effective: {self.config.batch_size * self.config.accumulation_steps})")
        print(f"[Trainer] Learning rate: {self.config.learning_rate}")
        print(f"[Trainer] Curriculum phases: {self.config.curriculum_phases}")
        print(f"[Trainer] Early stopping patience: {self.config.early_stopping_patience}")
        if self.config.time_budget_hours > 0:
            print(f"[Trainer] Time budget: {self.config.time_budget_hours:.1f} hours")
        if self.config.smoke_test:
            print(f"[Trainer] SMOKE TEST MODE: Running 2 epochs with limited batches")
        
        LOGGER.info(
            "Training started: epochs=%d, batch_size=%d, effective_batch=%d, lr=%.2e, curriculum=%s",
            self.config.epochs, self.config.batch_size, 
            self.config.batch_size * self.config.accumulation_steps,
            self.config.learning_rate, self.config.curriculum_phases
        )
        LOGGER.info(
            "Training optimizations: early_stopping_patience=%d, time_budget=%.1fh, adaptive_validation=%s, smoke_test=%s",
            self.config.early_stopping_patience, self.config.time_budget_hours,
            self.config.adaptive_validation, self.config.smoke_test
        )
        
        self.training_start_time = time.time()
        
        # Estimate total training time
        print(f"\n[Trainer] Dataset: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
        estimated_time_per_epoch = (len(self.train_loader) * 0.5)  # Rough estimate: 0.5s per batch
        estimated_total_hours = (estimated_time_per_epoch * self.config.epochs) / 3600
        print(f"[Trainer] Estimated total training time: {estimated_total_hours:.1f} hours")
        LOGGER.info("Dataset info: train_batches=%d, val_batches=%d, estimated_time=%.1fh",
                   len(self.train_loader), len(self.val_loader), estimated_total_hours)
        
        if self.config.time_budget_hours > 0 and estimated_total_hours > self.config.time_budget_hours:
            print(f"[Trainer] WARNING: Estimated time exceeds budget! Consider reducing epochs.")
            LOGGER.warning("Estimated training time (%.1fh) exceeds time budget (%.1fh)",
                          estimated_total_hours, self.config.time_budget_hours)

        for epoch in tqdm(range(self.current_epoch, self.config.epochs), 
                         desc="Overall Training Progress", 
                         initial=self.current_epoch,
                         total=self.config.epochs,
                         position=0,
                         leave=True,
                         ncols=100,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]'):
            epoch_start_time = time.time()
            self.current_epoch = epoch

            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            phase = self._describe_curriculum_phase(epoch)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Curriculum Phase: {phase}")
            if self.epoch_times:
                eta = self._estimate_time_remaining()
                elapsed_hours = (time.time() - self.training_start_time) / 3600
                print(f"Elapsed: {elapsed_hours:.1f}h | ETA: {eta}")
                LOGGER.info("Epoch %d/%d starting: phase=%s, elapsed=%.1fh, eta=%s",
                           epoch + 1, self.config.epochs, phase, elapsed_hours, eta)
            else:
                LOGGER.info("Epoch %d/%d starting: phase=%s", epoch + 1, self.config.epochs, phase)
            print(f"{'='*60}")

            # SMOKE TEST: Limit batches
            if self.config.smoke_test and epoch >= 2:
                print("[Smoke Test] Completed 2 epochs - stopping")
                LOGGER.info("Smoke test completed successfully after 2 epochs")
                break

            train_loss = self.train_epoch()
            
            # Print detailed epoch summary
            print(f"\n{'='*60}")
            print(f"📊 EPOCH {epoch + 1}/{self.config.epochs} SUMMARY")
            print(f"{'='*60}")
            print(f"  Train Loss: {train_loss:.4f}")
            
            LOGGER.info("Epoch %d train loss: %.4f", epoch + 1, train_loss)
            
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
                LOGGER.debug("Epoch %d loss components: ctc=%.4f, decor=%.4f, smooth=%.4f, attn=%.4f",
                            epoch + 1, avg_ctc, avg_decor, avg_smooth, avg_attn)
            self.epoch_metrics.append((epoch + 1, train_loss, avg_ctc, avg_decor, avg_smooth, avg_attn))
            self._save_loss_artifacts()

            # NOTE: scheduler.step() is now called after each optimizer.step() in _run_backward()
            # This is correct for OneCycleLR which needs per-batch updates
            
            # Track epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Calculate ETA
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.config.epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = eta_seconds / 3600
            
            print(f"  Epoch Time: {epoch_time/60:.1f} min")
            print(f"  Avg Time/Epoch: {avg_epoch_time/60:.1f} min")
            print(f"  ETA for {remaining_epochs} epochs: {eta_hours:.1f}h")
            
            LOGGER.info("Epoch %d completed in %.1f minutes", epoch + 1, epoch_time/60)

            # ADAPTIVE VALIDATION
            if self._should_validate_this_epoch(epoch):
                LOGGER.info("Running validation for epoch %d", epoch + 1)
                val_loss, val_cer = self.validate()
                
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Phoneme CER: {val_cer:.2%} (lower is better)")
                
                LOGGER.info("Epoch %d validation: val_loss=%.4f, val_cer=%.4f",
                            epoch + 1, val_loss, val_cer)
                self._save_loss_artifacts()

                val_improved = val_loss < self.best_val_loss
                if val_loss < self.best_val_loss:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    self.best_val_cer  = val_cer
                    self.best_epoch = epoch + 1  # Update best_epoch here too!
                    self.epochs_without_improvement = 0
                    
                    print(f"  🎉 NEW BEST MODEL! Improvement: {improvement:.4f}")
                    print(f"  Best Val Loss: {val_loss:.4f}")
                    print(f"  Best Val Phoneme CER: {val_cer:.2%}")
                    print(f"  Best Epoch: {self.best_epoch}")
                    
                    LOGGER.info("New best model: val_loss=%.4f (improvement=%.4f), val_cer=%.4f, best_epoch=%d",
                               val_loss, improvement, val_cer, self.best_epoch)
                    self._save_best(val_loss, val_cer, train_loss, epoch + 1)
                else:
                    print(f"  No improvement (best: {self.best_val_loss:.4f} at epoch {self.best_epoch})")
                    LOGGER.info("No improvement: current_val_loss=%.4f >= best_val_loss=%.4f (best_epoch=%d)",
                               val_loss, self.best_val_loss, self.best_epoch)
                
                print(f"{'='*60}\n")
                
                # EARLY STOPPING CHECK
                if (not val_improved) and self._check_early_stopping(val_loss):
                    print(f"\n[Trainer] Early stopping triggered - training complete!")
                    LOGGER.info("Training stopped early at epoch %d", epoch + 1)
                    break
            else:
                print(f"  (Validation skipped this epoch)")
                print(f"{'='*60}\n")
                LOGGER.debug("Skipping validation for epoch %d", epoch + 1)

            if (epoch + 1) % self.config.save_every == 0:
                print(f"[Trainer] 💾 Saving milestone checkpoint (epoch {epoch+1})")
                LOGGER.info("Saving milestone checkpoint at epoch %d", epoch + 1)
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            should_save_last = (
                (epoch + 1) % max(1, self.config.save_last_every) == 0
                or (epoch + 1) == self.config.epochs
            )
            if should_save_last:
                self.save_checkpoint("last_epoch.pt")
                LOGGER.debug("Saved last_epoch.pt checkpoint")
            
            # TIME BUDGET CHECK
            if self._check_time_budget():
                elapsed = (time.time() - self.training_start_time) / 3600
                print(f"\n[Trainer] Time budget of {self.config.time_budget_hours:.1f}h reached (elapsed: {elapsed:.1f}h)")
                print(f"[Trainer] Stopping training early to meet deadline")
                LOGGER.warning("Time budget reached: elapsed=%.1fh >= budget=%.1fh, stopping training",
                              elapsed, self.config.time_budget_hours)
                break

        self.save_checkpoint("final_model.pt")
        LOGGER.info("Saved final model checkpoint")
        
        total_time = (time.time() - self.training_start_time) / 3600
        print(f"\n[Trainer] Training complete!")
        print(f"[Trainer] Total time: {total_time:.2f} hours")
        print(f"[Trainer] Best epoch: {self.best_epoch} (val_loss={self.best_val_loss:.4f}, val_cer={self.best_val_cer:.4f})")
        LOGGER.info("Training complete: total_time=%.2fh, total_epochs=%d, best_epoch=%d, best_val_loss=%.4f, best_val_cer=%.4f",
                   total_time, self.current_epoch + 1, self.best_epoch, self.best_val_loss, self.best_val_cer)

        self.writer.close()
        LOGGER.info("TensorBoard writer closed")


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
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs (default: 300, early stopping recommended)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='AdamW weight decay (default: 0.01)')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Optional CTC label smoothing weight (default: 0.0)')

    parser.add_argument('--load_refiner', action='store_true',
                        help='Load Qwen2 refiner for joint training')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--load_weights_only', action='store_true',
                        help='Load only model weights from --resume and reset optimizer/epoch counters')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save milestone checkpoint every N epochs (default: 5)')
    parser.add_argument('--save_last_every', type=int, default=3,
                        help='Save resumable last_epoch.pt every N epochs (default: 3)')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Validate every N epochs when adaptive validation is disabled (default: 1)')
    parser.add_argument('--progress_every', type=int, default=100,
                        help='Print single-worker batch progress every N batches (default: 100)')

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
    parser.add_argument('--disable_curriculum', action='store_true',
                        help='Use the full dataset from epoch 1 instead of single-word/short-sentence curriculum')
    
    # OPTIMIZATION: New arguments for 20-hour deadline
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                        help='Stop training if no improvement for N epochs (default: 100, recommended for 300 epochs)')
    parser.add_argument('--time_budget_hours', type=float, default=0.0,
                        help='Maximum training time in hours (0 = no limit, recommend 19.5 for 20h deadline)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode (slower but fully reproducible)')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run quick smoke test (2 epochs, 10 batches each) to verify setup')
    parser.add_argument('--adaptive_validation', action='store_true', default=True,
                        help='Validate less frequently early on (default: True)')
    parser.add_argument('--no_adaptive_validation', '--no-adaptive-validation',
                        action='store_false', dest='adaptive_validation',
                        help='Use fixed --eval_every validation schedule')
    parser.add_argument('--freeze_visual_encoder', action='store_true',
                        help='Freeze the visual encoder and fine-tune temporal/head layers only')

    return parser.parse_args()


def _build_training_config(args) -> TrainingConfig:
    """Create TrainingConfig from CLI args."""
    config = TrainingConfig()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.accumulation_steps = args.accumulation_steps
    config.label_smoothing = args.label_smoothing
    config.checkpoint_dir = args.checkpoint_dir
    config.save_every = args.save_every
    config.save_last_every = args.save_last_every
    config.eval_every = args.eval_every
    config.progress_every = args.progress_every
    config.decorrelation_weight = args.decorrelation_weight
    config.decorrelation_anchor = args.decorrelation_anchor
    config.temporal_smoothness_weight = args.temporal_smoothness_weight
    config.attention_entropy_weight = args.attention_entropy_weight
    
    # OPTIMIZATION: New config options
    config.early_stopping_patience = args.early_stopping_patience
    config.time_budget_hours = args.time_budget_hours
    config.seed = args.seed
    config.deterministic = args.deterministic
    config.smoke_test = args.smoke_test
    config.adaptive_validation = args.adaptive_validation
    config.freeze_visual_encoder = args.freeze_visual_encoder
    
    if args.disable_curriculum:
        config.curriculum_phases = (0, 0)
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


def _ensure_split(data_dir: str, train_ratio: float = 0.80,
                  val_ratio: float = 0.10, seed: int = 42) -> None:
    """
    Create split.json in data_dir if it doesn't already exist.

    Split is done at the SOURCE VIDEO level (using source_video_id from
    metadata.json) so no speaker appears in more than one split.
    Ratios: 80% train / 10% val / 10% test.
    """
    import random as _random
    from collections import defaultdict as _defaultdict

    split_path = Path(data_dir) / "split.json"
    if split_path.exists():
        LOGGER.info("split.json already exists — skipping split generation.")
        return

    metadata_path = Path(data_dir) / "metadata.json"
    if not metadata_path.exists():
        LOGGER.warning("metadata.json not found — cannot create split, using all data.")
        return

    with open(metadata_path, encoding="utf-8") as f:
        samples = json.load(f)

    # Group clip IDs by source video
    video_to_clips: dict = _defaultdict(list)
    for s in samples:
        src = s.get("source_video_id") or s["id"].rsplit("_clip_", 1)[0]
        video_to_clips[src].append(s["id"])

    source_videos = sorted(video_to_clips.keys())
    rng = _random.Random(seed)
    rng.shuffle(source_videos)

    n = len(source_videos)
    n_val   = max(1, round(n * val_ratio))
    n_test  = max(1, round(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    train_vids = set(source_videos[:n_train])
    val_vids   = set(source_videos[n_train : n_train + n_val])
    # everything else → test

    clip_split: dict = {}
    for src, clips in video_to_clips.items():
        tag = "train" if src in train_vids else ("val" if src in val_vids else "test")
        for clip_id in clips:
            clip_split[clip_id] = tag

    counts = {t: sum(1 for v in clip_split.values() if v == t)
              for t in ("train", "val", "test")}

    split_data = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio,
                   "test": round(1.0 - train_ratio - val_ratio, 4)},
        "source_video_counts": {"train": n_train, "val": n_val,
                                "test": n - n_train - n_val},
        "clip_counts": counts,
        "clips": clip_split,
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    print(f"[Split] Created video-level split → {split_path}")
    print(f"[Split] Source videos — train: {n_train}  val: {n_val}  "
          f"test: {n - n_train - n_val}")
    print(f"[Split] Clips         — train: {counts['train']}  "
          f"val: {counts['val']}  test: {counts['test']}")
    LOGGER.info("Split created: %s", counts)


def main():
    """Main training function."""
    args = parse_args()
    log_system_info(LOGGER)
    LOGGER.info("Training arguments: %s", vars(args))

    print_device_info()
    device = get_device()
    LOGGER.info("Using device: %s", device)

    config = _build_training_config(args)
    LOGGER.info("Training configuration: batch_size=%d, epochs=%d, lr=%.2e, early_stopping=%d, time_budget=%.1fh, seed=%d",
               config.batch_size, config.epochs, config.learning_rate, 
               config.early_stopping_patience, config.time_budget_hours, config.seed)

    print("\n[Main] Creating model...")
    model_config = _build_model_config(args)
    model = create_model(load_refiner=args.load_refiner, config=model_config)
    LOGGER.info("Model created: load_refiner=%s, temporal_multiscale=%s, temporal_attention_layers=%d",
               args.load_refiner, model_config.temporal_multiscale, model_config.temporal_attention_layers)

    if args.load_refiner:
        LOGGER.info("Setting up LoRA for refiner")
        setup_lora(model, config)

    print("\n[Main] Creating dataloaders...")
    data_config = DataConfig()
    if args.disable_curriculum:
        data_config.curriculum_phases = (0, 0)
        LOGGER.info("Curriculum learning disabled")
    else:
        LOGGER.info("Curriculum learning enabled: phases=%s", data_config.curriculum_phases)

    # ── Ensure a video-level train/val/test split exists ─────────────────────
    LOGGER.info("Ensuring train/val/test split exists")
    _ensure_split(args.data_dir)

    train_loader = create_dataloader(
        args.data_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        epoch=0,
        training=True,
        config=data_config,
        split="train",
    )

    val_loader = create_dataloader(
        args.data_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        epoch=0,
        training=False,
        config=data_config,
        split="val",
    )

    LOGGER.info("Dataloaders created: train_batches=%d, val_batches=%d, num_workers=%d",
               len(train_loader), len(val_loader), args.num_workers)
    
    if len(train_loader) == 0:
        LOGGER.error("No training samples found in %s", args.data_dir)
        raise RuntimeError("No training samples found. Check data_dir and preprocessing outputs.")
    if len(val_loader) == 0:
        LOGGER.warning("Validation loader is empty; validation will be skipped.")
    
    # CRITICAL FIX: Pre-flight check - test loading first batch
    print("[Main] Testing data loading with first batch...")
    LOGGER.info("Pre-flight check: testing first batch load")
    try:
        import time
        start_time = time.time()
        first_batch = next(iter(train_loader))
        load_time = time.time() - start_time
        print(f"[Main] ✅ First batch loaded successfully in {load_time:.2f}s")
        print(f"[Main] Batch shape: video={first_batch['video'].shape}, texts={len(first_batch['text'])}")
        LOGGER.info("Pre-flight check passed: first batch loaded in %.2fs", load_time)
    except Exception as e:
        print(f"[Main] ❌ CRITICAL: Failed to load first batch!")
        print(f"[Main] Error: {e}")
        LOGGER.error("Pre-flight check FAILED: %s", e)
        log_exception(LOGGER, "Failed to load first training batch")
        raise RuntimeError(f"Data loading is broken: {e}")

    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Auto-resume: if --resume is given use that path; otherwise check for
    # last_epoch.pt in the checkpoint directory so a crash is transparent.
    resume_path = args.resume
    if not resume_path:
        auto = Path(config.checkpoint_dir) / "last_epoch.pt"
        if auto.exists():
            resume_path = str(auto)
            print(f"\n[Main] 🔄 Auto-resuming from {auto}")
            LOGGER.info("Auto-resume detected: loading checkpoint from %s", auto)
    
    if resume_path:
        print(f"[Main] Loading checkpoint: {resume_path}")
        LOGGER.info("Resuming training from checkpoint: %s", resume_path)
        trainer.load_checkpoint(resume_path, weights_only=args.load_weights_only)
        print(f"[Main] ✅ Resume successful - continuing training\n")
    else:
        print(f"[Main] Starting training from scratch (no checkpoint to resume)\n")
        LOGGER.info("Starting training from scratch (no checkpoint to resume)")

    LOGGER.info("Starting training loop")
    trainer.train()
    LOGGER.info("Training script completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_exception(LOGGER, "Training failed with an unhandled exception.")
        raise
