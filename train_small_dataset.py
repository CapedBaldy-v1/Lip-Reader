#!/usr/bin/env python3
"""
Train Swin-VALLR on small dataset (10 videos).
Optimized for quick training and evaluation.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_architecture import SwinVALLR, SwinConfig
from backend_manager import get_device, get_grad_scaler, is_mixed_precision_available

# Character vocabulary (same as Swin-VALLR)
VOCAB = " abcdefghijklmnopqrstuvwxyz'-"
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
BLANK_IDX = len(VOCAB)  # CTC blank token


class SmallDataset(Dataset):
    """Dataset for small number of videos with word-level alignment."""
    
    def __init__(self, data_dir: str, split: str = 'train', use_alignment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_alignment = use_alignment
        
        # Load manifest
        manifest_path = self.data_dir / split / 'manifest.json'
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        alignment_info = " (word-aligned)" if use_alignment else ""
        print(f"  Loaded {len(self.samples)} {split} samples{alignment_info}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video (ROIs)
        video_path = self.data_dir / sample['video_path']
        video = np.load(video_path)  # (50, 96, 96, 3)
        
        # Load label (transcript)
        label_path = self.data_dir / sample['label_path']
        with open(label_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Normalize video to [0, 1]
        video = video.astype(np.float32) / 255.0
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video = torch.from_numpy(video).permute(3, 0, 1, 2)  # (3, 50, 96, 96)
        
        return video, text, sample['video_id']


def text_to_indices(text: str) -> List[int]:
    """Convert text to character indices."""
    text = text.lower()
    indices = []
    for char in text:
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
    return indices


def indices_to_text(indices: List[int]) -> str:
    """Convert character indices to text."""
    return ''.join([IDX_TO_CHAR.get(i, '') for i in indices])


def decode_ctc(logits: torch.Tensor) -> str:
    """
    Decode CTC output using greedy decoding.
    
    Args:
        logits: (T, vocab_size) logits
    
    Returns:
        Decoded text
    """
    # Get most likely character at each timestep
    pred_indices = torch.argmax(logits, dim=-1)  # (T,)
    
    # Remove blanks and duplicates
    decoded = []
    prev_idx = None
    for idx in pred_indices.cpu().numpy():
        if idx != BLANK_IDX and idx != prev_idx:
            decoded.append(int(idx))
        prev_idx = idx
    
    return indices_to_text(decoded)


def calculate_cer(pred: str, target: str) -> float:
    """
    Calculate Character Error Rate.
    
    CER = (substitutions + deletions + insertions) / len(target)
    """
    # Simple Levenshtein distance
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    
    # Dynamic programming
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / len(target)


def collate_fn(batch):
    """Collate function for DataLoader."""
    videos, texts, video_ids = zip(*batch)
    
    # Stack videos: (B, C, T, H, W)
    videos = torch.stack(videos, dim=0)  # (B, 3, 50, 96, 96)
    
    # Convert texts to indices
    targets = []
    target_lengths = []
    for text in texts:
        indices = text_to_indices(text)
        targets.extend(indices)
        target_lengths.append(len(indices))
    
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return videos, targets, target_lengths, list(texts), list(video_ids)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for videos, targets, target_lengths, texts, video_ids in pbar:
        # Move to device
        videos = videos.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        logits = model(videos)  # (B, T, vocab_size)
        
        # CTC loss expects (T, B, vocab_size)
        logits = logits.permute(1, 0, 2)
        
        # Input lengths (all frames)
        input_lengths = torch.full((videos.size(0),), logits.size(0), dtype=torch.long)
        
        # Calculate loss
        loss = criterion(logits, targets, input_lengths, target_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    num_batches = 0
    num_samples = 0
    
    predictions = []
    
    with torch.no_grad():
        for videos, targets, target_lengths, texts, video_ids in tqdm(dataloader, desc="Validating"):
            # Move to device
            videos = videos.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            logits = model(videos)  # (B, T, vocab_size)
            
            # CTC loss
            logits_ctc = logits.permute(1, 0, 2)
            input_lengths = torch.full((videos.size(0),), logits_ctc.size(0), dtype=torch.long)
            loss = criterion(logits_ctc, targets, input_lengths, target_lengths)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Decode predictions
            for i in range(logits.size(0)):
                pred_text = decode_ctc(logits[i])
                target_text = texts[i].lower()
                
                # Calculate CER
                cer = calculate_cer(pred_text, target_text)
                total_cer += cer
                num_samples += 1
                
                predictions.append({
                    'video_id': video_ids[i],
                    'prediction': pred_text,
                    'target': target_text,
                    'cer': cer
                })
    
    avg_loss = total_loss / num_batches
    avg_cer = total_cer / num_samples
    
    return avg_loss, avg_cer, predictions


def main():
    print("="*70)
    print("TRAINING SWIN-VALLR ON SMALL DATASET")
    print("="*70)
    
    # Configuration
    data_dir = 'dataset'
    batch_size = 2  # Small batch for 10 videos
    num_epochs = 50  # More epochs for small dataset
    learning_rate = 1e-4
    
    # Device
    device = get_device()
    print(f"\n✓ Using device: {device}")
    
    # Create datasets
    print("\n[1/5] Loading datasets...")
    train_dataset = SmallDataset(data_dir, 'train')
    val_dataset = SmallDataset(data_dir, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for small dataset
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n[2/5] Creating model...")
    config = SwinConfig(
        img_size=96,
        patch_size=4,
        in_channels=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        num_phonemes=len(VOCAB) + 1,  # +1 for CTC blank
        num_frames=50
    )
    
    model = SwinVALLR(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    print("\n[3/5] Setting up training...")
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"  ✓ Loss: CTC Loss")
    print(f"  ✓ Optimizer: AdamW (lr={learning_rate})")
    print(f"  ✓ Scheduler: CosineAnnealing")
    
    # Training loop
    print(f"\n[4/5] Training for {num_epochs} epochs...")
    print("="*70)
    
    best_cer = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cer': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_cer, predictions = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        
        # Print stats
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val CER:    {val_cer:.2%}")
        
        # Show sample predictions
        if epoch % 10 == 0 or epoch == 1:
            print(f"\nSample predictions:")
            for pred in predictions[:2]:
                print(f"  Video: {pred['video_id']}")
                print(f"  Target: {pred['target'][:80]}...")
                print(f"  Pred:   {pred['prediction'][:80]}...")
                print(f"  CER:    {pred['cer']:.2%}")
                print()
        
        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_loss': val_loss,
            }, 'best_model.pt')
            print(f"  ✓ Saved best model (CER: {val_cer:.2%})")
    
    # Final evaluation
    print("\n" + "="*70)
    print("[5/5] FINAL EVALUATION")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    val_loss, val_cer, predictions = validate(model, val_loader, criterion, device)
    
    print(f"\nBest Model Performance:")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation CER:  {val_cer:.2%}")
    print(f"  Accuracy:        {(1 - val_cer) * 100:.2f}%")
    
    print(f"\nAll predictions:")
    for pred in predictions:
        print(f"\n  Video: {pred['video_id']}")
        print(f"  Target: {pred['target'][:100]}...")
        print(f"  Pred:   {pred['prediction'][:100]}...")
        print(f"  CER:    {pred['cer']:.2%}")
    
    # Save results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_cer': float(val_cer),
        'best_val_loss': float(val_loss),
        'accuracy': float((1 - val_cer) * 100),
        'history': history,
        'predictions': predictions
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to training_results.json")
    print(f"✓ Best model saved to best_model.pt")
    print()


if __name__ == '__main__':
    main()
