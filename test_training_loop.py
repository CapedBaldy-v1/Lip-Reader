#!/usr/bin/env python3
"""
Test that the training loop can iterate without hanging.
"""

import sys
import torch
from data_pipeline import create_dataloader, DataConfig
from model_architecture import create_model, SwinConfig
from train_swin_vallr import Trainer, TrainingConfig

print("=" * 80)
print("TRAINING LOOP TEST")
print("=" * 80)
print()

# Minimal configuration
data_dir = "final_preprocessed_dataset_fast"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[1/4] Creating model...")
model = create_model(load_refiner=False)
model = model.to(device)
print(f"✅ Model created and moved to {device}")
print()

print("[2/4] Creating dataloaders...")
train_loader = create_dataloader(
    data_dir=data_dir,
    batch_size=1,
    num_workers=0,
    epoch=0,
    training=True,
    config=DataConfig(),
    split="train"
)
val_loader = create_dataloader(
    data_dir=data_dir,
    batch_size=1,
    num_workers=0,
    epoch=0,
    training=False,
    config=DataConfig(),
    split="val"
)
print(f"✅ Dataloaders created: {len(train_loader)} train, {len(val_loader)} val")
print()

print("[3/4] Creating trainer...")
config = TrainingConfig()
config.smoke_test = True  # Only 10 batches
config.epochs = 1  # Only 1 epoch
trainer = Trainer(model, train_loader, val_loader, config, device)
print("✅ Trainer created")
print()

print("[4/4] Running one training epoch (10 batches)...")
try:
    train_loss = trainer.train_epoch()
    print(f"✅ Training epoch completed! Loss: {train_loss:.4f}")
    print()
except Exception as e:
    print(f"❌ Training epoch FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("✅ TRAINING LOOP TEST PASSED!")
print("=" * 80)
print()
print("Training is ready to run. Execute:")
print("  bash FINAL_TRAINING_COMMAND.sh")
print()
