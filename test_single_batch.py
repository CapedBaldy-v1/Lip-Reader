#!/usr/bin/env python3
"""Test if a single batch can be loaded"""

import sys
import torch
from data_pipeline import create_dataloaders

print("Creating dataloaders...")
train_loader, val_loader = create_dataloaders(
    data_dir="final_preprocessed_dataset_fast",
    batch_size=4,
    num_workers=0,
    epoch=0,
    training_phase="FULL_DATASET",
    disable_curriculum=True
)

print(f"Train loader created: {len(train_loader)} batches")
print("Loading first batch...")

try:
    batch = next(iter(train_loader))
    print(f"✅ SUCCESS! Batch loaded:")
    print(f"  Video shape: {batch['video'].shape}")
    print(f"  Texts: {len(batch['text'])} samples")
    print(f"  First text: {batch['text'][0]}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
