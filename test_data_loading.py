#!/usr/bin/env python3
"""
Quick test to verify data loading fixes work correctly.
This should complete in under 10 seconds if everything is working.
"""

import sys
import time
import torch
from data_pipeline import create_dataloader, DataConfig

print("=" * 80)
print("DATA LOADING TEST")
print("=" * 80)
print()

# Test configuration
data_dir = "final_preprocessed_dataset_fast"
batch_size = 1
num_workers = 0  # Single-threaded for testing

print(f"[1/4] Creating dataloader...")
print(f"  - data_dir: {data_dir}")
print(f"  - batch_size: {batch_size}")
print(f"  - num_workers: {num_workers}")
print()

try:
    start = time.time()
    train_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        epoch=0,
        training=True,
        config=DataConfig(),
        split="train"
    )
    elapsed = time.time() - start
    print(f"✅ Dataloader created in {elapsed:.2f}s")
    print(f"  - Total batches: {len(train_loader)}")
    print()
except Exception as e:
    print(f"❌ FAILED to create dataloader: {e}")
    sys.exit(1)

print(f"[2/4] Loading first batch...")
try:
    start = time.time()
    first_batch = next(iter(train_loader))
    elapsed = time.time() - start
    print(f"✅ First batch loaded in {elapsed:.2f}s")
    print(f"  - Video shape: {first_batch['video'].shape}")
    print(f"  - Text count: {len(first_batch['text'])}")
    print(f"  - Sample text: {first_batch['text'][0][:50]}...")
    print()
except Exception as e:
    print(f"❌ FAILED to load first batch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[3/4] Loading 5 more batches...")
try:
    start = time.time()
    count = 0
    for batch in train_loader:
        count += 1
        if count >= 5:
            break
    elapsed = time.time() - start
    print(f"✅ Loaded {count} batches in {elapsed:.2f}s ({elapsed/count:.2f}s per batch)")
    print()
except Exception as e:
    print(f"❌ FAILED to load batches: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[4/4] Testing validation loader...")
try:
    start = time.time()
    val_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        epoch=0,
        training=False,
        config=DataConfig(),
        split="val"
    )
    first_val_batch = next(iter(val_loader))
    elapsed = time.time() - start
    print(f"✅ Validation loader works in {elapsed:.2f}s")
    print(f"  - Total batches: {len(val_loader)}")
    print()
except Exception as e:
    print(f"❌ FAILED validation loader: {e}")
    sys.exit(1)

print("=" * 80)
print("✅ ALL TESTS PASSED - DATA LOADING IS WORKING!")
print("=" * 80)
print()
print("You can now run training with:")
print("  bash FINAL_TRAINING_COMMAND.sh")
print()
