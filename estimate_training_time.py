#!/usr/bin/env python3
"""
Estimate training time for Swin-VALLR based on dataset size and hardware.
Run this before starting training to get a realistic time estimate.
"""

import json
import argparse
from pathlib import Path


def estimate_training_time(
    data_dir: str,
    batch_size: int = 4,
    epochs: int = 50,
    num_workers: int = 4,
    gpu_type: str = "6800xt"
):
    """
    Estimate training time based on dataset and configuration.
    
    Args:
        data_dir: Path to preprocessed dataset
        batch_size: Training batch size
        epochs: Number of epochs
        num_workers: Number of data loading workers
        gpu_type: GPU type (6800xt, 3090, a100, etc.)
    """
    
    # Load dataset metadata
    metadata_path = Path(data_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ Error: {metadata_path} not found")
        return
    
    with open(metadata_path) as f:
        samples = json.load(f)
    
    total_samples = len(samples)
    
    # Calculate batches per epoch (80% train split)
    train_samples = int(total_samples * 0.8)
    batches_per_epoch = train_samples // batch_size
    
    # Estimate time per batch based on GPU type
    # These are rough estimates based on typical performance
    time_per_batch = {
        "6800xt": 0.5,    # AMD 6800XT (your GPU)
        "3090": 0.4,      # NVIDIA RTX 3090
        "a100": 0.3,      # NVIDIA A100
        "v100": 0.45,     # NVIDIA V100
        "4090": 0.35,     # NVIDIA RTX 4090
        "7900xtx": 0.45,  # AMD 7900 XTX
    }
    
    seconds_per_batch = time_per_batch.get(gpu_type.lower(), 0.5)
    
    # Calculate training time
    train_time_per_epoch = batches_per_epoch * seconds_per_batch
    
    # Validation time (10% of samples, faster since no backprop)
    val_samples = int(total_samples * 0.1)
    val_batches = val_samples // batch_size
    val_time_per_epoch = val_batches * seconds_per_batch * 0.3  # Validation is ~30% of training time
    
    # Total time per epoch
    total_time_per_epoch = train_time_per_epoch + val_time_per_epoch
    
    # Add overhead (checkpointing, logging, etc.)
    overhead_per_epoch = 30  # seconds
    total_time_per_epoch += overhead_per_epoch
    
    # Total training time
    total_training_seconds = total_time_per_epoch * epochs
    total_training_hours = total_training_seconds / 3600
    
    # Early stopping estimate (assume stops at 70% of epochs on average)
    early_stop_hours = total_training_hours * 0.7
    
    # Print results
    print("=" * 60)
    print("Training Time Estimation")
    print("=" * 60)
    print(f"\n📊 Dataset Information:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Train samples: {train_samples:,} (80%)")
    print(f"  Val samples: {val_samples:,} (10%)")
    print(f"  Test samples: {int(total_samples * 0.1):,} (10%)")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  GPU: {gpu_type.upper()}")
    print(f"  Estimated time per batch: {seconds_per_batch:.2f}s")
    
    print(f"\n⏱️  Time Estimates:")
    print(f"  Time per epoch: {total_time_per_epoch/60:.1f} minutes")
    print(f"  Total training time: {total_training_hours:.1f} hours")
    print(f"  With early stopping: {early_stop_hours:.1f} hours (estimated)")
    
    print(f"\n💡 Recommendations:")
    
    if total_training_hours > 20:
        print(f"  ⚠️  Estimated time ({total_training_hours:.1f}h) exceeds 20 hours!")
        recommended_epochs = int(epochs * 19.5 / total_training_hours)
        print(f"  ✅ Reduce epochs to {recommended_epochs} for 19.5h budget")
        print(f"  ✅ Or use: --time_budget_hours 19.5 (will stop automatically)")
    elif total_training_hours > 15:
        print(f"  ✅ Training time is reasonable for 20-hour deadline")
        print(f"  ✅ Use: --time_budget_hours 19.5 (safety buffer)")
    else:
        print(f"  ✅ Training time is well within 20-hour deadline")
        print(f"  ✅ Consider increasing epochs for better results")
    
    print(f"\n  ✅ Always run smoke test first: --smoke_test")
    print(f"  ✅ Enable early stopping: --early_stopping_patience 20")
    
    print(f"\n🚀 Recommended Command:")
    print(f"\n  python train_swin_vallr.py \\")
    print(f"      --data_dir {data_dir} \\")
    print(f"      --batch_size {batch_size} \\")
    print(f"      --epochs {epochs} \\")
    print(f"      --lr 3e-5 \\")
    print(f"      --num_workers {num_workers} \\")
    print(f"      --time_budget_hours 19.5 \\")
    print(f"      --early_stopping_patience 20 \\")
    print(f"      --seed 42")
    
    print("\n" + "=" * 60)
    
    # Return values for programmatic use
    return {
        "total_samples": total_samples,
        "train_samples": train_samples,
        "batches_per_epoch": batches_per_epoch,
        "time_per_epoch_minutes": total_time_per_epoch / 60,
        "total_training_hours": total_training_hours,
        "early_stop_hours": early_stop_hours,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate training time for Swin-VALLR"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="final_preprocessed_dataset_fast",
        help="Path to preprocessed dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="6800xt",
        choices=["6800xt", "3090", "a100", "v100", "4090", "7900xtx"],
        help="GPU type for time estimation"
    )
    
    args = parser.parse_args()
    
    estimate_training_time(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        gpu_type=args.gpu
    )


if __name__ == "__main__":
    main()
