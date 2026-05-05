"""
Calculate component-wise parameter counts for Swin-VALLR model.
"""

import torch
from model_architecture import SwinVALLR, SwinConfig

def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def main():
    # Create model with same config as training
    config = SwinConfig(
        temporal_attention_layers=1,
        temporal_attention_heads=16,
        temporal_attention_kernel=3,
        temporal_attention_dropout=0.1,
        temporal_multiscale=True  # This was enabled by default
    )
    
    model = SwinVALLR(config, load_refiner=False)
    
    # Calculate component-wise parameters
    visual_encoder_params = count_parameters(model.visual_encoder)
    temporal_adapter_params = count_parameters(model.temporal_adapter)
    temporal_multiscale_params = count_parameters(model.temporal_multiscale)
    temporal_attention_params = count_parameters(model.temporal_attention)
    feature_norm_params = count_parameters(model.feature_norm)
    phoneme_head_params = count_parameters(model.phoneme_head)
    
    total_params = count_parameters(model)
    
    print("=" * 70)
    print("SWIN-VALLR MODEL PARAMETER BREAKDOWN")
    print("=" * 70)
    print(f"Visual Front-End (Swin Encoder):     {visual_encoder_params:>12,} params")
    print(f"Temporal Adapter:                     {temporal_adapter_params:>12,} params")
    print(f"Temporal Multi-Scale Fusion:          {temporal_multiscale_params:>12,} params")
    print(f"Temporal Attention (Conformer):       {temporal_attention_params:>12,} params")
    print(f"Feature Normalization:                {feature_norm_params:>12,} params")
    print(f"Phoneme Head (CTC):                   {phoneme_head_params:>12,} params")
    print("-" * 70)
    print(f"TOTAL TRAINABLE PARAMETERS:           {total_params:>12,} params")
    print("=" * 70)
    print()
    print("NOTE: Linguistic Refiner (Qwen2-0.5B) was NOT loaded during training")
    print("      (load_refiner=False), so LoRA parameters = 0")
    print()
    
    # Verify total matches expected
    expected_total = 35829669
    if total_params == expected_total:
        print(f"✓ Total matches training log: {expected_total:,}")
    else:
        print(f"⚠ Warning: Total {total_params:,} != expected {expected_total:,}")

if __name__ == "__main__":
    main()
