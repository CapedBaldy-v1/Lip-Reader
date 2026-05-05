"""
Swin-VALLR Hardware Abstraction Layer (HAL)
============================================
Provides unified device management across CUDA, ROCm, DirectML, and CPU backends.
Implements DirectML-safe tensor operations for AMD Windows compatibility.
"""

import os
from typing import Tuple, Optional, Literal

import torch

from logging_utils import setup_logging, log_exception, log_system_info
from artifact_utils import get_useful_dir, save_json


LOGGER = setup_logging("backend_manager")

# Backend type definition
BackendType = Literal['cuda', 'rocm', 'directml', 'cpu']

# Global backend state
_BACKEND: Optional[BackendType] = None
_DEVICE: Optional[torch.device] = None


def _detect_cuda_or_rocm() -> Optional[Tuple[BackendType, torch.device]]:
    """Return CUDA/ROCm backend if available, otherwise None."""
    if not torch.cuda.is_available():
        LOGGER.debug("torch.cuda.is_available() returned False.")
        return None

    device = torch.device('cuda')
    LOGGER.info("CUDA backend detected. Probing GPU name...")

    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        LOGGER.info("Detected GPU name: %s", gpu_name)
        if 'amd' in gpu_name or 'radeon' in gpu_name:
            LOGGER.info("Classified backend as ROCm.")
            return 'rocm', device
    except Exception:
        log_exception(LOGGER, "Failed to read CUDA device name.")

    LOGGER.info("Classified backend as NVIDIA CUDA.")
    return 'cuda', device


def _detect_directml_device() -> Optional[torch.device]:
    """Return a DirectML device if available, otherwise None."""
    try:
        import torch_directml
    except ImportError:
        LOGGER.debug("torch_directml is not available.")
        return None

    try:
        return torch_directml.device()
    except Exception as e:
        LOGGER.warning("DirectML available but initialization failed: %s", e)
        print(f"[HAL] DirectML available but initialization failed: {e}")
        return None


def _detect_backend() -> Tuple[BackendType, torch.device]:
    """
    Hierarchically detect available compute backend.
    Priority: CUDA/ROCm -> DirectML -> CPU

    Returns:
        Tuple of (backend_name, torch_device)
    """
    cuda_result = _detect_cuda_or_rocm()
    if cuda_result is not None:
        return cuda_result

    directml_device = _detect_directml_device()
    if directml_device is not None:
        LOGGER.info("DirectML backend detected.")
        return 'directml', directml_device

    LOGGER.info("Falling back to CPU backend.")
    return 'cpu', torch.device('cpu')


def _apply_rocm_tweaks() -> None:
    """Enable ROCm-specific options for consumer AMD GPUs."""
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        LOGGER.info("Enabled TF32 for ROCm matmul (if supported).")

    if os.environ.get('HSA_OVERRIDE_GFX_VERSION') is None:
        LOGGER.info("HSA_OVERRIDE_GFX_VERSION not set; emitting guidance.")
        print("[HAL] TIP: For AMD 6800XT, set HSA_OVERRIDE_GFX_VERSION=10.3.0")


def initialize() -> None:
    """Initialize the HAL and detect backend. Call once at startup."""
    global _BACKEND, _DEVICE
    _BACKEND, _DEVICE = _detect_backend()

    if _BACKEND == 'rocm':
        _apply_rocm_tweaks()

    LOGGER.info("HAL initialized with backend=%s device=%s", _BACKEND, _DEVICE)
    print(f"[HAL] Initialized with backend: {_BACKEND}, device: {_DEVICE}")


def get_device() -> torch.device:
    """Get the active compute device."""
    if _DEVICE is None:
        initialize()
    return _DEVICE


def get_backend() -> BackendType:
    """Get the active backend name."""
    if _BACKEND is None:
        initialize()
    return _BACKEND


def get_backend_name() -> str:
    """Get human-readable backend name."""
    backend = get_backend()
    names = {
        'cuda': 'NVIDIA CUDA',
        'rocm': 'AMD ROCm',
        'directml': 'Microsoft DirectML',
        'cpu': 'CPU (Fallback)'
    }
    return names.get(backend, 'Unknown')


def is_mixed_precision_available() -> bool:
    """
    Check if mixed precision (AMP) training is available.
    Returns True for CUDA/ROCm, False for DirectML/CPU.
    """
    backend = get_backend()
    return backend in ('cuda', 'rocm')


def is_compile_available() -> bool:
    """
    Check if torch.compile is available and beneficial.
    Returns True for CUDA/ROCm with PyTorch 2.0+.
    """
    if get_backend() not in ('cuda', 'rocm'):
        return False
    return hasattr(torch, 'compile')


def _safe_roll_directml(x: torch.Tensor, shifts: int, dims: int) -> torch.Tensor:
    """DirectML-safe roll via slicing and concatenation."""
    size = x.size(dims)
    if size == 0:
        return x
    shifts = shifts % size

    if shifts == 0:
        return x

    split_point = size - shifts
    head = x.narrow(dims, 0, split_point)
    tail = x.narrow(dims, split_point, shifts)
    return torch.cat((tail, head), dim=dims)


def safe_roll(x: torch.Tensor, shifts: int, dims: int) -> torch.Tensor:
    """
    DirectML-safe alternative to torch.roll.

    torch.roll has performance issues and potential instability on DirectML.
    This implementation uses tensor slicing and concatenation, which are
    primitive operations with robust DirectML support.

    Args:
        x: Input tensor
        shifts: Number of positions to shift (positive = right shift)
        dims: Dimension along which to shift

    Returns:
        Rolled tensor with elements cyclically shifted
    """
    if get_backend() == 'directml':
        return _safe_roll_directml(x, shifts, dims)
    return torch.roll(x, shifts, dims)


def safe_roll_2d(x: torch.Tensor, shifts: Tuple[int, int], dims: Tuple[int, int]) -> torch.Tensor:
    """
    DirectML-safe 2D roll operation for Swin Transformer window shifting.

    Args:
        x: Input tensor
        shifts: Tuple of (shift_h, shift_w)
        dims: Tuple of (dim_h, dim_w)

    Returns:
        2D rolled tensor
    """
    if get_backend() == 'directml':
        x = safe_roll(x, shifts[0], dims[0])
        x = safe_roll(x, shifts[1], dims[1])
        return x
    return torch.roll(x, shifts, dims)


def to_device(tensor_or_module):
    """
    Move tensor or module to the active compute device.

    Args:
        tensor_or_module: torch.Tensor or nn.Module to move

    Returns:
        Object on the active device
    """
    device = get_device()
    return tensor_or_module.to(device)


def get_autocast_context():
    """
    Get the appropriate autocast context manager for mixed precision.

    Returns:
        Context manager for AMP, or nullcontext if not available
    """
    from contextlib import nullcontext

    if is_mixed_precision_available():
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def get_grad_scaler():
    """
    Get gradient scaler for mixed precision training.

    Returns:
        GradScaler if available, else None
    """
    if is_mixed_precision_available():
        return torch.amp.GradScaler('cuda')
    return None


def get_device_info() -> dict:
    """
    Get detailed information about the compute device.

    Returns:
        Dictionary with device specifications
    """
    backend = get_backend()
    device = get_device()

    info = {
        'backend': backend,
        'backend_name': get_backend_name(),
        'device': str(device),
        'mixed_precision': is_mixed_precision_available(),
        'compile_available': is_compile_available()
    }

    if backend in ('cuda', 'rocm'):
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['vram_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['vram_available_gb'] = (
            torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        ) / (1024**3)

    return info


def print_device_info() -> None:
    """Print formatted device information."""
    info = get_device_info()
    LOGGER.info("Device info: %s", info)
    try:
        out_dir = get_useful_dir("backend_manager", "device_info")
        save_json(out_dir / "device_info.json", info)
    except Exception:
        log_exception(LOGGER, "Failed to save device info artifact.")
    print("\n" + "="*50)
    print("SWIN-VALLR Hardware Configuration")
    print("="*50)
    print(f"Backend:          {info['backend_name']}")
    print(f"Device:           {info['device']}")
    print(f"Mixed Precision:  {'Enabled' if info['mixed_precision'] else 'Disabled'}")
    print(f"torch.compile:    {'Available' if info['compile_available'] else 'Unavailable'}")

    if 'gpu_name' in info:
        print(f"GPU:              {info['gpu_name']}")
        print(f"VRAM Total:       {info['vram_total_gb']:.1f} GB")
        print(f"VRAM Available:   {info['vram_available_gb']:.1f} GB")

    print("="*50 + "\n")


# Auto-initialize on import
initialize()


if __name__ == "__main__":
    # Test the HAL
    LOGGER.info("Running backend_manager self-test.")
    log_system_info(LOGGER)
    print_device_info()

    # Test safe_roll
    print("\nTesting safe_roll...")
    x = torch.arange(10, device=get_device())
    print(f"Original: {x}")
    print(f"safe_roll(x, 3, 0): {safe_roll(x, 3, 0)}")
    print(f"torch.roll(x, 3, 0): {torch.roll(x.cpu(), 3, 0)}")

    # Test 2D roll
    print("\nTesting safe_roll_2d...")
    x2d = torch.arange(9, device=get_device()).reshape(3, 3)
    print(f"Original 2D:\n{x2d}")
    print(f"safe_roll_2d(x2d, (1, 1), (0, 1)):\n{safe_roll_2d(x2d, (1, 1), (0, 1))}")
