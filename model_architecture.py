"""
Swin-VALLR Model Architecture
==============================
Visual Speech Recognition using Swin Transformer + Qwen2-0.5B LLM.

Components:
- TubeletEmbedding: 3D patch embedding for video input
- SwinTransformerBlock: Modified for DirectML compatibility
- VisualFrontEnd: Swin-Tiny encoder for lip features
- TemporalAdapter: 1D Conv for phoneme-rate alignment
- TemporalMultiScaleFusion: Multi-kernel temporal fusion for richer context
- PhonemeHead: CTC output for phoneme prediction
- LinguisticRefiner: Qwen2-0.5B for text correction
"""

import math
import re
import unicodedata
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend_manager import get_device, get_backend, safe_roll, safe_roll_2d, to_device
from logging_utils import setup_logging, log_exception, log_system_info
from artifact_utils import get_useful_dir, save_json, save_text


LOGGER = setup_logging("model_architecture")


# =============================================================================
# Phoneme Vocabulary
# =============================================================================

PHONEME_VOCAB = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', '<blank>'
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SwinConfig:
    """Swin-Tiny configuration for lip reading."""
    img_size: int = 96
    patch_size: int = 4
    in_channels: int = 3
    num_frames: int = 50
    temporal_patch: int = 1

    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7

    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    num_phonemes: int = 40

    # Temporal feature normalization
    temporal_feature_norm: bool = True
    temporal_feature_norm_eps: float = 1e-5
    temporal_feature_norm_affine: bool = True

    # Temporal multi-scale fusion
    temporal_multiscale: bool = False
    temporal_multiscale_kernels: Tuple[int, ...] = (3, 5, 7)
    temporal_multiscale_dropout: float = 0.1
    temporal_multiscale_gated: bool = True

    # Temporal attention (SwinLip-inspired)
    temporal_attention_layers: int = 0
    temporal_attention_heads: int = 16
    temporal_attention_kernel: int = 3
    temporal_attention_dropout: float = 0.1
    temporal_attention_use_mhsa: bool = True
    temporal_attention_ffn_mult: float = 4.0


# =============================================================================
# Tubelet Embedding
# =============================================================================

class TubeletEmbedding(nn.Module):
    """
    3D Patch Embedding for video input (Tubelet Embedding).

    Input: (B, C, T, H, W) -> Output: (B, T', H', W', D)
    """

    def __init__(self, config: SwinConfig):
        super().__init__()
        self.config = config

        self.proj = nn.Conv3d(
            config.in_channels,
            config.embed_dim,
            kernel_size=(config.temporal_patch, config.patch_size, config.patch_size),
            stride=(config.temporal_patch, config.patch_size, config.patch_size)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        self.temporal_dim = config.num_frames // config.temporal_patch
        self.spatial_dim = config.img_size // config.patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: Video tensor (B, C, T, H, W)

        Returns:
            Embedded tokens (B, T', H', W', D) and dimensions
        """
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)

        B, T, H, W, D = x.shape
        return x, T, H, W


# =============================================================================
# Window Attention
# =============================================================================

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.
    DirectML-compatible implementation.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            self._build_relative_position_index(window_size)
        )

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _build_relative_position_index(window_size: int) -> torch.Tensor:
        """Precompute indices for relative position bias lookup."""
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        return relative_coords.sum(-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features (num_windows*B, N, C) where N = window_size^2
            mask: Attention mask for shifted windows

        Returns:
            Attended features (num_windows*B, N, C)
        """
        batch_windows, tokens_per_window, channels = x.shape

        qkv = self.qkv(x).reshape(
            batch_windows, tokens_per_window, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens_per_window,
                tokens_per_window
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens_per_window, tokens_per_window)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens_per_window, channels)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# =============================================================================
# Swin Transformer Block
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    batch_size = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(batch_size, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(batch_size, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with DirectML-safe cyclic shift.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if self.shift_size > 0:
            self.shift_size = min(self.shift_size, self.window_size // 2)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )

        self.attn_mask = None
        self._input_resolution = None

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted window attention."""
        if self.shift_size <= 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, C) where L = H * W
            H, W: Spatial dimensions

        Returns:
            Output tensor (B, L, C)
        """
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} != {H}*{W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H_pad = H + pad_h
        W_pad = W + pad_w

        if self._input_resolution != (H_pad, W_pad):
            self._input_resolution = (H_pad, W_pad)
            self.attn_mask = self._compute_attn_mask(H_pad, W_pad, x.device)

        if self.shift_size > 0:
            shifted_x = safe_roll_2d(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)

        if self.shift_size > 0:
            x = safe_roll_2d(shifted_x, (self.shift_size, self.shift_size), (1, 2))
        else:
            x = shifted_x

        if pad_h or pad_w:
            x = x[:, :H, :W, :]

        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Stochastic Depth (drop path) for residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# =============================================================================
# Patch Merging
# =============================================================================

class PatchMerging(nn.Module):
    """Merge patches for downsampling (2x2 -> 1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input (B, H*W, C)

        Returns:
            Output (B, H/2*W/2, 2C), new H, new W
        """
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2


# =============================================================================
# Swin Encoder Stage
# =============================================================================

class SwinStage(nn.Module):
    """A stage of Swin Transformer blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] = None,
        downsample: bool = True
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if drop_path else 0.0
            )
            for i in range(depth)
        ])

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for block in self.blocks:
            x = block(x, H, W)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


# =============================================================================
# Visual Front-End
# =============================================================================

class VisualFrontEnd(nn.Module):
    """
    Swin-Tiny Visual Encoder for lip reading.

    Takes video input and produces frame-wise features.
    """

    def __init__(self, config: SwinConfig = None):
        super().__init__()
        self.config = config or SwinConfig()

        self.patch_embed = TubeletEmbedding(self.config)

        total_depth = sum(self.config.depths)
        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        dim = self.config.embed_dim

        for i, (depth, num_heads) in enumerate(zip(self.config.depths, self.config.num_heads)):
            stage = SwinStage(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=self.config.window_size,
                mlp_ratio=self.config.mlp_ratio,
                drop=self.config.drop_rate,
                attn_drop=self.config.attn_drop_rate,
                drop_path=dpr[sum(self.config.depths[:i]):sum(self.config.depths[:i+1])],
                downsample=(i < len(self.config.depths) - 1)
            )
            self.stages.append(stage)
            if i < len(self.config.depths) - 1:
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.output_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video input (B, C, T, H, W)

        Returns:
            Features (B, T', D) where T' is temporal dimension after tubelet embedding
        """
        B, C, T, H, W = x.shape

        x, T_out, H_out, W_out = self.patch_embed(x)

        x = x.view(B * T_out, H_out * W_out, -1)

        curr_H, curr_W = H_out, W_out
        for stage in self.stages:
            x, curr_H, curr_W = stage(x, curr_H, curr_W)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = x.view(B, T_out, -1)

        return x


# =============================================================================
# Temporal Adapter
# =============================================================================

class TemporalAdapter(nn.Module):
    """
    1D Convolutional adapter for temporal downsampling.

    Aligns frame-rate features (~25 FPS) to phoneme rate (~10-15/sec).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim
            # Use stride=1 to preserve temporal resolution for CTC
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_dim),
                nn.GELU()
            ])
            in_dim = out_dim

        self.conv_stack = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, T, D)

        Returns:
            Downsampled features (B, T', D')
        """
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        x = x.transpose(1, 2)
        return x


# =============================================================================
# Temporal Multi-Scale Fusion (SwinLip+ inspired)
# =============================================================================

class TemporalMultiScaleFusion(nn.Module):
    """Lightweight multi-kernel temporal fusion with optional gating."""

    def __init__(
        self,
        dim: int,
        kernels: Tuple[int, ...],
        dropout: float = 0.1,
        gated: bool = True
    ):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels if int(k) > 0)
        self.pre_norm = nn.LayerNorm(dim)
        self.branches = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim)
            for k in self.kernels
        ])
        self.post_pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.use_gating = gated and len(self.branches) > 1
        if self.use_gating:
            self.gate = nn.Linear(dim, len(self.branches))
        else:
            self.register_parameter("gate", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.branches or x.ndim != 3:
            return x
        residual = x
        x_norm = self.pre_norm(x)
        x_t = x_norm.transpose(1, 2)

        branch_outs = [conv(x_t) for conv in self.branches]
        if self.use_gating:
            pooled = x_norm.mean(dim=1)
            gate_logits = self.gate(pooled)
            weights = torch.softmax(gate_logits, dim=-1)
            fused = torch.zeros_like(branch_outs[0])
            for idx, out in enumerate(branch_outs):
                fused = fused + out * weights[:, idx].view(-1, 1, 1)
        else:
            fused = torch.zeros_like(branch_outs[0])
            for out in branch_outs:
                fused = fused + out
            fused = fused / float(len(branch_outs))

        fused = self.post_pointwise(fused)
        fused = F.gelu(fused)
        fused = self.dropout(fused)
        fused = fused.transpose(1, 2)
        return residual + fused


# =============================================================================
# Temporal Feature Normalization
# =============================================================================

class TemporalFeatureNormalizer(nn.Module):
    """Normalize features per-utterance along time for stable CTC training."""

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] == 0:
            return x
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.gamma + self.beta
        return x


# =============================================================================
# Temporal Convolutional Attention (SwinLip-style)
# =============================================================================

class TemporalConvModule(nn.Module):
    """Depthwise temporal convolution with pointwise projections."""

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.pointwise_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.activation = nn.GELU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class TemporalConformerBlock(nn.Module):
    """Conformer-style block for temporal attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        ffn_mult: float = 4.0,
        use_mhsa: bool = True
    ):
        super().__init__()
        hidden_dim = int(dim * ffn_mult)
        self.use_mhsa = use_mhsa

        self.ffn1_norm = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.mhsa_norm = nn.LayerNorm(dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.conv_norm = nn.LayerNorm(dim)
        self.conv = TemporalConvModule(dim, kernel_size=kernel_size, dropout=dropout)

        self.ffn2_norm = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        attn_weights = None
        if self.use_mhsa:
            if return_attn:
                attn_out, attn_weights = self.mhsa(
                    self.mhsa_norm(x),
                    self.mhsa_norm(x),
                    self.mhsa_norm(x),
                    need_weights=True,
                    average_attn_weights=False
                )
            else:
                attn_out, _ = self.mhsa(
                    self.mhsa_norm(x),
                    self.mhsa_norm(x),
                    self.mhsa_norm(x),
                    need_weights=False
                )
            x = x + attn_out

        x = x + self.conv(self.conv_norm(x))
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        x = self.final_norm(x)
        if return_attn:
            return x, attn_weights
        return x


class TemporalConformerStack(nn.Module):
    """Stack of temporal conformer blocks."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        ffn_mult: float = 4.0,
        use_mhsa: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalConformerBlock(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dropout=dropout,
                ffn_mult=ffn_mult,
                use_mhsa=use_mhsa
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_weights = [] if return_attn else None
        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, return_attn=True)
                if attn is not None:
                    attn_weights.append(attn)
            else:
                x = layer(x)
        if return_attn:
            return x, attn_weights
        return x

# =============================================================================
# Phoneme Head
# =============================================================================

class PhonemeHead(nn.Module):
    """
    CTC output head for phoneme prediction.
    """

    def __init__(self, input_dim: int, num_phonemes: int = 40):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_phonemes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, T, D)

        Returns:
            Log probabilities (B, T, num_phonemes)
        """
        return F.log_softmax(self.proj(x), dim=-1)


# =============================================================================
# Linguistic Refiner (Qwen2-0.5B)
# =============================================================================

class LinguisticRefiner:
    """
    Qwen2-0.5B-based phoneme-to-text refiner.

    Uses the LLM to correct homopheme ambiguities and produce coherent text.
    """

    PHONEME_MAP = {
        'AA': 'a', 'AE': 'a', 'AH': 'u', 'AO': 'o', 'AW': 'ow',
        'AY': 'ay', 'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th',
        'EH': 'e', 'ER': 'er', 'EY': 'ey', 'F': 'f', 'G': 'g',
        'HH': 'h', 'IH': 'i', 'IY': 'ee', 'JH': 'j', 'K': 'k',
        'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng', 'OW': 'o',
        'OY': 'oy', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
        'T': 't', 'TH': 'th', 'UH': 'oo', 'UW': 'oo', 'V': 'v',
        'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh', '<blank>': ''
    }

    ACCENT_VARIANTS = {
        "TH": ["T", "F", "D"],
        "DH": ["D", "Z", "V"],
        "T": ["D"],
        "D": ["T"],
        "R": ["W"],
        "V": ["W"],
        "W": ["V"],
        "S": ["Z"],
        "Z": ["S"],
        "AE": ["EH", "AH"],
        "AH": ["AE", "UH"],
        "IH": ["IY", "EH"],
        "IY": ["IH"],
        "UH": ["UW", "AH"],
        "UW": ["UH"],
        "OW": ["AO", "AA"],
        "AO": ["OW", "AA"],
        "CH": ["SH"],
        "SH": ["CH"],
        "NG": ["N"]
    }

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self, device: torch.device = None, adapter_path: Optional[str] = None):
        """Load the Qwen2 model (optionally with a LoRA adapter)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            backend = get_backend()
            if device is not None:
                self.device = device
            elif backend == "directml":
                self.device = torch.device("cpu")
            else:
                self.device = get_device()

            LOGGER.info("Loading refiner model: %s", self.model_name)
            print(f"[LinguisticRefiner] Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            dtype = torch.float16 if backend in ('cuda', 'rocm') else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map='auto' if backend in ('cuda', 'rocm') else None
            )

            if backend not in ('cuda', 'rocm'):
                self.model = self.model.to(self.device)

            if adapter_path:
                try:
                    from peft import PeftModel
                    adapter_path = str(adapter_path)
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    LOGGER.info("Loaded refiner adapter from %s", adapter_path)
                except Exception:
                    log_exception(LOGGER, "Failed to load refiner adapter.")

            self.model.eval()
            LOGGER.info("Refiner model loaded successfully on %s.", self.device)
            print(f"[LinguisticRefiner] Model loaded successfully")

        except Exception as e:
            log_exception(LOGGER, "Failed to load refiner model.")
            print(f"[LinguisticRefiner] Failed to load model: {e}")
            print("[LinguisticRefiner] Will use basic phoneme-to-grapheme conversion")
            self.model = None

    def _pad_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        return self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

    @staticmethod
    def romanize_text(text: str) -> str:
        """Romanize text into ASCII for language-agnostic prompting."""
        text = text.replace("\ufeff", " ")
        normalized = unicodedata.normalize("NFKD", text)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_text = ascii_text.lower()
        ascii_text = re.sub(r"[^a-z0-9' ]+", " ", ascii_text)
        ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
        return ascii_text

    @staticmethod
    def _normalize_for_scoring(text: str) -> str:
        text = LinguisticRefiner.romanize_text(text)
        return text.replace(" ", "")

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ch_a in enumerate(a, start=1):
            curr = [i]
            for j, ch_b in enumerate(b, start=1):
                cost = 0 if ch_a == ch_b else 1
                curr.append(min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost
                ))
            prev = curr
        return prev[-1]

    def _score_against_hint(self, candidate: str, hint: str) -> float:
        if not candidate:
            return 1.0
        cand_norm = self._normalize_for_scoring(candidate)
        hint_norm = self._normalize_for_scoring(hint)
        if not cand_norm and not hint_norm:
            return 0.0
        distance = self._edit_distance(cand_norm, hint_norm)
        denom = max(len(cand_norm), len(hint_norm), 1)
        return float(distance) / float(denom)

    @staticmethod
    def _clean_language_hint(language_hint: Optional[str]) -> Optional[str]:
        if not language_hint:
            return None
        hint = language_hint.strip()
        if not hint:
            return None
        if hint.lower() in {"auto", "unknown", "default"}:
            return None
        return hint

    @staticmethod
    def _format_reliability_tokens(tokens: Optional[List[str]]) -> str:
        if not tokens:
            return ""
        cleaned = [t.strip().upper()[:1] for t in tokens if t]
        return "".join(f"[{t}]" for t in cleaned if t)

    def phonemes_to_grapheme_hint(self, phonemes: List[str]) -> str:
        """Convert phoneme sequence to rough grapheme string for prompting."""
        return ''.join(self.PHONEME_MAP.get(p.upper(), p.lower()) for p in phonemes if p)

    def _build_phoneme_prompt(
        self,
        phoneme_sequence: List[str],
        hint: str,
        language_hint: Optional[str]
    ) -> str:
        phoneme_str = ' '.join(phoneme_sequence)
        language_hint = self._clean_language_hint(language_hint)
        accent_line = ""
        if language_hint is None or language_hint.lower() in {"english", "en"}:
            accent_line = "The speaker may have any English accent; interpret phonemes accordingly.\n\n"
        if language_hint:
            return (
                "You are an expert lip-reading assistant. Convert the following phoneme sequence "
                f"to proper {language_hint} text.\n\n"
                f"{accent_line}"
                f"Phoneme sequence: {phoneme_str}\n"
                f"Approximate sounds: {hint}\n\n"
                f"Output only the {language_hint} text, nothing else:"
            )
        return (
            "You are an expert lip-reading assistant. Convert the following phoneme sequence "
            "to proper English text.\n\n"
            f"{accent_line}"
            f"Phoneme sequence: {phoneme_str}\n"
            f"Approximate sounds: {hint}\n\n"
            "Output only the English text, nothing else:"
        )

    def _build_roman_prompt(self, roman_text: str, language_hint: Optional[str]) -> str:
        language_hint = self._clean_language_hint(language_hint) or "English"
        return (
            "You are an expert language assistant. Convert the following romanized speech "
            f"into proper {language_hint} text.\n\n"
            f"Roman text: {roman_text}\n\n"
            f"Output only the {language_hint} text, nothing else:"
        )

    def _generate_from_prompt(self, prompt: str, max_new_tokens: int) -> str:
        if self.model is None:
            return ""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._pad_token_id()
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        response = response.split('\n')[0].strip()
        return response

    def refine_phonemes(
        self,
        phoneme_sequence: List[str],
        language_hint: Optional[str] = None,
        mode: str = "phoneme"
    ) -> str:
        """
        Convert phoneme sequence to refined text.

        Args:
            phoneme_sequence: List of phoneme symbols (e.g., ['HH', 'EH', 'L', 'OW'])

        Returns:
            Refined text (e.g., "hello")
        """
        if mode == "roman":
            roman_text = self.phonemes_to_grapheme_hint(phoneme_sequence)
            return self.refine_roman(roman_text, language_hint=language_hint)
        if mode == "hybrid":
            return self.refine_hybrid(phoneme_sequence, language_hint=language_hint)

        hint = self.phonemes_to_grapheme_hint(phoneme_sequence)

        if self.model is None:
            return hint

        prompt = self._build_phoneme_prompt(phoneme_sequence, hint, language_hint)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self._pad_token_id()
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            response = response.split('\n')[0].strip()

            return response if response else hint

        except Exception as e:
            log_exception(LOGGER, "Refiner generation failed.")
            print(f"[LinguisticRefiner] Generation failed: {e}")
            return hint

    def refine_hybrid(
        self,
        phoneme_sequence: List[str],
        language_hint: Optional[str] = None
    ) -> str:
        """Refine phonemes using both phoneme + roman prompts and pick the best."""
        hint = self.phonemes_to_grapheme_hint(phoneme_sequence)

        if self.model is None:
            return hint

        phoneme_prompt = self._build_phoneme_prompt(phoneme_sequence, hint, language_hint)
        roman_prompt = self._build_roman_prompt(hint, language_hint)

        try:
            phoneme_out = self._generate_from_prompt(phoneme_prompt, max_new_tokens=50)
        except Exception:
            log_exception(LOGGER, "Hybrid phoneme prompt failed.")
            phoneme_out = ""

        try:
            roman_out = self._generate_from_prompt(roman_prompt, max_new_tokens=60)
        except Exception:
            log_exception(LOGGER, "Hybrid roman prompt failed.")
            roman_out = ""

        candidates = [c for c in [phoneme_out, roman_out] if c]
        if not candidates:
            return hint

        best = min(candidates, key=lambda c: self._score_against_hint(c, hint))
        return best if best else hint

    def refine_phonemes_with_alternatives(
        self,
        phoneme_sequence: List[str],
        alternatives: Optional[List[List[str]]] = None,
        confidences: Optional[List[float]] = None,
        language_hint: Optional[str] = None,
        mode: str = "phoneme"
    ) -> str:
        """Refine phonemes using optional alternatives for uncertain positions."""
        if mode == "roman":
            roman_text = self.phonemes_to_grapheme_hint(phoneme_sequence)
            return self.refine_roman(roman_text, language_hint=language_hint)
        if mode == "hybrid":
            return self.refine_hybrid(phoneme_sequence, language_hint=language_hint)

        if not alternatives or not any(alternatives):
            return self.refine_phonemes(
                phoneme_sequence,
                language_hint=language_hint,
                mode=mode
            )

        hint = self.phonemes_to_grapheme_hint(phoneme_sequence)
        if self.model is None:
            return hint

        target = self._clean_language_hint(language_hint) or "English"
        phoneme_str = " ".join(phoneme_sequence)
        prompt_lines = [
            "You are an expert lip-reading assistant.",
            f"Convert the following phoneme sequence to proper {target} text.",
            "",
            f"Phoneme sequence: {phoneme_str}",
            f"Approximate sounds: {hint}",
            "",
            "Uncertain positions (1-based index):"
        ]

        for idx, alt_list in enumerate(alternatives, start=1):
            if not alt_list:
                continue
            conf_text = ""
            if confidences and idx - 1 < len(confidences):
                conf_text = f" (confidence={confidences[idx-1]:.2f})"
            alt_text = ", ".join(alt_list)
            prompt_lines.append(f"{idx}. {phoneme_sequence[idx-1]} -> {alt_text}{conf_text}")

        prompt_lines.extend([
            "",
            f"Output only the {target} text, nothing else:"
        ])

        prompt = "\n".join(prompt_lines)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=self._pad_token_id()
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            response = response.split('\n')[0].strip()
            return response if response else hint

        except Exception as e:
            log_exception(LOGGER, "Refiner alternative prompt failed.")
            print(f"[LinguisticRefiner] Alternative prompt failed: {e}")
            return hint

    def refine_roman(self, roman_text: str, language_hint: Optional[str] = None) -> str:
        """Refine romanized speech into target language text."""
        if self.model is None:
            return roman_text

        prompt = self._build_roman_prompt(roman_text, language_hint)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            response = response.split('\n')[0].strip()
            return response if response else roman_text

        except Exception as e:
            log_exception(LOGGER, "Refiner romanization failed.")
            print(f"[LinguisticRefiner] Romanization failed: {e}")
            return roman_text

    def refine_candidates(
        self,
        candidates: List[List[str]],
        scores: Optional[List[float]] = None,
        language_hint: Optional[str] = None,
        mode: str = "phoneme"
    ) -> str:
        """
        Refine multiple phoneme hypotheses into a single best transcription.
        """
        if not candidates:
            return ""

        if self.model is None:
            if scores:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
            else:
                best_idx = 0
            if mode == "roman":
                return self.phonemes_to_grapheme_hint(candidates[best_idx])
            return " ".join(candidates[best_idx])

        if mode == "roman":
            target = self._clean_language_hint(language_hint) or "English"
            prompt_lines = [
                "You are an expert language assistant.",
                "Choose the best transcription based on the romanized hypotheses.",
                f"Output only the {target} text, nothing else.",
                ""
            ]
        else:
            if self._clean_language_hint(language_hint):
                target = self._clean_language_hint(language_hint)
                prompt_lines = [
                    "You are an expert lip-reading assistant.",
                    "Choose the best transcription based on the phoneme hypotheses.",
                    f"Output only the {target} text, nothing else.",
                    ""
                ]
            else:
                prompt_lines = [
                    "You are an expert lip-reading assistant.",
                    "Choose the best transcription based on the phoneme hypotheses.",
                    "Output only the final English text, nothing else.",
                    ""
                ]

        for idx, phonemes in enumerate(candidates, start=1):
            hint = self.phonemes_to_grapheme_hint(phonemes)
            score_text = ""
            if scores and idx - 1 < len(scores):
                score_text = f" | score={scores[idx-1]:.4f}"
            if mode == "roman":
                prompt_lines.append(f"{idx}. {hint}{score_text}")
            else:
                prompt_lines.append(f"{idx}. {(' '.join(phonemes))}{score_text}")
                prompt_lines.append(f"   hint: {hint}")

        prompt = "\n".join(prompt_lines)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            response = response.split('\n')[0].strip()
            return response

        except Exception as e:
            log_exception(LOGGER, "Refiner candidate selection failed.")
            print(f"[LinguisticRefiner] Candidate selection failed: {e}")
            if scores:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
            else:
                best_idx = 0
            return " ".join(candidates[best_idx])

    def refine_dual_hypotheses(
        self,
        primary_candidates: List[List[str]],
        secondary_candidates: List[List[str]],
        primary_scores: Optional[List[float]] = None,
        secondary_scores: Optional[List[float]] = None,
        primary_reliability: Optional[List[str]] = None,
        secondary_reliability: Optional[List[str]] = None,
        language_hint: Optional[str] = None,
        mode: str = "phoneme"
    ) -> str:
        """Refine dual hypothesis sets with reliability guidance."""
        if not primary_candidates and not secondary_candidates:
            return ""

        all_candidates = []
        all_scores = []
        for candidates, scores in (
            (primary_candidates, primary_scores),
            (secondary_candidates, secondary_scores)
        ):
            for idx, candidate in enumerate(candidates):
                all_candidates.append(candidate)
                if scores and idx < len(scores):
                    all_scores.append(scores[idx])
                else:
                    all_scores.append(float("-inf"))

        if self.model is None:
            if all_scores:
                best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
            else:
                best_idx = 0
            if mode == "roman":
                return self.phonemes_to_grapheme_hint(all_candidates[best_idx])
            return " ".join(all_candidates[best_idx])

        target = self._clean_language_hint(language_hint) or "English"
        if mode == "roman":
            prompt_lines = [
                "You are an expert language assistant.",
                "You are given two romanized hypothesis sets from the same video.",
                "Reliability tokens indicate which time chunks are reliable: [C]=reliable, [N]=noisy.",
                f"Output only the {target} text, nothing else.",
                ""
            ]
        else:
            prompt_lines = [
                "You are an expert lip-reading assistant.",
                "You are given two phoneme hypothesis sets from the same video.",
                "Reliability tokens indicate which time chunks are reliable: [C]=reliable, [N]=noisy.",
                "Prefer reliable segments and avoid hallucinating words not supported by the hypotheses.",
                f"Output only the {target} text, nothing else.",
                ""
            ]

        primary_tokens = self._format_reliability_tokens(primary_reliability)
        secondary_tokens = self._format_reliability_tokens(secondary_reliability)
        if primary_tokens:
            prompt_lines.append(f"Stream A reliability: {primary_tokens}")
        if secondary_tokens:
            prompt_lines.append(f"Stream B reliability: {secondary_tokens}")
        if primary_tokens or secondary_tokens:
            prompt_lines.append("")

        prompt_lines.append("Stream A hypotheses:")
        for idx, phonemes in enumerate(primary_candidates, start=1):
            hint = self.phonemes_to_grapheme_hint(phonemes)
            score_text = ""
            if primary_scores and idx - 1 < len(primary_scores):
                score_text = f" | score={primary_scores[idx-1]:.4f}"
            if mode == "roman":
                prompt_lines.append(f"{idx}. {hint}{score_text}")
            else:
                prompt_lines.append(f"{idx}. {(' '.join(phonemes))}{score_text}")
                prompt_lines.append(f"   hint: {hint}")

        prompt_lines.append("")
        prompt_lines.append("Stream B hypotheses:")
        for idx, phonemes in enumerate(secondary_candidates, start=1):
            hint = self.phonemes_to_grapheme_hint(phonemes)
            score_text = ""
            if secondary_scores and idx - 1 < len(secondary_scores):
                score_text = f" | score={secondary_scores[idx-1]:.4f}"
            if mode == "roman":
                prompt_lines.append(f"{idx}. {hint}{score_text}")
            else:
                prompt_lines.append(f"{idx}. {(' '.join(phonemes))}{score_text}")
                prompt_lines.append(f"   hint: {hint}")

        prompt = "\n".join(prompt_lines)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated[len(prompt):].strip()
            response = response.split('\n')[0].strip()
            return response

        except Exception as e:
            log_exception(LOGGER, "Refiner dual-hypothesis selection failed.")
            print(f"[LinguisticRefiner] Dual-hypothesis selection failed: {e}")
            if all_scores:
                best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
            else:
                best_idx = 0
            return " ".join(all_candidates[best_idx])


# =============================================================================
# Complete Swin-VALLR Model
# =============================================================================

class SwinVALLR(nn.Module):
    """
    Complete Swin-VALLR model for Visual Speech Recognition.

    Combines:
    - Swin-Tiny visual encoder
    - Temporal adapter
    - CTC phoneme head
    - Optional Qwen2 linguistic refiner
    """

    def __init__(self, config: SwinConfig = None, load_refiner: bool = False):
        super().__init__()
        self.config = config or SwinConfig()

        self.visual_encoder = VisualFrontEnd(self.config)
        self.temporal_adapter = TemporalAdapter(
            input_dim=self.visual_encoder.output_dim,
            hidden_dim=512,
            num_layers=2
        )
        if self.config.temporal_multiscale and self.config.temporal_multiscale_kernels:
            self.temporal_multiscale = TemporalMultiScaleFusion(
                dim=self.temporal_adapter.output_dim,
                kernels=self.config.temporal_multiscale_kernels,
                dropout=self.config.temporal_multiscale_dropout,
                gated=self.config.temporal_multiscale_gated
            )
        else:
            self.temporal_multiscale = nn.Identity()
        self.temporal_attention = TemporalConformerStack(
            dim=self.temporal_adapter.output_dim,
            num_layers=self.config.temporal_attention_layers,
            num_heads=self.config.temporal_attention_heads,
            kernel_size=self.config.temporal_attention_kernel,
            dropout=self.config.temporal_attention_dropout,
            ffn_mult=self.config.temporal_attention_ffn_mult,
            use_mhsa=self.config.temporal_attention_use_mhsa
        )
        if self.config.temporal_feature_norm:
            self.feature_norm = TemporalFeatureNormalizer(
                dim=self.temporal_adapter.output_dim,
                eps=self.config.temporal_feature_norm_eps,
                affine=self.config.temporal_feature_norm_affine
            )
        else:
            self.feature_norm = nn.Identity()
        self.phoneme_head = PhonemeHead(
            input_dim=self.temporal_adapter.output_dim,
            num_phonemes=self.config.num_phonemes
        )

        self.refiner = LinguisticRefiner() if load_refiner else None

    def extract_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract normalized temporal features for CTC."""
        features = self.visual_encoder(video)
        features = self.temporal_adapter(features)
        features = self.temporal_multiscale(features)
        features = self.temporal_attention(features)
        features = self.feature_norm(features)
        return features

    def extract_features_with_attention(
        self,
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract features with temporal attention weights for diagnostics."""
        features = self.visual_encoder(video)
        features = self.temporal_adapter(features)
        features = self.temporal_multiscale(features)
        if self.config.temporal_attention_layers > 0:
            features, attn_weights = self.temporal_attention(features, return_attn=True)
        else:
            attn_weights = []
        features = self.feature_norm(features)
        return features, attn_weights

    def forward(
        self,
        video: torch.Tensor,
        return_phoneme_logits: bool = True
    ) -> torch.Tensor:
        """
        Args:
            video: Input video (B, C, T, H, W)
            return_phoneme_logits: If True, return CTC logits; else return features

        Returns:
            Phoneme log-probabilities (B, T', num_phonemes) or features
        """
        features = self.extract_features(video)

        if return_phoneme_logits:
            logits = self.phoneme_head(features)
            return logits

        return features

    def decode_ctc(self, logits: torch.Tensor) -> List[List[str]]:
        """
        Decode CTC output to phoneme sequences.

        Args:
            logits: CTC log-probabilities (B, T, num_phonemes)

        Returns:
            List of phoneme sequences for each batch item
        """
        predictions = logits.argmax(dim=-1)

        results: List[List[str]] = []
        for seq in predictions:
            decoded: List[str] = []
            prev = -1
            for idx in seq.tolist():
                if idx != prev and idx != len(PHONEME_VOCAB) - 1:
                    decoded.append(PHONEME_VOCAB[idx])
                prev = idx
            results.append(decoded)

        return results

    @staticmethod
    def _logsumexp(a: float, b: float) -> float:
        if a == float("-inf"):
            return b
        if b == float("-inf"):
            return a
        if a < b:
            a, b = b, a
        return a + math.log1p(math.exp(b - a))

    def _ctc_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 10,
        blank_idx: Optional[int] = None
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """CTC prefix beam search for a single sample."""
        log_probs = log_probs.detach().cpu().float()
        time_steps, num_classes = log_probs.shape
        blank = blank_idx if blank_idx is not None else num_classes - 1

        beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, float("-inf"))}

        for t in range(time_steps):
            next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
            for prefix, (p_blank, p_nonblank) in beams.items():
                for c in range(num_classes):
                    p = float(log_probs[t, c])
                    if c == blank:
                        nb_blank, nb_nonblank = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        nb_blank = self._logsumexp(nb_blank, p_blank + p)
                        nb_blank = self._logsumexp(nb_blank, p_nonblank + p)
                        next_beams[prefix] = (nb_blank, nb_nonblank)
                        continue

                    new_prefix = prefix + (c,)
                    nb_blank, nb_nonblank = next_beams.get(new_prefix, (float("-inf"), float("-inf")))

                    if prefix and prefix[-1] == c:
                        nb_nonblank = self._logsumexp(nb_nonblank, p_blank + p)
                        prev_blank, prev_nonblank = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        prev_nonblank = self._logsumexp(prev_nonblank, p_nonblank + p)
                        next_beams[prefix] = (prev_blank, prev_nonblank)
                    else:
                        nb_nonblank = self._logsumexp(nb_nonblank, p_blank + p)
                        nb_nonblank = self._logsumexp(nb_nonblank, p_nonblank + p)

                    next_beams[new_prefix] = (nb_blank, nb_nonblank)

            sorted_beams = sorted(
                next_beams.items(),
                key=lambda item: self._logsumexp(item[1][0], item[1][1]),
                reverse=True
            )
            beams = dict(sorted_beams[:beam_width])

        scored: List[Tuple[Tuple[int, ...], float]] = []
        for prefix, (p_blank, p_nonblank) in beams.items():
            score = self._logsumexp(p_blank, p_nonblank)
            scored.append((prefix, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def decode_ctc_nbest(
        self,
        logits: torch.Tensor,
        beam_width: int = 10,
        nbest: int = 5
    ) -> List[List[Dict[str, object]]]:
        """
        Decode CTC output into N-best phoneme hypotheses.

        Returns:
            List of hypotheses per batch item with phonemes and log_prob.
        """
        results: List[List[Dict[str, object]]] = []
        blank_idx = len(PHONEME_VOCAB) - 1

        for sample in logits:
            beams = self._ctc_beam_search(sample, beam_width=beam_width, blank_idx=blank_idx)
            hypotheses: List[Dict[str, object]] = []
            for prefix, score in beams[:nbest]:
                phonemes = [PHONEME_VOCAB[idx] for idx in prefix if idx != blank_idx]
                hypotheses.append({
                    "phonemes": phonemes,
                    "log_prob": float(score)
                })
            results.append(hypotheses)

        return results

    def decode_ctc_with_alternatives(
        self,
        logits: torch.Tensor,
        top_k: int = 3,
        max_alternatives: int = 2,
        min_confidence: float = 0.4,
        min_margin: float = 0.08,
        accent_aware: bool = False,
        accent_max_alternatives: int = 2
    ) -> List[Dict[str, object]]:
        """Decode CTC output with alternative phonemes for uncertain positions."""
        results: List[Dict[str, object]] = []
        blank_idx = len(PHONEME_VOCAB) - 1

        for sample in logits:
            if sample.ndim != 2 or sample.shape[0] == 0:
                results.append({
                    "phonemes": [],
                    "alternatives": [],
                    "confidences": [],
                    "margins": [],
                    "uncertain_positions": []
                })
                continue

            preds = sample.argmax(dim=-1).tolist()
            segments: List[Tuple[int, List[int]]] = []
            current_label = None
            current_frames: List[int] = []

            for t, idx in enumerate(preds):
                if idx == blank_idx:
                    if current_frames:
                        segments.append((current_label, current_frames))
                        current_frames = []
                    current_label = None
                    continue
                if idx != current_label:
                    if current_frames:
                        segments.append((current_label, current_frames))
                    current_label = idx
                    current_frames = [t]
                else:
                    current_frames.append(t)

            if current_frames:
                segments.append((current_label, current_frames))

            phonemes: List[str] = []
            alternatives: List[List[str]] = []
            confidences: List[float] = []
            margins: List[float] = []

            for label_idx, frames in segments:
                segment_probs = sample[frames].exp().mean(dim=0)
                if segment_probs.numel() == 0:
                    continue
                segment_probs[blank_idx] = 0.0

                k = min(max(2, top_k), int(segment_probs.numel()))
                top_probs, top_idx = torch.topk(segment_probs, k=k)

                main_idx = int(top_idx[0].item())
                phonemes.append(PHONEME_VOCAB[main_idx])
                conf = float(top_probs[0].item())
                confidences.append(conf)
                if len(top_probs) > 1:
                    margin = float(top_probs[0].item() - top_probs[1].item())
                else:
                    margin = 1.0
                margins.append(margin)

                alt_list: List[str] = []
                for alt_idx in top_idx[1:].tolist():
                    if alt_idx == blank_idx:
                        continue
                    phoneme = PHONEME_VOCAB[int(alt_idx)]
                    if phoneme == PHONEME_VOCAB[main_idx]:
                        continue
                    if phoneme not in alt_list:
                        alt_list.append(phoneme)
                alternatives.append(alt_list[:max_alternatives])

            if not phonemes:
                results.append({
                    "phonemes": [],
                    "alternatives": [],
                    "confidences": [],
                    "margins": [],
                    "uncertain_positions": []
                })
                continue

            conf_mean = sum(confidences) / max(len(confidences), 1)
            margin_mean = sum(margins) / max(len(margins), 1)
            conf_thresh = max(min_confidence, conf_mean * 0.85)
            margin_thresh = max(min_margin, margin_mean * 0.5)

            uncertain_positions: List[int] = []
            filtered_alts: List[List[str]] = []
            for i, (conf, margin, alt_list) in enumerate(zip(confidences, margins, alternatives)):
                if conf < conf_thresh or margin < margin_thresh:
                    filtered_alts.append(alt_list)
                    if alt_list:
                        uncertain_positions.append(i)
                else:
                    filtered_alts.append([])

            if accent_aware:
                uncertain_set = set(uncertain_positions)
                for i, phoneme in enumerate(phonemes):
                    if i not in uncertain_set:
                        continue
                    accent_alts = LinguisticRefiner.ACCENT_VARIANTS.get(phoneme, [])
                    if not accent_alts:
                        continue
                    merged = list(filtered_alts[i])
                    for alt in accent_alts:
                        if alt not in merged:
                            merged.append(alt)
                        if len(merged) >= max_alternatives + accent_max_alternatives:
                            break
                    filtered_alts[i] = merged

            results.append({
                "phonemes": phonemes,
                "alternatives": filtered_alts,
                "confidences": confidences,
                "margins": margins,
                "uncertain_positions": uncertain_positions
            })

        return results

    def _temporal_smooth_log_probs(
        self,
        log_probs: torch.Tensor,
        kernel_size: int = 3
    ) -> torch.Tensor:
        if kernel_size <= 1:
            return log_probs
        if log_probs.ndim == 3:
            dim = 1
        else:
            dim = 0
        probs = log_probs.exp()
        smoothed = probs
        window = 1
        for shift in range(1, kernel_size // 2 + 1):
            smoothed = smoothed + safe_roll(probs, shift, dim) + safe_roll(probs, -shift, dim)
            window += 2
        smoothed = smoothed / float(window)
        smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return smoothed.clamp_min(1e-8).log()

    def _compute_reliability_tokens(
        self,
        log_probs: torch.Tensor,
        blank_idx: int,
        chunk_frames: int = 10
    ) -> List[str]:
        if log_probs.ndim != 2 or log_probs.shape[0] == 0:
            return []
        probs = log_probs.exp()
        max_probs = probs.max(dim=-1).values
        entropy = -(probs * log_probs).sum(dim=-1)
        blank_probs = probs[:, blank_idx]

        conf_base = max_probs.mean().item()
        ent_base = entropy.mean().item()
        blank_base = blank_probs.mean().item()

        conf_thresh = max(0.3, conf_base * 0.9)
        ent_thresh = ent_base * 1.1
        blank_thresh = blank_base * 1.1

        tokens = []
        chunk_frames = max(1, int(chunk_frames))
        for start in range(0, log_probs.shape[0], chunk_frames):
            end = min(log_probs.shape[0], start + chunk_frames)
            conf = max_probs[start:end].mean().item()
            ent = entropy[start:end].mean().item()
            blank = blank_probs[start:end].mean().item()
            if conf >= conf_thresh and ent <= ent_thresh and blank <= blank_thresh:
                tokens.append("C")
            else:
                tokens.append("N")
        return tokens

    def decode_ctc_dual_hypotheses(
        self,
        logits: torch.Tensor,
        beam_width: int = 10,
        nbest: int = 5,
        smooth_kernel: int = 3,
        chunk_frames: int = 10
    ) -> List[Dict[str, object]]:
        """Decode dual hypothesis sets with reliability tokens."""
        blank_idx = len(PHONEME_VOCAB) - 1
        primary = self.decode_ctc_nbest(logits, beam_width=beam_width, nbest=nbest)
        smooth_logits = self._temporal_smooth_log_probs(logits, kernel_size=smooth_kernel)
        secondary = self.decode_ctc_nbest(smooth_logits, beam_width=beam_width, nbest=nbest)

        results: List[Dict[str, object]] = []
        for idx in range(len(primary)):
            primary_tokens = self._compute_reliability_tokens(
                logits[idx],
                blank_idx=blank_idx,
                chunk_frames=chunk_frames
            )
            secondary_tokens = self._compute_reliability_tokens(
                smooth_logits[idx],
                blank_idx=blank_idx,
                chunk_frames=chunk_frames
            )
            results.append({
                "primary": primary[idx],
                "secondary": secondary[idx],
                "primary_reliability": primary_tokens,
                "secondary_reliability": secondary_tokens
            })

        return results

    def transcribe(
        self,
        video: torch.Tensor,
        language_hint: Optional[str] = None,
        refiner_mode: str = "phoneme"
    ) -> List[str]:
        """
        Full transcription pipeline: video -> text.

        Args:
            video: Input video (B, C, T, H, W)

        Returns:
            List of transcribed text strings
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(video)
            phoneme_sequences = self.decode_ctc(logits)

            if self.refiner is not None:
                texts = [
                    self.refiner.refine_phonemes(
                        seq,
                        language_hint=language_hint,
                        mode=refiner_mode
                    )
                    for seq in phoneme_sequences
                ]
            else:
                texts = [' '.join(seq) for seq in phoneme_sequences]

        return texts

    def load_refiner(self, adapter_path: Optional[str] = None):
        """Load the linguistic refiner model."""
        if self.refiner is None:
            self.refiner = LinguisticRefiner()
        LOGGER.info("Loading linguistic refiner.")
        self.refiner.load(get_device(), adapter_path=adapter_path)


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(load_refiner: bool = False, config: Optional[SwinConfig] = None) -> SwinVALLR:
    """
    Factory function to create Swin-VALLR model.

    Args:
        load_refiner: Whether to load Qwen2 refiner immediately

    Returns:
        Initialized SwinVALLR model on appropriate device
    """
    config = config or SwinConfig()
    model = SwinVALLR(config, load_refiner=load_refiner)
    model = to_device(model)

    LOGGER.info("Created Swin-VALLR model. load_refiner=%s", load_refiner)
    print(f"[Model] Created Swin-VALLR with {count_parameters(model):,} parameters")

    try:
        out_dir = get_useful_dir("model_architecture", "model_summary")
        save_text(out_dir / "model.txt", str(model))
        save_json(out_dir / "config.json", asdict(config))
        save_json(out_dir / "param_count.json", {"trainable": count_parameters(model)})
        param_stats = {}
        for name, param in model.named_parameters():
            data = param.detach().float()
            param_stats[name] = {
                "shape": list(data.shape),
                "mean": float(data.mean().item()),
                "std": float(data.std().item()),
                "min": float(data.min().item()),
                "max": float(data.max().item())
            }
        save_json(out_dir / "parameter_stats.json", param_stats)
    except Exception:
        log_exception(LOGGER, "Failed to save model artifacts.")

    return model


if __name__ == "__main__":
    # Test model creation
    LOGGER.info("Running model_architecture self-test.")
    log_system_info(LOGGER)
    print("Creating Swin-VALLR model...")
    model = create_model(load_refiner=False)

    # Test forward pass with dummy input
    device = get_device()
    dummy_video = torch.randn(2, 3, 50, 96, 96, device=device)

    print(f"\nInput shape: {dummy_video.shape}")

    with torch.no_grad():
        logits = model(dummy_video)

    print(f"Output shape: {logits.shape}")
    print(f"Expected: (2, T', 40) where T' ƒ%^ {50 // 2 // 4}")  # temporal_patch=2, 2 conv layers with stride 2

    # Test CTC decoding
    phonemes = model.decode_ctc(logits)
    print(f"\nSample decoded phonemes: {phonemes[0][:10]}...")
