"""MLX affine tensor IO and lossless int4 packing for the Edge0 converter.

The runtime conventions are documented by Edge0-AI/Edge0 (Apache-2.0),
revision a38d3dae7ed9c24d44f62c455f3c7ac67345d844. No MLX runtime is required.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np


class LossyCoefficients(ValueError):
    pass


class SafeTensors:
    def __init__(self, paths: list[Path]):
        self.items = {}
        self.maps = []
        for path in paths:
            with path.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if size > 100_000_000 or size + 8 > path.stat().st_size:
                    raise ValueError(f"invalid safetensors header: {path}")
                header = json.loads(f.read(size))
            mapping = np.memmap(path, dtype=np.uint8, mode="r")
            self.maps.append(mapping)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if name in self.items:
                    raise ValueError(f"duplicate tensor: {name}")
                dtype = {"U32": "<u4", "F32": "<f4", "F16": "<f2", "BF16": "<u2"}[info["dtype"]]
                start, end = info["data_offsets"]
                if start < 0 or end < start or end + size + 8 > mapping.size:
                    raise ValueError(f"invalid tensor bounds: {name}")
                if end - start != math.prod(info["shape"]) * np.dtype(dtype).itemsize:
                    raise ValueError(f"invalid tensor size: {name}")
                data = mapping[8 + size + start:8 + size + end].view(dtype).reshape(info["shape"])
                self.items[name] = (info["dtype"], data)

    def get(self, name: str) -> np.ndarray:
        dtype, data = self.items[name]
        if dtype == "BF16":
            return (data.astype(np.uint32) << 16).view(np.float32)
        return data


def affine_codes(weight: np.ndarray, bits: int) -> np.ndarray:
    if weight.dtype != np.dtype("<u4") or bits not in (4, 8):
        raise ValueError("expected packed uint32 affine int4 or int8")
    shifts = np.arange(0, 32, bits, dtype=np.uint32)
    return ((weight[..., None] >> shifts) & ((1 << bits) - 1)).astype(np.uint8).reshape(
        *weight.shape[:-1], weight.shape[-1] * (32 // bits))


def affine_dequant(weight, scales, biases, bits, group_size=64):
    codes = affine_codes(weight, bits)
    if scales.shape != biases.shape or scales.shape != (*codes.shape[:-1], codes.shape[-1] // group_size):
        raise ValueError("affine scale/bias shape mismatch")
    return (codes.reshape(*scales.shape, group_size).astype(np.float32) * scales[..., None]
            + biases[..., None]).reshape(codes.shape)


def affine_q4_1(weight, scales, biases, group_size=64):
    """Repack codes and duplicate group scales, without dequantizing/requantizing.

    Q4_1 stores 32 codes and two fp16 coefficients. MLX group-64 BF16
    coefficients must round-trip exactly through fp16; reject otherwise.
    """
    if group_size != 64:
        raise ValueError("only affine group size 64 is supported")
    for data in (scales, biases):
        with np.errstate(over="ignore", invalid="ignore"):
            if not np.all(np.isfinite(data)) or not np.array_equal(data, data.astype(np.float16).astype(np.float32)):
                raise LossyCoefficients("affine coefficients cannot be represented exactly as Q4_1 fp16")
    codes = affine_codes(weight, 4)
    if scales.shape != biases.shape or scales.shape != (*codes.shape[:-1], codes.shape[-1] // 64):
        raise ValueError("affine scale/bias shape mismatch")
    shape = (*codes.shape[:-1], codes.shape[-1] // 32)
    blocks = codes.reshape(*shape, 32)
    packed = blocks[..., :16] | (blocks[..., 16:] << 4)
    s = np.repeat(scales, 2, axis=-1).astype("<f2").reshape(*shape, 1).view(np.uint8)
    b = np.repeat(biases, 2, axis=-1).astype("<f2").reshape(*shape, 1).view(np.uint8)
    return np.concatenate((s, b, packed), axis=-1).reshape(*codes.shape[:-1], shape[-1] * 20)


def reorder_heads(data, axis, key_heads, value_heads, head_dim):
    if key_heads == value_heads:
        return data
    axis %= data.ndim
    shape = list(data.shape)
    if value_heads % key_heads or shape[axis] != value_heads * head_dim:
        raise ValueError("invalid grouped value-head layout")
    expanded = shape[:axis] + [key_heads, value_heads // key_heads, head_dim] + shape[axis + 1:]
    return data.reshape(expanded).swapaxes(axis, axis + 1).reshape(shape)


def qwen_layout(name, data, hp, packed=False, lora_part=None):
    """Apply the upstream Qwen3.5 grouped-to-tiled value-head permutation."""
    if ".linear_attn." not in name:
        return data
    nk, nv = hp["linear_num_key_heads"], hp["linear_num_value_heads"]
    dk, dv = hp["linear_key_head_dim"], hp["linear_value_head_dim"]
    def rows(x, d):
        return reorder_heads(x, 0, nk, nv, d)
    if ".out_proj." in name:
        if lora_part != "B":
            data = reorder_heads(data, 1, nk, nv, dv // 32 * 20 if packed else dv)
    elif lora_part != "A":
        if ".in_proj_qkv." in name or ".conv1d." in name:
            start = 2 * nk * dk
            data = np.concatenate((data[:start], rows(data[start:], dv)), axis=0)
        elif ".in_proj_z." in name:
            data = rows(data, dv)
        elif any(v in name for v in (".in_proj_a.", ".in_proj_b.", ".A_log", ".dt_bias")):
            data = rows(data, 1)
    return data
