# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PixelUMM-owned image conversion and aspect-preserving resize helpers."""

from __future__ import annotations

from typing import Any

import torch

from data.pixelumm_smart_resize import smart_resize_pixels


def pil_to_chw_uint8(image: Any) -> torch.Tensor:
    """Convert a PIL-compatible image to contiguous RGB ``uint8`` CHW."""

    if image.mode != "RGB":
        image = image.convert("RGB")
    return (
        torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        .view(image.height, image.width, 3)
        .permute(2, 0, 1)
        .contiguous()
    )


def resize_image_to_exact_hw(
    image: torch.Tensor,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """Resize one CHW tensor with the released bicubic-antialias contract."""

    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target H/W must be positive, got {target_hw!r}")
    if (int(image.shape[-2]), int(image.shape[-1])) == (target_h, target_w):
        return image.contiguous()
    resized = torch.nn.functional.interpolate(
        image.float().unsqueeze(0),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    if image.dtype == torch.uint8:
        return resized.round().clamp_(0, 255).to(torch.uint8).contiguous()
    return resized.to(dtype=image.dtype).contiguous()


def resize_image_native_smart_resize(
    image: torch.Tensor,
    *,
    patch_size: int,
    max_tokens: int | None,
    min_pixels: int,
    enabled: bool,
) -> torch.Tensor:
    """Preserve native aspect ratio under explicit raw-pixel bounds."""

    if not enabled:
        return image.contiguous()
    if max_tokens is None or int(max_tokens) <= 0:
        raise ValueError("native_smart_resize requires a positive image-token budget")
    max_pixels = int(max_tokens) * int(patch_size) * int(patch_size)
    min_pixels = int(min_pixels)
    if min_pixels <= 0:
        raise ValueError("native_smart_resize requires positive min_image_pixels")
    target_hw = smart_resize_pixels(
        int(image.shape[-2]),
        int(image.shape[-1]),
        factor=int(patch_size),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resize_image_to_exact_hw(image, target_hw)
