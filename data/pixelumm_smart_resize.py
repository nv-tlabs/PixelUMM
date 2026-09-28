# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared aspect-preserving smart-resize geometry for PixelUMM media."""

from __future__ import annotations

import math


_MAX_ASPECT_RATIO = 200.0


def smart_resize_pixels(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Resize H/W to a factor-aligned grid within explicit pixel bounds."""

    height = int(height)
    width = int(width)
    factor = int(factor)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    if height <= 0 or width <= 0 or factor <= 0:
        raise ValueError(
            f"Invalid resize geometry height={height} width={width} factor={factor}"
        )
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(f"Invalid pixel bounds min={min_pixels} max={max_pixels}")
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > _MAX_ASPECT_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be <= {_MAX_ASPECT_RATIO}, got {aspect_ratio}"
        )

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(
            factor, math.floor(height / scale / factor) * factor
        )
        resized_width = max(
            factor, math.floor(width / scale / factor) * factor
        )
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return int(resized_height), int(resized_width)
