# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-party preprocessing constants for the released R07 VLM suites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePreprocessContract:
    resize_mode: str = "native_smart_resize"
    min_image_pixels: int = 3_136
    max_image_tokens: int = 2_560_000 // (16**2)
    contract_name: str = "pixelumm-r07-image-vlm-v1"

    @property
    def recipe_names(self) -> tuple[str, ...]:
        """Compatibility label consumed by existing eval receipts."""

        return (self.contract_name,)


@dataclass(frozen=True)
class VideoPreprocessContract:
    target_fps: float = 1.0
    dense_target_fps: float = 4.0
    sparse_target_fps: float = 1.0
    max_sampled_frames: int = 96
    sparse_max_sampled_frames: int = 96
    dense_representation_probability: float = 0.0
    drop_incomplete_dense_tube: bool = False
    max_raw_patch_tokens: int = 96 * (448 // 16) ** 2
    max_frame_pixels: int = 448**2
    min_frame_pixels: int = 16**2
    temporal_patch_size: int = 4
    timestamp_mode: str = "tube_start"
    mrope_mode: str = "sequence_hw"
    contract_name: str = "pixelumm-r07-video4-sparse-v1"


IMAGE_VLM_CONTRACT = ImagePreprocessContract()
VIDEO_VLM_CONTRACT = VideoPreprocessContract()
