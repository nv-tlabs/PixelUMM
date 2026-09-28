# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""GPU-side materialization of PixelUMM raw vision segments into packed-patch tensors.

The local toy packer stashes raw uint8/float pixel tensors and skips CPU
patchify/tubeify; the trainer materializes them on each rank's GPU before
model.forward.

Design notes:
    * Free function, not nn.Module — operates on the data dict, returns same
      dict mutated. No model wrapping, no extra FSDP/compile layer.
    * Ops follow input tensor device. Caller must move data to GPU first
      via SimpleCustomBatch.cuda(device).
    * Should be called OUTSIDE torch.amp.autocast so einsum output stays fp32
      (matches CPU pack_sequence output, keeps unit tests bitwise stable).
    * @torch.no_grad() to drop autograd metadata for the patchify graph.
"""

import torch


_RAW_PIXEL_KEYS = (
    "raw_pixel_videos_gen", "raw_pixel_videos_und", "raw_pixel_images_gen",
    "raw_pixel_images_und",
)
_RAW_VIDEO_TEMPORAL_KEYS = (
    "raw_pixel_video_temporal_patch_sizes_gen",
    "raw_pixel_video_temporal_patch_sizes_und",
)
_RAW_PIXEL_RANGE_KEYS = (
    "raw_pixel_image_ranges_gen",
    "raw_pixel_video_ranges_gen", "raw_pixel_video_ranges_und", "raw_pixel_image_ranges_und",
)


def _pixel_float(tensor, value_range):
    """Normalize a pixel tensor to float [-1, 1] using explicit range metadata.

    Avoid data-dependent range guessing on GPU; it silently misclassifies
    float [0,1] and forces device synchronization via tensor.max().
    """
    tensor = tensor.contiguous()
    if value_range == "uint8_0_255":
        if tensor.dtype != torch.uint8:
            raise TypeError(f"value_range={value_range} expects uint8 tensor, got dtype={tensor.dtype}")
        return tensor.float().div_(127.5).sub_(1.0)
    if value_range == "float_neg1_pos1":
        if not tensor.is_floating_point():
            raise TypeError(f"value_range={value_range} expects floating tensor, got dtype={tensor.dtype}")
        return tensor.float()
    raise ValueError(f"Unknown raw pixel value_range={value_range!r}")


def _patchify_image_gpu(image, patch_size, value_range):
    """Flatten CHW pixels into row-major spatial patches with channels last."""
    image = _pixel_float(image, value_range)
    c, h, w = image.shape
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError(f"Image shape {(c, h, w)} is not divisible by patch_size={p}")
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    return image.reshape(-1, p * p * c)


def _tubeify_video_gpu(video, temporal_patch_size, patch_size, value_range):
    """GPU equivalent of data.data_utils.tubeify."""
    video = _pixel_float(video, value_range)
    t, c, h, w = video.shape
    p = int(patch_size)
    tp = int(temporal_patch_size)
    if t % tp != 0:
        raise ValueError(f"Video frames={t} is not divisible by temporal_patch_size={tp}")
    if h % p != 0 or w % p != 0:
        raise ValueError(f"Video shape {(t, c, h, w)} is not divisible by patch_size={p}")
    video = video.reshape(t // tp, tp, c, h // p, p, w // p, p)
    video = torch.einsum("utchpwq->uhwtpqc", video)
    return video.reshape(-1, tp * p * p * c)


def _materialize_images(data, raw_key, range_key, packed_key, patch_size):
    raw_images = data.pop(raw_key, None)
    if not raw_images:
        if range_key in data:
            raise ValueError(f"{range_key} is present without {raw_key}")
        return
    value_ranges = data.pop(range_key, None)
    if packed_key in data:
        raise ValueError(f"Both {raw_key} and {packed_key} are present in data")
    if value_ranges is None or len(value_ranges) != len(raw_images):
        raise ValueError(f"{range_key} must have one entry per {raw_key} segment")
    data[packed_key] = torch.cat(
        [
            _patchify_image_gpu(image, patch_size, value_range)
            for image, value_range in zip(raw_images, value_ranges)
        ],
        dim=0,
    )


def _materialize_videos(data, raw_key, temporal_key, range_key, packed_key, patch_size):
    raw_videos = data.pop(raw_key, None)
    if not raw_videos:
        if temporal_key in data:
            raise ValueError(f"{temporal_key} is present without {raw_key}")
        if range_key in data:
            raise ValueError(f"{range_key} is present without {raw_key}")
        return
    temporal_patch_sizes = data.pop(temporal_key, None)
    value_ranges = data.pop(range_key, None)
    if packed_key in data:
        raise ValueError(f"Both {raw_key} and {packed_key} are present in data")
    if temporal_patch_sizes is None or len(temporal_patch_sizes) != len(raw_videos):
        raise ValueError(f"{temporal_key} must have one entry per {raw_key} segment")
    if value_ranges is None or len(value_ranges) != len(raw_videos):
        raise ValueError(f"{range_key} must have one entry per {raw_key} segment")
    data[packed_key] = torch.cat(
        [
            _tubeify_video_gpu(video, tp, patch_size, value_range)
            for video, tp, value_range in zip(raw_videos, temporal_patch_sizes, value_ranges)
        ],
        dim=0,
    )


@torch.no_grad()
def materialize_packed_vision_on_gpu(data, *, patch_size):
    """In-place materialize raw_pixel_* keys into packed_pixel_* keys.

    Reads raw uint8/float pixel segments produced by the local PackedDataset
    and constructs the packed-patch tensors the model expects. Inference
    prepares patches directly and does not call this helper.

    Args:
        data: dict from collate. May contain raw_pixel_* lists (each a list of
            tensors with shape [C,H,W] for images or [T,C,H,W] for videos).
        patch_size: spatial patch size (model_args.pixel_token_patch_size).
    Returns:
        The same data dict, mutated. Inserts packed_pixel_* keys, removes
        raw_pixel_* and raw_pixel_video_temporal_patch_sizes_* keys.

    Notes:
        Ops follow input tensor device; caller must move data to GPU first
        (via SimpleCustomBatch.cuda(device)) for the operations to run on GPU.
        Call outside torch.amp.autocast for fp32 output (CPU-path parity).
    """
    _materialize_images(
        data, "raw_pixel_images_und", "raw_pixel_image_ranges_und",
        "packed_pixel_patches_und", patch_size,
    )
    _materialize_images(
        data, "raw_pixel_images_gen", "raw_pixel_image_ranges_gen",
        "packed_pixel_patches_gen", patch_size,
    )
    _materialize_videos(
        data,
        "raw_pixel_videos_und",
        "raw_pixel_video_temporal_patch_sizes_und",
        "raw_pixel_video_ranges_und",
        "packed_pixel_video_patches_und",
        patch_size,
    )
    _materialize_videos(
        data,
        "raw_pixel_videos_gen",
        "raw_pixel_video_temporal_patch_sizes_gen",
        "raw_pixel_video_ranges_gen",
        "packed_pixel_video_patches_gen",
        patch_size,
    )
    leftovers = [
        key for key in _RAW_PIXEL_KEYS + _RAW_VIDEO_TEMPORAL_KEYS + _RAW_PIXEL_RANGE_KEYS
        if key in data
    ]
    if leftovers:
        raise ValueError(f"Unmaterialized raw vision keys remain in data: {leftovers}")
    return data
