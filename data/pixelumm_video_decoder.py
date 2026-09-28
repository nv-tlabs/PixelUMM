# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PixelUMM-native video sampling and smart-resize utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import math
import re
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as transforms_F

if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder

from .pixelumm_smart_resize import smart_resize_pixels


_VIDEO_EXTENSIONS = frozenset({"mp4", "avi", "webm", "mov"})
# Keep the deterministic train-data split stable across the product rename.
_VIDEO_REPRESENTATION_COMPATIBILITY_SALT = bytes.fromhex(
    "706978656c626167656c2d766964656f2d726570726573656e746174696f6e2d763100"
)


def _open_video_decoder(*args: object, **kwargs: object) -> "VideoDecoder":
    """Import TorchCodec only when a caller actually decodes video media."""

    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Video decoding requires the torchcodec version pinned in "
            "requirements.txt and system FFmpeg libraries"
        ) from error
    return VideoDecoder(*args, **kwargs)


@dataclass(frozen=True)
class PixelUMMVideoLayout:
    """Deterministic model-input layout selected before decoding frames."""

    frame_indices: tuple[int, ...]
    conditioning_fps: float
    resized_height: int
    resized_width: int
    raw_patch_tokens: int
    packed_patch_tokens: int


class PixelUMMVideoRejectedError(ValueError):
    """Expected media rejection with a stable, machine-readable reason."""

    def __init__(self, reason: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.reason = str(reason)
        self.details = dict(details)


@dataclass(frozen=True)
class PixelUMMVideoRepresentation:
    """Per-video temporal representation selected before frame decoding."""

    mode: str
    nominal_frame_count: int
    max_sampled_frames: int
    temporal_patch_size: int
    selection_reason: str
    dense_probability: float
    selection_draw: float | None
    target_fps: float
    dense_target_fps: float
    sparse_target_fps: float
    dense_nominal_frame_count: int
    sparse_nominal_frame_count: int
    dense_grid_supported: bool
    sampling_mode: str


def stable_video_representation_draw(identity: str) -> float:
    """Map an immutable sample identity to a reproducible value in ``[0, 1)``."""

    digest = hashlib.blake2b(
        _VIDEO_REPRESENTATION_COMPATIBILITY_SALT + str(identity).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _decode_video_frames_with_timeout(
    decoder: VideoDecoder,
    frame_indices: tuple[int, ...],
    *,
    timeout_seconds: int,
) -> torch.Tensor | None:
    """Decode an explicit frame list without leaking uncancellable C++ work.

    TorchCodec frame decode is already executing in C++ when ``Future.cancel``
    returns false.  Returning at the timeout and calling ``shutdown(wait=False)``
    therefore leaves a ghost decode consuming the DataLoader worker's CPU.  The
    configured timeout is a slow-operation diagnostic threshold, not a false
    promise that a Python thread can hard-cancel native decode.
    """

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(decoder.get_frames_at, indices=list(frame_indices))
    try:
        _, unfinished = wait((future,), timeout=int(timeout_seconds))
        if unfinished:
            wait((future,))
        batch = future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    frames = batch.data
    if frames.ndim != 4 or int(frames.shape[0]) != len(frame_indices):
        raise ValueError(
            f"Unexpected torchcodec output shape {tuple(frames.shape)} for "
            f"{len(frame_indices)} requested frames"
        )
    return frames


def select_video_representation(
    *,
    total_frames: int,
    source_fps: float,
    target_fps: float,
    dense_target_fps: float | None = None,
    sparse_target_fps: float | None = None,
    dense_max_sampled_frames: int,
    dense_temporal_patch_size: int,
    sparse_max_sampled_frames: int = 0,
    dense_representation_probability: float = 1.0,
    representation_draw: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> PixelUMMVideoRepresentation:
    """Choose dense tubes or sparse frame-as-image encoding for one clip.

    ``target_fps`` is used when explicit dense/sparse cadences are absent.
    The release uses 24 FPS for generated 4-frame tubes and 1 FPS for sparse
    frame-as-image understanding. A dense route is legal only
    when the selected clip fits the dense cap and the source provides every
    requested timestamp as a unique frame; low-FPS media is never duplicated.
    A short clip that cannot provide 4 FPS but can provide 1 FPS falls back to
    the strict sparse timestamp grid.  Full-clip uniform sampling is used when
    the nominal dense grid exceeds its frame cap or the source cannot provide
    even the sparse cadence. A zero sparse cap selects dense-only behavior.
    """

    dense_max_sampled_frames = int(dense_max_sampled_frames)
    dense_temporal_patch_size = int(dense_temporal_patch_size)
    sparse_max_sampled_frames = int(sparse_max_sampled_frames)
    dense_representation_probability = float(dense_representation_probability)
    dense_target_fps = float(target_fps if dense_target_fps is None else dense_target_fps)
    sparse_target_fps = float(target_fps if sparse_target_fps is None else sparse_target_fps)
    if dense_max_sampled_frames <= 0 or dense_temporal_patch_size <= 0:
        raise ValueError(
            "Dense video frame and temporal-patch values must be positive: "
            f"frames={dense_max_sampled_frames} patch={dense_temporal_patch_size}"
        )
    if sparse_max_sampled_frames < 0:
        raise ValueError(
            "sparse_max_sampled_frames must be non-negative, got "
            f"{sparse_max_sampled_frames}"
        )
    if not 0.0 <= dense_representation_probability <= 1.0:
        raise ValueError(
            "dense_representation_probability must be in [0, 1], got "
            f"{dense_representation_probability}"
        )
    if representation_draw is not None and not 0.0 <= float(representation_draw) < 1.0:
        raise ValueError(f"representation_draw must be in [0, 1), got {representation_draw}")
    if dense_target_fps <= 0 or sparse_target_fps <= 0:
        raise ValueError(
            "dense_target_fps and sparse_target_fps must be positive: "
            f"dense={dense_target_fps} sparse={sparse_target_fps}"
        )

    # total_frames is a strict upper bound on unique sampled source frames, so
    # this returns the exact uncapped target-FPS grid without a second policy.
    dense_nominal_indices, _ = sample_frame_indices(
        total_frames=total_frames,
        source_fps=source_fps,
        target_fps=dense_target_fps,
        max_sampled_frames=int(total_frames),
        start_frame=start_frame,
        end_frame=end_frame,
    )
    sparse_nominal_indices, _ = sample_frame_indices(
        total_frames=total_frames,
        source_fps=source_fps,
        target_fps=sparse_target_fps,
        max_sampled_frames=int(total_frames),
        start_frame=start_frame,
        end_frame=end_frame,
    )
    clip_frames = (int(total_frames) if end_frame is None else int(end_frame)) - int(start_frame)
    clip_duration = clip_frames / float(source_fps)
    requested_dense_count = max(1, int(math.ceil(clip_duration * dense_target_fps - 1e-12)))
    requested_sparse_count = max(1, int(math.ceil(clip_duration * sparse_target_fps - 1e-12)))
    dense_nominal_frame_count = len(dense_nominal_indices)
    sparse_nominal_frame_count = len(sparse_nominal_indices)
    dense_grid_supported = (
        dense_nominal_frame_count == requested_dense_count
        and dense_nominal_frame_count >= dense_temporal_patch_size
    )
    sparse_grid_supported = sparse_nominal_frame_count == requested_sparse_count

    def sparse(reason: str) -> PixelUMMVideoRepresentation:
        return PixelUMMVideoRepresentation(
            mode="sparse_frame_image",
            nominal_frame_count=sparse_nominal_frame_count,
            max_sampled_frames=sparse_max_sampled_frames,
            temporal_patch_size=1,
            selection_reason=reason,
            dense_probability=dense_representation_probability,
            selection_draw=(
                float(representation_draw) if representation_draw is not None else None
            ),
            target_fps=sparse_target_fps,
            dense_target_fps=dense_target_fps,
            sparse_target_fps=sparse_target_fps,
            dense_nominal_frame_count=dense_nominal_frame_count,
            sparse_nominal_frame_count=sparse_nominal_frame_count,
            dense_grid_supported=dense_grid_supported,
            sampling_mode=(
                "uniform_clip"
                if reason == "dense_grid_above_cap" or not sparse_grid_supported
                else "timestamp_grid"
            ),
        )

    # Decide the long-clip fallback from the requested dense cadence, not from
    # the number of unique frames a low-FPS source happens to provide.
    if sparse_max_sampled_frames > 0 and requested_dense_count > dense_max_sampled_frames:
        return sparse("dense_grid_above_cap")
    if sparse_max_sampled_frames > 0 and not dense_grid_supported:
        return sparse("dense_grid_unavailable")
    if sparse_max_sampled_frames > 0 and dense_representation_probability < 1.0:
        if representation_draw is None and dense_representation_probability not in {0.0, 1.0}:
            raise ValueError(
                "A deterministic representation_draw is required when short-video "
                "dense_representation_probability is strictly between 0 and 1"
            )
        draw = float(representation_draw or 0.0)
        if draw >= dense_representation_probability:
            return sparse("short_probability_sparse")
    return PixelUMMVideoRepresentation(
        mode=(
            f"dense_tube_t{dense_temporal_patch_size}"
            if dense_temporal_patch_size > 1
            else "frame_image"
        ),
        nominal_frame_count=dense_nominal_frame_count,
        max_sampled_frames=dense_max_sampled_frames,
        temporal_patch_size=dense_temporal_patch_size,
        selection_reason=(
            "short_probability_dense"
            if sparse_max_sampled_frames > 0 and dense_representation_probability < 1.0
            else "dense_default"
        ),
        dense_probability=dense_representation_probability,
        selection_draw=(
            float(representation_draw) if representation_draw is not None else None
        ),
        target_fps=dense_target_fps,
        dense_target_fps=dense_target_fps,
        sparse_target_fps=sparse_target_fps,
        dense_nominal_frame_count=dense_nominal_frame_count,
        sparse_nominal_frame_count=sparse_nominal_frame_count,
        dense_grid_supported=dense_grid_supported,
        sampling_mode="timestamp_grid",
    )


def sample_frame_indices(
    *,
    total_frames: int,
    source_fps: float,
    target_fps: float,
    max_sampled_frames: int,
    start_frame: int = 0,
    end_frame: int | None = None,
    align_to_multiple: int = 1,
    uniform_over_clip: bool = False,
) -> tuple[tuple[int, ...], float]:
    """Sample a strict timestamp grid, uniformly thinning only above the cap.

    A nominal 1 FPS contract means timestamps ``0s, 1s, 2s, ...`` rather than
    choosing ``round(duration)`` frames with an endpoint-inclusive linspace.
    This preserves the requested cadence for short/medium clips.  Only when
    the timestamp grid exceeds ``max_sampled_frames`` do we uniformly thin the
    grid while retaining full-clip coverage. Temporal-tube divisibility is a
    later padding concern and must never remove valid sampled frames here.
    """

    total_frames = int(total_frames)
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    max_sampled_frames = int(max_sampled_frames)
    align_to_multiple = int(align_to_multiple)
    start_frame = int(start_frame)
    end_frame = total_frames if end_frame is None else int(end_frame)
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"source_fps must be positive and finite, got {source_fps}")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"target_fps must be positive and finite, got {target_fps}")
    if max_sampled_frames <= 0:
        raise ValueError(f"max_sampled_frames must be positive, got {max_sampled_frames}")
    if align_to_multiple <= 0:
        raise ValueError(f"align_to_multiple must be positive, got {align_to_multiple}")
    if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}) for total_frames={total_frames}"
        )

    available_frames = end_frame - start_frame
    duration = available_frames / source_fps
    if uniform_over_clip:
        count = min(available_frames, max_sampled_frames)
        indices_tensor = torch.linspace(
            start_frame,
            end_frame - 1,
            count,
            dtype=torch.float64,
        ).round().to(torch.int64)
    else:
        sample_period = 1.0 / target_fps
        requested_timestamps = torch.arange(
            0.0,
            duration,
            sample_period,
            dtype=torch.float64,
        )
        if requested_timestamps.numel() == 0:
            requested_timestamps = torch.zeros(1, dtype=torch.float64)
        # The timestamp grid is mathematically exact at equal source/target
        # rates, but binary floating point can place values such as 7/24 * 24
        # infinitesimally below 7.  Flooring those values silently duplicates
        # the preceding frame (96 @ 24 FPS previously collapsed to 85 unique
        # indices).  A tiny boundary tolerance preserves the intended integer
        # grid without moving any real sub-frame timestamp across a boundary.
        indices_tensor = torch.floor(
            requested_timestamps * source_fps + 1e-9
        ).to(torch.int64)
        indices_tensor = (indices_tensor + start_frame).clamp(
            min=start_frame,
            max=end_frame - 1,
        )
    indices_tensor = torch.unique_consecutive(indices_tensor)
    if indices_tensor.numel() > max_sampled_frames:
        keep = torch.linspace(
            0,
            indices_tensor.numel() - 1,
            max_sampled_frames,
        ).round().to(torch.int64)
        indices_tensor = indices_tensor[keep]
    if align_to_multiple > 1 and indices_tensor.numel() >= align_to_multiple:
        aligned = (indices_tensor.numel() // align_to_multiple) * align_to_multiple
        indices_tensor = indices_tensor[:aligned]
    indices = indices_tensor.tolist()
    sampled_frames = len(indices)
    conditioning_fps = sampled_frames / duration
    return tuple(int(index) for index in indices), float(conditioning_fps)


def compute_video_layout(
    *,
    total_frames: int,
    source_fps: float,
    source_height: int,
    source_width: int,
    target_fps: float,
    max_sampled_frames: int,
    max_raw_patch_tokens: int,
    patch_size: int,
    resize_multiple: int,
    min_frame_pixels: int,
    max_frame_pixels: int | None = None,
    temporal_patch_size: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    align_to_temporal_patch: bool = False,
    uniform_over_clip: bool = False,
) -> PixelUMMVideoLayout:
    """Resolve sampled frames and H/W under a valid-frame raw-patch budget.

    Temporal repeat padding changes the number of packed tube tokens, but must
    not lower the spatial resolution selected for the valid sampled frames.
    """

    max_raw_patch_tokens = int(max_raw_patch_tokens)
    patch_size = int(patch_size)
    resize_multiple = int(resize_multiple)
    min_frame_pixels = int(min_frame_pixels)
    temporal_patch_size = int(temporal_patch_size)
    if (
        max_raw_patch_tokens <= 0
        or patch_size <= 0
        or resize_multiple <= 0
        or temporal_patch_size <= 0
    ):
        raise ValueError(
            "PixelUMM video token, patch, resize, and temporal values must be positive: "
            f"raw_tokens={max_raw_patch_tokens} patch={patch_size} "
            f"multiple={resize_multiple} "
            f"temporal_patch_size={temporal_patch_size}"
        )
    if resize_multiple % patch_size != 0:
        raise ValueError(
            f"resize_multiple={resize_multiple} must be divisible by patch_size={patch_size}"
        )
    if min_frame_pixels <= 0:
        raise ValueError(f"min_frame_pixels must be positive, got {min_frame_pixels}")

    frame_indices, conditioning_fps = sample_frame_indices(
        total_frames=total_frames,
        source_fps=source_fps,
        target_fps=target_fps,
        max_sampled_frames=max_sampled_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        align_to_multiple=(temporal_patch_size if align_to_temporal_patch else 1),
        uniform_over_clip=uniform_over_clip,
    )
    num_valid_frames = len(frame_indices)
    temporal_groups = math.ceil(num_valid_frames / temporal_patch_size)
    per_frame_max_pixels = max_raw_patch_tokens * patch_size**2 // num_valid_frames
    if max_frame_pixels is not None:
        if int(max_frame_pixels) <= 0:
            raise ValueError(
                f"max_frame_pixels must be positive when set, got {max_frame_pixels}"
            )
        per_frame_max_pixels = min(per_frame_max_pixels, int(max_frame_pixels))
    if per_frame_max_pixels < min_frame_pixels:
        raise ValueError(
            "Video frame cap and minimum spatial budget exceed the total PixelUMM budget: "
            f"frames={num_valid_frames} temporal_groups={temporal_groups} "
            f"per_frame_max_pixels={per_frame_max_pixels} "
            f"min_frame_pixels={min_frame_pixels} "
            f"max_raw_tokens={max_raw_patch_tokens}"
        )
    resized_height, resized_width = smart_resize_pixels(
        int(source_height),
        int(source_width),
        factor=resize_multiple,
        min_pixels=min_frame_pixels,
        max_pixels=per_frame_max_pixels,
    )
    spatial_patch_tokens = (
        (int(resized_height) // patch_size) * (int(resized_width) // patch_size)
    )
    raw_patch_tokens = num_valid_frames * spatial_patch_tokens
    packed_patch_tokens = temporal_groups * spatial_patch_tokens
    if raw_patch_tokens > max_raw_patch_tokens:
        raise RuntimeError(
            "PixelUMM smart resize exceeded its raw-patch budget: "
            f"actual={raw_patch_tokens} max={max_raw_patch_tokens} "
            f"frames={num_valid_frames} temporal_groups={temporal_groups} "
            f"size={resized_height}x{resized_width}"
        )
    if packed_patch_tokens > raw_patch_tokens:
        raise RuntimeError(
            "Packed tube tokens cannot exceed valid-frame raw patch tokens: "
            f"packed={packed_patch_tokens} raw={raw_patch_tokens} "
            f"frames={num_valid_frames} temporal_patch_size={temporal_patch_size}"
        )
    return PixelUMMVideoLayout(
        frame_indices=frame_indices,
        conditioning_fps=conditioning_fps,
        resized_height=int(resized_height),
        resized_width=int(resized_width),
        raw_patch_tokens=int(raw_patch_tokens),
        packed_patch_tokens=int(packed_patch_tokens),
    )


def decode_pixelumm_video(
    *,
    key: str,
    data: bytes,
    min_fps: float,
    max_fps: float,
    target_fps: float,
    dense_target_fps: float | None = None,
    sparse_target_fps: float | None = None,
    max_sampled_frames: int,
    max_raw_patch_tokens: int,
    patch_size: int,
    resize_multiple: int,
    min_frame_pixels: int,
    max_frame_pixels: int | None = None,
    temporal_patch_size: int = 1,
    sparse_max_sampled_frames: int = 0,
    dense_representation_probability: float = 1.0,
    drop_incomplete_dense_tube: bool = False,
    representation_selector_key: str | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    required_sampled_frames: int = 0,
    num_threads: int = 1,
    decoding_timeout: int = 60,
) -> dict[str, object] | None:
    """Decode a video directly into the PixelUMM V2T model-input layout."""

    extension = re.sub(r".*[.]", "", str(key)).lower()
    if extension not in _VIDEO_EXTENSIONS:
        return None
    video_reader = _open_video_decoder(data, num_ffmpeg_threads=int(num_threads))
    total_frames = video_reader.metadata.num_frames
    source_fps = video_reader.metadata.average_fps
    if total_frames is None or source_fps is None:
        raise ValueError(
            "torchcodec missing video metadata "
            f"(num_frames={total_frames}, average_fps={source_fps})"
        )
    total_frames = int(total_frames)
    source_fps = float(source_fps)
    required_sampled_frames = int(required_sampled_frames)
    if required_sampled_frames < 0:
        raise ValueError(
            "required_sampled_frames must be non-negative, got "
            f"{required_sampled_frames}"
        )
    if source_fps < float(min_fps) or source_fps > float(max_fps):
        raise PixelUMMVideoRejectedError(
            "source_fps",
            f"Video source_fps={source_fps} is outside "
            f"[{float(min_fps)}, {float(max_fps)}]",
            source_fps=source_fps,
            min_fps=float(min_fps),
            max_fps=float(max_fps),
        )

    if (start_seconds is None) != (end_seconds is None):
        raise ValueError("start_seconds and end_seconds must be provided together")
    start_frame = 0
    end_frame = total_frames
    if start_seconds is not None:
        start = max(0.0, float(start_seconds))
        end = min(total_frames / source_fps, float(end_seconds))
        if not start < end:
            raise ValueError(
                f"invalid video clip [{start}, {end}] for {total_frames} "
                f"frames at {source_fps} FPS"
            )
        start_frame = min(total_frames - 1, int(math.floor(start * source_fps)))
        end_frame = min(
            total_frames,
            max(start_frame + 1, int(math.ceil(end * source_fps))),
        )

    representation_draw = (
        stable_video_representation_draw(representation_selector_key)
        if 0.0 < float(dense_representation_probability) < 1.0
        and representation_selector_key is not None
        else None
    )
    representation = select_video_representation(
        total_frames=total_frames,
        source_fps=source_fps,
        target_fps=target_fps,
        dense_target_fps=dense_target_fps,
        sparse_target_fps=sparse_target_fps,
        dense_max_sampled_frames=max_sampled_frames,
        dense_temporal_patch_size=temporal_patch_size,
        sparse_max_sampled_frames=sparse_max_sampled_frames,
        dense_representation_probability=dense_representation_probability,
        representation_draw=representation_draw,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    frame_indices, _ = sample_frame_indices(
        total_frames=total_frames,
        source_fps=source_fps,
        target_fps=representation.target_fps,
        max_sampled_frames=representation.max_sampled_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        align_to_multiple=(
            representation.temporal_patch_size
            if drop_incomplete_dense_tube and representation.temporal_patch_size > 1
            else 1
        ),
        uniform_over_clip=representation.sampling_mode == "uniform_clip",
    )

    # This is the authoritative fixed-window filter.  It uses the same
    # source metadata, clip bounds, representation policy, timestamp grid,
    # and temporal alignment as the real decode, but runs before any frame
    # payload is materialized.  Short dense samples therefore never pay the
    # expensive get_frames_at/decode cost, while acceptance semantics remain
    # exactly identical to the former post-decode shape check.
    sampled_frame_count = len(frame_indices)
    if (
        required_sampled_frames
        and sampled_frame_count != required_sampled_frames
    ):
        raise PixelUMMVideoRejectedError(
            "sampled_frame_count",
            "PixelUMM fixed-window sample has "
            f"{sampled_frame_count} sampled frames; required "
            f"{required_sampled_frames}",
            sampled_frame_count=sampled_frame_count,
            required_sampled_frames=required_sampled_frames,
            total_frames=total_frames,
            source_fps=source_fps,
            clip_start_frame=start_frame,
            clip_end_frame=end_frame,
        )

    frames = _decode_video_frames_with_timeout(
        video_reader,
        frame_indices,
        timeout_seconds=int(decoding_timeout),
    )
    if frames is None:
        return None
    source_height = int(frames.shape[-2])
    source_width = int(frames.shape[-1])
    layout = compute_video_layout(
        total_frames=total_frames,
        source_fps=source_fps,
        source_height=source_height,
        source_width=source_width,
        target_fps=representation.target_fps,
        max_sampled_frames=representation.max_sampled_frames,
        max_raw_patch_tokens=max_raw_patch_tokens,
        patch_size=patch_size,
        resize_multiple=resize_multiple,
        min_frame_pixels=min_frame_pixels,
        max_frame_pixels=max_frame_pixels,
        temporal_patch_size=representation.temporal_patch_size,
        start_frame=start_frame,
        end_frame=end_frame,
        align_to_temporal_patch=bool(drop_incomplete_dense_tube),
        uniform_over_clip=representation.sampling_mode == "uniform_clip",
    )
    if tuple(frames.shape[-2:]) != (layout.resized_height, layout.resized_width):
        frames = transforms_F.resize(
            frames,
            [layout.resized_height, layout.resized_width],
            interpolation=transforms_F.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return {
        "videos": frames.permute(1, 0, 2, 3).contiguous(),
        "source_fps": source_fps,
        "conditioning_fps": layout.conditioning_fps,
        "frame_indices": layout.frame_indices,
        "original_num_frames": total_frames,
        "source_height": source_height,
        "source_width": source_width,
        "source_duration_seconds": total_frames / source_fps,
        "clip_start_frame": start_frame,
        "clip_end_frame": end_frame,
        "clip_start_seconds": start_frame / source_fps,
        "clip_end_seconds": end_frame / source_fps,
        "sampling_mode": representation.sampling_mode,
        "video_representation": representation.mode,
        "nominal_target_fps_frame_count": representation.nominal_frame_count,
        "dense_nominal_frame_count": representation.dense_nominal_frame_count,
        "sparse_nominal_frame_count": representation.sparse_nominal_frame_count,
        "dense_grid_supported": representation.dense_grid_supported,
        "dense_max_sampled_frames": int(max_sampled_frames),
        "sparse_max_sampled_frames": int(sparse_max_sampled_frames),
        "dense_representation_probability": representation.dense_probability,
        "representation_selection_draw": representation.selection_draw,
        "representation_selection_reason": representation.selection_reason,
        "temporal_patch_size": representation.temporal_patch_size,
        "drop_incomplete_dense_tube": bool(drop_incomplete_dense_tube),
        "target_fps": representation.target_fps,
        "dense_target_fps": representation.dense_target_fps,
        "sparse_target_fps": representation.sparse_target_fps,
        "raw_patch_tokens": layout.raw_patch_tokens,
        "packed_patch_tokens": layout.packed_patch_tokens,
    }
