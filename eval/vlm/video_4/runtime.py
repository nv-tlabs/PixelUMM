# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-checkpoint runtime with R07 video preprocessing."""

from __future__ import annotations

import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms.functional as transforms_f
from PIL import Image

from eval.vlm.release_preprocess_contract import VIDEO_VLM_CONTRACT
from eval.vlm.r07_runtime import (
    validate_dcp_checkpoint,
)


@dataclass(frozen=True)
class VideoContract:
    target_fps: float
    dense_target_fps: float
    sparse_target_fps: float
    max_sampled_frames: int
    sparse_max_sampled_frames: int
    dense_representation_probability: float
    drop_incomplete_dense_tube: bool
    max_raw_patch_tokens: int
    max_frame_pixels: int
    min_frame_pixels: int
    patch_size: int
    resize_multiple: int
    temporal_patch_size: int
    timestamp_mode: str
    mrope_mode: str
    mixed_recipe: str
    recipe_names: tuple[str, ...]


@dataclass(frozen=True)
class VideoDecode:
    path: str
    source_fps: float
    source_frames: int
    metadata_source_frames: int
    metadata_frame_correction: int
    source_height: int
    source_width: int
    source_duration_seconds: float
    sampled_frame_indices: tuple[int, ...]
    sampled_frame_timestamps: tuple[float, ...]
    resized_height: int
    resized_width: int
    raw_patch_tokens: int
    packed_patch_tokens: int
    video_representation: str
    temporal_patch_size: int
    target_fps: float
    sampling_mode: str
    selection_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_source_fps(*, source_fps: float, path: Path) -> None:
    """Apply the source-FPS bounds of the frozen training recipe."""

    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise RuntimeError(f"Invalid video FPS {source_fps} ({path})")
    if source_fps < 1.0 or source_fps > 240.0:
        raise RuntimeError(
            f"Video FPS is outside the R07 training filter [1, 240]: "
            f"{source_fps} ({path})"
        )


@dataclass(frozen=True)
class VideoGeneration:
    raw_output: str
    scored_output: str
    generated_tokens: int
    max_new_tokens: int
    hit_eos: bool
    reasoning_started: bool
    reasoning_completed: bool
    reasoning_incomplete: bool
    decode: VideoDecode

    def to_dict(self) -> dict:
        result = asdict(self)
        result["decode"] = self.decode.to_dict()
        return result


def resolve_video_contract(model_args, data_args) -> VideoContract:
    """Resolve Video4 from checked-in release constants, not data recipes."""
    del data_args
    patch_size = int(model_args.pixel_token_patch_size)
    contract = VIDEO_VLM_CONTRACT
    if contract.temporal_patch_size != int(model_args.pixel_video_temporal_patch_size):
        raise RuntimeError(
            "Release/model temporal patch mismatch: "
            f"release={contract.temporal_patch_size} "
            f"model={model_args.pixel_video_temporal_patch_size}"
        )
    return VideoContract(
        target_fps=contract.target_fps,
        dense_target_fps=contract.dense_target_fps,
        sparse_target_fps=contract.sparse_target_fps,
        max_sampled_frames=contract.max_sampled_frames,
        sparse_max_sampled_frames=contract.sparse_max_sampled_frames,
        dense_representation_probability=contract.dense_representation_probability,
        drop_incomplete_dense_tube=contract.drop_incomplete_dense_tube,
        max_raw_patch_tokens=contract.max_raw_patch_tokens,
        max_frame_pixels=contract.max_frame_pixels,
        min_frame_pixels=contract.min_frame_pixels,
        patch_size=patch_size,
        resize_multiple=patch_size,
        temporal_patch_size=contract.temporal_patch_size,
        timestamp_mode=contract.timestamp_mode,
        mrope_mode=contract.mrope_mode,
        mixed_recipe="s8_f22_r07",
        recipe_names=(contract.contract_name,),
    )


def finalize_output(raw_output: str, *, generated_tokens: int, max_new_tokens: int, hit_eos: bool, decode: VideoDecode) -> VideoGeneration:
    raw_output = str(raw_output).strip()
    scored_output = raw_output
    reasoning_started = "<think>" in raw_output
    reasoning_completed = False
    reasoning_incomplete = False
    if reasoning_started:
        start = raw_output.find("<think>")
        end = raw_output.find("</think>", start + len("<think>"))
        if end < 0:
            scored_output = ""
            reasoning_incomplete = True
        else:
            reasoning_completed = True
            scored_output = raw_output[end + len("</think>") :].strip()
    answer = re.fullmatch(
        r"\s*<answer>\s*(.*?)\s*</answer>\s*",
        scored_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer is not None:
        scored_output = answer.group(1).strip()
    return VideoGeneration(
        raw_output=raw_output,
        scored_output=scored_output,
        generated_tokens=int(generated_tokens),
        max_new_tokens=int(max_new_tokens),
        hit_eos=bool(hit_eos),
        reasoning_started=reasoning_started,
        reasoning_completed=reasoning_completed,
        reasoning_incomplete=reasoning_incomplete,
        decode=decode,
    )


def validate_training_contract(model_args, data_args, training_args, contract: VideoContract) -> None:
    common_checks = {
        "visual_und": bool(training_args.visual_und),
        "und_space=pixel": training_args.und_space == "pixel",
        "layout=pixelumm_mot": data_args.packed_sequence_layout == "pixelumm_mot",
        "llm_backend=qwen3": model_args.llm_backend == "qwen3",
        "mrope=sensenova": model_args.qwen3_mrope_type == "sensenova",
        "pixel_patch_size=16": int(model_args.pixel_token_patch_size) == 16,
        "pixel_video_enabled": bool(model_args.enable_pixel_video),
        "separate_image_und_embedder": bool(model_args.pixel_separate_und_gen_embedder),
        "image_embedder=image_raw_patch_linear": (
            model_args.pixel_embedder_type == "image_raw_patch_linear"
        ),
        "separate_video_und_embedder": bool(
            model_args.pixel_video_separate_und_gen_embedder
        ),
        "video_embedder=video_raw_tube_linear": (
            model_args.pixel_video_embedder_type == "video_raw_tube_linear"
        ),
        "min_frame_pixels=256": contract.min_frame_pixels == 256,
        "temporal_patch=4": contract.temporal_patch_size == 4,
        "video_mrope=sequence_hw": contract.mrope_mode == "sequence_hw",
    }
    recipe_checks = {
        "mixed_recipe=s8_f22_r07": contract.mixed_recipe == "s8_f22_r07",
        "packed_attention=flex": model_args.packed_attention_impl == "flex",
        "target_fps=1": contract.target_fps == 1.0,
        "dense_target_fps=4": contract.dense_target_fps == 4.0,
        "sparse_target_fps=1": contract.sparse_target_fps == 1.0,
        "max_frames=96": contract.max_sampled_frames == 96,
        "sparse_max_frames=96": contract.sparse_max_sampled_frames == 96,
        "sparse_only": contract.dense_representation_probability == 0.0,
        "repeat_pad_incomplete_tube": not contract.drop_incomplete_dense_tube,
        "max_video_raw_tokens=75264": contract.max_raw_patch_tokens == 75_264,
        "max_frame_pixels=200704": contract.max_frame_pixels == 200_704,
        "timestamp=tube_start": contract.timestamp_mode == "tube_start",
    }
    checks = {**common_checks, **recipe_checks}
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Checkpoint does not satisfy R07 train-aligned Video4 contract: "
            + ", ".join(failures)
        )


class PixelUMMVideoRuntime:
    """One-GPU R07 runtime used by the Video4 lmms-eval adapter."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        code_root: str,
        device: str = "cuda:0",
        dtype: str = "bf16",
        max_new_tokens_cap: int = 64,
        decode_timeout_seconds: int = 120,
    ) -> None:
        self.code_root = Path(code_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        validate_dcp_checkpoint(self.checkpoint_path)
        self.device = torch.device(device)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.decode_timeout_seconds = int(decode_timeout_seconds)
        self.dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }[dtype.lower()]
        if str(self.code_root) not in sys.path:
            sys.path.insert(0, str(self.code_root))

        from train.model_factory import build_pixelumm_model
        from train.release_checkpoint import (
            load_checkpoint_weights,
            parse_release_arguments,
        )
        self.model_args, self.data_args, self.training_args = parse_release_arguments(
            self.code_root / "experiments" / "s8_f22_r07" / "release.yaml"
        )
        self.video_contract = resolve_video_contract(self.model_args, self.data_args)
        validate_training_contract(
            self.model_args,
            self.data_args,
            self.training_args,
            self.video_contract,
        )

        logger = logging.getLogger(
            f"pixelumm.video4.{self.checkpoint_path.name}"
        )
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        self.logger = logger

        model, tokenizer, new_token_ids = build_pixelumm_model(
            self.model_args, self.data_args, self.training_args, logger
        )
        model = model.to(dtype=self.dtype)
        model = load_checkpoint_weights(self.checkpoint_path, model, logger=logger)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        torch.cuda.empty_cache()

    @lru_cache(maxsize=1)
    def _decode_video_cached(self, path_string: str) -> tuple[tuple[Image.Image, ...], VideoDecode]:
        from torchcodec.decoders import VideoDecoder
        from data.pixelumm_video_decoder import (
            compute_video_layout,
            sample_frame_indices,
            select_video_representation,
        )

        path = Path(path_string).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark video is missing: {path}")
        decoder = VideoDecoder(str(path), num_ffmpeg_threads=1)
        total_frames = decoder.metadata.num_frames
        source_fps = decoder.metadata.average_fps
        if total_frames is None or source_fps is None:
            raise RuntimeError(f"torchcodec could not read video metadata: {path}")
        total_frames = int(total_frames)
        metadata_total_frames = total_frames
        source_fps = float(source_fps)
        if total_frames <= 0:
            raise RuntimeError(
                f"Invalid video metadata for {path}: frames={total_frames} fps={source_fps}"
            )
        validate_source_fps(
            source_fps=source_fps,
            path=path,
        )

        contract = self.video_contract
        representation_draw = math.nextafter(1.0, 0.0)
        # A few official MVBench/TVQA MP4s advertise one or more trailing
        # frames that FFmpeg/TorchCodec cannot actually decode.  Recompute the
        # representation and sampling grid against the corrected frame count
        # instead of clamping or repeating an invalid final frame.  This keeps
        # the same no-duplication sampling contract used by training.
        max_metadata_overcount = min(32, total_frames - 1)
        for metadata_frame_correction in range(max_metadata_overcount + 1):
            if metadata_frame_correction:
                total_frames = metadata_total_frames - metadata_frame_correction
                decoder = VideoDecoder(str(path), num_ffmpeg_threads=1)
            representation = select_video_representation(
                total_frames=total_frames,
                source_fps=source_fps,
                target_fps=contract.target_fps,
                dense_target_fps=contract.dense_target_fps,
                sparse_target_fps=contract.sparse_target_fps,
                dense_max_sampled_frames=contract.max_sampled_frames,
                dense_temporal_patch_size=contract.temporal_patch_size,
                sparse_max_sampled_frames=contract.sparse_max_sampled_frames,
                dense_representation_probability=contract.dense_representation_probability,
                representation_draw=representation_draw,
            )
            frame_indices, _ = sample_frame_indices(
                total_frames=total_frames,
                source_fps=source_fps,
                target_fps=representation.target_fps,
                max_sampled_frames=representation.max_sampled_frames,
                uniform_over_clip=representation.sampling_mode == "uniform_clip",
            )

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                decoder.get_frames_at, indices=list(frame_indices)
            )
            try:
                frame_batch = future.result(timeout=self.decode_timeout_seconds).data
                break
            except TimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"Video decode exceeded {self.decode_timeout_seconds}s: {path}"
                ) from exc
            except RuntimeError as exc:
                if (
                    "Requested next frame while there are no more frames left to decode"
                    not in str(exc)
                    or metadata_frame_correction == max_metadata_overcount
                ):
                    raise RuntimeError(
                        f"Video decode failed after correcting up to "
                        f"{metadata_frame_correction} metadata frames: {path}"
                    ) from exc
            finally:
                executor.shutdown(wait=False)
        else:  # pragma: no cover - the loop either decodes or raises above.
            raise AssertionError("unreachable video decode correction loop")
        if metadata_frame_correction:
            self.logger.warning(
                "Corrected over-reported video frame metadata: path=%s "
                "metadata_frames=%d decodable_frames=%d",
                path,
                metadata_total_frames,
                total_frames,
            )
        if frame_batch.ndim != 4 or int(frame_batch.shape[0]) != len(frame_indices):
            raise RuntimeError(
                f"Unexpected decoded frame shape {tuple(frame_batch.shape)} for {path}"
            )
        source_height, source_width = map(int, frame_batch.shape[-2:])
        layout = compute_video_layout(
            total_frames=total_frames,
            source_fps=source_fps,
            source_height=source_height,
            source_width=source_width,
            target_fps=representation.target_fps,
            max_sampled_frames=representation.max_sampled_frames,
            max_raw_patch_tokens=contract.max_raw_patch_tokens,
            patch_size=contract.patch_size,
            resize_multiple=contract.resize_multiple,
            min_frame_pixels=contract.min_frame_pixels,
            max_frame_pixels=contract.max_frame_pixels,
            temporal_patch_size=representation.temporal_patch_size,
            uniform_over_clip=representation.sampling_mode == "uniform_clip",
        )
        if tuple(layout.frame_indices) != tuple(frame_indices):
            raise RuntimeError("Video sampling changed between decode and layout")
        if tuple(frame_batch.shape[-2:]) != (layout.resized_height, layout.resized_width):
            frame_batch = transforms_f.resize(
                frame_batch,
                [layout.resized_height, layout.resized_width],
                interpolation=transforms_f.InterpolationMode.BICUBIC,
                antialias=True,
            )
        frame_batch = frame_batch.cpu()
        frames = tuple(
            Image.fromarray(frame.permute(1, 2, 0).numpy()).convert("RGB")
            for frame in frame_batch
        )
        timestamps = tuple(float(index) / source_fps for index in layout.frame_indices)
        decode = VideoDecode(
            path=str(path),
            source_fps=source_fps,
            source_frames=total_frames,
            metadata_source_frames=metadata_total_frames,
            metadata_frame_correction=metadata_frame_correction,
            source_height=source_height,
            source_width=source_width,
            source_duration_seconds=total_frames / source_fps,
            sampled_frame_indices=tuple(layout.frame_indices),
            sampled_frame_timestamps=timestamps,
            resized_height=layout.resized_height,
            resized_width=layout.resized_width,
            raw_patch_tokens=layout.raw_patch_tokens,
            packed_patch_tokens=layout.packed_patch_tokens,
            video_representation=representation.mode,
            temporal_patch_size=representation.temporal_patch_size,
            target_fps=representation.target_fps,
            sampling_mode=representation.sampling_mode,
            selection_reason=representation.selection_reason,
        )
        return frames, decode

    @torch.inference_mode()
    def generate(
        self,
        *,
        instruction: str,
        video_path: str,
        max_new_tokens: int,
        until: Iterable[str] = (),
    ) -> VideoGeneration:
        from train.eval_utils import _generate_v2t_one

        frames, decode = self._decode_video_cached(str(video_path))
        max_new_tokens = max(1, min(int(max_new_tokens), self.max_new_tokens_cap))
        raw_output = _generate_v2t_one(
            self.model,
            self.tokenizer,
            self.new_token_ids,
            list(frames),
            list(decode.sampled_frame_timestamps),
            str(instruction),
            self.device,
            max_new_tokens + 1,
            0.0,
            temporal_patch_size=decode.temporal_patch_size,
            timestamp_mode=self.video_contract.timestamp_mode,
        )
        token_ids = self.tokenizer.encode(raw_output, add_special_tokens=False)
        hit_eos = len(token_ids) < max_new_tokens
        generation = finalize_output(
            raw_output,
            generated_tokens=min(len(token_ids), max_new_tokens),
            max_new_tokens=max_new_tokens,
            hit_eos=hit_eos,
            decode=decode,
        )
        stop_positions = [
            generation.scored_output.find(stop)
            for stop in until
            if stop and stop in generation.scored_output
        ]
        if stop_positions:
            generation = VideoGeneration(
                **{
                    **generation.to_dict(),
                    "scored_output": generation.scored_output[: min(stop_positions)].rstrip(),
                    "decode": decode,
                }
            )
        return generation
