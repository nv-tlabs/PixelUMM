# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local JSONL dataset for PixelUMM toy training examples.

Records use relative media paths within the selected toy-data directory.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as transforms_f
from PIL import Image

from .pixelumm_smart_resize import smart_resize_pixels
from .pixelumm_video_decoder import decode_pixelumm_video


_CHATML_START = "<|im_start|>"
_CHATML_END = "<|im_end|>"
_SYSTEM_PROMPT = "You are a helpful assistant."
_T2I_INSTRUCTION = "Generate a high-quality image based on the following description:"
_T2V_INSTRUCTION = "Generate a high-quality video based on the following description:"
_I2T_INSTRUCTION = "Describe this image in detail."
_V2T_INSTRUCTION = "Describe this video in detail."
_TASKS = {"t2i", "t2v", "image_vlm", "video_vlm"}
_CONDITIONED_GENERATION_CFG_GROUP = "assistant_generation_context_v1"
_CONDITIONED_GENERATION_EMPTY_CONTEXT = (
    "<|im_start|>user\n<|im_end|>\n"
    "<|im_start|>assistant\n<|im_end|>"
)


def _encode(tokenizer, text: str) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(text)
    return [int(value) for value in values]


def _span(
    input_ids: list[int],
    *,
    loss_mask: list[bool] | None = None,
    cfg_dropout_input_ids: list[int] | None = None,
) -> dict[str, Any]:
    input_ids = [int(value) for value in input_ids]
    loss_mask = (
        [False] * len(input_ids)
        if loss_mask is None
        else [bool(value) for value in loss_mask]
    )
    if len(loss_mask) != len(input_ids):
        raise ValueError("text span input_ids/loss_mask length mismatch")
    result = {
        "input_ids": input_ids,
        "label_ids": input_ids[1:] + ([input_ids[-1]] if input_ids else []),
        "loss_mask": loss_mask,
    }
    if cfg_dropout_input_ids is not None:
        result["cfg_dropout_input_ids"] = [
            int(value) for value in cfg_dropout_input_ids
        ]
    return result


def _text_plan(*, loss: bool) -> dict[str, Any]:
    return {
        "type": "text_token_span",
        "enable_cfg": 0,
        "loss": int(loss),
        "special_token_loss": 0,
        "special_token_label": None,
    }


def _generation_text_plan() -> dict[str, Any]:
    return {
        **_text_plan(loss=False),
        "enable_cfg": 1,
        "cfg_dropout_group": _CONDITIONED_GENERATION_CFG_GROUP,
        "cfg_dropout_prob_source": "text",
    }


def _assistant_suffix(
    tokenizer,
    *,
    instruction: str,
    answer: str,
) -> dict[str, Any]:
    prompt_ids = _encode(
        tokenizer,
        f"{instruction}{_CHATML_END}\n{_CHATML_START}assistant\n",
    )
    answer_ids = _encode(tokenizer, answer)
    end_ids = _encode(tokenizer, _CHATML_END)
    input_ids = prompt_ids + answer_ids + end_ids
    loss_mask = [False] * len(input_ids)
    supervised_end = len(prompt_ids) + len(answer_ids) + len(end_ids)
    for target_index in range(len(prompt_ids), supervised_end):
        if target_index > 0:
            loss_mask[target_index - 1] = True
    return _span(input_ids, loss_mask=loss_mask)


def _pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float().div_(127.5).sub_(1.0).contiguous()


def _resize_image(
    image: Image.Image,
    *,
    patch_size: int,
    min_pixels: int,
    max_pixels: int,
) -> torch.Tensor:
    height, width = smart_resize_pixels(
        image.height,
        image.width,
        factor=patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = transforms_f.resize(
        image.convert("RGB"),
        [height, width],
        interpolation=transforms_f.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return _pil_to_normalized_tensor(image)


def _resolve_media(manifest_path: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{manifest_path}: {field} must be a non-empty relative path")
    value = value.strip()
    if "://" in value or Path(value).is_absolute():
        raise ValueError(
            f"{manifest_path}: {field} must be local and relative, got {value!r}"
        )
    root = manifest_path.parent.resolve()
    resolved = (root / value).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{manifest_path}: {field} escapes the manifest directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{manifest_path}: {field} is missing: {resolved}")
    return resolved


class PixelUMMLocalJSONLDataset(torch.utils.data.IterableDataset):
    """Cycle one or more tiny local manifests into canonical packed samples."""

    def __init__(
        self,
        tokenizer,
        manifest_paths: list[str],
        local_rank: int = 0,
        world_size: int = 1,
        allowed_task: str = "",
        seed: int = 42,
        patch_size: int = 16,
        generation_max_pixels: int = 480 * 480,
        image_vlm_max_tokens: int = 2_560_000 // (16**2),
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.patch_size = int(patch_size)
        self.generation_max_pixels = int(generation_max_pixels)
        self.image_vlm_max_tokens = int(image_vlm_max_tokens)
        self.allowed_task = str(allowed_task).strip()
        if self.allowed_task and self.allowed_task not in _TASKS:
            raise ValueError(f"Unknown allowed_task={self.allowed_task!r}")
        if not manifest_paths:
            raise ValueError("manifest_paths must contain at least one local JSONL")
        self.records = self._load_records(manifest_paths)
        if not self.records:
            raise ValueError("local JSONL manifests contain no examples")

    def _load_records(self, manifest_paths: list[str]) -> list[dict[str, Any]]:
        records = []
        seen_ids = set()
        for raw_path in manifest_paths:
            expanded = os.path.expandvars(os.path.expanduser(str(raw_path)))
            if "$" in expanded:
                raise ValueError(f"Unresolved environment variable in manifest: {raw_path}")
            manifest_path = Path(expanded).resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Local manifest is missing: {manifest_path}")
            for line_number, line in enumerate(
                manifest_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {manifest_path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{manifest_path}:{line_number} must be an object")
                task = str(record.get("task", "")).strip()
                if task not in _TASKS:
                    raise ValueError(
                        f"{manifest_path}:{line_number} has unknown task={task!r}"
                    )
                if self.allowed_task and task != self.allowed_task:
                    raise ValueError(
                        f"{manifest_path}:{line_number} expected task "
                        f"{self.allowed_task!r}, got {task!r}"
                    )
                sample_id = str(record.get("id", "")).strip()
                if not sample_id or sample_id in seen_ids:
                    raise ValueError(
                        f"{manifest_path}:{line_number} has missing/duplicate id={sample_id!r}"
                    )
                media_field = "image" if task in {"t2i", "image_vlm"} else "video"
                record = dict(record)
                record["_media_path"] = _resolve_media(
                    manifest_path, record.get(media_field), field=media_field
                )
                seen_ids.add(sample_id)
                records.append(record)
        return records

    def set_epoch(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.epoch += 1

    def _worker_records(self) -> list[dict[str, Any]]:
        order = list(range(len(self.records)))
        random.Random(self.seed + self.epoch).shuffle(order)
        rank_order = order[self.local_rank :: max(1, self.world_size)]
        if not rank_order:
            rank_order = order
        info = torch.utils.data.get_worker_info()
        if info is not None:
            sharded = rank_order[info.id :: info.num_workers]
            if sharded:
                rank_order = sharded
        return [self.records[index] for index in rank_order]

    def __iter__(self):
        records = self._worker_records()
        while True:
            for record in records:
                yield self._to_sample(record)

    def _load_image(self, path: Path, *, understanding: bool) -> torch.Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if understanding:
            return _resize_image(
                image,
                patch_size=self.patch_size,
                min_pixels=3_136,
                max_pixels=self.image_vlm_max_tokens * self.patch_size**2,
            )
        return _resize_image(
            image,
            patch_size=self.patch_size,
            min_pixels=256**2,
            max_pixels=self.generation_max_pixels,
        )

    def _load_video(self, path: Path, *, generation: bool) -> dict[str, Any]:
        decoded = decode_pixelumm_video(
            key=path.name,
            data=path.read_bytes(),
            min_fps=1.0 if not generation else 23.0,
            max_fps=240.0 if not generation else 60.0,
            target_fps=1.0 if not generation else 24.0,
            dense_target_fps=4.0 if not generation else 24.0,
            sparse_target_fps=1.0 if not generation else 24.0,
            max_sampled_frames=96,
            sparse_max_sampled_frames=96 if not generation else 0,
            dense_representation_probability=0.0 if not generation else 1.0,
            max_raw_patch_tokens=96 * (448 // 16) ** 2,
            patch_size=self.patch_size,
            resize_multiple=self.patch_size,
            min_frame_pixels=self.patch_size**2,
            max_frame_pixels=448**2,
            temporal_patch_size=4,
            required_sampled_frames=96 if generation else 0,
            drop_incomplete_dense_tube=False,
            representation_selector_key=str(path),
        )
        if decoded is None:
            raise RuntimeError(f"Could not decode local video: {path}")
        return decoded

    def _to_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        task = record["task"]
        if task == "t2i":
            sample = self._t2i(record)
        elif task == "t2v":
            sample = self._t2v(record)
        elif task == "image_vlm":
            sample = self._image_vlm(record)
        else:
            sample = self._video_vlm(record)
        return sample

    def _generation_span(self, prompt: str, *, instruction: str) -> dict[str, Any]:
        text = (
            f"{_CHATML_START}user\n{instruction}\n{prompt}"
            f"{_CHATML_END}\n{_CHATML_START}assistant\n"
        )
        ids = _encode(self.tokenizer, text) + [int(self.tokenizer.eos_token_id)]
        return _span(
            ids,
            cfg_dropout_input_ids=_encode(
                self.tokenizer, _CONDITIONED_GENERATION_EMPTY_CONTEXT
            ),
        )

    def _t2i(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"T2I example {record['id']} has no prompt")
        image = self._load_image(record["_media_path"], understanding=False)
        text = self._generation_span(prompt, instruction=_T2I_INSTRUCTION)
        tokens = len(text["input_ids"]) + (image.shape[-2] // self.patch_size) * (
            image.shape[-1] // self.patch_size
        )
        return {
            "image_tensor_list": [image],
            "image_value_range_list": ["float_neg1_pos1"],
            "video_tensor_list": [],
            "video_value_range_list": [],
            "text_ids_list": [text],
            "sequence_plan": [
                _generation_text_plan(),
                {
                    "type": "pixel_gen",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            ],
            "num_tokens": int(tokens),
        }

    def _t2v(self, record: dict[str, Any]) -> dict[str, Any]:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"T2V example {record['id']} has no prompt")
        decoded = self._load_video(record["_media_path"], generation=True)
        video = decoded["videos"].permute(1, 0, 2, 3).float().div_(127.5).sub_(1.0)
        text = self._generation_span(prompt, instruction=_T2V_INSTRUCTION)
        tokens = len(text["input_ids"]) + (
            video.shape[0]
            // 4
            * (video.shape[-2] // self.patch_size)
            * (video.shape[-1] // self.patch_size)
        )
        return {
            "image_tensor_list": [],
            "image_value_range_list": [],
            "video_tensor_list": [video.contiguous()],
            "video_value_range_list": ["float_neg1_pos1"],
            "text_ids_list": [text],
            "sequence_plan": [
                _generation_text_plan(),
                {
                    "type": "pixel_video_gen",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "temporal_patch_size": 4,
                },
            ],
            "num_tokens": int(tokens),
        }

    def _image_vlm(self, record: dict[str, Any]) -> dict[str, Any]:
        instruction = str(record.get("instruction") or _I2T_INSTRUCTION).strip()
        answer = str(record.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"image_vlm example {record['id']} has no answer")
        image = self._load_image(record["_media_path"], understanding=True)
        prefix = _span(
            _encode(
                self.tokenizer,
                f"{_CHATML_START}system\n{_SYSTEM_PROMPT}{_CHATML_END}\n"
                f"{_CHATML_START}user\n",
            )
        )
        suffix = _assistant_suffix(
            self.tokenizer, instruction=instruction, answer=answer
        )
        tokens = sum(len(span["input_ids"]) for span in (prefix, suffix)) + (
            image.shape[-2] // self.patch_size
        ) * (image.shape[-1] // self.patch_size)
        return {
            "image_tensor_list": [image],
            "image_value_range_list": ["float_neg1_pos1"],
            "video_tensor_list": [],
            "video_value_range_list": [],
            "text_ids_list": [prefix, suffix],
            "sequence_plan": [
                _text_plan(loss=False),
                {
                    "type": "pixel_und",
                    "enable_cfg": 0,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
                _text_plan(loss=True),
            ],
            "num_tokens": int(tokens),
        }

    def _video_vlm(self, record: dict[str, Any]) -> dict[str, Any]:
        instruction = str(record.get("instruction") or _V2T_INSTRUCTION).strip()
        answer = str(record.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"video_vlm example {record['id']} has no answer")
        decoded = self._load_video(record["_media_path"], generation=False)
        frames = decoded["videos"].permute(1, 0, 2, 3).float().div_(127.5).sub_(1.0)
        source_fps = float(decoded["source_fps"])
        timestamps = [
            float(index) / source_fps for index in decoded["frame_indices"]
        ]
        pad_frames = (-len(frames)) % 4
        if pad_frames:
            frames = torch.cat([frames, frames[-1:].repeat(pad_frames, 1, 1, 1)])
            timestamps.extend([timestamps[-1]] * pad_frames)

        prefix = _span(
            _encode(
                self.tokenizer,
                f"{_CHATML_START}system\n{_SYSTEM_PROMPT}{_CHATML_END}\n"
                f"{_CHATML_START}user\n",
            )
        )
        text_ids_list = [prefix]
        sequence_plan = [_text_plan(loss=False)]
        video_tensor_list = []
        media_tokens = 0
        local_origin = timestamps[0]
        for start in range(0, len(frames), 4):
            tube = frames[start : start + 4].contiguous()
            local_timestamp = float(timestamps[start] - local_origin)
            timestamp_span = _span(
                _encode(self.tokenizer, f"<{local_timestamp:.1f} seconds>")
            )
            text_ids_list.append(timestamp_span)
            sequence_plan.append(_text_plan(loss=False))
            video_tensor_list.append(tube)
            sequence_plan.append(
                {
                    "type": "pixel_video_und",
                    "enable_cfg": 0,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "append_start_delimiter": True,
                    "include_end_delimiter": True,
                    "split_start": True,
                    "split_end": True,
                    "attention_mode": "full",
                    "temporal_patch_size": 4,
                    "video_und_mrope_mode": "sequence_hw",
                    "video_und_timestamp_mode": "tube_start",
                    "tube_timestamp_seconds": local_timestamp,
                }
            )
            media_tokens += (tube.shape[-2] // self.patch_size) * (
                tube.shape[-1] // self.patch_size
            )
        suffix = _assistant_suffix(
            self.tokenizer, instruction=instruction, answer=answer
        )
        text_ids_list.append(suffix)
        sequence_plan.append(_text_plan(loss=True))
        tokens = sum(len(span["input_ids"]) for span in text_ids_list) + media_tokens
        return {
            "image_tensor_list": [],
            "image_value_range_list": [],
            "video_tensor_list": video_tensor_list,
            "video_value_range_list": ["float_neg1_pos1"] * len(video_tensor_list),
            "text_ids_list": text_ids_list,
            "sequence_plan": sequence_plan,
            "num_tokens": int(tokens),
        }
