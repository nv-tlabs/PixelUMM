# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train-aligned PixelUMM checkpoint runtime used by benchmark adapters."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from eval.vlm.release_preprocess_contract import (
    IMAGE_VLM_CONTRACT,
    ImagePreprocessContract,
)
from eval.vlm.r07_runtime import (
    validate_dcp_checkpoint,
)


@dataclass(frozen=True)
class Core13Generation:
    raw_output: str
    scored_output: str
    reasoning_mode: str
    assistant_prefill: str
    generated_tokens: int
    max_new_tokens: int
    hit_eos: bool
    reasoning_started: bool
    reasoning_completed: bool
    reasoning_incomplete: bool
    answer_wrapper_stripped: bool

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_image_preprocess_contract(data_args) -> ImagePreprocessContract:
    """Return the checked-in R07 image contract without a training recipe."""

    del data_args
    return IMAGE_VLM_CONTRACT


def finalize_reasoning_output(raw_output: str, metadata: dict) -> Core13Generation:
    """Separate the scoreable answer from an optional completed think trace."""
    raw_output = str(raw_output).strip()
    reasoning_mode = str(metadata["reasoning_mode"])
    if reasoning_mode != "standard_bare":
        raise ValueError(f"Unsupported Regular21 reasoning mode: {reasoning_mode}")

    reasoning_started = "<think>" in raw_output
    reasoning_completed = False
    reasoning_incomplete = False
    scored_output = raw_output
    if reasoning_started:
        start = raw_output.find("<think>")
        end = raw_output.find("</think>", start + len("<think>"))
        if end < 0:
            # Never score a partial reasoning trace as though it were a final
            # answer. The raw response remains available in the trace JSONL.
            scored_output = ""
            reasoning_incomplete = True
        else:
            reasoning_completed = True
            scored_output = raw_output[end + len("</think>") :].strip()

    answer_match = re.fullmatch(
        r"\s*<answer>\s*(.*?)\s*</answer>\s*",
        scored_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    answer_wrapper_stripped = answer_match is not None
    if answer_match is not None:
        scored_output = answer_match.group(1).strip()

    return Core13Generation(
        raw_output=raw_output,
        scored_output=scored_output,
        reasoning_mode=reasoning_mode,
        assistant_prefill=str(metadata.get("assistant_prefill", "")),
        generated_tokens=int(metadata["generated_tokens"]),
        max_new_tokens=int(metadata["max_new_tokens"]),
        hit_eos=bool(metadata["hit_eos"]),
        reasoning_started=reasoning_started,
        reasoning_completed=reasoning_completed,
        reasoning_incomplete=reasoning_incomplete,
        answer_wrapper_stripped=answer_wrapper_stripped,
    )


class PixelUMMImageRuntime:
    """One-GPU replicated runtime for PixelUMM I2T benchmark generation."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        code_root: str,
        device: str = "cuda",
        dtype: str = "bf16",
        max_new_tokens_cap: int = 256,
        reasoning_mode: str = "standard_bare",
    ) -> None:
        self.code_root = Path(code_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        if reasoning_mode != "standard_bare":
            raise ValueError(f"Unsupported Regular21 reasoning mode: {reasoning_mode}")
        self.reasoning_mode = reasoning_mode
        self._tiny_image_upscale_count = 0
        self.dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[dtype.lower()]

        validate_dcp_checkpoint(self.checkpoint_path)
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
        self._validate_training_contract()

        logger = logging.getLogger(
            f"pixelumm.regular21.{self.checkpoint_path.name}"
        )
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self.logger = logger

        model, tokenizer, new_token_ids = build_pixelumm_model(
            self.model_args,
            self.data_args,
            self.training_args,
            logger,
        )
        self._validate_checkpoint_vocab(model, tokenizer)
        model = model.to(dtype=self.dtype)
        model = load_checkpoint_weights(self.checkpoint_path, model, logger=logger)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        torch.cuda.empty_cache()

    def _validate_training_contract(self) -> None:
        image_contract = resolve_image_preprocess_contract(self.data_args)
        self.image_resize_mode = image_contract.resize_mode
        self.min_image_pixels = image_contract.min_image_pixels
        self.max_image_tokens = image_contract.max_image_tokens
        self.image_recipe_names = image_contract.recipe_names

        failures = []
        checks = {
            "visual_und": bool(self.training_args.visual_und),
            "und_space=pixel": self.training_args.und_space == "pixel",
            "layout=pixelumm_mot": (
                self.data_args.packed_sequence_layout == "pixelumm_mot"
            ),
            "llm_backend=qwen3": self.model_args.llm_backend == "qwen3",
            "mrope=sensenova": self.model_args.qwen3_mrope_type == "sensenova",
            "patch_size=16": int(self.model_args.pixel_token_patch_size) == 16,
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(name)
        if failures:
            raise RuntimeError(
                "Checkpoint experiment does not satisfy pixelumm-core13-v1: "
                + ", ".join(failures)
            )

    def _validate_checkpoint_vocab(self, model, tokenizer) -> None:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(
            str(self.checkpoint_path / "model")
        ).read_metadata().state_dict_metadata
        expected_rows = len(tokenizer)
        keys = (
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
        )
        for key in keys:
            if key not in metadata:
                raise RuntimeError(f"Checkpoint metadata is missing {key}")
            checkpoint_rows = int(metadata[key].size[0])
            if checkpoint_rows != expected_rows:
                raise RuntimeError(
                    f"Trimmed-vocab mismatch for {key}: checkpoint={checkpoint_rows} "
                    f"tokenizer={expected_rows}"
                )
        model_rows = model.language_model.model.embed_tokens.num_embeddings
        if int(model_rows) != expected_rows:
            raise RuntimeError(
                f"Eval model vocab mismatch: model={model_rows} tokenizer={expected_rows}"
            )

    def resize_images(self, images: Iterable[Image.Image]) -> list[Image.Image]:
        from data.pixelumm_image_resize import (
            pil_to_chw_uint8,
            resize_image_native_smart_resize,
        )
        images = [image.convert("RGB") for image in images]
        if not images:
            raise ValueError("Regular21 request has no images")
        per_image_budget = max(1, self.max_image_tokens // len(images))
        resized_images = []
        for image in images:
            if min(image.size) < int(self.model_args.pixel_token_patch_size):
                self._tiny_image_upscale_count += 1
                if self._tiny_image_upscale_count <= 8:
                    self.logger.warning(
                        "Regular21 minimum-patch upscale for tiny benchmark image: "
                        f"source={image.size} patch={self.model_args.pixel_token_patch_size}"
                    )
            resized = resize_image_native_smart_resize(
                pil_to_chw_uint8(image),
                patch_size=self.model_args.pixel_token_patch_size,
                max_tokens=per_image_budget,
                min_pixels=self.min_image_pixels,
                enabled=True,
            )
            resized_images.append(
                Image.fromarray(
                    resized.permute(1, 2, 0).cpu().numpy(), mode="RGB"
                )
            )
        return resized_images

    @torch.inference_mode()
    def generate(
        self,
        *,
        instruction: str,
        images: Iterable[Image.Image],
        max_new_tokens: int,
        until: Iterable[str] = (),
        image_placeholder_policy: str = "strict",
    ) -> Core13Generation:
        from train.eval_utils import generate_vlm_interleaved

        max_new_tokens = max(
            1,
            min(int(max_new_tokens), self.max_new_tokens_cap),
        )
        raw_output, metadata = generate_vlm_interleaved(
            model=self.model,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
            images=self.resize_images(images),
            instruction=str(instruction),
            device=self.device,
            max_length=max_new_tokens,
            temperature=0.0,
            fixed_length=False,
            image_placeholder_policy=image_placeholder_policy,
            reasoning_mode=self.reasoning_mode,
            exact_max_new_tokens=True,
            return_generation_metadata=True,
        )
        generation = finalize_reasoning_output(raw_output, metadata)
        output = generation.scored_output
        stop_positions = [output.find(stop) for stop in until if stop and stop in output]
        if stop_positions:
            output = output[: min(stop_positions)].rstrip()
            generation = Core13Generation(
                **{
                    **generation.to_dict(),
                    "scored_output": output,
                }
            )
        return generation
