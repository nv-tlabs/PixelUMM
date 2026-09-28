#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run PixelUMM inference directly from a released checkpoint.

The entrypoint is deliberately independent of experiment manifests and data
recipes.  It supports the four released surfaces: T2I, T2V, image VLM, and
video VLM.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from PIL import Image

from data.pixelumm_smart_resize import smart_resize_pixels
from data.pixelumm_video_decoder import decode_pixelumm_video
from eval.vlm.release_preprocess_contract import IMAGE_VLM_CONTRACT, VIDEO_VLM_CONTRACT
from train.eval_utils import (
    _generate_t2i_one,
    _generate_t2v_one,
    _generate_v2t_one,
    generate_vlm_interleaved,
    save_video_mp4,
)
from train.release_checkpoint import (
    DEFAULT_RELEASE_CONFIG,
    load_released_model,
    parse_release_arguments,
)


def _positive_multiple(value: str, *, name: str, multiple: int = 16) -> int:
    value = int(value)
    if value <= 0 or value % multiple:
        raise argparse.ArgumentTypeError(
            f"{name} must be a positive multiple of {multiple}; got {value}"
        )
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference from a complete released PixelUMM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_RELEASE_CONFIG))
    parser.add_argument(
        "--task",
        required=True,
        choices=("t2i", "t2v", "image-vlm", "video-vlm"),
    )
    parser.add_argument("--prompt", default="")
    negative = parser.add_mutually_exclusive_group()
    negative.add_argument("--negative-prompt", default="", help="Explicit T2V CFG reference prompt")
    negative.add_argument("--negative-prompt-file", help="UTF-8 file containing the T2V CFG reference prompt")
    parser.add_argument("--image", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--llm-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sampler", choices=("unipc", "dpm-solver"), default=None,
                        help="T2V: UniPC (default) or DPM-Solver++; T2I: DPM-Solver++ only")
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Disable the default Cosmos content checks for T2V inference",
    )
    parser.add_argument(
        "--cosmos-guardrails-python",
        help="Override the separate Cosmos guardrail Python interpreter",
    )
    args = parser.parse_args()

    args.height = _positive_multiple(str(args.height), name="height")
    args.width = _positive_multiple(str(args.width), name="width")
    if args.frames <= 0 or args.frames % 4:
        parser.error("--frames must be a positive multiple of 4")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.task == "t2i" and args.steps is not None and args.steps < 3:
        parser.error("T2I DPM-Solver requires --steps >= 3")
    if args.task == "t2i" and args.sampler not in (None, "dpm-solver"):
        parser.error("T2I supports only --sampler dpm-solver")
    if args.task not in ("t2i", "t2v") and args.sampler is not None:
        parser.error("--sampler applies only to t2i/t2v")
    if args.task == "t2v" and args.sampler == "dpm-solver" and args.steps is not None and args.steps < 3:
        parser.error("T2V DPM-Solver requires --steps >= 3")
    if not args.prompt.strip():
        parser.error("--prompt is required for every task")
    if args.task != "t2v" and (args.negative_prompt or args.negative_prompt_file):
        parser.error("negative prompts are currently supported only for --task t2v")
    if args.negative_prompt_file:
        try:
            args.negative_prompt = Path(args.negative_prompt_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"cannot read --negative-prompt-file: {exc}")
        if not args.negative_prompt:
            parser.error("--negative-prompt-file must not be empty")
    if args.task == "image-vlm" and not args.image:
        parser.error("--image is required for image-vlm")
    if args.task == "video-vlm" and not args.video:
        parser.error("--video is required for video-vlm")
    if args.task != "t2v" and (args.no_guardrails or args.cosmos_guardrails_python):
        parser.error("Guardrail options currently apply only to --task t2v")
    if args.no_guardrails and args.cosmos_guardrails_python:
        parser.error("--no-guardrails cannot be combined with --cosmos-guardrails-python")
    return args


def _guardrails_python(args: argparse.Namespace) -> str | None:
    if args.task != "t2v" or args.no_guardrails:
        return None
    configured = args.cosmos_guardrails_python or os.environ.get(
        "PIXELUMM_GUARDRAILS_PYTHON"
    )
    interpreter = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parent / ".venv-guardrails" / "bin" / "python"
    )
    if not interpreter.is_file():
        raise FileNotFoundError(
            f"Cosmos guardrail Python not found: {interpreter}. "
            "Install the separate environment as described in GUARDRAILS.md, "
            "set PIXELUMM_GUARDRAILS_PYTHON, or explicitly use --no-guardrails."
        )
    return str(interpreter.resolve())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resize_image_vlm(path: str, *, patch_size: int) -> Image.Image:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image input is missing: {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    target_h, target_w = smart_resize_pixels(
        image.height,
        image.width,
        factor=patch_size,
        min_pixels=IMAGE_VLM_CONTRACT.min_image_pixels,
        max_pixels=IMAGE_VLM_CONTRACT.max_image_tokens * patch_size**2,
    )
    return transforms_f.resize(
        image,
        [target_h, target_w],
        interpolation=transforms_f.InterpolationMode.BICUBIC,
        antialias=True,
    )


def _decode_video_vlm(path: str, *, patch_size: int) -> tuple[list[Image.Image], list[float], int]:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video input is missing: {video_path}")
    decoded = decode_pixelumm_video(
        key=video_path.name,
        data=video_path.read_bytes(),
        min_fps=1.0,
        max_fps=240.0,
        target_fps=VIDEO_VLM_CONTRACT.target_fps,
        dense_target_fps=VIDEO_VLM_CONTRACT.dense_target_fps,
        sparse_target_fps=VIDEO_VLM_CONTRACT.sparse_target_fps,
        max_sampled_frames=VIDEO_VLM_CONTRACT.max_sampled_frames,
        sparse_max_sampled_frames=VIDEO_VLM_CONTRACT.sparse_max_sampled_frames,
        dense_representation_probability=VIDEO_VLM_CONTRACT.dense_representation_probability,
        max_raw_patch_tokens=VIDEO_VLM_CONTRACT.max_raw_patch_tokens,
        patch_size=patch_size,
        resize_multiple=patch_size,
        min_frame_pixels=VIDEO_VLM_CONTRACT.min_frame_pixels,
        max_frame_pixels=VIDEO_VLM_CONTRACT.max_frame_pixels,
        temporal_patch_size=VIDEO_VLM_CONTRACT.temporal_patch_size,
        drop_incomplete_dense_tube=VIDEO_VLM_CONTRACT.drop_incomplete_dense_tube,
        representation_selector_key=str(video_path),
    )
    if decoded is None:
        raise RuntimeError(f"Video decoder rejected unsupported input: {video_path}")
    videos = decoded["videos"]
    frames = [
        Image.fromarray(
            videos[:, frame_index].permute(1, 2, 0).cpu().numpy()
        ).convert("RGB")
        for frame_index in range(int(videos.shape[1]))
    ]
    source_fps = float(decoded["source_fps"])
    timestamps = [float(index) / source_fps for index in decoded["frame_indices"]]
    return frames, timestamps, int(decoded["temporal_patch_size"])


def main() -> None:
    args = _parse_args()
    guardrails_python = _guardrails_python(args)
    output = Path(args.output).expanduser().resolve()
    if guardrails_python:
        if output.exists():
            raise FileExistsError(f"Refusing to replace an existing file: {output}")
        subprocess.run(
            [
                guardrails_python,
                str(Path(__file__).with_name("guardrails_cosmos.py")),
                "check-prompt",
                "--prompt-stdin",
                "--device",
                args.device,
            ],
            check=True,
            input=args.prompt,
            text=True,
        )
    model_args, _, _ = parse_release_arguments(args.config)
    if args.task == "video-vlm" and not model_args.pixel_video_separate_und_gen_embedder:
        raise ValueError(
            "This checkpoint profile has no video UND embedder; --task video-vlm "
            "requires the F22 profile and matching weights. T2V remains supported."
        )
    _seed_everything(args.seed)
    released = load_released_model(
        args.checkpoint,
        config_path=args.config,
        device=args.device,
        dtype=args.dtype,
        llm_path=args.llm_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = released.checkpoint_path
    device = next(released.model.parameters()).device

    with torch.inference_mode():
        if args.task == "t2i":
            image = _generate_t2i_one(
                released.model,
                released.tokenizer,
                released.new_token_ids,
                args.prompt,
                (args.height, args.width),
                device,
                args.steps or 50,
                3.0 if args.shift is None else args.shift,
                "dpm-solver",
                3.5 if args.cfg is None else args.cfg,
                1.0,
                (0.0, 1.0),
                "none",
            )
            image.save(output)
        elif args.task == "t2v":
            frames = _generate_t2v_one(
                released.model,
                released.tokenizer,
                released.new_token_ids,
                args.prompt,
                (args.frames, args.height, args.width),
                device,
                args.steps or (50 if args.sampler == "dpm-solver" else 35),
                10.0 if args.shift is None else args.shift,
                args.sampler or "unipc",
                6.0 if args.cfg is None else args.cfg,
                1.0,
                (0.0, 1.0),
                "none",
                video_fps=args.fps,
                negative_prompt=args.negative_prompt,
            )
            if guardrails_python:
                with tempfile.TemporaryDirectory(
                    prefix=".pixelumm-guardrails-", dir=output.parent
                ) as temporary:
                    raw = Path(temporary) / "raw.mp4"
                    saved = save_video_mp4(
                        frames, str(raw.parent), fps=args.fps, filename=raw.name
                    )
                    if saved is None:
                        raise RuntimeError("T2V inference produced no frames")
                    del frames
                    del released
                    gc.collect()
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                    subprocess.run(
                        [
                            guardrails_python,
                            str(Path(__file__).with_name("guardrails_cosmos.py")),
                            "filter-video",
                            "--input",
                            str(raw),
                            "--output",
                            str(output),
                            "--device",
                            args.device,
                        ],
                        check=True,
                    )
            else:
                saved = save_video_mp4(
                    frames, str(output.parent), fps=args.fps, filename=output.name
                )
                if saved is None:
                    raise RuntimeError("T2V inference produced no frames")
        elif args.task == "image-vlm":
            image = _resize_image_vlm(
                args.image,
                patch_size=int(released.model_args.pixel_token_patch_size),
            )
            text, metadata = generate_vlm_interleaved(
                model=released.model,
                tokenizer=released.tokenizer,
                new_token_ids=released.new_token_ids,
                images=[image],
                instruction=args.prompt,
                device=device,
                max_length=args.max_new_tokens,
                temperature=0.0,
                fixed_length=False,
                reasoning_mode="standard_bare",
                exact_max_new_tokens=True,
                return_generation_metadata=True,
            )
            output.write_text(text + "\n", encoding="utf-8")
            output.with_suffix(output.suffix + ".json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            frames, timestamps, temporal_patch_size = _decode_video_vlm(
                args.video,
                patch_size=int(released.model_args.pixel_token_patch_size),
            )
            text = _generate_v2t_one(
                released.model,
                released.tokenizer,
                released.new_token_ids,
                frames,
                timestamps,
                args.prompt,
                device,
                args.max_new_tokens + 1,
                0.0,
                temporal_patch_size=temporal_patch_size,
                timestamp_mode=VIDEO_VLM_CONTRACT.timestamp_mode,
            )
            output.write_text(text + "\n", encoding="utf-8")

    print(f"task={args.task} checkpoint={checkpoint_path} output={output}")


if __name__ == "__main__":
    main()
