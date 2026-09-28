#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos prompt and video guardrails, run in a separate environment.

The Cosmos package requires a different Transformers major version from
PixelUMM. Invoke this script with a dedicated Python interpreter; do not add
its requirements to the PixelUMM inference environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


class GuardrailRejected(Exception):
    """The input or generated video failed a safety check."""


def prepare_nltk_data() -> None:
    """Copy gated NLTK assets out of HF's symlink-based snapshot cache."""
    import nltk
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        "nvidia/Cosmos-1.0-Guardrail", allow_patterns=["blocklist/nltk_data/*"]
    ))
    source = snapshot / "blocklist" / "nltk_data"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing Cosmos NLTK data: {source}")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parent = cache_root / "pixelumm" / "guardrails"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"nltk_data-{snapshot.name}"
    if not destination.is_dir():
        with tempfile.TemporaryDirectory(prefix="nltk-", dir=parent) as temporary:
            staged = Path(temporary) / "nltk_data"
            shutil.copytree(source, staged, symlinks=False)
            try:
                staged.rename(destination)
            except FileExistsError:
                if not destination.is_dir():
                    raise
    nltk.data.path.insert(0, str(destination))


def check_prompt(prompt: str, device: str) -> None:
    from cosmos_guardrail.cosmos_guardrail import Blocklist, Qwen3Guard

    prepare_nltk_data()
    allowed, reason = Blocklist().is_safe(prompt)
    if not allowed:
        raise GuardrailRejected(reason)

    guard = Qwen3Guard().to(device)
    check_qwen_prompt(guard, prompt)


def check_qwen_prompt(guard, prompt: str) -> None:
    import torch

    messages = [{"role": "user", "content": prompt}]
    rendered = guard.tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = guard.tokenizer([rendered], return_tensors="pt").to(guard.model.device)
    with torch.inference_mode():
        generated = guard.model.generate(**inputs, max_new_tokens=128)
    response = guard.tokenizer.decode(
        generated[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    )
    label = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)\b", response)
    if label is None:
        raise RuntimeError("Qwen3Guard returned no recognized safety label")
    if label.group(1) != "Safe":
        raise GuardrailRejected(f"Qwen3Guard classified prompt as {label.group(1)}")


def classify_frame_strict(video_filter, frame) -> int:
    """Classify one frame and surface any inference failure."""
    import torch
    from PIL import Image

    with torch.inference_mode():
        encoder = video_filter.encoder
        encoder_model = encoder.model
        encoder_parameter = next(encoder_model.parameters())
        inputs = encoder.processor(images=Image.fromarray(frame), return_tensors="pt")
        inputs = inputs.to(device=encoder_parameter.device, dtype=encoder_parameter.dtype)
        features = encoder_model.get_image_features(**inputs)
        # Transformers 4 returns a tensor; Transformers 5 returns a model output.
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError("SigLIP returned invalid image features")
        features = torch.nn.functional.normalize(features, dim=-1)
        network = video_filter.model.network
        parameter = next(network.parameters())
        if features.shape[-1] != network.input_size:
            raise RuntimeError("SigLIP feature width does not match Cosmos classifier")
        logits = network(features.to(device=parameter.device, dtype=parameter.dtype))
        return int(torch.argmax(logits, dim=-1).item())


def verify_frames(frames, video_filter) -> None:
    if len(frames) == 0:
        raise ValueError("The video contains no frames")
    for index, frame in enumerate(frames):
        category = classify_frame_strict(video_filter, frame)
        if category != 0:
            raise GuardrailRejected(
                f"Cosmos video content filter rejected frame {index} (class {category})"
            )


def read_video(path: Path):
    import imageio
    import numpy as np

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        fps = float(reader.get_meta_data()["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source FPS: {fps}")
        frames = np.asarray([frame for frame in reader])
    finally:
        reader.close()
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected nonempty uint8 RGB video frames")
    return frames, fps


def write_video_exclusive(frames, fps: float, output: Path) -> None:
    import imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing file: {output}")
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-", suffix=".mp4", dir=output.parent, delete=False
    ) as temporary:
        staged = Path(temporary.name)
    try:
        with imageio.get_writer(
            str(staged), format="ffmpeg", fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None, quality=8,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        # A hard link is atomic and cannot silently replace another result.
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)


def filter_video(source: Path, output: Path, device: str) -> int:
    import torch
    from cosmos_guardrail.cosmos_guardrail import RetinaFaceFilter, VideoContentSafetyFilter

    if source.resolve() == output.resolve():
        raise ValueError("Input and output must differ")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing file: {output}")
    frames, fps = read_video(source)

    video_filter = VideoContentSafetyFilter().to(device)
    verify_frames(frames, video_filter)
    del video_filter
    if device.startswith("cuda"):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    face_filter = RetinaFaceFilter().to(device)
    processed = face_filter.postprocess(frames)
    if processed.shape != frames.shape or processed.dtype != frames.dtype:
        raise ValueError("Cosmos face filter changed video shape or dtype")
    write_video_exclusive(processed, fps, output)
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    text = subcommands.add_parser("check-prompt")
    prompt_source = text.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt")
    prompt_source.add_argument("--prompt-stdin", action="store_true")
    text.add_argument("--device", default="cuda:0")
    video = subcommands.add_parser("filter-video")
    video.add_argument("--input", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    try:
        if args.command == "check-prompt":
            prompt = sys.stdin.read() if args.prompt_stdin else args.prompt
            check_prompt(prompt, args.device)
            print(json.dumps({"guardrail": "cosmos", "prompt_allowed": True}))
        else:
            count = filter_video(args.input, args.output, args.device)
            print(json.dumps({"guardrail": "cosmos", "video_allowed": True, "frames_checked": count}))
        return 0
    except GuardrailRejected as exc:
        print(json.dumps({"guardrail": "cosmos", "allowed": False, "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
