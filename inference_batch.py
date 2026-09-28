#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an indexed T2V manifest with one model load per worker (Linux/CUDA)."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import yaml

from inference import _seed_everything
from train.eval_utils import _generate_t2v_one, save_video_mp4
from train.release_checkpoint import load_released_model


def read_items(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("version") != 1
        or payload.get("modality") != "video"
        or payload.get("includes")
    ):
        raise ValueError("Expected a self-contained version 1 video manifest")
    defaults = payload.get("defaults", {})
    items = []
    for group in payload["groups"]:
        for item in group["items"]:
            prompt = item.get("prompt")
            size = item.get("video_size", defaults.get("video_size"))
            fps = item.get("fps", defaults.get("fps"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Empty or invalid prompt")
            if (
                not isinstance(size, list)
                or len(size) != 3
                or any(type(n) is not int or n <= 0 for n in size)
            ):
                raise ValueError(
                    "video_size must be positive integer [frames,height,width]"
                )
            if size[0] % 4 or size[1] % 16 or size[2] % 16:
                raise ValueError("Video dimensions must align with t4/p16")
            if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
                raise ValueError("fps must be finite and positive")
            items.append(
                dict(
                    index=len(items),
                    group=group["name"],
                    prompt=prompt,
                    size=size,
                    fps=fps,
                )
            )
    if not items:
        raise ValueError("Empty video manifest")
    return items


def select_indices(value, count):
    indices = (
        list(range(count)) if value is None else [int(v) for v in value.split(",")]
    )
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= count for i in indices)
    ):
        raise ValueError("Indices must be unique valid zero-based manifest indices")
    return indices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


@contextlib.contextmanager
def loading_slot(path):
    if path is None:
        yield
        return
    import fcntl

    with Path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(90):
            mem = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            if int(mem["MemAvailable"].split()[0]) * 1024 >= 36 * 1024**3:
                break
            print("Waiting for CPU loading RAM headroom", flush=True)
            time.sleep(10)
        else:
            raise RuntimeError("Insufficient CPU loading RAM headroom after 15 minutes")
        yield


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--llm-path", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--negative-prompt-file", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--indices", help="Zero-based comma-separated IDs; default all")
    p.add_argument("--seed-base", type=int, default=4396)
    p.add_argument("--sampler", choices=("unipc", "dpm-solver"), default="unipc")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Default: 35 for UniPC, 50 for DPM-Solver++",
    )
    p.add_argument(
        "--load-lock", help="Shared external lock to serialize CPU weight loading"
    )
    args = p.parse_args(argv)
    if args.steps is None:
        args.steps = 50 if args.sampler == "dpm-solver" else 35
    if args.steps <= 0 or (args.sampler == "dpm-solver" and args.steps < 3):
        p.error("--steps must be positive; DPM-Solver requires --steps >= 3")
    if args.seed_base < 0:
        p.error("--seed-base must be nonnegative")
    return args


def main():
    args = parse_args()
    items = read_items(args.manifest)
    indices = select_indices(args.indices, len(items))
    negative = Path(args.negative_prompt_file).read_text(encoding="utf-8").strip()
    if not negative:
        raise ValueError("A non-empty negative prompt is required")
    # Keep generation arguments and provenance tied to the same settings.
    generation_options = dict(
        num_timesteps=args.steps,
        timestep_shift=10.0,
        sampler=args.sampler,
        cfg_text_scale=6.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_type="none",
    )
    sampling_metadata = dict(
        sampler=generation_options["sampler"],
        steps=generation_options["num_timesteps"],
        shift=generation_options["timestep_shift"],
        cfg=generation_options["cfg_text_scale"],
        cfg_img_scale=generation_options["cfg_img_scale"],
        cfg_interval=generation_options["cfg_interval"],
        cfg_renorm=generation_options["cfg_renorm_type"],
        dtype="bf16",
        noise_scale=1.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        config_sha256=sha(args.config),
        manifest_sha256=sha(args.manifest),
        negative_file_sha256=sha(args.negative_prompt_file),
        indices=indices,
        seed_rule="seed_base + original manifest index",
        seed_base=args.seed_base,
        **sampling_metadata,
    )
    write_json(args.output_dir / "request.json", identity)
    with loading_slot(args.load_lock):
        released = load_released_model(
            args.checkpoint,
            config_path=args.config,
            device="cuda:0",
            dtype=sampling_metadata["dtype"],
            llm_path=args.llm_path,
        )
    write_json(
        args.output_dir / "ready.json",
        dict(loaded=True, gpu=torch.cuda.get_device_name(0)),
    )
    print("MODEL_READY", flush=True)
    for index in indices:
        item = items[index]
        seed = args.seed_base + index
        _seed_everything(seed)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        print(f"GENERATE index={index} seed={seed} size={item['size']}", flush=True)
        with torch.inference_mode():
            frames = _generate_t2v_one(
                released.model,
                released.tokenizer,
                released.new_token_ids,
                item["prompt"],
                tuple(item["size"]),
                torch.device("cuda:0"),
                **generation_options,
                video_fps=item["fps"],
                negative_prompt=negative,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        count, height, width = item["size"]
        if len(frames) != count or any(
            frame.size != (width, height) for frame in frames
        ):
            raise RuntimeError(
                "Generated frame count or dimensions differ from manifest"
            )
        name = f"{index:03d}"
        output = args.output_dir / f"{name}.mp4"
        save_video_mp4(
            frames, str(args.output_dir), fps=item["fps"], filename=output.name
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("MP4 encoding failed")
        strip = Image.new("RGB", (width * 3, height))
        for col, frame_id in enumerate((0, count // 2, count - 1)):
            strip.paste(frames[frame_id], (col * width, 0))
        strip.save(args.output_dir / f"{name}-frames.jpg")
        record = dict(
            **item,
            seed=seed,
            **sampling_metadata,
            seconds=elapsed,
            mp4_sha256=sha(output),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
        write_json(args.output_dir / f"{name}.json", record)
        print(json.dumps(record), flush=True)
        del frames
    write_json(
        args.output_dir / "complete.json", dict(indices=indices, count=len(indices))
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
