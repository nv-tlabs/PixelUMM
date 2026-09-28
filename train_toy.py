#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-step local-data training example for a PixelUMM checkpoint.

Run this file under ``torchrun``. It loads model weights and starts a new
optimizer.
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from train.release_checkpoint import (
    DEFAULT_RELEASE_CONFIG,
    validate_released_checkpoint,
)
from train.toy_trainer import ToyTrainingConfig, run_toy_training


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_CONFIG = ROOT / "data" / "configs" / "pixelumm_toy.yaml"
_MANIFESTS = ("t2i.jsonl", "t2v.jsonl", "image_vlm.jsonl", "video_vlm.jsonl")


def _configure_single_node_distributed_defaults() -> dict[str, str]:
    """Use a reliable local transport for a single-node ``torchrun`` job.

    The released checkpoint is a DCP checkpoint, so loading it exercises the
    process group before FSDP wrapping. On hosts with an auto-detected but
    unusable IB/RDMA path, that first collective can otherwise terminate one
    rank without a Python traceback. Socket transport is a portable default
    for the toy job's single-node contract. Explicit user settings always win,
    and multi-node jobs are left untouched.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size <= 1 or local_world_size != world_size:
        return {}

    defaults = {
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
    }
    applied = {}
    for name, value in defaults.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue training from a complete released PixelUMM checkpoint "
            "using only finite local JSONL/media examples."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--toy-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_RELEASE_CONFIG))
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG))
    parser.add_argument(
        "--llm-path",
        default=None,
        help=(
            "Local Qwen3-8B tokenizer/config directory. Learned Qwen weights "
            "still come from --checkpoint."
        ),
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=4396)
    parser.add_argument("--max-num-tokens", type=int, default=82_000)
    parser.add_argument("--expected-num-tokens", type=int, default=60_000)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Validate inputs and print the complete public training config.",
    )
    return parser.parse_args()


def _validate_local_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path]:
    checkpoint = validate_released_checkpoint(args.checkpoint)
    toy_root = Path(args.toy_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    dataset_config = Path(args.dataset_config).expanduser().resolve()

    if not toy_root.is_dir():
        raise FileNotFoundError(f"Toy-data directory is missing: {toy_root}")
    missing = [name for name in _MANIFESTS if not (toy_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Toy-data directory lacks required manifests {missing}: {toy_root}"
        )
    if not config.is_file():
        raise FileNotFoundError(f"Release config is missing: {config}")
    if not dataset_config.is_file():
        raise FileNotFoundError(f"Toy dataset config is missing: {dataset_config}")
    if output == checkpoint or checkpoint in output.parents:
        raise ValueError("--output must not overwrite or nest under the released checkpoint")
    if output.exists():
        raise FileExistsError(f"Use a fresh --output directory: {output}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.max_num_tokens <= 0 or args.expected_num_tokens <= 0:
        raise ValueError("token limits must be positive")
    if args.expected_num_tokens > args.max_num_tokens:
        raise ValueError("--expected-num-tokens cannot exceed --max-num-tokens")
    return checkpoint, toy_root, output, config, dataset_config


def build_training_config(args: argparse.Namespace) -> ToyTrainingConfig:
    checkpoint, toy_root, output, config, dataset_config = _validate_local_inputs(args)
    llm_path = None
    if args.llm_path is not None:
        llm_path = Path(args.llm_path).expanduser().resolve()
        if not llm_path.is_dir():
            raise FileNotFoundError(
                f"Qwen tokenizer/config directory is missing: {llm_path}"
            )
    os.environ["PIXELUMM_TOY_ROOT"] = str(toy_root)
    return ToyTrainingConfig(
        checkpoint=checkpoint,
        toy_root=toy_root,
        output=output,
        model_config=config,
        dataset_config=dataset_config,
        llm_path=llm_path,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_num_tokens=args.max_num_tokens,
        expected_num_tokens=args.expected_num_tokens,
    )


def main() -> None:
    args = _parse_args()
    training_config = build_training_config(args)
    if args.print_config:
        print(json.dumps(training_config.to_dict(), indent=2, sort_keys=True))
        return

    # Import only after local inputs and the explicit no-cursor contract have
    # been validated.  The canonical trainer sees an ordinary local packer.
    applied_distributed_defaults = _configure_single_node_distributed_defaults()
    if applied_distributed_defaults:
        rendered = ", ".join(
            f"{name}={value}" for name, value in applied_distributed_defaults.items()
        )
        print(f"Applied single-node distributed defaults: {rendered}", flush=True)
    run_toy_training(training_config)


if __name__ == "__main__":
    main()
