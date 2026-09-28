# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only preflight for the released R07 Regular21 and Video4 runtimes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval.vlm.core13.runtime import resolve_image_preprocess_contract
from eval.vlm.r07_runtime import validate_dcp_checkpoint
from eval.vlm.video_4.runtime import (
    resolve_video_contract,
    validate_training_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--code-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = parser.parse_args()

    code_root = Path(args.code_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    validate_dcp_checkpoint(checkpoint)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from train.release_checkpoint import parse_release_arguments

    model_args, data_args, training_args = parse_release_arguments(
        code_root / "experiments" / "s8_f22_r07" / "release.yaml"
    )
    image_contract = resolve_image_preprocess_contract(data_args)
    video_contract = resolve_video_contract(model_args, data_args)
    validate_training_contract(
        model_args,
        data_args,
        training_args,
        video_contract,
    )
    generation_contract = {
        "num_steps": int(training_args.eval_num_timesteps),
        "cfg_renorm_type": str(training_args.eval_cfg_renorm_type),
        "t2i": {
            "scheduler": str(training_args.eval_sampler),
            "shift": float(training_args.eval_timestep_shift),
            "cfg_text_scale": float(training_args.eval_cfg_text_scale),
        },
        "t2v": {
            "scheduler": str(training_args.eval_video_sampler),
            "shift": float(training_args.eval_video_timestep_shift),
            "cfg_text_scale": float(training_args.eval_video_cfg_text_scale),
        },
    }
    expected_generation_contract = {
        "num_steps": 50,
        "cfg_renorm_type": "none",
        "t2i": {
            "scheduler": "dpm-solver",
            "shift": 3.0,
            "cfg_text_scale": 3.5,
        },
        "t2v": {"scheduler": "unipc", "shift": 10.0, "cfg_text_scale": 6.0},
    }
    if generation_contract != expected_generation_contract:
        raise RuntimeError(
            "PixelUMM-release qualitative generation contract mismatch: "
            f"observed={generation_contract!r} expected={expected_generation_contract!r}"
        )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "generation": generation_contract,
                "image_vlm": asdict(image_contract),
                "video_vlm": asdict(video_contract),
                "prediction_type": training_args.prediction_type,
                "loss_type": training_args.loss_type,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
