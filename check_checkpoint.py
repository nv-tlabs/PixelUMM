#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check complete checkpoint files and exact model keys/shapes without a GPU."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch

from train.model_factory import build_pixelumm_model
from train.release_checkpoint import (
    DEFAULT_RELEASE_CONFIG,
    parse_release_arguments,
    validate_checkpoint_model_schema,
    validate_released_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--llm-path", required=True, help="Pinned local Qwen config/tokenizer directory")
    parser.add_argument("--config", default=str(DEFAULT_RELEASE_CONFIG))
    args = parser.parse_args()
    checkpoint = validate_released_checkpoint(args.checkpoint)
    config_path = Path(args.config).expanduser().resolve()
    config = parse_release_arguments(config_path, extra_args=("--llm_path", args.llm_path))
    with torch.device("meta"):
        model, _, _ = build_pixelumm_model(*config, logging.getLogger("pixelumm.preflight"))
    validate_checkpoint_model_schema(checkpoint, model)
    print(json.dumps({
        "status": "schema_pass",
        "checkpoint": str(checkpoint),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_tensors": len(model.state_dict()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "note": "File structure and tensor schema only; no weight-content or GPU inference test.",
    }, indent=2))


if __name__ == "__main__":
    main()
