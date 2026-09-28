# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run lmms-eval behind the same singleton-Gloo contract as PixelUMM eval.

The lmms-eval CLI otherwise constructs ``Accelerator`` before it constructs the
model, which initializes NCCL even for a one-process worker. Bare DCP loading
uses CPU state dictionaries, so each independent GPU replica gets its own
world-size-one Gloo group instead. No worker communicates with another.
"""

from __future__ import annotations

import os
import runpy
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    if dist.is_initialized():
        raise RuntimeError("Regular21 singleton launcher received an initialized process group")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Regular21 singleton workers require WORLD_SIZE=1")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each Regular21 worker must see exactly one CUDA GPU via CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    # A local FileStore gives each singleton worker its own rendezvous.
    file_descriptor, store_name = tempfile.mkstemp(
        prefix="pixelumm-core13-singleton-",
        dir=os.environ.get("TMPDIR") or "/tmp",
    )
    os.close(file_descriptor)
    store_path = Path(store_name)
    store_path.unlink()
    dist.init_process_group(
        "gloo",
        init_method=store_path.as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        runpy.run_module("lmms_eval", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        store_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
