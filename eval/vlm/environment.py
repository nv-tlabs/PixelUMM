# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable benchmark-cache resolution shared by the 21+4 runners."""

from __future__ import annotations

import os
from pathlib import Path


def default_hf_home(environment: dict[str, str] | None = None) -> Path:
    environment = os.environ if environment is None else environment
    cache_root = environment.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "huggingface"
    home = environment.get("HOME")
    if not home:
        raise RuntimeError("HOME or XDG_CACHE_HOME is required for the default HF cache")
    return Path(home).expanduser() / ".cache" / "huggingface"
