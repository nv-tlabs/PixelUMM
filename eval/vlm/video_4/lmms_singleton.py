# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start lmms-eval with the world-size-one Gloo group required by DCP load."""

from __future__ import annotations

import os
import runpy
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _is_hub_rate_limit(error: BaseException) -> bool:
    """Return true only when an exception chain contains an HTTP 429."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 429 or "429" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _retry_rate_limited_snapshots() -> None:
    """Let an anonymous Hub snapshot resume after its fixed rate window resets."""
    import lmms_eval.api.task as task_module

    if getattr(task_module, "_pixelumm_snapshot_retry", False):
        return
    original = task_module.snapshot_download
    # Large video repos need several anonymous resolver windows even when each
    # retry resumes from the pinned-revision cache.
    retries = int(os.environ.get("PIXELUMM_VIDEO4_HF_429_RETRIES", "8"))
    wait_seconds = int(os.environ.get("PIXELUMM_VIDEO4_HF_429_WAIT_SECONDS", "310"))

    def snapshot_download(*args, **kwargs):
        for attempt in range(retries + 1):
            try:
                return original(*args, **kwargs)
            except Exception as error:
                if attempt >= retries or not _is_hub_rate_limit(error):
                    raise
                print(
                    "Hugging Face anonymous rate limit reached; "
                    f"waiting {wait_seconds}s before resumable snapshot retry "
                    f"{attempt + 1}/{retries}.",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise AssertionError("unreachable")

    task_module.snapshot_download = snapshot_download
    task_module._pixelumm_snapshot_retry = True


def _prefer_explicit_task_overlays() -> None:
    """Make --include_path override same-named built-in tasks.

    lmms-eval v0.7.1 indexes its built-ins before include paths and merges in
    reverse precedence, so an explicit external task cannot override only its
    dataset authentication setting.  Keep the pinned sources untouched and
    change precedence only inside this benchmark process.
    """
    from lmms_eval.tasks import TaskManager

    if getattr(TaskManager, "_pixelumm_overlay_precedence", False):
        return
    original = TaskManager.initialize_tasks

    def initialize_tasks(self, include_path=None, include_defaults=True):
        task_index = original(
            self, include_path=None, include_defaults=include_defaults
        )
        if include_path is None:
            return task_index
        paths = [include_path] if isinstance(include_path, str) else include_path
        for task_dir in paths:
            task_index.update(self._get_task_and_group(task_dir))
        return task_index

    TaskManager.initialize_tasks = initialize_tasks
    TaskManager._pixelumm_overlay_precedence = True


def main() -> None:
    if dist.is_initialized():
        raise RuntimeError("Video4 launcher received an initialized process group")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Video4 workers require WORLD_SIZE=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each Video4 worker must see exactly one CUDA GPU")

    torch.cuda.set_device(0)
    descriptor, store_name = tempfile.mkstemp(
        prefix="pixelumm-video-4-", dir=os.environ.get("TMPDIR") or "/tmp"
    )
    os.close(descriptor)
    store_path = Path(store_name)
    store_path.unlink()
    dist.init_process_group(
        "gloo", init_method=store_path.as_uri(), rank=0, world_size=1
    )
    try:
        _prefer_explicit_task_overlays()
        _retry_rate_limited_snapshots()
        runpy.run_module("lmms_eval", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        store_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
