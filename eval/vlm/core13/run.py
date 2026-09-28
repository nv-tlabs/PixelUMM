# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch independent one-GPU lmms-eval workers for the Regular21 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.vlm.core13.runtime import validate_dcp_checkpoint
from eval.vlm.r07_runtime import resolve_vlm_hf_home


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_PYTHON = Path(sys.executable)
REGULAR21_TASKS = (
    "mmmu_val", "mmstar", "realworldqa", "seedbench_image", "ai2d",
    "docvqa_val", "chartqa", "infovqa_val", "textvqa_val", "ocrbench",
    "mme", "gqa", "mmvp", "seedbench_2_plus", "cv_bench", "countbench",
    "pixmo_count", "vstar_bench", "mmmu_pro_standard", "blink", "muirbench",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--code-root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--response-cache-root",
        default="",
        help=(
            "Persistent lmms response-cache root. It must live outside "
            "--output-root so monitor retries can archive incomplete outputs "
            "without discarding completed deterministic responses."
        ),
    )
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument(
        "--lmms-root",
        required=True,
        help="Pinned external lmms-eval v0.7.1 checkout.",
    )
    parser.add_argument(
        "--contract",
        default=str(HERE / "contracts" / "regular21.yaml"),
        help="Pinned PixelUMM Regular21 suite contract.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--reasoning-mode",
        default="standard_bare",
        choices=("standard_bare",),
        help="Released train-aligned VLM decoding contract.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated evaluation tasks to run.",
    )
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Resume into an existing output directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_contract(contract_path: Path):
    contract_path = contract_path.resolve()
    raw = contract_path.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def distribute_groups(groups, workers):
    workers = max(1, min(int(workers), len(groups)))
    assigned = [[] for _ in range(workers)]
    for index, group in enumerate(groups):
        assigned[index % workers].extend(group)
    return [group for group in assigned if group]


def _cache_worker_component(worker_index: int, tasks: list[str]) -> str:
    """Return a readable, SQLite-safe persistent-cache path component."""
    task_label = "__".join(tasks)
    component = f"worker_{worker_index:02d}_{task_label}"
    if len(component) <= 96:
        return component
    digest = hashlib.sha256(task_label.encode("utf-8")).hexdigest()[:16]
    return f"worker_{worker_index:02d}_{task_label[:48]}_{digest}"


def _task_source_roots(contract: dict) -> tuple[str, ...]:
    """Map public lmms task IDs to the vendored source directories they execute."""
    overrides = {
        "mmmu_val": "mmmu",
        "seedbench_image": "seedbench",
        "docvqa_val": "docvqa",
        "infovqa_val": "infovqa",
        "textvqa_val": "textvqa",
        "mmmu_pro_standard": "mmmu_pro",
    }
    roots = []
    for task in contract["tasks"]:
        task_id = str(task["id"] if isinstance(task, dict) else task)
        roots.append(overrides.get(task_id, task_id))
    return tuple(sorted(set(roots)))


def _source_fingerprint(
    code_root: Path,
    lmms_root: Path,
    contract: dict,
) -> tuple[str, list[str]]:
    """Fingerprint every local source file that defines this eval contract."""
    paths = []
    for path in HERE.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(HERE).parts
        if ".venv" in relative_parts or "__pycache__" in relative_parts:
            continue
        paths.append(path)
    # Fingerprint every live source family that constructs the model topology,
    # loads DCP state, builds train-contract arguments, or performs generation.
    # Hashing only eval_utils.py would miss behavior changes in PixelUMM's decode
    # loop (including EOS handling) while claiming the same executable contract.
    runtime_source_roots = (code_root / "modeling" / "pixelumm",)
    for root in runtime_source_roots:
        paths.extend(
            path
            for path in root.rglob("*.py")
            if path.is_file() and "__pycache__" not in path.parts
        )
    runtime_source_files = (
        code_root / "data" / "pixelumm_image_resize.py",
        code_root / "data" / "pixelumm_smart_resize.py",
        code_root / "eval" / "vlm" / "release_preprocess_contract.py",
        code_root / "experiments" / "s8_f22_r07" / "release.yaml",
        code_root / "train" / "config.py",
        code_root / "train" / "model_factory.py",
        code_root / "train" / "eval_utils.py",
        code_root / "train" / "release_checkpoint.py",
    )
    paths.extend(runtime_source_files)

    # These vendored task definitions are part of the executable contract.  A
    # public-dataset auth change here can otherwise make an old receipt
    # impossible to reproduce even when the model code is unchanged.
    vendored_tasks = lmms_root / "lmms_eval" / "tasks"
    for task_root in _task_source_roots(contract):
        paths.extend(
            path
            for path in (vendored_tasks / task_root).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )

    unique_paths = sorted(set(path.resolve() for path in paths), key=str)
    digest = hashlib.sha256()
    labels = []
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing fingerprinted Regular21 source: {path}")
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = str(path)
        labels.append(label)
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), labels


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    validate_dcp_checkpoint(checkpoint)

    contract_path = Path(args.contract).resolve()
    contract, contract_sha256 = load_contract(contract_path)
    contract_tasks = tuple(str(task["id"]) for task in contract["tasks"])
    if contract.get("name") != "pixelumm-regular21-official-lmms-v1":
        raise ValueError(f"Unsupported PixelUMM VLM contract: {contract.get('name')!r}")
    if contract_tasks != REGULAR21_TASKS:
        raise ValueError(f"PixelUMM Regular21 task mismatch: {contract_tasks!r}")
    mode_contract = contract["generation"]["reasoning_modes"]["standard_bare"]
    # Official lmms task YAMLs own max_new_tokens. This is only a generous
    # fail-closed ceiling, not an alternate reasoning-mode policy.
    max_new_tokens_cap = 4096
    if args.tasks:
        groups = [[task.strip() for task in args.tasks.split(",") if task.strip()]]
        unknown = sorted(set(groups[0]) - set(REGULAR21_TASKS))
        if unknown:
            raise ValueError(f"Tasks are outside PixelUMM Regular21: {unknown}")
    else:
        groups = contract["worker_groups"]
    groups = distribute_groups(groups, args.workers)

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path.cwd()
        / "outputs"
        / "regular21"
        / args.reasoning_mode
        / checkpoint.name
    )
    response_cache_root = (
        Path(args.response_cache_root).resolve()
        if args.response_cache_root
        else None
    )
    if (
        response_cache_root is not None
        and (
            response_cache_root == output_root
            or response_cache_root.is_relative_to(output_root)
        )
    ):
        raise ValueError(
            "--response-cache-root must live outside --output-root so retries "
            "do not archive the cache"
        )
    if output_root.exists() and any(output_root.iterdir()):
        if not args.allow_existing_output:
            raise FileExistsError(
                f"Refusing to mix Regular21 attempts in non-empty output: {output_root}. "
                "Archive it or choose a fresh --output-root."
            )

    # Do not call Path.resolve() on the Python path: venv/bin/python is
    # intentionally a symlink to the container interpreter. Resolving it would
    # bypass the venv's site-packages in every worker subprocess.
    python = Path(args.python).expanduser().absolute()
    lmms_root = Path(args.lmms_root).resolve()
    code_root = Path(args.code_root).resolve()
    required = (
        python,
        lmms_root / "lmms_eval" / "__main__.py",
        code_root / "train" / "model_factory.py",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing Regular21 runtime dependency: {path}")
    source_sha256, source_files = _source_fingerprint(
        code_root,
        lmms_root,
        contract,
    )

    receipt = {
        "contract": contract["name"],
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "task_specs": contract["tasks"],
        "source_sha256": source_sha256,
        "source_files": source_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "code_root": str(Path(args.code_root).resolve()),
        "lmms_root": str(Path(args.lmms_root).resolve()),
        "workers": len(groups),
        "groups": groups,
        "limit": args.limit,
        "reasoning_mode": args.reasoning_mode,
        "reasoning_mode_contract": mode_contract,
        "response_cache_root": (
            str(response_cache_root) if response_cache_root is not None else None
        ),
        "master_port_base": int(os.environ.get("CORE13_MASTER_PORT_BASE", "29571")),
    }
    if args.dry_run:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )

    # Use a configurable base port to avoid collisions between workers.
    master_port_base = int(os.environ.get("CORE13_MASTER_PORT_BASE", "29571"))
    if not 1024 <= master_port_base <= 65535 - len(groups):
        raise ValueError(
            "CORE13_MASTER_PORT_BASE must leave one valid TCP port per worker; "
            f"got base={master_port_base}, workers={len(groups)}"
        )

    processes = []
    for worker_index, tasks in enumerate(groups):
        label = "__".join(tasks)
        worker_dir = output_root / f"worker_{worker_index:02d}_{label}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_cache_root = (
            response_cache_root
            / f"source_{source_sha256[:16]}"
            / _cache_worker_component(worker_index, tasks)
            if response_cache_root is not None
            else None
        )
        reasoning_trace_path = (
            worker_cache_root / "reasoning_outputs.jsonl"
            if worker_cache_root is not None
            else worker_dir / "reasoning_outputs.jsonl"
        )
        if worker_cache_root is not None:
            worker_cache_root.mkdir(parents=True, exist_ok=True)
        # A cache root exists on the first attempt as well.  Resume the
        # reasoning trace only after that worker has actually persisted one;
        # otherwise the model fails before it can create the initial trace.
        reasoning_trace_resume = reasoning_trace_path.is_file()
        model_args = ",".join(
            (
                f"checkpoint_path={checkpoint}",
                f"code_root={code_root}",
                "dtype=bf16",
                f"max_new_tokens_cap={max_new_tokens_cap}",
                f"reasoning_mode={args.reasoning_mode}",
                f"reasoning_trace_path={reasoning_trace_path}",
                f"reasoning_trace_resume={reasoning_trace_resume}",
            )
        )
        command = [
            str(python),
            "-m",
            "eval.vlm.core13.lmms_singleton",
            "--model",
            "pixelumm_core13",
            "--model_args",
            model_args,
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            "1",
            "--output_path",
            str(worker_dir),
            "--include_path",
            str(HERE / "pixelumm_lmms" / "tasks"),
            "--log_samples",
            "--verbosity",
            "INFO",
        ]
        if worker_cache_root is not None:
            command.extend(("--use_cache", str(worker_cache_root / "responses")))
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(master_port_base + worker_index)
        env["LMMS_EVAL_PLUGINS"] = "pixelumm_lmms"
        if worker_cache_root is not None:
            # Keep the layered cache shard identity stable across Slurm job
            # IDs. ResponseCache restores its periodically checkpointed shard
            # when a retry lands on a different node.
            env["LMMS_CACHE_RUN_ID"] = "resumable-singleton"
        python_paths = (
            str(code_root),
            str(HERE),
            str(lmms_root),
            env.get("PYTHONPATH", ""),
        )
        env["PYTHONPATH"] = ":".join(path for path in python_paths if path)
        env["HF_HOME"] = resolve_vlm_hf_home("regular21", env)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("PYTHONUNBUFFERED", "1")

        printable = " ".join(shlex_quote(token) for token in command)
        print(
            f"[worker {worker_index}] CUDA={worker_index} "
            f"MASTER_PORT={env['MASTER_PORT']} tasks={tasks}\n{printable}"
        )
        log_handle = (worker_dir / "worker.log").open("w")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_index, tasks, process, log_handle))

    failures = []
    for worker_index, tasks, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append((worker_index, tasks, return_code))
    if failures:
        raise RuntimeError(f"Regular21 workers failed: {failures}")

    if response_cache_root is not None:
        for worker_index, tasks in enumerate(groups):
            label = "__".join(tasks)
            worker_dir = output_root / f"worker_{worker_index:02d}_{label}"
            persistent_trace = (
                response_cache_root
                / f"source_{source_sha256[:16]}"
                / _cache_worker_component(worker_index, tasks)
                / "reasoning_outputs.jsonl"
            )
            if not persistent_trace.is_file():
                raise FileNotFoundError(
                    f"Missing persistent reasoning trace: {persistent_trace}"
                )
            shutil.copy2(persistent_trace, worker_dir / "reasoning_outputs.jsonl")

    subprocess.run(
        [str(python), str(HERE / "aggregate.py"), str(output_root)],
        check=True,
        env=os.environ.copy(),
    )


def shlex_quote(value):
    import shlex

    return shlex.quote(str(value))


if __name__ == "__main__":
    main()
