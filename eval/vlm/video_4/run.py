# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one or more independent one-GPU Video4 lmms-eval workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.vlm.video_4.runtime import validate_dcp_checkpoint
from eval.vlm.r07_runtime import resolve_vlm_hf_home
from eval.vlm.release_preprocess_contract import VIDEO_VLM_CONTRACT


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EVAL_CODE_ROOT = REPO_ROOT
DEFAULT_PYTHON = Path(sys.executable)

TASK_DIRS = {
    "mvbench": "mvbench",
    "videomme": "videomme",
    "longvideobench_val_v": "longvideobench",
    "lvbench": "lvbench",
}


def materialize_anonymous_task_configs(
    lmms_root: Path,
    destination: Path,
    groups: list[list[str]],
) -> tuple[list[list[str]], dict[str, list[str]]]:
    """Copy pinned task sources, preserving auth for gated datasets."""
    requested = {task for group in groups for task in group}
    task_root = lmms_root / "lmms_eval" / "tasks"
    unknown = sorted(requested - set(TASK_DIRS))
    if unknown:
        raise ValueError(f"Unsupported Video4 task override: {unknown}")
    for directory in sorted({TASK_DIRS[task] for task in requested}):
        source = task_root / directory
        shutil.copytree(source, destination / directory)

    patches = {
        "mvbench": destination / "mvbench" / "_default_template_yaml",
        "videomme": destination / "videomme" / "videomme.yaml",
        "longvideobench_val_v": (
            destination / "longvideobench" / "longvideobench_val_v.yaml"
        ),
        "lvbench": destination / "lvbench" / "lvbench.yaml",
    }
    auth_configs = dict(patches)
    gated_tasks = {"longvideobench_val_v"}
    for task in sorted(requested):
        path = auth_configs[task]
        text = path.read_text()
        # MVBench expands to 20 leaf tasks in one lmms process.  The pinned
        # loader otherwise lets every leaf remove and recreate the same shared
        # HF_HOME/mvbench_video symlink; concurrent profile jobs can observe the
        # link between unlink(2) and symlink(2) and fail before the first item.
        # The preflight owns the stable link, so task configs must not mutate it.
        if task == "mvbench":
            if "create_link: True" not in text:
                raise RuntimeError(
                    f"Could not disable mutable MVBench cache linking in {path}"
                )
            text = text.replace("create_link: True", "create_link: false", 1)
        if task in gated_tasks:
            if "token: True" not in text:
                raise RuntimeError(
                    f"Gated task must retain authenticated dataset access in {path}"
                )
            path.write_text(text)
            continue
        if "token: True" in text:
            text = text.replace("token: True", "token: false", 1)
        elif "token: false" not in text:
            raise RuntimeError(f"Could not disable authenticated dataset access in {path}")
        path.write_text(text)

    resolved: dict[str, list[str]] = {}
    for task in sorted(requested):
        if task == "mvbench":
            resolved[task] = [
                str(path.resolve())
                for path in sorted((destination / "mvbench").glob("mvbench_*.yaml"))
                if path.name != "mvbench.yaml"
            ]
        else:
            resolved[task] = [str(patches[task].resolve())]
    # lmms-eval v0.7.1 indexes external YAMLs through --include_path and then
    # selects them by their declared task/group name.  Supplying individual
    # YAML paths is not a supported selection mechanism in this revision.
    return ([list(group) for group in groups], resolved)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--code-root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument(
        "--lmms-root",
        required=True,
        help="Pinned external lmms-eval v0.7.1 checkout.",
    )
    parser.add_argument(
        "--contract", default=str(HERE / "contract_r07_video4.yaml")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-root", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--max-new-tokens-cap", type=int, default=64)
    parser.add_argument("--decode-timeout-seconds", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


FULL_TASK_DOCS_PER_LEAF = {
    "mvbench": (200, 20),
    "videomme": (2700, 1),
    "longvideobench_val_v": (1337, 1),
    "lvbench": (1549, 1),
}


def _trace_records(path: Path) -> list[dict]:
    records = []
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed generation trace {path}:{line_number}") from exc
    return records


def _trace_key(record: dict) -> tuple[str, str, str, str]:
    return (
        str(record["task"]), str(record["split"]), str(record["doc_id"]),
        str(record["instruction_sha256"]),
    )


def load_contract(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def distribute(groups: list[list[str]], workers: int) -> list[list[str]]:
    assigned = [[] for _ in range(max(1, min(int(workers), len(groups))))]
    for index, group in enumerate(groups):
        assigned[index % len(assigned)].extend(group)
    return [group for group in assigned if group]


def source_fingerprint(code_root: Path, lmms_root: Path, contract: dict) -> tuple[str, list[str]]:
    paths = [
        path
        for path in HERE.rglob("*")
        if path.is_file() and ".venv" not in path.parts and "__pycache__" not in path.parts
    ]
    for root in (code_root / "modeling" / "pixelumm",):
        paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    paths.extend(
        code_root / relative
        for relative in (
            "data/pixelumm_image_resize.py",
            "data/pixelumm_smart_resize.py",
            "data/pixelumm_video_decoder.py",
            "eval/vlm/release_preprocess_contract.py",
            "experiments/s8_f22_r07/release.yaml",
            "train/model_factory.py",
            "train/eval_utils.py",
            "train/config.py",
            "train/release_checkpoint.py",
        )
    )
    task_dir = lmms_root / "lmms_eval" / "tasks"
    for task in contract["tasks"]:
        directory = TASK_DIRS[str(task["id"])]
        source = task_dir / directory
        paths.extend(
            path
            for path in source.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    labels = []
    for path in sorted(set(path.resolve() for path in paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(f"Missing fingerprinted source: {path}")
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = str(path)
        labels.append(label)
        digest.update(label.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest(), labels


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    validate_dcp_checkpoint(checkpoint)
    contract_path = Path(args.contract).resolve()
    contract, contract_sha256 = load_contract(contract_path)
    model_input = contract.get("model_input", {})
    expected_input = {
        "patch_size": 16,
        "source_fps_min": 1,
        "source_fps_max": 240,
        "sparse_max_sampled_frames": VIDEO_VLM_CONTRACT.sparse_max_sampled_frames,
        "max_video_raw_patch_tokens": VIDEO_VLM_CONTRACT.max_raw_patch_tokens,
        "min_frame_pixels": VIDEO_VLM_CONTRACT.min_frame_pixels,
        "max_frame_pixels": VIDEO_VLM_CONTRACT.max_frame_pixels,
        "fallback_condition": "selected_duration_above_96s",
    }
    mismatched_input = {
        name: (model_input.get(name), expected)
        for name, expected in expected_input.items()
        if model_input.get(name) != expected
    }
    if mismatched_input:
        raise ValueError(
            f"Video4 YAML does not match R07 runtime: {mismatched_input}"
        )
    representation_profile = str(
        model_input.get("short_video_eval_arm", "")
    )
    if representation_profile != "short_image":
        raise ValueError(
            "PixelUMM-release Video4 requires short_video_eval_arm='short_image'; "
            f"got {representation_profile!r} in {contract_path}"
        )
    contract_tasks = tuple(str(task["id"]) for task in contract["tasks"])
    if contract_tasks != tuple(TASK_DIRS):
        raise ValueError(
            f"PixelUMM-release Video4 task mismatch: {contract_tasks!r}"
        )
    groups = (
        [[task.strip() for task in args.tasks.split(",") if task.strip()]]
        if args.tasks
        else contract["worker_groups"]
    )
    groups = distribute(groups, args.workers)
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    if args.shards > 1:
        if len(groups) != 1 or len(groups[0]) != 1:
            raise ValueError("Sharded execution requires exactly one requested task")
        if groups[0][0] not in FULL_TASK_DOCS_PER_LEAF:
            raise ValueError(f"No exact sharding cardinality for task {groups[0][0]!r}")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path.cwd() / "outputs" / "video4" / checkpoint.name
    )
    output_nonempty = output_root.exists() and any(output_root.iterdir())
    if output_nonempty and not args.resume:
        raise FileExistsError(f"Refusing non-empty Video4 output: {output_root}")
    if args.resume and (output_root / "COMPLETED.json").is_file():
        print(f"Video4 output already complete: {output_root}")
        return

    python = Path(args.python).expanduser().absolute()
    lmms_root = Path(args.lmms_root).resolve()
    code_root = Path(args.code_root).resolve()
    for required in (
        python,
        lmms_root / "lmms_eval" / "__main__.py",
        code_root / "train" / "model_factory.py",
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing Video4 runtime dependency: {required}")
    source_sha256, source_files = source_fingerprint(code_root, lmms_root, contract)
    receipt = {
        "contract": contract["name"],
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "source_sha256": source_sha256,
        "source_files": source_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "code_root": str(code_root),
        "lmms_root": str(lmms_root),
        "workers": len(groups),
        "shards": args.shards,
        "groups": groups,
        "limit": args.limit,
        "model_input": contract["model_input"],
        "representation_profile": representation_profile,
        "task_specs": contract["tasks"],
        "dataset_auth": "anonymous_public_task_overlay",
        "dataset_endpoint": os.environ.get(
            "HF_ENDPOINT", "https://huggingface.co"
        ),
    }
    if args.dry_run:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True, exist_ok=args.resume)
    receipt_path = output_root / "run_receipt.json"
    if args.resume:
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Missing resume receipt: {receipt_path}")
        original_receipt = json.loads(receipt_path.read_text())
        invariants = {
            "contract": receipt["contract"],
            "contract_sha256": receipt["contract_sha256"],
            "checkpoint": receipt["checkpoint"],
            "code_root": receipt["code_root"],
            "lmms_root": receipt["lmms_root"],
            "groups": receipt["groups"],
            "limit": receipt["limit"],
            "shards": receipt["shards"],
            "model_input": receipt["model_input"],
        }
        mismatched = [
            key for key, value in invariants.items()
            if original_receipt.get(key) != value
        ]
        if mismatched:
            raise RuntimeError(f"Resume receipt mismatch: {mismatched}")
        resolved_groups = [list(group) for group in groups]
        resolved_task_configs = original_receipt.get("resolved_task_configs")
        if not resolved_task_configs:
            raise RuntimeError("Resume receipt has no resolved task configs")
        resume_record = {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_sha256,
            "source_files": source_files,
        }
        with (output_root / "resume_attempts.jsonl").open("a") as handle:
            handle.write(json.dumps(resume_record, sort_keys=True) + "\n")
        receipt = original_receipt
    else:
        resolved_groups, resolved_task_configs = materialize_anonymous_task_configs(
            lmms_root,
            output_root / "task_configs",
            groups,
        )
        receipt["resolved_task_configs"] = resolved_task_configs
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    if args.shards > 1:
        task = groups[0][0]
        full_docs_per_leaf, leaf_count = FULL_TASK_DOCS_PER_LEAF[task]
        docs_per_leaf = (
            full_docs_per_leaf
            if args.limit is None
            else min(full_docs_per_leaf, int(args.limit))
        )
        if docs_per_leaf <= 0:
            raise ValueError("Sharded --limit must select at least one document")
        shard_span = math.ceil(docs_per_leaf / args.shards)
        shard_root = (
            Path(args.shard_root).resolve()
            if args.shard_root
            else output_root.with_name(f".{output_root.name}.shards")
        )
        shard_root.mkdir(parents=True, exist_ok=True)

        def worker_env(cuda_index: int) -> dict[str, str]:
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=str(cuda_index),
                RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                LMMS_EVAL_PLUGINS="pixelumm_lmms",
            )
            env["PYTHONPATH"] = ":".join(
                path for path in (
                    str(EVAL_CODE_ROOT), str(HERE), str(code_root), str(lmms_root),
                    env.get("PYTHONPATH", ""),
                ) if path
            )
            env["HF_HOME"] = resolve_vlm_hf_home("video4", env)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            return env

        def lmms_command(
            *, trace_path: Path, output_path: Path, resume_trace: bool,
            limit: int | float | None = None, offset: int | None = None,
        ) -> list[str]:
            model_args = ",".join(
                (
                    f"checkpoint_path={checkpoint}",
                    f"code_root={code_root}",
                    "dtype=bf16",
                    f"max_new_tokens_cap={args.max_new_tokens_cap}",
                    f"decode_timeout_seconds={args.decode_timeout_seconds}",
                    f"trace_path={trace_path}",
                    f"resume_trace={str(resume_trace).lower()}",
                )
            )
            command = [
                str(python), "-m", "eval.vlm.video_4.lmms_singleton",
                "--model", "pixelumm_video_4",
                "--model_args", model_args,
                "--tasks", task,
                "--include_path", str((output_root / "task_configs").resolve()),
                "--batch_size", "1",
                "--output_path", str(output_path),
                "--log_samples",
                "--verbosity", "INFO",
            ]
            if limit is not None:
                command.extend(("--limit", str(limit)))
            if offset is not None:
                command.extend(("--offset", str(offset)))
            return command

        shard_processes = []
        expected_total = docs_per_leaf * leaf_count
        for shard_index in range(args.shards):
            offset = shard_index * shard_span
            selected = max(0, min(shard_span, docs_per_leaf - offset))
            if selected == 0:
                continue
            shard_dir = shard_root / f"shard_{shard_index:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            trace_path = shard_dir / "generation_trace.jsonl"
            expected_shard = selected * leaf_count
            existing = _trace_records(trace_path)
            existing_keys = {_trace_key(record) for record in existing}
            if len(existing) != len(existing_keys) or len(existing) > expected_shard:
                raise RuntimeError(
                    f"Invalid shard trace coverage: {trace_path} "
                    f"records={len(existing)} unique={len(existing_keys)} "
                    f"expected_at_most={expected_shard}"
                )
            if len(existing) == expected_shard:
                print(f"[shard {shard_index}] trace already complete ({expected_shard})")
                continue
            log_handle = (shard_dir / "worker.log").open("a" if existing else "w")
            process = subprocess.Popen(
                lmms_command(
                    trace_path=trace_path,
                    output_path=shard_dir,
                    resume_trace=bool(existing),
                    limit=selected,
                    offset=offset,
                ),
                env=worker_env(shard_index),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            print(
                f"[shard {shard_index}] task={task} offset={offset} "
                f"docs_per_leaf={selected} CUDA={shard_index}"
            )
            shard_processes.append((shard_index, process, log_handle))

        failures = []
        for shard_index, process, log_handle in shard_processes:
            return_code = process.wait()
            log_handle.close()
            if return_code:
                failures.append((shard_index, return_code))
        if failures:
            raise RuntimeError(f"Video4 shard workers failed: {failures}")

        merged = []
        for shard_index in range(args.shards):
            merged.extend(_trace_records(
                shard_root / f"shard_{shard_index:02d}" / "generation_trace.jsonl"
            ))
        merged_by_key = {_trace_key(record): record for record in merged}
        if len(merged) != len(merged_by_key) or len(merged) != expected_total:
            raise RuntimeError(
                f"Merged shard trace coverage mismatch: records={len(merged)} "
                f"unique={len(merged_by_key)} expected={expected_total}"
            )
        merged = sorted(
            merged_by_key.values(),
            key=lambda record: (
                str(record["task"]), str(record["split"]),
                (0, int(record["doc_id"]))
                if str(record["doc_id"]).isdigit()
                else (1, str(record["doc_id"])),
            ),
        )
        final_dir = output_root / f"worker_00_{task}"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_trace = final_dir / "generation_trace.jsonl"
        final_trace.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged)
        )
        full_results = sorted(final_dir.glob("**/*_results.json"))
        if not full_results:
            log_handle = (final_dir / "worker.log").open("a")
            process = subprocess.Popen(
                lmms_command(
                    trace_path=final_trace,
                    output_path=final_dir,
                    resume_trace=True,
                    limit=args.limit,
                ),
                env=worker_env(0),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            return_code = process.wait()
            log_handle.close()
            if return_code:
                raise RuntimeError(f"Video4 exact full-score replay failed: {return_code}")
        subprocess.run(
            [str(python), str(HERE / "aggregate.py"), str(output_root)], check=True
        )
        return

    processes = []
    for worker_index, (tasks, resolved_tasks) in enumerate(zip(groups, resolved_groups)):
        label = "__".join(tasks)
        worker_dir = output_root / f"worker_{worker_index:02d}_{label}"
        worker_dir.mkdir(exist_ok=args.resume)
        trace_path = worker_dir / "generation_trace.jsonl"
        model_args = ",".join(
            (
                f"checkpoint_path={checkpoint}",
                f"code_root={code_root}",
                "dtype=bf16",
                f"max_new_tokens_cap={args.max_new_tokens_cap}",
                f"decode_timeout_seconds={args.decode_timeout_seconds}",
                f"trace_path={trace_path}",
                f"resume_trace={str(args.resume).lower()}",
            )
        )
        command = [
            str(python), "-m", "eval.vlm.video_4.lmms_singleton",
            "--model", "pixelumm_video_4",
            "--model_args", model_args,
            "--tasks", ",".join(resolved_tasks),
            "--include_path", str((output_root / "task_configs").resolve()),
            "--batch_size", "1",
            "--output_path", str(worker_dir),
            "--log_samples",
            "--verbosity", "INFO",
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(worker_index),
            RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
            LMMS_EVAL_PLUGINS="pixelumm_lmms",
        )
        env["PYTHONPATH"] = ":".join(
            path for path in (
                str(EVAL_CODE_ROOT), str(HERE), str(code_root), str(lmms_root),
                env.get("PYTHONPATH", ""),
            ) if path
        )
        env["HF_HOME"] = resolve_vlm_hf_home("video4", env)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_handle = (worker_dir / "worker.log").open("a" if args.resume else "w")
        print(f"[worker {worker_index}] tasks={tasks} CUDA={worker_index}")
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((worker_index, tasks, process, log_handle))
    failures = []
    for worker_index, tasks, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append((worker_index, tasks, return_code))
    if failures:
        raise RuntimeError(f"Video4 workers failed: {failures}")
    subprocess.run([str(python), str(HERE / "aggregate.py"), str(output_root)], check=True)


if __name__ == "__main__":
    main()
