# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and aggregate the four independent official video task outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from .dashboard import write_summary_html
except ImportError:  # Direct script execution inside the eval container.
    from dashboard import write_summary_html


def leaf_tasks(group: str, hierarchy: dict[str, list[str]]) -> set[str]:
    children = [str(child) for child in hierarchy.get(group, ())]
    if not children:
        return {group}
    leaves: set[str] = set()
    for child in children:
        leaves.update(leaf_tasks(child, hierarchy))
    return leaves


def videomme_duration_breakdown(output_root: Path, task: str) -> dict | None:
    """Recover official duration buckets from lmms per-example score records."""
    candidates = sorted(
        output_root.glob(f"**/worker_*/**/*_samples_{task}.jsonl")
    )
    best = None
    for path in candidates:
        buckets = {name: {"correct": 0.0, "questions": 0} for name in (
            "short", "medium", "long", "overall",
        )}
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                score_record = record["videomme_perception_score"]
                duration = str(score_record["duration"])
                score = float(score_record["score"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed Video-MME sample {path}:{line_number}") from exc
            if duration not in {"short", "medium", "long"}:
                raise ValueError(f"Unknown Video-MME duration {duration!r} in {path}")
            for bucket in (duration, "overall"):
                buckets[bucket]["correct"] += score
                buckets[bucket]["questions"] += 1
        if best is None or buckets["overall"]["questions"] > best["overall"]["questions"]:
            best = buckets
    if best is None:
        return None
    for values in best.values():
        count = values["questions"]
        values["accuracy"] = 100.0 * values["correct"] / count if count else 0.0
    return best


def main(output_root: str) -> None:
    output_root = Path(output_root).resolve()
    receipt_path = output_root / "run_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
    else:
        plan_path = output_root / "suite_plan.json"
        task_receipt_paths = sorted(output_root.glob("task_runs/*/run_receipt.json"))
        if not plan_path.is_file() or not task_receipt_paths:
            raise FileNotFoundError(
                f"Missing monolithic receipt or resumable task receipts under {output_root}"
            )
        plan = json.loads(plan_path.read_text())
        task_receipts = [json.loads(path.read_text()) for path in task_receipt_paths]
        reference = task_receipts[0]
        invariant_keys = (
            "contract", "contract_sha256", "checkpoint", "code_root",
            "lmms_root", "model_input",
        )
        for path, task_receipt in zip(task_receipt_paths[1:], task_receipts[1:]):
            mismatched = [
                key for key in invariant_keys
                if task_receipt.get(key) != reference.get(key)
            ]
            if mismatched:
                raise RuntimeError(f"Task receipt mismatch in {path}: {mismatched}")
        observed = [
            task for task_receipt in task_receipts
            for group in task_receipt["groups"] for task in group
        ]
        planned = [task for group in plan["groups"] for task in group]
        if sorted(observed) != sorted(planned) or len(observed) != len(set(observed)):
            raise RuntimeError(
                f"Task receipt coverage mismatch: planned={planned} observed={observed}"
            )
        for key in ("contract", "contract_sha256", "checkpoint"):
            if plan.get(key) != reference.get(key):
                raise RuntimeError(
                    f"Suite plan mismatch for {key}: "
                    f"plan={plan.get(key)!r} receipt={reference.get(key)!r}"
                )
        receipt = dict(reference)
        receipt.update(
            created_at_utc=plan["created_at_utc"],
            groups=plan["groups"],
            workers=len(plan["groups"]),
            execution_mode="resumable_task_jobs",
            task_run_receipts=[str(path.relative_to(output_root)) for path in task_receipt_paths],
            task_source_sha256={
                task: task_receipt["source_sha256"]
                for task_receipt in task_receipts
                for group in task_receipt["groups"] for task in group
            },
        )
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    expected = {task for group in receipt["groups"] for task in group}
    result_files = sorted(
        set(output_root.glob("**/worker_*/**/*_results.json"))
        | set(output_root.glob("**/worker_*/**/results_*.json"))
    )
    if not result_files:
        raise FileNotFoundError(f"No lmms-eval result JSON under {output_root}")

    requested_results = {}
    task_details = {}
    sources = {}
    trace_leaf_tasks: set[str] = set()
    mvbench_result_files: set[str] = set()
    for path in result_files:
        payload = json.loads(path.read_text())
        results = payload.get("results", {})
        hierarchy = {
            str(group): [str(child) for child in children]
            for group, children in payload.get("group_subtasks", {}).items()
        }
        requested_here = expected.intersection(results)
        allowed = set(requested_here)
        for requested in requested_here:
            leaves = leaf_tasks(requested, hierarchy)
            trace_leaf_tasks.update(leaves)
            allowed.update(leaves)
        if "mvbench" in expected and "mvbench" not in results:
            mvbench_leaves = {
                task for task in results if str(task).startswith("mvbench_")
            }
            if mvbench_leaves:
                allowed.update(mvbench_leaves)
                trace_leaf_tasks.update(mvbench_leaves)
                mvbench_result_files.add(str(path.relative_to(output_root)))
        unexpected = set(results).difference(allowed)
        if unexpected:
            raise RuntimeError(f"Unexpected lmms tasks in {path}: {sorted(unexpected)}")
        for task, metrics in results.items():
            if task in task_details:
                raise RuntimeError(f"Duplicate lmms result for task {task}")
            task_details[task] = metrics
            if task in expected:
                requested_results[task] = metrics
                sources[task] = str(path.relative_to(output_root))
    # lmms-eval v0.7.1 emits an alias-only placeholder for the MVBench group;
    # the official accuracy lives on its 20 leaf tasks.  Treat that placeholder
    # as missing so the count-weighted official group score is synthesized below.
    if "mvbench" in requested_results and not any(
        key.startswith("mvbench_accuracy,") and isinstance(value, (int, float))
        for key, value in requested_results["mvbench"].items()
    ):
        requested_results.pop("mvbench")
        sources.pop("mvbench", None)
    missing_before_trace = expected.difference(requested_results)
    if missing_before_trace.difference({"mvbench"}):
        raise RuntimeError(
            f"Video4 result coverage mismatch: expected={sorted(expected)} "
            f"observed={sorted(requested_results)}"
        )

    trace_files = sorted(output_root.glob("**/worker_*/generation_trace.jsonl"))
    if not trace_files:
        raise FileNotFoundError(f"No Video4 generation traces under {output_root}")
    records = []
    for path in trace_files:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed trace {path}:{line_number}") from exc
    observed_trace_tasks = {str(record["task"]) for record in records}
    if observed_trace_tasks != trace_leaf_tasks:
        raise RuntimeError(
            f"Video4 trace coverage mismatch: expected={sorted(trace_leaf_tasks)} "
            f"observed={sorted(observed_trace_tasks)}"
        )
    if "mvbench" in missing_before_trace:
        mvbench_leaves = sorted(
            task for task in task_details if task.startswith("mvbench_")
        )
        if len(mvbench_leaves) != 20:
            raise RuntimeError(
                f"MVBench requires all 20 leaf tasks, observed={mvbench_leaves}"
            )
        trace_counts = {
            task: sum(str(record["task"]) == task for record in records)
            for task in mvbench_leaves
        }
        metric_values = {}
        for task in mvbench_leaves:
            candidates = {
                key: value
                for key, value in task_details[task].items()
                if key.startswith("mvbench_accuracy,") and isinstance(value, (int, float))
            }
            if len(candidates) != 1 or trace_counts[task] <= 0:
                raise RuntimeError(
                    f"Invalid MVBench leaf metric/count: task={task} "
                    f"metrics={candidates} count={trace_counts[task]}"
                )
            metric_values[task] = float(next(iter(candidates.values())))
        total_count = sum(trace_counts.values())
        score = sum(
            metric_values[task] * trace_counts[task] for task in mvbench_leaves
        ) / total_count
        requested_results["mvbench"] = {"mvbench_accuracy,none": score}
        sources["mvbench"] = sorted(mvbench_result_files)
    if set(requested_results) != expected:
        raise RuntimeError(
            f"Video4 result coverage mismatch after MVBench aggregation: "
            f"expected={sorted(expected)} observed={sorted(requested_results)}"
        )
    sampled_frames = [len(record["decode"]["sampled_frame_indices"]) for record in records]
    packed_tokens = [int(record["decode"]["packed_patch_tokens"]) for record in records]
    representations = Counter(
        str(record["decode"].get("video_representation", "unknown"))
        for record in records
    )
    selection_reasons = Counter(
        str(record["decode"].get("selection_reason", "unknown"))
        for record in records
    )
    summary = {
        "tasks": requested_results,
        "task_details": task_details,
        "sources": sources,
        "num_tasks": len(requested_results),
        "num_lmms_leaf_tasks": len(trace_leaf_tasks),
        "num_generations": len(records),
        "preprocess": {
            "sampled_frames_min": min(sampled_frames),
            "sampled_frames_mean": statistics.fmean(sampled_frames),
            "sampled_frames_max": max(sampled_frames),
            "packed_video_tokens_min": min(packed_tokens),
            "packed_video_tokens_mean": statistics.fmean(packed_tokens),
            "packed_video_tokens_max": max(packed_tokens),
            "representations": dict(sorted(representations.items())),
            "selection_reasons": dict(sorted(selection_reasons.items())),
        },
        "reasoning": {
            "started": sum(bool(record["reasoning_started"]) for record in records),
            "completed": sum(bool(record["reasoning_completed"]) for record in records),
            "incomplete": sum(bool(record["reasoning_incomplete"]) for record in records),
        },
    }
    duration_breakdowns = {
        task: breakdown
        for task in ("videomme",)
        if task in expected
        and (breakdown := videomme_duration_breakdown(output_root, task)) is not None
    }
    if duration_breakdowns:
        summary["duration_breakdowns"] = duration_breakdowns
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# PixelUMM R07 train-aligned Video4 benchmark",
        "",
        f"- Checkpoint: `{receipt['checkpoint']}`",
        f"- Generations: {len(records)}",
        f"- Sampled frames: {summary['preprocess']['sampled_frames_min']} / "
        f"{summary['preprocess']['sampled_frames_mean']:.1f} / "
        f"{summary['preprocess']['sampled_frames_max']} (min/mean/max)",
        "- Evaluator: pinned official lmms-eval v0.7.1 tasks and scorers.",
        f"- Representation profile: `{receipt['representation_profile']}`.",
        f"- Model input: `{json.dumps(receipt.get('model_input', {}), sort_keys=True)}`.",
        "",
    ]
    for task in sorted(requested_results):
        lines.extend((f"## {task}", ""))
        for metric, value in sorted(requested_results[task].items()):
            if metric.endswith(",none") or not isinstance(value, (dict, list)):
                lines.append(f"- `{metric}`: {value}")
        breakdown = duration_breakdowns.get(task)
        if breakdown:
            for bucket in ("overall", "short", "medium", "long"):
                values = breakdown[bucket]
                lines.append(
                    f"- `{bucket}`: {values['accuracy']:.2f} "
                    f"({int(values['correct'])}/{int(values['questions'])})"
                )
        lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines))
    # Independent task workers may aggregate a leaf before the suite finishes.
    # Render the comparison dashboard only for the complete four-task contract.
    official_suite = {
        "mvbench", "videomme", "longvideobench_val_v", "lvbench",
    }
    if expected == official_suite:
        write_summary_html(output_root, summary, receipt)
    completion = {
        "contract": receipt["contract"],
        "contract_sha256": receipt["contract_sha256"],
        "source_sha256": receipt["source_sha256"],
        "checkpoint": receipt["checkpoint"],
        "num_tasks": len(requested_results),
        "num_generations": len(records),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(f"Aggregated official video benchmark results at {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    args = parser.parse_args()
    main(args.output_root)
