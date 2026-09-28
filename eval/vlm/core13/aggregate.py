# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate task metrics emitted by independent Regular21 workers."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

try:
    from .dashboard import write_step_summary, write_suite_dashboard
except ImportError:  # Direct script execution inside the eval container.
    from dashboard import write_step_summary, write_suite_dashboard


def _percentile(values: list[int], fraction: float) -> float:
    """Linear percentile without adding a numpy dependency to aggregation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate_reasoning_traces(output_root: Path, receipt: dict) -> dict:
    trace_files = sorted(output_root.glob("**/worker_*/reasoning_outputs.jsonl"))
    if not trace_files:
        raise FileNotFoundError(f"No reasoning trace JSONL under {output_root}")

    expected_mode = receipt.get("reasoning_mode")
    records_by_request = {}
    for path in trace_files:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("reasoning_mode") != expected_mode:
                raise RuntimeError(
                    f"Reasoning-mode mismatch in {path}:{line_number}: "
                    f"expected={expected_mode} observed={record.get('reasoning_mode')}"
                )
            identity_complete = all(
                record.get(key) is not None
                for key in ("task", "split", "doc_id", "instruction_sha256")
            )
            request_key = (
                (
                    str(record["task"]),
                    str(record["split"]),
                    str(record["doc_id"]),
                    str(record["instruction_sha256"]),
                    json.dumps(
                        record.get("generation_kwargs", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                if identity_complete
                else (str(path), str(line_number))
            )
            previous = records_by_request.get(request_key)
            if previous is not None:
                comparable_keys = (
                    "raw_output",
                    "scored_output",
                    "generated_tokens",
                    "hit_eos",
                    "reasoning_mode",
                )
                mismatched = [
                    key
                    for key in comparable_keys
                    if previous.get(key) != record.get(key)
                ]
                if mismatched:
                    raise RuntimeError(
                        "Conflicting duplicate reasoning trace for "
                        f"task={record.get('task')} doc_id={record.get('doc_id')}: "
                        f"keys={mismatched}"
                    )
                continue
            records_by_request[request_key] = record
    records = list(records_by_request.values())
    if not records:
        raise RuntimeError("Regular21 reasoning traces contain no generations")

    def summarize(items: list[dict]) -> dict:
        token_counts = [int(item["generated_tokens"]) for item in items]
        return {
            "samples": len(items),
            "reasoning_started": sum(bool(item["reasoning_started"]) for item in items),
            "reasoning_completed": sum(bool(item["reasoning_completed"]) for item in items),
            "reasoning_incomplete": sum(bool(item["reasoning_incomplete"]) for item in items),
            "hit_eos": sum(bool(item["hit_eos"]) for item in items),
            "length_truncated": sum(not bool(item["hit_eos"]) for item in items),
            "generated_tokens_mean": statistics.fmean(token_counts),
            "generated_tokens_p50": _percentile(token_counts, 0.50),
            "generated_tokens_p95": _percentile(token_counts, 0.95),
            "generated_tokens_max": max(token_counts),
        }

    by_task = {}
    for task in sorted({str(record["task"]) for record in records}):
        by_task[task] = summarize(
            [record for record in records if str(record["task"]) == task]
        )
    return {
        "reasoning_mode": expected_mode,
        "overall": summarize(records),
        "by_task": by_task,
        "trace_files": [str(path.relative_to(output_root)) for path in trace_files],
    }


def _load_or_compose_receipt(
    output_root: Path,
    *,
    allow_mixed_source_sha256: bool = False,
) -> dict:
    """Load a monolithic receipt or compose one from resumable task runs."""
    receipt_path = output_root / "run_receipt.json"
    if receipt_path.is_file():
        return json.loads(receipt_path.read_text())

    plan_path = output_root / "suite_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"Missing Core benchmark run receipt or suite plan under {output_root}"
        )
    plan = json.loads(plan_path.read_text())
    task_receipts = sorted(output_root.glob("task_runs/*/run_receipt.json"))
    if not task_receipts:
        raise FileNotFoundError(f"No completed task-run receipts under {output_root}")

    receipts = [json.loads(path.read_text()) for path in task_receipts]
    invariant_keys = (
        "contract",
        "contract_sha256",
        "checkpoint",
        "code_root",
        "lmms_root",
        "reasoning_mode",
        "reasoning_mode_contract",
    )
    reference = receipts[0]
    for path, receipt in zip(task_receipts[1:], receipts[1:]):
        mismatched = [
            key for key in invariant_keys if receipt.get(key) != reference.get(key)
        ]
        if receipt.get("source_sha256") != reference.get("source_sha256"):
            mismatched.append("source_sha256")
        if (
            allow_mixed_source_sha256
            and mismatched == ["source_sha256"]
        ):
            continue
        if mismatched:
            raise RuntimeError(
                f"Task-run receipt mismatch in {path}: keys={mismatched}"
            )

    planned_groups = plan.get("groups", [])
    planned_tasks = {
        task
        for group in planned_groups
        for task in group
    }
    observed_groups = [
        group
        for receipt in receipts
        for group in receipt.get("groups", [])
    ]
    observed_tasks = [
        task
        for group in observed_groups
        for task in group
    ]
    duplicates = sorted({
        task for task in observed_tasks if observed_tasks.count(task) > 1
    })
    if set(observed_tasks) != planned_tasks or duplicates:
        raise RuntimeError(
            "Core task-run receipt coverage mismatch: "
            f"missing={sorted(planned_tasks - set(observed_tasks))} "
            f"extra={sorted(set(observed_tasks) - planned_tasks)} "
            f"duplicates={duplicates}"
        )
    for key in ("checkpoint", "reasoning_mode", "contract_sha256"):
        if plan.get(key) != reference.get(key):
            raise RuntimeError(
                f"Core suite-plan mismatch for {key}: "
                f"plan={plan.get(key)!r} receipt={reference.get(key)!r}"
            )

    composed = dict(reference)
    source_sha256_by_task_run = {
        str(path.relative_to(output_root)): receipt.get("source_sha256")
        for path, receipt in zip(task_receipts, receipts)
    }
    source_sha256_values = sorted(set(source_sha256_by_task_run.values()))
    composed.update(
        created_at_utc=plan["created_at_utc"],
        workers=len(observed_groups),
        groups=planned_groups,
        task_run_receipts=[
            str(path.relative_to(output_root)) for path in task_receipts
        ],
        execution_mode="resumable_task_jobs",
    )
    if len(source_sha256_values) > 1:
        composed.update(
            source_sha256="mixed",
            source_sha256_values=source_sha256_values,
            source_sha256_by_task_run=source_sha256_by_task_run,
            mixed_source_sha256_explicitly_allowed=True,
        )
    receipt_path.write_text(json.dumps(composed, indent=2, sort_keys=True) + "\n")
    return composed


def _group_leaf_tasks(group: str, hierarchy: dict[str, list[str]]) -> set[str]:
    """Return the executable lmms leaves below a requested task or group."""
    children = [str(child) for child in hierarchy.get(group, ())]
    if not children:
        return {group}
    leaves = set()
    for child in children:
        leaves.update(_group_leaf_tasks(child, hierarchy))
    return leaves


def main(output_root: str, *, allow_mixed_source_sha256: bool = False):
    output_root = Path(output_root).resolve()
    receipt = _load_or_compose_receipt(
        output_root,
        allow_mixed_source_sha256=allow_mixed_source_sha256,
    )
    expected_tasks = {
        task
        for group in receipt.get("groups", [])
        for task in group
    }
    if not expected_tasks:
        raise RuntimeError("Regular21 run receipt contains no expected tasks")

    # The vendored Tuna lmms-eval fork writes TIMESTAMP_results.json.  Keep the
    # older results_TIMESTAMP.json spelling readable for existing receipts.
    result_files = sorted(
        set(output_root.glob("**/worker_*/**/*_results.json"))
        | set(output_root.glob("**/worker_*/**/results_*.json"))
    )
    if not result_files:
        raise FileNotFoundError(f"No lmms-eval results under {output_root}")

    tasks = {}
    task_details = {}
    sources = {}
    expected_trace_tasks = set()
    for path in result_files:
        payload = json.loads(path.read_text())
        results = payload.get("results", {})
        hierarchy = {
            str(group): [str(child) for child in children]
            for group, children in payload.get("group_subtasks", {}).items()
        }
        requested_here = expected_tasks.intersection(results)
        allowed_here = set(requested_here)
        for requested in requested_here:
            leaves = _group_leaf_tasks(requested, hierarchy)
            expected_trace_tasks.update(leaves)
            allowed_here.update(leaves)
        unexpected = set(results).difference(allowed_here)
        if unexpected:
            raise RuntimeError(
                f"Unexpected lmms result tasks in {path}: {sorted(unexpected)}"
            )
        for task, metrics in results.items():
            if task in task_details:
                raise RuntimeError(f"Duplicate Core task result: {task}")
            task_details[task] = metrics
            if task in expected_tasks:
                tasks[task] = metrics
                sources[task] = str(path.relative_to(output_root))
    observed_tasks = set(tasks)
    if observed_tasks != expected_tasks:
        missing = sorted(expected_tasks - observed_tasks)
        extra = sorted(observed_tasks - expected_tasks)
        raise RuntimeError(
            "Core result coverage mismatch: "
            f"missing={missing} extra={extra}"
        )
    reasoning = _aggregate_reasoning_traces(output_root, receipt)
    if set(reasoning["by_task"]) != expected_trace_tasks:
        raise RuntimeError(
            "Core reasoning trace task coverage mismatch: "
            f"expected={sorted(expected_trace_tasks)} "
            f"observed={sorted(reasoning['by_task'])}"
        )
    summary = {
        "tasks": tasks,
        "task_details": task_details,
        "sources": sources,
        "num_tasks": len(tasks),
        "num_lmms_leaf_tasks": len(expected_trace_tasks),
        "lmms_leaf_tasks": sorted(expected_trace_tasks),
        "reasoning": reasoning,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_step_summary(output_root, summary)
    write_suite_dashboard(output_root.parent)

    if not receipt["contract"].startswith("pixelumm-regular21"):
        raise ValueError(f"Unexpected PixelUMM image suite: {receipt['contract']!r}")
    suite_name = "Regular-21 official"
    lines = [f"# PixelUMM {suite_name} results", ""]
    lines.extend((
        f"- Reasoning mode: `{reasoning['reasoning_mode']}`",
        f"- Generations: {reasoning['overall']['samples']}",
        f"- Length-truncated: {reasoning['overall']['length_truncated']}",
        f"- Incomplete think traces: {reasoning['overall']['reasoning_incomplete']}",
        "",
    ))
    for task in sorted(tasks):
        lines.append(f"## {task}")
        lines.append("")
        for metric, value in sorted(tasks[task].items()):
            if metric.endswith(",none") or not isinstance(value, (dict, list)):
                lines.append(f"- `{metric}`: {value}")
        lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines))
    completion = {
        "contract": receipt["contract"],
        "contract_sha256": receipt["contract_sha256"],
        "source_sha256": receipt.get("source_sha256"),
        "checkpoint": receipt["checkpoint"],
        "num_tasks": len(tasks),
        "reasoning_mode": receipt["reasoning_mode"],
        "num_generations": reasoning["overall"]["samples"],
        "length_truncated": reasoning["overall"]["length_truncated"],
        "reasoning_incomplete": reasoning["overall"]["reasoning_incomplete"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(f"Aggregated {len(tasks)} tasks into {output_root / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument(
        "--allow-mixed-source-sha256",
        action="store_true",
        help=(
            "Explicitly allow task retries produced from different source "
            "snapshots when every other run-receipt invariant matches. The "
            "composed receipt records every source hash instead of hiding it."
        ),
    )
    args = parser.parse_args()
    main(
        args.output_root,
        allow_mixed_source_sha256=args.allow_mixed_source_sha256,
    )
