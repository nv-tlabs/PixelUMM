#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build self-contained HTML summaries for Core benchmark checkpoints."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from pathlib import Path

import yaml


_BASELINES_PATH = Path(__file__).with_name("comparison_baselines.json")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _metric_value(metrics: dict, metric_name: str) -> tuple[str, float] | None:
    matches = [
        (str(name), float(value))
        for name, value in metrics.items()
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and str(name).split(",", 1)[0] == metric_name
        )
    ]
    if not matches:
        return None
    matches.sort(key=lambda pair: (not pair[0].endswith(",none"), pair[0]))
    return matches[0]


def _display_metric(
    metrics: dict,
    display_spec: dict | None = None,
) -> tuple[str, str]:
    if display_spec:
        if display_spec.get("derived") == "sum":
            selected = [
                _metric_value(metrics, str(metric))
                for metric in display_spec.get("metrics", ())
            ]
            if selected and all(item is not None for item in selected):
                names = [item[0] for item in selected if item is not None]
                value = sum(item[1] for item in selected if item is not None)
                return f"{value:.2f}", " + ".join(names)
        elif "metric" in display_spec:
            selected = _metric_value(metrics, str(display_spec["metric"]))
            if selected is not None:
                name, value = selected
                shown = (
                    f"{value * 100:.2f}"
                    if 0.0 <= value <= 1.0
                    else f"{value:.2f}"
                )
                return shown, name
    scalars = [
        (str(name), value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    reported = [(name, value) for name, value in scalars if name.endswith(",none")]
    candidates = reported or scalars
    if not candidates:
        return "—", "no scalar metric"
    priority = (
        "accuracy", "acc", "exact_match", "anls", "relaxed_overall",
        "f1", "score",
    )
    candidates.sort(
        key=lambda pair: (
            next(
                (index for index, token in enumerate(priority) if token in pair[0].lower()),
                len(priority),
            ),
            pair[0],
        )
    )
    name, value = candidates[0]
    value = float(value)
    shown = f"{value * 100:.2f}" if 0.0 <= value <= 1.0 else f"{value:.2f}"
    return shown, name


def _load_comparison_baselines() -> dict:
    payload = json.loads(_BASELINES_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported comparison baseline schema in {_BASELINES_PATH}: "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _displayed_score(shown: str) -> float | None:
    try:
        value = float(shown)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _score_rankings(
    candidates: dict[str, list[float]],
) -> dict[str, tuple[float, float | None]]:
    rankings = {}
    for task, values in candidates.items():
        distinct = sorted(set(values), reverse=True)
        if distinct:
            rankings[task] = (
                distinct[0],
                distinct[1] if len(distinct) > 1 else None,
            )
    return rankings


def _ranked_score_html(
    shown: str,
    task: str,
    rankings: dict[str, tuple[float, float | None]],
    *,
    eligible: bool = True,
) -> str:
    escaped = html.escape(shown)
    value = _displayed_score(shown) if eligible else None
    if value is None or task not in rankings:
        return escaped
    best, second = rankings[task]
    if value == best:
        return f"<strong>{escaped}</strong>"
    if second is not None and value == second:
        return f"<u>{escaped}</u>"
    return escaped


def _baseline_rows(
    rows: list[dict],
    tasks: list[str],
    rankings: dict[str, tuple[float, float | None]],
) -> str:
    rendered = []
    for row in rows:
        cells = []
        scores = row.get("scores", {})
        for task in tasks:
            value = scores.get(task)
            if value is None:
                shown = "—"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                shown = f"{float(value):.2f}"
            else:
                shown = str(value)
            cells.append(
                f"<td>{_ranked_score_html(shown, task, rankings)}</td>"
            )
        rendered.append(
            f'<tr class="{html.escape(str(row.get("kind", "published")))}">'
            f'<td>{html.escape(str(row["model"]))}</td>'
            f'<td>{html.escape(str(row["source"]))}</td>'
            + "".join(cells)
            + "<td>—</td>"
            + "</tr>"
        )
    return "".join(rendered)


def _experiment_label(results_root: Path) -> str:
    for ancestor in (results_root, *results_root.parents):
        if ancestor.name != "eval":
            continue
        experiment = ancestor.parent.name
        match = re.search(r"(?:^|-)F(\d+)-R(\d+)(?:_|$)", experiment)
        if match:
            return f"PixelUMM F{match.group(1)}-R{match.group(2)}"
        return experiment
    return "PixelUMM checkpoint"


def _completed_task_payload(step_dir: Path) -> tuple[dict, dict]:
    tasks = {}
    reasoning_by_task = {}
    for summary_path in sorted((step_dir / "task_runs").glob("*/summary.json")):
        if not (summary_path.parent / "COMPLETED.json").is_file():
            continue
        summary = json.loads(summary_path.read_text())
        for task, metrics in summary.get("tasks", {}).items():
            if task in tasks:
                raise ValueError(f"Duplicate completed task summary for {task}")
            tasks[task] = metrics
        for task, metrics in summary.get("reasoning", {}).get("by_task", {}).items():
            if task in reasoning_by_task:
                raise ValueError(f"Duplicate completed reasoning summary for {task}")
            reasoning_by_task[task] = metrics
    reasoning = {"by_task": reasoning_by_task} if reasoning_by_task else {}
    return tasks, reasoning


def _step_payload(step_dir: Path, jobs_root: Path | None) -> dict:
    summary_path = step_dir / "summary.json"
    partial_summary_path = step_dir / "partial_summary.json"
    receipt_path = step_dir / "run_receipt.json"
    payload = {
        "step": int(step_dir.name.removeprefix("step_")),
        "status": "pending",
        "tasks": {},
        "reasoning": {},
    }
    if receipt_path.is_file():
        payload["receipt"] = json.loads(receipt_path.read_text())
    else:
        plan_path = step_dir / "suite_plan.json"
        if plan_path.is_file():
            payload["receipt"] = json.loads(plan_path.read_text())
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        payload.update(
            status="complete",
            tasks=summary.get("tasks", {}),
            reasoning=summary.get("reasoning", {}),
        )
        return payload
    if partial_summary_path.is_file():
        partial_summary = json.loads(partial_summary_path.read_text())
        payload.update(
            tasks=partial_summary.get("tasks", {}),
            reasoning=partial_summary.get("reasoning", {}),
            partial=partial_summary,
        )
    else:
        completed_tasks, completed_reasoning = _completed_task_payload(step_dir)
        payload.update(tasks=completed_tasks, reasoning=completed_reasoning)
    if jobs_root is not None:
        job_dir = jobs_root / step_dir.name
        failed = job_dir / "FAILED.json"
        submitted = job_dir / "SUBMITTED.json"
        if failed.is_file():
            payload["status"] = "failed"
            payload["failure"] = json.loads(failed.read_text())
        elif submitted.is_file():
            payload["status"] = "submitted"
        else:
            group_dirs = sorted((job_dir / "groups").glob("*"))
            if group_dirs:
                planned = len(group_dirs)
                complete = len(list((step_dir / "task_runs").glob("*/COMPLETED.json")))
                active = 0
                failed_groups = 0
                skipped_groups = 0
                for group_dir in group_dirs:
                    task_result = step_dir / "task_runs" / group_dir.name
                    if (task_result / "COMPLETED.json").is_file():
                        continue
                    if (task_result / "SKIPPED.json").is_file():
                        skipped_groups += 1
                        continue
                    attempts = sorted(group_dir.glob("attempt_*"))
                    if not attempts:
                        continue
                    latest = attempts[-1]
                    if (latest / "FAILED.json").is_file():
                        failed_groups += 1
                    elif (latest / "SUBMITTED.json").is_file():
                        active += 1
                if failed_groups and active == 0:
                    payload["status"] = (
                        "partial" if payload.get("partial") else "failed"
                    )
                else:
                    payload["status"] = "submitted"
                payload["status_detail"] = (
                    f"{complete}/{planned} task groups complete, "
                    f"{skipped_groups} skipped, {active} active, "
                    f"{failed_groups} failed attempts"
                )
                partial_coverage = payload.get("partial", {}).get("coverage", {})
                if partial_coverage:
                    coverage_detail = ", ".join(
                        f"{task} partial "
                        f"{int(coverage['samples']):,}/{int(coverage['total']):,}"
                        for task, coverage in sorted(partial_coverage.items())
                    )
                    payload["status_detail"] += f"; {coverage_detail}"
    return payload


def write_step_summary(step_dir: Path, summary: dict) -> Path:
    metadata_path = (
        step_dir / "run_receipt.json"
        if (step_dir / "run_receipt.json").is_file()
        else step_dir / "suite_plan.json"
    )
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    suite_name = (
        "Regular-21"
        if str(metadata.get("contract", "")).startswith("pixelumm-regular21")
        else "Core-11"
    )
    rows = []
    for task, metrics in sorted(summary.get("tasks", {}).items()):
        for metric, value in sorted(metrics.items()):
            if isinstance(value, (dict, list)):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(task))}</td>"
                f"<td><code>{html.escape(str(metric))}</code></td>"
                f"<td>{html.escape(str(value))}</td>"
                "</tr>"
            )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{suite_name} {html.escape(step_dir.name)}</title>
<style>body{{font:15px/1.5 system-ui;margin:28px;color:#1f2328}}table{{border-collapse:collapse}}th,td{{border:1px solid #d0d7de;padding:8px 12px;text-align:left}}th{{background:#f6f8fa}}code{{font-size:13px}}</style>
</head><body><h1>{suite_name} {html.escape(step_dir.name)}</h1><p><a href="../index.html">All checkpoints</a></p>
<table><thead><tr><th>Task</th><th>Metric</th><th>Value</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path = step_dir / "summary.html"
    _atomic_write(path, document)
    return path


def write_suite_dashboard(
    results_root: Path,
    jobs_root: Path | None = None,
    contract: dict | None = None,
) -> Path:
    results_root = Path(results_root)
    jobs_root = Path(jobs_root) if jobs_root is not None else None
    step_names = {
        path.name for path in results_root.glob("step_*") if path.is_dir()
    }
    if jobs_root is not None:
        step_names.update(
            path.name for path in jobs_root.glob("step_*") if path.is_dir()
        )
    step_dirs = [
        results_root / name
        for name in sorted(
            step_names,
            key=lambda name: int(name.removeprefix("step_")),
        )
    ]
    payloads = [_step_payload(path, jobs_root) for path in step_dirs]
    baselines = _load_comparison_baselines()
    receipts = [
        payload.get("receipt", {})
        for payload in payloads
        if payload.get("receipt")
    ]
    receipt_contract_names = {
        str(receipt.get("contract", "")) for receipt in receipts
    }
    receipt_contract_names.discard("")
    if len(receipt_contract_names) > 1:
        raise ValueError(
            "Dashboard cannot mix benchmark contracts: "
            f"{sorted(receipt_contract_names)}"
        )
    declared_contract_name = str((contract or {}).get("name", ""))
    if (
        declared_contract_name
        and receipt_contract_names
        and receipt_contract_names != {declared_contract_name}
    ):
        raise ValueError(
            "Dashboard contract does not match result receipts: "
            f"declared={declared_contract_name!r}, "
            f"receipts={sorted(receipt_contract_names)}"
        )
    contract_name = declared_contract_name or next(
        iter(receipt_contract_names), ""
    )
    receipt_task_specs = next(
        (
            receipt.get("task_specs", [])
            for receipt in receipts
            if receipt.get("task_specs")
        ),
        [],
    )
    task_specs = list((contract or {}).get("tasks", ())) or receipt_task_specs
    display_specs = {
        str(task["id"]): task.get("display", {})
        for task in task_specs
        if isinstance(task, dict)
    }
    contract_order = [
        str(task["id"] if isinstance(task, dict) else task)
        for task in task_specs
    ]
    baseline_order = contract_order or list(baselines["task_order"])
    hidden_tasks = set(baselines.get("hidden_tasks", ()))
    observed_tasks = {
        task
        for payload in payloads
        for task in payload["tasks"]
        if task not in hidden_tasks
    }
    tasks = baseline_order + sorted(observed_tasks.difference(baseline_order))
    suite_name = (
        "Regular-21"
        if contract_name.startswith("pixelumm-regular21")
        else "Core-11"
    )
    experiment_label = _experiment_label(results_root)
    rank_candidates = {task: [] for task in tasks}
    for payload in payloads:
        partial_coverage = payload.get("partial", {}).get("coverage", {})
        for task in tasks:
            if task not in payload["tasks"] or task in partial_coverage:
                continue
            shown, _ = _display_metric(
                payload["tasks"][task],
                display_specs.get(task),
            )
            value = _displayed_score(shown)
            if value is not None:
                rank_candidates[task].append(value)
    for baseline in baselines["rows"]:
        for task in tasks:
            value = baseline.get("scores", {}).get(task)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rank_candidates[task].append(
                    float(f"{float(value):.2f}")
                )
    rankings = _score_rankings(rank_candidates)
    rows = []
    for payload in payloads:
        cells = []
        for task in tasks:
            if task not in payload["tasks"]:
                cells.append("<td>—</td>")
                continue
            shown, metric = _display_metric(
                payload["tasks"][task],
                display_specs.get(task),
            )
            coverage = (
                payload.get("partial", {}).get("coverage", {}).get(task)
            )
            if coverage:
                metric += (
                    f"; partial {int(coverage['samples']):,}/"
                    f"{int(coverage['total']):,}"
                )
            cells.append(
                f'<td title="{html.escape(metric)}">'
                f"{_ranked_score_html(shown, task, rankings, eligible=not coverage)}"
                "</td>"
            )
        status = payload["status"]
        status_detail = payload.get("status_detail", "")
        if status == "failed":
            failure = payload.get("failure", {})
            failure_detail = failure.get(
                "slurm_state", failure.get("reason", "")
            )
            status_detail = failure_detail or status_detail
            status_detail = f" — {status_detail}" if status_detail else ""
        elif status_detail:
            status_detail = f" — {status_detail}"
        step_name = f"step_{payload['step']:07d}"
        step_label = (
            f'<a href="{step_name}/summary.html">{payload["step"]:,}</a>'
            if status == "complete"
            else f'{payload["step"]:,}'
        )
        rows.append(
            f'<tr class="{html.escape(status)}">'
            f'<td>{html.escape(experiment_label)}</td>'
            f'<td>{step_label}</td>'
            + "".join(cells)
            + f'<td>{html.escape(status + status_detail)}</td>'
            + "</tr>"
        )
    labels = baselines.get("task_labels", {})
    headers = "".join(
        f"<th>{html.escape(str(labels.get(task, task)))}</th>" for task in tasks
    )
    pixelumm_rows = [
        row
        for row in baselines["rows"]
        if row.get("kind") == "pixelumm_baseline"
    ]
    reference_rows = [
        row
        for row in baselines["rows"]
        if row.get("kind") != "pixelumm_baseline"
    ]
    pixelumm_comparison_rows = _baseline_rows(
        pixelumm_rows, tasks, rankings
    )
    reference_comparison_rows = _baseline_rows(
        reference_rows, tasks, rankings
    )
    separator = (
        f'<tr class="section"><td colspan="{len(tasks) + 3}">'
        "Reference baselines</td></tr>"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PixelUMM {suite_name} checkpoint dashboard</title>
<style>
body{{font:15px/1.5 system-ui;margin:28px;color:#1f2328}}.note{{max-width:1100px;color:#59636e}}.wrap{{overflow-x:auto;border:1px solid #d0d7de;border-radius:8px}}
table{{border-collapse:collapse;min-width:max-content;font-variant-numeric:tabular-nums}}th,td{{border-right:1px solid #d0d7de;border-bottom:1px solid #d0d7de;padding:9px 12px;text-align:right;white-space:nowrap}}th{{background:#f6f8fa;position:sticky;top:0}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{{text-align:left}}td strong{{font-weight:800;color:#b42318}}td u{{color:#b45309;text-decoration-color:#b45309;text-decoration-thickness:2px;text-underline-offset:2px}}tr.complete{{background:#eef7ff;font-weight:650}}tr.partial{{background:#fff8c5}}tr.failed{{background:#ffebe9}}tr.pending,tr.submitted{{color:#656d76}}tr.pixelumm_baseline{{background:#f6f8fa;font-weight:600}}tr.section td{{background:#24292f;color:#fff;font-weight:700;text-align:left;letter-spacing:.02em}}a{{color:#0969da}}
</style></head><body>
<h1>PixelUMM {suite_name} checkpoint dashboard</h1>
<p>Training-aligned <code>standard_bare</code>. Scores in [0,1] are shown as percentages; hover a checkpoint cell for its lmms-eval metric key. Across reference baselines and PixelUMM checkpoints together, best scores in each benchmark column are <strong style="color:#b42318">bold red</strong>; second-best distinct scores are <u style="color:#b45309;text-decoration-color:#b45309">underlined amber</u>.</p>
<div class="wrap"><table><thead><tr><th>Model</th><th>Checkpoint / source</th>{headers}<th>Status</th></tr></thead><tbody>{pixelumm_comparison_rows}{''.join(rows)}{separator}{reference_comparison_rows}</tbody></table></div>
<p class="note">Checkpoint cells whose hover text says <code>partial</code> are provisional scores over the displayed sample coverage, not complete official benchmark results. Published rows retain their source paper's split, prompt, decode policy, and scorer, so they are reference comparisons rather than strict reruns under the PixelUMM harness. In particular, a shared task name does not by itself guarantee an apples-to-apples score.</p>
</body></html>"""
    path = results_root / "index.html"
    _atomic_write(path, document)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--jobs-root", type=Path)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    contract = (
        yaml.safe_load(args.contract.read_text())
        if args.contract is not None
        else None
    )
    for step_dir in sorted(args.results_root.glob("step_*")):
        summary_path = step_dir / "summary.json"
        if summary_path.is_file():
            write_step_summary(step_dir, json.loads(summary_path.read_text()))
    print(write_suite_dashboard(args.results_root, args.jobs_root, contract))


if __name__ == "__main__":
    main()
