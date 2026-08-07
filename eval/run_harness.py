"""SentryMesh eval harness.

Runs the real system - supervisor, four specialists, event log, review queue -
against every seeded synthetic case, then scores it and interrogates the score.

    python eval/run_harness.py                 # full bank
    python eval/run_harness.py --limit 5       # smoke run
    python eval/run_harness.py --cases SM-001 SM-015
    python eval/run_harness.py --in-memory     # no Redis required

Writes ``eval/results/harness_report.json`` (machine-readable, served at
``/api/eval/report``) and ``eval/results/harness_report.md`` (the same content
as prose).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow `python eval/run_harness.py` from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import analyse_escalation, classify_run_errors, score, scored_breakdown, tally  # noqa: E402
from eval.report_markdown import render_markdown  # noqa: E402
from eval.scrutiny import build_honesty_section  # noqa: E402
from sentrymesh.agents.supervisor_agent import partition  # noqa: E402
from sentrymesh.case_bank import load_cases, raw_case  # noqa: E402
from sentrymesh.config import RESULTS_DIR  # noqa: E402
from sentrymesh.models import Case  # noqa: E402
from sentrymesh.orchestrator import SentryMesh  # noqa: E402

VERDICT_FOR_EXPECTATION = {"decline": "auto_decline", "approve": "auto_approve", "escalate": "escalate"}


async def run_case(mesh: SentryMesh, case: Case, run_id: str) -> Dict[str, Any]:
    """Investigate one seeded case and flatten the outcome into a result row."""
    seeded = raw_case(case.case_id) or {}
    # A fresh id per run keeps repeated harness runs from interleaving in one
    # case trace, while `case_id` stays the stable key the report is indexed by.
    run_case_id = f"{case.case_id}-{run_id}"
    investigated = Case.from_dict({**case.to_dict(), "case_id": run_case_id})

    started = time.perf_counter()
    harness_error: Optional[Dict[str, str]] = None
    try:
        verdict = await mesh.investigate(investigated)
    except Exception as exc:  # noqa: BLE001 - a crashed run is a reportable outcome
        await mesh.log.append(
            run_case_id, "harness", "error", {"stage": "investigate", "error": f"{type(exc).__name__}: {exc}"}
        )
        return {
            "case_id": case.case_id,
            "run_case_id": run_case_id,
            "category": seeded.get("category"),
            "ground_truth": seeded.get("ground_truth"),
            "expected_verdict": seeded.get("expected_verdict"),
            "difficulty": seeded.get("difficulty"),
            "summary": seeded.get("summary"),
            "planted_signals": seeded.get("planted_signals", []),
            "decision": None,
            "prediction_correct": False,
            "verdict_matches_expected": False,
            "findings": [],
            "run_errors": [{"actor": "harness", "error_class": "harness_exception", "error": f"{type(exc).__name__}: {exc}"}],
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    contributing, abstained, failed = partition(verdict.findings)
    expected = VERDICT_FOR_EXPECTATION.get(str(seeded.get("expected_verdict")), "escalate")

    truth = seeded.get("ground_truth")
    if truth == "fraud":
        prediction_correct = verdict.decision == "auto_decline"
    elif truth == "legitimate":
        prediction_correct = verdict.decision == "auto_approve"
    else:  # ambiguous cases are judged on whether the system declined to commit
        prediction_correct = verdict.decision == "escalate"

    return {
        "case_id": case.case_id,
        "run_case_id": run_case_id,
        "category": seeded.get("category"),
        "ground_truth": truth,
        "expected_verdict": seeded.get("expected_verdict"),
        "difficulty": seeded.get("difficulty"),
        "summary": seeded.get("summary"),
        "planted_signals": seeded.get("planted_signals", []),
        "decision": verdict.decision,
        "combined_risk": verdict.combined_risk,
        "confidence": verdict.confidence,
        "rationale": verdict.rationale,
        "escalation_reason": verdict.escalation_reason,
        "disagreement": verdict.disagreement.to_dict(),
        "verdict_matches_expected": verdict.decision == expected,
        "prediction_correct": prediction_correct,
        "contributing_specialists": len(contributing),
        "abstaining_specialists": len(abstained),
        "failed_specialists": len(failed),
        "findings": [f.to_dict() for f in verdict.findings],
        "run_errors": verdict.run_errors,
        "duration_ms": verdict.duration_ms,
    }


async def main_async(args: argparse.Namespace) -> int:
    run_id = uuid.uuid4().hex[:6]
    started_at = datetime.now(timezone.utc)

    cases = load_cases()
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c.case_id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    mesh = await SentryMesh.create(in_memory=args.in_memory)
    runtime = mesh.runtime()

    print(f"SentryMesh eval harness - run {run_id}")
    print(f"  cases          : {len(cases)}")
    print(f"  run mode       : {runtime['run_mode']}" + (f" ({runtime['model']})" if runtime["model"] else ""))
    print(f"  event log      : {runtime['event_log_backend']}")
    print(f"  review queue   : {runtime['review_queue_backend']}")
    print(f"  retrieval      : {runtime['retrieval_backend']}")
    print()

    results: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        row = await run_case(mesh, case, run_id)
        results.append(row)
        marker = "ok " if row["prediction_correct"] else "MISS"
        print(
            f"  [{index:>2}/{len(cases)}] {marker} {row['case_id']} "
            f"{str(row['ground_truth']):<11} -> {str(row['decision']):<13} "
            f"risk={row.get('combined_risk', 0) or 0:6.2f} conf={row.get('confidence', 0) or 0:.2f} "
            f"({row['duration_ms']}ms)"
        )
    wall_ms = int((time.perf_counter() - wall_start) * 1000)

    decisive = [r for r in results if r["ground_truth"] in {"fraud", "legitimate"}]
    overall = {
        "auto_only": score(decisive, escalation_as_flag=False).to_dict(),
        "escalation_as_flag": score(decisive, escalation_as_flag=True).to_dict(),
    }
    by_category = scored_breakdown(results, "category")
    by_difficulty = scored_breakdown(results, "difficulty")
    escalation = analyse_escalation(results).to_dict()
    errors = classify_run_errors(results)

    honesty = build_honesty_section(results, overall, by_category, by_difficulty, escalation, errors, runtime)

    report = {
        "report_version": 1,
        "run_id": run_id,
        "generated_at": started_at.isoformat(timespec="seconds"),
        "wall_clock_ms": wall_ms,
        "data_disclaimer": "All cases are SYNTHETIC. No real financial data is represented anywhere in "
        "this system or this report.",
        "runtime": runtime,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "case_bank": {
            "cases_run": len(results),
            "ground_truth_distribution": tally(results, "ground_truth"),
            "category_distribution": tally(results, "category"),
            "difficulty_distribution": tally(results, "difficulty"),
        },
        "overall": overall,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "escalation_analysis": escalation,
        "error_classification": errors,
        "honesty_and_self_scrutiny": honesty,
        "case_results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "harness_report.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")

    md_path = RESULTS_DIR / "harness_report.md"
    with md_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(report))

    print()
    print(f"  wall clock     : {wall_ms}ms")
    print(f"  auto-resolved  : {escalation['auto_resolved']}/{escalation['total_cases']}")
    print(f"  escalated      : {escalation['escalated']}/{escalation['total_cases']}")
    a = overall["auto_only"]
    f = overall["escalation_as_flag"]
    print(f"  auto_only      : P={a['precision']} R={a['recall']} F1={a['f1']} FPR={a['false_positive_rate']}")
    print(f"  esc-as-flag    : P={f['precision']} R={f['recall']} F1={f['f1']} FPR={f['false_positive_rate']}")
    print()
    print("  HONESTY SECTION")
    print(f"  {honesty['headline']}")
    for concern in honesty["concerns"]:
        print(f"   - {concern[:160]}{'...' if len(concern) > 160 else ''}")
    print()
    print(f"  wrote {json_path}")
    print(f"  wrote {md_path}")

    await mesh.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SentryMesh against the seeded synthetic case bank.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--cases", nargs="*", default=None, help="Run only these case ids.")
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use the in-memory event log and review queue instead of Redis.",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
