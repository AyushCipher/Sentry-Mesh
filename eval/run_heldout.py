"""Held-out evaluation for SentryMesh.

    python eval/run_heldout.py run --mode offline                  # deterministic solver baseline
    python eval/run_heldout.py run --mode live --repeats 3         # real LLM (cached, capped, backoff)
    python eval/run_heldout.py run --mode live --set bank --repeats 1
    python eval/run_heldout.py report                              # metrics + bootstrap CIs -> results/heldout_summary.json
    python eval/run_heldout.py tune                                # dev-split threshold tuning -> results/tuning.json

The held-out set (eval/heldout/cases.json) and its precedent corpus
(eval/heldout/precedents.json) are authored independently of the original case
bank.  Raw per-case outputs are written to eval/heldout/results/ and are never
overwritten by `report`/`tune`, which only read them.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.heldout.live_client import Budget, BudgetExceeded, DailyQuotaExhausted, ResilientClient, ResilientLLMClient  # noqa: E402
from eval.metrics import score  # noqa: E402
from sentrymesh.agents.llm_client import LLMClient  # noqa: E402
from sentrymesh.agents.supervisor_agent import (  # noqa: E402
    SupervisorAgent, aggregate_scores, bound_model_assessment, decide, detect_disagreement,
)
from sentrymesh.config import DecisionThresholds, SETTINGS  # noqa: E402
from sentrymesh.event_log import EventLog  # noqa: E402
from sentrymesh.hitl import ReviewQueue  # noqa: E402
from sentrymesh.models import Case, Finding  # noqa: E402
from sentrymesh.retrieval import LocalCosineIndex, load_corpus  # noqa: E402

HELDOUT = ROOT / "eval" / "heldout"
RESULTS = HELDOUT / "results"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------
def load_set(name: str) -> List[Dict[str, Any]]:
    """Return dicts with case inputs + labels."""
    if name == "heldout":
        return json.loads((HELDOUT / "cases.json").read_text(encoding="utf-8"))["cases"]
    from sentrymesh.case_bank import case_files

    out = []
    for p in case_files():
        d = json.loads(p.read_text(encoding="utf-8"))
        d["split"] = "bank"
        d["scenario_family"] = d.get("category")
        d["adversarial"] = None
        out.append(d)
    return out


def precedent_index(name: str) -> LocalCosineIndex:
    if name == "heldout":
        return LocalCosineIndex(load_corpus(HELDOUT / "precedents.json"))
    return LocalCosineIndex(load_corpus())


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------
INFRA_MARK = ("BudgetExceeded", "DailyQuotaExhausted")


async def run_one(mesh: Any, raw: Dict[str, Any], captured: Dict[str, Dict[str, Any]], tag: str) -> Optional[Dict[str, Any]]:
    run_id = f"{raw['case_id']}-{tag}"
    case = Case.from_dict({**raw, "case_id": run_id})
    t0 = time.perf_counter()
    verdict = await mesh.investigate(case)
    if any(any(m in (e.get("error") or "") for m in INFRA_MARK) for e in verdict.run_errors):
        return None  # budget abort: not a measurement
    cap = captured.get(run_id, {})
    return {
        "case_id": raw["case_id"], "split": raw.get("split"), "family": raw.get("scenario_family"),
        "ground_truth": raw["ground_truth"], "adversarial": raw.get("adversarial"),
        "decision": verdict.decision, "combined_risk": verdict.combined_risk, "confidence": verdict.confidence,
        "escalation_reason": verdict.escalation_reason,
        "findings": [{"specialist": f.specialist, "risk_score": f.risk_score, "confidence": f.confidence,
                      "error": f.error, "error_class": f.error_class, "tool_calls": f.tool_calls} for f in verdict.findings],
        "model_assessment": {k: cap.get(k) for k in ("model_combined_risk", "model_confidence", "recommended_action", "policy_overrides")}
        if cap else None,
        "run_errors": verdict.run_errors,
        "clean": not verdict.run_errors,
        "duration_s": round(time.perf_counter() - t0, 2),
        "verdict_rationale": verdict.rationale[:600],
    }


async def build_mesh(llm: LLMClient, set_name: str, captured: Dict[str, Dict[str, Any]]) -> Any:
    log = EventLog.in_memory()
    queue = ReviewQueue.in_memory(log)
    sup = SupervisorAgent(log, llm, precedent_index=precedent_index(set_name), review_queue=queue)
    orig = sup._assess

    async def spy(case, findings, disagreement, fb_r, fb_c):  # records the model's raw numbers for threshold replay
        out = await orig(case, findings, disagreement, fb_r, fb_c)
        captured[case.case_id] = dict(out[0])
        return out

    sup._assess = spy  # type: ignore[method-assign]
    return sup


async def run_mode(args: argparse.Namespace) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = load_set(args.set)
    if args.limit:
        cases = cases[: args.limit]
    budget = Budget(args.max_calls, args.max_cost)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path = RESULTS / f"{args.set}_{args.mode}_manifest.json"
    runs: List[Dict[str, Any]] = []

    for rep in range(1, args.repeats + 1):
        tag = f"r{rep}"
        out_path = RESULTS / f"{args.set}_{args.mode}_rep{rep}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(l)["case_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        todo = [c for c in cases if c["case_id"] not in done]
        captured: Dict[str, Dict[str, Any]] = {}
        if args.mode == "live":
            rc = ResilientClient(cache_path=RESULTS / "llm_cache" / f"{args.set}.jsonl", budget=budget, concurrency=args.llm_concurrency, repeat_tag=tag)
            llm: LLMClient = ResilientLLMClient(rc)
        else:
            llm = LLMClient(backend="offline")
        sup = await build_mesh(llm, args.set, captured)
        mesh = type("M", (), {"investigate": staticmethod(sup.investigate)})
        sem = asyncio.Semaphore(args.case_concurrency)
        completed = skipped_budget = 0
        stop = False
        stop_reason: List[str] = []
        lock = asyncio.Lock()
        print(f"[{args.set}/{args.mode}] repeat {rep}/{args.repeats}: {len(todo)} to run, {len(done)} already on disk", flush=True)

        async def worker(raw: Dict[str, Any]) -> None:
            nonlocal completed, skipped_budget, stop
            async with sem:
                if stop:
                    skipped_budget += 1
                    return
                try:
                    row = await run_one(mesh, raw, captured, tag)
                except BudgetExceeded as exc:
                    stop_reason.append(type(exc).__name__)
                    row = None
                if row is None:
                    stop = True
                    skipped_budget += 1
                    return
                async with lock:
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                    completed += 1
                    if completed % 5 == 0 or completed == len(todo):
                        print(f"  rep{rep}: {completed}/{len(todo)} done | calls={budget.live_calls} cache={budget.cache_hits} "
                              f"retries={budget.retries} est=${budget.cost:.3f}", flush=True)

        await asyncio.gather(*(worker(c) for c in todo))
        runs.append({"repeat": rep, "file": out_path.name, "cases_in_set": len(cases), "completed_this_invocation": completed,
                     "on_disk_total": len(done) + completed, "not_run_or_aborted_due_to_budget": skipped_budget,
                     "stop_reason": sorted(set(stop_reason))})
        if stop:
            print("  budget cap hit; stopping further repeats", flush=True)
            break

    manifest = {"set": args.set, "mode": args.mode, "model": SETTINGS.model if args.mode == "live" else "offline_deterministic",
                "provider_host": SETTINGS.provider_label() if args.mode == "live" else "none", "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repeats_requested": args.repeats,
                "case_concurrency": args.case_concurrency, "llm_concurrency": args.llm_concurrency,
                "specialist_concurrency": SETTINGS.max_concurrent_specialists,
                "thresholds": SETTINGS.thresholds.__dict__, "runs": runs, "budget": budget.snapshot()}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest["budget"], indent=2))
    return 0


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------
def _div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def compute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    auto = score(rows, escalation_as_flag=False)
    flag = score(rows, escalation_as_flag=True)
    n = len(rows)
    esc = sum(r["decision"] == "escalate" for r in rows)
    resolved = [r for r in rows if r["decision"] in ("auto_approve", "auto_decline")]
    ok_dec = sum((r["decision"] == "auto_decline" and r["ground_truth"] == "fraud") or
                 (r["decision"] == "auto_approve" and r["ground_truth"] == "legitimate") for r in resolved)
    resolved_dec = [r for r in resolved if r["ground_truth"] != "ambiguous"]
    amb = [r for r in rows if r["ground_truth"] == "ambiguous"]
    dec = [r for r in rows if r["ground_truth"] != "ambiguous"]
    fraud = [r for r in rows if r["ground_truth"] == "fraud"]
    return {
        "precision_auto_only": auto.precision, "recall_auto_only": auto.recall, "f1_auto_only": auto.f1,
        "fpr_auto_only": auto.false_positive_rate,
        "precision_escalation_as_flag": flag.precision, "recall_escalation_as_flag": flag.recall, "f1_escalation_as_flag": flag.f1,
        "fpr_escalation_as_flag": flag.false_positive_rate,
        "escalation_rate": _div(esc, n),
        "escalation_rate_on_ambiguous": _div(sum(r["decision"] == "escalate" for r in amb), len(amb)),
        "escalation_rate_on_decisive": _div(sum(r["decision"] == "escalate" for r in dec), len(dec)),
        "auto_resolve_rate": _div(len(resolved), n),
        "auto_resolve_accuracy_all": _div(ok_dec, len(resolved)),  # auto-resolved ambiguous cases count as wrong
        "auto_resolve_accuracy_decisive_only": _div(sum(
            (r["decision"] == "auto_decline" and r["ground_truth"] == "fraud") or
            (r["decision"] == "auto_approve" and r["ground_truth"] == "legitimate") for r in resolved_dec), len(resolved_dec)),
        "fraud_auto_approved_rate": _div(sum(r["decision"] == "auto_approve" for r in fraud), len(fraud)),
        "n_rows": n,
    }


def bootstrap(rows: Sequence[Dict[str, Any]], n_boot: int = 2000, seed: int = 11) -> Dict[str, Dict[str, Any]]:
    """Cluster bootstrap: resample CASES with replacement, keeping all repeats of a case together."""
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    ids = sorted(by_case)
    point = compute_metrics(rows)
    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        pick = [by_case[rng.choice(ids)] for _ in ids]
        m = compute_metrics(list(itertools.chain.from_iterable(pick)))
        for k, v in m.items():
            if v is not None:
                samples[k].append(v)
    out: Dict[str, Dict[str, Any]] = {}
    for k, pt in point.items():
        s = sorted(samples[k])
        if k == "n_rows" or len(s) < 20:
            out[k] = {"point": pt, "ci95": None, "valid_boot_samples": len(s)}
        else:
            out[k] = {"point": None if pt is None else round(pt, 4), "ci95": [round(s[int(.025 * len(s))], 4), round(s[int(.975 * len(s)) - 1], 4)],
                      "valid_boot_samples": len(s)}
    return out


def load_rows(set_name: str, mode: str) -> Dict[int, List[Dict[str, Any]]]:
    reps: Dict[int, List[Dict[str, Any]]] = {}
    for p in sorted(RESULTS.glob(f"{set_name}_{mode}_rep*.jsonl")):
        k = int(p.stem.rsplit("rep", 1)[1])
        reps[k] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return reps


def adversarial_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    adv = [r for r in rows if r.get("adversarial")]
    flipped = [r for r in adv if (r["adversarial"]["target"] == "approve" and r["decision"] == "auto_approve")
               or (r["adversarial"]["target"] == "decline" and r["decision"] == "auto_decline" and r["ground_truth"] != "fraud")]
    return {"rows": len(adv), "injection_succeeded_rows": len(flipped),
            "decisions": {d: sum(r["decision"] == d for r in adv) for d in ("auto_approve", "auto_decline", "escalate")}}


def stability(reps: Dict[int, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if len(reps) < 2:
        return None
    dec: Dict[str, List[str]] = {}
    for rows in reps.values():
        for r in rows:
            dec.setdefault(r["case_id"], []).append(r["decision"])
    multi = {k: v for k, v in dec.items() if len(v) == len(reps)}
    same = sum(len(set(v)) == 1 for v in multi.values())
    return {"cases_with_all_repeats": len(multi), "identical_decision_across_repeats": same,
            "fraction": round(same / len(multi), 4) if multi else None}


def report_mode(args: argparse.Namespace) -> int:
    summary: Dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "note": "All numbers are computed from the saved raw files in eval/heldout/results/. SYNTHETIC data authored by Claude."}
    for set_name in ("heldout", "bank"):
        for mode in ("live", "offline"):
            reps = load_rows(set_name, mode)
            if not reps:
                continue
            allrows = [r for rows in reps.values() for r in rows]
            n_cases = len({r["case_id"] for r in allrows})
            entry: Dict[str, Any] = {
                "repeats_on_disk": {str(k): len(v) for k, v in reps.items()},
                "unique_cases": n_cases,
                "rows_with_any_run_error": sum(not r["clean"] for r in allrows),
                "error_classes": _count(e["error_class"] for r in allrows for e in r["run_errors"]),
                "pooled": bootstrap(allrows),
                "per_repeat_point": {str(k): {m: (round(v, 4) if isinstance(v, float) else v) for m, v in compute_metrics(rows).items()}
                                      for k, rows in reps.items()},
                "clean_rows_only": bootstrap([r for r in allrows if r["clean"]]) if any(r["clean"] for r in allrows) else None,
                "adversarial": adversarial_report(allrows),
                "decision_stability_across_repeats": stability(reps),
                "decision_counts_by_truth": {t: _count(r["decision"] for r in allrows if r["ground_truth"] == t)
                                             for t in ("fraud", "legitimate", "ambiguous")},
            }
            if set_name == "heldout" and mode == "offline" and "heldout_live" in summary:
                live_ids = {r["case_id"] for rows in load_rows("heldout", "live").values() for r in rows}
                entry["matched_to_live_cases"] = bootstrap([r for r in allrows if r["case_id"] in live_ids])
                entry["matched_to_live_case_count"] = len(live_ids)
            if set_name == "heldout":
                for sp in ("dev", "test"):
                    sub = [r for r in allrows if r["split"] == sp]
                    if sub:
                        entry[f"split_{sp}"] = bootstrap(sub)
            mf = RESULTS / f"{set_name}_{mode}_manifest.json"
            if mf.exists():
                m = json.loads(mf.read_text(encoding="utf-8"))
                entry["run_manifest"] = {k: m[k] for k in ("model", "provider_host", "started_at", "finished_at", "repeats_requested", "runs", "budget")}
            summary[f"{set_name}_{mode}"] = entry
    (RESULTS / "heldout_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", RESULTS / "heldout_summary.json")
    for k, v in summary.items():
        if isinstance(v, dict) and "pooled" in v:
            p = v["pooled"]
            print(k, {m: (p[m]["point"], p[m]["ci95"]) for m in ("precision_auto_only", "recall_auto_only", "f1_auto_only", "escalation_rate", "fpr_auto_only")})
    return 0


def _count(it) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in it:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Threshold tuning on the dev split only
# ---------------------------------------------------------------------------
COST = {"fraud_auto_approved": 10.0, "legit_auto_declined": 5.0, "decisive_escalated": 1.0, "ambiguous_auto_resolved": 3.0,
        "fraud_auto_declined": 0.0, "legit_auto_approved": 0.0, "ambiguous_escalated": 0.0}


def decision_cost(rows: Sequence[Dict[str, Any]], decisions: Sequence[str]) -> float:
    total = 0.0
    for r, d in zip(rows, decisions):
        t = r["ground_truth"]
        if t == "fraud":
            total += COST["fraud_auto_approved"] if d == "auto_approve" else (COST["decisive_escalated"] if d == "escalate" else 0)
        elif t == "legitimate":
            total += COST["legit_auto_declined"] if d == "auto_decline" else (COST["decisive_escalated"] if d == "escalate" else 0)
        else:
            total += COST["ambiguous_auto_resolved"] if d != "escalate" else 0
    return total


def replay(row: Dict[str, Any], th: DecisionThresholds) -> str:
    ma = row.get("model_assessment")
    if not ma or ma.get("model_combined_risk") is None and ma.get("model_confidence") is None:
        return row["decision"]  # supervisor call failed in the live run: nothing to re-bound
    fs = [Finding(specialist=f["specialist"], risk_score=f["risk_score"], confidence=f["confidence"], reasoning_summary="",
                  error=f["error"], error_class=f["error_class"]) for f in row["findings"]]
    fb_r, fb_c = aggregate_scores(fs)
    dis = detect_disagreement(fs, th)
    b = bound_model_assessment(ma.get("model_combined_risk"), ma.get("model_confidence"), fs, fb_r, fb_c, th)
    return decide(b["combined_risk"], b["confidence"], dis, fs, th, b["force_escalate_reason"])[0]


def tune_mode(args: argparse.Namespace) -> int:
    reps = load_rows("heldout", "live")
    if not reps:
        print("no live heldout rows on disk", file=sys.stderr)
        return 2
    rows = [r for rs in reps.values() for r in rs]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    base = SETTINGS.thresholds
    mismatches = sum(replay(r, base) != r["decision"] for r in rows)

    def th(f: float, d: float) -> DecisionThresholds:
        return DecisionThresholds(**{**base.__dict__, "risk_floor_factor": f, "max_model_risk_divergence": d})

    floors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    divs = [10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 100.0]
    grid = []
    for f, d in itertools.product(floors, divs):
        t = th(f, d)
        dec = [replay(r, t) for r in dev]
        grid.append({"risk_floor_factor": f, "max_model_risk_divergence": d, "dev_cost": decision_cost(dev, dec),
                     "dev_escalation_rate": round(sum(x == "escalate" for x in dec) / max(1, len(dec)), 4)})
    default_cost = next(g["dev_cost"] for g in grid if g["risk_floor_factor"] == 0.9 and g["max_model_risk_divergence"] == 30.0)
    # prefer the default unless something beats it strictly; ties broken by closeness to defaults
    best = min(grid, key=lambda g: (g["dev_cost"], abs(g["risk_floor_factor"] - 0.9) + abs(g["max_model_risk_divergence"] - 30.0) / 100))
    tuned = th(best["risk_floor_factor"], best["max_model_risk_divergence"])

    def eval_split(sub: List[Dict[str, Any]], t: DecisionThresholds) -> Dict[str, Any]:
        d = [replay(r, t) for r in sub]
        rr = [{**r, "decision": x} for r, x in zip(sub, d)]
        return {"cost": decision_cost(sub, d), "cost_per_row": round(decision_cost(sub, d) / max(1, len(sub)), 4),
                "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in compute_metrics(rr).items()}, "n_rows": len(sub)}

    # paired bootstrap over test cases for the cost difference (tuned - default)
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in test:
        by_case.setdefault(r["case_id"], []).append(r)
    diff_per_case = {}
    for cid, rs in by_case.items():
        diff_per_case[cid] = (decision_cost(rs, [replay(r, tuned) for r in rs]) - decision_cost(rs, [replay(r, base) for r in rs])) / len(rs)
    rng, ids, boots = random.Random(5), sorted(diff_per_case), []
    for _ in range(2000):
        boots.append(sum(diff_per_case[rng.choice(ids)] for _ in ids) / len(ids))
    boots.sort()
    ovr = [r for r in rows if r.get("model_assessment") and r["model_assessment"].get("policy_overrides")]
    out = {
        "bounding_activity_at_default": {
            "rows_with_any_policy_override": len(ovr),
            "rows_floor_raised": sum(any("raised" in o for o in r["model_assessment"]["policy_overrides"]) for r in ovr),
            "rows_divergence_forced_escalation": sum(any("diverges" in o for o in r["model_assessment"]["policy_overrides"]) for r in ovr),
            "rows_total": len(rows)},
        "protocol": "grid search over (risk_floor_factor, max_model_risk_divergence) using ONLY dev-split rows of eval/heldout/cases.json; "
                    "evaluation on the test split; supervisor model outputs replayed from saved raw rows (no new LLM calls).",
        "cost_function": COST,
        "replay_sanity_default_thresholds_decision_mismatches": mismatches, "rows_checked": len(rows),
        "dev_rows": len(dev), "test_rows": len(test), "dev_cases": len({r['case_id'] for r in dev}), "test_cases": len({r['case_id'] for r in test}),
        "default": {"risk_floor_factor": base.risk_floor_factor, "max_model_risk_divergence": base.max_model_risk_divergence},
        "default_dev_cost": default_cost,
        "selected": {"risk_floor_factor": best["risk_floor_factor"], "max_model_risk_divergence": best["max_model_risk_divergence"],
                     "dev_cost": best["dev_cost"], "changed_from_default": (best["risk_floor_factor"], best["max_model_risk_divergence"]) != (0.9, 30.0)},
        "dev_before": eval_split(dev, base), "dev_after": eval_split(dev, tuned),
        "test_before": eval_split(test, base), "test_after": eval_split(test, tuned),
        "test_cost_per_row_diff_tuned_minus_default": {"mean": round(sum(diff_per_case.values()) / len(diff_per_case), 4),
                                                         "ci95": [round(boots[int(.025 * len(boots))], 4), round(boots[int(.975 * len(boots)) - 1], 4)]},
        "grid": grid,
        "caveat": "Dev split is small; the selected point can be dev-set noise. A CI on the test diff that spans 0 means no demonstrated benefit.",
    }
    (RESULTS / "tuning.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("replay mismatches vs recorded decisions:", mismatches, "/", len(rows))
    print("selected", out["selected"], "default dev cost", default_cost)
    print("test cost/row before", out["test_before"]["cost_per_row"], "after", out["test_after"]["cost_per_row"],
          "diff CI", out["test_cost_per_row_diff_tuned_minus_default"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["offline", "live"], required=True)
    r.add_argument("--set", choices=["heldout", "bank"], default="heldout")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-calls", type=int, default=6000)
    r.add_argument("--max-cost", type=float, default=2.5, help="estimated-spend hard cap in USD")
    r.add_argument("--case-concurrency", type=int, default=3)
    r.add_argument("--llm-concurrency", type=int, default=6)
    sub.add_parser("report")
    sub.add_parser("tune")
    args = ap.parse_args()
    if args.cmd == "run":
        return asyncio.run(run_mode(args))
    return report_mode(args) if args.cmd == "report" else tune_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
