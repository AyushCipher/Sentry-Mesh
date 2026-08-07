"""Render the harness report as Markdown.

Same content as the JSON, ordered the way it should be read: what the run
actually was, then the operational view, then the flattering view, then the
scrutiny. Putting the honesty section at the bottom of a long document is how it
gets skipped, so the headline appears near the top as well.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _matrix_row(name: str, m: Dict[str, Any]) -> str:
    return (
        f"| {name} | {m['scored_cases']} | {num(m['precision'])} | {num(m['recall'])} | "
        f"{num(m['f1'])} | {pct(m['false_positive_rate'])} | {pct(m['accuracy'])} | "
        f"{m['true_positive']}/{m['false_positive']}/{m['true_negative']}/{m['false_negative']} |"
    )


def render_markdown(report: Dict[str, Any]) -> str:
    rt = report["runtime"]
    overall = report["overall"]
    honesty = report["honesty_and_self_scrutiny"]
    esc = report["escalation_analysis"]
    err = report["error_classification"]

    out: List[str] = []
    add = out.append

    add("# SentryMesh eval harness report\n")
    add(f"**Run** `{report['run_id']}` - {report['generated_at']} - {report['wall_clock_ms']}ms wall clock\n")
    add(f"> {report['data_disclaimer']}\n")

    add("## What this run actually was\n")
    add("| | |")
    add("|---|---|")
    add(f"| Reasoning backend | `{rt['run_mode']}`" + (f" ({rt['model']})" if rt.get("model") else "") + " |")
    add(f"| Event log | `{rt['event_log_backend']}` |")
    add(f"| Review queue | `{rt['review_queue_backend']}` |")
    add(f"| Precedent retrieval | `{rt['retrieval_backend']}` |")
    add(f"| Cases run | {report['case_bank']['cases_run']} |")
    add(f"| Ground truth | {report['case_bank']['ground_truth_distribution']} |")
    add(
        f"| Thresholds | decline >= {rt['thresholds']['auto_decline_risk']}, "
        f"approve <= {rt['thresholds']['auto_approve_risk']}, "
        f"min confidence {rt['thresholds']['min_confidence']} |"
    )
    add("")

    add(f"**Headline from the self-scrutiny section:** {honesty['headline']}\n")
    add(f"**How to read this report:** {honesty['how_to_read_this']}\n")

    add("## Overall results\n")
    add("Two scoring views. `auto_only` excludes escalations; `escalation_as_flag` counts an ")
    add("escalation as a flag, because an escalated payment is still held and still costs a review.\n")
    add("| View | Scored | Precision | Recall | F1 | False-positive rate | Accuracy | TP/FP/TN/FN |")
    add("|---|---|---|---|---|---|---|---|")
    add(_matrix_row("auto_only", overall["auto_only"]))
    add(_matrix_row("escalation_as_flag", overall["escalation_as_flag"]))
    add("")

    for title, key, slice_key in (
        ("Results by category", "by_category", "category"),
        ("Results by difficulty", "by_difficulty", "difficulty"),
    ):
        add(f"## {title}\n")
        add("| Slice | Cases | Scored | Precision | Recall | F1 | FPR | Accuracy | TP/FP/TN/FN | Decisions |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for group, stats in report[key].items():
            m = stats["auto_only"]
            decisions = ", ".join(f"{k}:{v}" for k, v in stats["decisions"].items())
            add(
                f"| `{slice_key}={group}` | {stats['cases']} | {m['scored_cases']} | {num(m['precision'])} | "
                f"{num(m['recall'])} | {num(m['f1'])} | {pct(m['false_positive_rate'])} | "
                f"{pct(m['accuracy'])} | "
                f"{m['true_positive']}/{m['false_positive']}/{m['true_negative']}/{m['false_negative']} | "
                f"{decisions} |"
            )
        add("")
        add("*(auto_only view. Escalated cases are excluded from these matrices and counted in Decisions.)*\n")

    add("## Autonomy and escalation targeting\n")
    add(f"- Auto-resolved: **{esc['auto_resolved']}/{esc['total_cases']}**")
    add(f"- Escalated to human review: **{esc['escalated']}/{esc['total_cases']}** ({pct(esc['escalation_rate'])})")
    add(f"- Ambiguous cases escalated: **{esc['ambiguous_escalated']}/{esc['ambiguous_total']}** ({pct(esc['ambiguous_escalation_rate'])})")
    add(f"- Decisive cases escalated: **{esc['decisive_escalated']}/{esc['decisive_total']}** ({pct(esc['decisive_escalation_rate'])})")
    add(f"- Escalation lift on ambiguity: **{esc['lift'] if esc['lift'] is not None else 'n/a'}x**\n")
    for finding in esc["findings"]:
        add(f"> {finding}\n")

    add("## Failure accounting\n")
    add("Infrastructure failures, agent protocol failures and genuinely wrong decisions are counted ")
    add("separately. They are different findings and are never summed into one 'failed' bucket.\n")
    add("| Bucket | Count | What it means |")
    add("|---|---|---|")
    for key in ("infrastructure_failures", "agent_protocol_failures", "wrong_auto_decisions", "escalated_decisive_cases"):
        bucket = err[key]
        add(f"| `{key}` | {bucket['total']} | {bucket['meaning']} |")
    add("")
    if err["wrong_auto_decisions"]["cases"]:
        add("Wrong auto-decisions in detail:\n")
        add("| Case | Ground truth | Decision | Risk | Confidence |")
        add("|---|---|---|---|---|")
        for row in err["wrong_auto_decisions"]["cases"]:
            add(
                f"| {row['case_id']} | {row['ground_truth']} | `{row['decision']}` | "
                f"{row['combined_risk']} | {row['confidence']} |"
            )
        add("")

    add("## Honesty and self-scrutiny\n")
    add(f"{honesty['headline']}\n")

    if honesty["suspicious_results"]:
        add("### Slices at or above 95% - interrogated, not celebrated\n")
        for s in honesty["suspicious_results"]:
            add(f"**`{s['slice']}`** - accuracy {pct(s['accuracy'])} over {s['scored_cases']} scored case(s)\n")
            add(f"- Verdict: **{s['verdict']}**")
            for check in s["checks"]:
                add(f"- {check}")
            add("")
    else:
        add("No slice reached the 95% threshold, so there is no suspiciously-good result to investigate.\n")

    ab = honesty["ablation_study"]
    add("### Ablation: is the multi-agent architecture doing real work?\n")
    add(f"{ab['method']}\n")
    add(f"- Auto-decisions examined: **{ab['auto_decisions_examined']}**")
    add(f"- Survive removal of the loudest specialist: **{ab['survives_removal_of_loudest_specialist']}**")
    add(f"- Collapse without it: **{ab['collapses_without_loudest_specialist']}**")
    add(f"- Multi-signal share: **{pct(ab['multi_signal_share'])}**\n")
    if ab["single_signal_cases"]:
        add("| Case | Decision | Remove | Becomes | Risk without it |")
        add("|---|---|---|---|---|")
        for row in ab["single_signal_cases"]:
            add(
                f"| {row['case_id']} | `{row['decision']}` | {row['removed_specialist']} "
                f"({row['removed_risk_score']:.0f}) | `{row['decision_without_it']}` | {row['risk_without_it']} |"
            )
        add("")
    pd = ab["precedent_dependence"]
    add(
        f"Precedent dependence: **{pd['decisions_that_change_without_historical_case']}** auto-decision(s) "
        f"change when the historical-case specialist is removed. {pd['why_this_matters']}\n"
    )

    add("### Concerns about these results\n")
    for concern in honesty["concerns"]:
        add(f"- {concern}\n")

    add("---\n")
    add("Project by Ayush Verma - ayushv3533e@gmail.com\n")
    return "\n".join(out)
