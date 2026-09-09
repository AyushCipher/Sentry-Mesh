# SentryMesh eval harness report

**Run** `2f852a` - 2026-09-09T17:40:17+00:00 - 275ms wall clock

> All cases are SYNTHETIC. No real financial data is represented anywhere in this system or this report.

## What this run actually was

| | |
|---|---|
| Reasoning backend | `offline_deterministic` |
| Event log | `redis` |
| Review queue | `redis` |
| Precedent retrieval | `local_cosine` |
| Cases run | 23 |
| Ground truth | {'ambiguous': 2, 'fraud': 14, 'legitimate': 7} |
| Thresholds | decline >= 70.0, approve <= 35.0, min confidence 0.7 |

**Headline from the self-scrutiny section:** 7 slice(s) came back at or above 95% accuracy (category=account_takeover, category=card_testing, category=legitimate, category=money_muling, difficulty=easy, difficulty=hard, difficulty=medium). Each is investigated below rather than reported as a win.

**How to read this report:** Read the escalation rate first, then the escalation_as_flag false-positive rate, then the ablation study. The auto_only precision figure is the least informative number in this report and is printed first only because that is the convention.

## Overall results

Two scoring views. `auto_only` excludes escalations; `escalation_as_flag` counts an 
escalation as a flag, because an escalated payment is still held and still costs a review.

| View | Scored | Precision | Recall | F1 | False-positive rate | Accuracy | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|
| auto_only | 9 | 1.000 | 1.000 | 1.000 | 0.0% | 100.0% | 5/0/4/0 |
| escalation_as_flag | 21 | 0.824 | 1.000 | 0.903 | 42.9% | 85.7% | 14/3/4/0 |

## Results by category

| Slice | Cases | Scored | Precision | Recall | F1 | FPR | Accuracy | TP/FP/TN/FN | Decisions |
|---|---|---|---|---|---|---|---|---|---|
| `category=account_takeover` | 5 | 2 | 1.000 | 1.000 | 1.000 | n/a | 100.0% | 2/0/0/0 | auto_decline:2, escalate:3 |
| `category=card_testing` | 4 | 2 | 1.000 | 1.000 | 1.000 | n/a | 100.0% | 2/0/0/0 | auto_decline:2, escalate:2 |
| `category=legitimate` | 7 | 4 | n/a | n/a | n/a | 0.0% | 100.0% | 0/0/4/0 | auto_approve:4, escalate:3 |
| `category=money_muling` | 4 | 1 | 1.000 | 1.000 | 1.000 | n/a | 100.0% | 1/0/0/0 | auto_decline:1, escalate:3 |
| `category=synthetic_identity` | 3 | 0 | n/a | n/a | n/a | n/a | n/a | 0/0/0/0 | escalate:3 |

*(auto_only view. Escalated cases are excluded from these matrices and counted in Decisions.)*

## Results by difficulty

| Slice | Cases | Scored | Precision | Recall | F1 | FPR | Accuracy | TP/FP/TN/FN | Decisions |
|---|---|---|---|---|---|---|---|---|---|
| `difficulty=easy` | 4 | 4 | 1.000 | 1.000 | 1.000 | 0.0% | 100.0% | 3/0/1/0 | auto_approve:1, auto_decline:3 |
| `difficulty=hard` | 8 | 1 | n/a | n/a | n/a | 0.0% | 100.0% | 0/0/1/0 | auto_approve:1, escalate:7 |
| `difficulty=medium` | 11 | 4 | 1.000 | 1.000 | 1.000 | 0.0% | 100.0% | 2/0/2/0 | auto_approve:2, auto_decline:2, escalate:7 |

*(auto_only view. Escalated cases are excluded from these matrices and counted in Decisions.)*

## Autonomy and escalation targeting

- Auto-resolved: **9/23**
- Escalated to human review: **14/23** (60.9%)
- Ambiguous cases escalated: **2/2** (100.0%)
- Decisive cases escalated: **12/21** (57.1%)
- Escalation lift on ambiguity: **1.75x**

> All 2/2 deliberately-ambiguous cases were escalated.

> Ambiguous cases escalate 1.75x as often as decisive ones (100% vs 57%).

> Overall escalation rate is 61%. In production that is roughly 14 human reviews per 23 flagged payments, which is a staffing cost, not a free safety margin. Auto-decision quality below should be read against it.

## Failure accounting

Infrastructure failures, agent protocol failures and genuinely wrong decisions are counted 
separately. They are different findings and are never summed into one 'failed' bucket.

| Bucket | Count | What it means |
|---|---|---|
| `infrastructure_failures` | 0 | Transport or provider failures. These say nothing about agent reasoning quality and any metric computed over a run containing them is measured on degraded evidence. |
| `agent_protocol_failures` | 0 | The agent ran but did not produce a usable structured result (bad tool call, turn limit, tool execution error). A reasoning-adjacent failure, distinct from both infrastructure and from being wrong. |
| `wrong_auto_decisions` | 0 | The system ran correctly and committed to the wrong answer. This is the only bucket that measures reasoning quality, and it is the one that matters. |
| `escalated_decisive_cases` | 12 | Cases with a clear ground truth that the system would not commit to. Not wrong, but not autonomous either - each one is a human review the system failed to save. |

## Honesty and self-scrutiny

7 slice(s) came back at or above 95% accuracy (category=account_takeover, category=card_testing, category=legitimate, category=money_muling, difficulty=easy, difficulty=hard, difficulty=medium). Each is investigated below rather than reported as a win.

### Slices at or above 95% - interrogated, not celebrated

**`category=account_takeover`** - accuracy 100.0% over 2 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; survivorship: the hard cases were escalated out of the metric; cases were seeded to be easy; decisions rest on a single specialist**
- Only 2 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 2 of 4 decisive cases in this slice were escalated and are therefore excluded from the auto_only accuracy. The perfect score describes only the 2 case(s) the system was willing to commit to.
- Cases in this slice carry 4.0 planted signals on average. They were built to be detectable, and a system that missed them would be broken rather than merely imperfect. Getting them right is a floor, not an achievement.
- 2 of 2 auto-decisions here collapse when the loudest specialist is removed (SM-005, SM-006), so they rest on one signal rather than on corroboration across the panel.

**`category=card_testing`** - accuracy 100.0% over 2 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; survivorship: the hard cases were escalated out of the metric; cases were seeded to be easy; decisions rest on a single specialist**
- Only 2 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 2 of 4 decisive cases in this slice were escalated and are therefore excluded from the auto_only accuracy. The perfect score describes only the 2 case(s) the system was willing to commit to.
- Cases in this slice carry 5.0 planted signals on average. They were built to be detectable, and a system that missed them would be broken rather than merely imperfect. Getting them right is a floor, not an achievement.
- 2 of 2 auto-decisions here collapse when the loudest specialist is removed (SM-001, SM-002), so they rest on one signal rather than on corroboration across the panel.

**`category=legitimate`** - accuracy 100.0% over 4 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; survivorship: the hard cases were escalated out of the metric; decisions rest on a single specialist**
- Only 4 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 3 of 7 decisive cases in this slice were escalated and are therefore excluded from the auto_only accuracy. The perfect score describes only the 4 case(s) the system was willing to commit to.
- 1 of 4 auto-decisions here collapse when the loudest specialist is removed (SM-016), so they rest on one signal rather than on corroboration across the panel.

**`category=money_muling`** - accuracy 100.0% over 1 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; survivorship: the hard cases were escalated out of the metric; cases were seeded to be easy; decisions rest on a single specialist**
- Only 1 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 2 of 3 decisive cases in this slice were escalated and are therefore excluded from the auto_only accuracy. The perfect score describes only the 1 case(s) the system was willing to commit to.
- Cases in this slice carry 4.0 planted signals on average. They were built to be detectable, and a system that missed them would be broken rather than merely imperfect. Getting them right is a floor, not an achievement.
- 1 of 1 auto-decisions here collapse when the loudest specialist is removed (SM-012), so they rest on one signal rather than on corroboration across the panel.

**`difficulty=easy`** - accuracy 100.0% over 4 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; decisions rest on a single specialist**
- Only 4 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 3 of 4 auto-decisions here collapse when the loudest specialist is removed (SM-001, SM-002, SM-005), so they rest on one signal rather than on corroboration across the panel.

**`difficulty=hard`** - accuracy 100.0% over 1 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; survivorship: the hard cases were escalated out of the metric; cases were seeded to be easy**
- Only 1 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 5 of 6 decisive cases in this slice were escalated and are therefore excluded from the auto_only accuracy. The perfect score describes only the 1 case(s) the system was willing to commit to.
- Cases in this slice carry 5.0 planted signals on average. They were built to be detectable, and a system that missed them would be broken rather than merely imperfect. Getting them right is a floor, not an achievement.
- All 1 auto-decisions survive removal of the loudest specialist, and 4.0 specialists contributed a confident finding on average. The multi-signal reasoning is real for this slice.

**`difficulty=medium`** - accuracy 100.0% over 4 scored case(s)

- Verdict: **SUSPECT - sample too small to mean anything; survivorship: the hard cases were escalated out of the metric; cases were seeded to be easy; decisions rest on a single specialist**
- Only 4 case(s) were scored in this slice. At that size a single case moves the number by 20 points or more; this is not a measurement, it is an anecdote.
- 7 of 11 decisive cases in this slice were escalated and are therefore excluded from the auto_only accuracy. The perfect score describes only the 4 case(s) the system was willing to commit to.
- Cases in this slice carry 4.0 planted signals on average. They were built to be detectable, and a system that missed them would be broken rather than merely imperfect. Getting them right is a floor, not an achievement.
- 3 of 4 auto-decisions here collapse when the loudest specialist is removed (SM-006, SM-012, SM-016), so they rest on one signal rather than on corroboration across the panel.

### Ablation: is the multi-agent architecture doing real work?

Re-runs the supervisor's aggregation and decision functions over a reduced panel using the findings already recorded in the event log. No additional model calls, so this measures the decision logic exactly as it ran.

- Auto-decisions examined: **9**
- Survive removal of the loudest specialist: **3**
- Collapse without it: **6**
- Multi-signal share: **33.3%**

| Case | Decision | Remove | Becomes | Risk without it |
|---|---|---|---|---|
| SM-001 | `auto_decline` | identity_signal (100) | `escalate` | 69.81 |
| SM-002 | `auto_decline` | identity_signal (76) | `escalate` | 71.04 |
| SM-005 | `auto_decline` | identity_signal (80) | `escalate` | 47.2 |
| SM-006 | `auto_decline` | identity_signal (70) | `escalate` | 64.01 |
| SM-012 | `auto_decline` | network_analysis (90) | `escalate` | 32.29 |
| SM-016 | `auto_approve` | transaction_pattern (26) | `escalate` | 0.0 |

Precedent dependence: **3** auto-decision(s) change when the historical-case specialist is removed. The precedent corpus and the case bank were authored by the same person, so the historical-case specialist has a contamination advantage that would not survive contact with a real case archive. Decisions that depend on it should be discounted accordingly.

### Concerns about these results

- RUN MODE: this run used the offline deterministic reasoner, not an LLM, because no model provider was configured. Every accuracy number in this report therefore measures the explicit scoring functions inside each specialist, NOT agent reasoning. The tool calls, the event log, the supervisor's aggregation and the decision thresholds are all exercised exactly as they would be in an LLM run - but nothing here is evidence about model quality, and it must not be presented as such. Re-run with a provider configured to get numbers that mean what a reader will assume they mean.

- SAMPLE SIZE: 23 cases, 21 with a decisive ground truth. Every per-category figure below rests on 2-7 cases. Confidence intervals at that size are wide enough to swallow most of the differences between categories, and none of these numbers would survive being quoted as a production benchmark.

- SCORING VIEW: the false-positive rate is 0.0% counting only auto-declines, and 42.9% once escalations are counted as flags. The second number is the one a customer feels: an escalated payment is still held. Quoting only the first would be the single most misleading thing this report could do.

- AUTONOMY: 61% of cases were escalated to a human. High precision on the remaining auto-decisions is partly a selection effect - the system chose which cases to be judged on. A fraud team would not deploy this without pushing the escalation rate down, and doing so is exactly what would expose the error rate this report cannot currently see.

- MULTI-SIGNAL REASONING: only 33% of auto-decisions survive removal of their loudest specialist. The rest collapse to a different verdict, which means the four-specialist architecture is not earning its cost on those cases - a single specialist plus the threshold logic would have produced the same answers.

- RETRIEVAL CONTAMINATION: 3 auto-decision(s) change when the historical-case specialist is removed. The precedent corpus was written alongside the case bank by the same author, and retrieval is lexical TF-IDF, so precedent summaries share vocabulary with the cases they 'predict'. Retrieval hits at this rate would not transfer to a real case archive, and any per-category strength that traces back to this specialist should be discounted.

- FAILURE ACCOUNTING: no infrastructure or agent-protocol failures in this run. All 0 incorrect outcome(s) are genuine wrong decisions rather than plumbing, so the metrics above are measured on complete evidence.

- NOT MEASURED: the case bank was written by the same person who wrote the detectors, so it encodes the same assumptions about what fraud looks like. Nothing here tests the patterns neither the author nor the agents thought of, and that blind spot is where a real fraud system's losses actually come from. Adversarial cases authored by someone else would be the first thing to add.

---

Project by Ayush Verma - ayushv3533e@gmail.com
