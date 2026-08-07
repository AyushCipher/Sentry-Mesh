export type Decision = 'auto_approve' | 'auto_decline' | 'escalate'
export type EventType =
  | 'case_opened'
  | 'tool_call'
  | 'finding'
  | 'verdict'
  | 'escalation'
  | 'human_decision'
  | 'error'

export interface TraceEvent {
  event_id: string
  case_id: string
  actor: string
  event_type: EventType
  payload: Record<string, unknown>
  timestamp: string
}

export interface Runtime {
  event_log_backend: string
  review_queue_backend: string
  retrieval_backend: string
  llm_backend: string
  run_mode: string
  model: string | null
  thresholds: {
    auto_decline_risk: number
    auto_approve_risk: number
    min_confidence: number
  }
  data: string
}

export interface FindingPayload {
  specialist: string
  risk_score: number
  confidence: number
  reasoning_summary: string
  key_evidence: string[]
  tool_calls?: number
  duration_ms?: number
  error?: string | null
  error_class?: string | null
  status?: string
}

export interface Disagreement {
  pairwise_spread: number
  risk_stdev: number
  lowest_specialist_confidence: number
  split_verdict: boolean
  dissenting_specialists: string[]
  abstaining_specialists: string[]
  failed_specialists: string[]
  contributing_specialists: string[]
  reasons: string[]
  significant: boolean
}

export interface CaseStatus {
  case_id: string
  status: string
  opened_at: string
  summary: string
  category_hint: string
  run_mode: string | null
  verdict: Record<string, unknown> | null
  escalation: Record<string, unknown> | null
  human_decision: Record<string, unknown> | null
  findings: FindingPayload[]
  errors: Record<string, unknown>[]
  event_count: number
  in_flight?: boolean
}

export interface ReviewCase {
  case_id: string
  summary: string
  category_hint: string
  decision: Decision
  combined_risk: number
  confidence: number
  escalation_reason: string | null
  rationale: string
  disagreement: Disagreement
  findings: FindingPayload[]
  flagged_transaction: Record<string, unknown>
  status: string
  detail_requests: unknown[]
  escalated_at?: string
}

export interface SeededCase {
  case_id: string
  category: string
  ground_truth: string
  expected_verdict: string
  difficulty: string
  summary: string
  planted_signals: string[]
}

export interface Matrix {
  true_positive: number
  false_positive: number
  true_negative: number
  false_negative: number
  abstained: number
  precision: number | null
  recall: number | null
  f1: number | null
  false_positive_rate: number | null
  accuracy: number | null
  scored_cases: number
}

export interface SliceStats {
  cases: number
  auto_only: Matrix
  escalation_as_flag: Matrix
  decisions: Record<string, number>
  exact_verdict_matches: number
}

export interface Suspicion {
  slice: string
  accuracy: number
  scored_cases: number
  escalated_out_of_metric: number
  avg_planted_signals: number
  avg_contributing_specialists: number
  checks: string[]
  verdict: string
}

export interface HarnessReport {
  run_id: string
  generated_at: string
  wall_clock_ms: number
  data_disclaimer: string
  runtime: Runtime
  case_bank: {
    cases_run: number
    ground_truth_distribution: Record<string, number>
    category_distribution: Record<string, number>
    difficulty_distribution: Record<string, number>
  }
  overall: { auto_only: Matrix; escalation_as_flag: Matrix }
  by_category: Record<string, SliceStats>
  by_difficulty: Record<string, SliceStats>
  escalation_analysis: {
    total_cases: number
    escalated: number
    auto_resolved: number
    ambiguous_total: number
    ambiguous_escalated: number
    escalation_rate: number | null
    ambiguous_escalation_rate: number | null
    decisive_escalation_rate: number | null
    lift: number | null
    findings: string[]
  }
  error_classification: Record<
    string,
    { total: number; by_class?: Record<string, number>; meaning: string; cases?: unknown }
  >
  honesty_and_self_scrutiny: {
    headline: string
    how_to_read_this: string
    suspicious_results: Suspicion[]
    ablation_study: {
      method: string
      auto_decisions_examined: number
      survives_removal_of_loudest_specialist: number
      collapses_without_loudest_specialist: number
      multi_signal_share: number | null
      single_signal_cases: {
        case_id: string
        decision: string
        removed_specialist: string
        removed_risk_score: number
        decision_without_it: string
        risk_without_it: number
      }[]
      precedent_dependence: {
        decisions_that_change_without_historical_case: number
        why_this_matters: string
      }
    }
    concerns: string[]
  }
  case_results: CaseResult[]
}

export interface CaseResult {
  case_id: string
  run_case_id: string
  category: string
  ground_truth: string
  expected_verdict: string
  difficulty: string
  summary: string
  planted_signals: string[]
  decision: Decision | null
  combined_risk?: number
  confidence?: number
  rationale?: string
  escalation_reason?: string | null
  disagreement?: Disagreement
  verdict_matches_expected: boolean
  prediction_correct: boolean
  contributing_specialists?: number
  findings: FindingPayload[]
  run_errors: { actor: string; error_class: string; error: string }[]
  duration_ms: number
}
