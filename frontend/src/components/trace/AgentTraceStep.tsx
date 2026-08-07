import clsx from 'clsx'
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  FileSearch,
  Gavel,
  UserCheck,
  Wrench,
} from 'lucide-react'
import { useState } from 'react'
import type { TraceEvent } from '../../types'
import Badge, { toneForDecision, toneForRisk } from '../ui/Badge'

const ACTOR_LABEL: Record<string, string> = {
  supervisor: 'Supervisor',
  transaction_pattern: 'Transaction Pattern',
  identity_signal: 'Identity Signal',
  network_analysis: 'Network Analysis',
  historical_case: 'Historical Case',
  human_reviewer: 'Human Reviewer',
  harness: 'Eval Harness',
}

const ICONS: Record<string, typeof Wrench> = {
  case_opened: FileSearch,
  tool_call: Wrench,
  finding: Brain,
  verdict: Gavel,
  escalation: AlertTriangle,
  human_decision: UserCheck,
  error: AlertTriangle,
}

function num(value: unknown, digits = 2): string {
  const n = Number(value)
  return Number.isFinite(n) ? n.toFixed(digits) : '-'
}

/** One event from the shared log, rendered as a step in the investigation. */
export default function AgentTraceStep({ event }: { event: TraceEvent }) {
  const [open, setOpen] = useState(false)
  const Icon = ICONS[event.event_type] ?? Wrench
  const p = event.payload as Record<string, any>
  const actor = ACTOR_LABEL[event.actor] ?? event.actor
  const isError = event.event_type === 'error'

  return (
    <li className="relative pl-8">
      <span
        className={clsx(
          'absolute left-0 top-1 flex h-6 w-6 items-center justify-center rounded-full ring-1',
          isError
            ? 'bg-rose-500/15 text-rose-300 ring-rose-500/40'
            : event.event_type === 'verdict'
              ? 'bg-sky-500/15 text-sky-300 ring-sky-500/40'
              : 'bg-ink-800 text-slate-400 ring-ink-700',
        )}
      >
        <Icon size={13} />
      </span>

      <div className="rounded-lg border border-ink-700 bg-ink-900/70 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-slate-200">{actor}</span>
          <Badge tone="muted">{event.event_type}</Badge>
          {event.event_type === 'tool_call' && p.tool && (
            <code className="rounded bg-ink-850 px-1.5 py-0.5 font-mono text-xs text-sky-300">{p.tool}</code>
          )}
          {event.event_type === 'finding' && (
            <>
              <Badge tone={toneForRisk(Number(p.risk_score))}>risk {num(p.risk_score, 0)}</Badge>
              <Badge tone="muted">conf {num(p.confidence)}</Badge>
            </>
          )}
          {event.event_type === 'verdict' && (
            <>
              <Badge tone={toneForDecision(p.decision)}>{p.decision}</Badge>
              <Badge tone={toneForRisk(Number(p.combined_risk))}>risk {num(p.combined_risk, 1)}</Badge>
              <Badge tone="muted">conf {num(p.confidence)}</Badge>
              {p.policy_overrode_model && (
                <Badge tone="escalate" title={`model recommended ${p.model_recommended_action}`}>
                  policy overrode model
                </Badge>
              )}
            </>
          )}
          {event.event_type === 'human_decision' && <Badge tone={toneForDecision(p.action)}>{p.action}</Badge>}
          <span className="ml-auto font-mono text-[11px] text-slate-600">
            {new Date(event.timestamp).toLocaleTimeString()}
          </span>
        </div>

        {event.event_type === 'finding' && (
          <p className="mt-2 text-sm leading-relaxed text-slate-400">{p.reasoning_summary}</p>
        )}
        {event.event_type === 'verdict' && (
          <>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{p.rationale}</p>
            {p.disagreement_interpretation && (
              <p className="mt-2 border-l-2 border-amber-500/40 pl-3 text-sm italic leading-relaxed text-amber-200/80">
                {p.disagreement_interpretation}
              </p>
            )}
          </>
        )}
        {event.event_type === 'escalation' && (
          <p className="mt-2 text-sm leading-relaxed text-amber-200/80">{p.reason}</p>
        )}
        {isError && (
          <p className="mt-2 font-mono text-xs leading-relaxed text-rose-300">
            [{String(p.error_class)}] {String(p.error)}
          </p>
        )}
        {event.event_type === 'finding' && Array.isArray(p.key_evidence) && p.key_evidence.length > 0 && (
          <ul className="mt-2 space-y-1">
            {p.key_evidence.map((line: string, i: number) => (
              <li key={i} className="flex gap-2 text-xs text-slate-500">
                <CheckCircle2 size={12} className="mt-0.5 shrink-0 text-slate-600" />
                <span>{line}</span>
              </li>
            ))}
          </ul>
        )}

        <button
          onClick={() => setOpen((v) => !v)}
          className="mt-2 text-[11px] text-slate-600 underline-offset-2 hover:text-slate-400 hover:underline"
        >
          {open ? 'hide' : 'show'} raw event payload
        </button>
        {open && (
          <pre className="mt-2 max-h-72 overflow-auto rounded bg-ink-950 p-3 font-mono text-[11px] leading-relaxed text-slate-500">
            {JSON.stringify({ event_id: event.event_id, ...event.payload }, null, 2)}
          </pre>
        )}
      </div>
    </li>
  )
}
