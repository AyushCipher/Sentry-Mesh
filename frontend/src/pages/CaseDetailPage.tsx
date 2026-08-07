import { Loader2, ScrollText, Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getCase, getTrace, listCases } from '../api/client'
import AgentTraceStep from '../components/trace/AgentTraceStep'
import Badge, { toneForDecision, toneForRisk } from '../components/ui/Badge'
import ConfidenceMeter from '../components/ui/ConfidenceMeter'
import StatCard from '../components/ui/StatCard'
import type { CaseStatus, TraceEvent } from '../types'

export default function CaseDetailPage() {
  const { caseId } = useParams()
  const navigate = useNavigate()
  const [known, setKnown] = useState<{ case_id: string; status: string; summary: string }[]>([])
  const [status, setStatus] = useState<CaseStatus | null>(null)
  const [events, setEvents] = useState<TraceEvent[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    listCases()
      .then((data) => setKnown(data.cases))
      .catch(() => setKnown([]))
  }, [])

  useEffect(() => {
    if (!caseId) {
      setStatus(null)
      setEvents([])
      return
    }
    setLoading(true)
    setError(null)
    Promise.all([getCase(caseId), getTrace(caseId)])
      .then(([s, t]) => {
        setStatus(s)
        setEvents(t.events)
      })
      .catch(() => setError(`No events recorded for ${caseId}.`))
      .finally(() => setLoading(false))
  }, [caseId])

  const verdict = (status?.verdict ?? {}) as Record<string, any>
  const disagreement = (verdict.disagreement ?? {}) as Record<string, any>
  const filtered = known.filter(
    (c) => !filter || c.case_id.toLowerCase().includes(filter.toLowerCase()),
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Case Detail</h1>
        <p className="mt-1 text-sm text-slate-500">
          The full audit trail, reconstructed from the event log alone — first specialist tool call through
          final human decision. Nothing on this page comes from a side-table the agents also wrote to.
        </p>
      </div>

      <div className="rounded-lg border border-ink-700 bg-ink-900 p-4">
        <div className="flex items-center gap-2">
          <Search size={15} className="text-slate-600" />
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter recorded cases…"
            className="flex-1 bg-transparent text-sm text-slate-200 outline-none placeholder:text-slate-600"
          />
          <span className="text-xs text-slate-600">{known.length} cases in the log</span>
        </div>
        <div className="mt-3 flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
          {filtered.map((c) => (
            <button
              key={c.case_id}
              onClick={() => navigate(`/cases/${encodeURIComponent(c.case_id)}`)}
              className={`rounded border px-2 py-1 font-mono text-[11px] transition-colors ${
                c.case_id === caseId
                  ? 'border-sky-600 bg-sky-950/40 text-sky-300'
                  : 'border-ink-700 text-slate-400 hover:border-ink-600 hover:text-slate-200'
              }`}
            >
              {c.case_id}
            </button>
          ))}
          {filtered.length === 0 && (
            <span className="text-xs text-slate-600">no cases match — run an investigation first</span>
          )}
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 size={15} className="animate-spin" /> loading trace…
        </div>
      )}
      {error && <p className="text-sm text-rose-300">{error}</p>}

      {!caseId && !loading && (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-ink-700 bg-ink-900 py-16 text-slate-600">
          <ScrollText size={28} />
          <p className="text-sm">Pick a case above to reconstruct its audit trail.</p>
        </div>
      )}

      {status && (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="status" value={<span className="text-base">{status.status}</span>} hint={status.summary} />
            <StatCard
              label="combined risk"
              value={verdict.combined_risk !== undefined ? Number(verdict.combined_risk).toFixed(1) : '—'}
              tone={
                verdict.combined_risk === undefined
                  ? 'neutral'
                  : Number(verdict.combined_risk) >= 70
                    ? 'bad'
                    : Number(verdict.combined_risk) <= 35
                      ? 'good'
                      : 'warn'
              }
            />
            <StatCard
              label="decision"
              value={
                verdict.decision ? (
                  <Badge tone={toneForDecision(String(verdict.decision))} className="text-sm">
                    {String(verdict.decision)}
                  </Badge>
                ) : (
                  '—'
                )
              }
              hint={verdict.confidence !== undefined ? <ConfidenceMeter value={Number(verdict.confidence)} compact /> : undefined}
            />
            <StatCard label="events in trail" value={status.event_count} hint={`run mode: ${status.run_mode ?? 'unknown'}`} />
          </div>

          {status.findings.length > 0 && (
            <section className="rounded-lg border border-ink-700 bg-ink-900 p-4">
              <h2 className="mb-3 text-sm font-medium text-slate-200">Specialist panel</h2>
              <div className="grid gap-2 md:grid-cols-2">
                {status.findings.map((f, i) => (
                  <div key={`${f.specialist}-${i}`} className="rounded border border-ink-800 bg-ink-850/50 p-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-slate-300">{f.specialist}</span>
                      <Badge tone={toneForRisk(f.risk_score)}>{f.risk_score.toFixed(0)}</Badge>
                      {disagreement.abstaining_specialists?.includes(f.specialist) && (
                        <Badge tone="muted" title="reported that its category was uninformative">
                          abstained
                        </Badge>
                      )}
                      {disagreement.dissenting_specialists?.includes(f.specialist) && (
                        <Badge tone="escalate">dissent</Badge>
                      )}
                      <span className="ml-auto font-mono text-[11px] text-slate-500">
                        conf {f.confidence.toFixed(2)}
                      </span>
                    </div>
                    <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{f.reasoning_summary}</p>
                    {f.key_evidence?.length > 0 && (
                      <ul className="mt-1.5 space-y-0.5">
                        {f.key_evidence.map((e, j) => (
                          <li key={j} className="text-[11px] text-slate-600">
                            • {e}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
              {Array.isArray(disagreement.reasons) && disagreement.reasons.length > 0 && (
                <div className="mt-3 rounded border-l-2 border-amber-600/50 bg-amber-950/10 py-2 pl-3">
                  <p className="text-xs uppercase tracking-wide text-amber-400/70">disagreement</p>
                  {disagreement.reasons.map((r: string, i: number) => (
                    <p key={i} className="mt-0.5 text-xs text-amber-100/70">
                      • {r}
                    </p>
                  ))}
                </div>
              )}
            </section>
          )}

          {status.human_decision && (
            <section className="rounded-lg border border-emerald-800/40 bg-emerald-950/15 p-4">
              <h2 className="text-sm font-medium text-emerald-200">Human decision</h2>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <Badge tone={toneForDecision(String(status.human_decision.action))}>
                  {String(status.human_decision.action)}
                </Badge>
                <span className="text-xs text-slate-400">by {String(status.human_decision.reviewer)}</span>
                {status.human_decision.agreed_with_supervisor !== null && (
                  <Badge tone="muted">
                    {status.human_decision.agreed_with_supervisor ? 'agreed with' : 'overrode'} supervisor
                  </Badge>
                )}
              </div>
              {Boolean(status.human_decision.notes) && (
                <p className="mt-2 text-sm text-slate-400">{String(status.human_decision.notes)}</p>
              )}
            </section>
          )}

          <div>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-slate-500">
              Full audit trail — {status.event_count} events
            </h2>
            <ol className="space-y-3 border-l border-ink-800 pl-3">
              {events.map((e) => (
                <AgentTraceStep key={e.event_id} event={e} />
              ))}
            </ol>
          </div>
        </>
      )}
    </div>
  )
}
