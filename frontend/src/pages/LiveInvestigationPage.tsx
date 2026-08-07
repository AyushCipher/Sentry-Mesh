import clsx from 'clsx'
import { Loader2, Play, Radio } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { getCaseBank, submitCase } from '../api/client'
import AgentTraceStep from '../components/trace/AgentTraceStep'
import Badge, { toneForDecision } from '../components/ui/Badge'
import ConfidenceMeter from '../components/ui/ConfidenceMeter'
import StatCard from '../components/ui/StatCard'
import { useTraceStream } from '../hooks/useWebSocket'
import type { SeededCase } from '../types'

const SPECIALISTS = ['transaction_pattern', 'identity_signal', 'network_analysis', 'historical_case']

export default function LiveInvestigationPage() {
  const [bank, setBank] = useState<SeededCase[]>([])
  const [selected, setSelected] = useState<string>('')
  const [runCaseId, setRunCaseId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const { events, state, complete, error } = useTraceStream(runCaseId)

  useEffect(() => {
    getCaseBank()
      .then((data) => {
        setBank(data.cases)
        setSelected(data.cases[0]?.case_id ?? '')
      })
      .catch(() => setBank([]))
  }, [])

  const run = async () => {
    if (!selected) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const response = await submitCase({
        seeded_case_id: selected,
        case_id: `${selected}-live-${Date.now().toString(36)}`,
      })
      setRunCaseId(response.case_id)
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'submission failed')
    } finally {
      setSubmitting(false)
    }
  }

  const verdict = useMemo(
    () => events.filter((e) => e.event_type === 'verdict').slice(-1)[0]?.payload as Record<string, any> | undefined,
    [events],
  )
  const findings = useMemo(
    () => events.filter((e) => e.event_type === 'finding'),
    [events],
  )
  const reported = new Set(findings.map((f) => f.actor))
  const selectedCase = bank.find((c) => c.case_id === selected)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Live Investigation</h1>
        <p className="mt-1 text-sm text-slate-500">
          Submit a case and watch the supervisor fan it out to four specialists. Every step below is read
          from the shared event log as it is written — nothing here is replayed from a script.
        </p>
      </div>

      <div className="rounded-lg border border-ink-700 bg-ink-900 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="min-w-64 flex-1">
            <span className="mb-1 block text-xs uppercase tracking-wide text-slate-500">
              Seeded synthetic case
            </span>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="w-full rounded-md border border-ink-700 bg-ink-850 px-3 py-2 text-sm text-slate-200 outline-none focus:border-sky-600"
            >
              {bank.map((c) => (
                <option key={c.case_id} value={c.case_id}>
                  {c.case_id} — {c.category} / {c.difficulty} — {c.summary.slice(0, 70)}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={run}
            disabled={submitting || !selected}
            className="flex items-center gap-2 rounded-md bg-sky-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitting ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
            Investigate
          </button>
        </div>

        {selectedCase && (
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-ink-800 pt-3">
            <Badge tone="muted">ground truth: {selectedCase.ground_truth}</Badge>
            <Badge tone="muted">expected: {selectedCase.expected_verdict}</Badge>
            <Badge tone="muted">difficulty: {selectedCase.difficulty}</Badge>
            {selectedCase.planted_signals.map((s) => (
              <Badge key={s} tone="accent">
                {s}
              </Badge>
            ))}
            <span className="w-full pt-1 text-[11px] text-slate-600">
              Ground truth is shown here for the demo only. It is stripped from every agent prompt by
              <code className="mx-1 text-slate-500">Case.agent_view()</code>.
            </span>
          </div>
        )}
        {submitError && <p className="mt-2 text-sm text-rose-300">{submitError}</p>}
      </div>

      {runCaseId && (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard
              label="stream"
              value={
                <span className="flex items-center gap-2 text-base">
                  <Radio size={14} className={clsx(state === 'open' && !complete && 'animate-pulse text-sky-400')} />
                  {complete ? 'complete' : state}
                </span>
              }
              hint={`${events.length} events from the log`}
            />
            <StatCard
              label="specialists reported"
              value={`${SPECIALISTS.filter((s) => reported.has(s)).length}/4`}
              hint={SPECIALISTS.filter((s) => !reported.has(s)).join(', ') || 'all four in'}
            />
            <StatCard
              label="combined risk"
              value={verdict ? Number(verdict.combined_risk).toFixed(1) : '—'}
              tone={verdict ? (Number(verdict.combined_risk) >= 70 ? 'bad' : Number(verdict.combined_risk) <= 35 ? 'good' : 'warn') : 'neutral'}
              hint={verdict ? `deterministic cross-check: ${verdict.deterministic_fallback_risk}` : 'awaiting verdict'}
            />
            <StatCard
              label="decision"
              value={
                verdict ? (
                  <Badge tone={toneForDecision(String(verdict.decision))} className="text-sm">
                    {String(verdict.decision)}
                  </Badge>
                ) : (
                  '—'
                )
              }
              hint={verdict ? <ConfidenceMeter value={Number(verdict.confidence)} compact /> : 'awaiting verdict'}
            />
          </div>

          {verdict?.disagreement && Array.isArray((verdict.disagreement as any).reasons) &&
            (verdict.disagreement as any).reasons.length > 0 && (
              <div className="rounded-lg border border-amber-700/40 bg-amber-950/20 p-4">
                <h3 className="text-sm font-medium text-amber-200">Disagreement detected</h3>
                <ul className="mt-2 space-y-1">
                  {(verdict.disagreement as any).reasons.map((r: string, i: number) => (
                    <li key={i} className="text-sm text-amber-200/70">
                      • {r}
                    </li>
                  ))}
                </ul>
              </div>
            )}

          <div>
            <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-slate-500">
              Event log trace — {runCaseId}
            </h2>
            {error && <p className="mb-3 text-sm text-rose-300">{error}</p>}
            <ol className="space-y-3 border-l border-ink-800 pl-3">
              {events.map((e) => (
                <AgentTraceStep key={e.event_id} event={e} />
              ))}
            </ol>
            {events.length === 0 && state !== 'error' && (
              <p className="py-8 text-center text-sm text-slate-600">waiting for the first event…</p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
