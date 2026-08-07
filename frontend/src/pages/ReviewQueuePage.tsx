import { Check, HelpCircle, Inbox, Loader2, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getReviewQueue, submitReview } from '../api/client'
import Badge, { toneForRisk } from '../components/ui/Badge'
import ConfidenceMeter from '../components/ui/ConfidenceMeter'
import type { ReviewCase } from '../types'

const SPECIALISTS = ['transaction_pattern', 'identity_signal', 'network_analysis', 'historical_case']

export default function ReviewQueuePage() {
  const [queue, setQueue] = useState<ReviewCase[]>([])
  const [backend, setBackend] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState<string | null>(null)
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [detailTarget, setDetailTarget] = useState<Record<string, string>>({})
  const [message, setMessage] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const data = await getReviewQueue()
      setQueue(data.cases)
      setBackend(data.backend)
    } catch {
      setQueue([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const act = async (caseId: string, action: string) => {
    setBusy(caseId)
    setMessage(null)
    try {
      const result = await submitReview(caseId, {
        action,
        notes: notes[caseId] ?? '',
        requested_specialist: action === 'request_detail' ? detailTarget[caseId] ?? SPECIALISTS[0] : undefined,
      })
      setMessage(
        action === 'request_detail'
          ? `${caseId}: re-ran ${detailTarget[caseId] ?? SPECIALISTS[0]}; the case stays queued until a human commits to an outcome.`
          : `${caseId}: ${String(result.status)} — decision written to the event log.`,
      )
      await load()
    } catch (err) {
      setMessage(err instanceof Error ? err.message : 'review submission failed')
    } finally {
      setBusy(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Loader2 size={15} className="animate-spin" /> loading review queue…
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Review Queue</h1>
        <p className="mt-1 text-sm text-slate-500">
          Cases the supervisor would not commit to. Backend: <code className="text-slate-400">{backend}</code>.
          Every decision here is appended to the shared event log as a <code>human_decision</code> event, which
          is what closes the loop on the audit trail.
        </p>
      </div>

      {message && (
        <div className="rounded-lg border border-sky-800/50 bg-sky-950/30 px-4 py-3 text-sm text-sky-200">
          {message}
        </div>
      )}

      {queue.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-ink-700 bg-ink-900 py-16 text-slate-600">
          <Inbox size={28} />
          <p className="text-sm">Nothing awaiting review. Run the eval harness or a live case to populate it.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {queue.map((c) => (
            <article key={c.case_id} className="rounded-lg border border-ink-700 bg-ink-900 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <Link to={`/cases/${c.case_id}`} className="font-mono text-sm text-sky-300 hover:underline">
                  {c.case_id}
                </Link>
                <Badge tone="muted">{c.category_hint}</Badge>
                <Badge tone={toneForRisk(c.combined_risk)}>risk {c.combined_risk.toFixed(1)}</Badge>
                {c.disagreement?.significant && <Badge tone="escalate">disagreement</Badge>}
                {c.status === 'awaiting_detail' && <Badge tone="accent">detail requested</Badge>}
                <div className="ml-auto w-44">
                  <ConfidenceMeter value={c.confidence} compact />
                </div>
              </div>

              <p className="mt-2 text-sm text-slate-400">{c.summary}</p>

              <div className="mt-3 rounded border-l-2 border-amber-600/50 bg-amber-950/10 py-2 pl-3">
                <p className="text-xs uppercase tracking-wide text-amber-400/70">why it escalated</p>
                <p className="mt-0.5 text-sm text-amber-100/80">{c.escalation_reason}</p>
              </div>

              <div className="mt-3 grid gap-2 md:grid-cols-2">
                {c.findings.map((f) => (
                  <div key={f.specialist} className="rounded border border-ink-800 bg-ink-850/50 p-2.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-slate-300">{f.specialist}</span>
                      <Badge tone={toneForRisk(f.risk_score)}>{f.risk_score.toFixed(0)}</Badge>
                      <span className="ml-auto font-mono text-[11px] text-slate-500">
                        conf {f.confidence.toFixed(2)}
                      </span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-slate-500">{f.reasoning_summary}</p>
                  </div>
                ))}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-ink-800 pt-3">
                <input
                  value={notes[c.case_id] ?? ''}
                  onChange={(e) => setNotes((n) => ({ ...n, [c.case_id]: e.target.value }))}
                  placeholder="reviewer notes (recorded in the event log)"
                  className="min-w-56 flex-1 rounded border border-ink-700 bg-ink-850 px-3 py-1.5 text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-sky-600"
                />
                <select
                  value={detailTarget[c.case_id] ?? SPECIALISTS[0]}
                  onChange={(e) => setDetailTarget((d) => ({ ...d, [c.case_id]: e.target.value }))}
                  className="rounded border border-ink-700 bg-ink-850 px-2 py-1.5 text-xs text-slate-300 outline-none"
                >
                  {SPECIALISTS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
                <button
                  disabled={busy === c.case_id}
                  onClick={() => act(c.case_id, 'request_detail')}
                  className="flex items-center gap-1.5 rounded border border-ink-600 px-3 py-1.5 text-xs text-slate-300 hover:bg-ink-800 disabled:opacity-50"
                >
                  <HelpCircle size={13} /> more detail
                </button>
                <button
                  disabled={busy === c.case_id}
                  onClick={() => act(c.case_id, 'decline')}
                  className="flex items-center gap-1.5 rounded bg-rose-600/90 px-3 py-1.5 text-xs font-medium text-white hover:bg-rose-500 disabled:opacity-50"
                >
                  <X size={13} /> decline
                </button>
                <button
                  disabled={busy === c.case_id}
                  onClick={() => act(c.case_id, 'approve')}
                  className="flex items-center gap-1.5 rounded bg-emerald-600/90 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  <Check size={13} /> approve
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
