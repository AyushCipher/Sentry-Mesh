import { AlertTriangle, FlaskConical, Loader2, ShieldQuestion } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getEvalReport } from '../api/client'
import Badge, { toneForDecision } from '../components/ui/Badge'
import StatCard from '../components/ui/StatCard'
import type { HarnessReport, Matrix } from '../types'

const fmt = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined ? 'n/a' : v.toFixed(digits)
const pct = (v: number | null | undefined) => (v === null || v === undefined ? 'n/a' : `${(v * 100).toFixed(1)}%`)

function MatrixRow({ name, m }: { name: string; m: Matrix }) {
  return (
    <tr className="border-t border-ink-800">
      <td className="py-2 pr-3 font-mono text-xs text-slate-300">{name}</td>
      <td className="px-2 text-right font-mono text-xs text-slate-400">{m.scored_cases}</td>
      <td className="px-2 text-right font-mono text-xs text-slate-200">{fmt(m.precision)}</td>
      <td className="px-2 text-right font-mono text-xs text-slate-200">{fmt(m.recall)}</td>
      <td className="px-2 text-right font-mono text-xs text-slate-200">{fmt(m.f1)}</td>
      <td className="px-2 text-right font-mono text-xs text-amber-300">{pct(m.false_positive_rate)}</td>
      <td className="px-2 text-right font-mono text-xs text-slate-400">{pct(m.accuracy)}</td>
      <td className="pl-2 text-right font-mono text-[11px] text-slate-500">
        {m.true_positive}/{m.false_positive}/{m.true_negative}/{m.false_negative}
      </td>
    </tr>
  )
}

function ConfusionMatrix({ m }: { m: Matrix }) {
  const cells = [
    { label: 'True positive', value: m.true_positive, hint: 'fraud, correctly declined', tone: 'text-emerald-300' },
    { label: 'False positive', value: m.false_positive, hint: 'legitimate, wrongly flagged', tone: 'text-rose-300' },
    { label: 'False negative', value: m.false_negative, hint: 'fraud, wrongly let through', tone: 'text-rose-300' },
    { label: 'True negative', value: m.true_negative, hint: 'legitimate, correctly approved', tone: 'text-emerald-300' },
  ]
  return (
    <div className="grid grid-cols-2 gap-2">
      {cells.map((c) => (
        <div key={c.label} className="rounded border border-ink-800 bg-ink-850/50 p-3">
          <div className={`font-mono text-xl font-semibold ${c.tone}`}>{c.value}</div>
          <div className="text-xs text-slate-400">{c.label}</div>
          <div className="text-[11px] text-slate-600">{c.hint}</div>
        </div>
      ))}
    </div>
  )
}

export default function EvalLeaderboardPage() {
  const [report, setReport] = useState<HarnessReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getEvalReport()
      .then(setReport)
      .catch(() => setError('No harness report yet. Run `python eval/run_harness.py` and reload.'))
  }, [])

  if (error) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-lg border border-ink-700 bg-ink-900 py-16 text-slate-500">
        <FlaskConical size={28} />
        <p className="text-sm">{error}</p>
      </div>
    )
  }
  if (!report) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Loader2 size={15} className="animate-spin" /> loading harness report…
      </div>
    )
  }

  const auto = report.overall.auto_only
  const flagged = report.overall.escalation_as_flag
  const honesty = report.honesty_and_self_scrutiny
  const ab = honesty.ablation_study
  const esc = report.escalation_analysis

  const categoryChart = Object.entries(report.by_category).map(([name, stats]) => ({
    name,
    precision: stats.auto_only.precision ?? 0,
    recall: stats.auto_only.recall ?? 0,
    fpr: stats.auto_only.false_positive_rate ?? 0,
  }))
  const decisionChart = Object.entries(
    report.case_results.reduce<Record<string, number>>((acc, r) => {
      const key = r.decision ?? 'no_verdict'
      acc[key] = (acc[key] ?? 0) + 1
      return acc
    }, {}),
  ).map(([name, value]) => ({ name, value }))
  const DECISION_FILL: Record<string, string> = {
    auto_approve: '#34d399',
    auto_decline: '#f87171',
    escalate: '#fbbf24',
    no_verdict: '#64748b',
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Eval Leaderboard</h1>
        <p className="mt-1 text-sm text-slate-500">
          Run <code className="text-slate-400">{report.run_id}</code> · {report.generated_at} ·{' '}
          {report.case_bank.cases_run} synthetic cases · {report.wall_clock_ms}ms
        </p>
      </div>

      {report.runtime.run_mode !== 'anthropic' && (
        <div className="flex gap-3 rounded-lg border border-amber-700/40 bg-amber-950/20 p-4">
          <AlertTriangle size={18} className="mt-0.5 shrink-0 text-amber-400" />
          <p className="text-sm leading-relaxed text-amber-100/80">
            This run used the <strong>offline deterministic reasoner</strong>, not an LLM. Every number on this
            page measures the explicit scoring functions inside each specialist — not agent reasoning. The
            orchestration, event log and decision logic are exercised exactly as they would be with a model, but
            none of this is evidence about model quality.
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="precision (auto only)" value={fmt(auto.precision)} hint={`over ${auto.scored_cases} committed decisions`} tone="neutral" />
        <StatCard label="recall (auto only)" value={fmt(auto.recall)} hint="fraud caught among committed decisions" tone="neutral" />
        <StatCard
          label="false-positive rate"
          value={pct(auto.false_positive_rate)}
          hint={`${pct(flagged.false_positive_rate)} once escalations count as flags`}
          tone={(auto.false_positive_rate ?? 0) > 0.1 ? 'bad' : 'good'}
        />
        <StatCard
          label="escalation rate"
          value={pct(esc.escalation_rate)}
          hint={`${esc.escalated} of ${esc.total_cases} cases went to a human`}
          tone={(esc.escalation_rate ?? 0) > 0.4 ? 'warn' : 'good'}
        />
      </div>

      <section className="rounded-lg border border-ink-700 bg-ink-900 p-4">
        <h2 className="text-sm font-medium text-slate-200">Two scoring views</h2>
        <p className="mt-1 text-xs leading-relaxed text-slate-500">
          <code>auto_only</code> excludes escalations and flatters the system — it only grades the cases it chose
          to commit to. <code>escalation_as_flag</code> counts an escalation as a flag, because an escalated
          payment is still held and still costs a review. The gap between them is the most informative number here.
        </p>
        <table className="mt-3 w-full">
          <thead>
            <tr className="text-[11px] uppercase tracking-wide text-slate-600">
              <th className="pb-1 text-left font-normal">view</th>
              <th className="px-2 pb-1 text-right font-normal">scored</th>
              <th className="px-2 pb-1 text-right font-normal">precision</th>
              <th className="px-2 pb-1 text-right font-normal">recall</th>
              <th className="px-2 pb-1 text-right font-normal">f1</th>
              <th className="px-2 pb-1 text-right font-normal">fpr</th>
              <th className="px-2 pb-1 text-right font-normal">accuracy</th>
              <th className="pl-2 pb-1 text-right font-normal">tp/fp/tn/fn</th>
            </tr>
          </thead>
          <tbody>
            <MatrixRow name="auto_only" m={auto} />
            <MatrixRow name="escalation_as_flag" m={flagged} />
          </tbody>
        </table>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-lg border border-ink-700 bg-ink-900 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-200">Confusion matrix (auto_only)</h2>
          <ConfusionMatrix m={auto} />
          <p className="mt-2 text-[11px] text-slate-600">
            {auto.abstained} case(s) escalated and therefore excluded from this matrix.
          </p>
        </section>
        <section className="rounded-lg border border-ink-700 bg-ink-900 p-4">
          <h2 className="mb-3 text-sm font-medium text-slate-200">Decisions across the bank</h2>
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={decisionChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#233150" />
              <XAxis dataKey="name" tick={{ fill: '#7c93b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#7c93b8', fontSize: 11 }} allowDecimals={false} />
              <Tooltip contentStyle={{ background: '#0d1220', border: '1px solid #233150', fontSize: 12 }} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {decisionChart.map((d) => (
                  <Cell key={d.name} fill={DECISION_FILL[d.name] ?? '#64748b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <section className="rounded-lg border border-ink-700 bg-ink-900 p-4">
        <h2 className="mb-3 text-sm font-medium text-slate-200">Per-category performance</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={categoryChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#233150" />
            <XAxis dataKey="name" tick={{ fill: '#7c93b8', fontSize: 11 }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#7c93b8', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#0d1220', border: '1px solid #233150', fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="precision" fill="#60a5fa" radius={[3, 3, 0, 0]} />
            <Bar dataKey="recall" fill="#34d399" radius={[3, 3, 0, 0]} />
            <Bar dataKey="fpr" name="false-positive rate" fill="#f87171" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>

        <table className="mt-4 w-full">
          <thead>
            <tr className="text-[11px] uppercase tracking-wide text-slate-600">
              <th className="pb-1 text-left font-normal">slice</th>
              <th className="px-2 pb-1 text-right font-normal">scored</th>
              <th className="px-2 pb-1 text-right font-normal">precision</th>
              <th className="px-2 pb-1 text-right font-normal">recall</th>
              <th className="px-2 pb-1 text-right font-normal">f1</th>
              <th className="px-2 pb-1 text-right font-normal">fpr</th>
              <th className="px-2 pb-1 text-right font-normal">accuracy</th>
              <th className="pl-2 pb-1 text-right font-normal">tp/fp/tn/fn</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.by_category).map(([name, s]) => (
              <MatrixRow key={name} name={`category=${name}`} m={s.auto_only} />
            ))}
            {Object.entries(report.by_difficulty).map(([name, s]) => (
              <MatrixRow key={name} name={`difficulty=${name}`} m={s.auto_only} />
            ))}
          </tbody>
        </table>
      </section>

      {/* ---------------------------------------------------------------- */}
      <section className="rounded-lg border-2 border-amber-700/40 bg-amber-950/10 p-5">
        <div className="flex items-center gap-2">
          <ShieldQuestion size={18} className="text-amber-400" />
          <h2 className="text-base font-semibold text-amber-100">Honesty &amp; self-scrutiny</h2>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-amber-100/80">{honesty.headline}</p>
        <p className="mt-2 text-xs italic leading-relaxed text-amber-200/60">{honesty.how_to_read_this}</p>

        {honesty.suspicious_results.length > 0 && (
          <div className="mt-4 space-y-3">
            <h3 className="text-xs uppercase tracking-wide text-amber-400/70">
              Slices at or above 95% — interrogated, not celebrated
            </h3>
            {honesty.suspicious_results.map((s) => (
              <div key={s.slice} className="rounded border border-amber-800/30 bg-ink-900/60 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <code className="text-xs text-amber-200">{s.slice}</code>
                  <Badge tone="escalate">{pct(s.accuracy)} over {s.scored_cases} case(s)</Badge>
                  {s.escalated_out_of_metric > 0 && (
                    <Badge tone="muted">{s.escalated_out_of_metric} escalated out of the metric</Badge>
                  )}
                </div>
                <p className="mt-1.5 text-sm font-medium text-amber-200/90">{s.verdict}</p>
                <ul className="mt-1.5 space-y-1">
                  {s.checks.map((c, i) => (
                    <li key={i} className="text-xs leading-relaxed text-slate-400">
                      • {c}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}

        <div className="mt-4 rounded border border-ink-700 bg-ink-900/60 p-3">
          <h3 className="text-xs uppercase tracking-wide text-amber-400/70">
            Ablation — is the multi-agent architecture doing real work?
          </h3>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">{ab.method}</p>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <StatCard label="examined" value={ab.auto_decisions_examined} />
            <StatCard label="survive ablation" value={ab.survives_removal_of_loudest_specialist} tone="good" />
            <StatCard label="collapse" value={ab.collapses_without_loudest_specialist} tone="bad" />
            <StatCard
              label="multi-signal share"
              value={pct(ab.multi_signal_share)}
              tone={(ab.multi_signal_share ?? 0) >= 0.8 ? 'good' : 'warn'}
            />
          </div>
          {ab.single_signal_cases.length > 0 && (
            <table className="mt-3 w-full">
              <thead>
                <tr className="text-[11px] uppercase tracking-wide text-slate-600">
                  <th className="pb-1 text-left font-normal">case</th>
                  <th className="px-2 pb-1 text-left font-normal">decision</th>
                  <th className="px-2 pb-1 text-left font-normal">remove</th>
                  <th className="px-2 pb-1 text-left font-normal">becomes</th>
                </tr>
              </thead>
              <tbody>
                {ab.single_signal_cases.map((r) => (
                  <tr key={r.case_id} className="border-t border-ink-800">
                    <td className="py-1.5 font-mono text-xs text-sky-300">
                      <Link to={`/cases/${r.case_id}`} className="hover:underline">
                        {r.case_id}
                      </Link>
                    </td>
                    <td className="px-2">
                      <Badge tone={toneForDecision(r.decision)}>{r.decision}</Badge>
                    </td>
                    <td className="px-2 font-mono text-xs text-slate-400">
                      {r.removed_specialist} ({r.removed_risk_score.toFixed(0)})
                    </td>
                    <td className="px-2">
                      <Badge tone={toneForDecision(r.decision_without_it)}>{r.decision_without_it}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            <strong className="text-slate-400">
              {ab.precedent_dependence.decisions_that_change_without_historical_case}
            </strong>{' '}
            auto-decision(s) change when the historical-case specialist is removed.{' '}
            {ab.precedent_dependence.why_this_matters}
          </p>
        </div>

        <div className="mt-4">
          <h3 className="text-xs uppercase tracking-wide text-amber-400/70">Concerns about these results</h3>
          <ul className="mt-2 space-y-2">
            {honesty.concerns.map((c, i) => (
              <li key={i} className="border-l-2 border-amber-700/40 pl-3 text-sm leading-relaxed text-slate-400">
                {c}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="rounded-lg border border-ink-700 bg-ink-900 p-4">
        <h2 className="mb-3 text-sm font-medium text-slate-200">Per-case results</h2>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px]">
            <thead>
              <tr className="text-[11px] uppercase tracking-wide text-slate-600">
                <th className="pb-1 text-left font-normal">case</th>
                <th className="px-2 pb-1 text-left font-normal">category</th>
                <th className="px-2 pb-1 text-left font-normal">truth</th>
                <th className="px-2 pb-1 text-left font-normal">difficulty</th>
                <th className="px-2 pb-1 text-left font-normal">decision</th>
                <th className="px-2 pb-1 text-right font-normal">risk</th>
                <th className="px-2 pb-1 text-right font-normal">conf</th>
                <th className="px-2 pb-1 text-center font-normal">correct</th>
                <th className="pl-2 pb-1 text-right font-normal">ms</th>
              </tr>
            </thead>
            <tbody>
              {report.case_results.map((r) => (
                <tr key={r.case_id} className="border-t border-ink-800">
                  <td className="py-1.5 font-mono text-xs">
                    <Link to={`/cases/${r.run_case_id}`} className="text-sky-300 hover:underline">
                      {r.case_id}
                    </Link>
                  </td>
                  <td className="px-2 text-xs text-slate-400">{r.category}</td>
                  <td className="px-2 text-xs text-slate-400">{r.ground_truth}</td>
                  <td className="px-2 text-xs text-slate-500">{r.difficulty}</td>
                  <td className="px-2">
                    <Badge tone={toneForDecision(r.decision)}>{r.decision ?? 'no verdict'}</Badge>
                  </td>
                  <td className="px-2 text-right font-mono text-xs text-slate-300">
                    {r.combined_risk?.toFixed(1) ?? '—'}
                  </td>
                  <td className="px-2 text-right font-mono text-xs text-slate-400">
                    {r.confidence?.toFixed(2) ?? '—'}
                  </td>
                  <td className="px-2 text-center text-xs">
                    {r.prediction_correct ? (
                      <span className="text-emerald-400">✓</span>
                    ) : (
                      <span className="text-rose-400">✗</span>
                    )}
                  </td>
                  <td className="pl-2 text-right font-mono text-[11px] text-slate-600">{r.duration_ms}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
