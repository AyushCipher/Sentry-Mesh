import clsx from 'clsx'
import type { ReactNode } from 'react'

export type Tone = 'approve' | 'decline' | 'escalate' | 'neutral' | 'accent' | 'muted'

const TONES: Record<Tone, string> = {
  approve: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30',
  decline: 'bg-rose-500/15 text-rose-300 ring-rose-500/30',
  escalate: 'bg-amber-500/15 text-amber-300 ring-amber-500/30',
  neutral: 'bg-slate-500/15 text-slate-300 ring-slate-500/30',
  accent: 'bg-sky-500/15 text-sky-300 ring-sky-500/30',
  muted: 'bg-ink-800 text-slate-400 ring-ink-700',
}

export function toneForDecision(decision?: string | null): Tone {
  if (decision === 'auto_approve' || decision === 'approve') return 'approve'
  if (decision === 'auto_decline' || decision === 'decline') return 'decline'
  if (decision === 'escalate') return 'escalate'
  return 'neutral'
}

export function toneForRisk(risk: number): Tone {
  if (risk >= 70) return 'decline'
  if (risk <= 35) return 'approve'
  return 'escalate'
}

export default function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode
  tone?: Tone
  className?: string
  title?: string
}) {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset whitespace-nowrap',
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}
