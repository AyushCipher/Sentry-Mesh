import clsx from 'clsx'
import type { ReactNode } from 'react'

export default function StatCard({
  label,
  value,
  hint,
  tone = 'neutral',
  icon,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'neutral' | 'good' | 'warn' | 'bad'
  icon?: ReactNode
}) {
  return (
    <div className="rounded-lg border border-ink-700 bg-ink-900 p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>
      <div
        className={clsx(
          'mt-2 font-mono text-2xl font-semibold',
          tone === 'good' && 'text-emerald-300',
          tone === 'warn' && 'text-amber-300',
          tone === 'bad' && 'text-rose-300',
          tone === 'neutral' && 'text-slate-100',
        )}
      >
        {value}
      </div>
      {hint && <div className="mt-1 text-xs leading-relaxed text-slate-500">{hint}</div>}
    </div>
  )
}
