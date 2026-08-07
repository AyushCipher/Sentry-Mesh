import clsx from 'clsx'

/**
 * Confidence bar with the auto-resolution floor drawn on it.
 *
 * The threshold marker matters: a reader seeing "confidence 0.66" has no way to
 * know whether that was enough, and the whole escalation story turns on it.
 */
export default function ConfidenceMeter({
  value,
  threshold = 0.7,
  label = 'confidence',
  compact = false,
}: {
  value: number
  threshold?: number
  label?: string
  compact?: boolean
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const clears = value >= threshold
  return (
    <div className="w-full">
      {!compact && (
        <div className="mb-1 flex items-baseline justify-between text-xs">
          <span className="text-slate-400">{label}</span>
          <span className={clsx('font-mono', clears ? 'text-emerald-300' : 'text-amber-300')}>
            {value.toFixed(2)}
            <span className="text-slate-500"> / {threshold.toFixed(2)} floor</span>
          </span>
        </div>
      )}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-ink-800">
        <div
          className={clsx('h-full rounded-full transition-all', clears ? 'bg-emerald-400' : 'bg-amber-400')}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute top-0 h-full w-px bg-slate-300/70"
          style={{ left: `${threshold * 100}%` }}
          title={`auto-resolution floor ${threshold}`}
        />
      </div>
    </div>
  )
}
