import { AlertOctagon, Cpu, Database, Search } from 'lucide-react'
import type { HealthResponse } from '../../api/client'
import Badge from '../ui/Badge'

/**
 * Surfaces which backends this process is actually running on.
 *
 * Whether reasoning came from a live model or the offline deterministic
 * fallback changes what every number on every other page means, so it is shown
 * permanently rather than buried in a settings panel.
 */
export default function TopBar({ health }: { health: HealthResponse | null }) {
  const rt = health?.runtime
  const offline = rt?.run_mode !== 'anthropic'

  return (
    <header className="flex flex-wrap items-center gap-3 border-b border-ink-800 bg-ink-900/60 px-6 py-3">
      {!health ? (
        <Badge tone="muted">connecting to backend…</Badge>
      ) : (
        <>
          <Badge tone={offline ? 'escalate' : 'approve'} title="LLM reasoning backend">
            <Cpu size={12} />
            {offline ? 'offline deterministic reasoner' : `anthropic · ${rt?.model}`}
          </Badge>
          <Badge tone={rt?.event_log_backend === 'redis' ? 'accent' : 'muted'} title="Shared event log backend">
            <Database size={12} />
            event log: {rt?.event_log_backend}
          </Badge>
          <Badge tone="muted" title="Precedent retrieval backend">
            <Search size={12} />
            retrieval: {rt?.retrieval_backend}
          </Badge>
          <Badge tone={health.review_queue_depth > 0 ? 'escalate' : 'muted'}>
            {health.review_queue_depth} awaiting review
          </Badge>
          {health.in_flight_investigations > 0 && (
            <Badge tone="accent">{health.in_flight_investigations} investigating</Badge>
          )}
        </>
      )}
      {offline && health && (
        <span className="ml-auto flex items-center gap-1.5 text-[11px] text-amber-300/80">
          <AlertOctagon size={12} />
          No ANTHROPIC_API_KEY: findings come from the labelled deterministic fallback, not a model.
        </span>
      )}
    </header>
  )
}
