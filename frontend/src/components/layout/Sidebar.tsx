import clsx from 'clsx'
import { Activity, BarChart3, Inbox, ScrollText } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const LINKS = [
  { to: '/', label: 'Live Investigation', icon: Activity, end: true },
  { to: '/review', label: 'Review Queue', icon: Inbox, end: false },
  { to: '/eval', label: 'Eval Leaderboard', icon: BarChart3, end: false },
  { to: '/cases', label: 'Case Detail', icon: ScrollText, end: false },
]

export default function Sidebar({ queueDepth }: { queueDepth: number }) {
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-ink-800 bg-ink-900">
      <div className="border-b border-ink-800 px-5 py-4">
        <div className="text-lg font-semibold tracking-tight text-slate-100">SentryMesh</div>
        <div className="text-[11px] text-slate-500">multi-agent fraud orchestration</div>
      </div>
      <nav className="flex-1 space-y-1 p-3">
        {LINKS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-ink-800 text-slate-100'
                  : 'text-slate-400 hover:bg-ink-850 hover:text-slate-200',
              )
            }
          >
            <Icon size={16} />
            <span>{label}</span>
            {to === '/review' && queueDepth > 0 && (
              <span className="ml-auto rounded bg-amber-500/20 px-1.5 text-xs text-amber-300">{queueDepth}</span>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-ink-800 px-4 py-3 text-[11px] leading-relaxed text-slate-600">
        All case data is <span className="text-slate-500">synthetic</span>.
        <br />
        Project by Ayush Verma
      </div>
    </aside>
  )
}
