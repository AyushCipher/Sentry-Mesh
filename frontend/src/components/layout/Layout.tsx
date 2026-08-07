import { useEffect, useState } from 'react'
import { Outlet } from 'react-router-dom'
import { getHealth, type HealthResponse } from '../../api/client'
import Sidebar from './Sidebar'
import TopBar from './TopBar'

export default function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const data = await getHealth()
        if (alive) setHealth(data)
      } catch {
        if (alive) setHealth(null)
      }
    }
    void poll()
    const timer = setInterval(poll, 5000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar queueDepth={health?.review_queue_depth ?? 0} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar health={health} />
        <main className="flex-1 overflow-y-auto px-6 py-6">
          <Outlet context={{ health }} />
        </main>
      </div>
    </div>
  )
}
