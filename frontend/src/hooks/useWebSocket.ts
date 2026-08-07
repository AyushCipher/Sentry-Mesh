import { useCallback, useEffect, useRef, useState } from 'react'
import { traceSocketUrl } from '../api/client'
import type { Runtime, TraceEvent } from '../types'

export type SocketState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

interface TraceStream {
  events: TraceEvent[]
  state: SocketState
  runtime: Runtime | null
  complete: boolean
  error: string | null
  reset: () => void
}

/**
 * Tails a case's event log over the backend WebSocket.
 *
 * The server replays everything already recorded before it starts tailing, so
 * a late-connecting client still sees the full investigation. Events are
 * de-duplicated by event_id because that replay can overlap with what this
 * client already fetched over HTTP.
 */
export function useTraceStream(caseId: string | null): TraceStream {
  const [events, setEvents] = useState<TraceEvent[]>([])
  const [state, setState] = useState<SocketState>('idle')
  const [runtime, setRuntime] = useState<Runtime | null>(null)
  const [complete, setComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const seen = useRef<Set<string>>(new Set())
  const socketRef = useRef<WebSocket | null>(null)

  const reset = useCallback(() => {
    seen.current = new Set()
    setEvents([])
    setComplete(false)
    setError(null)
  }, [])

  useEffect(() => {
    if (!caseId) {
      setState('idle')
      return
    }
    reset()
    setState('connecting')

    const socket = new WebSocket(traceSocketUrl(caseId))
    socketRef.current = socket

    socket.onopen = () => setState('open')
    socket.onerror = () => {
      setState('error')
      setError('WebSocket connection failed - is the backend running on :8000?')
    }
    socket.onclose = () => setState((prev) => (prev === 'error' ? prev : 'closed'))
    socket.onmessage = (message) => {
      try {
        const parsed = JSON.parse(message.data as string)
        if (parsed.type === 'runtime') {
          setRuntime(parsed.runtime as Runtime)
        } else if (parsed.type === 'event') {
          const event = parsed.event as TraceEvent
          if (seen.current.has(event.event_id)) return
          seen.current.add(event.event_id)
          setEvents((prev) => [...prev, event])
        } else if (parsed.type === 'complete') {
          setComplete(true)
        } else if (parsed.type === 'error') {
          setError(String(parsed.error))
        }
      } catch {
        setError('received a malformed frame from the trace stream')
      }
    }

    return () => {
      socket.close()
      socketRef.current = null
    }
  }, [caseId, reset])

  return { events, state, runtime, complete, error, reset }
}
