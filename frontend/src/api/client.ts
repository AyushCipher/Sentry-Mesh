import axios from 'axios'
import type { CaseStatus, HarnessReport, ReviewCase, Runtime, SeededCase, TraceEvent } from '../types'

const api = axios.create({ baseURL: '/', timeout: 30000 })

export interface HealthResponse {
  status: string
  event_log_reachable: boolean
  review_queue_depth: number
  in_flight_investigations: number
  runtime: Runtime
}

export const getHealth = async (): Promise<HealthResponse> => (await api.get('/health')).data

export const getCaseBank = async (): Promise<{ count: number; cases: SeededCase[] }> =>
  (await api.get('/api/case-bank')).data

export const submitCase = async (body: {
  seeded_case_id?: string
  case_id?: string
  category?: string
  summary?: string
  transaction_data?: Record<string, unknown>
  device_data?: Record<string, unknown>
  network_data?: Record<string, unknown>
}): Promise<{ case_id: string; status: string; run_mode: string }> =>
  (await api.post('/api/cases', body)).data

export const getCase = async (caseId: string): Promise<CaseStatus> =>
  (await api.get(`/api/cases/${encodeURIComponent(caseId)}`)).data

export const getTrace = async (
  caseId: string,
): Promise<{ case_id: string; event_count: number; events: TraceEvent[] }> =>
  (await api.get(`/api/cases/${encodeURIComponent(caseId)}/trace`)).data

export const listCases = async (): Promise<{
  count: number
  cases: { case_id: string; status: string; summary: string; opened_at: string }[]
}> => (await api.get('/api/cases')).data

export const getReviewQueue = async (): Promise<{
  depth: number
  backend: string
  cases: ReviewCase[]
}> => (await api.get('/api/review-queue')).data

export const submitReview = async (
  caseId: string,
  body: { action: string; reviewer?: string; notes?: string; requested_specialist?: string },
): Promise<Record<string, unknown>> =>
  (await api.post(`/api/review-queue/${encodeURIComponent(caseId)}`, body)).data

export const getEvalReport = async (): Promise<HarnessReport> => (await api.get('/api/eval/report')).data

export const getEvalCase = async (caseId: string): Promise<Record<string, unknown>> =>
  (await api.get(`/api/eval/report/${encodeURIComponent(caseId)}`)).data

export const traceSocketUrl = (caseId: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/cases/${encodeURIComponent(caseId)}`
}

export default api
