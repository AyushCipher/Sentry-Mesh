import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/layout/Layout'
import CaseDetailPage from './pages/CaseDetailPage'
import EvalLeaderboardPage from './pages/EvalLeaderboardPage'
import LiveInvestigationPage from './pages/LiveInvestigationPage'
import ReviewQueuePage from './pages/ReviewQueuePage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<LiveInvestigationPage />} />
          <Route path="review" element={<ReviewQueuePage />} />
          <Route path="eval" element={<EvalLeaderboardPage />} />
          <Route path="cases" element={<CaseDetailPage />} />
          <Route path="cases/:caseId" element={<CaseDetailPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
