import React, { Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider, useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import Login from './pages/Login'

// Route-level code splitting
const Dashboard = React.lazy(() => import('./pages/Dashboard'))
const SubmitTransaction = React.lazy(() => import('./pages/SubmitTransaction'))
const CustomerProfiles = React.lazy(() => import('./pages/CustomerProfiles'))
const FeatureStore = React.lazy(() => import('./pages/FeatureStore'))
const ModelMonitor = React.lazy(() => import('./pages/ModelMonitor'))
const Overview = React.lazy(() => import('./pages/Overview'))
const AnomalyDetection = React.lazy(() => import('./pages/AnomalyDetection'))
const GraphSAGEDetection = React.lazy(() => import('./pages/GraphSAGEDetection'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 10_000 },
  },
})

/** Redirects to /login if not authenticated. */
function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

/** Redirects to / if user does not have the admin role. */
function RequireAdmin({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />
  if (user.role !== 'admin') return <Navigate to="/" replace />
  return <>{children}</>
}

/** Redirects already-authenticated users away from the login page. */
function GuestOnly({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  if (user) return <Navigate to="/" replace />
  return <>{children}</>
}

function AppRoutes() {
  return (
    <Routes>
      {/* Public login */}
      <Route
        path="/login"
        element={<GuestOnly><Login /></GuestOnly>}
      />

      {/* Protected shell */}
      <Route
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        {/* All authenticated users */}
        <Route index element={
          <Suspense fallback={<div className="p-8 text-slate-400">Loading Dashboard...</div>}>
            <Dashboard />
          </Suspense>
        } />
        <Route path="submit" element={
          <Suspense fallback={<div className="p-8 text-slate-400">Loading Form...</div>}>
            <SubmitTransaction />
          </Suspense>
        } />
        <Route path="overview" element={
          <Suspense fallback={<div className="p-8 text-slate-400">Loading Overview...</div>}>
            <Overview />
          </Suspense>
        } />
        <Route path="monitor" element={
          <Suspense fallback={<div className="p-8 text-slate-400">Loading Monitor...</div>}>
            <ModelMonitor />
          </Suspense>
        } />

        {/* Admin-only */}
        <Route
          path="customers"
          element={
            <RequireAdmin>
              <Suspense fallback={<div className="p-8 text-slate-400">Loading Profiles...</div>}>
                <CustomerProfiles />
              </Suspense>
            </RequireAdmin>
          }
        />
        <Route
          path="features"
          element={
            <RequireAdmin>
              <Suspense fallback={<div className="p-8 text-slate-400">Loading Feature Store...</div>}>
                <FeatureStore />
              </Suspense>
            </RequireAdmin>
          }
        />
        <Route
          path="anomaly"
          element={
            <RequireAdmin>
              <Suspense fallback={<div className="p-8 text-slate-400">Loading Anomaly Detection...</div>}>
                <AnomalyDetection />
              </Suspense>
            </RequireAdmin>
          }
        />
        <Route
          path="graphsage"
          element={
            <RequireAdmin>
              <Suspense fallback={<div className="p-8 text-slate-400">Loading GraphSAGE...</div>}>
                <GraphSAGEDetection />
              </Suspense>
            </RequireAdmin>
          }
        />
      </Route>

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
