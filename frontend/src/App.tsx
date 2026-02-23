import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider, useAuth } from './context/AuthContext'
import Layout from './components/Layout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import SubmitTransaction from './pages/SubmitTransaction'
import CustomerProfiles from './pages/CustomerProfiles'
import FeatureStore from './pages/FeatureStore'
import ModelMonitor from './pages/ModelMonitor'
import Overview from './pages/Overview'

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
        <Route index element={<Dashboard />} />
        <Route path="submit" element={<SubmitTransaction />} />
        <Route path="overview" element={<Overview />} />
        <Route path="monitor" element={<ModelMonitor />} />

        {/* Admin-only */}
        <Route
          path="customers"
          element={<RequireAdmin><CustomerProfiles /></RequireAdmin>}
        />
        <Route
          path="features"
          element={<RequireAdmin><FeatureStore /></RequireAdmin>}
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
