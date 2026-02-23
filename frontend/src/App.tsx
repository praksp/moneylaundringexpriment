import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import SubmitTransaction from './pages/SubmitTransaction'
import CustomerProfiles from './pages/CustomerProfiles'
import FeatureStore from './pages/FeatureStore'
import ModelMonitor from './pages/ModelMonitor'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 10_000 },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="submit" element={<SubmitTransaction />} />
            <Route path="customers" element={<CustomerProfiles />} />
            <Route path="features" element={<FeatureStore />} />
            <Route path="monitor" element={<ModelMonitor />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
