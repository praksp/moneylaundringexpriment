import {
  createContext, useContext, useState, useEffect,
  type ReactNode,
} from 'react'
import { api } from '../api/client'

export interface AuthUser {
  id: string
  username: string
  role: 'admin' | 'viewer'
  full_name?: string
  is_active: boolean
}

interface AuthState {
  user: AuthUser | null
  token: string | null
  isLoading: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

const TOKEN_KEY = 'aml_token'
const USER_KEY  = 'aml_user'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [user, setUser]   = useState<AuthUser | null>(() => {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? (JSON.parse(raw) as AuthUser) : null
  })
  const [isLoading, setIsLoading] = useState(false)

  // Inject token into every axios request
  useEffect(() => {
    const interceptor = api.interceptors.request.use(config => {
      if (token) config.headers.Authorization = `Bearer ${token}`
      return config
    })
    return () => api.interceptors.request.eject(interceptor)
  }, [token])

  // On 401 → auto-logout
  useEffect(() => {
    const interceptor = api.interceptors.response.use(
      res => res,
      err => {
        if (err.response?.status === 401) logout()
        return Promise.reject(err)
      },
    )
    return () => api.interceptors.response.eject(interceptor)
  })

  const login = async (username: string, password: string) => {
    setIsLoading(true)
    try {
      const res = await api.post<{ access_token: string; user: AuthUser }>(
        '/auth/login',
        { username, password },
      )
      const { access_token, user: u } = res.data
      setToken(access_token)
      setUser(u)
      localStorage.setItem(TOKEN_KEY, access_token)
      localStorage.setItem(USER_KEY, JSON.stringify(u))
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    setToken(null)
    setUser(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
