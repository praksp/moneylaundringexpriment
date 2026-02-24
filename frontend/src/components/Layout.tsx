import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  ShieldAlert, Send, Users, Activity,
  Database, BarChart3, Globe, LogOut, ChevronDown,
  Shield, Eye, ScanSearch, Network,
} from 'lucide-react'
import clsx from 'clsx'
import { useState } from 'react'
import { useAuth } from '../context/AuthContext'

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const isAdmin = user?.role === 'admin'

  const navItems = [
    { to: '/', label: 'Dashboard', icon: BarChart3, end: true, roles: ['admin', 'viewer'] },
    { to: '/submit', label: 'New Transaction', icon: Send, roles: ['admin', 'viewer'] },
    { to: '/overview', label: 'Global Overview', icon: Globe, roles: ['admin', 'viewer'] },
    // Admin-only
    { to: '/customers', label: 'Customer Profiles', icon: Users, roles: ['admin'] },
    { to: '/anomaly',   label: 'Anomaly Detection', icon: ScanSearch, roles: ['admin'] },
    { to: '/graphsage', label: 'GraphSAGE Mule',    icon: Network,   roles: ['admin'] },
    { to: '/features',  label: 'Feature Store', icon: Database, roles: ['admin'] },
    { to: '/monitor',   label: 'Model Monitor', icon: Activity, roles: ['admin', 'viewer'] },
  ]

  const visibleNav = navItems.filter(item => item.roles.includes(user?.role ?? ''))

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col">
        <div className="px-6 py-5 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <ShieldAlert className="text-blue-400" size={22} />
            <div>
              <p className="text-sm font-bold text-white leading-tight">AML Risk Engine</p>
              <p className="text-xs text-slate-500">Neo4j · Bayesian · XGBoost</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {/* Section header for admin tools */}
          {isAdmin && (
            <p className="px-3 pb-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Navigation
            </p>
          )}
          {visibleNav.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors',
                  isActive
                    ? 'bg-blue-600/20 text-blue-400 font-medium'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white',
                )
              }
            >
              <Icon size={17} />
              {label}
            </NavLink>
          ))}

          {/* Admin-only badge */}
          {isAdmin && (
            <div className="mt-3 pt-3 border-t border-slate-800">
              <p className="px-3 pb-1 text-xs font-semibold text-blue-400/70 uppercase tracking-wider flex items-center gap-1">
                <Shield size={10} />
                Admin Access
              </p>
            </div>
          )}
        </nav>

        {/* User section */}
        <div className="px-4 py-3 border-t border-slate-800">
          {/* Neo4j link */}
          <a
            href="http://localhost:7474"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors mb-3"
          >
            <Database size={13} />
            Neo4j Browser →
          </a>

          {/* User menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(v => !v)}
              className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg
                         hover:bg-slate-800 transition-colors text-left"
            >
              {/* Avatar */}
              <div className={clsx(
                'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0',
                isAdmin ? 'bg-blue-600/30 text-blue-400' : 'bg-emerald-600/30 text-emerald-400',
              )}>
                {user?.username?.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-white font-medium truncate">{user?.username}</p>
                <div className="flex items-center gap-1">
                  {isAdmin
                    ? <Shield size={10} className="text-blue-400" />
                    : <Eye size={10} className="text-emerald-400" />}
                  <p className={clsx(
                    'text-xs capitalize',
                    isAdmin ? 'text-blue-400' : 'text-emerald-400',
                  )}>
                    {user?.role}
                  </p>
                </div>
              </div>
              <ChevronDown
                size={14}
                className={clsx('text-slate-500 transition-transform', showUserMenu && 'rotate-180')}
              />
            </button>

            {showUserMenu && (
              <div className="absolute bottom-full left-0 right-0 mb-1
                              bg-slate-800 border border-slate-700 rounded-lg
                              shadow-xl overflow-hidden z-50">
                <div className="px-3 py-2 border-b border-slate-700">
                  <p className="text-xs text-slate-400">Signed in as</p>
                  <p className="text-sm text-white font-semibold">{user?.full_name || user?.username}</p>
                  <span className={clsx(
                    'inline-block text-xs px-1.5 py-0.5 rounded mt-0.5',
                    isAdmin ? 'bg-blue-600/20 text-blue-400' : 'bg-emerald-600/20 text-emerald-400',
                  )}>
                    {user?.role?.toUpperCase()}
                  </span>
                </div>
                <button
                  onClick={handleLogout}
                  className="w-full flex items-center gap-2 px-3 py-2.5
                             text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <LogOut size={14} />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
