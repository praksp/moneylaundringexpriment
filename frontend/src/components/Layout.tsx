import { NavLink, Outlet } from 'react-router-dom'
import {
  ShieldAlert, Send, Users, Activity,
  Database, BarChart3,
} from 'lucide-react'
import clsx from 'clsx'

const navItems = [
  { to: '/', label: 'Dashboard', icon: BarChart3, end: true },
  { to: '/submit', label: 'New Transaction', icon: Send },
  { to: '/customers', label: 'Customer Profiles', icon: Users },
  { to: '/features', label: 'Feature Store', icon: Database },
  { to: '/monitor', label: 'Model Monitor', icon: Activity },
]

export default function Layout() {
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
          {navItems.map(({ to, label, icon: Icon, end }) => (
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
        </nav>
        <div className="px-4 py-4 border-t border-slate-800">
          <a
            href="http://localhost:7474"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            <Database size={13} />
            Neo4j Browser →
          </a>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
