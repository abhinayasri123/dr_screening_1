import { Outlet, Link, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { 
  LayoutDashboard, FileImage, Users, CheckSquare, LineChart, 
  Network, LogOut, Eye, FileText, ShieldCheck, User,
  Cpu, Plus, Activity
} from 'lucide-react'

const Layout = () => {
  const { logout, user } = useAuth()
  const location = useLocation()

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'New Screening', path: '/screening/new', icon: FileImage },
    { name: 'Patients', path: '/patients', icon: Users },
    { name: 'Review Queue', path: '/reviews', icon: CheckSquare },
    { name: 'Explainability (XAI)', path: '/explainability', icon: Eye },
    { name: 'Clinical Reports', path: '/reports', icon: FileText },
    { name: 'Research Metrics', path: '/research', icon: LineChart },
    { name: 'Federated Learning', path: '/federated', icon: Network },
    { name: 'Administration', path: '/admin', icon: ShieldCheck },
  ]

  return (
    <div className="min-h-screen bg-transparent flex flex-col relative overflow-hidden font-sans antialiased text-[#D0CCC3]">
      <div className="relative z-10 flex flex-col flex-1">
        <div className="flex flex-1">
          {/* ── Elegant Midnight Sidebar (#0B0B0B, Border #292929) ── */}
          <aside className="w-64 glass-sidebar flex flex-col border-r border-[#292929] bg-[#0B0B0B]/96 backdrop-blur-md">
            {/* Brand Header */}
            <div className="p-5 border-b border-[#292929] bg-[#070707]/60">
              <div className="flex items-center space-x-3">
                <div className="w-9 h-9 rounded-lg bg-[#121212] border border-[#292929] flex items-center justify-center text-[#81958D] flex-shrink-0">
                  <Cpu className="w-5 h-5 text-[#81958D]" />
                </div>
                <div>
                  <div className="flex items-center space-x-1.5">
                    <span className="text-base font-bold text-[#F5F0E6] tracking-tight">Clinex</span>
                    <span className="text-base font-bold text-[#81958D] tracking-tight">AI</span>
                  </div>
                  <p className="text-[10px] font-medium text-[#85817A] tracking-wider uppercase">DR Decision Support</p>
                </div>
              </div>
            </div>
            
            {/* Navigation Links */}
            <nav className="flex-1 p-3.5 space-y-1 overflow-y-auto">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname.startsWith(item.path)
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center px-3.5 py-2.5 text-xs font-medium rounded-lg transition-colors ${
                      isActive 
                        ? 'bg-[#121212] text-[#F5F0E6] border border-[#81958D]/40 font-semibold' 
                        : 'text-[#D0CCC3] hover:bg-[#191919] hover:text-[#F5F0E6]'
                    }`}
                  >
                    <Icon className={`w-4 h-4 mr-3 flex-shrink-0 ${isActive ? 'text-[#81958D]' : 'text-[#85817A]'}`} />
                    <span>{item.name}</span>
                  </Link>
                )
              })}
            </nav>
            
            {/* Authenticated Clinician User Card */}
            <div className="p-3.5 border-t border-[#292929] bg-[#070707]/70">
              <div className="flex items-center space-x-2.5 mb-2.5 px-3 py-2 bg-[#121212] rounded-lg border border-[#292929]">
                <div className="w-7 h-7 rounded-full bg-[#191919] border border-[#292929] flex items-center justify-center text-[#81958D] flex-shrink-0">
                  <User className="w-3.5 h-3.5 text-[#81958D]" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold text-[#F5F0E6] truncate">{user?.email || 'clinician@hospital.org'}</p>
                  <p className="text-[10px] font-medium text-[#81958D] uppercase tracking-wider">{user?.role || 'Clinician'}</p>
                </div>
              </div>
              <button
                onClick={logout}
                className="flex items-center justify-center w-full px-3 py-2 text-xs font-medium text-[#F87171] bg-[#191919] hover:bg-[#222222] border border-[#292929] rounded-lg transition-colors"
              >
                <LogOut className="w-3.5 h-3.5 mr-2" />
                Sign Out
              </button>
            </div>
          </aside>

          {/* ── Main View Area With Professional Top Navigation ── */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Professional Top Navigation Bar */}
            <header className="h-16 px-6 border-b border-[#292929] bg-[#0B0B0B]/80 backdrop-blur-md flex items-center justify-between sticky top-0 z-20">
              <div className="flex items-center space-x-3">
                <span className="text-xs font-semibold text-[#F5F0E6]">Ophthalmology Clinical Suite</span>
                <span className="text-[#292929]">&bull;</span>
                <span className="text-xs text-[#85817A] hidden sm:inline">Hospital AI Node</span>
              </div>

              <div className="flex items-center space-x-4">
                {/* Status Indicator */}
                <div className="hidden md:flex items-center space-x-2 px-2.5 py-1 bg-[#121212] rounded-md border border-[#292929] text-[11px]">
                  <span className="w-2 h-2 rounded-full bg-[#81958D] animate-pulse"></span>
                  <span className="text-[#D0CCC3] font-medium">EfficientNet-B3</span>
                  <span className="text-[#85817A]">|</span>
                  <span className="text-[#81958D]">Active</span>
                </div>

                {/* Quick Action Button */}
                <Link
                  to="/screening/new"
                  className="inline-flex items-center px-3.5 py-1.5 bg-[#E8E1D5] hover:bg-[#F5F0E6] text-[#111111] text-xs font-semibold rounded-lg transition-colors"
                >
                  <Plus className="w-3.5 h-3.5 mr-1 text-[#111111]" />
                  New Screening
                </Link>
              </div>
            </header>

            {/* Main Workspace Content */}
            <main className="flex-1 p-5 md:p-8 overflow-y-auto bg-transparent flex flex-col justify-between">
              <div className="max-w-7xl mx-auto w-full">
                <Outlet />
              </div>
              <footer className="mt-10 py-3 px-4 bg-[#121212]/95 backdrop-blur-md rounded-lg text-center text-xs text-[#85817A] font-normal max-w-4xl mx-auto w-full border border-[#292929]">
                This tool provides AI-assisted screening support and is not a substitute for professional medical diagnosis. Final clinical decisions must be made by qualified healthcare professionals.
              </footer>
            </main>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Layout
