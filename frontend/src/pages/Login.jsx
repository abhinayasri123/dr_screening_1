import { useState } from 'react'
import { useNavigate, Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import { Activity, Lock, Mail, Shield, UserCheck, AlertCircle } from 'lucide-react'

const SEEDED_ROLES = [
  { id: 'clinician', label: 'Clinician', email: 'clinician@dr-screening.com', pass: 'clinician123', desc: 'Review & Grad-CAM' },
  { id: 'admin', label: 'Admin', email: 'admin@dr-screening.com', pass: 'admin123', desc: 'System Management' },
  { id: 'technician', label: 'Technician', email: 'technician@dr-screening.com', pass: 'tech123', desc: 'Image Intake & Triage' },
  { id: 'researcher', label: 'Researcher', email: 'researcher@dr-screening.com', pass: 'research123', desc: 'Federated Learning' },
]

const Login = () => {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('clinician@dr-screening.com')
  const [password, setPassword] = useState('clinician123')
  const [role, setRole] = useState('clinician')
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')

  if (user) {
    return <Navigate to="/dashboard" replace />
  }

  const selectRolePreset = (preset) => {
    setEmail(preset.email)
    setPassword(preset.pass)
    setRole(preset.id)
    setErrorMsg('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setErrorMsg('')
    setLoading(true)
    try {
      await login(email, password, role)
      toast.success('Clinical session authenticated successfully')
      navigate('/dashboard')
    } catch (err) {
      const msg = err.response?.data?.detail || 'Authentication failed. Please check your credentials.'
      setErrorMsg(msg)
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-transparent flex flex-col justify-center py-10 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="relative z-10 sm:mx-auto sm:w-full sm:max-w-md px-4 sm:px-0">
        
        {/* Portal Header */}
        <div className="text-center bg-[#121212]/95 backdrop-blur-md p-6 rounded-xl border border-[#292929] shadow-lg">
          <div className="inline-flex items-center justify-center p-3 bg-[#191919] rounded-xl text-[#81958D] border border-[#292929]">
            <Activity className="w-8 h-8 text-[#81958D]" />
          </div>
          <h1 className="mt-3 text-center text-2xl font-bold text-[#F5F0E6] tracking-tight">
            Clinex AI
          </h1>
          <p className="mt-1 text-center text-xs font-normal text-[#D0CCC3]">
            Diabetic Retinopathy Clinical Screening Portal
          </p>
          <div className="mt-2.5 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[11px] font-medium bg-[#191919] text-[#81958D] border border-[#292929]">
            <Shield className="w-3 h-3 text-[#81958D]" />
            <span>EfficientNet-B3 Radiographic Engine</span>
          </div>
        </div>

        {/* Form Card */}
        <div className="mt-5 bg-[#121212]/95 backdrop-blur-md py-6 px-6 rounded-xl border border-[#292929] sm:px-8 shadow-xl">
          
          {/* Preset Role Quick Selectors */}
          <div className="mb-5">
            <label className="block text-[11px] font-semibold text-[#85817A] uppercase tracking-wider mb-2">
              Select Demo Role Account:
            </label>
            <div className="grid grid-cols-2 gap-2">
              {SEEDED_ROLES.map((r) => {
                const isSelected = email === r.email
                return (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => selectRolePreset(r)}
                    className={`px-3 py-2 text-left rounded-lg text-xs transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-[#191919] border border-[#81958D] text-[#F5F0E6]'
                        : 'bg-[#0F0F0F] hover:bg-[#191919] border border-[#292929] text-[#85817A]'
                    }`}
                  >
                    <div className="font-semibold text-[#F5F0E6]">{r.label}</div>
                    <div className="text-[10px] text-[#85817A] truncate">{r.desc}</div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Error Banner */}
          {errorMsg && (
            <div className="mb-5 bg-rose-500/10 border border-rose-500/25 text-[#F87171] px-3.5 py-2.5 rounded-lg text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-[#F87171] flex-shrink-0 mt-0.5" />
              <div className="flex-1">{errorMsg}</div>
            </div>
          )}

          {/* Form */}
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-medium text-[#D0CCC3]">Email Address</label>
              <div className="mt-1 relative rounded-md">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#85817A]">
                  <Mail className="h-4 w-4" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@hospital.org"
                  className="block w-full pl-9 pr-3 py-2 bg-[#0F0F0F] border border-[#292929] rounded-lg text-[#F5F0E6] placeholder-[#85817A] focus:outline-none focus:border-[#81958D] text-xs"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#D0CCC3]">Password</label>
              <div className="mt-1 relative rounded-md">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#85817A]">
                  <Lock className="h-4 w-4" />
                </div>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="block w-full pl-9 pr-3 py-2 bg-[#0F0F0F] border border-[#292929] rounded-lg text-[#F5F0E6] placeholder-[#85817A] focus:outline-none focus:border-[#81958D] text-xs"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#D0CCC3]">Authorized Clinical Role</label>
              <div className="mt-1 relative rounded-md">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-[#85817A]">
                  <UserCheck className="h-4 w-4" />
                </div>
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="block w-full pl-9 pr-3 py-2 bg-[#0F0F0F] border border-[#292929] rounded-lg text-[#F5F0E6] focus:outline-none focus:border-[#81958D] text-xs"
                >
                  <option value="clinician">Clinician (Review & Screening)</option>
                  <option value="admin">Administrator / Lead Researcher</option>
                  <option value="technician">Ophthalmic Technician</option>
                  <option value="researcher">Clinical AI Researcher</option>
                </select>
              </div>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full flex justify-center items-center py-2.5 px-4 rounded-lg text-xs font-semibold text-[#111111] bg-[#E8E1D5] hover:bg-[#F5F0E6] border border-[#E8E1D5] disabled:opacity-50 transition-colors cursor-pointer"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-3.5 w-3.5 text-[#111111]" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                    </svg>
                    Authenticating Session...
                  </span>
                ) : (
                  'Sign In to Clinical Portal'
                )}
              </button>
            </div>
          </form>

          {/* Security & Regulatory Notice */}
          <div className="mt-5 border-t border-[#292929] pt-3 text-center">
            <p className="text-[10px] text-[#85817A] leading-relaxed">
              Confidential Medical Information &mdash; Authorized healthcare personnel only.
              All actions are monitored and audit-logged in compliance with HIPAA/GDPR.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Login
