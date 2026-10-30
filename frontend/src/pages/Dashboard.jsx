import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { 
  Users, FileImage, CheckSquare, Activity, ArrowRight, 
  Clock, ShieldCheck, Sparkles, TrendingUp
} from 'lucide-react'
import { 
  ResponsiveContainer, PieChart, Pie, Cell, Legend, Tooltip 
} from 'recharts'
import { getPredictions } from '../api/predictions'
import { getDashboardStats } from '../api/analytics'
import { useAuth } from '../context/AuthContext'

const Dashboard = () => {
  const { user } = useAuth()
  const [predictions, setPredictions] = useState([])
  const [stats, setStats] = useState({
    total_patients: 0,
    total_predictions: 0,
    pending_reviews: 0,
    completed_reviews: 0,
    agreement_rate: 0,
    class_distribution: []
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getDashboardStats(),
      getPredictions(0, 8)
    ]).then(([statsRes, predsRes]) => {
      if (statsRes) {
        setStats(statsRes)
      }
      setPredictions(Array.isArray(predsRes) ? predsRes : predsRes.items || [])
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  // Restrained Midnight Ivory chart palette: sage, soft ivory, warm grey, restrained amber, restrained crimson
  const CHART_COLORS = ['#81958D', '#A4B9B0', '#D0CCC3', '#D97706', '#DC2626']

  return (
    <div className="space-y-6 pb-10">
      {/* ── Welcome Section ── */}
      <div className="bg-[#121212]/95 backdrop-blur-md p-6 rounded-xl border border-[#292929] flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#191919] text-[#81958D] border border-[#292929] mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#81958D]"></span>
            <span>Diabetic Retinopathy Screening Portal &bull; Session Active</span>
          </div>
          <h1 className="text-2xl font-bold text-[#F5F0E6] tracking-tight">
            Welcome back{user?.full_name ? `, ${user.full_name}` : user?.email ? `, ${user.email.split('@')[0]}` : ''}
          </h1>
          <p className="text-xs text-[#D0CCC3] mt-1">
            Clinical decision-support suite powered by EfficientNet-B3 with dual-stage retinal validation and Grad-CAM explainability.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link 
            to="/screening/new"
            className="inline-flex items-center px-4 py-2 bg-[#E8E1D5] hover:bg-[#F5F0E6] text-[#111111] rounded-lg text-xs font-semibold transition-colors"
          >
            New Screening <ArrowRight className="w-3.5 h-3.5 ml-1.5 text-[#111111]" />
          </Link>
        </div>
      </div>

      {/* ── 4 Stat Cards (#121212, Border #292929) ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Patients */}
        <div className="bg-[#121212]/95 backdrop-blur-md p-5 rounded-xl border border-[#292929] flex items-center justify-between transition-colors hover:border-[#81958D]/40">
          <div>
            <p className="text-[11px] font-semibold text-[#85817A] uppercase tracking-wider">Total Patients</p>
            <p className="text-2xl font-bold text-[#F5F0E6] mt-1">{stats.total_patients}</p>
            <p className="text-[11px] text-[#85817A] mt-0.5">Registered in database</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-[#191919] border border-[#292929] flex items-center justify-center text-[#81958D]">
            <Users className="w-5 h-5" />
          </div>
        </div>

        {/* Total Screenings */}
        <div className="bg-[#121212]/95 backdrop-blur-md p-5 rounded-xl border border-[#292929] flex items-center justify-between transition-colors hover:border-[#81958D]/40">
          <div>
            <p className="text-[11px] font-semibold text-[#85817A] uppercase tracking-wider">Total Screenings</p>
            <p className="text-2xl font-bold text-[#F5F0E6] mt-1">{stats.total_predictions}</p>
            <p className="text-[11px] text-[#85817A] mt-0.5">AI evaluations executed</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-[#191919] border border-[#292929] flex items-center justify-center text-[#81958D]">
            <FileImage className="w-5 h-5" />
          </div>
        </div>

        {/* Pending Reviews */}
        <div className="bg-[#121212]/95 backdrop-blur-md p-5 rounded-xl border border-[#292929] flex items-center justify-between transition-colors hover:border-[#81958D]/40">
          <div>
            <p className="text-[11px] font-semibold text-[#85817A] uppercase tracking-wider">Pending Reviews</p>
            <p className="text-2xl font-bold text-[#FBBF24] mt-1">{stats.pending_reviews}</p>
            <p className="text-[11px] text-[#85817A] mt-0.5">Awaiting clinician review</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-[#191919] border border-[#292929] flex items-center justify-center text-[#FBBF24]">
            <Clock className="w-5 h-5" />
          </div>
        </div>

        {/* Model Performance & Clinician Agreement */}
        <div className="bg-[#121212]/95 backdrop-blur-md p-5 rounded-xl border border-[#292929] flex items-center justify-between transition-colors hover:border-[#81958D]/40">
          <div>
            <p className="text-[11px] font-semibold text-[#85817A] uppercase tracking-wider">Clinician Agreement</p>
            <p className="text-2xl font-bold text-[#81958D] mt-1">{stats.agreement_rate}%</p>
            <p className="text-[11px] text-[#85817A] mt-0.5">{stats.completed_reviews} reviews completed</p>
          </div>
          <div className="w-10 h-10 rounded-lg bg-[#191919] border border-[#292929] flex items-center justify-center text-[#81958D]">
            <CheckSquare className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* ── Recent Screenings & Analytics Charts ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Screening Activity Table */}
        <div className="lg:col-span-2 bg-[#121212]/95 backdrop-blur-md rounded-xl border border-[#292929] overflow-hidden flex flex-col justify-between">
          <div>
            <div className="p-4 border-b border-[#292929] flex justify-between items-center bg-[#191919]/60">
              <div>
                <h2 className="text-sm font-semibold text-[#F5F0E6]">Recent Screening Activity</h2>
                <p className="text-[11px] text-[#85817A]">Latest evaluated patient encounters</p>
              </div>
              <Link 
                to="/reviews" 
                className="text-xs font-medium text-[#81958D] hover:text-[#A4B9B0] flex items-center"
              >
                Review Queue <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="text-[11px] text-[#85817A] bg-[#101010] uppercase border-b border-[#292929]">
                  <tr>
                    <th className="px-4 py-3">Patient Code</th>
                    <th className="px-4 py-3">Predicted Class</th>
                    <th className="px-4 py-3">Confidence</th>
                    <th className="px-4 py-3">Review Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#292929]">
                  {predictions.length > 0 ? predictions.map((pred) => (
                    <tr key={pred.id} className="hover:bg-[#191919]/60 transition-colors">
                      <td className="px-4 py-3 font-medium text-[#F5F0E6]">
                        {pred.patient_code || pred.patient_name || (pred.patient_id ? pred.patient_id.substring(0, 8) : 'PAT-001')}
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-semibold text-[#F5F0E6]">
                          {pred.predicted_label || pred.prediction}
                        </span>
                        <span className="text-[11px] text-[#85817A] ml-1.5">(Class {pred.predicted_class})</span>
                      </td>
                      <td className="px-4 py-3 font-semibold text-[#81958D]">
                        {pred.confidence ? `${(pred.confidence * 100).toFixed(1)}%` : '---'}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-block px-2.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${
                          pred.review_status === 'APPROVED'
                            ? 'bg-[#81958D]/15 text-[#A8C4B8] border border-[#81958D]/30'
                            : pred.review_status === 'MODIFIED'
                            ? 'bg-amber-500/15 text-[#FBBF24] border border-amber-500/30'
                            : pred.review_status === 'REJECTED'
                            ? 'bg-rose-500/15 text-[#F87171] border border-rose-500/30'
                            : 'bg-[#191919] text-[#D0CCC3] border border-[#292929]'
                        }`}>
                          {pred.review_status || 'PENDING'}
                        </span>
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="4" className="px-4 py-8 text-center text-[#85817A] text-xs">
                        No screening records found in database.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="p-3.5 border-t border-[#292929] bg-[#191919]/40 flex justify-between items-center text-xs text-[#85817A]">
            <span>Validated AI classification with Grad-CAM explainability</span>
            <Link to="/screening/new" className="text-[#81958D] font-semibold hover:text-[#A4B9B0]">
              Start Screening &rarr;
            </Link>
          </div>
        </div>

        {/* Analytics Chart: DR Severity Distribution */}
        <div className="bg-[#121212]/95 backdrop-blur-md p-5 rounded-xl border border-[#292929] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-sm font-semibold text-[#F5F0E6]">DR Severity Distribution</h2>
              <span className="text-[10px] font-medium px-2 py-0.5 bg-[#191919] text-[#81958D] rounded border border-[#292929]">ICDR Scale</span>
            </div>
            <p className="text-[11px] text-[#85817A] mb-4">Actual diagnoses recorded in screening database</p>
            
            {stats.class_distribution && stats.class_distribution.some(d => d.value > 0) ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={stats.class_distribution}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={70}
                      innerRadius={42}
                      paddingAngle={3}
                    >
                      {stats.class_distribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} stroke="#121212" strokeWidth={2} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#121212', borderColor: '#292929', borderRadius: '6px', color: '#F5F0E6', fontSize: '11px' }} 
                      formatter={(value) => [value, 'Patients']} 
                    />
                    <Legend wrapperStyle={{ color: '#F5F0E6', fontSize: '11px', paddingTop: '8px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-56 flex flex-col items-center justify-center text-[#85817A] text-xs bg-[#191919]/50 rounded-lg border border-dashed border-[#292929]">
                <Activity className="w-6 h-6 mb-2 text-[#81958D]" />
                <p className="font-medium text-[#F5F0E6]">No cases recorded yet.</p>
              </div>
            )}
          </div>

          <div className="pt-3 border-t border-[#292929] text-[10px] text-[#85817A]">
            Based on 5-class International Clinical DR Severity grading scale.
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
