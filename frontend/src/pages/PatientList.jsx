import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getPatients, createPatient } from '../api/patients'
import { Search, Plus, UserCircle, Calendar, Activity, CheckCircle2, AlertCircle } from 'lucide-react'
import { format } from 'date-fns'
import toast from 'react-hot-toast'

const PatientList = () => {
  const [patients, setPatients] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [newPatient, setNewPatient] = useState({
    patient_code: '',
    full_name: '',
    date_of_birth: '1970-01-01',
    sex: 'M',
    contact_email: ''
  })

  useEffect(() => {
    fetchPatients()
  }, [])

  const fetchPatients = async () => {
    setLoading(true)
    try {
      const res = await getPatients()
      const list = Array.isArray(res) ? res : res.items || []
      setPatients(list)
    } catch (err) {
      console.error('Error fetching patients:', err)
      toast.error('Failed to load patient records')
    } finally {
      setLoading(false)
    }
  }

  const handleCreatePatient = async (e) => {
    e.preventDefault()
    try {
      const created = await createPatient(newPatient)
      toast.success(`Patient ${created.patient_code} created successfully`)
      setShowModal(false)
      setNewPatient({
        patient_code: '',
        full_name: '',
        date_of_birth: '1970-01-01',
        sex: 'M',
        contact_email: ''
      })
      fetchPatients()
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to create patient')
    }
  }

  const filtered = patients.filter(p => 
    (p.patient_code?.toLowerCase().includes(search.toLowerCase())) ||
    (p.full_name?.toLowerCase().includes(search.toLowerCase())) ||
    (p.name?.toLowerCase().includes(search.toLowerCase()))
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Patients Directory</h1>
          <p className="text-sm text-slate-500">Registered ophthalmology patients and screening history</p>
        </div>
        
        <div className="flex w-full sm:w-auto space-x-3">
          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
            <input 
              type="text" 
              placeholder="Search by name or code..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
            />
          </div>
          <button 
            onClick={() => setShowModal(true)}
            className="flex-shrink-0 flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium shadow-sm transition"
          >
            <Plus className="w-4 h-4 mr-1.5" /> Add Patient
          </button>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-slate-500 bg-slate-50 uppercase border-b border-slate-200">
              <tr>
                <th className="px-6 py-4">Patient ID / Code</th>
                <th className="px-6 py-4">Full Name</th>
                <th className="px-6 py-4">Age / Sex</th>
                <th className="px-6 py-4">Last Screening</th>
                <th className="px-6 py-4">Latest AI Result</th>
                <th className="px-6 py-4">Review Status</th>
                <th className="px-6 py-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan="7" className="px-6 py-12 text-center text-slate-500">
                    Loading patients from database...
                  </td>
                </tr>
              ) : filtered.length > 0 ? (
                filtered.map(patient => (
                  <tr key={patient.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-6 py-4 font-mono font-medium text-slate-800">
                      {patient.patient_code}
                    </td>
                    <td className="px-6 py-4 font-semibold text-slate-900 flex items-center">
                      <UserCircle className="w-7 h-7 text-slate-400 mr-2.5 flex-shrink-0" />
                      {patient.full_name || patient.name || 'Unnamed Patient'}
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {patient.age ? `${patient.age}y` : 'N/A'} / {patient.sex || 'U'}
                    </td>
                    <td className="px-6 py-4 text-slate-600">
                      {patient.last_screening 
                        ? format(new Date(patient.last_screening), 'MMM d, yyyy') 
                        : <span className="text-slate-400 text-xs">No screenings</span>}
                    </td>
                    <td className="px-6 py-4">
                      {patient.latest_ai_result ? (
                        <span className="font-semibold text-primary-700">
                          {patient.latest_ai_result}
                        </span>
                      ) : (
                        <span className="text-slate-400 text-xs">None</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      {patient.review_status ? (
                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                          patient.review_status === 'APPROVED' 
                            ? 'bg-emerald-100 text-emerald-800' 
                            : patient.review_status === 'MODIFIED'
                            ? 'bg-amber-100 text-amber-800'
                            : patient.review_status === 'REJECTED'
                            ? 'bg-rose-100 text-rose-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {patient.review_status}
                        </span>
                      ) : (
                        <span className="text-slate-400 text-xs">&mdash;</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right whitespace-nowrap">
                      <Link 
                        to={`/patients/${patient.id}`}
                        className="inline-flex items-center justify-center px-3 py-1.5 bg-[#191919] hover:bg-[#222222] border border-[#292929] hover:border-[#81958D] text-[#F5F0E6] font-semibold rounded-lg transition text-xs whitespace-nowrap"
                      >
                        View File
                      </Link>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="7" className="px-6 py-12 text-center text-slate-400">
                    No patients found. Click &quot;Add Patient&quot; above to register a patient.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Patient Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="modal-surface rounded-xl p-6 max-w-md w-full space-y-4">
            <h3 className="text-lg font-bold text-white">Register New Patient</h3>
            <form onSubmit={handleCreatePatient} className="space-y-3 text-sm">
              <div>
                <label className="block font-medium text-slate-300 mb-1">Patient ID / Code *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. P-2026-001"
                  value={newPatient.patient_code}
                  onChange={e => setNewPatient({...newPatient, patient_code: e.target.value})}
                  className="w-full px-3 py-2 border rounded-lg text-white"
                />
              </div>
              <div>
                <label className="block font-medium text-slate-300 mb-1">Full Name *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Ramesh Patel"
                  value={newPatient.full_name}
                  onChange={e => setNewPatient({...newPatient, full_name: e.target.value})}
                  className="w-full px-3 py-2 border rounded-lg text-white"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-slate-300 mb-1">Date of Birth</label>
                  <input
                    type="date"
                    value={newPatient.date_of_birth}
                    onChange={e => setNewPatient({...newPatient, date_of_birth: e.target.value})}
                    className="w-full px-3 py-2 border rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block font-medium text-slate-300 mb-1">Sex</label>
                  <select
                    value={newPatient.sex}
                    onChange={e => setNewPatient({...newPatient, sex: e.target.value})}
                    className="w-full px-3 py-2 border rounded-lg text-white"
                  >
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block font-medium text-slate-300 mb-1">Contact Email</label>
                <input
                  type="email"
                  placeholder="patient@example.com"
                  value={newPatient.contact_email}
                  onChange={e => setNewPatient({...newPatient, contact_email: e.target.value})}
                  className="w-full px-3 py-2 border rounded-lg text-white"
                />
              </div>
              <div className="flex justify-end gap-2 pt-3 border-t border-slate-700/50">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 border border-slate-700 rounded-lg text-slate-300 hover:bg-slate-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 font-medium"
                >
                  Save Patient
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default PatientList
