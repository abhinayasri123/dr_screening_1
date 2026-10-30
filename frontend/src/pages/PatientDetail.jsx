import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getPatient } from '../api/patients'
import { getPatientPredictions } from '../api/predictions'
import { UserCircle, FilePlus, Activity, Calendar, FileImage, CheckCircle2, AlertCircle } from 'lucide-react'
import { format } from 'date-fns'

const PatientDetail = () => {
  const { id } = useParams()
  const [patient, setPatient] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPatientData()
  }, [id])

  const fetchPatientData = async () => {
    setLoading(true)
    try {
      const [patRes, predRes] = await Promise.all([
        getPatient(id),
        getPatientPredictions(id)
      ])
      setPatient(patRes)
      const pList = Array.isArray(predRes) ? predRes : predRes.items || []
      setPredictions(pList)
    } catch (err) {
      console.error('Error fetching patient data:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="p-8 text-center text-slate-500 text-sm">Loading patient file...</div>
  }

  if (!patient) {
    return (
      <div className="p-8 text-center space-y-4">
        <p className="text-slate-600">Patient not found</p>
        <Link to="/patients" className="text-primary-600 underline text-sm">&larr; Back to Patients</Link>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Link to="/patients" className="text-xs text-primary-600 hover:underline">&larr; Patients Directory</Link>
          </div>
          <h1 className="text-2xl font-bold text-slate-800 mt-1">
            {patient.full_name || patient.name || 'Patient File'}
          </h1>
          <p className="text-xs text-slate-500">ID / Code: <code className="font-semibold text-slate-700">{patient.patient_code}</code></p>
        </div>

        <Link 
          to="/screening/new" 
          state={{ patientId: id }}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center text-sm font-medium shadow-sm transition"
        >
          <FilePlus className="w-4 h-4 mr-2" /> Run New Screening
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Patient Profile Information */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <div className="flex items-center mb-5">
              <UserCircle className="w-14 h-14 text-slate-300 mr-3 flex-shrink-0" />
              <div>
                <h2 className="text-lg font-bold text-slate-900 leading-tight">
                  {patient.full_name || patient.name}
                </h2>
                <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-mono mt-1 inline-block">
                  {patient.patient_code}
                </span>
              </div>
            </div>
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Date of Birth</span>
                <span className="font-medium text-slate-900">
                  {patient.date_of_birth ? format(new Date(patient.date_of_birth), 'MMM d, yyyy') : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Age / Sex</span>
                <span className="font-medium text-slate-900">
                  {patient.age ? `${patient.age} years` : 'N/A'} ({patient.sex || 'U'})
                </span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Email Contact</span>
                <span className="font-medium text-slate-900 truncate max-w-[150px]">
                  {patient.contact_email || 'None on file'}
                </span>
              </div>
              <div className="flex justify-between py-2 border-b border-slate-100">
                <span className="text-slate-500">Total Screenings</span>
                <span className="font-bold text-primary-700">
                  {predictions.length}
                </span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-slate-500">Registered On</span>
                <span className="font-medium text-slate-900">
                  {patient.created_at ? format(new Date(patient.created_at), 'MMM d, yyyy') : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Longitudinal Screening History */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="p-5 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
              <h3 className="text-base font-bold text-slate-800 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-primary-600" /> Longitudinal Screening History
              </h3>
              <span className="text-xs bg-white border border-slate-200 text-slate-600 px-2.5 py-0.5 rounded-full font-medium">
                {predictions.length} Records
              </span>
            </div>
            
            <div className="divide-y divide-slate-100">
              {predictions.length > 0 ? (
                predictions.map(pred => {
                  const isApproved = pred.review_status === 'APPROVED'
                  const isModified = pred.review_status === 'MODIFIED'
                  const isRejected = pred.review_status === 'REJECTED'
                  const isPending = !pred.review_status || pred.review_status === 'PENDING'

                  return (
                    <div key={pred.id} className="p-5 hover:bg-slate-50 transition flex flex-col sm:flex-row gap-4 items-start">
                      {/* Fundus Thumbnail */}
                      <div className="w-24 h-24 bg-black rounded-lg overflow-hidden flex-shrink-0 flex items-center justify-center">
                        {pred.image_url ? (
                          <img src={pred.image_url} alt="Fundus" className="w-full h-full object-cover" />
                        ) : (
                          <FileImage className="w-8 h-8 text-slate-500" />
                        )}
                      </div>

                      {/* Prediction Metadata */}
                      <div className="flex-1 space-y-1">
                        <div className="flex justify-between items-start">
                          <div>
                            <span className="text-xs text-slate-500 flex items-center gap-1">
                              <Calendar className="w-3.5 h-3.5" />
                              {pred.created_at ? format(new Date(pred.created_at), 'MMM d, yyyy - HH:mm') : 'N/A'}
                            </span>
                            <h4 className="text-base font-bold text-slate-900 mt-1">
                              AI Result: <span className="text-primary-700">{pred.predicted_label || pred.prediction}</span>
                            </h4>
                          </div>

                          <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                            isApproved 
                              ? 'bg-emerald-100 text-emerald-800' 
                              : isModified 
                              ? 'bg-amber-100 text-amber-800' 
                              : isRejected 
                              ? 'bg-rose-100 text-rose-800' 
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {pred.review_status || 'PENDING'}
                          </span>
                        </div>

                        <div className="text-xs text-slate-600 flex items-center gap-4 pt-1">
                          <span>
                            Confidence: <strong>{pred.confidence ? `${(pred.confidence * 100).toFixed(1)}%` : '---'}</strong>
                          </span>
                          <span>&bull;</span>
                          <span>
                            Severity: Class <strong>{pred.predicted_class}</strong> of 4
                          </span>
                          <span>&bull;</span>
                          <span>
                            Uncertainty: <strong>{pred.uncertainty_level || 'Low'}</strong>
                          </span>
                        </div>

                        <div className="text-[11px] text-slate-400 pt-1">
                          Screening ID: <code className="font-mono text-slate-500">{pred.id}</code>
                        </div>
                      </div>
                    </div>
                  )
                })
              ) : (
                <div className="p-12 text-center text-slate-400 text-sm">
                  <Activity className="w-8 h-8 mx-auto text-slate-300 mb-2" />
                  <p className="font-medium text-slate-600">No screening records available.</p>
                  <p className="text-xs text-slate-400 mt-1">Click &quot;Run New Screening&quot; above to perform this patient&apos;s first AI retinal screening.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PatientDetail
