import { useState, useEffect } from 'react'
import { getPendingReviews, createReview, getReviews } from '../api/reviews'
import { 
  CheckSquare, AlertCircle, Eye, CheckCircle2, Edit3, XCircle, 
  Calendar, User, FileText, ArrowLeft, Activity, ShieldCheck
} from 'lucide-react'
import { format } from 'date-fns'
import toast from 'react-hot-toast'

const ReviewQueue = () => {
  const [pendingList, setPendingList] = useState([])
  const [completedList, setCompletedList] = useState([])
  const [activeTab, setActiveTab] = useState('pending') // 'pending' | 'completed'
  const [loading, setLoading] = useState(true)
  const [selectedItem, setSelectedItem] = useState(null)
  
  // Clinician decision form state
  const [decision, setDecision] = useState('APPROVED') // 'APPROVED' | 'MODIFIED' | 'REJECTED'
  const [clinicianGrade, setClinicianGrade] = useState(0)
  const [finalDiagnosis, setFinalDiagnosis] = useState('')
  const [clinicianNotes, setClinicianNotes] = useState('')
  const [urgency, setUrgency] = useState('routine')
  const [submitting, setSubmitting] = useState(false)

  const severityLabels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [pendingRes, completedRes] = await Promise.all([
        getPendingReviews(),
        getReviews()
      ])
      const pend = Array.isArray(pendingRes) ? pendingRes : pendingRes.items || []
      const comp = Array.isArray(completedRes) ? completedRes : completedRes.items || []
      setPendingList(pend)
      setCompletedList(comp)
      if (pend.length > 0 && !selectedItem) {
        selectScreeningForReview(pend[0])
      }
    } catch (err) {
      console.error('Error fetching reviews:', err)
      toast.error('Failed to load review queue')
    } finally {
      setLoading(false)
    }
  }

  const [imageError, setImageError] = useState(false)

  const selectScreeningForReview = (item) => {
    setSelectedItem(item)
    setImageError(false)
    setDecision('APPROVED')
    setClinicianGrade(item.predicted_class ?? 0)
    setFinalDiagnosis(item.predicted_label || item.prediction || severityLabels[item.predicted_class ?? 0])
    setClinicianNotes('')
    setUrgency('routine')
  }

  const handleDecisionChange = (newDecision) => {
    setDecision(newDecision)
    if (newDecision === 'APPROVED' && selectedItem) {
      setClinicianGrade(selectedItem.predicted_class ?? 0)
      setFinalDiagnosis(selectedItem.predicted_label || severityLabels[selectedItem.predicted_class ?? 0])
    } else if (newDecision === 'REJECTED') {
      setFinalDiagnosis('Image Inconclusive / Rejected by Clinician')
    }
  }

  const handleGradeChange = (grade) => {
    setClinicianGrade(grade)
    setFinalDiagnosis(severityLabels[grade].split(' - ')[1])
  }

  const handleSubmitReview = async () => {
    if (!selectedItem || submitting) return
    setSubmitting(true)
    try {
      const savedReview = await createReview({
        prediction_id: selectedItem.id,
        agreed_with_ai: decision === 'APPROVED',
        decision: decision,
        clinician_grade: Number(clinicianGrade),
        clinician_notes: clinicianNotes,
        final_diagnosis: finalDiagnosis || selectedItem.predicted_label,
        requires_followup: urgency !== 'routine',
        followup_urgency: urgency
      })

      toast.success(`Screening ${selectedItem.id.substring(0, 8)} successfully ${decision.toLowerCase()}`)
      
      // Update local state by removing reviewed item and prepending to completed
      setPendingList(prev => prev.filter(p => p.id !== selectedItem.id))
      setCompletedList(prev => [savedReview, ...prev])
      setSelectedItem(null)
      
      // Sync from backend
      await loadData()
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to submit review'
      toast.error(msg)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Clinician Review Queue</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Human-in-the-loop diagnostic validation of AI-assisted retinal screenings
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center bg-slate-900/60 p-1 rounded-lg text-sm border border-slate-700/50">
          <button
            onClick={() => setActiveTab('pending')}
            className={`px-4 py-1.5 rounded-md font-semibold transition ${
              activeTab === 'pending'
                ? 'bg-primary-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Pending Review ({pendingList.length})
          </button>
          <button
            onClick={() => setActiveTab('completed')}
            className={`px-4 py-1.5 rounded-md font-semibold transition ${
              activeTab === 'completed'
                ? 'bg-primary-600 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Completed History ({completedList.length})
          </button>
        </div>
      </div>

      {/* Main Review Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: Screening List (Pending or Completed) */}
        <div className="lg:col-span-5 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col h-[750px]">
          <div className="p-4 bg-slate-50 border-b border-slate-200 flex justify-between items-center">
            <h3 className="font-semibold text-slate-800 text-sm">
              {activeTab === 'pending' ? 'Screenings Awaiting Doctor Sign-Off' : 'Validated Clinical Screenings'}
            </h3>
            <span className="text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full bg-primary-500/20 text-cyan-300 border border-primary-500/30">
              {activeTab === 'pending' ? pendingList.length : completedList.length} items
            </span>
          </div>

          <div className="overflow-y-auto flex-1 divide-y divide-slate-100">
            {loading ? (
              <div className="p-8 text-center text-sm text-slate-500">Loading screening queue...</div>
            ) : activeTab === 'pending' ? (
              pendingList.length === 0 ? (
                <div className="p-12 text-center text-slate-500">
                  <CheckSquare className="w-10 h-10 mx-auto text-emerald-400 mb-2" />
                  <p className="font-semibold text-slate-700">Review Queue is Empty</p>
                  <p className="text-xs text-slate-400 mt-1">All AI predictions have been reviewed. Run a New Screening to populate.</p>
                </div>
              ) : (
                pendingList.map(item => {
                  const isSelected = selectedItem?.id === item.id
                  return (
                    <div
                      key={item.id}
                      onClick={() => selectScreeningForReview(item)}
                      className={`p-4 cursor-pointer transition text-left border-l-4 ${
                        isSelected 
                          ? 'bg-primary-50/70 border-primary-600' 
                          : 'hover:bg-slate-50 border-transparent'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <span className="font-bold text-slate-900 text-sm">
                          {item.patient_code || item.patient_name || `Patient #${item.patient_id.substring(0, 8)}`}
                        </span>
                        <span className="text-[11px] px-2 py-0.5 bg-amber-100 text-amber-800 rounded font-semibold uppercase">
                          {item.review_status || 'PENDING'}
                        </span>
                      </div>

                      <div className="text-xs text-slate-500 mb-2 flex items-center gap-2">
                        <span>ID: {item.id.substring(0, 8)}</span>
                        <span>&bull;</span>
                        <span>{item.created_at ? format(new Date(item.created_at), 'MMM d, HH:mm') : 'Recent'}</span>
                      </div>

                      <div className="flex items-center justify-between text-xs">
                        <div className="text-slate-700">
                          AI: <strong className="text-primary-700">{item.predicted_label || item.prediction}</strong> (Class {item.predicted_class})
                        </div>
                        <span className="font-semibold text-slate-800">
                          {item.confidence ? `${(item.confidence * 100).toFixed(1)}%` : '---'}
                        </span>
                      </div>
                    </div>
                  )
                })
              )
            ) : (
              completedList.length === 0 ? (
                <div className="p-8 text-center text-sm text-slate-500">No completed reviews yet.</div>
              ) : (
                completedList.map(item => (
                  <div key={item.id} className="p-4 border-l-4 border-emerald-500 hover:bg-slate-50">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-semibold text-slate-900 text-sm">Screening {item.prediction_id.substring(0, 8)}</span>
                      <span className="text-[11px] px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded font-semibold">
                        {item.decision || (item.agreed_with_ai ? 'APPROVED' : 'MODIFIED')}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600">Diagnosis: <strong>{item.final_diagnosis}</strong></p>
                    <p className="text-[11px] text-slate-400 mt-1">
                      Reviewed on {format(new Date(item.review_timestamp), 'MMM d, yyyy HH:mm')}
                    </p>
                  </div>
                ))
              )
            )}
          </div>
        </div>

        {/* Right Side: Detailed Clinician Review Action Panel */}
        <div className="lg:col-span-7 bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between min-h-[750px]">
          {selectedItem ? (
            <div className="space-y-6">
              {/* Review Header */}
              <div className="flex justify-between items-start border-b pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-bold text-slate-800">Screening Clinical Validation</h2>
                    <span className="px-2 py-0.5 bg-amber-100 text-amber-800 text-xs font-semibold rounded">
                      PENDING REVIEW
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Screening ID: <code className="text-slate-700">{selectedItem.id}</code> &bull; Patient: <strong>{selectedItem.patient_code || selectedItem.patient_name || selectedItem.patient_id}</strong>
                  </p>
                </div>

                <div className="text-right text-xs text-slate-500">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5" />
                    {selectedItem.created_at ? format(new Date(selectedItem.created_at), 'MMMM d, yyyy - HH:mm') : 'N/A'}
                  </div>
                </div>
              </div>

              {/* Retinal Fundus & Grad-CAM Image Display */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-2">Original Retinal Fundus Image</p>
                  <div className="h-48 bg-black rounded-lg overflow-hidden flex items-center justify-center">
                    {selectedItem.image_url && !imageError ? (
                      <img 
                        src={selectedItem.image_url} 
                        alt="Fundus" 
                        className="w-full h-full object-contain" 
                        onError={() => setImageError(true)}
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-rose-400 p-4 text-center">
                        <AlertCircle className="w-8 h-8 mb-1.5 text-rose-500" />
                        <span className="text-xs font-semibold text-rose-300">Image File Unavailable</span>
                        <span className="text-[10px] text-slate-400 mt-0.5">
                          Original photograph missing or unreadable. Clinical approval blocked.
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-2">Grad-CAM Lesion Heatmap</p>
                  <div className="h-48 bg-black rounded-lg overflow-hidden flex items-center justify-center">
                    {selectedItem.gradcam_url ? (
                      <img src={selectedItem.gradcam_url} alt="Grad-CAM" className="w-full h-full object-contain" />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-slate-500 p-4 text-center">
                        <Eye className="w-6 h-6 mb-1 text-slate-600" />
                        <span className="text-xs text-slate-400">Heatmap Not Computed</span>
                        <span className="text-[10px] text-slate-500">Overlay generation requires verified original image</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* AI Diagnostic Prediction Breakdown */}
              <div className="p-4 bg-primary-50/50 border border-primary-100 rounded-xl space-y-3">
                <div className="flex justify-between items-center">
                  <div>
                    <span className="text-xs text-slate-500 uppercase font-semibold">AI Screening Prediction:</span>
                    <h3 className="text-xl font-extrabold text-primary-800">
                      {selectedItem.predicted_label || selectedItem.prediction} (Class {selectedItem.predicted_class})
                    </h3>
                  </div>
                  <div className="text-right">
                    <span className="text-xs text-slate-500 uppercase font-semibold">Model Confidence:</span>
                    <div className="text-2xl font-bold text-primary-700">
                      {selectedItem.confidence ? `${(selectedItem.confidence * 100).toFixed(1)}%` : '---'}
                    </div>
                  </div>
                </div>

                {/* Probabilities Row */}
                {selectedItem.probabilities && (
                  <div className="grid grid-cols-5 gap-2 pt-2 border-t border-primary-100 text-xs text-center">
                    {Object.entries(selectedItem.probabilities).map(([name, prob]) => (
                      <div key={name} className="bg-white p-1.5 rounded shadow-2xs">
                        <div className="text-[11px] text-slate-500 truncate">{name}</div>
                        <strong className="text-slate-800">{(prob * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Clinician Action Buttons: Approve, Modify, Reject */}
              <div className="space-y-4 pt-2">
                <label className="text-sm font-semibold text-slate-800 block">Clinician Decision Actions</label>
                <div className="grid grid-cols-3 gap-3">
                  <button
                    type="button"
                    disabled={imageError || !selectedItem.image_url}
                    onClick={() => handleDecisionChange('APPROVED')}
                    className={`py-3 px-4 rounded-xl border flex flex-col items-center justify-center text-sm font-semibold transition ${
                      imageError || !selectedItem.image_url
                        ? 'opacity-40 cursor-not-allowed bg-slate-100 text-slate-400 border-slate-200'
                        : decision === 'APPROVED'
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm'
                        : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                    }`}
                    title={imageError || !selectedItem.image_url ? 'Cannot approve AI prediction without verifiable retinal fundus image' : 'Approve AI Prediction'}
                  >
                    <CheckCircle2 className="w-5 h-5 mb-1" />
                    Approve AI Prediction
                  </button>

                  <button
                    type="button"
                    onClick={() => handleDecisionChange('MODIFIED')}
                    className={`py-3 px-4 rounded-xl border flex flex-col items-center justify-center text-sm font-semibold transition ${
                      decision === 'MODIFIED'
                        ? 'bg-amber-600 text-white border-amber-600 shadow-sm'
                        : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <Edit3 className="w-5 h-5 mb-1" />
                    Modify Prediction
                  </button>

                  <button
                    type="button"
                    onClick={() => handleDecisionChange('REJECTED')}
                    className={`py-3 px-4 rounded-xl border flex flex-col items-center justify-center text-sm font-semibold transition ${
                      decision === 'REJECTED'
                        ? 'bg-rose-600 text-white border-rose-600 shadow-sm'
                        : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <XCircle className="w-5 h-5 mb-1" />
                    Reject / Flag
                  </button>
                </div>

                {/* If Modified: Select Clinician Severity Grade */}
                {decision === 'MODIFIED' && (
                  <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl space-y-2">
                    <label className="block text-xs font-semibold text-amber-900 uppercase">
                      Select Final Clinician Severity Grade
                    </label>
                    <div className="grid grid-cols-5 gap-2 text-xs">
                      {[0, 1, 2, 3, 4].map(grade => (
                        <button
                          key={grade}
                          type="button"
                          onClick={() => handleGradeChange(grade)}
                          className={`p-2 rounded-lg font-medium border text-center transition ${
                            clinicianGrade === grade
                              ? 'bg-amber-600 text-white border-amber-600'
                              : 'bg-white text-slate-700 border-slate-200 hover:bg-amber-100'
                          }`}
                        >
                          <div>Class {grade}</div>
                          <div className="text-[10px] mt-0.5 truncate">{severityLabels[grade].split(' - ')[1]}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Form fields: Final Diagnosis & Notes */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">Final Clinician Diagnosis *</label>
                    <input
                      type="text"
                      value={finalDiagnosis}
                      onChange={e => setFinalDiagnosis(e.target.value)}
                      placeholder="e.g. Moderate Non-Proliferative DR"
                      className="w-full px-3 py-2 border rounded-lg text-sm text-slate-800"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-700 mb-1">Follow-Up Urgency</label>
                    <select
                      value={urgency}
                      onChange={e => setUrgency(e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg text-sm text-slate-800"
                    >
                      <option value="routine">Routine Follow-up (12 months)</option>
                      <option value="urgent">Urgent Referral (1-3 months)</option>
                      <option value="emergency">Emergency Immediate Consultation</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">Clinician Notes & Comments</label>
                  <textarea
                    rows={2}
                    value={clinicianNotes}
                    onChange={e => setClinicianNotes(e.target.value)}
                    placeholder="Document clinical observations, macular edema status, or referral instructions..."
                    className="w-full px-3 py-2 border rounded-lg text-sm text-slate-800"
                  />
                </div>
              </div>

              {/* Submit Final Review */}
              <div className="pt-4 border-t flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setSelectedItem(null)}
                  className="px-4 py-2 border rounded-lg text-slate-600 text-sm hover:bg-slate-50"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmitReview}
                  disabled={submitting}
                  className="px-6 py-2.5 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 text-sm shadow-sm"
                >
                  {submitting ? 'Committing Review...' : 'Commit Clinical Review Decision'}
                </button>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 p-8">
              <Eye className="w-12 h-12 text-slate-300 mb-3" />
              <h4 className="text-base font-semibold text-slate-600">No Screening Selected</h4>
              <p className="text-xs text-slate-400 mt-1 text-center max-w-sm">
                Select a pending screening record from the left panel to review fundus images, Grad-CAM activation, and submit clinician validation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ReviewQueue
