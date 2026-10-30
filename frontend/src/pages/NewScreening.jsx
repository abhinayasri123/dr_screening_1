import { useState, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, FileImage, Activity, Check, User, PlusCircle, AlertCircle, FileCheck, Layers, Eye } from 'lucide-react'
import EHRForm from '../components/EHRForm'
import PredictionResult from '../components/PredictionResult'
import GradCAMViewer from '../components/GradCAMViewer'
import { createPrediction } from '../api/predictions'
import { getPatients, createPatient, getPatientEHR, addEHRRecord } from '../api/patients'
import toast from 'react-hot-toast'
import { useNavigate, useLocation } from 'react-router-dom'

const NewScreening = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const [step, setStep] = useState(1)
  const [patients, setPatients] = useState([])
  const [selectedPatientId, setSelectedPatientId] = useState(location.state?.patientId || '')
  const [showNewPatientModal, setShowNewPatientModal] = useState(false)
  const [newPatientData, setNewPatientData] = useState({
    patient_code: '',
    full_name: '',
    date_of_birth: '1975-06-15',
    sex: 'M',
    contact_email: ''
  })

  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [invalidImageError, setInvalidImageError] = useState(null)
  const [ehrData, setEhrData] = useState({})
  const [hasExistingEhrRecord, setHasExistingEhrRecord] = useState(false)
  const [isSavingEhr, setIsSavingEhr] = useState(false)
  const [mode, setMode] = useState('image_only')
  
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  // Fetch patients on mount
  useEffect(() => {
    fetchPatients()
  }, [])

  const fetchPatients = async () => {
    try {
      const res = await getPatients()
      const list = Array.isArray(res) ? res : res.items || []
      setPatients(list)
      if (!selectedPatientId && list.length > 0) {
        setSelectedPatientId(list[0].id)
      }
    } catch (err) {
      console.error('Error fetching patients:', err)
    }
  }

  // Fetch patient EHR when selected patient changes
  useEffect(() => {
    if (selectedPatientId) {
      fetchPatientEHR(selectedPatientId)
    }
  }, [selectedPatientId, patients])

  const fetchPatientEHR = async (patientId) => {
    if (!patientId) return
    try {
      const ehr = await getPatientEHR(patientId)
      if (ehr && (ehr.hba1c !== undefined || ehr.age !== undefined)) {
        setEhrData({
          age: ehr.age,
          sex: ehr.sex !== undefined ? String(ehr.sex) : '1',
          diabetes_duration_years: ehr.diabetes_duration_years,
          hba1c: ehr.hba1c,
          systolic_bp: ehr.systolic_bp,
          diastolic_bp: ehr.diastolic_bp,
          bmi: ehr.bmi,
          smoking_status: ehr.smoking_status !== undefined ? String(ehr.smoking_status) : '0',
          insulin_use: Boolean(ehr.insulin_use),
          prev_laser_treatment: Boolean(ehr.prev_laser_treatment),
          nephropathy: Boolean(ehr.nephropathy),
          neuropathy: Boolean(ehr.neuropathy),
          cholesterol_ldl: ehr.cholesterol_ldl,
        })
        setHasExistingEhrRecord(true)
      } else {
        const curr = patients.find(p => p.id === patientId)
        setEhrData(prev => ({
          ...prev,
          age: curr?.age || '',
          sex: curr?.sex === 'F' ? '0' : '1',
        }))
        setHasExistingEhrRecord(false)
      }
    } catch (err) {
      console.warn('Could not fetch patient EHR:', err)
      setHasExistingEhrRecord(false)
    }
  }

  const handleSaveEHRToDatabase = async () => {
    if (!selectedPatientId) {
      toast.error('Please select a patient first')
      return
    }
    setIsSavingEhr(true)
    try {
      await addEHRRecord(selectedPatientId, ehrData)
      toast.success('Patient EHR record successfully saved/updated')
      setHasExistingEhrRecord(true)
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to save EHR record')
    } finally {
      setIsSavingEhr(false)
    }
  }

  const handleCreatePatientInline = async (e) => {
    e.preventDefault()
    if (!newPatientData.patient_code || !newPatientData.full_name) {
      toast.error('Patient code and full name are required')
      return
    }
    try {
      const created = await createPatient(newPatientData)
      toast.success(`Patient ${created.patient_code} registered`)
      setShowNewPatientModal(false)
      await fetchPatients()
      setSelectedPatientId(created.id)
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to create patient')
    }
  }

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0]
    if (file) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
      setInvalidImageError(null)
      setResult(null)
      setStep(2)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    maxFiles: 1
  })

  const handlePredict = async () => {
    if (!image) {
      toast.error('Please upload a retinal fundus image')
      return
    }

    if (mode === 'multimodal') {
      if (!ehrData.hba1c || !ehrData.systolic_bp || !ehrData.diastolic_bp || !ehrData.diabetes_duration_years) {
        toast.error('Please fill in the required Clinical EHR fields (HbA1c, Blood Pressure, Duration) for Multimodal inference')
        return
      }
    }

    setLoading(true)
    setInvalidImageError(null)
    try {
      const formData = new FormData()
      formData.append('file', image)
      if (selectedPatientId) {
        formData.append('patient_id', selectedPatientId)
      }
      formData.append('mode', mode)
      if (mode === 'multimodal') {
        formData.append('ehr_data', JSON.stringify(ehrData))
      }

      const res = await createPrediction(formData)
      setResult(res)
      toast.success(`Prediction generated: ${res.prediction || res.predicted_label}`)
      setStep(4) // Move to results step
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Failed to generate prediction'
      if (err.response?.status === 422 || (typeof msg === 'string' && msg.includes('INVALID IMAGE TYPE'))) {
        setInvalidImageError(
          "INVALID IMAGE TYPE\n\nThis image does not appear to be a suitable retinal fundus photograph.\n\nPlease upload a clear retinal fundus image.\n\nNo diabetic retinopathy prediction has been generated."
        )
        setResult(null)
      } else {
        toast.error(typeof msg === 'string' ? msg : 'Inference error')
      }
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmitForReview = () => {
    if (!result) return
    toast.success('Screening successfully logged in Clinical Review Queue (Status: PENDING)')
    navigate('/reviews')
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">New DR Screening</h1>
          <p className="text-sm font-medium text-slate-400">Multimodal Retinal Fundus Classification & Clinical Decision Support</p>
        </div>

      {/* Step indicator */}
      <div className="flex items-center space-x-2">
        {['Patient', 'Screening Mode', mode === 'multimodal' ? 'Clinical EHR' : null, 'Fundus Image', 'Analysis']
          .filter(Boolean)
          .map((name, i) => (
            <div key={i} className="flex items-center">
              <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                step === i + 1 
                  ? 'bg-[#E8E1D5] text-[#111111] font-semibold' 
                  : step > i + 1 
                  ? 'bg-[#81958D]/20 text-[#A8C4B8] border border-[#81958D]/30' 
                  : 'bg-[#191919] text-[#85817A] border border-[#292929]'
              }`}>
                {i + 1}. {name}
              </span>
              {i < (mode === 'multimodal' ? 4 : 3) && <span className="mx-1 text-[#85817A]">&rarr;</span>}
            </div>
          ))}
      </div>
      </div>

      {/* Main Screening Workflow */}
      {step < 4 && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-6">
            {/* 1. Patient Selector */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
                  <User className="w-4 h-4 text-primary-600" />
                  1. Select Registered Patient
                </label>
                <button
                  type="button"
                  onClick={() => setShowNewPatientModal(true)}
                  className="text-xs text-primary-600 hover:text-primary-800 font-medium flex items-center gap-1"
                >
                  <PlusCircle className="w-3.5 h-3.5" /> + New Patient
                </button>
              </div>
              
              {patients.length > 0 ? (
                <select
                  value={selectedPatientId}
                  onChange={(e) => setSelectedPatientId(e.target.value)}
                  className="w-full px-3 py-2.5 border border-slate-300 rounded-lg text-sm text-slate-800 focus:ring-2 focus:ring-primary-500 font-medium"
                >
                  {patients.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.patient_code} - {p.name || p.full_name} ({p.sex}, Age: {p.age || 'N/A'})
                    </option>
                  ))}
                </select>
              ) : (
                <div className="p-3 bg-slate-50 rounded-lg text-xs text-slate-500 flex justify-between items-center">
                  <span>No patients found in database. Create one to proceed:</span>
                  <button
                    onClick={() => setShowNewPatientModal(true)}
                    className="px-2 py-1 bg-primary-600 text-white rounded text-xs"
                  >
                    Create Patient
                  </button>
                </div>
              )}
            </div>

            {/* 2. Screening Mode Selector */}
            <div>
              <label className="text-sm font-semibold text-slate-700 flex items-center gap-1.5 mb-2">
                <Layers className="w-4 h-4 text-primary-600" />
                2. Select Screening Mode
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setMode('image_only')}
                  className={`p-3.5 rounded-xl border text-left text-sm transition ${
                    mode === 'image_only' 
                      ? 'border-primary-600 bg-primary-50/70 text-primary-900 shadow-sm ring-1 ring-primary-500' 
                      : 'border-slate-200 hover:bg-slate-50 text-slate-700'
                  }`}
                >
                  <div className="font-semibold flex items-center gap-1.5">
                    <Eye className="w-4 h-4 text-primary-600" />
                    Radiographic CNN (Image Only)
                  </div>
                  <div className="text-xs text-slate-500 font-normal mt-1">
                    Primary Retinal Fundus Photography Analysis using EfficientNet-B3.
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => setMode('multimodal')}
                  className={`p-3.5 rounded-xl border text-left text-sm transition ${
                    mode === 'multimodal' 
                      ? 'border-primary-600 bg-primary-50/70 text-primary-900 shadow-sm ring-1 ring-primary-500' 
                      : 'border-slate-200 hover:bg-slate-50 text-slate-700'
                  }`}
                >
                  <div className="font-semibold flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-primary-600" />
                    Multimodal Fusion (CNN + EHR)
                  </div>
                  <div className="text-xs text-slate-500 font-normal mt-1">
                    Fuses retinal photography with patient EHR covariates via Attention-Gated Fusion.
                  </div>
                </button>
              </div>
            </div>
          </div>

          {/* 3. Clinical EHR Form (Dynamically shown below patient/mode selection when multimodal) */}
          {mode === 'multimodal' && (
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-primary-600" />
                  3. Clinical Risk Covariates (EHR)
                </label>
              </div>
              <EHRForm 
                formData={ehrData} 
                onChange={setEhrData} 
                hasExistingRecord={hasExistingEhrRecord}
                onSaveToRecord={handleSaveEHRToDatabase}
                isSaving={isSavingEhr}
              />
            </div>
          )}

          {/* 4. Retinal Image Upload Area */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-4">
            <label className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
              <UploadCloud className="w-4 h-4 text-primary-600" />
              {mode === 'multimodal' ? '4. Retinal Fundus Photograph' : '3. Retinal Fundus Photograph'}
            </label>

            {/* Stage A Invalid Non-Retinal Rejection Alert */}
            {invalidImageError && (
              <div className="p-5 rounded-xl bg-[#191919] border border-rose-500/40 text-left animate-in fade-in duration-200">
                <div className="flex items-start space-x-3.5">
                  <AlertCircle className="w-5 h-5 text-[#F87171] flex-shrink-0 mt-0.5" />
                  <div className="space-y-1.5">
                    <h4 className="text-sm font-bold text-[#F87171] tracking-wider uppercase">
                      Invalid Image Type
                    </h4>
                    <p className="text-sm font-semibold text-[#F5F0E6]">
                      This image does not appear to be a suitable retinal fundus photograph.
                    </p>
                    <p className="text-xs text-[#D0CCC3]">
                      Please upload a clear retinal fundus image.
                    </p>
                    <div className="pt-2 border-t border-[#292929] text-xs font-semibold text-[#F87171]">
                      No diabetic retinopathy prediction has been generated.
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition ${
                invalidImageError ? 'border-rose-500/50 bg-rose-950/20' : isDragActive ? 'border-[#81958D] bg-[#191919]' : 'border-[#292929] bg-[#0F0F0F] hover:border-[#81958D]/40'
              }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <div className="flex flex-col items-center">
                  <img src={preview} alt="Fundus preview" className="max-h-56 rounded-lg border border-[#292929] mb-3 object-contain bg-black" />
                  <p className="text-xs text-[#85817A] font-medium">{image?.name} (Click or drag to change image)</p>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <UploadCloud className="w-12 h-12 text-[#85817A] mb-3" />
                  <p className="text-sm font-medium text-[#F5F0E6]">Drag & drop retinal fundus image here, or click to browse</p>
                  <p className="text-xs text-[#85817A] mt-1">Supports PNG, JPG, JPEG (APTOS / IDRiD format)</p>
                </div>
              )}
            </div>

            {/* Run AI Screening Action */}
            <div className="pt-3 flex justify-end">
              <button
                onClick={handlePredict}
                disabled={loading || !image}
                className="px-6 py-2.5 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center shadow-sm text-sm transition"
              >
                {loading ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-spin" />
                    {mode === 'multimodal' ? 'Running Multimodal Fusion Inference...' : 'Running CNN Inference...'}
                  </>
                ) : (
                  <>
                    <Check className="w-4 h-4 mr-2" />
                    Run AI Screening ({mode === 'multimodal' ? 'Multimodal' : 'Image Only'})
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Step 4: Results Display */}
      {step === 4 && result && (
        <div className="space-y-6">
          <PredictionResult prediction={result} onSubmitReview={handleSubmitForReview} />

          {/* Grad-CAM Viewer */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4">Explainable AI: Retinal Activation Heatmap (Grad-CAM)</h3>
            <GradCAMViewer 
              originalImage={result.image_url || preview} 
              gradcamImage={result.gradcam_url} 
            />
          </div>

          <div className="flex justify-between items-center bg-white p-4 rounded-xl border border-slate-200">
            <button
              onClick={() => { setStep(1); setImage(null); setPreview(null); setResult(null); }}
              className="px-4 py-2 border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 text-sm font-medium"
            >
              Start Another Screening
            </button>
            <button
              onClick={handleSubmitForReview}
              className="flex items-center px-5 py-2.5 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 shadow-sm text-sm"
            >
              <FileCheck className="w-4 h-4 mr-2" />
              Submit for Clinical Review &rarr;
            </button>
          </div>
        </div>
      )}

      {/* Inline New Patient Modal */}
      {/* New Patient Inline Modal */}
      {showNewPatientModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-card border border-slate-700/60 p-6 max-w-md w-full space-y-4 shadow-2xl">
            <h3 className="text-lg font-bold text-white">Add New Patient</h3>
            <form onSubmit={handleCreatePatientInline} className="space-y-3 text-sm">
              <div>
                <label className="block font-medium text-slate-300 mb-1">Patient ID / Code *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. P-2026-042"
                  value={newPatientData.patient_code}
                  onChange={e => setNewPatientData({...newPatientData, patient_code: e.target.value})}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block font-medium text-slate-300 mb-1">Full Name *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Eleanor Vance"
                  value={newPatientData.full_name}
                  onChange={e => setNewPatientData({...newPatientData, full_name: e.target.value})}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-slate-300 mb-1">Date of Birth</label>
                  <input
                    type="date"
                    value={newPatientData.date_of_birth}
                    onChange={e => setNewPatientData({...newPatientData, date_of_birth: e.target.value})}
                    className="w-full px-3 py-2 border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block font-medium text-slate-300 mb-1">Sex</label>
                  <select
                    value={newPatientData.sex}
                    onChange={e => setNewPatientData({...newPatientData, sex: e.target.value})}
                    className="w-full px-3 py-2 border rounded-lg"
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
                  value={newPatientData.contact_email}
                  onChange={e => setNewPatientData({...newPatientData, contact_email: e.target.value})}
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div className="flex justify-end gap-2 pt-3 border-t border-slate-700/60">
                <button
                  type="button"
                  onClick={() => setShowNewPatientModal(false)}
                  className="px-4 py-2 border border-slate-700 rounded-lg text-slate-300 hover:bg-slate-800/60 font-medium"
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

export default NewScreening
