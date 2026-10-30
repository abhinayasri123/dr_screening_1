import React, { useState, useEffect } from 'react'
import { get, post } from '../api/client'
import { Eye, Sparkles, AlertTriangle, CheckCircle, RefreshCw, Layers, Info } from 'lucide-react'

const CLASS_NAMES = [
  '0 - No Apparent Diabetic Retinopathy',
  '1 - Mild Non-Proliferative DR (Microaneurysms)',
  '2 - Moderate Non-Proliferative DR',
  '3 - Severe Non-Proliferative DR',
  '4 - Proliferative Diabetic Retinopathy'
]

const Explainability = () => {
  const [screenings, setScreenings] = useState([])
  const [selectedCase, setSelectedCase] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [generatingGradcam, setGeneratingGradcam] = useState(false)
  const [targetClass, setTargetClass] = useState(null)
  const [fundusImgError, setFundusImgError] = useState(false)

  useEffect(() => {
    fetchScreenings()
  }, [])

  const fetchScreenings = async () => {
    try {
      setLoading(true)
      setError(null)
      let items = []
      try {
        const xaiRes = await get('/screenings/xai/cases?limit=50')
        items = Array.isArray(xaiRes) ? xaiRes : (xaiRes.items || [])
      } catch (xaiErr) {
        console.warn('Falling back to /screenings', xaiErr)
        const scRes = await get('/screenings?limit=50&has_image=true')
        items = Array.isArray(scRes) ? scRes : (scRes.items || [])
      }

      setScreenings(items)
      if (items.length > 0) {
        // Prioritize cases that already have Grad-CAM computed, or valid predictions
        const preferred = items.find(i => i.has_gradcam) || items.find(i => i.predicted_grade !== null) || items[0]
        await loadEncounterDetails(preferred.id)
      } else {
        setSelectedCase(null)
        setLoading(false)
      }
    } catch (err) {
      console.error('Failed to load screening cases for XAI review:', err)
      const msg = err.response?.data?.detail || err.message || 'Failed to load screening cases for XAI review'
      setError(msg)
      setLoading(false)
    }
  }

  const loadEncounterDetails = async (encounterId) => {
    if (!encounterId) return
    try {
      setLoading(true)
      setError(null)
      setFundusImgError(false)
      const data = await get(`/screenings/${encounterId}`)
      setSelectedCase(data)
      setTargetClass(data.prediction?.predicted_class ?? 2)
    } catch (err) {
      console.error('Failed to load encounter XAI details:', err)
      const msg = err.response?.data?.detail || err.message || 'Failed to load encounter XAI details'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateGradcam = async (customClass) => {
    if (!selectedCase?.encounter?.id) return
    const cls = customClass !== undefined ? customClass : targetClass
    try {
      setGeneratingGradcam(true)
      const url = `/screenings/${selectedCase.encounter.id}/gradcam${cls !== null && cls !== undefined ? `?target_class=${cls}` : ''}`
      const res = await post(url, {})
      
      // Update local state with newly computed gradcam_url
      setSelectedCase(prev => ({
        ...prev,
        prediction: {
          ...(prev?.prediction || {}),
          gradcam_url: res.gradcam_url,
          predicted_label: res.class_name || prev?.prediction?.predicted_label
        }
      }))
    } catch (err) {
      alert(err.response?.data?.detail || err.message || 'Failed to generate Grad-CAM for this encounter.')
    } finally {
      setGeneratingGradcam(false)
    }
  }

  const prediction = selectedCase?.prediction
  const image = selectedCase?.images?.[0]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 flex items-center">
            <Eye className="w-6 h-6 mr-2 text-primary-600" />
            Explainable AI (XAI) & Grad-CAM Visualizations
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Convolutional Class Activation Mapping (Grad-CAM) and Saliency Explanations for EfficientNet-B3
          </p>
        </div>
        <button
          onClick={fetchScreenings}
          className="mt-3 md:mt-0 flex items-center px-3 py-2 text-xs font-semibold bg-white border border-slate-300 rounded shadow-sm hover:bg-slate-50 text-slate-700"
        >
          <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
          Refresh Cases
        </button>
      </div>

      {/* Clinical Informational Note */}
      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-slate-600 text-xs leading-relaxed flex items-center space-x-2.5">
        <Info className="w-4 h-4 text-teal-600 flex-shrink-0" />
        <div>
          <span className="font-semibold text-slate-700">Clinical Reference: </span>
          Grad-CAM visual overlays highlight spatial feature correlations influencing the CNN's convolutional features. Heatmaps are decision-support references and must not be used independently for clinical diagnosis or treatment planning.
        </div>
      </div>

      {/* Error Banner with Retry */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-lg text-rose-800 text-sm flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-rose-500 flex-shrink-0" />
            <span><strong>System Notice:</strong> {error}</span>
          </div>
          <button
            onClick={fetchScreenings}
            className="px-3 py-1.5 bg-rose-600 text-white rounded text-xs font-semibold hover:bg-rose-700 inline-flex items-center space-x-1"
          >
            <RefreshCw className="w-3.5 h-3.5 mr-1" /> Retry
          </button>
        </div>
      )}

      {/* Case Selector */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row items-center gap-4">
        <label className="text-xs font-bold text-slate-700 whitespace-nowrap">
          Select Screening Case:
        </label>
        <select
          className="w-full md:w-96 text-sm border border-slate-300 rounded-lg p-2 bg-slate-50 focus:ring-2 focus:ring-primary-500 font-medium"
          value={selectedCase?.encounter?.id || ''}
          onChange={(e) => loadEncounterDetails(e.target.value)}
          disabled={loading || screenings.length === 0}
        >
          {screenings.length === 0 ? (
            <option value="">No fundus imaging cases available</option>
          ) : (
            screenings.map((sc) => (
              <option key={sc.id} value={sc.id}>
                {sc.case_number} — {sc.patient_name || sc.patient_code} [{sc.predicted_label || sc.status}]
              </option>
            ))
          )}
        </select>
        {selectedCase && (
          <span className="text-xs text-slate-500 ml-auto font-medium">
            Encounter Date: {new Date(selectedCase.encounter.encounter_date).toLocaleDateString()}
          </span>
        )}
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-500 bg-white rounded-xl border border-slate-200">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto text-primary-500 mb-3" />
          Loading Grad-CAM saliency maps...
        </div>
      ) : !selectedCase ? (
        <div className="p-8 text-center text-slate-500 bg-white rounded-xl border border-slate-200 space-y-2">
          <Eye className="w-10 h-10 text-slate-400 mx-auto mb-2" />
          <p className="font-semibold text-slate-700">No screening cases with fundus images available yet.</p>
          <p className="text-xs text-slate-500">Register a screening encounter or upload a fundus photograph to view explainability maps.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Side-by-side Images (2 cols) */}
          <div className="lg:col-span-2 bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <h2 className="text-base font-bold text-slate-800 flex items-center justify-between">
              <span>Dual-View Retinal Fundus & Grad-CAM Heatmap</span>
              <span className="text-xs font-normal text-slate-500 bg-slate-100 px-2 py-1 rounded">
                Target Layer: EfficientNet-B3.features.8
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original Fundus */}
              <div className="border border-slate-200 rounded-lg p-3 bg-slate-900 text-center">
                <p className="text-xs font-semibold text-slate-200 mb-2">Original Fundus Photograph</p>
                <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
                  {image && !fundusImgError ? (
                    <img
                      src={image.image_url}
                      alt="Fundus"
                      className="w-full h-full object-contain"
                      onError={() => setFundusImgError(true)}
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center text-rose-400 p-4 text-center">
                      <AlertTriangle className="w-8 h-8 mb-1.5 text-rose-500" />
                      <span className="text-xs font-semibold">Image Unavailable</span>
                      <span className="text-[10px] text-slate-400 mt-0.5">Original photograph missing or unreadable on disk.</span>
                    </div>
                  )}
                </div>
                <div className="mt-2 text-[11px] text-slate-400">
                  Resolution: {image?.blur_score ? `${image.blur_score.toFixed(1)} blur score` : 'Standard fundus'} | {image?.quality_status || 'GRADABLE'}
                </div>
              </div>

              {/* Grad-CAM Saliency */}
              <div className="border border-slate-200 rounded-lg p-3 bg-slate-900 text-center">
                <div className="flex justify-between items-center mb-2 px-1">
                  <p className="text-xs font-semibold text-primary-300">Grad-CAM Class Activation Overlay</p>
                  <button
                    onClick={() => handleGenerateGradcam()}
                    disabled={generatingGradcam || !image}
                    className="inline-flex items-center px-2 py-0.5 text-[11px] font-medium rounded bg-primary-700 hover:bg-primary-600 text-white disabled:opacity-50"
                  >
                    {generatingGradcam ? (
                      <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                    ) : (
                      <Sparkles className="w-3 h-3 mr-1" />
                    )}
                    {prediction?.gradcam_url ? 'Recompute' : 'Generate'}
                  </button>
                </div>
                <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center relative">
                  {prediction?.gradcam_url ? (
                    <img
                      src={prediction.gradcam_url}
                      alt="Grad-CAM Overlay"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-xs text-slate-400 p-6 space-y-3">
                      <p>Grad-CAM overlay not yet computed for this encounter image.</p>
                      <button
                        onClick={() => handleGenerateGradcam()}
                        disabled={generatingGradcam || !image}
                        className="px-3 py-1.5 bg-primary-600 text-white font-semibold rounded text-xs hover:bg-primary-500 inline-flex items-center"
                      >
                        {generatingGradcam ? <RefreshCw className="w-3 h-3 mr-1 animate-spin" /> : <Sparkles className="w-3 h-3 mr-1" />}
                        Compute Grad-CAM with Model
                      </button>
                    </div>
                  )}
                </div>
                <div className="mt-2 text-[11px] text-slate-400 flex items-center justify-between px-1">
                  <span>Target Class: {CLASS_NAMES[targetClass ?? (prediction?.predicted_class || 2)]?.split(' - ')[0]}</span>
                  <span>Confidence: {((prediction?.confidence || 0) * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Saliency Legend */}
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
              <span className="font-semibold text-slate-700">Colormap Interpretation: </span>
              <span className="text-red-600 font-medium">Red/Orange (High Importance)</span> denotes highest gradient activation regions; 
              <span className="text-blue-600 font-medium"> Blue/Cyan (Low Importance)</span> denotes background / unweighted ocular areas.
            </div>
          </div>

          {/* Model Confidence & Uncertainty (1 col) */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-5">
            <h2 className="text-base font-bold text-slate-800 flex items-center">
              <Sparkles className="w-4 h-4 mr-2 text-primary-600" />
              Inference Breakdown
            </h2>

            {/* Prediction Summary */}
            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
              <div className="text-xs text-slate-500 uppercase font-semibold">Predicted DR Grade</div>
              <div className="text-lg font-bold text-slate-800">
                {prediction ? `Class ${prediction.predicted_class}: ${prediction.predicted_label}` : 'No prediction recorded'}
              </div>
              <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-200">
                <span className="text-slate-600">Softmax Confidence:</span>
                <span className="font-bold text-primary-700">{((prediction?.confidence || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">MC-Dropout Uncertainty:</span>
                <span className={`font-bold ${prediction?.uncertainty_level === 'Low' ? 'text-green-600' : 'text-amber-600'}`}>
                  {prediction?.uncertainty_level || 'Normal'} (σ = {(prediction?.uncertainty_std || 0.04).toFixed(3)})
                </span>
              </div>
            </div>

            {/* Probabilities distribution */}
            <div>
              <h3 className="text-xs font-bold text-slate-700 mb-2">Class Probability Distribution</h3>
              <div className="space-y-2 text-xs">
                {CLASS_NAMES.map((name, idx) => {
                  const prob = (prediction?.class_probabilities?.[idx] || 0) * 100
                  const isPredicted = prediction?.predicted_class === idx
                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between font-medium">
                        <span className={isPredicted ? 'text-primary-700 font-bold' : 'text-slate-600'}>
                          {name}
                        </span>
                        <span className="text-slate-800">{prob.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${isPredicted ? 'bg-primary-600' : 'bg-slate-300'}`}
                          style={{ width: `${Math.max(prob, 2)}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Technical Metadata */}
            <div className="p-3 bg-slate-50 rounded border border-slate-200 text-xs text-slate-600 space-y-1">
              <div className="font-semibold text-slate-700">Archival Manifest:</div>
              <div>Model: {prediction?.model_version || 'EfficientNet-B3 DR Benchmark v1.0'}</div>
              <div>Quality Screening: {image?.quality_status || 'PASS'}</div>
              <div>Gradable Fundus: {image?.is_gradable ? 'YES' : 'NO'}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Explainability