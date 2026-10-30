import { useState } from 'react'
import { AlertCircle, CheckCircle, Info, Database, Save } from 'lucide-react'

const EHRForm = ({ formData = {}, onChange, hasExistingRecord = false, onSaveToRecord, isSaving = false }) => {
  const [bloodGlucoseUnit, setBloodGlucoseUnit] = useState('mg/dL')

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target
    const updated = {
      ...formData,
      [name]: type === 'checkbox' ? checked : value
    }
    if (onChange) {
      onChange(updated)
    }
  }

  const handleBloodGlucoseChange = (e) => {
    const val = e.target.value
    const num = parseFloat(val)
    const updated = {
      ...formData,
      blood_glucose: val,
      blood_glucose_unit: bloodGlucoseUnit
    }
    // Auto-calculate HbA1c if not set or user wants suggestion: HbA1c = (eAG + 46.7) / 28.7
    if (!formData.hba1c && !isNaN(num) && num > 0) {
      const eAG = bloodGlucoseUnit === 'mmol/L' ? num * 18.0182 : num
      const estimatedA1c = ((eAG + 46.7) / 28.7).toFixed(1)
      if (estimatedA1c >= 4.0 && estimatedA1c <= 18.0) {
        updated.hba1c = estimatedA1c
      }
    }
    if (onChange) onChange(updated)
  }

  // Field validation helpers
  const getValidationWarning = (name, val) => {
    if (val === undefined || val === '' || val === null) return null
    const n = parseFloat(val)
    if (isNaN(n)) return 'Must be a valid number'
    switch (name) {
      case 'age':
        if (n < 1 || n > 120) return 'Age typically between 1 and 120'
        break
      case 'diabetes_duration_years':
        if (n < 0 || n > 70) return 'Duration typically 0 to 70 years'
        break
      case 'hba1c':
        if (n < 3.5 || n > 20) return 'HbA1c typically 3.5% to 20.0%'
        break
      case 'systolic_bp':
        if (n < 60 || n > 250) return 'Systolic BP typically 60-250 mmHg'
        break
      case 'diastolic_bp':
        if (n < 40 || n > 150) return 'Diastolic BP typically 40-150 mmHg'
        break
      case 'bmi':
        if (n < 10 || n > 70) return 'BMI typically 10-70 kg/m²'
        break
      case 'cholesterol_ldl':
        if (n < 40 || n > 500) return 'Cholesterol typically 40-500 mg/dL'
        break
      default:
        return null
    }
    return null
  }

  const renderStatusBadge = (fieldKey) => {
    const val = formData[fieldKey]
    const isRecorded = val !== undefined && val !== '' && val !== null
    if (isRecorded) {
      return (
        <span className="text-[10px] text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded font-medium flex items-center gap-0.5">
          <CheckCircle className="w-2.5 h-2.5" /> Recorded
        </span>
      )
    }
    return (
      <span className="text-[10px] text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded font-medium flex items-center gap-0.5">
        <AlertCircle className="w-2.5 h-2.5" /> Not recorded
      </span>
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header bar */}
      <div className="bg-slate-50 border-b border-slate-200 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-primary-600" />
          <h4 className="text-sm font-semibold text-slate-800">
            Patient Clinical EHR Profile
          </h4>
          {hasExistingRecord ? (
            <span className="text-xs bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-medium">
              EHR Record Synced
            </span>
          ) : (
            <span className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full font-medium">
              No Previous EHR Record (Enter Values Below)
            </span>
          )}
        </div>

        {onSaveToRecord && (
          <button
            type="button"
            onClick={onSaveToRecord}
            disabled={isSaving}
            className="text-xs flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-100 font-medium transition disabled:opacity-50"
          >
            <Save className="w-3.5 h-3.5 text-primary-600" />
            {isSaving ? 'Saving...' : 'Save to Patient EHR'}
          </button>
        )}
      </div>

      <div className="p-6 space-y-6">
        <div className="text-xs text-slate-500 bg-blue-50 border border-blue-100 rounded-lg p-3 flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" />
          <span>
            The Multimodal Attention-Fusion model combines deep retinal fundus features with 17 clinical risk covariates. Required fields are marked with <span className="text-rose-500 font-bold">*</span>. Values outside typical ranges will be highlighted for review.
          </span>
        </div>

        {/* 1. Primary Demographics & Diabetes Timeline */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Patient Age (years) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('age')}
            </div>
            <input
              type="number"
              name="age"
              placeholder="e.g. 58"
              value={formData.age ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('age', formData.age) && (
              <p className="text-[11px] text-amber-600 mt-0.5">{getValidationWarning('age', formData.age)}</p>
            )}
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Sex at Birth <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('sex')}
            </div>
            <select
              name="sex"
              value={formData.sex ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            >
              <option value="">Select Sex...</option>
              <option value="1">Male</option>
              <option value="0">Female</option>
            </select>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Diabetes Duration (years) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('diabetes_duration_years')}
            </div>
            <input
              type="number"
              step="0.5"
              name="diabetes_duration_years"
              placeholder="e.g. 8.5"
              value={formData.diabetes_duration_years ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('diabetes_duration_years', formData.diabetes_duration_years) && (
              <p className="text-[11px] text-amber-600 mt-0.5">
                {getValidationWarning('diabetes_duration_years', formData.diabetes_duration_years)}
              </p>
            )}
          </div>
        </div>

        {/* 2. Glycemic Control & Lab Values */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border-t pt-4">
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Glycated Hemoglobin HbA1c (%) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('hba1c')}
            </div>
            <input
              type="number"
              step="0.1"
              name="hba1c"
              placeholder="e.g. 7.8"
              value={formData.hba1c ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('hba1c', formData.hba1c) && (
              <p className="text-[11px] text-amber-600 mt-0.5">{getValidationWarning('hba1c', formData.hba1c)}</p>
            )}
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Blood Glucose (Optional)
              </label>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setBloodGlucoseUnit('mg/dL')}
                  className={`text-[10px] px-1 py-0.5 rounded ${bloodGlucoseUnit === 'mg/dL' ? 'bg-primary-600 text-white' : 'bg-slate-100 text-slate-600'}`}
                >
                  mg/dL
                </button>
                <button
                  type="button"
                  onClick={() => setBloodGlucoseUnit('mmol/L')}
                  className={`text-[10px] px-1 py-0.5 rounded ${bloodGlucoseUnit === 'mmol/L' ? 'bg-primary-600 text-white' : 'bg-slate-100 text-slate-600'}`}
                >
                  mmol/L
                </button>
              </div>
            </div>
            <input
              type="number"
              step="1"
              name="blood_glucose"
              placeholder={bloodGlucoseUnit === 'mg/dL' ? 'e.g. 155' : 'e.g. 8.6'}
              value={formData.blood_glucose ?? ''}
              onChange={handleBloodGlucoseChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            <p className="text-[10px] text-slate-400 mt-0.5">Calculates estimated HbA1c via eAG formula</p>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Body Mass Index (BMI kg/m²) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('bmi')}
            </div>
            <input
              type="number"
              step="0.1"
              name="bmi"
              placeholder="e.g. 27.4"
              value={formData.bmi ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('bmi', formData.bmi) && (
              <p className="text-[11px] text-amber-600 mt-0.5">{getValidationWarning('bmi', formData.bmi)}</p>
            )}
          </div>
        </div>

        {/* 3. Vitals & Lipids */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border-t pt-4">
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Systolic BP (mmHg) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('systolic_bp')}
            </div>
            <input
              type="number"
              name="systolic_bp"
              placeholder="e.g. 132"
              value={formData.systolic_bp ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('systolic_bp', formData.systolic_bp) && (
              <p className="text-[11px] text-amber-600 mt-0.5">{getValidationWarning('systolic_bp', formData.systolic_bp)}</p>
            )}
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                Diastolic BP (mmHg) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('diastolic_bp')}
            </div>
            <input
              type="number"
              name="diastolic_bp"
              placeholder="e.g. 84"
              value={formData.diastolic_bp ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('diastolic_bp', formData.diastolic_bp) && (
              <p className="text-[11px] text-amber-600 mt-0.5">{getValidationWarning('diastolic_bp', formData.diastolic_bp)}</p>
            )}
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-semibold text-slate-700">
                LDL Cholesterol (mg/dL) <span className="text-rose-500">*</span>
              </label>
              {renderStatusBadge('cholesterol_ldl')}
            </div>
            <input
              type="number"
              name="cholesterol_ldl"
              placeholder="e.g. 115"
              value={formData.cholesterol_ldl ?? ''}
              onChange={handleChange}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
            />
            {getValidationWarning('cholesterol_ldl', formData.cholesterol_ldl) && (
              <p className="text-[11px] text-amber-600 mt-0.5">{getValidationWarning('cholesterol_ldl', formData.cholesterol_ldl)}</p>
            )}
          </div>
        </div>

        {/* 4. Lifestyle & Medical History / Complications */}
        <div className="border-t pt-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-semibold text-slate-700">
                  Smoking Status <span className="text-rose-500">*</span>
                </label>
                {renderStatusBadge('smoking_status')}
              </div>
              <select
                name="smoking_status"
                value={formData.smoking_status ?? '0'}
                onChange={handleChange}
                className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-primary-500"
              >
                <option value="0">Never Smoked</option>
                <option value="1">Former Smoker</option>
                <option value="2">Current Smoker</option>
              </select>
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-2">
              Clinical Complications & Treatment History
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <label className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input
                  type="checkbox"
                  name="insulin_use"
                  checked={Boolean(formData.insulin_use)}
                  onChange={handleChange}
                  className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                />
                <div>
                  <span className="text-xs font-medium text-slate-800 block">Insulin Therapy Dependent</span>
                  <span className="text-[11px] text-slate-400">Exogenous insulin management</span>
                </div>
              </label>

              <label className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input
                  type="checkbox"
                  name="prev_laser_treatment"
                  checked={Boolean(formData.prev_laser_treatment)}
                  onChange={handleChange}
                  className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                />
                <div>
                  <span className="text-xs font-medium text-slate-800 block">Prior Retinal Laser Treatment</span>
                  <span className="text-[11px] text-slate-400">Panretinal or focal photocoagulation</span>
                </div>
              </label>

              <label className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input
                  type="checkbox"
                  name="nephropathy"
                  checked={Boolean(formData.nephropathy)}
                  onChange={handleChange}
                  className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                />
                <div>
                  <span className="text-xs font-medium text-slate-800 block">Diabetic Nephropathy</span>
                  <span className="text-[11px] text-slate-400">Microalbuminuria or renal impairment</span>
                </div>
              </label>

              <label className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-slate-50 cursor-pointer transition">
                <input
                  type="checkbox"
                  name="neuropathy"
                  checked={Boolean(formData.neuropathy)}
                  onChange={handleChange}
                  className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                />
                <div>
                  <span className="text-xs font-medium text-slate-800 block">Diabetic Neuropathy</span>
                  <span className="text-[11px] text-slate-400">Peripheral or autonomic nerve involvement</span>
                </div>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EHRForm
