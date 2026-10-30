import React, { useState, useEffect } from 'react'
import client, { get } from '../api/client'
import { FileText, Download, Filter, Search, RefreshCw, CheckCircle, Clock, AlertTriangle, ShieldCheck } from 'lucide-react'

const Reports = () => {
  const [screenings, setScreenings] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [downloadingId, setDownloadingId] = useState(null)
  const [selectedCaseDetail, setSelectedCaseDetail] = useState(null)

  useEffect(() => {
    fetchScreenings()
  }, [statusFilter])

  const fetchScreenings = async () => {
    try {
      setLoading(true)
      setError(null)
      let url = `/screenings?limit=50`
      if (statusFilter !== 'ALL') {
        url += `&status=${statusFilter}`
      }
      if (searchTerm) {
        url += `&search=${encodeURIComponent(searchTerm)}`
      }
      const res = await get(url)
      const items = Array.isArray(res) ? res : (res?.items || [])
      setScreenings(items)
      setLoading(false)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fetch clinical screening reports')
      setLoading(false)
    }
  }

  const handleDownloadPDF = async (screeningId, caseNumber) => {
    try {
      setDownloadingId(screeningId)
      const res = await client.get(`/screenings/${screeningId}/report/pdf`, {
        responseType: 'blob'
      })
      const blob = new Blob([res.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `Clinical_DR_Report_${caseNumber}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert('Failed to generate PDF clinical report. Please ensure encounter is completed with review.')
    } finally {
      setDownloadingId(null)
    }
  }

  const handleExportCSV = async () => {
    try {
      const res = await client.get('/screenings/export/csv', {
        responseType: 'blob'
      })
      const blob = new Blob([res.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `screenings_archive_${new Date().toISOString().slice(0, 10)}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert('Failed to export CSV archive.')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center pb-4 border-b border-slate-200">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 flex items-center">
            <FileText className="w-6 h-6 mr-2 text-primary-600" />
            Clinical Reports & Archival Export
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Download publication-grade ReportLab Platypus PDF clinical summaries and export audit datasets
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex items-center space-x-3">
          <button
            onClick={handleExportCSV}
            className="flex items-center px-3.5 py-2 text-xs font-semibold bg-white border border-slate-300 rounded-lg shadow-sm hover:bg-slate-50 text-slate-700"
          >
            <Download className="w-4 h-4 mr-1.5" />
            Export Archive CSV
          </button>
          <button
            onClick={fetchScreenings}
            className="flex items-center px-3.5 py-2 text-xs font-semibold bg-primary-600 rounded-lg shadow-sm hover:bg-primary-700 text-white"
          >
            <RefreshCw className="w-4 h-4 mr-1.5" />
            Refresh
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
          <input
            type="text"
            placeholder="Search by case reference (SCR-)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && fetchScreenings()}
            className="w-full pl-9 pr-4 py-2 text-sm border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-primary-500"
          />
        </div>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          <Filter className="w-4 h-4 text-slate-500" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="text-sm border border-slate-300 rounded-lg p-2 bg-slate-50 focus:ring-2 focus:ring-primary-500"
          >
            <option value="ALL">All Encounter Statuses</option>
            <option value="COMPLETED">Completed (Reviewed)</option>
            <option value="AWAITING_REVIEW">Awaiting Review</option>
            <option value="AWAITING_PREDICTION">Awaiting Prediction</option>
            <option value="UNGRADABLE">Ungradable</option>
          </select>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-center justify-between text-xs text-rose-800">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={fetchScreenings}
            className="px-3 py-1 bg-white border border-rose-300 rounded font-semibold text-rose-700 hover:bg-rose-50"
          >
            Retry Request
          </button>
        </div>
      )}

      {/* Reports Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-xs uppercase font-bold text-slate-700 border-b border-slate-200">
              <tr>
                <th className="px-5 py-3">Case Reference / Screening ID</th>
                <th className="px-5 py-3">Patient</th>
                <th className="px-5 py-3">Date</th>
                <th className="px-5 py-3">AI Prediction</th>
                <th className="px-5 py-3">Confidence</th>
                <th className="px-5 py-3">Review Decision</th>
                <th className="px-5 py-3">Status</th>
                <th className="px-5 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {loading ? (
                <tr>
                  <td colSpan="8" className="px-6 py-12 text-center text-slate-500">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto text-primary-500 mb-2" />
                    Loading clinical encounters...
                  </td>
                </tr>
              ) : screenings.length === 0 ? (
                <tr>
                  <td colSpan="8" className="px-6 py-8 text-center text-slate-500">
                    No clinical screenings found matching current filter.
                  </td>
                </tr>
              ) : (
                screenings.map((sc) => (
                  <tr key={sc.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-5 py-4">
                      <div className="font-bold text-slate-900 flex items-center">
                        <FileText className="w-4 h-4 mr-1.5 text-primary-600 flex-shrink-0" />
                        {sc.case_number}
                      </div>
                      <div className="text-[10px] text-slate-400 font-mono mt-0.5" title="Screening UUID">
                        ID: {sc.id}
                      </div>
                    </td>
                    <td className="px-5 py-4">
                      <div className="font-mono text-xs font-semibold text-slate-800">{sc.patient_code}</div>
                      <div className="text-[11px] text-slate-500">{sc.patient_name}</div>
                      {sc.patient_id && (
                        <div className="text-[9px] text-slate-400 font-mono" title="Patient UUID">
                          P-ID: {sc.patient_id?.slice(0, 8)}...
                        </div>
                      )}
                    </td>
                    <td className="px-5 py-4 text-xs whitespace-nowrap">
                      {new Date(sc.encounter_date).toLocaleDateString()}
                    </td>
                    <td className="px-5 py-4">
                      {sc.predicted_grade !== null && sc.predicted_grade !== undefined ? (
                        <div>
                          <span className="font-semibold text-slate-800">
                            Grade {sc.predicted_grade}: {sc.predicted_label || `Class ${sc.predicted_grade}`}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400 italic">Pending</span>
                      )}
                    </td>
                    <td className="px-5 py-4">
                      {sc.confidence !== null && sc.confidence !== undefined ? (
                        <span className="font-mono text-xs font-semibold text-primary-700">
                          {(sc.confidence * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-slate-400">---</span>
                      )}
                    </td>
                    <td className="px-5 py-4">
                      {sc.review_decision ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-100 text-emerald-800">
                          {sc.review_decision}
                        </span>
                      ) : (
                        <span className="text-slate-400 italic">Unreviewed</span>
                      )}
                    </td>
                    <td className="px-5 py-4">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                        sc.status === 'COMPLETED' || sc.status === 'APPROVED'
                          ? 'bg-green-100 text-green-800'
                          : sc.status === 'UNGRADABLE'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}>
                        {sc.status}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-right space-x-2 whitespace-nowrap">
                      {sc.class_probabilities && (
                        <button
                          onClick={() => setSelectedCaseDetail(sc)}
                          className="inline-flex items-center px-2 py-1 text-xs font-medium rounded bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300"
                        >
                          Probabilities
                        </button>
                      )}
                      <button
                        onClick={() => handleDownloadPDF(sc.id, sc.case_number)}
                        disabled={downloadingId === sc.id}
                        className="inline-flex items-center px-2.5 py-1 text-xs font-medium rounded bg-primary-50 text-primary-700 hover:bg-primary-100 border border-primary-200 transition-colors disabled:opacity-50"
                      >
                        {downloadingId === sc.id ? (
                          <RefreshCw className="w-3.5 h-3.5 mr-1 animate-spin" />
                        ) : (
                          <Download className="w-3.5 h-3.5 mr-1" />
                        )}
                        PDF
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Probabilities Breakdown Modal */}
      {selectedCaseDetail && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6 space-y-4">
            <div className="flex justify-between items-start border-b border-slate-200 pb-3">
              <div>
                <h3 className="font-bold text-slate-800 text-sm">Class Probabilities Breakdown</h3>
                <p className="text-xs text-slate-500 font-mono">{selectedCaseDetail.case_number} ({selectedCaseDetail.patient_code})</p>
              </div>
              <button onClick={() => setSelectedCaseDetail(null)} className="text-slate-400 hover:text-slate-600 text-xl font-bold">
                &times;
              </button>
            </div>

            <div className="space-y-2 text-xs">
              {['0 - No DR', '1 - Mild DR', '2 - Moderate DR', '3 - Severe DR', '4 - Proliferative DR'].map((label, idx) => {
                const prob = (selectedCaseDetail.class_probabilities?.[idx] || 0) * 100
                const isSelected = selectedCaseDetail.predicted_grade === idx
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between font-medium">
                      <span className={isSelected ? 'text-primary-700 font-bold' : 'text-slate-600'}>{label}</span>
                      <span className="font-mono text-slate-800">{prob.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${isSelected ? 'bg-primary-600' : 'bg-slate-300'}`}
                        style={{ width: `${Math.max(prob, 2)}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="pt-3 border-t border-slate-200 flex justify-between items-center text-xs">
              <div>
                Confidence: <strong className="text-primary-700">{((selectedCaseDetail.confidence || 0) * 100).toFixed(1)}%</strong>
              </div>
              <button
                onClick={() => setSelectedCaseDetail(null)}
                className="px-4 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold rounded-lg"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Reports