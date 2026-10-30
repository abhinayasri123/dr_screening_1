import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import {
  Play, Activity, Server, Database, CheckCircle2, AlertTriangle, Cpu, Layers, HardDrive, RefreshCw
} from 'lucide-react'
import { getFLStatus, startFLSimulation, getFLComparison } from '../api/federated'
import toast from 'react-hot-toast'

const FLDashboard = () => {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [comparison, setComparison] = useState(null)
  const [rounds, setRounds] = useState(3)
  const [localEpochs, setLocalEpochs] = useState(1)
  const [batchSize, setBatchSize] = useState(16)

  const fetchStatus = async () => {
    try {
      const res = await getFLStatus()
      setStatus(res)
      if (res?.results && Object.keys(res.results).length > 0) {
        setComparison(res.results)
      }
    } catch (err) {
      console.error('Error fetching FL status:', err)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 2500)
    return () => clearInterval(interval)
  }, [])

  const handleStart = async (e) => {
    e?.preventDefault()
    setLoading(true)
    try {
      const res = await startFLSimulation({
        rounds: Number(rounds),
        clients: 3,
        local_epochs: Number(localEpochs),
        batch_size: Number(batchSize),
        lr: 0.0001
      })
      toast.success(res?.message || 'Federated simulation initiated')
      await fetchStatus()
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Failed to start simulation'
      toast.error(detail)
    } finally {
      setLoading(false)
    }
  }

  const isRunning = Boolean(status?.running || status?.is_running)
  const history = status?.metrics_history || []
  const datasetInfo = status?.dataset_info
  const hospitalNodes = status?.hospital_nodes || status?.clients || {}
  const results = status?.results || comparison || {}
  const centralized = results?.centralized
  const federated = results?.federated || results?.federated_final

  return (
    <div className="space-y-6">
      {/* Simulation Accuracy & Methodology Notice */}
      <div className="p-3.5 glass-card rounded-xl text-xs text-[#D0CCC3] flex items-start space-x-3 border border-[#81958D]/25">
        <Server className="w-5 h-5 text-[#81958D] flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-[#F5F0E6]">REPRODUCIBLE SIMULATION ENVIRONMENT: </span>
          This module executes real PyTorch training and Federated Averaging (FedAvg) aggregation partitioned across local hospital nodes. Model weights are trained and exchanged locally without transmitting raw fundus images. Checkpoint persistence safeguards prevent automated overwrites of the verified production screening model.
        </div>
      </div>

      {/* Simulation Header & Live Control Panel */}
      <div className="glass-card p-6 rounded-xl border border-[#81958D]/25 shadow-sm">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-[#81958D]/20 pb-5 mb-5">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2.5">
              <h2 className="text-xl font-bold text-[#F5F0E6] tracking-tight">Federated Learning Simulation Controller</h2>
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold whitespace-nowrap flex-shrink-0 border ${
                isRunning 
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse' 
                  : status?.current_phase === 'Completed'
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  : 'bg-[#191919]/60 text-[#81958D] border-[#81958D]/30'
              }`}>
                {isRunning ? 'TRAINING IN PROGRESS' : status?.current_phase === 'Completed' ? 'COMPLETED' : 'IDLE'}
              </span>
            </div>
            <p className="text-[#D0CCC3] text-xs sm:text-sm mt-1">
              Multi-node FedAvg training with PyTorch CNN on partitioned retinal fundus images
            </p>
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            <button
              onClick={fetchStatus}
              title="Refresh Status"
              className="p-2 border border-[#81958D]/30 rounded-lg hover:bg-[#191919]/60 text-[#81958D] transition"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={handleStart}
              disabled={isRunning || loading || datasetInfo?.found === false}
              className="flex items-center px-5 py-2.5 bg-primary-600 text-[#F5F0E6] font-semibold text-sm rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition whitespace-nowrap"
            >
              {isRunning ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-pulse text-[#81958D]" />
                  Running Round {status?.current_round || 1}/{status?.total_rounds || rounds}...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2 text-[#81958D]" />
                  Start Real FL Training
                </>
              )}
            </button>
          </div>
        </div>

        {/* Real Training Parameters Form */}
        <form onSubmit={handleStart} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="flex flex-col justify-between">
            <label className="block text-xs font-semibold text-[#D0CCC3] uppercase tracking-wider mb-1.5">Communication Rounds</label>
            <input
              type="number"
              min="1"
              max="10"
              value={rounds}
              onChange={(e) => setRounds(e.target.value)}
              disabled={isRunning}
              className="w-full h-[38px] px-3 py-2 bg-[#0F0F0F]/80 border border-[#81958D]/25 rounded-lg text-[#F5F0E6] focus:border-[#81958D] disabled:opacity-50 text-sm"
            />
          </div>
          <div className="flex flex-col justify-between">
            <label className="block text-xs font-semibold text-[#D0CCC3] uppercase tracking-wider mb-1.5">Local Epochs / Round</label>
            <input
              type="number"
              min="1"
              max="5"
              value={localEpochs}
              onChange={(e) => setLocalEpochs(e.target.value)}
              disabled={isRunning}
              className="w-full h-[38px] px-3 py-2 bg-[#0F0F0F]/80 border border-[#81958D]/25 rounded-lg text-[#F5F0E6] focus:border-[#81958D] disabled:opacity-50 text-sm"
            />
          </div>
          <div className="flex flex-col justify-between">
            <label className="block text-xs font-semibold text-[#D0CCC3] uppercase tracking-wider mb-1.5">Batch Size</label>
            <input
              type="number"
              min="4"
              max="64"
              step="4"
              value={batchSize}
              onChange={(e) => setBatchSize(e.target.value)}
              disabled={isRunning}
              className="w-full h-[38px] px-3 py-2 bg-[#0F0F0F]/80 border border-[#81958D]/25 rounded-lg text-[#F5F0E6] focus:border-[#81958D] disabled:opacity-50 text-sm"
            />
          </div>
          <div className="flex flex-col justify-between">
            <label className="block text-xs font-semibold text-[#D0CCC3] uppercase tracking-wider mb-1.5">Aggregation Algorithm</label>
            <div className="w-full h-[38px] px-3 py-2 bg-[#0A2035]/90 border border-[#81958D]/20 rounded-lg text-[#81958D] font-mono text-xs flex items-center justify-between">
              <span className="font-bold">FedAvg</span>
              <span className="text-[11px] text-[#D0CCC3]">&Sigma;(n_k/N)&middot;W_k</span>
            </div>
          </div>
        </form>

        {/* Live Execution Phase Banner */}
        {isRunning && (
          <div className="mt-4 p-3.5 bg-[#191919]/60 border border-[#81958D]/40 rounded-xl flex items-center gap-3 text-[#F5F0E6] text-sm overflow-hidden">
            <Activity className="w-5 h-5 text-[#81958D] animate-spin flex-shrink-0" />
            <div className="min-w-0 flex-1 break-words">
              <span className="font-semibold text-[#81958D]">Current Phase: </span>
              <span className="text-[#F5F0E6]">{status?.current_phase || 'Synchronizing hospital nodes...'}</span>
            </div>
          </div>
        )}

        {status?.error && (
          <div className="mt-4 p-3.5 bg-red-950/60 border border-red-500/40 rounded-xl flex items-center gap-2 text-red-200 text-sm">
            <AlertTriangle className="w-5 h-5 flex-shrink-0 text-red-400" />
            <span className="break-words">{status.error}</span>
          </div>
        )}
      </div>

      {/* Dataset & Hardware Verification Card */}
      <div className="glass-card p-4 rounded-xl text-xs text-[#D0CCC3] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 border border-[#81958D]/25">
        <div className="flex items-start gap-2.5 min-w-0">
          <HardDrive className="w-4 h-4 text-[#81958D] flex-shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1 break-all overflow-wrap-anywhere">
            <strong className="text-[#F5F0E6] block">Dataset Path:</strong>
            <span className="text-[11px] font-mono">{datasetInfo?.path || 'Configured via APTOS_DATA_DIR'}</span>
          </div>
        </div>
        <div className="flex items-start gap-2.5 min-w-0">
          <Database className="w-4 h-4 text-[#81958D] flex-shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1">
            <strong className="text-[#F5F0E6] block">Total Images:</strong>
            <span>{datasetInfo?.total_images ?? 'Detecting...'} ({datasetInfo?.train_samples || 0} train / {datasetInfo?.test_samples || 0} test)</span>
          </div>
        </div>
        <div className="flex items-start gap-2.5 min-w-0">
          <Cpu className="w-4 h-4 text-[#81958D] flex-shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1">
            <strong className="text-[#F5F0E6] block">Compute Device:</strong>
            <span>{status?.device || 'CPU'}</span>
          </div>
        </div>
        <div className="flex items-start gap-2.5 min-w-0">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1">
            <strong className="text-[#F5F0E6] block">Classes:</strong>
            <span>5 (0: No DR &rarr; 4: Proliferative DR)</span>
          </div>
        </div>
      </div>

      {/* Hospital Partition Nodes Grid */}
      <div>
        <h3 className="text-lg font-bold text-[#F5F0E6] mb-3 flex items-center gap-2">
          <Layers className="w-5 h-5 text-[#81958D]" />
          Simulated Hospital Client Nodes (Real Data Partitions)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {[1, 2, 3].map((clientNum) => {
            const key = `client_${clientNum}`
            const node = hospitalNodes[key]
            const isNodeActive = isRunning && node?.status === 'Training...'

            return (
              <div
                key={clientNum}
                className={`glass-card p-5 rounded-xl border transition-all ${
                  isNodeActive 
                    ? 'border-[#81958D] shadow-[0_0_20px_rgba(40,229,242,0.3)] ring-1 ring-[#81958D]' 
                    : 'border-[#81958D]/25'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2 border-b border-[#81958D]/20">
                  <div className="flex items-center text-sm font-bold text-[#F5F0E6]">
                    <Database className="w-4 h-4 mr-2 text-[#81958D] flex-shrink-0" />
                    <span>Hospital {clientNum} Node</span>
                  </div>
                  <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold whitespace-nowrap border ${
                    node?.status === 'Training...' 
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse'
                      : node?.status === 'Completed'
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      : 'bg-[#191919]/60 text-[#D0CCC3] border-[#81958D]/20'
                  }`}>
                    {node?.status || (isRunning ? 'Waiting' : 'Standby')}
                  </span>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1 border-b border-[#81958D]/10 text-[#D0CCC3]">
                    <span>Local Data Size:</span>
                    <strong className="text-[#F5F0E6]">
                      {node?.data_size != null ? `${node.data_size} images` : '---'}
                    </strong>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#81958D]/10 text-[#D0CCC3]">
                    <span>Last Local Accuracy:</span>
                    <strong className="text-[#81958D]">
                      {node?.accuracy != null ? `${(node.accuracy * 100).toFixed(1)}%` : '---'}
                    </strong>
                  </div>
                  <div className="flex justify-between py-1 text-[#D0CCC3]">
                    <span>Local Macro F1:</span>
                    <strong className="text-[#81958D]">
                      {node?.f1 != null ? (node.f1).toFixed(3) : '---'}
                    </strong>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Global Model Performance Across Rounds */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-6 rounded-xl border border-[#81958D]/25">
          <div className="flex flex-wrap justify-between items-center gap-3 mb-4">
            <div>
              <h3 className="text-lg font-bold text-[#F5F0E6]">Global Model Accuracy vs. Communication Rounds</h3>
              <p className="text-xs text-[#D0CCC3]">Evaluated on independent held-out retinal test split after FedAvg weight aggregation</p>
            </div>
            {history.length > 0 && (
              <span className="text-xs font-semibold px-2.5 py-1 bg-[#191919] text-[#81958D] rounded-lg border border-[#81958D]/30">
                Latest: {(history[history.length - 1].global_accuracy * 100).toFixed(1)}%
              </span>
            )}
          </div>

          {history.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(40, 229, 242, 0.15)" />
                  <XAxis 
                    dataKey="round" 
                    stroke="#D0CCC3"
                    tick={{ fill: '#D0CCC3', fontSize: 11 }}
                    label={{ value: 'Round', position: 'insideBottom', offset: -5, fill: '#D0CCC3', fontSize: 11 }} 
                  />
                  <YAxis 
                    domain={[0, 1.0]} 
                    stroke="#D0CCC3"
                    tick={{ fill: '#D0CCC3', fontSize: 11 }}
                    tickFormatter={v => `${(v * 100).toFixed(0)}%`} 
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(6, 20, 38, 0.95)', borderColor: 'rgba(40, 229, 242, 0.3)', borderRadius: '8px', color: '#F5F0E6' }}
                    formatter={(value, name) => [
                      name === 'global_accuracy' 
                        ? `${(value * 100).toFixed(2)}%` 
                        : typeof value === 'number' ? value.toFixed(3) : value,
                      name === 'global_accuracy' ? 'Global Accuracy' : 'Global Loss'
                    ]} 
                  />
                  <Legend wrapperStyle={{ color: '#F5F0E6', paddingTop: '8px' }} />
                  <Line 
                    type="monotone" 
                    dataKey="global_accuracy" 
                    stroke="#81958D" 
                    strokeWidth={3} 
                    dot={{ r: 4, fill: '#81958D' }} 
                    name="Global Accuracy" 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="global_loss" 
                    stroke="#FBBF24" 
                    strokeWidth={2} 
                    strokeDasharray="4 4" 
                    dot={{ r: 3, fill: '#FBBF24' }} 
                    name="Global Loss" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-[#D0CCC3] bg-[#0A2035]/40 rounded-lg border border-dashed border-[#81958D]/25">
              <Activity className="w-8 h-8 mb-2 text-[#81958D]/60 animate-pulse" />
              <p className="text-sm font-medium">No round metrics available yet.</p>
              <p className="text-xs text-[#D0CCC3]/70 mt-1">Click &quot;Start Real FL Training&quot; to begin simulation.</p>
            </div>
          )}
        </div>

        {/* Round Summary Card */}
        <div className="glass-card p-6 rounded-xl border border-[#81958D]/25 flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-bold text-[#F5F0E6] mb-2">Latest Federated Evaluation</h3>
            <p className="text-xs text-[#D0CCC3] mb-4">Real test metrics computed on held-out fundus dataset</p>

            {history.length > 0 ? (
              <div className="space-y-3">
                <div className="bg-[#0A2035]/60 p-3 rounded-lg flex justify-between items-center border border-[#81958D]/15">
                  <span className="text-xs text-[#D0CCC3] font-medium">Test Accuracy</span>
                  <span className="text-base font-bold text-[#81958D]">
                    {(history[history.length - 1].global_accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-[#0A2035]/60 p-3 rounded-lg flex justify-between items-center border border-[#81958D]/15">
                  <span className="text-xs text-[#D0CCC3] font-medium">Quadratic Weighted Kappa (QWK)</span>
                  <span className="text-base font-bold text-[#F5F0E6]">
                    {history[history.length - 1].global_qwk != null 
                      ? Number(history[history.length - 1].global_qwk).toFixed(3) 
                      : '---'}
                  </span>
                </div>
                <div className="bg-[#0A2035]/60 p-3 rounded-lg flex justify-between items-center border border-[#81958D]/15">
                  <span className="text-xs text-[#D0CCC3] font-medium">Macro F1 Score</span>
                  <span className="text-base font-bold text-[#F5F0E6]">
                    {history[history.length - 1].global_f1 != null 
                      ? Number(history[history.length - 1].global_f1).toFixed(3) 
                      : '---'}
                  </span>
                </div>
                <div className="bg-[#0A2035]/60 p-3 rounded-lg flex justify-between items-center border border-[#81958D]/15">
                  <span className="text-xs text-[#D0CCC3] font-medium">Macro Precision</span>
                  <span className="text-base font-bold text-[#F5F0E6]">
                    {history[history.length - 1].global_precision != null 
                      ? Number(history[history.length - 1].global_precision).toFixed(3) 
                      : '---'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-sm text-[#D0CCC3]/70">
                Awaiting first communication round results
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-[#81958D]/15 text-xs text-[#D0CCC3]">
            Saved model checkpoint:{' '}
            <code className="text-[#81958D] font-mono break-all">models/federated_global_model.pth</code>
          </div>
        </div>
      </div>

      {/* Genuine Centralized vs. Federated Experimental Comparison */}
      <div className="glass-card p-6 rounded-xl border border-[#81958D]/25">
        <div className="mb-4">
          <h3 className="text-lg font-bold text-[#F5F0E6]">Genuine Centralized vs. Federated Learning Comparison</h3>
          <p className="text-xs text-[#D0CCC3]">
            Direct experimental evaluation comparing centralized baseline (pooled dataset) against federated FedAvg aggregation on identical held-out test split
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left min-w-[620px]">
            <thead className="text-xs text-[#D0CCC3] bg-[#0A2035]/80 uppercase border-y border-[#81958D]/20">
              <tr>
                <th className="px-4 py-3">Training Paradigm</th>
                <th className="px-4 py-3">Data Location</th>
                <th className="px-4 py-3">Test Accuracy</th>
                <th className="px-4 py-3">Macro F1</th>
                <th className="px-4 py-3">QWK Score</th>
                <th className="px-4 py-3">Privacy Guarantee</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#81958D]/10">
              <tr className="hover:bg-[#191919]/30 transition-colors">
                <td className="px-4 py-3 font-medium text-[#F5F0E6] flex items-center">
                  <Server className="w-4 h-4 mr-2 text-[#81958D]" />
                  Centralized Baseline (Pooled)
                </td>
                <td className="px-4 py-3 text-[#D0CCC3]">Centralized Server</td>
                <td className="px-4 py-3 font-semibold text-[#F5F0E6]">
                  {centralized?.accuracy != null ? `${(centralized.accuracy * 100).toFixed(1)}%` : 'Not evaluated yet'}
                </td>
                <td className="px-4 py-3 text-[#D0CCC3]">
                  {centralized?.f1 != null ? Number(centralized.f1).toFixed(3) : '---'}
                </td>
                <td className="px-4 py-3 text-[#D0CCC3]">
                  {centralized?.qwk != null ? Number(centralized.qwk).toFixed(3) : '---'}
                </td>
                <td className="px-4 py-3 text-amber-400 text-xs font-medium">
                  None (Full dataset uploaded to central server)
                </td>
              </tr>
              <tr className="bg-[#191919]/25 hover:bg-[#191919]/40 transition-colors">
                <td className="px-4 py-3 font-medium text-[#81958D] flex items-center">
                  <Activity className="w-4 h-4 mr-2 text-[#81958D]" />
                  Federated Learning (FedAvg)
                </td>
                <td className="px-4 py-3 text-[#D0CCC3]">3 Local Hospital Partitions</td>
                <td className="px-4 py-3 font-semibold text-[#81958D]">
                  {federated?.accuracy != null ? `${(federated.accuracy * 100).toFixed(1)}%` : 'Not evaluated yet'}
                </td>
                <td className="px-4 py-3 text-[#D0CCC3]">
                  {federated?.f1 != null ? Number(federated.f1).toFixed(3) : '---'}
                </td>
                <td className="px-4 py-3 text-[#D0CCC3]">
                  {federated?.qwk != null ? Number(federated.qwk).toFixed(3) : '---'}
                </td>
                <td className="px-4 py-3 text-emerald-400 text-xs font-medium">
                  High (Raw images remain on local hospital nodes)
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-3 bg-[#0A2035]/60 rounded-lg text-xs text-[#D0CCC3] border border-[#81958D]/20">
          <strong className="text-[#81958D]">Academic Observation:</strong> Federated learning allows multiple institutions to collaboratively train an ophthalmology diagnostic model without centralizing sensitive retinal fundus images, achieving performance competitive with centralized training while preserving institutional data privacy.
        </div>
      </div>
    </div>
  )
}

export default FLDashboard
