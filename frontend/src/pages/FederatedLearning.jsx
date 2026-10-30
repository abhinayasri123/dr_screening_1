import FLDashboard from '../components/FLDashboard'
import { Network, ShieldCheck, Cpu } from 'lucide-react'

const FederatedLearning = () => {
  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-10">
      <div>
        <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-[#191919] text-[#81958D] border border-[#292929] mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#81958D]"></span>
          <span>Federated Learning Network</span>
        </div>
        <h1 className="text-2xl font-bold text-[#F5F0E6] mb-1">Federated Learning Hub</h1>
        <p className="text-xs text-[#D0CCC3]">
          Privacy-preserving multi-institutional collaborative training for Diabetic Retinopathy screening
        </p>
      </div>

      <FLDashboard />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        <div className="bg-[#121212]/95 backdrop-blur-md p-6 rounded-xl border border-[#292929] shadow-sm">
          <ShieldCheck className="w-7 h-7 text-[#81958D] mb-3" />
          <h3 className="text-base font-bold text-[#F5F0E6] mb-2">Institutional Privacy Guarantee</h3>
          <p className="text-xs text-[#D0CCC3] leading-relaxed">
            Patient retinal images remain isolated within each local hospital node partition. Only model weight updates (tensors) are communicated to the central coordinator during federated rounds.
          </p>
          <div className="mt-3 p-2.5 bg-[#191919] rounded-lg border border-[#292929] text-[11px] text-[#85817A]">
            <strong className="text-[#F5F0E6]">Privacy Note:</strong> Multi-hospital federated training using partitioned retinal cohorts. No patient health records transfer across institutional boundaries.
          </div>
        </div>

        <div className="bg-[#121212]/95 backdrop-blur-md p-6 rounded-xl border border-[#292929] shadow-sm">
          <Network className="w-7 h-7 text-[#81958D] mb-3" />
          <h3 className="text-base font-bold text-[#F5F0E6] mb-2">Federated Averaging (FedAvg)</h3>
          <p className="text-xs text-[#D0CCC3] leading-relaxed mb-3">
            In each communication round, the server aggregates local client weights via sample-weighted parameter averaging:
          </p>
          <div className="bg-[#0F0F0F] border border-[#292929] text-[#F5F0E6] p-2.5 rounded-lg text-xs font-mono">
            W_global = &Sigma; (n_k / N) &middot; W_k
          </div>
          <p className="text-[11px] text-[#85817A] mt-2">
            Where n_k is the number of local fundus images at hospital node k, and N is total training images across all nodes.
          </p>
        </div>

        <div className="bg-[#121212]/95 backdrop-blur-md p-6 rounded-xl border border-[#292929] shadow-sm">
          <Cpu className="w-7 h-7 text-[#81958D] mb-3" />
          <h3 className="text-base font-bold text-[#F5F0E6] mb-2">Model Deployment & Persistence</h3>
          <p className="text-xs text-[#D0CCC3] leading-relaxed mb-3">
            The aggregated global checkpoint is saved to <code className="bg-[#0F0F0F] text-[#81958D] px-1.5 py-0.5 rounded font-mono text-[11px] border border-[#292929]">models/federated_global_model.pth</code>.
          </p>
          <p className="text-xs text-[#D0CCC3] leading-relaxed">
            The trained global model is directly integrated into the clinician inference pipeline for instant screening in the <em>New Screening</em> section.
          </p>
        </div>
      </div>
    </div>
  )
}

export default FederatedLearning
