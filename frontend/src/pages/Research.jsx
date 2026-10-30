import MetricsPanel from '../components/MetricsPanel'
import { LineChart, BookOpen, Layers, ShieldCheck } from 'lucide-react'

const Research = () => {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex items-center mb-6">
        <LineChart className="w-8 h-8 text-primary-600 mr-3" />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Research & Model Metrics</h1>
          <p className="text-sm text-slate-500">Empirical validation metrics computed on held-out fundus test partitions</p>
        </div>
      </div>

      <MetricsPanel />

      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm mt-8 space-y-4">
        <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
          <Layers className="w-5 h-5 text-primary-600" />
          Model Architecture & Experimental Methodology
        </h2>
        
        <p className="text-sm text-slate-600 leading-relaxed">
          The deep learning pipeline utilizes a transfer-learning convolutional neural network (<code className="text-slate-800 bg-slate-100 px-1 py-0.5 rounded">EfficientNet-B3</code>) trained with categorical cross-entropy and inverse class-frequency weighting to address Diabetic Retinopathy stage imbalance.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
            <h4 className="font-semibold text-slate-800 text-sm mb-1">Radiographic CNN</h4>
            <p className="text-xs text-slate-500">
              EfficientNet-B3 backbone pretrained on ImageNet, adapted with custom multi-stage classifier head for 5 DR severity grades.
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
            <h4 className="font-semibold text-slate-800 text-sm mb-1">Uncertainty Estimation</h4>
            <p className="text-xs text-slate-500">
              Monte Carlo Dropout with 10 stochastic forward passes at inference to approximate Bayesian posterior variance for high-risk clinical cases.
            </p>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
            <h4 className="font-semibold text-slate-800 text-sm mb-1">Explainability (XAI)</h4>
            <p className="text-xs text-slate-500">
              Gradient-weighted Class Activation Mapping (Grad-CAM) highlights lesion patterns and microaneurysms for clinician review.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Research
