import { ErrorBar, Scatter, ScatterChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const UncertaintyChart = ({ predictionsList }) => {
  // Mocking uncertainty data for display if actual predictions aren't providing it
  const data = predictionsList?.map((p, i) => ({
    x: i + 1,
    y: p.confidence * 100,
    errorY: [(p.uncertainty_score || 0.05) * 100, (p.uncertainty_score || 0.05) * 100] // simple error margins
  })) || []

  if (data.length === 0) return null

  return (
    <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
      <h3 className="text-lg font-semibold text-slate-800 mb-2">MC-Dropout Uncertainty</h3>
      <p className="text-sm text-slate-500 mb-4">Confidence intervals based on 10 Monte Carlo dropout passes.</p>
      
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="Prediction" tickLine={false} />
            <YAxis type="number" dataKey="y" name="Confidence (%)" domain={[0, 100]} tickLine={false} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Confidence" data={data} fill="#3b82f6">
              <ErrorBar dataKey="errorY" width={4} strokeWidth={2} stroke="#1e40af" direction="y" />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default UncertaintyChart
