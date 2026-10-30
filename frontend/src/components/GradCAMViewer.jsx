import { useState } from 'react'

const GradCAMViewer = ({ originalUrl, heatmapUrl, originalImage, gradcamImage }) => {
  const [opacity, setOpacity] = useState(0.5)

  const orig = originalUrl || originalImage
  const heat = heatmapUrl || gradcamImage

  if (!orig || !heat) return null

  return (
    <div className="bg-[#121212]/95 backdrop-blur-md p-5 rounded-xl border border-[#292929] shadow-md">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-[#292929]">
        <h3 className="text-base font-bold text-[#F5F0E6]">Grad-CAM Heatmap Visualization</h3>
        <span className="text-[11px] font-medium text-[#81958D] bg-[#191919] px-2 py-0.5 rounded border border-[#292929]">Explainable AI</span>
      </div>

      <div className="relative w-full aspect-square bg-black rounded-lg overflow-hidden group max-w-md mx-auto border border-[#292929]">
        <img 
          src={orig} 
          alt="Original Retinal Fundus" 
          className="absolute inset-0 w-full h-full object-contain"
        />
        <img 
          src={heat} 
          alt="Grad-CAM Activation" 
          className="absolute inset-0 w-full h-full object-contain transition-opacity"
          style={{ opacity }}
        />
      </div>
      
      <div className="mt-5 px-4 max-w-md mx-auto">
        <div className="flex justify-between items-center mb-2">
          <label className="text-xs font-semibold text-[#D0CCC3]">
            Heatmap Blend Opacity
          </label>
          <span className="text-xs font-mono text-[#81958D]">{Math.round(opacity * 100)}%</span>
        </div>
        <input 
          type="range" 
          min="0" 
          max="1" 
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-[#191919] rounded-lg appearance-none cursor-pointer accent-[#81958D]"
        />
        <div className="flex justify-between text-[11px] text-[#85817A] mt-1.5">
          <span>Natural Retinal Fundus</span>
          <span>Saliency Overlay</span>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-center space-x-2 text-[11px] text-[#85817A]">
        <span>Low Activation</span>
        <div className="w-32 h-2.5 bg-gradient-to-r from-blue-900 via-emerald-600 via-amber-500 to-rose-600 rounded"></div>
        <span>High Activation</span>
      </div>
      
      <div className="mt-5 flex justify-center">
        <a 
          href={heat} 
          download="gradcam_heatmap.jpg"
          className="px-4 py-2 bg-[#191919] text-[#F5F0E6] hover:bg-[#222222] rounded-lg text-xs font-medium transition-colors border border-[#292929]"
        >
          Download Heatmap Image
        </a>
      </div>
    </div>
  )
}

export default GradCAMViewer
