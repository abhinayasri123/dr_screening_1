import { AlertTriangle } from 'lucide-react'

const DisclaimerBanner = () => {
  return (
    <div className="bg-[#121212]/95 backdrop-blur-md border-b border-[#292929] text-[#D0CCC3] px-4 py-2 flex items-center justify-center text-xs md:text-sm font-medium sticky top-0 z-50 shadow-sm">
      <AlertTriangle className="w-4 h-4 text-[#81958D] mr-2 flex-shrink-0" />
      <span>AI Clinical Decision Support &mdash; NOT a replacement for professional medical diagnosis. Final decisions must be made by qualified healthcare professionals.</span>
    </div>
  )
}

export default DisclaimerBanner
