import { useState, useEffect, useRef } from 'react'

/**
 * AnimatedBackground Component
 * Newly selected black swirling particle-vortex moving background video:
 * - 12024715 / vortex_particles.mp4 (Full-screen, continuously moving, looping, muted)
 * - Zero bright distracting lights, pure black/cosmic particle vortex
 * - Matte-black professional interface overlay
 * - Non-blocking: pointer-events: none; z-index: 0
 */
const NEW_VORTEX_SOURCES = [
  '/videos/vortex_particles.mp4',
  '/videos/12024715_1920_1080_30fps.mp4',
  '/videos/12024715_1920_1080_30fps%20(1).mp4'
]

const AnimatedBackground = () => {
  const videoRef = useRef(null)
  const [sourceIdx, setSourceIdx] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    video.muted = true
    video.defaultMuted = true
    video.playsInline = true

    const attemptPlay = () => {
      const p = video.play()
      if (p !== undefined) {
        p.then(() => setIsPlaying(true))
         .catch((err) => {
           console.warn('[AnimatedBackground] Autoplay waiting for interaction:', err)
         })
      }
    }

    attemptPlay()
    video.addEventListener('loadeddata', attemptPlay)
    video.addEventListener('canplay', attemptPlay)
    video.addEventListener('playing', () => setIsPlaying(true))

    const onUserInteraction = () => {
      if (video && video.paused) {
        video.play()
          .then(() => setIsPlaying(true))
          .catch(() => {})
      }
    }

    window.addEventListener('click', onUserInteraction, { passive: true })
    window.addEventListener('keydown', onUserInteraction, { passive: true })
    window.addEventListener('touchstart', onUserInteraction, { passive: true })

    return () => {
      video.removeEventListener('loadeddata', attemptPlay)
      video.removeEventListener('canplay', attemptPlay)
      window.removeEventListener('click', onUserInteraction)
      window.removeEventListener('keydown', onUserInteraction)
      window.removeEventListener('touchstart', onUserInteraction)
    }
  }, [sourceIdx])

  const handleVideoError = () => {
    if (sourceIdx < NEW_VORTEX_SOURCES.length - 1) {
      setSourceIdx(sourceIdx + 1)
    }
  }

  return (
    <div
      id="dr-portal-background"
      style={{
        position: 'fixed',
        inset: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
        userSelect: 'none',
        backgroundColor: '#070707',
      }}
      aria-hidden="true"
    >
      {/* ── Newly Selected Black Swirling Particle-Vortex Video ── */}
      <video
        ref={videoRef}
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        src={NEW_VORTEX_SOURCES[sourceIdx]}
        onError={handleVideoError}
        style={{
          position: 'fixed',
          inset: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          zIndex: 1,
          pointerEvents: 'none',
          opacity: 0.90,
        }}
      />

      {/* ── Subtle Dark Scrim for Content Readability Behind Charcoal Cards ── */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          zIndex: 2,
          pointerEvents: 'none',
          backgroundColor: 'rgba(7, 7, 7, 0.40)',
        }}
      />
    </div>
  )
}

export default AnimatedBackground
