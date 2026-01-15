import { useRef, useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Video, VideoOff, Camera, Square, Upload, Scan, Circle, Play, X } from "lucide-react"

interface WebcamCaptureProps {
  onFramesCapture: (frames: string[]) => void
  isRecording: boolean
  setIsRecording: (recording: boolean) => void
}

export function WebcamCapture({ onFramesCapture, isRecording, setIsRecording }: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const previewVideoRef = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [recordedFrames, setRecordedFrames] = useState<string[]>([])
  const [frameCount, setFrameCount] = useState(0)
  const [recordingProgress, setRecordingProgress] = useState(0)
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null)
  const [isProcessingUpload, setIsProcessingUpload] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const recordingRef = useRef<number | null>(null)
  const framesRef = useRef<string[]>([])

  const startCamera = useCallback(async () => {
    // Clear any uploaded video preview
    if (uploadedVideoUrl) {
      URL.revokeObjectURL(uploadedVideoUrl)
      setUploadedVideoUrl(null)
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      })

      setStream(mediaStream)
      setIsCameraOn(true)

      // Set video source after state update
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
          videoRef.current.play().catch(console.error)
        }
      }, 100)
    } catch (err) {
      console.error("Error accessing camera:", err)
      alert("Could not access camera. Please check permissions.")
    }
  }, [uploadedVideoUrl])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
    setIsRecording(false)
    setRecordingProgress(0)
    setFrameCount(0)
    framesRef.current = []
    if (recordingRef.current) {
      cancelAnimationFrame(recordingRef.current)
      recordingRef.current = null
    }
  }, [stream, setIsRecording])

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null
    const canvas = canvasRef.current
    const video = videoRef.current

    if (video.readyState < 2 || video.videoWidth === 0) return null

    const ctx = canvas.getContext("2d")
    if (!ctx) return null

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)
    return canvas.toDataURL("image/jpeg", 0.8)
  }, [])

  const startRecording = useCallback(() => {
    framesRef.current = []
    setRecordedFrames([])
    setFrameCount(0)
    setRecordingProgress(0)
    setIsRecording(true)

    let count = 0
    const targetFps = 30
    const maxFrames = 150
    const frameDuration = 1000 / targetFps
    let lastTime = performance.now()

    const recordFrame = (currentTime: number) => {
      if (currentTime - lastTime >= frameDuration) {
        const frame = captureFrame()
        if (frame) {
          framesRef.current.push(frame)
          count++
          setFrameCount(count)
          setRecordingProgress((count / maxFrames) * 100)
        }
        lastTime = currentTime
      }

      if (count < maxFrames) {
        recordingRef.current = requestAnimationFrame(recordFrame)
      } else {
        setIsRecording(false)
        setRecordedFrames([...framesRef.current])
        onFramesCapture([...framesRef.current])
        recordingRef.current = null
      }
    }

    recordingRef.current = requestAnimationFrame(recordFrame)
  }, [captureFrame, onFramesCapture, setIsRecording])

  const stopRecording = useCallback(() => {
    if (recordingRef.current) {
      cancelAnimationFrame(recordingRef.current)
      recordingRef.current = null
    }

    setIsRecording(false)

    const capturedFrames = [...framesRef.current]
    setRecordedFrames(capturedFrames)

    if (capturedFrames.length >= 10) {
      onFramesCapture(capturedFrames)
    }
  }, [onFramesCapture, setIsRecording])

  const clearUploadedVideo = useCallback(() => {
    if (uploadedVideoUrl) {
      URL.revokeObjectURL(uploadedVideoUrl)
      setUploadedVideoUrl(null)
    }
    setRecordedFrames([])
    setFrameCount(0)
  }, [uploadedVideoUrl])

  const processVideoFile = useCallback(
    async (file: File) => {
      // Check if it's a video file
      if (!file.type.startsWith("video/")) {
        alert("Please drop a video file")
        return
      }

      // Stop camera if on
      if (isCameraOn) {
        stopCamera()
      }

      // Clear previous upload
      if (uploadedVideoUrl) {
        URL.revokeObjectURL(uploadedVideoUrl)
      }

      setIsProcessingUpload(true)

      const videoUrl = URL.createObjectURL(file)
      setUploadedVideoUrl(videoUrl)

      const video = document.createElement("video")
      video.src = videoUrl
      video.muted = true

      video.onloadedmetadata = async () => {
        const canvas = document.createElement("canvas")
        const ctx = canvas.getContext("2d")
        if (!ctx) {
          setIsProcessingUpload(false)
          return
        }

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        const frames: string[] = []
        const duration = video.duration
        const targetFrames = Math.min(Math.floor(duration * 30), 150)
        const interval = duration / targetFrames

        for (let i = 0; i < targetFrames; i++) {
          video.currentTime = i * interval
          await new Promise((resolve) => {
            video.onseeked = resolve
          })
          ctx.drawImage(video, 0, 0)
          frames.push(canvas.toDataURL("image/jpeg", 0.8))
        }

        framesRef.current = frames
        setRecordedFrames(frames)
        setFrameCount(frames.length)
        setIsProcessingUpload(false)
        onFramesCapture(frames)
      }

      video.onerror = () => {
        setIsProcessingUpload(false)
        alert("Error loading video file")
      }

      video.load()
    },
    [isCameraOn, stopCamera, uploadedVideoUrl, onFramesCapture]
  )

  const handleVideoUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return

      await processVideoFile(file)

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    },
    [processVideoFile]
  )

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    async (event: React.DragEvent) => {
      event.preventDefault()
      event.stopPropagation()
      setIsDragging(false)

      const file = event.dataTransfer.files?.[0]
      if (file) {
        await processVideoFile(file)
      }
    },
    [processVideoFile]
  )

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
      if (recordingRef.current) {
        cancelAnimationFrame(recordingRef.current)
      }
      if (uploadedVideoUrl) {
        URL.revokeObjectURL(uploadedVideoUrl)
      }
    }
  }, [stream, uploadedVideoUrl])

  // Determine what to show in the video area
  const showCamera = isCameraOn && !uploadedVideoUrl
  const showUploadedVideo = uploadedVideoUrl && !isCameraOn
  const showPlaceholder = !isCameraOn && !uploadedVideoUrl

  return (
    <div className="glass-card rounded-2xl overflow-hidden gradient-border">
      {/* Video Container */}
      <div
        className={`relative aspect-video bg-[hsl(220_25%_6%)] overflow-hidden transition-all duration-200 ${
          isDragging ? "ring-2 ring-primary ring-inset" : ""
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Live Camera Feed */}
        {showCamera && (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover z-0"
          />
        )}

        {/* Uploaded Video Preview */}
        {showUploadedVideo && (
          <video
            ref={previewVideoRef}
            src={uploadedVideoUrl}
            controls
            loop
            className="absolute inset-0 w-full h-full object-cover z-0"
          />
        )}

        {/* Camera Overlays */}
        {showCamera && (
          <>
            {/* Scan overlay */}
            <div className="absolute inset-0 pointer-events-none z-10">
              {/* Corner brackets */}
              <div className="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-primary/50" />
              <div className="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-primary/50" />
              <div className="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-primary/50" />
              <div className="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-primary/50" />

              {/* Center crosshair */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                <div className="w-16 h-16 border border-primary/30 rounded-full" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-8 bg-primary/30" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-1 bg-primary/30" />
              </div>
            </div>

            {/* Scan line animation when recording */}
            {isRecording && <div className="scan-line z-10" />}

            {/* Vignette overlay */}
            <div className="absolute inset-0 pointer-events-none z-10 bg-gradient-to-t from-background/30 via-transparent to-transparent" />
          </>
        )}

        {/* Uploaded Video Overlay */}
        {showUploadedVideo && (
          <>
            {/* Close button */}
            <button
              onClick={clearUploadedVideo}
              className="absolute top-4 left-4 z-20 p-2 rounded-lg bg-background/80 backdrop-blur-md border border-border/50 hover:bg-background transition-colors"
            >
              <X className="w-4 h-4 text-foreground" />
            </button>

            {/* Video badge */}
            <div className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-background/60 backdrop-blur-md border border-primary/30">
              <Play className="w-3 h-3 text-primary fill-primary" />
              <span className="text-xs font-mono text-primary">UPLOADED</span>
            </div>
          </>
        )}

        {/* Placeholder when nothing is active */}
        {showPlaceholder && !isDragging && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="relative"
            >
              <div className="w-24 h-24 rounded-2xl bg-secondary/30 flex items-center justify-center border border-border/30">
                <Camera className="w-10 h-10 text-muted-foreground" />
              </div>

              <motion.div
                className="absolute inset-0 rounded-2xl border border-primary/20"
                animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </motion.div>

            <p className="mt-6 text-muted-foreground font-medium">Camera Offline</p>
            <p className="text-sm text-muted-foreground/60 mt-1">Click below to enable or drop a video</p>
          </div>
        )}

        {/* Drag overlay */}
        <AnimatePresence>
          {isDragging && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-background/90 backdrop-blur-sm"
            >
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                className="w-24 h-24 rounded-2xl bg-primary/20 flex items-center justify-center border-2 border-dashed border-primary"
              >
                <Upload className="w-10 h-10 text-primary" />
              </motion.div>
              <p className="mt-6 text-foreground font-medium">Drop video here</p>
              <p className="text-sm text-muted-foreground mt-1">Release to upload</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Recording indicator */}
        <AnimatePresence>
          {isRecording && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-4 left-4 right-4 z-20"
            >
              <div className="flex items-center justify-between bg-background/80 backdrop-blur-md rounded-xl px-4 py-3 border border-red-500/30">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <Circle className="w-4 h-4 text-red-500 fill-red-500" />
                    <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-50" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-foreground">Recording</p>
                    <p className="text-xs text-muted-foreground font-mono">{frameCount} frames captured</p>
                  </div>
                </div>

                <div className="text-right">
                  <p className="text-lg font-mono font-bold text-red-400">
                    {Math.round(recordingProgress)}%
                  </p>
                </div>
              </div>

              <div className="mt-2 h-1 bg-secondary/50 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-red-500 to-red-400"
                  initial={{ width: 0 }}
                  animate={{ width: `${recordingProgress}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Processing upload indicator */}
        <AnimatePresence>
          {isProcessingUpload && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-background/80 backdrop-blur-md"
            >
              <div className="w-12 h-12 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              <p className="mt-4 text-foreground font-medium">Processing video...</p>
              <p className="text-sm text-muted-foreground">Extracting frames</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Status badge for live camera */}
        {showCamera && !isRecording && (
          <div className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-background/60 backdrop-blur-md border border-emerald-500/30">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-mono text-emerald-400">LIVE</span>
          </div>
        )}

        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Controls */}
      <div className="p-4 bg-secondary/20 border-t border-border/30">
        <div className="flex flex-wrap gap-3">
          {/* Camera toggle */}
          {!isCameraOn ? (
            <motion.button
              onClick={startCamera}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Video className="w-4 h-4" />
              <span>Enable Camera</span>
            </motion.button>
          ) : (
            <motion.button
              onClick={stopCamera}
              className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-secondary/50 text-foreground font-medium border border-border/50 hover:bg-secondary hover:border-border transition-all"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <VideoOff className="w-4 h-4" />
              <span>Disable</span>
            </motion.button>
          )}

          {/* Record button */}
          {isCameraOn && !isRecording && (
            <motion.button
              onClick={startRecording}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-r from-primary to-primary/80 text-primary-foreground font-medium glow-cyan hover:shadow-lg hover:shadow-primary/25 transition-all"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Scan className="w-4 h-4" />
              <span>Capture Sign</span>
            </motion.button>
          )}

          {/* Stop recording button */}
          {isRecording && (
            <motion.button
              onClick={stopRecording}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-red-500 text-white font-medium hover:bg-red-600 transition-colors"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
            >
              <Square className="w-4 h-4" />
              <span>Stop Recording</span>
            </motion.button>
          )}

          {/* Upload button */}
          <motion.button
            onClick={() => fileInputRef.current?.click()}
            disabled={isRecording || isProcessingUpload}
            className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-secondary/50 text-foreground font-medium border border-border/50 hover:bg-secondary hover:border-border transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            whileHover={{ scale: isRecording || isProcessingUpload ? 1 : 1.02 }}
            whileTap={{ scale: isRecording || isProcessingUpload ? 1 : 0.98 }}
          >
            <Upload className="w-4 h-4" />
            <span>Upload</span>
          </motion.button>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleVideoUpload}
            className="hidden"
          />
        </div>

        {/* Captured frames info */}
        <AnimatePresence>
          {recordedFrames.length > 0 && !isRecording && !isProcessingUpload && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-3 rounded-xl bg-primary/10 border border-primary/20"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary" />
                  <span className="text-sm text-foreground font-medium">
                    {recordedFrames.length} frames ready
                  </span>
                </div>
                <span className="text-xs font-mono text-primary">
                  BUFFER_LOADED
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
