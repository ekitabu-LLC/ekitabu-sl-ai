import { useState, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ModelSelector } from "@/components/ModelSelector"
import { WebcamCapture } from "@/components/WebcamCapture"
import { PredictionResults } from "@/components/PredictionResults"
import { predictSign } from "@/services/api"
import { Sparkles, Zap, Activity, Brain } from "lucide-react"

interface Prediction {
  label: string
  confidence: number
  rank: number
}

// Animated neural network background
function NeuralBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      {/* Gradient orbs */}
      <div className="absolute top-1/4 -left-1/4 w-[800px] h-[800px] rounded-full bg-[hsl(174_72%_56%/0.08)] blur-[120px] animate-pulse" />
      <div className="absolute bottom-1/4 -right-1/4 w-[600px] h-[600px] rounded-full bg-[hsl(280_100%_70%/0.05)] blur-[100px] animate-pulse animation-delay-500" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] rounded-full bg-[hsl(220_20%_8%)] blur-[80px]" />

      {/* Grid pattern */}
      <svg className="absolute inset-0 w-full h-full opacity-[0.03]" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
            <path d="M 60 0 L 0 0 0 60" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-primary" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>

      {/* Floating particles */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 rounded-full bg-primary/30"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.3, 0.8, 0.3],
          }}
          transition={{
            duration: 3 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  )
}

// Status indicator component
function StatusIndicator({ isConnected }: { isConnected: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-amber-400'}`} />
        {isConnected && (
          <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
        )}
      </div>
      <span className="text-xs font-mono text-muted-foreground">
        {isConnected ? 'NEURAL_LINK_ACTIVE' : 'CONNECTING...'}
      </span>
    </div>
  )
}

function App() {
  const [modelVersion, setModelVersion] = useState("v14")
  const [modelType, setModelType] = useState("numbers")
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [capturedFrames, setCapturedFrames] = useState<string[]>([])
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    // Simulate connection check
    const timer = setTimeout(() => setIsConnected(true), 1500)
    return () => clearTimeout(timer)
  }, [])

  const handleFramesCapture = useCallback(
    async (frames: string[]) => {
      setCapturedFrames(frames)
      setIsLoading(true)
      setError(null)
      setPredictions([])

      try {
        const result = await predictSign(frames, modelVersion, modelType)
        setPredictions(result.predictions)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to get prediction")
      } finally {
        setIsLoading(false)
      }
    },
    [modelVersion, modelType]
  )

  const handleRetry = useCallback(() => {
    if (capturedFrames.length > 0) {
      handleFramesCapture(capturedFrames)
    }
  }, [capturedFrames, handleFramesCapture])

  return (
    <div className="min-h-screen neural-bg">
      <NeuralBackground />

      {/* Noise overlay */}
      <div className="noise-overlay" />

      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative z-10 border-b border-border/30 backdrop-blur-xl bg-background/30"
      >
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Logo */}
              <motion.div
                className="relative"
                whileHover={{ scale: 1.05 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center glow-cyan">
                  <Brain className="w-6 h-6 text-primary" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-primary animate-pulse" />
              </motion.div>

              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  <span className="font-display italic text-gradient-cyan">KSL</span>
                  <span className="text-foreground/80 font-light ml-2">Neural</span>
                </h1>
                <p className="text-xs font-mono text-muted-foreground tracking-widest uppercase">
                  Sign Language Recognition Engine
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <StatusIndicator isConnected={isConnected} />

              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/50 border border-border/50">
                <Activity className="w-3 h-3 text-primary" />
                <span className="text-xs font-mono text-muted-foreground">
                  {modelVersion.toUpperCase()}_{modelType.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-10">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">ST-GCN Architecture Powered</span>
          </div>

          <h2 className="text-4xl md:text-5xl lg:text-6xl font-display italic text-foreground mb-4">
            Recognize Signs with
            <br />
            <span className="text-gradient">Neural Precision</span>
          </h2>

          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Advanced Kenya Sign Language recognition powered by
            Spatial-Temporal Graph Convolutional Networks
          </p>
        </motion.div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-5 gap-8">
          {/* Left Column - Controls */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Model Configuration Card */}
            <div className="glass-card rounded-2xl p-6 gradient-border">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Model Configuration</h3>
                  <p className="text-xs text-muted-foreground font-mono">SELECT_NEURAL_PATHWAY</p>
                </div>
              </div>

              <ModelSelector
                version={modelVersion}
                modelType={modelType}
                onVersionChange={setModelVersion}
                onTypeChange={setModelType}
              />
            </div>

            {/* Results Card */}
            <AnimatePresence mode="wait">
              <motion.div
                key={predictions.length > 0 ? 'results' : 'empty'}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3 }}
              >
                <PredictionResults
                  predictions={predictions}
                  isLoading={isLoading}
                  error={error}
                  onRetry={handleRetry}
                  onRunModel={handleRetry}
                  modelVersion={modelVersion}
                  modelType={modelType}
                  hasFrames={capturedFrames.length > 0}
                />
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Right Column - Video Capture */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="lg:col-span-3"
          >
            <WebcamCapture
              onFramesCapture={handleFramesCapture}
              isRecording={isRecording}
              setIsRecording={setIsRecording}
            />

            {/* Tech specs */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="mt-6 grid grid-cols-3 gap-4"
            >
              {[
                { label: "Frame Rate", value: "30 FPS" },
                { label: "Max Duration", value: "5 SEC" },
                { label: "Resolution", value: "640×480" },
              ].map((spec) => (
                <div
                  key={spec.label}
                  className="text-center p-3 rounded-xl bg-secondary/30 border border-border/30"
                >
                  <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-1">
                    {spec.label}
                  </p>
                  <p className="text-sm font-semibold text-foreground">{spec.value}</p>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>

        {/* Footer Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-16 pt-8 border-t border-border/30"
        >
          <div className="flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-muted-foreground">
            <div className="flex items-center gap-6">
              <span>KSL_NEURAL_v2.0</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">SPATIAL_TEMPORAL_GRAPH_CONV</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              <span>SYSTEM_OPERATIONAL</span>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

export default App
