import { motion, AnimatePresence } from "framer-motion"
import { Target, AlertCircle, Loader2, RefreshCw, Trophy, TrendingUp, Sparkles, Play } from "lucide-react"

interface Prediction {
  label: string
  confidence: number
  rank: number
}

interface PredictionResultsProps {
  predictions: Prediction[]
  isLoading: boolean
  error?: string | null
  onRetry?: () => void
  onRunModel?: () => void
  modelVersion?: string
  modelType?: string
  hasFrames?: boolean
}

// Animated loading spinner
function NeuralLoader() {
  return (
    <div className="flex flex-col items-center justify-center py-12">
      {/* Animated neural rings */}
      <div className="relative w-24 h-24">
        <motion.div
          className="absolute inset-0 border-2 border-primary/30 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute inset-2 border-2 border-primary/50 rounded-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute inset-4 border-2 border-primary/70 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            <Loader2 className="w-6 h-6 text-primary animate-spin" />
          </motion.div>
        </div>
      </div>

      <motion.p
        className="mt-6 text-foreground font-medium"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        Analyzing Sign Patterns
      </motion.p>

      <div className="flex items-center gap-2 mt-2">
        {["Processing", "frames", "..."].map((text, i) => (
          <motion.span
            key={i}
            className="text-xs font-mono text-muted-foreground"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 + i * 0.2 }}
          >
            {text}
          </motion.span>
        ))}
      </div>
    </div>
  )
}

// Empty state component
function EmptyState({ hasFrames, onRunModel }: { hasFrames?: boolean; onRunModel?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="w-20 h-20 rounded-2xl bg-secondary/30 flex items-center justify-center border border-border/30 mb-6"
      >
        <Target className="w-8 h-8 text-muted-foreground" />
      </motion.div>

      <p className="text-foreground font-medium mb-1">No Predictions Yet</p>
      <p className="text-sm text-muted-foreground max-w-[200px]">
        {hasFrames ? "Frames ready - run the model to analyze" : "Record or upload a video to see sign predictions"}
      </p>

      {hasFrames && onRunModel ? (
        <motion.button
          onClick={onRunModel}
          className="mt-6 flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-primary to-primary/80 text-primary-foreground font-medium glow-cyan hover:shadow-lg hover:shadow-primary/25 transition-all"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Play className="w-4 h-4" />
          <span>Run Model</span>
        </motion.button>
      ) : (
        <div className="flex items-center gap-2 mt-4 text-xs font-mono text-muted-foreground/60">
          <div className="w-1 h-1 rounded-full bg-muted-foreground/40" />
          <span>AWAITING_INPUT</span>
        </div>
      )}
    </div>
  )
}

// Error state component
function ErrorState({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="py-8">
      <div className="flex items-start gap-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
        <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center flex-shrink-0">
          <AlertCircle className="w-5 h-5 text-red-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium text-red-400 mb-1">Processing Error</p>
          <p className="text-sm text-red-400/80">{error}</p>
        </div>
      </div>

      {onRetry && (
        <motion.button
          onClick={onRetry}
          className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-secondary/50 text-foreground font-medium border border-border/50 hover:bg-secondary hover:border-border transition-all"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <RefreshCw className="w-4 h-4" />
          <span>Retry Analysis</span>
        </motion.button>
      )}
    </div>
  )
}

// Confidence bar component
function ConfidenceBar({ confidence, isTop = false }: { confidence: number; isTop?: boolean }) {
  const percentage = confidence * 100

  return (
    <div className="relative h-2 bg-secondary/50 rounded-full overflow-hidden">
      <motion.div
        className={`h-full rounded-full ${
          isTop
            ? "bg-gradient-to-r from-primary via-primary to-accent"
            : "bg-gradient-to-r from-muted-foreground/40 to-muted-foreground/20"
        }`}
        initial={{ width: 0 }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: 0.8, ease: "easeOut", delay: isTop ? 0.3 : 0.5 }}
      />
      {isTop && (
        <motion.div
          className="absolute inset-0 h-full rounded-full bg-primary/30"
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
          style={{ filter: "blur(8px)" }}
        />
      )}
    </div>
  )
}

export function PredictionResults({
  predictions,
  isLoading,
  error,
  onRetry,
  onRunModel,
  modelVersion,
  modelType,
  hasFrames,
}: PredictionResultsProps) {
  if (error) {
    return (
      <div className="glass-card rounded-2xl p-6 gradient-border">
        <ErrorState error={error} onRetry={onRetry} />
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="glass-card rounded-2xl p-6 gradient-border">
        <NeuralLoader />
      </div>
    )
  }

  if (predictions.length === 0) {
    return (
      <div className="glass-card rounded-2xl p-6 gradient-border">
        <EmptyState hasFrames={hasFrames} onRunModel={onRunModel} />
      </div>
    )
  }

  const topPrediction = predictions[0]
  const otherPredictions = predictions.slice(1, 5)

  return (
    <div className="glass-card rounded-2xl overflow-hidden gradient-border">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border/30 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <TrendingUp className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Analysis Results</h3>
            <p className="text-xs font-mono text-muted-foreground">PREDICTION_OUTPUT</p>
          </div>
        </div>

        {hasFrames && onRetry && (
          <motion.button
            onClick={onRetry}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <RefreshCw className="w-3 h-3" />
            Retry
          </motion.button>
        )}
      </div>

      {/* Top Prediction - Hero Style */}
      <AnimatePresence mode="wait">
        <motion.div
          key={topPrediction.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
          className="p-6 bg-gradient-to-br from-primary/10 via-transparent to-accent/5"
        >
          <div className="flex items-center gap-2 mb-4">
            <Trophy className="w-4 h-4 text-primary" />
            <span className="text-xs font-mono text-primary uppercase tracking-widest">
              Top Match
            </span>
          </div>

          <div className="flex items-end justify-between gap-4 mb-4">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            >
              <p className="text-5xl md:text-6xl font-display italic text-foreground text-glow-intense">
                {topPrediction.label}
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="text-right"
            >
              <p className="text-3xl font-bold text-primary text-glow">
                {(topPrediction.confidence * 100).toFixed(1)}%
              </p>
              <p className="text-xs font-mono text-muted-foreground">CONFIDENCE</p>
            </motion.div>
          </div>

          <ConfidenceBar confidence={topPrediction.confidence} isTop />
        </motion.div>
      </AnimatePresence>

      {/* Other Predictions */}
      {otherPredictions.length > 0 && (
        <div className="p-6 pt-4 border-t border-border/20">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-4">
            Alternative Matches
          </p>

          <div className="space-y-4">
            {otherPredictions.map((pred, index) => (
              <motion.div
                key={pred.label}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + index * 0.1 }}
                className="group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-md bg-secondary/50 flex items-center justify-center text-xs font-mono text-muted-foreground">
                      {pred.rank}
                    </span>
                    <span className="font-medium text-foreground/80 group-hover:text-foreground transition-colors">
                      {pred.label}
                    </span>
                  </div>
                  <span className="text-sm font-mono text-muted-foreground">
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <ConfidenceBar confidence={pred.confidence} />
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-6 py-3 border-t border-border/30 bg-secondary/10">
        <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
          <div className="flex items-center gap-2">
            <Sparkles className="w-3 h-3 text-primary" />
            <span>Powered by {modelVersion?.toUpperCase() || "V14"}</span>
          </div>
          <span className="uppercase">{modelType || "numbers"}_model</span>
        </div>
      </div>
    </div>
  )
}
