import { motion } from "framer-motion"
import { Hash, Type, ChevronDown, Check, Cpu, Sparkles } from "lucide-react"

interface ModelVersion {
  id: string
  name: string
  architecture: string
  description: string
  isRecommended?: boolean
}

const modelVersions: ModelVersion[] = [
  { id: "ensemble_6_uniform", name: "6-Model Ensemble", architecture: "ST-GCN Multi-Stream", description: "Best overall: 72.9% combined", isRecommended: true },
  { id: "v43", name: "V43", architecture: "ST-GCN + SupCon + R&R", description: "Best single model: 65.7% combined" },
  { id: "v41", name: "V41", architecture: "ST-GCN GroupNorm + R&R", description: "Best numbers: 67.8%" },
  { id: "v37", name: "V37", architecture: "ST-GCN GroupNorm + Speed Aug", description: "57.9% combined" },
  { id: "exp5", name: "Exp5", architecture: "ST-GCN + SupCon", description: "Best words: 65.4%" },
  { id: "exp1", name: "Exp1", architecture: "ST-GCN GroupNorm", description: "Baseline: 61.4%" },
]

const modelTypes = [
  { id: "numbers", name: "Numbers", icon: Hash, count: 15, description: "Numeric signs (9, 17, 22...)" },
  { id: "words", name: "Words", icon: Type, count: 15, description: "Word signs (Apple, Gift...)" },
]

interface ModelSelectorProps {
  version: string
  modelType: string
  onVersionChange: (value: string) => void
  onTypeChange: (value: string) => void
}

export function ModelSelector({ version, modelType, onVersionChange, onTypeChange }: ModelSelectorProps) {
  const selectedModel = modelVersions.find((m) => m.id === version)

  return (
    <div className="space-y-6">
      {/* Model Type Selection - Card Style */}
      <div className="space-y-3">
        <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          Sign Category
        </label>
        <div className="grid grid-cols-2 gap-3">
          {modelTypes.map((type) => {
            const Icon = type.icon
            const isSelected = modelType === type.id

            return (
              <motion.button
                key={type.id}
                onClick={() => onTypeChange(type.id)}
                className={`relative p-4 rounded-xl text-left transition-all duration-300 ${
                  isSelected
                    ? "bg-primary/10 border-primary/50"
                    : "bg-secondary/30 border-border/30 hover:bg-secondary/50 hover:border-border/50"
                } border`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isSelected && (
                  <motion.div
                    layoutId="typeIndicator"
                    className="absolute inset-0 rounded-xl bg-primary/10 border border-primary/30"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-2">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                      isSelected ? "bg-primary/20" : "bg-muted"
                    }`}>
                      <Icon className={`w-4 h-4 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
                    </div>
                    {isSelected && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="w-5 h-5 rounded-full bg-primary flex items-center justify-center"
                      >
                        <Check className="w-3 h-3 text-primary-foreground" />
                      </motion.div>
                    )}
                  </div>

                  <p className={`font-semibold ${isSelected ? "text-foreground" : "text-foreground/80"}`}>
                    {type.name}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {type.count} classes
                  </p>
                </div>
              </motion.button>
            )
          })}
        </div>
      </div>

      {/* Model Version Selection - Dropdown Style */}
      <div className="space-y-3">
        <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
          Neural Architecture
        </label>

        <div className="relative">
          <select
            value={version}
            onChange={(e) => onVersionChange(e.target.value)}
            className="w-full appearance-none bg-secondary/30 border border-border/50 rounded-xl px-4 py-3.5 pr-10 text-foreground font-medium cursor-pointer hover:bg-secondary/50 hover:border-border transition-all focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50"
          >
            {modelVersions.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} — {model.architecture} {model.isRecommended ? "★" : ""}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
        </div>

        {/* Selected Model Info */}
        {selectedModel && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            key={selectedModel.id}
            className="flex items-start gap-3 p-3 rounded-lg bg-secondary/20 border border-border/30"
          >
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
              <Cpu className="w-4 h-4 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-foreground">{selectedModel.architecture}</span>
                {selectedModel.isRecommended && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary/20 text-primary text-[10px] font-medium">
                    <Sparkles className="w-3 h-3" />
                    RECOMMENDED
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">{selectedModel.description}</p>
            </div>
          </motion.div>
        )}
      </div>

      {/* Quick Stats */}
      <div className="pt-4 border-t border-border/30">
        <div className="flex items-center justify-between text-xs">
          <span className="font-mono text-muted-foreground">ACTIVE_CONFIG</span>
          <span className="font-mono text-primary">
            {version.toUpperCase()}_{modelType.toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  )
}
