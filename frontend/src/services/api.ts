const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

interface PredictionResult {
  label: string
  confidence: number
  rank: number
}

interface PredictResponse {
  predictions: PredictionResult[]
  model_version: string
  processing_time_ms: number
}

export async function predictSign(
  video: Blob,
  modelVersion: string,
  modelType: string = "numbers"
): Promise<PredictResponse> {
  const form = new FormData()
  form.append("video", video, "sign.webm")
  form.append("model_version", modelVersion)
  form.append("model_type", modelType)

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: form,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }))
    throw new Error(error.detail || `HTTP error ${response.status}`)
  }

  return response.json()
}

export async function getAvailableModels(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/models`)

  if (!response.ok) {
    throw new Error("Failed to fetch available models")
  }

  const data = await response.json()
  return data.models
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
