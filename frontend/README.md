# KSL Model Testing Frontend

A React-based frontend for testing Kuwaiti Sign Language (KSL) recognition models.

## Features

- **Model Version Selection**: Dropdown to select different model versions (v7-v14)
- **Webcam Recording**: Record signs directly from your webcam
- **Video Upload**: Upload pre-recorded video files for testing
- **Real-time Predictions**: View prediction results with confidence scores
- **Responsive Design**: Works on desktop and mobile devices

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+ (for backend)

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup (Mock Server)

```bash
cd frontend/backend
pip install -r requirements.txt
python server.py
```

The API will be available at `http://localhost:8000`

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # Reusable UI components
│   │   ├── ModelSelector.tsx
│   │   ├── WebcamCapture.tsx
│   │   └── PredictionResults.tsx
│   ├── services/
│   │   └── api.ts           # API client
│   ├── lib/
│   │   └── utils.ts         # Utility functions
│   ├── App.tsx
│   └── index.css
├── backend/
│   ├── server.py            # Mock FastAPI server
│   └── requirements.txt
└── package.json
```

## Usage

1. Start both the frontend and backend servers
2. Select a model version from the dropdown
3. Either:
   - Click "Start Camera" and then "Record Sign" to capture a sign
   - Click "Upload Video" to upload an existing video
4. View the prediction results on the right side

## API Endpoints

- `GET /health` - Health check
- `GET /models` - List available model versions
- `POST /predict` - Make a prediction
  - Body: `{ frames: string[], model_version: string }`
  - Response: `{ predictions: [...], model_version: string, processing_time_ms: number }`

## Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```
