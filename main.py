from fastapi import FastAPI
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from preprocessing import preprocess_ecg
from modelloader import model, scaler
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ECG Edge-AI Backend")


latest_prediction = None
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 🔴 change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- ESP32 / Sensor → Backend ----------
@app.post("/ingest")
def ingest_ecg(data: dict):
    """
    Expected JSON:
    {
        "fs": 250,
        "signal": [
            [...],  // Lead 1
            [...],  // Lead 2
            [...]   // Lead 3
        ]
    }
    """
    global latest_prediction

    fs = data["fs"]
    signal = np.array(data["signal"], dtype=np.float32)

    ecg = preprocess_ecg(signal, fs, scaler)

    preds = model.predict(ecg)[0]

    latest_prediction = {
        "Normal": float(preds[0]),
        "MI": float(preds[1]),
        "STTC": float(preds[2]),
        "CD": float(preds[3]),
        "HYP": float(preds[4])
    }

    return {"status": "ECG received successfully"}

# ---------- Frontend → Backend ----------
@app.get("/latest_prediction")
def get_latest_prediction():
    if latest_prediction is None:
        return {"status": "waiting for ECG data"}
    return latest_prediction

# ---------- Health Check ----------
@app.get("/")
def root():
    return {"status": "ECG backend running"}
