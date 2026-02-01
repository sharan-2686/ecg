from fastapi import FastAPI
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from preprocessing import preprocess_ecg

app = FastAPI(title="ECG Edge-AI Backend")

# ---------- Load model & scaler ----------
model = load_model("model/model.h5")
scaler = joblib.load("model/scaler.pkl")

latest_prediction = None

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
