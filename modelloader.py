import os
import gdown
import pickle
from tensorflow.keras.models import load_model

# 🔁 Google Drive FILE IDs
MODEL_ID = "1Thi5PwfKpedNM26-A6Wgx_rsePNafDRK"
SCALER_ID = "1PUh9gklDiWIgBzKBv0z9zgDsLshNn3cf"

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

# Download model
if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_ID}",
        MODEL_PATH,
        quiet=False
    )

# Download scaler
if not os.path.exists(SCALER_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={SCALER_ID}",
        SCALER_PATH,
        quiet=False
    )

# Load them
model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
