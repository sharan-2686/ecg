import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample

# ---------- Filters ----------
def bandpass_filter(sig, fs, low=0.5, high=40, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, freq=50, Q=30):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, sig)

# ---------- Full preprocessing ----------
def preprocess_ecg(signal, fs, scaler, target_len=5000):
    """
    signal shape: (3, N)
    output shape: (1, 5000, 3)
    """
    processed = []

    for lead in signal:
        lead = bandpass_filter(lead, fs)
        lead = notch_filter(lead, fs)

        if len(lead) != target_len:
            lead = resample(lead, target_len)

        processed.append(lead)

    # (3, 5000) → (5000, 3)
    processed = np.array(processed).T

    # normalize per lead
    processed = scaler.transform(processed)

    return processed.reshape(1, target_len, 3)
