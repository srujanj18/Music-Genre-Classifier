import os
import librosa
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def audio_to_mel(path, sr=22050, duration=30, n_mels=128, hop_length=512, n_fft=2048):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.shape[0] < sr * duration:
        y = np.pad(y, (0, max(0, sr*duration - y.shape[0])))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return S_db.astype(np.float32)
