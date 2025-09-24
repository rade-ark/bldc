import numpy as np
import librosa
from .config import SR, N_MELS, HOP_LENGTH, N_FFT, DURATION

def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    else:
        y = y[:target_len]
    return y

def compute_mel(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
