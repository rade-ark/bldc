from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Audio params
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
DURATION = 2.0  # seconds (trim/pad)

TRAIN_VAL_SPLIT = 0.8
BATCH_SIZE = 32
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

SEED = 42
