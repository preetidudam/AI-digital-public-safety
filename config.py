import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Identifiers (Hugging Face)
ICFD_DATASET_ID = "rishia2220/icfd-31k"
SMS_DATASET_ID = "anmolshrivastav/scam-ham-india"

# Saved Model Paths
SMS_BASELINE_MODEL_PATH = MODELS_DIR / "sms_scam_baseline.joblib"
ICFD_BASELINE_MODEL_PATH = MODELS_DIR / "digital_scam_transcript_model.joblib"
ICFD_CONVERSATION_MODEL_PATH = MODELS_DIR / "scam_detector.joblib"
IMPROVED_MODEL_PATH = MODELS_DIR / "scam_detector_improved.joblib"

# Deepfake Audio Model (Hugging Face)
DEEPFAKE_AUDIO_MODEL_ID = "Vansh180/deepfake-audio-wav2vec2"

# Whisper Speech-to-Text Model
WHISPER_MODEL_NAME = "small"

# Sample Audio Path
DEFAULT_SAMPLE_AUDIO_PATH = DATA_DIR / "sample_scam_call.mp3"

# Model Thresholds
DEFAULT_CONVERSATION_THRESHOLD = 0.20
DEFAULT_IMPROVED_THRESHOLD = 0.30

# Random State
RANDOM_SEED = 42

# Label Mappings
SMS_LABEL_MAP = {
    "ham": 0,
    "legitimate": 0,
    "legit": 0,
    "spam": 1,
    "scam": 1,
    "fraud": 1,
}

ICFD_LABEL_MAP = {
    "Clear Fraud": 1,
    "Subtle Fraud": 1,
    "Clear Normal": 0,
    "Ambiguous but Ultimately Normal": 0,
}
