"""
Audio processing pipelines:
- Whisper Speech-to-Text transcription (Cells 55-57, 59, 67, 70)
- Deepfake audio classification with Wav2Vec2 (Cells 52-54)
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
import torch

from config import DEEPFAKE_AUDIO_MODEL_ID, WHISPER_MODEL_NAME


class WhisperTranscriber:
    """
    OpenAI Whisper wrapper for transcribing call recordings.
    """

    def __init__(self, model_name: str = WHISPER_MODEL_NAME, device: Optional[str] = None):
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. Please install it using: pip install openai-whisper"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Whisper model '{model_name}' on {self.device.upper()}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print("Whisper model loaded successfully.")

    def transcribe(self, audio_path: Union[str, Path], task: str = "transcribe") -> str:
        """
        Transcribe audio file to text transcript.
        Loads audio waveform into memory via librosa to avoid requiring system ffmpeg CLI.
        """
        audio_path_str = str(audio_path)
        if not os.path.exists(audio_path_str):
            raise FileNotFoundError(f"Audio file not found: {audio_path_str}")

        try:
            import librosa
            audio, _ = librosa.load(audio_path_str, sr=16000, mono=True)
            input_audio = audio
        except Exception:
            # Fallback to raw file path (requires ffmpeg CLI installed)
            input_audio = audio_path_str

        fp16 = self.device != "cpu"
        result = self.model.transcribe(input_audio, task=task, fp16=fp16)
        transcript = result.get("text", "").strip()
        return transcript


class DeepfakeAudioClassifier:
    """
    Wav2Vec2 audio classification pipeline for detecting synthetic/deepfake voices.
    Matches Cells 52-54 of original notebook.
    """

    def __init__(self, model_id: str = DEEPFAKE_AUDIO_MODEL_ID):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is not installed. Please install it using: pip install transformers"
            )

        audio_device = 0 if torch.cuda.is_available() else -1
        print(f"Loading deepfake audio detector '{model_id}' on {'GPU' if audio_device == 0 else 'CPU'}...")
        self.pipeline = pipeline(
            "audio-classification",
            model=model_id,
            device=audio_device
        )
        print("Deepfake audio detector loaded.")

    def predict(self, audio_path: Union[str, Path], sr: int = 16000) -> Any:
        """
        Classify audio file as real or deepfake.
        """
        import librosa

        audio_path_str = str(audio_path)
        if not os.path.exists(audio_path_str):
            raise FileNotFoundError(f"Audio file not found: {audio_path_str}")

        audio, sample_rate = librosa.load(audio_path_str, sr=sr, mono=True)

        if len(audio) < sr:
            print("Warning: audio is shorter than one second.")

        result = self.pipeline({"raw": audio, "sampling_rate": sample_rate})
        return result
