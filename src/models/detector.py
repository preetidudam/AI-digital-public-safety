"""
Inference wrapper for the Scam Detection model.
Provides prediction, score calculation, and warning message formatting.
Matches Cells 37, 39, 51, 60, 69, 70 of original notebook.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib

from config import IMPROVED_MODEL_PATH, DEFAULT_IMPROVED_THRESHOLD


class ScamDetector:
    """
    ScamDetector encapsulates model loading, inference, threshold comparison,
    and alert generation.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_obj=None,
        threshold: Optional[float] = None
    ):
        if model_obj is not None:
            self.model = model_obj
            self.threshold = threshold if threshold is not None else DEFAULT_IMPROVED_THRESHOLD
            self.model_info = {"model_type": type(model_obj).__name__}
        else:
            path = Path(model_path) if model_path else IMPROVED_MODEL_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found at: {path}\n"
                    f"Please run 'python scripts/train_icfd_improved.py' to train and save the model."
                )

            package = joblib.load(path)
            if isinstance(package, dict) and "model" in package:
                self.model = package["model"]
                self.threshold = threshold if threshold is not None else package.get("threshold", DEFAULT_IMPROVED_THRESHOLD)
                self.model_info = package
            else:
                self.model = package
                self.threshold = threshold if threshold is not None else DEFAULT_IMPROVED_THRESHOLD
                self.model_info = {"model_type": "Pipeline"}

    def predict_score(self, text: str) -> float:
        """
        Predict scam probability (0.0 to 1.0) for a given text.
        """
        if not text or not text.strip():
            return 0.0
        return float(self.model.predict_proba([text])[0, 1])

    def is_flagged(self, text: str) -> bool:
        """
        Check if text exceeds the decision threshold.
        """
        return self.predict_score(text) >= self.threshold

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis returning text, probability, alert decision,
        and standard warning message. Matches Gradio logic in Cell 70.
        """
        if not text or not text.strip():
            return {
                "text": "",
                "score": 0.0,
                "score_percent": 0.0,
                "is_scam": False,
                "alert": "Please enter a transcript or SMS."
            }

        score = self.predict_score(text)
        is_scam = score >= self.threshold

        if is_scam:
            alert = (
                "⚠️ POTENTIAL SCAM DETECTED\n"
                "Do not transfer money under pressure or share "
                "OTPs, PINs, or passwords. Verify independently."
            )
        else:
            alert = (
                "✅ No alert triggered at the current threshold.\n"
                "This does not guarantee that the message or call is safe."
            )

        return {
            "text": text,
            "score": score,
            "score_percent": round(score * 100, 2),
            "is_scam": is_scam,
            "alert": alert,
            "threshold": self.threshold
        }
