"""
Baseline SMS Scam Classification Pipeline using TF-IDF and Logistic Regression.
Matches Cells 11-13 of the original notebook.
"""
from pathlib import Path
from typing import Union
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import SMS_BASELINE_MODEL_PATH, RANDOM_SEED


def build_sms_baseline_pipeline(
    ngram_range=(1, 2),
    min_df: int = 2,
    max_features: int = 100_000,
    sublinear_tf: bool = True,
    max_iter: int = 2000,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Build scikit-learn Pipeline for SMS scam detection.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf
        )),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state
        ))
    ])


def save_sms_model(model: Pipeline, model_path: Union[str, Path] = SMS_BASELINE_MODEL_PATH):
    """
    Save trained SMS baseline model to disk.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved SMS baseline model to: {model_path}")


def load_sms_model(model_path: Union[str, Path] = SMS_BASELINE_MODEL_PATH) -> Pipeline:
    """
    Load trained SMS baseline model from disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SMS model file not found at: {model_path}")
    return joblib.load(model_path)
