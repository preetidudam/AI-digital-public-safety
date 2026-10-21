"""
Baseline ICFD Fraud Transcript Pipeline using TF-IDF and Logistic Regression.
Matches Cells 24 and 34-36 of the original notebook.
"""
from pathlib import Path
from typing import Dict, Any, Union
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import (
    ICFD_BASELINE_MODEL_PATH,
    ICFD_CONVERSATION_MODEL_PATH,
    DEFAULT_CONVERSATION_THRESHOLD,
    RANDOM_SEED
)


def build_icfd_baseline_pipeline(
    ngram_range=(1, 2),
    min_df: int = 2,
    max_features: int = 150_000,
    sublinear_tf: bool = True,
    max_iter: int = 1500,
    random_state: int = RANDOM_SEED,
    dtype: str = "float32"
) -> Pipeline:
    """
    Build scikit-learn Pipeline for ICFD turn-level scam detection.
    Matches Cell 24 of original notebook.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            dtype=dtype
        )),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state
        ))
    ])


def build_conversation_baseline_pipeline(
    ngram_range=(1, 2),
    min_df: int = 2,
    max_features: int = 150_000,
    sublinear_tf: bool = True,
    max_iter: int = 2000,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Build scikit-learn Pipeline for full-conversation scam detection.
    Matches Cell 34 of original notebook.
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


def package_model(
    model: Pipeline,
    threshold: float = DEFAULT_CONVERSATION_THRESHOLD,
    model_type: str = "TF-IDF + Logistic Regression",
    training_unit: str = "Complete conversation transcripts"
) -> Dict[str, Any]:
    """
    Package model and metadata into dictionary for serialization.
    Matches Cell 36 of original notebook.
    """
    return {
        "model": model,
        "threshold": threshold,
        "label_meanings": {
            0: "Legitimate",
            1: "Potential scam"
        },
        "model_type": model_type,
        "training_unit": training_unit
    }


def save_icfd_model(model_obj: Union[Pipeline, Dict[str, Any]], model_path: Union[str, Path]):
    """
    Save trained model or model package to disk.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, model_path)
    print(f"Saved ICFD model to: {model_path}")
