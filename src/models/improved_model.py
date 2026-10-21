"""
Improved Fraud Detection Pipeline using Word and Character n-gram FeatureUnion
with class-weighted Logistic Regression.
Matches Cells 47 and 50 of the original notebook.
"""
from pathlib import Path
from typing import Dict, Any, Union
import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import IMPROVED_MODEL_PATH, DEFAULT_IMPROVED_THRESHOLD, RANDOM_SEED


def build_improved_pipeline(
    word_ngram_range=(1, 2),
    word_min_df: int = 2,
    word_max_features: int = 100_000,
    char_ngram_range=(3, 5),
    char_min_df: int = 3,
    char_max_features: int = 100_000,
    sublinear_tf: bool = True,
    max_iter: int = 2000,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Build scikit-learn Pipeline with FeatureUnion combining:
    - Word-level TF-IDF n-grams (1, 2)
    - Character-level within-word (char_wb) TF-IDF n-grams (3, 5)
    followed by class-weighted Logistic Regression.
    Matches Cell 47 of original notebook.
    """
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=word_ngram_range,
                min_df=word_min_df,
                max_features=word_max_features,
                sublinear_tf=sublinear_tf
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=char_ngram_range,
                min_df=char_min_df,
                max_features=char_max_features,
                sublinear_tf=sublinear_tf
            ))
        ])),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state
        ))
    ])


def package_improved_model(
    model: Pipeline,
    threshold: float = DEFAULT_IMPROVED_THRESHOLD
) -> Dict[str, Any]:
    """
    Package improved model and metadata into dictionary for serialization.
    Matches Cell 50 of original notebook.
    """
    return {
        "model": model,
        "threshold": threshold,
        "label_meanings": {
            0: "Legitimate",
            1: "Potential scam"
        },
        "model_type": "Word + Character TF-IDF with Logistic Regression"
    }


def save_improved_model(
    model_package: Dict[str, Any],
    model_path: Union[str, Path] = IMPROVED_MODEL_PATH
):
    """
    Save improved model package to disk.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, model_path)
    print(f"Saved improved scam detector to: {model_path}")


def load_improved_model(
    model_path: Union[str, Path] = IMPROVED_MODEL_PATH
) -> Dict[str, Any]:
    """
    Load saved improved model package from disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Improved model not found at '{model_path}'. "
            f"Please run the training script first: python scripts/train_icfd_improved.py"
        )
    return joblib.load(model_path)
