"""
Interpretability and error analysis:
- Top feature weights for fraud and legitimate classes (Cells 27, 28, 41)
- Validation prediction errors and length checks (Cells 29, 30)
- False positive scenario audit (Cells 43-46)
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def get_top_features(model, n_top: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract top features associated with fraud and legitimate predictions
    from a scikit-learn pipeline (TF-IDF + Logistic Regression).
    Matches Cells 27, 28, 41 of original notebook.
    """
    clf = model.named_steps["classifier"]
    weights = clf.coef_[0]

    if "tfidf" in model.named_steps:
        feature_names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    elif "features" in model.named_steps:
        feature_names = np.array(model.named_steps["features"].get_feature_names_out())
    else:
        raise ValueError("Could not find 'tfidf' or 'features' step in pipeline.")

    fraud_idx = np.argsort(weights)[-n_top:][::-1]
    normal_idx = np.argsort(weights)[:n_top]

    top_fraud_df = pd.DataFrame({
        "Feature": feature_names[fraud_idx],
        "Weight": weights[fraud_idx]
    }).reset_index(drop=True)

    top_legit_df = pd.DataFrame({
        "Feature": feature_names[normal_idx],
        "Weight": weights[normal_idx]
    }).reset_index(drop=True)

    return top_fraud_df, top_legit_df


def analyze_prediction_errors(
    X_val: pd.Series,
    y_val: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    Analyze missed fraud (false negatives) and false alarms (false positives).
    Matches Cells 29-30 of original notebook.
    """
    error_df = pd.DataFrame({
        "text": X_val.reset_index(drop=True),
        "actual": y_val.reset_index(drop=True),
        "predicted": y_pred,
        "fraud_score": y_prob
    })

    errors = error_df[error_df["actual"] != error_df["predicted"]].copy()
    errors["actual_label"] = errors["actual"].map({0: "LEGITIMATE", 1: "FRAUD"})
    errors["predicted_label"] = errors["predicted"].map({0: "LEGITIMATE", 1: "FRAUD"})
    errors["text_length"] = errors["text"].str.len()

    missed_fraud = errors[errors["actual"] == 1].sort_values("fraud_score")
    false_alarms = errors[errors["actual"] == 0].sort_values("fraud_score", ascending=False)

    print(f"\nTotal validation errors: {len(errors)}")
    print(f"Missed fraud count: {len(missed_fraud)}")
    print(f"False alarms count: {len(false_alarms)}")

    return {
        "all_errors": errors,
        "missed_fraud": missed_fraud,
        "false_alarms": false_alarms
    }


def analyze_false_positives_by_scenario(
    X_val: pd.Series,
    y_val: pd.Series,
    y_prob: np.ndarray,
    conversation_df: pd.DataFrame,
    threshold: float = 0.30
) -> pd.DataFrame:
    """
    Analyze false alarms grouped by original ICFD case categories (Clear Normal vs Ambiguous).
    Matches Cells 43-46 of original notebook.
    """
    val_df = pd.DataFrame({
        "text": X_val.reset_index(drop=True),
        "actual_label": y_val.reset_index(drop=True),
        "scam_score": y_prob
    })

    # Filter legitimate records
    legit_df = val_df[val_df["actual_label"] == 0].copy()

    # Merge with original conversation metadata
    audit = legit_df.merge(
        conversation_df[["text", "case_type", "conversation_uid"]],
        on="text",
        how="left"
    )

    audit["above_threshold"] = audit["scam_score"] >= threshold

    summary = (
        audit.groupby("case_type")
        .agg(
            legitimate_examples=("text", "count"),
            above_threshold=("above_threshold", "sum"),
            mean_scam_score=("scam_score", "mean"),
            max_scam_score=("scam_score", "max")
        )
        .reset_index()
    )

    summary["false_alarm_rate"] = (
        summary["above_threshold"] / summary["legitimate_examples"]
    )

    print(f"\nFalse Positive Scenario Audit (Threshold = {threshold:.2f}):")
    print(summary.round(3).to_string(index=False))

    return summary
