"""
Model performance evaluation metrics and decision threshold analysis.
Matches Cells 12, 25, 34, 35, 48, 49 of the original notebook.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_classification_model(
    y_true,
    y_pred,
    y_prob=None,
    target_names: Optional[List[str]] = None,
    digits: int = 4
) -> Dict[str, Any]:
    """
    Compute standard classification metrics and return dictionary of results.
    Matches Cells 12, 25, 34 of original notebook.
    """
    if target_names is None:
        target_names = ["Legitimate", "Scam"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=target_names,
            digits=digits,
            zero_division=0
        )
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 50)
    print("CLASSIFICATION PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nClassification Report:")
    print(metrics["classification_report"])

    print("Confusion Matrix [TN, FP / FN, TP]:")
    print(metrics["confusion_matrix"])

    return metrics


def evaluate_threshold_sweep(
    y_true,
    y_prob,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Evaluate precision, recall, F1, false alarms, and missed scams
    across multiple decision thresholds.
    Matches Cells 35, 48, and 49 of the original notebook.
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    rows = []
    total_negatives = int((y_true == 0).sum())

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())

        row = {
            "Threshold": t,
            "Scam Precision": precision_score(y_true, pred, zero_division=0),
            "Scam Recall": recall_score(y_true, pred, zero_division=0),
            "Scam F1": f1_score(y_true, pred, zero_division=0),
            "False Alarms": fp,
            "Missed Scams": fn,
            "False Alarm Rate": fp / total_negatives if total_negatives > 0 else 0.0
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 50)
    print("DECISION THRESHOLD SWEEP")
    print("=" * 50)
    print(df.round(3).to_string(index=False))
    return df
