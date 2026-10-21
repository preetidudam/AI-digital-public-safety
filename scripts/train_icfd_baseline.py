"""
Training script for ICFD Turn-level Fraud Detection Baseline Model.
Matches Cells 22-28 of original notebook.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ICFD_BASELINE_MODEL_PATH
from src.data.loader import load_icfd_dataset
from src.data.preprocess import clean_icfd_training_data, split_icfd_grouped
from src.models.baseline_icfd import build_icfd_baseline_pipeline, save_icfd_model
from src.evaluation.metrics import evaluate_classification_model
from src.evaluation.error_analysis import get_top_features
from src.evaluation.test_cases import ICFD_BASELINE_TEST_CASES


def main():
    print("=== Training ICFD Baseline Model ===")

    # 1. Load data
    icfd = load_icfd_dataset()

    # 2. Clean training data
    train_df = clean_icfd_training_data(icfd["train"])
    print(f"Cleaned training rows: {len(train_df)}")
    print("Label distribution:\n", train_df["label"].value_counts())

    # 3. Grouped train/validation split by conversation_uid
    X_train, X_val, y_train, y_val, groups = split_icfd_grouped(train_df)
    print(f"Training rows: {len(X_train)}, Validation rows: {len(X_val)}")

    # 4. Build and train pipeline
    model = build_icfd_baseline_pipeline()
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    print(f"Vocabulary size: {len(model.named_steps['tfidf'].vocabulary_):,}")

    # 5. Evaluate on validation set
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    evaluate_classification_model(
        y_val,
        y_pred,
        y_prob=y_prob,
        target_names=["Legitimate", "Fraud"]
    )

    # 6. Save model
    save_icfd_model(model, ICFD_BASELINE_MODEL_PATH)

    # 7. Test on test cases (Cell 26)
    print("\n=== Testing on Example Cases (Cell 26) ===")
    for name, text in ICFD_BASELINE_TEST_CASES:
        scam_prob = model.predict_proba([text])[0][1]
        pred_label = "FRAUD" if scam_prob >= 0.5 else "LEGITIMATE"
        print(f"\n[{name}]")
        print(f"Prediction: {pred_label} (Fraud score: {scam_prob:.1%})")

    # 8. Feature weights inspection (Cell 27)
    print("\n=== Top Feature Weights (Cell 27) ===")
    top_fraud, top_legit = get_top_features(model, n_top=15)
    print("\nTop Fraud-Associated Features:")
    print(top_fraud.to_string(index=False))
    print("\nTop Legitimate-Associated Features:")
    print(top_legit.to_string(index=False))


if __name__ == "__main__":
    main()
