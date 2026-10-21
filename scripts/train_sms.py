"""
Training script for SMS Scam Detection Baseline Model.
Matches Cells 9-14 of original notebook.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SMS_BASELINE_MODEL_PATH
from src.data.loader import load_sms_dataset
from src.data.preprocess import clean_sms_data, split_sms_data
from src.models.baseline_sms import build_sms_baseline_pipeline, save_sms_model
from src.evaluation.metrics import evaluate_classification_model
from src.evaluation.test_cases import SMS_UNSEEN_TEST_CASES


def main():
    print("=== Training SMS Scam Baseline Model ===")

    # 1. Load data
    sms_raw = load_sms_dataset()

    # 2. Clean and preprocess
    sms_df = clean_sms_data(sms_raw["train"])
    print(f"Clean dataset shape: {sms_df.shape}")
    print("Label distribution:\n", sms_df["label"].value_counts())

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_sms_data(sms_df)
    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    # 4. Build and train pipeline
    model = build_sms_baseline_pipeline()
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # 5. Evaluate on held-out test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluate_classification_model(
        y_test,
        y_pred,
        y_prob=y_prob,
        target_names=["Legitimate (ham)", "Scam (spam)"]
    )

    # 6. Save model
    save_sms_model(model, SMS_BASELINE_MODEL_PATH)

    # 7. Test on unseen test cases (Cell 14)
    print("\n=== Testing on Unseen Example Messages (Cell 14) ===")
    test_texts = [text for _, text, _ in SMS_UNSEEN_TEST_CASES]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)[:, 1]

    for (name, text, expected), pred, prob in zip(SMS_UNSEEN_TEST_CASES, predictions, probabilities):
        label = "SCAM" if pred == 1 else "LEGITIMATE"
        print(f"\n[{name}]")
        print(f"Message: {text}")
        print(f"Prediction: {label} (Expected: {'SCAM' if expected == 1 else 'LEGITIMATE'})")
        print(f"Model scam score: {prob:.1%}")


if __name__ == "__main__":
    main()
