"""
Training script for Improved ICFD Fraud Detection Model (Word + Char FeatureUnion).
Matches Cells 31-36, 40, 42, 47-50 of original notebook.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import IMPROVED_MODEL_PATH, DEFAULT_IMPROVED_THRESHOLD
from src.data.loader import load_icfd_dataset
from src.data.preprocess import (
    clean_icfd_training_data,
    prepare_complete_conversations,
    split_by_scenario
)
from src.models.improved_model import (
    build_improved_pipeline,
    package_improved_model,
    save_improved_model
)
from src.models.detector import ScamDetector
from src.evaluation.metrics import evaluate_classification_model, evaluate_threshold_sweep
from src.evaluation.test_cases import SANITY_CHECK_TEST_CASES, CONTRAST_TEST_PAIRS


def main():
    print("=== Training Improved Fraud Detection Model (Word + Char TF-IDF) ===")

    # 1. Load data
    icfd = load_icfd_dataset()

    # 2. Clean training split
    train_df = clean_icfd_training_data(icfd["train"])

    # 3. Aggregate to complete conversation transcripts
    conversation_df = prepare_complete_conversations(train_df)
    print(f"Total complete conversations: {len(conversation_df)}")
    print("Class distribution:\n", conversation_df["label"].value_counts())

    # 4. Grouped split by scenario group
    X_train, X_val, y_train, y_val, groups = split_by_scenario(conversation_df)
    print(f"Training conversations: {len(X_train)}, Validation conversations: {len(X_val)}")
    overlap = len(set(groups.loc[X_train.index]) & set(groups.loc[X_val.index]))
    print(f"Scenario overlap between train and val: {overlap}")

    # 5. Build and train improved pipeline
    model = build_improved_pipeline()
    print("\nTraining Word + Character FeatureUnion model...")
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # 6. Evaluate probabilities on validation set
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= DEFAULT_IMPROVED_THRESHOLD).astype(int)

    evaluate_classification_model(
        y_val,
        y_pred,
        y_prob=y_prob,
        target_names=["Legitimate", "Scam"]
    )

    # 7. Threshold sweep evaluation (Cells 48-49)
    evaluate_threshold_sweep(
        y_val,
        y_prob,
        thresholds=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    )

    # 8. Package and save model
    package = package_improved_model(model, threshold=DEFAULT_IMPROVED_THRESHOLD)
    save_improved_model(package, IMPROVED_MODEL_PATH)

    # 9. Test with ScamDetector on sanity check cases (Cell 40)
    print("\n=== Sanity Checks Test Suite (Cell 40) ===")
    detector = ScamDetector(model_obj=model, threshold=DEFAULT_IMPROVED_THRESHOLD)
    passed = 0

    for name, expected, text in SANITY_CHECK_TEST_CASES:
        res = detector.analyze(text)
        predicted_binary = 1 if res["is_scam"] else 0
        is_pass = predicted_binary == expected
        if is_pass:
            passed += 1

        print(f"\n[{name}]")
        print(f"Expected: {'Scam' if expected else 'Legitimate'} | Predicted: {'Scam' if predicted_binary else 'Legitimate'}")
        print(f"Score: {res['score']:.1%} -> {'PASS' if is_pass else 'FAIL'}")

    print(f"\nSanity check pass rate: {passed}/{len(SANITY_CHECK_TEST_CASES)} ({passed / len(SANITY_CHECK_TEST_CASES):.1%})")

    # 10. Test with Contrast Pairs (Cell 42)
    print("\n=== Contrast Pairs Test Suite (Cell 42) ===")
    contrast_passed = 0
    for item in CONTRAST_TEST_PAIRS:
        res = detector.analyze(item["Text"])
        predicted_binary = 1 if res["is_scam"] else 0
        is_correct = predicted_binary == item["Expected_binary"]
        if is_correct:
            contrast_passed += 1

        print(f"\nPair: {item['Pair']} | Version: {item['Version']}")
        print(f"Expected: {item['Expected']} | Predicted: {'Scam' if predicted_binary else 'Legitimate'}")
        print(f"Score: {res['score']:.1%} -> {'CORRECT' if is_correct else 'INCORRECT'}")

    print(f"\nContrast pairs accuracy: {contrast_passed}/{len(CONTRAST_TEST_PAIRS)} ({contrast_passed / len(CONTRAST_TEST_PAIRS):.1%})")


if __name__ == "__main__":
    main()
