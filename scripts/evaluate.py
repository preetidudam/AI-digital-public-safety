"""
Evaluation script to test a saved model package against sanity checks,
contrast pairs, and feature importance analysis.
Matches Cells 40, 41, and 42 of original notebook.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import IMPROVED_MODEL_PATH
from src.models.detector import ScamDetector
from src.evaluation.error_analysis import get_top_features
from src.evaluation.test_cases import SANITY_CHECK_TEST_CASES, CONTRAST_TEST_PAIRS


def run_evaluation(model_path: Path):
    print(f"Loading model from: {model_path}...")
    detector = ScamDetector(model_path=model_path)
    print(f"Loaded successfully. Decision threshold: {detector.threshold:.2f}")

    # 1. Sanity Checks (Cell 40)
    print("\n" + "=" * 50)
    print("SANITY CHECK TEST SUITE (Cell 40)")
    print("=" * 50)

    rows = []
    for name, expected, text in SANITY_CHECK_TEST_CASES:
        res = detector.analyze(text)
        predicted = int(res["is_scam"])
        rows.append({
            "Test case": name,
            "Expected": "Scam" if expected else "Legitimate",
            "Predicted": "Scam" if predicted else "Legitimate",
            "Scam score": f"{res['score']:.1%}",
            "Result": "PASS" if predicted == expected else "FAIL"
        })

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))
    pass_rate = (results_df["Result"] == "PASS").mean()
    print(f"\nSanity-check accuracy: {pass_rate:.1%}")

    # 2. Contrast Pairs (Cell 42)
    print("\n" + "=" * 50)
    print("CONTRAST PAIRS TEST SUITE (Cell 42)")
    print("=" * 50)

    contrast_rows = []
    for item in CONTRAST_TEST_PAIRS:
        res = detector.analyze(item["Text"])
        predicted = int(res["is_scam"])
        correct = predicted == item["Expected_binary"]
        contrast_rows.append({
            "Pair": item["Pair"],
            "Version": item["Version"],
            "Expected": item["Expected"],
            "Scam score": f"{res['score']:.1%}",
            "Predicted": "Scam" if predicted else "Legitimate",
            "Correct": "PASS" if correct else "FAIL"
        })

    contrast_df = pd.DataFrame(contrast_rows)
    print(contrast_df.to_string(index=False))
    contrast_accuracy = (contrast_df["Correct"] == "PASS").mean()
    print(f"\nContrast pairs accuracy: {contrast_accuracy:.1%}")

    # 3. Top Feature Weights (Cell 41)
    print("\n" + "=" * 50)
    print("TOP FEATURE WEIGHTS (Cell 41)")
    print("=" * 50)
    try:
        top_fraud, top_legit = get_top_features(detector.model, n_top=15)
        print("\nFeatures most associated with scam predictions:")
        print(top_fraud.to_string(index=False))
        print("\nFeatures most associated with legitimate predictions:")
        print(top_legit.to_string(index=False))
    except Exception as e:
        print(f"Feature importance inspection skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved scam detector model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(IMPROVED_MODEL_PATH),
        help="Path to saved model (.joblib)"
    )
    args = parser.parse_args()
    run_evaluation(Path(args.model_path))


if __name__ == "__main__":
    main()
