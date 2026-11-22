"""
Evaluation script for the Indian Currency Verifier (MobileNetV3).
Loads the trained checkpoint from models/currency_verifier.pt and computes:
- Accuracy, Precision, Recall, F1-Score
- Full Confusion Matrix (TN, FP, FN, TP)
- Per-class classification report
- Confidence distribution and sample sanity checks
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from config import (
    CURRENCY_DATASET_DIR,
    CURRENCY_MODEL_PATH,
    CURRENCY_CLASSES,
    CURRENCY_IDX_TO_CLASS
)
from src.data.currency_loader import get_currency_dataloaders, get_currency_transforms
from src.models.currency_verifier import build_currency_model, CurrencyVerifier


def evaluate_currency_model(model_path: Path = CURRENCY_MODEL_PATH, batch_size: int = 32):
    print("=" * 65)
    print("      INDIAN CURRENCY VERIFIER: COMPREHENSIVE EVALUATION")
    print("=" * 65)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at: {model_path}")
        print("Please train the model first using: python scripts/train_currency_verifier.py")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model Path: {model_path}")
    print(f"Device:     {device}")

    # Load validation dataloader
    _, val_loader, stats = get_currency_dataloaders(
        data_dir=CURRENCY_DATASET_DIR,
        batch_size=batch_size
    )

    print(f"\nValidation Split Stats:")
    print(f"  - Genuine notes:     {stats['val_genuine']}")
    print(f"  - Counterfeit notes: {stats['val_counterfeit']}")
    print(f"  - Total validation:  {stats['val_genuine'] + stats['val_counterfeit']}")

    # Load model weights
    model = build_currency_model(num_classes=len(CURRENCY_CLASSES), pretrained=False)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print("\n" + "-" * 65)
    print("KEY PERFORMANCE INDICATORS")
    print("-" * 65)
    print(f"  Accuracy:                 {acc * 100.0:.2f}%")
    print(f"  Macro Precision:          {p_macro * 100.0:.2f}%")
    print(f"  Macro Recall:             {r_macro * 100.0:.2f}%")
    print(f"  Macro F1-Score:           {f1_macro * 100.0:.2f}%")
    print(f"  Weighted F1-Score:        {f1_weighted * 100.0:.2f}%")

    print("\n" + "-" * 65)
    print("CONFUSION MATRIX BREAKDOWN")
    print("-" * 65)
    print("                               Predicted Counterfeit   Predicted Genuine")
    print(f"Actual Counterfeit (Class 0):  {cm[0][0]:<23} {cm[0][1]}")
    print(f"Actual Genuine     (Class 1):  {cm[1][0]:<23} {cm[1][1]}")
    print(f"\n  True Counterfeit (TN): {tn:<5} (correctly flagged fake notes)")
    print(f"  False Genuine    (FP): {fp:<5} (counterfeit notes falsely marked genuine)")
    print(f"  False Counterfeit(FN): {fn:<5} (genuine notes falsely marked counterfeit)")
    print(f"  True Genuine     (TP): {tp:<5} (correctly authenticated genuine notes)")

    print("\n" + "-" * 65)
    print("PER-CLASS METRICS TABLE")
    print("-" * 65)
    print(f"{'Class':<22} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    for idx, cname in enumerate(CURRENCY_CLASSES):
        print(f"{cname:<22} {p_per_class[idx]*100.0:>6.2f}%      {r_per_class[idx]*100.0:>6.2f}%      {f1_per_class[idx]*100.0:>6.2f}%      {support[idx]}")

    print("\n" + "-" * 65)
    print("FULL CLASSIFICATION REPORT")
    print("-" * 65)
    print(classification_report(all_targets, all_preds, target_names=CURRENCY_CLASSES, digits=4))

    # 7. External Unseen Test Suite
    ext_test_dir = CURRENCY_DATASET_DIR / "external_test"
    ext_results = {}
    if ext_test_dir.exists():
        print("\n" + "=" * 65)
        print("EXTERNAL UNSEEN TEST EVALUATION (MULTI-SOURCE BENCHMARK)")
        print("=" * 65)
        try:
            _, val_transform = get_currency_transforms()
            from torchvision import datasets
            ext_dataset = datasets.ImageFolder(str(ext_test_dir), transform=val_transform)
            ext_loader = torch.utils.data.DataLoader(ext_dataset, batch_size=batch_size, shuffle=False)

            ext_preds, ext_targets = [], []
            with torch.no_grad():
                for images, labels in ext_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().tolist()
                    ext_preds.extend(preds)
                    ext_targets.extend(labels.tolist())

            ext_acc = accuracy_score(ext_targets, ext_preds)
            ext_cm = confusion_matrix(ext_targets, ext_preds)
            ext_tn, ext_fp, ext_fn, ext_tp = ext_cm.ravel() if ext_cm.size == 4 else (0, 0, 0, 0)
            ext_fgr = (ext_fp / (ext_tn + ext_fp)) * 100.0 if (ext_tn + ext_fp) > 0 else 0.0

            print(f"External Test Metrics (N={len(ext_targets)} completely unseen notes):")
            print(f"  Accuracy:                 {ext_acc * 100.0:.2f}%")
            print(f"  False Genuine Rate (FGR): {ext_fgr:.2f}% (CRITICAL: lower is safer)")
            print(f"  Total Counterfeit:        {ext_tn + ext_fp}")
            print(f"    - True Counterfeit (TN):{ext_tn} (correctly flagged fake notes)")
            print(f"    - False Genuine (FP):   {ext_fp} (counterfeits missed as genuine)")
            print(f"  Total Genuine:            {ext_fn + ext_tp}")
            print(f"    - True Genuine (TP):    {ext_tp} (correctly authenticated genuine)")
            print(f"    - False Counterfeit (FN):{ext_fn} (genuine notes falsely flagged)")

            print("\nExternal Confusion Matrix:")
            print("                               Predicted Counterfeit   Predicted Genuine")
            print(f"Actual Counterfeit (Class 0):  {ext_cm[0][0]:<23} {ext_cm[0][1]}")
            print(f"Actual Genuine     (Class 1):  {ext_cm[1][0]:<23} {ext_cm[1][1]}")

            ext_results = {
                "ext_accuracy": ext_acc,
                "ext_fgr": ext_fgr,
                "ext_confusion_matrix": ext_cm.tolist()
            }
        except Exception as e:
            print(f"Warning: Could not evaluate external test set: {e}")

    # Sanity inference test on User's Failed Sample
    print("\n" + "-" * 65)
    print("INSPECTION OF USER EXAMPLE IMAGE (Previously Failed Sample)")
    print("-" * 65)
    verifier = CurrencyVerifier(model_path=model_path, device=str(device))
    user_sample = Path("data/test_samples/user_failed_fake_sample.jpeg")
    if user_sample.exists():
        res_u = verifier.verify(user_sample)
        print(f"Image:       {user_sample.name}")
        print(f"Verdict:     {res_u['verdict']}")
        print(f"Confidence:  {res_u['confidence']}%")
        print(f"Is Genuine:  {res_u['is_genuine']}")
        if res_u["is_genuine"] is False:
            print("STATUS:      ✅ RESOLVED! Correctly flagged as POTENTIAL COUNTERFEIT.")
        else:
            print("STATUS:      ❌ STILL FAILING (flagged as genuine).")

    print("=" * 65)
    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm.tolist(),
        **ext_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Indian Currency Verifier")
    parser.add_argument("--model-path", type=str, default=str(CURRENCY_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate_currency_model(model_path=Path(args.model_path), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
