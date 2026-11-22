"""
Training script for Indian Currency Verification CNN (MobileNetV3).
Fine-tunes a pretrained vision model to classify genuine vs counterfeit currency notes.
"""
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from config import (
    CURRENCY_DATASET_DIR,
    CURRENCY_MODEL_PATH,
    CURRENCY_CLASSES,
    RANDOM_SEED
)
from src.data.currency_loader import (
    check_dataset_availability,
    get_currency_dataloaders,
    get_currency_transforms
)
from src.models.currency_verifier import build_currency_model


def train_currency_model(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    device_name: str = "auto"
):
    print("=" * 60)
    print("   INDIAN CURRENCY VERIFIER: CNN MODEL TRAINING")
    print("=" * 60)

    # 1. Check dataset availability
    stats = check_dataset_availability(CURRENCY_DATASET_DIR)
    print(f"\nChecking dataset at: {stats['dataset_dir']}")
    print(f"- Training Genuine notes:      {stats['train_genuine']}")
    print(f"- Training Counterfeit notes:  {stats['train_counterfeit']}")
    print(f"- Validation Genuine notes:    {stats['val_genuine']}")
    print(f"- Validation Counterfeit notes:{stats['val_counterfeit']}")
    print(f"- Total images found:          {stats['total_images']}")

    if not stats["is_ready"]:
        print("\n" + "!" * 60)
        print("DATASET NOT FOUND OR INSUFFICIENT IMAGES!")
        print("!" * 60)
        print("\nTo train the Currency Verifier, please place note images into:")
        print(f"  Genuine notes:     {CURRENCY_DATASET_DIR / 'train' / 'genuine'}")
        print(f"  Counterfeit notes: {CURRENCY_DATASET_DIR / 'train' / 'counterfeit'}")
        print("\nOptional validation images:")
        print(f"  Val Genuine:       {CURRENCY_DATASET_DIR / 'val' / 'genuine'}")
        print(f"  Val Counterfeit:   {CURRENCY_DATASET_DIR / 'val' / 'counterfeit'}")
        print("\nSupported formats: .jpg, .jpeg, .png, .bmp, .webp")
        print("Supported denominations: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000 (front or back)")
        print("\nOnce images are added, rerun:")
        print("  python scripts/train_currency_verifier.py")
        print("=" * 60)
        return False

    # 2. Setup Device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"\nTraining Device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 3. Load Datasets
    train_loader, val_loader, _ = get_currency_dataloaders(
        data_dir=CURRENCY_DATASET_DIR,
        batch_size=batch_size
    )

    # 4. Initialize Pretrained Model
    print("\nInitializing Pretrained MobileNetV3-Small backbone...")
    model = build_currency_model(num_classes=len(CURRENCY_CLASSES), pretrained=True)
    model.to(device)

    # Compute class weights with vigilance against counterfeit false-negatives
    n_gen = max(stats["train_genuine"], 1)
    n_fake = max(stats["train_counterfeit"], 1)
    total_train = n_gen + n_fake
    weight_fake = (total_train / (2.0 * n_fake)) * 1.35  # Higher penalty on missing counterfeit notes
    weight_gen = total_train / (2.0 * n_gen)
    class_weights = torch.tensor([weight_fake, weight_gen], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    best_weights = None

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train_samples = 0

        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train_samples += images.size(0)

            if batch_idx % 25 == 0 or batch_idx == total_batches:
                curr_acc = (train_correct / total_train_samples) * 100.0
                curr_loss = train_loss / total_train_samples
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Batch [{batch_idx:02d}/{total_batches:02d}] "
                      f"Loss: {curr_loss:.4f} | Running Acc: {curr_acc:.1f}%", flush=True)

        scheduler.step()

        epoch_train_loss = train_loss / total_train_samples
        epoch_train_acc = (train_correct / total_train_samples) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val_samples += images.size(0)

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0.0
        epoch_val_acc = (val_correct / total_val_samples * 100.0) if total_val_samples > 0 else 0.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%"
        )

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = model.state_dict().copy()

    # 5. Save Model
    CURRENCY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_state = best_weights if best_weights is not None else model.state_dict()
    torch.save(save_state, str(CURRENCY_MODEL_PATH))

    print("\n" + "=" * 60)
    print(f"Model saved successfully to: {CURRENCY_MODEL_PATH}")
    print(f"Best Validation Accuracy:    {best_val_acc:.1f}%")
    print("=" * 60)

    # 6. Final Evaluation on Validation Split
    print("\n" + "=" * 60)
    print("   CURRENCY VERIFIER: FINAL MODEL EVALUATION REPORT")
    print("=" * 60)

    model.load_state_dict(save_state)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.tolist())

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report
    )
    from torchvision import datasets

    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds)

    # Class 0: Counterfeit, Class 1: Genuine
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    val_fgr = (fp / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0

    print(f"\nOverall Validation Metrics (N={len(all_targets)} images):")
    print(f"  Accuracy:                 {acc * 100.0:.2f}%")
    print(f"  Precision (Macro):        {p_macro * 100.0:.2f}%")
    print(f"  Recall (Macro):           {r_macro * 100.0:.2f}%")
    print(f"  F1-Score (Macro):         {f1_macro * 100.0:.2f}%")
    print(f"  F1-Score (Weight):        {f1_weighted * 100.0:.2f}%")
    print(f"  False Genuine Rate (FGR): {val_fgr:.2f}%")

    print("\nConfusion Matrix:")
    print("               Predicted Counterfeit  Predicted Genuine")
    print(f"Actual Counterfeit:     {cm[0][0]:<15} {cm[0][1]}")
    print(f"Actual Genuine:         {cm[1][0]:<15} {cm[1][1]}")
    print(f"\n  True Counterfeit (TN): {tn}")
    print(f"  False Genuine (FP):    {fp} (suspicious note slipped through)")
    print(f"  False Counterfeit (FN):{fn} (genuine note falsely flagged)")
    print(f"  True Genuine (TP):     {tp}")

    print("\nDetailed Classification Report:")
    target_names = ["Counterfeit (Class 0)", "Genuine (Class 1)"]
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))

    # 7. Evaluate on Held-out External Test Set
    ext_test_dir = CURRENCY_DATASET_DIR / "external_test"
    if ext_test_dir.exists():
        print("\n" + "=" * 60)
        print("   CURRENCY VERIFIER: EXTERNAL UNSEEN TEST EVALUATION")
        print("=" * 60)
        val_transform = get_currency_transforms()[1]
        try:
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

            print(f"External Test Metrics (N={len(ext_targets)} completely unseen images):")
            print(f"  Accuracy:                 {ext_acc * 100.0:.2f}%")
            print(f"  False Genuine Rate (FGR): {ext_fgr:.2f}%")
            print(f"  Total Counterfeit:        {ext_tn + ext_fp}")
            print(f"    - Correctly Flagged:    {ext_tn} (True Counterfeit)")
            print(f"    - Misclassified:        {ext_fp} (False Genuine)")
            print(f"  Total Genuine:            {ext_fn + ext_tp}")
            print(f"    - Correctly Marked:     {ext_tp} (True Genuine)")
            print(f"    - Misclassified:        {ext_fn} (False Counterfeit)")

            print("\nExternal Confusion Matrix:")
            print("                       Pred Counterfeit  Pred Genuine")
            print(f"Actual Counterfeit:     {ext_cm[0][0]:<17} {ext_cm[0][1]}")
            print(f"Actual Genuine:         {ext_cm[1][0]:<17} {ext_cm[1][1]}")
            print("=" * 60)
        except Exception as e:
            print(f"Warning: Could not evaluate external test set: {e}")

    print("Currency Verifier is now active in the Gradio Web App!")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Indian Currency Verifier CNN Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Training device: 'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()

    train_currency_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
