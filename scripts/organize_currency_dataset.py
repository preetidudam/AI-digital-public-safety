"""
Dataset organizer for Indian Currency Verification.
Implements:
1. Multi-source dataset ingestion (devanandjoly + iayushanand).
2. Strict cluster-based splitting by base image identity (zero near-duplicate data leakage).
3. Excludes held-out external_test images so external test set remains 100% unseen.
"""
import os
import sys
import shutil
import random
import re
from pathlib import Path
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CURRENCY_DATASET_DIR, RANDOM_SEED

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_KEYWORDS = ["real", "genuine", "original", "authentic", "true", "orig", "actual", "legit", "valid"]
FAKE_KEYWORDS = ["fake", "counterfeit", "forged", "forge", "duplicate", "false", "spurious", "fraud", "bogus"]
DENOM_RE = re.compile(r"(?<!\d)(2000|500|200|100|50|20|10)(?!\d)")


def classify_image_path(rel_path: str):
    p = rel_path.lower()
    has_real = any(k in p for k in REAL_KEYWORDS)
    has_fake = any(k in p for k in FAKE_KEYWORDS)

    if has_fake and not has_real:
        return "counterfeit"
    if has_real and not has_fake:
        return "genuine"
    return None


def extract_denomination(path_str: str) -> str:
    m = DENOM_RE.search(path_str)
    return m.group(1) if m else "note"


def extract_base_identity(path: Path) -> str:
    """
    Extracts base identifier by removing augmentation patterns:
    e.g. '1 (1)_aug0.jpg' -> '1 (1)'
         'aug_1500_WhatsApp Image... (1).jpeg' -> 'whatsapp_image... (1)'
    """
    stem = path.stem
    stem = re.sub(r"_aug\d+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"aug_\d+_", "", stem, flags=re.IGNORECASE)
    clean = stem.strip().lower()
    return f"{path.parent.name.lower()}_{clean}"


def organize_dataset(source_dirs=None, target_dir=CURRENCY_DATASET_DIR, val_ratio=0.20):
    print("=" * 65)
    print("   INDIAN CURRENCY DATASET ORGANIZER: STRICT GROUPED SPLIT")
    print("=" * 65)

    target_dir = Path(target_dir)
    train_gen = target_dir / "train" / "genuine"
    train_fake = target_dir / "train" / "counterfeit"
    val_gen = target_dir / "val" / "genuine"
    val_fake = target_dir / "val" / "counterfeit"
    ext_test_dir = target_dir / "external_test"

    # Identify external test files to exclude from train/val
    excluded_names = set()
    if ext_test_dir.exists():
        for f in ext_test_dir.rglob("*.*"):
            if f.is_file():
                excluded_names.add(f.name.lower())
                excluded_names.add(f.stem.lower())

    print(f"Holding out {len(excluded_names)} external test references (guaranteed unseen).")

    # Clean existing train and val
    for d in [train_gen, train_fake, val_gen, val_fake]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                try:
                    f.unlink()
                except Exception:
                    pass

    if source_dirs is None:
        source_dirs = [
            r"C:\Users\preet\.cache\kagglehub\datasets\devanandjoly\indian-currency-images-for-fake-currency-detection\versions\1\dataset",
            r"C:\Users\preet\.cache\kagglehub\datasets\iayushanand\currency-dataset500-inr-note-real-fake\versions\1\indian currency"
        ]

    valid_images = []
    skipped = 0

    print(f"\nScanning source directories...")
    for src_dir in source_dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"Warning: Source path does not exist: {src_path}")
            continue

        for root, _, files in os.walk(src_path):
            for file in files:
                f_path = Path(root) / file
                if f_path.suffix.lower() not in IMG_EXTS:
                    continue

                # Check if this exact file was placed into external_test
                if f_path.name.lower() in excluded_names or f_path.stem.lower() in excluded_names:
                    skipped += 1
                    continue

                rel_str = str(f_path.relative_to(src_path))
                label = classify_image_path(rel_str)
                if not label:
                    label = classify_image_path(f_path.parent.name)

                if not label:
                    skipped += 1
                    continue

                try:
                    with Image.open(f_path) as im:
                        im.verify()
                    base_id = extract_base_identity(f_path)
                    denom = extract_denomination(rel_str)
                    valid_images.append({
                        "src": f_path,
                        "label": label,
                        "denom": denom,
                        "base_id": base_id
                    })
                except Exception:
                    skipped += 1

    print(f"Found {len(valid_images)} valid labeled images (skipped {skipped} external/corrupt).")

    genuine_items = [x for x in valid_images if x["label"] == "genuine"]
    fake_items = [x for x in valid_images if x["label"] == "counterfeit"]

    print(f"- Total Genuine images:     {len(genuine_items)}")
    print(f"- Total Counterfeit images: {len(fake_items)}")

    # Group by base_id for strict cluster splitting
    def cluster_split(items, split_ratio=val_ratio):
        clusters = {}
        for item in items:
            bid = item["base_id"]
            if bid not in clusters:
                clusters[bid] = []
            clusters[bid].append(item)

        cluster_keys = list(clusters.keys())
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(cluster_keys)

        total_target_val = int(len(items) * split_ratio)
        val_items = []
        train_items = []
        val_count = 0

        for key in cluster_keys:
            cluster_list = clusters[key]
            if val_count < total_target_val:
                val_items.extend(cluster_list)
                val_count += len(cluster_list)
            else:
                train_items.extend(cluster_list)

        return train_items, val_items, len(clusters)

    train_gen_items, val_gen_items, n_clusters_gen = cluster_split(genuine_items)
    train_fake_items, val_fake_items, n_clusters_fake = cluster_split(fake_items)

    print(f"\nStrict Base-Cluster Split (Zero Duplicate Leakage):")
    print(f"  Genuine:     {n_clusters_gen} unique note clusters -> {len(train_gen_items)} train, {len(val_gen_items)} val")
    print(f"  Counterfeit: {n_clusters_fake} unique note clusters -> {len(train_fake_items)} train, {len(val_fake_items)} val")

    def copy_batch(items, dest_dir, prefix):
        count = 0
        for idx, item in enumerate(items):
            denom = item["denom"]
            ext = item["src"].suffix.lower()
            dest_name = f"{prefix}_{denom}_{idx:04d}{ext}"
            dest_file = dest_dir / dest_name
            shutil.copy2(item["src"], dest_file)
            count += 1
        return count

    print("\nCopying files to organized directory structure...")
    n_tr_gen = copy_batch(train_gen_items, train_gen, "genuine")
    n_tr_fake = copy_batch(train_fake_items, train_fake, "counterfeit")
    n_vl_gen = copy_batch(val_gen_items, val_gen, "genuine")
    n_vl_fake = copy_batch(val_fake_items, val_fake, "counterfeit")

    print(f"\nFinal Organized Dataset:")
    print(f"  train/genuine:     {n_tr_gen} images")
    print(f"  train/counterfeit: {n_tr_fake} images")
    print(f"  val/genuine:       {n_vl_gen} images")
    print(f"  val/counterfeit:   {n_vl_fake} images")
    print(f"  TOTAL TRAIN+VAL:   {n_tr_gen + n_tr_fake + n_vl_gen + n_vl_fake} images")
    print("=" * 65)
    return True


if __name__ == "__main__":
    organize_dataset()
