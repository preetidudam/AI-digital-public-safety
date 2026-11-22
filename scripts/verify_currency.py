"""
Command-line tool to verify Indian currency note images (Genuine vs Counterfeit).
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

from config import CURRENCY_MODEL_PATH
from src.models.currency_verifier import CurrencyVerifier


def main():
    parser = argparse.ArgumentParser(description="Verify Indian currency note image for authenticity")
    parser.add_argument("--image", type=str, required=True, help="Path to currency note image (.jpg, .png)")
    parser.add_argument("--model-path", type=str, default=str(CURRENCY_MODEL_PATH), help="Path to trained model checkpoint")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at: {image_path}")
        sys.exit(1)

    verifier = CurrencyVerifier(model_path=args.model_path)
    result = verifier.verify(str(image_path))

    print("\n" + "=" * 55)
    print("       INDIAN CURRENCY VERIFICATION RESULT")
    print("=" * 55)
    print(f"\nImage File:  {image_path.name}")
    print(f"Verdict:     {result['verdict']}")
    print(f"Confidence:  {result['confidence']:.1f}%")
    print("\nAnalysis & Guidance:")
    print(result['analysis'])
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
