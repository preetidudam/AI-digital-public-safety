"""
Script to generate sample scam-call audio using gTTS.
Matches Cells 58 and 66 of original notebook.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_SAMPLE_AUDIO_PATH
from src.utils.audio_utils import generate_scam_call_audio, DEFAULT_SAMPLE_CALL_SCRIPT


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic call audio using gTTS")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_SAMPLE_AUDIO_PATH),
        help="Target output mp3 file path"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_SAMPLE_CALL_SCRIPT,
        help="Text to synthesize"
    )
    args = parser.parse_args()

    print(f"Synthesizing text:\n---\n{args.text}\n---")
    output_file = generate_scam_call_audio(text=args.text, output_path=args.output)
    print(f"Generated sample audio at: {output_file}")


if __name__ == "__main__":
    main()
