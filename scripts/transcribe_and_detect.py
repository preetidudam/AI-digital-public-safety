"""
Command-line utility to transcribe an audio call recording and analyze it for scam intent.
Matches Cells 59-60, 67-69 of original notebook.
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

from config import DEFAULT_SAMPLE_AUDIO_PATH, IMPROVED_MODEL_PATH
from src.models.audio_detector import WhisperTranscriber
from src.models.detector import ScamDetector


def transcribe_and_detect(audio_path: Path, model_path: Path):
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file '{audio_path}' does not exist. "
            f"Run 'python scripts/generate_sample_audio.py' to generate a sample recording."
        )

    # 1. Transcribe audio
    print(f"Transcribing '{audio_path}' with Whisper speech-to-text...")
    transcriber = WhisperTranscriber()
    transcript = transcriber.transcribe(audio_path)

    if not transcript:
        raise ValueError("No speech could be transcribed from the provided audio file.")

    # 2. Analyze transcript
    print(f"Analyzing transcript with ScamDetector...")
    detector = ScamDetector(model_path=model_path)
    score = detector.predict_score(transcript)
    threshold = detector.threshold

    # 3. Print Alert Banner (matches Cell 69)
    print("\n" + "=" * 45)
    print("       DIGITAL PUBLIC SAFETY ALERT")
    print("=" * 45)
    print("\nCall transcript:")
    print(transcript)
    print(f"\nPotential scam score: {score:.1%}")
    print(f"Alert threshold: {threshold:.0%}")

    if score >= threshold:
        print("\n⚠️ ALERT: POTENTIAL SCAM CALL DETECTED!")
        print("Recommended action:")
        print("- Do not transfer money under pressure.")
        print("- Never share OTPs, PINs, or passwords.")
        print("- Verify the caller through an official channel.")
    else:
        print("\n✅ No alert triggered by the current model.")
        print("This does not guarantee the call is safe.")

    print("\n" + "=" * 45)


def main():
    parser = argparse.ArgumentParser(description="Transcribe and analyze audio call for scam detection")
    parser.add_argument(
        "--audio",
        type=str,
        default=str(DEFAULT_SAMPLE_AUDIO_PATH),
        help="Path to audio file (.mp3, .wav, .m4a)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(IMPROVED_MODEL_PATH),
        help="Path to saved model package (.joblib)"
    )
    args = parser.parse_args()

    transcribe_and_detect(Path(args.audio), Path(args.model_path))


if __name__ == "__main__":
    main()
