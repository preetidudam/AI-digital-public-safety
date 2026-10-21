"""
Digital Public Safety - Interactive Gradio Web Application.
AI-assisted scam detection for messages and phone call recordings.
Matches Cell 70 of original notebook.
"""
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import IMPROVED_MODEL_PATH, DEFAULT_IMPROVED_THRESHOLD

# Global instances for lazy loading
_detector = None
_transcriber = None


def get_detector():
    """
    Lazy load the scam detector model.
    """
    global _detector
    if _detector is None:
        if not IMPROVED_MODEL_PATH.exists():
            return None
        from src.models.detector import ScamDetector
        _detector = ScamDetector(model_path=IMPROVED_MODEL_PATH)
    return _detector


def get_transcriber():
    """
    Lazy load the Whisper speech-to-text transcriber.
    """
    global _transcriber
    if _transcriber is None:
        try:
            from src.models.audio_detector import WhisperTranscriber
            _transcriber = WhisperTranscriber()
        except Exception as e:
            print(f"Warning: Whisper model could not be initialized: {e}")
            return None
    return _transcriber


def analyze_text(text: str):
    """
    Analyze text message or transcript. Matches Cell 70 of original notebook.
    """
    if not text or not text.strip():
        return "", 0.0, "Please enter a transcript or SMS."

    detector = get_detector()
    if detector is None:
        return (
            text,
            0.0,
            f"⚠️ Model not found at '{IMPROVED_MODEL_PATH}'. "
            f"Please run 'python scripts/train_icfd_improved.py' first."
        )

    res = detector.analyze(text)
    return res["text"], res["score_percent"], res["alert"]


def analyze_audio(audio_path: str):
    """
    Transcribe audio and analyze transcribed content. Matches Cell 70 of original notebook.
    """
    if not audio_path or not os.path.exists(audio_path):
        return "", 0.0, "Please upload an audio recording."

    transcriber = get_transcriber()
    if transcriber is None:
        return (
            "",
            0.0,
            "⚠️ Speech-to-text model is unavailable. Ensure 'openai-whisper' and 'torch' are installed."
        )

    try:
        transcript = transcriber.transcribe(audio_path)
        if not transcript:
            return "", 0.0, "No speech could be transcribed."

        return analyze_text(transcript)

    except Exception as e:
        return "", 0.0, f"Audio analysis error: {str(e)}"


def build_app():
    """
    Build the Gradio interface. Matches Cell 70 layout.
    """
    import gradio as gr

    with gr.Blocks(title="Digital Public Safety") as demo:
        gr.Markdown(
            """
            # 🛡️ Digital Public Safety
            ### AI-assisted scam detection for messages and phone calls

            Paste a message or upload a call recording to begin.
            """
        )

        with gr.Tab("📩 SMS / Transcript Detection"):
            text_input = gr.Textbox(
                label="Paste SMS or call transcript",
                placeholder="Paste suspicious message or conversation here...",
                lines=7
            )
            text_button = gr.Button("Analyze Text", variant="primary")

            text_output = gr.Textbox(label="Analyzed Text", lines=5)
            text_score = gr.Number(label="Potential Scam Score (%)")
            text_alert = gr.Textbox(label="Safety Alert", lines=3)

            text_button.click(
                fn=analyze_text,
                inputs=text_input,
                outputs=[text_output, text_score, text_alert]
            )

        with gr.Tab("🎙️ Audio Call Detection"):
            audio_input = gr.Audio(
                label="Upload a phone-call recording",
                type="filepath"
            )
            audio_button = gr.Button("Transcribe & Analyze", variant="primary")

            audio_transcript = gr.Textbox(
                label="Detected Speech / Transcript",
                lines=7
            )
            audio_score = gr.Number(label="Potential Scam Score (%)")
            audio_alert = gr.Textbox(label="Safety Alert", lines=3)

            audio_button.click(
                fn=analyze_audio,
                inputs=audio_input,
                outputs=[audio_transcript, audio_score, audio_alert]
            )

        gr.Markdown(
            """
            **Important:** Scores are model estimates, not proof of fraud.
            A low score does not establish that a call is safe.
            Audio analysis checks spoken content, not whether a voice is AI-generated.
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Digital Public Safety Gradio Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
