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
_currency_verifier = None


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


def get_currency_verifier():
    """
    Lazy load the Currency Verifier vision model.
    """
    global _currency_verifier
    if _currency_verifier is None:
        from src.models.currency_verifier import CurrencyVerifier
        _currency_verifier = CurrencyVerifier()
    return _currency_verifier


def analyze_currency(image):
    """
    Verify currency note image using transfer-learning CNN.
    """
    if image is None:
        return "Please upload a currency note image.", 0.0, "No image was provided."
    verifier = get_currency_verifier()
    res = verifier.verify(image)
    return res["verdict"], res["confidence"], res["analysis"]


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


# Design tokens: single source of truth for the UI's look.
# Neutral gray scale + one accent (indigo-600, 6.3:1 contrast on white).
CSS = """
:root {
    --ds-accent: #4f46e5;
    --ds-accent-hover: #4338ca;
    --ds-accent-ring: rgba(79, 70, 229, 0.35);
    --ds-bg: #fafafa;
    --ds-surface: #ffffff;
    --ds-border: #e5e7eb;
    --ds-text: #111827;
    --ds-text-muted: #4b5563;
    --ds-text-subtle: #6b7280;
    --ds-radius: 8px;
    --ds-transition: 150ms ease;
    color-scheme: light;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 32px 16px !important;
    background: var(--ds-bg);
}

/* Header */
.ds-header { margin-bottom: 24px; }
.ds-header h1 {
    font-size: 24px; font-weight: 600; color: var(--ds-text);
    margin: 0 0 4px; letter-spacing: -0.01em;
}
.ds-header p { font-size: 14px; color: var(--ds-text-muted); margin: 0; }

/* Section labels inside each tab */
.ds-section-title p {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--ds-text-subtle); margin: 8px 0 0;
}

/* Interaction states */
button, input, textarea { transition: border-color var(--ds-transition), background-color var(--ds-transition), box-shadow var(--ds-transition); }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible {
    outline: 2px solid var(--ds-accent) !important;
    outline-offset: 2px;
}
button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Footer note */
.ds-note {
    margin-top: 32px; padding-top: 16px;
    border-top: 1px solid var(--ds-border);
}
.ds-note, .ds-note p, .ds-note li { font-size: 12px !important; color: var(--ds-text-subtle) !important; line-height: 1.6; }
.ds-note strong { color: var(--ds-text-muted); font-weight: 600; }

@media (max-width: 640px) {
    .gradio-container { padding: 16px !important; }
    .ds-header h1 { font-size: 20px; }
}

/* Mobile: no sideways scrolling, finger-sized buttons and tabs */
html, body { max-width: 100%; overflow-x: hidden; }
.gradio-container img, .gradio-container video, .gradio-container audio { max-width: 100%; }
button { min-height: 44px; }
[role="tab"] { min-height: 44px; padding-left: 12px !important; padding-right: 12px !important; }
@media (max-width: 640px) {
    .gradio-container .row { flex-wrap: wrap !important; }
    .gradio-container .row > * { min-width: 100% !important; }
    .gradio-container button.primary { width: 100%; min-height: 48px; font-size: 16px !important; }
    /* 16px inputs stop phones from zooming in when you tap a text box */
    .gradio-container textarea, .gradio-container input { font-size: 16px !important; }
}
"""

# PWA + mobile tags added to the page <head>: viewport, app manifest, theme color,
# service worker registration, and rear-camera capture for the currency upload.
PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#4f46e5">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="ScamShield">
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/icon-192.png">
<script>
if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
        navigator.serviceWorker.register("/sw.js").catch((err) => console.warn("SW registration failed:", err));
    });
}
// Let the currency verifier open the phone's back camera directly.
new MutationObserver(() => {
    document.querySelectorAll('#currency-input input[type="file"]').forEach((el) => {
        el.setAttribute("accept", "image/*");
        el.setAttribute("capture", "environment");
    });
}).observe(document.documentElement, { childList: true, subtree: true });
</script>
"""

PWA_DIR = Path(__file__).resolve().parent / "pwa"

# Force light mode regardless of OS preference.
FORCE_LIGHT_JS = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.href);
    }
}
"""


def build_theme():
    """
    Minimal neutral theme with a single indigo accent.
    """
    import gradio as gr

    return gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="gray",
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
    ).set(
        body_background_fill="#fafafa",
        body_text_color="#111827",
        body_text_color_subdued="#4b5563",
        block_background_fill="#ffffff",
        block_border_color="#e5e7eb",
        block_border_width="1px",
        block_shadow="none",
        block_radius="8px",
        block_label_text_size="14px",
        block_label_text_weight="500",
        block_label_text_color="#374151",
        block_title_text_weight="500",
        input_background_fill="#ffffff",
        input_border_color="#d1d5db",
        input_border_color_focus="#4f46e5",
        input_shadow="none",
        input_shadow_focus="0 0 0 3px rgba(79, 70, 229, 0.15)",
        input_radius="6px",
        button_primary_background_fill="#4f46e5",
        button_primary_background_fill_hover="#4338ca",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#4f46e5",
        button_primary_shadow="none",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f3f4f6",
        button_secondary_border_color="#d1d5db",
        button_secondary_text_color="#374151",
        button_large_radius="6px",
        button_large_text_size="14px",
        button_large_text_weight="500",
        button_large_padding="10px 16px",
        shadow_drop="none",
        shadow_drop_lg="0 1px 2px rgba(0, 0, 0, 0.04)",
    )


def build_app():
    """
    Build the Gradio interface. Matches Cell 70 layout.
    """
    import gradio as gr

    with gr.Blocks(title="Digital Public Safety") as demo:
        gr.HTML(
            """
            <header class="ds-header">
                <h1>Digital Public Safety</h1>
                <p>AI-assisted scam detection for messages, phone calls, and counterfeit currency.</p>
            </header>
            """
        )

        with gr.Tab("Message / Transcript"):
            text_input = gr.Textbox(
                label="Paste SMS or call transcript",
                placeholder="Paste suspicious message or conversation here...",
                lines=7
            )
            text_button = gr.Button("Analyze Text", variant="primary")

            gr.Markdown("Results", elem_classes="ds-section-title")
            text_output = gr.Textbox(label="Analyzed Text", lines=5)
            with gr.Row():
                text_score = gr.Number(label="Potential Scam Score (%)", scale=1)
                text_alert = gr.Textbox(label="Safety Alert", lines=3, scale=2)

            text_button.click(
                fn=analyze_text,
                inputs=text_input,
                outputs=[text_output, text_score, text_alert]
            )

        with gr.Tab("Audio Call"):
            audio_input = gr.Audio(
                label="Upload a phone-call recording",
                type="filepath"
            )
            audio_button = gr.Button("Transcribe & Analyze", variant="primary")

            gr.Markdown("Results", elem_classes="ds-section-title")
            audio_transcript = gr.Textbox(
                label="Detected Speech / Transcript",
                lines=7
            )
            with gr.Row():
                audio_score = gr.Number(label="Potential Scam Score (%)", scale=1)
                audio_alert = gr.Textbox(label="Safety Alert", lines=3, scale=2)

            audio_button.click(
                fn=analyze_audio,
                inputs=audio_input,
                outputs=[audio_transcript, audio_score, audio_alert]
            )

        with gr.Tab("Currency Verifier"):
            currency_input = gr.Image(
                label="Upload Currency Note (Front or Back)",
                type="pil",
                elem_id="currency-input"
            )
            currency_button = gr.Button("Verify Banknote Authenticity", variant="primary")

            gr.Markdown("Results", elem_classes="ds-section-title")
            with gr.Row():
                currency_verdict = gr.Textbox(label="Verification Result", lines=2, scale=2)
                currency_score = gr.Number(label="Confidence Score (%)", scale=1)
            currency_analysis = gr.Textbox(label="Model Findings & Guidance", lines=9)

            currency_button.click(
                fn=analyze_currency,
                inputs=currency_input,
                outputs=[currency_verdict, currency_score, currency_analysis]
            )

        gr.Markdown(
            """
            **Important:**
            - Scam scores are model estimates, not proof of fraud. A low score does not guarantee that a call or message is safe.
            - Audio analysis checks spoken conversational content, not whether a voice is AI-synthesized.
            - Currency verification evaluates 2D surface printing patterns via CNN; physical security features (intaglio tactile ink relief, paper feel, UV fluorescence) require certified physical inspection.
            """,
            elem_classes="ds-note"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Digital Public Safety Gradio Web App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    demo = build_app()

    if args.share:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=True,
            theme=build_theme(),
            css=CSS,
            js=FORCE_LIGHT_JS,
            head=PWA_HEAD
        )
        return

    # Serve the PWA files from the site root, then mount the Gradio UI + API at "/".
    # Everything runs on ONE port, so a single "ngrok http <port>" link works.
    import uvicorn
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    server = FastAPI()

    @server.get("/manifest.json", include_in_schema=False)
    def manifest():
        return FileResponse(PWA_DIR / "manifest.json", media_type="application/manifest+json")

    @server.get("/sw.js", include_in_schema=False)
    def service_worker():
        return FileResponse(
            PWA_DIR / "sw.js",
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"}
        )

    @server.get("/icon-192.png", include_in_schema=False)
    def icon_192():
        return FileResponse(PWA_DIR / "icon-192.png", media_type="image/png")

    @server.get("/icon-512.png", include_in_schema=False)
    def icon_512():
        return FileResponse(PWA_DIR / "icon-512.png", media_type="image/png")

    server = gr.mount_gradio_app(
        server,
        demo,
        path="/",
        theme=build_theme(),
        css=CSS,
        js=FORCE_LIGHT_JS,
        head=PWA_HEAD,
        favicon_path=str(PWA_DIR / "icon-192.png")
    )

    print(f"* Running on http://{args.host}:{args.port}  (for phone: ngrok http {args.port})")
    uvicorn.run(server, host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
