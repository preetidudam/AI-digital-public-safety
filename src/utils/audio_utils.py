"""
Audio helper functions using gTTS for generating synthetic test calls.
Matches Cells 58 and 66 of original notebook.
"""
from pathlib import Path
from typing import Optional, Union

from config import DEFAULT_SAMPLE_AUDIO_PATH

DEFAULT_SAMPLE_CALL_SCRIPT = """
Hello sir, I am calling from the police department.
Your bank account will be blocked today.
You must transfer money immediately to verify your account.
Do not disconnect this call.
""".strip()


def generate_scam_call_audio(
    text: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    lang: str = "en"
) -> Path:
    """
    Generate an MP3 audio file from text using Google Text-to-Speech (gTTS).
    Matches Cells 58 and 66 of original notebook.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError(
            "gTTS is not installed. Please install it using: pip install gTTS"
        )

    if text is None:
        text = DEFAULT_SAMPLE_CALL_SCRIPT

    if output_path is None:
        target_path = Path(DEFAULT_SAMPLE_AUDIO_PATH)
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic call audio using gTTS...")
    tts = gTTS(text=text, lang=lang)
    tts.save(str(target_path))
    print(f"Sample audio saved successfully at: {target_path}")

    return target_path
