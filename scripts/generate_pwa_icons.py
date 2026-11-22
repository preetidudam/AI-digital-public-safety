"""
Generate the PWA app icons (192x192 and 512x512) as a simple white shield on the theme color.
Usage: python scripts/generate_pwa_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

PWA_DIR = Path(__file__).resolve().parent.parent / "pwa"
THEME_COLOR = "#4f46e5"


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), THEME_COLOR)
    draw = ImageDraw.Draw(img)
    s = size / 100  # draw on a 100x100 grid, then scale

    # Shield outline: flat top, curved sides narrowing to a point
    shield = [(25, 22), (50, 14), (75, 22), (75, 48), (70, 63), (60, 75), (50, 84),
              (40, 75), (30, 63), (25, 48)]
    draw.polygon([(x * s, y * s) for x, y in shield], fill="white")

    # Check mark inside the shield
    check = [(38, 48), (47, 58), (64, 38)]
    draw.line([(x * s, y * s) for x, y in check], fill=THEME_COLOR, width=max(2, int(7 * s)), joint="curve")
    return img


def main():
    PWA_DIR.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        path = PWA_DIR / f"icon-{size}.png"
        make_icon(size).save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
