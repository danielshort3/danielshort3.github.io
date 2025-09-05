#!/usr/bin/env python3
"""
Resize img/ui/logo.png into multiple smaller variants for favicons/UI.

Outputs (into img/ui/):
  - logo-32.png   (small)
  - logo-64.png   (medium)
  - logo-180.png  (apple touch)
  - logo-192.png  (large / PWA)

Requires: Pillow (pip install pillow)
"""
import sys
from pathlib import Path

SIZES = [
    ("logo-16.png", 16),
    ("logo-32.png", 32),
    ("logo-64.png", 64),
    ("logo-180.png", 180),
    ("logo-192.png", 192),
]

root = Path(__file__).resolve().parent.parent
src = root / "img" / "ui" / "logo.png"
out_dir = root / "img" / "ui"

def gen_main_logo():
    try:
        from PIL import Image
    except Exception as e:
        print("[resize_logo] Pillow is required. Install via: pip install pillow", file=sys.stderr)
        sys.exit(2)

    if not src.exists():
        print(f"[resize_logo] Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGBA")

    for name, size in SIZES:
        target = out_dir / name
        # Preserve aspect, fit inside square, then paste centered
        ratio = min(size / im.width, size / im.height)
        new_w = max(1, int(round(im.width * ratio)))
        new_h = max(1, int(round(im.height * ratio)))
        resized = im.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ox = (size - new_w) // 2
        oy = (size - new_h) // 2
        canvas.paste(resized, (ox, oy), resized)

        canvas.save(target, format="PNG", optimize=True)
        print(f"[resize_logo] Wrote {target.relative_to(root)}")

    # Write a multi-size favicon.ico at repo root
    ico_path = root / "favicon.ico"
    try:
        # Pillow will resample to the provided sizes
        im.save(ico_path, format="ICO", sizes=[(16,16), (32,32), (48,48), (64,64)])
        print(f"[resize_logo] Wrote {ico_path.relative_to(root)}")
    except Exception as e:
        print(f"[resize_logo] Failed to write favicon.ico: {e}", file=sys.stderr)


def gen_cert_logos():
    """Generate small variants for certification logos.
    For each PNG in img/cert_logos, write -24.png (1x) and -48.png (2x)
    keeping aspect ratio and transparent background.
    """
    try:
        from PIL import Image
    except Exception:
        print("[resize_logo] Pillow is required for cert logos.", file=sys.stderr)
        return

    cert_dir = root / "img" / "cert_logos"
    if not cert_dir.exists():
        return
    for src in cert_dir.glob("*.png"):
        name = src.stem
        if name.endswith("-24") or name.endswith("-48"):
            continue
        try:
            im = Image.open(src).convert("RGBA")
        except Exception as e:
            print(f"[resize_logo] Skip {src.name}: {e}")
            continue
        for target_h in (24, 48):
            ratio = target_h / im.height
            new_w = max(1, int(round(im.width * ratio)))
            new_h = target_h
            resized = im.resize((new_w, new_h), Image.LANCZOS)
            out = cert_dir / f"{name}-{target_h}.png"
            resized.save(out, format="PNG", optimize=True)
            print(f"[resize_logo] Wrote {out.relative_to(root)}")


def main():
    gen_main_logo()
    gen_cert_logos()

if __name__ == "__main__":
    main()
