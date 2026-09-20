"""Create secondary brand exports from the approved DS master and website tokens.

Optional authoring command, separate from the website build:
  python build/generate-brand-assets.py

Requires fonttools (and the repository's existing Node.js/sharp installation).
All production lettering is outlined from the bundled Inter font. Edit copy,
layout, or weights here rather than patching exported SVG paths. The approved
master, historical hero backgrounds, and native Android resources are untouched.
"""

import argparse
import copy
import json
import re
import subprocess
import sys
from functools import lru_cache
from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "tmp/pdfs/brand-guide-deps"
if LOCAL_DEPS.exists():
  sys.path.insert(0, str(LOCAL_DEPS))

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

TOKENS = (ROOT / "css/variables.css").read_text(encoding="utf-8")
PERSONAL = json.loads((ROOT / "content/audiences/personal.json").read_text(encoding="utf-8"))
SETTINGS = json.loads((ROOT / "content/site/settings.json").read_text(encoding="utf-8"))
MASTER = ET.parse(ROOT / "img/brand/00-ds-logo-master-full-color.svg").getroot()
NAME = SETTINGS["ownerName"]
TAGLINE = PERSONAL["brandTagline"]
DOMAIN = SETTINGS["siteOrigin"].split("://", 1)[-1].removeprefix("www.").rstrip("/")


def token(name):
  match = re.search(r"--" + re.escape(name) + r":\s*([^;]+);", TOKENS)
  if not match:
    raise ValueError(f"Missing token: {name}")
  value = match.group(1).strip()
  alias = re.fullmatch(r"var\(--([^)]+)\)", value)
  return token(alias.group(1)) if alias else value


NAVY = token("brand-midnight")
BLUE = token("brand-signal-blue")
SLATE = token("brand-slate")
MIST = token("brand-mist")
TEAL = token("category-tools")
COPPER = token("category-games")
GRAPHITE = token("brand-graphite")


@lru_cache(maxsize=None)
def font(weight):
  value = TTFont(ROOT / "build/fonts/Inter-Latin.ttf")
  if "fvar" in value:
    axes = {axis.axisTag: weight if axis.axisTag == "wght" else axis.defaultValue for axis in value["fvar"].axes}
    value = instantiateVariableFont(value, axes, inplace=True)
  return value


def text_width(value, size, weight=400, tracking=0):
  face = font(weight)
  cmap = face.getBestCmap()
  return sum(face["hmtx"].metrics[cmap[ord(char)]][0] for char in value) * size / face["head"].unitsPerEm + max(0, len(value) - 1) * tracking


def text(value, x, baseline, size, weight=400, fill=NAVY, tracking=0):
  """Outline the exact bundled Inter glyphs; never depend on installed fonts."""
  face = font(weight)
  glyphs = face.getGlyphSet()
  cmap = face.getBestCmap()
  scale = size / face["head"].unitsPerEm
  parts = [f'<g aria-label="{escape(value, quote=True)}" fill="{fill}">']
  cursor = x
  for char in value:
    name = cmap[ord(char)]
    pen = SVGPathPen(glyphs)
    glyphs[name].draw(pen)
    commands = pen.getCommands()
    if commands:
      parts.append(f'<path transform="translate({cursor:.4f} {baseline:.4f}) scale({scale:.8f} {-scale:.8f})" d="{commands}"/>')
    cursor += face["hmtx"].metrics[name][0] * scale + tracking
  parts.append("</g>")
  return "\n".join(parts)


def wrap(value, size, width, weight=400):
  lines = []
  current = ""
  for word in value.split():
    candidate = f"{current} {word}".strip()
    if current and text_width(candidate, size, weight) > width:
      lines.append(current)
      current = word
    else:
      current = candidate
  if current:
    lines.append(current)
  if len(lines) == 2:
    words = value.split()
    candidates = []
    for index in range(1, len(words)):
      pair = [" ".join(words[:index]), " ".join(words[index:])]
      widths = [text_width(line, size, weight) for line in pair]
      if max(widths) <= width:
        candidates.append((abs(widths[0] - widths[1]), pair))
    if candidates:
      lines = min(candidates, key=lambda candidate: candidate[0])[1]
  return lines


def lines(value, x, baseline, size, width, leading, weight=400, fill=NAVY):
  return "\n".join(text(line, x, baseline + index * leading, size, weight, fill) for index, line in enumerate(wrap(value, size, width, weight)))


def logo(x, y, height, white=False):
  scale = height / 408
  elements = []
  for node in MASTER:
    if node.tag.rsplit("}", 1)[-1] not in {"path", "g"}:
      continue
    duplicate = copy.deepcopy(node)
    for child in duplicate.iter():
      child.tag = child.tag.rsplit("}", 1)[-1]
      if white and child.get("fill") and child.get("fill") != "none":
        child.set("fill", "#FFFFFF")
    elements.append(ET.tostring(duplicate, encoding="unicode"))
  return f'<g transform="translate({x:.4f} {y:.4f}) scale({scale:.8f}) translate(8 8)">' + "\n".join(elements) + "</g>"


def svg(width, height, title, description, body):
  return f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by build/generate-brand-assets.py. Lettering is outlined Inter. -->
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title>
<desc id="desc">{escape(description)}</desc>
{body}
</svg>
'''


def tabs(x, y, width, height, label_size=13):
  """A flat brand motif based on the real five-category navigation."""
  items = [("ABOUT", NAVY), ("PROJECTS", BLUE), ("TOOLS", TEAL), ("GAMES", COPPER), ("CONTACT", GRAPHITE)]
  rail = width / len(items)
  parts = [f'<defs><clipPath id="tabs-clip"><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12"/></clipPath></defs>', '<g clip-path="url(#tabs-clip)">']
  for index, (label, color) in enumerate(items):
    left = x + rail * index
    parts.append(f'<rect x="{left}" y="{y}" width="{rail}" height="{height}" fill="{color}"/>')
    label_width = text_width(label, label_size, 650, .35)
    center_x = left + rail / 2
    center_y = y + height / 2
    parts.append(f'<g transform="translate({center_x} {center_y}) rotate(-90)">')
    parts.append(text(label, -label_width / 2, label_size * .35, label_size, 650, "#FFFFFF", .35))
    parts.append("</g>")
  parts.append("</g>")
  return "\n".join(parts)


def wordmark(analytics=False):
  height = 200 if analytics else 160
  parts = [logo(24, 28 if analytics else 20, 120)]
  parts.append(text(NAME, 174, 96 if analytics else 100, 58, 750, tracking=-.8))
  if analytics:
    parts.append(text("Data Analytics & BI", 176, 142, 26, 400, SLATE))
  label = "analytics wordmark" if analytics else "personal wordmark"
  return svg(540, height, f"{NAME} {label}", f"Full-color DS mark beside {NAME}" + (", with the professional descriptor Data Analytics & BI." if analytics else ". The personal version has no audience-specific descriptor."), "\n".join(parts))


def social_card():
  parts = ['<rect width="1200" height="630" fill="#FFFFFF"/>', logo(64, 64, 88), text(NAME, 176, 126, 46, 750, tracking=-.65)]
  parts.append(lines(TAGLINE, 72, 304, 43, 720, 58, 600))
  parts.append(f'<path d="M72 478H780" stroke="{MIST}" stroke-width="2"/>')
  parts.append(text(DOMAIN, 72, 543, 25, 400, SLATE))
  parts.append(tabs(884, 124, 240, 376, 13))
  return svg(1200, 630, f"{NAME} — personal website", TAGLINE + " Projects, tools, and games at " + DOMAIN + ".", "\n".join(parts))


def github_banner():
  parts = ['<rect width="1600" height="800" fill="#FFFFFF"/>', logo(96, 92, 100), text(NAME, 224, 164, 56, 750, tracking=-.8)]
  parts.append(lines(TAGLINE, 104, 360, 55, 940, 74, 600))
  parts.append(f'<path d="M104 598H1020" stroke="{MIST}" stroke-width="2"/>')
  parts.append(text("Projects, tools, and games", 104, 656, 26, 400, SLATE))
  parts.append(text(DOMAIN, 104, 709, 26, 600, BLUE))
  parts.append(tabs(1152, 158, 328, 482, 17))
  return svg(1600, 800, f"{NAME} — GitHub portfolio banner", TAGLINE + " A personal portfolio with projects, tools, and games.", "\n".join(parts))


def linkedin_banner():
  # The lower-left area stays empty for LinkedIn's overlapping profile photo.
  parts = ['<rect width="1584" height="396" fill="#FFFFFF"/>', logo(346, 61, 72), text(NAME, 439, 112, 44, 750, tracking=-.6)]
  parts.append(lines(TAGLINE, 352, 211, 30, 805, 42, 500))
  parts.append(text(DOMAIN, 352, 329, 23, 400, SLATE))
  parts.append(tabs(1243, 62, 245, 272, 12))
  return svg(1584, 396, f"{NAME} — LinkedIn banner", TAGLINE + " The lower-left profile-photo area remains clear.", "\n".join(parts))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=ROOT / "img/brand")
  parser.add_argument("--node", default="node", help="Node.js executable for the existing sharp rasterizer")
  parser.add_argument("--svg-only", action="store_true")
  args = parser.parse_args()
  destination = args.output_dir.resolve()
  destination.mkdir(parents=True, exist_ok=True)
  artwork = {
    "04c-ds-logo-all-white.svg": svg(397, 408, f"{NAME} all-white DS mark", "Single-color white DS artwork on a transparent canvas, for dark surfaces.", logo(0, 0, 408, white=True)),
    "06-wordmark-horizontal-assembly.svg": wordmark(),
    "06b-wordmark-analytics.svg": wordmark(analytics=True),
    "09-linkedin-banner-1584x396.svg": linkedin_banner(),
    "10-github-readme-portfolio-banner.svg": github_banner(),
    "personal-social-card.svg": social_card(),
  }
  for name, content in artwork.items():
    (destination / name).write_text(content, encoding="utf-8", newline="\n")
  if not args.svg_only:
    files = [str(destination / name) for name in artwork]
    script = '''const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const files = JSON.parse(fs.readFileSync(0, 'utf8'));
Promise.all(files.map(file => sharp(file).png({ compressionLevel: 9, adaptiveFiltering: true }).toFile(file.replace(/\\.svg$/, '.png'))))
  .catch(error => { console.error(error); process.exitCode = 1; });'''
    subprocess.run([args.node, "-e", script], input=json.dumps(files), cwd=ROOT, text=True, check=True)
  print(f"Generated {len(artwork)} SVG" + (" assets" if args.svg_only else " and PNG asset pairs") + f" in {destination}")


if __name__ == "__main__":
  main()
