"""Build the brand reference from editable copy, current tokens and real site assets.

Requires reportlab, Pillow and fonttools[woff]. Refresh screenshots with
node build/capture-brand-guide.cjs before regenerating after a visual change.
This document task is intentionally separate from the website build.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "tmp/pdfs/brand-guide-deps"
if LOCAL_DEPS.exists():
  sys.path.insert(0, str(LOCAL_DEPS))

from fontTools.ttLib import TTFont as SourceFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

W, H = 842, 595
M = 40
NAVY = "#091F3B"
BLUE = "#005FED"
SLATE = "#475569"
LINE = "#DCE3EC"
MIST = "#EEF2F7"
WHITE = "#FFFFFF"
ASSETS = ROOT / "docs/brand-guide-assets"
TOKENS = (ROOT / "css/variables.css").read_text(encoding="utf-8")


def token(name):
  match = re.search(r"--" + re.escape(name) + r":\s*([^;]+);", TOKENS)
  if not match:
    raise ValueError(f"Missing brand token: {name}")
  value = match.group(1).strip()
  alias = re.fullmatch(r"var\(--([^)]+)\)", value)
  return token(alias.group(1)) if alias else value


def load_fonts():
  folder = ROOT / "tmp/pdfs/brand-guide-fonts"
  folder.mkdir(parents=True, exist_ok=True)
  for name, weight in [("Inter", 400), ("InterMedium", 600), ("InterBold", 750)]:
    font = SourceFont(ROOT / "css/fonts/Inter-Latin.woff2")
    if "fvar" in font:
      axes = {axis.axisTag: (weight if axis.axisTag == "wght" else axis.defaultValue) for axis in font["fvar"].axes}
      font = instantiateVariableFont(font, axes, inplace=True)
    # Distinct PostScript names prevent PDF font subsetting from merging weights.
    for name_id, value in [(1, name), (2, "Regular"), (4, name), (6, name)]:
      font["name"].setName(value, name_id, 3, 1, 0x409)
      font["name"].setName(value, name_id, 1, 0, 0)
    font.flavor = None
    file = folder / f"{name}.ttf"
    font.save(file)
    pdfmetrics.registerFont(TTFont(name, str(file)))


class Guide:
  def __init__(self, data, output):
    self.data = data
    self.c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1)
    self.c.setTitle("Daniel Short - Brand & website guide")
    self.c.setAuthor("Daniel Short")
    self.c.setSubject("Current website identity, visual standards and proposed refinements")
    self.layouts = []

  def box(self, x, y, w, h, fill=WHITE, stroke=None, radius=0):
    self.c.setFillColor(colors.HexColor(fill))
    self.c.setStrokeColor(colors.HexColor(stroke or fill))
    self.c.setLineWidth(.7)
    self.c.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=bool(stroke))

  def line(self, x, y, w, color=LINE, width=.7):
    self.c.setStrokeColor(colors.HexColor(color))
    self.c.setLineWidth(width)
    self.c.line(x, H-y, x+w, H-y)

  def text(self, txt, x, y, width, size=11.3, color=NAVY, font="Inter", leading=None, limit=None):
    style = ParagraphStyle("brand", fontName=font, fontSize=size, leading=leading or size*1.38,
      textColor=colors.HexColor(color), splitLongWords=True, spaceAfter=0)
    p = Paragraph(txt, style)
    _, height = p.wrap(width, H)
    if limit is not None and height > limit:
      raise ValueError(f"Text overflow ({height:.1f} > {limit}): {txt[:75]}")
    if y+height > 552:
      raise ValueError(f"Text enters footer: {txt[:75]}")
    p.drawOn(self.c, x, H-y-height)
    self.layouts.append({"page": self.c.getPageNumber(), "x": x, "y": y, "w": width, "h": height, "text": txt})
    return height

  def logo(self, x, y, size, mode="color"):
    # Copy the approved SVG path geometry directly into vector PDF paths.
    svg = ElementTree.parse(ROOT / "img/brand/00-ds-logo-master-full-color.svg").getroot()
    self.c.saveState()
    scale = size/408
    self.c.translate(x+8*scale, H-y-8*scale)
    self.c.scale(scale, -scale)
    for element in svg.iter():
      if not element.tag.endswith("path"):
        continue
      commands = re.findall(r"[MLZ]|-?\d+(?:\.\d+)?", element.get("d", ""))
      if set(re.findall(r"[a-zA-Z]", element.get("d", ""))) - set("MLZ"):
        raise ValueError("Logo geometry changed; update the SVG path reader before publishing.")
      p = self.c.beginPath()
      index = 0
      while index < len(commands):
        command = commands[index]
        index += 1
        if command == "Z":
          p.close()
        else:
          px, py = float(commands[index]), float(commands[index+1])
          index += 2
          (p.moveTo if command == "M" else p.lineTo)(px, py)
      fill = element.get("fill", NAVY)
      if mode == "mono":
        fill = NAVY
      elif mode == "reversed" and fill.lower() != BLUE.lower():
        fill = WHITE
      self.c.setFillColor(colors.HexColor(fill))
      self.c.drawPath(p, fill=1, stroke=0, fillMode=0)
    self.c.restoreState()

  def image(self, file, x, y, w, h, border=False):
    image = Image.open(file)
    if image.mode not in ("RGB", "RGBA"):
      image = image.convert("RGBA")
    scale = min(w/image.width, h/image.height)
    iw, ih = image.width*scale, image.height*scale
    ix, iy = x+(w-iw)/2, y+(h-ih)/2
    self.c.drawImage(ImageReader(image), ix, H-iy-ih, iw, ih, mask="auto")
    if border:
      self.c.setStrokeColor(colors.HexColor(LINE))
      self.c.setLineWidth(.7)
      self.c.rect(ix, H-iy-ih, iw, ih, fill=0, stroke=1)

  def header(self, page, number):
    self.box(0, 0, W, H)
    self.text(f"DANIEL SHORT  /  {page['kicker'].upper()}", M, 25, 710, 9, BLUE, "InterMedium")
    self.logo(W-67, 23, 27)
    self.text(page["title"], M, 55, W-2*M, 28, font="InterBold", limit=40)
    self.text(page["subtitle"], M, 98, W-2*M, 11.5, SLATE, limit=32)
    self.line(M, 560, W-2*M)
    self.c.setFillColor(colors.HexColor(SLATE))
    self.c.setFont("Inter", 8)
    self.c.drawString(M, 18, f"Brand & website guide  |  Edition {self.data['edition']}  |  {self.data['updated']}")
    self.c.drawRightString(W-M, 18, f"{number:02d} / {len(self.data['pages']):02d}")

  def sections(self, page, y=368):
    sections = page["sections"]
    columns = 2 if len(sections) in (2, 4) else 3
    gap = 24
    width = (W-2*M-gap*(columns-1))/columns
    row_height = 113 if len(sections) == 4 else 178
    for i, section in enumerate(sections):
      x = M+(i % columns)*(width+gap)
      top = y+(i//columns)*row_height
      self.line(x, top, width, BLUE if page["kind"] == "improvements" else LINE, 1.2)
      th = self.text(section["title"], x, top+10, width, 12.2, font="InterMedium", limit=35)
      self.text(section["body"], x, top+16+th, width, 10.5 if len(sections)==4 else 11.1, SLATE,
        leading=14, limit=row_height-26-th)

  def figure(self, page):
    kind = page["kind"]
    if kind == "foundation":
      self.logo(47, 154, 62)
      self.text("Daniel Short", 125, 164, 260, 24, font="InterBold")
      self.text("Solving everyday problems<br/>with data and<br/>thoughtful tools.", 48, 239, 328, 21, leading=28, font="InterMedium")
      self.image(ASSETS/"home-desktop.png", 396, 141, 405, 214, True)
    elif kind == "logo":
      for x, mode, label in [(48, "color", "Master / light surfaces"), (302, "mono", "Monochrome / one color"), (556, "white", "All white / dark surfaces")]:
        self.box(x, 143, 234, 162, NAVY if mode == "white" else "#F8FAFC", radius=10)
        if mode == "white":
          self.image(ROOT/"img/brand/04c-ds-logo-all-white.png", x+57, 154, 126, 126)
        else:
          self.logo(x+57, 154, 126, mode)
        self.text(label, x, 314, 234, 10, SLATE)
      self.text("Header mark: 38px desktop / 40px mobile. Preserve the full artwork and adjacent name.", M, 340, 760, 10, SLATE)
    elif kind == "palette":
      swatches = [
        ("Midnight", "brand-midnight"), ("Signal / Projects", "brand-signal-blue"), ("Deep blue", "brand-deep-blue"),
        ("Tools", "category-tools"), ("Games", "category-games"),
        ("White", None), ("Canvas", "brand-canvas"), ("Mist", "brand-mist"), ("Graphite / Contact", "brand-graphite"), ("Slate", "brand-slate"),
        ("Ink", "brand-ink"), ("Amber accent", "brand-action-copper"), ("Warning text", "warning-text"), ("Success", "brand-success"), ("Error", "brand-risk")]
      for i, (label, name) in enumerate(swatches):
        x, y = M+(i % 5)*155, 142+(i//5)*70
        fill = token(name) if name else WHITE
        self.box(x, y, 141, 30, fill, LINE if fill in [WHITE,"#F9F9FA","#EEF2F7"] else None, 5)
        self.text(label, x, y+35, 143, 9, font="InterMedium")
        self.text(fill.upper(), x, y+49, 143, 8.2, SLATE)
    elif kind == "type":
      self.text("Inter", 45, 146, 295, 57, font="InterBold")
      self.text("Clear, direct, useful.", 47, 224, 315, 23, font="InterMedium")
      self.text("Aa Bb Cc 0123456789", 48, 270, 325, 19)
      rows = [("Page title", "28-36px / 750 / 1.15"), ("Masthead description", "16px / 400 / 1.5"),
        ("Library item title", "16px"), ("Library description", "14px"), ("Standard action", "14px / 44px target")]
      for i, (label, value) in enumerate(rows):
        y = 150+i*36
        self.text(label, 404, y, 180, 11, font="InterMedium")
        self.text(value, 594, y, 206, 10, SLATE)
        self.line(404, y+26, 394)
    elif kind == "layout":
      self.image(ASSETS/"projects-desktop.png", M, 140, 480, 210, True)
      for i, (label, value) in enumerate([("Homepage frame", "6px category border"), ("Library / detail frame", "4px category border"), ("Masthead divider", "2px tinted category line"), ("Internal boundary", "1px neutral line")]):
        self.text(label, 555, 149+i*48, 245, 11.5, font="InterMedium")
        self.text(value, 555, 168+i*48, 245, 10.5, SLATE)
    elif kind == "components":
      self.image(ASSETS/"text-compare-desktop.png", M, 140, 486, 210, True)
      self.box(563, 149, 160, 37, BLUE, radius=8)
      self.text("Primary action", 580, 159, 135, 11, WHITE, "InterMedium")
      self.box(563, 199, 160, 37, WHITE, LINE, 8)
      self.text("Secondary action", 576, 209, 145, 11, NAVY, "InterMedium")
      self.text("Tertiary text link", 563, 253, 218, 11, BLUE, "InterMedium")
      self.text("Success   Warning   Error", 563, 299, 230, 10.5, SLATE)
      self.text("Always include a visible label.", 563, 320, 230, 10, SLATE)
    elif kind == "imagery":
      for i, (file, label) in enumerate([("about-ai-network-v1.webp", "AI & machine learning"), ("about-family-frame-v1.webp", "Family"), ("about-french-horn-sheet-music-v1.webp", "French horn")]):
        x = M+i*257
        self.box(x+53, 145, 138, 138, "#F8FAFC", LINE, 10)
        self.image(ROOT/"img/hero"/file, x+65, 157, 114, 114)
        self.text(label, x+24, 297, 235, 12, font="InterMedium")
      self.text("Existing approved illustrations. Interface glyphs remain simple single-color outlines.", M, 333, 760, 10.5, SLATE)
    elif kind == "responsive":
      self.image(ASSETS/"projects-desktop.png", M, 141, 485, 210, True)
      self.image(ASSETS/"tools-mobile.png", 557, 139, 130, 215, True)
      self.text("1440px<br/>desktop", 714, 160, 85, 10, SLATE)
      self.text("390px<br/>mobile", 714, 216, 85, 10, SLATE)
      self.text("Also check<br/>320px", 714, 272, 85, 10, SLATE)
    elif kind == "voice":
      self.box(M, 143, W-2*M, 183, "#F8FAFC", radius=10)
      self.text("The personal introduction", 62, 160, 600, 10, BLUE, "InterMedium")
      self.text("I use data, AI, and machine learning to solve everyday problems. Family is important to me, and I've played French horn for 20 years.",
        62, 191, 702, 21, leading=29, font="InterMedium")
      self.text("Current About copy. Keep personal details specific and work claims evidence-based.", 62, 292, 695, 10, SLATE)
    elif kind == "applications":
      self.image(ROOT/"img/brand/personal-social-card.png", M, 141, 414, 217, True)
      self.text("One current identity", 489, 153, 298, 18, font="InterMedium")
      self.text("Refreshed social preview<br/>1200 x 630px", 489, 190, 298, 11, SLATE, leading=17)
      self.text("Plain personal wordmark", 489, 239, 298, 11, font="InterMedium")
      self.image(ROOT/"img/brand/06-wordmark-horizontal-assembly.png", 483, 260, 300, 66)
      self.text("Use the analytics variant for that audience.", 489, 340, 298, 10, SLATE)
    elif kind == "improvements":
      self.box(M, 143, W-2*M, 121, "#F8FAFC", LINE, 10)
      self.text("Current favicon exports", 58, 157, 260, 11, font="InterMedium")
      for x, size in [(63, 16), (113, 32), (177, 64)]:
        self.image(ROOT/f"img/ui/logo-{size}.png", x, 232-size, size, size)
        self.text(f"{size}px", x-3, 238, 50, 9, SLATE)
      self.text("The master stays. The kit improves.", 289, 163, 471, 19, font="InterMedium")
      self.text("Locally implemented: clearer favicon, production wordmarks, refreshed social assets, and flat header artwork.", 289, 201, 466, 11, SLATE, limit=42)
      self.sections(page, 289)
      return
    elif kind == "governance":
      entries = [("1", "Author", "Use approved assets and current tokens."), ("2", "Compare", "Review desktop, mobile and tiny icon sizes."),
        ("3", "Validate", "Check copy, contrast, clipping and controls."), ("4", "Release", "Approve, regenerate exports, then publish.")]
      for i, (n, title, desc) in enumerate(entries):
        y = 143+i*44
        self.box(M, y, 28, 28, BLUE, radius=6)
        self.text(n, M+9, y+5, 19, 12, WHITE, "InterMedium")
        self.text(title, 86, y+3, 113, 12, font="InterMedium")
        self.text(desc, 211, y+3, 570, 11, SLATE)
      self.text('Reference: <link href="https://www.ibm.com/design/language/ibm-logos/8-bar/" color="#005FED">IBM logo usage</link>  |  <link href="https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html" color="#005FED">Text contrast</link>  |  <link href="https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html" color="#005FED">Non-text contrast</link>  |  <link href="https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html" color="#005FED">Target size</link>  |  <link href="https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html" color="#005FED">Motion</link>', M, 328, 760, 10)
    else:
      raise ValueError(f"Unknown page kind: {kind}")
    self.sections(page)

  def build(self):
    for number, page in enumerate(self.data["pages"], 1):
      self.c.bookmarkPage(page["id"])
      self.c.addOutlineEntry(page["title"], page["id"], level=0)
      self.header(page, number)
      self.figure(page)
      self.c.showPage()
    self.c.save()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, default=ROOT / "documents/brand_guide.pdf")
  args = parser.parse_args()
  data = json.loads((ROOT / "docs/brand-guide.json").read_text(encoding="utf-8"))
  load_fonts()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  guide = Guide(data, args.output)
  guide.build()
  qa = ROOT / "tmp/pdfs/brand-guide-layout.json"
  qa.parent.mkdir(parents=True, exist_ok=True)
  qa.write_text(json.dumps(guide.layouts, indent=2), encoding="utf-8")
  print(f"Built {len(data['pages'])} pages: {args.output}")


if __name__ == "__main__":
  main()
