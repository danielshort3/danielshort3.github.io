# Brand guide maintenance

The current document is [`documents/brand_guide.pdf`](../documents/brand_guide.pdf).
Its editable copy is [`brand-guide.json`](brand-guide.json), and its visual renderer
is [`build/generate-brand-guide.py`](../build/generate-brand-guide.py). The PDF is
a concise reference for the current website, with proposals kept on a separate page.

## Source ownership

- Website colors and semantic geometry: [`css/variables.css`](../css/variables.css)
  and the relevant components listed in [the repository map](REPOSITORY_MAP.md).
- Copy and page organization: `docs/brand-guide.json`.
- Approved logo artwork: `img/brand/`. The PDF embeds the master's vector path
  geometry directly; it does not substitute an AI-generated logo.
- Actual website examples: `docs/brand-guide-assets/`, refreshed by
  `build/capture-brand-guide.cjs` from a running, freshly built local preview.
- Shared implementation conventions: [visual-style.md](visual-style.md).

The renderer reads palette values from the CSS tokens. Figure annotations that
describe typography, border sizes, or controls live in the Python renderer and
must be checked against the relevant CSS when those standards change. The page
`sourceNotes` fields record the evidence for each section without crowding the PDF.

## Regenerate

Use Python with `reportlab`, `Pillow`, and `fonttools[woff]`; the renderer derives
the Inter weights from the existing variable font. The normal website build
does not require Python and does not regenerate this document.

1. Build the website and run its local preview with the repository's normal commands.
2. Refresh examples with `node build/capture-brand-guide.cjs`. It defaults to
   `http://127.0.0.1:4173`; set `BRAND_GUIDE_URL` to another local preview if needed.
   Captures block external requests and APIs, and do not submit forms or run models.
3. Edit `docs/brand-guide.json` and update its edition/date when appropriate.
4. Run `python build/generate-brand-guide.py --output tmp/pdfs/brand-guide-review.pdf`.
5. Render every page with Poppler, inspect the text and images, and verify that
   proposed work is not described as an implemented standard.
6. Once checked, run `python build/generate-brand-guide.py` to update the existing PDF.

The generator checks text bounds and writes layout measurements to
`tmp/pdfs/brand-guide-layout.json`. These checks supplement a visual review.
Keep fonts and logo geometry sharp, text selectable, and reference links usable.

## Scope and publishing

This guide describes the website source and reviewed examples. Verify production
separately before making deployment claims. Native Android screens require their
own build and visual checks even when they share the website's identity or feed.

`docs/` is not published by the website build. The PDF remains at its existing
path; updating it does not add a public navigation link, deploy the website, or
approve any of the proposed logo refinements. Document publishing follows the
existing referenced-document rules in `build/copy-to-public.js`.
