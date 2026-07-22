# Chrome Web Store release kit

This directory contains the human-reviewed materials for an **unlisted** Chrome Web Store release of Job Application Copilot. It is deliberately excluded from the extension runtime package.

## Contents

- `listing.md` — paste-ready Chrome Web Store listing fields.
- `privacy-dashboard-disclosures.md` — single-purpose, permission, remote-code, and data-use declarations.
- `reviewer-instructions.md` — setup and test steps for Chrome Web Store review.
- `release-checklist.md` — item creation, stable-ID, validation, and publishing gates.
- `assets/icon-source.svg` — deterministic source for the 128 px store icon.
- `assets/small-promo-source.svg` — deterministic source for the 440 x 280 px promotional tile.
- `assets/icon-128.png` and `assets/small-promo-440x280.png` — generated upload assets.
- `assets/screenshots/01-reviewed-answers.png` — a 1280 x 800 browser capture of the actual built panel using synthetic data.
- `preview/index.html` — a 1280 x 800 synthetic ATS frame that embeds the extension's real `preview=1` side-panel build for screenshot capture.

Generate the PNG assets from the extension directory:

```text
node scripts/generate-store-assets.mjs
```

The generator resolves the repository's existing `sharp` installation and adds no extension dependency. Do not hand-edit generated PNGs.

The 128 px icon keeps its square artwork inside a 96 x 96 px visual area with 16 px of transparent padding on every side, following Chrome Web Store icon guidance. test/store-assets-requirements.test.js protects the PNG dimensions and this safe-area transform.

The preview page does not fabricate a screenshot. Build the extension first, serve the repository over HTTP, open `store/preview/index.html`, and capture the rendered 1280 x 800 page in a real browser. The right-hand panel is the actual built side-panel UI populated by its synthetic, non-production preview mode.

