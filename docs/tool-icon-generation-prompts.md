# Tool Icon Generation Prompts

Use these prompts to generate branded icon assets for the tools directory. The current tools UI supports generated image icons through an optional `iconImage` field on each `content/tools/*.json` record, while the existing `iconHtml` SVG remains the fallback when no image asset is set.

Recommended output:
- `1024x1024` PNG
- Transparent background
- Main mark centered with about `18-22%` safe padding
- Slug-matched filenames under `img/tools/icons/`, for example `img/tools/icons/text-compare.png`

## Shared Style Prefix

Paste this before each tool-specific prompt:

```text
Create a premium professional website tool icon for the Daniel Short analytics brand. 1:1 square composition, transparent background, clean vector-like illustration with subtle dimensional depth, rounded geometric forms, crisp edges, restrained shadows, and generous padding. Use midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, canvas white #F9F9FA, and one small copper accent #D97706. The icon must remain legible at 32px and 58px. No words, no letters, no numbers, no screenshots, no busy UI, no generic stock icon style.
```

## Negative Prompt

```text
No readable text, no fake UI screenshots, no tiny labels, no letters, no numbers, no mascots, no cartoon style, no purple gradients, no beige or brown palette, no cluttered dashboard, no photo-realism, no heavy shadows, no black background, no low-contrast white-on-white shapes.
```

## Tool-Specific Prompts

### Text Compare

```text
Two clean document panels side by side with aligned line marks, one blue insertion line and one copper deletion line, plus a subtle comparison connector between the panels.
```

Suggested asset path: `img/tools/icons/text-compare.png`

### Non-breaking Space Cleaner

```text
A document with paragraph lines and two spacing blocks being cleaned into one normal gap, with a small copper sweep mark to suggest cleanup.
```

Suggested asset path: `img/tools/icons/nbsp-cleaner.png`

### Oxford Comma Checker

```text
Three abstract list items in a row with a precise copper comma/check mark before the final item, presented as a grammar review tool without readable text.
```

Suggested asset path: `img/tools/icons/oxford-comma-checker.png`

### Point of View Checker

```text
A central eye or lens over three subtle perspective markers, suggesting first-person, second-person, and third-person detection without using numbers.
```

Suggested asset path: `img/tools/icons/point-of-view-checker.png`

### Word Frequency Analyzer

```text
A document transforming into ranked horizontal bars and small keyword dots, showing frequency analysis without readable words.
```

Suggested asset path: `img/tools/icons/word-frequency.png`

### UTM Batch Builder

```text
A spreadsheet-like grid feeding into linked campaign nodes, with a copper export arrow and clean URL/tag symbolism.
```

Suggested asset path: `img/tools/icons/utm-batch-builder.png`

### QR Code Generator

```text
A stylized QR pattern with three finder corners and a small centered brand-blue mark, abstract enough to not be scannable.
```

Suggested asset path: `img/tools/icons/qr-code-generator.png`

### Image Optimizer

```text
A photo frame compressed by inward arrows into a smaller sharp image tile, with a small quality gauge or sparkle.
```

Suggested asset path: `img/tools/icons/image-optimizer.png`

### Background Remover

```text
A product silhouette lifted from a checkerboard transparency field, with a clean selection outline and copper refinement brush.
```

Suggested asset path: `img/tools/icons/background-remover.png`

### Screen Recorder

```text
A monitor frame with a record dot, capture ring, and small audio waveform, simple enough for a small toolbar-style icon.
```

Suggested asset path: `img/tools/icons/screen-recorder.png`

### Job Application Tracker

```text
A briefcase and pipeline card with check marks, KPI bars, and calendar dots, showing progress tracking without dense dashboard detail.
```

Suggested asset path: `img/tools/icons/job-application-tracker.png`

### Short Links

```text
A compact chain link turning into a redirect arrow across two route nodes, with a small admin shield/key accent.
```

Suggested asset path: `img/tools/icons/short-links.png`

### GA4 UTM Performance

```text
Campaign link nodes flowing into a clean analytics bar chart, with a small admin shield and copper insight marker.
```

Suggested asset path: `img/tools/icons/ga4-utm-performance.png`

### File Transcriber

```text
An audio waveform and video/file tile converting into a transcript sheet, with a small cloud/check accent for AWS processing.
```

Suggested asset path: `img/tools/icons/transcribe.png`
