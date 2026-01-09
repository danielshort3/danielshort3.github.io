# UTM Batch Builder — Product Spec + Guide

## A) Product spec (UX flow + key decisions)

### Goal
Generate large, accurate batches of UTM URLs from a small set of structured inputs, with fast copy/export for ad platforms and trafficking.

### Primary users
Performance marketers and traffickers who:
- need consistent naming conventions,
- generate many links per campaign,
- want deterministic output they can re-run safely.

### Key decisions
- **Client-only**: no backend; no network calls with user URLs or CSV contents.
- **Deterministic output**: stable row ordering for the same inputs.
- **Performance**: generation runs in a **Web Worker**; results rendering uses **virtualized rows**.
- **Three combination modes**:
  - **A: Full Cartesian** — every base URL × every value combination.
  - **B: Zip by Row** — lists/CSV columns aligned by row index (strict length checks).
  - **C: Template + Rows** — “defaults + per-row overrides” (blank override cells fall back to defaults).
- **Normalization defaults**: `lowercase` + `spaces → underscores`.
- **Existing query params**: default behavior is **override** existing UTM/custom params (configurable).
- **Safety**: output is rendered as text (no HTML injection); CSV export escapes values.

### Out of scope (for v1)
- Server-side preset storage/auth.
- URL shortening, click tracking, or network validation of destination URLs.

## B) Wireframe-level UI sections

1) **Header**
   - Title, short description, privacy note (“runs locally”)

2) **Inputs**
   - Base URLs (multiline)
   - UTM fields: source, medium, campaign (+ optional content, term)
   - Custom parameters (add/remove rows)
   - Per-field input mode: Single / List / CSV Column (when CSV loaded)

3) **Campaign Name Builder**
   - Template input (e.g. `{initiative}_{market}_{channel}_{flight}`)
   - Auto-detected tokens, each with Single/List/CSV Column inputs
   - “Generate values” + “Use as `utm_campaign`”

4) **Combination Controls**
   - Mode A/B/C selector
   - Existing params handling (override vs preserve)
   - Exclude rules textarea (simple `key=value & key=value` per line)
   - Normalization toggles
   - Preview size + max rows warning

5) **CSV**
   - Upload CSV (read locally)
   - Shows detected columns + row count
   - Clear CSV

6) **Results**
   - Summary (rows generated, excluded, warnings)
   - Search filter
   - Copy all / Download CSV
   - Virtualized table (base_url, params…, final_url)

7) **Presets**
   - Save preset name
   - Load/delete presets (localStorage)

## C) Implementation plan (milestones)

1) Core library
   - URL builder with fragment/query handling + override rules
   - Normalization helpers (lowercase, separators, slugify)
   - Combination generators for A/B/C
   - CSV parser (quoted fields, BOM-safe)

2) Worker
   - Receive config, validate, generate rows in chunks, post progress

3) React UI
   - Input panels, validation, preview + generate
   - Presets storage + schema versioning
   - Virtualized results table + copy/export

4) Site integration
   - New tool page (`/tools/utm-batch-builder`)
   - Add tool card to Tools index
   - Add build step to bundle the React app
   - Add unit tests + contract checks

## E) How to use (examples)

### Example 1 — Full Cartesian
- Base URLs:
  - `https://example.com/landing-a`
  - `https://example.com/landing-b?ref=nav`
- `utm_source` (List):
  - `google`
  - `meta`
- `utm_medium` (Single): `cpc`
- `utm_campaign` (Single): `spring_sale`
- Mode: **Full Cartesian**

Result: `2 base URLs × 2 sources × 1 medium × 1 campaign = 4` URLs.

### Example 2 — Zip by Row (aligned lists)
- Base URL (Single): `https://example.com/landing`
- `utm_source` (List):
  - `google`
  - `meta`
- `utm_content` (List):
  - `banner_a`
  - `banner_b`
- Mode: **Zip by Row**

Result rows:
- row 1: google + banner_a
- row 2: meta + banner_b

If list lengths don’t match (and aren’t length 1), the tool shows an error.

### Example 3 — Template + Rows (defaults + overrides)
Use a default `utm_medium=cpc`, then override per row from CSV:

CSV columns:
- `base_url`
- `source`
- `campaign`
- `content` (optional)

Field mapping:
- Base URL: CSV Column `base_url`
- `utm_source`: CSV Column `source`
- `utm_campaign`: CSV Column `campaign`
- `utm_medium`: Single `cpc`
- `utm_content`: CSV Column `content` (blank cells fall back to default if provided)

### Exclude rules syntax
One exclusion per line; pairs must all match to exclude a row:
- `utm_medium=display & utm_source=google_search`
- `placement=feed & creative=video`

Keys match the final parameter keys (including custom params).

