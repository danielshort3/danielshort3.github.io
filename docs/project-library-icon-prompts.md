# Project Library icon generation prompts

Generated with the built-in `image_gen` tool on September 5, 2026. Each icon was generated individually and faithfully resized to a 256 × 256 PNG with Sharp. No image-generation CLI was used.

These homepage and library icons follow the Tools Library navy, signal-blue, white, and small copper-accent palette. The artwork has a white exterior and uses project-icon-only CSS multiply blending on the pale tiles. The generated images did not provide actual alpha transparency; the selected outputs contain no painted checkerboard.

All 16 published projects now have `iconImage` artwork in the full library. The homepage's four featured projects use those same icon files. Canonical project screenshots and demo previews remain in their existing `image` fields, including the Sheet Music comparison and Delivery Tip map.

The final revision adds a map-and-tip-coin icon for Delivery Tip, a music-page-and-eraser icon for Sheet Music Watermark Removal & Upscale, and replaces Baby Name Predictor's abstract name cards with a swaddled baby and a favorite name tag.

| Project | Saved asset |
| --- | --- |
| Smart Sentence Retriever | [smartSentence.png](../img/projects/icons/smartSentence.png) |
| Chatbot (LoRA + RAG) | [chatbotLora.png](../img/projects/icons/chatbotLora.png) |
| Shape Classifier Demo | [shapeClassifier.png](../img/projects/icons/shapeClassifier.png) |
| UFO Sightings Dashboard | [ufoDashboard.png](../img/projects/icons/ufoDashboard.png) |
| COVID-19 Outbreak Drivers | [covidAnalysis.png](../img/projects/icons/covidAnalysis.png) |
| Empty-Package Shrink Dashboard | [targetEmptyPackage.png](../img/projects/icons/targetEmptyPackage.png) |
| Handwriting Legibility Scoring | [handwritingRating.png](../img/projects/icons/handwritingRating.png) |
| Synthetic Digit Generator | [digitGenerator.png](../img/projects/icons/digitGenerator.png) |
| Sheet Music Watermark Removal & Upscale | [sheetMusicUpscale.png](../img/projects/icons/sheetMusicUpscale.png) |
| Delivery Tip | [deliveryTip.png](../img/projects/icons/deliveryTip.png) |
| Store-Level Loss & Sales ETL | [retailStore.png](../img/projects/icons/retailStore.png) |
| Pizza Tips Regression Modeling | [pizza.png](../img/projects/icons/pizza.png) |
| Baby Name Predictor | [babynames.png](../img/projects/icons/babynames.png) |
| Pizza Delivery Dashboard | [pizzaDashboard.png](../img/projects/icons/pizzaDashboard.png) |
| Nonogram Solver | [nonogram.png](../img/projects/icons/nonogram.png) |
| danielshort.me | [website.png](../img/projects/icons/website.png) |

## Final prompt set

For entries with a background correction, the creation prompt and final edit prompt together describe the selected asset. All background corrections used the built-in image editing tool.

### Smart Sentence Retriever

Asset: `img/projects/icons/smartSentence.png`

```text
Use case: logo-brand. Create one final Project Library icon for Smart Sentence Retriever, a website list thumbnail. Subject: a magnifying glass overlapping a white document, with exactly one blue horizontal sentence stripe highlighted beneath its lens; a few understated navy line strokes suggest other document lines; one very small copper accent at the lens hinge. Style: clean vector-like polished professional illustration, subtly rounded geometric forms, crisp thin navy edges, gentle dimensional shading and very restrained shadow. Composition: square 1:1 canvas, genuinely transparent alpha background (no painted checkerboard), centered compact single mark occupying 65-70% of canvas with 15-18% clear margin; legible at 58 pixels. Palette: midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA, only one small copper #D97706 accent. No words, letters, numbers, labels, watermarks, mascot, robot, brain, screenshots, dense dashboard, badge or tile backdrop, no photorealism. Render actual transparent PNG with alpha and a single cohesive silhouette.
```

Final background edit:

```text
Use case: background-extraction. Edit this icon only by removing its gray-and-white checkerboard background. The checkerboard is mistakenly painted into the image and must be completely absent. Return a true RGBA PNG with alpha=0 pixels outside the document-and-magnifier silhouette. Keep the icon colors, shape, geometry and materials exactly unchanged. Do not show or paint a transparency preview; no checkerboard anywhere. Transparent pixels outside the actual icon. If this image tool cannot actually return RGBA transparency, use a completely flat uniform white #FFFFFF background with absolutely no gray squares, no vignette, no ambient shading on the background, no shadow outside the icon. Square canvas.
```

### Chatbot (LoRA + RAG)

Asset: `img/projects/icons/chatbotLora.png`

```text
Use case: logo-brand. Create one final Project Library icon for Chatbot LoRA + RAG, a website list thumbnail. Subject: one large signal-blue chat bubble in front, connected by one simple navy elbow line to two overlapping white source document cards behind. Put two clean horizontal navy strokes on source cards and white two short rounded strokes within chat bubble. One very small copper linked-ring cue at the connection conveys citations. Style: clean vector-like polished professional illustration, subtly rounded geometric forms, crisp thin navy edges, gentle dimensional shading only within the artwork, no external shadow. Composition: square 1:1 canvas, perfectly flat uniform white #FFFFFF background edge to edge; centered compact single cohesive mark occupying 65-70% of canvas with 15-18% clear margin; legible at 58 pixels. Palette: midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA, only one small copper #D97706 accent. No words, letters, numbers, labels, watermarks, mascot, robot, brain, screenshots, dense dashboards, badge or tile backdrop, no photorealism, no checkerboard, no texture, no vignette, no background gradient.
```

### Shape Classifier Demo

Asset: `img/projects/icons/shapeClassifier.png`

```text
Use case: logo-brand. Create one final Project Library icon for Shape Classifier Demo, a website list thumbnail. Subject: a simple circle, triangle and square within four short navy recognition-frame corner brackets. The circle is upper left in mist-white, the square is upper right in deep blue, and one slightly larger signal-blue triangle sits lower center. One tiny copper glint or dot at a corner indicates recognition. Balanced geometric composition that reads as classifying shapes, no complex diagram. Style: clean vector-like polished professional illustration, subtly rounded geometric forms, crisp thin navy edges, gentle dimensional shading only within the artwork, no external shadow. Composition: square 1:1 canvas, perfectly flat uniform white #FFFFFF background edge to edge; centered compact single cohesive mark occupying 65-70% of canvas with 15-18% clear margin; legible at 58 pixels. Palette: midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA, only one small copper #D97706 accent. No words, letters, numbers, labels, watermarks, mascot, robot, brain, screenshots, dense dashboards, badge or tile backdrop, no photorealism, no checkerboard, no texture, no vignette, no background gradient.
```

### UFO Sightings Dashboard

Asset: `img/projects/icons/ufoDashboard.png`

```text
Use case: logo-brand. Create one final Project Library icon for UFO Sightings Dashboard, a website list thumbnail. Subject: a simple midnight-navy flying-saucer silhouette floating directly above a compact folded white-and-blue map. Saucer has one gentle blue dome and a broad clean oval rim; no alien, no space scene. Map consists of three folded panels with only two broad blue road paths and one small copper location dot. UFO dominates, map is supporting cue; cohesive centered silhouette. Style: clean vector-like polished professional illustration, subtly rounded geometric forms, crisp thin navy edges, gentle dimensional shading only within the artwork, no external shadow. Composition: square 1:1 canvas, perfectly flat uniform white #FFFFFF background edge to edge; centered compact single cohesive mark occupying 65-70% of canvas with 15-18% clear margin; legible at 58 pixels. Palette: midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA, only one small copper #D97706 accent. No words, letters, numbers, labels, watermarks, mascot, robot, brain, screenshots, dense dashboards, badge or tile backdrop, no photorealism, no checkerboard, no texture, no vignette, no background gradient, no beam, no stars.
```

### COVID-19 Outbreak Drivers

Asset: `img/projects/icons/covidAnalysis.png`

```text
Use case: logo-brand. Create one final Project Library icon for COVID-19 Outbreak Drivers hospital-utilization analysis, a website list thumbnail. Subject: one compact recognizable hospital bed in side view with signal-blue blanket, white pillow, midnight-navy headboard and simple legs. Immediately above it, a short clean rising blue utilization line with only three segments; one small copper dot at a threshold on the rising line. No patient, no human silhouette, no virus, no medical cross, no needles. The bed is the dominant symbol and chart line the only supporting cue. Style: clean vector-like polished professional illustration, subtly rounded geometric forms, crisp thin navy edges, gentle dimensional shading only within the artwork, no external shadow. Composition: square 1:1 canvas, perfectly flat uniform white #FFFFFF background edge to edge; centered compact single cohesive mark occupying 65-70% of canvas with 15-18% clear margin; legible at 58 pixels. Palette: midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA, only one small copper #D97706 accent. No words, letters, numbers, labels, watermarks, mascot, robot, brain, screenshots, dense dashboards, badge or tile backdrop, no photorealism, no checkerboard, no texture, no vignette, no background gradient.
```

### Empty-Package Shrink Dashboard

Asset: `img/projects/icons/targetEmptyPackage.png`

```text
Use case: logo-brand.
Asset type: Project Library icon on a professional light website, must remain identifiable at 58 pixels.
Primary request: A clearly empty open carton, with navy and white sides and four open flaps revealing its empty interior. One small copper descending inventory arrow near the lower right is the only supporting cue.
Scene/backdrop: Genuine transparent alpha background. Do not paint checkerboard, solid white backdrop, or ground plane.
Style/medium: Clean vector-like illustration, subtly rounded geometric forms, crisp thin navy outlines, gentle dimensional shading, exceptionally restrained soft shadow. Polished and professional, not cartoon.
Composition/framing: 1:1 square. A centered compact mark occupying 65-70% of the canvas, generous 15-18% clear padding. One dominant carton plus one small inventory cue. Easy silhouette.
Color palette: Midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA. Only one small copper #D97706 accent.
Constraints: No words, labels, letters, numerals, logos, watermark, mascot, robot, brain, faux screenshot, dense dashboard, badge or tile backdrop. No photorealism. Deliver an actual transparent PNG.
```

Final background edit:

```text
Use case: precise-object-edit.
Edit only the exterior background of this icon. Replace the entire painted checkerboard outside the open carton and copper downward arrow with perfectly solid white #FFFFFF. No checkerboard, gray exterior, gradients, texture, or shadows outside the artwork. Keep the existing carton silhouette, colors, empty interior, navy outlines, and copper descending arrow exactly the same. Keep the square framing and clear padding.
```

### Handwriting Legibility Scoring

Asset: `img/projects/icons/handwritingRating.png`

```text
Use case: logo-brand.
Asset type: Project Library icon on a professional light website, identifiable at 58 pixels.
Primary request: A handwritten blue digit "7" on one white rounded paper tile, beside a compact semicircular blue scoring gauge with a tiny copper needle. The handwritten numeral is the dominant subject; the score gauge is a simple supporting cue with no numbers, tick labels, or words.
Scene/backdrop: Perfectly solid flat white #FFFFFF exterior. No checkerboard, no gray background, no ground plane. No shadows outside artwork.
Style/medium: Clean vector-like illustration, subtly rounded geometric forms, crisp thin midnight navy edges, gentle dimensional shading only within forms. Polished and professional, not cartoon.
Composition/framing: 1:1 square. Centered compact mark occupying 65-70% of canvas, generous 15-18% clear padding. Digit tile slightly left, gauge slightly lower right, one cohesive mark.
Color palette: Midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA. Only one small copper #D97706 accent in gauge needle.
Constraints: The single handwritten numeral "7" is allowed as the subject. No other text, words, letters, logos, watermark, mascot, robot, brain, faux screenshot, dense dashboard, badge or encompassing tile backdrop. No photorealism.
```

### Synthetic Digit Generator

Asset: `img/projects/icons/digitGenerator.png`

```text
Use case: logo-brand.
Asset type: Project Library icon on a professional light website, identifiable at 58 pixels.
Primary request: One white rounded paper tile with a navy handwritten digit "3" branching through two short clean blue connector arrows into two smaller white paper tiles carrying visibly different handwritten samples of the same digit "3". Top output sample has rounded loops, bottom output sample is more angular. Clearly communicate synthetic variation of one handwritten digit. Tiny copper accent on just one small connector node.
Scene/backdrop: Perfectly solid flat white #FFFFFF exterior. No checkerboard, gray background, ground plane, or exterior shadows.
Style/medium: Clean vector-like illustration, subtly rounded geometric forms, crisp thin midnight navy edges, gentle dimensional shading only within forms. Polished and professional, not cartoon.
Composition/framing: 1:1 square. Centered compact mark occupying 65-70% of canvas, generous 15-18% clear padding. One larger input tile on left, two smaller output tiles vertically arranged on right; minimal connectors. Keep the 3 digits large and the silhouette clean.
Color palette: Midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA. Only one small copper #D97706 accent.
Constraints: Three handwritten "3" numerals as the essential subject; no other text, labels, words, letters, logos, watermark, mascot, robot, brain, faux screenshot, dense dashboard, badge or encompassing tile backdrop. No photorealism.
```

### Sheet Music Watermark Removal & Upscale

Asset: `img/projects/icons/sheetMusicUpscale.png`

```text
Use case: logo-brand.
Asset type: Project Library icon on a professional light website, identifiable at 58 pixels.
Input image: Style reference only, showing the existing navy/blue/white rounded artwork family. Do not reproduce the digit or scoring gauge.
Primary request: A crisp white music page with five evenly spaced dark navy staff lines and three distinct navy music notes. A simple blue eraser with one small copper end is rubbing away part of a faint diagonal gray watermark stripe across the lower area of the page. The score remains clearly visible and legible as music. Show the stripe only across a short portion and a clean erased gap beside the eraser, subtly communicating restoration of sheet music. No readable watermark text.
Scene/backdrop: Perfectly solid flat white #FFFFFF exterior. No checkerboard, gray background, ground plane, or shadows outside artwork.
Style/medium: Clean vector-like illustration, subtly rounded geometric forms, crisp thin midnight navy edges, gentle dimensional shading within forms only. Polished and professional, not cartoon. Match the restrained finish of the reference.
Composition/framing: 1:1 square. One centered compact icon occupying 65-70% of canvas with generous clear padding. The page is the dominant form, slightly angled with a subtle turned upper corner. The eraser is a smaller supporting shape overlapping the lower right corner. Staff lines and notes occupy a clear main area with bold readable shapes; avoid dense notation.
Color palette: Midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA. Only one small copper #D97706 accent on the eraser end.
Constraints: No words, letters, numerals, labels, logos, fake readable watermark, music player, headphones, mascot, robot, brain, faux screenshot, dashboard, badge or encompassing tile backdrop. No photorealism.
```

### Delivery Tip

Asset: `img/projects/icons/deliveryTip.png`

```text
Use case: logo-brand. Create one NEW final Project Library icon for the Delivery Tip geo-analytics project, which compares personal delivery tip earnings by neighborhood and shift. Input image is a STYLE REFERENCE ONLY: use its crisp navy edges, saturated blue, softly shaded white geometric forms, restrained polished dimensions and white exterior. Do not include its flying saucer or UFO. New subject: a compact folded neighborhood map with exactly three panels, a few bold simple blue street lines, one dominant signal-blue location pin, and one small copper coin with a single embossed dollar symbol as the tip-earnings cue. The map and pin form one cohesive clear silhouette, with the coin tucked at the lower right as supporting cue. The meaning should be monetary tips by location. No pizza, no delivery truck, no receipt, no arrows or graph, no numbers, no labels or words, no extra symbols other than the one dollar symbol on the coin. Professional clean vector-like illustration with slightly rounded geometric forms, thin crisp midnight-navy edges, gentle dimensional shading inside the forms only. Palette midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA; copper #D97706 only on the one small coin. Square 1:1 canvas, centered compact subject occupying roughly 70% canvas with 15% clear margin. Legible at 58px. Perfectly flat uniform white #FFFFFF background, no checkerboard, no gray squares, no gradient background, no shadow outside the artwork, no tile or badge backdrop, no photorealism, no watermark. Match reference illustration family and edge weight while using only the new map/pin/coin subject.
```

### Store-Level Loss & Sales ETL

Asset: `img/projects/icons/retailStore.png`

```text
Use case: logo-brand.
Asset type: Project Library icon on a professional light website, identifiable at 58 pixels.
Primary request: Three very short clean blue data-stream connector lines converge into one polished white and navy storefront with a blue awning. One tiny copper anomaly dot sits on one of the three incoming streams. Emphasize several sources of store data combined into one storefront, not empty packaging. Keep the storefront as the dominant subject with a few large simple shapes: striped blue/white awning, navy outlined walls, one door and one window. Incoming streams can be stepped horizontal strokes meeting into one connector, but no database cylinders or extra objects.
Scene/backdrop: Perfectly solid flat white #FFFFFF exterior. No checkerboard, gray background, ground plane, or shadows outside artwork.
Style/medium: Clean vector-like illustration, subtly rounded geometric forms, crisp thin midnight navy edges, gentle dimensional shading only within forms. Polished and professional, not cartoon.
Composition/framing: 1:1 square. Centered compact mark occupying 65-70% of canvas, generous 15-18% clear padding. Short incoming data strokes on left converge into dominant storefront on right.
Color palette: Midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA. Only one small copper #D97706 accent on one incoming stream.
Constraints: No words, labels, letters, numerals, logos, watermark, mascot, robot, brain, faux screenshot, dense dashboard, badge or tile backdrop. No photorealism.
```

### Pizza Tips Regression Modeling

Asset: `img/projects/icons/pizza.png`

```text
Use case: logo-brand.
Asset type: Project Library icon on a professional light website, identifiable at 58 pixels.
Primary request: A compact, simple blue scatterplot with a straight fitted regression line rising gently left to right. Only six blue data points loosely surround the line, with generous space. A small recognizable navy and blue pizza slice overlaps the lower right of the plot, with a tiny copper accent at its pointed tip. Clearly communicate pizza tips regression modeling. The scatterplot is dominant and the pizza slice is a small supporting anchor.
Scene/backdrop: Perfectly solid flat white #FFFFFF exterior. No checkerboard, gray background, ground plane, or shadows outside artwork.
Style/medium: Clean vector-like illustration, subtly rounded geometric forms, crisp thin midnight navy edges, gentle dimensional shading only within forms. Polished and professional, not cartoon.
Composition/framing: 1:1 square. Centered compact mark occupying 65-70% of canvas, generous 15-18% clear padding. Open navy L-shaped axes, six blue points, one blue fitted line. Absolutely no full rectangular chart card, no dashboard panels, no map.
Color palette: Midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA. Only one small copper #D97706 accent at the pizza slice tip.
Constraints: No words, labels, letters, numerals, logos, watermark, mascot, robot, brain, faux screenshot, dense dashboard, badge or tile backdrop. No photorealism.
```

### Baby Name Predictor

Asset: `img/projects/icons/babynames.png`

```text
Use case: precise-object-edit.
Asset type: final Baby Name Predictor icon for a professional Project Library. This project learns parents' preferences and recommends baby names.
Input image: existing baby-name icon is the edit target and style reference. Its two abstract cards do not make the baby-naming subject clear.
Primary request: replace the abstract letter-like cards with one clearly recognizable swaddled newborn, alongside one small blank name-tag card carrying a blue heart as a favorite-selection cue. The baby is the dominant subject, shown as a simple round white/mist face with a small navy curl, two gentle closed-eye strokes, wrapped in a signal-blue swaddle with clean navy edges. The name tag is a small supporting cue at the lower right with a tiny punched hole and a copper connector dot, and a blue heart. Keep the illustration restrained and icon-like, not a cartoon character portrait. No letters or name text, no tiny lists. Communicate baby plus a preferred name.
Style: match the input's thin midnight-navy outlines, gently rounded vector-like geometry, understated internal shading, clean polished professional finish. Use navy #091F3B, blue #005FED, deep blue #0145C8, mist #EEF2F7, white #F9F9FA, one small copper #D97706 accent.
Composition: square 1:1 canvas, centered compact mark occupying about 65-70% of the canvas, generous 15-18% padding. Legible as a single cohesive icon at 58px. Pure uniform white #FFFFFF exterior with no checkerboard, texture, gradient background, or outside shadows.
Constraints: no words, names, letters, numbers, ABC blocks, speech bubble, robot, brain, generic AI sparkle, pacifier, gender symbol, extra body parts, photorealism, large badge backdrop, busy details. Preserve the visual family's navy/blue/white palette while replacing the subject as requested.
```

### Pizza Delivery Dashboard

Asset: `img/projects/icons/pizzaDashboard.png`

```text
Use case: logo-brand. Create one production website project-library icon as a square 1:1 PNG with a perfectly uniform solid white #FFFFFF background. It must belong to a consistent professional navy/blue/white illustration family: clean vector-like shapes with subtly rounded geometry, crisp thin midnight-navy edges, very light dimensional shading, restrained shadow. Palette midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, canvas white #F9F9FA, one tiny copper #D97706 accent. Center the symbol occupying about 65-70% of the canvas with 15-18% padding. One dominant subject and at most one simple supporting cue. Legible at 58px. No framing tile or badge backdrop; background outside artwork is perfectly solid white #FFFFFF with NO painted checkerboard, no textured background and no gradient background. No words or labels, no miniature UI, no screenshots, no mascot, no cartoon, no photorealism, no brains, no generic AI sparkles, no clutter.
Subject: a small folded white/mist map outlined in navy, a large signal-blue location pin whose white center contains a stylized navy-and-copper pizza slice, and a single clean short blue delivery route connecting one small copper destination dot. Clear map plus pizza-location silhouette. No charts, clocks, cars, words, labels or numbers.
```

### Nonogram Solver

Asset: `img/projects/icons/nonogram.png`

```text
Use case: logo-brand. Create one production website project-library icon as a square 1:1 PNG with a perfectly uniform solid white #FFFFFF background. It must belong to a consistent professional navy/blue/white illustration family: clean vector-like shapes with subtly rounded geometry, crisp thin midnight-navy edges, very light dimensional shading, restrained shadow. Palette midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, canvas white #F9F9FA, one tiny copper #D97706 accent. Center the symbol occupying about 65-70% of the canvas with 15-18% padding. One dominant subject and at most one simple supporting cue. Legible at 58px. No framing tile or badge backdrop; background outside artwork is perfectly solid white #FFFFFF with NO painted checkerboard, no textured background and no gradient background. No words or labels, no miniature UI, no screenshots, no mascot, no cartoon, no photorealism, no brains, no generic AI sparkles, no clutter.
Subject: one compact square nonogram puzzle grid, exactly 5 by 5 cells, front-facing with very slight depth. Navy outer edge, pale gray/blue thin internal grid, selected signal-blue cells forming a simple stepped heart-like pattern; one unfilled next cell has a copper outline. The grid is the entire subject. No pencil, numbers, letters, checkmark, flag, bomb, or separate badges.
```

### danielshort.me

Asset: `img/projects/icons/website.png`

```text
Use case: logo-brand. Create one production website project-library icon as a square 1:1 PNG with a perfectly uniform solid white #FFFFFF background. It must belong to a consistent professional navy/blue/white illustration family: clean vector-like shapes with subtly rounded geometry, crisp thin midnight-navy edges, very light dimensional shading, restrained shadow. Palette midnight navy #091F3B, signal blue #005FED, deep blue #0145C8, mist #EEF2F7, canvas white #F9F9FA, one tiny copper #D97706 accent. Center the symbol occupying about 65-70% of the canvas with 15-18% padding. One dominant subject and at most one simple supporting cue. Legible at 58px. No framing tile or badge backdrop; background outside artwork is perfectly solid white #FFFFFF with NO painted checkerboard, no textured background and no gradient background. No words or labels, no miniature UI, no screenshots, no mascot, no cartoon, no photorealism, no brains, no generic AI sparkles, no clutter.
Subject: a simple white browser window with thin navy rounded frame, slim top bar and three tiny dots (one copper), containing the existing Daniel Short DS monogram from the provided image. EDIT TARGET: retain the recognizable overlapping navy D and signal-blue S, including ascending blue bars inside the D, in the central content area of the browser. Remove the existing gray background and large diffuse glow; use clean edges and perfectly solid white #FFFFFF outside the browser. Keep the monogram visually faithful to the supplied logo; no new text, no extra UI panels. The browser frame and existing DS logo are the complete icon.
```

Reference image: `img/ui/logo.png` (existing Daniel Short monogram).
