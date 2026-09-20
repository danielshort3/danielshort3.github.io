# Glowcap candidate source review

This review reads the original generated PNG and its measured rectangles. It does not modify pixels, rescale poses, or change game assets.

## Extraction and source integrity

- The source is 1254 x 1254 RGBA, not the requested uniform 1024-square sheet. Its nominal 4 x 4 cells would be 313.5 pixels wide/high.
- Uniform row boundaries cross pose pixels: frames 0 through 8 touch extraction edges. Some second-row extracts inherit first-row feet. The uniform extraction therefore cannot represent the authored poses faithfully.
- The metadata's measured row boundaries `[0, 354, 664, 969, 1254]` and suggested source rectangles preserve all source coverage, with no overlaps or omitted pixels. At alpha >= 16, all 16 measured rectangles avoid their edges and four-pixel gutters. No main-body clipping is detected with these rectangles.
- Main alpha silhouettes are 202-214 pixels wide and 145-213 pixels tall. The change in height corresponds to different squash/stretch/airborne drawings; do not normalize each frame's size.
- All 16 visible cells differ. Distinct pixel hashes establish different cells, not coherent animation or consistent character anatomy.
- Detached background patches are predominantly alpha 1/255: 100-2918 detached pixels per measured cell at the alpha > 0 threshold. They can inflate a raw nonzero-alpha bounding box dramatically. At alpha >= 16, the only detached pixel is one isolated pixel in frame 14. This faint background residue is a cleanup consideration, not evidence that the primary silhouettes are clipped.
- The candidate is an airborne hop study. The supplied GIF instead shows planted squash/stretch with cap wobble. Correcting extraction does not resolve that motion mismatch.

Detailed reports: `glowcap-hop-measured-check.json` and `glowcap-hop-uniform-check.json`.

## Root strategy for an eight-pose grounded study

1. Establish a single anatomical root on the ground between the same two supporting front leaf/foot pads in the neutral pose. Retain the original left/right foot identities across all drawings.
2. Author every planted drawing against the same source-space root crosshair and ground line. The cap, head, stalk, arms and leaf shapes may squash, lean and lag above that root. Require true drawn deformation; adding world-space bobbing to a still does not satisfy this study.
3. Record semantic contacts per frame: `leftFootContact`, `rightFootContact`, and whether each foot is planted. Select each contact at the sole's ground intersection, supported by the actual outline at substantial alpha (for example >= 128), not by a cap silhouette, shadow, particle, or faint alpha residue.
4. When both feet are planted, use the midpoint of those two anatomical contacts as the diagnostic root X and their common sole baseline as root Y. Do not use the center of the whole alpha bounding box or its alpha centroid: cap wobble must not move the world root. Changing foot width should not move the chosen contact point within the foot.
5. If a pose intentionally raises one foot, retain the planted foot's known world contact and the authored root. Do not average the lifted foot with the planted foot. A fully airborne frame belongs to a different action and should not be silently grounded.
6. For presentation, use one common scale derived from the neutral pose and one common stage origin. With measured source rectangles, account only for each rectangle's declared coordinate origin. Do not fit, independently scale, warp, or recenter every frame to make it appear consistent.
7. Treat incorrect contact placement as an authoring defect to expose and redraw, rather than hide it with per-frame motion offsets. A diagnostic contact overlay and frame-by-frame stepping make the discrepancy reviewable. Allow only the agreed pixel rounding tolerance at final native scale.
8. Review the last-to-first transition at the same root as every interior transition. Confirm that the cap spots, two glowing cap nubs, leaf attachment and left/right limb identities survive the entire loop.

Root-contact metadata describes a visual check; it does not create motion. The eight drawings themselves must supply the complete planted spring/wobble cycle.

## Checker usage

Uniform grid, default 4 x 4:

`node output/starfall-animation-samples/check-hop.cjs input.png report.json`

Measured rectangles from `frames[].suggestedSourceRect`, `frames[].sourceRect`, `frames[].rect`, or `sourceRects[]`:

`node output/starfall-animation-samples/check-hop.cjs input.png report.json layout.json`

A layout containing only `nominalGrid: { columns: 4, rows: 2 }` selects a uniform eight-frame grid. The checker never writes to the input image or layout file.
