# Ashglass terrain contact verification

The original Ashglass terrain had a real registration defect. Its flat source slabs begin at the first image row, but Canvas and Pixi applied the raised lip offset used by other biome art. The corrected Ashglass profile aligns flat terrain artwork directly with the collision surface. Collision geometry, player coordinates, and source images were not changed for this repair.

## Built-route measurement

The independent fixture loaded the actual built route at `http://127.0.0.1:4174/games/project-starfall`, without source-bundle substitution. It loaded Ashglass and the Cinder comparison, stopped the engine, placed the player at the selected platform, snapped the camera, and measured canvas geometry and runtime coordinates. All captures use a 1440×1000 viewport, internal canvas 1280×806, CSS canvas height 804.734375, and zoom 1.32. Runtime measurements are preserved in [ashglass-contact-measurements.json](contact-verification/ashglass-contact-measurements.json).

| Ashglass case | Collision world Y | Collision screen Y | Previous artwork offset | Corrected visible top, approximately |
| --- | ---: | ---: | ---: | ---: |
| Middle / ground | 1040 | 587.82 | 24 world px / 31.63 screen px upward | 588 |
| Upper / solid lane 01 | 880 | 538.46 | 12 world px / 15.82 screen px upward | 539 |

The screen positions apply to both renderers. Visible-edge readings are manual screenshot observations, rounded to whole pixels; runtime and draw-call positions are measured numbers. The corrected boots end just above the slab instead of inside it.

Individually inspected corrected originals, copied from `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/contact-final/`:

- [ashglassPass-middle-canvas.png](contact-verification/ashglassPass-middle-canvas.png)
- [ashglassPass-middle-pixi.png](contact-verification/ashglassPass-middle-pixi.png)
- [ashglassPass-upper-canvas.png](contact-verification/ashglassPass-upper-canvas.png)
- [ashglassPass-upper-pixi.png](contact-verification/ashglassPass-upper-pixi.png)

No new gap, buried feet, or material cap-end defect was visible in these scenes. This is a static standing-surface check, not a traversal or combat-animation test.

## Source pixels and regression

`img/project-starfall/environment/terrain/ashglass-pass.png` has 64×64 cells in eight columns. Ground repeating cells 1/2 and ledge repeating cells 5/6 begin with opaque artwork at source row zero in every column: 256 of 256 columns at alpha ≥64. The ledge cells require no further vertical source offset.

Left caps 0/4 and right caps 3/7 are tapered silhouettes, with inner 28-column plateaus at row zero and outer bevels reaching source row 60. Those bevels are not treated as flat support in the pixel assertion. The inspected actors stand on the continuous repeated plateau; these scenes do not establish an edge-of-cap traversal guarantee, and alpha geometry alone did not justify a further runtime change.

`node tests/project-starfall/project-starfall-terrain-contact.test.js` passes: 42 actual flat terrain dispatches agree between Canvas `drawImage` and Pixi texture placement, covering all Ashglass and Cinder flat platforms plus their generic flat dispatch. All eight Ashglass ground/ledge cells are exercised, and 18,392 drawn plateau columns meet collision within one world pixel. The test independently reads PNG alpha rather than reusing the asset importer's normalization logic. It also checks unchanged collision geometry and preserved Cinder lip offsets.

The initial [deep visual review](deep-review.md) missed this registration issue; its Ashglass acceptance has been explicitly superseded by this measured correction and the later corrected captures.
