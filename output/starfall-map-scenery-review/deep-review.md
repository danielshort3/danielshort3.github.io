# Deep renderer and portrait review

Individually inspected **60 original-resolution screenshots** of ten representative maps: 20 desktop Canvas views at 1440×1000, plus the corresponding 40 portrait/reduced-effects views at 390×844 in Canvas and Pixi. This extends the full-size map review and the separately reviewed 32 connector captures.

Desktop directory: `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/verified/`.

Portrait/reduced-effects directory: `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/verified-mobile-reduced/`.

## Findings

The initial screenshot pass identified no missing scenery or newly obstructed standing lane, but it missed an Ashglass surface-registration defect subsequently raised by the other reviewer. Its earlier conclusion that there was no blocking surface-art defect is superseded by the measured follow-up below. Foreground materials remain recognizable in each biome: grassy stone in Crossing/Verge, dark roots in Thornpath, metal-trimmed masonry in Rustcoil, burnt orange volcanic rock in Cinder, dark glass/basalt in Ashglass, snow-capped stone in Glacier, pale weathered rock in Stormbreak and the distinct celestial/void masonry in Astral/Eclipse. Cinder's distant illustration does not recreate the removed sharp foreground ledges. Ashglass's cap is continuous without the former pale steps; continuity alone did not prove that it met the collision plane.

The foreground geometry and regional material placement agree across the portrait Canvas/Pixi pairs. Enemy rosters, idle frames and portal frames vary because capture instances are separate; these files are not a pixel-diff equivalence test. Crossing's two portrait arrival captures also show different starting camera/player positions: Canvas begins farther east around Frontier Gate, while Pixi begins by the western workshop/storage area. Those arrival images were both inspected, but should not be described as a matched camera comparison. The Crossing upper pair does match.

Two remaining presentation concerns were reported to the implementation lead:

- **Crossing service labels:** `starfallCrossing-arrival-canvas.png` shows Storage and Shopkeeper as dark unboxed text against a textured platform/background. `starfallCrossing-upper-canvas.png` similarly shows Upgrade. These have weaker contrast than the corrected white-on-dark NPC nameplates. The sprites and service areas remain visible; this is a text-readability issue.
- **Portrait scale:** the canvas fits into roughly 374×234 CSS pixels at this viewport. Platform silhouettes and large enemies remain recognizable, but the player, world labels, minimap legend and canvas HUD text are very small. The separate touch movement/attack controls remain visible and do not cover the game canvas. This is an inherited portrait-layout limitation, so these screenshots support surface/context integrity rather than a claim of fully readable mobile gameplay.

Thornpath's dark root lanes have less background contrast than the brighter regions, especially at portrait size. Their continuous moss edge and actor outlines still identify the occupied lane. No further asset edit was made from this observation. The separately recorded Astral Stacks decorative-ladder concern is outside these ten maps; Astral Archive's foreground lanes remain clear in these views.

These are stopped-scene checks. Reduced-effects screenshots do not verify warning persistence during an actual attack, and screenshots cannot demonstrate smooth traversal, camera motion or touch-control interactions. No production files were edited for this review.

## Ashglass contact correction: superseding verification

Independent measurements of the final built route confirmed that the previous Ashglass ground artwork began 24 world pixels above collision, and field ledges began 12 pixels above collision. At the captured zoom and CSS scale these became approximately 31.63 and 15.82 screen pixels. The source art's repeated slabs begin at source row zero, so the generic terrain lip offset buried the character's feet instead of providing an intentional raised fringe.

The implementation now aligns Ashglass's flat artwork to collision in both renderers while leaving physics and other biome offsets unchanged. Reinspection of corrected middle/upper Canvas and Pixi captures finds the visible continuous slab at approximately screen Y588 on ground and Y539 on the upper lane, matching measured collision Y587.82 and Y538.46 within one screenshot pixel. The character's visible boots sit immediately above that edge. No new gap or buried feet was visible in these four corrected scenes.

See the [contact evidence and regression scope](terrain-contact-review.md), [recorded runtime coordinates](contact-verification/ashglass-contact-measurements.json), and preserved [ground Canvas](contact-verification/ashglassPass-middle-canvas.png), [ground Pixi](contact-verification/ashglassPass-middle-pixi.png), [upper Canvas](contact-verification/ashglassPass-upper-canvas.png), and [upper Pixi](contact-verification/ashglassPass-upper-pixi.png) captures. These replace the original Ashglass registration observation; the earlier 60-file coverage table remains a record of what was originally inspected. Endcap bevels are explicitly outside the repeated flat-plateau pixel assertion.

## Exact inspected files

Every filename below was opened at its original resolution; no contact-sheet substitution is counted.

| Map ID | Desktop Canvas, verified/ | Portrait/reduced effects, verified-mobile-reduced/ |
| --- | --- | --- |
| `starfallCrossing` | `starfallCrossing-arrival-canvas.png`<br>`starfallCrossing-upper-canvas.png` | `starfallCrossing-arrival-canvas.png`<br>`starfallCrossing-arrival-pixi.png`<br>`starfallCrossing-upper-canvas.png`<br>`starfallCrossing-upper-pixi.png` |
| `greenrootMeadow` | `greenrootMeadow-arrival-canvas.png`<br>`greenrootMeadow-upper-canvas.png` | `greenrootMeadow-arrival-canvas.png`<br>`greenrootMeadow-arrival-pixi.png`<br>`greenrootMeadow-upper-canvas.png`<br>`greenrootMeadow-upper-pixi.png` |
| `thornpathThicket` | `thornpathThicket-arrival-canvas.png`<br>`thornpathThicket-upper-canvas.png` | `thornpathThicket-arrival-canvas.png`<br>`thornpathThicket-arrival-pixi.png`<br>`thornpathThicket-upper-canvas.png`<br>`thornpathThicket-upper-pixi.png` |
| `rustcoilRuins` | `rustcoilRuins-arrival-canvas.png`<br>`rustcoilRuins-upper-canvas.png` | `rustcoilRuins-arrival-canvas.png`<br>`rustcoilRuins-arrival-pixi.png`<br>`rustcoilRuins-upper-canvas.png`<br>`rustcoilRuins-upper-pixi.png` |
| `cinderHollow` | `cinderHollow-arrival-canvas.png`<br>`cinderHollow-upper-canvas.png` | `cinderHollow-arrival-canvas.png`<br>`cinderHollow-arrival-pixi.png`<br>`cinderHollow-upper-canvas.png`<br>`cinderHollow-upper-pixi.png` |
| `ashglassPass` | `ashglassPass-arrival-canvas.png`<br>`ashglassPass-upper-canvas.png` | `ashglassPass-arrival-canvas.png`<br>`ashglassPass-arrival-pixi.png`<br>`ashglassPass-upper-canvas.png`<br>`ashglassPass-upper-pixi.png` |
| `glacierSpine` | `glacierSpine-arrival-canvas.png`<br>`glacierSpine-upper-canvas.png` | `glacierSpine-arrival-canvas.png`<br>`glacierSpine-arrival-pixi.png`<br>`glacierSpine-upper-canvas.png`<br>`glacierSpine-upper-pixi.png` |
| `stormbreakCliffs` | `stormbreakCliffs-arrival-canvas.png`<br>`stormbreakCliffs-upper-canvas.png` | `stormbreakCliffs-arrival-canvas.png`<br>`stormbreakCliffs-arrival-pixi.png`<br>`stormbreakCliffs-upper-canvas.png`<br>`stormbreakCliffs-upper-pixi.png` |
| `astralArchive` | `astralArchive-arrival-canvas.png`<br>`astralArchive-upper-canvas.png` | `astralArchive-arrival-canvas.png`<br>`astralArchive-arrival-pixi.png`<br>`astralArchive-upper-canvas.png`<br>`astralArchive-upper-pixi.png` |
| `eclipseFrontier` | `eclipseFrontier-arrival-canvas.png`<br>`eclipseFrontier-upper-canvas.png` | `eclipseFrontier-arrival-canvas.png`<br>`eclipseFrontier-arrival-pixi.png`<br>`eclipseFrontier-upper-canvas.png`<br>`eclipseFrontier-upper-pixi.png` |

See [maps 28–55 at full size](fullsize-second-half.md) and [connector coverage](visual-review.md#final-hunting-connector-checks) for the other explicitly reviewed subsets.
