# Project Starfall illustrated-v1 asset migration

This report records the local production migration to the approved clean illustrated direction. All 824 originally active asset paths are accounted for, and all 31 protected session images retain their original hashes. Playback and contact-sheet review remain available in the [interactive comparison](../../output/starfall-overhaul-review/index.html). This is a local implementation; it has not been deployed.

**Alignment follow-up:** the later Lava Tick review exposed horizontal drift that the migration's geometry and coverage checks did not detect. [The correction report](ASSET_ALIGNMENT_FOLLOWUP.md) documents 12 corrected identities and the limited review scope. The completed asset accounting below does not mean every animation is aligned or anatomically seamless.

The [asset guide](ASSET_GENERATION_GUIDE.md), its [confirmed references](ASSET_GENERATION_GUIDE.md#confirmed-art-direction-and-references), and [Combat Visual Language v1](ASSET_GENERATION_GUIDE.md#combat-visual-language-v1) remain the design authorities. Approved study images and their original prompts are immutable references. New production masters and exact provenance live in [overhaul-v1](../../asset-sources/project-starfall/overhaul-v1/).

## Original scope and accounting

The [immutable baseline](../../asset-sources/project-starfall/overhaul-v1/baseline.json) records 824 originally active raster paths and 31 protected session images, including formats beyond PNG. Coverage is path-based rather than a count of unique drawings. A retired compact-enemy path is complete only when its declared replacement exists, is active, and matches the recorded output hash.

| Production family | Outputs covered | Source / owner |
| --- | ---: | --- |
| Item icons | 209 | Icon catalog and importer |
| Skill icons | 85 | Icon catalog and importer |
| Monster-card icons | 21 | Icon catalog and importer |
| Menu icons | 18 | Icon catalog and importer; runtime aliases may share one file |
| Menu screens and character pedestal | 4 | Three screen PNGs and one alpha pedestal |
| Equipment atlases | 85 | Native editable SVG artwork, 856 angle/state cells |
| Map backgrounds | 40 | Individually authored scene masters |
| Terrain, props, ramps | 120 | 40 atlases of each type, from eleven biome kits |
| Town structures, world atlas, stations | 7 | One structure atlas, one world atlas, five stations |
| Combat/global FX, portals, projectile | 134 | 124 native combat FX sheets plus six global strips, three portals and one rigid projectile strip |
| Shared player, portrait, pet | 3 | Shared player atlas, portrait and fox atlas |
| Enemy actor atlases and portraits | 96 | 48 imported sheet/portrait pairs with shared scale and registration metadata; alignment review is separately scoped |
| Optimized start-screen alternatives | 2 | AVIF/WebP derived by the normal website image optimizer |
| **Baseline scope** | **824** | 776 changed in place; 48 retired sheets replaced by 48 active expanded sheets |

The icon/UI/equipment domain therefore owns 422 files; scenery owns 167. Enemy accounting comprises 44 canonical generated identities and four bandit compatibility copies, serving 51 gameplay IDs. The additional ID aliases are `glassback` → `shardling`, `riftLantern` → `void-mote`, and `faultSkitter` → `clockbug`. These aliases and copies are not separately generated character designs.

Historical inactive art, rejected generations, and earlier comparison variants remain on disk for provenance. Their presence does not make them active runtime assets. The icon ledger protects its 30 PNG study references; the top-level baseline is the authority for all 31 protected session images.

## Map consistency extension

The subsequent map pass extends scenery to **173 runtime outputs from 89 active raster masters**. This includes **44 backgrounds**: the original 40 location/trial panoramas plus four enclosed shop rooms shared by 24 regional vendors. The immutable 824-path baseline and the historical accounting above remain unchanged; the additional outputs are one Crossing structure atlas, four shop panoramas and one neutral indoor floor atlas, each explicitly registered in the scenery ledger.

- Ashglass now has its own basalt/volcanic-glass terrain, props and ramps, bringing the source kits to twelve. Its revised terrain uses a thin contact bevel and a measured common crop row to avoid per-tile contact-height steps while preserving different authored rock formations.
- Crossing uses a dedicated eight-cell landmark atlas, square presentation bounds, separate service kiosks and a frontier gate centered on its actual portal. The accepted atlas also supplies its recorded runtime fallback.
- Cinder's revised background replaces misleading foreground shelves with atmospheric volcanic distance. Earlier accepted Ashglass and Cinder masters remain reference history.
- Weapon, armor, supply and special shops now use distinct illustrated interiors with a clear actor lane. Their existing vendors, exits and floor geometry are preserved; outdoor houses and meadow dressing are removed from the rooms. A shared gray-taupe stone and walnut floor replaces the initially reused violet Ashglass strip; the four approved room backgrounds remain unchanged.

The [scenery source guide](../../asset-sources/project-starfall/overhaul-v1/scenery/README.md), [ledger](../../asset-sources/project-starfall/overhaul-v1/scenery/ledger.json) and [manifest](../../asset-sources/project-starfall/asset-generation-manifest.json) record current ownership and reproduction. The contextual scenery test covers the 24 shop assignments, unchanged actor/portal placement, Crossing's portal alignment, repeated terrain contact edges, source hashes and export contracts. Scenery validation reports 173 outputs with no failures; this does not replace gameplay-scale review in Canvas and Pixi or imply deployment. The [scenery comparison](../../output/starfall-map-scenery-review/index.html) shows the source-art changes.

## Source ownership and reproduction

| Domain | Evidence | Authoritative processor |
| --- | --- | --- |
| Player and fox | [ledger](../../asset-sources/project-starfall/overhaul-v1/players/ledger.json), measured registration, action-pair masters, original run/strike references | [process-project-starfall-overhaul-players.js](../../build/process-project-starfall-overhaul-players.js) |
| Enemy actors | [inventory](../../asset-sources/project-starfall/overhaul-v1/enemies/inventory.json), per-identity prompt, raw source, `source.json`, import report, reviewed packed sheet | [process-project-starfall-overhaul-enemies.js](../../build/process-project-starfall-overhaul-enemies.js) |
| FX | [ledger](../../asset-sources/project-starfall/overhaul-v1/fx/ledger.json), immutable raster masters and native semantic artwork | [FX importer](../../build/process-project-starfall-overhaul-fx.js), [combat FX generator](../../build/generate-project-starfall-combat-fx.js), [native art](../../build/lib/starfall-combat-language-art.js) |
| Scenery | [ledger](../../asset-sources/project-starfall/overhaul-v1/scenery/ledger.json), exact prompts, raw masters, measured partitions and seam treatment | [process-project-starfall-overhaul-scenery.js](../../build/process-project-starfall-overhaul-scenery.js) |
| Icons, screens and equipment | [catalog](../../asset-sources/project-starfall/overhaul-v1/icons/catalog.json), [ledger](../../asset-sources/project-starfall/overhaul-v1/icons/ledger.json), exact accepted/rejected briefs, raw PNGs and editable SVG masters | [icon importer](../../build/project-starfall-overhaul-icons.js), [equipment illustrations](../../build/project-starfall-equipment-illustrations.js) through [equipment generator](../../build/generate-project-starfall-equipment-atlases.js) |

New raster masters were created with the built-in image-generation tool. Equipment and the 124 combat FX sheets use native editable vector artwork rather than pretending to be generated raster sources. The accepted icon catalog uses 32 logical batches, including two batches for the 30 regional items absent from the old item processor and one explicit semantic correction batch. Eight rejected icon-generation attempts remain preserved. Recorded crop, alpha cleanup and padding operations do not modify original raw source bytes.

Legacy menu, coupon, item, skill, card, scenery, player and FX generation entry points must respect the new owners. The equipment generator calls its new native artwork module. Run the relevant owner instead of reprocessing an obsolete key-color sheet. Website builds publish checked-in outputs and derive screen formats; no deployment or remote verification is implied by a local build.

## Formats and playback

Player production is **8 columns × 10 rows of 160px cells**, or **1280×1600**, with rows idle, run, jump, fall, climb, basic, skill, party, hit and defeat. The twelve classes share the compact player body and portrait; equipment overlays preserve the 128px/eight-angle contract, with rest/draw/release rows for bows. The fox remains six columns × six rows of 160px cells.

Enemy production is **6 columns × 8 rows of 160px cells**, or **960×1280**, with rows idle, move, telegraph, attack, projectile, buff, hit and defeat. The enemy inventory maps previous `*-compact-sheet.png` paths to `*-sheet.png` replacements. Source dimensions can differ from runtime dimensions. The initial importer measured actor bounds but still used nominal horizontal column centers for many poses; those centers did not follow unevenly spaced bodies. The [follow-up](ASSET_ALIGNMENT_FOLLOWUP.md) replaces them with reviewed anatomical anchors for its stated scope.

[Animation data](../../js/games/project-starfall/data/animations.js) owns FPS, frame order and hold weights. A weight of four consumes four ticks at the specified FPS; it is not four independently drawn poses. Player run is eight frames at 40/3 FPS. Weighted jump/basic/party sequences preserve their intended contact and recovery phases. Do not replace source articulation with per-frame resizing, translated duplicate poses, or crossfades.

| Event | Local migration timing / rule |
| --- | --- |
| Player basic melee | Contact at 90 ms |
| Player ranged basic | Release at 130 ms; damage on projectile collision |
| Offensive skill | Prepare for 166.7 ms, then resolve its pending contact/release once |
| Normal healing cast | Gather for 350 ms, then show the restorative pulse and apply HP together |
| Ordinary enemy warning | Initial baselines: melee 420 ms, projectile release 540 ms, charge 750 ms |
| Major threat | About one second or more, preserving authored longer warnings |
| Final enemy commitment | Hold the last telegraph pose at least 200 ms; bosses/major threats 300 ms |
| Reactive defense / triggered recovery | Apply protection or recovery at its actual trigger with a same-frame functional cue; do not invent a normal-cast delay |

Prepared skill cancellation removes its pending resolution and corresponding preparation cue. Projectile release and collision remain separate. Persistent hazards and damage ticks retain active/status cues without restarting a full windup. Pose holds alone do not prove target locking or accurate danger geometry; those require runtime checks.

The canonical semantic palette is unchanged: mint healing, coral danger/damage, gold enhancement, cyan protection, violet impairment and blue resource recovery, with the required symbols. Actor, material and region colors remain identity accents. Combined effects retain each actual meaning, including direct drain or pack-call healing.

## Registration and visual review

Actor importers use source alpha, measured cells, foot/hover origins and shared scales. These properties do not establish per-pose horizontal alignment: the initial enemy fallback used nominal grid centers. Player registration and equipment sockets are derived from the packed output and reviewed at gameplay size. Enemy import reports record shared identity scale, source/output bounds, warnings and source hash, plus explicit `registrationReview` scope where anatomical anchoring has been reviewed. Alpha cleanup excludes detached extraction noise without recoloring the actor. Approved study PNGs are reused as immutable references or source inputs, never overwritten by the export.

Scenery separates atmospheric background paintings from readable platform art. The original 40 backgrounds, plus the four shop interiors in the map extension, are bounded, cover-scaled panoramas in Canvas and Pixi rather than edge-periodic textures. Repeatable terrain cells use a recorded premultiplied-alpha edge treatment; raw masters remain unchanged. Props, ramps, structures and stations retain their cell margins and aspect ratio.

The current rendering direction uses controlled illustrated edges and linear sprite filtering; it does not impose a universal strict pixel grid or fixed small palette. Verify both renderers, fractional camera scales, bright/dark scenery and small-screen readability. Stable feet do not by themselves prove stable faces, cap spots, crown points or other landmarks. Raw source review, frame stepping, loop/recovery playback, equipment attachment checks and alpha/bounds metrics complement one another.

## Validation status and completion gates

**Asset integration: complete.** The [coverage audit](../../asset-sources/project-starfall/overhaul-v1/coverage.json) verifies 824 active paths: 776 changed in place, 48 retired compact sheets and 48 active expanded replacements. There are zero unchanged baseline outputs, missing paths, unaccounted retirements or protected-image hash failures. The [enemy audit](../../asset-sources/project-starfall/overhaul-v1/enemies/validation-report.json) verifies 48 atlases, 2,304 frame slots, 44 distinct canonical identities, four compatibility copies and 96 original backup files.

Completed checks after enemy integration include assets (34,682 assertions plus class identity and asset URLs), smoke (20,359), systems (22,335 plus focused suites), combat (4,164), inventory (4,047), balance (4,012), progression (4,573), Eclipse, bundle/viewport, skill-FX quality/runtime, semantic contact/cancellation timing and focused fracture/rift progression. Icon/equipment owner validation covers 422 outputs; scenery validation covers 167 outputs, alpha, containment and repeat seams. The full website build and whitespace checks pass. All 824 active source assets match their built `public/` copies. Exact runs and the unrelated failure are recorded in the [regression summary](../../output/starfall-overhaul-validation/runtime-regression-summary.json).

The motion checks compare adjacent poses and preserve two declared Bristle Boar tail holds from its approved four-pose study; those repeated slots are timing holds, not additional independent drawings. The interactive review verifies 52 actor selections, 80 player frame boundaries and all 51 representative enemy commitment windows at four viewport widths. It shows isolated character rows; the running game is the reference for equipment, combined effects and combat scheduling.

The earlier [project-starfall-recurring-loop.test.js](../../tests/project-starfall/project-starfall-recurring-loop.test.js) failure was a clock-dependent fixture: its explicit July timestamps disagreed with wall-clock runtime normalization. The integrated rollback test now controls `Date.now()` and the supplied shop clock together, restores the real clock in `finally`, and preserves the assertions for rollback, returning to the stored week and the next forward reset. The focused test and `npm.cmd run test:starfall` now pass. The linked regression summary retains the earlier failure as historical evidence.

Live browser checks cover warm town/meadow scenery, the bright Frostfen healing sample and dark Eclipse scenery. Canvas and Pixi load the expanded sheets without missing assets. The staged Oracle case retains 477 HP during preparation and reaches 591 HP at the 350 ms restorative pose. Rising plus marks draw above actors while the segmented ring remains beneath them. Reduced-effects checks retain coral danger footprints and cyan safe-zone cues. A 390px game viewport has no document overflow; the comparison viewer also passes 320px, 390px, 736px and 1160px widths.

Pixi's existing dependency needed its matching official CSP interpreter to initialize under the site's script policy; the policy was not relaxed. Failed-initialization cleanup is guarded. The dependency source, version, license and checksum are in [its notice](../../js/vendor/pixi-unsafe-eval.NOTICE.md). Both strict-CSP regression tests and actual WebGL initialization pass.

Pixi's world-only canvas is clipped to the same playfield plus solid-band boundary as Canvas, so large ramp sprites cannot cover the HUD. The clipping regression covers viewport resizing, render-resolution changes and a band extending beyond the viewport. The HUD remains on its separate canvas.

The [browser verification record](../../output/starfall-overhaul-validation/browser-validation.json) links the local game evidence. Final full-run retries encountered Windows `UNKNOWN` write errors in unrelated generated site files. The failed digest stage and remaining public-copy/preview-validation stages subsequently completed individually, and the final `dc3be71b` game bundle was verified in the browser. Source/public asset parity remains 824 of 824.

Reproduction checks:

```powershell
node build/verify-project-starfall-overhaul-icons.js
node build/process-project-starfall-overhaul-scenery.js --validate
node asset-sources/project-starfall/overhaul-v1/scenery/validate-scenery.cjs
node build/audit-project-starfall-overhaul.js --complete
npm.cmd run validate:project-starfall-assets
npm.cmd run validate:project-starfall-class-skills
npm.cmd run test:starfall:full
node tests/project-starfall/project-starfall-combat-visual-language.test.js
npm.cmd run build
node build/audit-project-starfall-overhaul.js --complete --public
git diff --check
```

When changing a source or importer, repeat the affected checks and inspect the local game in Canvas and Pixi. After the deterministic clock-fixture fix, `test:starfall` and `test:starfall:assets` pass independently; current skill-FX, enemy-FX and viewport checks also pass. The aggregate `test:starfall:full` has not been rerun in this validation pass, and the earlier bundle result remains recorded separately. Coverage requires no missing paths, no unaccounted retirements, no unchanged baseline outputs and no protected-image hash failures. Generated artwork still requires visual judgment for faces, clothing landmarks and motion; the comparison viewer supports further targeted feedback. Production deployment and remote parity have not been performed.
