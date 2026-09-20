# Starfall icon, UI and equipment sources

This directory owns 422 active production files: 209 item icons, 85 skill icons, 21 monster-card icons, 18 menu icons, three menu screens, one selection pedestal, and 85 equipment atlases. Historical inactive variants are retained.

`catalog.json` maps 32 logical raster batches to every active icon/UI output, including the 30 regional items missing from the previous item processor. `prompts/` contains the exact accepted generation briefs. `raw/` preserves original built-in imagegen PNG bytes; `rejected/` preserves eight unusable attempts and their briefs. `ledger.json` records baseline/output/source hashes, actual reference inputs, extraction method, original generation location, and the 30 immutable session references.

All skill briefs use current gameplay descriptions. Coral damage, mint healing pluses, gold enhancement chevrons, cyan protection, violet impairment and blue resource symbols follow the canonical combat visual language. Three corrected icons are owned by `icon-semantic-corrections`: Ratchet Repeater is a mechanical crossbow, Sniper mastery is a longbow, and Heat Vent has damage/resource symbolism.

Equipment is native editable SVG artwork, produced by `build/project-starfall-equipment-illustrations.js` through the existing equipment generator. It retains all 128px cells, eight angle slots, the three bow states and the published grip pivots. The 85 masters are in `equipment/`; the atlas contains 856 validated cells. No generated raster sprite has been repainted by these vector routines.

The importer uses native alpha or processor-safe flat green, groups connected components without clipping diagonal objects, crops visible art, and adds consistent runtime padding. Coupon source backgrounds required an explicitly recorded silhouette extraction mask; the physical ticket artwork is unchanged. Rebuilds preserve raw sources.

```powershell
node build/project-starfall-overhaul-icons.js rebuild all
node build/generate-project-starfall-equipment-atlases.js --all
node build/verify-project-starfall-overhaul-icons.js
```

The final command verifies all source hashes, dimensions, margins, changed output hashes, equipment angles and session-image preservation. It also writes small-size contact sheets and `validation.json` under `output/starfall-overhaul-icons/`.

Legacy item, skill and card processors route to this owner. The old visual sweep routes its menu and coupon branches here. Item derivative generation may consume the new artwork but cannot overwrite it with recolored aliases. The existing equipment generator calls the new vector masters. Start-screen AVIF/WebP deployment alternatives remain derived through `build/optimize-site-images.js` from the replaced canonical PNG.
