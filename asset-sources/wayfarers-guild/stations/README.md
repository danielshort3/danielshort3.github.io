# Stacked-station production art

The approved station-and-area-upgrades and earned-station-expansion concepts are the art anchors. The six generated area masters preserve mountain scenery, timber and rock depth, lanterns, machinery, and aqua accents. Icon masters use stable upgrade subjects, not atlas position, to assign the 180 local illustrations. The worker master supplies six roles and four separate poses.

`prompts.json` records the generation prompts; `sources.json` records selected masters, measured crop boundaries, and icon subjects. Regenerate exports with `node build/process-wayfarers-station-art.cjs`. The processor extracts, trims, and resizes these generated layers; it does not draw substitute scenery or controls.

Published flat WebP files live in `img/wayfarers-guild/`. `station-art.json` records source and output SHA-256 hashes, dimensions, crop rectangles, byte sizes, and anchors. `station-art.js` is the browser/Android runtime catalog. Each station keeps a 384 × 208 logical segment; the first station also has a 112-pixel surface backdrop. Worker cells are 64 × 96. UI text and buttons remain live DOM controls.

Focused validation: `node --test tests/games/wayfarers-guild-station-art.test.cjs`.
