# Wayfarers' Guild pixel artwork

These are original production assets generated with the built-in OpenAI ImageGen tool on September 29, 2026. The approved direction is **C2 Pocket Rooms**: coarse side-view pixel characters, sparse navy structures, cream skies/walls, amber highlights, and blue clothing. The accepted reference boards were `early-progression.png`, `connected-guild-v2.png`, and the original `c-pocket-rooms.png`. Their app interfaces are references only; none is flattened into the game.

The October 1, 2026 **C · Living Guild** composition (`c-living-guild-v3.png`) adds a tall automatic-travel scene and stacked, working room interiors. Three original ImageGen backgrounds extend the existing actor and prop atlases. The backgrounds contain no workers, controls, text, or status values; all activity and interface state remains live.

## Runtime files

The October 3 collection adds eighteen individual card portraits and eight equipment icons. Each has its own transparent 96 × 96 PNG named after its stable content ID. [collection-art.json](collection-art.json) records every final prompt, original filename, source checksum and shipped checksum. The built-in ImageGen tool generated each asset separately; [process-wayfarers-collection-art.cjs](../../build/process-wayfarers-collection-art.cjs) applies only nearest-neighbor sizing and preserves source alpha. Card frames, rarity, ranks and inventory badges remain semantic HTML. `Icons.COLLECTION_ART` registers all twenty-six IDs, and the collection-art test checks uniqueness plus exact offline APK-bundle inclusion.

| File | Layout | Purpose |
| --- | --- | --- |
| `actors.png` | 128 × 160; four columns, five rows; 32 × 32 transparent cells | Adventurer walk/idle/inspection, miner and smith work pairs, scholar/cook work pairs, forager, cartographer, leader, alchemist, fox, owl, tortoise |
| `props.png` | 128 × 96; four columns, three rows; 32 × 32 transparent cells | Ore, furnace, anvil, lantern, crate, cauldron, desk, map table, basket, banner, alchemy table, sign |
| `realms.png` | 192 × 256; two columns, four rows; 96 × 64 opaque cells | Greenway, Copper Hills, Mistwood, Frostpass, Sunken Reach, Starfall Heights, Endless Frontier, empty guild room |
| `asset-manifest.json` | Versioned JSON | Exact cell names/coordinates, palette, generator identity, source IDs and SHA-256 hashes |
| `ui-icons.png` | 288 × 288; six columns and rows; 48 × 48 transparent cells | Resource, profession, equipment, relic and caravan icons generated September 30, 2026 |
| `living-trail.webp` | 384 × 480; lossless WebP | Portrait Greenway path with an unstaffed future mine |
| `living-room.webp` | 384 × 148; lossless WebP | Empty timber-framed room with space for live stations and left-side HTML labels |
| `living-mine.webp` | 384 × 148; lossless WebP | Empty underground mine floor for the live miner and ore props |

The game icon at `../home-icons/wayfarers-guild.svg` is a mechanical, pixel-exact SVG trace of the generated adventurer idle sprite on a cream tile, grouped into horizontal color runs. It is not a separate illustration. The 640 × 360 lossless WebP at `../home-previews/games/wayfarers-guild.webp` is a screenshot of the actual canvas renderer's opening Greenway scene.

## Export contract

The generated masters remain outside the repository in the ImageGen output directory. Their immutable filenames and checksums are recorded in the manifest. Only prepared runtime assets are published.

The masters were mechanically extracted with Sharp, resized with nearest-neighbor sampling, mapped to the manifest's 22-color palette, and given binary alpha (threshold 150) without dithering. Human bodies occupy roughly 16 × 24 logical pixels, with a few extra pixels for tools; complete actors fit within 22 × 25. Cells align to a common feet baseline of 30. Animals fit within 19 × 14. Props fit within 29 × 29. Transparent pixels remain transparent.

The actor master is 1122 × 1402. Its four columns are equal-width, and the inspected row boundaries are `[0, 308, 600, 884, 1146, 1402]`; these boundaries avoid neighboring feet entering another row. Each cell is trimmed against alpha before proportional nearest-neighbor scaling and centered on its runtime cell. The prop master is 1448 × 1086 with an equal 4 × 3 grid. The background master is 1086 × 1448, resized to its 192 × 256 runtime grid.

Keep the source atlas coordinates and alpha stable when replacing art. Do not smooth the images, introduce gradients/dither, add background scenery to actor cells, or bake controls and labels into sprites.

Each atlas URL in the renderer uses the first 12 characters of that file's SHA-256 as its cache version. Update the corresponding URL when replacing an atlas; the integration check verifies that the version matches the authored file.

### Living Guild background provenance

The original masters remain outside the repository under the task's `generated_images/01a0eb4e-bd54-7c61-86b3-bc8ab1f4cbb6/` directory. They were inspected before export, resized directly with Sharp's `nearest` kernel (`fit: 'fill'`) to the dimensions above, and encoded as lossless WebP (`effort: 6`). No objects, semantic content, colors, transparency, or interface elements were edited during conversion. The near-identical source and output aspect ratios preserve the authored composition.

| Runtime file | Original master | Source dimensions | Source SHA-256 | Runtime SHA-256 |
| --- | --- | --- | --- | --- |
| `living-trail.webp` | `exec-622e3a69-74e5-49d1-b7f8-b1073ba93d45.png` | 1122 × 1402 | `2340754169dc3a2f6aaf9ade9ee71421cd8a31a00faa8d19b31534bbdf3e5c2b` | `7a796037857ecc1e3d49431ff51c4879e8c2fac4cd7106ef365bbfff2d5fb72f` |
| `living-room.webp` | `exec-d06ca231-3ee2-46d2-896a-80fb3a65c041.png` | 2018 × 779 | `18f132f8acb56e358150013bcbc4be02005c97f4ee930eb81e0def7c4271a12e` | `d6a6aa91c6ce8705cff16db2b3979eb1d004d35ab31b0ebd1bebf10a1260dc1a` |
| `living-mine.webp` | `exec-649d9c91-3860-494b-834b-c6e7428851e1.png` | 2022 × 778 | `7990c6f494d659319512e5906663c822a1bd47cc2a429e3dfa00f1295c281a0a` | `bbd69124a37517d08f0228b0e40438d7e6744e9ca20dc45e8e5cd2e95198d485` |

## Renderer

`js/games/wayfarers-guild/scene.js` supports three geometries. The default panorama keeps the original 64-pixel-high logical stage with expanded side columns. `layout: 'portrait'` uses a 96-pixel-wide logical stage and adds vertical space without enlarging the character; `layout: 'room'` uses the same logical width for the wide stacked floors. All actor and prop cells remain square, and the complete stage uses one uniform nearest-neighbor scale with smoothing disabled. Fractional display scales can vary individual pixel widths slightly.

Optional `sceneArt` URLs supply separate backgrounds, for example `{trail: '/img/wayfarers-guild/living-trail.webp?v=7a796037857e', room: '/img/wayfarers-guild/living-room.webp?v=d6a6aa91c6ce', mine: '/img/wayfarers-guild/living-mine.webp?v=bbd69124a375'}`. Specific room and realm IDs can override the generic background. The generic portrait trail applies to Greenway; other realms keep their original art unless explicitly supplied. Backgrounds cover their scene independently from the sprite transform. The generated portrait's actor feet sit at 86% height; the new room and mine stations use the painted floor at 80% height. Legacy fallback floors retain their original alignment. Workers and props occupy the right side of stacked rooms, leaving the left wall for accessible HTML labels.

`WayfarersScene.create(canvas, options)` returns `update(view)`, `setRoom(id)`, `getStatus()`, `retryAssets()`, and `destroy()`. Options are optional `assetBase`, `layout`, `sceneArt`, `reducedMotion`, and `onStatus(status)` callback. The explicit view is `{ room, realm, progress, workers, companion, reducedMotion }`. The scene never mutates game state.

- Rooms: `trail`, `mine`, `forge`, `forage`, `kitchen`, `study`, `cartography`, `hall`; the spare `alchemy` art is also supported.
- Realms: `greenway`, `copperhills`, `mistwood`, `frostpass`, `sunkenreach`, `starfall`, `frontier`.
- Workers: numeric assigned-worker count or an array; a positive count depicts one representative worker. Zero leaves the station unstaffed.
- Companion: `fox`, `owl`, `tortoise`, or `null`.
- Progress: normalized 0–1; UI progress labels remain in accessible HTML.
- Status: `loading`, `ready`, or `error`, with `loaded`, `total`, failed asset names, and `layout`. The total includes optional background files. Errors can be retried explicitly or on the browser's `online` event; already-loaded atlases remain available as a visual fallback.

Work/walk frames change approximately five times per second. The renderer schedules no animation while the document or canvas is hidden, respects system and game reduced-motion preferences, and removes observers, loading callbacks, and timers on destruction. No artwork implies direct movement controls. Canvas descriptions and all interactive controls belong to the HTML app.

## Production prompts

Three separate ImageGen passes created the actor, background, and prop atlases. Full prompt specifications are in [PROMPTS.md](PROMPTS.md). The primary palette is navy `#14283a`, cream `#f5e9cc`, amber `#ffbd43`, and blue `#2071b2` with restrained supporting colors. The sparse silhouettes and blank walls are intentional.
