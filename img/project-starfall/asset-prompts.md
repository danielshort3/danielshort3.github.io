# Project Starfall Asset Prompts

Generated on 2026-05-16 for the static browser prototype. Source sheets were produced with the image generation tool, then cropped into individual deployable assets under `img/project-starfall/`.

Canonical forward art direction: **Fractured Starfront Sci-Fantasy**. Future Project Starfall assets should use painterly 2.5D side-scroller environments, cel-shaded sprites, crisp contour lines, readable silhouettes, shattered observatory stone, dark iron repair structures, luminous cyan star-tech, restrained ember-gold utility light, asymmetrical frontier construction, and practical expedition materials. Avoid ornate blue-and-gold medieval guild architecture, cute village motifs, generic fantasy-market dressing, and decorative clutter that weakens gameplay silhouettes. Historical prompt summaries below may describe earlier passes, but replacement assets should follow this original Project Starfall language.

## Shared Bordered-Sheet Rule

Updated on 2026-05-24 with `build/project-starfall-sheet-grid.js`.

All future AI-generated source sheets that are processed as grids must include thin solid `#00ffff` guide lines around the whole sheet and between every cell. The guide color must not appear inside the art. Source processors detect these borders before removing chroma-key backgrounds, slice only from the interiors of the detected cells, and fail loudly when an expected row, column, or outer border cannot be found. This applies to enemy animation sheets, compact test sheets, projectile sheets, item icon sheets, skill icon sheets, and any future non-enemy grid source for UI icons, equipment layers, props, portals, or VFX. Legacy code-native generators may still exist for provenance, but replacement assets should keep source images in the workspace and flow through a processor.

## Character Sheet

Workspace source: `img/project-starfall/characters/source/ai-class-sheet.png`

Prompt summary: 3 by 3 grid of full-body side-view sprites for Fighter, Mage, Archer, Guardian, Berserker, Fire Mage, Rune Mage, Sniper, and Trapper. Starlit Frontier Fantasy, readable silhouettes, non-photoreal painterly style, flat `#FF00FF` chroma-key background, no labels or watermark.

Updated on 2026-07-21 with `build/process-project-starfall-player-ai-assets.js`.

Playable portraits and runtime sheets now derive from three v5 Fracture Runner family sources under `asset-sources/project-starfall/players/class-families/`: plated Fighter, long-mantled Mage, and light field-leather Archer. All three keep the approved athletic 5.5-to-6-head adult frontier-operative anatomy, compact realistic head, practical hair, charcoal field materials, cyan star-tech seams, restrained burnt-orange utility accents, and the same registered 60-frame skeleton. Advanced classes reuse their base family sheet. The v4 body and earlier chibi/anime passes are recovery or legacy provenance, not current class art. Keep `build/process-project-starfall-player-ai-assets.js` or `build/generate-project-starfall-action-sprites.js --only players` as the sole refresh path for `img/project-starfall/characters/` and `img/project-starfall/animations/players/`.

## Enemy Sheet

Source: `/home/sd205521/.codex/generated_images/019e31c3-9335-7fb1-918d-7c0126258f36/ig_0efed1a3122171dd016a08aeb4c7d08194ba8976a18d514b9d.png`

Prompt summary: 4 by 4 grid containing Slimelet, Mossback, Thorn Sprout, Bristle Boar, Dust Imp, Clockbug, Ember Wisp, Bandit Cutter, Bandit Thrower, Oreback Beetle, Glowcap Healer, Cracked Mimic, and Emberjaw Golem. Side-view enemy silhouettes, Starlit Frontier Fantasy, flat `#FF00FF` chroma-key background.

## Gear Sheets

Source: `/home/sd205521/.codex/generated_images/019e31c3-9335-7fb1-918d-7c0126258f36/ig_0efed1a3122171dd016a08aefc94f08194b81577d26cc71d02.png`

Prompt summary: 4 by 4 grid of inventory icons for the base shop gear: Training Sword, Training Wand, Training Bow, Copper Sword, Birch Wand, Simple Bow, Stitched Vest, Traveler Boots, Plain Ring, Iron Sword, Iron Axe, Apprentice Staff, Oak Longbow, Guardian Tower Shield, Ember Core, and Trap Kit.

Source: `/home/sd205521/.codex/generated_images/019e31c3-9335-7fb1-918d-7c0126258f36/ig_0efed1a3122171dd016a08b20202b88194ad6395f4e0874bae.png`

Prompt summary: 3-icon row for the missing advanced offhand gear: Berserker War Grip, Rune-Etched Focus, and Deadeye Scope.

## Map Backdrop Sheet

Source: `/home/sd205521/.codex/generated_images/019e31c3-9335-7fb1-918d-7c0126258f36/ig_0efed1a3122171dd016a08af3dc28481948d22b0a6ff943aed.png`

Prompt summary: 4 by 2 grid of side-scrolling map backdrops for Starfall Crossing, Greenroot Meadow, Thornpath Thicket, Rustcoil Ruins, Cinder Hollow, Bandit Ridge Camp, Oreback Quarry, and a spare frontier panel.

## Individual Seamless Map Background Pass

Updated on 2026-05-17 with individually generated 2:1 map backdrops and a Sharp seam-processing pass.

Workspace source directory: `img/project-starfall/maps/source/field/`

Prompt summary: one image per Project Starfall map, using readable pixel-fantasy side-scroller backgrounds with no characters, monsters, UI, labels, or watermarks. Each prompt required a horizontally seamless tile where the left and right edges match for endless canvas repetition. The final WebP files were normalized to 1280 by 640 and processed so the wrap boundary is exact. Late-game maps now have unique files for Ashglass Pass, Stormbreak Cliffs, Astral Archive, Eclipse Frontier, and Endless Rift instead of reusing older backgrounds.

Updated on 2026-05-21 with `build/process-project-starfall-map-backgrounds.js`, then revised on 2026-05-22 to remove the half-width repeated-slice seam.

The source-backed full-map WebP backdrops use full-width seamless wrap blending instead of reflected edge synthesis or half-tile repetition. Each tiled output keeps the original generated image through the center, blends in a half-width circular offset only near the outer edges, and locks the final edge columns so the left and right sides meet cleanly when the renderer draws adjacent copies. The processor does not use x-axis flipping, flopping, negative scale, or reversed sampling. Starfall Crossing is the deliberate exception: `img/project-starfall/maps/source/field/starfall-crossing-fractured-observatory-v1.png` becomes a single 2560 by 640 panorama so its fractured observatory ring pans across the hub once rather than repeating. The Crossing prompt uses an original frontier repair settlement built through broken observatory ribs, suspended stone plates, angular iron gantries, survey equipment, cyan star-tech, and sparse ember utility light, with no characters, enemies, UI, text, logos, medieval village stalls, flower boxes, or ornate guild banners.

Updated on 2026-05-22 with dedicated safe-zone hub backgrounds for Rustcoil Outpost, Cinder Refuge, Frostfen Camp, Stormbreak Haven, and Astral Observatory.

Workspace source directory: `img/project-starfall/maps/source/safe-zones/`

Prompt summary: one built-in image generation prompt per regional town, using polished side-scrolling MMORPG town backdrops with no characters, monsters, UI, labels, logos, or watermarks. The final WebP files were normalized to 1280 by 640 under `img/project-starfall/maps/`, giving every regional safe-zone map a dedicated background instead of reusing its neighboring combat field.

These safe-zone WebPs are also processed by the full-width seamless wrap pass so regional towns tile in normal orientation without mirrored halves or repeated 640px slices.

Updated on 2026-05-25 with `build/process-project-starfall-non-enemy-ai-assets.js`.

The source-backed non-enemy rebuild command runs map backgrounds, imagegen environment terrain and props, AI item icons, and skill icons while refusing to invoke enemy-owned processors. The map processor now reads named workspace source files rather than `$CODEX_HOME/generated_images` paths, so project-bound map sources are portable and reviewable. Bramble Depths, Gearworks Vault, and Emberjaw Lair were added to the source-backed map pass so their backgrounds match the surrounding field art instead of relying on the legacy generated-map fallback.

## Station Sheet

Source: `/home/sd205521/.codex/generated_images/019e31c3-9335-7fb1-918d-7c0126258f36/ig_0efed1a3122171dd016a08af70e2408194a3171c8b07e17d32.png`

Prompt summary: Horizontal row of safe-zone station props for Starter Outfitter, Upgrade Artisan, and Class Supplier. Side-view platformer prop art with flat `#FF00FF` chroma-key background.

## Animation Sprite Sheets

Character and enemy animation sheets were generated after the static character and enemy assets, preserving priority order. Each deployable sheet is a transparent PNG under `img/project-starfall/animations/`.

Player sheets: 6 columns by 10 action rows for idle, run, jump, fall, climb, basic attack, skill, party buff, hit, and defeat. The current source-backed family sheets use authored sequential poses and the shared registration contract rather than whole-body transforms.

Original enemy sheets: 6 columns by 7 action rows for idle, move, telegraph, attack, projectile, hit, and defeat. This was superseded by the Enemy Pixel-Art Animation Pass below.

FX source: `/home/sd205521/.codex/generated_images/019e31c3-9335-7fb1-918d-7c0126258f36/ig_0efed1a3122171dd016a08c422bee08194b572eb226b035930.png`

FX prompt summary: 6 by 1 horizontal sprite sheet of a melee slash arc, blue arcane cast burst, golden arrow streak, party aura, impact spark, and smoky defeat burst. Starlit Frontier Fantasy, transparent-ready flat `#FF00FF` background.

## AI Skill Combat FX Pass

Updated on 2026-05-30 with `build/process-project-starfall-ai-skill-fx.js`.

Workspace source directory: `img/project-starfall/animations/combat-fx/skills/source/`

Generated field-skill sources:
- Ground Glyph: `/home/sd205521/.codex/generated_images/019e7acc-e769-7210-b933-f0a8ac8a72ef/ig_04d25c7eb75e4c4d016a1b5c1c52988193be57756cc0a6f9b3.png`
- Grand Inscription: `/home/sd205521/.codex/generated_images/019e7acc-e769-7210-b933-f0a8ac8a72ef/ig_04d25c7eb75e4c4d016a1b5c4f60c081939973cadf37ab9044.png`

Prompt summary: one generated 6 by 4 sprite sheet per active skill, ordered by rows `cast`, `projectile`, `impact`, and `area`, with six frames per row. Sources should use Project Starfall's Starlit Frontier Fantasy spell and weapon VFX language: readable silhouettes, luminous star-magic accents, custom motifs per skill, no UI, no text, no labels, no watermarks, no character bodies, and transparent-ready flat `#00ff00` or `#ff00ff` backgrounds. Bordered sources may use the shared `#00ffff` guide color; the processor removes chroma-key and guide pixels, normalizes each sheet to 960 by 640, validates visible art, and writes the runtime sheets under `img/project-starfall/animations/combat-fx/skills/`.

`build/process-project-starfall-ai-visual-sweep.js` now owns the broad source-backed visual replacement pass for global FX, basic attack FX, enemy combat FX, enemy projectiles, portals, menu icons, town landmark structures, class trial maps, missing derived map backgrounds, and the rate coupon sheet. It intentionally does not rewrite playable player assets, which are owned by `build/process-project-starfall-player-ai-assets.js`. Equipment-layer redesign remains deferred so base character likeness and animation can be approved before gear is rebuilt. The generated source sheets live under the matching `img/project-starfall/**/source/` folders, old deterministic outputs are preserved under `img/project-starfall/backups/procedural/`, and runtime asset loading can automatically use those backup images if an AI primary fails to load.

Mage projectile rows now use `img/project-starfall/animations/combat-fx/projectiles/source/ai-mage-projectile-rows.png`, generated from `/home/sd205521/.codex/generated_images/019e7ac8-62d4-7fe0-b3c9-2019cc7c9fd1/ig_0db365bbc683087e016a1b9878237481958b9aff9da407075c.png`. The source is a 6 by 4 bordered sheet for magic, fire, rune, and lightning projectile rows; every frame must be centered, right-facing, and horizontally level so mage projectile visuals do not imply an upward shot.

## Action Sprite Refinement

Updated on 2026-05-16 with `build/generate-project-starfall-action-sprites.js`.

The player sheets were regenerated deterministically from the existing character PNGs. The first two frames of walk/run, jump, fall, basic attack, and skill rows now include explicit foot, leg, weapon, bow, casting, and lunge overlays so action rows read as distinct poses instead of small whole-body tilts. The sheet grid and filenames were preserved for the prototype.

The melee slash FX sheet was regenerated as a right-facing forward stroke. The engine flips it only for left-facing attacks and anchors player attack FX in front of the character.

## Fighter Pixel-Art Review Pass

Updated on 2026-05-16 with `build/generate-project-starfall-action-sprites.js`.

Only the base Fighter/warrior assets were replaced for review. `img/project-starfall/characters/fighter.png` and `img/project-starfall/animations/players/fighter-sheet.png` now use deterministic limited-color block pixel art with a compact side-view warrior, chunky armor, scarf, simple sword, and two-frame walk, jump, fall, attack, skill, party, hit, and defeat poses. Mage, Archer, advanced classes, enemies, gear, maps, and FX assets were left unchanged.

## Generic Player and Equipment Layers

Updated on 2026-05-16 with `build/generate-project-starfall-action-sprites.js`.

The playable-character visual model uses a deterministic generic body as the shared rig source: `img/project-starfall/characters/generic-player.png` and `img/project-starfall/animations/players/generic-player-sheet.png`. The sheet is a 6 by 10 grid covering idle, run, jump, fall, climb, basic attack, skill cast, party/buff cast, hit, and defeat.

Each shop item now has a matching transparent equipment layer sheet under `img/project-starfall/equipment-layers/`. The sheets share the exact player frame grid, so equipped weapons, chest armor, boots, rings, shields, focuses, scopes, cores, grips, and kits are composited on top of the same generic body during walking, jumping, attacking, climbing, and buff casting.

Updated on 2026-07-21 with the Fracture Runner identity pass in `build/process-project-starfall-player-ai-assets.js`.

The v5 Fighter, Mage, and Archer sources live under `asset-sources/project-starfall/players/class-families/`. The processor writes cache-safe `*-v5.png` portraits and `*-sheet-v5.png` sheets, maps every advanced class to its base family, and retains `generic-player-v4` only for failed-load recovery. Validation protects cross-family alpha-mask differences, authored material separation, transparent frame edges, planted registration, all equipment sockets, starter-equipment composites, and semantic action progression.

The standard runtime player contract remains a 6 by 10 grid of 160px frames ordered as idle, run, jump, fall, climb, basic, skill, party, hit, and defeat. Player source sheets use the shared `#00ffff` guide grid with transparent cells so frames do not pick up chroma-key fringe. The player validator requires planted idle registration, alternating run stride, jump lift/landing, falling brace, climb reach/pull, attack windup/strike, skill charge/cast/release, party channel, hit recoil, and defeat collapse. Existing equipment atlases remain modular and can be retuned without changing body physics.

Updated on 2026-06-08 with the connected player animation pass in `build/process-project-starfall-player-ai-assets.js`.

The source-backed processor keeps all three Fracture Runner families as connected full-body silhouettes across every row. The runtime rig continues to expose one named, per-frame attachment contract while each body stays weaponless. The broad action-sprite generator delegates player writes to this processor, preventing a default rebuild from silently restoring deprecated procedural player art.

Updated on 2026-07-21 with `build/generate-project-starfall-equipment-atlases.js`.

Runtime equipment now uses the 85 versioned `*-atlas-v2.png` modular atlases under `img/project-starfall/equipment-atlases/`; the older full player-grid equipment layers and unversioned atlas filenames are provenance only. Starter and early Fighter, Mage, and Archer gear share the `fractured-starfront` art-direction profile: compact adult-scale silhouettes, charcoal field materials, muted iron plates, cyan star-tech seams, and restrained ember utility marks. Weapons retain the authored grip pivots, while boots, helmets, gloves, and chest pieces use explicit neutral-frame size budgets so modular gear does not overwhelm the Fracture Runner body. The generator validates palette depth, cyan and ember accents, transparent cell borders, all eight socket angles, the v2 cache-busting filename contract, and the complete 85-atlas file set on every refresh.

## Enemy Pixel-Art Animation Pass

Updated on 2026-05-16 with `build/generate-project-starfall-action-sprites.js`.

All enemy portraits and enemy animation sheets are now regenerated deterministically in the same simplified blocky pixel direction as the generic player. The enemy sheets use a 6 by 8 grid for idle, move, telegraph, attack, projectile, buff, hit, and defeat. Glowcap Healer support actions and Emberjaw Golem overheat cues use the dedicated buff row, while melee, ranged, flying, turret, elite, and boss enemies each get readable movement, attack/projectile, hit, and death frames.

## Drop Item Icon Pass

Superseded on 2026-05-24 by the AI item icon sheet pass below. The historical deterministic notes are retained only for provenance.

Updated on 2026-05-16 with `build/generate-project-starfall-action-sprites.js`.

Missing ground-drop icons are now generated deterministically as transparent blocky pixel-art PNGs under `img/project-starfall/items/`: coins, minor health potion, minor resource tonic, equipment slot coupon, upgrade dust, upgrade catalyst, fracture dust, gel drop, and ore chunks. These icons are intended to render without a dropped-item frame; the engine supplies only a tier-colored aura from the canvas shadow around the icon.

Updated on 2026-05-17 with the same deterministic generator for the remaining item icons: pet whistle, potential cube, base skill manual, advanced skill manual, SP reset scroll, warding scroll, and refinement core. The generator now supports `--only items` for refreshing item PNGs without rewriting player, enemy, map, or animation assets.

Updated on 2026-05-18 with deterministic boss-drop item icons for boss equipment that had no reusable visual asset. This superseded pass added transparent 180px PNG icons for the Thorncrown, Furnaceheart, Titanwork, Deepcore, Stormcaller, Astral, and Eclipse head/glove drops while leaving boss drops that already reused weapon, chest, or boot visuals unchanged.

## AI Item Icon Sheet Pass

Updated on 2026-05-25 with the built-in image generation tool and `build/process-project-starfall-ai-item-icons.js`.

Workspace source sheets:
- `img/project-starfall/items/source/ai-items-consumables-materials.png`
- `img/project-starfall/items/source/ai-items-potion-tiers.png`
- `img/project-starfall/items/source/ai-items-rate-coupons.png`
- `img/project-starfall/items/source/ai-items-slot-prisms-plinko.png`
- `img/project-starfall/items/source/ai-items-mob-materials-core.png`
- `img/project-starfall/items/source/ai-items-mob-materials-late.png`
- `img/project-starfall/items/source/ai-items-shop-boss-forest.png`
- `img/project-starfall/items/source/ai-items-boss-core-storm.png`
- `img/project-starfall/items/source/ai-items-boss-astral-eclipse.png`

Generated sources:
- `/home/sd205521/.codex/generated_images/019e4330-b02e-7a02-89c6-1c10e5514eb6/ig_02e102501c98f839016a14444fe28c8193b8914c4b2184a64a.png`
- `/home/sd205521/.codex/generated_images/019e4330-b02e-7a02-89c6-1c10e5514eb6/ig_02e102501c98f839016a14469cc3c481938014be47cf106f00.png`
- `/home/sd205521/.codex/generated_images/019e4330-b02e-7a02-89c6-1c10e5514eb6/ig_02e102501c98f839016a1445e2058c8193b8bbbfc151a14867.png`
- `/home/sd205521/.codex/generated_images/019e7acc-e769-7210-b933-f0a8ac8a72ef/ig_0f4d91151dab977d016a1c9e54b9308193b6909a24493cfc70.png`
- `/home/sd205521/.codex/generated_images/019e4330-b02e-7a02-89c6-1c10e5514eb6/ig_02e102501c98f839016a14463df49481939391b2ddff4f752e.png`
- `/home/sd205521/.codex/generated_images/019e7ac8-62d4-7fe0-b3c9-2019cc7c9fd1/ig_0db365bbc683087e016a1b97d3e5588195b289d0c9f44148fb.png`
- `/home/sd205521/.codex/generated_images/019e7ac8-62d4-7fe0-b3c9-2019cc7c9fd1/ig_0db365bbc683087e016a1b982634cc819583c4164dfdc4f75e.png`
- `/home/sd205521/.codex/generated_images/019e7ac8-62d4-7fe0-b3c9-2019cc7c9fd1/ig_00de5723ad24ef89016a1c54528fb4819bba7b92d2f94dd7db.png`

Prompt summary: clean RPG inventory sprite sheets on flat `#00ff00` chroma-key backgrounds covering every entry in `ITEM_ASSETS`: consumables, tiered HP/MP/hybrid potion restoratives, currencies, rate coupons, materials, shop gear, random world-drop gear, and all boss-set equipment. Replacement sources must include the shared `#00ffff` bordered-cell grid, one centered item per cell, generous padding, no item art crossing cell borders, and no intentional `#00ff00` or neon chroma green inside item art. The processors detect that grid, slice only cell interiors, globally remove chroma-key green plus post-resize key-color remnants, reject visible edge spillover, crop to the visible icon, and write transparent 64px PNGs or item sheets from AI source art.

## Base Skill Icon Sheets

Updated on 2026-05-16 with source sheets copied into `img/project-starfall/skills/source/` and processed by `build/process-project-starfall-skill-icons.js`.

Sources:
- Fighter: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a09095f4a808196806f89efc23091ae.png`
- Mage: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a0909982e208196a108f128f0b473b3.png`
- Archer: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a0909c820048196b313994b61fa5cee.png`

Prompt summary: three 3 by 2 sheets covering Fighter, Mage, and Archer base skills. Replacement sheets must include the shared `#00ffff` bordered-cell grid. Each icon uses a centered blocky RPG symbol, flat `#ff00ff` chroma-key background, no text, no numerals, no watermark, and enough separation to crop into individual 256px PNGs under `img/project-starfall/skills/base/`. The processor removes chroma-key and corner-connected sheet backgrounds, including black fallback backgrounds, so deployable icons use real transparent alpha.

## Card Icon Sheet Pass

Updated on 2026-05-31 with the built-in image generation tool and `build/process-project-starfall-card-icons.js`.

Workspace source sheet:
- `img/project-starfall/cards/source/ai-card-icons.png`

Generated source:
- `/home/sd205521/.codex/generated_images/019e7ade-a02d-78f1-bafe-f03d0c8a1c74/ig_09a17062ccb86316016a1c9a2b68c481948983fa24bc869ab5.png`

Prompt summary: one 7 by 3 card icon sheet covering every `CARD_DEFINITIONS` entry in data order. Icons use Project Starfall's Starlit Frontier Fantasy item-symbol style, with centered readable card motifs that match the card names, tags, and guide role. The source uses the shared `#00ffff` bordered-cell grid, flat `#ff00ff` chroma-key background, no text, no labels, no UI frames, and one isolated icon per cell. The processor removes chroma-key and guide pixels, crops visible art, normalizes each icon to a transparent 64px PNG, and writes runtime assets under `img/project-starfall/cards/icons/`.

## Advanced Skill Icon and Class Variant Pass

Updated on 2026-05-16 with `build/process-project-starfall-skill-icons.js` and `build/generate-project-starfall-action-sprites.js`.

Advanced skill sheet sources:
- Guardian: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a0909f5f6e08196956ffd034da58384.png`
- Berserker: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a090a256df88196a081143dc8b381cf.png`
- Fire Mage: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a090a54fd208196bb5edaf53de27555.png`
- Rune Mage: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a090a7f23188196ae1003a9c2a042b5.png`
- Sniper: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a090aad5b30819694df9333a8e313ab.png`
- Trapper: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_007fc2b5eb25b35b016a090adc37208196b812ba8ae5504f6b.png`
- Duelist, Storm Mage, and Beast Archer: `/home/sd205521/.codex/generated_images/019e3274-f207-7663-a990-3f25e4a8438a/ig_0dbc0bfda4bdd488016a090b7d74ac81939255cfe6e0856f21.png`

Prompt summary: generated icon sheets for every advanced skill branch, using blocky pixel-art symbols, the shared `#00ffff` bordered-cell grid, no text, no labels, no UI frames, and transparent-ready flat `#ff00ff` backgrounds. Cropped icons are stored under `img/project-starfall/skills/advanced/<class>/`.

Superseded by the v5 class-family pass. Playable classes no longer point at the generic fallback or decode 12 redundant body sheets: Fighter branches reuse the plated Fighter family, Mage branches reuse the long-mantled Mage family, and Archer branches reuse the light field-leather Archer family. Specialization identity remains in skills and modular equipment until a future advanced-family art pass can meet the same 60-frame socket contract.

## Boss and Dungeon Variant Pass

Updated on 2026-05-16 with `build/generate-project-starfall-action-sprites.js`.

Brambleking, Clockwork Titan, and Quarry Colossus now have unique deterministic transparent boss portraits and 6 by 8 enemy animation sheets instead of reusing Thorn Sprout, Clockbug, and Oreback Beetle visuals. Bramble Depths, Gearworks Vault, and Emberjaw Lair now have unique blocky WebP dungeon backgrounds instead of reusing the base thicket, ruins, and hollow maps.

## Environment Terrain and Prop Atlas Pass

Updated on 2026-05-17 with `build/generate-project-starfall-environment-assets.js`.

All maps now have deterministic transparent pixel-art terrain and prop atlases under `img/project-starfall/environment/`. Terrain sheets are 8 by 4 grids of 64px natural-platform cells with multiple top, body, deep-body, underside, side-cap, shadow, and detail variants. Prop sheets are 6 by 2 grids of 64px sprites for grass, bushes, trees, rocks, flowers, small/tall objects, crates, crystals, vines, signs, and glow accents. The engine places these assets deterministically from map id and platform geometry so platforms, ground, and foreground read as richer map-specific scenery without changing collision or pathing.

Updated later on 2026-05-17 for gameplay readability. The pass kept deterministic terrain and transparent prop atlas paths, but scaled props down in the generator and restricted runtime placement so scenery never overlaps the walkable foot line. Full-height decorative props draw behind gameplay silhouettes or on safe ground-layer positions only, and there is no actor-covering foreground scenery pass after enemies, loot, projectiles, or the player. The 2026-05-24 imagegen pass re-enabled a small front-lane pass for low-profile props only.

Updated on 2026-05-22 for biome polish. The deterministic atlas generator now adds distinct platform and prop details for town, forest, construct, volcanic, quarry, frost, storm, astral, eclipse, and rift themes while preserving the same 64px cell layout, transparent prop corners, and runtime placement rules.

Updated on 2026-05-24 with the built-in image generation tool and `build/process-project-starfall-environment-imagegen-assets.js`.

Imagegen source: `/home/sd205521/.codex/generated_images/019e3725-52b4-7572-ac7e-b04887306014/ig_0a2871911ec1d435016a13cf4147848196acafb27d2caeffdc.png`

Workspace source copy: `img/project-starfall/environment/source/imagegen-environment-source.png`

Prompt summary: generated a large side-scroller MMO environment source sheet on a flat `#00ff00` chroma-key background. It includes distinct terrain strips and isolated props for guild town cobbles, meadow roots, thorn brambles, rusted gearworks, volcanic basalt, frost camp ice, storm cliffs, astral observatory structures, eclipse rift stone, and crystal quarry pieces. The processor removes chroma key, builds exact 8 by 4 natural terrain atlases and 6 by 2 transparent prop atlases, and outputs one terrain/prop pair for every playable map plus each class trial. Map profiles now use map-specific terrain IDs so repeated areas such as frost, cinder, storm, astral, and ruins no longer share the same standardized atlas.

Updated on 2026-05-30 for natural side-scroller terrain. The runtime keeps authored collision rectangles but renders terrain with overhanging lips, side caps, varied body chunks, ragged undersides, deterministic detail patches, and no repeated grass trim on organic maps. Greenroot Road I / Starter Field is the baseline for Project Starfall's own layered route readability and silhouette-safe platform treatment.

Updated on 2026-05-31 with `build/generate-project-starfall-forest-terrain-atlases.js`.

Greenroot Meadow and Thornpath Thicket now have dedicated custom AI terrain source sheets instead of cropped strips from the shared environment source. Workspace sources live at `img/project-starfall/environment/source/greenroot-meadow-terrain-imagegen-source.png` and `img/project-starfall/environment/source/thornpath-thicket-terrain-imagegen-source.png`; keyed processor outputs are stored beside them. The generator removes `#ff00ff` chroma-key from the source copies, then assembles exact 8 by 4 64px low-frequency modular terrain atlases with tighter repeated-cell seams so forest platforms render as consistent natural tiles with smoother ends and no visible magenta square artifacts.

Future replacement prompt family: "Create a high-quality side-scroller pixel-art environment source sheet on a perfectly flat #00ff00 chroma-key background. Include structurally different platform terrain strips and isolated environmental props for each biome, not just recolors. Props only, no characters, no monsters, no UI, no text, no shadows, generous padding, clean silhouettes, and low-profile variants suitable for platform edges without covering enemies." Crop into the same terrain and prop atlas formats, then validate dimensions and transparent prop corners before replacing any project asset.

Updated on 2026-05-25 for map-asset uniqueness validation. The imagegen environment processor now fails validation if a map or trial reuses another map's structural terrain/prop source key. Hue and brightness shifts can harmonize a biome, but every playable map and class trial must have a unique terrain/prop/offset combination so map assets are not just recolored copies.

Updated on 2026-05-18 for class advancement instances. `build/generate-project-starfall-action-sprites.js --only trial-maps` adds generated 1280 by 640 WebP backgrounds under `img/project-starfall/maps/trials/` for Guardian, Berserker, Duelist, Fire Mage, Rune Mage, Storm Mage, Sniper, Trapper, and Beast Archer trials. `build/generate-project-starfall-environment-assets.js --only trials` adds matching deterministic terrain and transparent prop atlases for each trial theme. The trial maps use class-specific motifs while preserving the same readable side-scroller rules: no characters, monsters, UI, labels, logos, watermarks, or foreground cover that would obscure combat.

## Portal Sprite Pass

Updated on 2026-05-17 with `build/generate-project-starfall-action-sprites.js --only portals`.

Standard, boss, and locked portal idle sheets are deterministic generated assets under `img/project-starfall/animations/portals/`. They use the existing 6-frame 160px Project Starfall animation grid and are not externally sourced or AI-generated prompt outputs.

## World Map Atlas Pass

Updated on 2026-05-18 with the built-in image generation tool, then normalized with Sharp to `img/project-starfall/world-map/starfall-atlas.webp` at 1920 by 1080.

Source: `/home/sd205521/.codex/generated_images/019e36b6-9b13-78e3-b246-30fe63de738e/ig_0f974c4664ce234b016a0a6af722888190969e38b3eddd8f3d.png`

Prompt summary: one painterly top-down fantasy MMORPG atlas background with no baked text, labels, icons, UI frames, characters, monsters, logos, or watermarks. The image arranges Starfall Crossing, Greenroot Wilds, Rustcoil Expanse, Cinder Basin, Frostfen Tundra, Stormbreak Reach, and Astral Dominion as visually distinct connected regions so interactive route markers can be rendered by the game on top.

## Boss Drop Item Icon Pass

Superseded on 2026-05-24 by `build/process-project-starfall-ai-item-icons.js`.

Updated on 2026-05-18 with `build/generate-project-starfall-action-sprites.js --only items`.

All boss set equipment moved from this superseded deterministic 180px pass to the AI-generated 64px item icon sheet pipeline above.

## Character Select Screen Pass

Updated on 2026-05-22 with the built-in image generation tool and normalized with Sharp.

Background source: `/home/sd205521/.codex/generated_images/019e36b6-9b13-78e3-b246-30fe63de738e/ig_046dc4d44cfea8f6016a0fb59a52788193a30a274177c38205.png`

Pedestal source: `/home/sd205521/.codex/generated_images/019e36b6-9b13-78e3-b246-30fe63de738e/ig_046dc4d44cfea8f6016a0fb5cf6368819398ed76625f5f967d.png`

Prompt summary: generated a polished Adventurer Hall and open guild balcony character-select backdrop with blue-and-gold Starfall banners, portal glow, lantern light, and calm central floor space for selectable character slots. The pedestal prompt generated a compact stone-and-wood display plinth with blue crystal inlays on a flat `#00ff00` chroma-key background; the final transparent PNG is used under each character slot.

## Town Structure Landmark Atlas Pass

Updated on 2026-05-23 with `build/generate-project-starfall-environment-assets.js --only structures`.

The town landmark atlas is cell-aligned at 4 by 3 256px transparent sprites. Its first two rows retain the legacy Starfall hall, Rustcoil workshop, Cinder forge, Frostfen lodge, Stormbreak gate, Astral observatory, market awning, and lantern arch. The third row is source-backed Crossing art for the Fractured Observatory Core, Expedition Depot, Lens Workshop, and Frontier Gate. New Crossing compositions should prioritize those four original structures, combine them asymmetrically, and use Project Starfall's fractured starfront palette without adding collision or blocking gameplay silhouettes.

## AI Enemy Sprite Sheet Pipeline

Updated on 2026-05-25 with `build/process-project-starfall-compact-bandits.js`.

Enemy visuals are now treated as compact source sprite sheets instead of outputs from the deterministic blocky sprite generator. The runtime contract is one transparent `384x1024` PNG per enemy under `img/project-starfall/animations/enemies/`, laid out as a 3 by 8 grid of 128px frames for idle, move, telegraph, attack, projectile, buff, hit, and defeat. Every row supplies three runtime frames, including hit reactions. Raw generated compact sheets belong in `asset-sources/project-starfall/enemies/compact/` as `<enemy-file-id>-compact-source.png` and are processed into final `*-compact-sheet.png` files plus 320px portraits with the compact sheet processor. The processor derives one shared character scale for the entire sheet, preserves a four-pixel safe gutter by constraining effect-heavy render bounds around the stable body anchor, and uses high-quality downsampling so animation frames do not pump or develop nearest-neighbor stair steps. Peripheral VFX that extends outside that stable body window is softly feathered at the safe boundary instead of forcing the character to shrink or leaving a hard-cut edge. Validation also checks all 24 cells for inner-gutter crowding, adjacent near-duplicates, and excessive body framing drift in stable rows. When evaluating a replacement source, use `--enemy <id> --source <path>` to process exactly one candidate without replacing the tracked source image. When a new raw compact source is not available, the processor can only migrate a previous finalized `*-sheet.png` into the compact runtime format; it no longer has a procedural enemy-art refresh path.

Shared prompt direction: one original charming 2D side-scroller MMO monster per sheet, clean dark outline, rounded readable silhouette, soft cel shading, simple expressive animation, consistent proportions and markings across all 24 source cells, right-facing only, no UI/text/watermark, and no copied characters or external game assets. Use the shared `#00ffff` bordered-cell grid and a perfectly flat chroma-key background, usually `#00ff00` and `#ff00ff` for green/plant/ooze subjects, with no shadows, gradients, floor plane, or texture.

### Compact Enemy Sheets

Updated on 2026-05-25 with all current enemies using compact runtime sheets.

Prompt summary: generate each compact enemy as its own clean 3 by 8 sprite sheet on a flat chroma-key background, usually `#00ff00` or `#ff00ff` for green/plant/ooze subjects, with the shared `#00ffff` bordered-cell grid. Rows are exactly idle, walk, wind-up, attack, projectile or ranged action, buff or special action, hit, and defeat; each row has only three frames. Keep one right-facing enemy with the same head size, outfit or markings, weapon, and body proportions in every frame. Center the character in each cell, keep grounded enemies' feet on the same invisible baseline, keep flyers centered consistently, avoid camera zoom changes, avoid cropped limbs, avoid duplicate characters, and keep attack slashes/projectiles separate enough that processing can preserve the body scale. Projectile prompts use separate 3 by 1 bordered sheets with only the projectile on the same chroma-key background. The processor detects the guide borders before cleanup, removes the chroma key and guide lines, clears transparent color data, normalizes final runtime sheets, and validates that every used frame contains visible art without touching the cell edge.

Use the same method for newly generated items or visual variants: isolated asset, flat chroma-key background, visible cyan bordered cells when multiple frames are needed, consistent scale across every cell, no labels or UI, and a processor pass before committing runtime assets. Single-frame inventory icons can be generated as isolated square assets, but animated items, enemy projectiles, and effects should keep the bordered-cell source format so trimming and scaling are repeatable.
