# Project Starfall scenery masters

These are the authoritative scenery sources for the illustrated asset overhaul. All 89 active raster masters were generated with the built-in `image_gen` tool. Exact prompts are in `prompts/`; the world atlas also records its geographic reference. The 44 background paintings comprise 40 location/trial scenes and four purpose-specific shop interiors shared across 24 regional shops. Twelve biome kits supply 40 terrain, 40 prop, and 40 ramp atlases without hue-shifted derivatives; a separate terrain-only interior kit adds one neutral stone/timber floor atlas. Together with two structure atlases, five station objects and the world atlas, the importer owns 173 runtime outputs.

`ledger.json` maps every original active path to its canonical output and records the baseline, source, prompt, reference, and output SHA-256 hashes. It also records reviewed additions, measured extraction rectangles and the existing runtime cell contracts. Existing paths and dimensions remain compatible; the Crossing atlas, four shop panoramas and indoor floor atlas are additive. Reviewed animation sample PNGs are outside this pipeline.

Build every scenery output with:

```powershell
node build/process-project-starfall-overhaul-scenery.js
node build/process-project-starfall-overhaul-scenery.js --validate
node asset-sources/project-starfall/overhaul-v1/scenery/validate-scenery.cjs
```

Optional `--group backgrounds|terrain|props|ramps|structures|stations|world-map` limits the operation. Legacy scenery commands route to this importer while their corresponding ledger groups are enabled, preventing regeneration from old key-color or tinted sources. Normal website builds publish the checked-in runtime outputs.

Source atlases have genuine alpha. Extraction uses measured empty gutters and preserves source colors. Props use a four-pixel cell margin, ramps two, and landmarks eight. Objects retain aspect ratio. Terrain components fill their established square tile contract; their shared body tiles and caps are intentionally reused within each biome kit. Repeatable terrain pieces receive an eight-pixel transition to shared horizontal edge colors in premultiplied alpha, preserving the established seamless tile contract. Character animation frames never pass through this static export treatment. Station objects fit a 320-pixel square with an eight-pixel outer margin.

`measure-layouts.cjs` can recompute source partitions when a master changes, but its results must be visually checked before building. `layout-measurements.json` records source alpha bounds; `validation.json` records all 173 output checks. The check confirms changed baseline outputs, explicitly registered additions, expected groups and dimensions, contained components, and no opaque cell-border contact in transparent props/ramps/landmarks/stations. The Crossing fallback is rebuilt from the accepted atlas and checked against its output hash; its legacy `backups/procedural` directory name does not describe its generated-art provenance.

Backgrounds are complete 2:1 scene paintings. They are not claimed to be edge-periodic textures; a scrolling renderer should use mirrored alternating panorama repeats or a bounded panorama rather than blending unrelated scene edges. The world atlas preserves the original region layout and leaves route/node overlays to the game.

The map consistency pass adds a dedicated Ashglass basalt/volcanic-glass kit and a Crossing landmark atlas selected by `townScene.structureTheme`. Ashglass's continuity revision uses a thin contact bevel and a measured common `terrainContactTop` crop before packing. Its environment profile sets `terrainStyle.contactAligned`; the shared scenery helper places ground and elevated caps at their collision plane in Canvas and Pixi. Export normalization and runtime registration must be reviewed together to avoid sinking actors into a correctly packed stone slab. Cinder's depth revision removes foreground-like background ledges. The prior Ashglass and Cinder masters remain historical references; they do not add to the active-master count. Approved actor studies stay immutable.

Shop panoramas are `shop-weapon-interior`, `shop-armor-interior`, `shop-supply-interior` and `shop-special-interior`. The shop builder selects the room by vendor type and preserves vendor, exit and floor positions. Their explicit indoor environment uses the neutral gray-taupe stone and walnut shop-interior terrain atlas, with no outdoor structures or procedural plant dressing. The approved room panoramas are unchanged by this material correction.

Every new scenery brief must state indoor/outdoor context, room or biome purpose, actual floor and portal positions, and actor clear areas. Keep background shelves, stairs and rock ledges visually behind the playable plane; avoid background edges that promise nonexistent footholds. Repeated terrain must share a level contact edge and compatible material transitions while retaining authored interior variation. Edge-color matching alone does not establish geometric continuity. Validate at gameplay scale in both renderers, including floor/ramp joins, foreground overlap and the real vendor or combat lane. Contextual screenshots remain a separate acceptance gate from export checks.
