# Project Starfall Map Editor Integration Guide

This guide defines a practical, project-specific map editor workflow for Project Starfall. It is written for the current browser prototype and its existing JavaScript data pipeline, Pixi/canvas renderer, platform collision system, slope support, Worldwright admin tooling, and map-design guidance.

Primary project files reviewed:

- `project_starfall_gdd_v0_5.md`
- `MAP_AND_LEVEL_DESIGN_GUIDE.md`
- `docs/project-starfall-map-optimization-audit.md`
- `docs/project-starfall-map-optimization-implementation.md`
- `img/project-starfall/asset-prompts.md`
- `pages/games/project-starfall.html`
- `package.json`
- `test.js`
- `build/validate-project-starfall-maps.js`
- `js/games/project-starfall/project-starfall-data.js`
- `js/games/project-starfall/project-starfall-engine.js`
- `js/games/project-starfall/project-starfall-renderer-pixi.js`
- `js/games/project-starfall/core/geometry.js`
- `js/games/project-starfall/core/storage.js`
- `js/games/project-starfall/engine/map-runtime.js`
- `js/games/project-starfall/engine/movement.js`
- `js/games/project-starfall/engine/viewport.js`
- `js/games/project-starfall/engine/generated-maps.js`
- `js/games/project-starfall/engine/map-analytics.js`
- `js/games/project-starfall/engine/map-mechanics.js`
- `js/games/project-starfall/engine/state.js`
- `js/games/project-starfall/data/index.js`
- `js/games/project-starfall/data/map-content.js`
- `js/games/project-starfall/data/map-catalog.js`
- `js/games/project-starfall/data/map-publication.js`
- `js/games/project-starfall/data/map-presentation.js`
- `js/games/project-starfall/data/map-portals.js`
- `js/games/project-starfall/data/map-assembly.js`
- `js/games/project-starfall/data/map-builders.js`
- `js/games/project-starfall/data/map-geometry.js`
- `js/games/project-starfall/data/world.js`
- `js/games/project-starfall/data/environment.js`
- `js/games/project-starfall/ui/admin-config.js`
- `js/games/project-starfall/ui/input.js`
- `js/games/project-starfall/ui/panels.js`
- `js/games/project-starfall/ui/canvas-viewport.js`
- `img/project-starfall/environment/terrain/*`
- `img/project-starfall/environment/ramps/*`
- `img/project-starfall/environment/props/*`
- `img/project-starfall/maps/*`

Official editor references reviewed:

- Tiled documentation: <https://doc.mapeditor.org/>
- Tiled layers: <https://doc.mapeditor.org/en/stable/manual/layers/>
- Tiled objects: <https://doc.mapeditor.org/en/stable/manual/objects/>
- Tiled custom properties: <https://doc.mapeditor.org/en/stable/manual/custom-properties/>
- Tiled terrain brush: <https://doc.mapeditor.org/en/stable/manual/terrain/>
- Tiled automapping: <https://doc.mapeditor.org/en/stable/manual/automapping/>
- Tiled map format: <https://doc.mapeditor.org/en/stable/reference/tmx-map-format/>
- LDtk layers: <https://ldtk.io/docs/general/editor-components/layers/>
- LDtk IntGrid layers: <https://ldtk.io/docs/general/intgrid-layers/>
- LDtk auto layers: <https://ldtk.io/docs/general/auto-layers/>
- LDtk JSON overview: <https://ldtk.io/docs/game-dev/json-overview/>
- LDtk layer instances: <https://ldtk.io/docs/game-dev/json-overview/layer-instances/>
- LDtk Super Simple export: <https://ldtk.io/docs/game-dev/super-simple-export/>

Assumptions:

- Project Starfall remains a static browser prototype using the current plain JavaScript, Pixi/canvas, localStorage, and data-module architecture.
- The first editor is for the game creator/developer, not for public user-generated content.
- Editable maps should be version-controlled source data, while runtime maps should remain compiled/normalized Starfall map records.
- Existing item pickup, checkpoint, door, switch, moving-platform, and hazard runtime systems are incomplete or map-specific, so the editor should support those object types as source data first and connect runtime behavior in phases.
- Tiled and LDtk are optional future authoring tools; the first source-of-truth format should be Project Starfall JSON because current maps are not external tilemaps.

## 1. Project-specific map system analysis

### Current Map Storage

Project Starfall maps are currently code-defined JavaScript data, not external tilemap files.

The source path is:

1. `js/games/project-starfall/data/map-catalog.js` creates public map records.
2. `js/games/project-starfall/data/map-assembly.js` expands maps with generated town, field, dungeon, and boss geometry.
3. `js/games/project-starfall/data/map-builders.js` creates reusable platform layouts, lane clusters, slope connectors, terrain visuals, spawn sections, and party-play geometry.
4. `js/games/project-starfall/data/map-geometry.js` defines platform helper functions, including slope platform creation and ramp connection generation.
5. `js/games/project-starfall/data/map-publication.js` attaches map assets, environment profiles, route metadata, field compositions, design intent, spawn sections, town service plans, and portals.
6. `js/games/project-starfall/data/map-content.js` composes the map data into `MAPS`.
7. `js/games/project-starfall/data/index.js` exposes `MAPS` through `ProjectStarfallData`.
8. `js/games/project-starfall/project-starfall-data.js` assigns the final data object to `global.ProjectStarfallData`.

The editor must respect this pipeline. The editor should not bypass `map-publication.js`, `map-runtime.js`, or the validation scripts. A custom map should be normalized into the same map record shape that the runtime already consumes.

### Current Level Loading

Maps are loaded by ID through `changeMap()` in `project-starfall-engine.js`. That method:

- Resolves the map from `Data.MAPS`.
- Sets `state.mapId`.
- Rebuilds `this.runtime` using `createMapRuntime(map.id, this.getViewportMetrics())`.
- Clears enemies, projectiles, effects, and map-local runtime objects.
- Places the player at an entry portal or fallback platform.
- Spawns initial enemies.
- Starts map analytics tracking.
- Emits UI and debug updates.

`createMapRuntime()` lives in both the large engine file and the modular file `js/games/project-starfall/engine/map-runtime.js`. The modular helper is the cleaner integration target. It accepts either a map ID or a map object, resolves viewport metrics, creates runtime platforms, generates ramp connections, aligns portals, aligns climbables, aligns spawn points, builds the platform graph, and creates a training-route contract.

The editor should preview maps by calling the same runtime path. A map that cannot survive `createMapRuntime()` should not be exportable.

### Current Tilemap Representation

The game does not currently use `.tmx`, `.tsx`, `.tmj`, `.ldtk`, `.ldtk2`, Unity scenes, Godot scenes, or engine-native tilemaps for Project Starfall. A search of the repo found no external map editor source files for Starfall.

The current "tilemap" layer is effectively split into two systems:

- Gameplay terrain is explicit platform geometry in JS.
- Visual terrain is rendered from biome atlases and procedural strip logic.

The renderer in `project-starfall-renderer-pixi.js` uses:

- Environment terrain atlases from `img/project-starfall/environment/terrain/`.
- Environment prop atlases from `img/project-starfall/environment/props/`.
- Ramp atlases from `img/project-starfall/environment/ramps/`.
- Cell definitions such as `ENVIRONMENT_TERRAIN_CELLS` and `ENVIRONMENT_PROP_CELLS`.
- `terrainVisual` metadata on platform records.
- Field compositions, town structures, landmarks, and generated scenery placements.

That means an editor should expose tile-like painting for designers, but export the authoritative gameplay surface as platforms, slopes, climbables, portals, spawns, stations, NPCs, and map metadata.

### Current Tileset Organization

Tilesets are organized as generated image atlases rather than external editor tileset files.

Important asset groups:

- Terrain atlases: `img/project-starfall/environment/terrain/*.png`
- Ramp atlases: `img/project-starfall/environment/ramps/*.png`
- Prop atlases: `img/project-starfall/environment/props/*.png`
- Structure/landmark atlases: configured through environment content modules
- Map backdrop images: `img/project-starfall/maps/*.png`
- Source map images: `img/project-starfall/maps/source/*`

Important data files:

- `js/games/project-starfall/data/environment.js`
- `js/games/project-starfall/data/environment-content.js`
- `js/games/project-starfall/data/assets.js`
- `js/games/project-starfall/data/map-presentation.js`

Current environment atlases use cell metadata:

- Terrain cell size is generally `64`.
- Prop cell size is generally `64`.
- Ramp cell size is generally `128`.
- Terrain cells include `groundLeft`, `groundMid`, `groundRight`, `platformLeft`, `platformMid`, `platformRight`, `body`, `bodyDeep`, `underside`, `left`, `right`, `cap`, `detail`, `shadow`, `top`, and `topAlt`.
- Prop cells include `grass`, `bush`, `tree`, `rock`, `flower`, `small`, `tall`, `crate`, `crystal`, `vine`, `sign`, and `glow`.

The editor should not invent unrelated tile IDs. It should load its palette from Starfall's existing environment assets and cell definitions.

### Current Collision Handling

Collision is platform based:

- Flat platforms are arrays like `[x, y, w, h]` or objects with `x`, `y`, `w`, and `h`.
- Sloped platforms are objects with `shape: "slope"` and `y2`.
- `core/geometry.js` provides `isSlopePlatform()` and `getPlatformSurfaceY()`.
- Movement, enemy placement, spawn alignment, fall recovery, and runtime pathing all sample platform surfaces.
- `map-runtime.js` builds runtime platform objects with IDs, indexes, drop-through behavior, platform graph links, ramp links, and footholds.

The editor must treat collision as a first-class authored layer, not as an automatic side effect of art. Visual tiles may suggest collision, but explicit collision shapes are the source of truth.

### Current Slope Handling

Slopes are gameplay platforms:

- Source helper: `makeSlopePlatformDef(x, y, y2, w, h, visual)` in `map-geometry.js`.
- Runtime shape: `{ shape: "slope", x, y, y2, w, h }`.
- Surface sampling: linear interpolation from `y` to `y2`.
- Runtime ramp links: `makeRampConnections()` in map data and `createRuntimeRampConnections()` in runtime.
- Rendering: `drawRampPlatformTerrain()` currently routes slope visuals through fallback platform drawing. Ramp atlas assets exist, but the renderer path still relies on fallback slope geometry.
- Validation: `build/validate-project-starfall-maps.js` enforces slope budgets, max grade, ramp connections, no spawns on slopes, and enough broad flat platforms.

This is a critical editor requirement: every edited slope must be both visual and collision-correct, connected to adjacent platforms, and validated against the slope-density rules in `MAP_AND_LEVEL_DESIGN_GUIDE.md`.

### Current Enemies, Items, Hazards, Checkpoints, Spawns, And Exits

Enemies:

- Map records include enemy pools such as `enemies: [...]`.
- Combat maps include `spawnPoints`, each with `x`, `platformIndex`, `weight`, and now section metadata in the optimized maps.
- Some admin/test maps include `fixedEnemySpawns`.
- Engine spawning resolves spawn points into runtime enemies, tracks spawn sections, and uses section-aware respawn scoring.
- Admin commands can spawn enemies through the Worldwright Console.

Items and pickups:

- Project Starfall currently emphasizes loot drops, inventory grants, shop items, consumables, materials, and equipment.
- There is no mature hand-authored per-map pickup layer comparable to `spawnPoints`.
- The editor should add a future `items` or `pickups` object layer and export it separately from enemy drops.
- Early implementation may support editor pickups as metadata and keep them disabled until runtime pickup spawning is implemented.

Hazards:

- Hazards are mostly runtime mechanics, boss mechanics, projectiles, effects, map mechanic definitions, and arena section calls.
- `map-mechanics.js` tracks live map mechanics for Oreback Quarry, Stormbreak Cliffs, Endless Rift, and boss/dungeon spatial calls.
- `map-presentation.js` defines `MAP_MECHANIC_DEFINITIONS`.
- There is no universal authored hazard collision layer yet.
- The editor should introduce hazard objects as typed triggers that feed map mechanics, boss mechanics, or future hazard runtime code.

Checkpoints:

- There is no explicit checkpoint placement layer in the current map data.
- Current map transitions, respawn, save/load, and fall recovery rely on map IDs, platform placement, portals, and runtime recovery.
- The editor should support checkpoint objects as future-ready data. In the first implementation, checkpoints can validate and preview but should not affect runtime until checkpoint state is added.

Spawns:

- Player map entry is controlled by portals and fallback platform placement.
- Enemy spawns are `spawnPoints`.
- Fall recovery uses nearest viable platform logic.
- Class trials and benchmark maps create their own temporary generated map records in `generated-maps.js`.

Exits and transitions:

- Route transitions are portals in `map-portals.js`.
- Runtime portals are aligned to platforms in `map-runtime.js`.
- World routes and world-map edges live in `world.js`.
- `changeMap()` drives actual transitions.

The editor must treat exits as portal objects that can export either into `MAP_PORTALS` or into a map-local `portals` array before publication.

### Current Camera Bounds And Viewport

Viewport constants:

- `PLAYFIELD_WIDTH = 1280`
- `PLAYFIELD_HEIGHT = 640`
- `DEFAULT_WORLD_ZOOM = 1.32`
- Canvas view width is `1280`.
- Canvas playfield height is `640`.
- The status HUD sits below the playfield.

Runtime world bounds come from map geometry:

- `worldWidth` is derived from authored platforms, map width, stations, spawns, and map sizing rules.
- `worldHeight` is derived from viewport height, authored world height, platform bottoms, and solid platform height.
- The camera follows the player and is snapped on teleport/map change.

The editor should provide camera-safe overlays:

- 1280x640 playfield frame.
- Current zoom/world-view frame.
- Spawn camera frame.
- Portal/exit camera frame.
- Vertical movement warning frame for long slope chains or climb shafts.

### Current Map Type

Project Starfall maps are:

- Data-driven.
- Code-defined.
- Generated and augmented through JS modules.
- Platform-geometry based.
- Not currently JSON-authored.
- Not currently external tilemap based.
- Not engine-native scene based.

The correct editor output is not raw Tiled/LDtk data as the runtime source of truth. The correct output is Starfall map JSON that compiles or normalizes into the current JS map record shape.

### What Map Customization Is Currently Easy

Current easy customization for a programmer:

- Change map names, purpose text, palettes, route metadata, and enemy pools in data modules.
- Adjust platform arrays or builder parameters.
- Add spawn points if platform indexes are known.
- Add portals if platform positions are understood.
- Adjust world-route metadata.
- Add design intent, spawn sections, and map mechanic metadata.
- Run `npm run validate:starfall:maps` to catch slope and spawn issues.
- Use Worldwright Console commands to teleport, spawn enemies, kill enemies, grant items, change rates, and benchmark runtime behavior.

### What Map Customization Is Currently Painful

Current painful customization:

- Editing maps requires raw JS edits across multiple files.
- Platform indexes are fragile for spawns, portals, stations, and climbables.
- Slopes require manual `y2` math and ramp connection correctness.
- Visual terrain and collision are not edited together in a visual interface.
- There is no drag/drop object placement.
- There is no map preview before changing code.
- There is no in-browser save/export workflow for maps.
- There is no object property panel for enemy spawns, portals, hazards, or checkpoints.
- There is no non-programmer-friendly palette.
- There is no selectable map template library in the UI.
- There is no single JSON file per map.
- Item pickups, hazards, and checkpoints are not mature map object types yet.

### What Must Change To Support An Editor Workflow

Required changes:

1. Add a versioned Starfall editor map schema.
2. Store editable maps as JSON files instead of hand-editing map JS.
3. Add a loader/compiler that converts editor JSON into the existing runtime map record shape.
4. Add validation shared by the editor UI and Node scripts.
5. Add platform IDs so objects can attach to stable platform references instead of only numeric indexes.
6. Add a developer-only editor UI, preferably under the existing Worldwright/admin surface.
7. Add import/export/download support for editor JSON.
8. Add preview/playtest mode that uses `createMapRuntime()`.
9. Add build/test scripts for editor maps.
10. Add documentation for map authors, map reviewers, and asset authors.

## 2. Best-practice research for map editors

### Research Summary

Official Tiled and LDtk documentation supports several practices that are directly useful for Project Starfall:

- Use explicit layers for visual tiles, object placement, background images, foreground images, collision, and editor notes.
- Keep object/entity placement separate from tile painting.
- Use custom properties or fields for gameplay metadata.
- Support terrain brushes, stamps, random variation, and automated tile rules to reduce repetitive hand painting.
- Use object templates or entity definitions for repeated gameplay objects.
- Store maps in version-control-friendly formats.
- Validate exports instead of assuming every visual layout is playable.
- Treat JSON exports as a stable interchange format only when the runtime has a clear parser.
- Support layer locking to prevent accidental edits to system layers.
- Support preview/playtest loops close to the runtime.

Tiled is strong at flexible tile maps, object layers, image layers, custom properties, terrain brushes, automapping, tile collision editing, and TMX/JSON export. LDtk is strong at layer definitions, entity fields, IntGrid layers, auto layers, and clean JSON-oriented workflows.

The key lesson for Project Starfall: adopt the workflow principles, not necessarily the raw file format. Starfall is not currently a grid-first tilemap game. It is a platform-geometry game with atlas-driven visual terrain.

### Tile Palette Workflow

Best practice:

- Designers should choose tiles or cells from a palette grouped by biome and layer.
- Palettes should display only valid cells for the active layer.
- Tiles should have readable names, not only numeric IDs.
- Brush tools should support single cell, strip, stamp, and random variation modes.

Project Starfall requirement:

- Build palettes from `ENVIRONMENT_TERRAIN_CELLS`, `ENVIRONMENT_PROP_CELLS`, and `ENVIRONMENT_ASSETS`.
- Group palettes by biome/theme: `greenroot`, `rustcoil`, `cinder`, `frostfen`, `stormbreak`, `astral`, `eclipse`, `rift`, and `town`.
- Hide prop cells when editing collision.
- Hide collision shapes when editing decorative layers unless collision overlay is enabled.

### Layer-Based Editing

Best practice:

- Separate background, terrain visuals, collision, objects, foreground, and editor notes.
- Use layer order to match render order.
- Allow visibility toggles and locking.
- Make gameplay-critical layers harder to accidentally damage.

Project Starfall requirement:

- Collision layers must be explicit and lockable.
- Portal, spawn, and station layers must be object layers.
- Field composition, spawn sections, and map mechanic regions should be visible overlays.
- Visual prop layers must not affect collision unless a deliberate collider object exists.

### Collision Editing

Best practice:

- Collision should be visible as a clean overlay.
- Collision should be editable independently from art.
- The tool should warn when collision does not match visual terrain.
- Collision should use simple shapes where possible.

Project Starfall requirement:

- Collision is not tile collision. It is runtime platform geometry.
- Collision editing must create flat platforms, slope platforms, climbables, drop-through platforms, and future hazard volumes.
- The editor should show the sampled surface line for every platform.
- The editor should show player-height clearance and broad flat combat zones.

### Slope Editing

Best practice:

- Slopes need constrained angles, clean transitions, and visible collision previews.
- Slope collision should not be inferred from noisy art.
- Slopes should be treated as authored connectors, not random terrain.

Project Starfall requirement:

- Slopes must be explicit platform objects with `shape: "slope"` and `y2`.
- The editor must warn when slope density violates `build/validate-project-starfall-maps.js`.
- The editor must display the slope grade and endpoint heights.
- The editor must prevent enemy/player spawn placement on slope platforms.
- The editor must require flat pads before and after slopes.

### Object And Entity Placement

Best practice:

- Use object layers for spawns, enemies, pickups, NPCs, hazards, triggers, checkpoints, and exits.
- Use property panels for object metadata.
- Use templates or presets for common objects.
- Use icons and labels, not raw coordinate tables.

Project Starfall requirement:

- Object placement should write stable IDs and platform references.
- Enemy spawn objects should resolve to existing `spawnPoints`.
- Portal objects should resolve to `MAP_PORTALS` or map-local portals.
- NPC objects should resolve to `questNpcs`.
- Station objects should resolve to `stations`.
- Checkpoint, hazard, and pickup objects should be included in editor JSON even before full runtime support is implemented.

### Enemy Placement

Best practice:

- Let designers place spawn regions and encounter presets, not only individual enemy instances.
- Show difficulty, density, patrol/leash area, section, and respawn behavior.
- Validate reaction time and safe spawn distances.

Project Starfall requirement:

- Use `spawnPoints` plus map enemy pools for standard fields.
- Support `fixedEnemySpawns` for labs, set pieces, and tutorials.
- Attach `sectionId` and `sectionLabel` to spawn points.
- Show spawn pressure by section for maps with live mechanics.
- Warn if spawns cluster too tightly or sit on slopes.

### Item Placement

Best practice:

- Place pickups as objects with item IDs, quantity, visibility, respawn state, and conditions.
- Validate that items are reachable and visible.
- Keep rewards on stable, readable surfaces.

Project Starfall requirement:

- Add a future `pickups` object layer.
- Use item IDs from current item/material/consumable/equipment data.
- Mark early pickups as `editorOnly` or `runtimeDisabled` until runtime pickup support exists.
- Avoid placing pickups on slopes unless tested.

### Trigger Placement

Best practice:

- Triggers should be rectangular or polygon regions with visible bounds.
- Triggers should have clear names, event types, and target IDs.
- Triggers should validate their target references.

Project Starfall requirement:

- Add trigger objects for map mechanics, boss spatial calls, tutorial prompts, checkpoint activation, hazard toggles, and event sections.
- Use `sectionId` to tie triggers to spawn sections and mechanics.
- Do not tie triggers to visual prop placement.

### Spawn, Checkpoint, And Exit Placement

Best practice:

- Spawn/checkpoint/exit objects should have icons, facing direction, platform attachment, and validation.
- Entry camera frame should be previewable.
- Exits should be labeled by destination.

Project Starfall requirement:

- Player spawn is currently portal/fallback driven. The editor should still add `playerStart` for new maps and use portals for normal route flow.
- Checkpoints need a future runtime state, but the editor should support them now.
- Exits must export destination map ID, route ID, dungeon ID, required map ID, label, platform anchor, and return/boss flags.

### Background And Parallax Editing

Best practice:

- Background layers should be separate from gameplay terrain.
- Parallax factors should be visible and testable.
- Background clutter should not compete with gameplay.

Project Starfall requirement:

- Background editing should choose existing map backdrops and environment profiles.
- Parallax parameters should remain simple at first: far image, mid image, color palette, opacity, and horizontal repeat.
- Gameplay collision must remain readable against backgrounds.

### Undo, Redo, Selection, And Copy/Paste

Best practice:

- Every edit should be undoable.
- Dragging, moving, resizing, deleting, property changes, layer changes, and paste operations must be undoable.
- Multi-select is essential for moving platform clusters and encounter groups.

Project Starfall requirement:

- The editor must support at least 100 undo steps in memory.
- Undo records should be operation based, not full-map snapshots for every mouse move.
- Platform clusters should be movable with attached objects when selected as a group.

### Grid Snapping

Best practice:

- Support snap-to-grid, snap-to-object, and free placement.
- Let designers temporarily disable snapping.
- Use grid sizes that match asset and gameplay units.

Project Starfall requirement:

- Default grid: `64` px for terrain atlas cells.
- Fine grid: `16` px for object placement and platform endpoints.
- Slope endpoints should snap to `16` px vertical increments by default.
- Major camera/section grid: `1280` px horizontal frames.
- Spawn and portal anchors should snap to platform surfaces.

### Validation Warnings

Best practice:

- Validation should run continuously and before export.
- Errors should block export.
- Warnings should be visible but not always blocking.
- Clicking a validation message should focus the relevant object.

Project Starfall requirement:

- The editor should reuse or mirror `build/validate-project-starfall-maps.js`.
- Errors should block saving to production map manifests.
- Warnings should allow draft saves.
- The validator should understand slope budgets, platform reachability, spawn validity, portal validity, missing assets, missing metadata, and camera framing.

### Preview And Playtest Mode

Best practice:

- Designers need a fast switch between edit and play.
- Preview should use real runtime movement and collision.
- Playtest should allow starting from selected spawn/checkpoint.

Project Starfall requirement:

- Preview must call `createMapRuntime()` and render through the existing Pixi/canvas path.
- Playtest must use the real player controller in `movement.js`.
- Add one-click test from player start, selected checkpoint, selected portal, and selected spawn section.
- Editor playtest should not corrupt normal player save data.

### Saving, Loading, Exporting, And Versioning

Best practice:

- Use versioned data.
- Keep source maps human-readable.
- Support import/export/download.
- Keep generated runtime output separate from editable source.

Project Starfall requirement:

- Store editable source maps as JSON.
- Use localStorage autosaves for browser editing.
- Use download/import for portable files.
- Use a build step to compile JSON into a JS manifest.
- Keep schema versions and migration helpers.
- Do not store editor maps inside the normal player save payload unless they are local custom maps.

### Preventing Invalid Map States

Best practice:

- Use constrained tools and object templates.
- Validate references on edit.
- Lock system layers.
- Limit dangerous operations such as deleting a platform with attached spawns.

Project Starfall requirement:

- Deleting a platform should either delete attached objects or reassign them through a confirmation.
- A portal cannot save without a destination.
- A spawn cannot attach to a slope.
- A slope cannot save without adjacent flat pads or an approved exception.
- A production map cannot save without map metadata, environment profile, at least one player start or entry portal, and runtime-valid collision.

### Most Important Practices For Starfall

The most important practices for this project are:

1. Runtime preview through `createMapRuntime()`, because collision/pathing is custom.
2. Explicit platform collision editing, because visuals do not define gameplay surfaces.
3. Slope validation, because slope overuse is already a known map-quality problem.
4. Stable platform IDs, because platform indexes are fragile.
5. Object property panels, because enemies, portals, stations, and sections carry game-specific metadata.
6. Versioned JSON source maps, because raw JS map edits are the main customization barrier.
7. Layer locking and validation, because accidental edits to portals/spawns/collision can break progression.

## 3. Recommended editor approach

### Option A: Build An In-Game Map Editor

Description:

Create a map editor directly inside the Project Starfall browser page. It would draw editor UI over the real game canvas and edit maps with Starfall-specific tools.

Pros:

- Uses the real renderer, camera, runtime, movement, and collision.
- Immediate preview/playtest loop.
- Can reuse Worldwright admin access.
- Can validate with current game data.
- Best possible alignment with custom slopes, portals, spawn sections, and map mechanics.

Cons:

- More UI code inside the game.
- Needs careful separation from normal player save data.
- Browser file writing requires import/export/download or a dev-only save endpoint.
- More code to maintain than a pure external tool workflow.

Complexity:

- Medium for a developer/debug editor.
- High only if trying to make it a polished public creator.

Risk:

- Moderate if built inside existing game UI without boundaries.
- Low if kept developer-only, feature-gated, and source maps remain JSON.

Fit for this project:

- Excellent. Starfall's map system is custom and runtime-dependent.

Required file/data changes:

- Add editor map schema and loader modules.
- Add editor UI modules.
- Add editor map validator.
- Add localStorage draft storage and import/export.
- Add optional build script for JSON maps.

Long-term maintainability:

- Good if the editor is built as small modules and talks to runtime through stable map JSON, not engine internals.

### Option B: Build A Debug/Developer Editor Mode

Description:

Add editor mode as an extension of the existing Worldwright/admin tooling. Access it with the Worldwright Console item, a query parameter, or a dev flag.

Pros:

- Reuses existing admin mental model.
- Keeps the tool out of normal play.
- Can integrate with current admin teleport/spawn/debug log workflows.
- Smaller first implementation than a full standalone editor.
- Can be built iteratively.

Cons:

- UI may feel dense if squeezed into the current admin panel.
- Needs a larger editor canvas or panel layout to be comfortable.
- Still needs import/export and validation.

Complexity:

- Low to medium.

Risk:

- Low if editor state is isolated from player state.

Fit for this project:

- Best first implementation.

Required file/data changes:

- Add `mapEditor` panel or a new Worldwright Console tab.
- Add editor modules under `js/games/project-starfall/ui/`.
- Add editor state separate from player save state.
- Add schema/validator modules under `js/games/project-starfall/data/` or `engine/`.

Long-term maintainability:

- Very good if kept modular and developer-only.

### Option B2: Extend Existing Engine Editor Tools

Description:

Extend the current Worldwright Console, admin settings panel, debug log, map analytics, performance debug, and admin teleport/spawn commands into a dedicated map-creation workflow.

Pros:

- Builds on real tools already present in `project-starfall-engine.js`, `ui/admin-config.js`, `ui/input.js`, and `ui/panels.js`.
- Keeps map creation aligned with existing admin permission logic and debug output.
- Reuses current commands for teleporting, spawning enemies, clearing runtime objects, and collecting performance/debug information.
- Gives the editor a clear home inside the game's current tool-like RPG UI.

Cons:

- Current Worldwright tools are command/debug oriented, not visual editing tools.
- The existing admin console should not become the entire editor UI; it needs a larger canvas-oriented editing surface.
- Existing tools do not solve source-map serialization, layer editing, object property forms, or validation by themselves.

Complexity:

- Low for adding editor launch commands and debug hooks.
- Medium for turning those hooks into a full visual editor panel.

Risk:

- Low if editor state is isolated from gameplay state.
- Medium if editor commands mutate live player state during map editing.

Fit for this project:

- Excellent as the integration shell. This is the best way to make the editor feel native without building a separate app.

Required file/data changes:

- Add a Worldwright `Map Editor` tab or panel.
- Add editor command aliases such as `map editor`, `map clone`, `map validate`, and `map playtest`.
- Add editor state outside normal player save data.
- Reuse debug-log output for validation and playtest reports.

Long-term maintainability:

- Good if the visual editor is modular and the Worldwright Console only launches/controls it.

### Option C: Use Tiled

Description:

Use Tiled as an external editor and import/export maps into Starfall.

Pros:

- Mature tile palette, layer, object, terrain, automapping, custom property, and collision workflows.
- Designers get a familiar 2D map editor.
- Good object layers for portals, spawns, hazards, and notes.
- Good tile painting for visual mockups.

Cons:

- Starfall is not currently a Tiled runtime.
- Tiled tile collision does not match Starfall's platform/slope runtime directly.
- Requires an importer/exporter.
- Requires a Starfall-specific Tiled project template.
- Designers can create visually valid maps that are runtime-invalid without strong validation.
- Slopes would still need custom object templates or polygon/line interpretation.

Complexity:

- Medium to high.

Risk:

- Medium. Import conversion errors could be subtle.

Fit for this project:

- Good as a mockup/interchange tool later, not as the first source of truth.

Required file/data changes:

- Add `.tiled-project`, `.tmx` or `.tmj` map files, `.tsx` tilesets, and a conversion script.
- Add Starfall Tiled object templates for platforms, slopes, portals, spawns, hazards, and NPCs.
- Add import validation.

Long-term maintainability:

- Good if the importer is stable and editor rules are strict.
- Poor if maps are edited in both Tiled and Starfall JSON without a single source of truth.

### Option D: Use LDtk

Description:

Use LDtk as an external editor with JSON export, layers, entities, fields, IntGrid layers, and auto layers.

Pros:

- Strong entity-field model.
- JSON-oriented workflow.
- IntGrid and auto layers are useful for collision intent and visual rule layers.
- Good for level metadata and editor validation.
- Better fit than pure tile editors when objects and fields matter.

Cons:

- Starfall still needs a parser/converter.
- Current Starfall terrain is platform geometry, not grid cells.
- LDtk auto layers do not directly create Starfall platform/ramp geometry.
- Slope editing still requires custom entities.

Complexity:

- Medium.

Risk:

- Medium.

Fit for this project:

- Good future option if you want a strong external editor. Not the simplest first implementation.

Required file/data changes:

- Add LDtk project file, entity definitions, enums, and import script.
- Define `Platform`, `Slope`, `SpawnPoint`, `Portal`, `Climbable`, `Hazard`, `Checkpoint`, `Station`, and `NPC` entities.
- Convert LDtk layers to Starfall map JSON.

Long-term maintainability:

- Good if LDtk JSON is treated as source of truth.
- Risky if mixed with handwritten JS maps.

### Option E: Use An Engine-Native Tilemap Editor

Description:

Move Starfall maps to an engine such as Godot, Unity, or another native tilemap environment.

Pros:

- Mature editor tooling if the game moves engines.
- Engine-native collision, scene objects, prefabs, and debugging.

Cons:

- Project Starfall is a static browser prototype.
- Current architecture is plain JS, Pixi/canvas, and custom data modules.
- This would be a major engine migration, not a map editor integration.

Complexity:

- Very high.

Risk:

- Very high.

Fit for this project:

- Poor for the current prototype.

Required file/data changes:

- Full map/runtime/tooling migration.

Long-term maintainability:

- Only reasonable if the entire game is ported.

### Option F: Create A Custom Standalone Editor

Description:

Build a separate local web app or desktop tool that edits Starfall maps.

Pros:

- Clean separation from game runtime UI.
- Can be focused entirely on editing.
- Can write files more directly if backed by a local dev server.

Cons:

- Duplicates renderer, camera, asset loading, and runtime preview unless embedded.
- Larger setup.
- More maintenance.
- Slower feedback loop than in-game preview.

Complexity:

- High.

Risk:

- Medium to high.

Fit for this project:

- Not justified for the first implementation.

Required file/data changes:

- Standalone app shell, asset loader, preview renderer, serializer, validator, and dev server endpoint.

Long-term maintainability:

- Good only if map creation becomes a large separate product.

### Option G: Simple JSON/YAML Editing Workflow

Description:

Write maps in JSON or YAML manually and compile them into runtime data.

Pros:

- Simple implementation.
- Easy to validate.
- Easy to version control.
- Good stepping stone.

Cons:

- Does not meet the goal of easy customization without raw file edits.
- Still requires coordinate/platform knowledge.
- Hard to visualize slopes, collision, portals, and camera.

Complexity:

- Low.

Risk:

- Low.

Fit for this project:

- Good as the data foundation, not enough as the whole editor.

Required file/data changes:

- Add schema, example maps, loader, compiler, and validator.

Long-term maintainability:

- Excellent as the source format behind a UI.

### Option H: Hybrid External Editor Plus In-Game Validation/Preview

Description:

Use Tiled or LDtk for initial layout editing, convert to Starfall JSON, then validate and playtest in-game.

Pros:

- Combines mature external editing with Starfall runtime correctness.
- Good long-term path if map volume grows.
- Lets designers use tile/object workflows.

Cons:

- Requires import/export tooling.
- Requires strict conventions.
- Adds another app to the workflow.
- Still needs a Starfall preview layer.

Complexity:

- Medium to high.

Risk:

- Medium.

Fit for this project:

- Good second phase, not first phase.

Required file/data changes:

- Same Starfall JSON schema plus importers for Tiled/LDtk.

Long-term maintainability:

- Good if Starfall JSON remains the normalized handoff format.

### Recommended Approach

Use a phased hybrid with a project-native source format:

1. **Phase 1: Starfall editor JSON schema plus validator.**
   - Store editable maps as versioned JSON.
   - Add a loader/compiler to produce current Starfall map records.
   - Add Node validation and test coverage.

2. **Phase 2: Developer-only in-game editor mode inside Worldwright.**
   - Load, edit, validate, preview, and playtest Starfall JSON maps.
   - Use the current renderer/runtime for preview.
   - Use import/export/download/localStorage for map drafts.

3. **Phase 3: Optional Tiled/LDtk importers.**
   - Add external editor support only after the Starfall schema is stable.
   - Treat external editor files as authoring convenience, not the runtime format.

This is the simplest reliable workflow for the current project because it avoids a full engine migration, avoids forcing a grid tilemap onto a platform-geometry game, and still gives you a visual editor for map customization.

## 4. Required editor features

### Must-Have Features

Create new map:

- Start from a template: town hub, starter field, training field, deep field, dungeon, boss arena, endless field, class trial, or admin lab.
- Auto-fill required metadata from existing `MAP_LAYOUT_ROLES`.
- Assign a unique map ID.

Load existing map:

- Load editor JSON maps from `content/project-starfall/maps/`.
- Import a downloaded JSON file.
- Load a runtime map from `Data.MAPS` as a read-only reference or clone it into editable JSON.

Save map:

- Autosave drafts to localStorage under a separate editor key.
- Export JSON as a downloadable file.
- In local dev, optionally support a dev-only endpoint for writing to `content/project-starfall/maps/`.
- Never write editor drafts into `projectStarfallPrototypeSave.v1`.

Edit tile terrain:

- Provide a biome terrain palette using existing environment terrain cells.
- Support painting visual terrain strips, platform visual styles, decorative cells, and foreground trim.
- Keep visual terrain separate from collision.

Edit slopes:

- Add, drag, shorten, flip, and delete slope platforms.
- Show grade, endpoints, and sampled collision line.
- Warn on excessive slope density.
- Require flat pads at both ends.

Edit collision:

- Add flat platforms.
- Resize platforms.
- Convert flat platforms to slope platforms and back.
- Mark platform role: ground, combat lane, optional ledge, one-way/drop-through, connector, boss lane, service floor, or editor note.
- Show player height/clearance overlay.

Place player spawn:

- Add a `playerStart` object for new maps.
- Show facing direction and initial camera frame.
- Validate that it rests on a flat platform.

Place exits/transitions:

- Add portal objects with label, destination map, destination route, dungeon ID, required map ID, portal role, and platform anchor.
- Support return portals and boss portals.
- Validate destination map IDs against `WORLD_MAP_NODES`, `WORLD_MAP_EDGES`, and `MAPS`.

Place enemies:

- Add spawn points and fixed spawns.
- Choose enemy pool entries from `Data.ENEMIES`.
- Assign spawn section, weight, platform, and spawn role.
- Preview spawn density and section labels.

Place items/pickups:

- Add future-ready pickup objects with item ID, quantity, rarity, respawn behavior, and visibility.
- Validate item IDs against current item, material, equipment, and consumable data.
- Mark runtime-disabled pickups until pickup runtime support is added.

Place hazards:

- Add hazard regions and hazard emitters.
- Choose hazard type: lava, spike, wind, falling debris, rift surge, frost slip, boss call, projectile emitter, timed gate, or custom mechanic hook.
- Validate collision/trigger bounds.

Place checkpoints:

- Add checkpoint objects with ID, label, platform anchor, facing direction, and camera frame.
- Mark as future runtime until checkpoint state is implemented.
- Validate rest-space requirements from `MAP_AND_LEVEL_DESIGN_GUIDE.md`.

Set map bounds:

- Edit width, height, authored ground Y, and camera-safe bounds.
- Show 1280 px section divisions.
- Warn if world height creates excessive vertical camera travel without rest zones.

Validate map:

- Run validation continuously.
- Display errors and warnings by object/layer.
- Block production export on errors.

Preview/playtest map:

- Preview with current Pixi renderer.
- Playtest from player start, selected checkpoint, selected portal, or selected spawn section.
- Reset playtest without affecting normal save state.

### Should-Have Features

Undo/redo:

- At least 100 steps.
- Covers object movement, platform resizing, slope edits, property changes, layer edits, and deletes.

Tile palette:

- Biome-filtered terrain, ramp, prop, and structure cells.
- Search by cell name.
- Favorites for frequently used cells.

Object palette:

- Player start, checkpoint, portal, enemy spawn, fixed enemy, pickup, hazard, trigger, climbable, station, NPC, note, spawn section, mechanic region.

Brush size:

- Single tile/cell.
- Strip brush.
- Area fill for visual decoration only.
- Object placement is one-at-a-time unless using templates.

Eraser:

- Active-layer eraser.
- Selected-object delete.
- Protected system layers require confirmation.

Selection/move/delete:

- Drag objects.
- Resize platforms.
- Multi-select objects.
- Group move with attached platform objects.

Copy/paste:

- Copy platform clusters with attached spawns, props, and notes.
- Pasted objects receive new IDs.
- Pasted portals require destination review.

Layer visibility toggles:

- Show/hide background, terrain visual, collision, slopes, objects, hazards, foreground, camera, validation, and notes.

Entity property editor:

- Form-based properties with dropdowns from game data.
- No raw JSON editing required for normal use.

Map metadata editor:

- Map ID, name, role, level range, biome, route, purpose, design intent, party role target, farming-abuse risk, palette, environment profile, music, ambience, and notes.

Background/parallax editing:

- Select existing backdrop asset.
- Set palette colors.
- Set parallax strength.
- Preview contrast against gameplay terrain.

Basic map templates:

- Safe town hub.
- Starter field.
- Three-lane training field.
- Switchback terrace.
- Vertical canopy/climb.
- Dungeon arena.
- Boss echo.
- Endless quadrant field.
- Class trial.

### Nice-To-Have Features

Auto-tiling:

- Auto-pick visual terrain cells based on platform type and neighboring shapes.

Slope brush:

- Drag from low point to high point.
- Snap to approved grades.
- Auto-create flat pads if needed.

Rule tiles:

- Place terrain details based on biome, platform role, and surrounding space.

Smart terrain painting:

- Convert rough blockout platforms into terrain visuals.
- Suggest cliff, ledge, bridge, or ramp visuals based on shape.

Enemy encounter templates:

- Greenroot starter pull.
- Bandit high-low pressure.
- Oreback scaffold farm.
- Cinder flyer pressure.
- Frostfen brute/oracle pairing.
- Stormbreak ranged/airspace pressure.
- Astral elite/rift pocket.

Item placement presets:

- Visible reward ledge.
- Post-combat recovery pickup.
- Optional route prize.
- Tutorial pickup.

Heatmap/validation overlays:

- Slope density.
- Broad flat combat zones.
- Spawn density.
- Platform reachability.
- Camera bob risk.
- Unused space.
- Reward visibility.

One-click test from spawn:

- Right-click any spawn/checkpoint/portal and choose playtest from here.

Screenshot/export preview:

- Export a full-map image.
- Export first-screen preview.
- Export section previews.

Map version history:

- Keep local autosave history.
- Store `lastValidatedAt`, `lastExportedAt`, and `previousVersionId`.

## 5. Map data model and file format

### Recommended Format

Use versioned JSON as the Starfall map editor source format.

Reason:

- The project is already JavaScript/data driven.
- JSON is easy to validate, diff, version, import, and export.
- JSON can compile into existing map records.
- JSON is easier for future Tiled/LDtk importers to target.
- JSON avoids adding YAML parsing or external dependencies.

Recommended source directory:

- Editable map source: `content/project-starfall/maps/*.json`
- Map templates: `content/project-starfall/map-templates/*.json`
- Editor tileset metadata, if needed: `content/project-starfall/tilesets/*.json`
- Generated runtime manifest: `js/games/project-starfall/data/editor-map-manifest.js`

Do not store editor source maps directly in `js/games/project-starfall/data/map-catalog.js`. The purpose of the editor is to stop hand-authoring raw map JS.

### Map Schema Requirements

Every editor map must include:

- `schema`: fixed string such as `project-starfall-map-v1`.
- `version`: integer.
- `id`: camelCase runtime map ID.
- `fileSlug`: kebab-case file slug.
- `name`: display name.
- `role`: one of `town`, `starterField`, `trainingField`, `deepField`, `dungeon`, `bossArena`, `endlessField`, or `trial`.
- `levelRange`: two integers.
- `biome`: world/visual biome ID.
- `routeId`: optional world route ID.
- `width`: authored world width.
- `height`: authored world height.
- `tileSize`: normally `64`.
- `authoredGroundY`: baseline ground Y.
- `palette`: color array compatible with current maps.
- `environment`: profile IDs and asset references.
- `layers`: ordered layer list.
- `platforms`: explicit collision platforms.
- `climbables`: climbable objects.
- `spawns`: player/enemy/fixed spawn data.
- `portals`: exits/transitions.
- `stations`: town service objects.
- `npcs`: quest/NPC objects.
- `pickups`: future item/pickup objects.
- `hazards`: future hazard objects.
- `checkpoints`: future checkpoint objects.
- `triggers`: tutorial/mechanic/section triggers.
- `sections`: route/spawn/mechanic sections.
- `camera`: authored camera metadata.
- `metadata`: editor and design metadata.

### Platform Data

Use stable IDs for all platforms. Numeric platform indexes can be generated during export.

Flat platform:

```json
{
  "id": "ground_main_01",
  "type": "platform",
  "shape": "flat",
  "role": "main_ground",
  "x": 0,
  "y": 520,
  "w": 2200,
  "h": 80,
  "dropThrough": false,
  "visual": {
    "kind": "ground",
    "theme": "greenroot-meadow"
  }
}
```

Slope platform:

```json
{
  "id": "slope_pond_to_canopy_01",
  "type": "platform",
  "shape": "slope",
  "role": "major_transition",
  "x": 1680,
  "y": 520,
  "y2": 440,
  "w": 320,
  "h": 24,
  "requiresFlatPads": true,
  "visual": {
    "kind": "slope",
    "theme": "greenroot-meadow",
    "rampCell": "gentleUp"
  }
}
```

The compiler should convert these to the current runtime shape:

- Flat: `{ id, x, y, w, h, terrainVisual }`
- Slope: `{ id, shape: "slope", x, y, y2, w, h, terrainVisual }`

### Object Attachment

Object layers should attach to platform IDs, not platform indexes.

Example enemy spawn:

```json
{
  "id": "spawn_moss_lane_01",
  "type": "enemySpawn",
  "x": 980,
  "platformId": "ground_main_01",
  "sectionId": "moss_lane",
  "sectionLabel": "Moss Lane",
  "weight": 2,
  "enemyPool": ["slimelet", "mossback"],
  "spawnRole": "starter_melee"
}
```

The compiler should resolve `platformId` to `platformIndex` after platforms are sorted and normalized. The editor should preserve both in exported debug output, but source maps should trust IDs.

### Example Editor Map File

```json
{
  "schema": "project-starfall-map-v1",
  "version": 1,
  "id": "greenrootMeadowEditorTest",
  "fileSlug": "greenroot-meadow-editor-test",
  "name": "Greenroot Meadow Editor Test",
  "role": "starterField",
  "levelRange": [1, 8],
  "biome": "greenroot",
  "routeId": "forest",
  "width": 6400,
  "height": 1120,
  "tileSize": 64,
  "authoredGroundY": 520,
  "palette": ["#77bf65", "#dff7ff", "#f3d86d"],
  "environment": {
    "profileId": "greenrootMeadow",
    "terrain": "greenroot-meadow",
    "props": "greenroot-meadow",
    "ramps": "greenroot-meadow",
    "backgroundAsset": "img/project-starfall/maps/greenroot-meadow.png"
  },
  "camera": {
    "startX": 0,
    "startY": 0,
    "minX": 0,
    "maxX": 6400,
    "minY": 0,
    "maxY": 1120,
    "previewFrameW": 1280,
    "previewFrameH": 640
  },
  "sections": [
    {
      "id": "entry_pond",
      "label": "Starter Pond",
      "x": 0,
      "w": 1600,
      "tier": "entry",
      "spawnRole": "starter"
    },
    {
      "id": "moss_lane",
      "label": "Moss Lane",
      "x": 1600,
      "w": 1800,
      "tier": "training",
      "spawnRole": "melee"
    },
    {
      "id": "canopy_reward",
      "label": "Canopy Reward",
      "x": 3400,
      "w": 1400,
      "tier": "optional",
      "spawnRole": "ranged"
    },
    {
      "id": "exit_gate",
      "label": "Thornpath Gate",
      "x": 4800,
      "w": 1600,
      "tier": "advance",
      "spawnRole": "exit"
    }
  ],
  "platforms": [
    {
      "id": "ground_entry",
      "type": "platform",
      "shape": "flat",
      "role": "main_ground",
      "x": 0,
      "y": 520,
      "w": 1880,
      "h": 80,
      "visual": { "kind": "ground" }
    },
    {
      "id": "slope_entry_to_mid",
      "type": "platform",
      "shape": "slope",
      "role": "major_transition",
      "x": 1880,
      "y": 520,
      "y2": 456,
      "w": 300,
      "h": 24,
      "visual": { "kind": "slope" }
    },
    {
      "id": "mid_moss_lane",
      "type": "platform",
      "shape": "flat",
      "role": "combat_lane",
      "x": 2180,
      "y": 456,
      "w": 1320,
      "h": 24,
      "visual": { "kind": "solidLane" }
    },
    {
      "id": "canopy_reward_ledge",
      "type": "platform",
      "shape": "flat",
      "role": "optional_reward",
      "x": 3780,
      "y": 328,
      "w": 760,
      "h": 24,
      "visual": { "kind": "island" }
    },
    {
      "id": "ground_exit",
      "type": "platform",
      "shape": "flat",
      "role": "main_ground",
      "x": 4540,
      "y": 520,
      "w": 1860,
      "h": 80,
      "visual": { "kind": "ground" }
    }
  ],
  "climbables": [
    {
      "id": "vine_canopy_reward",
      "type": "climbable",
      "x": 3960,
      "y": 328,
      "w": 28,
      "h": 128,
      "fromPlatformId": "mid_moss_lane",
      "toPlatformId": "canopy_reward_ledge"
    }
  ],
  "spawns": {
    "playerStart": {
      "id": "player_start",
      "x": 160,
      "platformId": "ground_entry",
      "facing": "right"
    },
    "enemySpawns": [
      {
        "id": "spawn_entry_01",
        "x": 920,
        "platformId": "ground_entry",
        "sectionId": "entry_pond",
        "sectionLabel": "Starter Pond",
        "weight": 1
      },
      {
        "id": "spawn_moss_lane_01",
        "x": 2720,
        "platformId": "mid_moss_lane",
        "sectionId": "moss_lane",
        "sectionLabel": "Moss Lane",
        "weight": 2
      }
    ],
    "fixedEnemySpawns": []
  },
  "enemies": ["slimelet", "mossback", "thornSprout"],
  "pickups": [
    {
      "id": "pickup_canopy_reward_01",
      "type": "material",
      "itemId": "upgradeDust",
      "quantity": 4,
      "x": 4160,
      "platformId": "canopy_reward_ledge",
      "runtimeDisabled": true
    }
  ],
  "hazards": [],
  "checkpoints": [
    {
      "id": "checkpoint_moss_lane",
      "label": "Moss Lane Rest",
      "x": 2240,
      "platformId": "mid_moss_lane",
      "facing": "right",
      "runtimeDisabled": true
    }
  ],
  "portals": [
    {
      "id": "greenroot_editor_return",
      "label": "Return to Starfall Crossing",
      "x": 96,
      "platformId": "ground_entry",
      "destinationMapId": "starfallCrossing",
      "portalRole": "return"
    },
    {
      "id": "greenroot_editor_thornpath",
      "label": "Thornpath Gate",
      "x": 6120,
      "platformId": "ground_exit",
      "destinationMapId": "thornpathThicket",
      "requiredMapId": "greenrootMeadow",
      "portalRole": "advance"
    }
  ],
  "stations": [],
  "npcs": [],
  "triggers": [],
  "metadata": {
    "purpose": "Editor schema smoke-test map for the Greenroot starter route.",
    "designIntent": {
      "archetype": "starter training field",
      "routeSummary": "Safe entry, one major ramp transition, flat moss combat lane, optional canopy reward, clear exit.",
      "partyRoleTarget": "solo starter",
      "visualIdentity": "pond bridges and soft Greenroot grass"
    },
    "author": "Project Starfall Editor",
    "createdAt": "2026-06-19T00:00:00.000Z",
    "updatedAt": "2026-06-19T00:00:00.000Z"
  }
}
```

### Exported Runtime Shape

The compiler should output a normal Starfall map record:

- `id`
- `name`
- `levelRange`
- `safeZone`
- `isDungeon`
- `bossRoom`
- `layoutRole`
- `palette`
- `asset`
- `environment`
- `purpose`
- `enemies`
- `platforms`
- `rampConnections`
- `climbables`
- `spawnPoints`
- `fixedEnemySpawns`
- `stations`
- `questNpcs`
- `portals`
- `fieldComposition`
- `designIntent`
- `spawnSections`
- `townServicePlan`
- `arenaSkeleton`
- `arenaMechanic`
- `editorSource`

The compiler should call or mirror:

- `makeRampConnections()`
- `getPlatformDefSurfaceY()`
- map publication helpers
- environment profile lookup
- portal normalization
- spawn section assignment

## 6. Tilemap and layer rules

### Required Layer Names

Use these layer IDs in editor JSON and UI:

| Layer ID | Type | Collision | Purpose |
| --- | --- | --- | --- |
| `background_far` | image/parallax | No | Sky, mountains, far atmosphere. |
| `background_mid` | image/parallax | No | Mid-distance structures and biome silhouettes. |
| `background_near` | image/parallax | No | Near decorative parallax, never gameplay-blocking. |
| `section_guides` | guide | No | Route sections, spawn sections, mechanic regions. |
| `terrain_visual` | tile/strip | No | Visible terrain surfaces and bodies. |
| `terrain_decoration_rear` | props | No | Rear props that sit behind actors. |
| `collision_solid` | platforms | Yes | Flat solid platforms. |
| `collision_slopes` | platforms | Yes | Slope platforms. |
| `collision_one_way` | platforms | Yes | Future one-way/drop-through platforms. |
| `climbables` | objects | Yes | Vines, ladders, ropes, chains, lifts. |
| `hazards` | objects | Yes/trigger | Lava, spikes, wind, rift, boss hazards. |
| `entities_spawns` | objects | No | Player start, enemy spawn points, fixed enemies. |
| `entities_items` | objects | No | Future pickups/rewards. |
| `entities_npcs` | objects | No | Quest NPCs and service NPCs. |
| `entities_stations` | objects | No | Shops, storage, upgrade, class, slots, Plinko. |
| `portals` | objects | Trigger | Exits, dungeon doors, return portals, boss portals. |
| `checkpoints` | objects | Trigger | Future checkpoint markers. |
| `triggers` | objects | Trigger | Tutorial, mechanic, event, and camera triggers. |
| `terrain_decoration_front` | props | No | Foreground props that do not hide gameplay. |
| `foreground_mask` | image/props | No | Very limited foreground silhouettes. |
| `editor_notes` | notes | No | Non-runtime notes and review markers. |

### Required Layer Order

Render/edit order:

1. `background_far`
2. `background_mid`
3. `background_near`
4. `section_guides`
5. `terrain_visual`
6. `terrain_decoration_rear`
7. `collision_solid`
8. `collision_slopes`
9. `collision_one_way`
10. `climbables`
11. `hazards`
12. `entities_spawns`
13. `entities_items`
14. `entities_npcs`
15. `entities_stations`
16. `portals`
17. `checkpoints`
18. `triggers`
19. `terrain_decoration_front`
20. `foreground_mask`
21. `editor_notes`

The editor UI may hide collision layers during art editing, but the source order should stay stable.

### Visual-Only Layers

These layers must never create collision automatically:

- `background_far`
- `background_mid`
- `background_near`
- `terrain_visual`
- `terrain_decoration_rear`
- `terrain_decoration_front`
- `foreground_mask`
- `editor_notes`

If a prop needs collision, create a separate collision or hazard object.

### Collision Layers

These layers create runtime collision or interaction:

- `collision_solid`
- `collision_slopes`
- `collision_one_way`
- `climbables`
- `hazards`
- `portals`
- `checkpoints`
- `triggers`

Collision layers should be locked by default in art mode. They should be visible as overlays in gameplay mode.

### Interactive Object Layers

These layers export gameplay metadata:

- `entities_spawns`
- `entities_items`
- `entities_npcs`
- `entities_stations`
- `portals`
- `checkpoints`
- `triggers`
- `hazards`

Each object must have:

- Stable `id`
- `type`
- `x`
- Optional `y`
- Optional `platformId`
- Layer-specific fields
- Optional `notes`

### Tile References

Use symbolic tile/cell references instead of raw atlas indexes whenever possible.

Recommended visual reference:

```json
{
  "assetGroup": "terrain",
  "assetId": "greenroot-meadow",
  "cell": "platformMid",
  "variant": 0
}
```

The renderer/compiler can resolve this through:

- `ENVIRONMENT_ASSETS.terrain`
- `ENVIRONMENT_TERRAIN_CELLS`
- `ENVIRONMENT_ASSETS.props`
- `ENVIRONMENT_PROP_CELLS`

Raw indexes may be stored as compiled output, but source maps should use names.

### Missing Tile Handling

If a visual cell is missing:

- Editor should show a magenta/black missing-tile placeholder.
- Validator should warn for draft maps.
- Production export should fail if the missing asset affects a visible runtime layer.

If a terrain visual is missing but collision is valid:

- Runtime may fall back to `drawPlatformFallback()`.
- Editor should still warn because excessive fallback terrain makes maps look cheap.

### Tile ID Management

Rules:

- Do not assign arbitrary numeric tile IDs in editor source.
- Use named cells from data files.
- Use `assetId` values that match `ENVIRONMENT_ASSETS`.
- Keep map-specific custom cells in `content/project-starfall/tilesets/` until they are promoted into `data/environment.js`.

### Auto-Tiling And Rule Tiles

Initial implementation:

- Auto-generate terrain strip visuals from platform role and biome.
- Use existing renderer strip behavior.

Future implementation:

- Add rules that choose left/mid/right/top/body/underside cells based on platform dimensions.
- Add prop scatter rules that respect `ENVIRONMENT_READABILITY_DEFAULTS`.
- Add slope visual rules that choose ramp art by grade and direction.

Rules must not create gameplay collision automatically. They can suggest collision, but the source of truth remains explicit platforms.

### Alignment Rules

Visual terrain and collision must stay aligned:

- Platform surface visual top should be within 8 px of collision surface for flat platforms.
- Slope visual line should be within 6 px of collision sample line at 25%, 50%, and 75%.
- Prop foot positions must not float more than 8 px above platform surface.
- Climbable top/bottom should reach target platform surfaces within 16 px.
- Portal base should land on a platform surface within 12 px.
- Checkpoint base should land on a flat platform within 8 px.

### Locked/System Layers

Lock these by default:

- `collision_solid`
- `collision_slopes`
- `portals`
- `entities_spawns`
- `entities_stations`
- `section_guides`

Unlocking should require switching to the matching editor tool. For example, selecting the Slope tool unlocks `collision_slopes` while leaving portal and spawn layers locked.

### Naming Conventions

Map IDs:

- Runtime map ID: camelCase, matching existing style, e.g. `greenrootMeadow`.
- File slug: kebab-case, e.g. `greenroot-meadow`.
- New custom maps: use a prefix or suffix until promoted, e.g. `greenrootMeadowEditorTest`.

Platform IDs:

- `ground_<section>_<number>`
- `lane_<section>_<tier>_<number>`
- `slope_<from>_to_<to>_<number>`
- `ledge_<section>_<number>`
- `bridge_<section>_<number>`
- `boss_<role>_<number>`

Object IDs:

- `spawn_<section>_<number>`
- `portal_<from>_<to>`
- `checkpoint_<section>`
- `hazard_<section>_<kind>_<number>`
- `pickup_<section>_<item>_<number>`
- `npc_<map>_<role>`
- `station_<map>_<kind>`
- `trigger_<section>_<event>`

Section IDs:

- snake_case, readable, stable, e.g. `starter_pond`, `moss_lane`, `rift_core`.

Asset IDs:

- Keep existing kebab-case style, e.g. `greenroot-meadow`, `stormbreak-cliffs`, `eclipse-throne`.

## 7. Slope editing guidance

### Slope Representation

In the editor, a slope is a collision platform object with a linked visual.

Required fields:

- `id`
- `shape: "slope"`
- `x`
- `y`
- `y2`
- `w`
- `h`
- `role`
- `visual.kind: "slope"`
- `requiresFlatPads`

Do not represent slopes as only tiles. Do not represent slopes as decorative art with hidden inferred collision. Starfall movement, enemy pathing, spawn placement, and platform graph logic need explicit slope platforms.

### Slope Tools

The editor should provide:

- Add slope by dragging from left endpoint to right endpoint.
- Flip slope direction.
- Snap endpoint height to 16 px increments.
- Snap width to 16 px increments.
- Convert selected flat platform edge into a slope connector.
- Split a long slope into slope plus flat rest.
- Convert selected slope to flat platform.
- Show slope grade as a number.
- Show warnings while dragging if grade or density exceeds the map budget.

### Recommended Slope Angles

Use grade, not degrees, in validation:

- Grade = `abs(y2 - y) / w`.
- Comfortable routine slope: `0.20` to `0.45`.
- Strong terrain transition: `0.45` to `0.60`.
- Dramatic slope: `0.60` to `0.72`.
- Production maximum: `0.72`, matching the current validator.

Slope grade rules:

- Starter fields should usually stay below `0.55`.
- Towns should usually stay below `0.45`.
- Boss arenas should usually stay below `0.50` unless the arena mechanic needs a dramatic ramp.
- Cinder, Stormbreak, and Eclipse can use steeper slopes as landmarks, but not as default lane connectors.
- Anything above `0.72` must fail validation.

### Current Slope Budget Targets

Use the current validator in `build/validate-project-starfall-maps.js` as the baseline:

| Map Role | Max Slopes | Max Grade | Max Slopes In Any 1200 px Window |
| --- | ---: | ---: | ---: |
| Shop interior | 0 | 0.00 | 0 |
| Town / safe zone | 4 | 0.70 | 3 |
| Boss arena | 4 | 0.72 | 3 |
| Dungeon | 4 | 0.72 | 3 |
| Starter field | 6 | 0.72 | 3 |
| Training/deep field | 8 | 0.72 | 3 |
| Endless/deep field | 8 | 0.72 | 3 |

The editor should show both hard validation and softer design guidance:

- Ideal town slopes: 0-2.
- Ideal dungeon/boss slopes: 0-3.
- Ideal starter field slopes: 2-4.
- Ideal training/deep field slopes: 3-6.
- Ideal endless field slopes: 4-6, with rift/portal transitions carrying the rest.

### Slope Visual And Collision Linkage

Every slope must show three overlays:

- Collision line from `(x, y)` to `(x + w, y2)`.
- Visual surface line from the ramp art or fallback drawing.
- Walkable pad zones at both ends.

The editor should sample the slope at 25%, 50%, and 75%. If the visual surface differs from collision by more than 6 px at any sample, show a warning.

### Flat Pad Requirements

Every slope needs flat terrain before and after it.

Minimum pads:

- Town/service slope: 160 px each side.
- Starter field slope: 128 px each side.
- Combat route slope: 192 px each side.
- Boss/dungeon slope: 240 px each side.
- Optional route slope: 96 px each side.

Do not allow a slope to connect directly into another slope unless the designer marks it as `approvedLongTransition: true`, and even then the chain must remain below the density limit.

### Excessive Slope Detection

The editor should warn when:

- A map exceeds the role-based slope budget.
- More than 3 slopes appear within any 1200 px window.
- Two slopes touch without at least 96 px of flat rest.
- More than 40% of a section's playable main route is sloped.
- More than one combat spawn is near a slope end in the same section.
- A pickup, checkpoint, station, or required NPC is placed on a slope.
- A slope grade is above `0.60` in a starter/tutorial section.
- A boss arena's primary lane is not mostly flat.

### Long Slope Chain Detection

A slope chain is any connected sequence of slopes and short flats where each flat rest is under 96 px.

Rules:

- Chain length over 2 slopes: warning.
- Chain length over 3 slopes: error unless marked as a special set piece.
- Chain horizontal span over 1000 px: warning.
- Chain vertical rise over 240 px: warning.
- Chain through combat spawns: error.

### Flat Areas For Combat And Rest

The editor should calculate broad flats:

- Broad flat = non-slope platform with `w >= 640`.
- Party combat flat = non-slope platform with `w >= 900`.
- Boss main lane = non-slope platform with `w >= 1200`.

Rules:

- Every combat map needs at least 2 broad flats.
- Slope count should not exceed broad flat count.
- Ideal broad-flat-to-slope ratio is at least `1.5:1`.
- Boss arenas need one dominant flat primary lane.
- Spawn sections should have at least one flat combat area unless they are explicitly traversal-only.

### Slope Collision Preview

Preview mode should display:

- Green line for valid slope collision.
- Yellow line for steep but valid slope.
- Red line for invalid grade.
- Blue endpoint handles.
- Flat pad zones.
- Actor foot sample dots.
- Surface Y label at mouse position.

The preview should use the same `getPlatformSurfaceY()` calculation as runtime.

### Slope Movement Testing

The editor should include a slope test mode:

1. Place the player at the lower pad.
2. Walk up the slope without jumping.
3. Walk down the slope.
4. Stop mid-slope.
5. Attack on the slope.
6. Jump while moving uphill.
7. Jump while moving downhill.
8. Drop from the upper platform onto the slope.
9. Spawn a melee enemy on each adjacent flat pad.
10. Confirm camera motion does not bob distractingly.

Do not ship a slope-heavy section without running this test.

### Avoiding Cheap Or Auto-Generated Maps

Slope overuse creates the "cheap map generator" look when:

- Every height change becomes a ramp.
- Slopes repeat the same width and grade.
- Flat rests are missing.
- Diagonal lines dominate the silhouette.
- Enemies and pickups sit awkwardly on diagonal surfaces.
- Biomes share the same ramp pattern.
- Slopes appear without landmarks or route purpose.

To avoid this:

- Use slopes primarily for major terrain transitions.
- Use ledges, steps, cliffs, ladders, bridges, vines, lifts, portals, and drop routes for smaller changes.
- Put combat on flats.
- Give each slope a reason: reveal, route change, biome identity, pacing shift, or set piece.
- Break long routes with landmarks and horizontal rest.
- Vary slope width and grade only when it supports the area.

### Slope Editor Defaults

Default slope tool settings:

- Width: 300 px.
- Height delta: 64 px.
- Grade warning above `0.55`.
- Grade error above role max.
- Endpoint snap: 16 px.
- Minimum pad warning: 128 px.
- Auto-link adjacent platforms if endpoints match within 36 px horizontally and 24 px vertically.
- Auto-run ramp connection generation after every slope edit.

## 8. Entity and object placement

### Object Model

Every entity/object should use this base shape:

```json
{
  "id": "object_id",
  "type": "objectType",
  "x": 0,
  "y": 0,
  "platformId": "platform_id",
  "sectionId": "section_id",
  "label": "Readable Label",
  "properties": {}
}
```

The editor property panel should hide irrelevant fields and show dropdowns from project data.

### Object Placement Specification Matrix

Use this matrix as the minimum property contract for the editor property panel. Object types marked "future runtime" should still be editable, validated, exported, and documented now, but their gameplay behavior should remain disabled until the corresponding runtime system is implemented.

| Object Type | Placement Rules | Required Properties | Optional Properties | Visual Editor Marker | Validation Rules | Runtime Reference Format | Naming Convention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Player spawn | Flat platform, safe camera frame, no nearby hazard/enemy. | `id`, `x`, `platformId`, `facing` | `cameraX`, `cameraY`, `notes` | Blue player arrow with facing chevron. | Must resolve to flat platform; must be inside map bounds; must have safe flat space. | `spawns.playerStart`, preview fallback placement. | `player_start` or `player_start_<section>` |
| Checkpoint | Flat rest zone after challenge or before branch. | `id`, `label`, `x`, `platformId` | `safeRadius`, `facing`, `runtimeDisabled` | Cyan flag/star marker. | Must be reachable; no enemy/hazard in safe radius; not on slope. | `checkpoints[]` future runtime. | `checkpoint_<section>` |
| Exit/portal | Flat approach space; readable destination; route edge if production. | `id`, `label`, `x`, `platformId`, `destinationMapId` | `routeId`, `dungeonId`, `requiredMapId`, `returnPortal`, `bossPortal` | Gold portal door with destination label. | Destination exists or is editor-only; platform exists; route gate metadata valid. | `portals[]` or compiled `MAP_PORTALS`. | `portal_<from>_<to>` |
| Enemy spawn | Flat combat lane, section-owned, outside immediate player safe zone. | `id`, `x`, `platformId`, `sectionId`, `weight` | `enemyPool`, `spawnRole`, `leashRadius`, `minLevel`, `maxLevel` | Red spawn diamond with enemy-role badge. | No slopes; platform reachable; enemy IDs valid; density sane. | `spawnPoints[]`. | `spawn_<section>_<number>` |
| Fixed enemy | Set piece, tutorial, lab, boss add, or hand-authored encounter. | `id`, `enemyId`, `x`, `platformId` | `count`, `facing`, `respawn`, `sectionId`, `temporary` | Red enemy silhouette. | Enemy ID valid; not inside collision; reachable or intentionally airborne. | `fixedEnemySpawns[]`. | `fixed_enemy_<section>_<enemy>_<number>` |
| Boss | Boss arena marker or encounter start, not a generic field spawn. | `id`, `bossId`, `x`, `platformId`, `arenaId` | `dungeonId`, `encounterId`, `introTriggerId`, `phaseRegionIds` | Purple crown/skull marker. | Boss ID valid; arena has flat primary lane; boss portal exists. | `bossId`, `bossEncounterId`, `arenaSkeleton`, `arenaMechanic`. | `boss_<bossId>_anchor` |
| Item | Hand-authored visible reward; flat or clearly reachable ledge. | `id`, `kind`, `itemId`, `quantity`, `x`, `platformId` | `rarity`, `upgrade`, `condition`, `runtimeDisabled` | Green pickup sparkle. | Item ID valid; reachable; not inside collision; not on awkward slope. | `pickups[]` future runtime. | `pickup_<section>_<itemId>_<number>` |
| Pickup | Same as item but can represent currency/material bundle. | `id`, `pickupType`, `x`, `platformId` | `currency`, `materials`, `consumables`, `respawnSeconds` | Green bag/gem marker. | Reward references valid; visible or hinted; reachable. | `pickups[]` future runtime. | `pickup_<section>_<type>_<number>` |
| Hazard | Telegraphable danger with avoid path and warning space. | `id`, `hazardType`, `x`, `y`, `w`, `h` | `damage`, `tickSeconds`, `telegraphSeconds`, `pattern`, `mechanicId` | Orange/red translucent region. | Not on player start/checkpoint; has warning space; bounds valid. | `hazards[]`, `triggers[]`, or map mechanic hook. | `hazard_<section>_<kind>_<number>` |
| Trigger | Invisible event or mechanic region with explicit target. | `id`, `triggerType`, `x`, `y`, `w`, `h` | `event`, `targetId`, `condition`, `once`, `sectionId` | Yellow outlined rectangle. | Bounds valid; target references valid; no accidental overlap with start. | `triggers[]` future runtime or onboarding/map mechanic hook. | `trigger_<section>_<event>` |
| Door | Interior transition, shop door, dungeon gate, or locked route door. | `id`, `doorType`, `x`, `platformId`, `destinationMapId` | `requiredMapId`, `stationId`, `lockedLabel`, `assetId` | Door icon with lock/arrow badge. | Destination valid; approach flat; lock condition valid. | Compiled as portal/station/shop interior metadata. | `door_<from>_<to>` |
| Switch | Mechanic activator tied to hazard, door, boss phase, or platform. | `id`, `switchType`, `x`, `platformId`, `targetId` | `cooldown`, `oneShot`, `requiredItemId`, `sectionId` | Lever/button icon. | Target exists; reachable; not inside enemy-only zone unless intentional. | `triggers[]` or future `switches[]`. | `switch_<section>_<target>` |
| Moving platform | Future authored platform motion, lift, ferry, or elevator. | `id`, `x`, `y`, `w`, `h`, `path` | `speed`, `pauseSeconds`, `loop`, `platformRole`, `runtimeDisabled` | Blue platform with path line. | Path inside bounds; no impossible gaps; collision preview valid. | `movingPlatforms[]` future runtime, compile to special platform later. | `moving_<section>_<number>` |
| NPC | Quest, service, class, lore, or event character. | `id`, `npcType`, `name`, `x`, `platformId` | `questId`, `stationId`, `dialogueId`, `role` | NPC bust marker. | Platform valid; station/quest references valid; approach space clear. | `questNpcs[]` or station host metadata. | `npc_<map>_<role>` |
| Dialogue trigger | Region that starts or advances dialogue/onboarding. | `id`, `dialogueId`, `x`, `y`, `w`, `h` | `npcId`, `once`, `condition`, `onCompleteEvent` | Speech-bubble region. | Dialogue/NPC target valid; not spam-triggered; bounds readable in editor. | `triggers[]` future runtime or onboarding event. | `dialogue_<section>_<topic>` |
| Secret area | Optional route or hidden pocket with reveal rule. | `id`, `x`, `y`, `w`, `h`, `revealType` | `hintObjectId`, `rewardContainerId`, `condition` | Dashed teal region. | Reachable; not required for main route; reward valid; camera can reveal. | `secretAreas[]` future runtime metadata. | `secret_<section>_<number>` |
| Reward container | Chest, ore node, cache, shrine, or breakable reward. | `id`, `containerType`, `x`, `platformId`, `rewardTableId` | `oneShot`, `respawnSeconds`, `condition`, `assetId`, `runtimeDisabled` | Chest/cache icon. | Reward table valid; reachable; not on slope unless approved; visible or hinted. | `rewardContainers[]` future runtime or pickup generator. | `reward_<section>_<type>_<number>` |

Example reward container entry:

```json
{
  "id": "reward_canopy_cache_01",
  "type": "rewardContainer",
  "containerType": "cache",
  "x": 4210,
  "platformId": "canopy_reward_ledge",
  "sectionId": "canopy_reward",
  "rewardTableId": "greenroot_optional_cache",
  "oneShot": true,
  "runtimeDisabled": true,
  "notes": "Future optional route prize. Keep visible from the main lane."
}
```

### Player Start

Purpose:

- Defines the default start point for newly created custom maps.

Fields:

- `id`
- `x`
- `platformId`
- `facing`
- `cameraX`
- `cameraY`

Rules:

- Must attach to a flat platform.
- Must have at least 480 px of safe flat terrain nearby.
- Must not overlap an enemy spawn, hazard, or portal.
- For production route maps, portal entry can override player start.

Runtime integration:

- Used by custom map preview and fallback placement.
- Can later feed map entry logic.

### Enemy Spawn Point

Purpose:

- Standard field/dungeon respawn source.

Fields:

- `id`
- `x`
- `platformId`
- `weight`
- `sectionId`
- `sectionLabel`
- `enemyPool`
- `spawnRole`
- `minLevel`
- `maxLevel`
- `leashRadius`

Rules:

- Must attach to a flat platform.
- Must not be on a slope.
- Must not be within 240 px of player start.
- Must not be inside a checkpoint safe zone.
- Must include section metadata for combat maps.

Runtime integration:

- Exports to `spawnPoints`.
- Compiler resolves `platformId` to `platformIndex`.

### Fixed Enemy Spawn

Purpose:

- Tutorial, admin lab, set piece, boss add, or hand-authored encounter.

Fields:

- `id`
- `enemyId`
- `x`
- `platformId`
- `sectionId`
- `count`
- `facing`
- `respawn`
- `temporary`

Rules:

- Use sparingly in fields.
- Good for class trials, labs, boss rooms, and scripted tutorials.
- Validate `enemyId` against `Data.ENEMIES`.

Runtime integration:

- Exports to `fixedEnemySpawns` where supported.

### Pickup

Purpose:

- Future hand-authored rewards.

Fields:

- `id`
- `kind`: `material`, `consumable`, `equipment`, `currency`, `quest`
- `itemId`
- `quantity`
- `rarity`
- `upgrade`
- `x`
- `platformId`
- `visible`
- `respawnSeconds`
- `condition`
- `runtimeDisabled`

Rules:

- Must be reachable.
- Should be visible or hinted.
- Should sit on flat terrain.
- Should not be required for progression until runtime support is complete.

Runtime integration:

- Phase 1: editor metadata only.
- Phase 2: add pickup runtime spawning.

### Hazard

Purpose:

- Damage, crowd control, movement modifier, or map mechanic trigger.

Fields:

- `id`
- `hazardType`
- `x`
- `y`
- `w`
- `h`
- `sectionId`
- `damage`
- `tickSeconds`
- `telegraphSeconds`
- `activePattern`
- `mechanicId`
- `runtimeDisabled`

Recommended hazard types:

- `lava`
- `spikes`
- `wind`
- `iceSlip`
- `riftSurge`
- `fallingDebris`
- `projectileEmitter`
- `bossCall`
- `slowField`
- `damageField`

Rules:

- Must be telegraphed.
- Must not overlap player start or checkpoint safe zone.
- Must allow a readable avoidance route.
- Must fit biome identity.

Runtime integration:

- Can feed future hazard runtime or existing map mechanic definitions.

### Checkpoint

Purpose:

- Future respawn/rest marker.

Fields:

- `id`
- `label`
- `x`
- `platformId`
- `facing`
- `safeRadius`
- `cameraX`
- `cameraY`
- `runtimeDisabled`

Rules:

- Must be on a flat platform.
- Needs 480-720 px safe flat rest depending on map role.
- No enemies or hazards inside safe radius.
- Should have a landmark or clear prop.

Runtime integration:

- Phase 1: editor/playtest marker only.
- Phase 2: save checkpoint state and respawn there.

### Portal/Exit

Purpose:

- Map transition, dungeon entrance, return portal, boss portal, shop door, or route gate.

Fields:

- `id`
- `label`
- `x`
- `platformId`
- `destinationMapId`
- `routeId`
- `dungeonId`
- `requiredMapId`
- `entryPortalId`
- `portalRole`
- `returnPortal`
- `bossPortal`
- `shopInterior`

Rules:

- Must attach to a platform.
- Must have a valid destination unless marked editor-only.
- Must have readable label.
- Must have enough flat approach space.
- Route portals should match `WORLD_MAP_EDGES`.

Runtime integration:

- Exports to map portals and aligns through runtime.

### Climbable

Purpose:

- Vines, ladders, ropes, chains, stairs, lifts, vertical traversal helpers.

Fields:

- `id`
- `kind`
- `x`
- `y`
- `w`
- `h`
- `fromPlatformId`
- `toPlatformId`
- `sectionId`

Rules:

- Top and bottom should align to reachable platforms.
- Must not overlap portals or stations.
- Should be easier to read than a hidden jump route.
- Use climbables instead of slopes for strong vertical movement.

Runtime integration:

- Exports to `climbables`.

### Station

Purpose:

- Town service or interactable service point.

Fields:

- `id`
- `stationType`
- `name`
- `x`
- `platformId`
- `asset`
- `serviceFrequency`
- `townServiceRole`

Station types:

- `storage`
- `shop`
- `upgrade`
- `class`
- `slots`
- `plinko`

Rules:

- Frequent services should be on main flat ground.
- Stations should not sit on slopes.
- Station host NPCs should be colocated with the station.
- Town layout should follow service pocket rules from `docs/project-starfall-map-optimization-audit.md`.

Runtime integration:

- Exports to `stations`.
- Tied to `townServicePlan`.

### NPC

Purpose:

- Quest, guide, service host, lore, class master, event NPC.

Fields:

- `id`
- `npcType`
- `name`
- `x`
- `platformId`
- `questId`
- `stationId`
- `dialogueId`
- `role`

Rules:

- NPCs should have clear approach space.
- Service NPCs should be near their station.
- Quest NPCs should not block the main route.

Runtime integration:

- Exports to `questNpcs`.

### Trigger

Purpose:

- Tutorial messages, map mechanics, arena calls, camera cues, one-shot events, route locks.

Fields:

- `id`
- `triggerType`
- `x`
- `y`
- `w`
- `h`
- `sectionId`
- `event`
- `targetId`
- `once`
- `condition`

Rules:

- Trigger bounds should be visible in editor.
- Must validate target references.
- Should not fire before the player can understand the context.

Runtime integration:

- Phase 1: editor metadata and validation.
- Phase 2: hook into onboarding, map mechanics, boss mechanics, and quest events.

## 9. Editor UX and workflow

### Recommended Entry Point

Add editor access through the existing Worldwright/admin surface.

Recommended access methods:

- Query parameter: `/project-starfall?editor=1`
- Worldwright Console tab: `Map Editor`
- Admin command: `map editor`
- Optional keyboard shortcut only in editor mode.

Access rules:

- Editor mode requires Worldwright/admin access in normal UI.
- Local development can allow query-param access.
- Production should hide editor controls unless explicitly enabled.

### Layout

Use a work-tool layout, not a marketing page.

Recommended editor screen:

- Top toolbar: mode, tool, map name, validation status, save/export, playtest.
- Left sidebar: layers and palettes.
- Center: game canvas/editor canvas.
- Right sidebar: selected object properties and map metadata.
- Bottom bar: validation messages, coordinates, selected object ID, zoom, grid snap, current section.

The canvas should be large. Avoid forcing all editing through a small modal.

### Tool Modes

Required modes:

- Select
- Pan
- Terrain brush
- Platform collision
- Slope
- Climbable
- Entity/object
- Portal
- Hazard
- Trigger
- Section guide
- Note
- Playtest

### Toolbar Commands

Top toolbar commands:

- New
- Open
- Clone Current Runtime Map
- Import JSON
- Export JSON
- Validate
- Playtest
- Stop Playtest
- Screenshot
- Undo
- Redo
- Grid
- Collision Overlay
- Camera Overlay
- Slope Overlay
- Snap
- Help

### Keyboard Shortcuts

Use familiar editor shortcuts:

| Shortcut | Action |
| --- | --- |
| `V` | Select |
| `H` or Space-drag | Pan |
| `B` | Brush |
| `E` | Eraser |
| `P` | Platform tool |
| `S` | Slope tool |
| `C` | Collision overlay |
| `O` | Object tool |
| `R` | Portal/route tool |
| `T` | Trigger tool |
| `G` | Toggle grid |
| `L` | Toggle layer panel |
| `F` | Frame selection |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |
| `Ctrl+C` | Copy |
| `Ctrl+V` | Paste |
| `Delete` | Delete selected |
| `Ctrl+S` | Save/export draft |
| `Enter` | Playtest from selected start |
| `Esc` | Cancel tool or stop playtest |
| `Shift` while dragging | Constrain line/axis |
| `Alt` click | Eyedropper |

Do not override existing gameplay keybinds while playtest mode is active unless editor mode owns focus.

### Layer Panel

Each layer row should show:

- Visibility toggle.
- Lock toggle.
- Layer type icon.
- Layer name.
- Object count or tile count.
- Warning badge.

Clicking a warning badge filters validation messages to that layer.

### Palette Panel

Palette categories:

- Terrain
- Slopes/ramps
- Platforms
- Props
- Structures
- Backgrounds
- Spawns
- Enemies
- Pickups
- Hazards
- Portals
- Stations
- NPCs
- Checkpoints
- Triggers
- Notes

Terrain and prop palettes should change when biome/theme changes.

### Property Panel

The property panel should:

- Show selected object ID and type.
- Use dropdowns for known IDs.
- Use number steppers for coordinates.
- Show platform attachment and "snap to surface".
- Show section attachment.
- Show validation state.
- Provide "focus", "duplicate", "delete", and "convert" actions.

Do not require normal object edits through raw JSON.

### Map Creation Workflow

Recommended workflow for a new map:

1. Choose template.
2. Enter map ID, display name, role, biome, and level range.
3. Choose environment profile.
4. Place main ground route.
5. Add route sections.
6. Add broad flat combat lanes.
7. Add only necessary slopes/climbables.
8. Add player start and portals.
9. Add enemy pools and spawn points.
10. Add pickups, hazards, checkpoints, and triggers if needed.
11. Add visual terrain and props.
12. Validate.
13. Playtest from start.
14. Playtest from each section.
15. Export JSON.
16. Run Node validation and project tests.

### Editing Existing Maps

Recommended workflow:

1. Clone current runtime map into editor JSON.
2. Keep original map ID read-only until ready.
3. Rename draft ID with `EditorDraft` suffix.
4. Clean platform IDs.
5. Mark main route and sections.
6. Reduce slopes according to guide budgets.
7. Move spawns off slopes and into flat combat zones.
8. Attach portals/stations/NPCs to platform IDs.
9. Validate and playtest.
10. Promote to production map ID only after validation passes.

### Save Model

Use three save levels:

Draft autosave:

- localStorage key: `projectStarfallMapEditorDrafts.v1`
- Browser-only.
- Can include invalid maps.

Exported source:

- JSON download/import.
- Version-controlled in `content/project-starfall/maps/`.
- Must pass schema validation.

Compiled runtime:

- Generated JS manifest.
- Must pass runtime validation and project tests.
- Used by production gameplay.

### Playtest Isolation

Editor playtest must not mutate normal player progress.

Rules:

- Do not write to `projectStarfallPrototypeSave.v1`.
- Use a cloned test state.
- Disable persistent map analytics writes during editor playtest or mark them `editorOnly`.
- Disable item grants/drops from editor pickups unless explicitly testing.
- Provide reset-to-editor button.

## 10. Validation system

### Validator Architecture

Use shared validation in two contexts:

- Browser editor validation.
- Node/build validation.

Recommended modules:

- `js/games/project-starfall/data/map-editor-schema.js`
- `js/games/project-starfall/data/map-editor-validation.js`
- `build/validate-project-starfall-editor-maps.js`

The Node validator should import the same rules where possible. If browser compatibility prevents direct import, keep rule constants in a data-only module.

### Validation Severity

Use four severities:

- `error`: blocks runtime preview and production export.
- `warning`: allows draft save and preview, blocks production export only if configured.
- `notice`: non-blocking polish suggestion.
- `info`: design note or measurement.

### Core Validation Rule Matrix

Use these rule IDs in browser validation, Node validation, and documentation. "Blocks save" means the editor should not allow the map to be stored as a clean source map. "Blocks export" means the map cannot be promoted into compiled runtime content.

| Rule Name | What It Checks | Severity | How To Fix | Save/Export Behavior |
| --- | --- | --- | --- | --- |
| `schema.requiredFields` | Required top-level map fields exist. | Error | Fill missing schema, version, ID, name, role, dimensions, environment, platforms, and metadata. | Blocks save and export. |
| `schema.invalidObjectProperty` | Object fields have valid types and allowed values. | Error | Correct the property in the object panel. | Blocks save and export. |
| `schema.duplicateIds` | Map, layer, platform, section, and object IDs are unique. | Error | Rename duplicate IDs with editor-generated names. | Blocks save and export. |
| `map.missingPlayerSpawn` | Map has player start or valid entry portal. | Error | Add `playerStart` or at least one valid entry portal. | Blocks export; draft save allowed only as invalid draft. |
| `map.missingExit` | Non-arena route map has at least one exit/portal. | Error | Add a portal with valid destination or mark map as isolated arena/trial. | Blocks export. |
| `map.missingMetadata` | Role, level range, biome, purpose, and design intent exist. | Warning | Complete map metadata panel. | Allows draft save; blocks production export. |
| `map.sizeBounds` | Width and height are within project limits for role. | Warning | Resize map or choose a more appropriate template. | Allows save; blocks export if too small to play. |
| `tiles.brokenReference` | Terrain/prop/ramp tile references exist. | Error | Choose a valid cell from the palette or add the asset metadata. | Blocks export; draft save allowed. |
| `tiles.missingBackground` | Required background/profile is missing for production map. | Warning | Choose a map backdrop or environment profile. | Allows save; blocks production export. |
| `collision.missing` | Map has playable collision platforms. | Error | Add at least one valid flat platform. | Blocks save and export. |
| `collision.visualMismatch` | Visual terrain surface differs from collision beyond tolerance. | Warning | Move visual terrain, adjust platform, or approve a deliberate mismatch. | Allows save; blocks export when mismatch affects main route. |
| `collision.negativeSize` | Collision/object rectangles have non-positive dimensions. | Error | Resize or delete invalid shapes. | Blocks save and export. |
| `slope.excessiveDensity` | Slope count exceeds role budget. | Error | Remove, shorten, or replace slopes with ledges, stairs, cliffs, or climbables. | Blocks export. |
| `slope.longChain` | Connected slope chain exceeds project threshold. | Warning/Error | Insert flat rests or break the chain with a ledge/climbable. | Blocks export if chain is unapproved and affects main route. |
| `slope.gradeTooSteep` | Slope grade exceeds role maximum. | Error | Lower rise, widen slope, or replace with ladder/ledge. | Blocks export. |
| `slope.visualMismatch` | Ramp art does not match slope collision samples. | Warning | Adjust visual ramp cell, platform endpoints, or renderer mapping. | Allows save; blocks export for required route slopes. |
| `slope.spawnAdjacent` | Spawn/checkpoint/exit/major encounter lacks nearby flat ground. | Error | Add flat pads or move object to a flat combat/rest platform. | Blocks export. |
| `combat.noFlatAreas` | Combat map lacks broad flat combat/rest spaces. | Error | Add platforms at least 640 px wide; boss main lane should be wider. | Blocks export. |
| `combat.awkwardArenaSlope` | Important combat arena uses awkward slope placement. | Warning/Error | Move encounter to flats or mark slope-based encounter as intentional. | Blocks export for boss/dungeon primary arena. |
| `enemy.insideSolid` | Enemy/fixed spawn overlaps solid collision. | Error | Move spawn to platform surface or valid airborne marker. | Blocks export. |
| `enemy.invalidId` | Referenced enemy ID is missing from game data. | Error | Pick a valid enemy from `Data.ENEMIES`. | Blocks save and export. |
| `item.insideSolid` | Item/pickup/reward overlaps solid collision. | Error | Move reward to surface or valid container. | Blocks export when runtime-enabled. |
| `item.invalidId` | Referenced item/material/equipment/consumable is missing. | Error | Choose a valid item ID from current data. | Blocks export when runtime-enabled. |
| `spawn.unsafeArea` | Player/enemy spawn is too close to hazard/enemy/portal conflict. | Warning/Error | Move spawn or add safe zone spacing. | Blocks export for player start and checkpoints. |
| `spawn.onSlope` | Spawn is attached to a slope platform. | Error | Move spawn to a flat platform. | Blocks export. |
| `exit.unreachable` | Required exit cannot be reached from player start. | Error | Add route connection, platform, climbable, portal, or adjust jump gaps. | Blocks export. |
| `checkpoint.unreachable` | Checkpoint cannot be reached from player start or prior route. | Error | Move checkpoint or connect route. | Blocks export when runtime-enabled. |
| `camera.boundsInvalid` | Camera bounds are too small, too large, or outside world. | Warning/Error | Resize map/camera bounds; confirm first-screen and exit frames. | Blocks export for invalid start/exit frames. |
| `hazard.noWarningSpace` | Hazard can damage player before readable warning. | Warning/Error | Add telegraph space, signposting, safe lane, or delay. | Blocks export for required route hazards. |
| `performance.objectCount` | Object/tile/prop count exceeds role budget. | Suggestion/Warning | Reduce decoration density, split map, or use generated props. | Warns; may block export if frame budget fails. |

### Required Validation Categories

Schema validation:

- Required fields exist.
- Types are correct.
- IDs are unique.
- Schema version is supported.

Reference validation:

- Platform references exist.
- Destination maps exist.
- Enemy IDs exist.
- Item IDs exist.
- Station types are valid.
- Section IDs exist.
- Environment assets exist.
- Tile/cell names exist.

Collision validation:

- At least one valid platform.
- Player start has a platform.
- Portal bases resolve to platforms.
- Climbables connect reachable surfaces.
- No zero-width platforms.
- No negative-size objects.
- Platforms are inside map bounds or intentionally marked external.

Slope validation:

- Slope count within role budget.
- Max grade within role budget.
- No more than 3 slopes in any 1200 px window.
- Every slope has generated ramp connection.
- No spawns/checkpoints/stations on slopes.
- Broad flat count is sufficient.
- Slope pads exist.
- No unapproved long slope chains.

Reachability validation:

- Player start can reach required exit.
- Broad combat lanes are reachable.
- Spawn platforms are reachable by the player or valid for enemy-only roles.
- Optional reward routes are reachable and returnable.
- Required quest/NPC/station portals are reachable.

Combat validation:

- Combat maps have enemy pool or fixed spawns.
- Spawn points are distributed by section.
- Spawns have enough flat ground.
- Spawn density is not too high near player start or checkpoints.
- Ranged/flyer spawns have enough sightline and ceiling clearance.

Camera validation:

- Player start camera frame is valid.
- Exits are visible before activation.
- Long vertical transitions have rest platforms.
- No required object sits outside expected playfield visibility.
- No dense foreground object hides combat lanes.

Visual validation:

- Environment profile exists.
- Palette exists.
- Terrain visuals use valid cells.
- Collision and visuals align.
- Props do not block important objects.
- Background contrast remains readable.

Progression validation:

- Route IDs match `WORLD_ROUTES`.
- World-map node metadata exists for promoted maps.
- Dungeon/boss IDs match current dungeon/boss data.
- Required map IDs exist.
- Safe zones have service plan if they include stations.

Editor validation:

- No orphaned objects.
- No duplicate IDs.
- No objects on locked/system layers with invalid types.
- No stale generated indexes in source maps.

### Validation Messages

Validation messages should include:

- Severity.
- Rule ID.
- Human-readable message.
- Object ID or layer ID.
- Suggested fix.
- "Focus" action.

Example:

```json
{
  "severity": "error",
  "ruleId": "slope.spawnOnSlope",
  "message": "Enemy spawn spawn_moss_lane_03 is attached to slope_pond_to_canopy_01.",
  "objectId": "spawn_moss_lane_03",
  "layerId": "entities_spawns",
  "suggestedFix": "Move the spawn to a flat combat platform or add a flat pad after the slope."
}
```

### Production Export Gates

Production export must fail when:

- Schema validation fails.
- Runtime map creation fails.
- Required references are missing.
- Required route exit is unreachable.
- Slope budget is exceeded.
- Any spawn/checkpoint/station sits on a slope.
- No player start or valid entry portal exists.
- Combat map has no valid enemy spawns.
- Map has no broad flat combat/rest zones.
- Map has missing production assets.

Draft save can allow these states, but the editor must keep warnings visible.

### Integration With Existing Validator

`build/validate-project-starfall-maps.js` already validates:

- `MAP_AND_LEVEL_DESIGN_GUIDE.md` existence and sections.
- Required terrain/ramp/prop/data/runtime paths.
- Map `layoutRole`.
- Slope counts.
- Max slope grade.
- Max slopes in a 1200 px window.
- Ramp connection count.
- Slope ramp connection coverage.
- Spawn points not on slopes.
- Minimum broad flat count.
- Combat maps having spawn points.

The editor validator should extend these rules rather than replacing them.

Add a new script:

```json
{
  "validate:starfall:editor-maps": "node build/validate-project-starfall-editor-maps.js"
}
```

Then update `npm test` or `test.js` to include editor map validation when editor source maps exist.

## 11. Runtime integration

### Integration Boundary

The editor should integrate at the map record boundary:

Editor JSON -> compiler/loader -> Starfall map record -> `createMapRuntime()` -> renderer/movement/gameplay.

Do not make the renderer or movement system read raw editor JSON directly. Keep `createMapRuntime()` as the runtime gate.

### Recommended New Modules

Data/schema:

- `js/games/project-starfall/data/map-editor-schema.js`
- `js/games/project-starfall/data/map-editor-loader.js`
- `js/games/project-starfall/data/map-editor-validation.js`
- `js/games/project-starfall/data/editor-map-manifest.js`

Build scripts:

- `build/compile-project-starfall-editor-maps.js`
- `build/validate-project-starfall-editor-maps.js`

UI:

- `js/games/project-starfall/ui/map-editor.js`
- `js/games/project-starfall/ui/map-editor-state.js`
- `js/games/project-starfall/ui/map-editor-tools.js`
- `js/games/project-starfall/ui/map-editor-layers.js`
- `js/games/project-starfall/ui/map-editor-palettes.js`
- `js/games/project-starfall/ui/map-editor-properties.js`
- `js/games/project-starfall/ui/map-editor-validation-panel.js`

Styles:

- `css/games/project-starfall-map-editor.css`

Content:

- `content/project-starfall/maps/*.json`
- `content/project-starfall/map-templates/*.json`
- `content/project-starfall/tilesets/*.json`

Tests:

- `tests/project-starfall-map-editor.test.js`
- `tests/fixtures/project-starfall/maps/*.json`

### Loader Responsibilities

`map-editor-loader.js` should:

- Parse editor JSON.
- Normalize schema versions.
- Resolve environment profiles.
- Resolve platform IDs.
- Sort platforms into runtime order.
- Convert platform IDs to platform indexes for runtime fields.
- Generate `rampConnections`.
- Generate `fieldComposition` from sections if not authored.
- Generate `spawnSections`.
- Attach `designIntent`.
- Attach `townServicePlan` for towns.
- Attach map assets.
- Return a map record compatible with `createMapRuntime()`.

### Data Pipeline Integration

Recommended flow:

1. Build script reads `content/project-starfall/maps/*.json`.
2. Build script validates editor maps.
3. Build script compiles them to `js/games/project-starfall/data/editor-map-manifest.js`.
4. `map-content.js` optionally merges compiled editor maps into `MAPS`.
5. Production maps are included only when `published: true`.
6. Draft maps can be loaded only in editor mode.

This keeps production gameplay stable while allowing editor experimentation.

### Runtime Preview Integration

Preview flow:

1. Editor creates in-memory editor map.
2. Loader converts it to a Starfall map record.
3. Validator runs.
4. If runtime-safe, call `createMapRuntime(mapRecord, viewport)`.
5. Renderer draws the runtime map.
6. Editor overlays collision, slopes, objects, camera frames, and validation.

Do not call `changeMap()` for every edit. Use runtime preview for editing, then use isolated playtest mode when the designer presses Play.

### Playtest Integration

Playtest flow:

1. Freeze current editor map into a runtime map record.
2. Create isolated playtest state.
3. Set player position from selected start/checkpoint/portal.
4. Use real movement/combat systems.
5. Disable normal save writes.
6. Show "Return to Editor" control.
7. On stop, discard playtest runtime and return to edit state.

### Runtime Loading Responsibilities

| Runtime Area | Current Project Hook | Editor Integration Rule |
| --- | --- | --- |
| Map file loading | `map-content.js`, `data/index.js`, compiled `Data.MAPS` | Editor JSON loads through `map-editor-loader.js`, then compiles into a normal map record. |
| Tilemap instantiation | `project-starfall-renderer-pixi.js` terrain strips and environment atlases | Do not create a separate tilemap runtime first; instantiate visual terrain through existing terrain/ramp/prop atlas metadata. |
| Collision creation | `createRuntimePlatformLayout()` in `map-runtime.js` | Compile editor platforms into `platforms`; never infer collision only from art. |
| Slope collision creation | `shape: "slope"`, `y2`, `getPlatformSurfaceY()` | Compile slopes as explicit platform objects and generate ramp connections before preview/export. |
| Entity spawning | Runtime alignment helpers for stations, portals, NPCs, climbables | Compile object layers into `stations`, `questNpcs`, `portals`, and `climbables` with platform indexes resolved from IDs. |
| Enemy spawning | `spawnPoints`, `fixedEnemySpawns`, engine spawn/respawn logic | Compile enemy spawn objects to `spawnPoints`; preserve `sectionId`, `sectionLabel`, and weight. |
| Item spawning | Loot/drop systems; no mature authored pickup layer yet | Store `pickups[]` in editor maps; keep runtime-disabled until item pickup runtime is implemented. |
| Trigger setup | `map-mechanics.js`, onboarding/progress event hooks, future trigger runtime | Store `triggers[]` with event and target metadata; connect to runtime in a later phase. |
| Camera bounds setup | `viewport.js`, runtime `worldWidth`, `worldHeight`, camera snap/follow | Compile map dimensions and preview camera frames; validate start/exit frames. |
| Background/parallax setup | `renderBackground()`, `map.asset`, `environment`, palette | Choose existing backdrop/environment profile and expose palette/parallax metadata. |
| Save/load compatibility | `projectStarfallPrototypeSave.v1`, `serialize()`, `restore()` | Editor drafts use separate keys; playtest must not write normal save data. |
| Level transitions | `MAP_PORTALS`, `changeMap()`, `WORLD_MAP_EDGES` | Compile exits to portal records and validate destinations/route gates. |
| Error handling | Toasts, debug log, validation scripts | Invalid maps show validation errors, block production export, and fall back to `starfallCrossing` if a saved map ID is unavailable. |
| Development loading | Local JSON import/export, optional dev endpoint | Allow draft imports, localStorage autosaves, and optional localhost-only file writes. |
| Production loading | Compiled JS manifest merged into `MAPS` | Only `published: true` maps that pass validation should enter production data. |
| Hot reload | Not currently available for maps | Feasible in editor mode by recompiling in-memory JSON and rebuilding preview runtime without calling `changeMap()`. |
| Playtest mode | Current movement/combat/runtime systems | Use isolated cloned state and real runtime; discard state on return to editor. |

### Save/Load Compatibility

Normal player saves store `state.mapId`. If a player save references a custom map that is no longer available, restore should fall back safely.

Rules:

- Production map IDs must be stable.
- Draft map IDs must not be used in normal saves.
- Custom local maps should use a namespace such as `custom_<slug>`.
- Restore should detect missing map ID and fall back to `starfallCrossing`.
- Editor playtest should not write normal save data.

### Portal Integration

The editor should support two portal workflows:

Production route portal:

- Export to `MAP_PORTALS` or compiled map portals.
- Validate against `WORLD_MAP_EDGES`.
- Used by normal progression.

Local/editor portal:

- Stored inside the map JSON.
- Valid only for editor preview or local custom maps.
- Can target other local maps loaded in editor mode.

### World Map Integration

Promoted maps need world metadata:

- `WORLD_MAP_NODES`
- `WORLD_MAP_EDGES`
- `WORLD_ROUTES`
- `REGIONAL_TOWN_IDS` if a new town
- Portal fiction/labels in map presentation data

The editor can create draft world metadata, but production promotion should be manual or reviewed because world progression affects saves and route gates.

### Asset Loading Integration

`engine/assets.js` collects map and environment assets for preload. Editor maps must expose asset paths in a way asset collection can find.

Rules:

- Every map must specify environment profile or asset IDs.
- Every custom asset path must be under allowed project asset directories.
- Production build should include map source and assets in `public/`.
- If new asset folders are introduced, update `build/copy-to-public.js`.

### Renderer Integration

The editor should not duplicate `project-starfall-renderer-pixi.js`. It should:

- Let the existing renderer draw base map terrain.
- Draw editor overlays on top.
- Add overlay rendering for collision, platform IDs, sections, object icons, slope samples, camera frames, and validation.

Future renderer improvement:

- Wire ramp atlas rendering into `drawRampPlatformTerrain()` so slope visuals do not always use fallback polygons.
- Until then, the editor should warn when a map depends visually on many slopes.

## 12. Editor implementation plan

### Phase 0: Confirm Contracts

Goal:

- Lock down source format, export boundary, and validation rules.

Tasks:

1. Add `content/project-starfall/maps/` and `content/project-starfall/map-templates/`.
2. Add one minimal template for each role.
3. Add `map-editor-schema.js` constants.
4. Add `map-editor-loader.js` with platform ID resolution.
5. Add `map-editor-validation.js`.
6. Add fixtures for valid and invalid maps.
7. Add `build/validate-project-starfall-editor-maps.js`.
8. Add npm script `validate:starfall:editor-maps`.

Exit criteria:

- A valid JSON map compiles to a runtime map record.
- Invalid slope, spawn, portal, and asset references fail validation.
- `createMapRuntime()` succeeds for valid fixtures.

### Phase 1: JSON Import/Export And Runtime Preview

Goal:

- Make maps editable through files and previewable in the browser.

Tasks:

1. Add editor state module.
2. Add load/import JSON UI.
3. Add export/download JSON UI.
4. Add runtime preview renderer overlay.
5. Add validation panel.
6. Add current map clone function.
7. Add localStorage autosave.

Exit criteria:

- You can clone `greenrootMeadow`, export JSON, import it, preview it, and validate it.
- Editor drafts do not affect normal saves.

### Phase 2: Collision And Slope Editing

Goal:

- Enable core map shape editing.

Tasks:

1. Add platform selection.
2. Add platform creation/resizing.
3. Add slope creation/resizing/flipping.
4. Add flat pad and slope-density overlays.
5. Add platform ID generation.
6. Add object reattachment when platform IDs change.
7. Add undo/redo for geometry edits.

Exit criteria:

- You can build a simple valid combat map without editing raw JSON.
- Slope validator runs live.
- Runtime preview matches collision overlay.

### Phase 3: Object Placement

Goal:

- Place gameplay objects visually.

Tasks:

1. Add object palette.
2. Add player start placement.
3. Add enemy spawn placement.
4. Add portal placement.
5. Add climbable placement.
6. Add station/NPC placement.
7. Add checkpoint, pickup, hazard, and trigger draft layers.
8. Add property panel dropdowns from game data.

Exit criteria:

- You can create a playable start-to-exit map with enemies and portals.
- Object property editing does not require raw JSON.

### Phase 4: Terrain Visual Editing

Goal:

- Make maps look polished without manual asset math.

Tasks:

1. Add biome palette.
2. Add terrain visual assignment per platform.
3. Add prop painting/scattering.
4. Add landmark placement.
5. Add background/profile picker.
6. Add visual/collision alignment warnings.
7. Add first-screen preview export.

Exit criteria:

- You can make a map visually distinct using existing Starfall assets.
- Terrain and collision remain aligned.

### Phase 5: Playtest Mode

Goal:

- Test maps with real movement and combat.

Tasks:

1. Add isolated playtest state.
2. Start from player start, checkpoint, portal, or selected section.
3. Spawn enemies from edited spawn points.
4. Allow admin debug tools in playtest.
5. Return to editor without saving player progress.
6. Add playtest warnings from runtime observations.

Exit criteria:

- You can edit a map, press Play, traverse it, fight enemies, return to editor, adjust, and repeat.

### Phase 6: Production Promotion

Goal:

- Promote editor maps into the real data pipeline safely.

Tasks:

1. Compile published editor maps to JS manifest.
2. Merge compiled maps into `MAPS`.
3. Add optional world node/edge metadata.
4. Add portals to production route data.
5. Add full test coverage.
6. Update build copy if new assets are used.

Exit criteria:

- Published editor maps appear in the game and pass `npm test`.
- `npm run validate:starfall:maps` and `npm run validate:starfall:editor-maps` pass.

### Phase 7: Optional External Editor Import

Goal:

- Support Tiled or LDtk if needed.

Tasks:

1. Create Starfall Tiled/LDtk templates.
2. Define object/entity templates.
3. Add importer to Starfall editor JSON.
4. Add import validation.
5. Document external editor conventions.

Exit criteria:

- External maps import into Starfall JSON and pass the same runtime validation.

## 13. Technical implementation guidance

### Keep The Runtime Shape Stable

Do not refactor the entire map runtime for the editor. The editor should adapt to the current runtime:

- Current runtime wants map records.
- Current maps use `platforms`, `rampConnections`, `climbables`, `spawnPoints`, `stations`, `questNpcs`, and `portals`.
- Current renderer expects `environment`, `fieldComposition`, `townScene`, and platform `terrainVisual`.

The editor loader is the translation layer.

### Add Platform IDs Everywhere

Current platform indexes are fragile. The editor source must use platform IDs.

Compiler behavior:

1. Sort/normalize platforms.
2. Assign indexes.
3. Build `{ platformId -> platformIndex }`.
4. Convert all object references.
5. Keep `platformId` in runtime debug metadata where useful.

### Use Existing Geometry Helpers

Use existing helpers wherever possible:

- `makePlatformDef()`
- `makeSlopePlatformDef()`
- `getPlatformDefSurfaceY()`
- `makeRampConnections()`
- `isSlopePlatform()`
- `getPlatformSurfaceY()`
- `createMapRuntime()`

Do not duplicate slope math in the editor if the core helpers can be reused.

### Avoid Direct DOM File Writes

In a static browser app, the editor cannot safely write repo files directly.

Use:

- localStorage autosaves for drafts.
- Download/export for source JSON.
- Import JSON for loading.
- Optional local dev endpoint in `build/dev.js` later.

### Local Dev Save Endpoint

Optional later feature:

- Add a dev-only route to `build/dev.js`, such as `POST /__starfall-map-editor/save`.
- Only enable on localhost.
- Validate payload before writing.
- Write only under `content/project-starfall/maps/`.
- Reject paths with `..`, absolute paths, or unexpected extensions.

### Editor State Shape

Recommended state:

```json
{
  "activeMapId": "greenrootMeadowEditorTest",
  "drafts": {},
  "selectedLayerId": "collision_solid",
  "selectedObjectIds": [],
  "tool": "select",
  "snap": {
    "enabled": true,
    "grid": 16
  },
  "overlays": {
    "collision": true,
    "slopes": true,
    "camera": true,
    "validation": true
  },
  "history": {
    "undo": [],
    "redo": []
  },
  "validation": {
    "messages": []
  },
  "playtest": {
    "active": false,
    "startObjectId": ""
  }
}
```

### Undo/Redo Implementation

Use operations:

```json
{
  "type": "moveObject",
  "objectId": "spawn_moss_lane_01",
  "before": { "x": 920, "platformId": "ground_entry" },
  "after": { "x": 1040, "platformId": "ground_entry" }
}
```

Do not push a full map snapshot for every mouse move. Collapse drag operations into one history entry.

### Coordinate System

Use world coordinates:

- X increases right.
- Y increases downward.
- Platform `y` is the collision surface for flat platforms.
- Slope `y` is left endpoint surface.
- Slope `y2` is right endpoint surface.
- Platform `h` is thickness/body depth for collision/rendering.

UI should show:

- World X/Y under cursor.
- Surface Y under cursor.
- Platform ID under cursor.
- Section under cursor.
- Camera frame bounds.

### Grid Sizes

Use:

- 16 px fine grid.
- 32 px object grid.
- 64 px terrain cell grid.
- 128 px platform tuning grid.
- 1280 px camera section grid.

### Export Compiler Pseudocode

```js
function compileEditorMap(editorMap, data) {
  const normalized = normalizeEditorMap(editorMap);
  const validation = validateEditorMap(normalized, data);
  if (validation.errors.length) {
    return { ok: false, errors: validation.errors };
  }

  const platforms = normalized.platforms.map(compilePlatform);
  const platformIndexById = createPlatformIndex(platforms);
  const spawnPoints = normalized.spawns.enemySpawns.map((spawn) =>
    compileSpawnPoint(spawn, platformIndexById)
  );
  const climbables = normalized.climbables.map((climbable) =>
    compileClimbable(climbable, platformIndexById)
  );
  const portals = normalized.portals.map((portal) =>
    compilePortal(portal, platformIndexById)
  );
  const stations = normalized.stations.map((station) =>
    compileStation(station, platformIndexById)
  );
  const questNpcs = normalized.npcs.map((npc) =>
    compileNpc(npc, platformIndexById)
  );

  return {
    ok: true,
    map: {
      id: normalized.id,
      name: normalized.name,
      levelRange: normalized.levelRange,
      safeZone: normalized.role === 'town',
      isDungeon: normalized.role === 'dungeon',
      bossRoom: normalized.role === 'bossArena',
      layoutRole: normalized.role,
      palette: normalized.palette,
      asset: normalized.environment.backgroundAsset,
      environment: resolveEnvironmentProfile(normalized, data),
      purpose: normalized.metadata.purpose,
      enemies: normalized.enemies,
      platforms,
      rampConnections: makeRampConnections(platforms),
      climbables,
      spawnPoints,
      fixedEnemySpawns: normalized.spawns.fixedEnemySpawns || [],
      stations,
      questNpcs,
      portals,
      fieldComposition: compileFieldComposition(normalized),
      designIntent: normalized.metadata.designIntent,
      spawnSections: compileSpawnSections(normalized.sections),
      editorSource: {
        schema: normalized.schema,
        version: normalized.version,
        fileSlug: normalized.fileSlug
      }
    }
  };
}
```

### Script Integration

Add scripts:

```json
{
  "compile:starfall:editor-maps": "node build/compile-project-starfall-editor-maps.js",
  "validate:starfall:editor-maps": "node build/validate-project-starfall-editor-maps.js"
}
```

Recommended test sequence after editor map changes:

```bash
npm run validate:starfall:editor-maps
npm run validate:starfall:maps
npm test
```

Run `npm run build` when compiled data, public assets, script lists, CSS bundles, or public output change.

### Page Script Integration

`pages/games/project-starfall.html` currently loads many Starfall modules explicitly. If editor modules are added:

- Load schema/loader/validation before UI editor modules.
- Load UI editor modules after current UI modules.
- Keep editor modules deferred.
- Gate editor activation by query parameter/admin state so normal load cost stays low.

Example order:

1. Core/data/runtime modules.
2. Map editor schema/loader/validation.
3. Existing UI modules.
4. Map editor UI modules.
5. Main bootstrap.

### CSS Integration

Add editor CSS as a separate route-specific file:

- `css/games/project-starfall-map-editor.css`

Follow repo conventions:

- 2-space indentation.
- Use existing visual language.
- Keep dense tool UI.
- Avoid marketing-style panels.
- Use clear icons and tooltips.
- Do not make cards inside cards.

### Security And Production Safety

Rules:

- Editor mode must not expose arbitrary file writes in production.
- Import JSON must validate schema and reject scripts/functions.
- Asset paths must be relative and under allowed asset roots.
- Dev save endpoint must be localhost-only.
- Editor maps should not include secrets or remote URLs.

## 14. Testing plan

### Unit Tests

Add tests for:

- Schema normalization.
- Platform ID uniqueness.
- Platform ID to index resolution.
- Flat platform compilation.
- Slope platform compilation.
- Ramp connection generation.
- Spawn point compilation.
- Portal compilation.
- Missing reference detection.
- Missing asset detection.
- Invalid role detection.
- Version migration.

Recommended fixtures:

- `valid-greenroot-editor-map.json`
- `valid-town-editor-map.json`
- `valid-dungeon-editor-map.json`
- `invalid-missing-player-start.json`
- `invalid-missing-platform-reference.json`
- `invalid-spawn-on-slope.json`
- `invalid-too-many-slopes.json`
- `invalid-slope-grade.json`
- `invalid-missing-portal-destination.json`
- `invalid-missing-environment-asset.json`

Recommended fixture structure:

```text
tests/
  fixtures/
    project-starfall/
      editor-maps/
        valid/
          valid-greenroot-editor-map.json
          valid-town-editor-map.json
          valid-dungeon-editor-map.json
        invalid/
          invalid-missing-player-start.json
          invalid-missing-platform-reference.json
          invalid-spawn-on-slope.json
          invalid-too-many-slopes.json
          invalid-slope-grade.json
          invalid-missing-portal-destination.json
          invalid-missing-environment-asset.json
```

### Runtime Tests

For each valid fixture:

- Compile to map record.
- Call `createMapRuntime()`.
- Assert platforms exist.
- Assert runtime world width/height are valid.
- Assert spawn points align.
- Assert portals align.
- Assert climbables align.
- Assert platform graph exists.
- Assert training route contract is valid for combat maps.

### Validator Tests

Test all core validation rules:

- Slope budget.
- Slope grade.
- Slope density window.
- Spawn on slope.
- No broad flats.
- Missing platform ID.
- Duplicate object IDs.
- Missing section ID.
- Missing enemy ID.
- Missing item ID.
- Portal destination missing.
- Missing environment profile.
- Collision/visual mismatch.

### Editor UI Tests

If browser automation is available later:

- Open editor mode.
- Create a new map from template.
- Place platform.
- Place slope.
- Place spawn.
- Place portal.
- Validate.
- Export JSON.
- Import JSON.
- Confirm imported map matches expected structure.

### Playtest Tests

Manual or automated:

- Start playtest from player start.
- Start from selected checkpoint.
- Start from selected portal.
- Walk across every platform.
- Walk up/down every slope.
- Jump from flat to slope and slope to flat.
- Spawn enemies from each section.
- Use portal transitions in editor-only mode.
- Stop playtest and verify normal save remains unchanged.

### Regression Tests

Existing commands to keep passing:

```bash
npm run validate:starfall:maps
npm test
npm run test:starfall:smoke
```

Add editor map validation to `npm test` once editor fixtures exist.

### Visual Review Tests

Before promoting a map:

- Screenshot first camera frame.
- Screenshot every 1280 px section.
- Screenshot every portal/checkpoint.
- Screenshot slope-heavy sections with collision overlay.
- Confirm background does not compete with gameplay.
- Confirm slope visuals match slope collision.
- Confirm props do not hide spawns, portals, stations, NPCs, or pickups.

### Save/Load Tests

Test:

- Normal player save/load unaffected by editor drafts.
- Editor draft autosaves and reloads.
- Export/import preserves IDs and metadata.
- Missing custom map fallback returns normal save to `starfallCrossing`.
- Editor playtest does not write to `projectStarfallPrototypeSave.v1`.

### Manual UX Test Checklist

Before calling the editor usable:

- Open editor mode from the Worldwright surface and from the query parameter.
- Create a new map from each template.
- Clone an existing runtime map and confirm the clone is clearly marked as a draft.
- Paint terrain without accidentally editing collision.
- Edit collision without accidentally changing visual-only layers.
- Add a slope and read its grade, endpoints, flat pads, and validation warnings.
- Place player start, checkpoint, portal, enemy spawn, pickup, hazard, trigger, NPC, and note markers.
- Edit every placed object's properties from the property panel without touching raw JSON.
- Undo and redo placement, deletion, movement, slope edits, and property edits.
- Multi-select platforms and attached objects, move them, and confirm attachments remain valid.
- Toggle every layer and overlay.
- Trigger validation errors intentionally and confirm messages focus the right object.
- Start playtest from player start, selected checkpoint, selected portal, and selected spawn section.
- Return from playtest and confirm the editor state is unchanged.
- Export JSON, re-import it, and confirm the map is identical.
- Confirm normal player save/load is unaffected by editor drafts.

### Build Tests

When editor maps are promoted:

```bash
npm run compile:starfall:editor-maps
npm run validate:starfall:editor-maps
npm run validate:starfall:maps
npm test
npm run build
```

## 15. Documentation requirements

### Required Documents

Create or update:

- `MAP_EDITOR_INTEGRATION_GUIDE.md`
- `MAP_AND_LEVEL_DESIGN_GUIDE.md`
- `docs/project-starfall-map-editor-user-guide.md`
- `docs/project-starfall-map-editor-schema.md`
- `docs/project-starfall-map-editor-validation.md`
- `docs/project-starfall-map-editor-promotion-checklist.md`

### Required How-To Pages

Add these topics either as sections in the user guide or as small focused docs:

- Opening the map editor.
- Creating a new map from a template.
- Keyboard shortcut reference.
- Tile layer reference.
- Object placement reference.
- Slope usage rules.
- Validation error reference.
- Adding a new tileset or environment atlas to the editor palette.
- Adding a new enemy type to the editor object palette.
- Adding a new item/pickup type to the editor object palette.
- Adding a new hazard, switch, door, moving platform, or trigger type.
- Promoting a draft map into production data.
- Troubleshooting common editor problems.

### User Guide Contents

The editor user guide should cover:

- Opening editor mode.
- Creating a new map.
- Loading/cloning maps.
- Saving drafts.
- Exporting/importing JSON.
- Editing platforms.
- Editing slopes.
- Placing objects.
- Using validation.
- Playtesting.
- Promoting maps.
- Troubleshooting common errors.

### Schema Documentation

The schema doc should cover:

- Every top-level field.
- Every layer type.
- Every object type.
- Required and optional fields.
- ID conventions.
- Versioning.
- Migration rules.
- Example maps.

### Validation Documentation

The validation doc should cover:

- Rule IDs.
- Severity definitions.
- What blocks export.
- How to fix each error.
- Slope budgets.
- Reachability requirements.
- Runtime preview requirements.

### Promotion Checklist

Before a map becomes production content:

- Source JSON validates.
- Runtime map validates.
- Slope budget passes.
- Broad flat requirements pass.
- Spawn sections pass.
- Portals pass.
- World route data is reviewed.
- Map matches GDD tone and biome identity.
- Map matches `MAP_AND_LEVEL_DESIGN_GUIDE.md`.
- Playtest from start/checkpoints/portals passes.
- Screenshots reviewed.
- `npm test` passes.
- `npm run build` passes if public output/assets changed.

### In-Editor Help

Add concise tooltips:

- Platform tool: "Draw flat collision platforms. Use for combat and rest space."
- Slope tool: "Draw explicit slope collision. Use sparingly for major transitions."
- Spawn tool: "Place enemy spawn points on flat platforms."
- Portal tool: "Place exits and route transitions."
- Section tool: "Mark route, spawn, and mechanic sections."
- Validation panel: "Fix errors before production export."

Avoid long tutorial text inside the canvas. Keep detailed guidance in docs.

## 16. Final Markdown deliverable

This guide is the implementation manual for adding a practical Project Starfall map editor. The recommended path is:

1. Use versioned Starfall JSON as the editor source format.
2. Compile editor JSON into existing Starfall map records.
3. Validate maps through shared browser and Node validation.
4. Add a developer-only Worldwright map editor mode.
5. Preview and playtest through the real runtime.
6. Keep slope editing explicit, limited, and validated.
7. Add Tiled or LDtk import only after the native schema is stable.

The editor must serve the current game, not a generic tilemap ideal. Project Starfall maps are platform-geometry action RPG spaces with route metadata, spawn sections, portal progression, environment atlases, and slope-sensitive movement. The editor is successful when you can customize those maps visually, safely, and repeatedly without touching raw map JS.
