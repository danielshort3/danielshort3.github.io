# Project Starfall Map And Level Design Guide

This guide is for designing, cleaning up, and expanding Project Starfall maps so they look authored, readable, and professional. It is based on the current project files, with the game guide Markdown files treated as the source of truth for tone and player experience.

Primary files reviewed:

- `project_starfall_gdd_v0_5.md`
- `docs/project-starfall-map-optimization-audit.md`
- `docs/project-starfall-map-optimization-implementation.md`
- `img/project-starfall/asset-prompts.md`
- `js/games/project-starfall/data/map-catalog.js`
- `js/games/project-starfall/data/map-layouts.js`
- `js/games/project-starfall/data/map-geometry.js`
- `js/games/project-starfall/data/map-builders.js`
- `js/games/project-starfall/data/map-assembly.js`
- `js/games/project-starfall/data/map-presentation.js`
- `js/games/project-starfall/data/world.js`
- `js/games/project-starfall/data/environment.js`
- `js/games/project-starfall/core/geometry.js`
- `js/games/project-starfall/engine/movement.js`
- `js/games/project-starfall/engine/map-runtime.js`
- `js/games/project-starfall/engine/viewport.js`
- `js/games/project-starfall/project-starfall-renderer-pixi.js`
- `img/project-starfall/environment/terrain/*`
- `img/project-starfall/environment/ramps/*`
- `img/project-starfall/environment/props/*`
- `all-map-terrain-format-contact-sheet.png`
- `grass-map-redo-contact-sheet.png`

## 1. Project-Specific Map Design Analysis

### Game Type

Project Starfall is a 2D side-scrolling action RPG prototype with MMO-inspired structure. The GDD defines a side-view 2D action perspective, readable party positioning, skill timing, dodge windows, multi-platform fields, vertical routes, ladders, environmental hazards, boss arenas, towns, dungeons, and repeatable field maps.

The visual target from the GDD and `img/project-starfall/asset-prompts.md` is "Starlit Frontier Fantasy": luminous, readable, painterly fantasy with clean silhouettes, blue/gold star magic, warm guild-town details, crafted structures, runes, lanterns, crystals, and strong foreground/background separation. Maps should feel hand-placed and adventure-readable, not procedural.

### Current Level Design Style

Current map layouts are authored as JavaScript data, not external editor tilemaps. The main authored source is `js/games/project-starfall/data/map-catalog.js`, with layout helpers in `map-layouts.js`, platform helpers in `map-geometry.js`, and assembly transforms in `map-assembly.js`.

The current level style is lane-based:

- A main ground or bottom route.
- One or more higher platform lanes.
- Side lanes for ranged enemies, rewards, or party roles.
- Ramps/slopes connecting many lane changes.
- Climbables and jump links layered on top of platform routes.
- Town hubs with service platforms.
- Boss arenas with repeated shelf patterns.

This is structurally correct for Starfall, but many maps currently share a similar "flat lane plus ramp chain" shape. The map optimization audit already called out sameness and weak authored route identity, especially in later maps. The current slope integration amplifies that issue because slopes appear in similar quantities and similar purposes across biomes.

### Current Terrain Structure

The terrain system is platform-definition driven:

- Flat platforms are represented as `[x, y, width, height]` arrays or object definitions.
- Slope platforms are object definitions with `shape: "slope"` and a `y2` endpoint.
- `makeSlopePlatformDef(x, y, y2, width, height, visual)` in `map-geometry.js` creates gameplay slope surfaces.
- `getPlatformDefSurfaceY()` samples slope height linearly.
- `map-runtime.js` builds `rampConnections` and path graph links from slope platforms.
- `movement.js` and `core/geometry.js` resolve actor landing against slope surface Y.

This means slopes are not only visual decoration. They directly affect traversal, enemy pathing, camera motion, and combat footing.

### Geometry And Spawn Authoring Contract

Maps must now declare how their collision geometry is produced:

- `geometryMode: "authored"` preserves the map's declared platforms, climbables, and spawn points. Use it for bespoke rooms and layouts whose exact platform arrangement matters.
- `geometryMode: "generated"` opts into a named `geometryGenerator`. Use it only when the selected builder is the intentional source of truth for that map.
- Every published platform receives a stable `id`. Spawn points and ramp connections also retain the matching platform IDs so later geometry edits do not silently redirect a mob territory.

Combat population is authored through `spawnGroups`, not by relying on the order of the map-wide `enemies` array. A group declares a section label, `platformIds`, weighted enemy types, target population, respawn cadence, leash distance, party scaling, and actor traversal permissions. The runtime keeps replacements inside that territory and applies the group's cadence independently. Maps without explicit groups receive backward-compatible groups from their spawn sections, but priority fields should always use named groups.

Use platform territories to create readable roles:

- Ground melee packs belong on broad lower lanes.
- Ranged packs belong on ledges with an obvious approach.
- Flyers belong in open upper airspace.
- Support or rare enemies belong in smaller, visually distinct pockets.
- Regroup sections should have lower pressure than the neighboring combat territories.

The world map and minimap are player-facing summaries of this contract. Keep group labels short, make portal destinations explicit, and expose approximate population and respawn timing before a player enters a field. Exact rare-drop odds remain a Monster Guide mastery reward; the map panel should still reveal common/signature drop categories immediately.

### How Players Move Through Maps

Player movement is side-scroller traversal through horizontal lanes, jumps, drops, climbables, and slope connections. The engine uses default jump-link limits around 128 px vertical jump and 300 px drop. Camera behavior in `viewport.js` targets the player with a forward-biased X anchor and smoothed Y movement. Because the camera follows vertical movement, long chains of slopes can create constant subtle camera bobbing.

Movement readability depends on clear flat landings, predictable route options, and combat spaces where the player can read enemies without fighting terrain.

### Current Slope Usage

Map generation previously used slopes as standard connectors between nearly every lane. The current builders cap that pattern and combine ramps with climbables, drops, hops, and broad flat tiers. New maps should keep that variety instead of reintroducing a ramp at every lane change.

Current slope counts from the built map data:

| Map Type | Current Slope Pattern | Design Risk |
| --- | --- | --- |
| Town hubs | 4 slopes with 8-9 broad flats | Keep services flat and ramps outside interaction clusters. |
| Starter/standard fields | 5-6 slopes with at least 10 broad flats | Good baseline; use ramps for section transitions, not every tier change. |
| Party/farm fields | 6 slopes with 10-13 broad flats | Maintain clear regroup pockets and lane-specific mob territories. |
| Vertical/deep fields | 6 slopes plus climbables | Let ladders, vines, lifts, ledges, and drops carry most vertical identity. |
| Endless Rift | 5 slopes with 14 broad flats | Preserve quadrant breaks, floating islands, and rune-style transitions. |
| Dungeon/boss arenas | 4 slopes with 7 broad flats | Keep the primary boss lane flat and mechanic-readable. |

Current slope grades are also aggressive in some biomes. Greenroot has gentler 64-128 px rises over about 260 px width. Cinder, Quarry, Stormbreak, and Endless Rift often use 160-180 px rises over 240-300 px, which is visually steep and should feel dramatic, not routine.

### Current Map Readability

The screenshot/contact-sheet assets show that Project Starfall is strongest when maps have:

- Wide flat playable lanes.
- Readable painterly silhouettes.
- Clear foreground terrain against atmospheric backgrounds.
- Distinct biome palettes.
- Strong horizontal shelf structure.

Readability gets weaker when too many ramps appear close together, because the player sees repeated diagonal terrain with similar angles. This reduces the clarity of "main path", "optional path", "combat space", and "route transition".

### Current Visual Polish

The terrain and prop atlases are strong enough to support polished maps. The environment sheets under `img/project-starfall/environment/terrain/`, `props/`, and `ramps/` are biome-specific and consistent with the GDD. The issue is not asset quality; it is composition.

Important implementation note: `data/environment.js` registers biome ramp atlases and `project-starfall-renderer-pixi.js` now renders slope cells from those atlases in `drawRampPlatformTerrain()`, retaining the fallback only when an asset is unavailable. Slope budgets still matter: repeated ramps create visual rhythm and camera movement even when the artwork is polished.

### Current Tile And Terrain Consistency

Project Starfall already has distinct terrain identities by map:

- Greenroot: soft grassy platforms, roots, vines, warm meadow shapes.
- Rustcoil: stone/metal industrial ruins and gearwork structures.
- Cinder: basalt, lava glow, ashglass, furnace silhouettes.
- Frostfen: icy shelves, snowy marsh flats, pale cyan highlights.
- Stormbreak: cliff shelves, wind-cut rocks, storm masts.
- Astral/Eclipse: runes, archive platforms, star magic, rift islands.

Terrain consistency is weakened when the same slope composition is used across all of these areas. Rustcoil should not climb like Greenroot. Cinder should not have every lane connected by the same ramp logic as Frostfen. Astral maps should use rune platforms, lifts, and floating shelves more than natural hills.

### Current Risks To Guard Against

- Generated maps can become samey if different biomes reuse the same lane grammar without a named route purpose.
- Boss arenas still share a common skeleton and need mechanic-specific iteration over time.
- Steep slopes should remain memorable terrain events rather than ordinary connectors.
- Long vertical chains can still cause camera oscillation even when traversal validation passes.
- Enemy and pickup placement should remain off slope surfaces.
- Platform IDs referenced by spawn groups must stay stable when geometry is edited.
- World-map population, cadence, and drop previews must be updated whenever a spawn group changes.

## 2. Core Level Design Principles For Project Starfall

### Rule 1: Design The Main Route First

Every map must have a main route that can be understood in under 3 seconds after entering a section.

- Use the widest, clearest, most continuous terrain for the main route.
- Keep the main route mostly flat or gently stepped.
- Use lighting, background landmarks, platform width, and enemy facing to point forward.
- Do not make the main route depend on reading repeated diagonal terrain.
- Branches should visibly leave the main route and visibly return or reward.

### Rule 2: Separate Main Route, Optional Route, And Combat Route

Each route type should have a different terrain language:

- Main route: broad, readable, low-friction, mostly flat.
- Optional reward route: narrower, slightly higher, visually hinted by crystals, chests, glowing plants, ore, runes, or props.
- Combat route: broad flat lane, predictable edges, enough room to dodge.
- Mobility route: ladders, vines, lifts, one-way drops, short ramps, or stepped platforms.

Avoid making all routes a chain of slopes. If every path is diagonal, no route feels designed.

### Rule 3: Give Combat Flat Ground

Combat in Starfall depends on readable spacing, dodging, ranged pressure, party positioning, and skill timing. For that reason:

- Standard combat pockets need at least 640 px of mostly flat ground.
- Party combat areas should target 900-1400 px of stable flat ground.
- Boss arenas should make the primary lane flat for most of its length.
- Sloped combat should be intentional and rare, such as a rolling boulder hazard, uphill chase, or ranged high-ground encounter.
- Do not place important attack tutorials, elite pulls, healer pockets, or boss mechanics on awkward slopes.

### Rule 4: Use Verticality As A Reward And Threat Structure

Verticality should create decisions:

- Bottom route: safe, direct, more melee pressure.
- Middle route: standard combat and progress.
- Upper route: ranged enemies, resources, scouting, or shortcuts.
- Drop route: fast but risky.
- Climb route: slower but safer or reward-rich.

Verticality should not be "three flat lanes connected by identical slopes" in every map. Use biome-appropriate connectors.

### Rule 5: Teach, Test, Vary

Every new level mechanic should follow this sequence:

1. Teach in a safe area with no punishment.
2. Test with one enemy or one hazard.
3. Combine with movement, verticality, or route choice.
4. Vary the setup once before the exit.
5. Reward mastery with a visible pickup, shortcut, or cleaner combat angle.

### Rule 6: Make Enemy Placement Serve The Terrain

Enemy placement should explain the terrain rather than clutter it.

- Place melee enemies on flat lanes where the player can read distance.
- Place ranged enemies on upper ledges with clear approach routes.
- Place flyers above open pockets, not under low ceilings or directly over steep ramps.
- Place brutes near chokepoints only when there is a readable dodge or retreat space.
- Use enemies to mark optional routes, not to hide the main path.
- Do not stack enemies at slope exits unless the slope is meant to be a deliberate ambush.

### Rule 7: Make Items Visible Before They Are Earned

Project Starfall benefits from RPG map readability: players should see rewards before they choose to pursue them.

- Put reward items on visible ledges, behind short traversal tests, or above the main route.
- Use props and lighting to frame rewards.
- Do not put key pickups on diagonal surfaces unless the pickup radius and actor movement have been tested.
- Put healing or recovery items on flat rests after challenge sections.

### Rule 8: Checkpoints Need Rest Space

Checkpoint zones should be visually and mechanically calm.

- Minimum flat ground: 480 px for solo maps, 720 px for party maps.
- No slopes crossing the checkpoint center.
- No enemies inside immediate checkpoint activation space.
- Place a landmark, banner, lantern, shrine, campfire, signal post, or starfall marker nearby.
- Let the camera settle before the next challenge.

### Rule 9: Hazards Must Be Readable Before They Punish

Hazards should have warning space.

- Lava, spikes, vents, lightning rods, falling ice, rift surges, or thorn traps need a visual tell.
- Place the first instance in a safe or low-pressure context.
- Avoid hazards at slope bottoms unless the player has a flat reaction pad.
- Never hide a hazard behind foreground decoration.

### Rule 10: Design For The Camera

The camera has a forward-biased X anchor and smoothed Y follow. Maps should avoid rapid vertical oscillation.

- Keep long traversal sections on a stable Y band.
- Break climbs into clear plateaus.
- Avoid slope-slope-slope chains that keep nudging the camera vertically.
- Put major reveals at flat overlooks, not halfway through a ramp.
- Avoid enemies that attack from just outside the vertical camera comfort zone.

### Rule 11: Preserve Visual Hierarchy

The player, enemies, hazards, interactables, and main route must read first.

- Gameplay terrain should have the strongest silhouette.
- Background details should be softer, lower contrast, and slower parallax.
- Foreground decorations should not cover collision edges.
- Props should cluster around landmarks and rest spaces, not evenly cover every platform.
- Use lighting accents to support route decisions.

### Rule 12: Pace Challenge, Rest, Exploration, And Reward

A finished Starfall field map should alternate between:

- Safe entry.
- Simple traversal.
- First combat read.
- Route branch or vertical choice.
- Harder encounter or hazard test.
- Rest/checkpoint.
- Reward or secret.
- Final mixed challenge.
- Clean exit.

Do not fill every screen with enemies, slopes, and decorations. Empty readable space is part of professional pacing.

## 3. Slope Design Guidance

### Core Slope Philosophy

Slopes in Project Starfall should be major terrain transitions, not default terrain filler. A slope should answer one of these design questions:

- How does the player move from one authored lane to another without breaking flow?
- How does the environment express a biome identity, such as a meadow hill, ashglass incline, glacier shelf, or wind-cut cliff?
- How does the map guide the player into a reveal, encounter, shortcut, or rest area?
- How does the slope change pacing in a way a flat platform cannot?

If a slope does not answer one of those questions, replace it.

### Slope Budget Targets

Use these targets when cleaning up existing maps or authoring new ones.

| Map Type | Initial Audit Tendency | Target Slope Count | Rule |
| --- | --- | --- | --- |
| Town hub | About 4 slopes | 2-4 | Slopes may connect service terraces, but all service stations need flat footing. |
| Starter field | About 9-12 slopes | 4-6 | Use slopes to introduce terrain flow, not to connect every lane. |
| Standard field | About 9-12 slopes | 5-8 | Keep slopes as section transitions. Use ledges, bridges, vines, and drops elsewhere. |
| Party farm field | About 10-12 slopes | 5-8 | Prioritize flat party combat and healer/rest pockets. |
| Vertical field | About 9 slopes | 6-9 only if biome-appropriate | Use climbables, lifts, shelves, and one-way drops for most vertical movement. |
| Dungeon | About 6 slopes | 2-4 | Dungeons should feel constructed, rooted, carved, or magical, not like rolling hills. |
| Boss arena | About 6 slopes | 2-4 | The primary boss lane must be flat. Slopes can connect side shelves or hazard vents. |
| Endless Rift | 13 slopes | 5-8 | Use floating islands, portals, rune steps, and quadrant breaks instead of repeated ramps. |

As a practical target, most maps should have at least two broad flat combat/platform sections for every one slope. If a map has one slope per broad lane, slope density is too high.

Executable enforcement lives in `build/validate-project-starfall-maps.js` and runs through:

```bash
npm run validate:starfall:maps
```

The same validation is also called from the `Project Starfall fast contracts` section in `test.js`. It fails if a map exceeds its slope budget, has a slope grade above 0.72, places more than 3 slopes in a 1200 px section, has slope platforms without matching `rampConnections`, places spawns on slopes, loses required map-design paths, or removes required sections from this guide.

### When To Use Slopes

Use a slope when:

- The player is moving between major height bands.
- The slope frames a biome identity, such as Greenroot meadow rolls or Cinder ashglass ramps.
- The slope creates a readable transition into a new section.
- The slope is part of a landmark reveal, such as climbing to see a meteor crater, forge, glacier spine, storm mast, or astral archive.
- The slope gives melee players a smooth approach to a higher enemy shelf.
- The slope supports a chase, escape, or flow section.
- The slope is visually anchored by cliffs, roots, pipes, ruins, bridges, crystals, or carved supports.

### When Flat Platforms Are Better

Use flat ground instead of a slope when:

- The player is expected to fight.
- The player is expected to aim, dodge, cast, or time skills.
- Enemies need stable patrol or readable attack spacing.
- A checkpoint, vendor, portal, class trainer, quest NPC, campfire, shrine, or interactable is nearby.
- The section teaches a mechanic.
- The player must make a route decision.
- A pickup, chest, or reward should be collected cleanly.
- The camera needs to settle before a reveal or encounter.

### When Stairs, Ledges, Ramps, Or Terrain Breaks Are Better

Use a different connector when the slope would make the map look repetitive.

- Use stairs for towns, ruins, archives, crafted structures, and boss facilities.
- Use ledges for platforming tests and short height changes.
- Use ladders or vines for Greenroot verticality and controlled climbs.
- Use bridges or catwalks for Rustcoil, Bandit Ridge, Quarry, and Stormbreak.
- Use lifts, rune platforms, or portals for Astral and Rift maps.
- Use one-way drops for shortcuts, danger routes, and return loops.
- Use cliffs for strong silhouettes, reveals, and biome drama.
- Use stepped ice shelves for Frostfen instead of long slippery ramps.

### How Many Slopes Are Too Many

A map has too many slopes when any of these are true:

- The main route reads as rolling hills instead of authored spaces.
- There is a slope on nearly every screen.
- Two or more slopes connect in sequence without a flat rest.
- Combat frequently happens on diagonal ground by accident.
- Pickups sit on slopes because there is no flat reward ledge.
- The camera constantly moves up and down during normal traversal.
- The map's biome identity is mostly "ramps" instead of roots, ruins, lava shelves, ice, storm cliffs, or astral platforms.
- The same slope angle appears three or more times in a row.

### Slope Density And Visual Quality

High slope density creates the "cheap map generator" look because diagonal terrain repeats a mathematical pattern. Professional 2D side-scroller terrain usually alternates:

- Flat rest.
- Strong vertical edge.
- Short connector.
- Landmark or prop cluster.
- Broad combat surface.
- Optional ledge.
- Another terrain type.

If slopes are the primary outline everywhere, the map loses authored rhythm. The player sees the generator, not the world.

### Slope Density And Gameplay Readability

Too many slopes reduce readability because:

- Ground contact is less predictable.
- Enemy spacing changes on the incline.
- Jump starts and landings feel less consistent.
- Pickups and interactables are harder to frame.
- Route hierarchy becomes ambiguous.
- Camera Y movement increases.
- The difference between path, decoration, and hazard becomes less clear.

### Slope Collision Feel

Because Project Starfall uses real slope collision through `getPlatformDefSurfaceY()`, every slope must be tested as gameplay.

Rules:

- Put 96-160 px of flat ground before entering a slope.
- Put 96-160 px of flat ground after exiting a slope.
- Do not end a slope directly at a platform edge unless the edge is the point of the challenge.
- Do not put enemy spawn centers, interactables, portals, or checkpoints on slope surfaces.
- Avoid steep slopes in ice movement maps unless sliding is the intended challenge.
- Avoid placing ranged enemies on slopes because their spacing and line reads are weaker.
- Test both uphill and downhill movement with player, melee enemy, ranged enemy, and follower/pathing behavior.

### Slope Connection Rules

A professional slope connection should have a clear start, body, and end.

- Start: flat approach, visual anchor, readable foot of slope.
- Body: one clean incline with no pickup clutter.
- End: flat landing, ledge, gate, bridge, cliff top, cave mouth, platform, or landmark.

Do not connect slope to slope unless building a deliberate long climb. Even then, break the climb with plateaus and scenery.

### How Slopes Should Transition Into Terrain Features

- Into cliffs: slope should end at a flat ledge before the cliff edge.
- Into platforms: slope should meet the platform flush, with no visual gap or collision bump.
- Into caves: slope should guide downward or upward into a mouth framed by rock, roots, crystals, or supports.
- Into bridges: slope should end before the bridge, not continue diagonally under it.
- Into structures: use stairs or platforms more often than natural slopes.
- Into boss arenas: slope should deposit the player on a flat arena floor before the encounter starts.
- Into reward ledges: slope should end before the reward, giving the player a flat collection space.

### Avoiding The Cheap Map Generator Look

To avoid the cheap generated look:

- Do not use slopes for every small height change.
- Do not repeat identical slope width/rise values across a whole map.
- Do not build "flat, slope, flat, slope, flat, slope" as the entire route grammar.
- Break long diagonals with flat rests, small cliffs, props, roots, pipes, ruins, caves, or ledges.
- Use one major slope per section, not three minor slopes in every cluster.
- Let biome-specific structures do the work: Greenroot vines, Rustcoil catwalks, Cinder basalt shelves, Frostfen ice ledges, Stormbreak wind cliffs, Astral rune platforms.
- Keep combat spaces flat unless slope combat is the named gimmick.

### Current Map Cleanup State

The initial audit found Greenroot Meadow at 12 slopes, Endless Rift at 13 slopes, most standard/vertical fields at 9-12 slopes, and dungeon/boss arenas at 6 slopes. The current generator in `js/games/project-starfall/data/map-builders.js` has been cleaned up so upper-lane ramp chains are no longer added by default. Current assembled-map targets are:

| Map Group | Current Validated State |
| --- | --- |
| Town hubs | 4 slopes, with widened high-lane town ramps and flat service spaces. |
| Greenroot Meadow | 6 slopes, keeping gentle meadow transitions while relying on flats, ledges, and vines. |
| Standard/vertical fields | 5-7 slopes, with repeated upper-lane chains replaced by climbables, ledges, and broad flats. |
| Deep fields | 6-7 slopes, keeping only major biome transitions and special approach ramps. |
| Endless Rift | 5 slopes, after removing the stacked central peak ramp that created a local slope pileup. |
| Dungeons | 4 slopes, with broad flat lanes and climbables carrying upper shelf access. |
| Boss arenas | 4 slopes, preserving side transitions while keeping the main combat floor flat. |

Continue using these cleanup rules for future maps:

- Greenroot Meadow: keep a few gentle meadow slopes, but convert excess connector ramps into flat lanes, short ledges, vines, and root steps.
- Thornpath Thicket: reduce repeated canopy ramping. Use vines, branch platforms, and drop loops.
- Rustcoil Ruins: replace natural slopes with stairs, broken catwalks, gear lifts, and scaffold steps.
- Cinder Hollow: keep only dramatic ashglass or basalt transitions, with flat safe pockets near vents.
- Bandit Ridge Camp: use bridges, camp terraces, rope platforms, and barricade steps instead of constant slopes.
- Oreback Quarry: use mine carts, scaffold lifts, ore ledges, and vertical shafts.
- Ashglass Pass: keep a few dangerous ashglass inclines, but avoid making the crossing a rolling ramp chain.
- Frostfen Outskirts and Glacier Spine: avoid slopes where ice movement makes control feel mushy. Use flat ice shelves and stepped glacier ledges.
- Stormbreak Cliffs: use cliffs, wind lifts, bridges, and ledges. Slopes should be wind-carved approaches to major cliffs.
- Astral Archive and Eclipse Frontier: use rune platforms, floating shelves, portal steps, and archive stairs.
- Endless Rift: use island gaps, portals, quadrant platforms, rift surges, and rune stairs instead of stacked central slopes.
- Boss arenas: keep slopes at or below 4. Primary combat floor stays flat.

## 4. Terrain Composition Rules

### Ground Silhouettes

The playable silhouette should be simple and readable at a glance.

- Use long flat tops for main lanes.
- Use strong vertical cuts for cliffs and drops.
- Use occasional diagonal slopes for meaningful transitions.
- Avoid noisy sawtooth outlines.
- Avoid micro bumps on playable collision.
- Keep decorative texture detail inside the terrain mass, not along the collision edge.

### Flat Areas

Flat areas are not boring. They are where the game breathes.

- Use flats for combat, checkpoints, NPCs, portals, pickups, and route decisions.
- Vary flat areas with background and prop composition, not collision noise.
- Add visual interest below and behind the flat collision surface.
- Keep the top edge clear enough that the player always knows where they can stand.

### Cliffs

Cliffs should be used to create strong map identity.

- Greenroot cliffs: root-wrapped dirt cuts, moss ledges, flower patches.
- Rustcoil cliffs: broken masonry, metal braces, pipes, gear fragments.
- Cinder cliffs: basalt columns, ember cracks, lava glow.
- Frostfen cliffs: ice slabs, snow caps, frozen roots.
- Stormbreak cliffs: wind-cut rock, banners, lightning rods.
- Astral cliffs: floating stone, rune plates, star-glass edges.

Use cliffs to punctuate terrain. Do not soften every cliff into a slope.

### Overhangs

Overhangs make maps look authored when used sparingly.

- Use overhangs to frame caves, enemy perches, secrets, or traversal gates.
- Keep underside decoration non-colliding unless clearly marked.
- Avoid low overhangs above combat lanes unless they intentionally block jumps.

### Platforms

Platforms should communicate purpose through width and placement.

- 160-320 px: small step, perch, or optional pickup.
- 320-640 px: short encounter, ranged perch, or route connector.
- 640-1200 px: standard combat lane.
- 1200+ px: party battle, boss floor, town plaza, or major rest area.

Do not make all platforms the same length. Vary length based on gameplay purpose.

### Ramps And Slopes

Use slopes as large readable gestures.

- Best use: section transitions and landmark approach.
- Worst use: every height change.
- Slope surfaces should be clean, quiet, and mostly free of props.
- Decorate around slopes, not on top of them.

### Steps

Steps are best for authored or artificial spaces.

- Towns: stairs to services, plazas, guild platforms.
- Rustcoil: metal steps, gearwork risers, scaffolds.
- Astral: rune stair plates, archive mezzanines.
- Boss arenas: symmetrical side steps or mechanic shelves.

Steps should be collision-clean. Avoid tiny step noise that catches movement.

### Caves

Caves should be strong transitions.

- Put a flat entrance lip before the cave mouth.
- Frame entrances with props and lighting.
- Use darker backgrounds and foreground edges to show enclosure.
- Do not run a slope directly into a dark hazard without a preview.

### Bridges

Bridges are excellent alternatives to slopes.

- Use bridges for Bandit Ridge, Rustcoil, Quarry, Stormbreak, and Astral archive spans.
- Keep bridge collision flat unless it is explicitly broken.
- Let bridges cross over optional lower routes.
- Use bridge supports to show route hierarchy.

### Ledges

Ledges should support jumps, rewards, and tactical high ground.

- Give ledges flat tops.
- Make ledge edges visually crisp.
- Place ranged enemies on ledges only if approach routes are readable.
- Use one-way drops from ledges to create loops.

### Foreground Elements

Foreground decoration should frame, not obscure.

- Use foreground roots, rocks, railings, crystals, banners, or machinery at the bottom of the screen or behind non-critical edges.
- Do not cover platform collision edges.
- Do not hide hazards or enemies.
- Use foreground clusters around landmarks, not evenly across every section.

### Background Layers

Backgrounds should support depth and biome identity.

- Keep gameplay-adjacent backgrounds low contrast.
- Reserve brightest background accents for route direction, landmarks, or exits.
- Use parallax to imply scale, not to compete with characters.
- Avoid background clutter directly behind combat silhouettes.

### Decorative Props

Props should explain the place.

- Greenroot: flowers, mushrooms, roots, camp signs, small lanterns.
- Rustcoil: gears, pipes, crates, pressure plates, brass frames.
- Cinder: furnace doors, vents, cooled lava ridges, ashglass shards.
- Frostfen: snowdrifts, ice crystals, frozen posts, pine silhouettes.
- Stormbreak: banners, ropes, lightning rods, cliff markers.
- Astral: books, rune pedestals, star lenses, floating tablets.

Use props in clusters with purpose. A prop every 200 px looks sprinkled rather than designed.

### Natural Landmarks

Every field map should have at least one landmark visible from the main route:

- Meteor crater.
- Giant root arch.
- Gear tower.
- Furnace silhouette.
- Glacier spine.
- Storm mast.
- Astral lens.
- Rift core.

Landmarks help players remember where they are and prevent maps from feeling like repeated lanes.

### Artificial Structures

Structures should imply builders and function.

- Town structures should be flat, stable, and service-friendly.
- Rustcoil and Quarry structures should use right angles, braces, steps, and catwalks.
- Astral structures should use clean platforms, rune supports, and symmetrical moments.
- Bandit camps should use rough bridges, barricades, tents, and rope platforms.

Use structures to replace slopes when the biome is built or civilized.

### Tile Variation

Variation should be controlled.

- Use alternate edge tiles every few platform lengths, not every tile.
- Keep collision edges clean and readable.
- Put heavy detail in terrain interiors, underside chunks, and background.
- Use biome accent tiles at section landmarks and rewards.
- Avoid visible repeating stamp patterns on long slopes.

### Making Terrain Look Hand-Designed

Before considering a section finished, ask:

- What is this section's gameplay purpose?
- What is the dominant terrain shape?
- What is the visual anchor?
- Where does the player rest?
- Where does the player fight?
- What makes this section different from the previous one?

If the answer is "another slope to another lane", redesign the section.

## 5. Tilemap And Collision Guidance

### Current Structure

Project Starfall currently uses JS-authored platform data instead of external tilemap files. Treat this as a layered tilemap system in code:

1. Gameplay collision layer: flat platforms, slope platforms, climbables, hazards.
2. Route graph layer: ramp connections, jump links, drops, AI traversal.
3. Visual terrain layer: terrain atlas cells, ramp art, trims, platform bodies.
4. Decoration layer: props, foreground pieces, background accents.
5. Encounter layer: spawn points, enemies, elites, boss mechanics.
6. Reward layer: pickups, resources, chests, shortcuts.
7. Camera/pacing layer: section anchors, reveals, rest areas, exit zones.

Keep those layers conceptually separate even when authoring in JavaScript.

### Collision Layer Rules

- Collision must be simpler than art.
- Do not trace every decorative bump.
- Flat platform collision should remain flat even if the art has grass, snow, pipes, roots, or crystal chips.
- Slope collision should match the visible top edge closely.
- Do not put decorative-only ramp art where the player cannot walk unless it is clearly background.
- Do not put invisible slope collision under flat-looking art.
- Check every slope with player movement, enemy movement, and follower/path graph behavior.

### Slope Collision Rules

- Slopes need clear visual start/end points.
- Slope collision must not overlap adjacent flat collision in a way that creates bumps.
- Slope endpoints must align with flat platform surfaces.
- Use flat pads before and after slopes.
- Avoid putting two slopes with different grades directly adjacent.
- Avoid steep slopes in maps with slippery movement unless intended.
- Test jumps from the slope midpoint, slope top, and slope bottom.
- Test landing on the slope from above.
- Test movement while attacking or being hit on the slope.

### Decorative Layer Rules

- Decoration should never define collision by accident.
- Props on playable terrain should have obvious non-blocking status unless designed as blockers.
- Foreground props must not hide hazards or platform edges.
- Background props can imply paths, but should not look more walkable than actual terrain.

### Foreground And Background Separation

Use this hierarchy:

1. Player/enemies/VFX.
2. Hazards and pickups.
3. Playable terrain top edges.
4. Foreground framing.
5. Midground structures.
6. Background scenery.
7. Atmosphere and sky.

If background platforms look as crisp as playable platforms, reduce contrast, blur detail, darken them, or shift color temperature.

### One-Way Platforms

If one-way/drop-through platforms are added or formalized:

- Use a consistent visual language, such as thinner ledges, wooden planks, branch shelves, metal catwalks, or rune plates.
- Do not mix one-way and solid platforms with identical art.
- Keep one-way platforms flat whenever possible.
- Use one-way platforms for optional paths, drop loops, arena side shelves, and vertical climbs.

### Naming Conventions

Use names that describe both map and purpose:

- `greenroot-meadow-main-flat-01`
- `greenroot-meadow-reward-ledge-01`
- `cinder-hollow-ashglass-slope-01`
- `rustcoil-ruins-catwalk-mid-02`
- `frostfen-outskirts-ice-shelf-safe-01`
- `stormbreak-cliffs-wind-lift-01`
- `astral-archive-rune-platform-03`

Avoid names like `platform7`, `ramp2`, or `hillA` in new authored data.

### Terrain Tile Requirements

Each biome tileset should include:

- Flat top tile.
- Left and right edge/cap tiles.
- Inner fill tiles.
- Underside tiles.
- Cliff face tiles.
- Corner transitions.
- Short ledge tiles.
- Long platform body variants.
- 2-3 slope angles only if the biome needs them.
- Slope start and slope end cap tiles.
- Flat-to-slope transition tiles.
- Slope-to-cliff transition tiles.
- Damaged/broken variants.
- Low-noise collision-readable versions.

Do not generate a slope set without matching flat transition caps.

### Autotile Or Rule-Tile Recommendations

If a future tilemap editor or rule-tile system is introduced:

- Autotile flat ground and cliff interiors.
- Do not fully autotile gameplay slopes without manual review.
- Give slopes strict rule variants: start, middle, end, upper cap, lower cap.
- Require hand-authored collision polygons for every slope variant.
- Keep decorative noise out of collision edges.
- Create biome-specific rule sets instead of one universal terrain rule.

### Testing Collision And Readability

For each map, test:

- Walk left/right across every flat-to-slope transition.
- Jump from the bottom, middle, and top of each slope.
- Land on each slope from above.
- Attack while standing on each slope.
- Let melee enemies chase across each slope.
- Let ranged enemies target across slope elevation changes.
- Verify the camera does not jitter or overcorrect on slope chains.
- Verify pickups do not slide visually or feel awkward to collect.
- Verify the visual edge matches the collision edge at runtime zoom.

### Automated Map Validation

Run this command after changing map data, map builders, slope helpers, terrain assets, spawn rules, or this guide:

```bash
npm run validate:starfall:maps
```

The validator reads the assembled `ProjectStarfallData.MAPS` output, not only the raw authored arrays. It checks the maps after `applyPartyPlayGeometry()` has generated platforms, `rampConnections`, climbables, and spawn points.

Validation currently enforces:

- `MAP_AND_LEVEL_DESIGN_GUIDE.md` exists and keeps the required sections.
- Required map-design source paths exist.
- Slope counts stay within role-specific budgets.
- Slope grade stays at or below 0.72.
- No 1200 px section contains more than 3 slopes.
- Every slope platform has a matching `rampConnections` entry.
- Spawn points do not reference missing platforms.
- Spawn points are not placed on slope platforms.
- Non-shop maps keep enough broad flat lanes for combat/rest space.

`npm test` also runs the validator through `test.js`, so map-design regressions should fail the normal project checks.

## 6. Map Layout Patterns

### Introductory Safe Area

Purpose: Let the player orient, see the biome, and start moving without pressure.

- Use broad flat ground.
- Avoid slopes in the first screen unless the biome is being introduced through one gentle landmark slope.
- Place a landmark in the background or midground.
- No enemies until the player has moved forward.
- Use a clear exit direction.
- Camera should settle immediately.

### Flat Traversal Section

Purpose: Give players movement confidence and visual rhythm.

- Use a clean flat lane with small prop clusters.
- Add one optional upper ledge or lower pocket.
- Avoid unnecessary ramps.
- Use a simple pickup trail only if it points to a route or reward.
- Good place for low-risk enemies.

### Gentle Slope Transition

Purpose: Move the player between major height bands while maintaining flow.

- Use one slope, not a chain.
- Give flat approach and flat exit pads.
- Anchor the slope with roots, rocks, rails, crystals, pipes, or supports.
- Avoid enemies on the slope body.
- Place the next encounter after the flat exit.

### Combat Arena

Purpose: Let combat mechanics breathe.

- Use a wide flat floor.
- Add side shelves only if they support enemy roles or boss mechanics.
- Avoid diagonal terrain in the primary combat lane.
- Place melee enemies on the floor, ranged enemies on ledges, flyers in open air.
- Put hazards at edges or in telegraphed lanes.
- Camera should not need to climb or descend during normal combat.

### Vertical Climb

Purpose: Create upward progression, shortcuts, and route decisions.

- Use ledges, ladders, vines, lifts, or rune platforms as the primary connectors.
- Use slopes only at the base or top of the climb to transition into a new section.
- Add rest ledges every few jumps.
- Keep enemies on flat shelves.
- Use background height cues to show progress.

### Optional Reward Path

Purpose: Reward curiosity without confusing the main path.

- Branch from a visible point on the main route.
- Use narrower ledges, a short climb, one slope, or a small hazard test.
- Show or hint at the reward before the branch.
- Return to the main route or provide a clear drop back.
- Avoid making optional paths look larger than the main route.

### Hazard Corridor

Purpose: Test timing and observation.

- Use predictable flat or stepped ground.
- Avoid slopes unless the slope is the named hazard mechanic.
- Give a safe preview of the hazard.
- Use warning props, lighting, particles, or terrain color shifts.
- Place rewards after the corridor, not inside unreadable hazard clutter.

### Rest Area

Purpose: Let the player recover, read the next section, or interact.

- Use flat ground.
- Add a campfire, shrine, lantern cluster, signal post, or similar landmark.
- No active enemies.
- No slopes through the center.
- Place optional lore, healing, checkpoint, or route preview here.

### Landmark Reveal

Purpose: Create memory and direction.

- Put the player on a flat overlook, bridge, or plateau.
- Use background composition to reveal the next goal.
- Do not trigger the reveal on a slope because the camera is still moving vertically.
- Frame the landmark with foreground edges, lighting, and negative space.

### Checkpoint Area

Purpose: Save progress and reset pacing.

- Use at least 480 px of flat terrain for solo maps and 720 px for party maps.
- Put the checkpoint on the main route.
- Keep enemies and hazards outside immediate activation range.
- Use a strong visual marker that matches the biome.
- Let the player see the next route direction.

### Final Challenge Section

Purpose: Combine the map's main ideas before the exit.

- Use the level's taught mechanic plus one variation.
- Keep the exit visible or strongly implied.
- Use slopes only if they were intentionally introduced earlier.
- Give flat combat and recovery space before the final pull.
- Make the final challenge memorable through layout, not just enemy count.

### Exit Or Transition Zone

Purpose: End cleanly and prepare the next destination.

- Use a flat portal, gate, bridge, cave mouth, town road, or boss door.
- Avoid placing the exit on a slope.
- Use lighting and props to make the exit unambiguous.
- Do not clutter the exit with loot, enemies, and slope transitions at the same time.

## 7. Biome And World-Specific Map Guidance

### Starfall Crossing

Terrain identity: warm guild-town hub built around a central meteor plaza.

- Use flat plazas, short stairs, service terraces, banners, lanterns, and starfall stonework.
- Slopes should be rare and gentle, used only for plaza-to-terrace transitions.
- Do not place class trainers, shops, portals, or upgrade stations on slopes.
- Use blue/gold star motifs and readable service silhouettes.
- Avoid wild rolling terrain inside the hub.

### Greenroot Wilds

Includes Greenroot Meadow, Thornpath Thicket, Bramble Depths, and Brambleking routes.

- Terrain identity: soft meadows, roots, vines, branch shelves, moss lanes, flower accents.
- Slopes are appropriate here, but only as gentle meadow rolls or root-backed transitions.
- Keep starter combat mostly flat.
- Use vines and branch ledges for verticality.
- Use thorn hazards on flat or stepped terrain first, then vary later.
- Backgrounds should show canopy, sky breaks, and soft forest depth.
- Avoid making every hill a slope. Greenroot should feel organic, not generated.

### Rustcoil Expanse

Includes Rustcoil Outpost, Rustcoil Ruins, Gearworks Vault, Titan Foundry, and Deepcore routes.

- Terrain identity: construct ruins, brass machinery, stone terraces, scaffolds, pipes, gears.
- Prefer stairs, catwalks, lifts, and industrial shelves over natural slopes.
- Slopes should be broken ramps, collapsed masonry, or ore conveyors, not grassy hills.
- Enemy placement should use armored melee on flat lower lanes and sentries on catwalks.
- Hazards should be pressure plates, gears, steam, crushers, or machinery tells.
- Use teal glow, brass, gray stone, and dark oil accents.
- Avoid organic rolling terrain unless it is a collapsed ruin section.

### Cinder Basin

Includes Cinder Refuge, Cinder Hollow, Ashglass Pass, Emberjaw Lair, and Emberjaw Furnace.

- Terrain identity: basalt shelves, ashglass crossings, furnace structures, lava vents, smoke.
- Slopes can be dramatic ashglass inclines or cooled lava flows, but should be few and memorable.
- Use flat safe pockets before and after vents or lava hazards.
- Combat should happen on basalt plates or furnace floors, not on steep inclines.
- Flyers need open air above flat terrain.
- Use ember orange, basalt black, smoke gray, and molten gold.
- Avoid long chains of steep lava ramps, which make the map look like a random mountain path.

### Frostfen Tundra

Includes Frostfen Camp, Frostfen Outskirts, Glacier Spine, Rimewarden Sanctum, and Rimewarden Vault.

- Terrain identity: icy marsh flats, snow shelves, glacier ledges, frozen ruins.
- Because movement has slippery characteristics in ice contexts, use slopes carefully.
- Prefer flat ice shelves, stepped glacier chunks, and short ledges.
- Use slopes only for major glacier transitions or scenic descents.
- Do not place precision pickups or important fights on icy slopes.
- Put recovery and checkpoint spaces on flat snow or stone.
- Use pale cyan, ice blue, muted pine, silver, and soft snow contrast.
- Avoid slippery downhill combat unless it is a deliberate encounter gimmick.

### Stormbreak Reach

Includes Stormbreak Haven, Stormbreak Cliffs, and Stormbreak Aerie.

- Terrain identity: wind-cut cliffs, suspended bridges, storm masts, lightning rods, exposed high routes.
- Use cliffs, rope bridges, ledges, wind lifts, and narrow shelves.
- Slopes should be wind-carved approaches to major overlooks or cliff transitions.
- Ranged pressure belongs on bridges and high ledges with readable approach paths.
- Flyers need airspace that does not hide above the camera.
- Use blue-gray stone, violet storm light, white lightning, and strong sky silhouettes.
- Avoid overusing ramps; this biome should feel like cliffs and vertical exposure.

### Astral Dominion

Includes Astral Observatory, Astral Archive, Astral Stacks, Eclipse Frontier, Eclipse Throne, and Endless Rift.

- Terrain identity: floating platforms, rune architecture, archives, star lenses, portals, rift islands.
- Prefer rune platforms, floating shelves, stairs, portals, and clean geometric layouts.
- Slopes should be rare and magical, such as tilted star-stone bridges or collapsed archive ramps.
- Use strong visual hierarchy because astral backgrounds can get busy.
- Enemy placement should use elite pockets, line-of-sight shelves, and portal loops.
- Hazards should be rift surges, star beams, sigils, void zones, or timed platforms.
- Use indigo, luminous cyan, parchment gold, violet, corona gold, void magenta, and cold white.
- Avoid making Astral maps look like natural hills with a star texture.

### Bandit Ridge Camp

Terrain identity: split-lane camp with lower cutters, middle throwers, upper rope bridge, campfire regroup.

- Use rough flat terraces, rope bridges, barricades, tents, and camp props.
- Slopes should be dirt paths into the camp, not the main combat grammar.
- Keep regroup/campfire points flat.
- Put ranged bandits on upper ledges with readable routes.
- Avoid cluttering slopes with tents, props, and enemies.

### Oreback Quarry

Terrain identity: material farm with ore carts, scaffold sentries, healer mushroom pockets, and mine events.

- Use mine shafts, scaffolds, cart tracks, flat quarry floors, and stepped ore ledges.
- Slopes should be haul roads or collapsed ore slides, not every connector.
- Keep resource nodes on flat pads.
- Put healer pockets on clearly safe flat shelves.
- Avoid steep repeated slopes because they make material farming feel sloppy.

## 8. Before-And-After Improvement Workflow

Use this workflow on every existing map before adding new content.

1. Identify the main player route.
   - Draw or list the route from spawn to exit.
   - Mark the intended safe entry, first encounter, branch, checkpoint, final challenge, and exit.
   - If the main route is not obvious, widen it and reduce competing diagonal paths.

2. Mark all slope sections.
   - Count every `shape: "slope"` platform.
   - Group slopes by screen or section.
   - Mark any slope used for combat, pickups, checkpoints, services, portals, or boss mechanics.

3. Remove unnecessary slopes.
   - Delete slopes that only solve tiny height changes.
   - Delete slopes that duplicate a nearby ladder, jump, bridge, or ledge.
   - Delete slopes that appear only because the generator added one per lane.

4. Convert rolling terrain into authored shapes.
   - Replace repeated slopes with flat platforms, ledges, cliffs, stairs, bridges, vines, lifts, portals, or one-way drops.
   - Preserve one major transition slope where it improves flow.
   - Add visual anchors where slopes remain.

5. Keep only slopes that improve flow or identity.
   - Greenroot may keep gentle meadow/root slopes.
   - Cinder may keep dramatic ashglass inclines.
   - Frostfen may keep glacier transitions but not slippery combat slopes.
   - Rustcoil, Astral, and town maps should use more artificial connectors.

6. Add flat combat zones.
   - Insert or widen flat platforms before enemy groups.
   - Move enemies off slope bodies.
   - Place ranged enemies on flat ledges.
   - Place recovery items and checkpoints on stable ground.

7. Add visual anchors and landmarks.
   - Every section with a remaining slope needs a reason to exist visually.
   - Use roots, rocks, machines, bridges, vents, ice slabs, storm markers, or rune supports.
   - Add one memorable landmark per map.

8. Improve tile variation.
   - Keep top collision edges clean.
   - Add texture variation to platform interiors and undersides.
   - Use biome accents near landmarks, rewards, and section transitions.
   - Avoid repeating the same ramp art stamp.

9. Clean up collision.
   - Ensure slope endpoints meet flat terrain without bumps.
   - Ensure visual slope edge matches collision slope.
   - Remove invisible snag points.
   - Confirm enemies and path graph links still work after slope removal.

10. Playtest traversal and camera movement.
    - Walk the main route without fighting.
    - Run from spawn to exit.
    - Jump through optional routes.
    - Fight every encounter.
    - Watch the camera on climbs and descents.
    - Confirm the map has challenge/rest/reward pacing.

### Slope Decision Rules

For every existing slope, choose one outcome:

| Decision | Keep If | Replace With |
| --- | --- | --- |
| Keep | It moves between major lanes, supports biome identity, has flat entry/exit, and does not host accidental combat. | None. Add better visual anchors if needed. |
| Shorten | The slope is useful but too dominant or too long. | Shorter slope plus flat landing or ledge. |
| Replace With Flat Ground | The slope is in a combat, checkpoint, service, pickup, or decision area. | Broad platform with decorative terrain depth. |
| Replace With Stairs | The area is town, ruin, archive, facility, or boss structure. | Biome-specific stairs or stepped platforms. |
| Replace With Ledges | The height change is small or should create a jump test. | Short ledges with clean landing pads. |
| Replace With Cliff/Drop | The section needs a stronger silhouette, shortcut, or one-way return. | Cliff face, drop loop, lower route. |
| Replace With Bridge/Catwalk | The route crosses a gap or artificial space. | Rope bridge, metal catwalk, stone span, rune bridge. |
| Replace With Climbable/Lift/Portal | The map needs vertical movement without diagonal terrain. | Vine, ladder, lift, wind current, rune platform, portal. |
| Use As Major Transition | The slope marks a new section, reveal, or biome set piece. | Keep one strong slope and remove nearby duplicates. |

## 9. Prompt Templates For Map And Tileset Assets

Use these prompts when generating new map-related art. Keep prompts consistent with the GDD and existing asset prompts.

### Master Map Style Prompt

```text
Project Starfall 2D side-scrolling action RPG environment, Starlit Frontier Fantasy, clean luminous painterly fantasy, crisp playable silhouettes, readable side-view platformer terrain, hand-authored map shapes, broad flat combat lanes, intentional ledges and cliffs, sparse meaningful slopes, blue and gold star magic accents where appropriate, warm crafted materials, atmospheric parallax background, clear foreground/background separation, game-ready terrain, professional 2D platformer readability, no UI, no characters unless requested.
```

### Master Negative Prompt

```text
random generated hills, excessive slopes, rolling terrain everywhere, noisy terrain outline, jagged unreadable collision edge, cluttered gameplay path, blurry tiles, inconsistent perspective, mismatched lighting, over-detailed background competing with gameplay, non-seamless tile edges, AI-looking artifacts, warped geometry, unusable collision shapes, hidden hazards, overpainted foreground covering platforms, generic fantasy mush, low contrast playable terrain.
```

### Tileset Prompt

```text
Create a game-ready Project Starfall biome tileset for [BIOME NAME], 2D side-scroller side-view, 64 px tile logic, clean readable collision tops, painterly but crisp, includes flat top tiles, left and right caps, inner fills, underside tiles, cliff faces, corner transitions, short ledge pieces, long platform body variants, subtle biome-specific accent tiles, consistent lighting from upper left, seamless tile edges, no characters, no UI. Style must match Starlit Frontier Fantasy: luminous, readable, hand-crafted, not procedural.
```

### Slope Tile Prompt

```text
Create a small Project Starfall slope tile set for [BIOME NAME], side-view 2D platformer, only 2-3 intentional slope angles, includes lower start cap, repeating slope middle, upper end cap, flat-to-slope transition, slope-to-flat transition, slope-to-cliff transition, clean collision-readable top edge, minimal texture noise on playable edge, biome-specific support details such as roots, basalt, metal braces, ice slabs, wind-cut stone, or rune plates. Slopes must look hand-placed and rare, not like rolling generated hills.
```

### Cliff Tile Prompt

```text
Create Project Starfall cliff terrain tiles for [BIOME NAME], side-view 2D action RPG, strong vertical silhouettes, clean top ledges, readable cliff faces, interior fill variation, underside shadow pieces, corner caps, painterly fantasy detail inside the terrain mass, crisp playable edges, seamless tiles, consistent lighting, no excessive bumps along collision.
```

### Platform Prompt

```text
Create Project Starfall platform assets for [BIOME NAME], side-view 2D platformer, flat readable gameplay tops, left/right caps, long center pieces, short ledges, underside pieces, optional one-way platform variant that is visually distinct from solid ground, biome-specific materials, crisp silhouette, clean collision expectation, painterly luminous fantasy style, seamless edges.
```

### Background Prompt

```text
Create a parallax background layer for Project Starfall [BIOME NAME], side-view fantasy action RPG, atmospheric depth, lower contrast than gameplay terrain, readable negative space behind player silhouettes, biome landmarks in the distance, luminous but not cluttered, no foreground collision-looking platforms, no UI, no characters, consistent lighting and color palette.
```

### Foreground Decoration Prompt

```text
Create Project Starfall foreground decoration props for [BIOME NAME], transparent background, side-view 2D game assets, roots/rocks/rails/crystals/banners/machinery/runes as appropriate, designed to frame gameplay without covering platform edges, clean silhouettes, painterly fantasy, consistent lighting, no collision ambiguity, no large opaque clutter.
```

### Prop Prompt

```text
Create Project Starfall biome props for [BIOME NAME], 2D side-scroller RPG, game-ready object sheet, includes small, medium, and landmark props, clean silhouettes, readable at gameplay zoom, transparent background, consistent palette and lighting, crafted/luminous fantasy style, props should support level storytelling without cluttering the playable path.
```

### Hazard Prompt

```text
Create Project Starfall hazard tiles for [BIOME NAME], 2D side-scroller, readable danger shapes, clear warning silhouettes, animated-ready design, includes safe inactive state and active dangerous state, consistent lighting, no hidden edges, no background clutter, hazard must be understandable before it punishes the player. Examples: thorns, vents, lava cracks, ice spikes, lightning rods, rift sigils.
```

### Parallax Background Prompt

```text
Create layered parallax background art for Project Starfall [BIOME NAME], separated into far, mid, and near background layers, side-view fantasy action RPG, atmospheric depth, soft contrast, landmark shapes that support navigation, no walkable-looking foreground terrain, no busy detail behind combat spaces, luminous Starlit Frontier Fantasy color palette.
```

### Full-Level Concept Prompt

```text
Create a full 2D side-scroller level concept for Project Starfall [MAP NAME], showing the whole level as a readable side-view map mockup, clear main route, optional upper/lower route, broad flat combat areas, one checkpoint rest area, one landmark reveal, sparse intentional slopes only at major terrain transitions, biome-specific terrain and props, clean playable silhouettes, no random rolling hills, no noisy collision edges, professional hand-designed platformer composition.
```

### Map Blockout Prompt

```text
Create a simple side-view blockout map for Project Starfall [MAP NAME], grayscale gameplay-first layout, broad flat main route, optional reward branch, vertical route where appropriate, clear combat arenas, checkpoint area, final challenge, exit zone, slopes limited to major transitions and marked clearly, no decoration clutter, no random hills, camera-friendly plateaus and sightlines.
```

## 10. Final Map Polish Checklist

Use this checklist before calling a map complete.

### Route And Pacing

- [ ] The level has a clear main path from spawn to exit.
- [ ] The first screen tells the player where to go.
- [ ] Optional routes are visible or hinted.
- [ ] Optional routes return cleanly or reward clearly.
- [ ] The level alternates challenge, rest, exploration, and reward.
- [ ] The level teaches, tests, and varies its main mechanic.
- [ ] The final section combines earlier ideas without becoming cluttered.
- [ ] The exit or transition zone is unmistakable.

### Slope Usage

- [ ] Slopes are used intentionally rather than everywhere.
- [ ] The map meets its slope budget target.
- [ ] No section has repeated slope chains without flat rests.
- [ ] Major combat spaces are flat.
- [ ] Checkpoints, shops, portals, NPCs, and rewards are not on slopes.
- [ ] Each remaining slope has a visual reason to exist.
- [ ] Slopes have flat entry and exit pads.
- [ ] Slope collision matches slope visuals.
- [ ] The map does not look like random rolling hills.

### Terrain Readability

- [ ] There are enough flat spaces for combat, jumping, and decision-making.
- [ ] Terrain shapes are readable at gameplay zoom.
- [ ] Ground silhouettes are clean and not noisy.
- [ ] Cliffs, ledges, bridges, caves, and structures break up the terrain.
- [ ] Playable platforms are visually distinct from background shapes.
- [ ] Foreground decoration does not hide collision edges.
- [ ] Background detail does not compete with gameplay.

### Encounters And Hazards

- [ ] Enemies have enough reaction space.
- [ ] Melee enemies fight on readable flat ground.
- [ ] Ranged enemies have clear approach routes.
- [ ] Flyers have visible airspace.
- [ ] Hazards are readable before they punish the player.
- [ ] Hazards are not hidden by props or foreground.
- [ ] Rewards are visible or hinted before the player commits.
- [ ] Recovery items appear after meaningful challenge.

### Tile And Collision Quality

- [ ] Collision is simpler than decoration.
- [ ] No invisible bumps exist at flat-to-slope transitions.
- [ ] No decorative art implies false walkable terrain.
- [ ] One-way platforms, if present, have distinct visuals.
- [ ] Tile variation avoids obvious repetition.
- [ ] Terrain interiors carry detail while gameplay edges stay clear.
- [ ] Ramp art and slope collision align.
- [ ] Enemy pathing still works after terrain changes.

### Camera And Feel

- [ ] The camera behaves well through each section.
- [ ] Climbs are broken into stable plateaus.
- [ ] Major reveals happen from flat overlooks or rests.
- [ ] The camera does not bob constantly from slope chains.
- [ ] Players can run the main route without snagging.
- [ ] Players can jump, land, dodge, and attack reliably on required terrain.

### World And Style

- [ ] The map matches the GDD's Starlit Frontier Fantasy tone.
- [ ] The map uses the correct biome palette and material language.
- [ ] The map has at least one memorable landmark.
- [ ] Props support the theme instead of creating clutter.
- [ ] The level feels hand-designed, not auto-generated.
- [ ] The map matches the intent described in `map-presentation.js`.
- [ ] The map supports Project Starfall's action RPG combat, party roles, exploration, and progression.

## 11. Practical Design Rules To Keep Visible While Building

- Build flat combat first.
- Add verticality second.
- Add slopes third, only where they improve flow.
- Add enemies after terrain reads clearly.
- Add rewards after routes read clearly.
- Add decoration after collision is final.
- Never let decoration hide the game.
- Never let slopes become the map's default texture.
- One strong slope is better than four forgettable slopes.
- A flat platform with a good landmark is more professional than a noisy hill.
- If a terrain shape does not teach, test, guide, rest, reward, or reveal, simplify it.
