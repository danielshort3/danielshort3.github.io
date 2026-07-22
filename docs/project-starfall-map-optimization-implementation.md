# Project Starfall Map Optimization Implementation Notes

This companion note tracks implementation progress against `docs/project-starfall-map-optimization-audit.md` and `project-starfall-map-optimization-implementation-goal.txt`.

## Implemented In This Slice

- Fixed Starfall Crossing's Plinko placement contract by adding an explicit `plinko` entry to town station placement. The Starfall Plinko station and Plinko Host now resolve to the same town platform and `x` position.
- Moved high-frequency town stations into an intentional ground-level service pocket:
  - `storage` is a high-frequency ground service.
  - `shop` is a high-frequency ground service and represents supply, repair, and equipment restock.
  - `slots`, `upgrade`, and `plinko` are medium-frequency systems.
  - `class` is lower-frequency class/onboarding flavor.
- Added `townServicePlan` metadata for every current public town:
  - Starfall Crossing
  - Rustcoil Outpost
  - Cinder Refuge
  - Frostfen Camp
  - Stormbreak Haven
  - Astral Observatory
- Added `designIntent` metadata for every current non-admin combat map. Each record includes intended archetype, use case, route summary, party role target, farming-abuse risk, visual identity tag, spawn-section model, party scaling, and priority-redesign flag.
- Baseline combat-map metadata and runtime plumbing now cover: Greenroot Meadow (`greenrootMeadow`), Thornpath Thicket (`thornpathThicket`), Bramble Depths (`brambleDepths`), Rustcoil Ruins (`rustcoilRuins`), Gearworks Vault (`gearworksVault`), Cinder Hollow (`cinderHollow`), Emberjaw Lair (`emberjawLair`), Bandit Ridge Camp (`banditRidgeCamp`), Oreback Quarry (`orebackQuarry`), Ashglass Pass (`ashglassPass`), Frostfen Outskirts (`frostfenOutskirts`), Glacier Spine (`glacierSpine`), Rimewarden Sanctum (`rimewardenSanctum`), Stormbreak Cliffs (`stormbreakCliffs`), Astral Archive (`astralArchive`), Eclipse Frontier (`eclipseFrontier`), Endless Rift (`endlessRift`), Brambleking Court (`bramblekingCourt`), Titan Foundry (`titanFoundry`), Deepcore Core (`deepcoreCore`), Emberjaw Furnace (`emberjawFurnace`), Rimewarden Vault (`rimewardenVault`), Stormbreak Aerie (`stormbreakAerie`), Astral Stacks (`astralStacks`), and Eclipse Throne (`eclipseThrone`). Individual maps still require authored geometry, encounter, and presentation passes before they should be considered professionally complete.
- Added `spawnSections` metadata to non-safe maps and assigned every spawn point a `sectionId` and `sectionLabel`.
- Preserved `sectionId` and service metadata through runtime map normalization so engine systems can consume them.
- Added section-aware field respawn scoring:
  - Killed enemies remember their originating spawn section.
  - Replacement respawns prefer the same section when safe.
  - Existing safe-spawn, view-edge, crowding, and hot-area pressure checks still apply.
- Added bespoke `geometry-spawn-v1` layouts for the audit's highest-priority field maps:
  - Greenroot Meadow now has a named Starter Pond Loop before Moss Lane, Canopy Practice, and Thornpath Gate sections.
  - Bandit Ridge Camp now uses non-repeating authored geometry for its lower cutter, middle thrower, high rope-bridge, and campfire-regroup beats. Combat groups own explicit non-overlapping platforms, while the campfire remains a zero-routine-population recovery space.
  - Oreback Quarry now has ore-cart, scaffold, mushroom-pocket, and mine-event sections for small-party/material farming.
  - Cinder Hollow and Ashglass Pass now use different lava-region route shapes and section fiction instead of sharing the same lava-shaft grammar.
  - Stormbreak Cliffs now separates low ram, mid archer, high harrier, and lightning-rod support sections.
  - Endless Rift now uses quadrant sections plus a non-spawning rift-core regroup section for future surge rules.
- Added bespoke `arena-skeleton-v1` dungeon and boss layouts for every required dungeon/boss map:
  - Bramble Depths: root lanes, root shelves, and crown approach.
  - Gearworks Vault: tank lane, sentry catwalk, and gear switch shelf.
  - Emberjaw Lair: vent side, safe pockets, and overheat shelf.
  - Rimewarden Sanctum: brute lane, oracle shelf, and sentinel shelf.
  - Brambleking Court: root lane, thorn pod shelf, and crown platform.
  - Titan Foundry: gear floor, armor switches, and sentry catwalk.
  - Deepcore Core: tank chamber, healer lane, turret lane, and ore core.
  - Emberjaw Furnace: lava cracks, valve shelf, and safe pocket.
  - Rimewarden Vault: whiteout brute/oracle/sentinel lanes.
  - Stormbreak Aerie: ram lane, rod perch, and harrier airspace.
  - Astral Stacks: left stacks, center rune shelf, and right stacks.
  - Eclipse Throne: solar lane, eclipse dais, lunar lane, and mote shelf.
- Exposed `arenaSkeleton` and `arenaMechanic` through runtime snapshots and renderer map metadata for future tuning and UI.
- Exposed map intent and spawn-section coverage in `tests/project-starfall-balance-harness.js` field reports so tuning output can group maps by intended archetype, use case, party role, farming risk, visual identity, and section coverage.
- Added `field.mapTuning` to the balance harness. The report now covers every current non-admin combat map and exposes:
  - kills per minute
  - EXP per minute
  - drop value per minute
  - idle time
  - travel share
  - route cycle time
  - damage taken
  - potion use
  - death rate
  - platform coverage
  - spawn vacancy
  - non-combat traversal share
  - class performance spread
  - party overlap
  - party efficiency versus solo
  - elite/miniboss clear time
  - abandonment risk
  - repeat-visitation index
- Added high-risk farming abuse-control coverage to the tuning report. Oreback Quarry, Eclipse Frontier, and Endless Rift now surface explicit control sections, including Eclipse Frontier's Elite Pocket.
- Added live `mapMechanic` definitions and runtime progress for the highest-priority farm/party maps:
  - Oreback Quarry now has a rotating Ore Cart Material Rush that advances through real spawn sections and grants capped ore rewards.
  - Stormbreak Cliffs now has a Lightning Rod Charge objective that requires split low/mid/anti-air section participation before rewarding the party route.
  - Endless Rift now has a four-quadrant surge rotation, a Rift Core regroup marker, and an anti-camping reward scale that feeds Rift score gains.
- Added live boss/dungeon spatial mechanics for every bespoke dungeon and boss echo:
  - Bramble Depths and Brambleking Court now route root waves, thorn/root pod calls, vine cages, and crown exposes to authored arena sections.
  - Gearworks Vault and Titan Foundry now route gear switches, sentry catwalks, tank lanes, overclock calls, and plate/core pulse windows to authored sections.
  - Deepcore Core now routes tank, healer, turret, and ore-core chamber calls to authored sections.
  - Emberjaw Lair and Emberjaw Furnace now route furnace vents, lava cracks, valve shelves, overheat shelves, and safe pockets to authored sections.
  - Rimewarden Sanctum and Rimewarden Vault now route ice-wall lane locks, whiteout shelves, brute lanes, and sentinel shelves to authored sections.
  - Stormbreak Aerie now routes storm rod perches, harrier airspace, ram lanes, wind lanes, and divebomb calls to authored sections.
  - Astral Stacks now routes mirrored archive shelves, rune calls, memory seals, and scribe adds to authored sections.
  - Eclipse Throne now routes solar lanes, lunar lanes, the eclipse dais, and mote shelf calls to authored sections.
- Boss telegraphs and resolved boss hazards now expose spatial mechanic metadata in runtime effects and boss snapshots.
- Added a tracked dungeon `Spatial Control` objective that advances when players resolve live spatial boss calls during dungeon runs.
- Added persisted runtime `mapAnalytics` hooks for real player-session map tuning:
  - map entries and repeat visits are recorded on actual map transitions.
  - active visit time and field kills are tracked during play.
  - short low-kill field exits become real abandonment signals.
  - repeat-visitation and abandonment rates are exposed through snapshots and survive save/restore.
- Added test coverage for town service plans, Plinko station/host colocation, combat design intent, spawn-section metadata, runtime respawn section preservation, priority field geometry/spawn-section implementation, Cinder/Ashglass route differentiation, dungeon/boss arena skeleton identity, live boss/dungeon spatial hooks, runtime abandonment/repeat-visitation analytics, map tuning metric coverage, farming abuse-control coverage, and live map-mechanic progression.

## Deferred Work

No audit-required implementation items are knowingly deferred. Future tuning work should use the new runtime `mapAnalytics` snapshots alongside the balance-harness estimates to validate real player behavior over longer sessions.

## Verification Run

Fast metadata verifier:

- Public towns checked: 6.
- Non-admin combat maps checked: 25.
- Plinko station and host colocated at `x=2260`, `platformIndex=5`.
- No metadata coverage failures found.
- `npm test` passed after the data/runtime/test changes.
- `node -c` passed for `project-starfall-data.js`, `project-starfall-engine.js`, and `test.js` after the boss spatial hook changes.
- `npm test` passed again after adding persisted runtime abandonment/repeat-visitation analytics.
- `npm run build` passed and regenerated the static public output.
- Final goal verifier passed:
  - Public towns checked: 6.
  - Non-admin combat maps checked: 25.
  - Boss spatial maps checked: 12.
- Final public source-copy checks passed for Project Starfall data and engine files.
- Scoped `git diff --check` passed for the touched Project Starfall/data/test/doc/public files.
- Full `git diff --check` still reports unrelated pre-existing trailing whitespace in `start-local-build-wsl.bat`, `start-local-dev-wsl.bat`, and `yarn.cmd`.

Required follow-up verification for later slices:

- Run `npm test`.
- Run `npm run build` if build inputs, CSS bundles, copied assets, or public output change.
- Run `git diff --check`.
- Inspect the playable Project Starfall page after geometry redesigns to confirm town stations, portals, redesigned maps, and UI panels still render and interact.
