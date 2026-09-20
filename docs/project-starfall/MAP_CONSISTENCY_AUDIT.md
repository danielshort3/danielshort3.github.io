# Starfall map consistency audit

Local implementation and evidence for the 56-map layout, training, and scenery pass. Map, portal, quest, and save identities are preserved. This report does not describe a deployment.

## Layout and encounters

All 56 maps have reachable outward and return routes for an ordinary character under the conservative 30 FPS jump model. The structural audit covers 634 platforms and 332 climbables. Navigation filters jump links using actual jump velocity, horizontal speed and body width: 166 of 653 potential graph jumps are usable by the baseline 510/240/40 actor. Thirteen hunting bridges, one shallow Cinder connection and 69 access climbables were appended without renumbering original platforms or portals. Actual player input exercises both directions across the hunting connections in 84 cases at 30, 60 and 120 FPS; the map encounter suite passes 23,991 assertions. The shallow Cinder ramp reaches both original shelves without falling through, but descending player gravity produces intermittent airborne flags within 4.1px of its surface. This is not a claim of continuous grounding. Enemy locomotion checks pass across 111 authored combat ramps; enemy and companion checks also cover representative jump and return routes at those frame rates.

The final targeted encounter adjustments give Meadow's basin four enemies across equally weighted lower and upper spawn points, for eleven total while preserving its two-enemy arrival and all cadences. Quarry replaces underleveled ratchets in the ore-cart and support pockets with its existing beetle/warden roster, preserving population, replacement timing and the one-healer cap. The added connections have no routine spawn points or group ownership; eastern and upper optional routes retain their geometry.

Thirteen public fields, including the separately judged Endless Rift, have authored combat groups, a regroup approach with no authored combat group and an optional combat branch. Quarry support pockets cap simultaneous healers; restorative pulses exclude other support healers and use two-dimensional range. Glacier scales from 32 solo enemies to 40 for a three-member party. Cinder and Bandit optional branches now use stronger on-level native rosters; their populations and cadences are unchanged. The starter beacon removes its weakest encounter variant. Spawn occupancy deferral and quotas remain active. Pursuit and return honor actor traversal permissions and bounded leashes, with no reacquisition while returning home.

Grounded down+jump now takes precedence over automatic ladder mounting, so ladder-top drops move down. Down alone still climbs, and jumping while already climbing retains its upward exit; 36 actual-input cases cover four maps at 30/60/120 FPS. Drops also clear overlapping or adjoining pieces of the same support surface, fixing the Rustcoil catwalk trap. Twenty-seven assertions preserve solid ground, separate lower floors, ramp intersections and real lower-step landings.

Companion state now preserves wounded and defeated HP. New companions start with full class/level health; defeated companions remain down until their four-second recovery expires, revive once, and do not increase the living-party spawn target while down. Sixteen focused cases cover both state implementations and actual damage/recovery at 30/60/120 FPS.

## Visual changes

- Shared Canvas/Pixi placement anchors props to actual surfaces and omits broad unsupported props. Zero density suppresses decoration.
- All ramp atlases paint their contact edge onto the authored collision slope. Transparent padding and repeated wedge resets no longer shift the visible ledge.
- Crossing uses distinct survey, depot, gate, workshop and service artwork with the frontier gate centered on its portal.
- Ashglass uses basalt and volcanic glass; Cinder has a distant panorama without misleading foreground shelves. Full-size review also corrected Ashglass's flat terrain registration: its normalized caps now start at the collision plane in both renderers, removing the old 24px ground / 12px ledge lift without moving gameplay geometry.
- All 24 shops have vendor-specific interiors and a neutral stone/timber floor instead of outdoor fallback scenery.
- Full-size review caught five generated hunt Wardens using geometric placeholders. Their missing mapping now reuses the existing illustrated actor artwork; NPC identities, dimensions, footing and quests are unchanged. All thirteen measured field NPC definitions remain identical. [Before](../../output/starfall-map-scenery-review/scenes/warden-before.png) / [after](../../output/starfall-map-scenery-review/scenes/warden-after.png).
- Scenery continues through the reserved world-bottom boundary; both renderers fade into a dark HUD backdrop. Asset request revisions refresh previously cached scenery. NPC labels remain readable on light and dark scenery.

The pass changes ten selected runtime images: six additions and four replacements. The full scenery ledger validates 173 outputs from 89 active masters. All 31 protected session images retain their original hashes.

## Training evidence

Formula estimates now use actual spawn-group populations, weighted rosters and respawn cadences; wave minutes are converted to seconds before comparing with respawn delays. Formula output is explicitly labeled, includes the authored normal-kill XP multiplier, and is not measured throughput.

The deterministic harness uses actual movement inputs, combat, healing, deaths, loot and respawns, with 60 seconds warmup and 300 measured seconds across three fixed seeds. It holds level, shop equipment budget and legal skill ranks consistent; account/admin boosts and accumulating mastery bonuses are disabled. Both clocks and process-local runtime state are controlled so worker order cannot change results. A mechanical route gate precedes numerical comparisons.

The fixed party consists of the fighter leader and the built-in ranged/magic companions. Their authored stat model is reported separately from human-player equipment. Leader and companion rewards are not silently combined. Incomplete route coverage or excessive controller stalls disqualify a run from balance acceptance. Actual pickups also report class/level-equippable equipment, crafting materials, consumables and cards. Potential shop resale uses the production appraisal but is not counted as earned currency or an equipment upgrade. Incoming damage describes the leader and includes shield absorption; it is not the sum of party HP loss.

The final evidence contains **1,575 fresh actual-engine runs** after the Cinder XP adjustment and final visual-only source corrections: 525 current main-route runs, 525 reconstructed earlier-authored main-route runs, and 525 current optional-route runs. Each matrix has 175 map/level/class-or-party rows averaged across three seeds. All runs complete an ordered circuit, visit 100% of their selected platforms and pass the unchanged stall gate; every main comparison has a matched earlier row. Earlier authoring runs on the same corrected engine, not a historical executable. The [previous complete matrix](../../output/starfall-map-scenery-review/training-pre-cinder-adjustment/README.md) remains preserved with its original bytes and source hashes. No earlier scenario rows were reused or silently rehashed. Phase-one party measurements preceded the companion HP fix and are not acceptance evidence.

**Balance acceptance is not complete.** Of 145 comparable ordinary-field rows, 41 meet both XP and pacing targets and 104 require tuning. Seventy exceed 30% travel; 62 miss their XP target (these groups overlap). Seventeen standalone rows and thirteen Endless Rift rows require separate judgement. Respawn-only waiting passes across the sample, with a maximum row mean of 1.17%. Of 162 ordinary-field optional-branch rows, 125 have higher measured pressure, but only one earns the intended 10–20% XP premium. The thirteen Rift optional rows are retained as diagnostics, not ordinary-field acceptance.

The table shows the range across eligible solo classes and the fixed companion party at each level; these are observed changes against earlier authored geometry/encounters, not estimated formula gains.

| Map | Level | XP/min change across modes | Current travel across modes | Modes passing pacing |
|---|---:|---:|---:|---:|
| Starfall Verge | 6 | 0.1% to 14.7% | 46.2–49.8% | 0 / 4 |
| Cinder Hollow | 19 | 32.0% to 43.4% | 33.2–39.3% | 0 / 4 |
| Cinder Hollow | 28 | 23.4% to 54.9% | 24.1–52.0% | 2 / 13 |
| Oreback Quarry | 28 | 3.0% to 18.2% | 21.8–36.9% | 6 / 13 |
| Ashglass Pass | 50 | 16.7% to 48.7% | 25.4–46.5% | 2 / 13 |
| Glacier Spine | 60 | 2.1% to 29.2% | 18.7–29.3% | 13 / 13 |
| Stormbreak Cliffs | 60 | 7.9% to 53.8% | 31.3–50.7% | 0 / 13 |

There were no leader deaths under this stocked-consumable controller. That is not a claim of risk-free human play: each run starts with 99 level-appropriate health potions and 99 resource tonics, outside the equipment budget, and reports their actual consumption and replacement cost. Current main runs record 1,297 companion knockouts across 57 party runs and 5,196 combined companion down member-seconds. Companion fragility remains visible in the review rather than disappearing into leader-only death counts.

Cinder now uses an authored **0.93 normal-kill XP multiplier**, after route/encounter tuning and rejection of an ineffective roster alternative. The reconstructed before profile stays at 1. Random elites are protected by the engine's existing boss-or-elite predicate, so the observed aggregate reduction is 6.35% on the main circuit and 6.34% on the optional circuit, rather than exactly 7%. The [observed confirmation](../../output/starfall-map-scenery-review/training-cinder-observed-adjustment.json) verifies 1,473 unchanged scenario payloads and 102 Cinder scenarios whose only changes are earned-XP fields. Boss, elite, dungeon, trial, endless, rare-drop and one-time rewards remain separate. The scalar does not repair class-specific encounter deficits or optional-branch travel; differing protected-elite mixes also mean main and optional aggregate reductions need not match exactly.

Remaining priorities are excessive transit, optional-branch rosters/rewards, ranged deep-field deficits and the all-class Thornpath-over-Verge advantage at level 6. Cinder no longer exceeds Rustcoil by 25% across every eligible class, but individual class/cohort misses remain. The [pre-adjustment diagnosis](../../output/starfall-map-scenery-review/training-dominance-findings.md) and [scalar decision history](../../output/starfall-map-scenery-review/training-cinder-scalar-projection.md) retain their historical numbers; final acceptance uses the fresh observed reports.

[Interactive comparisons](../../output/starfall-map-scenery-review/scene-review.html) · [All main rows](../../output/starfall-map-scenery-review/training-comparison.csv) · [All optional rows](../../output/starfall-map-scenery-review/training-optional-comparison.csv) · [Protocol, source hashes and caveats](../../output/starfall-map-scenery-review/training-comparison.json) · [Summary](../../output/starfall-map-scenery-review/training-review-summary.json)

## Verification and limits

- Structural audit: 23,991 assertions across all 56 maps, including arrivals, population scaling, territory policy, saved-position recovery and real traversal.
- Ramp motion: 222,364 checks over 111 ramps at 30/60/120 FPS; retained spawn, crowd, flyer and walk-home regressions.
- Scenery: Canvas/Pixi placement equality for 372 props, including 56 on slopes; 47,680 sampled contact columns across 160 ramp cells. The separate [flat-terrain contact check](../../output/starfall-map-scenery-review/terrain-contact-review.md) covers 42 dispatches and 18,392 painted plateau columns, plus built-route Ashglass measurements in both renderers.
- Visual review: 448 arrival, middle, elevated and exit captures across all maps and both renderers. Every map received original-resolution Pixi review (200 distinct scenes, excluding duplicate shop views), with additional Canvas detail checks and deep desktop/mobile/reduced-effects coverage. [Exact reviewed files](../../output/starfall-map-scenery-review/visual-review.md) distinguish captured views from manually inspected subsets. Sixteen overlapping charge/heal fixtures cover bright/dark backgrounds and both renderers/settings. These are controlled fixtures, not human playtest balance ratings.
- Local build, all 830 active assets and [169 runtime JavaScript files](../../output/starfall-map-scenery-review/runtime-public-parity.json) match public output. All 31 protected artwork hashes are unchanged. A separate [bounded code review](../../output/starfall-map-scenery-review/code-review.md) found no new blocking defect in the reviewed navigation, return, companion and build paths.
- Smoke, systems, combat, inventory, Eclipse, asset and map suites pass. The broader progression command reaches an existing date-dependent recurring-shop rollback assertion; that unrelated failure is recorded rather than counted as a pass.

## Map-by-map coverage

Every row has connected outward/return routing and reviewed key views. Population columns describe authored maxima before occupied-spawn deferral. Towns, shops, dungeons, the admin laboratory and Endless Rift retain their distinct purposes; ordinary-field balance gates do not apply to every row.

| Map | Purpose | Platforms | Solo / party population | Respawn seconds | Encounter or retained purpose |
|---|---|---:|---:|---:|---|
| Starfall Crossing | town | 17 | — | — | Services, landmarks and exits retained |
| Rustcoil Outpost | town | 17 | — | — | Services, landmarks and exits retained |
| Cinder Refuge | town | 17 | — | — | Services, landmarks and exits retained |
| Frostfen Camp | town | 17 | — | — | Services, landmarks and exits retained |
| Stormbreak Haven | town | 17 | — | — | Services, landmarks and exits retained |
| Astral Observatory | town | 17 | — | — | Services, landmarks and exits retained |
| Starfall Verge | starterField | 12 | 11 / 11 | 6, 7 | Arrival Shelf; Glass Basin; Fractured Bridge; Beacon Approach |
| Thornpath Thicket | trainingField | 21 | 26 / 26 | 5, 6 | Rootfall Ground Packs; Canopy Ambush; Briar Guard Branch |
| Bramble Depths | dungeon | 15 | 8 / 8 | 8 | Ridge Return; Root Lanes; Court Gate |
| Rustcoil Ruins | trainingField | 22 | 28 / 28 | 5, 6 | Ratchet Patrol; Coil Crossfire; Warden Gearwell |
| Gearworks Vault | dungeon | 15 | 9 / 9 | 8 | Tank Lane; Sentry Catwalk; Gear Switch Shelf |
| Cinder Hollow | trainingField | 26 | 24 / 24 | 5, 6 | Ash Floor Ground Packs; Lava Tick Vent Run; Ember Crossfire |
| Emberjaw Lair | dungeon | 15 | 8 / 8 | 8 | West Vent; Safe Pockets; Overheat Shelf |
| Bandit Ridge Camp | deepField | 24 | 30 / 42 | 4, 5 | Lower Cutters; Thrower Camp; Rope Bridge |
| Bandit Animation Lab | admin review | 6 | 5 / 5 | 5 | Entry; Admin Lab; Exit |
| Oreback Quarry | deepField | 25 | 26 / 34 | 5, 6, 8 | Ore Cart Beetles; Scaffold Sentries; Glowcap Support Pocket; Mimic Mine |
| Ashglass Pass | deepField | 26 | 30 / 30 | 5, 7 | Basalt Bridge Patrol; Vent Crossfire; Glass Shelf Airspace; Obsidian Elite Pocket |
| Frostfen Outskirts | trainingField | 21 | 31 / 31 | 5, 6 | Frozen Marsh Scouts; Rimeglass Airspace; Oracle Grove Guards |
| Glacier Spine | deepField | 30 | 32 / 40 | 5, 6 | Lower Ridge Scouts; Glacier Sentry Circuit; High Ridge Elite Guard |
| Rimewarden Sanctum | dungeon | 15 | 9 / 9 | 8 | Brute Lane; Oracle Shelf; Sentinel Shelf |
| Stormbreak Cliffs | deepField | 27 | 32 / 40 | 5, 7 | Thunder Ram Lane; Archer Bridge; Harrier Airspace; Lightning Rod |
| Astral Archive | trainingField | 30 | 34 / 34 | 5, 6 | Reading Room Guards; Index Shelf Crossfire; Sealed Archive Pocket |
| Eclipse Frontier | deepField | 30 | 34 / 42 | 5, 6 | Solar Sentinel Patrol; Lunar Airspace; Gate Crossfire; Eclipse Elite Guard |
| Endless Rift | endlessField | 28 | 36 / 46 | 5, 7 | Western Sentinel Circuit; Upper Rift Airspace; Eastern Duelist Circuit; Lower Rift Crossfire; Optional Rift Surge |
| Brambleking Court | bossArena | 15 | 8 / 8 | 8 | Root Lane; Thorn Pod Shelf; Crown Platform |
| Titan Foundry | bossArena | 15 | 8 / 8 | 8 | Gear Floor; Armor Switches; Sentry Catwalk |
| Deepcore Core | bossArena | 15 | 8 / 8 | 8 | Tank Chamber; Healer Lane; Turret Lane; Ore Core |
| Emberjaw Furnace | bossArena | 15 | 8 / 8 | 8 | Lava Cracks; Valve Shelf; Safe Pocket |
| Rimewarden Vault | bossArena | 15 | 8 / 8 | 8 | Brute Lane; Oracle Shelf; Sentinel Shelf |
| Stormbreak Aerie | bossArena | 15 | 8 / 8 | 8 | Ram Lane; Rod Perch; Harrier Airspace |
| Astral Stacks | bossArena | 15 | 8 / 8 | 8 | Left Stacks; Center Rune Shelf; Right Stacks |
| Eclipse Throne | bossArena | 15 | 8 / 8 | 8 | Solar Lane; Eclipse Dais; Lunar Lane; Mote Shelf |
| Crossing Weapon Shop | shop | 1 | — | — | weapon interior; vendor and exit retained |
| Crossing Armor Shop | shop | 1 | — | — | armor interior; vendor and exit retained |
| Crossing Supply Shop | shop | 1 | — | — | supply interior; vendor and exit retained |
| Crossing Special Shop | shop | 1 | — | — | special interior; vendor and exit retained |
| Rustcoil Weapon Shop | shop | 1 | — | — | weapon interior; vendor and exit retained |
| Rustcoil Armor Shop | shop | 1 | — | — | armor interior; vendor and exit retained |
| Rustcoil Supply Shop | shop | 1 | — | — | supply interior; vendor and exit retained |
| Rustcoil Special Shop | shop | 1 | — | — | special interior; vendor and exit retained |
| Cinder Weapon Shop | shop | 1 | — | — | weapon interior; vendor and exit retained |
| Cinder Armor Shop | shop | 1 | — | — | armor interior; vendor and exit retained |
| Cinder Supply Shop | shop | 1 | — | — | supply interior; vendor and exit retained |
| Cinder Special Shop | shop | 1 | — | — | special interior; vendor and exit retained |
| Frostfen Weapon Shop | shop | 1 | — | — | weapon interior; vendor and exit retained |
| Frostfen Armor Shop | shop | 1 | — | — | armor interior; vendor and exit retained |
| Frostfen Supply Shop | shop | 1 | — | — | supply interior; vendor and exit retained |
| Frostfen Special Shop | shop | 1 | — | — | special interior; vendor and exit retained |
| Stormbreak Weapon Shop | shop | 1 | — | — | weapon interior; vendor and exit retained |
| Stormbreak Armor Shop | shop | 1 | — | — | armor interior; vendor and exit retained |
| Stormbreak Supply Shop | shop | 1 | — | — | supply interior; vendor and exit retained |
| Stormbreak Special Shop | shop | 1 | — | — | special interior; vendor and exit retained |
| Astral Weapon Shop | shop | 1 | — | — | weapon interior; vendor and exit retained |
| Astral Armor Shop | shop | 1 | — | — | armor interior; vendor and exit retained |
| Astral Supply Shop | shop | 1 | — | — | supply interior; vendor and exit retained |
| Astral Special Shop | shop | 1 | — | — | special interior; vendor and exit retained |

## Reproduce

- `npm run test:starfall:maps`
- `npm run test:starfall:map-visuals`
- `node tests/project-starfall/project-starfall-map-encounters.test.js --report`
- `node build/analyze-project-starfall-training.js --route-gate --workers=6 --output=<preflight.json>`
- `node build/analyze-project-starfall-training.js --levels=3,6,14,19,28,50,60,85,100 --workers=6 --output=<current.json>`
- `node build/compare-project-starfall-training.js --current=<current.json> --before=<before.json> --optional=<optional.json> --output=<comparison.json> --csv=<comparison.csv> --optional-csv=<optional-comparison.csv>`
- Exact full-matrix commands and preserved authoring fixtures: [training baseline README](../../output/starfall-map-scenery-review/training-baseline/README.md).

[Before/after scenes](../../output/starfall-map-scenery-review/scene-review.html) · [Structural evidence](../../output/starfall-map-scenery-review/map-audit.json) · [Scenery sources and preview](../../output/starfall-map-scenery-review/index.html) · [Detailed visual notes](../../output/starfall-map-scenery-review/visual-review.md)
