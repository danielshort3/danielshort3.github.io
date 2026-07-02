# Project Starfall Map Optimization Audit

Evidence inspected:
- `project_starfall_gdd_v0_5.md`, especially current world, class, combat, and prototype constraints.
- `js/games/project-starfall/project-starfall-data.js` for maps, world graph, towns, portals, enemies, layout generators, town scenes, and field compositions.
- `js/games/project-starfall/project-starfall-engine.js` for runtime map contracts, spawn cadence, initial spawning, wave replacement behavior, and safe respawn logic.
- `pages/games/project-starfall.html`, `css/games/project-starfall.css`, `js/games/project-starfall/project-starfall-renderer-pixi.js`, and `js/games/project-starfall/project-starfall-ui.js` for the playable canvas shell, Pixi/environment rendering path, UI panels, minimap, combat metrics hooks, and service interaction surface.
- `test.js` Starfall map assertions for current automated guarantees.
- `tests/project-starfall-balance-harness.js` for class-vs-map fit, field efficiency, boss mechanic, and party-scaling analysis hooks.
- `img/project-starfall/maps/` and `img/project-starfall/environment/` for the current map, terrain, ramp, prop, and landmark asset coverage.
- `all-map-terrain-format-contact-sheet.png` for terrain and region material identity.

Important current-state notes:
- Most authored map arrays are not the final runtime maps. `applyPartyPlayGeometry()` replaces safe-zone, dungeon, and most field geometry with shared generators.
- Runtime combat maps currently pass the engine's automated training-route checks: loopable movement, valid spawn placement, distributed spawn coverage, low downtime, and no unreachable broad platforms.
- Passing those checks does not mean the maps are optimized for MMORPG-style long-term play. The biggest issues are sameness, service friction, weak party roles, and insufficient authored route identity in later maps.
- Current field respawns use `FIELD_RESPAWN_DELAY_SECONDS = 3`, `FIELD_RESPAWN_BATCH_SIZE = 3`, safe-player spacing, view-edge preference, local respawn radius, and spawn-pressure decay. Map `waveDelay` is usually 5-8 seconds, but training fields use the lower 3-second field respawn cap.

## A. Executive Summary

Biggest strengths:
- The world graph is already readable: starter town -> forest route -> regional towns -> thematic fields -> dungeons -> late ascension route.
- Every playable combat map has valid runtime traversal and spawn anchors. The current route contract says all non-admin combat maps are loopable and viable.
- The enemy roster has useful behavior families: hoppers, bruisers, chargers, turrets, skirmishers, flyers, healers, armored blockers, elites, and bosses.
- Terrain art and prop files give each biome a visual material set: grass, thorn, ruins, cinder, frost, storm, astral, eclipse, and rift.
- Portals already communicate route direction through labels like Greenroot Gate, Rustcoil Ruins, Gearworks Vault, Glacier Spine, and Endless Rift.

Biggest problems:
- Towns share nearly the same `townVerticalHub` skeleton. Rustcoil Outpost, Cinder Refuge, Frostfen Camp, Stormbreak Haven, and Astral Observatory all use the same service positions and service hierarchy.
- High-frequency town services are split vertically. Storage is at `x=430/p0`, shop is at `x=440/p1`, slots at `x=1020/p4`, upgrade at `x=1750/p8`, and the quest/Plinko NPCs sit around `x=2140-2260/p5`. That is atmospheric, but too much vertical errand tax for repeat visits.
- Starfall Crossing has a likely service-placement bug: the `plinko` station resolves to the default placement at `x=430/p0`, while the Plinko Host NPC is at `x=2260/p5`.
- Many fields are mechanically valid but over-templated. Most vertical fields are `5200` world-width, `31` platforms, `9` broad platforms, `9` spawn points, and similar loop grammar. Shared/switchback fields are usually `8400` width, `37` platforms, and repeated connector density.
- Dungeon and boss rooms are especially same-shaped: most use `dungeonArena`, `4600` width, `19` platforms, `6` broad platforms, `6` spawn points, and `waveMax` 8-9. Their mechanics differ on paper, but the spatial behavior is too similar.
- Current active monster caps are low for true party grinding. Field caps range from 24 to 36, while the current layouts often have 9-12 broad platforms. That gives good solo safety but weak duo/trio lane ownership and weak full-party scaling.
- Later regions lean heavily into verticality plus flyers/ranged mobs. That is exciting for mobile/ranged classes, but melee, tank, and low-mobility classes will feel overtaxed unless routes add shortcuts, reset drops, teleporters, or lane-local objectives.
- Only Greenroot Meadow, Thornpath Thicket, and Bramble Depths have authored field composition metadata. Most later maps use generic Entry/Route/Exit composition, which weakens navigation identity.

Highest-priority fixes:
1. Fix Starfall Crossing Plinko placement and regroup high-frequency services into an obvious ground-level service pocket.
2. Add authored `MAP_FIELD_COMPOSITIONS` for every major field and dungeon, not just the starter forest.
3. Split field generators into distinct solo-loop, party-lane, vertical-shaft, and high-density-farm variants instead of using one broad vertical grammar.
4. Give each dungeon and boss echo a unique arena skeleton tied to its mechanic.
5. Add spawn tuning by section: lane-local caps, per-section respawn pressure, and optional party scaling instead of only global `waveMax`.

Quick wins:
- Add `plinko` to `getTownStationPlacement()` or set `xOverride/platformIndex` for the Plinko station so the station and host agree.
- Put storage, supply shop, repair/equipment shop, and town-return travel on the ground-level main street in every town.
- Move slots, upgrade, broker/market, and crafting one short transition or one stair away.
- Put class/lore/guild/cosmetic NPCs farther out or inside interiors.
- Author portal roles and section labels for Rustcoil Ruins, Oreback Quarry, Cinder Hollow, Ashglass Pass, Frostfen Outskirts, Glacier Spine, Stormbreak Cliffs, Astral Archive, Eclipse Frontier, and Endless Rift.
- Raise spawn caps modestly on party-intended fields before touching damage/EXP values.

Longer-term redesign recommendations:
- Treat each town as a different service culture, not a reskinned hub.
- Convert some late fields into true split-lane party maps with 3-6 sections and central regroup points.
- Add hidden/high-density farming pockets with caps, keys, rotating events, or diminishing returns so they do not obsolete normal maps.
- Instrument KPM, EXP/min, idle time, travel share, platform coverage, class spread, and party overlap before final spawn tuning.

## B. Town-By-Town Audit

### Starfall Crossing

Current role:
- Starter town, class onboarding, shops, storage, upgrade, slots, Plinko, first Greenroot exit.

Layout strengths:
- Strongest town landmark in the game: central meteor/adventurer hall grammar.
- Wider than other towns at `3800` width.
- Has 5 meaningful outbound/interior connections: 4 shop doors and Greenroot Gate.
- Shop interiors are diegetic enough to feel like doors instead of pure UI buttons.

Navigation problems:
- The town has 17 platforms and 10 stairs, but the first-time service path is not clearly prioritized.
- The Greenroot exit is on the ground at `x=2040`, while service NPCs and class/Plinko interactions are split across higher platforms.
- A new player can cross the map within the target travel budget by raw speed, but finding the right vertical service can still feel menu-like or arbitrary.

Service placement issues:
- Storage and shop share almost the same `x` on different platforms.
- Class Supplier and Class Master are separated: station at `x=1290/p10`, Class Master at `x=2140/p5`.
- Plinko station resolves to `x=430/p0`, but the Plinko Host is at `x=2260/p5`. This should be fixed first.

Landmark/identity quality:
- Good: "central meteor plaza/adventurer hall" is memorable.
- Improve: make the meteor or Starfall Guild Hall visually dominate the spawn camera and anchor class onboarding.

Door/portal/interior quality:
- Good shop-door pattern. Keep it.
- Greenroot Gate should preview trees, grass, and low-level slime habitat more explicitly at the right exit.

Suggested layout changes:
- Ground service pocket: storage, potion/supply shop, repair/equipment shop, town travel.
- Upper/class pocket: Class Master, Class Supplier, training dummy, class trial door.
- Market pocket: slots, broker/market, upgrade/crafting, Plinko.
- Put Plinko station and host together, preferably in the market pocket, not split across town.

Suggested visual identity improvements:
- Add a meteor crater/fountain plaza as the social pocket.
- Use banners and class silhouettes around the class hall.
- Give each shop door a distinct facade icon and awning color.

### Rustcoil Outpost

Current role:
- First regional town, construct route hub, Rustcoil Ruins/Oreback Quarry access.

Layout strengths:
- Strong route logic: return to Thornpath on the left, Rustcoil Ruins and Oreback Quarry to the right.
- Gear tower/workshop identity fits armored construct enemies.

Navigation problems:
- It repeats the same town skeleton and station coordinates as later regional towns.
- Ruins and Quarry exits are only `180px` apart on the ground, so the difference between "training route" and "deep route" may be visually undercommunicated.

Service placement issues:
- Storage and shop are close but vertically split.
- Upgrade belongs here thematically, but it is on a high/upper platform. Good for flavor, slightly costly for frequent use.

Landmark/identity quality:
- "Gear tower" is strong, but needs silhouette dominance: large rotating gear, crane arm, or steam stack visible from spawn.

Door/portal/interior quality:
- Shop doors work.
- Rustcoil Ruins exit should look like a broken gear gate. Oreback Quarry should look like a mine lift or rail cart, not just another portal.

Suggested layout changes:
- Ground: storage, supply, repair.
- Workshop upper floor: upgrade/crafting.
- Far-right split: low right gate to Ruins, elevator/rail cart to Quarry.

Suggested visual identity improvements:
- Add scrap piles, conveyor belts, broken clockwork, and a Kafra-like "outpost dispatcher" near the exit split.

### Cinder Refuge

Current role:
- Volcanic safe hub between Oreback Quarry, Cinder Hollow, and Ashglass Pass.

Layout strengths:
- Strong theme: furnace shelter, ash market, Hollow Gate.
- Three route connections create a real regional crossroads.

Navigation problems:
- It uses the same geometry/service pattern as Rustcoil.
- Cinder Hollow and Ashglass Pass exits are close together, but Ashglass is gated by level/dungeon; this needs stronger visual lock state.

Service placement issues:
- Upgrade/crafting should be visually tied to the forge, but frequent potion/repair should remain near spawn.
- Plinko/quest NPC location repeats other towns.

Landmark/identity quality:
- "Furnace shelter" is clear. Make it the social pocket: warm light, inn/tavern, repair/forge, and ember banners.

Door/portal/interior quality:
- Cinder Hollow should be a cave/vent door.
- Ashglass Pass should be a glass bridge/ash road with visible heat haze and a gate lock until requirements are met.

Suggested layout changes:
- Place healer/supply/repair around the furnace shelter.
- Put upgrade artisan in or beside the forge interior.
- Put Ashglass Pass on an elevated bridge so the route feels like a later unlock.

Suggested visual identity improvements:
- Use black stone, orange rim lighting, furnace smoke, glowing vents, and NPCs wearing heat masks.

### Frostfen Camp

Current role:
- Frozen regional camp with Ashglass return, Frostfen Outskirts, Glacier Spine, storage, slots, and upgrade.

Layout strengths:
- The theme naturally supports a compact camp with supply tents and a signal lodge.
- The route split from Outskirts to Glacier is readable in the world map.

Navigation problems:
- Same station and town skeleton as previous hubs.
- The "camp" fantasy is weakened by too many vertical platforms for a temporary shelter.

Service placement issues:
- High-frequency survival services should be clustered around a campfire/signal lodge.
- Upgrade can be a cold-forge or rune heater slightly off the main path.

Landmark/identity quality:
- "Ice signal post" is good but should be taller and more visible.

Door/portal/interior quality:
- Ashglass return and Frostfen field portals are clear labels, but the biome preview should differ: ash road left, snowfield right, glacier lift upper-right.

Suggested layout changes:
- Ground: central fire/signal lodge with supply, storage, quartermaster.
- Upper route: glacier lift with clear warning signage.
- Add a compact inn/safe-room style interior for a Castlevania-like respite before ice maps.

Suggested visual identity improvements:
- Add wind socks, sled crates, frost lanterns, ice crystals, insulated tents, and muffled blue-white lighting.

### Stormbreak Haven

Current role:
- High-altitude town before Stormbreak Cliffs and Astral Observatory.

Layout strengths:
- The "storm mast" landmark can be memorable.
- Route connection to Astral Observatory supports a strong vertical ascension fantasy.

Navigation problems:
- It is still a service clone.
- It needs more vertical town identity than earlier hubs, but high-frequency services should not require extra climbing.

Service placement issues:
- Storage/shop should stay near a sheltered landing platform.
- Upgrade can live in the mast workshop.
- Astral transfer should feel like a sky elevator, airship, or storm bridge.

Landmark/identity quality:
- Potentially strong, currently underused in layout behavior.

Door/portal/interior quality:
- Stormbreak Cliffs and Astral Observatory exits should not be same-height adjacent portals. One should be a field gate, the other a dramatic sky transit object.

Suggested layout changes:
- Ground/shelter platform: supplies, storage, repair.
- Mast platform: upgrade, storm captain, party board.
- Far/upper exit: sky elevator to Astral Observatory.

Suggested visual identity improvements:
- Add cables, wind turbines, lightning rods, cloud layers, flags, and gust audio.

### Astral Observatory

Current role:
- Late-game town for Astral Archive, Eclipse Frontier path, shared account services.

Layout strengths:
- "Star lens" is one of the strongest late-game landmarks.
- Shop interiors and archive market fit the setting.

Navigation problems:
- Same service layout as regional towns, despite being the late-game identity hub.
- Only one field exit from the town, so the extra town width can feel underused.

Service placement issues:
- It needs late-game services: broker/auction, crafting/upgrade, storage, party board, endgame challenge board.
- Current station set lacks differentiated endgame service placement.

Landmark/identity quality:
- Good landmark concept. Make the star lens visible as a moving centerpiece with the archive/portal beneath it.

Door/portal/interior quality:
- Astral Archive door should feel like a library gate. The Stormbreak return should feel like a sky bridge/elevator.

Suggested layout changes:
- Central star-lens plaza: social pocket, party finder, endgame board.
- Left ground: basic services and Stormbreak return.
- Right/upper: Astral Archive gate, Eclipse/Rift foreshadowing, lore library.

Suggested visual identity improvements:
- Add glass floors, star charts, floating books, telescopes, runic elevators, and soft violet/cyan lighting.

## C. Combat-Map-By-Combat-Map Audit

### Field And Dungeon Route Maps

| Map | Intended vs actual | Route, spawn, and terrain read | Class and party read | Practical fix |
|---|---|---|---|---|
| Greenroot Meadow | Intended starter field. Actual shared-lane multi-tier loop with `37` platforms, `24` spawn points, `waveMax 24`. | Valid loop, but too wide and structurally advanced for the first map. The player is introduced to a large multi-tier grind before a simpler lane rhythm is established. | Good solo map, weak party map. Ranged can over-clear long lanes; melee is okay but walks more. | Compress the early section into a true starter loop: bottom slime pond, mid moss lane, one canopy optional route. Keep full width only after the first route milestone. Raise cap to 32-36 only after tutorial completion. |
| Thornpath Thicket | Intended vertical canopy. Actual `verticalCanopy`, `31` platforms, `9` spawn points, `waveMax 26`. | Good theme and route fork, but 9 spawn anchors over 31 platforms can make high/peak travel feel under-rewarded. | Mobility/ranged favored. Turrets plus verticality may punish melee. Party coverage is limited because the map has one vertical route, not separable lanes. | Add one-way drop resets and 2-3 hidden canopy pockets. Put turrets on visible perches with flank paths. Add section-local spawns for low, mid, high. |
| Bramble Depths | Intended dungeon route. Actual generic `dungeonArena`, `6` broad platforms, `waveMax 8`. | Compact and valid, but too similar to other dungeons. Boss staging exists in data, less in terrain. | Solo viable. Party has limited spatial jobs beyond spreading. | Make it a root-lane arena: lower root floor, mid thorn shelf, upper healer pod ledge, boss gate. Use root walls to force lane swaps. |
| Rustcoil Ruins | Intended construct training. Actual `industrialStack`, `9` spawn anchors, `waveMax 28`. | Valid vertical industrial route, but only one return portal and no forward field portal inside the map. It behaves more like a self-contained grind loop than a road. | Armor-break and ranged/turret mix is good. Low-mobility melee gets ladder/ramp tax. | Add gear elevators or short teleporter lifts every third section. Add an optional lower conveyor lane for melee/tank classes. |
| Gearworks Vault | Intended construct dungeon. Actual generic `dungeonArena`, `waveMax 9`, two boss IDs in enemy list. | Good armor-check enemy mix, but the room shape is not a vault-specific encounter. | Party potential is high because tanks, armor-break, and priority turret control exist. Current cap is too low for full party. | Convert to 3-lane factory: lower tank lane, mid sentry catwalk, upper gear-control switches. Spawn 18-22 adds in waves, not 9 static mobs. |
| Cinder Hollow | Intended volcanic cave. Actual `lavaShaft`, `9` spawn anchors, `waveMax 24`. | Strong vertical fire identity, but current cap is low for a 31-platform map. Flyers and throwers can make travel feel like punishment. | Ranged and mobile classes favored; melee needs safe approach paths. | Add vent shortcuts and ground-level ash crawler lanes. Put flyers around route turns, not on every tier. Raise cap to 40-46 when KPM supports it. |
| Emberjaw Lair | Intended lava dungeon. Actual generic `dungeonArena`, `waveMax 8`. | Boss theme exists, but arena does not yet express furnace/overheat behavior spatially. | Solo boss prep okay. Party roles are weak until boss mechanics start. | Add two furnace platforms, vent-safe pockets, and a center overheat lane. Adds should arrive from side vents, not generic spawn points. |
| Bandit Ridge Camp | Intended deep forest/ridge field. Actual `switchbackTerraces`, `19` spawn points, `waveMax 30`. | One of the better party candidates because it has wider lanes and ranged priority mobs. | Supports duo/trio well. Throwers give ranged responsibility, blockers give melee/tank work. Risk: best lower/mid lane may dominate. | Make it explicit split-lane party content: lower cutter lane, mid thrower lookout, high rope bridge, central campfire regroup. Increase cap to 54-60 with lane-local respawns. |
| Oreback Quarry | Intended deep mining/material route. Actual `quarryShaft`, `9` spawn anchors, `waveMax 26`. | Good material fantasy, but vertical shaft plus tanks/healers can feel slow. It should be a party/material farm, not just another vertical stack. | Armor-break classes and ranged target priority benefit. Melee may spend too much time climbing to healers/sentries. | Redesign as mining sections: bottom ore carts, mid scaffold lanes, upper healer mushroom pockets. Make it small-party/high-density with controlled abuse risk. |
| Ashglass Pass | Intended high-level transition route. Actual `lavaShaft`, `waveMax 30`. | Strong bridge/pass fantasy, but it shares Cinder Hollow's vertical grammar. It should feel like a dangerous crossing, not another lava shaft. | Flyers and skirmishers favor range/mobility. Material/elite value can make it an abuse target. | Make it a hub-and-branch route: main ashglass bridge, two side pockets, one elite shortcut. Add hazard bursts sparingly and route-safe alcoves. |
| Frostfen Outskirts | Intended ice training. Actual `switchbackTerraces`, `19` spawn points, `waveMax 31`. | Good use of wide terraces with ice movement. Needs clearer section identity so sliding is tactical rather than friction. | Good solo/duo candidate. Healers/flyers create target priority. | Divide into marsh flats, broken ice shelf, and oracle grove. Add short reset drops so sliding mistakes do not force long backtracking. |
| Glacier Spine | Intended deep glacier vertical route. Actual `glacierClimb`, `9` spawn anchors, `waveMax 32`. | Strong vertical theme, but high travel burden plus ice movement can be tiring. | Mobility/ranged heavily favored. Party can split by height, but regroup is weak. | Add lift/elevator checkpoints and one-way drops. Put sentinels on clear chokepoints and flyers in open air, not hidden high corners. |
| Rimewarden Sanctum | Intended frost dungeon. Actual generic `dungeonArena`, `waveMax 9`. | Ice footing gives identity, but the geometry is still the same arena skeleton. | Party roles exist in enemy mix: brutes, flyers, sentinels, oracle. Cap too low for party pressure. | Make a frozen vault with 3 lanes: cracked lower ice, mid oracle ledge, upper sentinel shelf. Rimewarden should lock/unlock lanes with whiteout waves. |
| Stormbreak Cliffs | Intended storm cliff deep field. Actual `stormClimb`, `9` spawn anchors, `waveMax 32`. | Strong theme, but flyers/throwers/chargers/healers all stacked into a vertical climb risks class bias. | Best for ranged/mobile. Melee needs anchors and short approaches. Party potential is high if lanes are separated. | Add anti-air perches and wind-lift shortcuts. Split into low ram lane, mid archer bridge, high harrier airspace, central lightning rod objective. |
| Astral Archive | Intended late-game archive training. Actual `astralStack`, `waveMax 34`. | Visually distinct through materials, but the map is still a vertical stack. It should feel like library rooms/loops. | Ranged and AoE can dominate if index scribes/void motes cluster. Tanks useful against sentinels. | Make 4 reading-room loops connected by rune lifts. Add line-of-sight shelves for scribes and a center regroup/archive console. |
| Eclipse Frontier | Intended elite-density route. Actual `astralStack`, `waveMax 34`, elite in pool. | Good pre-endgame danger, but it risks being "Astral Archive with darker paint." | High class bias toward ranged/mobility due void motes plus duelists. Farming abuse risk is high if elites are too common. | Make it a frontier patrol map: three outposts, eclipse gate, hidden elite pocket. Use capped elite pulses and rotating sigil zones. |
| Endless Rift | Intended scaling rift training. Actual `riftStack`, `waveMax 36`. | Good endgame concept, but needs stronger replayable structure than another stack. | Can support solo/party if scaling and rewards are controlled. Abuse risk is very high. | Convert to modular circular loop with rift surges. Spawn cap should scale by party count, but rewards need diminishing returns, event timers, or instability stacks. |

### Boss Echo Rooms

| Map | Current issue | Recommended spatial identity |
|---|---|---|
| Brambleking Court | Same `dungeonArena` skeleton as other boss echoes. | Root-court arena with three horizontal root lanes, destroyable thorn pods, and a crown-exposure platform. |
| Titan Foundry | Same skeleton; gear fantasy is not spatial enough. | Rotating gear floor, two side armor-plate switches, upper sentry catwalk. |
| Deepcore Core | Same skeleton; split-lane add pressure is mostly descriptive. | Four mining chambers around a central ore core. Healer lane and turret lane should pull party members apart. |
| Emberjaw Furnace | Same skeleton; furnace cracks need terrain roles. | Lower lava cracks, mid safe platforms, upper vent valves. Boss overheat pulls players to safe pockets. |
| Rimewarden Vault | Same skeleton; ice walls/whiteout should shape movement. | Frozen vault with lane-locking ice walls, central safe room, upper oracle/sentinel ledges. |
| Stormbreak Aerie | Same skeleton; flying boss needs airspace. | Tall open aerie, lightning rods on separate perches, wind lanes that force vertical repositioning. |
| Astral Stacks | Same skeleton; action-memory mechanic needs room grammar. | Library stacks with mirrored left/right platforms and rune shelves that punish repeated skill/position use. |
| Eclipse Throne | Same skeleton; totality mechanic needs zone contrast. | Solar left lane, lunar right lane, central eclipse dais. Players rotate zones instead of staying on one platform. |

### Admin Map

Bandit Animation Lab is correctly admin-only. Keep it isolated from progression and do not tune it for grinding. Its current long shared-lane generator is fine for asset comparison but should not be counted as a player content archetype.

### Combat Viability, Exploit Risk, And Monster Behavior Notes

This table makes the audit criteria explicit for each monster map. The spawn table below gives the numerical spawn recommendations.

| Map | Solo viability | Party viability | Main exploit/class-bias risk | Monster behavior changes |
|---|---|---|---|---|
| Greenroot Meadow | Strong, but current map is larger than a first grind needs. | Low; keep early Greenroot mostly solo. | Long lanes favor ranged line attacks if rewards are too high. | Keep hoppers dominant. Put Thorn Sprouts in visible optional perches, not mandatory early path blockers. |
| Thornpath Thicket | Medium-good for mobile classes, fair for ranged, more taxing for melee. | Duo only unless sections are added. | Safe turret perches and vertical dead travel can make one class/path optimal. | Use Vine Snapper and Briar Stag as route pressure. Add flank paths to Thorn Sprouts so melee has an answer. |
| Bramble Depths | Medium. Boss dungeon should be clearable solo with care. | Small party viable, full party not yet. | Upper safe ledges could trivialize Brambleking/adds. | Make healers and thorn pods phase/objective spawns, not static filler. Boss roots should force lane swaps. |
| Rustcoil Ruins | Medium-good, slower for melee because armored mobs plus vertical travel. | Duo viable if lower/mid lanes have independent spawns. | Armor-break classes may overperform if all high-value mobs are slow tanks. | Put Coil Sentries in line-of-sight posts, Scrap Wardens in chokepoints, Rust Ratchets as movement pressure. |
| Gearworks Vault | Medium solo, but should push armor-break readiness. | Strong candidate for small/full party. | Current low cap makes extra players overlap instead of cooperate. | Separate boss/add roles: sentry control, tank lane, armor-plate switch adds. |
| Cinder Hollow | Medium for ranged/mobile, harder for melee. | Duo viable, weak larger party. | Flyers plus shaft layout can make melee inefficient. | Put flyers at turns/open pockets, not hidden high shelves. Use Ash Crawlers/Lava Ticks to keep grounded routes valuable. |
| Emberjaw Lair | Medium solo boss route. | Small party viable if vents and safe pockets exist. | Safe upper platform can invalidate ground slams. | Spawn adds from side vents. Emberjaw should punish static ledge camping with overheat/fissure patterns. |
| Bandit Ridge Camp | Strong solo/duo. | Best current small-party field candidate. | One mid lane can become the best thrower farm. | Bandit Throwers should kite between lookout platforms; Cutters should hold lower chokepoints; Briar Stag charges should telegraph across long lanes. |
| Oreback Quarry | Medium solo due tank/healer time-to-kill. | Strong small-party material map candidate. | Material drops plus healer pockets can create a single best farm shelf. | Put Glowcap Healers in guarded pockets, Sentries on scaffolds, Beetles on ore-cart lanes. Mimics should be event/cooldown spawns. |
| Ashglass Pass | Medium-high risk route, good solo if hazards are readable. | Duo viable, not full party without branches. | Elite/material value can make side pockets the only farm. | Use Lava Ticks for pressure and Wisps/Spitters sparingly around hazard windows. Add elite only through timed glassstorm pockets. |
| Frostfen Outskirts | Good solo/duo if ice movement is readable. | Small party possible after lane ownership. | Sliding can make ranged safer than melee if every lane has flyers/healers. | Put Shardlings/Frostlings on ground lanes; keep Oracles in reachable pockets; Flyers should patrol over open shelves. |
| Glacier Spine | Medium; high vertical/ice travel is tiring. | Small party viable by height split. | Ranged/mobile classes can own high lanes while melee climbs. | Sentinels should guard chokepoints, Brutes should hold lower lanes, Wisps should be anti-air targets, not constant harassment. |
| Rimewarden Sanctum | Medium solo, better as party dungeon. | Strong with lane-lock mechanics. | Static ledge camping during boss waves. | Rimewarden should lock lanes with ice walls; Oracles and Sentinels should spawn as priority objectives. |
| Stormbreak Cliffs | Medium for melee, strong for ranged/mobile. | Strong small-party potential. | Flyers/throwers/healers together create a major ranged/mobility bias. | Give Thunder Rams long low lanes, Archers mid sightlines, Harriers high airspace, Acolytes objective pockets. |
| Astral Archive | Strong for ranged/AoE, medium for melee. | Duo/small party if rooms are separated. | Chain/AoE classes can dominate if enemies cluster in one vertical stack. | Use line-of-sight shelves for Index Scribes, durable Sentinels as anchors, Void Motes as periodic disruptors. |
| Eclipse Frontier | Medium-high difficulty solo. | Strong small-party/high-risk map. | Elite density and void flyers can create farm abuse and class spread. | Eclipse Duelists should patrol ground/mid lanes; Void Motes should pulse during sigils; elites should be capped events. |
| Endless Rift | Scalable solo if cap and rewards are controlled. | Potential full-party/endgame map. | Highest abuse risk because scaling, elites, and high density combine. | Rift Aberrations should be surge-only. Void/Eclipse mobs should rotate quadrants to force movement. |
| Brambleking Court | Solo boss challenge. | Small party boss. | One safe root shelf can negate adds. | Root waves, Thorn Sprout pods, and Glowcap Healers should force target priority. |
| Titan Foundry | Solo boss challenge. | Small/full party boss. | Armor-break classes may trivialize if switches are optional. | Clockbugs/Sentries should support armor phases; Scrap Wardens guard switches. |
| Deepcore Core | Hard solo, better as party boss. | Full party candidate. | Healer lane can be ignored if boss DPS race is better. | Glowcap Healers and Coil Sentries should force split control during quake anchors. |
| Emberjaw Furnace | Solo boss challenge. | Small party boss. | Static upper safe spots during lava charges. | Lava Tick side spawns and Cinder Spitters should deny ledge camping during overheat. |
| Rimewarden Vault | Medium-hard solo. | Small/full party boss. | Ice movement may punish melee too hard if every add is ranged/flying. | Mix lower Brute pressure with reachable Oracle/Sentinel objectives. |
| Stormbreak Aerie | Hard solo for melee, good ranged challenge. | Full party boss potential. | Ranged uptime can dominate if lightning rods are optional. | Harriers/Roc airspace should be paired with ground rod duties and Ram lane pressure. |
| Astral Stacks | Medium-hard solo. | Small party boss. | Repeated skill/position mechanics may not matter if arena is generic. | Index Scribes and Void Motes should mirror player routes and punish staying in one stack. |
| Eclipse Throne | Endgame solo/party boss. | Full party capstone. | Solar/lunar zones become cosmetic if no spatial rotation is required. | Eclipse Duelists, Void Motes, and Sentinels should enforce solar/lunar lane swaps around the central dais. |

## D. Map Archetype Classification Table

| Map | Current archetype | Intended archetype | Best use case | Main problem | Recommended fix |
|---|---|---|---|---|---|
| Greenroot Meadow | C/F long shared lanes | A -> F starter lane loop | Starter solo | Too large/advanced for first grind | Compress first loop and unlock wider route later |
| Thornpath Thicket | D vertical canopy | D with F reset loop | Solo/duo vertical mastery | High travel for 9 anchors | Add reset drops and canopy pockets |
| Bramble Depths | H generic arena | H/E root dungeon | Dungeon staging | Same arena skeleton | Root lane swaps and thorn pod roles |
| Rustcoil Ruins | D industrial stack | C/F industrial terrace | Solo/duo construct grind | Vertical tax on armored mobs | Gear lifts and lower melee lane |
| Gearworks Vault | H generic arena | H/E armor-check party arena | Small/full party dungeon | Too few adds and generic room | 3-lane factory with switches |
| Cinder Hollow | D lava shaft | C/F hazard-lite route | Solo/duo fire grind | Flyers plus verticality overbias range | Add vent shortcuts and grounded lanes |
| Emberjaw Lair | H generic arena | H/I furnace arena | Boss prep | Furnace mechanic not spatial | Vent-side add waves and overheat safe pockets |
| Bandit Ridge Camp | C switchback terraces | E split-lane party map | Duo/trio party grind | Party roles not explicit | Lower/mid/high camp lanes and regroup fire |
| Oreback Quarry | D quarry shaft | E/J material party map | Small-party farming | Slow vertical tank/healer route | Mining sections with lane-local objectives |
| Ashglass Pass | D lava shaft | G/F dangerous crossing | Exploration/elite route | Too similar to Cinder Hollow | Main bridge plus side pockets |
| Frostfen Outskirts | C switchback terraces | C/E ice solo/duo route | Solo/duo training | Sliding lacks section identity | Marsh/shelf/grove sections and reset drops |
| Glacier Spine | D glacier climb | D/G deep route | Progression and party split | Ice plus vertical travel fatigue | Lifts, one-way drops, sentinel chokepoints |
| Rimewarden Sanctum | H generic arena | H/I frost vault | Party dungeon | Geometry not vault-like | Lane-locking ice walls and oracle shelf |
| Stormbreak Cliffs | D storm climb | D/E anti-air party field | Small party | Strong ranged/mobility bias | Wind lifts, anti-air perches, rod objective |
| Astral Archive | D astral stack | C/F room-loop archive | Late solo/duo grind | Feels like stack, not archive | Reading-room loops and rune lifts |
| Eclipse Frontier | D astral stack | G/J elite frontier | High-risk farming | Too close to Astral Archive | Patrol outposts and capped elite pocket |
| Endless Rift | D rift stack | F/J scaling loop | Endgame farming | High abuse risk, repetitive stack | Modular circular loop with surge rules |
| Brambleking Court | H generic arena | H/E root boss room | Boss mastery | Same geometry as other echoes | Root-court lanes and destroyable thorn pods |
| Titan Foundry | H generic arena | H/E factory boss room | Boss mastery and armor-break party play | Gear fantasy is not spatial | Gear floor, armor switches, and sentry catwalk |
| Deepcore Core | H generic arena | H/E quarry core boss room | Full-party control check | Split-lane pressure is mostly descriptive | Four chambers around ore core with healer/turret lanes |
| Emberjaw Furnace | H generic arena | H/I furnace boss room | Boss mastery with hazards | Furnace cracks do not shape movement | Lower lava cracks, vent valves, and safe pockets |
| Rimewarden Vault | H generic arena | H/I frost vault boss room | Party dungeon boss | Ice walls are not spatially enforced | Lane-locking ice walls and oracle/sentinel shelves |
| Stormbreak Aerie | H generic arena | H/D vertical flying boss room | Full-party anti-air fight | Flying boss lacks true airspace | Tall aerie with rod perches and wind lanes |
| Astral Stacks | H generic arena | H/G mirrored archive boss room | Position-memory boss mastery | Generic room weakens action-memory mechanic | Mirrored stack shelves and rune lanes |
| Eclipse Throne | H generic arena | H/I capstone zone-rotation room | Endgame boss mastery | Solar/lunar zones can become cosmetic | Solar/lunar lanes around central eclipse dais |

## E. Spawn Tuning Table

These are starting targets for MMORPG-style testing. They should be validated against actual KPM, EXP/min, damage taken, and class spread before finalizing.

| Map | Recommended total active spawn count | Recommended distribution | Respawn timing | Elite/miniboss/event use | Farming abuse risk |
|---|---:|---|---|---|---|
| Greenroot Meadow | 32-36 | 12 pond/low, 10 moss/mid, 8-10 canopy, 2 tutorial turrets | 4s, batch 3 | No elite until route clear; rare harmless shiny slime later | Low |
| Thornpath Thicket | 40-44 | 14 low vines, 12 mid sprout shelves, 10 high canopy, 2-4 briar chargers | 4s, section-local | Briar Stag pulse every 4-5 minutes | Medium |
| Bramble Depths | 14-18 adds plus boss | Root floor adds, mid thorns, upper healer pod | 8s waves | Brambleking phase adds | Medium |
| Rustcoil Ruins | 42-48 | 16 lower ratchets/clockbugs, 14 mid sentries, 10 high wardens, 2 turret posts | 4-5s | Scrap Warden patrol | Medium |
| Gearworks Vault | 18-22 adds plus boss targets | Lower tanks, mid sentries, upper switches | 8s waves | Clockwork Titan/Quarry Colossus phases | Medium |
| Cinder Hollow | 40-46 | 12 ground crawlers, 12 lava ticks, 10 flyers, 4-6 spitters | 4-5s | Ember vent burst | Medium-high |
| Emberjaw Lair | 16-20 adds plus boss | Side vents, mid platforms, overheat safe pockets | 8s waves | Emberjaw overheat add wave | Medium |
| Bandit Ridge Camp | 54-60 | 14 lower cutters, 12 mid throwers, 12 high rope bridge, 12 deep camp, 2 elite patrols | 4s, lane-local | Bandit Captain pulse | Medium |
| Oreback Quarry | 56-64 | 16 ore lane, 14 scaffold, 12 mushroom/healer, 12 upper sentry, 2-4 mimic/ore events | 5s | Cracked Mimic and ore-cart event | High |
| Ashglass Pass | 46-54 | 18 bridge route, 12 vent pockets, 10 high glass shelf, 4-6 elite side pocket | 5s | Cracked Mimic or glassstorm event | High |
| Frostfen Outskirts | 50-58 | 16 marsh flats, 14 ice shelf, 12 oracle grove, 8-12 flyer/sentinel | 4-5s | Icebloom pulse | Medium |
| Glacier Spine | 52-60 | 14 lower climb, 14 mid ridge, 12 high ridge, 8 flyers, 4 sentinels/oracles | 5s | Rime crystal objective | Medium-high |
| Rimewarden Sanctum | 18-22 adds plus boss | Lower brutes, mid oracle, upper sentinel/flyer | 8s waves | Rimewarden ice-wall phases | Medium |
| Stormbreak Cliffs | 54-62 | 14 ram lane, 14 archer bridge, 12 harrier airspace, 10 support/rod zone, 2 elite | 4-5s | Lightning rod event | High |
| Astral Archive | 58-66 | 14 per reading-room loop across 4 rooms plus 2 elite/mote anchors | 4s | Archive index event | High |
| Eclipse Frontier | 60-72 | 16 outpost A, 16 outpost B, 14 eclipse gate, 10 high void lane, 4-8 elite pulses | 5-6s | Eclipse sigil, capped mimic/duelist elite | Very high |
| Endless Rift | 64-76 solo/duo, 88-96 party instance | 4 loop quadrants, each 16-24, with surge pockets | 4-5s, scales by party | Rift surge, instability stacks, reward caps | Very high |
| Brambleking Court | 12-16 adds plus boss | Lower root lane, mid thorn shelf, upper pod shelf | 8-10s phase waves | Thorn pod/root bind phases | Medium |
| Titan Foundry | 14-18 adds plus boss | Lower gear floor, side switch pads, upper sentry catwalk | 8-10s phase waves | Armor-plate switch events | Medium |
| Deepcore Core | 16-20 adds plus boss | Four chambers around core: tank, healer, turret, ore | 8-12s phase waves | Quake anchors and healer lane pulses | Medium-high |
| Emberjaw Furnace | 14-18 adds plus boss | Side vents, mid safe shelves, upper valve platforms | 8-10s phase waves | Overheat vent and fissure events | Medium |
| Rimewarden Vault | 14-18 adds plus boss | Lower brutes, mid oracle lane, upper sentinel shelves | 8-12s phase waves | Ice-wall and whiteout phases | Medium |
| Stormbreak Aerie | 16-20 adds plus boss | Low ram lane, rod perches, high harrier airspace | 8-10s phase waves | Lightning rod charge windows | Medium-high |
| Astral Stacks | 12-18 adds plus boss | Mirrored left/right stacks and center rune shelf | 8-10s phase waves | Memory-rune punish windows | Medium-high |
| Eclipse Throne | 16-20 adds plus boss | Solar lane, lunar lane, central eclipse dais, upper mote shelf | 8-12s phase waves | Totality rotation and elite pulse | High |

Spawn principle for implementation:
- Keep current safe-spawn logic. It is useful.
- Add section IDs to spawn points so respawns can fill the lane the player or party is actually clearing.
- For party maps, scale by occupied sections and party count, not just total map cap.
- Do not raise rewards until KPM and EXP/min are measured by class and party size.

## F. Party-Play Recommendations

Solo only:
- Greenroot Meadow early section.
- Bandit Animation Lab, because it is admin-only.

Solo/duo:
- Thornpath Thicket, Rustcoil Ruins, Cinder Hollow, Ashglass Pass, Frostfen Outskirts, Astral Archive.
- Requirements: clear personal loop, enough spawns per visited section, and reset drops to reduce ladder tax.

Small party:
- Bandit Ridge Camp, Oreback Quarry, Glacier Spine, Stormbreak Cliffs, Eclipse Frontier, Endless Rift.
- Requirements: 3-5 sections, lane-local spawns, support regroup point, and at least one role-specific objective.

Full party:
- Gearworks Vault, Rimewarden Sanctum, boss echo rooms, future Endless Rift party instance.
- Requirements: 4-6 meaningful sections, boss/add split, anti-air/ranged duty, tank chokepoints, and support-safe regroup pockets.

Party-only:
- None should be hard party-only in the current prototype. Keep content solo-accessible, but make dungeons and boss echoes noticeably better with coordinated roles.

High-density farming:
- Oreback Quarry, Ashglass Pass, Eclipse Frontier, Endless Rift.
- Add caps, event timers, anti-AFK pressure, contested objectives, or reward decay so these do not become the only optimal maps.

Recommended party map patterns:
- Bandit Ridge: lower melee/tank lane, middle thrower lane, high rope bridge for ranged, central campfire support point, elite patrol crossing all lanes.
- Oreback Quarry: lower ore carts for tanks, scaffold sentries for ranged, healer mushroom pocket for burst classes, central ore lift regroup.
- Stormbreak Cliffs: low thunder ram lane, mid archer bridge, high harrier airspace, lightning rod objective that support/control classes can manage.
- Endless Rift: 4 rotating quadrants with surge warnings; party splits to hold quadrants, then regroups at the rift core for elite pulses.

Avoid:
- Raising global spawn caps without section ownership.
- Letting Storm Mage, Fire Mage, or Trapper erase every section from one safe platform.
- Letting Sniper/Archer solve all flyer maps without ground threats.
- Letting tanks have no spatial job except taking contact damage.

### Progression Flow Audit

Current strengths:
- The macro progression is sensible: Starfall Crossing -> Greenroot -> Thornpath/Bandit -> Bramble dungeon -> Rustcoil/Oreback -> Cinder -> Frostfen -> Stormbreak/Astral -> Eclipse/Rift.
- Enemy behavior complexity rises over time: hoppers and bruisers first, then turrets/chargers, then armored/tanks/healers, then flyers/throwers/elites.
- Regional towns appear at good breakpoints and can serve as service resets between routes.

Current progression problems:
- Greenroot Meadow starts with a large multi-tier map instead of a very simple first hunting grammar.
- Thornpath introduces verticality and ranged turrets quickly, before a pure circular solo loop is established.
- Midgame party structure is not explicit enough. Bandit Ridge and Oreback Quarry want to be duo/trio maps, but current data only marks them as deep fields.
- Late-game vertical maps stack movement friction, flyers, ranged pressure, and elite value. That risks making mobility/ranged classes feel mandatory.
- Mobility upgrades currently make long/vertical maps more tolerable, but the maps need more routes that are actually unlocked or optimized by mobility: shortcuts, one-way drops, wind lifts, rune stairs, and hidden ledges.

Recommended progression shape:
- Levels 1-10: simple lane loops and one optional upper platform.
- Levels 10-25: terraces, visible ranged enemies, and first route forks.
- Levels 25-45: split-lane duo maps, dungeon staging, armor-break and healer priority.
- Levels 45-70: hazards and verticality, but with shortcut infrastructure.
- Levels 70+: elite objectives, rotating events, high-density pockets, and party section ownership.

## G. Example Redesigned Layouts

### Starfall Crossing - Service Plaza Redesign

Platform arrangement:
- Ground main street from spawn to Greenroot Gate.
- Central meteor plaza around `x=900-1500`.
- Upper class hall balcony over the plaza.
- Market roofwalk to the right.

Player route:
- Spawn -> storage/supply/repair in first 5 seconds.
- Continue -> meteor social plaza/class hall.
- Continue -> upgrade/slots/Plinko market.
- Far right -> Greenroot Gate.

Portal/interior placement:
- Four shop doors grouped at the market facade, but the supply door is closest to spawn.
- Class hall door beneath the Adventurer Hall landmark.
- Greenroot Gate framed by trees and slime habitat preview.

Intended behavior:
- New players immediately find survival services.
- Returning players can dump loot, buy potions, repair, and exit without climbing.
- Social and lower-frequency systems still give the town a place-like structure.

### Greenroot Meadow - Starter Loop Redesign

Platform arrangement:
- Bottom slime pond lane, one short mid moss platform, one optional canopy platform.
- Later section expands into the current multi-tier route after a visible sign/quest threshold.

Player route:
- Clear bottom lane left to right.
- Jump to mid moss lane.
- Drop to bottom near the start.
- Repeat as enemies repopulate.

Spawn sections:
- Pond: Dew Slime and Slimelet.
- Moss lane: Mossback.
- Canopy: 1-2 Thorn Sprouts, visible and optional.

Class interactions:
- Melee learns contact and knockback.
- Ranged learns line shots.
- Mage learns small AoE without needing vertical control.

### Bandit Ridge Camp - Split-Lane Party Redesign

Platform arrangement:
- Lower cutter lane with crates and chokepoints.
- Middle camp lane with Bandit Thrower platforms.
- Upper rope bridge with sniper/archer sightlines.
- Central campfire/tent as regroup point.

Player route:
- Solo: lower -> middle -> drop -> repeat.
- Duo/trio: one player lower, one middle/high, regroup at campfire for elite patrol.

Spawn sections:
- Lower: Bandit Cutters and Vine Snappers.
- Mid: Bandit Throwers and Cutters.
- High: Throwers and Briar Stag charge lanes.
- Elite pulse: Bandit Captain patrol crosses sections.

Party strategy:
- Tank/frontliner holds lower chokepoints.
- Ranged handles throwers.
- Support regroups at campfire and rotates buffs.

### Oreback Quarry - Material Party Farm Redesign

Platform arrangement:
- Lower ore-cart loop.
- Mid scaffold loop.
- Upper mushroom/healer pocket.
- Side mine shaft hidden farm room.

Player route:
- Solo: ore cart -> scaffold -> drop through -> repeat.
- Party: split ore/scaffold/mushroom, regroup at lift for ore-cart event.

Spawn sections:
- Ore lane: Oreback Beetles and Scrap Wardens.
- Scaffold: Coil Sentries.
- Mushroom pocket: Glowcap Healers plus guarded ore nodes.
- Hidden pocket: short high-density beetle room with cooldown/entry cost.

Abuse controls:
- Hidden farm room has a timed ore cart, resource key, or diminishing reward after repeated clears.

### Stormbreak Cliffs - Anti-Air Party Field

Platform arrangement:
- Low ram lane with long telegraph distance.
- Mid archer bridge with cover gaps.
- High harrier airspace with wind lifts.
- Central lightning rod objective.

Player route:
- Solo: clear low ram lane, take wind lift to mid, drop to low.
- Party: split low/mid/high, regroup at lightning rod when it charges.

Monster types:
- Thunder Ram on low lanes.
- Stormbound Archer on mid bridges.
- Gale Harrier in high airspace.
- Cloudcall Acolyte near rod objective.

Class interactions:
- Melee gets clean charger lanes.
- Ranged gets anti-air duty.
- Supports/control classes manage rod/acolyte pressure.

### Endless Rift - Scaling Circular Loop

Platform arrangement:
- Four quadrant loop around a rift core.
- One-way drops and short teleporters return players to the next quadrant.
- Rift core opens during surge events.

Player route:
- Clear quadrant A -> teleporter -> B -> drop -> C -> lift -> D -> core.
- Return to A as enemies repopulate.

Spawn sections:
- Each quadrant has a local cap.
- Rift Aberrations spawn during surge windows only.
- Void Motes and Eclipse Duelists are distributed to avoid one safe platform.

Abuse controls:
- Instability stacks increase danger and reward for a limited time.
- Rewards normalize if players camp one quadrant without rotating.

## H. Visual Identity Recommendations

| Region/town | Landmark | Architecture | Palette/lighting | Props/background | Audio mood | NPC flavor | Nearby monster family | Portal/interior style |
|---|---|---|---|---|---|---|---|---|
| Starfall Crossing | Meteor plaza / Adventurer Hall | Guild hall, market awnings | Warm gold, teal sky | Flowers, banners, meteor fragments | Safe town theme, light crowd | Guides, class trainers, merchants | Slimes, moss beasts, thorns | Forest gate, shop doors, class hall |
| Rustcoil Outpost | Gear tower | Scrap workshops, cranes | Brass, steel, teal sparks | Gears, rails, crates, steam | Clanks, low machine hum | Foremen, salvagers, mechanics | Constructs, beetles, sentries | Ruins gate, quarry lift, workshop interiors |
| Cinder Refuge | Furnace shelter | Basalt walls, forge shelters | Black stone, orange glow | Vents, ash, heat shimmer | Furnace pulse, muffled drums | Smiths, heat-masked scouts | Wisps, crawlers, lava ticks | Cave vent, ashglass bridge, forge interior |
| Frostfen Camp | Ice signal lodge | Tents, lodges, watchposts | Blue-white, warm campfire | Sleds, frost lanterns, crystals | Wind, soft bells, fire crackle | Trackers, quartermasters | Frostlings, wisps, brutes | Tundra gate, glacier lift, safe-room lodge |
| Stormbreak Haven | Storm mast | Suspended platforms, cables | Slate, cyan lightning, gold | Rods, flags, turbines, clouds | High wind, distant thunder | Captains, sky engineers | Harriers, archers, rams | Cliff span, sky elevator, mast workshop |
| Astral Observatory | Star lens | Observatory, archive arches | Violet, cyan, glass highlights | Star charts, floating books, lenses | Celestial pads, quiet chimes | Scribes, astronomers, endgame brokers | Scribes, sentinels, void motes | Archive gate, rune lift, lore library |
| Eclipse Frontier/Rift | Eclipse gate / rift lens | Broken astral ruins | Black, solar gold, cyan/violet | Sigils, fractured horizons | Tension drone, pulse surges | Watchers, frontier scouts | Duelists, aberrations, void | Eclipse gate, unstable rift portals |

Reusable layout skeletons can stay, but swap:
- Landmark silhouette.
- Portal fiction.
- Background horizon.
- Lighting and ambient audio.
- NPC clothing/culture.
- Spawn family.
- Verticality profile.
- Traversal object: vines, ropes, gear lifts, chains, frost ladders, wind lifts, rune stairs.

## I. Future Map Design Checklist

Town checklist:
- Does the town have one landmark the player can name?
- Can a new player find storage, supply/potions, repair/shop, quick travel, and core quest turn-ins without opening a menu?
- Are high-frequency services within 5-10 seconds of spawn or the main return portal?
- Are medium-frequency systems one short transition away?
- Are flavor/lore/guild/cosmetic services farther out or in interiors?
- Are doors and portals diegetic?
- Do exits preview the next biome?
- Can the town be crossed without mobility skills in roughly 20-40 seconds?
- Does the town have a social pocket?
- Can the town be described in one phrase?

Combat map checklist:
- What is the intended archetype: flat lane, dense chamber, terrace, vertical shaft, split-lane party, circular loop, hub-and-branch, arena, hazard map, hidden farm?
- What is the one-sentence player route?
- Does the player return to each section as enemies repopulate?
- Is travel share under 30 percent?
- Is idle time under 10-15 percent?
- Are spawns distributed by section, not just by total map?
- Are any spawn points unreachable, hidden, or too close to safe platforms?
- Does each platform have a job?
- Does each ladder/rope/lift have a payoff?
- Which classes are favored, and is the reward gap likely above 20-25 percent?
- Can a party split without overlap above 25-40 percent?
- Is there a support regroup point?
- Does the map have a risk/reward reason to revisit?
- Is any high-density pocket capped or otherwise controlled?

Spawn tuning checklist:
- Measure route cycle time.
- Measure KPM, EXP/min, drop value/min, damage taken/min, potion use, death rate.
- Measure platform coverage and spawn vacancy.
- Measure ladder/rope time.
- Measure best-class vs median-class reward gap.
- Measure party reward efficiency against solo.
- Tune spawn count, spawn section, respawn timing, and reward together.

## J. Prioritized Implementation Roadmap

Must fix now:
1. Fix Plinko station placement in Starfall Crossing so the station and host are colocated.
2. Move or duplicate high-frequency town services into a clearly marked ground-level service pocket.
3. Add authored field compositions for all non-starter route maps.
4. Add section IDs to spawn points so respawn tuning can target lane-local behavior.
5. Document intended archetype per map in data, not just inferred from `layoutStyle`.

Should improve soon:
1. Redesign Bandit Ridge Camp as the first true split-lane party map.
2. Redesign Oreback Quarry as a small-party/material farm with abuse controls.
3. Convert Cinder Hollow and Ashglass Pass into different route shapes instead of two lava-shaft variants.
4. Add shortcuts/reset drops to Thornpath Thicket, Glacier Spine, Stormbreak Cliffs, Astral Archive, Eclipse Frontier, and Endless Rift.
5. Increase field wave caps in staged passes and measure KPM/EXP/min before changing rewards.

Nice-to-have polish:
1. Add more town-specific props and NPC cultures.
2. Add route-specific portal visuals: mine carts, cave mouths, sky elevators, rune gates, rift lenses.
3. Add social pockets to every town.
4. Add hidden farming pockets with event timers.
5. Add map-specific ambient audio loops.

Future expansion ideas:
1. Party-size spawn scaling by occupied section.
2. Pylon/Kafra-style regional travel unlocks tied to town discovery and quest completion.
3. Safe rooms before major dungeons with boss prep services.
4. Weekly rotating high-density maps with reward caps.
5. World map overlays for "solo", "duo", "party", "elite", and "farm" tags.
6. A map analytics panel showing idle time, travel share, platform coverage, and class spread from real play sessions.
