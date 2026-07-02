# Project Starfall Balance Optimization Audit

Date: 2026-06-11

## Executive Summary

Project Starfall already has a strong RPG systems foundation: three readable base classes, nine advanced jobs, role profiles, class resources, movement skills, party skills, boss encounter metadata, multi-layout maps, monster guide progression, accomplishments, daily rewards, cards, upgrades, attunement, and combat metrics.

The biggest balance risk is not missing systems. It is uncontrolled interaction between broad systems: advanced jobs can inherit base skills, party buffs add multipliers, attunement adds specialized damage bonuses, monster guide mastery adds damage, and map geometry strongly changes real class value. Without guardrails, one map/skill/class loop can quietly become the only efficient path.

Highest-priority fixes completed in this pass:
- Reworked the Starfall balance harness so sparse advanced jobs preserve inherited base-skill filler instead of appearing like dead-end one-button kits.
- Limited low-cooldown trainer AoE, explosive projectiles, and trap skills in the benchmark so they no longer model as full-pack damage every cast.
- Added field-map efficiency and progression pacing benchmarks using real map level ranges, enemy mixes, layout archetypes, wave pacing, and class-map fit.
- Added scenario and map spread guardrails to `test.js` so future changes catch runaway class dominance, trap-level underperformance, stale progression fixtures, and pacing drift.
- Expanded combat metrics so kills, boss kills, elite kills, drops, potion cost, deaths, damage taken, skill casts, skill damage, and loot kinds are persisted and rate-tracked.
- Added boss/party benchmark coverage for encounter mechanic categories, deterministic progress, chase rewards, dungeon objectives, party loadouts, and sublinear party HP scaling.
- Added monster TTK and reward-cadence estimates to the field-map and progression benchmarks.
- Fixed optional monster drop tables so normal, elite, and boss base drop chances actually scale non-guaranteed tables, then tuned the normal base chance from 12% to 8%.
- Refined field spawn-area keys so vertically separated platform lanes count as distinct spawn areas instead of collapsing into the same section bucket.
- Added level-curve spike auditing so time-to-level increases above 25% must be justified by an authored unlock or content breakpoint.
- Added reward-source health auditing so field loot is split by source table, reward family, rarity, deterministic progress, optional rare cadence, and source dominance risk.
- Added field survivability and potion-pressure auditing so damage taken, purchased potion replacement cost, and death-risk outliers are checked by class and map.
- Added encounter-specific boss HP budgets plus boss TTK auditing so custom bosses benchmark at 6-12 minute median solo clear pacing instead of collapsing into short DPS checks.
- Added boss chase dry-streak auditing so Epic/Relic set drops are evaluated through the actual soft-pity curve instead of base drop chance alone.
- Added support-contribution auditing so party value is scored across mitigation, party damage, control/debuffs, sustain, mobility/uptime, and objective utility instead of personal DPS alone.
- Surfaced map-tuning and retention-friction auditing so combat maps report idle time, travel share, spawn vacancy, class spread, party overlap, abuse-control coverage, and abandonment/repeat-visitation risk.
- Added equipment/upgrade health auditing, filled missing level-25 class-supplier offhands for Duelist, Storm Mage, and Beast Archer, and made non-destroy upgrade failures salvage partial Upgrade Dust.
- Added economy health auditing plus runtime currency-spend telemetry so coin faucets, coin sinks, item purpose coverage, market listings, and net currency rate can be tracked.
- Added retention/chore health auditing so daily login rewards, season goal time, long-term progression lanes, player-type coverage, catch-up sources, and cash-shop power risk are measured.
- Added damage/stat formula auditing so additive source coverage, positive multiplier buckets, base-class stat identity, attunement caps, and defense mitigation are checked automatically.
- Added skill-system health auditing so every playable owner is checked for core rotation, movement/survival, role utility, identity cooldown, authored purpose, prerequisite validity, inherited base-loop preservation, and obsolete base-skill risk.
- Added enemy ecosystem health auditing so live combat enemies, archetypes, map distribution, bracket variety, and counter-tool coverage are checked before one enemy/map pattern can dominate progression.

## Current Benchmark

Source: `npm run analyze:starfall:balance`, level 50, rank 10, equal gear assumptions.

Class/job DPS leaders after this pass:

| Scenario | Top classes/jobs | Balance read |
| --- | --- | --- |
| Single boss | Archer 1210, Duelist 1209, Berserker 1190, Sniper 1184, Storm Mage 1079 | Boss lane has several viable leaders. Archer's base sustained damage is now the top watch item. |
| Clustered pack | Storm Mage 2103, Trapper 2032, Fire Mage 1695 | Mobbing lane now has three leaders instead of one runaway Trapper result. |
| Spread pack | Storm Mage 2013, Trapper 1645, Fire Mage 1511, Archer 1508, Rune Mage 1409 | Storm still leads wide pack estimates; verify in real maps because route time may favor Archer. |
| Armored target | Guardian 1364, Fighter 1191, Sniper 1169, Archer 1157, Rune Mage 1121 | Guardian has a clear armor/control lane without replacing all boss jobs. |
| Flying pack | Storm Mage 2289, Archer 1732, Fire Mage 1705, Mage 1494, Rune Mage 1275 | Flying remains a deliberate melee weakness. Watch Storm's ceiling in flyer-heavy brackets. |
| Sustain boss | Guardian 1776, Archer 1129, Berserker 1129, Fighter 1076, Sniper 1060 | Guardian's support/survival lane is visible; count it as utility value, not raw DPS only. |

Field-map efficiency leaders after this pass:

| Map | Level | Top hourly efficiency | TTK | Reward cadence | Balance read |
| --- | ---: | --- | ---: | --- | --- |
| Greenroot Meadow | 4 | Archer 103, Mage 100, Fighter 89 | 3.2s | 2.9/13.7m | Early fields now benchmark base classes only; no level-25 subclass leakage. |
| Cinder Hollow | 28 | Storm Mage 139, Fire Mage 121, Rune Mage 112 | 4.6s | 4.9/22.8m | Vertical flyer route correctly favors elemental mobbers, but Storm remains a watch item. |
| Oreback Quarry | 30 | Trapper 115, Rune Mage 110, Guardian 106 | 5.7s | 6.1/28.3m | Armored/support route no longer defaults to Storm chain clearing. |
| Frostfen Outskirts | 52 | Trapper 119, Fire Mage 113, Storm Mage 110 | 7.1s | 4.3/20.2m | Wide support-heavy route rewards trap routing and fire pressure. |
| Stormbreak Cliffs | 63 | Storm Mage 135, Fire Mage 117, Rune Mage 113 | 7.1s | 4.3/19.9m | Flyer/ranged cliff route remains Storm-favored without exceeding the map guardrail. |
| Endless Rift | 100 | Storm Mage 126, Fire Mage 111, Rune Mage 109 | 11.5s | 4.1/19.0m | Endgame mixed route keeps a Storm edge but not a runaway hourly-value edge. |

Progression pacing estimate:

| Bracket | Median time-to-level | Median TTK | Small/medium reward | Target |
| --- | ---: | ---: | --- | --- |
| Early | 5.1 minutes | 3.2s | 2.9/13.7m | 3-8 minutes, 2-4s TTK |
| Midgame | 17.5 minutes | 5.5s | 5.8/27.2m | 10-25 minutes, 4-7s TTK |
| Late | 38.5 minutes | 7.1s | 4.2/19.8m | 25-60 minutes, 6-10s TTK |
| Endgame | 60.1 minutes | 11.5s | 4.1/19.0m | 60-240 minutes per major step proxy, 8-15s TTK |

Level-curve spike audit:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Checked levels | 2-112 | Covers the authored map/enemy/equipment/boss-source range currently represented in data. |
| Max time-to-level increase | 30.9% at level 9 | The only >25% spike is the authored early-to-midgame bracket transition. |
| Unjustified spikes | 0 | No level step exceeds 25% without an unlock/content breakpoint. |

Damage/stat formula health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Formula buckets | 9 | Base offense, additive damage, class/status windows, crit, target type, execute, defense mitigation, monster mastery, and variance are now named. |
| Positive multiplier buckets | 7 | Still controlled, but this is the ceiling to watch before adding new damage systems. |
| Additive stat sources | 11 | Base class, level, gear, cards, roster, party, specialization, stat upgrades, permanent bonuses, passives, and buffs are visible. |
| Base-class raw power spread | 7.4% max | Warrior, Archer, and Mage keep tight raw-offense parity before skill/map identity modifiers. |
| Warrior effective HP floor | 107.6 index | Warrior keeps a measurable durability edge without becoming mathematically immune. |
| Mage MP floor | 108.9 index | Mage resource identity is visible even when raw power stays close. |
| Archer speed/range floors | 106.8 / 113.2 index | Archer route identity is visible through movement and range rather than raw damage. |
| Attunement multiplier caps | Rare 7, Relic 23, Celestial 54 | Early percent lines remain modest; high-end values stay prestige-gated. |
| Defense mitigation samples | 100 raw -> 100/64/47/31/18 at 0/25/50/100/200 defense | Mitigation is monotonic and meaningful without creating near-immunity. |
| Formula issues | 0 | Current data passes formula-bucket, class-identity, multiplier-cap, and mitigation-curve checks. |

Skill-system health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Skills checked | 85 | Skill audit covers all authored base and advanced job skills. |
| Active/passive split | 70 / 15 | Active kits are broad enough for rotations while passive count stays controlled. |
| Purpose coverage | 100% | Every skill has an authored purpose such as trainer, mobility, setup, defense, finisher, or party. |
| Accessible family coverage | 12/12 owners | Every playable class/job can access core rotation, movement/survival, role utility, and identity cooldown tools. |
| Class-owned family coverage | 12/12 owners | Even lean jobs have their own trainer, movement/survival, role utility, and identity cooldown hooks. |
| Advanced base-loop preservation | 100% floor | Advanced jobs retain access to their full base-class skill loop instead of invalidating earlier learning. |
| Max owned/accessed active skills | 8 / 14 | Current kits stay below the skill-bloat guardrail while allowing inherited base skills. |
| Primary trainers | 12/12 owners | Every playable owner has a reliable low-cooldown trainer/basic rotation entry. |
| Prerequisite gaps | 0 missing, 0 over-cap | Skill dependencies currently resolve and do not ask for ranks above a prerequisite's max rank. |
| Obsolete base skills | 0 | Older base skills remain relevant through purpose, prerequisites, passive scaling, or benchmark rotations. |

Enemy ecosystem health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Live combat enemies used | 42 / 48 | Most authored enemies are placed in non-admin combat routes; the unused six are comparison/variant assets, not progression-critical gaps. |
| Behavior variety | 12 behaviors | Hoppers, bruisers, turrets, skirmishers, chargers, armored enemies, blockers, flyers, throwers, healers, elites, and bosses are all represented. |
| Enemy families | 27 | Enemy identity is regionally varied enough that the game is not only reskinning one stat block. |
| Required archetypes | 11 / 11 | Rushers, shielded blockers, flyers, ranged casters, armored targets, elemental enemies, elites, swarms, supports, high-threat melee, and bosses all pass minimum coverage. |
| Non-boss map variety | 17 / 17 | Every field/dungeon route has at least three unique enemies, two behaviors, and two archetypes. |
| Boss arenas | 8 | Boss rooms are tracked separately so single-boss arenas do not weaken field-route variety checks. |
| Bracket variety | 4/3/5 early, 22/12/11 mid, 32/12/11 late, 7/6/9 endgame | Enemy/archetype pressure remains varied across level brackets instead of collapsing into one efficient farming pattern. |
| Counter-tool coverage | 8 / 8 | AoE, mobility, range, control, armor break, burst windows, elemental prep, and defense/sustain all have authored counter use cases. |
| Ecosystem issues | 0 | Current enemy/map data passes roster, archetype, bracket, map-variety, and counter-category guardrails. |

Reward-source health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Field maps checked | 13 | Every benchmarked field route now has reward-source estimates. |
| Optional drops | 19.9/hour median | Visible loot is frequent without becoming a drop shower. |
| Deterministic progress | 211.7/hour median | Primary monster materials give reliable forward progress and anti-dry-streak value. |
| Optional rare-or-better cadence | 27 minutes median | Rare optional progress remains session-visible without making rare drops feel common. |
| Max optional source share | 59.8% | Coins are the largest optional source, but no single optional table exceeds the 65% dominance guardrail. |
| Source health issues | 0 | No field route currently fails deterministic progress, optional cadence, dry-streak, or source-dominance checks. |

Field survivability and potion pressure:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Field maps checked | 13 | Every benchmarked field route now has class-by-map survivability estimates. |
| Median purchased potion replacement cost | 14.1% of farming earnings | Baseline attrition is visible without crossing the 25% affordability threshold. |
| Max purchased potion replacement cost | 23.2% of farming earnings | Worst-case field pressure remains below the potion-cost investigation threshold. |
| Max death risk | 0.139/hour | No class/map pair currently creates a meaningful death-risk outlier in normal field farming. |
| Survivability issues | 0 | Guardian safety and ranged spacing advantages are preserved while risky jobs still feel more expensive to farm. |

Map tuning and retention friction:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Combat maps checked | 25 | Includes 13 field maps, 4 dungeon maps, and 8 boss arenas with route/spawn/friction metrics. |
| Median idle time | 1.9% | Field and dungeon routes are not generally spawn-starved. |
| Median travel share | 29.5% | Travel is close to the 30% investigation threshold, so vertical/boss routes should stay watched. |
| Median class spread | 2.0% | Across the tuning model, median class/map efficiency spread is controlled. |
| Median party overlap | 19.8% | Party routes are not generally collapsing into one overcrowded lane. |
| High-risk farm maps | 3 | Oreback Quarry, Eclipse Frontier, and Endless Rift are recognized as farming-abuse-sensitive maps. |
| Abuse-control coverage | 100% | Every high-risk farming map has authored map controls instead of pure static spawn loops. |
| Warning mix | 31 total | 12 travel-share, 12 idle-time, 6 class-spread, and 1 party-overlap warnings are investigation targets, not failing blockers. |
| Top abandonment risks | 27.9, 27.7, 27.0 | Bramble Depths, Deepcore Core, and Emberjaw Furnace are the highest current friction-watch maps. |

Equipment and upgrade health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Equipment items | 115 | Covers 49 shop items, 17 random world drops, and 49 boss-set items. |
| Boss sets | 7 | Every configured boss equipment source maps to a seven-piece set. |
| Advanced offhand coverage | 9/9 | Every advanced job now has a level-25 Class Supplier offhand instead of only six jobs. |
| Deterministic weapon ceiling | Level 70 | Fighter, Mage, and Archer all have deterministic shop weapons through Astral-tier content. |
| Max deterministic weapon gap | 15 levels | Baseline weapon replacement is paced without requiring rare drops every few levels. |
| Baseline safe upgrade ceiling | +10 | Upgrade bands through +10 have 0% destroy chance and positive expected upgrade movement. |
| Prestige destroy risk | 8-16% | Destroy risk exists only in +11-20 prestige bands. |
| Protected final destroy chance | 0% | Warding Scrolls convert destroy rolls into protected failures. |
| Failure salvage | Partial Upgrade Dust refund | Failed/protected upgrades now return some dust, so bad rolls are not pure material erasure. |
| Upgrade material enemy sources | 5/17/4/5/4 | Upgrade Dust, Catalyst, Warding Scroll, Refinement Core, and Prism Shard all have enemy drop sources. |
| Early attunement multiplier cap | Rare 7, Relic 23 | Early multiplier lines stay modest; Celestial 54 remains a prestige-only ceiling. |
| Equipment issues | 0 | Current data passes deterministic baseline, build-expression, and prestige-risk guardrails. |

Economy health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Field coin faucet | 4,097/hour median | Field coin income is visible without dwarfing repeatable sink costs. |
| Field faucet range | 1,147-8,342/hour | Map-level coin value has spread but remains auditable by field route. |
| Deterministic currency rewards | 42,090 coins, 2,453 Star Tokens | Authored rewards add currency outside field kills and are now counted separately. |
| Coin sinks | 108 | The prototype has broad coin outflow instead of only gear buying. |
| Sink types | 7 | Vendor gear, vendor consumables, vendor bundles, market listings, Plinko balls, cosmetics, and inventory slots all remove coins. |
| Repeatable sinks | 104 | Most sinks can keep absorbing coins after one-time account purchases are done. |
| Median repeatable sink | 7.2 field minutes | Repeatable sink pacing is session-visible without making every reward instantly vanish. |
| Market listings | 15 | Current market has 11 repeatable listings and 4 one-time account-service listings. |
| Item purpose coverage | 100% | No item definition is currently classified as dead loot. |
| Cash-shop power items | 0 | Premium shop data does not inject permanent combat power or materials. |
| Runtime spend telemetry | Present | Session metrics now track `currencySpent`, `currencySinks`, `currencySpentPerHour`, and `netCurrencyPerHour`. |
| Economy issues | 0 | Current data passes faucet visibility, sink breadth, item-purpose, market, and cash-shop economy guardrails. |

Retention and chore health:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Daily login rewards | 7 | Login rewards are a short claim cadence, not a task checklist. |
| Mandatory daily checklist | 0 tasks, about 1 minute | Current daily pressure is below the 20-30 minute high-value target and avoids chore creep. |
| Active season goals | 3 goals, about 70 minutes | The Beta Foundations season fits the 60-90 minute weekly target. |
| Long-term lanes | 15 | Accomplishments, monster guide, cards, mastery, roster, daily login, seasons, cosmetics, boss sets, dungeons, endless scaling, crafting, subclasses, party objectives, and target farms are represented. |
| Player-type coverage | 6/6 | Casual, grinder, completionist, bosser, economy, and competitive lanes all have at least one supported progression path. |
| Accomplishments | 41 across 9 categories and 5 tiers | The accomplishment spine has enough breadth for onboarding, collection, combat, crafting, bossing, dungeons, classes, and mastery. |
| Monster guide entries | 42 live collectibles | Collection depth is meaningful without relying only on rare drops. |
| Cards and mastery | 21 cards, 12 mastery tracks | Both collection and class-loyalty lanes have measurable long-tail structure. |
| Roster growth | 11 traits, 4 synergies | Account-wide growth exists but remains bounded and authored. |
| Cash shop power items | 0 | Current cash-shop data is cosmetic or earnable buff bundles rather than direct permanent power. |
| Earnable buff bundles | 4, max weekly limit 3 | Convenience buffs are capped and tagged as earnable in game. |
| Catch-up sources | 10 | Coupons, slots, and reset-style sources provide returning-player relief without season permanent power. |
| Retention issues | 0 | Current retention data passes chore, coverage, cash-shop, and season-pressure guardrails. |

Boss and party validation:

| Check | Current result | Balance read |
| --- | ---: | --- |
| Custom boss encounters | 8 | Every encounter has a boss room, phases, add pool, break profile, and deterministic progress path. |
| Minimum mechanic categories per boss | 6 | Bosses are not only DPS checks in data; every fight combines positioning, movement pressure, adds, control, and attrition. |
| Boss chase rewards | 7 | Seven encounters have configured boss-set chase rewards; Rimewarden currently leans deterministic/material until a set is added. |
| Boss TTK target | 6-12 minutes | Static solo benchmark now checks boss clear pacing against the requested tier target. |
| Median solo boss clear | 8.91 minutes | Boss HP budgets now give enough fight length for phases, adds, breaks, and positioning mechanics to matter. |
| Fastest specialist clear | 6.15 minutes | Top boss specialists no longer erase entry bosses before mechanics execute. |
| Slowest floor clear | 19.23 minutes | Weak solo matchups are slower but remain inside the tested floor tolerance. |
| Encounter HP scale | 49-64x | HP scaling is applied only to custom boss encounters, not normal field enemies or add pools. |
| Boss TTK issues | 0 | Every current boss has configured HP scale and median solo clear pacing inside the 6-12 minute envelope. |
| Boss chase dry-streak sources | 7 | Every configured boss set source is covered by the dry-streak audit. |
| Epic boss-set p95 | 21 clears | Epic set drops remain rare, but soft pity keeps the 95th percentile from drifting into extreme frustration. |
| Relic boss-set p95 | 29 clears | Relic chase items stay longer-tail than Epic without becoming unprotected mandatory power. |
| Max boss-set p99 | 38 clears | The 99th percentile remains bounded by soft pity instead of relying on pure RNG. |
| Max p95 chase time | 5.29 solo hours | Worst-case p95 chase time is long-term but still below the 8-hour guardrail. |
| Boss dry-streak issues | 0 | No configured boss set violates p95, p99, or soft-pity coverage thresholds. |
| Support contribution classes | 12 | Every base and advanced class is included in support-value scoring. |
| Support contributors | 9 | Support value is distributed across several jobs instead of only one formal support. |
| Support categories covered | 6/6 | Mitigation, party damage, control, sustain, mobility, and objective utility all have measurable sources. |
| Max support share | 13.7% | No single class owns enough support value to become mandatory by utility alone. |
| Support contribution issues | 0 | Current party support model has breadth without collapsing into one required class. |
| Four-player HP multiplier | 2.78x | Sublinear curve gives about 143.8% theoretical throughput vs one solo baseline, inside the 125-160 target band. |

## Class Review

Fighter is the current Warrior analogue. Its base identity is close to target: high HP, high defense, short range, control, guard, dash, ground slam, and power break. Guardian cleanly extends safety/control. Berserker and Duelist now benchmark as strong boss paths without dominating pack lanes.

Archer has strong range, speed, mark pressure, pierce, roll movement, and advanced jobs with Sniper bossing, Trapper control, and Beast Archer sustain. Sniper is now near other boss specialists instead of being far above them. Trapper is still strong in grounded packs, but the new guardrails prevent it from dwarfing Fire Mage and Storm Mage.

Mage has the clearest AoE/control identity. Fire Mage is the burn/explosion path, Rune Mage is setup/control/support, and Storm Mage is chain-clearing. The previous benchmark made Storm Mage look like a trap in single-target content; inherited Magic Bolt filler is now modeled for boss/armored/sustain scenarios.

## Skill System Review

Confirmed strengths:
- Every playable owner has all four requested skill families: core rotation, movement/survival, role utility, and identity cooldown.
- Every playable owner has a primary trainer/basic rotation entry; current audit finds 12/12 trainer coverage.
- Skill purposes are authored for 85/85 skills and prerequisite health has 0 missing or over-cap requirements.
- Advanced jobs preserve 100% access to their base-class active loop, so job advancement does not invalidate earlier learning.
- Older base skills remain relevant through purpose, prerequisites, passive scaling, inherited access, or benchmark rotations; current audit finds 0 obsolete base skills.
- Skill bloat is currently controlled: the largest class-owned active kit has 8 skills, and the largest inherited accessible kit has 14 active skills.

Risks:
- Sparse advanced jobs can feel shallow if the UI or default primary-skill behavior hides inherited base skills.
- Low-cooldown AoE trainers can become the whole rotation if resource pressure and setup friction are not felt in real combat.
- Party skills are solo self-buffs today, so their future party value needs explicit telemetry before multiplayer balancing.
- Duelist, Storm Mage, and Beast Archer intentionally have only 3 class-owned active skills; they pass family coverage because their party skill acts as identity/utility and the base loop remains accessible.

Recommendation:
- Keep inherited base skills visible in advanced-job recommendations, keybind prompts, and default rotations.
- For every future class or subclass, require 0 skill-system issues, all four skill families, at least one trainer, valid prerequisites, base-loop preservation >= 60%, class-owned active skills <= 8, and accessible active skills <= 14 unless the UI is explicitly redesigned for larger kits.
- Add telemetry for skill damage contribution and cast frequency before further coefficient tuning.

## Progression Review

Confirmed systems include leveling, skill points, advanced class trials, level 25 advanced jobs, level 60 specializations, quests, dungeons, routes, accomplishments, daily rewards, monster guide, cards, and roster/class mastery.

Risks:
- Many progression lanes exist at once. Without session-level guidance, players may perceive the game as a checklist.
- XP/hour, time-to-level, monster TTK, small/medium reward cadence, and level-to-level spike risk are now benchmarked.
- Field reward sources are now split by source table, family, rarity, deterministic backup, optional rare cadence, and source-dominance share.
- Field survivability now estimates damage taken, death risk, and purchased potion replacement cost against farming earnings.
- Daily rewards exist; keep them modest and optional so they do not become mandatory power chores.

Recommendation:
- Extend the progression simulator with deterministic token progress beyond primary monster materials, then validate the static attrition model against live class/map telemetry.
- Keep the level-spike audit tied to data-defined breakpoints so future map, quest, equipment, dungeon, and specialization changes automatically justify or fail pacing jumps.

## Damage And Stat Formula Review

Confirmed:
- `getStats()` uses mostly additive stat sources across gear, cards, roster, party, specialization, stat upgrades, permanent bonuses, and passives.
- Damage uses line counts, line damage scale, defense reduction, crit, role bonuses, boss/elite/execute bonuses, monster mastery, and variance.
- The harness now names 9 formula buckets and keeps positive multiplier buckets at 7: additive damage, class/status windows, crit, target type, execute, monster mastery, and variance.
- Base-class stat identity now has automated floors: Warrior effective HP 107.6+, Mage MP 108.9+, Archer speed/range 106.8/113.2+, with raw power spread capped at 8%.
- Runtime mitigation uses `100 / (100 + defense * 2.25)` before capped damage-reduction bonuses; 100 raw damage currently becomes 47 at 50 defense and 18 at 200 defense.
- Attunement multiplier lines are capped in data at Rare 7, Relic 23, and Celestial 54, with runtime boss/elite/execute damage bonuses capped at 100% and damage reduction capped at 25%.

Risks:
- A few independent multiplier paths remain: active buff power, Fire Mage ignition, role/status windows, crit, boss/elite/execute attunement bonuses, monster mastery, and variance.
- Gear plus cards plus mastery plus attunement can become hard to reason about without bucket caps.
- The 7 positive multiplier buckets are acceptable now, but adding another independent final-damage bucket should require removing or merging an older bucket.

Recommendation:
- Keep the formula report as a required balance gate: no formula issues, positive multiplier buckets <= 7, base raw-power spread <= 8%, and Rare/Relic/Celestial attunement caps <= 7/23/54.
- Keep percent stats additive inside their bucket wherever possible; reserve new independent multipliers for explicit class identity windows, boss mechanics, or prestige-only systems.
- Surface the formula bucket names in designer-facing docs or debug UI so balance changes can be reviewed by bucket instead of by tooltip DPS alone.

## Map And Monster Review

Confirmed map variety:
- Shared lane maps: Greenroot Meadow, Bandit Ridge Camp, Oreback Quarry, Frostfen Outskirts, Stormbreak Cliffs, Astral Archive, Endless Rift.
- Switchback/terrace maps: Rustcoil Ruins, Cinder Hollow, Ashglass Pass, Eclipse Frontier.
- Vertical maps: Thornpath Thicket, Glacier Spine.
- Compact dungeons and boss rooms exist across route tiers.

Confirmed enemy variety includes chargers/rushers, flyers, casters/ranged enemies, armored/heavy enemies, healers, mimics, bosses, and route-specific families.

Confirmed enemy-ecosystem coverage:
- The harness now evaluates 42 used combat enemies out of 48 authored enemies, 12 behavior types, 27 enemy families, and 11 required enemy archetypes.
- Required archetypes cover rushers, shielded blockers, flyers, ranged/caster pressure, armored targets, elemental preparation, elites, swarms, support casters, high-threat melee, and bosses.
- Every non-boss field/dungeon route currently has mixed enemy, behavior, and archetype pressure, with 17/17 routes passing the variety guardrail.
- Bracket enemy/behavior/archetype coverage is 4/3/5 early, 22/12/11 midgame, 32/12/11 late, and 7/6/9 endgame.
- Counter-tool coverage is complete across AoE, mobility, range, control, armor break, burst windows, elemental preparation, and defense/sustain.

Confirmed map-tuning coverage:
- The harness now evaluates all 25 combat maps across route viability, spawn sections, idle time, travel share, route cycle time, damage pressure, party overlap, party efficiency, class spread, farming-abuse risk, abandonment risk, and repeat-visitation value.
- High-risk farm maps currently have complete abuse-control coverage, including Eclipse Frontier's elite-pocket routing.

Risks:
- Storm Mage still leads most flyer-heavy vertical routes. The hourly-value compression keeps it inside current guardrails, but real telemetry should confirm chain targeting, MP pressure, and travel downtime.
- Trapper may overperform where grounded enemies path predictably into traps.
- Archer route efficiency should be measured on actual wide maps, not only spread-pack DPS.
- Travel share is the main map-friction watch item: 12 combat maps currently exceed the travel-share warning threshold, and the median travel share is close to the 30% investigation line.
- The early bracket has only one current field benchmark, so its enemy variety passes the minimum but still needs a second low-level combat route before launch.

Recommendation:
- Keep the new field-map benchmark in CI and add explicit authored map archetype tags if the heuristic becomes hard to reason about.
- Keep the enemy ecosystem report in CI and treat archetype loss, missing counter categories, or non-boss map variety warnings as balance blockers.
- Use the map-tuning warning mix to prioritize route polish before adding more rewards to underused maps; high travel or idle time should be fixed structurally before loot is inflated.
- Track map usage rate and investigate any map with 20%+ activity in a level bracket.

## Loot And Economy Review

Confirmed:
- Items generally fit equip, sell, consume, material, card, cosmetic, or upgrade purposes.
- Drops include currency, materials, consumables, equipment, cards, Plinko items, attunement items, and rare global valuables.
- Combat metrics now track DPS, currency/hour, XP/hour, level ETA, kills/hour, generated drops/hour, looted drops/hour, potion use/hour, estimated potion replacement cost/hour, deaths/hour, damage taken/hour, skill cast counts, skill damage contribution, and loot kind counts.
- Runtime economy metrics now track coins gained, coins spent, spend by sink category, spent/hour, and net coins/hour.
- Guaranteed primary monster materials remain deterministic, while optional currency, consumable, equipment, card, bonus material, and Plinko tables now respect the configured normal/elite/boss base drop chance.
- Current field estimates put small visible rewards around 2.9-6.2 minutes and medium forward progress around 13.7-28.8 minutes, inside the tested pacing envelope.
- Reward-source estimates now put optional field drops at 19.9/hour median, equipment at 1.13/hour median, cards at 0.78/hour median, bonus materials at 1.34/hour median, Plinko balls at 1.38/hour median, rare valuables at 0.79/hour median, and optional rare-or-better rewards at 27 minutes median.
- Field survivability estimates keep purchased potion replacement cost at 14.1% median and 23.2% maximum of farming earnings.
- Economy-health estimates put field coin income at 4,097/hour median, 108 coin sinks across 7 sink types, 104 repeatable sinks, 7.2 field minutes per median repeatable sink, 15 market listings, 100% item-purpose coverage, and 0 economy issues.

Risks:
- No player-trading or auction house exists yet, so auction taxes are not applicable today but must be added before player-to-player trading.
- Rare power drops need deterministic backup as more mandatory upgrades are added.
- Dead loot risk rises as material and card catalogs expand.
- Drop telemetry counts generated and looted drops, while the static reward-source audit now splits by rarity, source table, mandatory power, and prestige/utility. Live dry-streak percentiles still need runtime telemetry.
- Boss set chase drops now have a static dry-streak audit using soft-pity probabilities: Epic p95 is 21 clears, Relic p95 is 29 clears, and max p99 is 38 clears.

Recommendation:
- Add live rare dry-streak counters, drop source/rarity splits, and power-vs-cosmetic classifications to development metrics.
- Keep mandatory power progression deterministic or pity-backed; reserve extreme rarity for cosmetics/prestige.
- Keep the new faucet/sink report in CI and investigate if median repeatable sink pacing drifts below 3 minutes or above 30 minutes of median field income.
- Before adding player trading, require auction tax, listing fee, market volume, essential-item volatility, and hoarding telemetry.

## Equipment Review

Confirmed systems include equipment slots, deterministic shop equipment, random equipment, boss equipment, sets, upgrades, upgrade aides, attunement lines, cards, cash-shop cosmetics, and equipment-driven visuals.

Confirmed equipment health:
- Fighter, Mage, and Archer each have deterministic shop weapons from level 1 through level 70, with no gap above 15 levels.
- All nine advanced jobs now have level-25 Class Supplier offhands.
- Baseline upgrades through +10 are non-destructive and have positive expected upgrade movement.
- Failure and protected-destroy outcomes salvage partial Upgrade Dust, while true destroy risk remains reserved for +11-20 prestige bands.
- Warding Scrolls eliminate final destroy chance when players choose to protect a risky prestige attempt.
- Upgrade Dust, Upgrade Catalyst, Warding Scroll, Refinement Core, and Prism Shards all have both enemy drop sources and deterministic reward paths.

Risks:
- Upgrade destruction still exists in the prestige data model. Keep it out of baseline power and keep Warding Scroll access practical before players enter +11 or higher attempts.
- The +16-20 band remains a long-tail prestige sink even when protected and boosted; do not make that band mandatory for normal boss access.
- Attunement lines include many useful stats; one rare multiplier can dominate if caps are not enforced.

Recommendation:
- Separate baseline gear progression, build-expression affixes, and prestige enhancement in UI and data.
- Keep failed/protected upgrade salvage visible in the UI and telemetry so players understand that bad rolls still return some material progress.

## Boss And Party Review

Confirmed:
- Boss encounters have mechanics, phases, adds, break profiles, room ambience, rewards, and route maps.
- Simulated party members, party commands, party assists, class-specific AI party loadouts, support loadouts, and party skills exist.
- Dungeon bonus objectives check boss breaks, add control, party survival, and timed clears.
- Boss/party benchmark now validates eight encounters, seven mechanic category families, deterministic progress, and the recommended `BaseHP * (1 + 0.7 * (players - 1)^0.85)` scaling curve.
- Boss encounters now carry encounter-only HP budgets, and the balance harness validates 6-12 minute median solo clear pacing.
- Support contribution is now scored separately from DPS across shields/mitigation, party damage, control/debuffs, sustain/recovery, mobility/uptime, and objective utility.

Risks:
- The boss benchmark is still a static TTK model, not a full fight simulator. It validates clear-time budget, encounter design breadth, and authored support coverage, but not live potion pressure, death reasons, mechanic failure rates, or real multiplayer support contribution.
- Party scaling is not yet a live multiplayer economy problem, but party utility already affects solo balance through self-buffs and simulated allies.
- Rimewarden has no boss set chase source yet, so its long-tail boss reward identity is weaker than the other high-tier echoes.

Recommendation:
- Add live boss simulators for burst windows, add phases, interrupt/control windows, incoming damage, potion use, failure reasons, and measured support contribution.
- Keep future multiplayer HP scaling sublinear and test 1-4 player multipliers before reward tuning.

## Retention Review

Confirmed long-term lanes include accomplishments, monster guide, cards, class mastery, roster synergy, daily rewards, seasonal data, cosmetics, boss sets, dungeons, endless scaling, crafting, subclasses, party objectives, and target farms.

The new retention report currently finds 15 long-term lanes, 6/6 player-type coverage, 41 accomplishments across 9 categories and 5 tiers, 42 live monster-guide entries, 21 cards, 12 class mastery tracks, 11 roster traits, 4 roster synergies, and 0 retention issues. The active season is estimated at about 70 minutes of goals, which fits the requested 60-90 minute weekly range. Daily pressure is intentionally low: 7 login rewards, 0 checklist tasks, and about 1 minute of mandatory daily time.

Risks:
- Daily rewards plus accomplishments plus cards plus mastery can become too many parallel prompts.
- Account-wide bonuses should remain capped and selectable, not uncapped mandatory alt leveling.
- Login milestone permanent stats exist. Keep them tiny and capped, and do not add uncapped attendance power.
- Future seasons should not require every lane at once. A season that asks for bosses, dungeons, crafting, cards, and daily tasks simultaneously will feel like a chore board.

Recommendation:
- Keep daily high-value time to 20-30 minutes and weekly goals to 60-90 minutes.
- Use optional uncapped grind for dedicated players, with catch-up for returning players.
- Preserve the current cash-shop boundary: cosmetics are fine, buff bundles must remain earnable in game, and permanent power should stay out.

## Telemetry Plan

Existing:
- DPS, currency/hour, currency spent/hour, net currency/hour, currency sink totals, XP/hour, level ETA, kills/hour, boss kills/hour, elite kills/hour, generated drops/hour, looted drops/hour, consumables/hour, potions/hour, potion replacement cost/hour, deaths/hour, damage taken/hour, skill cast totals, skill damage totals, loot-kind totals, scenario spread guardrails, skill-system health estimates, enemy ecosystem health estimates, field-map efficiency estimates, map-tuning and retention-friction estimates, monster TTK estimates, reward-cadence estimates, reward-source health estimates, economy-health estimates, field survivability and potion-pressure estimates, damage/stat formula health estimates, equipment/upgrade health estimates, retention/chore health estimates, boss TTK estimates, boss chase dry-streak estimates, party support-contribution estimates, progression bracket estimates, and level-curve spike audits.

Add next:
- Metrics segmentation by class, subclass, level range, map, gear tier, and party size.
- Kill/hour and deaths/hour by enemy behavior, enemy archetype, and counter category.
- Live drops/hour by rarity, source table, power relevance, and deterministic backup path.
- Boss clear time and failure reason.
- Map usage by level bracket.
- Upgrade success/failure/destroy rates.
- Rare item dry streak percentiles.
- Class population, subclass pick rates, and reroll rates.
- Player-trading metrics only after trading exists: listing fees, auction tax, market prices, market volume, essential-item volatility, and hoarding behavior.

Investigation thresholds:
- EXP/hour class outlier: 12-15% above median across several popular maps.
- Boss clear outlier: 10%+ faster/slower after controlling for gear.
- Deaths/hour: 20%+ higher without compensating efficiency.
- Potion cost: above 25% of farming earnings.
- Map usage: 20%+ of activity in a bracket.
- Essential item volatility: 15%+ weekly without content cause.

## Prioritized Action Plan

Immediate:
- Keep the new balance harness guardrails passing.
- Use the new class/map bracket benchmarks to tune Storm Mage, Trapper, Fire Mage, and Rune Mage against real map telemetry.
- Use the surfaced map-tuning warning mix to triage high-travel and high-idle maps before compensating them with extra rewards.
- Add segmentation to the new combat/economy telemetry so kills, drops, potion cost, deaths, skill contribution, coin gains, coin spends, and sink categories can be compared by class, level, map, and gear tier.

Medium term:
- Extend field survivability checks into live segmented telemetry, dungeon routes, and boss simulations.
- Add deterministic token progress checks beyond primary monster materials.
- Extend economy checks with live sink-to-faucet ratios by map, level bracket, and play style.
- Extend boss checks from static TTK/data/support validation into live simulations for failures, potion pressure, add phases, and actual support contribution.
- Extend deterministic/pity review to future mandatory power drops beyond current boss set sources.

Long term:
- Add player-trading dashboards, auction tax, listing fees, market volume, price volatility, and hoarding alerts before enabling a true player market.
- Convert authored support-contribution scoring into live support credit for shields, mitigation, buffs, debuffs, CC, and party damage contribution.
- Add seasonal/catch-up tuning so returning players can re-enter without destroying baseline economy.

## Concrete Recommendations

Class targets:
- Fighter/Warrior analogue: keep best safety and close-range boss learning; do not let wide-map clear exceed Archer/Mage specialists.
- Archer: keep open-map route value and boss precision; prevent off-screen safe farming from bypassing rushers/flyers/casters.
- Mage: keep dense-map control and burst; preserve cast/resource/interruption risk.

Balance guardrails now tested:
- Single boss top advanced class <= 1.35x median, floor >= 0.5x median.
- Clustered pack top <= 1.7x median, floor >= 0.3x median.
- Spread pack top <= 1.85x median, floor >= 0.35x median.
- Armored target top <= 1.4x median, floor >= 0.5x median.
- Flying pack top <= 3.05x median, floor >= 0.22x median.
- Sustain boss top <= 1.85x median, floor >= 0.5x median.
- Cinder Hollow top hourly efficiency <= 165, floor >= 70.
- Oreback Quarry top hourly efficiency <= 125, floor >= 75.
- Stormbreak Cliffs top hourly efficiency <= 155, floor >= 70.
- No one class may top more than seven field maps in the current benchmark.
- Skill-system health: require 0 skill issues, 100% authored purpose coverage, every playable owner to have core rotation, movement/survival, role utility, and identity cooldown coverage, every owner to have at least one trainer, advanced base-loop preservation >= 60%, max class-owned active skills <= 8, max accessible active skills <= 14, 0 missing or over-cap prerequisites, and 0 obsolete base skills. Current audit finds 85 skills, 70 active, 15 passive, 12/12 family coverage, 100% advanced base-loop preservation, 8/14 max active owned/accessed skills, 0 prerequisite gaps, and 0 obsolete base skills.
- Enemy ecosystem health: require 0 ecosystem issues, at least 40 used combat enemies, at least 10 behavior types, at least 20 enemy families, all 11 required archetypes covered, every non-boss combat route to include at least three unique enemies/two behaviors/two archetypes, every progression bracket to pass enemy/behavior/archetype minimums, and all 8 counter-tool categories covered. Current audit finds 42 used enemies, 12 behaviors, 27 families, 11/11 archetypes, 17/17 non-boss routes passing variety, 8/8 counter categories, and 0 ecosystem issues.
- Map tuning: every combat map must expose route, spawn, economy, survival, class-spread, party-overlap, abandonment, and repeat-visitation metrics; median idle time, median travel share, and median class spread must stay below their warning thresholds; high-risk farm maps require abuse-control coverage; free-farm, missing-control, and route-contract warnings must stay at 0. Current audit finds 25 combat maps, 1.9% median idle, 29.5% median travel, 2.0% median class spread, 3 high-risk farm maps, and 100% abuse-control coverage.
- Damage/stat formula health: require 0 formula issues, at least 9 named buckets, positive multiplier buckets <= 7, additive stat sources >= 10, base raw-power spread <= 8%, Warrior effective HP index >= 106, Mage MP index >= 108, Archer speed/range indices >= 105/110, Rare/Relic/Celestial attunement multiplier caps <= 7/23/54, runtime damage bonus cap <= 100%, runtime damage reduction cap <= 25%, and monotonic defense mitigation that does not reduce 100 raw damage below 10 at 200 defense. Current audit finds 9 buckets, 7 positive multipliers, 11 additive sources, 7.4% max base power spread, 107.6 Warrior eHP floor, 108.9 Mage MP floor, 106.8/113.2 Archer speed/range floors, 100 raw damage -> 47 at 50 defense and 18 at 200 defense, and 0 formula issues.
- Equipment/upgrade health: deterministic class weapons must reach level 70 with max 15-level gaps, all 9 advanced jobs need class-supplier offhands, baseline upgrades through +10 must have 0% destroy chance and positive expected movement, prestige destroy risk must be protectable to 0% final destroy chance, failed/protected upgrades must salvage partial Upgrade Dust, all upgrade aide materials need enemy and deterministic sources, and Rare/Relic attunement multiplier caps must stay <= 7/23. Current audit finds 115 equipment items, 9/9 offhand coverage, +10 baseline safety, 8-16% prestige destroy risk, Warding Scroll protection, and 0 equipment issues.
- Economy health: field coin faucets must be visible, coin sinks must span at least six categories, median repeatable sink pacing should stay between 3-30 minutes of median field income, item-purpose coverage must stay effectively complete, market listings must include repeatable and one-time sinks, cash-shop power items must stay at 0, and runtime telemetry must include gained/spent/net currency plus sink categories. Current audit finds 4,097 coins/hour median faucet, 108 sinks across 7 types, 7.2-minute median repeatable sink pacing, 100% item-purpose coverage, 15 market listings, 0 cash-shop power items, and 0 economy issues.
- Retention/chore health: keep daily login rewards at 7, mandatory daily checklist tasks at 0 unless deliberately added, estimated mandatory daily time <= 5 minutes for login-only systems, active season goal time in the 60-90 minute weekly range, at least 12 long-term lanes, 6/6 player-type coverage, 0 cash-shop power items, earnable buff bundles capped at 3/week, and 0 season permanent-power rewards. Current audit finds 15 lanes, 6/6 coverage, 70-minute active season goals, 10 catch-up sources, 0 cash-shop power items, and 0 retention issues.
- Progression median targets: early 3-8 minutes, midgame 10-25 minutes, late 25-60 minutes, endgame 60-240 minutes.
- Boss-party data health: every boss encounter must have at least four mechanic categories, three phases, four unique phase actions, three adds, a break profile, deterministic reward progress, and valid chase rewards when a chase rarity is configured.
- Boss mechanic roster coverage must include positioning, movement/dodge, burst windows, add control, utility/control, attrition, and class-role moments.
- Party HP scaling must follow `BaseHP * (1 + 0.7 * (players - 1)^0.85)`: 1p 1.00x, 2p 1.70x, 3p about 2.26x, 4p about 2.78x.
- Boss TTK: every custom boss encounter must have encounter-specific HP scale, median solo clear time between 6-12 minutes, fastest specialist clear >= 3.9 minutes, slowest weak-matchup clear <= 20 minutes, and 0 clear-time issues. Current audit finds 8.91 minute median solo clear, 6.15 fastest, 19.23 slowest, and 49-64x encounter HP scale.
- Boss chase dry streaks: every boss set source must have soft-pity coverage, p95 <= 40 clears, p99 <= 60 clears, p95 solo time <= 8 hours, and 0 dry-streak issues. Current audit finds Epic p95 21 clears, Relic p95 29 clears, max p99 38 clears, max p95 time 5.29 hours, and 0 dry-streak issues.
- Party support contribution: score mitigation/shields, party damage, control/debuffs, sustain/recovery, mobility/uptime, and objective utility separately; require at least 7 support contributors, all 6 categories covered, max single-class support share <= 32%, and 0 support issues. Current audit finds 9 contributors, 6/6 categories, 13.7% max share, and 0 issues.
- Combat/economy telemetry persistence: session metrics must preserve kills, boss kills, elite kills, generated drops, looted drops, consumable use, potion use, potion replacement cost, deaths, damage taken, skill casts, skill damage, loot-kind totals, currency gained, currency spent, and currency sink totals.
- Combat/economy telemetry rolling rates: live HUD rates must decay over the rolling window for kills, drops, potions, deaths, damage taken, currency gained, currency spent, and net currency, not only DPS/XP.
- Field TTK guardrails: early 2-4 seconds, midgame 4-7 seconds, late 6-10 seconds, endgame 8-15 seconds.
- Field reward cadence guardrails: small visible rewards roughly every 3-7 minutes after the starter floor, medium forward progress roughly every 10-45 minutes in the current prototype harness.
- Optional monster drop tables must multiply through the configured base drop chance; only guaranteed primary etc materials should bypass that base chance.
- Level-to-level spike cap: no time-to-level step may rise by more than 25% unless the current level has an authored unlock or content breakpoint. Current audit checks levels 2-112 and finds 0 unjustified spikes.
- Reward-source health: every field map must have deterministic primary progress, optional field loot between 5-40 drops/hour, optional rare-or-better cadence under 180 minutes, max optional table share <= 65%, and nonzero equipment, card, bonus-material, Plinko, and rare-valuable tables in the aggregate.
- Field survivability: purchased potion replacement cost must stay <= 25% of farming earnings; death-risk outliers require both >0.25 deaths/hour and 20%+ over median without compensating efficiency. Current audit finds 14.1% median potion cost, 23.2% max potion cost, 0.139 max death risk/hour, and 0 survivability issues.

Next numerical tests to add:
- Live segmented validation for field potion cost, death rate, and damage taken by class, map, level, and gear tier.
- Live boss failure reasons, potion pressure, measured support contribution, and add-phase outcomes.
- Live dry-streak validation for rare mandatory power drops and chase items after runtime telemetry accumulates real player distributions.
