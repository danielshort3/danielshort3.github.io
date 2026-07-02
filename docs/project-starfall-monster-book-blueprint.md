# Project Starfall Monster Book Blueprint

This blueprint follows `project-starfall-monster-book-goal.txt` and is grounded in the current browser prototype:

- Game data: `js/games/project-starfall/project-starfall-data.js`
- Runtime/state/loot logic: `js/games/project-starfall/project-starfall-engine.js`
- Canvas UI: `js/games/project-starfall/project-starfall-ui.js`
- Product direction: `project_starfall_gdd_v0_5.md`

Project Starfall already has a strong start: enemies, maps, drops, boss phases, target farming, and a Monster Guide panel exist. The optimization should evolve that system into a combat-first, collection-second guide that gives useful fighting and farming information early, then rewards long-term mastery with collection visibility, exact odds, lore, cosmetics, and tracking tools.

## A. Current-State Diagnosis

### What The Game Already Does Well

- The current player-facing name is already **Monster Guide**, exposed as a panel and keybind. That fits the game's practical RPG-client tone better than a purely lore-heavy name.
- The game already tracks monster kills by enemy id through `state.monsterGuide.killsByEnemyId`.
- The Monster Guide already has a searchable list, selected monster detail view, portrait rendering, kill count, progress bar, tier display, and target-farm button.
- The current guide already unlocks information by kill tier:
  - Tier 1: family, role, level range.
  - Tier 2: HP, mechanic, counter.
  - Tier 3: damage, defense, aggro, behavior.
  - Tier 4: maps and drop preview.
  - Tier 5: calculated drop table summaries.
  - Tier 6: monster-specific damage mastery bonus.
- The game has 48 current enemy definitions, including 8 bosses. Four Bandit Cutter variants are asset-test/admin variants and should be excluded from normal collection completion.
- Enemy definitions already include useful base fields: `id`, `name`, `levelRange`, `role`, `family`, `hpMult`, `damageMult`, `defenseMult`, `expMult`, `speed`, `behavior`, `mechanic`, `counter`, `drops`, `dropPool`, `asset`, and animation metadata.
- Maps already define enemy placement context through `MAPS[].enemies`, `levelRange`, `purpose`, `safeZone`, `isDungeon`, `movementProfile`, `areaMechanic`, `spawnPoints`, and `questNpcs`.
- Bosses have richer data in `BOSS_ENCOUNTERS`: encounter map, set id, adds, phases, mechanics, intros, clear text, and summaries.
- Loot is already implementation-friendly. Monsters use structured drop pools, which the engine converts into drop tables: guaranteed primary material, coins, potions, equipment, cards, bonus materials, plinko balls, and global rare valuables.
- Boss-set equipment already has explicit source records with drop chances, rarity, set ids, and pity data.
- Target farming already exists as a short-term anti-frustration mechanic. `targetFarm` tracks the selected enemy, map, streak, bonus timer, and loot bonus.
- Map modifiers already include `lucent_cache`, which increases target farming and Monster Guide research progress.
- The game has item icons, enemy portraits, compact enemy sheets, card icons, boss equipment, cosmetics, achievements/accomplishments, daily rewards, and profile-style reward hooks that can support collection rewards.

### What Is Weak Or Missing

- The guide state only stores kills and selected enemy. It does not track `sighted`, `firstSeen`, `firstKilled`, `lastKilled`, per-drop obtained state, duplicate drop counts, field unlocks, milestone claim state, or mastery reward state.
- The current list shows monster names and portraits before any sighting/kill. It dims zero-kill portraits but does not support silhouette, unknown nameplate, or partial identification.
- Current regular milestones are `[1, 5, 15, 30, 60, 120]`, and boss milestones are `[1, 2, 3, 5, 10, 20]`. These work mechanically, but they do not match the desired 1/5/10/25/50/100/250 structure for long-term research.
- The current guide is mostly a vertical list of info rows. It lacks a full combat/habitat/drop/completion layout, region pages, family pages, and a global completion summary.
- Search exists, but filters and sorting do not. There are no filters for region, biome, family, level range, threat tier, weakness, drop item, rarity, completion, quest relevance, event-only status, or boss/elite/normal type.
- The game has text counters like `counter: "Fire, armor break."`, but it does not have structured weaknesses, resistances, elements, status vulnerabilities, or attack pattern tags.
- Current drop display shows table summaries and the first few table rows, not an item-level collection table with hidden slots, ghosted known-but-unobtained items, obtained counts, quantity ranges, and exact odds.
- Boss-set drops are generated separately before the normal drop tables and are not clearly represented in the Monster Guide drop info. This is a major farming-UX gap.
- Drop acquisition is not recorded. The game can show theoretical drops, but it cannot show "you have found this", "you have found 12", or "this remains missing".
- The current guide's mastery reward is combat power: +2% or +5% damage to that monster. This is tolerable in prototype form because it is monster-specific, but long-term MMO design should shift most guide rewards toward prestige, cosmetics, tracking, and convenience.
- The prototype save is local and character-centric. The GDD wants account-style progression, but Monster Guide progress is currently stored inside the active save state, not in a clear account-wide profile layer.
- `bristleBoar` and `dustImp` are defined as enemies but are not present in current map enemy lists. They should be either placed, hidden as future content, or excluded from completion.
- Asset-test enemies such as `banditCutterDirect`, `banditCutterReference`, `banditCutterHybrid`, and `banditCutterPuppet` should not count toward normal Monster Guide completion.

## B. Recommended System Overview

### Recommended Name

Keep the player-facing name **Monster Guide**.

Reasons:

- It already exists in the UI and GDD, so keeping it avoids churn.
- "Guide" fits Project Starfall's practical, RPG-client tone: combat, farming, map routes, target farming, and progression.
- "Bestiary" leans more lore-fantasy than the current system needs.
- "Codex" is a strong broader lore/system word, but Project Starfall already uses "Starbound Codex" as an item and has many systems that could eventually live in a larger codex.
- "Monster Book" is readable but too directly reminiscent of MapleStory.
- "Field Guide" is good, but slightly narrower than bosses, dungeons, rift enemies, and astral content.
- "Enemy Journal" is clear but less iconic.
- "Monster Encyclopedia" is too formal and bulky.

Use **Monster Guide** for the panel title and **Monster Research** for the progression mechanic inside it.

### Full Structure

The Monster Guide should have four primary views:

1. **Monsters**
   - Main list and detail view.
   - Search, filters, sorting, and target-farm entry points.

2. **Regions**
   - Region/route completion cards such as Greenroot, Rustcoil, Cinder, Frostfen, Stormbreak, Astral, Eclipse, Rift.
   - Each region shows monsters sighted, researched, mastered, drops obtained, bosses cleared, and active target-farm suggestions.

3. **Families**
   - Family/type pages such as Ooze, Plant, Beast, Construct, Humanoid, Spirit, Frost, Storm, Astral, Void, Boss.
   - Useful for players farming a theme of cards or materials.

4. **Completion**
   - Global progress, region progress, family progress, boss progress, drop collection, recent discoveries, pinned targets, and unclaimed guide rewards.

### Desktop Layout

Use a dense three-zone panel:

- Left rail: search, filters, sort, monster list.
- Center: selected monster header, combat overview, habitat, drop table.
- Right rail: completion, next milestone, target farming, relevant quests, region/family links, rewards.

The existing canvas panel can keep the left-list/right-detail structure as the first iteration, but the detail side needs more substructure.

### Mobile / Narrow Layout

Use stacked sections:

- Search and filter chips at top.
- Monster list cards.
- On selection, open a full-height detail sheet.
- Use segmented tabs: Overview, Combat, Habitat, Drops, Mastery.
- Keep touch targets at least 44 px high.

Project Starfall is PC-first, but this layout also helps narrow browser windows and Steam Deck-like displays.

## C. UI Wireframe Description

### Monster List Page

Top summary band:

- Global completion percentage.
- Monsters sighted / total.
- Monsters mastered / total.
- Drops obtained / known.
- Current pinned target.
- Unclaimed rewards count.

Search and filters:

- Search input with placeholder "Search monster, drop, map, mechanic".
- Quick chips: All, Sighted, Unseen, Ready Milestone, Missing Drops, Bosses, Elites, Current Region.
- Filter drawer for region, biome, family/type, level range, threat tier, weakness, drop item, rarity, quest relevance, event-only, boss/elite/normal.
- Sort: Recommended, Level, Region, Kills, Completion, Missing Drops, Name.

List cards:

- Portrait or silhouette.
- Name or unknown nameplate.
- Family/type.
- Level range or broad threat.
- Kill count.
- Research rank.
- Completion ring or bar.
- Missing-drop indicator.
- Quest/target-farm pin.

Locked treatment:

- Unseen: silhouette, "Unknown Monster", broad region if known from map.
- Sighted: silhouette or partial portrait, family, biome, rough threat.
- Killed: full name and portrait.

### Monster Detail Page

Header:

- Monster portrait/sprite/silhouette.
- Name or unknown nameplate.
- Family/type.
- Normal/elite/boss badge.
- Threat tier.
- Level range.
- Kill count.
- Completion percentage.
- Research rank.
- Region/biome tags.
- Quest/task tags.
- Target Farm button.
- Pin button.

Overview:

- One short flavor line.
- Basic behavior summary.
- First seen location.
- Common habitats.
- Tags: aggressive, passive, social, ranged, magic, flying, turret, charger, healer, armored, blocker, boss.

Combat:

- HP, attack, defense, speed, aggro range.
- Approximate bars at low research, exact values later.
- Weaknesses and resistances.
- Status vulnerabilities.
- Attack pattern chips.
- Special mechanics.
- Recommended approach.
- Boss phase links when applicable.

Habitat:

- Spawn locations with map names.
- Map level ranges.
- Spawn conditions.
- Time/weather/event requirements when those systems exist.
- Respawn class: field pack, elite, dungeon boss, echo boss, rift.
- Rare spawn rules.
- Map pin / travel hint.

Drops:

- Item-level table, not only table-level summary.
- Columns: icon, item name, state, quantity, rarity, odds, conditions, personally obtained count.
- Toggle: percentage / 1-in-X.
- Toggle: current odds / base odds / boosted odds.
- Grouping: Guaranteed, Materials, Equipment, Cards, Consumables, Rare Valuables, Boss Set.

Collection footer:

- Entry completion.
- Drops discovered.
- Drops obtained.
- Next unlock milestone.
- Rewards earned.
- Mastery status.

### Completion Page

Global rows:

- Overall Monster Guide completion.
- Research rank completion.
- Drop discovery.
- Drop obtained.
- Boss research.
- Region completion.
- Family completion.

Cards:

- Recently discovered drops.
- Dry-streak assistance available.
- Pinned monsters.
- Recommended next milestones.
- Unclaimed cosmetics/titles/badges.

### UX Recommendations

Desktop:

- Keep the guide dense and scannable, matching the existing RPG-client UI.
- Use a persistent left list and right detail view so players can compare monsters quickly.
- Use section headers inside the detail view instead of hiding everything behind tabs on wide screens.
- Keep the drop table visible without excessive scrolling once a monster is researched.
- Put Target Farm, Pin, and Travel/Map Hint actions near the header.

Mobile and narrow layouts:

- Use a stacked detail sheet with segmented tabs: Overview, Combat, Habitat, Drops, Mastery.
- Collapse filters into a drawer after the primary search field and quick chips.
- Keep all touch targets at least 44 px high.
- Avoid dense multi-column drop tables; use row cards with odds and obtained count on a second line.

Tabbed vs scroll:

- Desktop should use a scroll-first layout with sticky local section navigation.
- Mobile should use tabs/segments to avoid a very long single scroll.
- Boss entries should use an extra Phases segment or collapsible section.

Locked and partial information:

- Locked fields should show a short reason and the next milestone, not only `???`.
- Partial information should be useful: "Volcanic ranged attacker" is better than "Unknown".
- Completed entries should get a badge, strong completion state, and quick access to personal drop history.

Accessibility:

- Do not rely on color for rarity, completion, or lock state.
- Add text labels and shape markers for every rarity tier.
- Use consistent focus order: search, filters, monster list, detail actions, detail rows.
- Support keyboard/controller list navigation and activation.
- Tooltips should also be reachable from focus, not only hover.
- Keep progress bars paired with text such as `25/50` and `3 drops missing`.
- Avoid tiny canvas text in drop rows; use shorter labels and secondary lines rather than shrinking below readable sizes.

## D. Unlock Milestone Plan

Use these normal-monster milestones:

| Stage | Requirement | Unlocks |
|---|---:|---|
| Sighted | Seen on screen or targeted | Silhouette, family, broad biome, rough threat tier, unknown nameplate, region hint. |
| Identified | 1 kill | Name, portrait, kill counter, flavor line, broad habitat, one behavior tag, target-farm eligibility. |
| Familiar | 5 kills | Approximate HP/attack/defense bars, obvious weakness/counter, basic attack pattern tags, known spawn-region names. |
| Studied | 10 kills | Exact core stats, resistances, weaknesses, notable attacks, quest/task relevance, exact map list. |
| Cataloged | 25 kills | Full drop list, quantity ranges, rarity tiers, known-but-unobtained silhouettes, hidden slot count. |
| Researched | 50 kills | Exact odds for common, uncommon, rare, and very rare drops; conditional-drop notes; base/current odds toggle. |
| Expert | 100 kills | Ultra-rare odds, advanced behavior notes, rare spawn/respawn details, personal drop-history summary. |
| Mastered | 250 kills | Mastery badge, cosmetic/lore reward, profile title/frame, tracking perk, optional account-level collection perk. |

Bosses should use lower clear thresholds because each kill takes much longer:

| Stage | Requirement | Unlocks |
|---|---:|---|
| Sighted | Enter encounter or see boss | Silhouette, boss family, arena/region, rough threat. |
| Identified | 1 clear | Name, portrait, phase names, boss material, kill counter. |
| Familiar | 2 clears | Core mechanics, basic counter plan, adds, rough stats. |
| Studied | 3 clears | Exact stats, phase thresholds, break profile, dungeon objectives. |
| Cataloged | 5 clears | Full boss drop list, set silhouettes, quantity ranges, rarity tiers. |
| Researched | 10 clears | Exact common through very rare odds, boss-set base chance and pity rules. |
| Expert | 20 clears | Ultra-rare odds, advanced phase notes, personal drop history, duplicate set counts. |
| Mastered | 50 clears | Boss mastery badge, cosmetic/title/profile frame, trophy, expanded lore. |

Important tuning rule: useful combat information arrives by 5-10 kills for normal monsters and 2-3 clears for bosses. The late grind should refine farming and completion, not withhold basic survival knowledge.

## E. Drop Table Rules

### Drop States

Use four item states:

1. **Hidden slot**
   - The player knows an undiscovered slot exists.
   - Show a question icon, source group, and broad rarity when appropriate.
   - Example: "Unknown rare valuable".

2. **Revealed but not obtained**
   - The item is visible as a ghosted/silhouetted icon.
   - Show category text before item acquisition if the milestone allows it.
   - Example: "Unknown crafting material", "Boss set weapon", or "Storm card".

3. **Obtained**
   - Show full icon, name, item type, description, quantity range, rarity, and personal count.
   - If exact odds are not researched yet, show rarity tier instead of exact odds.

4. **Fully researched**
   - Show exact odds, conditions, table source, quantity range, and current boosted odds.
   - Show both percentage and 1-in-X through a toggle.

### Exact Odds Recommendation

Use **milestone-based exact odds**, with personal acquisition as an early reveal override.

- Before 25 kills: no full drop list; only broad previews from current `enemy.drops`.
- At 25 kills: full list, rarity, quantity, and silhouettes appear.
- If an item drops before 25 kills: reveal that item immediately in obtained state.
- At 50 kills: exact common through very rare odds appear.
- At 100 kills: exact ultra-rare odds, global rare table odds, and rare spawn/respawn rules appear.
- Boss set odds should appear at 10 boss clears, including base chance, current pity-adjusted chance, and missing-piece weighting.

This prevents wiki dependence while preserving a sense of research.

### Rarity Treatment

For the Monster Guide, use these collection-display tiers:

- Common
- Uncommon
- Rare
- Very Rare
- Ultra Rare

Map current item rarities into the guide as:

- `Common` -> Common
- `Uncommon` -> Uncommon
- `Rare` -> Rare
- `Epic` -> Very Rare
- `Relic`, `Mythic`, `Ascendant`, `Celestial` -> Ultra Rare

Keep the original item rarity in item tooltips when needed. Example: display "Ultra Rare - Relic item".

Do not rely on color alone. Each rarity row should include:

- Text label.
- Shape/border pattern.
- Small icon marker.
- Tooltip.
- Optional color.

Suggested non-color markers:

- Common: single dot.
- Uncommon: double dot.
- Rare: diamond.
- Very Rare: star.
- Ultra Rare: starburst.

### Project Starfall Drop Table Adaptation

The UI should flatten current drop tables into item rows while preserving source groups:

- `primaryEtc`: guaranteed material.
- `coins`: currency.
- `potions`: consumables.
- `equipment`: world equipment group.
- `cards`: card drops.
- `bonusMaterials`: extra materials.
- `plinkoBalls`: plinko item drops.
- `rareValuables`: global rare table.
- `bossSet`: boss equipment generated from `BOSS_EQUIPMENT_SOURCES` and `BOSS_EQUIPMENT_ITEMS`.

Each row should calculate:

- `tableChance`: chance the source table rolls.
- `entryChance`: chance within that table.
- `chancePerKill`: `tableChance * entryChance`, adjusted by current boosts when showing current odds.
- `baseChancePerKill`: same calculation without target-farm, coupon, map modifier, or admin boosts.
- `quantityMin` / `quantityMax`.
- `conditions`: boss only, elite only, class-weighted, level requirement, quest requirement, dungeon clear, current class weighting, pity-adjusted boss set chance.

## F. Completion And Reward System

### Progress Ownership

Use a hybrid model:

- **Account-wide**: sighted, kills, research tier, drops obtained, drop counts, completion, mastery badges, cosmetics, titles, lore, and guide rewards.
- **Character-specific**: target farm selection/streak, current pins, current UI filters, combat mastery effects if any remain.
- **World/server-specific later**: seasonal event monsters, server-first discovery metadata, market-driven collection stats.

For the current local prototype, store this inside the save state now, but shape it as `account.monsterGuide` or `collection.monsterGuide` so it can migrate cleanly to account-wide storage later.

### Completion Formula

Monster entry completion should be weighted so combat research matters more than rare-drop perfection:

- 50% research milestone progress.
- 25% drop discovery.
- 15% drop obtained.
- 10% mastery reward claimed.

Example:

```text
entryCompletion =
  researchTierProgress * 0.50 +
  discoveredDropSlots / totalKnownDropSlots * 0.25 +
  obtainedDropSlots / totalObtainableDropSlots * 0.15 +
  masteryClaimed * 0.10
```

Region completion:

- Average eligible monster entry completion in that region.
- Bosses can be weighted 2x in boss-focused regions.
- Exclude admin-only asset-test monsters.
- Event-only monsters count only during their event or in an event archive category.

Family completion:

- Average eligible monsters in that family.
- Multi-family monsters can contribute to all listed families at reduced weight or to their primary family only.

Global completion:

- Weighted average of normal monsters, bosses, drop collection, and mastery rewards.
- Keep a separate "All Drops" completion for dedicated collectors so rare-drop chasing does not block normal guide progress.

### Rewards

Prefer rewards that feel good but are not mandatory:

- Monster mastery badges.
- Region titles.
- Profile frames.
- Damage-number cosmetics themed to regions.
- Housing trophies/decorations if housing arrives.
- Monster portrait stamps.
- Expanded lore pages.
- Map hints.
- Tracking pins.
- Improved guide filters.
- Target-farm presets.
- Cosmetic aura variants.
- Accomplishment entries.

Avoid:

- Large global combat stat bonuses.
- Mandatory best-in-slot power.
- Rewards that make players grind the guide before normal leveling.

Current +2%/+5% monster-specific damage can remain temporarily as a prototype validation reward, but the long-term version should either:

- Cap it at a small monster-specific convenience bonus, or
- Replace it with non-power rewards once the cosmetic/profile/research systems are mature.

### Anti-Frustration Mechanics

Add these mechanics:

- Daily first-kill research bonus: first kill of a monster each day grants +2 research progress, capped once per monster per day.
- Weekly research tasks: "Study 3 Construct monsters", "Catalog 1 boss drop table", "Defeat 30 Frost monsters".
- Dry-streak reveal assist: after enough kills without a new guide discovery, reveal one missing non-ultra drop slot or condition note.
- Soft pity for collection-page unlocks, not necessarily for rare item drops.
- Target-farm pin: from the guide, pin a monster to the HUD and world map.
- Missing-drop focus: target farm can improve research/drop visibility before it improves actual item odds.
- Boss-set duplicate tracking: show missing set pieces and personal duplicates clearly.

## G. Data Model / Implementation Plan

### Static MonsterDefinition

Extend current `ENEMIES` into a richer definition while preserving existing fields:

```js
{
  id: 'cinderSpitter',
  name: 'Cinder Spitter',
  family: 'Volcanic Beast',
  type: 'normal',
  category: 'thrower',
  levelRange: [28, 55],
  threatTier: 'standard',
  hpMult: 0.88,
  damageMult: 1.12,
  defenseMult: 0.9,
  expMult: 1.26,
  speed: 62,
  behavior: 'thrower',
  behaviorTags: ['Ranged', 'Arcing Projectile', 'Volcanic'],
  elements: ['Fire'],
  weaknesses: ['Cold', 'Stun', 'Gap Closer'],
  resistances: ['Fire'],
  statusVulnerabilities: ['Stun', 'Slow'],
  attackPatterns: ['Mid-range lob', 'Retreat after cast'],
  mechanics: ['Arcing cinder globs'],
  counter: 'Gap closers, vertical movement, and stuns.',
  spawnLocations: ['cinderHollow'],
  spawnConditions: [],
  respawnClass: 'fieldPack',
  lore: 'A furnace-fed scavenger that spits burning slag from safe ledges.',
  asset: '...',
  tags: ['Cinder', 'Field', 'Ranged'],
  excludeFromCollection: false
}
```

### Static MonsterDrop

Create normalized rows from current drop pools and boss sources:

```js
{
  id: 'cinderSpitter:cinderGland',
  monsterId: 'cinderSpitter',
  itemId: 'cinderGland',
  itemKind: 'material',
  tableId: 'primaryEtc',
  tableLabel: 'Guaranteed Etc',
  quantityMin: 1,
  quantityMax: 1,
  rarityTier: 'Common',
  tableChance: 1,
  entryChance: 1,
  dropChance: 1,
  conditions: [],
  unlockVisibilityMilestone: 25,
  oddsVisibilityMilestone: 50,
  requiresQuest: '',
  requiresLevel: 0,
  isHiddenUntilFound: false
}
```

Boss-set row:

```js
{
  id: 'emberjawGolem:furnaceheart_arsenal',
  monsterId: 'emberjawGolem',
  itemKind: 'bossSet',
  setId: 'furnaceheart_arsenal',
  tableId: 'bossSet',
  quantityMin: 1,
  quantityMax: 1,
  rarityTier: 'Very Rare',
  dropChance: 0.1,
  conditions: ['Boss clear', 'Weighted toward missing pieces'],
  unlockVisibilityMilestone: 5,
  oddsVisibilityMilestone: 10
}
```

### PlayerMonsterProgress

Replace the current `killsByEnemyId`-only model with a normalized progress object:

```js
{
  monsterId: 'cinderSpitter',
  sighted: true,
  kills: 50,
  firstSeen: 1781140000000,
  firstSeenMapId: 'cinderHollow',
  firstKilled: 1781140200000,
  lastKilled: 1781145400000,
  researchTier: 'researched',
  researchPoints: 50,
  dropsObtained: {
    cinderGland: true,
    upgradeCatalyst: true,
    ashflare_core: false
  },
  dropCounts: {
    cinderGland: 47,
    upgradeCatalyst: 3
  },
  unlockedFields: {
    name: true,
    combatStats: true,
    dropList: true,
    commonOdds: true
  },
  completedMilestones: [1, 5, 10, 25, 50],
  claimedMilestoneRewards: [25],
  masteryComplete: false,
  dryStreaksByDropGroup: {
    cards: 18,
    rareValuables: 50
  },
  dailyResearchBonusDate: '2026-06-11'
}
```

### Account Monster Guide State

```js
{
  selectedEnemyId: 'cinderSpitter',
  pinnedEnemyIds: ['cinderSpitter', 'emberjawGolem'],
  filters: {
    region: '',
    family: '',
    completion: '',
    dropItemId: ''
  },
  entriesByMonsterId: {
    cinderSpitter: PlayerMonsterProgress
  },
  regionRewardsClaimed: {},
  familyRewardsClaimed: {},
  globalRewardsClaimed: {}
}
```

### Dynamically Calculated

Calculate these at snapshot time:

- Research tier from kills/research points.
- Exact stat values from existing balance functions.
- Drop rows from current drop pools and economy settings.
- Current odds after target-farm, coupon, map modifier, admin multiplier, and boss pity.
- Completion percentages.
- Available filters and sort counts.
- Recommended target-farm maps.
- Quest relevance from `QUESTS` and `questNpcs`.

### Frontend / Backend Ownership

Current prototype:

- Keep static definitions in `project-starfall-data.js`.
- Keep derived guide snapshots in `project-starfall-engine.js`.
- Keep visual rendering, focus regions, filters, and table drawing in `project-starfall-ui.js`.
- Persist through the current local save format with migration support.

Future account/backend version:

- Store `MonsterDefinition` and `MonsterDrop` as server-authored static data or versioned JSON.
- Store `PlayerMonsterProgress` in account-scoped persistence.
- Let the client calculate display-only filters, completion percentages, lock labels, and visual grouping from authoritative progress.
- Let the server validate kills, drops, milestone claims, and account rewards once multiplayer/server authority exists.
- Include a content version field so changed drop tables can migrate or preserve historical obtained counts safely.

### Stored Per Player / Account

Store these:

- Sighted state.
- Kills/research points.
- First seen/killed and last killed timestamps.
- Drop obtained booleans.
- Drop duplicate counts.
- Claimed milestone rewards.
- Pinned targets.
- Dry-streak counters for reveal assist.

### Visibility Rules

Expose immediately:

- Broad threat, family, biome, and silhouette after sighting.
- Name, portrait, kill count, broad habitat, and one behavior tag after first kill.
- Basic tactical counter by 5 kills.

Hide until milestones:

- Exact stats until 10 kills.
- Full drop list until 25 kills, except items already obtained.
- Exact common through very rare odds until 50 kills.
- Exact ultra-rare, global rare, boss pity, and rare-spawn odds until 100 kills or the boss-equivalent milestone.
- Mastery cosmetics, expanded lore, and profile rewards until mastery.

### Refactor Targets

1. `createMonsterGuideState`
   - Migrate from `killsByEnemyId` to `entriesByMonsterId`.
   - Preserve old saves by copying old kills into new entries.

2. `recordMonsterGuideDefeat`
   - Increment kills/research.
   - Set first/last killed.
   - Claim sighted if missing.
   - Check milestone unlock events.

3. Enemy spawn / camera awareness
   - Add `recordMonsterSighted(enemy, mapId)` when a monster appears on screen, attacks, or is targeted.

4. Loot generation / pickup
   - Add `recordMonsterDrop(enemyId, item)` when a generated monster drop is picked up by the player or pet.
   - Track stack quantities and duplicate equipment/card counts.

5. `getMonsterGuideDropInfo`
   - Include boss-set sources.
   - Return item-level drop rows with lock state, not only table summaries.

6. `drawMonsterGuideCanvas`
   - Split into list, filter bar, detail header, tab/section renderers, and drop table renderer.
   - Add keyboard/controller-friendly regions for filters and table rows.

7. Tests
   - Add tests for save migration, milestone tiers, sighted state, drop-row generation, boss-set inclusion, completion math, and asset-test exclusion.

## H. Example Monster Entry

Example entry: **Cinder Spitter**

### Header

- Name: Cinder Spitter
- State: Researched at 50 kills
- Family: Volcanic Beast
- Type: Normal / Ranged
- Threat tier: Standard
- Level range: 28-55
- Region tags: Cinder, Field, Volcanic
- Quest tags: Cinder route, material sample if active quest references Cinder drops
- Kill count: 50
- Completion: 68%
- Current research rank: Researched
- Next milestone: 100 kills, Expert

### Overview

Flavor:

"Cinder Spitters cling to furnace ledges and lob half-cooled slag at anything crossing the hollow."

Behavior:

- Mid-range thrower.
- Prefers distance.
- Lobs arcing cinder projectiles.
- Vulnerable after a throw.

First seen:

- Cinder Hollow

Common habitats:

- Cinder Hollow
- Emberjaw Lair as an add
- Emberjaw Furnace as an echo-boss add

Tags:

- Ranged
- Arcing Projectile
- Volcanic
- Field
- Add

### Combat Notes

Unlocked at 5 kills:

- Avoid standing still on lower platforms.
- Jump or close distance when it begins a lob.
- Fast melee, stuns, traps, and vertical movement are effective.

Unlocked at 10 kills:

- Exact HP, damage, defense, and speed values by level range.
- Behavior: thrower.
- Aggro range: 620 px.
- Attack range: 620 px.
- Counter: gap closers, vertical movement, and stuns.

Recommended approach:

- Fighter: dash through the arc and punish after the throw.
- Mage: use blink or ranged spells to avoid platform tax.
- Archer: maintain horizontal distance and fire during the recovery window.
- Trapper: place a snare where it retreats.

### Habitat

- Spawn maps: Cinder Hollow, Emberjaw Lair, Emberjaw Furnace.
- Spawn class: field pack / dungeon add / boss echo add.
- Spawn conditions: normal route availability.
- Respawn category: standard field wave or dungeon wave.
- Map hint: use Cinder Refuge routes, then pin Cinder Hollow for repeated farming.

### Drop Table

Illustrative researched state using the current table model:

| State | Item | Source | Quantity | Rarity | Odds |
|---|---|---|---:|---|---:|
| Obtained | Cinder Gland | Guaranteed Etc | 1 | Common | 100% |
| Obtained | Coins | Coins | variable | Common | 50% base table |
| Revealed | Potions / utility consumables | Potions | 1 | Common-Uncommon | 12% base table |
| Revealed not obtained | Focus accessory | Equipment | 1 | Uncommon-Rare | Table odds at 50 kills |
| Revealed not obtained | Cinder gear | Equipment | 1 | Uncommon-Rare | Table odds at 50 kills |
| Obtained | Upgrade Catalyst | Bonus Materials | 1 | Rare | 6% base table |
| Revealed not obtained | Ember Glint card | Cards | 1 | Common | 3.5% table, weighted |
| Revealed not obtained | Ashflare Core card | Cards | 1 | Uncommon | 3.5% table, weighted |
| Hidden slot | Rare Valuable | Rare Valuables | 1 | Very Rare+ | Exact ultra-rare odds at 100 kills |

### Unlock State

- Sighted: complete.
- 1 kill: complete.
- 5 kills: complete.
- 10 kills: complete.
- 25 kills: complete.
- 50 kills: complete.
- 100 kills: incomplete.
- 250 kills: incomplete.

### Completion Progress

- Research: 6/8 stages.
- Drops discovered: 8/10.
- Drops obtained: 3/10.
- Personal duplicate counts: Cinder Gland x47, Upgrade Catalyst x3.
- Next unlock: 100 kills reveals ultra-rare odds, advanced behavior notes, and personal drop-history summary.
- Reward status: 25-kill Cinder Spitter badge claimed; 250-kill mastery cosmetic locked.

## I. Pitfalls To Avoid

- Do not make players wait until 50 kills for basic combat counters. The current system is close, but the optimized system should reveal behavior/counter by 5 kills and exact tactical stats by 10.
- Do not show only table summaries. Farmers need item rows, personal counts, missing slots, and odds.
- Do not omit boss-set drops from the guide. Boss farmers must see set-piece silhouettes, drop chance, pity rules, and duplicates.
- Do not count admin-only asset-test enemies toward completion.
- Do not make rare-drop completion block normal region completion. Use separate "drop collection" and "research mastery" tracks.
- Do not rely on color-only rarity indicators.
- Do not turn the Monster Guide into mandatory power progression. Keep rewards mostly cosmetic, prestige, and convenience.
- Do not overfit the guide to current localStorage. Shape data as account-wide now so future backend migration is clean.
- Do not hide rare drops forever. Silhouettes and category labels should appear at the drop-list milestone.
- Do not expose boosted odds without explaining why they changed. Show base odds and current odds separately.
- Do not add all filters at once if the canvas UI becomes cluttered. Start with search, region, family, completion, boss/normal, and drop item.

## J. Prioritized Action Plan

### Quick Wins

1. Rename nothing: keep **Monster Guide** and add "Monster Research" language inside the panel.
2. Exclude admin-only asset-test enemies from normal guide completion.
3. Add `excludeFromCollection` and `collectionCategory` fields to enemy definitions.
4. Add structured `tags`, `weaknesses`, `resistances`, `behaviorTags`, `spawnConditions`, and `lore` fields to enemies.
5. Add a sighted state and unknown/silhouette treatment for zero-kill monsters.
6. Add region/family/boss/completion filter chips above the existing search list.
7. Include boss-set drop sources in Monster Guide drop info.
8. Add a drop-row builder that flattens current drop tables into item rows for the selected monster.
9. Add tests for milestone calculation, boss-set drop inclusion, and collection exclusion.

### Medium-Term Improvements

1. Migrate `monsterGuide.killsByEnemyId` to `monsterGuide.entriesByMonsterId` while preserving old saves.
2. Track first seen, first killed, last killed, drops obtained, and drop counts.
3. Implement the normal milestone ladder: sighted, 1, 5, 10, 25, 50, 100, 250.
4. Implement the boss milestone ladder: sighted, 1, 2, 3, 5, 10, 20, 50.
5. Build item-level drop table UI with hidden, revealed, obtained, and researched states.
6. Add odds display toggles: percent, 1-in-X, base odds, current odds.
7. Add region and family completion pages.
8. Add target-farm pins that appear in the guide, world map, and HUD.
9. Add daily first-kill research bonus and weekly research tasks.

### Long-Term System Upgrades

1. Move Monster Guide progress to an account-wide profile model.
2. Add cosmetic rewards: monster badges, titles, frames, trophies, and themed damage-splat/aura variants.
3. Add dry-streak reveal assistance for missing collection entries.
4. Add event-only and quest-only monster categories with archive rules.
5. Add map pins and navigation hints unlocked by research.
6. Add richer boss pages with phase diagrams, break windows, add waves, and personal clear/drop history.
7. Add backend-ready schemas for account monster progress, drop history, and seasonal monster collections.
8. Add accessibility refinements: full keyboard navigation, controller focus rings, screen-reader labels for rarity and lock states, and non-color rarity symbols.

## Concrete Developer Blueprint

Build the optimized system in this order:

1. Extend enemy and drop metadata in `project-starfall-data.js`.
2. Add save migration from `killsByEnemyId` to `entriesByMonsterId`.
3. Add sighting and drop acquisition recording in the engine.
4. Replace table-only drop guide data with item-level `MonsterDropRow` snapshots.
5. Redesign the Monster Guide UI around search/filter/list/detail/drop-table sections.
6. Add completion calculations and reward claim state.
7. Add tests for migration, unlocks, drop states, odds display, and completion math.

The end state should be a Monster Guide that helps a new player fight better after a few kills, helps a farmer make informed route/drop decisions after moderate research, and gives completionists long-term goals without turning monster research into mandatory power grinding.
