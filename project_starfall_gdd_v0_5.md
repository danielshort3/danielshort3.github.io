# Project Starfall — Detailed Game Design Document

**Version:** 0.6 current prototype sync and future roadmap  
**Document type:** Detailed Game Design Document  
**Working title:** Project Starfall  
**Genre:** 2D side-scrolling multiplayer action RPG  
**Platform target:** PC-first, with possible future companion/mobile features  
**Design direction:** Character-first progression RPG with intuitive classes, branching advancement, party buffs, chance-based gear upgrades, and long-term multi-character account progression.

> **Core pitch:** A PC-first 2D multiplayer action RPG where players choose a simple starting class, permanently branch into distinct advanced classes, party with other players for dungeons and bosses, and chase risky gear upgrades that can improve, scar, mutate, or transform equipment instead of only increasing stats.

---

## Table of Contents

0. [Current Prototype State](#0-current-prototype-state)
1. [High-Level Vision](#1-high-level-vision)
2. [Design Pillars](#2-design-pillars)
3. [Target Audience](#3-target-audience)
4. [Game Format](#4-game-format)
5. [Differentiation Goals](#5-differentiation-goals)
6. [World, Tone, and Fantasy](#6-world-tone-and-fantasy)
7. [Core Gameplay Loops](#7-core-gameplay-loops)
8. [Player Progression Overview](#8-player-progression-overview)
9. [Class System Overview](#9-class-system-overview)
10. [Base Classes](#10-base-classes)
11. [Advanced Classes and Skills](#11-advanced-classes-and-skills)
12. [Party Skills and Buff Design](#12-party-skills-and-buff-design)
13. [Specializations](#13-specializations)
14. [Multiple Character / Account Roster System](#14-multiple-character--account-roster-system)
15. [Gear System](#15-gear-system)
16. [Risk-Based Upgrade System](#16-risk-based-upgrade-system)
17. [Class-Reactive Gear Traits](#17-class-reactive-gear-traits)
18. [Combat Content](#18-combat-content)
19. [Multiplayer Structure](#19-multiplayer-structure)
20. [Economy and Trading](#20-economy-and-trading)
21. [Questing, Leveling, and World Structure](#21-questing-leveling-and-world-structure)
22. [UI / UX Direction](#22-ui--ux-direction)
23. [Art Direction](#23-art-direction)
24. [Audio Direction](#24-audio-direction)
25. [Monetization Philosophy](#25-monetization-philosophy)
26. [Prototype Scope And MVP Definition](#26-prototype-scope-and-mvp-definition)
27. [Roadmap](#27-roadmap)
28. [Balance Principles](#28-balance-principles)
29. [Risks and Mitigations](#29-risks-and-mitigations)
30. [Open Design Questions](#30-open-design-questions)
31. [Appendix A: Skill Summary Tables](#appendix-a-skill-summary-tables)
32. [Appendix B: Buff Category Rules](#appendix-b-buff-category-rules)
33. [Appendix C: Upgrade Outcome Tables](#appendix-c-upgrade-outcome-tables)
34. [Appendix D: Glossary](#appendix-d-glossary)
35. [Appendix E: Skill Role and Prerequisite Reference](#appendix-e-skill-role-and-prerequisite-reference)
36. [Appendix F: Enemy Implementation Quick Reference](#appendix-f-enemy-implementation-quick-reference)
37. [Appendix G: Equipment and Shop Quick Reference](#appendix-g-equipment-and-shop-quick-reference)
38. [Appendix H: AI Asset Production Standards](#appendix-h-ai-asset-production-standards)
39. [Appendix I: Character and NPC Visual Prompt Library](#appendix-i-character-and-npc-visual-prompt-library)
40. [Appendix J: Enemy and Boss Visual Prompt Library](#appendix-j-enemy-and-boss-visual-prompt-library)
41. [Appendix K: Equipment, Item, and Shop Visual Prompt Library](#appendix-k-equipment-item-and-shop-visual-prompt-library)
42. [Appendix L: Skill Icon, VFX, and Animation Prompt Library](#appendix-l-skill-icon-vfx-and-animation-prompt-library)
43. [Appendix M: Map, Tileset, and UI Visual Prompt Library](#appendix-m-map-tileset-and-ui-visual-prompt-library)

---

# 0. Current Prototype State

This document is now synced against the browser prototype in the repository. When this section conflicts with older aspirational text later in the document, this section describes the current implementation and the later text should be read as design direction or future work.

## 0.1 Current Build

The current Project Starfall implementation is a static, no-framework browser prototype served at `/project-starfall`. It uses a Canvas 2D stage with route-specific CSS, plain JavaScript modules, generated image assets, and local persistence through `localStorage` under `projectStarfallPrototypeSave.v1`.

The prototype currently includes:

- Class selection for **Fighter**, **Mage**, and **Archer**.
- Leveling, XP, coins, uncapped/endless scaling support, base skill points, and advanced skill points.
- Base Skill Batch and Advanced Skill Batch rank progression.
- Permanent advanced class selection after completing Level 20 class trials, with advanced jobs becoming usable at Level 25.
- Explicit role profiles for all base and advanced classes.
- One low-cooldown **primary training skill** for every advanced class.
- Canvas combat with movement, jump, gravity, broad training platforms, ladders, camera follow, melee, ranged, magic, chain, trap, buff, animated projectile trails, area, field, burn, and impact effects.
- Save, load, reset, and character roster controls. Saved characters include keybinds.
- Inventory, Equipment, Usable, Etc, Pet, shop, upgrade station, world map, monster guide, quests, dungeons, class trials, party/self-buff panel, keybinds, minimap, admin settings, first-session guide, combat metrics, performance debug, and session log.
- Local simulated party finder slots with prototype ally stat bonuses, timed assist behaviors, and visible ally support.
- Procedural browser audio cues for attacks, skills, buffs, loot, level-ups, upgrades, damage, travel, defeat, party assists, and UI confirmation.
- Equipment-driven character visuals using a generic layered runtime rig plus item-specific weapon and gear visuals.
- Generated character portraits, player animation sheets, enemy animation sheets, skill icons, item icons, map art, station art, FX sheets, and equipment-layer sheets.

## 0.2 Current Prototype Constraints

The current prototype is local and single-player. Multiplayer, server authority, live economy, trading, true networked party members, matchmaking, music, and production monetization are design goals only. Party skills currently apply solo value and can also affect local simulated allies, but this is still a role, UI, and balance placeholder rather than multiplayer infrastructure.

The prototype should continue to favor direct implementation clarity over MMO-scale architecture until the core combat, class identity, progression, UI, and upgrade loops are validated.

## 0.3 Current Class And Role Summary

| Class | Resource | Current Role Profile | Current Prototype Focus |
|---|---|---|---|
| Fighter | Momentum | Hybrid / melee control | Close-range impact, guard, slam, break, burst. |
| Mage | Energy | Hybrid / area setup | Stable-footed spellcasting, teleport, area magic, marks. |
| Archer | Focus | Hybrid / mobile range | Jump attacks, marks, piercing shots, mobility. |
| Guardian | Stored Impact | Support / Control | Stagger, crack, shield, defensive counter-pressure. |
| Berserker | Rage | Bossing | Low-HP scaling, cleave, lifesteal, aggressive sustain. |
| Duelist | Tempo | Bossing | Same-target combo tempo and fast melee uptime. |
| Fire Mage | Heat | Mobbing | Fireball training, burn spread, wildfire, inferno. |
| Rune Mage | Runic Energy | Support / Control | Rune marks, links, glyph fields, detonation. |
| Storm Mage | Charge | Mobbing | Chain Bolt as clustered-enemy clearing. |
| Sniper | Aim | Bossing | Aimed shots, weak points, armor pierce, execution. |
| Trapper | Preparation | Mobbing / Control | Low-cooldown traps, lure, detonate, kill zones. |
| Beast Archer | Bond | Support / Hybrid | Companion strikes, pack marks, sustain. |

## 0.4 Advanced Primary Training Skills

Each advanced class has exactly one immediate low-cooldown primary training skill. The primary skill input prefers this skill once the job is selected.

| Advanced Class | Primary Training Skill | Prototype Cooldown | Current Purpose |
|---|---|---:|---|
| Guardian | Shield Bash | 0.48s | Basic guarded strike that builds Stored Impact and applies control. |
| Berserker | Blood Cleave | 0.56s | Wide training cleave with low-HP scaling. |
| Duelist | Quick Cut | 0.42s | Fast same-target Tempo builder. |
| Fire Mage | Fireball | 0.62s | Explosive projectile and burn starter. |
| Rune Mage | Rune Mark | 0.58s | Rune bolt that marks, slows, and links targets. |
| Storm Mage | Chain Bolt | 0.82s | Multi-target training attack for mobbing. |
| Sniper | Aimed Shot | 0.65s | Precision ranged training attack for bossing. |
| Trapper | Snare Trap | 0.55s | Low-cooldown trap placement and control. |
| Beast Archer | Companion Strike | 0.56s | Coordinated shot that marks for companion pressure. |

## 0.5 Current World And Progression

The prototype currently contains a starter safe zone, multiple connected field maps, dungeon maps, class trials, world routes, route portals, and an extended late-prototype Ascension training path.

Implemented maps and route structure:

- **Starfall Crossing**: safe-zone hub with shop, class, and upgrade stations.
- **Greenroot Meadow** and **Thornpath Thicket**: early forest training.
- **Bramble Depths**: forest dungeon route.
- **Rustcoil Ruins** and **Gearworks Vault**: ruins and mechanical dungeon route.
- **Cinder Hollow** and **Emberjaw Lair**: fire route and boss dungeon.
- **Bandit Ridge Camp** and **Oreback Quarry**: midgame field progression.
- **Ashglass Pass**, **Frostfen Outskirts**, **Glacier Spine**, **Stormbreak Cliffs**, **Astral Archive**, **Eclipse Frontier**, and **Endless Rift**: late prototype route progression and endurance-style training.

Field maps use multiple training structures instead of a single repeated layout:

- **Shared lane maps**: Greenroot Meadow, Bandit Ridge Camp, Oreback Quarry, Frostfen Outskirts, Stormbreak Cliffs, Astral Archive, and Endless Rift.
- **Switchback terrace maps**: Rustcoil Ruins, Cinder Hollow, Ashglass Pass, and Eclipse Frontier.
- **Vertical canopy maps**: Thornpath Thicket and Glacier Spine.
- Dungeon maps stay compact arenas: Bramble Depths, Gearworks Vault, Emberjaw Lair, and Rimewarden Sanctum.

Implemented enemies:

- Slimelet, Mossback, Thorn Sprout, Bristle Boar, Dust Imp, Clockbug, Ember Wisp, Bandit Cutter, Bandit Thrower, Oreback Beetle, Glowcap Healer, Cracked Mimic.
- Frostling Scout, Rimeback Brute, Snowglare Wisp, and Rimewarden.
- Boss and dungeon enemies currently include Brambleking, Clockwork Titan, Quarry Colossus, Emberjaw Golem, and Rimewarden.

## 0.6 Current UI / UX Implementation

The current UI is intentionally dense and tool-like, closer to an RPG client than a landing page. The browser prototype includes:

- HUD meters for HP, MP, class resource, XP, coins, level, and selected class.
- Skill tree tabs for base and advanced jobs, visible locked prerequisites, ranks, MP costs, cooldowns, tooltips, role tags, and primary training labels.
- Equipment popup with equipment icons organized by slot on the left and a single-column stat sheet on the right.
- Inventory tabs for Equipment, Usable, and Etc items, item tooltips, stat roll tier indicators, item comparison, consumables, material stacks, inventory section coupons, equipment locks, and double-click equip/use behavior.
- Pet panel with unlockable Pet Assist automation, HP/MP potion slot assignment, loot filtering, and potion thresholds.
- Upgrade station with a gear-only selection list, movable upgrade window, outcome percentages, cost, material requirements showing current/required counts, selected upgrade aides, animation, and detailed result previews.
- World map panel on its own keybind, map travel, dungeons, portals, route progress, and boss gates.
- Keybinding UI with movement keys fixed, save-file persistence, skills reset on new characters, SKL on `K`, CHR on comma, world map on `W`, and dedicated NPC talk on `Y`.
- NPC quest prompts support `Y` to talk/accept/advance and `N` to decline.
- Minimap, command menu, party skill/self-buff panel, simulated party finder controls, guide strip, audio toggle, quests, class trials, dungeons, monster guide, accomplishments, beta/prototype settings, admin rate controls, optional DPS/money/hour/XP/hour combat metrics, and performance debug capture.

## 0.7 Current Visual And Asset Direction

The current visual direction has moved from purely generated portraits toward a **generic character body with equipment-driven appearance**:

- The same generic runtime rig is used for all playable classes.
- Equipped weapons, shields, armor, boots, rings, amulets, and class items influence the character's look.
- Character animation is deliberately simple and blocky, with walking, jumping, attacking, climbing, skill, buff, hit, and defeat states.
- Weapon animation differs by weapon family: melee swings, bow releases, spell casting, and buff casting use different character action rows and effects.
- Enemies use generated animation sheets with idle, move, telegraph, attack, projectile, buff, hit, and defeat rows.
- Loot drops show the item icon with a rarity aura, not a solid item background or border.
- Skill icons, item icons, map images, station images, FX sheets, and equipment-layer sheets are present under `img/project-starfall/`.
- AI-generated grid sheets now use a shared bordered-sheet pipeline: source sheets include visible `#00ffff` guide lines around every cell, processors detect those borders before chroma cleanup, and final transparent assets are audited for consistent sizing and clean cell edges.

## 0.8 Current Technical Architecture

The prototype uses separated static modules:

- Data owns classes, skills, enemies, maps, items, shops, upgrades, visuals, and constants.
- Engine owns movement, physics, combat, enemy behavior, loot, upgrades, save/load, class mechanics, projectiles, traps, fields, and drawing.
- UI owns panels, keybinds, tooltips, canvas windows, inventory, equipment, shop, upgrade station, world map, and HUD.
- Rig owns the code-drawn layered character/equipment rendering.
- Tests in `test.js` enforce route contracts, data contracts, skill prerequisites, save key, assets, animation shape, primary training skills, upgrade tables, UI hooks, and several combat behaviors.

## 0.9 Current Player Guide

This guide describes what is currently implemented in the browser prototype. Older sections of this document may describe intended future multiplayer or economy systems.

### Starting Out

1. Choose a character slot, look, name, and base class.
2. Pick **Fighter**, **Mage**, or **Archer**.
3. Start in **Starfall Crossing**, the safe-zone hub.
4. Use the first-session guide strip, World Map, Quests, and route portals to move into Greenroot Meadow.
5. Defeat monsters, pick up loot, equip upgrades, rank skills, and return to town for shops, class progression, and upgrades.

### Default Controls

| Action | Default |
|---|---|
| Move | Arrow keys |
| Jump | Space |
| Attack / primary skill | `J` or Left Shift |
| Party skill | `L` |
| Interact | `F` |
| Talk to NPC / accept / advance chat | `Y` |
| Decline NPC prompt | `N` |
| Loot nearby drops | Hold `Z` |
| Menu / close top window / close Attunement UI | Escape |
| Keybinds | Backslash |
| Character | Comma |
| Equipment | `E` |
| Party | `P` |
| Pet | `T` |
| World Map | `W` |
| Monster Guide | `N` |
| Skills | `K` |
| Quests | `Q` |
| Inventory | `I` |
| Shop | `O` |
| Upgrade | `U` |
| Session Log | `G` |
| Save Character | `F6` |
| Character Select / Load | `F7` |
| Delete Character | `F8` |
| Performance Debug | `F3` |
| Combat Metrics | `F4` |
| Level 100 Boost | `B` |

Keybinds can be changed in the Keybinds window and are saved with the character. Movement remains fixed to the arrow keys.

### Combat And Aggro

- Basic attacks and offensive skills trigger a short action lock to keep attacks readable.
- Fighter, Mage, and Archer attacks use different animation poses and effects.
- Projectiles use animated trails for magic, fire, rune, lightning, and arrows.
- Fire Mage burn damage ticks at fixed intervals and contributes to combat metrics.
- Player attacks alert nearby monsters, so enemies in range can become aggressive even if the player is high level.
- Enemies can route across platforms with jumps, drops, ladders, and ropes.
- The optional Combat Metrics panel shows DPS, money/hour, XP/hour, and estimated time to level.

### Classes, Trials, And Specializations

- Base classes are **Fighter**, **Mage**, and **Archer**.
- Advanced class trials unlock at Level 20 and are started from the Quests/Trials UI or the Class Master flow.
- Advanced jobs become the permanent branch for that character and are meant to be usable from Level 25 onward.
- Level 60 specializations currently grant stat bonuses and class identity direction.

Advanced branches:

- Fighter branches into **Guardian**, **Berserker**, or **Duelist**.
- Mage branches into **Fire Mage**, **Rune Mage**, or **Storm Mage**.
- Archer branches into **Sniper**, **Trapper**, or **Beast Archer**.

Each advanced class has one low-cooldown primary training skill:

| Advanced Class | Primary Skill | Role |
|---|---|---|
| Guardian | Shield Bash | Guarded control and impact generation |
| Berserker | Blood Cleave | Wide melee training and low-HP pressure |
| Duelist | Quick Cut | Fast single-target tempo |
| Fire Mage | Fireball | Explosive fire projectile and burn starter |
| Rune Mage | Rune Mark | Rune setup, slow, and link pressure |
| Storm Mage | Chain Bolt | Multi-target lightning clearing |
| Sniper | Aimed Shot | Precision ranged damage |
| Trapper | Snare Trap | Trap placement and lane control |
| Beast Archer | Companion Strike | Marked companion pressure |

### World, Dungeons, And Map Layouts

- Starfall Crossing contains the Starter Outfitter, Upgrade Artisan, Class Supplier, and Class Master.
- Field maps are connected by physical portals and the World Map route graph.
- Several maps now use long shared training lanes so different groups can occupy different platform bands.
- Dungeons require an advanced class and have boss respawn timers after clears.

Current dungeon progression:

| Dungeon | Level | Bosses |
|---|---:|---|
| Bramble Depths | 25 | Brambleking |
| Emberjaw Lair | 25 | Emberjaw Golem |
| Gearworks Vault | 35 | Clockwork Titan, Quarry Colossus |
| Rimewarden Sanctum | 58 | Rimewarden |

### Inventory, Loot, And Items

- Inventory is split into **Equipment**, **Usable**, and **Etc** tabs.
- Equipment can be double-clicked to equip.
- Usable items can be double-clicked to use.
- Non-usable, non-equipment drops such as Upgrade Dust, Upgrade Catalyst, Warding Scroll, Refinement Core, Prism Shards, Gel Drops, and Ore Chunks live in Etc.
- Stack drops use an in-game quantity prompt instead of the browser prompt.
- Item tooltips show comparisons, level requirements, stat roll tiers, attunement lines, and rarity styling.
- The Monster Guide tracks kills by enemy, shows stats/drop information, and grants mastery bonuses at milestones.

Important usable items:

- **Minor Health Potion** restores HP.
- **Minor MP Tonic** restores MP.
- **Town Return Scroll** returns to Starfall Crossing.
- **Camp Ration** restores both HP and MP.
- **Guard Tonic**, **Swiftstep Oil**, and **Magnet Charm** grant temporary utility buffs.
- **Pet Whistle** unlocks Pet Assist.
- **Equipment/Usable/Etc Section Coupons** expand inventory sections.
- **Base SP Manual**, **Advanced SP Manual**, and **SP Reset Scroll** manage skill points.

### Upgrades, Aides, And Attunement

- Gear upgrades use Upgrade Dust and have Success, Failure, and Destroy outcomes.
- Higher upgrade ranges become riskier, and destroy risk appears at high ranges.
- The upgrade UI shows required materials as current/required counts before confirming.
- Upgrade animation plays in a movable upgrade window.

Upgrade aides:

- **Upgrade Catalyst**: adds +10% success chance to one selected upgrade attempt.
- **Warding Scroll**: turns one destroy result into a failure so the item survives.
- **Refinement Core**: makes a successful upgrade gain +2 instead of +1.

Attunement:

- **Attunement Prism** retunes bonus lines on selected gear.
- **Echo Prism** previews one new roll and then lets the player keep the current attunement or apply the new one.
- **Prism Shards** are generated from prism use and drops.
- 10 Prism Shards combine into 1 Attunement Prism.
- 25 Prism Shards combine into 1 Echo Prism.
- Attunement tiers are Rare, Epic, and Relic.
- Rare has 1 line, Epic has 2 lines, and Relic has 3 lines.
- Attunement lines can roll stats such as power, defense, HP, speed, range, crit, block, area damage, and resource gain.

### Party, Pet, And Account Systems

- Party Finder fills local simulated ally slots for testing party UI, bonuses, and assist timing.
- Simulated allies can attack, shield, or support depending on their generated role.
- Party skills can affect the player and visible simulated allies.
- Party members earn simulated shared XP/drop counters while the player keeps real rewards.
- Pet Assist can auto-use assigned HP/MP potions and apply loot filtering once unlocked.
- Character roster slots, looks, names, roster traits, and per-character save data are implemented.

### Admin And Debug Tools

- Admin settings can change XP and drop rates for prototype testing.
- Performance Debug captures frame timing, update/draw/overlay phases, counts, canvas draw requests, and likely bottlenecks.
- Combat Metrics can be toggled independently from Performance Debug.
- Audio can be toggled and volume is clamped to a safe saved range.

# 1. High-Level Vision

# 1. High-Level Vision

Project Starfall is a **character-first multiplayer progression RPG** built around the satisfaction of choosing a class, leveling that character, branching into a permanent advanced path, collecting gear, upgrading equipment through exciting chance-based systems, and playing with other people in parties.

The game should feel familiar enough that players immediately understand the basics:

- Choose **Fighter**, **Mage**, or **Archer**.
- Level up through quests, maps, dungeons, and bosses.
- Branch into an advanced class at a milestone level.
- Learn dedicated skills and a signature party buff.
- Improve gear through drops, crafting, trading, and risky upgrades.
- Make additional characters to experience other class paths and unlock limited account-wide benefits.

The game should feel distinct through its execution:

- Classes are intuitive in name, but unique in mechanics.
- Each class has a resource that shapes combat flow.
- Branch choices permanently change how that resource works.
- Skills are intentionally tagged for **mobbing**, **bossing**, **hybrid**, **control**, or **support** value.
- Skill batches reveal a group of nodes at once, while rank prerequisites shape the order of investment.
- Party skills benefit both the caster and allies.
- Gear upgrades can create scars, mutations, class-reactive effects, and useful failure outcomes.
- Multiple characters are valuable, but the game avoids making every player feel forced to level every class.

---

# 2. Design Pillars

## 2.1 Intuitive First, Unique Second

Starting classes must be instantly understandable. A new player should not need lore or a guide to understand the basic fantasy of Fighter, Mage, or Archer.

However, advanced classes should become mechanically distinct. A **Guardian** should not simply be “Fighter with more defense.” A Guardian should convert blocked damage into Stored Impact, then release that impact as shields, shockwaves, or counters.

## 2.2 Permanent Class Identity

Characters choose a class path and stay that path. This creates attachment and replayability.

The game can include limited respecs for mistakes, but class advancement should not be casually swappable. If a player wants to experience a different branch, making a new character should be part of the game’s appeal.

## 2.3 Party Skills That Matter

Every advanced class has a signature party skill. These skills should:

- Benefit the caster when solo.
- Benefit nearby party members.
- Make the class desirable in parties.
- Complement certain classes more strongly than others.
- Avoid becoming absolutely required for all content.

Flat buffs are allowed. The key is controlled stacking, cooldowns, categories, and encounter designs that allow multiple valid party compositions.

## 2.4 Risk-Based Gear Progression

Gear upgrades should be exciting because outcomes are uncertain. Success should feel rewarding. Failure should feel tense, but not always like pure wasted progress.

Failures may downgrade, scar, mutate, destabilize, or rarely break items. Even severe failures can create useful materials or alternate build options.

## 2.5 Multiplayer Without Mandatory Scheduling

The game should support party play and social progression, but players should still be able to make meaningful solo progress. Standard multiplayer should use 4-player parties, with larger world bosses or raids added later.

## 2.6 Multiple Characters Without Chore Pressure

Making multiple characters should be beneficial and fun, not mandatory and exhausting.

The account roster system should reward class variety through limited, selectable bonuses rather than uncapped passive stacking.

## 2.7 No Pay-to-Win

Players should never be able to buy combat power, upgrade odds, best-in-slot gear, boss clears, or progression shortcuts that undermine the core RPG loop.

Monetization should focus on cosmetics, account cosmetics, emotes, housing-style cosmetics if added, battle-pass cosmetics, and convenience that does not affect power.

## 2.8 Mobbing and Bossing Identity

Class kits should deliberately support different content strengths.

Some skills should be designed for **mobbing**: broad damage, map clearing, wave control, enemy grouping, and efficient leveling. Other skills should be designed for **bossing**: single-target damage, weak-point exploitation, armor break, burst windows, sustained uptime, and survival during dangerous mechanics.

Every class must be functional in both contexts, but not every class or skill should be equally optimal in both contexts. A Fire Mage can clear enemy packs better than a Sniper, while a Sniper can bring stronger single-target boss pressure. A Trapper can dominate enemy waves through setup and control, while a Berserker can excel during boss burst windows.

This distinction should be visible in the skill UI through simple role tags such as **Mobbing**, **Bossing**, **Hybrid**, **Control**, **Support**, and **Party**.

---

# 3. Target Audience

## 3.1 Primary Audience

Players who enjoy:

- Long-term character progression.
- Class identity and branching class trees.
- Multiplayer party content.
- Gear progression and risky upgrades.
- Bossing, grinding, and visible power growth.
- Making alternate characters to experience different playstyles.

## 3.2 Secondary Audience

Players who enjoy:

- Action RPG combat.
- Buildcrafting.
- Social hubs.
- Trading and item hunting.
- Cosmetic collection.
- Casual but long-term online games.

## 3.3 Player Motivations

| Motivation | Game Response |
|---|---|
| “I want to main one class.” | Permanent class identity, deep branch progression, dedicated skills. |
| “I want to make alts.” | Multiple class paths, roster traits, shared storage, account achievements. |
| “I want to play with friends.” | 4-player parties, party buffs, dungeons, bosses, social towns. |
| “I want exciting gear upgrades.” | Chance-based upgrade system, scars, mutations, item stories. |
| “I want to be useful in parties.” | Every advanced class has a party skill. |
| “I want to chase rare items.” | Boss drops, upgrade materials, class-reactive traits, cosmetics. |
| “I want deep systems but readable classes.” | Simple class names, distinct resource mechanics, limited but meaningful skills. |

---

# 4. Game Format

## 4.1 Genre

2D side-scrolling multiplayer action RPG.

## 4.2 Camera and Perspective

- Side-view 2D action perspective.
- Multi-platform maps with verticality, ladders, jump routes, environmental hazards, and boss arenas.
- Combat should emphasize readable spacing, dodging, skill timing, and party positioning.

## 4.3 Platform

Primary target: **PC**.

Possible future support:

- Mobile companion app for market, cosmetics, character inspection, or offline expedition-like systems if added.
- Steam Deck support if controls are designed cleanly.
- Controller support as a stretch goal.

## 4.4 Session Structure

The game should support several play patterns:

| Session Type | Duration | Activities |
|---|---:|---|
| Quick session | 10–20 minutes | Daily quests, material runs, upgrade attempts, short dungeon. |
| Standard session | 30–60 minutes | Leveling maps, party dungeon, boss practice, trading. |
| Long session | 1–3 hours | Boss progression, farming, multiple characters, events. |

## 4.5 Party Size

- Standard party size: **4 players**.
- Duo content supported where possible.
- Larger content later: 8–12 player world bosses or raids.

## 4.6 PvP

No PvP at launch.

Reason: PvP would create large balance costs and distract from the core progression RPG loop. Casual arena modes can be considered later after PvE identity is stable.

---

# 5. Differentiation Goals

The game can share high-level genre DNA with classic 2D online action RPGs, but it should avoid feeling like a clone.

## 5.1 Keep the Appealing Structure

Keep:

- Permanent class choice.
- Branching class advancement.
- Leveling and skill-batch progression.
- Party buffs.
- Social hubs.
- Gear drops.
- Chance-based upgrades.
- Multiple characters.
- Boss progression.

## 5.2 Avoid Direct Derivative Feel

Avoid:

- Copying specific class kits or job fantasy structures.
- Making classes defined only by flashy attack spam.
- Making gear upgrades only “stat number goes up.”
- Making party buffs mandatory in a rigid meta.
- Creating dozens of shallow classes before a few deep ones.
- Making multiple characters mandatory through uncapped passive stacking.
- Using a chibi visual style that feels too close to existing games.

## 5.3 Distinctive Features

Distinctive elements:

1. **Class resources:** Fighter uses Momentum, Mage uses Energy, Archer uses Focus.
2. **Resource transformation:** Branches alter the base resource into a more specific mechanic.
3. **Self-and-party buffs:** Every advanced class has a buff useful solo and in groups.
4. **Gear scars and mutations:** Failed upgrades can produce alternate item identities.
5. **Class-reactive gear:** Some item traits behave differently depending on class.
6. **Limited roster benefits:** Multiple characters matter, but bonuses are capped and selectable.

---

# 6. World, Tone, and Fantasy

## 6.1 Tone

The tone is **Starlit Frontier Fantasy**: a clean, luminous, side-scrolling MMO fantasy style where frontier guild life meets fallen-star magic.

It should be:

- Colorful.
- Adventurous.
- Slightly whimsical.
- Friendly enough for casual players.
- Deep enough for progression-focused players.
- Not overly grim, but not childish.
- Cozy around towns and guild spaces, but dangerous at the edges of each route.

The world should feel like a practical frontier built around magical materials. Stars, portals, runes, lanterns, crystal inlays, guild banners, and upgrade devices should appear repeatedly as visual anchors.

## 6.2 World Premise

The world is built around frontier regions where magic, monsters, ruins, and settlements overlap. Players are adventurers exploring dangerous lands, clearing hostile areas, recovering materials, defeating bosses, and improving their equipment.

Magic is practical and physical. It can be forged into gear, stored in runes, released as elemental power, or destabilized through risky enhancement.

## 6.3 Setting Goals

The setting should support:

- Gear upgrading as a normal part of the world.
- Distinct regions with monster themes.
- Dungeons and boss arenas.
- Social towns.
- Crafting and trading.
- Seasonal events.
- Bright environments with occasional danger.

## 6.4 Example Regions

| Region | Level Range | Theme | Example Enemies | Main Materials |
|---|---:|---|---|---|
| Greenhollow Fields | 1–20 | beginner forest and farms | slimes, beetles, sproutlings | soft ore, green sap, training charms |
| Ember Quarry | 20–40 | mining zone with fire vents | fire bats, ore golems, furnace imps | emberstone, heated iron |
| Moonwell Marsh | 40–60 | wetlands and old ruins | frogs, wisps, mud spirits | moon reeds, mist crystals |
| Highwind Cliffs | 60–80 | vertical wind zone | harpies, storm bugs, cliff drakes | wind glass, sky feathers |
| Ashen Clockworks | 80–100 | ancient magical machines | clock soldiers, spark drones | gear cores, charged plates |

---

# 7. Core Gameplay Loops

## 7.1 Moment-to-Moment Combat Loop

1. Move through a side-scrolling combat map.
2. Engage enemies using basic attacks and skills.
3. Build class resource: Momentum, Energy, or Focus.
4. Spend resource on stronger skills or finishers.
5. Use movement and defensive skills to avoid damage.
6. Use branch mechanics to create unique combat rhythm.
7. Collect drops, currency, materials, and quest progress.

## 7.2 Session Loop

1. Log in at a town or safe area.
2. Check quests, market, events, and party finder.
3. Choose activity: leveling, dungeon, boss, farming, crafting, upgrading, or alt progression.
4. Run content solo or with a party.
5. Upgrade gear, learn skills, allocate progression rewards.
6. Return to town, trade, socialize, or queue for more content.

## 7.3 Long-Term Loop

1. Level a character through base class and advanced class milestones.
2. Unlock skill batches, invest skill points, improve party buffs, and choose specializations.
3. Farm and upgrade gear.
4. Defeat harder bosses.
5. Build account roster by creating additional characters.
6. Unlock limited roster traits and shared cosmetics.
7. Chase rare upgrade outcomes, class-reactive traits, and prestige cosmetics.

---

# 8. Player Progression Overview

## 8.1 Progression Layers

| Layer | Description |
|---|---|
| Character Level | Main vertical progression; unlocks skill batches, class branches, and content. |
| Class Advancement | Permanent branching at milestone levels. |
| Skill Growth | Skill batches unlock at milestone levels; individual skills are learned and strengthened through earned skill points until the next batch opens. |
| Gear | Equipment stats, upgrade levels, traits, scars, mutations. |
| Account Roster | Limited account-wide benefits from multiple characters. |
| Collection | Cosmetics, mounts, titles, emotes, badges, visual effects. |
| Boss Progression | Access and mastery of increasingly difficult bosses. |

## 8.2 Level Milestones and Skill Batches

Project Starfall should not fully unlock individual skills at scattered level breakpoints. Instead, major levels unlock a **batch of available skills**, and the player earns skill points throughout the level range to learn, rank up, and customize those skills.

| Level | Milestone | Skill Batch Opened |
|---:|---|---|
| 1 | Choose starting class: Fighter, Mage, or Archer. | **Base Skill Batch** opens. Base combat skills become visible; core starter nodes are trainable immediately and later nodes use rank prerequisites. |
| 20 | Class trial becomes available. | No new full batch; late base levels are used to finish or diversify Base Skill Batch investment. |
| 25 | Choose advanced class. | **Advanced Skill Batch** opens. Dedicated advanced skills and the party skill become visible; core branch nodes are trainable immediately and later nodes use rank prerequisites. |
| 50 | Advanced class capstone period begins. | No separate skill unlock; players should be finishing key Advanced Skill Batch ranks and experimenting with alternate investment. |
| 60 | Choose specialization. | **Specialization Skill Batch** opens. Specialization skills and modifiers become visible; later specialization nodes can use rank prerequisites. |
| 100 | Unlock mastery system. | **Mastery Skill Batch** opens. Mastery nodes, skill modifiers, and late-game build refinements become visible and are gated by mastery prerequisites. |

## 8.3 Skill Batch Progression Model

A **Skill Batch** is a group of skills unlocked together at a milestone level. The batch gives players access to a set of skills, but those skills are not automatically completed. Players must spend earned skill points to unlock ranks, improve effects, reduce cooldowns, add utility, or strengthen party benefits.

### Core Rules

- Skills become **available in batches**, not one-by-one at fixed levels.
- A skill becomes usable at **Rank 1**.
- Higher ranks improve the skill through damage, cooldown, resource generation, range, duration, secondary effects, or party-skill scaling.
- Players earn skill points regularly while leveling through the current batch range.
- The player should not be able to max every skill in a batch before the next batch opens.
- Each batch should support at least two reasonable investment styles: focused specialization or broad utility.
- Respeccing within the current batch should be cheap during leveling so players can experiment.

### Skill Availability vs. Skill Trainability

When a batch opens, the player should be able to **see the whole batch immediately**. This preserves the desired feeling of unlocking a complete group of skills at a milestone level.

However, not every visible node needs to be immediately trainable. Individual skills can require investment in related skills before they can receive points. This creates natural progression inside the batch without reverting to level-by-level skill unlocks.

Definitions:

| Term | Meaning |
|---|---|
| Visible | The skill appears in the batch UI and can be previewed. |
| Trainable | The player can spend points into the skill because prerequisites are met. |
| Usable | The skill has reached Rank 1 and can be used in combat. |
| Ranked | The skill has additional investment beyond Rank 1. |

Example:

```text
Level 25: Fire Mage Advanced Skill Batch opens.
All Fire Mage skills are visible.
Fireball is trainable immediately.
Wildfire is visible, but requires Burning Mark Rank 5.
Inferno Burst is visible, but requires Heat Vent Rank 5 and Energy Release Rank 10 from the Base Skill Batch.
```

### Batch Ranges

| Batch | Level Range | Purpose |
|---|---:|---|
| Base Skill Batch | 1–24 | Teach the base class resource, core combat, mobility, defense, and first finisher. |
| Advanced Skill Batch | 25–59 | Define the advanced class, resource twist, party skill, and main combat loop. |
| Specialization Skill Batch | 60–99 | Deepen the class fantasy with build-defining specializations. |
| Mastery Skill Batch | 100+ | Add late-game refinement, skill modifiers, prestige progression, and difficult build choices. |

### Point Budget Philosophy

By the end of a batch range, an active character should be able to:

- Max a small number of favorite skills.
- Keep several utility skills at medium rank.
- Leave some skills lightly invested or untouched.

This creates build identity without forcing permanent mistakes. A Guardian might heavily invest in **Impact Guard**, **Shield Wall**, and **Retaliation Wave**, while another Guardian might prioritize **Oath Barrier**, **Hold the Line**, and party protection.

### Prototype Skill Point Economy

For the first prototype, use a simple point model that is easy to tune:

| Batch | Possible Skill Ranks | Suggested Points Earned Before Next Batch | Design Result |
|---|---:|---:|---|
| Base Skill Batch | ~60 total ranks | ~28–34 points | Player can max 2–3 skills or spread points across the whole base kit. |
| Advanced Skill Batch | ~70 total ranks | ~38–46 points | Player chooses between combat focus, party-skill focus, or balanced investment. |
| Specialization Skill Batch | TBD | TBD | Tuning depends on the number of specialization nodes. |
| Mastery Skill Batch | TBD | Ongoing | Long-term horizontal and prestige growth. |

Prototype assumptions:

- Most ranks cost 1 skill point.
- Major breakpoint ranks may cost 2 points if they add a powerful effect.
- Class trials, milestone quests, and first boss clears can award bonus skill points.
- Skill points should be batch-specific at first for easier balance: Base Points, Advanced Points, Specialization Points, and Mastery Points.
- Later testing can determine whether a unified skill point pool feels better.

### Skill Rank Structure

Prototype rank structure:

| Rank Range | Meaning |
|---:|---|
| 0 | Available but not learned. |
| 1 | Skill becomes usable. |
| 2–4 | Core improvements: damage, duration, resource gain, cooldown, or reliability. |
| 5 | Minor breakpoint: adds a small secondary effect. |
| 6–9 | Scaling improvements for committed investment. |
| 10 | Major breakpoint: adds a signature modifier, stronger party component, or build-defining bonus. |

Not every skill needs exactly 10 ranks in the final design, but the prototype should use a consistent structure for testing.

### Skill Prerequisite Rules

Skill prerequisites are allowed and encouraged, as long as they serve readable build progression rather than hiding skills behind arbitrary level gates.

Prototype prerequisite types:

| Prerequisite Type | Example | Design Purpose |
|---|---|---|
| Same-batch rank requirement | **Burning Mark Rank 5** required for **Wildfire** | Teaches the player to invest in the setup skill before the payoff skill. |
| Prior-batch max requirement | **Energy Release Rank 10** required for **Inferno Burst** | Rewards commitment to a core base skill and ties advanced finishers to base identity. |
| Either/or requirement | **Ground Glyph Rank 5** or **Arcane Link Rank 5** required for **Rune Detonation** | Supports multiple build routes inside the same class. |
| Party-skill breakpoint requirement | **Shield Bash Rank 5** improves **Shield Wall** at Rank 5 | Encourages party-focused builds without making the party skill useless at Rank 1. |
| Specialization gateway | **One Perfect Shot Rank 10** required for a later Deadeye mastery node | Creates long-term build identity. |

Rules:

- Skill prerequisites should use **skill ranks**, not extra character-level gates, once the batch has opened.
- Rank 1 of each class's simplest core skill should be trainable immediately.
- Most utility and party skills should be trainable early, but their stronger rank breakpoints may require related skill investment.
- Finishers, major area skills, and major bossing skills can require Rank 5 or Rank 10 in earlier skills.
- At least two valid routes should exist in every batch: one leaning toward mobbing/leveling and one leaning toward bossing/single-target play.
- Prior-batch max requirements should be used sparingly so players do not feel forced to max every base skill.
- The UI must show locked nodes, missing prerequisites, and suggested paths clearly.

### Party Skill Investment

Party skills are part of the Advanced Skill Batch. They should not be automatically fully unlocked at a specific level.

At Rank 1, the party skill becomes usable with a modest effect. Higher ranks increase its self benefit, party benefit, duration, range, cooldown efficiency, or synergy effect. This lets players choose how much they want to lean into party support while still making the skill useful for solo play.

## 8.4 Advancement Philosophy

Advancement should be a meaningful moment. The player should complete a short trial that teaches the playstyle of the branch.

Examples:

- Guardian trial: survive waves while protecting an NPC.
- Berserker trial: defeat enemies while health is limited.
- Fire Mage trial: manage Heat without overheating.
- Rune Mage trial: place and detonate runes in sequence.
- Sniper trial: defeat targets using weak-point marks.
- Trapper trial: control enemy paths with traps.

---

# 9. Class System Overview

## 9.1 Starting Classes

The game launches with three starting classes:

```text
Fighter
Mage
Archer
```

These names are intentionally simple and readable.

## 9.2 Advanced Classes

At Level 25, each starting class branches into three advanced classes:

```text
Fighter
├── Guardian
├── Berserker
└── Duelist

Mage
├── Fire Mage
├── Rune Mage
└── Storm Mage

Archer
├── Sniper
├── Trapper
└── Beast Archer
```

## 9.3 Current Branch Scope

The current browser prototype implements all nine advanced classes. The original six-branch MVP scope has been superseded by a broader exploration build, so the next milestone should focus on polish, balance, and onboarding rather than adding more class branches.

Current branch spread:

| Role | Class |
|---|---|
| Defense | Guardian |
| Risky melee damage | Berserker |
| Precision melee | Duelist |
| Area damage | Fire Mage |
| Setup magic | Rune Mage |
| Chain mobbing | Storm Mage |
| Boss precision | Sniper |
| Control | Trapper |
| Companion sustain | Beast Archer |

## 9.4 Base Class Resources

| Class | Resource | How It Builds | How It Spends |
|---|---|---|---|
| Fighter | Momentum | Attacking, blocking, staying in combat | Shockwaves, finishers, empowered strikes |
| Mage | Energy | Casting, marking, spell combos | Burst spells, shields, empowered magic |
| Archer | Focus | Positioning, marking, avoiding damage | Precision shots, traps, volleys |

## 9.5 Resource Transformation by Branch

| Advanced Class | Base Resource | Branch Resource / Twist |
|---|---|---|
| Guardian | Momentum | Stored Impact from blocked damage |
| Berserker | Momentum | Rage from damage dealt/taken and low HP |
| Duelist | Momentum | Combo Momentum from timed attacks and counters |
| Fire Mage | Energy | Heat from fire spells and Overheat risk |
| Rune Mage | Energy | Runic Energy used for placed runes |
| Storm Mage | Energy | Charge from fast casting and movement |
| Sniper | Focus | Aim from long-range hits and steady positioning |
| Trapper | Focus | Preparation from trap placement and triggers |
| Beast Archer | Focus | Bond from coordinated companion attacks |

## 9.6 Mobbing vs. Bossing Role Tags

Each skill should have a role tag. These tags help players understand whether a skill is mainly for clearing regular enemies, damaging bosses, controlling the map, supporting allies, or filling multiple roles.

| Role Tag | Meaning |
|---|---|
| **Mobbing** | Best for hitting multiple enemies, clearing maps, farming materials, and wave content. |
| **Bossing** | Best for single-target damage, burst windows, weak points, armor break, or sustained boss uptime. |
| **Hybrid** | Useful in both mobbing and bossing, usually with moderate efficiency in each. |
| **Control** | Slows, roots, pulls, stuns, zones, or redirects enemies. |
| **Support** | Shields, healing, mitigation, cleanse, resource generation, or party utility. |
| **Party** | Explicitly benefits nearby allies in addition to the caster. |

Advanced class orientation:

| Advanced Class | Primary Content Strength | Secondary Strength | Notes |
|---|---|---|---|
| Guardian | Boss survival / party safety | Hybrid mob control | Strong for dangerous boss mechanics and protecting fragile allies. |
| Berserker | Boss burst / risky sustain | Mobbing through wide cleaves | High damage windows, especially when health is low. |
| Duelist | Boss uptime / precision | Fast small-pack clearing | Rewards timing and constant contact with priority targets. |
| Fire Mage | Mobbing / area damage | Burst against burnable bosses | Excellent wave clear; boss value depends on burn uptime and Heat control. |
| Rune Mage | Hybrid setup | Control / flexible support | Can build toward rune fields for mobbing or detonations/seals for bossing. |
| Storm Mage | Mobbing / mobility | Multi-hit boss pressure | Fast clear and chain damage; bossing improves through Charge management. |
| Sniper | Bossing / crit burst | Line-clearing with precision shots | Best single-target identity among Archer branches. |
| Trapper | Mobbing / control | Positional boss utility | Strong when enemies can be slowed, routed, or kept inside fields. |
| Beast Archer | Hybrid sustained damage | Sustain / coordinated party play | Companion creates steady boss damage and useful solo consistency. |

---

# 10. Base Classes

# 10.1 Fighter

## Identity

Fighter is the close-range weapon class. It is sturdy, physical, direct, and easy to learn.

Fighter should feel satisfying through impact, momentum, and controlled aggression.

## Resource: Momentum

Fighters gain Momentum by:

- Landing basic attacks.
- Using melee skills.
- Blocking or parrying attacks.
- Staying near enemies.

Momentum is spent on stronger attacks, finishers, shockwaves, and branch-specific effects.

## Base Fighter Skill Batch

All Fighter base skills become available as trainable nodes in the **Base Skill Batch** at Level 1. The tutorial can recommend an early path, but the skills are improved through points rather than fully unlocked at fixed levels.

| Skill | Type | Combat Role | Prerequisite | Description |
|---|---|---|---|---|
| Heavy Strike | Basic attack | Hybrid | None | A close-range weapon attack. Consecutive hits build Momentum. Rank investment improves Momentum gain and combo reliability. |
| Dash Slash | Mobility / damage | Hybrid | Heavy Strike Rank 3 | Dash forward and strike the first enemy hit. Rank investment improves range, cooldown, and Momentum generation on hit. |
| Guard | Defense | Support / Bossing | Heavy Strike Rank 3 | Briefly reduce incoming damage. Rank investment improves reduction, timing window, and bonus Momentum from timed guards. |
| Ground Slam | Area damage | Mobbing | Heavy Strike Rank 5 | Slam the ground, damaging nearby enemies. Rank investment improves shockwave size, damage, and Momentum spending efficiency. |
| Power Break | Debuff | Bossing | Ground Slam Rank 3 or Heavy Strike Rank 5 | Heavy hit that lowers enemy defense briefly. Rank investment improves debuff duration and resource cost. |
| Momentum Burst | Finisher | Hybrid | Heavy Strike Rank 5 plus any Fighter spender Rank 3 | Spend Momentum for a powerful attack. Rank investment improves damage scaling and adds finisher utility. |

## Beginner Skill Flow

This is the recommended tutorial flow, not a level-gated unlock order.

1. Use Heavy Strike to build Momentum.
2. Use Dash Slash to engage or reposition.
3. Use Guard against dangerous attacks.
4. Use Ground Slam or Momentum Burst to spend Momentum.

---

# 10.2 Mage

## Identity

Mage is the ranged spellcasting class. It is more fragile than Fighter, but has strong area damage, utility, and burst potential.

Mage should feel powerful but resource-aware.

## Resource: Energy

Mages gain Energy by:

- Casting Magic Bolt.
- Marking enemies.
- Landing spell combos.
- Using branch-specific spell interactions.

Energy is spent on stronger spells, shields, and finishers.

## Base Mage Skill Batch

All Mage base skills become available as trainable nodes in the **Base Skill Batch** at Level 1. The player learns the full shape of the class early, then decides which spells to strengthen while leveling toward the advanced class choice.

| Skill | Type | Combat Role | Prerequisite | Description |
|---|---|---|---|---|
| Magic Bolt | Basic attack | Hybrid | None | Simple ranged magic projectile. Rank investment improves Energy gain, projectile speed, and reliability. |
| Blink | Mobility | Support | Magic Bolt Rank 3 | Short teleport. Rank investment improves cooldown, distance, and branch-compatible effects. |
| Arcane Burst | Area damage | Mobbing | Magic Bolt Rank 5 | Small explosion at target area. Rank investment improves radius, damage, and Energy spending efficiency. |
| Mana Shield | Defense | Support / Bossing | Magic Bolt Rank 3 | Absorb some damage using Energy. Rank investment improves absorption, Energy cost, and emergency survival value. |
| Spell Mark | Setup | Bossing | Magic Bolt Rank 5 | Mark an enemy so the next spell has an added effect. Rank investment improves mark duration and bonus effect strength. |
| Energy Release | Finisher | Hybrid | Arcane Burst Rank 5 or Spell Mark Rank 5 | Spend Energy to fire a stronger projectile or explosion. Rank investment improves scaling and finisher utility. |

## Beginner Skill Flow

This is the recommended tutorial flow, not a level-gated unlock order.

1. Use Magic Bolt to build Energy.
2. Use Spell Mark to prepare a target.
3. Use Arcane Burst or Energy Release for damage.
4. Use Blink and Mana Shield to survive.

---

# 10.3 Archer

## Identity

Archer is the ranged physical class. It rewards spacing, marking, mobility, and target selection.

Archer should feel precise and tactical rather than simply “ranged Fighter.”

## Resource: Focus

Archers gain Focus by:

- Hitting enemies from range.
- Applying and attacking marks.
- Avoiding damage.
- Using positioning skills well.

Focus is spent on strong shots, volleys, traps, and branch-specific effects.

## Base Archer Skill Batch

All Archer base skills become available as trainable nodes in the **Base Skill Batch** at Level 1. The player can lean into mobility, marks, precision, or Focus spending before choosing an advanced class.

| Skill | Type | Combat Role | Prerequisite | Description |
|---|---|---|---|---|
| Quick Shot | Basic attack | Hybrid | None | Fast ranged attack. Rank investment improves Focus gain, attack consistency, and mark interaction. |
| Roll Shot | Mobility / damage | Hybrid | Quick Shot Rank 3 | Roll backward or sideways and fire an arrow. Rank investment improves cooldown, distance, and Focus generation after dodging. |
| Marked Shot | Setup | Bossing | Quick Shot Rank 3 | Mark an enemy. Marked enemies take bonus damage from Focus spenders. Rank investment improves mark duration and payoff. |
| Piercing Arrow | Damage | Mobbing | Quick Shot Rank 5 | Fire an arrow that hits enemies in a line. Rank investment improves range, damage, and Focus efficiency. |
| Eagle Stance | Buff | Bossing | Marked Shot Rank 3 or Piercing Arrow Rank 3 | Briefly increase range and critical chance. Rank investment improves duration, crit scaling, and uptime. |
| Focused Volley | Finisher | Hybrid | Marked Shot Rank 5 or Piercing Arrow Rank 5 | Spend Focus to fire several arrows at marked targets. Rank investment improves scaling and target priority. |

## Beginner Skill Flow

This is the recommended tutorial flow, not a level-gated unlock order.

1. Use Quick Shot to build Focus.
2. Use Marked Shot on priority enemies.
3. Use Piercing Arrow or Focused Volley to spend Focus.
4. Use Roll Shot to maintain spacing.

---

# 11. Advanced Classes and Skills

This section reflects the current Level 25 Advanced Skill Batch implementation. Choosing an advanced class opens its visible batch and grants the no-prerequisite starter ranks for that branch. The current prototype includes all nine advanced branches, not only the original six-class MVP subset.

Current advanced class rules:

- Advanced choice is permanent in the prototype save.
- Advanced classes unlock at Level 25 after the matching class trial.
- Each advanced class has one immediate **primary training attack** with cooldown below 1 second.
- Other advanced skills serve as mobility, setup, burst, sustain, defense, passive, field, finisher, or party/self-buff skills.
- Party skills currently function as solo self-buffs and document future party effects in their tooltips.
- Role tags and class role profiles are part of the skill UI and balance language.

## 11.1 Advanced Class Summary

| Base | Advanced Class | Resource | Current Role | Primary Training Skill |
|---|---|---|---|---|
| Fighter | Guardian | Stored Impact | Support / Control | Shield Bash |
| Fighter | Berserker | Rage | Bossing | Blood Cleave |
| Fighter | Duelist | Tempo | Bossing | Quick Cut |
| Mage | Fire Mage | Heat | Mobbing | Fireball |
| Mage | Rune Mage | Runic Energy | Support / Control | Rune Mark |
| Mage | Storm Mage | Charge | Mobbing | Chain Bolt |
| Archer | Sniper | Aim | Bossing | Aimed Shot |
| Archer | Trapper | Preparation | Mobbing / Control | Snare Trap |
| Archer | Beast Archer | Bond | Support / Hybrid | Companion Strike |

## 11.2 Guardian

Guardian is a defensive Fighter branch that converts blocks and guarded hits into Stored Impact. The current prototype emphasizes control, armor cracking, shields, and counter-burst rather than passive tanking.

| Skill | Current Purpose |
|---|---|
| Shield Bash | Primary training attack. Low-cooldown guarded strike that staggers, slows, cracks, and builds Stored Impact. |
| Shield Dash | Mobility/control dash that claims space behind a shield. |
| Impact Guard | Defensive self-buff that grants shield, invulnerability window, and Stored Impact. |
| Oath Barrier | Converts Stored Impact into a larger shield. |
| Retaliation Wave | Forward shockwave that spends or builds defensive pressure. |
| Hold the Line | Passive defense/block/resource identity. |
| Guardian's Verdict | Stored Impact finisher with defensive explosion and shield payoff. |
| Shield Wall | Party skill prototype; currently a solo shield/damage-reduction self-buff with future ally protection. |

Future Guardian work should add clearer blocked-hit timing, boss-specific guard windows, and UI segments for Stored Impact.

## 11.3 Berserker

Berserker is a risky Fighter branch focused on bossing, low-HP damage scaling, cleaves, recovery, and brief survival windows.

| Skill | Current Purpose |
|---|---|
| Blood Cleave | Primary training attack. Low-cooldown wide cleave that gets stronger as HP drops. |
| Rage Surge | Self-buff mapped to War Cry-style power behavior. |
| Reckless Leap | Aggressive mobility skill for gap crossing and engagement. |
| Crimson Recovery | Bossing sustain hit that heals from damage dealt. |
| Pain to Power | Passive power/resource identity. |
| Last Stand | High-damage low-HP bossing finisher with a short invulnerability window. |
| War Cry | Party skill prototype; currently a solo offensive self-buff with future ally damage/resource support. |

Future Berserker work should add clearer danger thresholds, better Rage UI, and stronger differentiation between safe training and risky boss burst.

## 11.4 Duelist

Duelist is a precision Fighter branch that rewards repeatedly hitting the same target. The current prototype uses Tempo as a same-target combo mechanic.

| Skill | Current Purpose |
|---|---|
| Quick Cut | Primary training attack. Low-cooldown multi-line melee hit that builds Tempo on the same target. |
| Flash Step | Fast mobility dash that preserves Tempo flow. |
| Rallying Flourish | Party skill prototype; currently a solo haste/crit/Tempo self-buff with future party haste. |

Future Duelist work should expand the branch beyond the current compact MVP implementation with parry/riposte timing, mark payoffs, and a true finisher.

## 11.5 Fire Mage

Fire Mage is the Mage mobbing branch. It uses Heat and burn spread to make clustered enemies feel meaningfully different from single-target fights.

| Skill | Current Purpose |
|---|---|
| Fireball | Primary training attack. Low-cooldown explosive projectile that applies burn. |
| Flame Trail | Mobility effect that leaves a damaging fire field. |
| Burning Mark | Fire projectile that marks and burns, enabling spread. |
| Heat Vent | Cone-like role resolver that vents Heat into area fire damage. |
| Wildfire | Cluster burn-spread skill for mobbing. |
| Inferno Burst | Larger burn-scaling explosion and Heat dump. |
| Ignition Aura | Party skill prototype; currently a solo fire/burn self-buff with future ally burn synergy. |

Future Fire Mage work should add a more explicit Overheat state, clearer Heat UI, enemy fire resistance, and safer tutorial messaging for self-risk mechanics.

## 11.6 Rune Mage

Rune Mage is the Mage setup/control branch. It links enemies, creates rune fields, and detonates prepared rune states.

| Skill | Current Purpose |
|---|---|
| Rune Mark | Primary training attack. Low-cooldown rune projectile that marks, slows, and links. |
| Rune Blink | Advanced teleport/mobility option. |
| Ground Glyph | Control field that slows and marks enemies. |
| Arcane Link | Links two nearby enemies and shows a visual chain. |
| Rune Detonation | Bursts linked or marked enemies and clears rune state. |
| Mana Seal | Control projectile with slow/mark behavior. |
| Grand Inscription | Large rune field and support/control finisher. |
| Rune Circle | Party skill prototype; currently a solo resource/support self-buff with future selectable party rune modes. |

Future Rune Mage work should add visible active rune counters, selectable Rune Circle modes, and more deliberate setup/payoff tutorials.

## 11.7 Storm Mage

Storm Mage is the Mage mobbing branch built around Chain Bolt. Chain Bolt is intentionally strong for clustered enemy clearing and intentionally weaker as pure boss damage.

| Skill | Current Purpose |
|---|---|
| Chain Bolt | Primary training attack. Low-cooldown delayed lightning chain that jumps between nearby enemies. |
| Static Shift | Advanced teleport/mobility option. |
| Stormfront | Party skill prototype; currently a solo Charge/area-damage self-buff with future ally haste/lightning support. |

Future Storm Mage work should add more non-chain utility and bossing alternatives so the class remains fun when enemies are spread out or boss-only.

## 11.8 Sniper

Sniper is the Archer bossing branch. It uses weak-point states, armor pierce, long-range shots, and execution-style payoffs.

| Skill | Current Purpose |
|---|---|
| Aimed Shot | Primary training attack. Low-cooldown precision projectile with bossing identity. |
| Combat Roll | Mobility repositioning skill. |
| Weak Point Mark | Projectile that applies weak-point state. |
| Steady Breath | Passive range/crit/resource identity. |
| Pierce Armor | Piercing debuff shot that cracks enemy defense. |
| Execution Shot | High-value weak-point or low-HP payoff. |
| One Perfect Shot | Single-target ultimate-style payoff. |
| Eagle Eye | Party skill prototype; currently a solo crit/range self-buff with future ally crit/weak-point payoff. |

Future Sniper work should add aim-state feedback, boss weak-point telegraphs, and better jump-attack/bossing tuning.

## 11.9 Trapper

Trapper is the Archer mobbing/control branch. It places active skill objects that arm, trigger, slow, damage, and can be detonated manually.

| Skill | Current Purpose |
|---|---|
| Snare Trap | Primary training attack. Low-cooldown trap placement and control. |
| Grapple Dash | Mobility skill for positioning around trap zones. |
| Spike Trap | Higher-damage trap. |
| Lure Shot | Projectile that marks/slows and supports trap routing. |
| Tripwire | Multi-charge trap object. |
| Detonate | Manual trigger for active traps. |
| Kill Zone | Large multi-charge trap field. |
| Tactical Field | Party skill prototype; currently a solo trap/control self-buff with future ally safety/damage support. |

Future Trapper work should improve trap visibility, enemy lure behavior, boss immunity rules, and map layouts that reward preparation without slowing training too much.

## 11.10 Beast Archer

Beast Archer is the Archer support/hybrid branch. The current prototype represents the companion through coordinated strike visuals, pack marks, and sustain behavior rather than a separate pathing pet entity.

| Skill | Current Purpose |
|---|---|
| Companion Strike | Primary training attack. Low-cooldown coordinated shot that marks enemies for pack pressure. |
| Pounce Roll | Mobility skill with companion-themed retreat/repositioning. |
| Pack Call | Party skill prototype; currently a solo sustain/resource self-buff with future ally sustain and resource recovery. |

Future Beast Archer work should add a true companion actor, companion positioning, pet-safe boss mechanics, and more active Bond spenders.

---

# 12. Party Skills and Buff Design

## 12.1 Goals

Party skills should create multiplayer identity without making any one class mandatory.

Each party skill must:

1. Be useful to the caster while solo.
2. Provide a meaningful party effect.
3. Have clear synergies with some classes.
4. Have limitations through duration, cooldown, range, stacking rules, or encounter dependency.

## 12.2 Buff Categories

Buffs are divided into categories to prevent excessive stacking.

```text
Attack Buff
Critical Buff
Defense Buff
Speed Buff
Sustain Buff
Field Buff
Enemy Debuff
Resource Buff
```

## 12.3 Stacking Rules

Prototype stacking rules:

- Only the strongest **Attack Buff** applies at full value.
- Only the strongest **Critical Buff** applies at full value.
- Defense buffs stack partially with diminishing returns.
- Speed buffs do not stack fully; highest applies, second gives 25% value.
- Healing-on-hit effects share a per-second cap.
- Field buffs can overlap visually, but only one damage amplification field applies at full value.
- Enemy debuffs stack up to a cap.
- Resource buffs stack only from different categories or with reduced value.

## 12.4 Party Skill Rank Gates

Party skills should be useful at Rank 1, but their stronger solo and party effects can use prerequisites. This keeps party skills valuable while still making investment choices meaningful.

Prototype party-skill rank structure:

| Party Skill Rank | Requirement Pattern | Result |
|---:|---|---|
| 1 | Advanced class chosen | Skill becomes usable with modest self and party value. |
| 5 | One related core advanced skill at Rank 5 | Adds stronger self scaling, longer duration, or improved range. |
| 10 | Signature advanced skill at Rank 5 plus one relevant prior-batch skill at Rank 10 | Adds the major party breakpoint or strongest synergy effect. |

Examples:

| Party Skill | Rank 5 Requirement | Rank 10 Requirement | Major Breakpoint |
|---|---|---|---|
| Shield Wall | Impact Guard Rank 5 | Oath Barrier Rank 5 and Guard Rank 10 | Larger ally shield and stronger anti-knockback. |
| War Cry | Rage Surge Rank 5 | Pain to Power Rank 5 and Momentum Burst Rank 10 | Stronger party attack buff during burst windows. |
| Eagle Eye | Weak Point Mark Rank 5 | Execution Shot Rank 5 and Focused Volley Rank 10 | Higher crit damage against marked weak points. |
| Tactical Field | Snare Trap Rank 5 | Kill Zone Rank 5 and Piercing Arrow Rank 10 | Better field control and bonus damage to trapped enemies. |

## 12.5 Party Skill Summary

| Advanced Class | Party Skill | Self Benefit | Party Benefit | Main Role |
|---|---|---|---|---|
| Guardian | Shield Wall | strong shield / damage reduction | smaller shield / damage reduction | survival |
| Berserker | War Cry | attack power, Rage, lifesteal | smaller attack/resource/lifesteal buff | burst offense |
| Duelist | Rallying Flourish | movement speed, crit, Tempo gain | smaller haste/crit/resource buff | rhythm DPS |
| Fire Mage | Ignition Aura | fire damage, Heat, burn power | bonus damage vs burning enemies | area damage |
| Rune Mage | Rune Circle | full selected rune effect | reduced selected rune effect | flexible support |
| Storm Mage | Stormfront | Charge, area damage, chain reach | smaller skill haste/shocked-target damage buff | mobility / mobbing |
| Sniper | Eagle Eye | crit/range/Aim | smaller crit/weak-point buff | boss damage |
| Trapper | Tactical Field | trap power/safety/Focus | damage reduction/control bonus | control |
| Beast Archer | Pack Call | Bond, max HP, MP recovery | minor sustain/resource recovery | sustain |

## 12.6 Party Composition Examples

### Balanced Beginner Party

```text
Guardian + Fire Mage + Sniper + Trapper
```

- Guardian provides survival.
- Fire Mage provides area damage.
- Sniper provides boss damage and crit.
- Trapper provides control and safe zones.

### Burst Damage Party

```text
Berserker + Duelist + Fire Mage + Sniper
```

- Berserker provides War Cry.
- Duelist provides Rallying Flourish.
- Sniper provides Eagle Eye.
- Fire Mage uses Ignition Aura and burst spells.

This party has high damage but lower safety.

### Control Party

```text
Guardian + Rune Mage + Trapper + Fire Mage
```

- Trapper slows and groups enemies.
- Rune Mage sets Rune Circle.
- Fire Mage burns clustered enemies.
- Guardian keeps the group safe.

### Mobility Party

```text
Duelist + Storm Mage + Berserker + Archer branch
```

- Storm Mage provides Stormfront.
- Duelist maintains combo flow.
- Berserker stays attached to mobile bosses.
- Archer branch benefits from repositioning.

---

# 13. Specializations

At Level 60, each advanced class chooses a specialization. Specializations deepen identity without replacing the advanced class fantasy.

## 13.1 Specialization Tree

```text
Fighter
├── Guardian
│   ├── Shield Knight
│   └── Ironbreaker
├── Berserker
│   ├── Blood Warrior
│   └── Warbringer
└── Duelist
    ├── Sword Dancer
    └── Counterblade

Mage
├── Fire Mage
│   ├── Pyromancer
│   └── Ashcaller
├── Rune Mage
│   ├── Rune Scholar
│   └── Sealbinder
└── Storm Mage
    ├── Lightning Mage
    └── Windcaller

Archer
├── Sniper
│   ├── Deadeye
│   └── Longshot
├── Trapper
│   ├── Bomb Hunter
│   └── Snare Hunter
└── Beast Archer
    ├── Beastmaster
    └── Spirit Hunter
```

## 13.2 Fighter Specializations

| Advanced Class | Specialization | Direction |
|---|---|---|
| Guardian | Shield Knight | More protection, party shielding, boss survival. |
| Guardian | Ironbreaker | Defensive bruiser, armor break, impact damage. |
| Berserker | Blood Warrior | Lifesteal, low-HP scaling, risky sustain. |
| Berserker | Warbringer | Bigger burst windows, group aggression, rage explosions. |
| Duelist | Sword Dancer | Speed, combo chains, multi-target mobility. |
| Duelist | Counterblade | Parry, counterattacks, single-target precision. |

## 13.3 Mage Specializations

| Advanced Class | Specialization | Direction |
|---|---|---|
| Fire Mage | Pyromancer | Explosions, direct fire damage, Heat burst. |
| Fire Mage | Ashcaller | Burn spread, smoke, lingering damage, safer Heat control. |
| Rune Mage | Rune Scholar | More rune combinations, combo payoff, battlefield setup. |
| Rune Mage | Sealbinder | Control, debuffs, anti-boss utility, curse cleansing. |
| Storm Mage | Lightning Mage | Chain lightning, fast hits, burst through Charge. |
| Storm Mage | Windcaller | Mobility, evasion, party movement, sustained casting. |

## 13.4 Archer Specializations

| Advanced Class | Specialization | Direction |
|---|---|---|
| Sniper | Deadeye | Crit damage, weak points, boss execution. |
| Sniper | Longshot | Range, charged shots, safe positioning. |
| Trapper | Bomb Hunter | Explosive traps, area burst, combo detonation. |
| Trapper | Snare Hunter | Control, roots, slows, defensive utility. |
| Beast Archer | Beastmaster | Physical companion damage and coordination. |
| Beast Archer | Spirit Hunter | Spirit companion, utility, healing, status effects. |

---

# 14. Multiple Character / Account Roster System

The game should support multiple characters without turning alt-leveling into a required chore.

## 14.1 Goals

- Encourage players to try multiple classes.
- Let alternate characters provide limited account value.
- Avoid forcing players to level every branch.
- Keep the player’s main character meaningful.

## 14.2 Shared Account Features

| Feature | Description |
|---|---|
| Shared Storage | Materials, currencies, and eligible gear can be moved between characters. |
| Shared Cosmetics | Unlocked cosmetics are account-wide unless intentionally character-specific. |
| Account Achievements | Milestones reward cosmetics, titles, and small roster benefits. |
| Roster Traits | Each character can unlock a limited selectable trait for the account. |
| Gear Inheritance | Some gear can become account-bound and passed to alts. |
| Catch-Up Boosts | Lower-level alts receive mild leveling help after the account reaches milestones. |

## 14.3 Roster Traits

At certain milestones, each character unlocks a Roster Trait.

Prototype milestones:

| Character Level | Roster Trait Unlock |
|---:|---|
| 50 | Basic class trait |
| 100 | Advanced class trait |
| 150+ | Cosmetic title / prestige trait |

Only a limited number of Roster Traits can be equipped at once.

Prototype rule:

```text
Account may equip 3 Roster Traits at first.
Additional slots unlock slowly through account progression, up to a hard cap of 5.
```

## 14.4 Example Roster Traits

| Class | Roster Trait | Effect |
|---|---|---|
| Guardian | Steady Guard | Slightly reduced knockback and small defense bonus. |
| Berserker | Battle Hunger | Slightly increased damage against low-health enemies. |
| Duelist | Quick Reflex | Small dodge recovery improvement. |
| Fire Mage | Ember Touch | Small bonus damage against burning enemies. |
| Rune Mage | Focused Study | Slightly increased class-resource generation. |
| Storm Mage | Light Step | Small movement speed bonus. |
| Sniper | Sharp Aim | Small critical chance bonus. |
| Trapper | Careful Prep | Small bonus against slowed/rooted enemies. |
| Beast Archer | Shared Instinct | Small heal-on-hit cap increase or companion utility bonus. |

## 14.5 Roster Trait Balance Rules

- Traits must be helpful but modest.
- Traits should not be stronger than gear or class skills.
- Traits should be capped by equipped slots.
- Traits should not require leveling every class to be competitive.
- Duplicate trait types should not stack fully.

---

# 15. Gear System

## 15.1 Gear Philosophy

Gear should be a major long-term progression system. It should provide stats, identity, visible upgrades, rare drops, and risky enhancement excitement.

However, gear should not only be about linear stat increases. Items can develop histories through upgrades, failures, scars, mutations, and class-reactive traits.

## 15.2 Equipment Slots

Prototype equipment slots:

| Slot | Notes |
|---|---|
| Weapon | Main class-defining item. Most important combat item. |
| Offhand | Shield, focus, quiver, charm, catalyst, etc. Depends on class. |
| Head | Armor / cosmetic silhouette. |
| Chest | Main armor. |
| Gloves | Attack speed, crit, skill modifiers. |
| Boots | Movement, jump, dodge, mobility stats. |
| Ring 1 | Utility stats and traits. |
| Ring 2 | Utility stats and traits. |
| Amulet | Resource and class modifiers. |
| Badge / Charm | Special effects, boss drops, prestige items. |

## 15.3 Item Rarity

| Rarity | Description |
|---|---|
| Common | Basic leveling gear. |
| Uncommon | Slight stat variation, early traits. |
| Rare | Notable stat rolls or simple traits. |
| Epic | Multiple traits, stronger upgrade potential. |
| Relic | Boss/dungeon items with unique effects. |
| Mythic | Endgame chase items, limited sources, strong identity. |

## 15.4 Item Stats

Core stats:

- Attack Power.
- Magic Power.
- Defense.
- Max HP.
- Critical Chance.
- Critical Damage.
- Attack Speed.
- Cast Speed.
- Movement Speed.
- Class Resource Generation.
- Skill Damage.
- Boss Damage.
- Area Damage.
- Burn / trap / companion / block / mark modifiers.

## 15.5 Item Traits

Traits are special item effects.

Examples:

| Trait | Effect |
|---|---|
| Impact Core | Fighter finishers create a small shockwave. |
| Ember Vein | Attacks can apply minor burn. |
| Clear Lens | Marks last longer. |
| Guard Plate | Blocking restores a small amount of resource. |
| Volatile Edge | Higher damage, but resource costs increase. |
| Worn But Lucky | Slightly worse stats, slightly higher rare material drop chance. |

---


## 15.6 Equipment Philosophy: Readable Gear, Deep Progression

Equipment should be readable at a glance but deep over time.

A player should immediately understand that:

- A **sword** is a balanced Fighter weapon.
- A **staff** is a slower but stronger Mage weapon.
- A **longbow** is a precision Archer weapon.
- Heavy armor protects better than cloth.
- Jewelry provides utility and build modifiers.

Depth comes from level tiers, rarity, stat budgets, traits, upgrade history, scars, and class-reactive effects.

Gear should support three purposes:

| Purpose | Design Requirement |
|---|---|
| Leveling reliability | Shops sell understandable baseline gear so unlucky players are not stuck. |
| Build expression | Drops, crafting, traits, and upgrades create specialized builds. |
| Long-term chase | High-rarity and upgraded gear provide aspirational progression. |

Shop gear should create a **power floor**, not the best possible endgame. Rare drops, crafted gear, boss gear, and upgraded gear should create the power ceiling.

## 15.7 Gear Level Requirements and Tier Bands

Gear is divided into level bands. Each tier has a minimum character level and an expected power budget.

| Gear Tier | Level Requirement | Example Names | Primary Sources | Upgrade Cap Guideline | Design Role |
|---|---:|---|---|---:|---|
| Tier 0: Training | 1 | Training, Recruit, Worn | Starting quests, starter shop | +5 | Tutorial gear. Very simple, no traits. |
| Tier 1: Copper | 5 | Copper, Stitched, Simple, Birch | Starter shop, early mobs | +7 | First real gear. Teaches slots and upgrades. |
| Tier 2: Iron | 15 | Iron, Apprentice, Hardened, Oak | Regional shop, field drops | +10 | Early build choices begin. |
| Tier 3: Steel | 25 | Steel, Adept, Reinforced, Ashwood | Advanced class shops, dungeons | +12 | Supports first advanced class identity. |
| Tier 4: Silver | 40 | Silver, Expert, Tempered, Moonlit | Regional shop, rare drops, dungeon tokens | +15 | Midgame gear with clearer stat specialization. |
| Tier 5: Runed | 60 | Runed, Champion, Emberglass, Stormcarved | Dungeons, bosses, crafting | +18 | Specialization gear, multiple traits. |
| Tier 6: Starforged | 80 | Starforged, Masterwork, Deepsteel | Bosses, high-end crafting | +20 | Early endgame chase gear. |
| Tier 7: Ancient | 100 | Ancient, Mythic, Awakened, Relicbound | Endgame bosses, awakening | +20+ | Build-defining gear. Not sold for basic currency. |

Prototype level requirement rule:

```text
A character may equip gear if:
- Character level >= item level requirement, and
- Character class matches the item's class requirement if one exists, and
- Advanced/specialization requirement is met if the item is branch-specific.
```

Most leveling gear should only require a base class. Some Tier 3+ gear can require an advanced class if it directly modifies branch mechanics.

Examples:

| Item | Requirement |
|---|---|
| Iron Sword | Level 15, Fighter. |
| Apprentice Staff | Level 15, Mage. |
| Ashwood Bow | Level 25, Archer. |
| Guardian Tower Shield | Level 25, Fighter advanced into Guardian. |
| Rune-Etched Focus | Level 25, Mage advanced into Rune Mage. |
| Deadeye Longbow | Level 25, Archer advanced into Sniper. |

## 15.8 Gear Rarity and Trait Slots

Rarity controls stat budget, trait access, visual flair, and upgrade potential.

| Rarity | Stat Budget Multiplier | Trait Slots | Visual Identity | Typical Source |
|---|---:|---:|---|---|
| Common | 1.00x | 0 | Plain silhouette, muted colors. | Shops, common drops. |
| Uncommon | 1.08x | 0–1 minor | Slight trim, cleaner model. | Shops, field drops. |
| Rare | 1.18x | 1 | Distinct color accents. | Field elites, dungeons, regional shops. |
| Epic | 1.32x | 1–2 | Strong silhouette, glow accents. | Dungeons, bosses, crafting. |
| Relic | 1.45x | 2 + unique effect | Unique model, lore identity. | Bosses, rare events. |
| Mythic | 1.60x | 2–3 + build-defining effect | Animated effects, iconic silhouette. | Endgame awakening, major bosses. |

Shop vendors should generally sell Common and Uncommon gear. Regional or token vendors may sell limited Rare gear. Epic, Relic, and Mythic gear should primarily come from gameplay achievements, crafting, bosses, or rare drops.

## 15.9 Stat Budget Model

Each item has a stat budget based on tier, rarity, slot, and specialization.

Prototype formula:

```text
Item Stat Budget = Tier Budget × Slot Weight × Rarity Multiplier
```

Tier budget guideline:

| Tier | Level Requirement | Base Budget |
|---|---:|---:|
| Tier 0 | 1 | 10 |
| Tier 1 | 5 | 16 |
| Tier 2 | 15 | 30 |
| Tier 3 | 25 | 48 |
| Tier 4 | 40 | 75 |
| Tier 5 | 60 | 115 |
| Tier 6 | 80 | 165 |
| Tier 7 | 100 | 230 |

Slot weights:

| Slot | Slot Weight | Balance Purpose |
|---|---:|---|
| Weapon | 1.00 | Main offense source. |
| Offhand | 0.55 | Class utility, resource tuning, defense or secondary offense. |
| Chest | 0.70 | Main defense and HP source. |
| Head | 0.45 | Secondary defense and utility. |
| Gloves | 0.45 | Attack/cast speed, crit, skill modifiers. |
| Boots | 0.45 | Movement, jump, dodge, stability. |
| Ring | 0.30 each | Focused utility or secondary damage. |
| Amulet | 0.40 | Resource generation, class modifiers, survivability. |
| Badge / Charm | 0.35 | Special effects, boss trophies, niche traits. |

Example:

```text
Tier 3 Steel Sword, Rare
Base Budget: 48
Weapon Slot Weight: 1.00
Rare Multiplier: 1.18
Total Budget: 56.64, rounded to 57
```

A 57-budget sword might allocate its budget as:

| Stat | Allocation |
|---|---:|
| Attack Power | 42 |
| Critical Chance | 6 |
| Attack Speed | 5 |
| Minor trait value | 4 |

Final numeric conversion should be tuned through combat testing. The purpose of the budget is to prevent items from having too many strong stats at once.

## 15.10 Stat Categories and Conversion Costs

Different stats should cost different amounts of item budget. High-impact stats require higher budget cost.

Prototype conversion table:

| Stat | Budget Cost Guideline | Notes |
|---|---:|---|
| Attack Power / Magic Power | 1 budget = 1 point | Main weapon scaling. |
| Defense | 1 budget = 1.5 points | Armor gets more raw defense per budget. |
| Max HP | 1 budget = 8–12 HP | Scales by level bracket. |
| Critical Chance | 4 budget = 1% | Keep crit chance controlled. |
| Critical Damage | 3 budget = 2% | Stronger on crit-heavy builds. |
| Attack Speed / Cast Speed | 5 budget = 1% | Powerful feel stat; cap carefully. |
| Movement Speed | 4 budget = 1% | Useful but should not trivialize maps. |
| Boss Damage | 4 budget = 2% | Single-target orientation. |
| Area Damage | 4 budget = 2% | Mobbing orientation. |
| Resource Generation | 5 budget = 1% | Can break rotations if too high. |
| Skill Cooldown Reduction | 8 budget = 1% | Very high-impact; use sparingly. |
| Element/branch modifier | 4 budget = 2% | Fire damage, trap damage, block value, mark damage, etc. |

Balance note: gear should not easily stack **Attack Power**, **Boss Damage**, **Critical Chance**, and **Critical Damage** all on the same item unless that item sacrifices defense, speed, or area clearing.

## 15.11 Equipment Slot Details

### Weapon

Weapons are the primary source of offense and should visibly change the character silhouette.

| Class | Weapon Type | Visual Description | Main Stats | Balance Identity |
|---|---|---|---|---|
| Fighter | Sword | One-handed or two-handed blade, clean outline. | Attack Power, attack speed, crit chance. | Balanced offense. |
| Fighter | Axe | Heavy head, broad swing silhouette. | High Attack Power, armor break, lower speed. | Bossing and elite damage. |
| Fighter | Hammer | Large blunt weapon, exaggerated impact. | Attack Power, area damage, stagger. | Mobbing and control. |
| Fighter | Spear | Long reach, narrow point. | Attack Power, range, line damage. | Hybrid reach and spacing. |
| Mage | Wand | Small focus implement, fast casting. | Magic Power, cast speed, Energy generation. | Fast spell loops. |
| Mage | Staff | Tall magical staff, large headpiece. | High Magic Power, area damage, lower speed. | Burst and mobbing. |
| Mage | Orb | Floating catalyst held or orbiting. | Magic Power, resource generation, crit. | Technical/bossing builds. |
| Mage | Tome | Spellbook or grimoire. | Magic Power, skill damage, rune/utility modifiers. | Setup and support builds. |
| Archer | Shortbow | Compact bow, fast animation. | Attack Power, attack speed, Focus generation. | Mobbing and movement. |
| Archer | Longbow | Tall bow, strong draw animation. | Attack Power, crit damage, range. | Bossing precision. |
| Archer | Crossbow | Mechanical silhouette, slower reload. | High Attack Power, crit chance, armor pierce. | Burst and elites. |
| Archer | Thrower | Wrist launcher or compact projectile device. | Attack speed, trap/mark modifiers. | Hybrid utility. |

### Offhand

Offhands are secondary identity items. They should not outscale weapons, but they should meaningfully support branch mechanics.

| Class / Branch | Offhand Type | Visual Description | Main Stats | Example Use |
|---|---|---|---|---|
| Fighter | Shield | Round, kite, or tower shield. | Defense, HP, block value. | Guardian survivability. |
| Fighter | Buckler | Small parry shield. | Crit chance, attack speed, parry value. | Duelist timing builds. |
| Fighter | War Grip | Gauntlet or weapon charm. | Attack Power, Rage/Momentum generation. | Berserker aggression. |
| Mage | Focus | Crystal, charm, or small relic. | Magic Power, Energy generation. | General caster utility. |
| Mage | Seal | Floating sigil plate. | Rune duration, debuff strength. | Rune Mage setup. |
| Mage | Ember Core | Small burning catalyst. | Heat capacity, burn damage. | Fire Mage risk/reward. |
| Archer | Quiver | Visible arrow pack. | Focus generation, attack speed. | General Archer DPS. |
| Archer | Scope | Lens or sighting device. | Crit, range, weak-point duration. | Sniper bossing. |
| Archer | Trap Kit | Belt or satchel of tools. | Trap damage, arming speed. | Trapper control. |
| Archer | Beast Token | Whistle, claw charm, spirit mark. | Companion damage, Bond gain. | Beast Archer later. |

### Head

Head gear contributes defense and one secondary utility stat.

Names and appearances:

| Type | Possible Names | Visual Look | Stat Lean |
|---|---|---|---|
| Fighter helm | Guard Helm, Iron Visor, Steel Barbute | Metal helmet, strong outline. | Defense, HP, knockback resistance. |
| Mage hood | Apprentice Hood, Rune Cowl, Ember Veil | Cloth hood, glowing trim. | Magic Power, resource generation, status resistance. |
| Archer cap | Scout Cap, Hunter Hood, Longshot Visor | Light hood/cap, goggles or feathers. | Crit, movement, Focus generation. |

### Chest

Chest armor is the main defensive item.

| Armor Weight | Possible Names | Visual Look | Stat Lean | Tradeoff |
|---|---|---|---|---|
| Heavy | Plate Mail, Tower Coat, Iron Harness | Bulky armor, large shoulder plates. | Defense, HP, knockback resistance. | Lower movement/cast speed. |
| Medium | Scale Vest, Scout Jacket, Reinforced Coat | Practical layered armor. | Balanced defense, crit, speed. | Lower max defense. |
| Light / Cloth | Rune Robe, Ember Robe, Windwrap | Cloth, glowing trim, lighter silhouette. | Magic Power, resource generation, cast speed. | Low defense. |

### Gloves

Gloves should support action feel.

| Type | Stat Lean |
|---|---|
| Grip Gloves | Attack speed, weapon damage. |
| Casting Gloves | Cast speed, Energy generation. |
| Precision Gloves | Critical chance, mark damage. |
| Trap Gloves | Trap arming speed, control duration. |
| Guard Gauntlets | Block value, defense. |

### Boots

Boots control movement feel.

| Type | Stat Lean |
|---|---|
| Iron Boots | Defense, knockback resistance. |
| Scout Boots | Movement speed, dodge recovery. |
| Wind Boots | Jump/mobility stats, speed. |
| Rune Sandals | Cast speed, resource generation. |
| Spiked Boots | Momentum generation, close-range control. |

### Rings

Rings are small stat-customization items. They should not dominate builds individually, but two rings together can support a build direction.

| Ring Type | Stat Lean |
|---|---|
| Power Ring | Attack/Magic Power. |
| Guard Ring | Defense/HP. |
| Focus Ring | Resource generation. |
| Sharp Ring | Critical chance. |
| Ember Ring | Burn damage. |
| Hunter Ring | Boss damage or mark damage. |
| Sweep Ring | Area damage. |

### Amulet

Amulets are build-shaping but not always raw damage.

| Amulet Type | Stat Lean |
|---|---|
| Momentum Amulet | Fighter resource generation. |
| Energy Amulet | Mage resource generation. |
| Focus Amulet | Archer resource generation. |
| Vital Amulet | HP and sustain. |
| Precision Amulet | Crit damage and boss damage. |
| Field Amulet | Area damage and control duration. |

### Badge / Charm

Badges and charms are special-effect items from bosses, achievements, and events.

Examples:

| Item | Source | Effect |
|---|---|---|
| Emberjaw Badge | Emberjaw Golem | Minor fire resistance and bonus damage to burning enemies. |
| Cart Defender Charm | Escort challenge | Reduced knockback while near objectives. |
| Old Hunter Token | Rare field elite | Slightly increased rare enemy detection. |
| Cracked Core Charm | Failed high-risk upgrade | Slight resource burst after taking heavy damage. |

## 15.12 Mobbing Gear vs Bossing Gear

Gear should support both mobbing and bossing without making one set universally best.

| Gear Orientation | Common Stats | Good For | Tradeoff |
|---|---|---|---|
| Mobbing | Area Damage, attack/cast speed, movement, resource generation, on-hit effects. | Leveling, wave maps, dungeons. | Lower single-target burst. |
| Bossing | Boss Damage, crit chance, crit damage, armor pierce, weak-point damage. | Bosses, elites, challenge rooms. | Lower map clear and less utility. |
| Defensive | Defense, HP, damage reduction, status resistance. | Hard bosses, solo survival. | Lower damage. |
| Utility | Resource generation, cooldown recovery, control duration, movement. | Smooth rotations, class mechanics. | Lower raw stats. |
| Hybrid | Balanced stats. | General play. | Not best at any one job. |

Example balancing rule:

```text
An item with Boss Damage and Critical Damage should not also roll high Area Damage.
An item with high Area Damage may roll attack speed or resource generation, but should have lower boss burst stats.
```

## 15.13 Shop Gear vs Drop Gear vs Crafted Gear

| Gear Source | Strength | Weakness | Purpose |
|---|---|---|---|
| Shop gear | Reliable, affordable, predictable. | Lower trait access, lower upgrade ceiling. | Prevent bad luck from blocking progression. |
| Field drops | Random, can be better than shop gear. | Inconsistent rolls. | Reward farming and exploration. |
| Dungeon drops | More specialized stats. | Requires party or dungeon clears. | Encourage multiplayer and objectives. |
| Boss drops | Unique traits and badges. | Time-gated or difficult. | Create aspirational progression. |
| Crafted gear | Targeted stat families. | Requires materials. | Let players work toward specific builds. |
| Upgraded gear | Highest personal investment. | Risk of downgrade/scar/break. | Long-term chase and item identity. |

Shop gear should generally be 5–10% weaker than comparable well-rolled drops, but strong enough to clear expected story content at that level.

## 15.14 Example Implementable Gear Items

### Tier 1: Level 5 Shop Gear

| Item | Slot | Requirement | Appearance | Stats | Notes |
|---|---|---|---|---|---|
| Copper Sword | Fighter Weapon | Level 5 Fighter | Simple copper blade with leather grip. | +16 Attack Power, +2% attack speed. | Balanced starter weapon. |
| Birch Wand | Mage Weapon | Level 5 Mage | Pale wood wand with small blue crystal. | +15 Magic Power, +2% cast speed. | Fast starter casting. |
| Simple Bow | Archer Weapon | Level 5 Archer | Plain shortbow with stitched grip. | +15 Attack Power, +2% Focus gain. | General starter ranged weapon. |
| Stitched Vest | Chest | Level 5 Any | Basic brown padded vest. | +18 Defense, +60 HP. | General leveling armor. |
| Traveler Boots | Boots | Level 5 Any | Soft leather boots. | +8 Defense, +3% movement speed. | Teaches mobility gear. |

### Tier 2: Level 15 Regional Gear

| Item | Slot | Requirement | Appearance | Stats | Notes |
|---|---|---|---|---|---|
| Iron Axe | Fighter Weapon | Level 15 Fighter | Dark iron axe with heavy head. | +34 Attack Power, +4% armor break effect, -2% attack speed. | Bossing/elite lean. |
| Apprentice Staff | Mage Weapon | Level 15 Mage | Staff with wrapped handle and faint glow. | +32 Magic Power, +4% Area Damage. | Mobbing lean. |
| Oak Longbow | Archer Weapon | Level 15 Archer | Tall wooden bow with reinforced limbs. | +31 Attack Power, +5% crit damage, +1 range unit. | Bossing lean. |
| Guard Helm | Head | Level 15 Any | Iron half-helm. | +22 Defense, +80 HP. | Defensive option. |
| Focus Ring | Ring | Level 15 Any | Small silver ring with plain gem. | +3% class resource generation. | Utility option. |

### Tier 3: Level 25 Advanced Class Gear

| Item | Slot | Requirement | Appearance | Stats | Notes |
|---|---|---|---|---|---|
| Guardian Tower Shield | Offhand | Level 25 Guardian | Tall rectangular shield with reinforced rim. | +40 Defense, +180 HP, +6% block value. | Supports Shield Wall and Impact Guard. |
| Berserker War Grip | Offhand | Level 25 Berserker | Spiked gauntlet wrapped in red cloth. | +18 Attack Power, +6% Rage generation. | Aggressive offhand. |
| Ember Core | Offhand | Level 25 Fire Mage | Small floating ember crystal. | +16 Magic Power, +8% burn damage, +5 Heat capacity. | Fire Mage identity item. |
| Rune-Etched Focus | Offhand | Level 25 Rune Mage | Stone disk with glowing symbols. | +14 Magic Power, +8% rune duration, +4% resource generation. | Rune setup item. |
| Deadeye Scope | Offhand | Level 25 Sniper | Lens attachment or monocular sight. | +6% crit chance, +8% weak-point duration. | Bossing utility. |
| Trap Kit | Offhand | Level 25 Trapper | Belt pouch with springs and wires. | +10% trap arming speed, +7% trap damage. | Mobbing/control utility. |

### Tier 4: Level 40 Specialized Regional Gear

| Item | Slot | Requirement | Appearance | Stats | Notes |
|---|---|---|---|---|---|
| Tempered Hammer | Fighter Weapon | Level 40 Fighter | Large square-headed hammer with orange metal trim. | +73 Attack Power, +8% Area Damage, +5% stagger chance. | Mobbing/control Fighter weapon. |
| Moonlit Orb | Mage Weapon | Level 40 Mage | Floating silver-blue orb. | +68 Magic Power, +5% crit chance, +6% Energy generation. | Technical caster weapon. |
| Hunter Crossbow | Archer Weapon | Level 40 Archer | Compact metal crossbow with crank. | +76 Attack Power, +7% crit chance, -4% attack speed. | Bossing burst weapon. |
| Expert Gloves | Gloves | Level 40 Any | Class-colored gloves. | +4% attack/cast speed, +3% crit chance. | General DPS gloves. |
| Field Amulet | Amulet | Level 40 Any | Green amulet with map-like engraving. | +6% Area Damage, +4% movement speed. | Mobbing accessory. |

### Tier 5: Level 60 Specialization Gear

| Item | Slot | Requirement | Appearance | Stats | Notes |
|---|---|---|---|---|---|
| Runed Plate | Chest | Level 60 Fighter | Heavy armor with glowing seam lines. | +125 Defense, +550 HP, +5% status resistance. | Defensive midgame armor. |
| Emberglass Staff | Weapon | Level 60 Fire Mage | Red glass staff head with contained flame. | +112 Magic Power, +10% burn damage, +8% Area Damage. | Strong mobbing/fire option. |
| Longshot Bow | Weapon | Level 60 Sniper | Elegant longbow with sight mark carvings. | +118 Attack Power, +12% crit damage, +8% Boss Damage. | Bossing weapon. |
| Control Band | Ring | Level 60 Any | Dark ring with wire-like pattern. | +7% damage to slowed/rooted enemies. | Trapper/Rune synergy. |

## 15.15 Gear Appearance and Readability Rules

Gear should communicate function through silhouette and animation.

| Gear Type | Visual Rule |
|---|---|
| Defensive gear | Larger, heavier, broader silhouettes. |
| Fast gear | Slimmer silhouettes, lighter colors, smaller profiles. |
| Fire gear | Warm glow, ember particles, scorched accents. |
| Rune gear | Geometric lines, symbols, pulsing inscriptions. |
| Storm gear | Sharp shapes, wind or spark accents. |
| Sniper gear | Lenses, long limbs, focused clean lines. |
| Trapper gear | Belts, pouches, wires, small mechanisms. |
| Boss gear | Contains visible trophy parts from the boss. |
| Scarred gear | Cracks, asymmetry, flickering effects, imperfect glow. |

Important readability rule:

```text
A player should be able to tell the broad type of another player's weapon and armor weight from their silhouette, even if exact stats are hidden.
```

## 15.16 Gear Balance Rules

- Weapons should drive damage identity, but armor and accessories should shape survivability and build utility.
- Shop gear must be good enough for story progression but not best-in-slot.
- Every level band should offer at least one generalist set and at least one mobbing or bossing option.
- Bossing gear should not also be the best mobbing gear.
- Mobbing gear should improve clear speed but should not trivialize bosses.
- Defensive gear should be valuable for progression and solo play without becoming mandatory for optimized farming.
- Class-specific gear should strengthen identity but not make a class unplayable without it.
- Rare traits should be exciting but numerically controlled through the stat budget.
- Upgrade potential should be part of item value. A lower-rarity item with high upgrade success may compete with a higher-rarity item temporarily.
- Visible gear changes should reward progression without harming animation clarity.


# 16. Risk-Based Upgrade System

## 16.1 Goals

The upgrade system should create excitement, risk, and long-term goals.

It should avoid pure frustration by making many failures produce useful outcomes.

## 16.2 Upgrade Types

| Upgrade Type | Risk | Purpose |
|---|---|---|
| Reinforce | Low | Increase base stats. |
| Refine | Medium | Improve stat rolls or add minor traits. |
| Infuse | Medium-high | Add element or class-reactive trait. |
| Overcharge | High | Major stat increase, risk of instability. |
| Awaken | Very high | Unlock or transform unique item effect. |
| Restore | Utility | Repair scars, stabilize instability, or recover downgrade. |

## 16.3 Upgrade Outcomes

| Outcome | Description |
|---|---|
| Clean Success | Upgrade succeeds normally. |
| Great Success | Upgrade succeeds with bonus stat or improved trait. |
| Mutation | Upgrade succeeds but changes or adds an unusual trait. |
| Scar | Item gains a drawback and a hidden or visible upside. |
| Instability | Item becomes stronger but harder to control or riskier to use. |
| Downgrade | Upgrade level decreases. |
| Fracture | Item loses power but creates rare material. |
| Break | Rare severe failure; item is destroyed or becomes unusable, but yields valuable material. |

## 16.4 Failure Philosophy

Failure should create drama, not only loss.

Examples:

- A failed fire infusion creates **Ash Scar**: lower direct damage, stronger burn duration.
- A failed reinforcement creates **Cracked Core**: item loses one upgrade level, but gains a chance to trigger a class-resource burst.
- A severe failure creates **Fracture Dust**, a rare material used for restoration or crafting.

## 16.5 Upgrade Level Example

Prototype upgrade ladder:

```text
+0 to +5: low risk, no break chance
+6 to +10: moderate risk, downgrade possible
+11 to +15: high risk, scars and instability possible
+16 to +20: very high risk, fracture or break possible unless protected
```

## 16.6 Protection Items

Protection items can exist but must not be sold for real money in a pay-to-win way.

Possible sources:

- Boss drops.
- Event rewards.
- Crafting.
- Weekly content.
- Account achievement rewards.

Protection types:

| Protection | Effect |
|---|---|
| Stabilizer | Prevents break, but not downgrade. |
| Repair Seal | Prevents new scar once. |
| Memory Charm | Preserves one selected trait during mutation. |
| Safety Thread | Reduces severity of failure. |

## 16.7 Upgrade Pity and Fairness

To prevent extreme frustration:

- Track failed attempts at high upgrade levels.
- Increase chance of non-catastrophic outcomes after repeated failures.
- Provide Fracture Dust or similar materials on severe failures.
- Allow slow deterministic progress through restoration materials.
- Clearly show risks before the player confirms an upgrade.

## 16.8 Upgrade UI Requirements

The upgrade UI must show:

- Success chance.
- Great success chance.
- Failure outcomes.
- Downgrade chance.
- Scar/mutation chance.
- Break/fracture chance.
- Protection item effects.
- Preview of possible trait changes where appropriate.

The game should not hide major risks from players.

Current prototype status:

- Implemented outcomes are success, great success, fail, downgrade, scar, fracture, and rare break.
- The upgrade station presents a detailed chance table, cost, materials, and selected-item state before the player upgrades.
- Mutation, protection-item selection, pity messaging, and deterministic restoration are future work.

---

# 17. Class-Reactive Gear Traits

## 17.1 Purpose

Class-reactive traits make gear more interesting across characters. The same item can behave differently depending on the class using it.

This supports:

- Buildcrafting.
- Alt characters.
- Trading.
- Unique upgrade outcomes.
- Replayability.

## 17.2 Example Trait: Volatile Core

| Class | Effect |
|---|---|
| Fighter | Momentum finishers create a shockwave. |
| Mage | Energy spenders have a chance to overload for bonus damage. |
| Archer | Focus spenders trigger a delayed secondary hit. |

## 17.3 Example Trait: Cracked Lens

| Class | Effect |
|---|---|
| Guardian | Perfect guard stores extra Impact. |
| Fire Mage | Burning Mark spreads farther. |
| Sniper | Weak Point Mark lasts longer. |
| Trapper | Traps reveal hidden weak points on bosses. |

## 17.4 Example Trait: Scarred Ember Ring

Base effect:

- +Magic Power.
- +Burn damage.
- Scar: taking damage builds elemental pressure.

Class-reactive effects:

| Class | Effect |
|---|---|
| Fighter | Momentum Burst adds fire damage. |
| Mage | Heat or Energy spenders create a small explosion. |
| Archer | Marked targets can ignite when hit by Focus spenders. |

## 17.5 Balance Rules

- Class-reactive traits should be flavorful, not mandatory.
- A trait should not be best-in-slot for every class.
- Effects should support multiple builds.
- Some traits can be more valuable to certain branches.
- Tooltips must clearly explain class-specific behavior.

---

# 18. Combat Content

Current prototype status: combat content is represented through authored map data, enemy tables, boss flags, route unlocks, and local Canvas 2D behavior. The current implementation covers field maps, dungeon-like maps, boss arenas, drops, ladders, platforms, safe-zone stations, minimap context, and world-map navigation, but encounters are still prototype systems rather than fully authored MMO content.

Near-term combat content work should focus on making maps read better at first glance, tightening enemy pathing edge cases, improving telegraph clarity, and turning boss maps into deliberately scripted fights with retry/reward flow.

## 18.1 Content Types

| Content Type | Description |
|---|---|
| Field Maps | Repeatable leveling and farming areas. |
| Story Quests | Guided progression, region unlocks, class trials. |
| Party Dungeons | 4-player instanced content with objectives. |
| Bosses | Single or party fights with mechanics and rare drops. |
| Challenge Rooms | Short combat tests for materials or cosmetics. |
| Events | Limited-time maps, bosses, cosmetics, upgrade materials. |
| World Bosses | Larger public encounters added later. |

## 18.2 Field Map Design

Field maps should support:

- Solo leveling.
- Small party farming.
- Material collection.
- Hidden elites.
- Environmental hazards.
- Occasional rare monster spawns.

Field maps should avoid being only static grind platforms. Add light objectives:

- Clear waves.
- Protect a cart.
- Break monster nests.
- Activate map shrines.
- Defeat roaming elites.

## 18.3 Dungeon Design

Standard dungeon length: 8–15 minutes.

Dungeon structure:

1. Entry room.
2. Combat rooms with optional objectives.
3. Mini-boss or puzzle-control room.
4. Final boss.
5. Reward chest and score breakdown.

Dungeon objectives can include:

- Defeat enemies.
- Hold a zone.
- Protect an object.
- Avoid traps.
- Split party paths briefly.
- Use class roles to make objectives easier, but never impossible without a specific class.

## 18.4 Boss Design

Boss fights should reward:

- Dodging.
- Party positioning.
- Buff timing.
- Resource management.
- Burst windows.
- Learning patterns.

Bosses should have:

- Clear telegraphs.
- Enrage or soft timer.
- Mechanics that reward defensive, offensive, mobility, and control tools.
- Rare drops and upgrade materials.

## 18.5 Example Boss: Emberjaw Golem

Level range: 30–40.

Mechanics:

- Slams ground, creating fire cracks.
- Charges across arena.
- Summons ore minions.
- Overheats, becoming vulnerable to control or burst.

Class interactions:

- Guardian can shield slam waves.
- Berserker can burst during Overheat vulnerability.
- Fire Mage is less effective with direct fire but can use burn spread on minions.
- Rune Mage can seal minion portals.
- Sniper can mark exposed core.
- Trapper can slow ore minions and control charge paths.

---


## 18.6 Enemy Design Goals

Enemies should be simple enough to understand quickly, but varied enough to make class strengths matter.

Each enemy should teach or test at least one skill:

| Enemy Function | What It Tests |
|---|---|
| Basic melee | Movement, attack timing, basic damage rotation. |
| Swarm enemy | Mobbing skills, area damage, resource generation. |
| Ranged enemy | Dodging, target priority, closing distance. |
| Charger | Telegraph reading, vertical movement, control skills. |
| Armored enemy | Bossing damage, armor break, sustained single-target pressure. |
| Flying enemy | Ranged attacks, anti-air skills, mobility. |
| Support enemy | Target priority and interruption. |
| Elite enemy | Mini-boss mechanics, burst windows, party coordination. |

Enemies should be implemented with clear data fields so designers can tune them without rewriting behavior.

## 18.7 Enemy Data Schema

Each enemy should have the following implementation fields:

| Field | Description |
|---|---|
| Enemy ID | Internal identifier. |
| Display Name | Name shown to players. |
| Family | Slime, beast, construct, humanoid, spirit, etc. |
| Level Range | Intended player level range. |
| Role Tags | Swarm, melee, ranged, tank, charger, support, elite, flying. |
| HP Multiplier | Multiplier against baseline enemy HP for its level. |
| Damage Multiplier | Multiplier against baseline enemy damage for its level. |
| Defense Multiplier | Multiplier against baseline enemy defense for its level. |
| EXP Multiplier | Multiplier against baseline EXP for its level. |
| Move Speed | Slow, medium, fast, flying, stationary. |
| Aggro Range | Distance at which the enemy reacts. |
| Attack Range | Melee, short, medium, long. |
| Attack Pattern | Basic AI loop and special actions. |
| Telegraph Time | Time between warning and dangerous attack. |
| Resistances | Element or damage-type resistance. |
| Weaknesses | Element, status, or mechanic vulnerability. |
| Drops | Currency, gear type, crafting material, rare drop. |
| Spawn Rules | Solo, small pack, large pack, mixed group, rare spawn. |
| Balance Purpose | Why the enemy exists in the game. |

## 18.8 Enemy Baseline Formulas

The following formulas are prototype tuning targets. Final numbers should be adjusted after combat testing.

```text
Baseline Enemy HP at Level L = 42 × L^1.35
Baseline Enemy Damage at Level L = 5 + (L × 1.8)
Baseline Enemy Defense at Level L = 2 + (L × 0.7)
Baseline Enemy EXP at Level L = 7 × L^1.15
```

Enemy archetypes multiply these values.

| Archetype | HP Mult | Damage Mult | Defense Mult | EXP Mult | Notes |
|---|---:|---:|---:|---:|---|
| Swarm | 0.45 | 0.55 | 0.60 | 0.45 | Appears in groups. Good for mobbing skills. |
| Basic Melee | 1.00 | 1.00 | 1.00 | 1.00 | Standard enemy. |
| Ranged | 0.80 | 0.90 | 0.80 | 1.00 | Lower HP, forces movement. |
| Charger | 1.15 | 1.25 | 1.00 | 1.15 | Telegraphs heavy movement attack. |
| Tank | 1.80 | 0.85 | 1.60 | 1.40 | Tests single-target and armor break. |
| Support | 0.75 | 0.50 | 0.70 | 1.10 | Buffs or heals enemies; high priority. |
| Elite | 4.00 | 1.60 | 1.40 | 4.00 | Rare or objective enemy. Mini-boss behavior. |
| Flying | 0.85 | 0.85 | 0.70 | 1.10 | Tests range, mobility, and anti-air. |

Health and damage should also be adjusted by map density. A map with many swarm enemies should not also use high-damage ranged enemies unless the map is meant to be dangerous.

## 18.9 Early Enemy Bestiary

The following enemies are enough to support early implementation from Level 1 to Level 35.

### 18.9.1 Slimelet

| Field | Value |
|---|---|
| Enemy ID | `enemy_slimelet_001` |
| Family | Ooze |
| Level Range | 1–6 |
| Role Tags | Basic melee, tutorial, swarm-light |
| HP / Damage / Defense / EXP | 0.75 / 0.60 / 0.60 / 0.70 |
| Move Speed | Slow hop |
| Aggro Range | Short |
| Attack Range | Melee contact |
| Telegraph Time | 0.4 sec hop windup |
| Resistances | None |
| Weaknesses | Area damage, knockback |
| Drops | Small currency, Training gear, Gel Drop material |
| Spawn Rules | Groups of 2–5 on starter maps |
| Balance Purpose | Teaches basic attacking, movement, and mobbing without high danger. |

Behavior:

- Idles until player enters short range.
- Hops toward target every 1.2 seconds.
- Contact hit on landing.
- Can be knocked back by most attacks.

Tuning note: Slimelets should die quickly to basic attacks and faster to early area skills. They should make Fighter, Mage, and Archer all feel competent at Level 1.

### 18.9.2 Mossback

| Field | Value |
|---|---|
| Enemy ID | `enemy_mossback_001` |
| Family | Beast |
| Level Range | 4–10 |
| Role Tags | Basic melee, tank-light |
| HP / Damage / Defense / EXP | 1.25 / 0.85 / 1.20 / 1.10 |
| Move Speed | Slow |
| Aggro Range | Medium |
| Attack Range | Melee |
| Telegraph Time | 0.5 sec head-lower animation |
| Resistances | Minor resistance to knockback |
| Weaknesses | Fire, armor break |
| Drops | Copper gear, Moss Hide, small chance of Guard Ring |
| Spawn Rules | 1–3 mixed with Slimelets or Thorn Sprouts |
| Balance Purpose | Introduces tougher enemies that reward Power Break, marks, and focus fire. |

Behavior:

- Walks toward target and performs short headbutt.
- Briefly braces after taking multiple hits, reducing knockback.
- Does not deal high damage, but takes longer to kill.

### 18.9.3 Thorn Sprout

| Field | Value |
|---|---|
| Enemy ID | `enemy_thorn_sprout_001` |
| Family | Plant |
| Level Range | 5–12 |
| Role Tags | Stationary ranged |
| HP / Damage / Defense / EXP | 0.70 / 0.85 / 0.70 / 1.00 |
| Move Speed | Stationary |
| Aggro Range | Medium-long |
| Attack Range | Medium projectile |
| Telegraph Time | 0.6 sec stem glow before shot |
| Resistances | Root/slow immune because stationary |
| Weaknesses | Fire, line attacks, long range |
| Drops | Thorn Fiber, Copper/Uncommon gloves, low chance of Focus Ring |
| Spawn Rules | Placed on platforms or behind melee enemies |
| Balance Purpose | Teaches projectile dodging and target priority. |

Behavior:

- Rotates toward player.
- Fires a single thorn projectile every 2.0 seconds.
- Projectile is slow enough to dodge or jump over.
- Vulnerable after firing for 0.5 seconds.

### 18.9.4 Bristle Boar

| Field | Value |
|---|---|
| Enemy ID | `enemy_bristle_boar_001` |
| Family | Beast |
| Level Range | 8–16 |
| Role Tags | Charger |
| HP / Damage / Defense / EXP | 1.15 / 1.25 / 1.00 / 1.20 |
| Move Speed | Medium, fast during charge |
| Aggro Range | Medium |
| Attack Range | Horizontal charge |
| Telegraph Time | 0.9 sec paw scrape before charge |
| Resistances | Minor resistance to frontal knockback during charge |
| Weaknesses | Traps, slows, vertical movement, attacks from behind |
| Drops | Bristle Hide, Iron materials, chance of Traveler Boots |
| Spawn Rules | 1–2 per group, often in horizontal lanes |
| Balance Purpose | Teaches telegraphs, jump avoidance, and control. |

Behavior:

- Tracks player for 1 second.
- Scrapes paw and locks direction.
- Charges horizontally for a fixed distance.
- Briefly crashes/stuns itself if it hits a wall or shielded target.

Class interactions:

- Guardian can block the charge and gain Stored Impact.
- Trapper can stop or slow the charge.
- Sniper can punish the recovery window.

### 18.9.5 Dust Imp

| Field | Value |
|---|---|
| Enemy ID | `enemy_dust_imp_001` |
| Family | Imp |
| Level Range | 10–18 |
| Role Tags | Fast melee, evasive |
| HP / Damage / Defense / EXP | 0.85 / 1.00 / 0.75 / 1.10 |
| Move Speed | Fast |
| Aggro Range | Medium |
| Attack Range | Short melee leap |
| Telegraph Time | 0.45 sec crouch before leap |
| Resistances | Minor resistance to slow duration |
| Weaknesses | Area damage, fast attacks, traps |
| Drops | Dust Claw, Uncommon boots/gloves, minor upgrade material |
| Spawn Rules | Groups of 2–4, often mixed with ranged enemies |
| Balance Purpose | Pressures stationary players and tests fast target switching. |

Behavior:

- Dashes in short bursts.
- Performs small leap slash.
- Occasionally jumps backward after being hit.
- Low HP, but annoying if ignored.

### 18.9.6 Clockbug

| Field | Value |
|---|---|
| Enemy ID | `enemy_clockbug_001` |
| Family | Construct |
| Level Range | 12–22 |
| Role Tags | Armored tank |
| HP / Damage / Defense / EXP | 1.70 / 0.80 / 1.70 / 1.40 |
| Move Speed | Slow |
| Aggro Range | Medium |
| Attack Range | Melee bite, short gear burst |
| Telegraph Time | 0.7 sec gear spin |
| Resistances | Poison/burn duration reduced, high defense |
| Weaknesses | Armor break, lightning/storm, bossing skills |
| Drops | Clockwork Scrap, Iron gear, chance of Focus Amulet |
| Spawn Rules | 1 per group early; 2 in later construct maps |
| Balance Purpose | Gives bossing-oriented skills value during leveling. |

Behavior:

- Slowly approaches target.
- Alternates bite with short radial gear burst.
- Takes reduced damage until armor is cracked.
- Armor cracks after enough hits or an armor-break skill.

### 18.9.7 Ember Wisp

| Field | Value |
|---|---|
| Enemy ID | `enemy_ember_wisp_001` |
| Family | Spirit |
| Level Range | 16–26 |
| Role Tags | Flying, ranged, elemental |
| HP / Damage / Defense / EXP | 0.85 / 0.95 / 0.70 / 1.15 |
| Move Speed | Floating medium |
| Aggro Range | Medium-long |
| Attack Range | Medium fire projectile |
| Telegraph Time | 0.6 sec glow pulse |
| Resistances | Fire resistance, reduced trap trigger rate if airborne |
| Weaknesses | Marks, lightning, ranged attacks |
| Drops | Ember Dust, Ember Ring, Fire Mage materials |
| Spawn Rules | 2–3 in vertical maps; mixed with ground enemies |
| Balance Purpose | Teaches anti-air, ranged targeting, and elemental resistance. |

Behavior:

- Floats within range, maintaining distance.
- Fires slow firebolt projectile.
- Occasionally rises above ground attacks.
- Can be pulled lower by certain control effects.

### 18.9.8 Bandit Cutter

| Field | Value |
|---|---|
| Enemy ID | `enemy_bandit_cutter_001` |
| Family | Humanoid |
| Level Range | 18–28 |
| Role Tags | Melee, blocker |
| HP / Damage / Defense / EXP | 1.05 / 1.05 / 1.00 / 1.10 |
| Move Speed | Medium |
| Aggro Range | Medium |
| Attack Range | Melee slash |
| Telegraph Time | 0.5 sec weapon raise |
| Resistances | Brief frontal block reduces damage |
| Weaknesses | Back attacks, stuns, marks, fire |
| Drops | Bandit Cloth, Steel weapon chance, currency |
| Spawn Rules | 2–3 with Bandit Throwers |
| Balance Purpose | Introduces humanoid enemies that use defensive actions. |

Behavior:

- Runs toward target.
- Uses two-hit slash combo.
- Occasionally blocks from the front for 1 second.
- Vulnerable from behind while blocking.

### 18.9.9 Bandit Thrower

| Field | Value |
|---|---|
| Enemy ID | `enemy_bandit_thrower_001` |
| Family | Humanoid |
| Level Range | 20–30 |
| Role Tags | Ranged, support-light |
| HP / Damage / Defense / EXP | 0.80 / 0.90 / 0.75 / 1.05 |
| Move Speed | Medium, retreats from close targets |
| Aggro Range | Long |
| Attack Range | Long projectile arc |
| Telegraph Time | 0.7 sec throw windup |
| Resistances | None |
| Weaknesses | Gap closers, long-range shots, burst |
| Drops | Throwing Knife Scrap, Archer gloves, small chance of Sharp Ring |
| Spawn Rules | 1–2 behind melee enemies or on platforms |
| Balance Purpose | Teaches priority targeting and vertical map awareness. |

Behavior:

- Keeps distance from player.
- Throws arcing knives.
- Repositions backward if player gets too close.
- Low HP but dangerous if left alone.

### 18.9.10 Oreback Beetle

| Field | Value |
|---|---|
| Enemy ID | `enemy_oreback_beetle_001` |
| Family | Beast / Mineral |
| Level Range | 24–35 |
| Role Tags | Tank, material enemy |
| HP / Damage / Defense / EXP | 1.90 / 0.90 / 1.80 / 1.50 |
| Move Speed | Slow |
| Aggro Range | Medium |
| Attack Range | Melee horn, short stomp |
| Telegraph Time | 0.8 sec shell lift before stomp |
| Resistances | High defense, fire resistance if red variant |
| Weaknesses | Armor break, hammer attacks, weak-point marks |
| Drops | Ore Chunks, Iron/Steel materials, rare upgrade catalyst |
| Spawn Rules | 1 per pack; rare elite variant can spawn |
| Balance Purpose | Provides crafting materials and tests single-target damage. |

Behavior:

- Slow movement but high durability.
- Stomp creates short ground shockwave.
- Shell can be cracked, temporarily reducing defense.

### 18.9.11 Glowcap Healer

| Field | Value |
|---|---|
| Enemy ID | `enemy_glowcap_healer_001` |
| Family | Plant |
| Level Range | 22–34 |
| Role Tags | Support, stationary-light |
| HP / Damage / Defense / EXP | 0.75 / 0.40 / 0.70 / 1.20 |
| Move Speed | Slow shuffle |
| Aggro Range | Medium |
| Attack Range | Short poison puff |
| Telegraph Time | 0.8 sec glow before heal pulse |
| Resistances | Poison resistance |
| Weaknesses | Fire, burst, silence/seal effects |
| Drops | Glow Spores, potion materials, support accessories |
| Spawn Rules | 1 behind enemy packs; never more than 2 in early maps |
| Balance Purpose | Teaches target priority and interruption. |

Behavior:

- Periodically heals nearby enemies.
- If threatened, releases weak poison puff.
- Heal can be interrupted by stun, seal, or enough burst damage.

### 18.9.12 Cracked Mimic

| Field | Value |
|---|---|
| Enemy ID | `enemy_cracked_mimic_elite_001` |
| Family | Construct / Treasure |
| Level Range | 15–35 scaling rare elite |
| Role Tags | Elite, ambush, loot |
| HP / Damage / Defense / EXP | 4.00 / 1.50 / 1.20 / 4.00 |
| Move Speed | Medium-fast once active |
| Aggro Range | Triggered by proximity/interact |
| Attack Range | Melee bite, short-range tongue lash |
| Telegraph Time | 1.0 sec chest shake before first attack |
| Resistances | Minor resistance to all control after first stun |
| Weaknesses | Burst windows after failed bite, weak-point marks |
| Drops | Currency burst, rare gear chance, Fracture Dust low chance |
| Spawn Rules | Rare spawn replacing treasure chest or field object |
| Balance Purpose | Adds excitement, rare loot, and mini-boss challenge to field maps. |

Behavior:

- Appears as a cracked chest until approached.
- Opens with a lunging bite.
- If bite misses, becomes vulnerable for 2 seconds.
- Attempts to flee at low HP unless stunned or blocked.

## 18.10 Enemy Packs and Map Composition

Enemy balance depends on combinations, not individual monsters alone.

Example early map compositions:

| Map Level | Pack Composition | Purpose |
|---|---|---|
| 1–5 | 3 Slimelets | Basic movement and attacking. |
| 5–10 | 2 Slimelets + 1 Thorn Sprout | Teaches melee/ranged priority. |
| 8–14 | 2 Mossbacks + 1 Bristle Boar | Teaches durability and charge telegraphs. |
| 12–20 | 3 Dust Imps + 1 Clockbug | Mixes swarm pressure with tank target. |
| 16–24 | 2 Ember Wisps + 2 Mossbacks | Teaches anti-air and elemental resistance. |
| 20–30 | 2 Bandit Cutters + 1 Bandit Thrower | Teaches platform targeting and enemy roles. |
| 24–35 | 1 Oreback Beetle + 2 Glowcap Healers/Plants | Teaches priority and armor break. |

Mobbing-focused classes should shine against larger packs. Bossing-focused classes should shine when packs include tanks, elites, or priority enemies.

## 18.11 Enemy Resistance and Weakness Guidelines

Resistances should create variety without making classes feel useless.

Rules:

- Early enemies should not exceed 30% resistance to a major damage type.
- Bosses and elites can exceed 30% resistance, but must have other vulnerabilities.
- No required story enemy should hard-counter a single class path.
- Resistance should be communicated visually and through tooltips/combat text.
- Weaknesses should reward build diversity.

Prototype resistance categories:

| Resistance Type | Example Enemy | Counterplay |
|---|---|---|
| High defense | Clockbug, Oreback Beetle | Armor break, weak-point marks, magic damage. |
| Fire resistance | Ember Wisp | Non-fire damage, control, marks. |
| Knockback resistance | Mossback, Bristle Boar during charge | Slows, vertical movement, burst. |
| Airborne | Ember Wisp | Ranged attacks, anti-air, pulls. |
| Control resistance | Cracked Mimic after first stun | Burst windows, positioning. |

## 18.12 Drop Table Guidelines

Enemy drops should support gear, crafting, shops, and upgrades.

| Enemy Type | Common Drops | Uncommon Drops | Rare Drops |
|---|---|---|---|
| Basic melee | Currency, basic materials. | Common/Uncommon gear. | Minor upgrade material. |
| Ranged | Projectile materials, light gear. | Gloves, rings. | Crit or Focus accessory. |
| Tank | Heavy materials. | Armor, shields, hammers. | Upgrade catalyst. |
| Flying | Elemental dust. | Mage/Archer gear. | Element trait material. |
| Support | Potion materials. | Utility jewelry. | Cleanse/status accessory. |
| Elite | Currency burst, materials. | Rare gear. | Scar/upgrade materials, cosmetics. |

Drop tables should avoid forcing one enemy farm forever. Materials should have multiple sources, with different efficiency.


# 19. Multiplayer Structure

## 19.1 Social Hubs

Towns are shared multiplayer spaces.

Town features:

- Quest NPCs.
- Vendors.
- Upgrade stations.
- Storage.
- Market access.
- Party finder.
- Cosmetic preview.
- Social areas.

## 19.2 Party Finder

Current prototype status: the browser build includes a local simulated party finder. It fills up to three ally slots from authored prototype members, grants small stat bonuses, and periodically triggers simple assist behaviors such as shielding, damage, mark pressure, or control fields. This proves party-role readability and solo assist pacing, but it does not represent real players, networking, matchmaking, dungeon queues, party authority, invite flow, or server-side buff ownership.

The party finder should support:

- Dungeon queues.
- Boss practice groups.
- Boss clear groups.
- Farm parties.
- Role tags based on party skill category.

Instead of rigid tank/healer/DPS roles, use utility tags:

```text
Defense
Attack Buff
Crit Buff
Control
Mobility
Sustain
Flexible Support
```

## 19.3 Party Buff Visibility

The UI should show:

- Which party buffs are active.
- Duration remaining.
- Buff category.
- Whether another buff is partially suppressed by stacking rules.

## 19.4 Multiplayer Balance Goals

- Parties should clear content faster and safer than solo players.
- Solo players should still be able to progress through most leveling content.
- Bosses can have solo, normal party, and challenge modes.
- No class should be excluded from general content.

---

# 20. Economy and Trading

## 20.1 Economy Goals

The economy should support trading, farming, item chase, and social value without making real-money power or botting the dominant path.

## 20.2 Tradeable Items

| Item Type | Trade Rule |
|---|---|
| Basic gear | Tradeable. |
| Unupgraded rare gear | Tradeable. |
| Materials | Tradeable unless special boss material. |
| Cosmetics | Tradeable if designed as market items. |
| Upgraded gear | Tradeable until certain upgrade threshold, then bound. |
| High-end awakened gear | Account-bound or character-bound. |
| Protection items | Mostly account-bound or earned. |

## 20.3 Binding Rules

Recommended compromise:

- Gear is tradeable before major enhancement.
- Equipping may bind certain items.
- Upgrading beyond a threshold makes gear account-bound.
- Awakening or adding high-end traits makes gear character-bound or account-bound.

## 20.4 Market Options

The game can use:

- Auction house for convenience.
- Player shops for nostalgia/social atmosphere.
- Direct trade with restrictions.

Recommendation:

- Use an auction house for core trade.
- Add player shop cosmetics or stalls later if social identity becomes important.

## 20.5 Anti-Botting and Anti-Abuse

Design considerations:

- Limit raw currency farming efficiency from low-level mobs.
- Make valuable materials come from varied content.
- Add account-level trade restrictions for new accounts.
- Monitor suspicious market behavior.
- Avoid unlimited trade of premium-like upgrade protections.

---


## 20.6 Shops and Vendor Economy

Shops are a core progression safety net. They ensure players can always buy baseline equipment for their level even if drops are unlucky.

Shops should sell:

- Basic weapons for each class.
- Armor for each level tier.
- Offhands after advanced class unlock.
- Low-rarity accessories.
- Consumables.
- Basic upgrade materials.
- Limited weekly materials or token gear from special vendors.

Shops should generally not sell:

- Best-in-slot gear for normal currency.
- High-end protection items in unlimited quantities.
- Relic or Mythic gear for basic currency.
- Upgrade odds boosts for real money.

## 20.7 Shop Types

| Shop Type | Unlock Timing | Inventory | Design Purpose |
|---|---|---|---|
| Starter Outfitter | Level 1 | Training and Tier 1 armor, starter accessories. | Teaches equipment slots. |
| Weapon Smith | Level 1+ | Base class weapons by tier. | Ensures players can replace outdated weapons. |
| Regional Armorer | Each town/region | Armor and boots appropriate to region level. | Provides defensive catch-up gear. |
| Class Supplier | After Level 25 advancement | Class offhands and branch-themed gear. | Supports advanced class identity. |
| Arcane Jeweler | Level 10+ | Rings, amulets, charms. | Offers utility and build adjustment. |
| Consumable Vendor | Level 1+ | Potions, antidotes, repair kits, town scrolls. | Basic survival and convenience. |
| Upgrade Artisan | Level 10+ | Low-tier upgrade materials, restoration services. | Introduces gear progression. |
| Dungeon Quartermaster | Dungeon unlock | Token gear, rare materials, cosmetics. | Gives deterministic dungeon rewards. |
| Event Vendor | Event periods | Cosmetics, limited materials, catch-up gear. | Supports live events without pay-to-win. |
| Reputation Vendor | Region completion | Region-themed gear and cosmetics. | Rewards questing and exploration. |

## 20.8 Vendor Gear Rules

Vendor gear must be balanced carefully.

Rules:

- Basic vendors sell Common and Uncommon gear.
- Regional vendors can sell a small number of Rare items at high currency cost or reputation requirement.
- Dungeon Quartermasters sell Rare gear and some Epic-quality cosmetics or materials through tokens.
- Vendor gear should have predictable stats and fewer trait slots than drops.
- Vendor gear should usually have lower upgrade caps than dungeon/boss gear of the same tier.
- Vendor inventory should update by region and player level band, not random cash-shop style rotation.

Prototype vendor gear limitations:

| Vendor Gear Type | Max Rarity | Max Upgrade Cap | Trait Access |
|---|---|---:|---|
| Starter vendor gear | Common | +5 | None |
| Regional basic gear | Uncommon | +7 to +10 | Minor trait chance only |
| Advanced class vendor gear | Uncommon/Rare | +10 to +12 | One branch-relevant trait on Rare items |
| Dungeon token gear | Rare/Epic | +12 to +15 | One guaranteed trait |
| Reputation gear | Rare | +12 | Region-themed trait |

## 20.9 Example Shop Inventories

### Starter Outfitter: Level 1–10

| Item | Cost Guideline | Requirement | Purpose |
|---|---:|---|---|
| Training Sword / Wand / Bow | Very low | Level 1 class | Replacement starter weapons. |
| Copper Sword / Birch Wand / Simple Bow | Low | Level 5 class | First purchased weapon upgrade. |
| Stitched Vest | Low | Level 5 any | Basic defense. |
| Traveler Boots | Low | Level 5 any | Movement option. |
| Plain Ring | Low | Level 5 any | Teaches accessories. |
| Small Potion | Very low | None | Early sustain. |

### Regional Armorer: Level 15–25

| Item | Cost Guideline | Requirement | Purpose |
|---|---:|---|---|
| Iron Helm | Medium | Level 15 any | Defensive head gear. |
| Reinforced Coat | Medium | Level 15 any | Balanced chest armor. |
| Scout Boots | Medium | Level 15 any | Movement-focused boots. |
| Guard Gloves | Medium | Level 15 any | Defensive gloves. |
| Iron Ring | Medium-high | Level 15 any | HP/defense accessory. |

### Weapon Smith: Level 15–40

| Item | Cost Guideline | Requirement | Purpose |
|---|---:|---|---|
| Iron Sword | Medium | Level 15 Fighter | Balanced Fighter weapon. |
| Iron Axe | Medium-high | Level 15 Fighter | Bossing/elite option. |
| Apprentice Staff | Medium | Level 15 Mage | Area magic option. |
| Oak Longbow | Medium | Level 15 Archer | Precision ranged option. |
| Steel Hammer | High | Level 25 Fighter | Mobbing/control option. |
| Adept Wand | High | Level 25 Mage | Faster casting option. |
| Ashwood Shortbow | High | Level 25 Archer | Mobbing/mobility option. |

### Class Supplier: Level 25+

| Item | Cost Guideline | Requirement | Purpose |
|---|---:|---|---|
| Guardian Tower Shield | High | Level 25 Guardian | Defensive branch item. |
| Berserker War Grip | High | Level 25 Berserker | Aggressive branch item. |
| Duelist Buckler | High | Level 25 Duelist | Timing/crit branch item. |
| Ember Core | High | Level 25 Fire Mage | Heat/burn branch item. |
| Rune-Etched Focus | High | Level 25 Rune Mage | Rune setup branch item. |
| Storm Charm | High | Level 25 Storm Mage | Charge/mobility branch item. |
| Deadeye Scope | High | Level 25 Sniper | Weak-point branch item. |
| Trap Kit | High | Level 25 Trapper | Trap branch item. |
| Beast Token | High | Level 25 Beast Archer | Companion branch item. |

### Dungeon Quartermaster: Token Gear

| Item Type | Currency | Requirement | Purpose |
|---|---|---|---|
| Rare weapon box | Dungeon tokens | Level bracket | Deterministic weapon catch-up. |
| Rare armor piece | Dungeon tokens | Level bracket | Reliable armor progression. |
| Upgrade catalyst | Dungeon tokens | Weekly cap | Controlled upgrade progression. |
| Cosmetic dye/accent | Dungeon tokens | None | Non-power chase. |
| Boss practice charm | Dungeon tokens | Level bracket | Minor defensive utility for boss learning. |

## 20.10 Shop Pricing Guidelines

Prices should be tuned so that players naturally replace gear every level band without grinding excessive currency.

Prototype pricing rules:

```text
Basic vendor weapon cost ≈ currency earned from 20–30 minutes of level-appropriate play.
Basic armor piece cost ≈ currency earned from 10–20 minutes of level-appropriate play.
Accessory cost ≈ currency earned from 20–40 minutes of level-appropriate play.
Class offhand cost ≈ currency earned from 30–45 minutes of level-appropriate play.
```

Dungeon token prices:

```text
Rare token weapon = 3–5 dungeon clears.
Rare token armor = 2–4 dungeon clears.
Upgrade catalyst = 1–2 dungeon clears, with weekly cap if needed.
```

## 20.11 Shop Balance and Player Trust

Shops should make progression feel fair.

- Players should never be stuck using a Level 5 weapon at Level 20 because drops were unlucky.
- Shop gear should reduce frustration but preserve excitement from drops.
- Shop previews should show level requirements, class requirements, stats, trait slots, and upgrade caps.
- Shops should teach stat differences by offering clear options, such as bossing weapon vs mobbing weapon.
- Shop gear should be visually simpler than rare drops so rare gear still feels special.

Example shop choice at Level 15:

| Class | Mobbing Option | Bossing Option | Generalist Option |
|---|---|---|---|
| Fighter | Iron Hammer | Iron Axe | Iron Sword |
| Mage | Apprentice Staff | Apprentice Orb | Apprentice Wand |
| Archer | Ash Shortbow | Oak Longbow | Simple Crossbow |

This gives players agency without overwhelming them.


# 21. Questing, Leveling, and World Structure

## 21.1 Leveling Philosophy

Leveling should combine:

- Main quests.
- Field combat.
- Party dungeons.
- Bosses.
- Class trials.
- Optional exploration objectives.

The game should not rely entirely on repetitive mob grinding, but grinding can remain a valid progression path.

## 21.2 Main Quest Structure

Each region has:

- Arrival quest.
- Local conflict.
- Dungeon unlock.
- Boss encounter.
- Region material unlock.
- Class or gear upgrade tutorial where relevant.

## 21.3 Class Trials

At Level 20–25, players complete a trial before selecting an advanced class.

The trial previews the full advanced skill batch instead of unlocking individual skills. It teaches:

- The branch’s resource twist.
- Main skill loop.
- Party skill fantasy.
- Example investment paths, such as damage-focused, utility-focused, or party-support-focused builds.

## 21.4 Daily and Weekly Content

Daily content should be useful but not exhausting.

Examples:

| Content | Reward |
|---|---|
| Daily field challenge | EXP, materials. |
| Daily dungeon bonus | Upgrade materials. |
| Weekly boss clear | Rare materials, gear. |
| Weekly class challenge | Skill mastery currency. |
| Weekly account roster objective | Cosmetic progress, small roster currency. |

Avoid creating too many mandatory chores.

---

# 22. UI / UX Direction

## 22.1 Core UI Requirements

Combat UI must show:

- HP.
- Class resource.
- Skill cooldowns.
- Party buff status.
- Active debuffs.
- Gear trait triggers where relevant.

Current prototype status:

- The HUD shows HP, MP, class resource, XP, level, coins, class name, hotbar/menu access, cooldowns, buffs, minimap, and quest/map context through canvas windows.
- The UI is intentionally compact and client-like, with draggable canvas windows, modal/panel fallbacks, and dense RPG information rather than landing-page presentation.
- Equipment, inventory, shop, upgrade, skill tree, world map, party/self-buff, keybind, quest, dungeon, trial, accomplishment, log, and admin panels are implemented.
- The equipment window places equipment icons on the left and a single compact stat column on the right.
- The world map is a dedicated panel/keybind rather than being nested under Character.
- Reset behavior clears skill keybinds while preserving non-skill defaults.

## 22.2 Class Resource UI

Each class resource should have a distinct visual identity:

| Class | Resource UI |
|---|---|
| Fighter | Momentum bar with impact pulses. |
| Mage | Energy orb/bar with branch-specific transformation. |
| Archer | Focus meter with mark indicators. |

Branch examples:

- Guardian Stored Impact appears as shield segments.
- Berserker Rage appears as a red rising gauge.
- Fire Mage Heat appears as a thermometer-like gauge.
- Rune Mage shows active rune count.
- Sniper Aim shows charge/steady indicators.
- Trapper Preparation shows trap slots and trigger readiness.

Current prototype note: advanced resources share the main class resource meter location and change label/color by selected branch. More specialized per-branch resource widgets are future UI polish.

## 22.3 Skill Tree UI

The class tree UI should be simple at the top level:

```text
Starting Class → Advanced Class → Specialization → Mastery
```

The skill UI underneath should be organized by batches:

```text
Base Skill Batch → Advanced Skill Batch → Specialization Skill Batch → Mastery Skill Batch
```

Each skill node should clearly show:

- Current rank and maximum rank.
- Cost to learn the next rank.
- What changes at the next rank.
- Major breakpoint ranks.
- Whether the skill is active, passive, party-oriented, or resource-oriented.
- Combat role tags: Mobbing, Bossing, Hybrid, Control, Support, or Party.
- Whether the node is visible, trainable, usable, or locked by prerequisites.
- Missing prerequisite ranks, such as “Requires Burning Mark Rank 5.”
- Example build tags such as Defense, Burst, Mobility, Control, Party, Resource, Mobbing, or Bossing.

Players should be able to preview:

- Full skill batches before committing to an advanced class.
- Skill prerequisite chains and recommended investment routes.
- Mobbing-focused, bossing-focused, and balanced sample builds.
- Resource twist.
- Party skill and its rank scaling.
- Specialization options.
- Example playstyle and recommended investment paths.

## 22.4 Upgrade UI

The upgrade UI is critical. It should clearly show:

- Upgrade level.
- Required materials.
- Success chance.
- Failure outcomes.
- Possible scars/mutations.
- Protection items.
- Binding consequences.
- Confirmation step for high-risk upgrades.

Current prototype status:

- The upgrade station shows a detailed chance table rather than a single upgrade button.
- It displays material costs, coin costs, selected item state, and outcome chances before the player commits.
- Implemented outcomes are success, great success, fail, downgrade, scar, fracture, and rare break.
- Scar and fracture outcomes are represented as item state and materials in the prototype; mutation and full protection-item systems remain future work.

Future upgrade UI work should add stronger before/after stat previews, protection item selection, pity/fairness messaging, and clearer explanations for scars, fractures, and irreversible risk.

## 22.5 Party Finder UI

Party finder should show:

- Class.
- Advanced class.
- Party skill category.
- Player role tags.
- Gear score or recommended power, if used.
- Practice/clear/farm intent.

Current prototype status: party finder is implemented as a local simulated-party panel. It shows available ally slots, fills those slots with authored prototype allies, allows dismissing the party, and exposes the party state in both DOM and Canvas UI paths. The current UI intentionally does not show real queue intent, gear score, live player status, invites, or server-authenticated party membership.

## 22.6 First-Session Guide

The prototype should help new players understand the first loop without turning the page into a tutorial landing page. The current implementation uses a compact guide strip with persistent local milestones:

- Choose a class.
- Open the world map.
- Travel to Greenroot Meadow.
- Defeat an enemy.
- Pick up loot.
- Equip an item.
- Rank a skill.
- Attempt an upgrade.
- Find prototype allies.
- Start a class trial.
- Choose an advanced class.
- Clear a dungeon.

The guide can be hidden, but completed step ids remain in the local save. Future work should connect these milestones to richer quest copy, NPC prompts, and contextual station highlights.

---

# 23. Art Direction

## 23.1 Starlit Frontier Fantasy Style Bible

**Canonical style phrase:** Starlit Frontier Fantasy.

Project Starfall should look like a welcoming side-scrolling MMO frontier shaped by fallen-star magic. The target is AI-generated 2D fantasy art with painterly environments, clean cel-shaded sprite assets, crisp contour outlines, luminous star-magic accents, and equipment-driven character identity.

Core style pillars:

- **Luminous frontier:** towns use warm lanterns, wood, stone, banners, market awnings, forge smoke, and blue-gold Starfall guild motifs.
- **Practical magic:** magic appears in gear, upgrade tools, rune plates, crystal cores, portal rings, station props, and terrain details rather than only as abstract glow.
- **Readable side-scroller:** silhouettes, walkable platforms, enemy tells, projectiles, loot, and damage text must read instantly at gameplay scale.
- **Equipment identity:** playable characters use a shared compact body, while weapons, armor layers, class VFX, cosmetics, and damage splats create visual variety.
- **Cozy danger:** safe zones should feel social and inviting; field routes should become more hazardous through sharper shapes, stronger contrast, and unstable magic.

Signature visual motifs:

- Fallen stars, star fragments, star-white spark cores, and fractured halos.
- Portal rings, rune circles, floating glyph plates, and blue crystal inlays.
- Guild banners, lantern arches, carved stone, honey wood, brass gearwork, and arcane shop devices.
- Regional landmarks that make each route identifiable at a glance.

Core palette:

| Use | Colors |
|---|---|
| Global darks | midnight navy, deep blue, blue-black shadow |
| Global lights | star-white, lantern gold, crystal cyan |
| Frontier materials | warm tan stone, honey wood, brass, worn leather |
| Magic accents | cyan portals, gold runes, pale violet astral light |

Regional palette families:

| Region Family | Palette |
|---|---|
| Greenroot | soft greens, flower yellow, pale sky blue, bark brown |
| Rustcoil | brass, teal glow, gray stone, dark oil accents |
| Cinder | basalt black, ember orange, smoke gray, molten gold |
| Frostfen | ice blue, muted pine, silver, pale cyan |
| Stormbreak | blue-gray, violet storm light, white lightning |
| Astral | indigo, luminous cyan, parchment gold, soft violet |
| Eclipse / Rift | black-violet, pale corona gold, void magenta, cold white |

Do:

- Use readable silhouettes, broad value blocks, crisp edges, and controlled glow.
- Keep UI panels dark navy with gold, star-white, and crystal-cyan accents.
- Make magic look like a craft, tool, material, or machine the world understands.
- Let rarity, upgrades, scars, and cosmetics add clear visual accents without obscuring gameplay.

Avoid:

- Photorealism, muddy dark fantasy, generic procedural-looking shapes, and excessive texture noise.
- Chibi proportions, oversized anime heads, or tiny unreadable sprite bodies.
- Baked text, labels, logos, UI lettering, or watermarks in generated assets.
- Modern sci-fi UI language that looks disconnected from guilds, runes, portals, and frontier craft.

## 23.2 Visual Style

Stylized 2D fantasy with clear silhouettes.

Avoid a chibi look. Characters can be readable, expressive, and appealing, but should feel distinct from existing cute side-scrolling RPGs.

Current prototype direction: characters are simple, blocky, and pixel-readable. The prototype deliberately uses a generic player body and lets equipped items provide most visual identity. This should remain the near-term direction until the equipment-driven system is validated.

## 23.3 Character Proportions

Recommended:

- Compact but readable blocky characters.
- Strong weapon silhouettes.
- Clear armor and equipment overlays.
- Shared body animation with class and gear variation layered on top.
- Cosmetic gear visible without making readability messy.

Current animation coverage:

- Player states: idle, run, jump, fall, climb, basic attack, skill, party/buff cast, hit, defeat.
- Weapon action differences: melee swing, bow release, spell cast, and buff cast should read differently.
- Climb animation should play only while moving on a ladder.
- Enemy states: idle, move, telegraph, attack, projectile, buff, hit, defeat.

## 23.4 Skill Effects

Skill effects should be:

- Flashy but readable.
- Color-coded by branch.
- Distinct by class resource.
- Scalable in opacity for party content.

Examples:

| Class | Visual Identity |
|---|---|
| Guardian | blue/steel shields, shockwaves, impact rings. |
| Berserker | red/orange slashes, rage pulses. |
| Fire Mage | fire, ash, heat shimmer. |
| Rune Mage | geometric glyphs, glowing circles. |
| Storm Mage | lightning chains, shocked bursts, static forks. |
| Sniper | clean aim lines, weak-point flashes. |
| Trapper | wires, trap symbols, mechanical/nature devices. |
| Beast Archer | companion arrows, claw sparks, pack calls. |

## 23.5 Gear Visuals

Gear should visually change through:

- Weapon shape.
- Glow effects.
- Upgrade aura.
- Scars or cracks.
- Elemental infusions.
- High-end cosmetic skins.

Scars and mutations can be represented with subtle effects, not always permanent ugly visuals.

Current prototype status:

- Gear affects the character through code-drawn runtime attachments and equipment-layer sheets.
- Weapons, shields, armor, boots, rings, amulets, class items, bows, wands, staffs, and trap kit visuals exist.
- Dropped items use icon-only presentation with a rarity aura around the icon.
- Future work should improve armor silhouette variety, upgrade/scar visual language, and final item-set cohesion.

---

# 24. Audio Direction

Current prototype status: the browser build includes lightweight procedural Web Audio cues for core actions. These are functional feedback sounds, not final sound design. They cover UI confirmation, basic attacks, skill casts, buff casts, loot pickup, level-up, upgrade success/failure, player damage, enemy defeat, map travel, and simulated party assists. Music, region loops, authored sound assets, mix settings beyond a simple toggle/volume state, and final class-specific audio libraries remain future work.

## 24.1 Music

Music should support bright magical adventure.

Region styles:

- Towns: warm, social, memorable melodies.
- Fields: upbeat exploration loops.
- Dungeons: tense but not oppressive.
- Bosses: energetic and readable combat themes.

## 24.2 Skill Sounds

Each class needs distinctive audio identity:

| Class | Audio Feel |
|---|---|
| Fighter | impacts, metal, weight, shockwaves. |
| Mage | magical tones, crackles, bursts, chimes. |
| Archer | strings, snaps, wind, trap clicks. |

## 24.3 Upgrade Sounds

Upgrade attempts need strong audio feedback:

- Material placement.
- Charge-up tension.
- Success chime.
- Great success flourish.
- Scar/mutation distortion.
- Fracture crack.
- Break sound with valuable material drop feedback.

The sound design should make upgrades feel dramatic.

---

# 25. Monetization Philosophy

## 25.1 Core Rule

No pay-to-win.

Players should not be able to buy:

- Upgrade success chance.
- Gear power.
- Boss clears.
- Combat stats.
- Best-in-slot items.
- Exclusive power progression.

## 25.2 Acceptable Monetization

| Monetization | Notes |
|---|---|
| Cosmetic outfits | Account-wide where possible. |
| Weapon skins | Visual only. |
| Skill effect skins | Must preserve combat readability. |
| Emotes | Social. |
| Mount cosmetics | Visual/travel only, if mounts exist. |
| Battle pass cosmetics | No power. |
| Name changes | Service fee. |
| Extra cosmetic presets | Convenience only. |
| Character slots | Acceptable if base slots are generous. |

## 25.3 Risky Monetization to Avoid

Avoid:

- Paid upgrade protections.
- Paid loot boxes with power.
- Paid drop-rate boosts that affect economy.
- Paid stat pets.
- Paid convenience that becomes mandatory.
- Selling materials that affect endgame gear.

## 25.4 Character Slots

Because multiple characters are part of the game, the free account should include enough slots to enjoy the core system.

Prototype:

```text
Free slots: 6
Purchasable extra slots: cosmetic/convenience, up to account cap
```

This lets players try all MVP branches without paying.

---

# 26. Prototype Scope And MVP Definition

The current browser prototype has moved beyond the original six-advanced-class MVP. It now validates the core local gameplay loop across all base classes and all nine advanced branches, but it is not yet an MMO MVP because true multiplayer, account services, trading, music, and server authority are not implemented.

## 26.1 Current Prototype Scope

Implemented:

- Static `/project-starfall` route with no framework dependency.
- Fighter, Mage, Archer base classes.
- Guardian, Berserker, Duelist, Fire Mage, Rune Mage, Storm Mage, Sniper, Trapper, Beast Archer advanced classes.
- Primary training skill for every advanced class.
- Base and advanced skill ranks, prerequisites, role tags, and party/self-buff skills.
- Canvas movement, platforms, ladders, enemy pathing, projectiles, chain lightning, traps, fields, melee, magic, arrows, drops, loot, inventory, equipment, shop, upgrades, save/load/reset, and world map.
- First-session guide milestones, simulated local party finder allies, and procedural action feedback audio.
- Generated assets for characters, enemies, maps, items, stations, skill icons, equipment layers, and FX.

Not yet implemented:

- True multiplayer, real-player parties, matchmaking queues, or server-side party buffs.
- Music, authored SFX libraries, or production audio mixing.
- Production economy, trading, market, or anti-abuse systems.
- Full account roster, shared storage, monetization, cosmetics, or seasonal framework.
- Production-ready quest writing, boss scripting, encounter telegraphs, or content pacing.

## 26.2 MVP Reframing

The next MVP should prove:

1. The first 30 minutes are understandable and fun.
2. Each base class has a readable identity before advanced selection.
3. Every advanced class has a satisfying primary training loop.
4. Mobbing and bossing differences are visible but not punitive.
5. Gear drops, shop gear, and upgrades form a trustworthy progression floor.
6. The UI supports repeated play without overwhelming new players.

Multiplayer should not be treated as MVP-complete until the solo combat and progression loop is strong enough to justify networking and service complexity.

## 26.3 MVP Content Target

| Area | Current / Target |
|---|---|
| Town hub | Starfall Crossing implemented. |
| Base classes | Fighter, Mage, Archer implemented. |
| Advanced classes | All nine branches implemented in prototype form. |
| Field maps | Early, mid, dungeon, and late training paths implemented; pacing still needs tuning. |
| Dungeons / bosses | Dungeon maps and boss enemies exist; encounter scripting needs depth. |
| Gear | Shop gear, drops, requirements, equipment visuals, and upgrades implemented. |
| UI | Broad prototype UI implemented; onboarding and clarity need polish. |
| Save/load | LocalStorage prototype save implemented. |
| Multiplayer | Local simulated party placeholder implemented; true multiplayer remains future work. |

---

# 27. Roadmap

## 27.1 Immediate: Prototype Polish

Goals:

- Improve first-session onboarding from class select through first map, first skill rank, first item drop, first equipment change, and first upgrade.
- Tune the new guide strip so it complements quests and station prompts without covering combat information.
- Tune simulated party assists so they demonstrate support value without trivializing solo combat.
- Balance primary training skills so each advanced class can level comfortably without erasing its mobbing/bossing identity.
- Tune Chain Bolt against other mobbing and bossing tools so Storm Mage remains strong but not uniquely dominant.
- Add clearer UI explanations for advanced choice permanence, class trials, primary training skills, and party skills as current self-buffs.
- Improve early maps, ladders, enemy spawn pacing, loot readability, and safe-zone station prompts.
- Review all generated assets for scale, readability, silhouette, and equipment layering consistency.

## 27.2 Next Milestone: Vertical Slice

Goals:

- Create a polished Level 1-30 flow with one complete route from Starfall Crossing through advanced class selection.
- Convert dungeons from map variants into authored encounters with boss phases, telegraphs, rewards, and retry flow.
- Expand class trials into meaningful branch previews rather than simple gates.
- Add stronger quest structure, route goals, and map unlock messaging.
- Add richer upgrade previews, protection item placeholder behavior, and better scar/fracture explanations.
- Harden save compatibility before adding more persistent systems.

## 27.3 Later: Systems Depth

Goals:

- Add deeper gear tiers, class-reactive traits, crafted items, deterministic fallback progression, and boss-specific drops.
- Expand Duelist, Storm Mage, and Beast Archer beyond compact prototype kits.
- Add specialization or mastery layers only after advanced classes feel complete.
- Add account roster traits, shared storage, and alt-friendly progression.
- Replace procedural audio cues with authored hit sounds, skill sounds, upgrade sounds, mix controls, and region music.

## 27.4 Multiplayer And Economy

Goals:

- Add multiplayer only after combat, roles, bossing, and UI are stable in solo prototype.
- Define party authority, buff ownership, encounter scaling, revive/fail states, and party finder UX.
- Add market/trading only after item binding, rarity, upgrade failure, and anti-abuse rules are finalized.
- Keep monetization cosmetic-only and never tied to upgrade odds, combat power, boss clears, or best-in-slot gear.

## 27.5 Stretch / Launch Candidate

Goals:

- Controller and Steam Deck support.
- World bosses, raids, seasonal events, cosmetics, emotes, and social hub features.
- Production backend, accounts, telemetry, moderation, anti-botting, and live balance tooling.
- Final art pass replacing or refining prototype-generated assets where needed.

---

# 28. Balance Principles

## 28.1 Class Balance

Every class should be good at something.

No class should be:

- Best at all damage types.
- Best at all party support.
- Required for all bosses.
- Useless in solo play.
- Only valuable because of one buff.

## 28.2 Skill Batch Balance

Skill batches should give players meaningful choice without overwhelming them.

Rules:

- A new batch should expose the full class direction immediately.
- Skill points should be limited enough that choices matter.
- Mandatory-feeling skills should either be free baseline actions or have low early investment cost.
- Party skills should be useful at Rank 1 but meaningfully better with investment.
- Rank breakpoints should be clear in the UI so players understand why they are investing.
- No build should be invalid because it did not perfectly spend points while leveling; respecs should be accessible during early testing and early game.

## 28.3 Mobbing vs. Bossing Balance

Mobbing and bossing roles should be explicit but not absolute.

Rules:

- Mobbing-focused classes should level efficiently and feel satisfying in maps, waves, and farming content.
- Bossing-focused classes should have stronger single-target scaling, burst timing, weak-point payoff, or sustained uptime.
- Every class must have at least one acceptable mobbing route and one acceptable bossing route through skill investment.
- Skill prerequisites should allow players to invest toward either direction before the next batch opens.
- Content should rotate incentives: some dungeons favor area damage, some bosses favor burst, and some encounters reward control or survival.
- No class should be rejected from basic party content because it is not the optimal mobbing or bossing pick.

## 28.4 Skill Prerequisite Balance

Prerequisites should create satisfying progression, not trap builds.

Rules:

- The simplest core skill in each batch should be trainable immediately.
- Prerequisites should usually require Rank 3 or Rank 5; Rank 10 requirements should be reserved for finishers, capstones, and major breakpoints.
- Prior-batch max requirements should be rare and thematic.
- Each batch should offer at least two readable paths, such as a mobbing path and a bossing path.
- The player should be able to preview locked skills and understand exactly how to unlock them.
- Early-game respecs should be cheap enough that a player can correct mistaken investments.

## 28.5 Party Skill Balance

Party skills should be desirable but not mandatory.

Balancing tools:

- Cooldowns.
- Duration.
- Buff categories.
- Range limitations.
- Encounter-specific strengths.
- Self vs party scaling.
- Diminishing returns.

## 28.6 Gear Upgrade Balance

Upgrade progression should be exciting, but not abusive.

Rules:

- Low-level upgrades should be forgiving.
- High-level upgrades should be risky.
- Severe failure should often produce useful materials.
- Players should understand risks.
- Real money should not improve odds.

## 28.7 Solo vs Party Balance

Solo play:

- Viable for leveling and basic farming.
- Slower or harder for elite content.
- Allows all classes to progress.

Party play:

- Faster dungeon clears.
- Safer bosses.
- Better use of buffs and synergies.
- Access to higher challenge rewards.

## 28.8 Alt Balance

Multiple characters should provide:

- More playstyle options.
- Limited account benefits.
- Shared progression convenience.
- Cosmetic and achievement rewards.

Multiple characters should not provide:

- Uncapped stat stacking.
- Mandatory daily chores across many alts.
- Huge advantage impossible for single-main players to approach.

---

# 29. Risks and Mitigations

## 29.1 Risk: Classes Feel Too Generic

Mitigation:

- Keep names intuitive, but make mechanics distinct.
- Make resources central to class identity.
- Give each branch a signature loop, a primary training skill, and a party/self-buff.
- Use class-reactive gear traits.
- Preserve visible differences between melee swings, bow attacks, spell casts, traps, fields, and buffs.

Current prototype note: all nine advanced branches now exist, which raises the bar for making every class feel worthwhile. The next balance pass should compare each branch's primary training skill, bossing option, utility value, and resource loop against Chain Bolt's high mobbing impact.

## 29.2 Risk: UI Becomes Too Dense

Mitigation:

- Keep the client-like density, but improve onboarding and default panel layout.
- Introduce concepts in order: movement, attack, first drop, equip, first skill point, map travel, upgrade.
- Keep world map, equipment, stats, skills, keybinds, and upgrade details discoverable through stable buttons and default keys.
- Avoid adding new panels until existing panels are readable at common laptop resolutions.

## 29.3 Risk: Upgrade System Feels Too Punishing

Mitigation:

- Use scars, mutations, and fracture materials.
- Show risks clearly.
- Add protection items earned through gameplay.
- Add long-term restoration paths.
- Make severe failures rare and meaningful.

Current prototype note: the upgrade station already shows outcome chances and cost. Future work should add protection-item preview, pity/catch-up explanation, and clearer post-failure recovery paths before higher upgrade tiers are balanced.

## 29.4 Risk: Too Much Scope

Mitigation:

- Treat the current prototype as an exploration build, not as locked production scope.
- Freeze the next playable milestone around a polished Level 1-30 route.
- Delay specializations until base and advanced class loops are proven.
- Build one strong vertical slice first.
- Avoid PvP at launch.
- Keep future party size at 4 unless encounter design proves otherwise.

## 29.5 Risk: Multiple Characters Feel Like Chores

Mitigation:

- Limit roster trait slots.
- Avoid daily chores per character.
- Add account-wide catch-up.
- Make alts fun through different class playstyles, not required stats.

## 29.6 Risk: Prototype Art Direction Drifts

Mitigation:

- Keep the player body generic, compact, blocky, and readable.
- Let equipment, weapons, class effects, portraits, and skill VFX carry identity.
- Review every generated asset at gameplay scale before accepting it.
- Maintain a stable animation rig so new gear layers do not require redrawing the whole character.

## 29.7 Risk: Economy Abuse

Mitigation:

- Bind high-end upgraded gear.
- Restrict new account trading.
- Avoid paid power items.
- Monitor high-value materials.
- Use sinks for currency and materials.

Current prototype note: there is no live economy yet. Economy design should wait until item rarity, upgrade risk, binding, and deterministic progression rules are stable.

## 29.8 Risk: Multiplayer Is Added Before The Solo Loop Is Ready

Mitigation:

- Validate combat feel, UI readability, class identity, and gear progression locally first.
- Define server authority and party buff rules before implementing online combat.
- Treat party/self-buffs in the prototype as UX placeholders, not proof that networked party combat is solved.

---

# 30. Open Design Questions

These do not need to be solved immediately, but they should be decided before full production.

1. Should the next playable target be a polished Level 1-30 vertical slice, or a broader systems sandbox with less authored pacing?
2. Should advanced class choice remain fully permanent, or should testing include a rare reset path?
3. Which of the current maps are canonical progression content, and which are prototype test maps?
4. How close should the final visual language stay to blocky side-scroller RPG inspiration versus a more original custom style?
5. Should each advanced class have exactly one primary training skill, or can some branches rely on a low-cooldown combo of two skills?
6. Should Storm Mage's Chain Bolt stay as the defining primary mobbing skill, or should its power be split across setup and payoff skills?
7. How much enemy pathing complexity is necessary before multiplayer work begins?
8. Should class trials be solo-only, party-assisted, or split between solo tutorial and party challenge versions?
9. Which upgrade failure states should be reversible, and which should become permanent item identity?
10. Should gear have fully visible stat ranges, hidden rolls, or both?
11. Should the market be auction-house only, player-shop only, or both?
12. Should boss difficulty scale by party size?
13. How many daily or weekly tasks are acceptable before the game feels like a chore?
14. Should companion-style skills such as Beast Archer become true pet actors, or remain coordinated attack effects?
15. What is the minimum audio pass required before the prototype feels representative?

---

# Appendix A: Skill Summary Tables

## A.1 Base Class Skills

| Class | Skill | Role |
|---|---|---|
| Fighter | Heavy Strike | Basic attack |
| Fighter | Dash Slash | Mobility / damage |
| Fighter | Guard | Defense |
| Fighter | Ground Slam | Area damage |
| Fighter | Power Break | Debuff |
| Fighter | Momentum Burst | Finisher |
| Mage | Magic Bolt | Basic attack |
| Mage | Blink | Mobility |
| Mage | Arcane Burst | Area damage |
| Mage | Mana Shield | Defense |
| Mage | Spell Mark | Setup |
| Mage | Energy Release | Finisher |
| Archer | Quick Shot | Basic attack |
| Archer | Roll Shot | Mobility / damage |
| Archer | Marked Shot | Setup |
| Archer | Piercing Arrow | Damage |
| Archer | Eagle Stance | Buff |
| Archer | Focused Volley | Finisher |

## A.2 Skill Batch Summary

| Batch | Opens At | Skill Examples | Player Activity Until Next Batch |
|---|---:|---|---|
| Base Skill Batch | Level 1 | Heavy Strike, Blink, Marked Shot, class finishers | Invest points into base combat, mobility, defense, and resource flow. |
| Advanced Skill Batch | Level 25 | Guardian skills, Fire Mage skills, Sniper skills, party skills | Invest points into the advanced class loop, party skill, resource twist, and finisher. |
| Specialization Skill Batch | Level 60 | Specialization skills and modifiers | Deepen a chosen branch identity and create build-defining variations. |
| Mastery Skill Batch | Level 100+ | Mastery nodes and late-game modifiers | Refine endgame builds, add prestige power, and customize skill behavior. |

## A.3 Advanced Class Signature Skills

| Class | Primary / Signature Combat Skill | Party / Buff Skill |
|---|---|---|
| Guardian | Shield Bash / Retaliation Wave | Shield Wall |
| Berserker | Blood Cleave / Last Stand | War Cry |
| Duelist | Quick Cut / Riposte | Rallying Flourish |
| Fire Mage | Heat Vent / Inferno Burst | Ignition Aura |
| Rune Mage | Rune Detonation | Rune Circle |
| Storm Mage | Chain Bolt | Stormfront |
| Sniper | One Perfect Shot | Eagle Eye |
| Trapper | Kill Zone | Tactical Field |
| Beast Archer | Companion Strike / Coordinated Assault | Pack Call |

---

# Appendix B: Buff Category Rules

## B.1 Buff Categories

| Category | Examples |
|---|---|
| Attack Buff | War Cry, Power Rune |
| Critical Buff | Eagle Eye, Rallying Flourish |
| Defense Buff | Shield Wall, Guard Rune |
| Speed Buff | Stormfront, Haste Rune |
| Sustain Buff | Pack Call, War Cry lifesteal |
| Field Buff | Rune Circle, Tactical Field, Ignition Aura |
| Enemy Debuff | Pierce Armor, Power Break, Weak Point Mark |
| Resource Buff | Focus Rune, Rallying Flourish resource gain |

## B.2 Stacking Examples

### Example 1: War Cry + Power Rune

War Cry and Power Rune are both Attack Buffs.

Suggested behavior:

- Stronger buff applies at full value.
- Weaker buff applies at 25–50% value.

### Example 2: Eagle Eye + Rallying Flourish

Both provide crit-related benefits, but Rallying Flourish also provides haste and Tempo support.

Suggested behavior:

- Highest critical chance buff applies fully.
- Second critical chance buff applies partially.
- Haste from Rallying Flourish applies normally unless another Speed Buff is active.

### Example 3: Shield Wall + Guard Rune

Both are defense buffs.

Suggested behavior:

- Both can apply, but total damage reduction is subject to a cap.
- Example cap: 35% general damage reduction from temporary buffs.

---

# Appendix C: Upgrade Outcome Tables

## C.1 Example Reinforce Table

| Upgrade Range | Success | Great Success | Downgrade | Scar | Fracture / Break |
|---|---:|---:|---:|---:|---:|
| +0 to +5 | High | Low | None | None | None |
| +6 to +10 | Medium | Low | Low | Low | None |
| +11 to +15 | Medium-low | Low | Medium | Medium | Very low |
| +16 to +20 | Low | Low | Medium | Medium | Low |

Exact rates must be tuned through testing.

## C.2 Example Scar Types

| Scar | Drawback | Upside |
|---|---|---|
| Cracked | Slightly reduced durability or defense | Higher resource generation on skill use |
| Burnt | Minor self-damage on some triggers | Stronger fire/burn effects |
| Heavy | Reduced movement | Increased stagger or impact damage |
| Unstable | Random resource fluctuation | Chance for bonus skill trigger |
| Faded | Lower base stat | Reduced upgrade material cost |

## C.3 Example Mutation Types

| Mutation | Effect |
|---|---|
| Echoing | Skills have a chance to repeat at reduced power. |
| Splintering | Attacks can split into smaller secondary hits. |
| Guarded | Defensive skills grant minor resource. |
| Burning | Attacks can apply burn. |
| Focused | Marks, runes, or weak points last longer. |
| Volatile | Higher damage but increased resource cost. |

---

# Appendix D: Glossary

| Term | Definition |
|---|---|
| Advanced Class | The Level 25 branch chosen from a starting class. |
| Batch Range | The level range between one skill batch opening and the next. |
| Branch Resource | Modified version of a base class resource. |
| Class-Reactive Trait | Item trait that behaves differently depending on class. |
| Energy | Mage base class resource. |
| Focus | Archer base class resource. |
| Momentum | Fighter base class resource. |
| Mutation | Upgrade outcome that changes or adds unusual item behavior. |
| Party Skill | Class skill that benefits the caster and nearby allies. Party skills are trainable nodes within the Advanced Skill Batch. |
| Rank | A skill's investment level. Rank 1 makes a skill usable; higher ranks improve its effects or add breakpoints. |
| Roster Trait | Limited account-wide bonus unlocked by leveling a character. |
| Scar | Upgrade failure result with both drawback and possible upside. |
| Skill Batch | A group of skills that becomes available together at a milestone level and is improved through earned skill points. |
| Skill Point | Currency earned through leveling and selected challenges that is spent to learn or rank up skills. |
| Skill Prerequisite | A required skill rank or prior investment needed before another visible node becomes trainable. |
| Trainable | A skill node that can currently receive skill points because its prerequisites are met. |
| Visible | A skill node shown in the batch UI, even if it is not yet trainable. |
| Mobbing | Combat focused on clearing many regular enemies, waves, maps, or farming routes. |
| Bossing | Combat focused on high-value single targets, burst windows, weak points, and sustained boss damage. |
| Specialization | Level 60 branch that deepens advanced class identity. |
| Stored Impact | Guardian branch resource generated from blocked damage. |
| Heat | Fire Mage branch resource generated by fire spells. |
| Aim | Sniper branch resource generated through precision and range. |
| Preparation | Trapper branch resource generated through traps and control. |

---


# Appendix E: Skill Role and Prerequisite Reference

## E.1 Combat Role Tag Definitions

| Tag | Use Case | Example Skills |
|---|---|---|
| Mobbing | Clearing many enemies, waves, farming maps, or dungeon rooms. | Ground Slam, Wildfire, Chain Bolt, Kill Zone. |
| Bossing | Single-target damage, burst windows, weak points, armor break, or sustained boss pressure. | Power Break, Last Stand, One Perfect Shot, Execution Shot. |
| Hybrid | Good in both mobbing and bossing but usually not best-in-slot for either. | Momentum Burst, Fireball, Rune Detonation, Coordinated Assault. |
| Control | Enemy movement, slows, roots, zoning, stuns, grouping, or area denial. | Snare Trap, Ground Glyph, Mana Seal, Tactical Field. |
| Support | Survival, resource generation, shields, healing, cleanse, utility, or party contribution. | Shield Wall, Rune Circle, Pack Call, Stormfront. |
| Party | Explicitly benefits nearby party members in addition to the caster. | War Cry, Eagle Eye, Rallying Flourish. |

## E.2 Advanced Class Role Summary

| Class | Mobbing Strength | Bossing Strength | Party / Utility Strength | Intended Identity |
|---|---:|---:|---:|---|
| Guardian | Medium | Medium-high | High | Protects allies, survives heavy mechanics, converts defense into offense. |
| Berserker | Medium | High | Medium | Risky burst damage and sustain, especially during low-health windows. |
| Duelist | Medium | High | Medium | Timing-based uptime, crit tempo, and single-target pressure. |
| Fire Mage | High | Medium | Medium | Excellent area damage, burn spread, and Heat burst. |
| Rune Mage | Medium-high | Medium | High | Flexible setup caster with control, support, and combo payoff. |
| Storm Mage | High | Medium | Medium | Fast clearing, mobility, chains, and speed support. |
| Sniper | Low-medium | High | Medium | Best precision bossing and crit support. |
| Trapper | High | Medium-low | High | Control-heavy mobbing, safe zones, and positional utility. |
| Beast Archer | Medium | Medium-high | Medium-high | Companion-based sustained damage, solo consistency, and light sustain. |

## E.3 Example Skill Investment Paths

These examples show how prerequisite chains can steer players toward different builds without unlocking individual skills at fixed character levels.

### Fighter Base Batch

```text
Mobbing path:
Heavy Strike R5 → Ground Slam R5 → Momentum Burst R1+

Bossing path:
Heavy Strike R5 → Guard R5 → Power Break R5 → Momentum Burst R1+
```

### Mage Base Batch

```text
Mobbing path:
Magic Bolt R5 → Arcane Burst R5 → Energy Release R1+

Bossing path:
Magic Bolt R5 → Spell Mark R5 → Energy Release R1+
```

### Archer Base Batch

```text
Mobbing path:
Quick Shot R5 → Piercing Arrow R5 → Focused Volley R1+

Bossing path:
Quick Shot R3 → Marked Shot R5 → Eagle Stance R5 → Focused Volley R1+
```

### Guardian Advanced Batch

```text
Protection path:
Impact Guard R5 → Oath Barrier R5 → Shield Wall R5+

Counterattack path:
Impact Guard R5 → Retaliation Wave R5 → Guardian’s Verdict R1+
```

### Berserker Advanced Batch

```text
Mobbing path:
Blood Cleave R5 → Reckless Leap R5 → War Cry R1+

Bossing path:
Blood Cleave R5 → Rage Surge R5 → Pain to Power R5 → Last Stand R1+
```

### Fire Mage Advanced Batch

```text
Mobbing path:
Fireball R5 → Burning Mark R5 → Wildfire R5 → Ignition Aura R1+

Bossing path:
Fireball R5 → Heat Vent R5 → Inferno Burst R1+ → Ignition Aura R5+
```

### Sniper Advanced Batch

```text
Bossing path:
Aimed Shot R5 → Weak Point Mark R5 → Execution Shot R5 → One Perfect Shot R1+

Party crit path:
Weak Point Mark R5 → Eagle Eye R5+ → Pierce Armor R5
```

### Trapper Advanced Batch

```text
Mobbing path:
Snare Trap R5 → Spike Trap R5 → Tripwire R5 → Kill Zone R1+

Control support path:
Snare Trap R5 → Lure Shot R5 → Tactical Field R5+
```

## E.4 Design Notes for Prerequisite Chains

- A prerequisite chain should describe a gameplay relationship. Example: **Burning Mark** feeds **Wildfire**, so Wildfire requiring Burning Mark Rank 5 is intuitive.
- A chain should never require unrelated investment just to slow players down.
- Either/or prerequisites should be used when two different playstyles can reasonably lead to the same skill.
- The first version should support respecs so these requirements can be tested without punishing players.
- Advanced finishers can require a maxed base finisher to make base investment matter beyond the first 25 levels.

# Closing Direction

Project Starfall should focus first on making a small set of classes feel excellent, social, and replayable. The class names should remain intuitive, but the mechanics should be distinctive enough that each branch has its own rhythm and value.

The first major prototype should prove this loop:

```text
Choose class → level → open skill batch → invest points through prerequisite paths → branch → open advanced skill batch → choose mobbing/bossing/party investment emphasis → run content with others → upgrade gear → chase better outcomes → make another character for a new playstyle
```

The strongest differentiator is the combination of:

1. Simple readable class structure.
2. Resource-driven class identity.
3. Explicit mobbing, bossing, hybrid, control, support, and party skill roles.
4. Skill-batch progression with rank prerequisites instead of scattered level unlocks.
5. Party skills that help both self and allies.
6. Risk-based gear upgrades with scars and mutations.
7. Limited, non-mandatory multi-character account progression.


# Appendix F: Enemy Implementation Quick Reference

This appendix summarizes the enemy implementation details for quick production use.

## F.1 Early Enemy Summary

| Enemy | Level | Role | HP Mult | DMG Mult | DEF Mult | Key Mechanic | Primary Counter |
|---|---:|---|---:|---:|---:|---|---|
| Slimelet | 1–6 | Basic/swarm-light | 0.75 | 0.60 | 0.60 | Slow hop contact. | Any basic attack, AoE. |
| Mossback | 4–10 | Durable melee | 1.25 | 0.85 | 1.20 | Braces against knockback. | Fire, armor break. |
| Thorn Sprout | 5–12 | Stationary ranged | 0.70 | 0.85 | 0.70 | Fires dodgeable thorns. | Fire, range, line attacks. |
| Bristle Boar | 8–16 | Charger | 1.15 | 1.25 | 1.00 | Telegraph charge. | Jumping, traps, blocks. |
| Dust Imp | 10–18 | Fast melee | 0.85 | 1.00 | 0.75 | Leap slash and retreat. | AoE, traps, fast hits. |
| Clockbug | 12–22 | Armored tank | 1.70 | 0.80 | 1.70 | Armor crack state. | Armor break, bossing skills. |
| Ember Wisp | 16–26 | Flying ranged | 0.85 | 0.95 | 0.70 | Floating firebolt. | Marks, lightning, ranged attacks. |
| Bandit Cutter | 18–28 | Melee blocker | 1.05 | 1.05 | 1.00 | Frontal block. | Back attacks, burst, stun. |
| Bandit Thrower | 20–30 | Ranged priority | 0.80 | 0.90 | 0.75 | Retreats and throws arcs. | Gap closers, long range. |
| Oreback Beetle | 24–35 | Tank/material | 1.90 | 0.90 | 1.80 | Shell crack. | Armor break, weak points. |
| Glowcap Healer | 22–34 | Support | 0.75 | 0.40 | 0.70 | Heals nearby enemies. | Fire, seal, target priority. |
| Cracked Mimic | 15–35 | Rare elite | 4.00 | 1.50 | 1.20 | Ambush and flee at low HP. | Burst windows, stuns. |

## F.2 Enemy Role Balance Checklist

Before shipping a new enemy, confirm:

- It has a readable silhouette.
- It has at least one clear behavior players can learn.
- It has a defined counterplay option.
- Its drops support at least one progression loop.
- It does not hard-counter a class path in required content.
- It has a clear role in map composition.
- Its HP and damage multipliers match expected spawn density.

# Appendix G: Equipment and Shop Quick Reference

## G.1 Gear Tier Summary

| Tier | Level | Shop Availability | Expected Rarity | Role |
|---|---:|---|---|---|
| Training | 1 | Starter shops | Common | Tutorial. |
| Copper | 5 | Starter shops | Common/Uncommon | Early replacement gear. |
| Iron | 15 | Regional shops | Common/Uncommon/Rare | First real stat choices. |
| Steel | 25 | Weapon Smith/Class Supplier | Uncommon/Rare | Advanced class support. |
| Silver | 40 | Regional shops/token vendors | Rare | Midgame specialization. |
| Runed | 60 | Dungeons/bosses/tokens | Rare/Epic | Specialization gear. |
| Starforged | 80 | Bosses/crafting | Epic/Relic | Endgame start. |
| Ancient | 100 | Awakening/endgame bosses | Relic/Mythic | Build-defining chase. |

## G.2 Main Equipment Slot Summary

| Slot | Main Purpose | Common Stats |
|---|---|---|
| Weapon | Main offense identity. | Attack/Magic Power, speed, crit, boss/area damage. |
| Offhand | Branch utility. | Defense, resource generation, branch mechanic modifiers. |
| Head | Secondary defense/utility. | Defense, HP, status resistance, resource. |
| Chest | Main defense. | Defense, HP, damage reduction. |
| Gloves | Action feel and precision. | Attack/cast speed, crit, skill modifiers. |
| Boots | Movement and positioning. | Movement speed, dodge recovery, knockback resistance. |
| Rings | Small stat customization. | Crit, resource, HP, element, boss/area damage. |
| Amulet | Build-shaping utility. | Resource, crit damage, sustain, utility. |
| Badge/Charm | Special effects. | Boss trophies, rare effects, niche traits. |

## G.3 Vendor Design Checklist

Before adding a vendor item, confirm:

- The item has a clear level requirement.
- The item has a clear class or branch requirement if applicable.
- The item is not best-in-slot unless it is token-gated and intentionally designed as deterministic progression.
- The item has fewer traits or lower upgrade cap than comparable rare drops.
- The price matches expected playtime for its level band.
- The item teaches a useful choice, such as mobbing vs bossing vs defense.
- The item visually matches its tier and function.



# Appendix H: AI Asset Production Standards

This appendix converts the GDD into a **practical AI asset brief**. The goal is to make visual generation consistent enough that sprites, icons, effects, enemies, equipment, maps, and UI can be generated with minimal ambiguity.

## H.1 Global Style Rules

**Overall visual style:**

- Canonical theme: **Starlit Frontier Fantasy** - a clean, luminous side-scrolling MMO fantasy style where frontier guild life meets fallen-star magic.
- 2D side-scrolling fantasy action RPG with AI-generated painterly environments and clean cel-shaded sprite assets.
- Blocky, pixel-readable sprite look with simple shapes, clean materials, and limited fine detail. It may use smooth Canvas rendering, but it should read closer to compact sprite art than over-rendered illustration.
- Clean silhouettes, readable proportions, strong class color coding.
- Compact readable body proportions; approximately **3.5 to 4 heads tall**, not tiny chibi and not realistic.
- Luminous frontier materials: midnight navy, warm lantern gold, star-white, crystal cyan, honey wood, brass, stone, leather, and class-specific accent colors.
- Repeating motifs: fallen stars, portals, guild banners, lanterns, rune circles, crystal inlays, practical magic tools, upgrade devices, and regional landmarks.
- Crisp contour separation, restrained outlines, broad value blocks, and minimal face detail on generic player bodies.
- Effects should be vivid but not so large that they hide enemy telegraphs.
- Equipment should be readable at gameplay scale first, decorative second.
- Player identity should primarily come from equipped item layers, weapon silhouettes, class VFX, portraits, and skill effects rather than a unique face/body for every character.

**Global negative instructions for AI:**

- Do not generate text, labels, logos, watermarks, UI lettering, or signatures.
- Do not use photorealism.
- Do not use extreme anatomy exaggeration or oversized anime heads.
- Do not make sprites too dark, muddy, noisy, or over-rendered.
- Do not make assets look like generic procedural shapes when an authored or AI-generated fantasy material treatment is possible.
- Do not use perspective inconsistent with side-view gameplay.
- Do not place props behind the character that could be mistaken for limbs or gear.
- Do not generate isometric or top-down assets when a side-view sprite is requested.
- Do not use modern sci-fi UI styling unless it is filtered through Starfall's rune, guild, portal, and frontier craft language.

## H.2 Rendering Targets by Asset Type

| Asset Type | Recommended Canvas | Background | Notes |
|---|---:|---|---|
| Character sprite concept | 1024x1024 | transparent or flat neutral | Full-body side-view and 3/4 reference if needed. |
| Gameplay sprite frame | 256x256 per frame | transparent | Body centered, feet aligned to baseline. |
| Sprite sheet | varies; e.g. 2048x2048 | transparent | Consistent frame spacing and pivot. |
| Enemy sprite concept | 1024x1024 | transparent or flat neutral | One full body, one expression/attack pose if needed. |
| Skill icon | 256x256 | transparent or dark neutral | Single strong symbol, no text. |
| Item icon / equipment sprite | 256x256 | transparent | Item centered, minimal shadow. |
| Map concept art | 1920x1080 | full scene | For mood and layout reference. |
| Tileset sheet | 2048x2048 | transparent | Clean edges, modular pieces. |
| UI panel | 1920x1080 or component sheet | transparent | Export as separate layered components where possible. |
| Shop interior scene | 1920x1080 | full scene | Used for vendor screen backgrounds and concept art. |

## H.2.1 Bordered Source Sheet Standard

All AI-generated assets that are processed as a grid must use the same bordered-source methodology. This applies to enemy animation sheets, compact enemy review sheets, projectile sheets, item icon sheets, skill icon sheets, and any future generated grid of UI icons, equipment icons, props, or VFX cells.

Source sheet contract:

- Use thin solid `#00ffff` guide lines around the entire sheet and between every cell.
- The guide color is reserved for processing only. Do not use `#00ffff` inside the artwork, glow, shadow, spell effect, item gem, or background.
- Use a flat chroma-key background that is not present in the artwork. Current defaults are `#00ff00` for most item and enemy sheets and `#ff00ff` when green would conflict with the subject.
- Keep one complete asset per cell. Do not let limbs, feet, weapons, projectiles, shadows, glows, labels, or decorative fragments cross into adjacent cells.
- Keep scale, facing direction, baseline, and padding consistent across all cells in the same row or sheet.
- Do not rely on proportional slicing, manual cropping, or approximate equal cell sizes. The processor must detect the visible guide borders first and cut only from the cell interiors.
- If a source sheet is missing a required guide line, has broken guide spacing, or has art bleeding across borders, the correct response is to regenerate or repair the source sheet instead of forcing it through the pipeline.

Processor contract:

- Detect the guide grid before chroma-key removal.
- Remove guide pixels and chroma-key background after detection.
- Normalize each output cell or icon to the runtime asset contract.
- Audit final outputs for transparent corners, visible content, consistent dimensions, and no opaque pixels touching animation cell edges.
- Generated deployable assets should keep the runtime file paths and dimensions stable so game data does not need to change when art is regenerated.

Current processor ownership:

| Asset Family | Source Location | Processor | Output Contract |
|---|---|---|---|
| Enemy animation sheets | `tmp/project-starfall-ai-enemies/raw/` | `build/process-project-starfall-ai-enemy-sheets.js` | Transparent animation sheets and portraits. |
| Compact enemy/prototype sheets | `asset-sources/project-starfall/enemies/compact/` | `build/process-project-starfall-compact-bandits.js` | Compact enemy sheets, portraits, and projectile sheets. |
| Item icons | `img/project-starfall/items/source/` | `build/process-project-starfall-ai-item-icons.js` | Transparent 64px item icons. |
| Skill icons | `img/project-starfall/skills/source/` | `build/process-project-starfall-skill-icons.js` | Transparent 256px skill icons. |

For non-enemy rebuilds, use `npm run build:project-starfall-assets`. This command runs only the source-backed map, environment, item, and skill processors and refuses to call enemy-owned processors.

## H.3 Animation Package Requirements

Every player class and humanoid NPC should support a standard animation pack.

### H.3.1 Player animation list

| Animation | Frames | Notes |
|---|---:|---|
| Idle | 6 | subtle breathing, cloth and hair motion |
| Walk / Run | 8 | clean contact frames, readable foot placement |
| Jump start | 2 | crouch and launch |
| Jump loop | 2 | airborne pose |
| Fall | 2 | controlled fall silhouette |
| Land | 2 | impact recovery |
| Basic attack 1 | 6 | quick, readable |
| Basic attack 2 | 6 | alternate rhythm |
| Skill cast / skill attack | 8–12 | depends on class |
| Hit react | 3 | light recoil |
| Knockdown | 4 | launch / downed sequence |
| Victory / emote | 6 | class personality |

### H.3.2 Enemy animation list

| Animation | Frames | Notes |
|---|---:|---|
| Idle | 4–6 | looping identity motion |
| Move | 4–8 | hop, crawl, float, walk, or slide |
| Attack | 5–8 | clear telegraph and release |
| Hit react | 2–3 | brief response |
| Death | 4–8 | collapse, burst, dissolve, or break apart |
| Special / elite tell | 4–8 | used for elite or boss mechanics |

## H.4 AI Prompt Formula Templates

### H.4.1 Character sprite concept template

> Generate a clean 2D side-view fantasy action RPG character sprite concept for **[character name or generic base]**. Style: Starlit Frontier Fantasy, clean cel-shaded sprite asset, compact non-realistic proportions, about 3.5-4 heads tall, readable silhouette, crisp contour separation, broad value blocks, and luminous star-magic accents. Show full body, neutral standing pose, facing right, transparent background. Keep the base body generic with no detailed face unless this is a portrait. Emphasize **[class fantasy]**, **[weapon]**, **[materials]**, and **[signature colors]** through equipment layers, weapon shape, and VFX. Keep the design gameplay readable at small size.

### H.4.2 Skill icon template

> Generate a 256x256 fantasy RPG skill icon for **[skill name]** in the Starlit Frontier Fantasy style. Show a single centered symbol with strong silhouette on a dark navy or transparent background. Use bold color contrast, readable glow, and class-specific magic color. Represent **[core action]** using **[visual symbol]**. No text, no border text, no watermark.

For sheet generation, use the bordered-source standard: arrange icons in the exact requested rows and columns on a flat chroma-key background, add solid `#00ffff` guide lines around every cell and the outer sheet border, keep one centered icon per cell, and do not use `#00ffff` inside the icon art.

### H.4.3 VFX key art template

> Generate a 2D side-view action RPG effect reference for **[skill name]** in the Starlit Frontier Fantasy style. Show the effect in side view on transparent background. Emphasize **[element / color]**, **[motion shape]**, **[impact moment]**, and clean luminous edges. The effect should feel readable in gameplay and not overly large.

### H.4.4 Equipment icon template

> Generate a single equipment icon sprite for a **[tier] [slot]** used by a **[class]** in a 2D side-scrolling fantasy action RPG. Transparent background, centered item, readable at small size. Material palette: **[material description]**. Style: Starlit Frontier Fantasy, clean cel-shaded item art, lightly outlined, bright but not noisy, with practical crafted materials and optional star/rune/crystal accent details.

For production item sheets, prefer batched bordered sheets over unrelated one-off generations. List the item IDs in exact processing order, use one item per cell, keep comparable item scale across the sheet, and reserve `#00ffff` only for guide borders.

### H.4.4.1 Bordered item sheet template

> Generate a clean RPG item icon source sheet for Project Starfall. Sheet layout: **[columns] by [rows]**. Use a perfectly flat `#00ff00` chroma-key background and thin solid `#00ffff` guide lines around the outside and between every cell. Do not use `#00ffff` inside any item. Put exactly one centered item in each cell, in this order: **[ordered item ID/name list]**. Style: Starlit Frontier Fantasy, clean cel-shaded inventory art, readable silhouettes at small size, lightly outlined, consistent scale across similar slots, no text, no labels, no UI frames, no watermark, no cropped edges, no shadows or glow crossing cell borders.

### H.4.5 Environment template

> Generate a side-scrolling 2D fantasy map concept for **[map name]** in the Starlit Frontier Fantasy style. Show layered platforms, traversable lanes, background depth, and readable landmarks. Style: AI-generated painterly side-scroller background, luminous frontier materials, modular level-design friendly. Include **[region props]**, **[hazards]**, **[enemy habitat cues]**, **[dominant colors]**, and one clear Starfall motif such as a lantern, banner, rune, crystal inlay, portal remnant, or fallen-star fragment.

# Appendix I: Character and NPC Visual Prompt Library

## I.1 Playable Character Base Proportions and Shared Rules

- Body proportion: 3.5–4 heads tall.
- Head large enough for readability, but simple and generic.
- Hands and feet slightly oversized for animation clarity.
- Face detail removed or minimized on gameplay sprites; portraits may carry expression.
- All classes must have a strong side silhouette and a weapon silhouette distinct from the body.
- Class and item identity should come from equipment layers, weapon silhouettes, class VFX, and accent colors rather than unique base bodies.

## I.2 Base Classes

### Fighter

**Visual identity:** grounded frontline adventurer. Short cloak or shoulder cape optional, fitted leather and steel armor, practical boots, reinforced gloves, broad belt, visible weapon harness.

**Palette:** steel gray, warm brown leather, muted crimson accent.

**Silhouette cues:** broad shoulders, gauntlets, weapon-forward stance, planted feet.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for a Fighter class in a bright fantasy side-scrolling RPG. The character should look like a practical frontline adventurer with medium steel armor, brown leather straps, a crimson cloth accent, one-handed sword or short heavy blade, and a grounded ready stance. Non-chibi, readable silhouette, stylized hand-painted sprite style, transparent background.

### Mage

**Visual identity:** mobile spellcaster with layered robe-tunic hybrid, sash, light boots, arcane focus, visible mana charms.

**Palette:** navy, ivory, soft teal, pale gold accents.

**Silhouette cues:** staff or wand, hanging sleeves or sash tails, arcane accessories, upright posture.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for a Mage class in a bright fantasy side-scrolling RPG. The character should wear a layered robe-tunic with a sash, light boots, arcane charms, and carry a wand or short staff with a glowing focus. Use navy, ivory, teal, and gold accents. Non-chibi, readable silhouette, stylized hand-painted sprite style, transparent background.

### Archer

**Visual identity:** agile field ranger. Light armor, short mantle, fitted tunic, bracers, quiver, travel boots, scouting accessories.

**Palette:** forest green, tan leather, charcoal, pale amber accent.

**Silhouette cues:** bow and quiver, narrow frame, capelet or scarf trailing backward, poised stance.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for an Archer class in a bright fantasy side-scrolling RPG. The character should wear light leather field gear, a short mantle or scarf, fitted tunic, bracers, and carry a longbow with a visible quiver. Use forest green, tan, charcoal, and amber accents. Non-chibi, readable silhouette, stylized hand-painted sprite style, transparent background.

## I.3 Advanced Classes

### Guardian

**Identity:** defensive Fighter with shield-bearing or fortress-like body language.

**Visual additions:** heavier pauldrons, reinforced kite shield or tower shield, impact plates, square shield motifs, protective crest.

**Palette:** iron gray, deep red, brass accents.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Guardian advanced class. The character is a defensive Fighter wearing reinforced medium-heavy armor with large pauldrons, layered arm guards, a kite shield or tower shield, and a protective crimson tabard. Emphasize stability, defense, and impact absorption. Bright fantasy hand-painted sprite style, transparent background.

### Berserker

**Identity:** aggressive Fighter with risk-reward style.

**Visual additions:** asymmetrical armor, exposed arms or damaged armor edges, red wraps, jagged blade or heavy axe, feral forward lean.

**Palette:** dark iron, blood red, bone, ember orange.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Berserker advanced class. The character should look reckless and dangerous, with asymmetrical armor, red war wraps, a scarred heavy blade or axe, and a forward-leaning aggressive pose. The design should imply low-defense high-offense without looking shirtless or barbaric-only. Stylized hand-painted sprite art, readable silhouette, transparent background.

### Duelist

**Identity:** refined fast Fighter.

**Visual additions:** lighter fitted armor, rapier or slim blade, long gloves, split coat tails, elegant waist sash.

**Palette:** silver, navy, white, crimson accent.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Duelist advanced class. The character should look agile and precise, with lighter fitted armor, a slender sword, long gloves, split coat tails, and an elegant but combat-ready stance. Use silver, navy, white, and crimson accents. Stylized hand-painted sprite style, transparent background.

### Fire Mage

**Identity:** heat-building caster.

**Visual additions:** ember-lined robe trim, glowing vent seams, staff core like a coal ember, heat shimmer motifs.

**Palette:** blackened red, orange, ash gray, gold.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Fire Mage advanced class. The character should look like an explosive elemental caster with ember-lined robes, warm orange glow details, ash-gray cloth layers, and a wand or staff with a furnace-like core. Suggest heat build-up and controlled danger. Stylized hand-painted sprite style, transparent background.

### Rune Mage

**Identity:** geometric support caster.

**Visual additions:** rune plates, hanging talismans, geometric hems, belt scroll cases, staff or spellbook with glowing rune circles.

**Palette:** indigo, parchment, cyan glow, gold trim.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Rune Mage advanced class. The character should wear layered arcane robes with geometric rune motifs, hanging talismans, a book or rune staff, and glowing cyan inscriptions. The design should feel intelligent, composed, and tactical. Stylized hand-painted sprite style, transparent background.

### Storm Mage

**Identity:** fast electric caster.

**Visual additions:** swept cloth shapes, wind-torn scarf, lightning-shaped focus, lighter silhouette, charged gauntlet or rod.

**Palette:** sky blue, white, deep navy, electric yellow accents.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Storm Mage advanced class. The character should look swift and energetic, with a wind-swept scarf, light arcane clothing, a rod or focus charged with lightning, and electric motion motifs. Use sky blue, white, navy, and electric yellow. Stylized hand-painted sprite style, transparent background.

### Sniper

**Identity:** bossing-focused precision Archer.

**Visual additions:** longer bow, sighting monocle or hawk-eye headband, stabilized bracer, long coat-tail or cloak wedge, clean line silhouette.

**Palette:** dark green, black-brown leather, steel gray, pale gold.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Sniper advanced class. The character should be a precision archer with a longbow, refined field gear, a sighting accessory such as a monocle or headband, and a disciplined steady stance. Use dark green, leather brown, gray steel, and pale gold accents. Stylized hand-painted sprite style, transparent background.

### Trapper

**Identity:** battlefield-control Archer.

**Visual additions:** trap satchel, leg pouches, coiled wire, hooked arrows, utility harness, compact bow.

**Palette:** olive, brown, rust orange, steel gray.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Trapper advanced class. The character should look like a tactical ranger carrying trap satchels, wire tools, hooked arrows, and a compact bow. Use olive, brown, rust orange, and steel gray. Emphasize preparedness and battlefield control. Stylized hand-painted sprite style, transparent background.

### Beast Archer

**Identity:** partner-based Archer.

**Visual additions:** animal-bond motifs, feather or claw ornaments, lighter armor, companion harness or charm tokens.

**Palette:** leaf green, cream, chestnut, turquoise accent.

**AI prompt:**

> Generate a full-body 2D side-view sprite concept for the Beast Archer advanced class. The character should look like a wilderness archer bonded to a companion, with practical ranger gear, claw or feather ornaments, and a visible companion charm or harness accessory. Use leaf green, cream, chestnut, and turquoise accents. Stylized hand-painted sprite style, transparent background.

## I.4 NPC Vendor Visual Prompts

### Starter Outfitter

Friendly traveling clothier. Wooden rack, folded tunics, patched banners, beginner warmth.

> Generate a 2D side-view NPC vendor concept for the Starter Outfitter. Friendly fantasy clothier, middle-aged or young adult, practical apron, colorful folded cloth, sewing kit, and a warm welcoming expression. Include a simple market stall with training tunics and leather caps.

### Weapon Smith

Smith with sparks, anvils, hanging blades, and practical metalwork.

> Generate a 2D side-view NPC vendor concept for the Weapon Smith. Muscular or stocky blacksmith in leather apron with forge gloves, hammer, and a stall full of swords, bows, and staves hanging behind them. Bright fantasy hand-painted style.

### Regional Armorer

Town armorer with display mannequins and polished breastplates.

> Generate a 2D side-view NPC concept for a Regional Armorer. Show a professional armorer with polished armor displays, helmets, gloves, and boots arranged neatly in a high-quality storefront.

### Class Supplier

A specialist instructor selling class-coded equipment and manuals.

> Generate a 2D side-view NPC concept for the Class Supplier. The vendor should look like a knowledgeable adventuring quartermaster with organized racks of class-specific gear and manuals for Fighter, Mage, and Archer.

### Arcane Jeweler

Elegant magical shopkeeper, gem trays, floating accessories.

> Generate a 2D side-view NPC concept for the Arcane Jeweler. The character should run an elegant magical accessory shop with rings, amulets, glowing gemstones, and softly floating arcane trinkets.

### Consumable Vendor

Travel supplies, potions, field food.

> Generate a 2D side-view NPC concept for the Consumable Vendor. Show a practical merchant with potion flasks, dried herbs, packed rations, lanterns, and organized supply crates.

### Upgrade Artisan

Mystical-forge technician combining smithing and enchantment.

> Generate a 2D side-view NPC concept for the Upgrade Artisan. The character should look like a master craftsperson combining runes and metalwork, with a forge table, catalysts, glowing tongs, and a focused serious expression.

### Dungeon Quartermaster

Disciplined military-adjacent supplier, tokens and boss gear displays.

> Generate a 2D side-view NPC concept for the Dungeon Quartermaster. Show an orderly vendor with token boards, dungeon trophies, and limited high-grade equipment on display.

### Event / Reputation Vendor

Seasonal or regionally themed merchant with premium decorative wares.

> Generate a 2D side-view NPC concept for an Event or Reputation Vendor. The vendor should feel more decorative and collectible, with special banners, unique accessories, and themed event merchandise.

# Appendix J: Enemy and Boss Visual Prompt Library

## J.1 Enemy Sprite Rules

- Readability first: enemies need clear attack tell poses.
- Each enemy should have one dominant identifying feature visible in silhouette.
- Use class counters visually: armored enemies look armored, flying enemies look airborne, support enemies look vulnerable.
- Elite variants should add one extra glow color and one shape change.

## J.2 Early Enemy Prompts

### Slimelet

**Look:** small translucent slime blob with a shiny core and a tiny wobbling mouth. Rounded, cute but not babyish.

**Palette:** mint green or pale blue with glossy highlights.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for a Slimelet, a small translucent fantasy slime for a side-scrolling RPG. Show a rounded jelly body with a shiny inner core, simple eyes, and a hop-ready pose. Keep it readable, cute, and easy to animate. Transparent background.

### Mossback

**Look:** squat boar-like forest beast with moss on its back, bark-like hide, and small tusks.

**Palette:** moss green, bark brown, muted beige.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Mossback, a small boar-like forest beast with bark-textured hide, moss growing along its back, and short tusks. Make it sturdy and a little tanky-looking. Transparent background.

### Thorn Sprout

**Look:** animated plant turret with bulb head, thorn pod mouth, root base.

**Palette:** green stem, dark thorns, red or purple bud core.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Thorn Sprout, a stationary plant enemy. It should have a rooted base, a flower-bulb head, and a thorn-shooting opening. Make it clearly a ranged plant turret. Transparent background.

### Bristle Boar

**Look:** leaner boar with bristled mane and visible charge posture.

**Palette:** dark brown, tan, gray, red eyes.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Bristle Boar, a fast charging beast enemy. Show a lean boar with a raised bristled mane, scraping foreleg pose, and a silhouette that communicates horizontal charges. Transparent background.

### Dust Imp

**Look:** small goblin-imp hybrid with oversized hands or claws, bandaged feet, mischievous face.

**Palette:** dusty tan, muted red, charcoal.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Dust Imp, a quick evasive imp-like enemy. Show a wiry body, oversized clawed hands, and a crouched leap-ready pose. Make it annoying but readable, not grotesque. Transparent background.

### Clockbug

**Look:** beetle-like construct with brass shell, visible gears, and mechanical jaw.

**Palette:** brass, iron gray, cyan or yellow gear glows.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Clockbug, a slow armored clockwork insect enemy. It should have a brass shell, visible gears, and a mechanical bite or gear-burst mechanism. Make it clearly durable. Transparent background.

### Ember Wisp

**Look:** floating flame spirit with a bright core and trailing ember tail.

**Palette:** orange, gold, red, dim charcoal outline.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Ember Wisp, a floating fire spirit. Show a glowing central flame, soft ember tail, and simple expressive eye shapes. It should look aerial and slightly evasive. Transparent background.

### Bandit Cutter

**Look:** masked or hooded humanoid melee bandit with short blade and guarded stance.

**Palette:** dark cloth, tan leather, muted steel.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Bandit Cutter, a melee bandit enemy. Show light armor, a hood or half-mask, short sword, and a guarded combat stance. Transparent background.

### Bandit Thrower

**Look:** agile bandit with knife bandolier and throwing pose.

**Palette:** dusty brown, dark green, gray steel.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Bandit Thrower, a ranged bandit enemy. Show a slim humanoid with a throwing knife bandolier, light travel clothes, and a backward-leaning pose ready to throw. Transparent background.

### Oreback Beetle

**Look:** large beetle-crab hybrid with ore growths embedded in its shell.

**Palette:** iron gray, rusty red, crystal ore highlights.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Oreback Beetle, a heavy mineral beast enemy. Show a thick shell with embedded ore chunks, a horned front, and a stomp-ready body shape. Transparent background.

### Glowcap Healer

**Look:** mushroom-healer creature with luminous cap, soft body, and puff spores.

**Palette:** cream stem, blue or green glow cap, pale spores.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Glowcap Healer, a support mushroom creature. It should have a luminous cap, a soft body, and a visual language of healing spores rather than aggression. Transparent background.

### Cracked Mimic

**Look:** treasure chest monster with broken lid, teeth, and cracked magical lock.

**Palette:** dark wood, bronze trim, purple or teal magical crack glow.

**Prompt:**

> Generate a 2D side-view enemy sprite concept for Cracked Mimic, an elite chest monster. Show a treasure chest body with cracked trim, teeth inside the lid, a magical lock, and a tongue-lash or bite-ready pose. Transparent background.

## J.3 Boss Prompt

### Emberjaw Golem

**Look:** hulking volcanic construct, stone plates with magma seams, furnace mouth, heavy fists.

**Palette:** basalt black, lava orange, ash gray, brass highlights.

**Prompt:**

> Generate a 2D side-view boss sprite concept for Emberjaw Golem, a large volcanic construct boss in a fantasy side-scrolling RPG. Show a massive body made of dark stone plates with glowing magma cracks, a furnace-like mouth, and huge heavy fists. Emphasize weight, slow power, and readable attack telegraphs. Transparent background.

# Appendix K: Equipment, Item, and Shop Visual Prompt Library

## K.1 Gear Style Rules

- Each item icon should be centered and readable at 64x64 UI scale.
- Shop-tier gear should look simpler than dungeon or boss gear.
- High-tier gear should not just be bigger; it should show better material quality and more precise silhouette language.
- The same slot across classes should share overall scale consistency.
- Generated item icons should be produced from bordered source sheets, not ad hoc manual crops, so future item batches stay consistent.

## K.1.1 Item Icon Generation Workflow

All generated item icons should follow the same pipeline as the current item icon processor.

Design and data setup:

- Add or update the item in the item data first, including item ID, slot/category, rarity/tier intent, class restrictions, and `ITEM_ASSETS` path.
- Add the item ID to the correct source sheet definition in `build/process-project-starfall-ai-item-icons.js`, keeping the source order identical to the prompt order.
- Group items by visual family when possible: consumables/materials, starter shop gear, boss set gear, region gear, event cosmetics, or class-specific equipment. Mixed sheets are allowed only when the prompt clearly lists each cell.

Source generation requirements:

- Source sheets live under `img/project-starfall/items/source/`.
- Use exact declared sheet dimensions and cell count from the processor definition.
- Use a flat `#00ff00` chroma-key background unless the item art requires a different key color.
- Draw solid `#00ffff` guide borders around the sheet and between every cell.
- Put one item per cell. No labels, quantities, rarity frames, UI boxes, text, watermark, drop shadow that crosses borders, or duplicate alternate views.
- Keep comparable slots at comparable scale. A helmet, ring, potion, book, and weapon may have different silhouettes, but each should fill the icon canvas enough to read at inventory and ground-drop scale.
- Item identity should come from silhouette, material, tier motifs, class motifs, region motifs, and color. Do not rely on tiny decoration that disappears at 64x64.

Processing and validation:

- Run `node build/process-project-starfall-ai-item-icons.js` after adding or replacing item source sheets.
- The processor detects the `#00ffff` grid before removing chroma key, slices only inside cell borders, removes small artifacts, normalizes icon fill, and writes transparent 64px PNGs to `img/project-starfall/items/`.
- Run `npm test` after regeneration. Tests should confirm every item asset exists, is 64px, has transparent corners, and fills most of the icon canvas.
- If any icon is too small, cropped, contaminated by a neighboring cell, or missing, fix the bordered source sheet or the prompt order. Do not manually crop around the failed output as the primary workflow.

## K.2 Tier Visual Language

| Tier | Look | Materials | Accent Cues |
|---|---|---|---|
| Training | plain, humble | wood, cloth, stitched leather | minimal decoration |
| Copper | early crafted, warm metal | copper, rough leather | simple rivets |
| Iron | practical soldiering | iron, dark leather | squared forms |
| Steel | refined and balanced | polished steel, layered cloth | cleaner edges, blue-gray steel |
| Silver | elegant and bright | silver, pale leather, etched trim | light magical inlay |
| Runed | enchanted utility | dark steel, rune carvings, glow lines | visible glowing symbols |
| Starforged | premium heroic gear | midnight steel, gold, crystal cores | luminous star motifs |
| Ancient | relic-grade | weathered alloys, stone, precious cores | ceremonial or mysterious motifs |

## K.3 Fighter Equipment Visual Prompts

### Fighter weapons

- **Training Sword:** plain wooden practice sword with wrapped grip.
- **Arming Sword:** practical straight blade, simple crossguard.
- **Broadsword:** heavier single broad blade, thicker spine.
- **War Hammer:** short one-handed hammer with metal head.
- **Kite Shield:** tapering shield with iron rim.
- **Tower Shield:** tall rectangular shield with reinforced center boss.

**Generic prompt formula:**

> Generate a single transparent-background item sprite for a **[tier] Fighter [weapon/shield]** in a 2D fantasy side-scrolling RPG. The item should be centered, readable, and designed for small icon use. Use **[tier material language]** and keep the silhouette clear.

### Fighter armor

- **Head:** leather cap, steel circlet, enclosed helm depending on tier.
- **Chest:** tunic to brigandine to plated cuirass.
- **Gloves:** leather gloves to armored gauntlets.
- **Boots:** travel boots to plated greaves.

## K.4 Mage Equipment Visual Prompts

### Mage weapons

- **Training Wand:** smooth wood wand with tiny crystal bead.
- **Focus Rod:** slim metal-tipped wand.
- **Short Staff:** compact staff with glowing focus head.
- **Spellbook:** floating or hand-held tome with clasp.
- **Orb Offhand:** glass or crystal orb in a ring mount.

**Prompt formula:**

> Generate a transparent-background item sprite for a **[tier] Mage [wand/staff/spellbook/orb]**. The design should be elegant, arcane, and readable, using **[tier material language]** with magic-ready details.

### Mage armor

- **Head:** hood, circlet, pointed short cap, rune crown.
- **Chest:** robe-tunic, layered coat, mantle, arcane sash.
- **Gloves:** casting gloves, rune cuffs.
- **Boots:** light boots with straps or glowing trims.

## K.5 Archer Equipment Visual Prompts

### Archer weapons

- **Training Bow:** simple wooden bow.
- **Recurve Bow:** balanced field bow.
- **Longbow:** tall precision bow for Sniper.
- **Compact Bow:** tactical bow for Trapper.
- **Spirit Bow:** nature-styled branch-and-core bow for Beast Archer.
- **Quiver:** visible as class accessory sprite or paper-doll component.

**Prompt formula:**

> Generate a transparent-background item sprite for a **[tier] Archer [bow/quiver]**. The item should look mobile and field-ready, with a strong bow silhouette and readable tier materials.

### Archer armor

- **Head:** hood, band, scout cap, feathered field cap.
- **Chest:** fitted leather coat, ranger tunic, layered vest.
- **Gloves:** archery gloves, trap gloves.
- **Boots:** flexible travel boots.

## K.6 Universal Accessories

### Rings

- Small silhouette, gem or engraved signet.
- Stat identity should influence color: crit rings use sharp gem facets; defense rings use thicker bands.

### Amulets

- Pendant, charm, or small relic on a chain.
- Magical amulets can have floating halo shapes.

### Badges

- Crest plates, insignias, or monster trophy emblems.

### Charms

- Small dangling token, carved bone, feather bundle, rune coin, or gemstone knot.

**Prompt formula:**

> Generate a transparent-background fantasy accessory sprite for a **[tier] [ring/amulet/badge/charm]**. Show one centered item with a clear silhouette and materials that communicate **[stat fantasy]** such as crit, defense, fire, support, or bossing.

## K.7 Example Item-Specific Prompts

### Traveler Boots

> Generate a transparent-background item sprite for Traveler Boots, an early-game Archer/Fighter footwear item. Show sturdy brown travel boots with reinforced toe caps, ankle straps, and a practical adventurer style.

### Guard Ring

> Generate a transparent-background item sprite for Guard Ring, an early-game defensive ring. Use a thick iron band with a small shield-like face and minimal blue gem detail.

### Focus Ring

> Generate a transparent-background item sprite for Focus Ring, an early-game accuracy and precision ring. Use a slimmer silver or iron band with an eye-shaped amber gem.

### Sharp Ring

> Generate a transparent-background item sprite for Sharp Ring, a crit-focused accessory. Use a sleek band with a faceted pale gold or red gem and slightly aggressive angular detailing.

### Emberjaw Badge

> Generate a transparent-background item sprite for Emberjaw Badge, a boss trophy badge. Show a metal insignia inspired by a golem jaw and magma lines, with ember orange highlights.

## K.8 Shop Interior Visual Prompts

### Starter Outfitter Shop

> Generate a cozy 2D fantasy shop interior for the Starter Outfitter in a side-scrolling RPG. Show folded beginner clothes, wooden shelves, simple armor mannequins, stitched banners, and warm lantern light. Design it as a side-view scene suitable for a shop interface background.

### Weapon Smith Shop

> Generate a 2D side-view fantasy blacksmith shop interior. Include forge glow, hanging swords, bows on racks, shield displays, metal tools, and a clear customer counter area.

### Arcane Jeweler Shop

> Generate a 2D side-view fantasy jeweler interior focused on magical accessories. Include gem trays, rings, amulets, floating arcane trinkets, polished wood cabinetry, and cool magical lighting.

### Upgrade Artisan Forge

> Generate a 2D side-view fantasy upgrade workshop interior. Show rune-inscribed anvils, catalyst jars, glowing tongs, worktables, and a dedicated item enhancement pedestal.

# Appendix L: Skill Icon, VFX, and Animation Prompt Library

## L.1 Icon Style Rules

- Each icon should show one main symbol, one supporting effect, and one color identity.
- Use dark background fields or transparent backgrounds for extraction.
- No text or numerals.
- Mobbing skills should use wider shapes or multiple targets.
- Bossing skills should use narrow, pointed, or focused shapes.
- Party skills should include an aura, circle, or radiating support cue.

## L.2 Fighter Base Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Heavy Strike | A sword or heavy blade descending with a bright impact spark and short arc. | Side-view slash effect with a compact white-yellow impact flash and a short dust line at contact. |
| Dash Slash | A diagonal blade slash laid over speed lines or a lunging silhouette. | Fast horizontal dash trail with a narrow slash arc and brief afterimage. |
| Guard | A shield silhouette or crossed forearms with a blue-white block flash. | Short frontal barrier flare with a metallic spark on contact. |
| Ground Slam | Weapon striking ground with cracks radiating outward. | Downward impact, brown dust burst, low circular shockwave. |
| Power Break | Blunt strike icon cracking a shield or armor plate. | Heavy hit flash with fractured orange-white shards indicating defense reduction. |
| Momentum Burst | Condensed glowing impact orb or blade thrust exploding outward. | Short charge glow followed by a larger circular shockwave tuned as a finisher. |

## L.3 Guardian Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Shield Bash | Shield edge striking with a gold-white spark. | Forward shield hit with squat impact flash and brief stun stars or square burst. |
| Impact Guard | Shield absorbing a hit, showing stored energy as glowing lines. | Defensive pose with attack sparks flowing into a red-gold energy reservoir on shield. |
| Oath Barrier | Protective crest or shield dome around multiple figures. | Expanding golden-red barrier ring around player and nearby allies. |
| Retaliation Wave | Shockwave bursting from a shield front. | Forward rectangular shockwave with red-gold force lines. |
| Hold the Line | Banner or shield planted into ground with sturdy aura. | Grounded stance with low red aura and faint defensive crest behind player. |
| Shield Wall | Large shield aura covering allies. | Semi-transparent layered shield field that hugs nearby party members. |
| Guardian’s Verdict | Crest-shaped explosion and shield shell. | Stored energy eruption upward and forward, then a protective shell remains briefly. |

## L.4 Berserker Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Blood Cleave | Jagged blade crescent with red slash. | Wide heavy red slash trail with blood-red energy edges. |
| Rage Surge | Roaring face or upward flame of fury. | Red aura ignition around body with surging lines and rising sparks. |
| Reckless Leap | Lunging silhouette crashing downward. | Arc leap with red afterimage and heavy landing burst. |
| Crimson Recovery | Blade siphoning red energy back into the user. | Struck enemy emits red stream flowing back to player. |
| Pain to Power | Broken heart or cracked icon feeding strength. | Passive: faint red pulse intensifies as HP lowers. |
| War Cry | Roaring mouth with radiating red lines. | Circular red-orange shout wave expanding around caster and allies. |
| Last Stand | A nearly broken blade glowing intensely. | Brief red time-freeze flash when lethal hit is prevented, followed by empowered aura. |

## L.5 Duelist Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Quick Cut | Slim blade with sharp white slash. | Fast narrow slash with clean silver streak. |
| Blade Step | Elegant diagonal footstep and blade trail. | Short phasing dash through target with faint blue-white afterimage. |
| Riposte | Countering blade crossing a strike line. | Defensive slip followed by immediate reverse stab flash. |
| Precision Mark | Small crosshair or marked weak point. | Mark appears over enemy as a clean red-white sigil. |
| Flurry Window | Repeating blade icons or swirling cuts. | Short self-buff with rapid silver glints around weapon arm. |
| Rallying Flourish | Circular rhythm aura with blade and spark motifs. | Party aura pulse in silver-crimson waves around nearby allies. |
| Final Thrust | Focused rapier point piercing a target. | Quick charge line into long narrow piercing beam-like stab effect. |

## L.6 Mage Base Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Magic Bolt | Simple glowing projectile with arcane tail. | Mid-speed magic orb in blue-teal with soft trail. |
| Blink | Small warp ring or blinking silhouette. | Instant short teleport with blue arcane burst at exit and entry. |
| Arcane Burst | Circular arcane detonation rune. | Small area explosion with teal-violet magical ring. |
| Mana Shield | Transparent magical shell around figure. | Short-lived blue-violet protective sphere around player. |
| Spell Mark | Floating rune or mark above target. | Small arcane sigil appears above enemy. |
| Energy Release | Concentrated magical orb bursting outward. | Short charge then stronger projectile or compact explosion. |

## L.7 Fire Mage Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Fireball | Flaming orb with ember sparks. | Orange-red projectile with ember tail and small blast. |
| Flame Trail | Burning line left behind a movement arc. | Blink leaves a brief ground flame strip. |
| Burning Mark | Target sigil wrapped in fire. | Enemy receives a fiery brand on body center. |
| Heat Vent | Flame cone venting from caster. | Forward cone blast that dumps stored heat in orange-white jets. |
| Wildfire | Flames jumping between enemies. | Fire arcs leap from one burning enemy to another. |
| Ignition Aura | Circular burning aura around caster and allies. | Warm orange field pulse with embers drifting outward. |
| Inferno Burst | Exploding fire core or sunburst. | Large spherical explosion with layered orange-yellow-white fire petals. |

## L.8 Rune Mage Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Rune Mark | Geometric rune on target. | Cyan rune stamp over enemy. |
| Ground Glyph | Arcane floor sigil. | Large geometric rune appears beneath enemy or on floor tile. |
| Arcane Link | Two runes connected by an energy line. | Cyan line links targets or runes with pulsing nodes. |
| Rune Detonation | Rune bursting into shards. | Existing runes explode into angular light fragments. |
| Mana Seal | Lock-like glyph or closed circle rune. | Binding circular seal snaps shut on enemy. |
| Rune Circle | Large support circle with configurable symbol. | Wide floor rune in different colors depending on mode: red power, blue guard, green cleanse, gold focus, white haste. |
| Grand Inscription | Complex layered rune field. | Multi-ring rune array fills a wide area under the caster or target zone. |

## L.9 Storm Mage Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Chain Bolt | Zig-zag electricity arcing between targets. | Primary bolt then chained arcs branching to nearby enemies. |
| Static Shift | Wind-swept warp with lightning edge. | Longer blink with pale blue wind ribbon left behind. |
| Static Field | Charged circular field. | Field of crackling electricity and rising sparks. |
| Thunder Snap | Target-focused lightning strike. | Instant vertical strike onto marked or charged enemy. |
| Stormfront | Wind ring and feather/lightning motif. | Party buff aura with light blue airflow ribbons around allied feet and shoulders. |
| Stormfall | Multiple lightning bolts descending from above. | Repeating vertical bolts and circular impact flashes. |

## L.10 Archer Base Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Quick Shot | Arrow in motion with speed line. | Fast arrow release with thin trail. |
| Roll Shot | Rolling silhouette with arrow. | Backstep or side roll with a fast release shot. |
| Marked Shot | Arrow plus target mark. | Enemy receives amber mark after hit. |
| Piercing Arrow | Long arrow line hitting multiple targets. | Straight bright projectile line passing through enemies. |
| Eagle Eye | Eye symbol with arrow or sight line. | Self and party receive pale gold focus glint over eyes. |
| Focused Volley | Cluster of arrows raining into a marked target. | Multiple arrows rapidly converge on selected targets. |

## L.11 Sniper Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Aimed Shot | Drawn longbow and focused arrow tip. | Slower draw, narrow white-gold release, strong single impact spark. |
| Weak Point Mark | Crosshair over vulnerable spot. | Mark appears on boss or elite weak point in pale gold-red. |
| Steady Breath | Calm eye and breath swirl. | Subtle self-buff with stillness ring and gold focus pulse. |
| Pierce Armor | Arrow breaking a plate. | Impact on target creates gray shard burst and defense-break symbol. |
| Execution Shot | Finisher arrow striking a vulnerable target. | High-speed finisher shot with dark-green trail and intense hit flash. |
| Eagle Eye (advanced emphasis) | Sharper eye icon radiating focus. | Stronger party crit aura in pale gold around heads or weapons. |
| One Perfect Shot | Single luminous arrow and focused starburst. | Charged stance then one huge clean projectile with dramatic impact bloom. |

## L.12 Trapper Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Snare Trap | Open trap jaws or rope snare. | Trap object appears on ground with brief setup motion. |
| Spike Trap | Ground spikes or trap plate. | Trap pops with upward spikes and dust burst. |
| Lure Shot | Arrow dragging enemies toward a point. | Special projectile leaves a small pulsing lure marker on impact. |
| Tripwire | Thin line connecting anchors. | Ground-level glowing wire appears between two points. |
| Detonate | Trigger switch or remote blast symbol. | Existing traps flash, then detonate in sequence. |
| Tactical Field | Grid field or trap zone aura. | Ground field with subtle olive-orange tactical pattern. |
| Kill Zone | Large interlocking trap cluster. | Wide field where multiple traps arm and chain-trigger with controlled bursts. |

## L.13 Beast Archer Skills

| Skill | Icon Prompt | Animation / VFX Prompt |
|---|---|---|
| Companion Strike | Animal claw plus arrow. | Companion lunges as player attacks. |
| Hunter’s Mark | Paw print or beast sigil on target. | Mark appears over enemy as nature-tinted insignia. |
| Pack Step | Player and companion repositioning together. | Short synchronized movement with mirrored afterimages. |
| Guard Beast | Companion shielding the player. | Companion moves in front; a soft green-gold shield arc appears. |
| Coordinated Assault | Arrow and beast strike crossing together. | Simultaneous ranged and melee impact lines. |
| Pack Call | Linked paw and bow aura. | Soft green-gold party aura that pulses when companion attacks. |
| Primal Bond | Large spirit-beast emblem. | Temporary transformation-like glow around player and companion. |

# Appendix M: Map, Tileset, and UI Visual Prompt Library

## M.1 Environment Style Rules

- All map, tileset, and UI prompts inherit **Starlit Frontier Fantasy** as the canonical art direction.
- Use layered parallax backgrounds with a readable foreground and midground.
- Platform edges should be readable and contrasted from the background.
- Hazard objects must have visual warning language.
- Regions should support enemy identity visually.
- Town spaces should feel friendly and legible, with strong location landmarks.
- Safe zones should combine guild-town warmth with star fragments, banners, lanterns, crystal inlays, and practical magic devices.
- Combat routes should keep painterly background depth behind clean, readable gameplay lanes.
- UI surfaces should use dark navy translucent panels with gold, star-white, and crystal-cyan accents; avoid generic web-app cards and modern sci-fi chrome.

## M.2 Core Regions and Maps

### Starfall Crossing (starter town)

**Purpose:** first town hub, shops, quest handoffs, upgrade artisan.

**Look:** cheerful frontier town built around a crashed star fragment fountain. Wood-and-stone buildings, hanging banners, warm lamps, forge smoke, market square, arcane shop with crystal signage.

**Palette:** warm tan stone, honey wood, sky blue banners, gold-white magical accent.

**Prompt:**

> Generate a 2D side-scrolling fantasy town concept called Starfall Crossing in the Starlit Frontier Fantasy style. Show a welcoming frontier guild town hub centered on a glowing star-fragment fountain, wooden and stone buildings, blue-and-gold guild banners, market stalls, a forge, an arcane jeweler, lantern arches, crystal inlays, and clean side-view traversable streets and platforms.

### Greenroot Meadow

**Purpose:** Level 1–6 starter field.

**Look:** open grassy platforms, giant leaves, small flowers, low cliffs, shallow ponds, slime habitats.

**Palette:** soft greens, pale blue water, yellow flowers.

**Prompt:**

> Generate a 2D side-scrolling fantasy field map concept called Greenroot Meadow in the Starlit Frontier Fantasy style. Show beginner-friendly grassy platforms, small ponds, low cliffs, giant leaves, soft flowers, distant guild-road signs, and a gentle whimsical environment suited for Slimelets. Keep the platforms readable and the background painterly.

### Thornpath Thicket

**Purpose:** Level 5–16 forest route.

**Look:** denser foliage, thorn bushes, wooden footbridges, angled tree roots, layered paths.

**Palette:** deeper greens, bark brown, berry red.

**Prompt:**

> Generate a 2D side-scrolling fantasy forest map concept called Thornpath Thicket. Include dense foliage, thorny plants, layered platforms made from roots and branches, and clear ranged-enemy perches for Thorn Sprouts.

### Dustburrow Trail

**Purpose:** Level 10–18 mixed evasive enemy field.

**Look:** dry trail, eroded rock, windblown dust, imp hideouts.

**Palette:** ochre, tan, muted red, weathered brown.

**Prompt:**

> Generate a 2D side-scrolling fantasy field map concept called Dustburrow Trail. Show dry dirt paths, broken rock shelves, windy dust effects, and small hideout structures suitable for Dust Imps and Bristle Boars.

### Rustcoil Ruins

**Purpose:** Level 12–22 construct region.

**Look:** old machine ruins, brass gears, stone corridors, elevated catwalks, moving mechanisms.

**Palette:** brass, gray stone, teal glow.

**Prompt:**

> Generate a 2D side-scrolling fantasy ruin map concept called Rustcoil Ruins. Show old construct machinery, brass gears, cracked stone platforms, catwalks, and mechanical ambience suitable for Clockbugs.

### Cinder Hollow

**Purpose:** Level 16–26 fire spirit region and first boss region.

**Look:** volcanic cave, ember vents, black rock, glowing lava seams, hanging chains, heat shimmer.

**Palette:** basalt black, orange lava, smoky gray.

**Prompt:**

> Generate a 2D side-scrolling fantasy cave map concept called Cinder Hollow. Show black volcanic rock, glowing magma cracks, ember vents, layered platforms, and dramatic but readable side-view terrain suitable for Ember Wisps and Emberjaw Golem.

### Bandit Ridge Camp

**Purpose:** Level 18–30 humanoid enemy zone.

**Look:** cliffside camp, rope bridges, watch platforms, crates, ragged banners.

**Palette:** dusty brown, gray wood, muted red cloth.

**Prompt:**

> Generate a 2D side-scrolling fantasy bandit camp map concept called Bandit Ridge Camp. Include rope bridges, lookout platforms, crates, tents, and narrow lanes appropriate for Bandit Cutters and Bandit Throwers.

### Oreback Quarry

**Purpose:** Level 24–35 mining and heavy-enemy zone.

**Look:** carved quarry walls, mine scaffolds, rails, ore veins, glowcap caverns.

**Palette:** stone gray, rust red, mineral blue-green glow.

**Prompt:**

> Generate a 2D side-scrolling fantasy quarry map concept called Oreback Quarry. Show mining platforms, ore veins, wood scaffolds, cave pockets with glowing mushrooms, and heavy traversable lanes suitable for Oreback Beetles and Glowcap Healers.

## M.3 Tileset Production Prompts

### Grassland tileset

> Generate a modular 2D side-scrolling tileset for a fantasy grassland map. Include ground tiles, grassy platform edges, dirt undersides, small rocks, flowers, bushes, low fences, tree roots, and decorative leaf clusters. Transparent background, clean tile separation.

### Forest / thorn tileset

> Generate a modular 2D side-scrolling tileset for a thorn forest. Include root platforms, thorn bushes, bark walls, hanging vines, mushroom props, wooden bridges, and branch platforms. Transparent background.

### Ruins / machine tileset

> Generate a modular 2D side-scrolling tileset for ancient mechanical ruins. Include stone blocks, brass inlays, gear decorations, broken catwalks, chain supports, and glowing machine cores. Transparent background.

### Volcanic cave tileset

> Generate a modular 2D side-scrolling tileset for a volcanic cave. Include basalt rock walls, magma crack platforms, ember vents, chain anchors, broken stone pillars, and lava-side props. Transparent background.

### Quarry / mine tileset

> Generate a modular 2D side-scrolling tileset for a mining quarry. Include cut-stone walls, scaffold pieces, mine rails, ore clusters, cave beams, crate props, and glowcap mushrooms. Transparent background.

## M.4 UI Visual Prompts

### Main HUD

**Elements:** HP bar, MP/Energy-style bar, class resource bar, skill hotbar, buff tray, quest tracker, minimap, party frames.

**Style:** clean fantasy UI with dark neutral panels, soft metallic edging, subtle class-colored accents.

> Generate a 2D fantasy action RPG HUD concept for a side-scrolling game. Include HP, resource bars, skill hotbar, buff icons, party frames, minimap, and quest tracker. The UI should be clean, readable, and lightly ornate, with subtle fantasy metal and cloth motifs.

### Inventory window

> Generate a fantasy RPG inventory UI panel. Show equipment slots for head, chest, gloves, boots, weapon, offhand, rings, amulet, badge, and charm, plus a grid inventory. Style should match a bright but polished fantasy RPG and remain highly readable.

### Skill tree / skill batch window

> Generate a fantasy RPG skill batch UI panel. Show grouped skill nodes, visible prerequisites, rank pips, role tags, and a detail panel. The visual should clearly communicate that a batch of skills is unlocked together and then improved with points.

### Upgrade UI

> Generate a fantasy RPG equipment upgrade interface for a side-scrolling online RPG. Show item preview, upgrade level, catalyst slots, success/failure outcome preview, and a dramatic but readable enhancement pedestal.

### Shop UI

> Generate a fantasy RPG shop interface. Show vendor portrait, category tabs, item list, equipment comparison panel, currency display, and a warm thematic backdrop tied to a frontier fantasy town.

## M.5 Portrait Prompt Standards

Portraits for UI should be chest-up 3/4 view illustrations with stronger facial readability than gameplay sprites.

**Prompt template:**

> Generate a stylized fantasy RPG portrait for **[character / enemy / vendor]**. Show a chest-up 3/4 view portrait with strong facial expression, clean class identity, and background color treatment that matches the role. Non-photorealistic, polished, readable at small UI size.

## M.6 Asset Naming Convention

| Asset Type | Example |
|---|---|
| Character sprite | `char_fighter_base_idle_sheet_v01.png` |
| Enemy sprite | `enemy_slimelet_idle_attack_v01.png` |
| Skill icon | `icon_skill_guardian_shield_wall_v01.png` |
| VFX sheet | `vfx_skill_firemage_inferno_burst_v01.png` |
| Item icon | `item_weapon_fighter_copper_arming_sword_v01.png` |
| Map tileset | `tileset_greenroot_meadow_v01.png` |
| UI panel | `ui_inventory_window_v01.png` |

## M.7 Recommended Asset Creation Order

1. Base class sprite concepts.
2. Base class animation sheets.
3. Early enemy sprite set.
4. Starter town and first three region concepts.
5. Equipment and item icon baseline by tier using bordered source sheets.
6. Base skill icon set using bordered source sheets.
7. Advanced class sprite concepts.
8. Advanced skill icons, item expansions, and VFX.
9. Shop interiors and vendor portraits.
10. UI panels and polish pass.
