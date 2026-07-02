# Project Starfall Class and Skill Design Guide

This guide is the working instruction manual for designing, improving, balancing, and implementing Project Starfall's class and skill system.

Primary design authority: `project_starfall_gdd_v0_5.md`.

Primary implementation authority:

- `js/games/project-starfall/data/classes.js`
- `js/games/project-starfall/data/skills.js`
- `js/games/project-starfall/data/class-skill-design.js`
- `js/games/project-starfall/data/specializations.js`
- `js/games/project-starfall/project-starfall-engine.js`
- `js/games/project-starfall/engine/skills.js`
- `js/games/project-starfall/engine/combat-formulas.js`
- `js/games/project-starfall/engine/movement.js`
- `js/games/project-starfall/engine/skill-modifiers.js`
- `js/games/project-starfall/data/enemies.js`
- `js/games/project-starfall/data/animations.js`
- `js/games/project-starfall/data/combat-fx.js`
- `js/games/project-starfall/ui/skill-*.js`
- `docs/project-starfall-balance-optimization-audit.md`
- `docs/project-starfall-class-skill-system-audit.md`
- `docs/project-starfall-monster-book-blueprint.md`
- `docs/project-starfall-map-optimization-audit.md`
- `docs/project-starfall-ui-visual-iteration-audit.md`
- `img/project-starfall/asset-prompts.md`
- `build/validate-project-starfall-class-skills.js`

Assumptions:

- The current browser prototype remains the implementation target: static HTML, Canvas/Pixi rendering support, plain JavaScript modules, data objects, generated image assets, and `localStorage` persistence.
- The intended class roster is the existing roster from the GDD and live data: Fighter, Mage, Archer, Guardian, Berserker, Duelist, Fire Mage, Rune Mage, Storm Mage, Sniper, Trapper, and Beast Archer.
- The guide treats the current GDD class fantasy and progression structure as the product direction, even where the live prototype has placeholder or simplified behavior.
- Do not add new class families until the existing 12 class identities are readable, balanced, visually supported, and tutorialized.

## 1. Project-Specific Combat And Class Analysis

### Current Combat Type

Project Starfall is a character-first 2D side-scrolling action RPG with MMO-inspired progression. The GDD describes it as a "PC-first 2D side-scrolling multiplayer action RPG" with class choice, permanent advanced branching, gear collection, party buffs, and long-term character progression. The live prototype supports that direction through canvas combat, side-scroller movement, class resources, enemy packs, field maps, dungeons, bosses, skill ranks, gear, upgrades, and party-skill placeholders.

The current combat is a hybrid of:

- Action platformer: real-time movement, jumping, ladders, slopes, dashes, blinks, roll movement, enemy telegraphs, projectile travel, hit reactions, and platform positioning.
- RPG: levels, stats, XP, coins, class skill points, skill ranks, gear, rarity, attunement, class mastery, boss sets, consumables, monster drops, and stat formulas.
- Metroidvania-adjacent traversal: side-scrolling maps with vertical terrain, broad platforms, ladders, ropes, drop-through behavior, map routes, portals, and regional safe zones.
- MMO-lite progression: base classes, advanced classes, party skills, monster guide, quests, dungeons, boss encounters, service towns, shops, storage, upgrades, and account/roster concepts.

It should not be designed as a pure roguelite or pure brawler. The correct class system is not "run-based random builds" and not "one-screen combo fighter." The correct target is a readable side-scrolling RPG where class expression comes from movement rhythm, range profile, resource use, skill routing, encounter matchups, gear traits, and progression choices.

### Current Player Movement Options

Observed in `project-starfall-engine.js` and `engine/movement.js`:

- Horizontal movement on platforms.
- Jumping and falling with platform collision.
- Slope support through platform surface queries.
- Ladders and climbables.
- Drop-through platform behavior.
- Ground-aware and platform-aware movement skills.
- Blinks, dashes, rolls, leaps, and grapple-style movement through `movementEffect`.
- Short invulnerability windows on selected movement skills.
- Air momentum preservation after mobility skills.
- Movement restrictions for some skills, such as mage airborne projectile blocking and teleport-on-ground constraints.

Design implication: every class must be balanced around terrain. A skill that is fine on flat ground can be broken on vertical maps if it ignores ladders, slopes, spawn lanes, or boss hazard placement.

### Current Attack And Skill Structure

Observed in `data/skills.js`, `engine/skills.js`, `engine/skill-modifiers.js`, and `project-starfall-engine.js`:

- Basic attack exists for each base class style: Fighter melee, Mage projectile, Archer projectile.
- Active skills are data-driven objects in `SKILLS`.
- Skills include `owner`, `batch`, `type`, `category`, `roleTags`, `prerequisites`, `maxRank`, `defaultRank`, `resourceCost`, `cooldown`, `lineCount`, `lineDamageScale`, `iconAsset`, `visualId`, `purpose`, `targeting`, `movementEffect`, `partyEffect`, and passive data.
- Base classes keep their base skills after advanced branching. `getClassSkills` returns inherited base skills plus advanced skills.
- Primary skill selection prefers the advanced primary training skill once the player has an advanced class.
- Special mechanics currently live in `tryUseSignatureSkill`, with generic helpers for movement, projectile, chain, field, trap, glyph, mark, buff, and area skills.
- Skill modifiers and gear traits can modify damage, cooldown, resource cost, mark duration, rune duration, armor break, burn duration, target farming, and marked/broken target scaling.

Design implication: new skills should first be expressible through data and existing generic helpers. Only add branch-specific engine logic when the skill genuinely changes behavior.

### Current Class And Build Structure

Observed in the GDD, `data/classes.js`, `data/skills.js`, `data/specializations.js`, `engine/class-mastery.js`, and `data/progression.js`:

- Base classes: Fighter, Mage, Archer.
- Advanced branches:
  - Fighter branches: Guardian, Berserker, Duelist.
  - Mage branches: Fire Mage, Rune Mage, Storm Mage.
  - Archer branches: Sniper, Trapper, Beast Archer.
- Advanced class trials unlock around level 20.
- Advanced class selection is intended around level 25.
- Specialization unlocks are represented in live data at the specialization stage, with one specialization per advanced class today.
- Class mastery grants long-term stat bonuses.
- Gear, boss sets, attunement lines, consumables, skill modifiers, and stat upgrades support build identity.

Design implication: the game should use fixed base classes with permanent advanced specializations, not weapon-only classes or a fully classless skill board. Build flexibility should come from skill rank choices, limited active loadouts, passive traits, gear, attunement, specialization bonuses, and class mastery.

### Current Enemy Behavior And Combat Demands

Observed in `data/enemies.js`, `data/combat-modifiers.js`, `project-starfall-engine.js`, and `docs/project-starfall-monster-book-blueprint.md`:

- Enemy roles include hoppers, bruisers, chargers, turrets, skirmishers, flyers, ranged attackers, healers, blockers, armored enemies, elites, and bosses.
- Enemy families and counters already point to class relevance: fire against plants, armor break against Clockbugs and tanks, marks and range against flyers, traps and AoE against swarms, burst against priority supports, and sustain/defense against boss pressure.
- Bosses use break gauges, phases, hazards, adds, and map-specific mechanics.
- Direct attacks can alert nearby monsters, which makes range and AoE a spawn-management decision rather than pure convenience.

Design implication: each class needs at least one answer to flying, armored, swarm, ranged, support, charger, and boss-pressure scenarios, but no class should answer all of them equally well.

### Current Resource Systems

Observed in `data/classes.js`, `data/skills.js`, `engine/combat-formulas.js`, `project-starfall-engine.js`, and the GDD:

- HP.
- MP, used by `resourceCost` in skill data.
- Class resource, with branch-specific names and colors:
  - Fighter: Momentum.
  - Mage: Energy.
  - Archer: Focus.
  - Guardian: Stored Impact.
  - Berserker: Rage.
  - Duelist: Tempo.
  - Fire Mage: Heat.
  - Rune Mage: Runic Energy.
  - Storm Mage: Charge.
  - Sniper: Aim.
  - Trapper: Preparation.
  - Beast Archer: Bond.
- Cooldowns.
- Party skill cooldowns.
- Skill ranks and skill points.
- Gear attunement modifiers such as resource cost reduction, cooldown recovery, resource gain, trap speed, shield strength, buff duration, and mobility tuning.

Important implementation note: in current `SKILLS`, `resourceCost` functions as MP cost. Branch resources are usually generated or spent inside engine skill logic and class-specific effects.

### Current Progression Systems

Observed in the GDD, `engine/skills.js`, `data/progression.js`, `data/specializations.js`, `engine/class-mastery.js`, `data/equipment-catalog.js`, `data/upgrades.js`, and `data/consumables.js`:

- Leveling with XP and coins.
- Base skill points and advanced skill points.
- Base Skill Batch at level 1.
- Advanced class trials around level 20.
- Advanced Skill Batch around level 25.
- Specialization around level 60.
- Mastery goals around level 100.
- Skill rank breakpoints at ranks 5 and 10.
- Class mastery milestone bonuses.
- Gear progression through shops, drops, boss sets, and random world drops.
- Attunement and upgrade systems.
- Consumables, SP manuals, and reset scrolls.

Design implication: skills should not be scattered as isolated level rewards. The batch-based skill model in the GDD is the correct structure: show the player a complete direction, then let points and prerequisites create choices.

### Current Skill UI And Hotbar Structure

Observed in `ui/skill-state.js`, `ui/skill-display.js`, `ui/skill-metadata.js`, `ui/skill-prerequisites.js`, `ui/hud.js`, `ui/input.js`, and `docs/project-starfall-ui-visual-iteration-audit.md`:

- Skill UI can show visible, trainable, usable, and locked states.
- Skill rows/cards can show rank, role tags, current effect, next-rank preview, and Level Up controls.
- Skill prerequisite helpers already explain missing requirements.
- Skill icons have generated assets and fallback rendering.
- The HUD can show HP, MP, class resource, XP, coins, class, buffs, cooldowns, boss state, and other combat metadata.
- Input supports attack, primary skill, party skill, command panels, and skill use through UI/keybind actions.

Design implication: class and skill complexity must be communicated through the Skills panel, HUD resource widgets, cooldown indicators, and concise tooltips. The UI should make inherited base skills visible for advanced classes, because the balance audit identifies hidden inherited skills as a major risk.

### Current Problems And Gaps

These are the most important class/skill gaps to fix before expanding scope:

- Some advanced branches are mechanically shallow in live data. Duelist, Storm Mage, and Beast Archer pass balance checks because inherited base skills fill their rotations, but their class-owned active kits are sparse.
- Branch resources are named and colored, but several need more distinct HUD widgets and clearer spend/build loops.
- Party skills are currently solo/self-buff or simulated-support placeholders. They need real party telemetry and encounter value later.
- Some GDD mechanics are aspirational or incomplete: Guardian blocked-hit timing, Berserker danger thresholds, Duelist parry/riposte and finisher identity, Fire Mage Overheat, Rune Mage active rune counters, Storm Mage boss alternatives, Sniper aim-state feedback, Trapper trap visibility and lure AI, and Beast Archer true companion positioning.
- Skill timing is partly implicit. Startup, active, recovery, cancel windows, and hitbox shape should become more explicit for high-polish combat tuning.
- Current movement skills are very low cooldown in data. They feel responsive, but they can invalidate terrain if not constrained by MP cost, action lock, grounded/air rules, platform targeting, and hazard design.
- Skill VFX and icons have a source-backed pipeline, but each new skill still needs clear art direction, silhouette testing, and clutter limits.

## 2. Core Design Goals For The Class System

### Recommended Class System Structure

Use this structure:

1. Fixed base classes: Fighter, Mage, Archer.
2. Permanent advanced branches at the character level: Guardian, Berserker, Duelist, Fire Mage, Rune Mage, Storm Mage, Sniper, Trapper, Beast Archer.
3. Batch-based skill unlocks, not scattered single-skill level unlocks.
4. Skill rank investment with ranks 1, 5, and 10 as meaningful behavior breakpoints.
5. Limited active loadouts/keybinds so builds have tradeoffs.
6. Passive traits and class mastery for long-term identity.
7. Equipment, boss sets, attunement, and skill modifiers as build support, not replacement class identity.
8. One specialization layer after advanced class mastery, initially one specialization per advanced branch until the live game can support more.

Do not switch the project to:

- Pure weapon-based classes. Weapons should reinforce class identity, not define it alone.
- A classless freeform skill web. It would weaken the GDD's character-first advanced-branch fantasy.
- A roguelite random skill pool. It conflicts with the current long-term RPG progression.
- Huge early skill trees. They would overload new players and hide the side-scroller action feel.

### What Each Class Should Let The Player Express

Each class should express a different answer to the same combat question: "How do I control danger while staying productive?"

- Fighter: express commitment, timing, and front-line control.
- Mage: express setup, area planning, elemental routing, and safe spell windows.
- Archer: express spacing, target selection, mobility, and mark management.
- Guardian: express protection, stagger control, stored impact, and counter-pressure.
- Berserker: express risk, health-threshold decision-making, lifesteal, and burst windows.
- Duelist: express tempo, same-target pressure, precise positioning, and finishers.
- Fire Mage: express burn setup, heat pressure, pack detonation, and area denial.
- Rune Mage: express preparation, marks, glyph placement, links, seals, and delayed payoff.
- Storm Mage: express chaining, clustered target routing, mobility, and electric tempo.
- Sniper: express aim discipline, weak points, long-range priority kills, and boss focus.
- Trapper: express preparation, lure routing, kill zones, and terrain ownership.
- Beast Archer: express companion timing, pack pressure, hybrid support, and sustain.

### Moment-To-Moment Difference

Classes should feel different before the damage number appears:

- Different approach route to enemies.
- Different safe distance.
- Different mobility rhythm.
- Different moment to commit.
- Different recovery vulnerability.
- Different resource decision.
- Different best target.
- Different reaction to enemy telegraphs.
- Different preferred terrain.
- Different sound and animation cadence.

If two classes solve the same encounter with the same input rhythm, same distance, same resource cadence, and same target priority, one of them is not finished.

### Mechanical Complexity

Use a three-layer complexity model:

- Base class layer: simple, readable, low text burden. One primary damage skill, one movement or defense tool, one AoE/control option, one passive direction.
- Advanced class layer: one branch resource loop, one identity skill, one movement/defense adjustment, one party skill, one clear weakness.
- Mastery layer: optional optimizations through skill ranks, modifiers, gear traits, attunement, boss sets, class mastery, and specializations.

New players should understand "what button keeps me alive" and "what button does my class thing" within one map. Skilled players should still have timing, spacing, combo, resource, and encounter-routing decisions after dozens of hours.

### Strengths, Weaknesses, And Decision-Making

Every class must have:

- One strong enemy matchup.
- One weak enemy matchup.
- One preferred terrain type.
- One uncomfortable terrain type.
- One efficient farming pattern.
- One bossing pattern.
- One defensive answer.
- One panic problem that cannot be solved by mashing the same skill.
- At least one skill that has a good non-damage use.
- At least one skill that is bad if used at the wrong time.

### Visual, Mechanical, And Animation Identity

Class identity must show up in these channels:

- Mechanics: resource loop, range, mobility, defense, target preference, and utility.
- Animation: stance, anticipation, weapon/casting pose, recoil, and recovery.
- VFX: color, shape language, impact pattern, projectile behavior, and field markers.
- Sound: attack texture, cast tone, impact weight, warning cues, and resource state.
- UI: resource widget, icon shapes, cooldown clarity, buff/debuff state, and skill tags.

No class should be a stat swap. A Guardian with only more defense than Fighter is not a class. A Fire Mage with only higher fire damage than Mage is not a class. A Sniper with only more Archer range is not a class.

## 3. Class Identity Framework

Use this framework for every class design sheet.

```md
## Class Name

- Core fantasy:
- Combat role:
- Movement identity:
- Range profile:
- Defensive profile:
- Resource profile:
- Skill expression style:
- Skill ceiling:
- Weaknesses:
- Visual language:
- Animation language:
- VFX language:
- Sound design direction:
- Ideal player type:
- Best situations:
- Worst situations:
- Enemy matchups:
- Environmental interactions:
- Progression path:
- Required basic kit:
- Required advanced kit:
- Required UI feedback:
- Required tutorial moment:
```

### Class Distinction Rules

When designing or revising a class:

- Start from range, movement, and resource, not damage.
- Give the class one "I am safe if I do this correctly" answer and one "I am punished if I autopilot" weakness.
- Tie the main resource to player behavior, not passive time only.
- Make the primary training skill useful but incomplete.
- Give each class one skill that changes enemy behavior, map control, or player movement.
- Give each class at least one boss-relevant option and one pack-relevant option.
- Keep inherited base skills visible and valuable after advanced branching.
- Avoid passive-only identity. If the class fantasy is not felt in active play, the class is unfinished.

## 4. Skill Design Framework

### Current Skill Data Contract

Live skill definitions already support:

- Identity: `id`, `name`, `owner`, `batch`, `type`, `category`, `purpose`, `roleTags`.
- Progression: `maxRank`, `defaultRank`, `prerequisites`, skill point pool logic.
- Cost and cadence: `resourceCost`, `cooldown`.
- Damage: `lineCount`, `lineDamageScale`.
- Presentation: `iconAsset`, `iconKind`, `visualId`, `description`.
- Behavior: `movementEffect`, `targeting`, `targetCaps`, `partyEffect`, `passiveStats`, `primaryTraining`.

Future high-polish combat tuning should add optional fields instead of hardcoding more timing:

- `startupSeconds`
- `activeSeconds`
- `recoverySeconds`
- `cancelRules`
- `hitbox`
- `airUse`
- `facingLock`
- `movementLock`
- `riskLevel`
- `skillTier`
- `statusEffects`
- `assetRequirements`

### Skill Lifecycle

Every active skill should be designed as this sequence:

1. Input request.
2. Usability check: unlocked, rank > 0, not passive, not action-locked, enough MP, enough class resource if needed, movement rules pass, cooldown ready.
3. Startup: anticipation pose, cast frame, aim state, trap arm time, guard raise, or leap windup.
4. Active event: hitbox, projectile, trap, field, buff, shield, mark, summon, or movement.
5. Enemy reaction: hit stop, knockback, stagger, burn, slow, crack, mark, link, lure, or break-gauge progress.
6. Recovery: player vulnerability, landing recovery, weapon recoil, casting settle, trap aftercast.
7. Cooldown and resource feedback.
8. Upgrade hooks and telemetry.

### Skill Design Sheet Template

```md
## Skill Name

- Class or archetype:
- Input method:
- Skill type:
- Purpose:
- Player decision:
- Startup time:
- Active frames:
- Recovery time:
- Range:
- Hitbox shape:
- Damage profile:
- Knockback:
- Cost:
- Cooldown:
- Charges:
- Cancel rules:
- Movement lock or freedom:
- Animation requirements:
- VFX requirements:
- Sound requirements:
- Enemy reaction:
- Upgrade hooks:
- Balance risks:
- Implementation notes:
- UI tooltip summary:
- Playtest questions:
```

### Skill Type Rules

- Basic attack: low commitment, no resource cost, class-readable range and animation.
- Primary training skill: low cooldown, low-to-medium damage, reinforces class loop, never solves every situation alone.
- Mobility skill: offensive or defensive use, clear terrain constraints, controlled invulnerability, no free level-skip.
- Defensive skill: strong if timed, inefficient if spammed, visible protection state.
- Utility skill: changes enemy movement, marks, fields, traps, party state, or resource economy.
- Special skill: branch-defining payoff that consumes setup, resource, or positioning.
- Ultimate or finisher: long cooldown or high resource cost, clear anticipation, strong but not encounter-invalidating.
- Passive trait: changes decisions, not just stats.
- Party skill: useful at rank 1, stronger with investment, subject to stacking rules.

## 5. Skill Expression And Player Mastery

### What Makes A Skill Expressive

A skill is expressive when the player can use it differently based on timing, positioning, terrain, enemy type, resource state, and future plan.

Bad pattern:

- Press skill on cooldown for damage.

Good patterns:

- Press now for safety, or wait for a better cluster.
- Use as movement, dodge, or engage.
- Use to mark a target now and detonate later.
- Use to reposition enemies into a trap, field, or party burst.
- Use early for resource generation, or late for execution.
- Use on ground for reliability, or from a platform for reach.
- Use during a boss break window for payoff, or save for add control.

### Movement And Combat Interaction

Movement skills should not be generic teleports. They must have class-specific meaning:

- Fighter Dash Slash: commit through a lane and threaten melee contact.
- Mage Blink/Rune Blink: create casting distance and platform repositioning.
- Archer Roll Shot/Combat Roll: maintain aim space while attacking backward.
- Guardian Shield Dash: close distance while protecting the front.
- Berserker Reckless Leap: enter danger for rage and lifesteal payoff.
- Duelist Flash Step: keep same-target tempo and create back-angle pressure.
- Fire Mage Flame Trail: reposition while leaving area denial.
- Storm Mage Static Shift: preserve chain angle and escape clustered danger.
- Trapper Grapple Dash: route enemies through prepared trap space.

### Reward Vectors

Every class should reward at least three of these:

- Timing: guard, parry, execution, boss break, dodge window.
- Positioning: trap placement, chain angle, mark target, ranged line, melee spacing.
- Aim: sniper shots, piercing arrows, projectile lanes.
- Spacing: melee risk, caster safety, archer kite distance.
- Routing: map loops, spawn lanes, trap corridors, vertical shortcuts.
- Resource management: rage, heat, runes, focus, bond, MP.
- Risk-taking: low HP Berserker, close-range Guardian bash, Duelist tempo.
- Combo planning: mark into burst, crack into armor payoff, burn into wildfire, trap into detonate.

### Avoiding "Press Button For Damage"

For every new damaging skill, add at least one of:

- Setup requirement.
- Positional requirement.
- Enemy-state requirement.
- Terrain interaction.
- Resource tradeoff.
- Recovery risk.
- Cooldown opportunity cost.
- Utility alternative use.
- Different behavior at rank 5 or rank 10.

Examples:

- Short-range high-risk attack with cancel: Duelist Quick Cut can be safe only if chained into Flash Step or timed after an enemy whiff.
- Mobility skill as offense/defense: Fire Mage Flame Trail escapes a charger while setting burn terrain.
- Area control: Rune Mage Ground Glyph blocks a lane and sets up Rune Detonation.
- Defensive timing: Guardian Impact Guard stores impact only when used before real pressure.
- Projectile tradeoff: Sniper Aimed Shot is strongest when stationary and lined up, weaker during panic movement.
- Passive approach change: Berserker Pain to Power changes whether the player saves or spends healing windows.
- Terrain interaction: Trapper Snare Trap is stronger near ledges, ladders, and spawn lanes but weaker in open boss rooms unless adds route through it.

### Synergy Without One Obvious Combo

Use "soft synergy" instead of mandatory one-button chains:

- Fire Mage can burn with Fireball, Burning Mark, Flame Trail, or Wildfire before Inferno Burst.
- Rune Mage can create marks through Rune Mark, Arcane Link, Ground Glyph, or Grand Inscription before detonation.
- Archer/Sniper can choose Marked Shot, Weak Point Mark, or gear traits for priority target pressure.
- Trapper can use Snare Trap, Spike Trap, Lure Shot, Tripwire, or Kill Zone depending on terrain.

Avoid exact mandatory rotations such as "always press A, then B, then C." The same class should route differently against flyers, tanks, swarms, support enemies, and bosses.

### Casual And Mastery Support

For casual players:

- Primary training skill must be reliable.
- UI must label the best use case.
- Defensive/mobility skill must be easy to find.
- Rank 1 skills must work.
- Early enemies should visibly react to correct tools.

For mastery players:

- Add timing breakpoints.
- Add terrain-sensitive value.
- Add boss break optimization.
- Add resource overcap prevention.
- Add skill modifiers that alter behavior.
- Add high-rank upgrades with meaningful side effects, not only damage.

## 6. Recommended Class Archetypes

The recommended roster is the existing roster. Do not add extra archetypes until the following are polished.

### Fighter - Balanced Starter

- Class fantasy: grounded weapon fighter who wins through commitment, impact, and control.
- Mechanical identity: melee pressure, Momentum building, stagger, armor break, short-range AoE.
- Primary attack style: close melee basic plus Heavy Strike.
- Secondary mechanic: Momentum Burst as a resource payoff.
- Resource model: Momentum gained through melee contact and pressure.
- Mobility profile: Dash Slash for lane commitment.
- Defensive tool: Guard.
- Utility role: Power Break and Ground Slam for control.
- Example skill kit: Heavy Strike, Dash Slash, Guard, Ground Slam, Power Break, Momentum Burst.
- Passive trait: Fighter Damage Mastery.
- Strengths: stable onboarding, armored enemies, close-range packs, simple boss uptime.
- Weaknesses: flyers, high vertical spread, ranged enemies on separated platforms.
- Required animations: planted idle, run, jump, melee basic, heavy swing, dash slash, guard raise, ground slam, hit recoil.
- Required VFX: steel slash, ground impact ring, guard flash, crack marker.
- Required UI elements: Momentum meter, guard status, break status.
- Implementation notes: keep Fighter readable as the base melee grammar for Guardian, Berserker, and Duelist.
- Balance risks: if Heavy Strike plus Dash Slash solves everything, advanced Fighter branches lose identity.

### Mage - Area Setup Starter

- Class fantasy: arcane caster who controls space and uses marks to create burst.
- Mechanical identity: ranged projectiles, MP management, Energy generation, spell marks, area magic.
- Primary attack style: Magic Bolt.
- Secondary mechanic: mark into Energy Release.
- Resource model: Energy gained through spell pressure and spent on burst.
- Mobility profile: Blink to maintain casting distance.
- Defensive tool: Mana Shield.
- Utility role: Spell Mark and Arcane Burst.
- Example skill kit: Magic Bolt, Blink, Arcane Burst, Mana Shield, Spell Mark, Energy Release.
- Passive trait: Mage Damage Mastery.
- Strengths: safe lane control, groups, marked payoff, enemy clusters.
- Weaknesses: interrupted casting, airborne restrictions, fast melee rushers if Blink is wasted.
- Required animations: cast windup, projectile release, blink vanish/reappear, shield cast, burst release.
- Required VFX: blue arcane bolts, mark sigil, blink sparkle, shield shell.
- Required UI elements: MP clarity, Energy meter, mark indicator.
- Implementation notes: preserve caster foot-planting so spell windows have risk.
- Balance risks: Blink plus long range can trivialize melee maps if enemy pressure and MP costs are too low.

### Archer - Mobile Range Starter

- Class fantasy: agile ranged fighter who wins through spacing, marks, and movement.
- Mechanical identity: high mobility, Focus, quick projectiles, marks, piercing lines.
- Primary attack style: Quick Shot and ranged basic.
- Secondary mechanic: marks into Focused Volley.
- Resource model: Focus gained through sustained ranged pressure and target control.
- Mobility profile: Roll Shot for spacing while attacking.
- Defensive tool: movement, distance, and Eagle Stance positioning.
- Utility role: Marked Shot and Piercing Arrow.
- Example skill kit: Quick Shot, Roll Shot, Marked Shot, Piercing Arrow, Eagle Stance, Focused Volley.
- Passive trait: Archer Damage Mastery.
- Strengths: flyers, priority targets, safe kiting, vertical ranged lanes.
- Weaknesses: cramped arenas, blockers, enemies that force close-range panic, swarms from both sides.
- Required animations: bow draw, quick release, roll-shot, stance, volley.
- Required VFX: golden arrow streaks, mark flash, pierce trail.
- Required UI elements: Focus meter, mark state, stance buff.
- Implementation notes: keep Archer mobile but require line of sight, platform management, and target choice.
- Balance risks: too much range plus too much mobility makes melee enemy design irrelevant.

### Guardian - Tank / Defender / Control

- Class fantasy: shield vanguard who stores enemy impact and turns pressure into control.
- Mechanical identity: Stored Impact, stagger, crack, shields, counters, party protection.
- Primary attack style: Shield Bash.
- Secondary mechanic: block or absorb pressure to empower Retaliation Wave and Guardian's Verdict.
- Resource model: Stored Impact from guarding, blocking, and close-range contact.
- Mobility profile: Shield Dash, controlled forward engage.
- Defensive tool: Impact Guard and Oath Barrier.
- Utility role: Shield Wall party protection, stagger and armor crack.
- Example skill kit: Shield Bash, Shield Dash, Impact Guard, Oath Barrier, Retaliation Wave, Guardian's Verdict, Shield Wall.
- Passive trait: Hold the Line.
- Strengths: armored enemies, boss sustain, charger enemies, party safety.
- Weaknesses: spread flyers, enemies that refuse contact, long-range turret maps.
- Required animations: shield idle, bash, shield dash, guard brace, barrier plant, counter wave.
- Required VFX: blue/steel shields, shockwaves, impact rings, crack icons.
- Required UI elements: Stored Impact segments, block window flash, active barrier timer.
- Implementation notes: add explicit guard timing and blocked-hit feedback before adding more Guardian skills.
- Balance risks: permanent high defense can remove skill expression; Guardian must still need timing and positioning.

### Berserker - Risk Melee / Sustain Bossing

- Class fantasy: reckless heavy attacker who converts danger into rage and recovery.
- Mechanical identity: Rage, cleave, low-HP pressure, lifesteal, burst windows.
- Primary attack style: Blood Cleave.
- Secondary mechanic: low-health and rage thresholds.
- Resource model: Rage gained by dealing and taking damage, spent on burst/sustain.
- Mobility profile: Reckless Leap.
- Defensive tool: Crimson Recovery, lifesteal, timed invulnerability windows.
- Utility role: War Cry party damage pressure.
- Example skill kit: Blood Cleave, Rage Surge, Reckless Leap, Crimson Recovery, Last Stand, War Cry.
- Passive trait: Pain to Power.
- Strengths: boss pressure, dense melee packs, sustained solo fights.
- Weaknesses: bursty enemies, ranged kite maps, hazard-heavy arenas, heal denial.
- Required animations: heavy cleave, rage pulse, leap slam, recovery slash, last stand burst.
- Required VFX: red/orange slash arcs, rage aura, bloodless crimson energy, danger pulse.
- Required UI elements: Rage meter, low-HP threshold marker, recovery cooldown, danger state.
- Implementation notes: distinguish safe training from risky boss burst. The player must know when they are in the danger band.
- Balance risks: if sustain is too high, low-HP play becomes fake risk. If too low, the class feels unfair.

### Duelist - Mobility Assassin / Tempo Bossing

- Class fantasy: precise blade specialist who wins by staying on one target and exploiting tempo.
- Mechanical identity: Tempo, same-target combo pressure, fast cuts, repositioning, finishers.
- Primary attack style: Quick Cut.
- Secondary mechanic: same-target chain and tempo maintenance.
- Resource model: Tempo gained by repeated clean hits on the same target, lost by whiffing or disengaging.
- Mobility profile: Flash Step.
- Defensive tool: intended parry/riposte and evasive repositioning; current live kit needs expansion.
- Utility role: Rallying Flourish party tempo/crit support.
- Example skill kit: Quick Cut, Flash Step, Rallying Flourish, plus future Parry, Riposte Mark, and True Finisher.
- Passive trait: Duelist Damage Mastery.
- Strengths: single boss, elite duels, priority targets, tight execution windows.
- Weaknesses: large spread packs, flyers on separate lanes, heavy unavoidable area damage.
- Required animations: quick triple slash, step-through, parry flash, riposte, finisher pose.
- Required VFX: narrow silver/gold cuts, tempo afterimages, target lock glint.
- Required UI elements: Tempo meter, same-target counter, finisher ready cue.
- Implementation notes: this branch needs more class-owned active depth. Add parry/riposte before adding raw damage.
- Balance risks: same-target bonuses can become invisible math unless UI and hit feedback show tempo clearly.

### Fire Mage - Burn / Mobbing Caster

- Class fantasy: volatile elemental caster who spreads flame and detonates burned packs.
- Mechanical identity: Heat, burn spread, area denial, wildfire payoff.
- Primary attack style: Fireball.
- Secondary mechanic: burn state into area detonation.
- Resource model: Heat generated by fire hits and burned enemies, vented or spent for burst.
- Mobility profile: Flame Trail.
- Defensive tool: spacing, burn terrain, emergency reposition.
- Utility role: Ignition Aura party damage and burn pressure.
- Example skill kit: Fireball, Flame Trail, Burning Mark, Heat Vent, Wildfire, Inferno Burst, Ignition Aura.
- Passive trait: Fire Mage Damage Mastery.
- Strengths: clustered mobs, plants, low-fire-resistance packs, predictable spawn lanes.
- Weaknesses: fire-resistant enemies, mobile ranged enemies, bosses without add or burn windows.
- Required animations: fire cast, trail dash, heat vent, wildfire spread, inferno burst.
- Required VFX: fire, ash, ember trails, heat shimmer, burn icons.
- Required UI elements: Heat meter, burn counters, Overheat warning if implemented.
- Implementation notes: add explicit Overheat or Heat cap behavior only if it creates choices, not punishment for normal play.
- Balance risks: burn spread can dominate field maps if target caps, duration, and tick damage are not controlled.

### Rune Mage - Setup / Control / Support

- Class fantasy: tactical glyph caster who prepares runes, links enemies, and detonates planned fields.
- Mechanical identity: Runic Energy, marks, glyphs, links, seals, delayed payoff.
- Primary attack style: Rune Mark.
- Secondary mechanic: active rune count and detonation windows.
- Resource model: Runic Energy from rune application and field upkeep.
- Mobility profile: Rune Blink.
- Defensive tool: Mana Seal and controlled fields.
- Utility role: Rune Circle party support.
- Example skill kit: Rune Mark, Rune Blink, Ground Glyph, Arcane Link, Rune Detonation, Mana Seal, Grand Inscription, Rune Circle.
- Passive trait: Rune Mage Damage Mastery.
- Strengths: controlled arenas, armored/elite setups, party utility, boss preparation.
- Weaknesses: chaotic fast maps, enemies that leave fields, immediate burst checks.
- Required animations: rune inscribe, blink, ground glyph, linking cast, seal channel, detonation.
- Required VFX: geometric glyphs, circles, linked lines, runic pulses.
- Required UI elements: active rune counter, rune duration, linked target state, field timer.
- Implementation notes: add visible active rune counters before increasing Rune Mage complexity.
- Balance risks: delayed payoff can feel weak in solo leveling if setup time is not compensated.

### Storm Mage - Chain / Cluster Mobbing

- Class fantasy: kinetic lightning caster who routes energy through enemy clusters.
- Mechanical identity: Charge, chain targeting, clustered pack damage, static repositioning.
- Primary attack style: Chain Bolt.
- Secondary mechanic: chain angle, chain target count, charge buildup.
- Resource model: Charge generated by chained hits and spent for stronger storm effects.
- Mobility profile: Static Shift.
- Defensive tool: mobility and slows from electric effects.
- Utility role: Stormfront party speed/pressure.
- Example skill kit: Chain Bolt, Static Shift, Stormfront, plus future single-target condenser and anti-boss charge spender.
- Passive trait: Storm Mage Damage Mastery.
- Strengths: flyers, clustered packs, vertical chains, dense field routes.
- Weaknesses: isolated bosses, spread enemies, lightning-resistant or chain-breaking mechanics.
- Required animations: lightning cast, static blink, stormfront channel.
- Required VFX: lightning forks, chain arcs, static fields, charge sparks.
- Required UI elements: Charge meter, chain count feedback, no-chain warning.
- Implementation notes: this branch needs boss alternatives so it is not only a field-clear class.
- Balance risks: chain skills can lead every field metric if target caps and falloff are too generous.

### Sniper - Long-Range Bossing / Priority Kill

- Class fantasy: disciplined marksman who lines up weak points and ends priority targets.
- Mechanical identity: Aim, weak points, long range, armor pierce, execution.
- Primary attack style: Aimed Shot.
- Secondary mechanic: weak-point setup and steady aim.
- Resource model: Aim built by steady shots, marks, and disciplined spacing.
- Mobility profile: Combat Roll.
- Defensive tool: distance, roll, target deletion.
- Utility role: Eagle Eye party crit/vision support.
- Example skill kit: Aimed Shot, Combat Roll, Weak Point Mark, Pierce Armor, Execution Shot, One Perfect Shot, Eagle Eye.
- Passive trait: Steady Breath.
- Strengths: bosses, elites, flyers, priority ranged/support enemies.
- Weaknesses: swarm pressure, cramped arenas, enemies behind blockers, forced constant movement.
- Required animations: aim stance, recoil, roll, weak-point mark, execution shot.
- Required VFX: aim line, weak-point flash, piercing arrow trail, critical hit flare.
- Required UI elements: Aim meter, steady indicator, weak-point duration, execution-ready cue.
- Implementation notes: add aim-state feedback and boss weak-point telegraphs before increasing sniper burst.
- Balance risks: extreme range can trivialize enemies unless line of sight, blockers, adds, and hazard pressure matter.

### Trapper - Preparation / Area Control

- Class fantasy: field engineer who wins before the enemy reaches them.
- Mechanical identity: Preparation, traps, lure routing, detonation, kill zones.
- Primary attack style: Snare Trap.
- Secondary mechanic: trap placement and detonation timing.
- Resource model: Preparation generated through setup and trap interactions.
- Mobility profile: Grapple Dash.
- Defensive tool: traps, slows, lures, reposition.
- Utility role: Tactical Field party control.
- Example skill kit: Snare Trap, Grapple Dash, Spike Trap, Lure Shot, Tripwire, Detonate, Kill Zone, Tactical Field.
- Passive trait: Trapper Damage Mastery.
- Strengths: spawn lanes, swarms, chargers, platform chokepoints, party setup.
- Weaknesses: open boss arenas, flying enemies that ignore traps, enemies that cannot be lured.
- Required animations: trap toss/place, grapple movement, lure shot, detonate gesture, field deploy.
- Required VFX: wires, trap circles, arming pulses, mechanical/nature devices, detonation burst.
- Required UI elements: active trap count, trap readiness, arming state, trigger radius.
- Implementation notes: improve trap visibility and enemy lure behavior before adding more trap variants.
- Balance risks: traps can either be useless on bosses or oppressive in fields. Boss-specific partial value is required.

### Beast Archer - Companion / Hybrid Support

- Class fantasy: archer bonded with a companion, blending ranged pressure, sustain, and pack support.
- Mechanical identity: Bond, companion strikes, pack marks, hybrid sustain.
- Primary attack style: Companion Strike.
- Secondary mechanic: companion timing and pack-mark payoff.
- Resource model: Bond gained through coordinated hits, healing, and companion actions.
- Mobility profile: Pounce Roll.
- Defensive tool: sustain, companion peel, movement.
- Utility role: Pack Call party support.
- Example skill kit: Companion Strike, Pounce Roll, Pack Call, plus future Companion Guard, Sic Command, Recall, and Bond spender.
- Passive trait: Beast Archer Damage Mastery.
- Strengths: solo sustain, hybrid party contribution, marked targets, long fights.
- Weaknesses: companion-hostile boss mechanics, very fast target swaps, high burst damage.
- Required animations: archer shot with companion cue, pounce roll, pack call, companion attack, companion recall.
- Required VFX: claw sparks, pack marks, green/gold bond aura, companion trail.
- Required UI elements: Bond meter, companion status, companion cooldown/position, pack mark state.
- Implementation notes: this branch needs a true companion actor or at least a visible companion proxy before its fantasy is complete.
- Balance risks: if companion damage is passive and safe, the class becomes low-interaction. Make companion timing active.

## 7. Progression And Unlock Structure

### Recommended Pacing

- Level 1: base class selected. Player starts with basic attack, one primary training skill, HP/MP/class resource HUD, and one clear class fantasy prompt.
- Levels 2-5: first movement or defensive tool unlocks. This teaches survival before damage complexity.
- Levels 6-10: first AoE/control/setup option unlocks.
- Levels 11-19: first real build choice appears through rank investment, passive choice, and role-tagged skills.
- Level 20: advanced class trials become available. Trials preview mechanics and enemy matchups for each branch.
- Level 25: advanced class selection. Treat this as a permanent character identity choice after confirmation and preview.
- Levels 25-35: advanced primary training skill, branch movement/defense, branch resource widget, and first party skill access.
- Levels 35-50: branch payoff skills and first meaningful skill modifier choices.
- Level 60: specialization layer. Start with one specialization per advanced class until the class kits are complete.
- Levels 60-100: mastery, boss sets, attunement, advanced modifiers, party contribution, encounter mastery.

### Loadout Recommendation

Use limited active availability even if the Skills panel can show all learned skills:

- Basic attack: always available.
- Primary skill: always available and auto-selected from current class/advanced class.
- Mobility/defense slot: one dedicated survivability input.
- Active skill slots: 3 to 4 equipped class skills.
- Party skill slot: one advanced branch party skill.
- Consumable/utility slots: separate from class skill slots.

Reason: the current data can support many inherited skills. If every active skill is equally available with no loadout pressure, advanced choices become rotation clutter instead of build expression.

### Skill Tree Structure

Keep the GDD's batch model:

- Base Skill Batch: visible from level 1.
- Advanced Skill Batch: visible after branch selection.
- Specialization Skill Batch: visible after specialization unlock.
- Mastery/Modifier layer: visible after the player understands the core loop.

Each node should show:

- Current rank.
- Max rank.
- Pool cost.
- MP cost.
- Cooldown.
- Purpose tags.
- Prerequisites.
- Rank 5 breakpoint.
- Rank 10 breakpoint.
- Next rank behavior.

### Upgrade Philosophy

Rank upgrades should change behavior at breakpoints:

- Add one target, not just 5 percent damage.
- Add shorter arm time, not just more trap damage.
- Add wider guard timing, not just more defense.
- Add a new detonation interaction, not just bigger rune number.
- Add improved burn spread logic, not just burn damage.
- Add a mobility cancel or landing safety window, not just cooldown reduction.

Use pure number scaling for ranks 2-4 and 6-9. Use behavior changes for ranks 5 and 10.

### Respecs And Permanence

- Skill ranks should be respec-friendly through reset scrolls or town services.
- Advanced branch should be permanent per character in the main design, because the GDD is character-first and roster-driven.
- During beta, allow an admin or expensive reset path for testing.
- Gear, attunement, skill modifiers, and loadouts should remain adjustable.

## 8. Combat Readability And Game Feel

### Animation Timing Rules

- Basic attacks: 4 to 8 frames of anticipation, fast impact, short recovery.
- Primary training skills: readable but snappy. Do not exceed the current low-cooldown class rhythm.
- Heavy melee: longer anticipation, strong impact pause, visible recovery.
- Caster fields: clear cast pose, visible ground marker before full damage.
- Projectile skills: muzzle/cast flash, readable projectile body, clear impact.
- Defensive skills: protection must appear before the dangerous hit lands.
- Mobility skills: start direction must be legible before movement blur or blink.
- Ultimates/finishers: larger anticipation and recovery are acceptable if payoff is clear.

### Impact Rules

- Use short hit pause on meaningful hits, not every tiny burn tick.
- Use screen shake only for heavy melee, boss break, slam, ultimate, or major crits.
- Keep normal repeated hits stable to avoid visual fatigue.
- Enemy hit reaction should identify hit type: stagger, knockback, burn, crack, mark, slow, or defeat.

### VFX Clarity Rules

- Skill VFX must not hide enemy telegraphs, hazards, ledges, ladders, or player feet.
- Keep field effects translucent enough to see enemy silhouettes.
- Use different shape language by class:
  - Guardian: shields, rings, steel-blue shockwaves.
  - Berserker: red arcs, rage pulses, aggressive slash trails.
  - Duelist: thin cuts, afterimages, lock glints.
  - Fire Mage: flame tongues, ash, heat shimmer.
  - Rune Mage: geometric glyphs, circles, linked lines.
  - Storm Mage: forks, chains, static sparks.
  - Sniper: aim lines, weak-point flashes, precision trails.
  - Trapper: wires, trap rings, mechanical/nature devices.
  - Beast Archer: claw sparks, companion trails, pack aura.
- Do not stack multiple full-screen effects from normal skills.
- Do not use VFX as decoration if it does not communicate timing, area, or status.

### Side-Scroller Terrain Rules

- Hitboxes must visually match slopes and platform height.
- Ground effects must snap to platform surfaces, not float at actor center.
- Projectiles should make their lane readable.
- Vertical skills need clear target rules: same platform, nearest platform, line of sight, chain range, or field radius.
- Traps and glyphs must show arming state and trigger radius.
- Movement skills must show start and end points clearly enough that the player understands why they did or did not land safely.

## 9. Balance Framework

### Balance Goals

- Every class has a viable solo route.
- Every class has a party reason.
- Every class has a mobbing path and bossing path, but not equal excellence in both.
- No class is best at damage, safety, range, mobility, and support at the same time.
- No skill invalidates map traversal, boss mechanics, or enemy roles.
- No early enemy hard-counters a class.
- No class depends on rare gear to function.

### Current Benchmarks To Preserve

From `docs/project-starfall-balance-optimization-audit.md`:

- Skill system health currently reports 85 skills, 70 active and 15 passive.
- All owners have a core rotation, movement/survival, role utility, and identity cooldown.
- Boss TTK target is 6 to 12 minutes, with current median solo boss around 8.91 minutes.
- Current strongest field clear classes include Storm Mage, Trapper, and Fire Mage.
- Current strongest single-target leaders include Archer, Duelist, Berserker, and Sniper.
- Guardian leads sustain and armored-pressure profiles.

Use these as direction checks, not as final locked values.

### Power Budget Rules

- Damage vs utility: a skill with slow, mark, crack, shield, field, or trap utility must deal less immediate damage than a pure damage skill.
- Range vs safety: longer range needs lower burst, aim requirement, projectile travel, line of sight, or recovery.
- Mobility vs damage: a skill with invulnerability and repositioning should not also be top-tier damage unless it has high cost or strict condition.
- Crowd control vs cooldown: hard control needs longer cooldown, shorter duration on bosses, or target caps.
- Defense vs vulnerability: strong defense needs timing, resource cost, recovery, directionality, or limited duration.
- Resource generation vs payoff: generators should be safe and frequent but modest; spenders should create clear windows.
- Passive scaling vs active identity: passive bonuses should support a playstyle, not outdamage active skill mastery.

### Cooldown Rules

- Primary trainers: low cooldown, low-to-medium damage, core loop reinforcement.
- Mobility skills: low cooldown can work only with MP cost, terrain rules, and controlled invulnerability.
- Area-control skills: medium cooldown, visible duration, target caps.
- Defensive reaction skills: medium cooldown, strong feedback, timing requirement.
- Party skills: long cooldown, useful rank 1, clear stacking category.
- Finishers/ultimates: long cooldown or high class-resource cost, clear setup and recovery.

### Boss Rules

- Bosses should resist or scale down hard crowd control, not ignore entire class kits.
- Armor break, weak point, burn, rune, trap, and companion mechanics should have boss-specific partial value.
- Break windows should reward preparation without making pre-break time irrelevant.
- Boss adds should occasionally let mobbing classes contribute their identity.
- Boss arenas should include at least one terrain feature that matters but does not favor only ranged or mobility classes.

### Skill Tuning Table Format

Use this table for every skill balance pass:

| Skill | Owner | Tier | Damage | Cooldown | MP Cost | Class Resource | Range | Startup | Recovery | Area Size | CC Value | Mobility Value | Risk | Best Use | Weakness |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| Fireball | Fire Mage | 1 | Medium | 0.62s | 5 | Builds Heat | 500 | Short | Short | Small impact | Burn | None | Low | Burn starter | Fire-resistant targets |
| Shield Bash | Guardian | 1 | Low-Medium | 0.48s | 4 | Builds Stored Impact | Short | Short | Short | Narrow front | Stagger/Crack | None | Medium | Frontline control | Flyers/spread targets |
| Chain Bolt | Storm Mage | 1 | Low per hit | 0.78s | 6 | Builds Charge | 380 chain | Short | Short | Chain | Slow | None | Low | Clustered packs | Isolated bosses |

### Tuning Process

1. Test each skill alone against dummy targets.
2. Test each skill in its intended class rotation.
3. Test against swarm, flyer, armored, support, ranged, charger, elite, and boss enemies.
4. Test on flat, vertical, slope, ladder, and split-platform maps.
5. Test at rank 1, rank 5, and rank 10.
6. Test with no gear, expected gear, strong gear, and extreme attunement.
7. Compare field KPM, boss TTK, damage taken, potion use, and idle time.
8. Nerf safety before nerfing identity.
9. Buff utility before buffing raw damage if a class feels weak but distinctive.

## 10. Implementation Guidance

### Technical Direction

Use the current data-driven JavaScript module structure. Do not introduce Unity-style ScriptableObjects or a new framework.

Represent class and skill content through:

- Class definitions in `data/classes.js`.
- Skill definitions in `data/skills.js`.
- Guide-backed design contracts in `data/class-skill-design.js`.
- Specializations in `data/specializations.js`.
- Skill modifiers in `engine/skill-modifiers.js`.
- Combat formulas in `engine/combat-formulas.js`.
- Movement rules in `engine/movement.js`.
- Enemy data in `data/enemies.js`.
- Animation contracts in `data/animations.js`.
- FX contracts in `data/combat-fx.js`.
- UI summaries and states in `ui/skill-*.js`.

### Executable Design Contract

`js/games/project-starfall/data/class-skill-design.js` is the machine-readable bridge between this guide and runtime data. Keep it updated when changing class resources, status effects, tooltip structure, loadout assumptions, balance table fields, encounter test coverage, or debug scenario coverage.

Required exported contract groups:

- `CLASS_SKILL_GUIDE_CONTRACT`
- `CLASS_RUNTIME_REQUIRED_FIELDS`
- `ADVANCED_CLASS_REQUIRED_FIELDS`
- `SKILL_RUNTIME_REQUIRED_FIELDS`
- `SKILL_DESIGN_OPTIONAL_FIELDS`
- `CLASS_RESOURCE_DEFINITIONS`
- `STATUS_EFFECT_DEFINITIONS`
- `CLASS_SKILL_TOOLTIP_FORMAT`
- `CLASS_SKILL_BALANCE_TUNING_FIELDS`
- `CLASS_SKILL_LOADOUT_SLOTS`
- `CLASS_SKILL_ENCOUNTER_TEST_CASES`
- `CLASS_SKILL_DEBUG_SCENARIOS`

Run the guide-backed validator after every class, skill, animation, VFX, status, resource, tooltip, or progression change:

```bash
npm run validate:project-starfall-class-skills
```

The validator must prove:

- The guide still contains the required sections.
- All expected base and advanced classes exist.
- Every class has required runtime fields, role profile data, and a resource definition.
- Every advanced class has a valid base class, level requirement, and party skill.
- Every skill has required schema fields, valid owner, role tags, purpose, cost, cooldown, icon, prerequisites, and behavior hook.
- Every active skill has generated combat FX metadata.
- Every passive skill has passive stats.
- Advanced branches keep exactly one primary training skill and a documented party skill.
- Advanced branches inherit their base skill list through `engine/skills.js`.
- Status definitions, tooltip concepts, loadout slots, balance fields, encounter tests, and debug scenarios remain complete.

### New Skill Implementation Checklist

1. Add the skill to `SKILLS` in `data/skills.js`.
2. Assign `owner`, `batch`, `type`, `category`, `purpose`, `roleTags`, `maxRank`, cost, cooldown, damage lines, icon, and visual id.
3. Define prerequisites using the existing prerequisite helpers.
4. Use existing generic behavior when possible: movement, projectile, chain, trap, field, mark, buff, area, or passive.
5. Add branch-specific logic in `tryUseSignatureSkill` only if generic helpers cannot express the skill.
6. Add or reuse `SKILL_VISUALS` and `SKILL_FX_ANIMATION_ASSETS`.
7. Add icon source and processed icon path.
8. Add UI summary text in skill display helpers if generic summaries are unclear.
9. Add skill modifier hooks if ranks or gear should change behavior.
10. Add balance harness coverage if the skill changes role performance.
11. Verify save/load does not break learned ranks or equipped keybinds.
12. Test on at least one flat map, one vertical map, one dungeon, and one boss.

### Skill State Machine Guidance

The engine should treat active skills as states:

- `requested`
- `validated`
- `startup`
- `active`
- `recovery`
- `cooldown`

Current code already validates cooldown, MP, action lock, climb state, movement rules, and rank. The next polish step is to make startup/recovery data explicit so designers can tune feel without rewriting engine branches.

### Hitbox And Projectile Guidance

- Keep melee hitboxes tied to facing and lane.
- Keep projectiles lane-readable and right/left facing.
- Use `targeting` fields for range, speed, count, pierce, chain, explosion, slow, mark, burn, crack, and homing.
- Use `targetCaps` for every field, chain, trap, and area skill.
- Use platform surface checks for traps, glyphs, and ground effects.
- Keep boss hitbox interactions predictable. Boss size should not multiply every field tick unless explicitly intended.

### Resource Cost Validation

- MP cost comes from `resourceCost`.
- Class resource costs or spenders should be explicit in branch logic or future `classResourceCost`.
- Skills that generate class resource should show generation in UI or combat feedback.
- Do not hide class resource generation inside passive math if it affects player decisions.

### Save/Load Integration

When adding class/skill features, preserve:

- Learned ranks.
- Skill points.
- Advanced class selection.
- Specialization selection.
- Class mastery.
- Keybinds/loadout.
- Active modifiers.
- Gear traits and attunement.

Do not rename skill ids after release without migration code.

### Debug Tools

Add or maintain debug views for:

- Current class resource.
- Cooldowns.
- Active buffs/debuffs.
- Trap/glyph/field counts.
- Projectile count.
- Enemy role and target state.
- Skill damage lines.
- Boss break gauge contribution.
- KPM and TTK.
- Damage taken by source.
- Skill usage frequency.

## 11. Required Animation And Asset Guidance For Skills

### Current Asset Contracts

Observed in `data/animations.js`, `data/combat-fx.js`, and `img/project-starfall/asset-prompts.md`:

- Player runtime sheets use a 6 by 10 grid of 160px frames.
- Player rows: idle, run, jump, fall, climb, basic, skill, party, hit, defeat.
- Enemy sheets use a 6 by 8 grid.
- Enemy rows: idle, move, telegraph, attack, projectile, buff, hit, defeat.
- Skill FX sheets use a 6 by 4 grid.
- Skill FX rows: cast, projectile, impact, area.
- Source sheets should include thin `#00ffff` guide lines around cells.
- Chroma backgrounds are generally flat `#ff00ff` or `#00ff00`, depending on processor.
- Icons should be centered, readable, text-free, label-free, and watermark-free.

### Required Assets By Skill Type

- Basic melee: player basic row, slash FX, impact FX, hit sound.
- Projectile: cast row, projectile row, impact row, projectile sprite, impact sound.
- Mobility: movement pose, start burst, travel trail, end burst, foot/landing sound.
- Defensive: guard/channel pose, shield/aura loop, absorb impact, break/end cue.
- Trap: placement pose, trap object sprite, arming pulse, trigger VFX, detonation VFX.
- Field/glyph: cast pose, ground marker, active loop, expiration cue, detonation cue.
- Mark/debuff: cast or shot pose, target icon, status pulse, expiration cue.
- Party skill: party row animation, aura, buff icon, sound cue, HUD timer.
- Finisher: unique startup pose, impact FX, hit pause cue, recovery pose, special sound.

### Recommended Frame Counts

- Basic/primary: 6 frames are enough if anticipation, strike, and recovery are distinct.
- Heavy/finisher: use 6 frames but exaggerate frame 2 or 3 for anticipation and frame 4 for impact.
- Projectile loops: 6 projectile frames should read as motion, not flicker.
- Field effects: 6 area frames should loop softly and not cover actors.
- Hit effects: 3 to 6 visible frames, fast fade.

### Asset Prompt Templates

#### Class Idle Pose

```text
Create a Project Starfall playable class sprite source cell for [CLASS NAME]. Style: Starlit Frontier Fantasy, clean cel-shaded side-scroller sprite, crisp contour outline, readable silhouette, practical crafted gear, luminous fallen-star accents. Pose: idle side-view stance facing right, full body connected silhouette, no detached limbs, no attack motion. Background: transparent-ready flat #ff00ff. No text, no labels, no watermark.
```

#### Basic Attack Animation Sheet

```text
Create a 6-frame horizontal sprite animation row for [CLASS NAME] using Project Starfall's 160px player frame style. Action: [basic melee swing / bow shot / spell cast]. Include clear anticipation, strike or release, and recovery. Keep feet and weapon readable in side view, facing right. Use transparent-ready flat #ff00ff background and thin #00ffff guide lines around each cell. No text, no UI, no watermark.
```

#### Special Skill FX Sheet

```text
Create a 6 by 4 bordered sprite sheet for Project Starfall skill VFX: [SKILL NAME]. Rows in order: cast, projectile, impact, area. Six frames per row. Style: Starlit Frontier Fantasy, readable side-scroller combat VFX, [CLASS VFX LANGUAGE], clear silhouettes, no character bodies, no UI, no text, no labels, no watermark. Use flat #ff00ff or #00ff00 chroma background and thin #00ffff guide lines. Do not use #00ffff inside the art.
```

#### Skill Icon

```text
Create one centered RPG skill icon for Project Starfall: [SKILL NAME]. Symbol should communicate [PURPOSE] for [CLASS NAME]. Style: Starlit Frontier Fantasy, clean cel-shaded icon, readable at 64px, crisp outline, no UI frame, no text, no numerals, no watermark. Use flat #ff00ff chroma background with generous padding.
```

#### Buff Aura

```text
Create a 6-frame Project Starfall buff aura row for [BUFF NAME]. The aura should sit around a side-scroller character without hiding the character silhouette. Visual language: [CLASS COLORS AND SHAPES]. Soft loop, readable at gameplay scale, no text, no UI, no watermark, transparent-ready chroma background.
```

#### Impact Effect

```text
Create a 6-frame Project Starfall impact VFX row for [HIT TYPE]. The impact should read instantly as [stagger / burn / crack / weak point / electric chain / trap detonation]. Keep the effect compact enough to avoid hiding enemies or hazards. No text, no UI, no watermark, transparent-ready chroma background.
```

### Avoiding AI-Generated-Looking Assets

- Keep silhouettes functional, not ornamental.
- Avoid over-detailed noise that disappears at 160px or 64px.
- Do not let glows cover feet, ledges, ladders, or enemy telegraphs.
- Keep class palettes consistent across icon, skill FX, and animation.
- Use practical equipment shapes and clear weapon profiles.
- Reject assets with fake text, pseudo-runes that look like letters, uneven cell framing, broken anatomy, detached limbs, or cropped VFX.

## 12. Enemy And Encounter Interaction

### Class Matchup Matrix

| Class | Counters | Struggles Against | Encounter Requirement |
| --- | --- | --- | --- |
| Fighter | bruisers, armored enemies, close packs | flyers, spread ranged enemies | give reachable platforms and armor-break value |
| Mage | clustered packs, marked targets | fast rushers, interrupted casting | provide cast windows and threats that punish bad Blink use |
| Archer | flyers, priority targets, ranged lanes | cramped swarm pressure | include blockers and flank pressure so range is not free |
| Guardian | chargers, bosses, armored enemies | spread flyers, turret maps | include blockable hits and meaningful boss pressure |
| Berserker | bosses, dense melee packs | burst hazards, ranged kiting | give sustain windows but punish reckless overcommitment |
| Duelist | elites, bosses, priority targets | spread packs, flying swarms | include same-target payoff and parryable attacks |
| Fire Mage | plants, swarm packs, lane clusters | fire resistance, isolated bosses | include burnable packs and some burn-resistant checks |
| Rune Mage | controlled arenas, armored elites | chaotic fast maps | give setup windows and enemies that can be linked/sealed |
| Storm Mage | flyers, clustered vertical enemies | isolated bosses | include clusters but add chain falloff and boss alternatives |
| Sniper | bosses, flyers, supports | blockers, swarms, cramped rooms | include weak points and line-of-sight puzzles |
| Trapper | chargers, spawn lanes, swarms | flying enemies, open bosses | include chokepoints and boss partial trap value |
| Beast Archer | sustained fights, hybrid support | pet-hostile hazards, burst | include companion-safe positions and pack mark value |

### Encounter Design Rules

- Every map should have at least one route where melee can function without unfair punishment.
- Every map should have at least one threat that asks ranged classes to move.
- Every dungeon should include mixed enemy roles so no class uses one skill forever.
- Flyers should reward Archer, Sniper, Storm Mage, and Mage tools without making Fighter/Guardian useless.
- Armored enemies should reward Guardian, Fighter, Sniper, Rune Mage, and armor-break gear.
- Support enemies should force priority targeting and mobility decisions.
- Chargers should reward blocks, traps, jumps, slows, and lane awareness.
- Bosses should test class mastery through timing, resource spending, defensive decisions, and terrain use.

### Preventing Trivialization

- Ranged classes need line-of-sight, blockers, projectile travel, target priority, flank pressure, or hazard pressure.
- Mobility classes need cooldowns, MP costs, landing risk, ground/air constraints, and no free bypass of boss gates.
- Trap classes need arming time, visibility, target caps, and enemies that sometimes route around or fly over traps.
- Defensive classes need attacks that are blockable and attacks that require movement.
- AoE classes need target caps, falloff, resistant enemies, or spread formations.

### Avoiding Unfair Melee Punishment

- Do not stack unreachable flyers, ranged turrets, floor hazards, and knockback in the same early encounter.
- Give melee classes platform routes, jump windows, armor-break rewards, and safe recovery after correct defense.
- Bosses should enter meleeable states or expose adds/resources close to the player.
- Use telegraphs with enough startup for ground classes to react.

## 13. UI And Player Communication

### Class Selection UI

Class cards should show:

- Role.
- Range.
- Mobility.
- Defense.
- Resource.
- Difficulty.
- Best situations.
- Weak situations.
- Example skills.
- Future advanced branches.

Do not describe classes only as stat bundles. Use short mechanical promises:

- Fighter: "Control the front line with heavy hits, guard timing, and Momentum."
- Mage: "Create safe spell windows, mark enemies, and burst grouped targets."
- Archer: "Stay mobile, mark priority targets, and punish from range."

### Advanced Class Selection UI

At level 25, show:

- Base class inherited skills remain available.
- Branch resource and HUD preview.
- Primary training skill.
- Defensive/mobility tool.
- Party skill.
- Best enemy matchups.
- Weak enemy matchups.
- Required playstyle.
- Permanent choice warning.

### Skill Tree UI

Every skill card should show:

- Name.
- Rank and max rank.
- Active/passive/party tag.
- Purpose tags.
- MP cost.
- Cooldown.
- Range or area if relevant.
- Current effect.
- Next-rank effect.
- Breakpoint previews.
- Prerequisites.
- Trainable/usable/locked state.
- Why it matters in plain language.

### Skill Tooltip Copy Format

Use this format:

```md
[Skill Name] - Rank [current]/[max]
[Type] | [Purpose Tags]

[One-sentence fantasy and use case.]

Cost: [MP/class resource/charges]
Cooldown: [seconds]
Range: [short/medium/long or exact if useful]

Current: [specific behavior at current rank]
Next Rank: [specific change]
Breakpoint: [Rank 5/10 behavior if relevant]

Use when: [decision prompt]
Weak when: [counter-situation]
Prerequisite: [missing requirement or "Ready"]
```

Example:

```md
Rune Detonation - Rank 3/10
Active | Setup | Area | Burst

Detonates marked or linked enemies after you prepare a rune cluster.

Cost: 35 MP
Cooldown: 9s
Range: Medium field radius

Current: Hits up to 6 prepared targets with 3 damage lines.
Next Rank: Slightly increases detonation damage.
Breakpoint: Rank 5 extends rune duration before detonation.

Use when: several enemies are marked, linked, or standing in your glyph.
Weak when: enemies are scattered or leaving your field.
Prerequisite: Ready
```

### HUD Requirements

- HP and MP must remain readable during combat.
- Class resource must use branch-specific color and shape.
- Cooldowns must show remaining time and ready flash.
- Party buffs must show category and duration.
- Debuffs must be visible on enemies without covering hitboxes.
- Traps, glyphs, marks, weak points, burns, cracks, and links need distinct indicators.
- Boss break gauge contribution should be readable when relevant.

### Tutorial Prompts

Teach one idea at a time:

- "Use your movement skill to avoid the charger."
- "Guard just before impact to store Guardian impact."
- "Burn enemies before using Wildfire."
- "Place traps where enemies will cross, not where they stand now."
- "Keep attacking the same target to build Duelist Tempo."
- "Use Weak Point Mark before Execution Shot."

## 14. Class And Skill Documentation Templates

### Class Design Sheet

```md
# [Class Name]

- Base or advanced:
- Parent class:
- Unlock level:
- Trial or unlock source:
- Core fantasy:
- Combat role:
- Resource:
- Range:
- Mobility:
- Defense:
- Utility:
- Primary training skill:
- Party skill:
- Passive identity:
- Strengths:
- Weaknesses:
- Best enemy matchups:
- Worst enemy matchups:
- Best map types:
- Worst map types:
- Required animations:
- Required VFX:
- Required sounds:
- Required UI:
- Tutorial encounter:
- Balance notes:
- Implementation files:
```

### Skill Design Sheet

```md
# [Skill Name]

- ID:
- Owner:
- Batch:
- Type:
- Category:
- Purpose:
- Role tags:
- Input:
- Rank max:
- Default rank:
- Prerequisites:
- MP cost:
- Class resource cost/gain:
- Cooldown:
- Startup:
- Active:
- Recovery:
- Range:
- Hitbox:
- Target cap:
- Damage lines:
- Damage scale:
- Status effects:
- Movement behavior:
- Cancel rules:
- Upgrade breakpoints:
- Skill modifiers:
- Gear interactions:
- Enemy reactions:
- Boss rules:
- UI summary:
- Animation assets:
- VFX assets:
- Sound assets:
- Implementation notes:
- Balance risks:
```

### Passive Trait Sheet

```md
# [Passive Name]

- Owner:
- Unlock:
- Fantasy:
- Behavior change:
- Stat changes:
- Resource interaction:
- Skill interactions:
- Gear interactions:
- UI indicator:
- Balance risk:
- Test cases:
```

### Skill Tree Node

```md
# [Node Name]

- Skill id:
- Rank requirement:
- Point cost:
- Prerequisites:
- Unlocks:
- Current-rank text:
- Next-rank text:
- Rank 5 breakpoint:
- Rank 10 breakpoint:
- UI state labels:
```

### Upgrade Modifier

```md
# [Modifier Name]

- Applies to:
- Unlock condition:
- Effect:
- Behavior change:
- Numeric change:
- Tradeoff:
- UI wording:
- Balance risk:
```

### Status Effect

```md
# [Status Name]

- Source classes:
- Duration:
- Stack rule:
- Tick rate:
- Boss scaling:
- Visual indicator:
- Sound cue:
- Cleanse/expire cue:
- Enemy reaction:
- Balance cap:
```

### Resource Type

```md
# [Resource Name]

- Class:
- Max value:
- Gain sources:
- Spend sources:
- Decay rules:
- Overcap rules:
- HUD shape:
- Ready cue:
- Tutorial explanation:
- Balance risk:
```

### Encounter Test Case

```md
# [Encounter Name]

- Map:
- Enemy mix:
- Terrain features:
- Tested classes:
- Intended class strengths:
- Intended class weaknesses:
- Skill checks:
- Resource checks:
- Expected clear time:
- Expected potion use:
- Failure signs:
- Tuning notes:
```

### Balance Tuning Entry

```md
# [Skill/Class/Encounter]

- Date:
- Build:
- Test level:
- Gear profile:
- Skill ranks:
- Enemy:
- Map:
- TTK:
- KPM:
- Damage taken:
- Potion use:
- Resource uptime:
- Cooldown waste:
- Player notes:
- Change made:
- Retest result:
```

### Animation/VFX Requirement Entry

```md
# [Asset Name]

- Skill/class:
- Asset type:
- Sheet format:
- Row order:
- Frame count:
- Frame size:
- Chroma color:
- Guide color:
- Visual language:
- Gameplay read:
- Must avoid:
- Processor:
- Runtime path:
```

## 15. Playtesting Checklist

Use this checklist for every class and major skill pass.

### Class Identity

- [ ] Does the class feel distinct before damage numbers appear?
- [ ] Does the class have a clear range profile?
- [ ] Does the class have a clear movement rhythm?
- [ ] Does the class have a clear defensive answer?
- [ ] Does the class resource change decisions?
- [ ] Does the class have a meaningful weakness?
- [ ] Does the class match the GDD fantasy?
- [ ] Are inherited base skills visible and useful?

### Skill Purpose

- [ ] Does each skill have a clear purpose?
- [ ] Does each active skill create a decision?
- [ ] Is any skill always correct to use on cooldown?
- [ ] Is any skill never worth using?
- [ ] Does the primary training skill support the class without replacing the whole kit?
- [ ] Does the defensive or mobility skill feel responsive?
- [ ] Does the skill have appropriate risk and reward?

### Readability And Feel

- [ ] Is the skill readable before use?
- [ ] Is the active hit or effect readable during use?
- [ ] Is the recovery or cooldown readable after use?
- [ ] Does the animation match the hit timing?
- [ ] Does the VFX show the actual area or lane?
- [ ] Does the sound cue communicate impact?
- [ ] Are cooldown and resource changes visible?
- [ ] Does hit pause feel satisfying without slowing repeated hits too much?
- [ ] Does screen shake appear only for meaningful impact?

### Terrain And Encounter Fit

- [ ] Does the skill work correctly on slopes?
- [ ] Does it work correctly on platforms?
- [ ] Does it work correctly near ladders and climbables?
- [ ] Does it work correctly against flying enemies?
- [ ] Does it work correctly against armored enemies?
- [ ] Does it work correctly against bosses?
- [ ] Does it trivialize any map route?
- [ ] Does it unfairly fail in common terrain?

### Progression And Upgrades

- [ ] Can new players understand the class by level 5?
- [ ] Is the first build choice meaningful by levels 10-19?
- [ ] Does the advanced trial preview the branch honestly?
- [ ] Does rank 5 change behavior?
- [ ] Does rank 10 change behavior?
- [ ] Are upgrades more than damage bumps?
- [ ] Are respec/loadout rules understandable?

### Balance

- [ ] Does the class have acceptable mobbing?
- [ ] Does the class have acceptable bossing?
- [ ] Does the class have a party contribution?
- [ ] Does it avoid being best at everything?
- [ ] Does it avoid being useless without rare gear?
- [ ] Does the skill stay fair with strong attunement lines?
- [ ] Does the skill stay fair with boss sets?
- [ ] Does the class perform within field KPM and boss TTK targets?

### UI And Communication

- [ ] Does the tooltip explain when to use the skill?
- [ ] Does the tooltip explain when the skill is weak?
- [ ] Are prerequisites clear?
- [ ] Are resource costs clear?
- [ ] Are status effects named consistently?
- [ ] Are icons readable at gameplay size?
- [ ] Is the class resource widget understandable?
- [ ] Are party skill stacking rules clear?

## 16. Final Design Rules

- Preserve the current 3 base class and 9 advanced branch structure.
- Make the existing classes deeper before adding new ones.
- Keep base skills inherited and visible after advanced branching.
- Build skills through data first and engine branches second.
- Treat movement as part of combat balance, not a convenience feature.
- Give every skill a decision, not only a damage number.
- Give every class a weakness the player can understand and play around.
- Use rank 5 and rank 10 for behavior changes.
- Keep party skills useful at rank 1 but controlled by stacking rules.
- Keep class identity visible in animation, VFX, sound, UI, resource loops, and encounter matchups.
- Balance around maps, enemy roles, boss mechanics, gear, and attunement together.
- Never let one class invalidate terrain, enemy roles, or party composition.
- Never let a class fantasy depend on hidden math that the player cannot see.
