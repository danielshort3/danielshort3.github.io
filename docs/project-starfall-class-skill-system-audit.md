# Project Starfall Class And Skill System Audit

Last updated: 2026-06-19

This audit tracks how the current repository satisfies `CLASS_AND_SKILL_DESIGN_GUIDE.md`.

## Current State

- Guide source of truth: `CLASS_AND_SKILL_DESIGN_GUIDE.md`
- Machine-readable design contract: `js/games/project-starfall/data/class-skill-design.js`
- Runtime class data: `js/games/project-starfall/data/classes.js`
- Runtime skill data: `js/games/project-starfall/data/skills.js`
- Runtime progression hooks: `js/games/project-starfall/engine/skills.js`
- Runtime combat hooks: `js/games/project-starfall/project-starfall-engine.js`
- UI skill hooks: `js/games/project-starfall/ui/skill-*.js`
- Resource widget hooks: `js/games/project-starfall/ui/resource-widgets.js`
- Validation command: `npm run validate:project-starfall-class-skills`

## Completed

- The current roster is fixed at 3 base classes and 9 advanced classes.
- Every playable class has runtime data, role profile data, and a guide-backed resource definition.
- Every advanced class has a base class, level requirement, party skill id, and primary training skill.
- Existing skill data remains data-driven through `data/skills.js`.
- Every skill has owner, batch, type, category, role tags, prerequisites, rank data, cost, cooldown, icon metadata, purpose, and description.
- Every active skill has generated combat FX metadata.
- Every passive skill has passive stat metadata.
- Advanced classes inherit base skills through `engine/skills.js`.
- The guide-backed design contract defines status effects, tooltip structure, balance tuning fields, loadout slots, encounter test cases, and debug scenarios.
- `test.js` runs the guide-backed validation during Project Starfall fast contracts.

## Validation Coverage

`build/validate-project-starfall-class-skills.js` currently verifies:

- Required guide sections exist.
- Guide contract file paths exist.
- Base and advanced class ids match the intended roster.
- Class runtime fields are present.
- Class resources match class resource names and include gain, spend, feedback, player decision, and UI widget guidance.
- Advanced class base links, level requirements, and party skill ids are valid.
- Status effects are defined and reference valid classes.
- Skill ids are unique.
- Skill owners, batches, categories, purposes, role tags, costs, cooldowns, damage lines, icons, descriptions, and prerequisites are valid.
- Active skills have behavior hooks and combat FX references.
- Passive skills have passive stats.
- Each advanced class has exactly one primary training skill.
- Each advanced party skill documents current and future party behavior.
- Advanced branches inherit base skills and prefer their advanced trainer as the primary skill candidate.
- Tooltip concepts, balance fields, loadout slots, debug scenarios, and encounter test cases are present.

## Remaining Implementation Placeholders

These are intentionally not complete runtime features yet. They are now documented and validated as design targets.

- Guardian: add clearer blocked-hit timing, Stored Impact segment animation, and boss guard-window feedback.
- Berserker: add stronger danger-band UI, clearer low-HP threshold feedback, and safer distinction between training sustain and risky boss burst.
- Duelist: add class-owned parry, riposte, same-target finisher, and Tempo drop/recovery feedback.
- Fire Mage: add explicit Overheat rules only if they create decisions, plus Heat cap warnings and burn-count UI.
- Rune Mage: add active rune counters, linked-target visibility, and clearer Rune Circle mode selection.
- Storm Mage: add a non-chain boss spender so isolated bosses do not collapse the branch identity.
- Sniper: add aim-state feedback, steady aim cue, and boss weak-point telegraphs.
- Trapper: improve trap visibility, arming state, trigger radius, lure routing, and boss partial trap value.
- Beast Archer: add a true companion actor or stronger visible companion proxy before adding more passive companion damage.
- Party skills: replace solo/self-buff prototype behavior with real party telemetry and stacking enforcement when multiplayer/AI party systems mature.
- Timing data: add optional `startupSeconds`, `activeSeconds`, `recoverySeconds`, `cancelRules`, and `hitbox` fields to runtime skills when the combat system is ready for frame-level tuning.

## Required Checks For Future Changes

Run these after changing class, skill, resource, status, animation, VFX, tooltip, progression, or UI data:

```bash
npm run validate:project-starfall-class-skills
npm run test:starfall:smoke
```

Run the broader default suite before merging:

```bash
npm test
```
