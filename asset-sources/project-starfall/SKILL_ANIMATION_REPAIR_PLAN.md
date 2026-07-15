# Project Starfall Skill Animation Repair Plan

## Scope

Repair every active combat-skill effect while preserving the four-row runtime contract:
`cast`, `projectile`, `impact`, and `area`, with six 160 x 160 frames per row.
Passive skills do not receive effect sheets because they never activate directly.

## Baseline findings

- 85 skills exist; 70 are active and map one-to-one to 70 runtime sheets.
- The baseline runtime set contains 1,680 frames and occupies 49,666,293 bytes on disk.
- Gross geometry was valid, but the artwork was semantically generic: melee strikes,
  guards, rolls, traps, arrows, fields, and spells reused the same cast starburst,
  flying crystal, crystal impact, and humanoid barrier.
- Three unrelated skill pairs were byte-identical, 90 cross-skill frame-hash groups
  existed, and several mage projectile rows were exact copies.
- Only two old ImageGen sources were preserved, and neither reproduced its deployed
  runtime sheet. The old broad visual sweep was the real owner and could overwrite
  dedicated source art with shared palette-shifted rows.
- Eleven mobility skills did not pass their skill ID into the effect request, so their
  dedicated sheets were never selected.
- The current-map asset set omitted skill sheets. Pixi could lazily request one after a
  missed frame, while the Canvas renderer had no matching on-demand load path.
- Every state looped with a delay even when it was a one-shot cast or impact. Common
  effect lifetimes ended before all six frames could be shown.
- Legacy skill backup sheets were registered as fallback URLs even though the default
  public build excludes them, so a failed primary request could only fall through to a 404.

## Repair strategy

1. Make the semantic combat-FX generator the canonical owner of all active skill sheets.
   Derive an action profile and motif from each skill's type, icon, targeting, movement,
   visual ID, description, and class palette. Keep deterministic per-skill signatures so
   unrelated skills cannot collapse to identical pixels.
2. Keep ImageGen as an intentional override path. Process preserved source art frame by
   frame, remove connected chroma-key background, clear hidden key RGB, fit each frame
   inside an eight-pixel gutter, and apply stable row-aware registration.
3. Validate every output for dimensions, all 24 nonempty and distinct frames, safe
   margins, clean transparency, absence of key-color residue, and unique rendered sheet
   pixels. Re-run the generator twice to prove deterministic output.
4. Pass the activating skill ID through every runtime path, align one-shot timing with
   player/equipment contact frames, selectively preload only the current base/advanced
   class and party sheets, and make Canvas and Pixi choose and fade the same animation state.
5. Remove the obsolete skill-backup fallback mapping. Canonical deterministic generation
   is the recovery mechanism, while deployed primaries remain the only runtime texture set.

## Pilot and rollout gates

- Pilot: Fighter `Ground Slam`, because it is immediately available and exercises cast,
  impact, grounded-area registration, and the equipped melee pose.
- Projectile gate: Fire Mage `Fireball`, because it exercises all four rows in live combat.
- Full rollout: regenerate all 70 active sheets, build a representative contact sheet,
  run the quality audit and complete Project Starfall tests, then play the pilot and a
  projectile/area path in both Canvas/Pixi-capable browser rendering.

## Acceptance criteria

- 70 of 70 active skills have deterministic, unique, semantically appropriate sheets.
- No active or hidden chroma-key pixels remain; no frame crosses the eight-pixel gutter.
- All active runtime paths resolve the correct skill sheet, including mobility and party buffs.
- One-shot cast and impact sequences can display all six frames consistently in Canvas and Pixi.
- Project Starfall focused tests, full tests, build, visual-sweep validation, and live browser
  play complete without new errors or missing skill-animation requests.
