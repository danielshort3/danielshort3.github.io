# Project Starfall animation audit and repair plan

## Scope evaluated

The audit covered all 278 unique runtime animation images referenced by the active game manifests and renderer:

| Family | Active sheets | Evaluation |
| --- | ---: | --- |
| Player bodies | 3 families + 1 recovery sheet | Replaced. Fighter, Mage, and Archer now have distinct v5 silhouettes on one registered skeleton; advanced classes reuse their base family instead of decoding redundant copies. |
| Equipped-item atlases | 85 | Replaced. Each item now uses eight reusable angle cells instead of duplicating the item across all 60 body-animation frames. Bows add rest, draw, and release rows. |
| Combat effects | 127 | Retained. Frame progression, cell bounds, and gameplay timing are coherent. |
| Enemies and projectiles | 49 | Repaired in a follow-up pass. See `MONSTER_ANIMATION_REPAIR_PLAN.md` for the converter, renderer, cadence, and validation changes. |
| Portals | 3 | Retained. Loops are registered consistently. |
| Pet | 1 | Retained. Motion is coherent; one jump frame reaches the left cell boundary but does not show visible runtime clipping. |

Legacy and source-only sheets that are not referenced by an active manifest were excluded from runtime counts, but were retained as provenance.

## Root causes

1. Player art was authored as a standing cutout and mechanically transformed, so run, jump, climb, hit, and defeat frames lacked real limb articulation.
2. Class weapons were painted into the body sheet while the renderer also drew equipped-item layers, causing duplicate or contradictory weapons.
3. Canvas rendering stretched each square source frame into the player's narrow collision box.
4. Pixi trimmed every frame independently, discarded authored offsets, and refit the changing trim rectangle, so the character's scale and ground line shifted between frames.
5. Equipment layers did not share a frame-by-frame attachment rig with the body art.

## Repair plan

### 1. Stabilize the runtime registration

- Keep the complete 160 x 160 authored frame for player and party-member bodies; render equipment from separate 128 x 128 atlas cells.
- Use a fixed bottom-center pivot and one uniform scale derived from the authored body height.
- Apply identical registration in Canvas and Pixi renderers.

Status: complete.

### 2. Replace the player body animation

- Use one weaponless registration contract with distinct Fighter, Mage, and Archer family silhouettes, preserving the existing 6-column by 10-row contract.
- Author meaningful sequential motion for idle, run, jump, fall, climb, basic, skill, party, hit, and defeat.
- Map each advanced class to its base family sheet so changing specialization preserves anatomy, baseline, equipment sockets, and cache reuse.

Status: complete. The v5 class-family pass adds readable armor, mantle, and field-leather silhouettes while preserving the Fracture Runner registration, collision, and 60-frame attachment contract. The v4 sheet remains only as a recovery asset.

### 3. Make equipment reflect game state

- Keep weapons, shields, armor, boots, helmets, gloves, jewelry, and specialist gear out of the body sheets.
- Generate one modular atlas from each real item visual definition: eight registered angles for every item and rest, draw, and release rows for bows.
- Use the shared 60-frame attachment table to select the nearest atlas angle and place the item on the correct weapon, body, head, hand, foot, or accessory socket for the current action.
- Draw only the layers for currently equipped item IDs; unequipped inventory items remain invisible.
- Keep item colors, silhouettes, handedness, and slot ordering tied to their definitions.
- Do not rebuild 60-frame item sheets when player poses change; update the shared sockets or the item's compact atlas instead.

Status: complete.

### 4. Prevent regressions

- Validate the exact 85-atlas inventory, 128 px cells, eight angle columns, bow variant rows, alpha, non-empty cells, and transparent cell edges.
- Run the asset manifest validator, smoke tests, and browser QA for idle, run, jump, basic attack, and equipped-state compositing.
- Review the contact sheet whenever body poses or attachment anchors change.

Status: complete. The defeat row is fully authored and validated, although normal gameplay currently restores the player before that state is displayed for long.

## Acceptance checks

```powershell
node build/process-project-starfall-player-ai-assets.js --validate
npm run validate:project-starfall-equipment-atlases
npm run test:starfall:assets
npm run validate:project-starfall-assets
npm run test:starfall:smoke
node tests/project-starfall-class-identity.test.js
```

Source review artifacts: `asset-sources/project-starfall/players/class-families/*-v5-generated-chroma.png`. The processor normalizes their variable source dimensions into guided source sheets and exact 960 by 1600 runtime sheets.
