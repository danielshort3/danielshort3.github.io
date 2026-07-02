# Project Starfall Item Visual Audit

This audit compares the current repository state against `ITEM_VISUAL_DESIGN_GUIDE.md` and `img/project-starfall/items/item-visual-manifest.json`.

## Complete and Active

- Runtime inventory item icons exist under `img/project-starfall/items/icons/`.
- Runtime card icons exist under `img/project-starfall/cards/icons/`.
- Item icons and card icons use the current 64x64 PNG contract.
- AI item source sheets exist under `img/project-starfall/items/source/`.
- Processed item sheets exist under `img/project-starfall/items/sheets/`.
- Item icon references are centralized through `ITEM_ASSETS`.
- Card icon references are centralized through `CARD_ASSETS`.
- Current rarity colors are centralized through `ITEM_RARITY_VISUALS` and mirrored in the manifest.
- The item visual manifest now defines icon, pickup, rarity, category, theme, prompt, metadata, generated asset, and validation contracts.
- Generated direct equipment icon variants exist for the class/regional equipment records that previously shared inventory icons.
- Generated pickup sprites exist for current item and card icons.
- Generated rarity frames exist for implemented item rarities.
- Generated category markers exist for all manifest item categories.
- Generated item VFX sheets and animated pickup sheets exist for current generic pickup/drop treatments.
- `npm run validate:project-starfall-item-visuals` validates the manifest against current data and assets.

## Generated Derivative Structure

- Dedicated pickup sprites: `img/project-starfall/items/pickups/`.
- Pickup source sheets: `img/project-starfall/items/pickups/source/`.
- Animated pickup sheets: `img/project-starfall/items/pickups/animated/`.
- Rarity frame overlays: `img/project-starfall/items/rarity-frames/`.
- Generic pickup/drop VFX: `img/project-starfall/animations/item-vfx/`.
- Item UI category marker overlays: `img/project-starfall/ui/item-overlays/`.

Generated derivative assets are produced by `npm run build:project-starfall-item-visual-assets`. Runtime still reuses inventory icons until item data and renderers support dedicated pickup/VFX/overlay references.

## Current Runtime Constraints

- World loot currently reuses inventory icons.
- Equipment and card drops receive runtime rarity aura rings.
- Stackable items use larger world icon presentation than tier-aura items.
- Dedicated `pickup_sprite_path`, `animated_pickup_path`, and `vfx_path` fields are documented in the manifest and generated assets exist, but these fields are not yet runtime-critical.
- Mythic item rarity is documented as future and is not implemented in runtime item rarity data.

## Resolved Visual Identity Gaps

The following class/regional equipment records previously resolved to shared inventory icons. They now have generated direct icon variants under `img/project-starfall/items/icons/`:

- `duelist_parry_medal` previously used `deadeye_scope`.
- `storm_charge_focus` previously used `rune_etched_focus`.
- `beast_bond_charm` previously used `wanderer_charm`.
- Rustcoil regional gear previously shared early/world-drop gear icons.
- Cinder regional gear previously shared iron, apprentice, oak, furnace, scorch, and ring icons.
- Frostfen regional gear previously shared forest boss-set and plain ring icons.
- Stormbreak regional gear previously shared Stormbreak boss-set and charm icons.
- Astral regional gear previously shared Astral/Eclipse boss-set and charm icons.

Run this command to verify the current warning list is clean:

```bash
npm run validate:project-starfall-item-visuals -- --warnings
```

## Still Awaiting Final Generated Art

- Runtime integration for dedicated world pickup sprites.
- Runtime integration for animated pickup sheets.
- Runtime integration for item-specific pickup burst VFX.
- Hand-authored rarity frame overlays, if the UI later needs them instead of generated frames.
- Quest/key/lore collectible item icons after those runtime item types are formalized.
- Hand-authored replacements for generated direct icon variants if art direction wants fully bespoke AI/manual source art later.

## Current Acceptance Gate

An item visual pass is considered structurally ready when:

- `npm run validate:project-starfall-item-visuals` passes.
- `npm run test:starfall:assets` passes.
- New item art follows `ITEM_VISUAL_DESIGN_GUIDE.md`.
- New source art follows the cyan guide-line and chroma-key sheet rules.
- New data references keep existing item loading and save/load behavior intact.
