# Project Starfall Item Visual Workflow

`ITEM_VISUAL_DESIGN_GUIDE.md` is the source of truth for item visual direction. This folder exists so the asset workflow is discoverable from the docs tree and so validation can prove that item visual structure is not just implicit.

## Active Contract Files

- Guide: `ITEM_VISUAL_DESIGN_GUIDE.md`
- Manifest: `img/project-starfall/items/item-visual-manifest.json`
- Prompt templates: `asset-sources/project-starfall/prompts/item-visual-prompts.md`
- Current audit: `docs/project-starfall-item-visuals/audit.md`
- Identity template: `docs/project-starfall-item-visuals/item-visual-identity-template.md`
- Runtime item icons: `img/project-starfall/items/icons/`
- AI source sheets: `img/project-starfall/items/source/`
- Processed runtime sheets: `img/project-starfall/items/sheets/`
- Generated pickup sprites: `img/project-starfall/items/pickups/`
- Generated item VFX: `img/project-starfall/animations/item-vfx/`
- Generated UI overlays: `img/project-starfall/ui/item-overlays/`

## Validation

Run:

```bash
npm run validate:project-starfall-item-visuals
```

The validator checks the manifest, required folders, current icon dimensions, card icon dimensions, rarity colors, category/theme coverage, generated direct icon variants, generated pickup sprites, rarity frames, category markers, item VFX sheets, animated pickup sheets, and guide references.

`npm test` also runs the item visual validator as part of the Project Starfall asset suite.

To confirm no equipment records resolve to shared inventory icon art, run:

```bash
npm run validate:project-starfall-item-visuals -- --warnings
```

## Current Runtime Policy

World pickups currently reuse inventory icons in runtime rendering. Dedicated pickup sprites, animated pickup sheets, rarity frame art, item VFX, and category markers are generated derivative assets and are available for renderer/data integration when the game adds explicit fields for them.
