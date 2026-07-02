# Project Starfall Asset Sources

This folder contains the source-facing side of the Project Starfall asset pipeline. Treat `ASSET_GENERATION_GUIDE.md` as the human-readable contract and `asset-generation-manifest.json` as the machine-readable contract.

## Authoritative Files

- `ASSET_GENERATION_GUIDE.md` - full project-specific art, prompt, naming, folder, and export guide.
- `asset-sources/project-starfall/asset-generation-manifest.json` - validator-readable asset contract.
- `asset-sources/project-starfall/prompts/README.md` - reusable generation prompt templates.
- `img/project-starfall/asset-prompts.md` - historical prompt and processor provenance.

## Validation

Run this after changing source folders, generated assets, prompts, or Starfall asset data:

```bash
npm run validate:project-starfall-assets
npm run test:starfall:assets
```

The validator checks required source folders, runtime folders, manifest contracts, generated asset dimensions, source sheet presence, and runtime data references.

## Source Folder Ownership

- `players/classes/` holds source-backed playable class sheets.
- `players/equipment/` is reserved for generated equipment-layer source sheets.
- `enemies/compact/` holds compact enemy source sheets.
- `enemy-projectiles/` is reserved for enemy projectile source sheets.
- `prompts/` holds prompt templates that mirror the guide.

Do not place generated runtime PNGs here. Runtime outputs belong under `img/project-starfall/`.

