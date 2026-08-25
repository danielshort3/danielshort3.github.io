# Project Starfall Design Guides

Canonical design contracts for Project Starfall. Build validators in
`build/validate-project-starfall-*.js` and runtime contracts in
`js/games/project-starfall/data/class-skill-design.js` load these files at build/test time. The
runtime **does not** fetch them in the browser — they are referenced only by build/test tooling,
contracts, and documentation.

## Files

| Guide | Consumed by |
| ----- | ----------- |
| `ASSET_GENERATION_GUIDE.md` | `build/validate-project-starfall-asset-generation.js`, `asset-sources/project-starfall/asset-generation-manifest.json` |
| `CLASS_AND_SKILL_DESIGN_GUIDE.md` | `build/validate-project-starfall-class-skills.js`, `js/games/project-starfall/data/class-skill-design.js`, `test.js` |
| `ITEM_VISUAL_DESIGN_GUIDE.md` | `build/validate-project-starfall-item-visuals.js`, `img/project-starfall/items/item-visual-manifest.json`, item-visuals audit READMEs |
| `MAP_AND_LEVEL_DESIGN_GUIDE.md` | `build/validate-project-starfall-maps.js`, `test.js` |
| `MAP_EDITOR_INTEGRATION_GUIDE.md` | cross-referenced from `MAP_AND_LEVEL_DESIGN_GUIDE.md` |
| `project_starfall_gdd_v0_5.md` | cited by the other 5 guides as primary source of truth; audit docs, test.js |

## Conventions

- Consumers at the repo root use absolute-path form `docs/project-starfall/<FILE>`.
- Cross-refs between these files use the file name directly (e.g. `ASSET_GENERATION_GUIDE.md`)
  because they're all in the same directory.
- If you add a new contract file here, add its validator in `build/` and any runtime references in
  `js/games/project-starfall/data/` simultaneously. The 21,842-check test suite enforces this.
