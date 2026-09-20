# Enemy artwork overhaul sources

This directory records 44 new canonical enemy artwork sources and four historical Bandit comparison aliases, covering all 48 existing enemy portrait paths and 51 runtime enemy IDs. The aliases keep their existing paths and use byte-identical copies of `bandit-cutter` artwork.

## Rebuild and provenance

From the repository root:

```powershell
node build/process-project-starfall-overhaul-enemies.js --enemy <file-id>
node build/process-project-starfall-overhaul-enemies.js --enemy <file-id> --import
```

The first command writes `review-sheet.png` and `import-report.json` in that enemy's folder. Import requires `source.json.visualReview.status` to be `approved-for-import`. Importing `bandit-cutter` also updates its four declared compatibility copies. Runtime definitions and deployment builds are separate integration steps.

Each canonical folder preserves the unmodified image-generation `source.png`, exact `generation-prompt.md`, source checksum and reference provenance in `source.json`. Supplemental sources preserve their own exact prompts and checksums. `originals/` preserves the previous portraits and compact sheets with the hashes recorded in `inventory.json`.

`inventory.json` maps every previous compact sheet to its new expanded sheet. `enemy-registration.json` is the plain registration map consumed by runtime integration. `validation-report.json` records the final source/packed-image audit, including all output checksums and registration values.

## Export contract

- Output atlas: 960 by 1280 pixels, six columns and eight rows of 160-pixel cells.
- Rows: idle, move, telegraph, attack, projectile, buff, hit and defeat.
- Portrait: 320 by 320 pixels at the existing portrait path.
- Grounded root: `(80, 150)`; floating center: `(80, 80)`. Each identity has its own measured `authoredBodyHeight`.
- One fixed scale applies across an identity. A supplemental source has one fixed resolution conversion factor for all of its poses. No frame is independently stretched or resized to conceal motion.
- Actor components are extracted from actual alpha geometry, with reviewed row boundaries when needed. Nominal generated grid dimensions are not reliable crop boundaries.
- Export isolates the selected actor and adjacent faint edge pixels, discards alpha below 8, and clears invisible RGB. It does not remove semantic colors with a chroma key.
- All actor art is separate from gameplay warnings, projectiles, impact effects and healing auras.

Attack frame 0 is the authored contact pose; projectile frame 0 is the authored release gesture. Early sources are explicitly reordered to meet this event contract. Preparation remains in the telegraph row. The healer buff release pose is frame 4; runtime timing owns the preparation and pulse event.

## Approved studies and explicit pose reuse

The original Oracle, Glowcap and Bristle Boar study PNGs remain byte-for-byte unchanged. Their reuse is documented under `protectedReuse` and `supplementalRows` with approved anatomical anchors:

- Oracle buff uses approved study poses 0, 1, 2, 3, 4 and 7.
- Glowcap idle uses approved spring poses 0, 1, 2, 4, 5 and 7.
- Bristle Boar telegraph and attack use the approved charge study. Repeated final drawings are deliberate commitment/recovery holds.

Rust Ratchet and Lava Tick receive separately authored missing hit rows. Brambleking, Rimeback Brute, Bandit Cutter, Astral Archivist and Eclipse Sovereign receive separated replacement action rows where raw poses touched. Frostling Scout receives a single-dagger telegraph row. Its buff guard reuses one valid pose from that row. Snowglare Wisp and Stormbound Archer reuse valid authored release gestures from their attack rows to exclude a baked effect or a missing bow. These are declared pose mappings, not manufactured in-between drawings.

The Eclipse Sovereign attack extraction explicitly groups its detached staff with the first actor. This prevents an accidental component-count match from concealing a split accessory and a merged pair elsewhere.

## Validation scope

Every packed atlas was visually inspected. Image checks cover all 2,304 frame slots, expected dimensions, alpha, no occupied cell-edge pixels, 96 original backup hashes, all source checksums, three reused study hashes and the 48 registration entries. There are 44 distinct rendered atlases and exactly four declared compatibility copies. Runtime playback, effect timing, game-size readability and deployment parity are validated by the integration checks.
