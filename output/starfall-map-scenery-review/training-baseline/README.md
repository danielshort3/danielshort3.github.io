# Reproducing the authored-map comparison

The three final matrices each contain 525 scenarios across all 13 public training fields at levels 3, 6, 14, 19, 28, 50, 60, 85 and 100, wherever each field is level-eligible. Every eligible solo class and the fixed fighter-led archer/mage companion party uses three fixed seeds, one minute of warmup and five measured minutes at 30 FPS.

The before fixture reconstructs the authored publication and map-builder sources from before this pass. Both profiles run on the same final corrected runtime, including movement and companion recovery fixes. It is not a historical game-build benchmark. The selected main circuit uses identical original platform IDs in both profiles; missing or unsuccessful routes cannot establish paired acceptance. The optional matrix uses the final current authoring and its separately selected optional circuit. Cinder normal non-elite kills use the approved trainingXpMultiplier of 0.93 in both current circuits; the preserved before publication leaves it at its default 1. All three final matrices were rerun uniformly after the scalar and final visual-only edits. The previous complete matrix remains under training-pre-cinder-adjustment with unchanged source hashes.

Run from the repository root after restoring the matching source revision recorded in the raw reports:

```powershell
node build/analyze-project-starfall-training.js --levels=3,6,14,19,28,50,60,85,100 --route=main --route-plans=output/starfall-map-scenery-review/training-baseline/approved-route-plans.json --workers=4 --output=output/starfall-map-scenery-review/training-current.json
node build/analyze-project-starfall-training.js --levels=3,6,14,19,28,50,60,85,100 --route=main --route-plans=output/starfall-map-scenery-review/training-baseline/approved-route-plans.json --baseline-spawns=output/starfall-map-scenery-review/training-baseline/map-publication.before.js --baseline-layouts=output/starfall-map-scenery-review/training-baseline/map-builders.before.js --workers=4 --output=output/starfall-map-scenery-review/training-before.json
node build/analyze-project-starfall-training.js --levels=3,6,14,19,28,50,60,85,100 --route=optional --route-plans=output/starfall-map-scenery-review/training-baseline/approved-route-plans.json --workers=4 --output=output/starfall-map-scenery-review/training-optional.json
node build/compare-project-starfall-training.js --current=output/starfall-map-scenery-review/training-current.json --before=output/starfall-map-scenery-review/training-before.json --optional=output/starfall-map-scenery-review/training-optional.json --output=output/starfall-map-scenery-review/training-comparison.json --csv=output/starfall-map-scenery-review/training-comparison.csv --optional-csv=output/starfall-map-scenery-review/training-optional-comparison.csv
```

Each scenario runs in a fresh Node process with seeded randomness and simulated wall and monotonic clocks. Reports record Node/platform/architecture, all source hashes, exact windows, seeds, loadouts and selected routes. Source changes during a pool abort it.

Each character begins with 99 health potions and 99 resource tonics of its legal level tier. Actual use, inventory depletion, cooldowns and replacement prices remain active. The starting stock has no purchase-budget restriction; its cost is separate from the equal equipment budget. Loot quantities and potentially equippable equipment are measured, while potential resale is an appraisal and does not credit currency.

Numerical balance targets are separate from coverage and controller confidence. Earlier phase 1 party runs are superseded by these uniform reruns after the companion HP/recovery correction.
