# Starfall art review

Open `index.html` through a repository static server. Rebuild with:

```powershell
node build/generate-project-starfall-overhaul-review.js
```

The generator reads the current runtime registry, including animation rows, frame holds, sequences, registration and configured timing. Rerun it after activating additional replacement enemy sheets. `review-data.json` is the generated review contract; no gameplay files are changed.

The original player sheet was extracted from Git and matched against `asset-sources/project-starfall/overhaul-v1/baseline.json`. Its original animation definitions and frame registrations are captured in `before/animation-contracts.json` so later commits do not change the comparison. Original enemy sheets come from the immutable enemy source archive and are also verified against baseline on every build.

The viewer shows sprite art at the game render height (56 pixels for the player; enemy profile height from the renderer), with a 2× option. Actions repeat for inspection using their configured timing. Updated enemy telegraphs use clearly labeled representative windups of 420/540/750/1000 ms by behavior, fitting the early poses before the renderer's final 200 ms commitment hold (300 ms for bosses). Original telegraphs retain native timing. Other combat state durations, attack scheduling, equipment and visual effects are not simulated here. Shared frame stepping and the seek slider follow the updated drawing sequence, with the original evaluated at the same elapsed time. Center-registered floating actors use the same half-height root placement as the renderer.

Light and dark studio backgrounds and all forty authored scenery paintings are available. Full contact sheets cover scenery, items, skills, cards, equipment and interface art. Selected character, action, background, speed, scale, playback and position are restored in the same browser. The read-only `window.starfallOverhaulReview.snapshot()` and `seek(ms)` inspection hooks support repeatable visual checks.
