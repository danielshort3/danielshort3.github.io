# Area techniques and progression contract

The six original production tracks per area remain in `progression-content.js` with their IDs and established formulas. `area-skills-content.js` adds nine techniques per area. The combined fifteen entries occupy five reveal modules of three; `TRACK_LAYOUT` and each technique's `functionalRole` drive presentation. UI code must not infer gates from rank or elapsed time.

## Learning and costs

Every foundation is visible from the area's first visit. The second requires rank 4 in the first and local output; the third requires rank 3 in the second and more local output. Output thresholds are Trail 3/18 deliveries, Quarry 6/18 refined units, Tower 60/180 survey work, Workshop 6/18 assembled units, Ruins 3/9 recovered finds, and Harbor 1/3 voyage equivalents. Convoys count their actual voyage equivalents. Learned historical foundations remain learned. Quarry creation waits for Scouting's explicit claim and first purchased rank; Tower likewise waits for Refining. No timer or wallet balance substitutes for these achievements.

Operations techniques require all three foundation claims, actual local production, and—for techniques two and three—the preceding technique's retained rank 2. Later techniques additionally require their named cross-area project. The definitions are the canonical catalogue for names, local output gates, project prerequisites, effects, choices, and unique icon IDs.

New techniques cap at rank 10. A rank's effect is `start + (end - start) * log(rank) / log(10)`, with rank zero inactive. Coin cost is `ceil(areaBase * moduleFactor * 2.4^rank)`; module factors are 8, 32, 128 and 512. Non-Trail techniques also spend their area's resource equal to `ceil(coins * .025)`. Rebuilding ranks below retained high rank uses the existing renewed-run 50% discount. Added throughput bonuses in one bucket cap at +75%; actual-input refunds cap at 25%. Storage capacity is a separate rule: Stockpiles and Field Camps reach +150%.

`quote(state, id, quantity)` is the shared source for previews, tutorials and purchases. Quantities must be earned existing batch modes. Transactions require the whole exact quantity, sufficient resources, space below rank 10 and the current optional quote token. They never silently buy fewer ranks. A descriptor supplies `fittingQuantityAction` when the selected quantity exceeds remaining ranks.

## State and actions

Core schema 7 owns `areaSkills`, whose own version is 1. Its exact schema contains `revision`, `unlocked`, `ranks`, `highRanks`, `output`, `configs`, and `runtime`. The ledger is separate from old production tracks and does not change their validators. Ordinary reset preserves discoveries, local achievement counters, configurations and highest ranks; it clears current technique ranks and all temporary/prepaid production. Remembered operations remain dormant until rank 1 is rebuilt.

- `area-skill-unlock`: explicitly claims an eligible technique without buying a rank.
- `area-skill-buy`: performs an exact, atomic purchase using the canonical quote.
- `area-skill-config`: selects an earned option after rank 1; Standard Tools and Spare Parts share one mutually exclusive allocation.

`view(state)` returns `{items, ready, areas, operations, configurations}`. Item identities are `skill:<id>` while action IDs use the bare technique ID. Each item includes module/role metadata, prerequisites, state, costs, impact, canonical actions and configuration options. `foundations` and `foundationRequirement` provide the existing tracks' actual requirement counters; Tiers enriches their explicit claim status. Views are read-only.

## Production and offline ownership

`Progression.tick` advances actual output counters and technique runtime from one pre-step rate snapshot. `Progression.nextEvent` includes kiln payout, drill charge/burst, shift, recipe rotation, discovery rotation and prepaid hopper boundaries. Only unlearned foundation output thresholds add event boundaries; merely discovering a future technique does not perturb the released economy. `Progression.reset` owns technique reset, including reset previews.

Trail deliveries call `trailStarted` and `trailArrived`. Express speed and arrival cargo are frozen for the trip, and earned journal maps or funded research are paid only at actual arrival. Core calls `Progression.sync` after arrival before synchronizing tier notices, so a newly met foundation is ready in the same returned state.

Harbor launch freezes the provision bill, cargo, target and port. Paid receipts record Return Cargo refunds, Bonded Routes' continued domestic diversion, and Exchange Houses' domestic support; switching the next voyage configuration cannot rewrite those receipts. Receipts must match distinct funded manifests and are consumed exactly once on arrival. Only skills needing these effects create receipts. Fast fleets retain the existing convoy consolidation rule.

Material Hoppers purchase ore above the protected reserve; manufacturing consumes that prepaid quantity before the wallet. Offcut Recovery refunds only ore actually consumed. Kilns defer their actual earned ore/byproduct payout and settle on their boundary or when switched off. Field Notebooks cannot finance a commission and contribute only the lesser of stored survey work and the project's bounded work share. Archive Network diverts recovery yield only while a paid commission exists.

## Verification

`node --test tests/games/wayfarers-guild-area-skills.test.cjs` checks the complete catalogue, actual unlock and purchase actions, caps, exact quantities, saved configuration dormancy, physical production effects, paid manifest conservation, protected reserves and offline/foreground/reload agreement for each new clock family. Existing Harbor, Trail delivery, schema migration and released-v5 economy tests protect previous behavior when techniques are inactive.
