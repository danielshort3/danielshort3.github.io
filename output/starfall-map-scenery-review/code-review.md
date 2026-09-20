# Bounded map-pass code review

Reviewed on 2026-09-19 against the current local source and `project-starfall.6d600c6a.js` build. This was a read-only review: no runtime, data, asset, or test changes were made.

## Outcome

No new blocking correctness defect was found in the reviewed paths for the currently authored maps. This is a bounded review, not a claim that all map, balance, and visual acceptance criteria pass.

## Traversal, cache policy, and return behavior

- `js/games/project-starfall/project-starfall-engine.js:22793` derives ladder, ramp, stationary, territory, home, and leash policy.
- `project-starfall-engine.js:22835` applies route permissions and physical jump eligibility to candidate links. `engine/map-runtime.js:346` invokes the route filter at every expanded graph edge, not only the first hop.
- `project-starfall-engine.js:30235` caches enemy routes by from/to platform, full policy key, return state, body width, and movement speed. Replacing the graph clears the cache. A more permissive actor's result is therefore not reused for a restricted actor.
- `project-starfall-engine.js:30594` prevents a denied authored route from falling through to the old proximity-jump behavior.
- `project-starfall-engine.js:22865` begins leash return by clearing aggro and pending attack/charge warnings together. The return gate at `:22882` requires home-lane grounding (or flyer home-height proximity) and horizontal proximity before clearing the return state, then applies a one-second reacquisition delay.
- `project-starfall-engine.js:23010` also rejects direct contact/attack-alert aggro while returning or during that delay. Return movement at `:30419` uses terrain routing rather than teleporting into the patrol bounds.

Existing coverage inspected: `tests/project-starfall/project-starfall-map-encounters.test.js:334` onward checks ladder/ramp policy cache separation, denied fallback, strict territory, actual climbing, and 30/60/120 FPS walk-home and contact-reacquisition behavior.

## Companion knockout and recovery

- `project-starfall-engine.js:22399` preserves explicit zero HP during normalization and only initializes absent/nonfinite values.
- `engine/party.js:244` preserves zero HP when creating/restoring member state. The standalone engine mirrors this behavior in its embedded fallback.
- `project-starfall-engine.js:23416` processes defeated members before below-world recovery. It holds the member down until the deadline and clears the consumed deadline on revival at `:23432`.
- `project-starfall-engine.js:30691` ignores further damage to an already defeated companion, so repeated hits cannot keep extending the recovery deadline.
- `project-starfall-engine.js:27798` excludes downed companions from the living party count used by spawn population scaling.

Existing coverage inspected: `tests/project-starfall/project-starfall-party-recovery.test.js` exercises both the module and embedded fallback, missing/zero/nonfinite HP, correct class-level initial health, real lethal damage, downed bodies below the world, recovery deadlines, one-time revival, and living population counts at 30/60/120 FPS.

## Shared scenery inclusion and current build parity

The shared helper is imported at `build/entries/project-starfall.entry.js:10`, after core geometry/math and before the engine/renderers. Canvas and Pixi both resolve and call that helper. `build/copy-to-public.js:558` includes the complete `js` directory in recursive publication. Both authored and public game HTML point to the same current hashed bundle.

Read-only inspection of the generated bundle confirmed the shared scenery module and its surface/footing helpers are present. The current bundle is 4,828,104 bytes. Exact SHA-256 values:

| File | SHA-256 |
|---|---|
| `js/games/project-starfall/engine/scenery-placement.js` | `f36b9ce4f1e1ffdc341778743d7325d3820884ae82062a243abc876f825ce3c7` |
| `public/js/games/project-starfall/engine/scenery-placement.js` | `f36b9ce4f1e1ffdc341778743d7325d3820884ae82062a243abc876f825ce3c7` |
| `dist/project-starfall.6d600c6a.js` | `6d600c6ad0bb252fea1a8a73458bbf0ded23c6e8018a2a63b66e2387d82fcec1` |
| `public/dist/project-starfall.6d600c6a.js` | `6d600c6ad0bb252fea1a8a73458bbf0ded23c6e8018a2a63b66e2387d82fcec1` |

The raw helper and bundled output each match their public copy byte-for-byte. This establishes current local publication inclusion; it is not deployment evidence.

## Limits

No additional test suite or training matrix was started for this review; existing focused tests were read, and source/build inclusion was verified directly. The review did not perform a new live pursuit animation check or exhaustively simulate every route and actor combination. New future policy combinations, dynamic mutation of graph edges in place, and balance outcomes are outside this certification. Training reports and the separate full-size/deep visual reviews retain their own scope and acceptance findings. Line references describe the source at review time and may shift after later edits.

