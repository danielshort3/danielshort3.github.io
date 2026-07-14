# Project Starfall monster animation repair plan

## Scope evaluated

The active monster catalog contains 48 portraits and 48 compact runtime sheets. Every runtime sheet uses a `3 x 8` grid of `128 x 128` frames (`384 x 1024` total) for idle, move, telegraph, attack, projectile, buff, hit, and defeat states. The audit also covered the compact source sheets, enemy combat-effect sheets, Canvas rendering, and Pixi rendering.

## Root causes

1. Canvas drew square monster frames into non-square boxes, stretching ordinary enemies and rendering most bosses at ordinary-enemy scale.
2. The compact-sheet processor calculated a different scale for every pose. Crouches, wide effects, and imperfect body detection therefore made the creature grow or shrink between frames.
3. The processor constrained the detected body but not the complete visible bounds. Wide tails, wings, attacks, and projectiles were silently clipped at the cell edge.
4. Compact validation stopped after checking dimensions and the forcibly cleared outermost pixel, so it did not detect inner-edge clipping, center drift, scale pumping, or all adjacent duplicates.
5. Compact animation metadata ignored species timing overrides and exposed only one of the three authored hit frames.
6. Four enemy combat-effect pairs collided to the same generated color identity.

## Repair sequence

### 1. Representative pilot

- Use Rift Aberration because its move row had the largest stable-row body variance.
- Preserve its identity and all 24 action meanings while normalizing torso scale and ground registration.
- Process the candidate through the same deterministic converter used by the full catalog.
- Compare all three frames in idle and move, both adjacent frame pairs, transparent gutter, center drift, height variance, and rendered aspect ratio.

### 2. Deterministic sheet conversion

- Derive a stable scale from the creature's body across its non-effect poses instead of independently fitting every frame.
- Constrain the complete visible bounds to a safe transparent gutter.
- Preserve a fixed authored origin for grounded and flying monsters.
- Use high-quality downsampling for painterly source artwork.
- Keep body registration stable while retaining genuine attack, projectile, buff, hit, and defeat motion.

### 3. Renderer parity

- Use one visual profile for Canvas and Pixi.
- Preserve the square source-frame aspect ratio.
- Share boss sizing, flyer registration, and authored baseline handling between both renderers.

### 4. Catalog rollout

- Regenerate all 48 compact monster sheets and portraits only after the pilot passes.
- Enable all three hit frames and apply compact timing overrides.
- Make generated combat-effect identities unique across the catalog.
- Retain the 44 legacy standard sheets as source provenance only; active runtime definitions continue to use compact sheets.

### 5. Regression coverage

- Validate exact catalog coverage and sheet dimensions.
- Reject meaningful inner-edge occupancy, unintended adjacent duplicates, unstable body scale, excessive center drift, and renderer profile mismatch.
- Run asset, runtime, smoke, and full repository tests.
- Play representative grounded, flying, ranged, elite, and boss encounters in both Pixi and forced Canvas modes.

## Acceptance criteria

- Every active runtime frame preserves a `1:1` render aspect ratio.
- Canvas and Pixi use the same monster size and registration profile.
- Stable idle and movement poses do not visibly pump in scale.
- Used frames retain a transparent safety gutter without clipping meaningful art.
- Both adjacent pairs (`1 -> 2` and `2 -> 3`) are checked for unintended duplication.
- All three hit poses are reachable at runtime.
- The full 48-monster catalog passes deterministic regeneration and validation.

