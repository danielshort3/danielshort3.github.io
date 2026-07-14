# Rift Aberration compact source repair prompt

## Generation record

- Mode: identity-preserving image edit
- Input: `asset-sources/project-starfall/enemies/compact/rift-aberration-compact-source.png`
- Candidate output: `asset-sources/project-starfall/enemies/compact/rift-aberration-v2-generated-source.png`
- Required dimensions: `887 x 1774`

## Prompt

Precise sprite-sheet correction edit. Preserve the exact same Rift Aberration creature identity, silhouette language, purple-black palette, glowing vertical eye, tentacles, painterly pixel-art/game-sprite style, action meanings, facing direction, and all 24 poses. Preserve the exact 3-column by 8-row layout, order, margins, bright flat chroma-green background, thin cyan divider grid, and no text or labels. Correct only animation consistency: keep the creature's core torso, head, and eye at one consistent apparent scale across frames, especially all three idle frames in row 1 and all three locomotion frames in row 2; keep a consistent ground baseline within each grounded row; prevent squashing, stretching, or accidental redesign; retain genuine tentacle, dust, projectile, impact, buff, hit, and defeat motion. Each cell must contain exactly one pose and nothing may cross a cyan divider. Maintain generous safe padding around every cell. Output a clean production-ready source sprite sheet matching the reference aspect ratio and full sheet composition.

## Review requirements

- The cyan guide grid must still be machine-detectable as 3 columns by 8 rows.
- No frame may cross a guide divider.
- The creature identity and row semantics must match the original.
- The candidate must pass the compact processor and pilot quality checks before replacing the tracked source.

