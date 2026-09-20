# Rejected player run candidate - not included in corrected samples

Built-in image_gen was used with the existing classic generic-player-sheet.png as the identity reference. The retained rejected candidate is the first generation, copied unchanged for provenance only. Both attempts were rejected: the near-side leading boot repeats instead of exchanging clearly with the far leg after the half-cycle. The targeted second iteration introduced a more abrupt arm/leg pose change without reliably solving that bilateral gait.

Actual retained output: 1448 x 1086 PNG with alpha and a nominal 4-column x 3-row layout. Requested 1024 x 768 was not honored. Although the dimensions divide into 362 x 362 cells, pixel analysis shows that the first-row boots enter the next nominal row: visible pixels touch row-0 bottom and row-1 top in all four columns. The generated spacing therefore does not obey the required export grid either.

Do not present this as corrected, approved, seamless, or ready for runtime import. Character identity was preserved, but bilateral articulation and registration failed. It is excluded from the corrected feedback samples. No game assets or runtime code were replaced.

## Initial prompt

+Use case: identity-preserve.
Asset type: ONE review sprite sheet for the shared Project Starfall player RUN animation.
Input image 1 is a strict character-identity and art-style reference, NOT a pose sequence to copy.
Create a completely redrawn, genuine twelve-pose bilateral running gait for exactly the same small heroic character: brown spiky hair, cream short-sleeved tunic/shirt, dark trousers, brown gloves/bracers and brown boots, small gold belt buckle, no weapons. Keep the exact compact classic chibi proportions, face, hair silhouette, costume, colors and crisp pixel-like painted shading of the reference. Fixed right-facing 3/4 side-view in every pose; upper-left light.
Output one PNG with actual transparent alpha background, no backdrop. Prefer exact image size 1024 by 768, arranged as FOUR columns by THREE rows of equal 256x256 cells. There must be exactly 12 complete separate sprites, one per cell, strictly even invisible grid. Reading order is chronological left-to-right across each row, then next row; it is ONE continuous run cycle. No labels/grid lines/letters/numbers.
Every cell uses the same scale: character roughly210px tall, pelvis/root horizontal registration x128 within the cell. The ground-contact baseline is y232 within every cell. Keep body centered in place and permit only2-4px intentional vertical running bob. Do not translate the character sideways between frames. Entire hair, hands and boots remain inside the cell with generous clear margins.
AUTHORED TIMING, 12 sequential poses, one full left-right stride:
0 near leg extended forward contacting ground, far leg behind, opposite arm forward;
1 weight compresses onto near leg, far heel lifts behind;
2 near leg supports body while far knee passes forward under pelvis;
3 push-off near leg, far knee drives high forward;
4 brief airborne flight, near leg trailing and far leg extending ahead;
5 far boot approaches its forward landing, near heel folds back;
6 far leg extended forward contacting ground, near leg behind, arms now fully opposite to pose0;
7 weight compresses onto far leg, near heel lifts behind;
8 far leg supports body while near knee passes forward under pelvis;
9 push-off far leg, near knee drives high forward;
10 brief airborne flight, far leg trailing and near leg extending ahead;
11 near boot approaches forward landing and far heel folds back, leading naturally into pose0.
Show REAL leg articulation at hip/knee/ankle and opposing arm swings; poses0 and6 are opposite-leg contacts, NOT copies or mirrored whole characters. Silhouette must visibly alternate. Keep head/torso scale and clothing details invariant while legs move. The final frame is a lead-in to the first, NOT a duplicated endpoint.
No idle/attack/jump/climb rows. No weapons, effects, sparks, dust, ground plane, cast shadows, motion blur, text, border, checkerboard painted into the image, or background color. Transparent alpha is essential. This is a motion/registration correction preserving approved classic artwork, not a character redesign.

## Targeted revision prompt (not selected)

+Use case: precise-object-edit / identity-preserve.
Edit only the running gait articulation of this12-pose sprite sheet. Preserve this exact classic brown-haired cream-shirted brown-gloved brown-booted adventurer, all colors/shading, the four-column three-row invisible equal-square grid, twelve cells, right-facing camera, genuine transparent alpha, uniform character scale and head silhouette. One running cycle in chronological row-major order0..11.
CRITICAL CORRECTION: the near leg closest to camera must swing both ahead AND behind the pelvis over the cycle. The generated sheet currently suggests the same leading boot repeatedly. Make the opposite-leg half-cycle unmistakable. Keep the near limb slightly lighter and outlined in front of the far limb so identity is readable.
Cells0..5: near leg goes from forward heel strike to compression, to vertical under hip, then toe-off behind body, then lifted folded behind, then beginning forward recovery. Far leg simultaneously progresses from trailing to knee driving forward and finishes forward heel strike.
Cells6..11: swap limb roles without turning or mirroring the character. In cell6 the NEAR leg is visibly extended BACK to the LEFT of the pelvis while the FAR leg extends FORWARD to the RIGHT and lands. Near arm swings FORWARD to the right as near leg is back; far arm swings back. Cell7 near heel lifts behind; cell8 near knee passes the stance far leg; cell9 near knee drives ahead; cell10 near leg extends ahead in flight; cell11 approaches near-leg landing leading into cell0. Make this near-leg-behind stance very clear, including a boot behind the torso at cell6.
Each sprite is a different intentional anatomically plausible in-between. Smooth contact/down/passing/up/flight/landing for both legs, opposing arm swings. Same pelvis x within EVERY cell, shared planted-foot baseline, small3px vertical bob only, consistent body/head size. Entire figure contained with clear margin. No horizontal sliding, no stretched limbs, no costume changes.
No extra figures, no text/numbers/labels, no grid, no background, no baked checkerboard, no shadows, no ground, no weapon, no effects. Keep actual transparency. First and last poses must connect naturally; do not duplicate endpoint.
