# Project Starfall player animation v2

Generation mode: built-in image generation, using `plain-adventurer-base.png` as the identity and style reference.

Final generation prompt:

> Create one exact sprite-sheet image for Project Starfall, preserving the attached plain adventurer's identity, proportions, brown spiky hair, cream tunic, dark trousers, brown boots, warm painterly pixel-art shading, and right-facing three-quarter view. Output an exact 6-column by 10-row grid: 60 equal cells, one character per cell, consistent scale and registration, transparent-ready solid chroma-green background in every cell, and thin bright-cyan grid lines including the outer border. Rows from top to bottom must be: 1 idle breathing, 2 run with a real alternating stride, 3 jump from anticipation through takeoff and apex, 4 fall with open descent and landing brace, 5 climb with alternating reach and pull, 6 unarmed basic attack with windup/contact/follow-through/recovery, 7 unarmed skill channel and release, 8 party-support raise/flare/lower, 9 hit reaction recoil and recovery, 10 defeat from stagger to kneel to controlled collapse and still final pose. Six sequential frames left to right in every row. Fixed bottom-center registration for grounded poses, stable anatomy, stable head/body size, no stretching, no duplicated limbs, no text, no labels, no embedded weapons, no armor, no shield, no bow, no staff, no ground shadow, no scenery, no particles crossing cell borders. Crisp readable game sprite silhouettes with meaningful limb articulation and no rigid whole-body rotation.

Generated source:

- `generic-player-v2-generated-chroma.png` is the untouched built-in generation result.
- `generic-player-v2-generated-source.png` is the chroma-keyed transparent source consumed by the player asset processor.

The runtime body sheet is intentionally weaponless. Equipped weapons and armor are generated as independent sheets from the shared attachment rig so the visible character always reflects the live equipment slots.
