# Project Starfall Prompt Templates

Use these templates with `ASSET_GENERATION_GUIDE.md`. The guide remains authoritative for full prompts, frame counts, dimensions, and review rules.

## Master Style Prompt

```text
Project Starfall Starlit Frontier Fantasy, original 2D side-scroller fantasy RPG asset, clean readable silhouette, crisp dark contour outline where appropriate, soft cel-shaded sprite rendering, subtle painterly color, luminous fallen-star magic accents, blue-and-gold guild motifs, practical crafted fantasy materials, readable at gameplay scale, consistent upper-left/front lighting, no copied IP.
```

For map backgrounds, replace sprite outline language with:

```text
painterly side-scroller background depth, clear gameplay lanes, atmospheric perspective, readable foreground/midground/background separation
```

## Master Negative Prompt

```text
text, labels, numbers, watermark, signature, logo, fake UI, unrelated background clutter, photorealism, 3D render, isometric gameplay view, top-down gameplay view, copied IP, copied MapleStory sprite, inconsistent style, warped anatomy, extra limbs, missing limbs, broken hands, broken feet, melted outlines, noisy unreadable details, over-bloom, muddy colors, cropped subject, cell overflow, inconsistent lighting, inconsistent proportions, changing costume, changing weapon, changing face
```

For source sheets, also add:

```text
guide grid color inside artwork, chroma key color inside artwork, row labels, column labels, frame numbers, character crossing grid lines
```

## Player Sheet Template

```text
Create an original Project Starfall player family animation source sheet for [FIGHTER, MAGE, OR ARCHER], an athletic adult 2D side-scroller sci-fantasy frontier operative.
Style: Fractured Starfront Sci-Fantasy, clean cel-shaded 2D game sprite, crisp dark contour, realistic 5.5-to-6-head proportions, readable family silhouette, restrained painterly material detail, charcoal field gear, cyan star-tech seams, ember utility accents, no copied IP.
Character lock: [HAIR], [FACE VISIBILITY], [FAMILY OUTFIT SILHOUETTE], [ARMOR OR MANTLE PROFILE], [FAMILY COLORS]. Keep anatomy, face, materials, and costume identical in every frame. Keep the body weaponless; weapons, helmets, chest pieces, gloves, boots, jewelry, and specialist gear are separate runtime atlases.
Registration lock: preserve the approved Fracture Runner feet, torso, head, main-hand, off-hand, and ground sockets in every pose. Fighter shoulder armor and Mage mantle must not hide the hand sockets or extend below the two-pixel ground tolerance.
Sheet layout: 6 columns x 10 rows, 160px normalized frame target, one centered character per cell, right-facing, transparent-ready, consistent baseline, generous padding, no labels. Use solid #00ffff outer and cell guide lines plus a flat processor-safe chroma background outside the character.
Rows: idle, run, jump, fall, climb, basic, skill, party, hit, defeat.
```

## Enemy Compact Sheet Template

```text
Create one original Project Starfall compact enemy animation source sheet for [ENEMY_NAME].
Style: Starlit Frontier Fantasy, charming 2D side-scroller RPG monster, clean cel-shaded sprite, crisp dark contour outline, readable silhouette at 128px, compact proportions, no copied IP.
Enemy design: [VISUAL DESCRIPTION]. Gameplay role: [ROLE]. Attack tell: [ATTACK TELL]. Regional palette: [REGION PALETTE].
Sheet layout: 3 columns x 8 rows, 128px frame target, right-facing, one enemy centered per cell, consistent baseline or hover center, generous padding. Use cyan #00ffff guide grid lines. Use flat [#00ff00 or #ff00ff] processor-safe background outside the art.
Rows: idle, move, telegraph, attack, projectile, buff, hit, defeat.
```

## Item Icon Template

```text
Create a Project Starfall 64x64 transparent fantasy RPG item icon for [ITEM_NAME].
Style: Starlit Frontier Fantasy, clean painted 2D icon, crisp silhouette, controlled cel-shaded detail, dark edge separation, readable at 64px, [TIER MATERIALS], [REGION OR CLASS MOTIF].
Composition: one centered item, no background, no UI border, no rarity frame, no quantity number, no label, no watermark.
Use palette: [PALETTE]. Leave transparent padding around the item.
```

## Map Background Template

```text
Create a 1280x640 Project Starfall side-scroller panoramic background for [MAP_NAME].
Style: Starlit Frontier Fantasy, painterly 2D fantasy background, readable side-scroller depth, clear gameplay lanes, layered parallax feel, no characters, no monsters, no UI, no text.
Region: [REGION DESCRIPTION]. Palette: [PALETTE]. Required landmarks: [LANDMARKS].
Composition: foreground gameplay platform areas must remain visually clear, midground supports the theme, background has atmospheric depth. Left and right edges must tile or blend seamlessly.
```

## Skill FX Template

```text
Create a Project Starfall skill VFX source sheet for [SKILL_NAME], 960x640 target, 6 columns x 4 rows, 160x160 frames.
Style: clean 2D fantasy combat VFX, Starlit Frontier Fantasy, luminous but readable, no character body, no UI, no text, no background clutter.
Rows: cast, projectile, impact, area.
Palette and motif: [CLASS COLOR AND MOTIF].
Keep every frame centered and inside cell bounds.
```

## UI Icon Template

```text
Create a 64x64 transparent Project Starfall menu icon for [ICON_ID].
Style: clean fantasy RPG UI icon, dark navy/gold/cyan Starfall palette, readable silhouette, simple symbol, subtle bevel, no text, no numbers, no background panel unless the symbol requires a small internal shape.
The icon must be clear at 32px and 64px.
```
