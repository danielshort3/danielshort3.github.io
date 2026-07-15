# Fighter Ground Slam V2 source prompt

- Mode: built-in ImageGen with an existing Project Starfall VFX sheet used only as a layout and rendering-style reference.
- Reference image: `img/project-starfall/animations/combat-fx/skills/source/rune-mage-ground-glyph-sheet.png`
- Intended output: `img/project-starfall/animations/combat-fx/skills/source/fighter-ground-slam-sheet.png`

## Final prompt

> Use case: style-transfer
> Asset type: Project Starfall combat skill VFX sprite-sheet source
> Input images: Image 1 is only a grid, animation-density, and luminous painterly rendering reference; do not preserve its rune shapes, teal palette, or spell identity.
> Primary request: Create a new Fighter Ground Slam VFX source sheet with an exact 6-column by 4-row grid and six sequential animation frames in every row. The overall canvas must be 3:2 landscape so all 24 cells are equal. Row 1 is CAST: physical force gathers downward from a compact spark into a charged circular pulse, clearly building frame by frame. Row 2 is PROJECTILE: a low horizontal ground shockwave travels to the right, growing a sharp gold-white leading edge and orange/steel-blue energy wake. Row 3 is IMPACT: a downward ground strike erupts from a small contact flash into a broad radial impact with dust, stone fragments, and a clean recovery frame. Row 4 is AREA: a grounded elliptical shock ring expands outward across the floor, peaks, and dissipates; it must remain low and wide rather than becoming a character-shaped aura.
> Style/medium: polished 2D fantasy action-RPG VFX, painterly pixel-art readability, crisp silhouettes, bright cores, restrained bloom, compatible with Project Starfall's existing sprites.
> Composition/framing: one effect centered inside each equal cell with at least 8 percent padding; no effect may cross a cell boundary. Grounded rows share a stable bottom-center registration. The projectile row stays horizontally centered and points right.
> Color palette: ember orange, warm gold, white-hot highlights, restrained steel blue, and a little dust brown. Do not use green in the effect.
> Scene/backdrop: perfectly flat solid #00ff00 chroma-key background filling every cell, with no shadows, gradients, texture, reflections, floor plane, or lighting variation.
> Constraints: exactly 24 distinct sequential frames; consistent scale and registration; readable anticipation, contact, peak, and recovery; no character body, humanoid silhouette, weapon, UI, text, labels, watermark, scenery, border, or grid line. No duplicated frames, clipped fragments, rigid whole-sheet rotation, or particles crossing cell boundaries.
