# Project Starfall Rarity Frames

Generated rarity frame overlays live here. They are produced by:

```bash
npm run build:project-starfall-item-visual-assets
```

Current runtime policy: rarity is rendered through data/CSS/canvas aura values, not baked into inventory icons. These frames are available as separate UI overlays if a future inventory/shop/tooltip view needs authored frame PNGs.

Implemented rarity colors:

- Common: `#d8e5ec`
- Uncommon: `#74d680`
- Rare: `#68a9ff`
- Epic: `#c794ff`
- Relic: `#ffbe55`

Do not bake these frames into inventory icons. Rarity frames must remain separate UI overlays so the item silhouette remains the primary read.
