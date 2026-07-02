# Project Starfall Item UI Overlays

Generated category marker overlays live here. They are produced by:

```bash
npm run build:project-starfall-item-visual-assets
```

Current runtime policy:

- Inventory icons do not include baked frames.
- Rarity aura and many overlays are handled by CSS/canvas.
- Category and theme must read from the item art itself.

Generated overlays remain separate from item icons so one icon can work in inventory, shop, tooltip, hotbar, reward, and loot contexts.
