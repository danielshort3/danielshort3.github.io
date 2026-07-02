# Project Starfall Item Pickup Sprites

Generated pickup sprites live here. They are derived from approved runtime item icons and card icons by:

```bash
npm run build:project-starfall-item-visual-assets
```

Current runtime policy: world pickups still reuse inventory icons from `img/project-starfall/items/icons/`. These generated 64x64 pickup sprites are available for future renderer/data integration through a field such as `pickup_sprite_path` or `pickupAsset`.

Generated static pickup contract:

- `pickup-<item-id-kebab>.png` for item icons.
- `pickup-card-<card-id-kebab>.png` for card icons.
- 64x64 transparent PNG.
- Stronger world-read silhouette treatment than the source icon.
- Runtime rarity-colored aura/shadow baked into the derived pickup asset.
- No text, watermark, fake UI, or background scene.
