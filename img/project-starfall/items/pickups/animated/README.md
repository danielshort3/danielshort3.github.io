# Project Starfall Animated Pickup Sheets

Generated generic animated pickup sheets live here. They are produced by:

```bash
npm run build:project-starfall-item-visual-assets
```

Current generated frame counts:

- Common pickup idle: 4 frames.
- Rare pickup glow: 6 frames.
- Quest item pulse: 8 frames.
- Currency sparkle: 4 frames.
- Health/mana pickup idle: 4 frames.
- Equipment drop reveal: 8 frames.
- Boss item drop: 12 frames.

Runtime does not yet consume these sheets directly. Add explicit item data and renderer references before relying on them in-game.
