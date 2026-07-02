# Project Starfall Item Source Art

This folder is for source references, working files, and generated source material that should not be treated as final runtime art.

Runtime item icons belong in:

```text
img/project-starfall/items/icons/
```

AI source sheets belong in:

```text
img/project-starfall/items/source/
```

Processed runtime sheets belong in:

```text
img/project-starfall/items/sheets/
```

Follow `ITEM_VISUAL_DESIGN_GUIDE.md` and `img/project-starfall/items/item-visual-manifest.json` before creating or importing any item art.

Source sheet rules:

- Use `#00ffff` guide lines around and between cells.
- Use a flat chroma-key background such as `#00ff00` or `#ff00ff` when the processor expects key removal.
- Keep one item per cell.
- Do not include labels, text, fake UI, watermarks, or background scenes.
- Match the cell order in `build/process-project-starfall-ai-item-icons.js`.
