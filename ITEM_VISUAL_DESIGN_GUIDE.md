# Project Starfall Item Visual Design Guide

This guide is the visual production manual for Project Starfall item assets. It is tailored to the current static browser prototype, the Project Starfall GDD, the item data, the inventory and loot UI, and the existing asset pipeline.

## Repository Execution Contract

The guide is implemented in the repo through these companion files:

- Visual manifest: `img/project-starfall/items/item-visual-manifest.json`.
- Prompt templates: `asset-sources/project-starfall/prompts/item-visual-prompts.md`.
- Workflow docs: `docs/project-starfall-item-visuals/README.md`.
- Generated pickup sprites: `img/project-starfall/items/pickups/`.
- Generated item VFX sheets: `img/project-starfall/animations/item-vfx/`.
- Generated item UI overlays: `img/project-starfall/ui/item-overlays/`.
- Derived asset generator: `npm run build:project-starfall-item-visual-assets`.
- Validation command: `npm run validate:project-starfall-item-visuals`.

The manifest is the machine-checkable contract for current icon dimensions, generated pickup sprites, rarity frames, category markers, item VFX sheets, animated pickup sheets, category/theme inference, rarity colors, prompt template locations, metadata fields, and required item visual folders. Update the manifest whenever item visual folders, generated asset patterns, categories, themes, prompt paths, or runtime pickup policies change.

## 1. Project-Specific Item Visual Analysis

### Primary Sources Inspected

- `project_starfall_gdd_v0_5.md`: primary source for tone, mechanics, itemization, gear tiers, rarity, art direction, and AI asset standards.
- `img/project-starfall/asset-prompts.md`: source-of-truth notes for AI source sheets, cyan guide-line grids, chroma-key backgrounds, player/equipment sheets, and current item icon generation batches.
- `js/games/project-starfall/data/items.js`: authoritative item icon mapping, material items, rarity visuals, item asset root paths, and file-name conversion.
- `js/games/project-starfall/data/consumables.js`: consumables, potions, rations, coupons, scrolls, skill manuals, Plinko balls, and account/progression utility items.
- `js/games/project-starfall/data/equipment-catalog.js`: shop equipment, world equipment, boss sets, drop economy, tiers, classes, and item sources.
- `js/games/project-starfall/data/equipment-visuals.js`: equipment-layer naming, 960x1600 animation-sheet contract, visual-slot layering, and theme inference.
- `js/games/project-starfall/data/cards.js`: monster card definitions, rarity, card icon roots, card tags, and card themes.
- `js/games/project-starfall/data/monster-drops.js`: enemy drop pools, card material drops, biome material drops, equipment groups, and boss rewards.
- `js/games/project-starfall/ui/inventory-config.js`: inventory tabs, grid sizing, stack caps, visible rows, and tab categories.
- `js/games/project-starfall/ui/item-assets.js`: runtime icon lookup, item drawing, rarity aura behavior, and canvas icon rendering.
- `js/games/project-starfall/engine/loot.js`, `js/games/project-starfall/project-starfall-engine.js`, and `js/games/project-starfall/renderer-pixi.js`: world loot draw size, shadows, hover motion, aura rings, pickup fly effects, and ownership dimming.
- `css/games/project-starfall/inventory.css`: inventory tile visual treatment, icon scale, level badges, locks, arrows, and parchment/gold UI language.
- `css/games/project-starfall/item-theme.css`: rarity aura CSS and attunement border treatment.
- `build/process-project-starfall-ai-item-icons.js`: current item source sheet definitions, runtime icon output size, chroma cleanup, validation, and backup behavior.
- `docs/project-starfall-balance-optimization-audit.md`: current equipment count and progression balance context.
- `docs/project-starfall-monster-book-blueprint.md`: enemy family and monster-card context.
- `docs/project-starfall-ui-visual-iteration-audit.md`: UI visual direction and readability issues.

### Current Item Types

The current game uses these item types in data and UI:

- Equipment: weapons, armor, boots, gloves, helms, robes, plates, bows, staffs, offhands, rings, charms, and class-specific gear.
- Consumables: health potions, resource tonics, rations, Town Return Scroll, Guard Tonic, Swiftstep Oil, Magnet Charm, Pet Whistle, coupons, skill manuals, reset scrolls, Plinko balls, and admin/service items.
- Materials: Upgrade Dust, Upgrade Catalyst, Warding Scroll, Refinement Core, Line Catalyst, Prism Shards, card star materials, enemy drops, biome materials, and boss materials.
- Currency: coins and coin-stack variants.
- Cards: monster cards and star-card materials.
- Boss drops: named equipment sets and boss-specific materials.
- Utility/account items: inventory slot coupons, card slot coupons, XP/drop coupons, Plinko balls, and admin console.

The GDD also implies future or partially represented categories:

- Quest-critical items and keys.
- Lore collectibles.
- Environmental pickups.
- Dedicated world pickup sprites separate from inventory icons.
- Mythic rarity items, currently described in the GDD but not implemented in `ITEM_RARITY_VISUALS`.

### Current Visual Style

Project Starfall's intended item style is "Starlit Frontier Fantasy": clean luminous side-scrolling MMO fantasy where frontier guild life meets fallen-star magic. The GDD specifies painterly environments, clean cel-shaded sprite assets, crisp contour outlines, luminous star-magic accents, and equipment-driven character identity.

The current item icons follow this direction unevenly but generally use:

- Isolated 2D item art.
- Transparent runtime PNG output.
- Readable 64x64 icon silhouettes.
- Bright fantasy materials such as brass, leather, crystals, steel, parchment, potions, runes, and star fragments.
- Rarity auras drawn by the renderer instead of baked into most icons.
- Strong item-type silhouettes for potions, coins, swords, bows, staffs, scrolls, cards, and materials.

### Current Icon Style

Runtime item icons are 64x64 PNGs under `img/project-starfall/items/icons/`. The current processor outputs transparent 64px icons from AI source sheets under `img/project-starfall/items/source/`, then writes processed sheets under `img/project-starfall/items/sheets/`.

The current item icon contract is:

- Source concept target: usually 256x256 transparent isolated item art, or a larger AI source sheet with bordered cells.
- Runtime output: 64x64 transparent PNG.
- File path: `img/project-starfall/items/icons/<kebab-id>.png`.
- Mapping: `ITEM_ASSETS` in `js/games/project-starfall/data/items.js`.
- Naming conversion: item IDs are converted from snake_case or camelCase concepts to kebab-case file names by `itemIconFileName`.
- No text, no fake UI, no baked inventory slot, no watermark, and no background scene.

Cards use a separate path:

- Runtime card icons: `img/project-starfall/cards/icons/`.
- Source sheet: `img/project-starfall/cards/source/ai-card-icons.png`.
- Runtime size: 64x64 PNG.

Equipment character overlays use a separate contract:

- Runtime sheets: `img/project-starfall/equipment-layers/<visual-id-kebab>-sheet.png`.
- Sheet size: 960x1600 PNG.
- Grid: 6 columns by 10 rows.
- Cell size: 160x160.
- Animation row order: idle, run, jump, fall, climb, basic, skill, party, hit, defeat.

### Current World Pickup Style

There are no dedicated standalone world pickup sprite assets for most items. The current world loot presentation reuses inventory icons and adds runtime behavior:

- Loot object data uses `w: 34` and `h: 34`.
- Tier aura items, such as equipment and cards, draw in a 44px box with a 5px icon pad, leaving about 34px for the icon.
- Non-tier stackable items draw in a 58px box with a 4px icon pad, leaving about 50px for the icon.
- Settled loot bobs vertically by about 5px.
- Loot has a shadow ellipse under the item.
- Equipment and card drops receive rarity aura rings and glow.
- Other-player loot is dimmed and grayscale.
- Pickup uses a fly-to-target effect with shrink, fade, arc, and optional aura ring.

Because world drops currently reuse inventory icons, every item icon must remain readable both at 64x64 inventory size and at roughly 34px to 50px world size.

### Current Inventory and Menu Presentation

Inventory is a parchment and gold fantasy UI over dark navy translucent panels. Relevant constraints:

- Inventory tabs: Equipment, Usable, Etc, Cards.
- DOM inventory grid: 4 columns.
- Canvas inventory cell size: 58px with 5px gap.
- Visible storage rows: 6.
- Tile styling: warm parchment/gold surfaces with beveled fantasy borders.
- Icon wrapper max size in CSS: about 86x86 inside the tile.
- Level badge appears at top-left.
- Lock badge appears at bottom-left.
- Directional/upgrade arrows can appear on the right/top area.
- Empty tiles have a faint star motif.

Icons must not place essential information in corners that are likely to be covered by level badges, lock badges, stack counts, or comparison arrows.

### Current Rarity and Progression Indicators

Current implemented item rarity visuals are defined in `ITEM_RARITY_VISUALS`:

| Rarity | Color | Use |
| --- | --- | --- |
| Common | `#d8e5ec` | Plain, muted, low aura |
| Uncommon | `#74d680` | Slight green aura and cleaner accents |
| Rare | `#68a9ff` | Blue aura, distinct accents |
| Epic | `#c794ff` | Purple aura, stronger glow and pulse |
| Relic | `#ffbe55` | Gold aura, strongest current drop presence |

The GDD also describes Mythic as an aspirational/future endgame rarity with animated effects and iconic silhouettes. Mythic should not be used for runtime item drops unless `ITEM_RARITY_VISUALS`, CSS rarity classes, renderer logic, sorting, tests, and item data all support it.

Attunement uses additional border language in CSS for Rare, Epic, Relic, Mythic, Ascendant, and Celestial line states. Do not confuse attunement border states with implemented item rarity unless code explicitly maps them.

### Current Naming Conventions

There are two naming layers:

- Data IDs: most consumables and equipment use snake_case, such as `minor_health_potion`, `copper_sword`, and `thorncrown_greatsword`.
- Some material definitions use camelCase keys, such as `upgradeDust`, `cubeFragment`, and `gelDrop`; asset IDs are still normalized through mapping and kebab file names.

Runtime icon file names are kebab-case:

- `minor_health_potion` becomes `minor-health-potion.png`.
- `copper_sword` becomes `copper-sword.png`.
- `stormbreak_plume` becomes `stormbreak-plume.png`.

Current item icon assets live at:

- `img/project-starfall/items/icons/<kebab-id>.png`.

Current item source sheets live at:

- `img/project-starfall/items/source/ai-items-<batch-name>.png`.

Current processed sheets live at:

- `img/project-starfall/items/sheets/ai-items-<batch-name>-sheet.png`.

### Current Folder Structure

Use these current folders as authoritative unless code is changed:

| Purpose | Current folder |
| --- | --- |
| Runtime item icons | `img/project-starfall/items/icons/` |
| Runtime item sheets | `img/project-starfall/items/sheets/` |
| AI item source sheets | `img/project-starfall/items/source/` |
| Runtime card icons | `img/project-starfall/cards/icons/` |
| AI card source sheets | `img/project-starfall/cards/source/` |
| Equipment overlay sheets | `img/project-starfall/equipment-layers/` |
| Equipment source references | `asset-sources/project-starfall/players/equipment/` |
| Skill icons | `img/project-starfall/skills/base/` and `img/project-starfall/skills/advanced/` |
| UI menu icons | `img/project-starfall/ui/menu-icons/` |
| Combat/item VFX | `img/project-starfall/animations/fx/` and `img/project-starfall/animations/combat-fx/` |
| Environment art | `img/project-starfall/environment/` |
| Map art | `img/project-starfall/maps/` |

Recommended future folders, after code support is added:

- `img/project-starfall/items/pickups/`
- `img/project-starfall/items/pickups/source/`
- `img/project-starfall/animations/items/`
- `img/project-starfall/animations/item-vfx/`
- `img/project-starfall/ui/item-overlays/`
- `asset-sources/project-starfall/items/`

### Current Strengths

- The 64x64 icon contract is clear and enforced through `build/process-project-starfall-ai-item-icons.js`.
- The cyan guide-line sheet workflow is documented in the GDD and `asset-prompts.md`.
- Runtime rarity aura is centralized in data, CSS, and rendering, so icons do not need baked frames.
- Item taxonomy is rich: shop gear, random world gear, boss sets, consumables, upgrade materials, monster materials, cards, star-card materials, coupons, and utility items.
- Biomes and enemy families already have strong material names: moss, thorn, gears, coils, ash, rime, storm, runes, void, eclipse, and rift.
- Gear progression has a strong material ladder: Training, Copper, Iron, Steel, Silver, Runed, Starforged, Ancient.
- The loot renderer already gives high-value drops a distinct aura and pickup effect.

### Current Weaknesses and Inconsistencies

- Some item visuals are generated in different batches and may vary in contour thickness, perspective, render detail, saturation, and material finish.
- Some gameplay-distinct gear currently reuses existing `assetId` or `visualId` values. Examples include class offhands and regional variants that borrow icons from similar items. This is acceptable for prototype coverage but should be replaced for final item identity.
- Current world pickups reuse inventory icons, so a visually complex inventory icon can become unreadable in the world.
- Root-level star-card images exist under `img/project-starfall/items/`, but current runtime item assets should be considered authoritative under `img/project-starfall/items/icons/`.
- Mythic rarity is described in the GDD but is not currently part of implemented item rarity visuals.
- Quest and key item visual categories are implied by the game structure but do not yet have a complete runtime item asset category.
- Equipment-layer regeneration is deferred in `asset-prompts.md` until the base player rig direction is accepted. Do not mass-regenerate equipment overlays until the base rig is locked.

### Gaps Between Visuals and the GDD

- The GDD asks for item-specific weapon and gear visuals to support equipment-driven identity. Current data supports this, but some items still use shared or borrowed visuals.
- The GDD calls for boss gear and relic gear to have unique lore identity. Current boss item names support this strongly; final icons should exaggerate each boss motif more than current generic fantasy items.
- The GDD says shop-tier gear should be simpler than dungeon or boss gear. Final icon production must preserve that hierarchy instead of making every generated item equally ornate.
- The GDD says high-tier gear should not just be bigger. Final tier upgrades must show better materials, cleaner craftsmanship, runes, crystal inlays, and precise silhouettes.
- The GDD's world tone is luminous frontier fantasy, not generic dark fantasy, modern sci-fi, or high-noise AI ornamentation. Every item must keep the friendly but dangerous frontier tone.

## 2. Core Item Visual Design Goals

Every Project Starfall item visual must communicate five things at a glance:

1. What it is.
2. What category it belongs to.
3. How important it is.
4. What theme, element, faction, class, biome, enemy, or progression tier it connects to.
5. Whether it is consumable, equippable, collectible, upgrade-related, quest-related, currency-related, or material-related.

### Required Qualities

- Instantly recognizable at 64x64 inventory size.
- Still recognizable at 34px to 50px world loot size.
- Thematically obvious before the player reads the tooltip.
- Visually distinct from similar items.
- Internally consistent across item categories.
- Clean, polished, and professional.
- Strong silhouette before interior detail.
- Compatible with Starlit Frontier Fantasy.
- Readable against parchment/gold inventory tiles and dark navy gameplay scenes.
- Clear category and importance without relying only on color.
- Free of clutter, fake UI, text, watermarking, over-rendering, and AI artifacts.

### Communication Rules

- Item identity comes first from silhouette.
- Category comes second from object family: bottle, scroll, blade, bow, staff, crystal, coin, card, gear part, pelt, shell, feather, claw, badge, or crown.
- Theme comes third from material and motif: moss, thorn, brass, gear teeth, ember cracks, frost facets, storm feathers, astral runes, eclipse corona, or rift splinters.
- Rarity comes fourth from material richness, aura, trim, and detail complexity.
- Color supports the read but cannot replace the read.
- Glows support magic, rarity, and special function but must not hide the object's shape.

## 3. Global Item Art Style Rules

### Canvas and Export Rules

| Asset type | Source recommendation | Runtime requirement |
| --- | --- | --- |
| Inventory item icon | 256x256 transparent isolated item, or AI source sheet cell | 64x64 transparent PNG |
| Card icon | Source sheet cell | 64x64 transparent PNG |
| Equipment overlay | 6x10 sheet of 160x160 cells | 960x1600 transparent PNG |
| Future world pickup sprite | 64x64 or 96x96 source, simplified | Add code support before runtime use |
| Future animated pickup sheet | 4 to 8 frames, equal cells | Add data and renderer references first |
| Future pickup burst VFX | 6 to 12 frames, equal cells | Use PNG sheet with alpha |

Current item icons must export to:

- `img/project-starfall/items/icons/<kebab-id>.png`
- 64x64 PNG.
- Transparent background.
- Clean transparent corners.
- No baked slot frame.
- No text.
- No fake UI.
- No background scene.

### Source Sheet Rules

Use the existing Project Starfall source sheet contract:

- Put generated item art in cells separated by `#00ffff` guide lines.
- Guide lines must border cells and must not appear inside art.
- Use a flat chroma-key background, usually `#00ff00` or `#ff00ff`, when generating sheet art for the processor.
- Use one item per cell.
- Do not let art bleed across cell borders.
- Do not include labels, text, numbers, UI frames, watermarks, drop shadows that touch borders, or cropped fragments.
- Match the item order exactly to the `items` list in `build/process-project-starfall-ai-item-icons.js`.

### Icon Layout Rules

- Place the item centered in the canvas.
- Keep 8% to 14% safe padding around the silhouette in the 256x256 source.
- In the final 64x64 icon, the visible object should usually occupy 46px to 56px on its longest side.
- Long weapons may use diagonal composition to fit without becoming tiny.
- Potions, cards, coins, materials, and scrolls should sit upright or slightly angled.
- Do not crop tips, handles, feathers, straps, auras, or spark details.
- Leave corners clear when possible because inventory UI can place badges and stack text there.

### Silhouette Rules

- The item must read in flat black before any interior detail is added.
- Each category needs a distinct silhouette family.
- Important items need a unique silhouette, not only a recolor.
- Never rely on tiny engravings, small runes, or thin internal lines for primary identity.
- Avoid mirrored blobs and over-rounded AI shapes.
- Avoid shape noise that creates a fuzzy outer contour.

### Outline Rules

- Use a crisp dark contour outline on most item icons.
- Keep outline thickness consistent across a batch.
- At 64x64, the outline should usually resolve as 1 to 3 pixels.
- Inner material separation lines should be thinner than the outer contour.
- Relic and boss items can have broken, glowing, or gilded outline accents, but the base contour must remain readable.

### Lighting Rules

- Use one consistent lighting direction: upper-left key light, lower-right form shadow.
- Highlights should be clean, simple, and material-specific.
- Cast shadows should be minimal and soft, not a visible scene floor.
- Do not use dramatic multi-light studio rendering.
- Do not use random rim lights unless the item is magical, relic, or boss-tier.

### Color and Palette Rules

Project Starfall's core palette:

- Dark foundation: midnight navy, deep blue, blue-black.
- Bright magic: star-white, lantern gold, crystal cyan, pale violet.
- Frontier material: tan stone, honey wood, brass, leather, parchment.
- Interface accent: gold, parchment, warm amber, star-white.

Use regional palette rules:

- Greenroot and Thornpath: soft greens, bark brown, moss, yellow light, thorn shadows.
- Rustcoil and Gearworks: brass, teal oxidation, gray steel, amber lamps, oil-black accents.
- Cinder and Emberjaw: basalt, ember orange, smoke gray, molten red, ash beige.
- Frostfen and Rimewarden: ice blue, pine green, silver, cyan, snow white.
- Stormbreak: blue-gray, violet, white lightning, cloud silver, charged yellow-white.
- Astral and Archive: indigo, crystal cyan, parchment, pale violet, star-white.
- Eclipse and Rift: black-violet, corona gold, void magenta, cold white.
- Quarry and Deepcore: stone gray, ore blue, muted gold, geode cyan, dusty brown.
- Bandit and frontier: dark leather, cloth red, iron, rope, bone, worn brass.

Color rules:

- Do not make icons one-note monochrome unless category clarity is still strong.
- Do not let rarity color overwrite category and theme color.
- Reserve intense saturation for potions, magic crystals, boss items, and relic accents.
- Common materials should use lower saturation and simpler material separation.
- Relic and boss items may use higher contrast, luminous cores, and richer trim.

### Texture and Detail Rules

- Use broad material planes first.
- Add detail only where it remains visible at 64x64.
- Keep micro-scratches, filigree, threads, tiny runes, and particles rare.
- Use 2 to 4 interior detail accents for ordinary items.
- Use 4 to 7 interior accents for Epic or Relic items.
- Boss items may have one signature motif repeated clearly, not random ornamentation.

### Transparency and Background Rules

- Runtime icons must have transparent backgrounds.
- Do not bake inventory tile frames into item icons.
- Do not bake full rarity frames into item icons because rarity is already handled by CSS and renderer aura.
- Do not use rectangular backplates except for item types whose object is a card, coupon, book, badge, or scroll.
- Do not use blurry ambient glows that fill the whole canvas.

### Glow, Frame, and Badge Rules

- Glows are allowed for magic, rarity, attunement, boss drops, star materials, and active consumables.
- Common materials usually have no glow.
- Uncommon items may have a small edge accent.
- Rare items may have a small controlled magical glow.
- Epic items may have a stronger contained aura.
- Relic items may have a distinct lore glow and bright core.
- Do not bake category badges into icons unless the category object is itself a badge or token.
- Rarity frames should be UI overlays, not baked into icon art.

### World Pickup vs Inventory Icon Rules

Current implementation reuses inventory icons for world pickups. Therefore:

- Inventory icons must remain readable at 34px to 50px.
- Avoid tiny interior details that vanish in the world.
- Make material silhouettes simple enough to read while bobbing.
- Place important color and shape mass near the center.
- Keep glows and particles inside the object footprint unless the item is rare, boss, or quest-critical.

Future dedicated pickup sprites should be simpler than inventory icons:

- Fewer internal details.
- Larger silhouette.
- Stronger outline.
- More contrast against the biome.
- Optional small idle glow or sparkle for valuable items.

### Anti-AI Artifact Rules

Reject any generated asset with:

- Text, pseudo-letters, fake UI labels, numbers, or watermark marks.
- Melted handles, inconsistent blade edges, impossible straps, or broken symmetry.
- Extra duplicate pieces attached to the item.
- Unclear front/back perspective.
- Random filigree that does not match category, theme, rarity, or function.
- Blurry outlines.
- Chroma-key residue.
- Over-sharpened noisy texture.
- Multiple unrelated objects in one cell unless the item is explicitly a stack or kit.
- Background scenes, smoke clouds, or floor shadows.

## 4. Item Category Visual Language

### Weapons

Purpose: Primary combat identity for Fighter, Mage, Archer, and advanced classes.

Visual identity:

- Strong directional silhouettes.
- Weapon type readable from outline.
- Class fantasy visible through materials and shape.
- Tier visible through craftsmanship and material quality.

Shape language:

- Fighter swords and axes: broad confident diagonals, angular metal, sturdy grips.
- Hammers and mauls: heavy top mass, squared or stone-like heads.
- Spears and polearms: long clean shaft, distinct tip.
- Mage wands and staffs: vertical or diagonal rod, crystal or rune focus near top.
- Orbs and tomes: centered magical mass with clear circular or book silhouette.
- Bows and crossbows: crescent or mechanical arc, string line, arrow/bolt suggestion.
- Throwers: compact crescent, blade fan, or star-like silhouette.

Primary colors:

- Training: wood, cloth, worn leather.
- Copper/Iron/Steel/Silver: tier material color.
- Runed/Starforged/Ancient: dark metal, gold, cyan crystal, pale violet, stone.

Accent colors:

- Class accents can appear as small cloth wrap or glow: Fighter red/gold, Mage cyan/violet, Archer green/teal.
- Element accents should stay secondary.

Texture and material:

- Weapons need clean metal planes, not noisy grunge.
- High-tier blades may include engraved runes, crystal cores, or star inlays.
- Shop weapons should be plain and practical.
- Boss weapons should carry the boss motif in the main silhouette.

Icon framing:

- No baked frame.
- Diagonal placement is preferred for long weapons.
- Keep weapon tips inside safe padding.

Pickup sprite rules:

- Current: use inventory icon with runtime aura.
- Future: weapon pickup can use simplified diagonal silhouette with small ground glint.

Rarity treatment:

- Common: plain blade/wood, no glow.
- Uncommon: cleaner trim, one accent.
- Rare: distinct guard, small glow or gem.
- Epic: sharper silhouette, richer material, visible magic channel.
- Relic: signature shape and lore motif.

Avoid:

- Generic fantasy sword with no tier, class, or biome signal.
- Blades hidden by large glow.
- Too many spikes on common weapons.
- Recolor-only variants.

Prompt template:

```text
Clean professional 2D game inventory icon of [weapon name], [weapon type] for Project Starfall, Starlit Frontier Fantasy, [tier material] with [theme motif], strong readable silhouette, crisp dark contour outline, upper-left lighting, simple cel-shaded material planes, centered on transparent background, no frame, no text, no watermark, readable at 64x64.
```

Negative prompt:

```text
text, letters, numbers, watermark, fake UI, background scene, inventory frame, cropped weapon, blurry outline, noisy texture, random spikes, excessive glow, extra blades, broken handle, photorealism, 3D render, AI artifacts.
```

File naming:

- Data ID: `weapon_name` or equipment item ID in snake_case.
- Runtime icon: `img/project-starfall/items/icons/<weapon-name>.png`.
- Equipment layer: `img/project-starfall/equipment-layers/<visual-id-kebab>-sheet.png`.

### Armor and Clothing

Purpose: Defensive gear, class fantasy, progression identity, and visible equipment overlays.

Visual identity:

- Armor icons should read as body-slot objects: helm, chest, gloves, boots, robe, plate, mask, crown.
- Material tier and biome motif must be visible without over-rendering.

Shape language:

- Helms: compact head silhouettes with clear crown/visor/mask shape.
- Chest armor: broad torso silhouette with shoulder and chest mass.
- Robes: vertical cloth silhouette with sleeves and rune panels.
- Gloves: paired gauntlets or a single readable glove at slight angle.
- Boots: paired boots or one larger boot with a second behind it.
- Crowns/masks: ceremonial centered shape with boss motif.

Primary colors:

- Cloth/leather for early gear.
- Iron, steel, silver for martial armor.
- Parchment, cyan, violet for mage robes.
- Green, bark, and thorn for forest armor.
- Brass and teal for construct armor.
- Basalt and ember for cinder armor.
- Ice blue and silver for frost armor.
- Indigo, star-white, and violet for astral armor.
- Black-violet and corona gold for eclipse armor.

Icon framing:

- No baked frame.
- Chest pieces should fill the center mass.
- Gloves and boots can use paired compositions but must not become cluttered.

Pickup sprite rules:

- Current: inventory icon reused.
- Future: simplify paired gloves/boots for world pickup with one dominant item and one small secondary item.

Rarity treatment:

- Increase material precision and trim quality by tier.
- Add controlled glow for runed, starforged, relic, or boss armor.
- Boss armor should carry an unmistakable boss motif.

Avoid:

- Flat shirt icons for armor with no defense identity.
- Random shoulder spikes that do not match theme.
- Tiny symmetrical details that disappear at 64x64.

Prompt template:

```text
Clean professional 2D game inventory icon of [armor name], [slot type] armor for Project Starfall, Starlit Frontier Fantasy, [material tier], [biome or boss motif], centered transparent background, strong slot-readable silhouette, crisp dark contour outline, upper-left lighting, clean cel-shaded planes, readable at 64x64, no frame, no text.
```

Negative prompt:

```text
text, watermark, UI frame, full character, mannequin, background, noisy fabric, tiny unreadable embroidery, melted straps, asymmetrical accident, excessive particles, photorealism, 3D render.
```

File naming:

- Icon: `img/project-starfall/items/icons/<armor-item-id-kebab>.png`.
- Equipment layer: `img/project-starfall/equipment-layers/<visual-id-kebab>-sheet.png`.

### Accessories and Offhands

Purpose: Utility identity, class specialization, secondary stats, and advanced class fantasy.

Visual identity:

- Small but iconic object silhouettes.
- Offhand items should be more specific than generic jewelry.
- Accessories must be readable even when small.

Shape language:

- Rings: circular band plus one large gem or rune notch.
- Amulets/charms: pendant silhouette with cord or metal loop.
- Badges: shield, star, medal, or faction mark.
- Shields: clear broad shield silhouette.
- Focus items: orb, seal, crystal, ember core, rune disk.
- Archer tools: scope, trap kit, quiver, beast token.
- Berserker grips: thick wrapped handgrip, metal studs, red cloth.
- Duelist medals: compact polished medal with parry/star motif.

Primary colors:

- Brass, leather, silver, cyan crystal, gold trim, class-specific accents.

Pickup sprite rules:

- Current: inventory icon reused.
- Future: accessories need enlarged simplified pickups because rings and charms can become too small.

Rarity treatment:

- Common accessories use simple metal and one stone.
- Rare and above can have active rune cores or star inlays.
- Relic accessories need unique silhouette, not just stronger glow.

Avoid:

- Tiny rings that occupy only the center of the canvas.
- Decorative shapes with no function.
- Multiple small trinkets in one icon unless item is a kit.

Prompt template:

```text
Clean professional 2D game inventory icon of [accessory/offhand name], [functional object type], Project Starfall Starlit Frontier Fantasy, [class or theme], large readable silhouette, crisp outline, one clear focal gem or emblem, upper-left lighting, centered transparent background, no frame, no text.
```

Negative prompt:

```text
tiny jewelry, unreadable ornament, text, watermark, UI frame, background, excessive sparkle, random symbols, blurred metal, duplicate objects.
```

File naming:

- Icon: `img/project-starfall/items/icons/<accessory-id-kebab>.png`.

### Consumables

Purpose: Immediate player action, survival, travel, buffs, account utilities, and progression helpers.

Visual identity:

- Must read fast in hotbar or inventory.
- Shape and container should show function.
- Tier progression should be visible without changing base identity.

Subtypes:

- Health potions.
- Resource tonics.
- Rations.
- Scrolls.
- Oils and tonics.
- Coupons.
- Manuals and reset scrolls.
- Pet whistle.
- Plinko balls.
- Slot coupons.
- Admin/service items.

Shape language:

- Health potion: rounded bottle, red liquid, cork or metal cap.
- Resource tonic: bottle with blue/violet liquid, sharper magic silhouette.
- Ration: wrapped food bundle, bread, jerky, or compact provision kit.
- Scroll: rolled parchment with seal.
- Oil: narrow vial or flask with slick highlight.
- Magnet charm: horseshoe magnet or charm shape with cyan/gold pull motif.
- Coupon: rectangular ticket with torn edge and foil color, no readable text.
- Manual: book silhouette, star or class mark, no letters.
- Reset scroll: distinct folded/rolled parchment with circular arrow-like rune, no actual UI arrow if too small.
- Plinko ball: polished orb with tier material.

Primary colors:

- HP: red and warm highlights.
- MP/resource: blue, cyan, violet.
- Rations: tan, brown, warm gold.
- Utility scrolls: parchment, red/gold seals, cyan runes.
- Coupons: distinct main colors already defined in processor constants.
- Manuals: parchment, leather, star-white/cyan marks.

Pickup sprite rules:

- Current: inventory icon reused.
- Future: potions and scrolls can use almost the same art but simplified with bigger liquid/readable seal.

Rarity treatment:

- Consumables should usually communicate tier by cap material, bottle shape, liquid brightness, and label/seal shape, not a full rarity aura.
- Coupons can show value through foil material and border richness, but no text.

Avoid:

- Tiny labels or unreadable pseudo-text.
- Potions differentiated only by hue.
- Overly elaborate bottles that blur at 64x64.

Prompt template:

```text
Clean professional 2D game inventory icon of [consumable name], [consumable subtype], Project Starfall Starlit Frontier Fantasy, [function color], clear readable container silhouette, simple magical accent showing [effect], crisp dark outline, upper-left lighting, centered transparent background, no text, no frame, readable at 64x64.
```

Negative prompt:

```text
text, letters, labels, watermark, UI frame, background scene, too many bottles, unreadable tiny details, liquid spill, excessive glow, photorealistic glass, AI artifacts.
```

File naming:

- Icon: `img/project-starfall/items/icons/<consumable-id-kebab>.png`.

### Potions and Tonics

Purpose: Clear HP/MP recovery and tiered survival reads.

Visual identity:

- Health and resource lines must share a family structure but never be confused.
- The player should recognize potion tier by silhouette upgrade plus cap material.

Tier rules:

- Minor: small round bottle, simple cork, bronze or plain cap.
- Standard: taller bottle, cleaner neck, silver cap.
- Greater: larger faceted bottle, gold cap, brighter liquid.
- Superior: heroic bottle, strongest shape, gold/star cap, controlled glow.

HP:

- Red liquid.
- Warm highlight.
- Rounder bottle.
- Small cross-like or heart-like silhouette only if abstract and non-textual.

MP/resource:

- Blue/violet liquid.
- Cyan highlight.
- Sharper bottle or star-rune cap.
- Arcane liquid swirl.

Avoid:

- Same silhouette with only color changes.
- Labels with letters.
- Glass reflection noise.

Prompt template:

```text
Clean professional 2D game icon of [tier] [health/resource] potion for Project Starfall, same family as other potion tiers, [red HP liquid or blue-violet resource liquid], [tier-specific bottle silhouette], [cap material], crisp outline, upper-left lighting, transparent background, no text, no frame, readable at 64x64.
```

Negative prompt:

```text
text, label, watermark, fake UI, background, clutter, multiple potions, unreadable cork, over-rendered glass, random ornament, excessive glow.
```

File naming:

- `minor-health-potion.png`
- `standard-health-potion.png`
- `greater-health-potion.png`
- `superior-health-potion.png`
- `minor-resource-tonic.png`
- `standard-resource-tonic.png`
- `greater-resource-tonic.png`
- `superior-resource-tonic.png`

### Crafting Materials and Enemy Drops

Purpose: Loot identity, crafting economy, biome memory, monster guide reinforcement, and upgrade progression.

Visual identity:

- Materials should look like harvested parts, refined scraps, crystals, hides, fibers, feathers, plates, glands, pages, or sigils.
- Each material should clearly belong to its enemy family or biome.

Shape language:

- Drops: bead, claw, fang, horn, feather, hide, plate, silk, page, shard, core, crown, badge.
- Natural materials: organic curves, bark, hide texture, moss edges.
- Construct materials: hard mechanical geometry, bolts, gear teeth, brass, teal oxidation.
- Cinder materials: jagged ember shapes, cracked basalt, ash, molten edge.
- Frost materials: sharp facets, translucent blue, snow rim.
- Storm materials: feathered arcs, lightning forks, cloud silk curls.
- Astral materials: parchment pages, runes, cyan crystal, star fragments.
- Eclipse materials: black-violet shards, gold corona edge, void magenta core.

Primary colors:

- Follow biome or enemy family palette.
- Keep Common materials muted.
- Reserve bright glow for magical cores, boss materials, and rare crafting items.

Pickup sprite rules:

- Current: use icon.
- Future: material pickups should use simple, chunky silhouettes and low glow.

Rarity treatment:

- Common enemy parts: no aura, simple material.
- Rare materials: one glow vein, cleaner silhouette, stronger contrast.
- Boss materials: unique silhouette and small controlled aura.

Avoid:

- Generic rocks for every material.
- Piles that look identical except color.
- Materials that look like equipment.
- Unreadable loose fragments.

Prompt template:

```text
Clean professional 2D game inventory icon of [material name], crafting material from [enemy family or biome], Project Starfall Starlit Frontier Fantasy, [organic/metal/crystal/parchment] material, strong simple silhouette, clear [theme motif], crisp outline, upper-left lighting, centered transparent background, no frame, no text, readable at 64x64.
```

Negative prompt:

```text
text, watermark, UI frame, background scene, pile of random debris, too many fragments, blurry edges, noisy texture, unrelated ornaments, fake letters, excessive glow.
```

File naming:

- Icon: `img/project-starfall/items/icons/<material-id-kebab>.png`.

### Upgrade and Attunement Materials

Purpose: Gear enhancement, risk systems, stat-line progression, and long-term item investment.

Current examples:

- Upgrade Dust.
- Upgrade Catalyst.
- Warding Scroll.
- Refinement Core.
- Prism Shards.
- Attunement Prism.
- Echo Prism.
- Line Catalyst.

Visual identity:

- These should look more magical and technical than ordinary materials.
- They need distinct shapes because players will use them in high-stakes upgrade decisions.

Shape language:

- Upgrade Dust: small pile or vial of luminous star grains.
- Upgrade Catalyst: faceted crystal reagent or charged orb.
- Warding Scroll: parchment with protective seal, shield-like rune.
- Refinement Core: compact metallic/crystal core.
- Prism Shard: angular shard with rainbow/star refraction.
- Attunement Prism: larger cube/prism with contained star light.
- Echo Prism: similar family but with mirrored/echo ring motif.
- Line Catalyst: narrow luminous reagent, line/filament visual.

Primary colors:

- Star-white, cyan, gold, pale violet.
- Warding: parchment, gold, blue shield glow.
- Dangerous enhancement: controlled ember or violet edge can show risk.

Pickup sprite rules:

- Current: use inventory icon.
- Future: upgrade materials should have small pulse or sparkle if high-value.

Rarity treatment:

- Upgrade Dust: low glow.
- Catalyst/Core/Prisms: stronger central glow.
- Echo/Attunement items: clean geometric shape and highest clarity.

Avoid:

- Making all upgrade materials crystal piles.
- Excessive particles.
- Confusing Prism Shards with star-card materials.

Prompt template:

```text
Clean professional 2D game inventory icon of [upgrade material], Project Starfall gear enhancement reagent, [crystal/prism/core/scroll/dust] silhouette, star-magic cyan and lantern-gold accents, crisp dark outline, upper-left lighting, centered transparent background, no frame, no text, readable at 64x64.
```

Negative prompt:

```text
text, watermark, UI frame, background, random magic clutter, too many particles, unreadable crystal pile, fake letters, excessive glow, photorealism.
```

File naming:

- `upgrade-dust.png`
- `upgrade-catalyst.png`
- `warding-scroll.png`
- `refinement-core.png`
- `cube-fragment.png`
- `potential-cube.png`
- `preservation-cube.png`
- `line-catalyst.png`

### Currency

Purpose: Immediate reward, shop economy, and readable loot value.

Visual identity:

- Coins should be warm, bright, and readable against dark scenes.
- Stack size should imply value.

Shape language:

- Small: a few coins.
- Medium: clear stack plus loose coin.
- Large: taller stack or pouch.
- Huge: larger pile, chest-like stack, or star-stamped coin bundle.

Primary colors:

- Gold, amber, brass, star-white highlights.

Pickup sprite rules:

- Current: coin icons are reused.
- Future: currency pickups should sparkle lightly with short idle glint.

Rarity treatment:

- Currency value uses stack size and sparkle, not item rarity color.

Avoid:

- Coins that look like cookies, buttons, or generic yellow blobs.
- Text or numbers on coins.
- Large sparkle clouds for low coin values.

Prompt template:

```text
Clean professional 2D game inventory icon of [small/medium/large/huge] coin stack for Project Starfall, warm gold frontier fantasy coins with subtle star stamp shapes, clear stack silhouette, crisp outline, upper-left lighting, centered transparent background, no text, no frame, readable at 64x64.
```

Negative prompt:

```text
numbers, text, watermark, UI frame, background, excessive sparkle, melted coins, photorealism, random gems, unreadable pile.
```

File naming:

- `coins.png`
- `coins-small.png`
- `coins-medium.png`
- `coins-large.png`
- `coins-huge.png`

### Cards and Star Cards

Purpose: Collection, build customization, monster identity, and long-term reward chase.

Visual identity:

- Cards should read as cards first, then as enemy/theme icons.
- Star-card materials should read as star fragments/cards and should not be confused with monster cards.

Shape language:

- Monster cards: upright card silhouette, inner symbol or creature mark, controlled decorative border.
- Star cards: star-shaped or card-like material with color-coded tier.
- Boss cards: stronger border, boss motif, rare material.

Primary colors:

- Common: white.
- Uncommon: green.
- Rare: blue.
- Epic: purple.
- Relic: orange/gold.

Pickup sprite rules:

- Current: cards use runtime item/card icons and tier aura.
- Future: cards can use a small flip or shimmer animation.

Rarity treatment:

- Card rarity is already important and should use the existing rarity colors.
- Do not cover the card face with aura.

Avoid:

- Fake readable text on card faces.
- Cards that look like UI buttons instead of collectible objects.
- Overly detailed creature portraits that vanish at 64x64.

Prompt template:

```text
Clean professional 2D collectible card icon for Project Starfall, [monster or star-card name], upright fantasy card silhouette, simple symbolic [enemy family or rarity] mark, crisp outline, limited decorative border, upper-left lighting, transparent background, no readable text, no watermark, readable at 64x64.
```

Negative prompt:

```text
readable text, letters, numbers, watermark, fake UI button, full background scene, tiny portrait details, excessive border clutter, blurry card.
```

File naming:

- Cards: `img/project-starfall/cards/icons/<card-id>.png`.
- Star-card materials: `img/project-starfall/items/icons/<rarity>-star-card.png`.

### Quest Items and Keys

Purpose: Future quest-critical interaction, gates, dungeon objectives, and story progression.

Current status: implied by systems and GDD, but not a complete current item asset category.

Visual identity:

- Quest items must be unmistakably special but not confused with high-rarity gear.
- Keys should use strong literal silhouettes with biome-specific heads and teeth.
- Lore objects should use readable object forms: sealed letter, guild token, map shard, relic lens, crest, star compass.

Shape language:

- Keys: large key silhouette, themed bow, distinct teeth.
- Quest relics: centered ceremonial object, controlled glow.
- Lore collectibles: parchment, badge, seal, page, small codex.

Primary colors:

- Parchment, gold, cyan star light.
- Use biome colors for regional quest items.

Icon framing:

- Quest items may have a subtle baked star glint, but not a full UI frame.
- UI can add a quest marker overlay separately if needed.

Pickup sprite rules:

- Future quest pickups should pulse slowly and have a larger interaction prompt.

Rarity treatment:

- Quest-critical importance should use a steady cyan/gold pulse, not normal rarity color.

Avoid:

- Making quest keys look like ordinary crafting metal.
- Using text labels.
- Overusing relic glow until every quest item looks like a boss drop.

Prompt template:

```text
Clean professional 2D game inventory icon of [quest item name], quest-critical [key/relic/document/token] for Project Starfall, [biome or story motif], strong readable silhouette, subtle star-magic importance glow, crisp outline, upper-left lighting, centered transparent background, no frame, no text, readable at 64x64.
```

Negative prompt:

```text
text, letters, numbers, watermark, UI frame, background scene, generic key, excessive glow, tiny inscriptions, random ornaments, photorealism.
```

File naming:

- Recommended future icon: `img/project-starfall/items/icons/<quest-item-id-kebab>.png`.
- Recommended future pickup: `img/project-starfall/items/pickups/pickup-quest-<quest-item-id-kebab>.png`.

### Class Items

Purpose: Advanced class identity, specialization, offhand fantasy, and build readability.

Current examples:

- Guardian Tower Shield.
- Berserker War Grip.
- Duelist Parry Medal.
- Ember Core.
- Rune-Etched Focus.
- Storm Charge Focus.
- Deadeye Scope.
- Trap Kit.
- Beast Bond Charm.

Visual identity:

- Each class item should be category-readable and class-readable.
- These should not be simple recolors of generic accessories.

Shape language:

- Guardian: broad shield, blue-gold warding marks.
- Berserker: heavy wrapped grip, iron studs, red leather, aggressive mass.
- Duelist: polished medal, slim blade-parry motif, elegant metal.
- Ember Mage: glowing ember core, basalt metal cradle.
- Rune caster: seal/focus disk, etched cyan runes.
- Storm caster: charged rod/core, white-violet arc marks.
- Deadeye: brass/steel scope, clear lens.
- Trap specialist: compact kit, clamp/trap teeth, leather roll.
- Beast class: carved charm/token, claw or paw motif, warm leather.

Rarity treatment:

- Class items should communicate specialization through shape before glow.
- Higher rarity adds trim and magic, not a different class read.

Avoid:

- Shared icons for final class-defining items.
- Generic amulets for all class accessories.
- Tiny emblems that do not read at 64x64.

Prompt template:

```text
Clean professional 2D game inventory icon of [class item name], class-specific [offhand/accessory] for [class name] in Project Starfall, strong functional silhouette, [class motif], crisp outline, upper-left lighting, clean cel-shaded materials, transparent background, no frame, no text, readable at 64x64.
```

Negative prompt:

```text
generic charm, unreadable emblem, text, watermark, UI frame, background, excessive glow, duplicate accessories, random ornament, blurred lens or rune.
```

File naming:

- `img/project-starfall/items/icons/<class-item-id-kebab>.png`.

### Tools and Service Items

Purpose: Account systems, utility, pet interaction, Plinko, admin/testing, slots, and meta progression.

Visual identity:

- These items should feel like magical guild tools, not modern app icons.
- They can be more symbolic but still must be objects.

Examples:

- Pet Whistle: whistle with paw/star charm.
- Slot Coupons: ticket/card silhouettes with category color and icon mark, no text.
- XP/drop coupons: foil tickets, no letters or numbers.
- Plinko Balls: tiered orbs with different materials.
- Admin Worldwright Console: magical slate/device, not a laptop or phone.

Shape language:

- Tool object first.
- Category icon second.
- Magic effect third.

Avoid:

- Modern UI symbols detached from the world.
- Readable text on tickets.
- Neon sci-fi devices.

Prompt template:

```text
Clean professional 2D game inventory icon of [tool/service item], magical guild utility object for Project Starfall, [function motif], readable object silhouette, crisp outline, upper-left lighting, transparent background, no readable text, no frame, no watermark, readable at 64x64.
```

Negative prompt:

```text
modern phone, laptop, app icon, text, letters, numbers, watermark, fake UI, background scene, excessive glow, random icons.
```

File naming:

- `img/project-starfall/items/icons/<tool-id-kebab>.png`.

### Boss and Relic Drops

Purpose: Peak reward identity, boss memory, long-term progression, and visual excitement.

Visual identity:

- Boss drops must be unmistakably tied to their boss family.
- Relic drops must have unique silhouette and lore identity.
- Boss icons should look better and more specific than shop or random world gear.

Boss motif examples:

- Brambleking: crown thorns, bark plates, roots, green/gold poison-nature accents.
- Emberjaw Golem: furnace core, jaw shape, basalt, molten cracks.
- Clockwork Titan: brass gear teeth, teal energy, clockwork plates.
- Quarry Colossus: stone slabs, ore veins, geode cores.
- Stormbreak Roc: feathers, talons, cloud arcs, lightning.
- Astral Archivist: pages, indexes, star runes, archive metal.
- Eclipse Sovereign: black-violet plate, corona gold, void magenta, cold white.

Rarity treatment:

- Epic boss gear: strong boss motif plus controlled purple aura.
- Relic boss gear: iconic silhouette plus gold/star aura.
- Do not depend on aura alone. The boss should still be recognizable if the aura is removed.

Avoid:

- Generic legendary sword with unrelated ornaments.
- Overcrowding with all motifs at once.
- Effects that obscure the item shape.

Prompt template:

```text
Clean professional 2D game inventory icon of [boss drop name], [weapon/armor/accessory] from [boss name] in Project Starfall, unmistakable [boss motif], Epic or Relic reward identity, strong unique silhouette, rich but readable material planes, controlled magical glow, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

Negative prompt:

```text
generic legendary item, text, watermark, UI frame, background, excessive particles, glow covering silhouette, random filigree, too many motifs, broken anatomy, photorealism, 3D render.
```

File naming:

- `img/project-starfall/items/icons/<boss-drop-id-kebab>.png`.

## 5. Theme and Motif System

### Element Motifs

| Theme | Shape language | Palette | Materials | Glow/VFX | Avoid |
| --- | --- | --- | --- | --- | --- |
| Fire/Cinder | Jagged cracks, furnace vents, ember cores | Basalt, ember orange, smoke gray, molten red | Black stone, hot metal, ash, magma | Inner crack glow, small ember flecks | Full flame clouds on every item |
| Frost/Ice | Sharp facets, snow rims, crystalline points | Ice blue, silver, pine, cyan, white | Ice crystal, pale metal, frozen hide | Cold edge glow, small frost glint | Making all frost items pale blue blobs |
| Storm/Lightning | Forks, feathers, cloud arcs, charged rods | Blue-gray, violet, white, yellow-white | Feather, silver, cloud silk, charged metal | Thin lightning accents, small arcs | Thick zigzags that hide shape |
| Earth/Quarry | Heavy blocks, geode cores, ore veins | Stone gray, dusty brown, ore blue, muted gold | Stone, ore, crystal, mineral plate | Subtle geode glow | Generic rocks with no item identity |
| Forest/Nature | Roots, thorns, leaves, bark, moss | Soft green, bark brown, yellow light | Wood, hide, vine, moss, horn | Soft green/gold life glow | Overly cute leaf icons |
| Astral/Arcane | Stars, runes, orbit rings, pages | Indigo, cyan, parchment, pale violet | Crystal, parchment, silver, star metal | Star-white/cyan core glow | Generic purple magic clutter |
| Eclipse/Rift | Crescent, corona, broken star, shard | Black-violet, corona gold, void magenta, cold white | Dark metal, void crystal, silk, crown metal | Gold rim plus magenta void core | Making everything black and unreadable |
| Light/Warding | Shields, halos, seal circles | Gold, star-white, pale cyan | Parchment, polished metal, crystal | Clean protective aura | Religious text or unreadable sigils |
| Poison/Corruption | Thorns, glands, ooze beads, sickly veins | Green, yellow-green, dark violet | Gland, thorn, slime, cracked crystal | Low toxic rim glow | Gross detail that clashes with friendly tone |
| Water/Dew/Ooze | Droplets, beads, soft curves | Cyan, teal, wet green, star-white | Gel, dew, polished shell | Tiny wet highlight | Featureless blue orb |

### Biome and Enemy Family Motifs

#### Greenroot, Thornpath, and Bramble

- Use bark, moss, vine fiber, thorn hooks, antlers, bramble crowns, and soft green/gold light.
- Gear should look grown or wrapped, not industrial.
- Materials can have organic asymmetry but must keep clean silhouettes.
- Boss Brambleking items need crown-thorn silhouettes and root mass.

#### Rustcoil and Construct

- Use brass, gear teeth, coils, clock plates, teal oxidation, rivets, and oil-black shadow.
- Shapes should be mechanical and precise.
- Avoid sci-fi neon panels. This is magical clockwork, not cyberpunk.

#### Cinder and Emberjaw

- Use basalt, furnace jaws, molten cracks, ash carapaces, ember dust, and red-orange inner light.
- Boss Emberjaw gear should include jaw, furnace, or molten core shapes.
- Avoid huge flame plumes on small icons.

#### Frostfen and Rimewarden

- Use ice facets, frozen hide, snow dust, rime shards, pine-green shadows, and pale silver.
- Rimewarden items should feel ceremonial and frozen.
- Avoid low-contrast white-on-transparent silhouettes.

#### Stormbreak

- Use feathers, talons, sky bows, cloud silk, lightning grips, and blue-gray/violet storms.
- Stormbreak Roc gear should carry bird-of-prey silhouette elements.
- Avoid random lightning bolts that do not connect to the object.

#### Astral Archive

- Use codex pages, runic index tabs, star maps, orbit boots, comet strings, and cyan crystal.
- Astral Archivist items should look scholarly and magical, not just purple.
- Avoid readable text.

#### Eclipse and Rift

- Use corona crowns, black-violet metal, void silk, rift splinters, eclipse plates, and gold rim light.
- Eclipse Sovereign items should be regal, sharp, and high contrast.
- Avoid making silhouettes disappear into black fill.

#### Quarry and Deepcore

- Use heavy stone, ore, geodes, colossus plates, deepcore helms, and bedrock boots.
- Quarry gear should feel massive and grounded.
- Avoid generic gray lumps.

#### Bandit and Frontier

- Use cloth wraps, knives, rope, worn leather, iron scraps, and improvised metal.
- Bandit materials should feel practical and looted.
- Avoid making bandit gear look like premium boss gear.

### Class Motifs

| Class family | Motifs | Materials | Colors | Notes |
| --- | --- | --- | --- | --- |
| Fighter | Blades, shields, grips, plate, medals | Steel, iron, leather, gold trim | Red, gold, steel, navy | Strongest silhouettes and weight |
| Mage | Staffs, wands, cores, seals, books | Crystal, parchment, silver, rune metal | Cyan, violet, indigo, star-white | Clear magical focus shapes |
| Archer | Bows, scopes, traps, quivers, feathers | Wood, leather, brass, fletching | Green, teal, sky, tan | Long arc silhouettes and precision tools |
| Guardian | Tower shield, ward seal | Steel, gold, blue crystal | Blue, gold, silver | Defensive, broad, stable |
| Berserker | War grip, heavy axe/cleaver | Iron, red leather, dark steel | Red, iron, ember | Aggressive but clean |
| Duelist | Medal, slim blades, polished trim | Silver, leather, gold | White, gold, navy | Elegant and precise |
| Ember caster | Core, furnace, flame crystal | Basalt, ember, brass | Orange, black, gold | Contained heat |
| Rune caster | Focus, seal, etched disk | Silver, crystal, parchment | Cyan, violet, indigo | Runes as shapes, not text |
| Storm caster | Charge core, rod, arcs | Silver, cloud silk, crystal | Blue-gray, violet, white | Thin controlled lightning |
| Deadeye | Scope, longbow, crossbow | Brass, steel, leather | Green, brass, teal | Precision silhouette |
| Trapper | Trap kit, clamps, rope | Iron, leather, wood | Brown, steel, moss | Compact utility |
| Beast class | Token, claw, charm | Bone, hide, leather, carved wood | Warm brown, green, gold | Natural companion motif |

### Origin Motifs

Natural:

- Organic curves.
- Bark, hide, fiber, feather, horn.
- Lower polish, irregular but clean silhouette.

Crafted:

- Symmetric structure.
- Metal, leather, cloth, rivets, tool marks.
- Clear functional construction.

Magical:

- Crystal, runes, star cores, orbit rings.
- Glow contained inside object or on edge.

Corrupted:

- Broken symmetry.
- Void cracks, magenta seams, black-violet material.
- Must remain readable and not become visual noise.

Ancient:

- Weathered alloys, stone, ceremonial geometry, precise inlays.
- Rich but not over-detailed.

Starforged:

- Midnight steel, lantern gold, crystal cyan, star-white.
- Cleanest high-tier craftsmanship.

## 6. Rarity and Importance Readability

### Implemented Rarity Rules

Rarity should enhance the item, not replace item design.

Common:

- Plain silhouette.
- Muted material.
- Minimal trim.
- No glow or very faint neutral glint.
- Runtime aura color: `#d8e5ec`.

Uncommon:

- Slightly cleaner shape.
- One small accent material.
- Mild green or natural magic accent.
- Runtime aura color: `#74d680`.

Rare:

- Clear upgraded silhouette.
- Better material and one magical detail.
- Controlled blue accent or gem.
- Runtime aura color: `#68a9ff`.

Epic:

- Strong silhouette change.
- Rich material, rune channel, or visible elemental feature.
- Controlled purple aura and pulse at runtime.
- Runtime aura color: `#c794ff`.

Relic:

- Unique silhouette.
- Lore or boss identity visible from outline.
- Gold/star aura.
- Strong material hierarchy.
- Runtime aura color: `#ffbe55`.

Mythic:

- GDD-only/future item rarity.
- Do not ship Mythic item icons until runtime item rarity support is added.
- Future rule: iconic silhouette, restrained animated accent, starforged or celestial material.

### Importance Types

Quest-critical:

- Use cyan/gold steady pulse or quest UI overlay.
- Do not use normal rarity aura as the only signal.
- Shape should be key, relic, seal, document, map, or token.

Boss drops:

- Use unique boss silhouette plus Epic or Relic aura.
- Add small landing burst and pickup notification.
- Avoid generic loot glow.

Class-specific items:

- Category and class motif must read before rarity.
- Use class material and silhouette markers.

Upgrade materials:

- Use geometric magic/reagent shapes.
- Importance comes from core glow and clean form.
- Do not make Upgrade Dust look as important as Attunement Prism.

Currency:

- Importance comes from stack size and sparkle, not rarity color.

Consumables:

- Importance comes from bottle shape, cap material, liquid brightness, and tier size.
- Avoid heavy rarity aura.

### Rarity Visual Controls

Use these controls in order:

1. Silhouette complexity.
2. Material richness.
3. Accent material.
4. Controlled glow.
5. Runtime aura.
6. Animation or VFX, only for special drops.

Do not:

- Apply maximum glow to every item.
- Recolor a Common item and call it Epic.
- Let rarity color hide biome, class, or category color.
- Use particles for all rarities.
- Add more spikes as the only rarity progression.

## 7. Item Uniqueness Framework

Every item must define these identity layers:

1. Core silhouette: the object family and exact outline.
2. Category marker: weapon, armor, potion, scroll, material, coin, card, key, tool, or relic.
3. Theme marker: biome, element, enemy family, class, tier, or origin.
4. Rarity marker: material richness, trim, aura, or glow.
5. Functional marker: heals, restores resource, upgrades, protects, unlocks, buffs, summons, collects, equips, crafts, or sells.
6. Material identity: leather, brass, steel, crystal, bark, ember, frost, parchment, void silk, etc.
7. Color identity: main color and 1 to 2 accent colors.
8. Icon framing: usually transparent and unframed.
9. Pickup behavior: normal, aura, pulse, sparkle, burst, or boss reveal.
10. Differentiator: what makes it different from the closest similar item.
11. Consistency anchor: what makes it belong to Project Starfall.

### Reusable Item Visual Identity Template

```markdown
## Item Visual Identity: [Item Name]

- Data ID:
- Runtime icon file:
- Category:
- Subcategory:
- Gameplay purpose:
- Source or drop context:
- Class restriction, if any:
- Biome/theme:
- Element:
- Rarity/importance:
- Core silhouette:
- Category marker:
- Theme marker:
- Rarity marker:
- Functional marker:
- Primary material:
- Secondary material:
- Main colors:
- Accent colors:
- Glow/VFX:
- Inventory icon notes:
- World pickup notes:
- Similar items to compare against:
- Required uniqueness difference:
- Consistency anchor with Project Starfall:
- Rejection risks:
```

### Uniqueness Rules

Avoid ten items that look like minor recolors:

- Change silhouette, not only color.
- Change material construction by tier.
- Change focal feature by function.
- Change boss motif by source.

Avoid over-detailed items:

- If a detail disappears at 64x64, remove or enlarge it.
- Keep one focal motif.
- Use fewer, larger shapes.

Avoid different-game drift:

- Keep crisp outline.
- Keep upper-left light.
- Keep frontier fantasy materials.
- Avoid photorealism, modern sci-fi, grimdark horror, and generic mobile-game neon.

Avoid glow abuse:

- Glow only where magic or importance justifies it.
- Keep object edges visible through glow.
- Use runtime aura for rarity.

Avoid inconsistent perspective:

- Use a three-quarter or slight top-down item presentation consistently.
- Long weapons can be diagonal, but do not switch to full side-view for one batch unless needed.

## 8. Inventory Icon Guidance

### Runtime Requirements

- Size: 64x64 PNG.
- Background: transparent.
- Object scale: usually 46px to 56px on longest side.
- Safe padding: 4px to 8px at final 64px size.
- Corners: keep important details away from corners.
- No baked UI frame.
- No text.
- No watermark.
- No scene background.

### UI Compatibility

Inventory icons must work in:

- Equipment tab.
- Usable tab.
- Etc tab.
- Cards tab.
- Hotbar/action UI.
- Shop lists.
- Crafting/upgrade station.
- Reward notifications.
- Tooltips.
- Canvas-rendered inventory views.
- World loot render.

### Stack Count and Overlay Rules

- Do not put essential detail at bottom-right where stack counts may appear.
- Do not put essential detail at top-left where level badges may appear.
- Do not put essential detail at bottom-left where locks may appear.
- Keep the center 60% of the icon as the main read.
- Disabled/grayscale states must still preserve silhouette.
- Hover/selected states are UI effects and should not require alternate item art.

### Equipment Icons

- Use strong slot silhouette.
- Put item type before decoration.
- Use tier material and class motif.
- Do not include character body parts.
- Do not include equipment-layer pose.

### Consumable Icons

- Use function color and container shape.
- Tier by silhouette and cap material.
- Avoid tiny labels.
- Potions should be recognizable when grayscale by bottle silhouette.

### Quest Item Icons

- Use unique object silhouette.
- Add subtle importance glow.
- Avoid normal rarity confusion.
- Use tooltip and UI marker for quest status if implemented.

### Crafting Material Icons

- Use large single material shape or small cluster with one dominant piece.
- Tie the shape to enemy family or biome.
- Avoid generic piles unless the item is explicitly dust, scraps, or chunks.

### Currency Icons

- Use stack size and coin count, not text.
- Bright but controlled highlights.
- Keep outline strong against dark gameplay.

### Skill/Class Item Icons

- If it is an item, use item icon rules.
- If it is a skill icon, keep it under `img/project-starfall/skills/` and preserve the 256x256 skill icon contract.
- Class items should have class-specific silhouette markers.

### Rare and Boss Item Icons

- Must pass silhouette-only recognition.
- Must show boss or source motif.
- Use richer materials and controlled glow.
- Do not rely on the renderer aura to make the item special.

### Small-Size Readability Test

Before import, view each icon at:

- 64x64.
- 48x48.
- 34x34.
- Grayscale 64x64.
- Against parchment/gold UI.
- Against dark navy gameplay background.
- Next to three similar items.

Reject the icon if the object category, source theme, or function is unclear.

## 9. World Pickup Sprite Guidance

### Current Runtime Behavior

Project Starfall currently draws world pickups from item icons:

- Tier aura item draw box: 44px.
- Tier aura icon inner size: about 34px.
- Non-tier item draw box: 58px.
- Non-tier icon inner size: about 50px.
- Settled bob motion: about 5px vertical.
- Shadow ellipse appears under the item.
- Rare equipment/cards receive aura rings and glow.
- Pickup effect flies the item toward the player or UI target, shrinks, fades, and may draw an aura ring.

### Current Design Implication

Inventory icons must be pickup-ready. If an icon only reads at 64x64 and fails at 34px, it is not acceptable for equipment, cards, or other tier-aura loot.

### Future Dedicated Pickup Sprite Rules

Only add dedicated pickup sprites after item data and renderers support a `pickupAsset` or equivalent reference.

Recommended dimensions:

- Static pickup sprite: 64x64 transparent PNG.
- High-value pickup sprite: 96x96 source reduced or scaled by renderer.
- Animated pickup: 4 to 8 frames, equal cell size, transparent PNG sheet.

Visual rules:

- Larger, simpler silhouette than inventory icon.
- Stronger outline.
- Fewer interior details.
- Drop shadow controlled by renderer when possible.
- Low-value pickups use minimal glow.
- High-value pickups use contained aura, pulse, or sparkle.
- Quest pickups use steady cyan/gold pulse.
- Currency pickups use brief sparkle.
- Boss pickups use reveal burst and aura.

### Recommended Pickup Animations

| Pickup type | Frames | Motion | VFX |
| --- | --- | --- | --- |
| Common pickup idle | 4 | 2px to 4px bob | None or tiny glint |
| Rare pickup glow | 6 | 4px bob | Soft aura pulse |
| Quest item pulse | 8 | Slow hover | Cyan/gold pulse ring |
| Currency sparkle | 4 | Quick glint | Small star sparkle |
| Health/mana pickup idle | 4 | Soft bob | Liquid shine |
| Equipment drop reveal | 8 | Drop squash and settle | Rarity ring flash |
| Boss item drop | 10 to 12 | Slow reveal | Landing burst, aura ring, sparkle |

### Visibility by Biome

- Against forest: avoid dark green-only icons.
- Against cinder: avoid red/orange-only icons without outline.
- Against frost: avoid pale blue/white-only icons without dark contour.
- Against storm: avoid gray/violet-only icons without star-white highlight.
- Against astral/eclipse: use gold/cyan edge highlights for dark items.

## 10. Item Impact and Reward Feel

Important items should feel satisfying without cluttering the screen.

### Drop Anticipation

- Rare or boss drops can delay visibility briefly with a small glow at the landing point.
- Do not delay common materials.
- Boss drops may use a short reveal pulse after the enemy defeat effect.

### Drop Animation

- Common drops: normal scatter arc and settle.
- Rare equipment/cards: scatter arc plus aura ring.
- Boss drops: higher arc, slower settle, stronger landing ring.
- Quest items: controlled drop or spawn, not random scatter unless story-appropriate.

### Landing Effect

- Common: shadow and settle only.
- Uncommon/Rare: small glint.
- Epic: ring flash and pulse.
- Relic/Boss: ring flash, brief star burst, controlled particles.

### Pickup Burst

- Common: quick fly and fade.
- Currency: small sparkle.
- Consumables: soft color puff matching function.
- Upgrade materials: small star-prism burst.
- Equipment: rarity ring trail.
- Boss item: short pause or notification emphasis if the UI supports it.

### UI Notification

- Use item icon plus name, rarity, and category.
- Do not show a giant duplicated icon for common materials.
- Boss and quest items may get larger reveal treatment.
- Tooltip reveal should show category, rarity, source, and comparison.

### Sound and VFX Direction

- Common: light tick or soft pickup.
- Currency: bright coin chime.
- Consumable: bottle/cloth magic puff.
- Upgrade material: crystalline chime.
- Equipment: metal/star resonance.
- Boss drop: layered chime plus low impact.

### Avoid Cheap Excess

- Do not fill the screen with particles for common rewards.
- Do not use large glow blobs that obscure nearby enemies.
- Do not make every drop pulse at the same intensity.
- Let rarity hierarchy create contrast.

## 11. Prompt Engineering Rules for Item Assets

### Locked Traits Across All Prompts

Always preserve:

- Project Starfall Starlit Frontier Fantasy.
- Clean professional 2D game asset style.
- Strong readable silhouette.
- Crisp dark contour outline.
- Upper-left lighting.
- Simple cel-shaded material planes.
- Transparent final background.
- No text.
- No watermark.
- No fake UI.
- No background scene.
- No clutter.
- No AI-looking artifacts.
- Readable at 64x64.

Variables that may change:

- Item category.
- Object subtype.
- Biome or enemy family.
- Element.
- Rarity.
- Tier material.
- Class identity.
- Functional motif.
- Main and accent colors.
- Glow intensity.
- Silhouette complexity.

### Master Item Art Prompt

```text
Clean professional 2D game inventory icon for Project Starfall, Starlit Frontier Fantasy, [item name], [item category and subtype], [gameplay function], [biome/class/element/theme], [rarity or tier], strong readable silhouette, crisp dark contour outline, upper-left lighting, clean cel-shaded material planes, controlled highlights, centered isolated object, transparent background, no frame, no text, no watermark, no background scene, readable at 64x64.
```

### Master Negative Prompt

```text
text, letters, numbers, watermark, logo, fake UI, inventory frame, background scene, scenery, character holding item, hands, cropped object, blurry outline, over-rendered texture, excessive glow, excessive particles, random ornamentation, duplicated parts, melted geometry, inconsistent perspective, photorealism, 3D render, noisy AI artifacts, chroma residue.
```

### Inventory Icon Prompt Template

```text
Create a single centered transparent-background 2D inventory icon for Project Starfall: [item name], a [category/subtype]. It should communicate [function] and [theme] using [shape motif] and [materials]. Use Starlit Frontier Fantasy style, crisp contour outline, upper-left lighting, clean cel shading, limited readable details, no frame, no text, no watermark, no background, readable at 64x64.
```

### World Pickup Sprite Prompt Template

```text
Create a simplified 2D world pickup sprite for Project Starfall: [item name], [category], transparent background, stronger silhouette than inventory icon, crisp outline, fewer interior details, readable at 34px to 50px, [optional rarity glow or quest pulse], no text, no frame, no background scene, no watermark.
```

### Animated Pickup Sprite Sheet Prompt

```text
Create a [frame count]-frame 2D sprite sheet for Project Starfall showing [item name] as a world pickup idle animation, equal-size cells, transparent or chroma-key background, cyan #00ffff guide lines between cells if using source-sheet processing, consistent scale and center point in every frame, subtle [bob/glow/sparkle] motion, no text, no UI, no watermark, no background scene.
```

### Weapon Icon Prompt

```text
Clean professional 2D game icon of [weapon name], [weapon type], [tier material], [class or biome motif], Project Starfall Starlit Frontier Fantasy, strong diagonal readable silhouette, crisp outline, upper-left lighting, controlled [element] accent, transparent background, no frame, no text, readable at 64x64.
```

### Armor Icon Prompt

```text
Clean professional 2D game icon of [armor name], [slot], [material tier], [biome or boss motif], Project Starfall Starlit Frontier Fantasy, clear slot-readable silhouette, crisp dark outline, upper-left lighting, clean material planes, transparent background, no character mannequin, no frame, no text, readable at 64x64.
```

### Consumable Prompt

```text
Clean professional 2D game icon of [consumable name], [bottle/scroll/ration/oil/coupon/manual/tool], communicates [effect], Project Starfall Starlit Frontier Fantasy, readable object silhouette, [function color], crisp outline, upper-left lighting, transparent background, no readable text, no frame, no watermark, readable at 64x64.
```

### Crafting Material Prompt

```text
Clean professional 2D game icon of [material name], [material type] from [enemy family/biome], Project Starfall Starlit Frontier Fantasy, strong simple silhouette, [theme motif], [material colors], crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Quest Item Prompt

```text
Clean professional 2D game icon of [quest item name], quest-critical [key/relic/document/token], Project Starfall Starlit Frontier Fantasy, [story or biome motif], unique strong silhouette, subtle cyan and gold importance glow, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Currency Prompt

```text
Clean professional 2D game icon of [value tier] coin stack for Project Starfall, warm gold fantasy coins, clear stack silhouette, subtle star-stamped shapes without readable text, crisp outline, upper-left lighting, transparent background, no frame, no numbers, readable at 64x64.
```

### Relic Prompt

```text
Clean professional 2D game icon of [relic name], unique Relic-tier [category], Project Starfall Starlit Frontier Fantasy, iconic lore silhouette, [origin theme], rich materials, controlled gold/star glow, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Upgrade Material Prompt

```text
Clean professional 2D game icon of [upgrade material], Project Starfall gear enhancement reagent, [prism/core/dust/scroll/catalyst] silhouette, star-white, crystal cyan, and lantern-gold accents, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Class-Specific Item Prompt

```text
Clean professional 2D game icon of [class item name], class-specific [offhand/accessory/tool] for [class], Project Starfall Starlit Frontier Fantasy, strong functional silhouette, clear [class motif], [materials], crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Elemental Variant Prompt

```text
Clean professional 2D game icon of [base item] with [element] attunement, Project Starfall Starlit Frontier Fantasy, keep the original [category] silhouette recognizable, add [element motif] through material accents and controlled glow, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Boss Drop Prompt

```text
Clean professional 2D game icon of [boss drop name], [category] dropped by [boss name], Project Starfall Starlit Frontier Fantasy, unmistakable [boss motif], Epic or Relic reward identity, unique strong silhouette, rich readable materials, controlled magical glow, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

### Rarity Variant Prompt

```text
Create a [rarity] version of [item name] for Project Starfall. Keep the base [category] silhouette recognizable, increase rarity through [material richness/trim/glow/silhouette feature], preserve [theme motif], no recolor-only change, crisp outline, upper-left lighting, transparent background, no frame, no text, readable at 64x64.
```

## 12. Item Asset Generation Workflow

1. Identify gameplay purpose.
   - Is the item consumed, equipped, crafted, upgraded, collected, sold, opened, slotted, or used for a quest?

2. Assign item category.
   - Use current UI categories when possible: Equipment, Usable, Etc, Cards.
   - If adding a new category, update UI and tests before relying on it.

3. Assign theme or motif.
   - Choose biome, class, enemy family, element, tier, origin, or boss.
   - Use the motif system in this guide.

4. Assign rarity or importance.
   - Use implemented rarities only: Common, Uncommon, Rare, Epic, Relic.
   - Treat Mythic as future until runtime support is added.

5. Define silhouette.
   - Sketch the item as a black shape.
   - Confirm it reads at 64px and 34px.

6. Define color palette.
   - Pick 1 main color, 1 material color, and 1 to 2 accents.
   - Keep rarity color secondary.

7. Generate inventory icon.
   - Use the correct prompt template.
   - For source sheets, maintain cyan guide lines and exact cell order.
   - Avoid baked frames, labels, and backgrounds.

8. Generate world pickup sprite, if needed.
   - Current implementation usually does not need separate pickup art.
   - Add renderer/data support before adding `pickupAsset`-style art.

9. Generate animation or VFX, if needed.
   - Only for quest, boss, Relic, currency sparkle, or special pickup effects.
   - Keep frame count modest.

10. Review readability at small size.
    - Test 64x64, 48x48, 34x34, grayscale, parchment UI, and dark gameplay.

11. Compare against existing item categories.
    - Check similar icons in `img/project-starfall/items/icons/`.
    - Confirm the new item is not a recolor of a neighbor.

12. Check consistency with the GDD.
    - Confirm Starlit Frontier Fantasy tone.
    - Confirm tier, biome, class, and rarity rules.

13. Export correctly.
    - Runtime item icon: 64x64 PNG with alpha.
    - Transparent corners.
    - No chroma residue.
    - Correct kebab-case file name.

14. Place in correct folder.
    - Runtime icon: `img/project-starfall/items/icons/`.
    - Source sheet: `img/project-starfall/items/source/`.
    - Processed sheet: `img/project-starfall/items/sheets/`.

15. Connect to item data.
    - Add or confirm item in data file.
    - Add item to `ITEM_ASSETS`.
    - If source-sheet generated, add to `build/process-project-starfall-ai-item-icons.js`.
    - Add to shop, drop table, reward table, stack cap, or UI list as needed.

16. Run processing and checks.
    - Run `node build/process-project-starfall-ai-item-icons.js` after source-sheet changes.
    - Run `npm test`.
    - Run `npm run build` before deployment.

17. Test in context.
    - Inventory.
    - Tooltip.
    - Shop or reward list.
    - World drop.
    - Pickup effect.
    - Dark and bright biome backgrounds.

### Rejection Criteria

Reject generated assets if:

- The category is unclear at 64x64.
- The category is unclear at world drop size.
- The silhouette is weak or muddy.
- The theme does not match the item data or GDD.
- The item looks like a generic fantasy asset outside Project Starfall.
- It uses text, labels, watermark, or fake UI.
- It has background art.
- It has cropped parts.
- It has noisy AI detail.
- It uses glow to hide a weak shape.
- It is a recolor-only version of a similar item.
- It violates source sheet grid or naming order.
- It leaves chroma-key residue after processing.

## 13. Asset Organization and Naming Conventions

### Current Required Structure

```text
img/project-starfall/items/icons/<item-id-kebab>.png
img/project-starfall/items/source/ai-items-<batch-name>.png
img/project-starfall/items/sheets/ai-items-<batch-name>-sheet.png
img/project-starfall/cards/icons/<card-id>.png
img/project-starfall/cards/source/ai-card-icons.png
img/project-starfall/equipment-layers/<visual-id-kebab>-sheet.png
asset-sources/project-starfall/players/equipment/<visual-id-kebab>-source.png
```

### Runtime Item Icon Naming

Use current kebab-case runtime naming:

```text
<item-id-kebab>.png
```

Examples:

```text
minor-health-potion.png
upgrade-dust.png
thorncrown-greatsword.png
stormbreak-plume.png
potential-cube.png
preservation-cube.png
```

Do not use final runtime names like `icon_weapon_copper_sword_rare.png` unless the codebase is changed. The current runtime expects direct kebab item IDs through `ITEM_ASSETS`.

### Recommended Future Source File Names

Use descriptive batch names:

```text
ai-items-consumables-materials.png
ai-items-potion-tiers.png
ai-items-mob-materials-core.png
ai-items-mob-materials-late.png
ai-items-shop-boss-forest.png
ai-items-world-drops.png
ai-items-boss-core-storm.png
ai-items-boss-astral-eclipse.png
ai-items-quest-keys.png
ai-items-future-mythic.png
```

### Recommended Future Pickup Names

Only after code support exists:

```text
img/project-starfall/items/pickups/pickup-consumable-<item-id-kebab>.png
img/project-starfall/items/pickups/pickup-material-<item-id-kebab>.png
img/project-starfall/items/pickups/pickup-equipment-<item-id-kebab>.png
img/project-starfall/items/pickups/pickup-quest-<item-id-kebab>.png
img/project-starfall/items/pickups/pickup-boss-<item-id-kebab>-idle-sheet.png
```

### Recommended Future VFX Names

```text
img/project-starfall/animations/item-vfx/vfx-item-<item-id-kebab>-pickup-burst.png
img/project-starfall/animations/item-vfx/vfx-rarity-common-drop.png
img/project-starfall/animations/item-vfx/vfx-rarity-uncommon-drop.png
img/project-starfall/animations/item-vfx/vfx-rarity-rare-drop.png
img/project-starfall/animations/item-vfx/vfx-rarity-epic-drop.png
img/project-starfall/animations/item-vfx/vfx-rarity-relic-drop.png
img/project-starfall/ui/item-overlays/frame-rarity-<tier>.png
```

### Data Naming Rules

- Keep item IDs stable once saved data may reference them.
- Use snake_case for new item IDs.
- Runtime icon files should be kebab-case generated from the item ID.
- If a material definition uses camelCase, provide an explicit asset mapping to avoid ambiguity.
- Do not rename an existing asset without updating `ITEM_ASSETS`, drop tables, shop data, tests, and any saved-data migration needs.

## 14. Technical Implementation Guidance

### Current Framework

Project Starfall is a static no-framework browser prototype. It uses:

- Plain JavaScript data modules.
- Canvas 2D and Pixi-style rendering paths.
- CSS for inventory/menu presentation.
- LocalStorage and server-backed systems for some account/tool state.
- PNG assets under `img/project-starfall/`.

### Connecting Item Visuals to Data

For new item icons:

1. Add the item definition in the correct data file:
   - Consumables: `js/games/project-starfall/data/consumables.js`.
   - Materials and shared item assets: `js/games/project-starfall/data/items.js`.
   - Equipment: `js/games/project-starfall/data/equipment-catalog.js`.
   - Cards: `js/games/project-starfall/data/cards.js`.
   - Drops: `js/games/project-starfall/data/monster-drops.js`.

2. Add or confirm the asset in `ITEM_ASSETS`.

3. Use `assetId` or `visualId` only when intentionally sharing visuals.

4. For equipment overlays, add or confirm `EQUIPMENT_VISUALS` and the sheet path in `equipment-visuals.js`.

5. For loot drops, confirm `getItemAsset(item)` can resolve the asset through:
   - `currency`.
   - `id`.
   - `itemId`.
   - `consumableId`.
   - `materialId`.
   - `assetId`.
   - `visualId`.
   - Equipment visual fallback.

6. Add to source sheet definitions in `build/process-project-starfall-ai-item-icons.js` if generated from a sheet.

7. Run the processor to create/update 64x64 runtime PNGs.

8. Run tests.

### Rarity References

- Use `ITEM_RARITY_VISUALS` for current rarity aura values.
- Do not bake full rarity frames into icons.
- If adding a new rarity, update:
  - `ITEM_RARITY_VISUALS`.
  - CSS rarity classes in `item-theme.css`.
  - Renderer aura logic.
  - Inventory sorting and filtering if applicable.
  - Tests.
  - Tooltips and comparison styling.

### Category References

Current inventory tab categories:

- `equipment`.
- `usable`.
- `etc`.
- `cards`.

If adding quest keys or lore collectibles, decide whether they belong under `etc` or require a new tab. Do not add a new visual category without checking inventory UI layout.

### Theme Tags

Use item ID, source, boss set, drop pool, card tags, and equipment visual theme inference consistently:

- Forest/nature: thorn, briar, bark, root, moss.
- Cinder: ember, magma, cinder, ashen, furnace, lava, scorch.
- Construct: gear, chrono, ratchet, titan, clock, gyro, spring.
- Quarry: colossus, geode, ore, deepcore, bedrock, quarry, stone.
- Storm: storm, cloud, sky, roc, tempest, lightning, gale.
- Astral: astral, star, comet, archive, scribe, orbit, index.
- Eclipse: eclipse, umbral, corona, sovereign, penumbra, sunfall.
- Martial: vanguard, bulwark, breaker, sentinel.
- Ranger: ranger, pathfinder, deadeye, windrunner.

### Import Settings

Use PNG for item icons:

- Preserve alpha.
- Avoid JPEG.
- Avoid lossy compression for icons.
- Keep edges crisp.
- Avoid chroma-key residue.
- Transparent corners should remain transparent.

For source sheets:

- Keep guide-line color exactly `#00ffff`.
- Keep chroma-key backgrounds clean.
- Do not place art on guide lines.
- Do not use `#00ff00` or `#00ffff` as intentional item colors in source art.

### Pixel-Perfect Handling

The current item art is not strict pixel art. Treat it as clean cel-shaded 2D art:

- Do not generate low-resolution pixel-art icons unless the whole UI direction changes.
- Keep downscaled edges crisp.
- Avoid excessive blur.
- Test scaling in inventory and world loot.

### Save/Load References

- Save data should reference item IDs, not file names.
- Do not change item IDs casually.
- If a visual file changes but item ID remains stable, saved data should continue to work.

### Tooltip, Inventory, Hotbar, Shop, and Loot Scaling

- Tooltip icons should use the same runtime asset.
- Inventory grid may scale icons inside tile wrappers.
- Hotbar icons need the same corner-safe read.
- Shop lists should show the same icon as inventory.
- Loot rendering uses icon assets with runtime aura and shadow.
- Do not create alternate icon sizes unless the renderer supports them.

## 15. Item Review Checklist

Use this checklist before importing or accepting every item visual.

### Identity

- Does the item read clearly at 64x64?
- Does it still read at 48x48?
- Does it still read at 34x34?
- Does the silhouette communicate the category?
- Is it visually distinct from similar items?
- Does the theme read immediately?
- Does the function read without tooltip text?
- Does the rarity or importance read without overwhelming identity?

### Project Fit

- Does it match Starlit Frontier Fantasy?
- Does it match the GDD's tone: colorful, adventurous, slightly whimsical, friendly but dangerous?
- Does it fit the correct biome, class, element, enemy family, or tier?
- Does it look professional next to existing icons?
- Does it avoid generic fantasy drift?

### Rendering Consistency

- Is the lighting direction upper-left?
- Is the outline thickness consistent?
- Are materials rendered with clean cel-shaded planes?
- Are colors controlled and readable?
- Is glow contained and justified?
- Is texture detail visible but not noisy?
- Are edges clean?

### Technical

- Is the file 64x64 PNG for runtime item icons?
- Is the background transparent?
- Are transparent corners clean?
- Is there no chroma-key residue?
- Is the file name kebab-case and correct?
- Is it in the correct folder?
- Is it mapped in `ITEM_ASSETS` or the appropriate card/equipment data?
- If source-sheet generated, does it match processor cell order?
- Does `npm test` pass after adding it?

### UI and Gameplay

- Does it work on parchment/gold inventory tiles?
- Does it work against dark navy gameplay backgrounds?
- Does it avoid overlay conflict with stack counts, locks, badges, and arrows?
- Does the pickup sprite/icon match the inventory icon?
- Does it read during bobbing and aura effects?
- Does it remain recognizable when disabled or grayscale?

### AI Artifact Review

- No text, letters, numbers, watermark, or fake UI.
- No melted geometry.
- No duplicated accidental parts.
- No random ornamentation.
- No blurry outline.
- No excessive particles.
- No inconsistent perspective.
- No background scene.

## 16. Item Visual Documentation Templates

### Item Visual Identity Sheet

```markdown
## Item Visual Identity: [Item Name]

- Data ID:
- Runtime icon:
- Category:
- Subcategory:
- Gameplay purpose:
- Source/drop/shop:
- Rarity/importance:
- Required level:
- Class restriction:
- Biome/theme:
- Element:
- Origin: Natural / Crafted / Magical / Corrupted / Ancient / Starforged
- Core silhouette:
- Category marker:
- Theme marker:
- Functional marker:
- Rarity marker:
- Primary material:
- Secondary material:
- Main colors:
- Accent colors:
- Glow/VFX:
- Inventory icon requirements:
- World pickup requirements:
- Similar existing items:
- Required uniqueness difference:
- Final approval notes:
```

### Item Category Sheet

```markdown
## Item Category: [Category Name]

- Purpose:
- Current data files:
- Runtime folder:
- Source folder:
- Shape language:
- Primary colors:
- Accent colors:
- Materials:
- Icon scale:
- Pickup scale:
- Rarity treatment:
- UI overlay concerns:
- Do:
- Avoid:
- Prompt template:
- Negative prompt:
```

### Theme/Motif Sheet

```markdown
## Theme/Motif: [Theme Name]

- Used by:
- Biomes/enemies/classes:
- Shape language:
- Materials:
- Main colors:
- Accent colors:
- Glow/VFX:
- Surface details:
- Icon treatment:
- Pickup treatment:
- Example item applications:
- What not to do:
```

### Rarity Treatment Sheet

```markdown
## Rarity Treatment: [Rarity]

- Runtime rarity color:
- Silhouette complexity:
- Material richness:
- Trim style:
- Glow intensity:
- Particle/VFX allowance:
- Animation allowance:
- Inventory read:
- World pickup read:
- Example items:
- Avoid:
```

### Inventory Icon Spec

```markdown
## Inventory Icon Spec: [Item Name]

- Runtime file:
- Size:
- Background:
- Object scale:
- Safe padding:
- Corner safety:
- Silhouette read:
- Color palette:
- Outline:
- Lighting:
- Glow:
- Overlay conflicts:
- Small-size test result:
- Approval:
```

### World Pickup Sprite Spec

```markdown
## World Pickup Sprite Spec: [Item Name]

- Current implementation: Reuse icon / Dedicated pickup
- Runtime file:
- Size:
- Frame count:
- Idle motion:
- Glow behavior:
- Shadow behavior:
- Collision/interaction area:
- Prompt behavior:
- Rarity feedback:
- Biome visibility:
- Pickup VFX:
- Approval:
```

### Animated Pickup Spec

```markdown
## Animated Pickup Spec: [Item Name]

- Sheet file:
- Frame size:
- Frame count:
- FPS:
- Looping:
- Motion description:
- Glow frames:
- Sparkle frames:
- Anchor point:
- Export format:
- Renderer hookup:
- Approval:
```

### Boss Drop Spec

```markdown
## Boss Drop Visual Spec: [Boss Drop Name]

- Boss:
- Item ID:
- Category:
- Rarity:
- Boss motif:
- Silhouette requirement:
- Material requirement:
- Color palette:
- Glow/VFX:
- Drop animation:
- Landing effect:
- Pickup effect:
- UI notification:
- Similar boss drops to compare:
- Approval:
```

### Quest Item Spec

```markdown
## Quest Item Visual Spec: [Quest Item Name]

- Quest:
- Item ID:
- Function:
- Required visual object:
- Story motif:
- Biome/faction motif:
- Importance treatment:
- Inventory icon:
- World pickup:
- Interaction prompt:
- VFX:
- Avoid:
- Approval:
```

### Item VFX Spec

```markdown
## Item VFX Spec: [VFX Name]

- Used by:
- Trigger:
- Runtime file:
- Frame size:
- Frame count:
- Duration:
- Color palette:
- Shape motif:
- Intensity:
- Screen-space limit:
- Sound pairing:
- Avoid:
- Approval:
```

### Final Asset Review Record

```markdown
## Final Asset Review: [Item Name]

- Reviewer:
- Date:
- Item ID:
- File path:
- Category:
- Theme:
- Rarity:
- 64x64 readability: Pass / Fail
- 48x48 readability: Pass / Fail
- 34x34 readability: Pass / Fail
- Grayscale readability: Pass / Fail
- Inventory UI test: Pass / Fail
- World loot test: Pass / Fail
- Tooltip test: Pass / Fail
- Naming test: Pass / Fail
- Data mapping test: Pass / Fail
- Source sheet/order test: Pass / Fail / N/A
- AI artifact check: Pass / Fail
- GDD consistency check: Pass / Fail
- Notes:
- Approved:
```

## 17. Assumptions and Final Rules

### Assumptions

- The current 64x64 item icon runtime contract remains active.
- World pickups continue to reuse inventory icons until renderer/data support for dedicated pickup sprites is added.
- Current implemented item rarities are Common, Uncommon, Rare, Epic, and Relic.
- Mythic remains a future rarity for item visuals unless the codebase is updated.
- Quest/key/lore item visuals are future-facing categories implied by the game design, not a complete current runtime asset set.
- Equipment-layer sheets should not be mass-regenerated until the base player rig and equipment-layer direction are locked.

### Final Rules

- Use the GDD's Starlit Frontier Fantasy direction as the top-level art authority.
- Use existing data files and asset processors as the top-level technical authority.
- Runtime item icons are 64x64 transparent PNGs under `img/project-starfall/items/icons/`.
- Source sheets must follow the cyan guide-line and chroma-key rules.
- Category must read from silhouette.
- Theme must read from motif and material.
- Rarity must enhance the design, not replace it.
- Boss and Relic items must have unique silhouettes.
- Shop gear must stay simpler than dungeon, boss, and relic gear.
- Current world loot readability is mandatory because icons are reused in-world.
- No text, no fake UI, no watermark, no background scene, no noisy AI artifacts.
- Reject anything that does not look clean, readable, professional, and native to Project Starfall.
