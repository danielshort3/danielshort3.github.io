# Project Starfall Asset Generation Guide

The `illustrated-v1` migration replaces the classic production artwork while preserving the shared compact adventurer identity, gameplay IDs, and equipment atlas contract. Its source authority is `asset-sources/project-starfall/overhaul-v1/`. Older classic, Fracture Runner, v2 equipment, and compact-enemy sources remain historical references. See the [migration report](ASSET_OVERHAUL_V1.md) for scope, source provenance, and validation status.

This guide is the asset-generation contract for Project Starfall. It is based on the current GDD, Starfall asset prompt notes, runtime data modules, build processors, CSS/UI tokens, and the existing asset folders.

Assumptions are marked with `[Assumption]`. Everything else should be treated as a current project requirement.

For new or revised assets, start with [Confirmed art direction and references](#confirmed-art-direction-and-references). Combat work also follows [Combat visual language v1](#combat-visual-language-v1) and the [required combat generation brief](#required-combat-generation-brief). The formats and ownership below describe the illustrated production migration; acceptance requirements do not certify every generated landmark or untested gameplay path as perfect.

## 1. Project-Specific Visual Style Rules

### Core Style

Use the Project Starfall style: **Starlit Frontier Fantasy — clean illustrated fantasy, expressive characters, warm adventure, and readable danger**.

All generated assets must feel like they belong in a charming 2D side-scrolling browser action RPG with:

- Crisp contours, controlled shading, clear features, and little visual noise.
- Compact readable gameplay forms.
- Warm guild-town frontier materials.
- Luminous fallen-star magic.
- Blue-and-gold Starfall motifs.
- Crafted wood, stone, cloth, brass, crystal, lanterns, runes, portals, and practical magic devices.
- Atmospheric depth in distant scenery, with clearer nearby platforms and props and strong separation around characters and hazards.

### Confirmed Art Direction and References

This direction governs new and revised art, taking precedence over conflicting style instructions in older generation prompts. Preserve the approved references' appearance and motion while maintaining character identity. The overall cast is charming with variety: expressive small creatures, capable compact adventurers, and imposing bosses. Welcoming towns coexist with mysterious regions and credible threats.

| Reference | Role in future work | Remaining review limits |
| --- | --- | --- |
| [Glowcap spring study](../../output/starfall-animation-samples/glowcap-spring-study.png) | Clean finish, clear expressions, deliberate grounded compression and spring motion | Stable cap spots, leaves, contour landmarks, and loop transitions still require cleanup |
| [Oracle cast study](../../output/starfall-animation-samples/oracle-cast-study.png) | Clean finish, expressive gathering and release poses, flexible plant anatomy | Crown landmarks, fine highlights, proportions, and recovery transitions still require cleanup; healing FX follow the separate semantic standard |
| [Approved player run](../../output/starfall-animation-samples/review-v2/player-run-study.png) and [strike](../../output/starfall-animation-samples/review-v2/player-strike-study.png) | Compact player identity, clean finish, articulated action poses | Original study files remain immutable; production packing, equipment sockets, holds, and contact timing require separate verification |

The studies are approved direction references, not production-ready atlases or promises of perfect frame consistency. Their [original Glowcap prompt](../../output/starfall-animation-samples/glowcap-spring-generation-prompt.md) and [original Oracle prompt](../../output/starfall-animation-samples/oracle-cast-generation-prompt.md) remain historical records. Do not promote their strict pixel grids, 24–32-color limits, or hard two/three-tone shading into general rules. Use the images as references and write future briefs from this guide.

The approved studies and their original prompts remain immutable. The explicitly scoped illustrated migration changes production packing and timing through the processors and runtime metadata below; editing this guide alone never changes those contracts.

Do not copy MapleStory or any other existing game. MapleStory is only a high-level reference for side-scroller readability and action coverage.

### Camera and Perspective

- Gameplay sprites: side-scroller camera, right-facing by default.
- Characters and enemies: slight readable 3/4 side-view is allowed, but the pose must still read as side-scroller gameplay.
- UI portraits: chest-up or full-body presentation inside a square frame, 3/4 fantasy portrait angle.
- Maps/backgrounds: side-scroller panoramic view, not isometric or top-down unless generating the world-map atlas.
- World-map atlas: painterly top-down fantasy map, no labels baked into the image.

### Character Scale and Proportions

- Player classes: preserve the current compact adventurer's silhouette and proportions as the baseline. Use the approved player reference rather than imposing a fixed head-count ratio or making the player taller or more exaggerated.
- Faces: clear expressive features at gameplay scale, with readable hair/helmet/costume shapes. Expression and cuteness should fit the character's role; bosses can remain imposing.
- Hands, weapons, shields, hats, and class props can be slightly oversized for readability.
- Enemies: compact silhouettes normalized by the enemy processor. Bosses should feel larger through shape, posture, horns, shoulders, wings, or aura, not by exceeding the cell.
- Items/icons: object fills most of the icon space but leaves transparent padding.

### Line Weight and Outlines

- Gameplay sprites: crisp dark contour outline, visually around 2 to 4 px at 160px frame scale.
- Enemy sprites: crisp outline, readable at 160px frame scale and the smaller gameplay draw size.
- Item icons: thin dark rim or painted edge separation, no heavy UI border unless the asset is itself a UI frame.
- Backgrounds: no hard sprite outline; use painterly edge control and depth separation instead.

### Palette

Use the existing Starfall UI and world palette as the base:

- Ink/dark frame: `#1c2631`, `#102033`, `#071323`
- Parchment panels: `#f2dfbf`, `#f7ebd2`
- Gold accents: `#d8a531`, `#f5cf72`
- Starfall cyan/blue: `#2aaad3`, `#7bdff2`, `#9be7ff`
- Green/nature: `#4e9d61`
- Red/fire material accent: `#d9584a`
- Teal/magic: `#2aa79a`

These are world, character, and material colors. Combat effect cores, symbols, and feedback use the functional palette in [Combat visual language v1](#combat-visual-language-v1); regional and class colors are secondary accents in those effects.

Regional palettes:

- Starfall Crossing: warm parchment, guild wood, soft cyan crystals, gold banners.
- Greenroot: fresh greens, meadow gold, pond cyan.
- Thornpath/Bramble: deep greens, bark browns, thorn crimson.
- Rustcoil/Gearworks: brass, iron gray, teal energy.
- Cinder/Emberjaw: dark volcanic rock, ember orange, molten gold.
- Bandit Ridge: canvas tan, rope brown, muted forest green.
- Oreback Quarry: stone gray, ore gold, mushroom teal.
- Frostfen/Rimewarden: icy blue, pale snow, deep glacier navy.
- Stormbreak: slate cliff gray, bright sky cyan, lightning gold.
- Astral/Eclipse/Rift: deep indigo, violet, cyan star magic, solar gold, void magenta.

### Shading and Lighting

- Sprites: clean illustrated rendering with controlled shading, crisp contours, and clear features. Keep material highlights subordinate to readable forms; avoid noisy surface texture and soft haze over silhouettes.
- Backgrounds: atmospheric fantasy lighting and depth gradients in distant scenery. Nearby platforms and props use clearer silhouettes, simpler texture, and readable contact edges so they support gameplay visibility.
- UI: clean illustrated fantasy UI, not noisy or over-rendered.
- `[Assumption]` Use upper-left/front lighting for gameplay sprites unless a specific VFX or map calls for a different light source.
- Do not let rim lights or glow effects erase the silhouette.

### Pixel Density and Resolution Style

Generate clean illustrated source art that retains the approved samples' crispness at runtime size. Match apparent detail density and edge treatment across an asset family. There is no universal strict pixel grid, fixed small palette count, or hard two/three-tone shading requirement; controlled gradients may support form without blurring contours. When extending an approved character, preserve its identity and agreed finish. Clean, distinct poses and stable identity take priority over extra surface detail.

- Player and FX frame size: 160px.
- Enemy frame size: 160px.
- Item/menu icons: 64px.
- Skill icons: 256px.
- Map backgrounds: 1280x640 WebP.

Avoid tiny noisy detail that collapses at these sizes.

### Transparency and Background Rules

- Runtime sprites, icons, FX, props, stations, portraits, and UI cutouts must have transparent backgrounds.
- Illustrated actor and scenery masters use genuine alpha and empty gutters, with no cyan grid. Their processors locate authored cells and preserve the source palette.
- Some icon batches use a recorded flat `#00ff00` extraction key; select native alpha when a subject's colors conflict with that key. Historical magenta/cyan-grid sources are not the default for the new owners. Never add a key or grid without explicit support in that batch's importer.
- Runtime files must not contain chroma pixels or cyan guide pixels.
- Map backgrounds and splash/start screens are full rectangular images and should not have alpha.

### Animation Readability

- Every animation must read clearly in silhouette before color is considered.
- Attack tells must have obvious windup frames.
- Player attack and skill rows must show preparation, commitment, contact, and recovery, with timing specified in milliseconds rather than inferred from frame count.
- Enemy attacks must make the danger type obvious before contact or projectile release.
- Keep feet, baselines, and body scale stable across frames.
- Match motion to anatomy: mushrooms compress and spring, plants flex through stems and petals, and humanoid or armored characters convey more solid weight through joints, foot contact, recoil, and follow-through.
- Use distinct poses, intentional holds, and smooth transitions; add enough authored articulation for the action rather than forcing every character into the same bounce or pose count. Preserve recognizable anatomy throughout squash, stretch, and expression changes.
- Do not add camera shake, motion blur, UI labels, text, hit numbers, or particles that cross cell borders.

### How To Avoid AI-Generated Artifacts

- Generate a coherent action strip or action-pair sheet using the approved identity reference; pack complete runtime sheets from those authored poses through the domain importer.
- Use the same reference image, seed, style prompt, and proportions across all animations for one character or enemy.
- Lock costume details, colors, accessories, weapons, and silhouette in every prompt.
- After generation, compare all frames at 100 percent zoom and at in-game display size.
- Reject frames with warped limbs, changing faces, changing weapons, inconsistent outlines, melted fingers, extra accessories, unstable baselines, or inconsistent lighting.
- Manually clean minor edge artifacts before processing.

## 2. Global Asset Generation Rules

These rules apply to every Project Starfall asset.

### Required Global Rules

- Use consistent side-scroller perspective for gameplay assets.
- Use consistent scale within an asset family.
- Use consistent outlines for sprites and icons.
- Use upper-left/front lighting for sprites unless the asset is an emissive VFX.
- Use the regional palette for biome assets and the class palette for character identity. Combat effects use meaning-first colors with those palettes as secondary accents.
- Keep silhouettes clean and readable.
- Do not include text, logos, watermarks, labels, signatures, UI artifacts, random symbols, frame numbers, or background clutter unless the asset is explicitly a UI asset.
- Do not include copyrighted characters, copied costumes, or recognizable third-party IP.
- Do not include a ground plane or cast shadow in transparent sprites unless the runtime asset specifically requires it.
- Do not let art cross the cell boundaries of a source sheet.
- Leave padding around each cell.
- Use kebab-case filenames for generated files.
- Keep IDs aligned with the existing runtime registry in `js/games/project-starfall/data/assets.js`.

### Sprite Sheet Rules

- Use one animation row per semantic action.
- Use exact row order from the runtime contract.
- Use exact column count from the runtime contract.
- Keep the character/enemy centered consistently within every frame.
- Keep the feet or hover center aligned to the same registration point.
- Use transparent runtime backgrounds.
- Use transparent gutters rather than drawn guide lines for the illustrated actor importers. A source atlas may differ in dimensions from its runtime export; record the measured source cells and shared transform.
- Do not place action labels or row labels in the image.

### Anatomical Registration and Scoped Motion Review

Generated gutters are often unequal even when the prompt requests an exact grid. Detecting each complete actor prevents cropping, but a nominal column center is not evidence of a stable body anchor. Record an explicit source-space X anchor for each pose using a reviewed anatomical landmark: for example, a rigid shell crown, torso center, pelvis, or planted body base. Pair it with the declared foot/hover Y anchor and the existing shared scale. Keep authored lean, root travel, squash, recoil, limb articulation and vertical motion; do not automatically recenter silhouettes by their bounding boxes or force every pose to match a rigid template.

Verify the packed pixels independently of the anchors used to pack them. Review a second stable landmark or a rigid interior RGB/alpha patch that excludes moving limbs, weapons and effects. Inspect every action being accepted, adjacent frames, the last-to-first seam for loops, and transitions into and out of the action at gameplay size. Translation-fit metrics are screening evidence, not automatic corrections or proof that changing anatomy is correct. Alpha margins, identical grid spacing, shared scale and stable bounding boxes alone cannot certify registration.

Record the exact reviewed actions, axes, landmarks, intended motion, evidence paths and remaining limitations in `source.json` under `registrationReview`; carry that scope into the import report. An `idle-horizontal only` review must not be described as approval of all eight rows, vertical stability, anatomy or gameplay timing. For a packing-only correction, retain before files and verify unchanged raw-source hashes, shared scale, unaffected axes and unaffected row pixels; compare the corrected pixels after undoing the intended translation. See the [Lava Tick follow-up](ASSET_ALIGNMENT_FOLLOWUP.md) for the defect that established this requirement.

### Enemy Combat Body Masks

An enemy's combat body follows the visible actor in its current animation frame. Generate exact one-source-pixel occupancy from the production PNG at **alpha >= 64/255**, preserving transparent gaps between limbs and inside the silhouette. Use the same animation frame, registration, shared scale, facing and hit-reaction render transform as Canvas and Pixi. Do not enlarge the combat body to the full cell, a silhouette bounding rectangle, or an arbitrary interaction margin. Transparent packing gutters, shadows, auras and separate FX do not count as enemy anatomy.

Keep the stable `x/y/w/h` terrain body for platform movement, navigation and separation. It is not the enemy's combat hurtbox. Attack reach and declared hazards remain separate gameplay geometry; changing an actor sheet must not silently expand either. Review mask overlays at gameplay size in both facings and through recoil, including both visible extremities and empty corners/gaps. A faint antialiased pixel below the threshold is excluded; this raster-edge rule is not permission for a broad invisible border.

[The mask generator](../../build/generate-project-starfall-enemy-hurtboxes.js) owns [generated enemy hurtbox data](../../js/games/project-starfall/data/enemy-hurtboxes.js); [the runtime helper](../../js/games/project-starfall/engine/enemy-hurtboxes.js) transforms and queries it. [Enemy activation](../../build/integrate-project-starfall-overhaul-enemies.js) automatically regenerates masks after applying accepted imports, and [the JavaScript build](../../build/build-js.js) requires `--check` before bundling. For other accepted repaint/repack workflows, regenerate explicitly and require the same freshness and collision checks below. Registration-only changes also require collision review because they move the mask and drawing together. See the [enemy hitbox audit](ENEMY_HITBOX_AUDIT.md) for evidence and scope.

```bash
node build/generate-project-starfall-enemy-hurtboxes.js
node build/generate-project-starfall-enemy-hurtboxes.js --check
npm run test:starfall:hitboxes
```

### Individual Frame Rules

Use individual frames only for manual cleanup or review. Runtime import should use the sheet formats below unless the code is intentionally updated.

### Export Formats

- Runtime sprites/icons/FX/UI cutouts: PNG with alpha.
- Source sheets: PNG.
- Map backgrounds: WebP, 1280x640, no alpha.
- World map atlas: WebP, 1920x1080, no alpha.
- Do not export JPEG for sprites, icons, FX, or UI cutouts.

### Folder Structure

Use the existing folder structure:

```text
asset-sources/project-starfall/overhaul-v1/players/
asset-sources/project-starfall/overhaul-v1/enemies/
asset-sources/project-starfall/overhaul-v1/fx/
asset-sources/project-starfall/overhaul-v1/scenery/
asset-sources/project-starfall/overhaul-v1/icons/
img/project-starfall/animations/players/
img/project-starfall/animations/enemies/
img/project-starfall/animations/enemy-projectiles/
img/project-starfall/animations/combat-fx/basic/
img/project-starfall/animations/combat-fx/enemies/
img/project-starfall/animations/combat-fx/projectiles/source/
img/project-starfall/animations/combat-fx/skills/
img/project-starfall/animations/combat-fx/skills/source/
img/project-starfall/animations/fx/
img/project-starfall/animations/fx/source/
img/project-starfall/animations/pets/
img/project-starfall/animations/pets/source/
img/project-starfall/animations/portals/
img/project-starfall/animations/portals/source/
img/project-starfall/cards/icons/
img/project-starfall/cards/source/
img/project-starfall/characters/
img/project-starfall/enemies/
img/project-starfall/environment/terrain/
img/project-starfall/environment/props/
img/project-starfall/environment/ramps/
img/project-starfall/environment/source/
img/project-starfall/environment/structures/
img/project-starfall/environment/structures/source/
img/project-starfall/items/icons/
img/project-starfall/items/source/
img/project-starfall/items/sheets/
img/project-starfall/maps/
img/project-starfall/maps/source/field/
img/project-starfall/maps/source/safe-zones/
img/project-starfall/skills/base/
img/project-starfall/skills/advanced/
img/project-starfall/skills/source/
img/project-starfall/stations/
img/project-starfall/ui/
img/project-starfall/ui/menu-icons/
img/project-starfall/ui/source/
img/project-starfall/world-map/
asset-sources/project-starfall/prompts/
```

If a new asset folder is added, update `build/copy-to-public.js` so deployment includes it.

### Actionable Repo Metadata

The guide is mirrored into these repo-local execution files:

- Machine-readable manifest: `asset-sources/project-starfall/asset-generation-manifest.json`
- Source folder notes: `asset-sources/project-starfall/README.md`
- Reusable prompt templates: `asset-sources/project-starfall/prompts/README.md`
- Validator: `build/validate-project-starfall-asset-generation.js`
- NPM command: `npm run validate:project-starfall-assets`

When this guide changes a dimension, folder, source sheet, processor, naming rule, or required asset category, update the manifest and validator in the same change.

### Illustrated Production Ownership

| Domain | Authoritative sources and evidence | Writer / processor |
| --- | --- | --- |
| Shared player, portrait, and fox | `overhaul-v1/players/ledger.json`, `registration.json`, plus immutable approved run/strike studies | `build/process-project-starfall-overhaul-players.js` |
| Enemy actors and portraits | `overhaul-v1/enemies/inventory.json`, per-identity `source.json`, `source.png`, prompt and import report | `build/process-project-starfall-overhaul-enemies.js --enemy <file-id>`; reviewed imports add `--import` |
| Global FX, portals, projectile | `overhaul-v1/fx/ledger.json` and raw masters | `build/process-project-starfall-overhaul-fx.js` |
| Basic, skill, and enemy combat FX | Native editable `build/lib/starfall-combat-language-art.js`; output hashes in `overhaul-v1/fx/ledger.json` | `build/generate-project-starfall-combat-fx.js` |
| Maps, terrain, props, ramps, structures, stations, world atlas | `overhaul-v1/scenery/ledger.json`, exact prompts and raw masters | `build/process-project-starfall-overhaul-scenery.js` |
| Items, skill/card/menu icons, screens, pedestal | `overhaul-v1/icons/catalog.json`, exact prompts, immutable raw batches and `ledger.json` | `build/project-starfall-overhaul-icons.js rebuild all` |
| Equipment overlays | `overhaul-v1/icons/equipment/` editable SVG masters and icon-domain ledger | `build/project-starfall-equipment-illustrations.js` via `build/generate-project-starfall-equipment-atlases.js --all` |

Paths beginning `overhaul-v1/` above are relative to `asset-sources/project-starfall/`. Legacy generation entry points defer to these owners and must not silently overwrite their outputs. Historical source folders remain for provenance and compatibility. The website build publishes checked-in raster outputs and derives start-screen AVIF/WebP alternatives; it does not authorize a new art generation.

The immutable baseline records 824 originally active paths and 31 protected session images. Completion requires `node build/audit-project-starfall-overhaul.js --complete`: every original path must be changed or explicitly retired to a verified replacement, no current output may be missing, and every protected image hash must match. Generated-art identity and motion still require visual review; dimension, alpha, registration, and coverage checks alone do not prove seamless anatomy or landmark consistency.

## 3. Character Asset Instructions

### Current Player Runtime Contract

All player class animation sheets use:

- Runtime path: `img/project-starfall/animations/players/<file-id>-sheet.png`
- Source authority: `asset-sources/project-starfall/overhaul-v1/players/ledger.json`; action-pair masters and approved run/strike study references
- Portrait path: `img/project-starfall/characters/<file-id>.png`
- Sheet size: `1280x1600`
- Frame size: `160x160`
- Layout: 8 columns x 10 rows
- Background: transparent in runtime PNG
- Facing: right-facing
- Portrait size: `320x320`, transparent PNG

Rows must be exactly:

| Row | Action | Frames | FPS | Loop |
| --- | --- | ---: | ---: | --- |
| 0 | idle | 8 | 8 | yes |
| 1 | run | 8 | 40/3 | yes |
| 2 | jump | 8 | 16, weighted holds | no |
| 3 | fall | 8 | 12 | no |
| 4 | climb | 8 | 10 | yes |
| 5 | basic | 8 | 100, weighted holds | no |
| 6 | skill | 8 | 24 | no |
| 7 | party | 8 | 80, weighted holds | no |
| 8 | hit | 8 | 24 | no |
| 9 | defeat | 8 | 10 | no |

`data/animations.js` is authoritative for FPS, sequences, and holds. A hold weight consumes that many FPS ticks: jump `[1,1,1,1,2,2,4,4]`, basic `[1,4,4,5,6,7,5,6]`, and party `[7,7,7,7,5,5,5,5]`. Basic melee contact is 90 ms; ranged release retains 130 ms. Offensive skill preparation resolves at 166.7 ms; normal healing resolves at its 350 ms pulse. These gameplay events are independent of nominal frame count. Enemy telegraphs retain the final committed pose for at least 200 ms, or 300 ms for bosses/major threats.

Do not add a new runtime row unless `js/games/project-starfall/data/animations.js` and the processing scripts are updated.

`[Assumption]` Interaction animation currently maps best to the `party` row for friendly interactions or the `idle` row for neutral interact prompts. If a true `interact` animation is required later, generate it as a separate 6-frame source strip first and update the runtime contract before import.

### Required Player Class Files

All twelve classes currently share `generic-player-sheet.png` and `generic-player.png`; the table below describes gameplay and equipment identities, not twelve active body sheets. Class-specific body sheets require an explicit later registry migration.

| Class | File ID | Visual Identity |
| --- | --- | --- |
| Fighter | `fighter` | Practical sword-and-guard adventurer, sturdy red/gold/steel accents |
| Mage | `mage` | Robed arcane caster, cyan/gold star magic focus |
| Archer | `archer` | Leather ranger, bow, green/gold travel gear |
| Guardian | `guardian` | Shield defender, heavy armor, oath glow, blue/gold guard motifs |
| Berserker | `berserker` | Heavy weapon, red rage accents, rugged armor |
| Duelist | `duelist` | Agile blade fighter, refined coat, silver/crimson tempo cues |
| Fire Mage | `fire-mage` | Ember robes, heat core, orange/gold flame motifs |
| Rune Mage | `rune-mage` | Glyph caster, teal/cyan runes, floating focus |
| Storm Mage | `storm-mage` | Lightning caster, storm cloak, blue/yellow charge |
| Sniper | `sniper` | Precision archer, longbow/crossbow silhouette, focused aim |
| Trapper | `trapper` | Tool belt, snares, practical field kit, olive/amber accents |
| Beast Archer | `beast-archer` | Ranger with companion cues, nature bond, feather/fur accents |

### Master Player Sheet Prompt

Use this prompt template for a complete class sheet:

```text
Create an original Project Starfall player class animation source sheet for [CLASS_NAME], a compact heroic 2D side-scroller fantasy RPG character.

Apply the completed required combat generation brief to each combat action. Preserve the approved reference style and character identity. Generate actor poses separately from functional FX, using the specified shared anchors and phase timing.

Style: Starlit Frontier Fantasy, clean illustrated 2D game sprite, crisp dark contour outline, controlled shading, clear features and little surface noise. Preserve [APPROVED COMPACT PLAYER REFERENCE] silhouette and proportions, with solid humanoid weight and articulated motion. Luminous fallen-star magic accents, no copied IP, no MapleStory costume copying.

Character lock: [EXACT HAIR/HELMET], [EXACT FACE VISIBILITY], [EXACT OUTFIT], [EXACT WEAPON OR FOCUS], [CLASS COLORS], [SIGNATURE ACCESSORY]. Keep these identical in every frame.

Runtime layout: 8 columns x 10 rows, 160px cells, 1280x1600. Generate [REQUESTED ACTION ROWS] as coherent eight-pose strips with genuine alpha, empty gutters, one complete actor per cell, right-facing, consistent scale and declared foot/hover anchor, no grid or labels. The importer packs these into the runtime row order.

Rows in order:
1 idle: subtle breathing and cloth/hair motion.
2 run: alternating stride cycle with clear contact and passing poses.
3 jump: crouch takeoff, upward lift, airborne stretch.
4 fall: downward bracing, cloak/hair rising, landing-ready posture.
5 climb: hands reaching and pulling on a ladder or ledge without drawing the ladder.
6 basic: [CLASS BASIC ATTACK], windup, strike/release, recovery.
7 skill: [CLASS SIGNATURE SKILL], charge, cast/release, follow-through.
8 party: supportive buff or rally gesture, coordinated with a separately generated semantic effect.
9 hit: recoil, flinch, stagger, recover.
10 defeat: collapse/downed pose, readable but non-gory.

Background: genuine alpha with empty gutters. No text, no labels, no numbers, no watermark, no UI, no scenery, no cast shadow, no drawn grid.
```

### Master Player Negative Prompt

```text
text, labels, numbers, watermark, signature, logo, UI, speech bubble, background scenery, floor shadow, frame labels, isometric view, top-down view, front-facing idle only, photorealism, 3D render, low-quality pixel mush, noisy texture, inconsistent costume, changing hair, changing weapon, changing face, extra limbs, missing limbs, warped hands, broken feet, melted outline, flickering proportions, inconsistent lighting, copied MapleStory sprite, copyrighted character, cell overflow, cropped weapon, cyan guide color inside artwork, chroma key color inside artwork
```

### Player Animation Rows

| Action | Pose Requirements | Motion Principles | Review Checklist | Filename |
| --- | --- | --- | --- | --- |
| Idle | Stable stance, subtle breathing, weapon held naturally | Small loop, no foot sliding | Same baseline, no costume drift, frame 8 returns to frame 1 | Row 0 in the production atlas |
| Walk | `[Assumption]` Not a separate runtime row | If needed later, generate separate 6-frame strip | Do not replace run unless code changes | `<file-id>-walk-source.png` only after code update |
| Run | Clear contact, passing, airborne/recovery poses | Strong side-scroller stride, readable legs | Alternating feet, stable head size, no sliding | Row 1 |
| Jump | Crouch, launch, rise, apex | Anticipation then upward stretch | No ground drawn, no clipped weapon | Row 2 |
| Fall | Airborne bracing, downward cloth/hair motion | Slower non-loop hold readability | Reads as falling, not jumping | Row 3 |
| Land | `[Assumption]` Covered by final jump/fall frames | Compression and recovery if separated later | Do not add runtime row without code update | Future `<file-id>-land-source.png` |
| Climb | Reach, pull, knee lift, reset | Loopable alternating arms | No ladder drawn, hands align consistently | Row 4 |
| Attack | Class basic attack with windup, hit frame, recovery | Anticipation, impact, follow-through | Hit frame is unmistakable, weapon does not resize | Row 5 |
| Hurt | Recoil, flinch, stagger, recover | Fast snap then recovery | Does not look like attack or defeat | Row 8 |
| Death | Collapse, downed, fade-ready final pose | Non-gory, readable defeat | Final pose stable, no gore | Row 9 |
| Interaction | `[Assumption]` Use party/idle in current runtime | Friendly gesture, hand raise, channel object | Generate separately only if code adds row | Future `<file-id>-interact-source.png` |
| Skill | Charge, cast, release, recovery | Strong class identity and readable VFX anchor | VFX does not hide body | Row 6 |
| Party | Buff, rally, shield, or supportive pulse gesture | Clear supportive intent | Separate FX aligns with the gesture and stays inside its own cell | Row 7 |

### Class-Specific Attack/Skill Prompt Inserts

Use these action descriptions to plan the actor poses and their separately generated FX. Effect descriptions belong in the FX prompt, not painted into the actor sheet. Every effect retains its functional core and symbol from Combat Visual Language v1; class colors below describe secondary accents only.

- Fighter basic: `short sword slash with grounded stance and momentum trail`.
- Fighter skill: `wide guarded power strike with coral impact arc and gold accents`.
- Mage basic: `small arcane bolt cast from hand or focus, coral damage core with cyan accents`.
- Mage skill: `larger star-rune burst with a bright casting circle`.
- Archer basic: `quick bow shot with clear release pose`.
- Archer skill: `focused multi-arrow or piercing shot with coral damage cue and gold-green accents`.
- Guardian basic: `shield bash or guarded weapon strike`.
- Guardian skill: `oath barrier pulse and shield-wall stance`.
- Berserker basic: `heavy cleave with coral damage arc and crimson accents`.
- Berserker skill: `gold rage-enhancement cue, two-handed smash, coral impact with crimson accents`.
- Duelist basic: `fast precise cut with coral contact core and silver slash accents`.
- Duelist skill: `flash-step flourish with repeated blade afterimage`.
- Fire Mage basic: `small firebolt with coral damage core and orange ember trail accents`.
- Fire Mage skill: `inferno burst or wildfire cast with heat aura`.
- Rune Mage basic: `rune-marked arcane shot`.
- Rune Mage skill: `ground glyph detonation with coral danger/contact core and cyan rune accents`.
- Storm Mage basic: `chain-bolt hand cast`.
- Storm Mage skill: `stormfront charge with lightning arcs`.
- Sniper basic: `aimed arrow shot with steady posture`.
- Sniper skill: `one perfect shot, coral damage cue with narrow gold aim accents`.
- Trapper basic: `quick shot or snare toss`.
- Trapper skill: `trap deployment with field kit, violet snare cue, secondary green material accents`.
- Beast Archer basic: `bow shot with companion-bond accent`.
- Beast Archer skill: `pack-call aura or companion strike cue without drawing a full companion unless required`.

### Player Portrait Prompt

```text
Create a 320x320 transparent PNG character portrait for Project Starfall class [CLASS_NAME].
Show the same character design as the animation sheet: [LOCKED CHARACTER DETAILS].
Style: Starlit Frontier Fantasy, clean 2D fantasy game portrait, crisp silhouette, compact heroic proportions, warm guild fantasy materials, luminous Starfall accents, readable at small UI size.
Pose: confident class-select pose, right-facing or 3/4 view, weapon/focus visible but not cropped awkwardly.
Background: transparent. No text, no logo, no watermark, no UI frame.
```

## 4. Enemy Asset Instructions

### Current Enemy Runtime Contract

Illustrated enemy sheets use:

- Runtime path: `img/project-starfall/animations/enemies/<file-id>-sheet.png`; the inventory records each retired compact path and replacement
- Source path: `asset-sources/project-starfall/overhaul-v1/enemies/<file-id>/source.png`, with `source.json` and `import-report.json`
- Portrait path: `img/project-starfall/enemies/<file-id>.png`
- Sheet size: `960x1280`
- Frame size: `160x160`
- Layout: 6 columns x 8 rows
- Facing: right-facing
- Runtime background: transparent
- Source background: genuine alpha, empty gutters, no colored grid or chroma key
- Packing: one scale per identity, measured source cells and declared foot/hover registration; never resize each pose independently

Rows must be exactly:

| Row | Action | Source Frames | Runtime Notes |
| --- | --- | ---: | --- |
| 0 | idle | 6 | Breathing, bobbing, stance |
| 1 | move | 6 | Hop, walk, crawl, fly, or charge preparation |
| 2 | telegraph | 6 | Warning ends in the held committed pose |
| 3 | attack | 6 | Melee, bite, slash, slam, charge contact |
| 4 | projectile | 6 | Throw, spit, cast, shoot, or no-op pose if enemy has no projectile |
| 5 | buff | 6 | Actual heal, shield, enrage, phase, summon, or special |
| 6 | hit | 6 | Recoil and recovery |
| 7 | defeat | 6 | Collapse, dissolve, shatter, burn out, non-gory |

The 51 runtime IDs resolve to 48 sheet/portrait pairs: 44 canonical generated identities, four compatibility copies for admin bandit variants, and three enemy-ID aliases. Consult the enemy inventory before generating a duplicate identity. Raw review approval and successful import measurements are separate requirements; the migration report records remaining validation work.

### Master Enemy Sheet Prompt

```text
Create one original Project Starfall compact enemy animation source sheet for [ENEMY_NAME].

Apply the completed required combat generation brief to each combat action. Preserve approved references and generate actor poses separately from functional FX; name the actual gameplay effect of the buff row and coordinate its event timing and anchors.

Style: Starlit Frontier Fantasy, clean illustrated 2D side-scroller RPG monster, crisp dark contour outline, controlled shading and clear features, readable silhouette at 160px and gameplay size, approved proportions, little visual noise, no copied IP, no text. Match [CHARACTER PERSONALITY AND ANATOMY]: expressive small creatures, flexible plants, solid humanoids, or imposing bosses as appropriate.

Enemy design: [VISUAL DESCRIPTION]. Gameplay role: [ROLE]. Attack tell: [ATTACK TELL]. Regional palette: [REGION PALETTE].

Runtime layout: 6 columns x 8 rows, 160px cells, 960x1280. Source: six distinct authored poses per action row, genuine alpha, empty gutters, no grid lines, one complete enemy per cell, right-facing, consistent scale and declared foot/hover anchor. Preserve all visible anatomy and landmarks across the action.

Rows in order:
1 idle: 6 readable idle/breath/bob poses.
2 move: 6 articulated movement poses.
3 telegraph: 6 warning poses ending in the held committed direction.
4 attack: 6 strike/contact/recovery poses.
5 projectile: 6 projectile launch/cast poses or a clear no-projectile special pose.
6 buff: 6 support/enrage/shield/heal/special poses; functional FX remain separate.
7 hit: 6 recoil/recovery poses.
8 defeat: 6 collapse/dissolve/shatter poses, non-gory.

No labels, no UI, no scenery, no cast shadow, no watermark, no cell overflow, no guide color inside artwork.
```

### Master Enemy Negative Prompt

```text
text, labels, numbers, watermark, signature, logo, UI, background scenery, ground shadow, photorealism, 3D render, isometric, top-down, front-view-only, copied Pokemon, copied MapleStory, copied IP, inconsistent body size, changing horns, changing weapon, changing colors, extra limbs, missing limbs, melted outline, noisy detail, tiny unreadable face, cropped body, cell overflow, cyan guide color in art, chroma key color in art, gore
```

### Enemy Generation Table

Use this table as the source list. File IDs use kebab-case and should match or be mapped from `ASSET_REGISTRY.enemies`.

| Enemy ID | File ID | Role | Visual Description | Attack Tell Requirement |
| --- | --- | --- | --- | --- |
| `slimelet` | `slimelet` | Basic swarm-light ooze | Small translucent mint/pale-blue slime with shiny star core and simple eyes | Squash downward before hop contact |
| `dewSlime` | `dew-slime` | Starter swarm ooze | Wet dew-colored slime with glossy droplet crown | Compress and lean forward before quick hop |
| `mossback` | `mossback` | Durable beast bruiser | Squat boar-like forest beast with bark hide, moss back, small tusks | Brace shoulders and lower head before shove |
| `thornSprout` | `thorn-sprout` | Stationary plant turret | Rooted bulb plant, thorn-pod mouth, leaf fins | Pod inflates and thorns glow before shot |
| `vineSnapper` | `vine-snapper` | Ambush plant skirmisher | Low vine creature with snapping blossom head | Coil body backward before lunge |
| `bristleBoar` | `bristle-boar` | Charger beast | Lean boar with bristled mane and dirt-scraped hooves | Paw ground, lower tusks, long charge line posture |
| `briarStag` | `briar-stag` | Heavy plant charger | Stag with briar antlers, bark plates, thorn trail | Antlers glow and head lowers before charge |
| `dustImp` | `dust-imp` | Fast melee imp | Wiry tan/red imp with oversized claw hands | Crouch with claws pulled back before leap |
| `clockbug` | `clockbug` | Armored construct tank | Beetle construct, brass shell, exposed gears, mechanical jaw | Shell locks and gear teeth spin before snap |
| `rustRatchet` | `rust-ratchet` | Construct skirmisher | Small gear-driven ratchet crawler with skate-wheel legs | Wheel sparks and body tilts before snap dash |
| `coilSentry` | `coil-sentry` | Construct turret | Brass coil turret with teal electrical core | Coil expands and cyan charge builds before bolt |
| `scrapWarden` | `scrap-warden` | Armored blocker | Humanoid scrap construct with shield plate and heavy arm | Shield raises, counter arm cocks back |
| `emberWisp` | `ember-wisp` | Flying ranged spirit | Floating flame spirit, orange/gold/red ember tail | Flame core contracts, firebolt forms at front |
| `ashCrawler` | `ash-crawler` | Volcanic bruiser | Low volcanic beast with ash carapace and ember plates | Plates flare before heavy bite or slam |
| `lavaTick` | `lava-tick` | Fast burn skirmisher | Tiny molten tick with hot abdomen and ember legs | Abdomen glows and legs tense before bite |
| `cinderSpitter` | `cinder-spitter` | Volcanic thrower | Stocky cave creature with cinder throat pouch | Throat pouch swells before lob |
| `banditCutter` | `bandit-cutter` | Melee blocker humanoid | Hooded/masked bandit with short blade and guarded stance | Blade arm pulls back while guard stays up |
| `banditThrower` | `bandit-thrower` | Ranged priority humanoid | Slim bandit with knife bandolier and backward-leaning throw pose | Knife lifted high with clear throw arc |
| `orebackBeetle` | `oreback-beetle` | Tank/material beast | Thick beetle shell with ore chunks and horned front | Shell lowers and horn points forward |
| `glowcapHealer` | `glowcap-healer` | Plant support | Mushroom healer with luminous cap, spores, small staff-like stem | Cap brightens, spores spiral outward |
| `crackedMimic` | `cracked-mimic` | Rare elite treasure construct | Broken treasure chest monster, teeth, magical cracked lock | Lid opens wider and lock flashes before bite |
| `brambleking` | `brambleking` | Plant boss | Crowned bramble monarch, root beard, thorn crown, vine arms | Root crown flares before root wave/thorn volley |
| `clockworkTitan` | `clockwork-titan` | Construct boss | Heavy brass/iron titan with armor plates and gear core | Gear core winds visibly before slam |
| `quarryColossus` | `quarry-colossus` | Mineral construct boss | Massive stone/ore golem with quarry plates and crystal seams | Ore plates lift and fists rise before slam |
| `emberjawGolem` | `emberjaw-golem` | Volcanic construct boss | Dark stone golem, glowing magma cracks, furnace mouth, heavy fists | Furnace mouth glows and fists overheat |
| `frostlingScout` | `frostling-scout` | Frostkin fast melee | Small frost scout with ice dagger, fur hood, quick stance | Dagger hand freezes over before dash |
| `shardling` | `shardling` | Frost swarm | Crystalline ice hopper with tiny eyes and shard fins | Shards vibrate before hop |
| `rimebackBrute` | `rimeback-brute` | Frost beast tank | Heavy frost beast with rime shell and plated back | Shell frosts over before body slam |
| `glacierSentinel` | `glacier-sentinel` | Frozen turret | Ice obelisk sentry with glowing lance core | Lance point forms and brightens |
| `snowglareWisp` | `snowglare-wisp` | Frost spirit flyer | Pale floating frost wisp with snow-glow eye | Eye narrows, ice mote forms |
| `icebloomOracle` | `icebloom-oracle` | Frost plant healer | Elegant ice flower oracle with glowing petals | Petals open and healing motes gather |
| `galeHarrier` | `gale-harrier` | Storm flyer | Wind spirit bird shape with ribbon-like gust wings | Wings fold back before gust dive |
| `stormboundArcher` | `stormbound-archer` | Storm ranged humanoid | Hooded storm archer with lightning bowstring | Bowstring crackles and aim line forms |
| `thunderRam` | `thunder-ram` | Storm charger beast | Ram with storm horns, charged hooves, cloud mane | Horns spark and hooves scrape before charge |
| `cloudcallAcolyte` | `cloudcall-acolyte` | Storm support | Robed acolyte with cloud charm and staff | Staff lifts and cloud ring forms |
| `indexScribe` | `index-scribe` | Astral thrower | Floating scribe with loose pages, ink-star quill | Pages orbit and quill points before throw |
| `lumenSentinel` | `lumen-sentinel` | Astral construct tank | Polished star-metal sentinel with luminous plates | Chest gem charges before beam or shield |
| `voidMote` | `void-mote` | Void flyer | Small dark violet star mote with cyan core | Core inverts and shadow ring pulses |
| `eclipseDuelist` | `eclipse-duelist` | Astral blocker | Elegant masked duelist with solar/lunar blade | Parry stance, blade crescent flashes |
| `riftAberration` | `rift-aberration` | Void elite | Warped but readable rift beast with split star limbs | Body tears open with magenta/cyan warning |
| `rimewarden` | `rimewarden` | Frost construct boss | Ancient frost guardian with ice crown and heavy shield | Crown glows, shield plants into ground |
| `stormbreakRoc` | `stormbreak-roc` | Storm beast boss | Huge roc condensed to compact sprite, storm feathers, lightning talons | Wings spread and lightning gathers under talons |
| `astralArchivist` | `astral-archivist` | Astral humanoid boss | Grand archive mage with floating books and star robes | Books fan open and runes align |
| `eclipseSovereign` | `eclipse-sovereign` | Astral royalty boss | Regal eclipse ruler with corona crown, dark star mantle | Crown halo darkens and solar edge flashes |

Admin-only IDs `bandit-cutter-direct`, `bandit-cutter-reference`, `bandit-cutter-hybrid`, and `bandit-cutter-puppet` retain compatibility copies of the new canonical bandit output. They do not represent four separately generated identities.

### Enemy Archetype Readability Rules

- Hoppers: show squash before hop and stretch at launch.
- Bruisers/tanks: show weight through low stance and delayed impact.
- Turrets: keep base fixed; telegraph through glowing pod/barrel/lance.
- Skirmishers: use lean, crouch, and quick recoil poses.
- Chargers: telegraph must be long and unmistakable.
- Blockers: shield/guard must be readable in idle and telegraph.
- Flyers: maintain hover center instead of ground baseline.
- Healers/support: the buff row must show a distinct support gesture coordinated with separate semantic FX. Healing uses mint plus marks and a restorative pulse; spores, cloud rings, or runes may provide secondary accents for the actual effect category.
- Elites: silhouette can be more complex but must remain readable at 160px and gameplay size.
- Bosses: use crown/core/armor/wing/aura to imply status, not oversized cells.

## 5. Item, Pickup, and Interactable Asset Instructions

### Item Icon Runtime Contract

- Final icon path: `img/project-starfall/items/icons/<item-id>.png`
- Size: `64x64`
- Format: PNG with alpha
- Source authority: `asset-sources/project-starfall/overhaul-v1/icons/catalog.json`, exact batch prompts and immutable `raw/` PNGs
- Processor: `build/project-starfall-overhaul-icons.js`; 209 standalone item outputs, with six-pixel outer padding at 64px
- Historical processed sheets under `img/project-starfall/items/sheets/` are not active icon-generation masters.
- Style: clean fantasy item icon, centered, readable, no UI frame unless the item is itself a coupon/card.

### Item Icon Global Prompt

```text
Create a Project Starfall 64x64 transparent fantasy RPG item icon for [ITEM_NAME].
Style: Starlit Frontier Fantasy, clean illustrated 2D icon, crisp silhouette, controlled shading, clear material features, minimal noise, dark edge separation, readable at 64px, [TIER MATERIALS], [REGION OR CLASS MOTIF].
Composition: one centered item, no background, no UI border, no rarity frame, no quantity number, no label, no watermark.
Use palette: [PALETTE]. Leave transparent padding around the item.
```

### Item Icon Negative Prompt

```text
text, numbers, label, watermark, signature, UI frame, inventory slot, rarity border, background, scenery, character hand, photorealism, 3D render, clutter, multiple unrelated items, cropped item, drop shadow crossing icon edge, copied IP, unreadable tiny detail
```

### Historical Item Source Sheets

The old source sheet names below are retained for provenance. Do not rebuild active icons from them. The current catalog owns grouped consumables, materials, currency, coupons, gear, star cards, and two regional-equipment batches covering the 30 previously unmapped items. It also records semantic correction cells and their exact source indices.

| Source Sheet | Layout | Items |
| --- | ---: | --- |
| `ai-items-star-cards-sheet.png` | 5x1 | white, green, blue, purple, orange star cards |
| `ai-items-consumables-materials.png` | 5x5 source, 4 output rows | coins, town return scroll, guard tonic, swiftstep oil, magnet charm, pet whistle, cube fragment, skill manuals, reset scrolls, admin console, upgrade materials, gel drop, ore chunks, line catalyst |
| `ai-items-potion-tiers.png` | 4x3 | health potions, resource tonics, rations by tier |
| `ai-items-mob-materials-core.png` | 5x4 | early/mid mob and boss materials |
| `ai-items-mob-materials-late.png` | 5x4 | late frost/storm/astral/eclipse materials |
| `ai-items-coin-stacks.png` | 4x1 | small, medium, large, huge coin stacks |
| `ai-items-rate-coupons.png` | 3x2 | XP and drop coupons |
| `ai-items-slot-prisms-plinko.png` | 3x3 | slot coupons, cubes, Plinko balls |
| `ai-items-shop-boss-forest.png` | 5x5 | early shop and forest boss gear |
| `ai-items-world-drops.png` | 5x4 | broad world drop gear |
| `ai-items-boss-core-storm.png` | 5x5 | Emberjaw, Titan, Colossus, Stormbreak gear |
| `ai-items-boss-astral-eclipse.png` | 5x4 | Astral and Eclipse boss gear |

The five Star Card item icons now belong to the `items-star-cards` batch in the new catalog. The former external processed sheet is historical. Keep physical green card facets, leaves, and materials intact while removing actual extraction-key residue; the combat semantic palette does not recolor ordinary item materials.

### Card Icon Contract

Monster/Star card icons are a separate asset family from item icons.

- Source authority: `overhaul-v1/icons/catalog.json`, `monster-cards` batch and its preserved raw PNG
- Runtime icon path: `img/project-starfall/cards/icons/<card-id>.png`
- Runtime size: `64x64`
- Format: PNG with alpha
- Source layout: 7 columns x 3 rows
- Processor: `build/project-starfall-overhaul-icons.js`; the legacy card command routes to this owner
- Runtime IDs: every entry in `CARD_DEFINITIONS` in `js/games/project-starfall/data/cards.js`

Card icon prompt:

```text
Create a Project Starfall card icon source sheet, 7 columns x 3 rows, covering every CARD_DEFINITIONS entry in data order.
Style: Starlit Frontier Fantasy item-symbol art, clean illustrated fantasy card motifs, one centered readable symbol per cell, genuine alpha or the catalog's explicitly configured extraction key, empty gutters with no drawn grid, no text, no labels, no card UI frame, no numerals, no watermark.
Each icon should match the card name, tags, rarity mood, and gameplay role through symbol, element, and color. Keep every symbol readable at 64x64.
```

Card icon negative prompt:

```text
text, initials, numbers, card frame, UI border, labels, watermark, logo, character portrait, scenery, clutter, copied IP, cropped symbol, cell overflow, guide color inside artwork, chroma key color inside artwork
```

### Item Tier Visual Rules

- Training: plain wood, cloth, leather, humble shapes.
- Copper: warm metal, simple rivets, beginner crafted.
- Iron: practical soldier gear, darker gray metal.
- Steel: refined, cleaner bevels, stronger silhouette.
- Silver: elegant bright metal, polished.
- Runed: glowing glyphs, teal/cyan magic seams.
- Starforged: midnight steel, gold, crystal, star motifs.
- Ancient: relic-grade weathered alloys, stone, precious cores.

### Pickup Rules

Pickups should use the item icon as the base visual. Do not bake pickup glow, rarity ring, count text, or inventory slot into the icon unless the runtime specifically needs a separate pickup sprite.

`[Assumption]` If dedicated world pickup sprites are added later, use `64x64` PNG alpha under `img/project-starfall/items/pickups/<item-id>-pickup.png` and keep the object centered with a small Starfall glow.

### Interactables and Stations

Current station assets:

- `img/project-starfall/stations/shop.png` - `320x320`, PNG alpha
- `img/project-starfall/stations/storage.png` - `320x320`, PNG alpha
- `img/project-starfall/stations/slots.png` - `320x320`, PNG alpha
- `img/project-starfall/stations/upgrade.png` - `320x320`, PNG alpha
- `img/project-starfall/stations/class.png` - `320x320`, PNG alpha

Station prompt:

```text
Create a 320x320 transparent PNG Project Starfall town station prop for [STATION_NAME].
Style: Starlit Frontier Fantasy, clean 2D side-scroller prop, warm guild-town craft, crisp readable silhouette, practical fantasy construction, blue-and-gold Starfall accents, lantern/crystal/rune details as appropriate.
Composition: one centered interactable station, no character, no text label, no UI, no background, no cast shadow.
Function cue: [SHOP/STORAGE/SLOTS/UPGRADE/CLASS] should be obvious through shape and props, not written words.
```

Station negative prompt:

```text
text, shop sign words, labels, numbers, watermark, logo, character, scenery, floor, UI frame, photorealism, 3D render, clutter, unreadable details, cropped object
```

Other interactables:

- Portals: use the portal animation contract in the VFX section.
- Signs: use terrain prop cell 10, no readable text baked in.
- Chests: `[Assumption]` Not currently a main runtime contract. If added, use `128x128` PNG alpha for static chests or 6-frame `160x160` strips for open animations.
- Doors/switches/checkpoints: `[Assumption]` Use Starfall portal, station, or prop conventions unless code introduces a specific object contract.

## 6. Environmental and Level Asset Instructions

Use soft depth and crisp gameplay surfaces: distant scenery may retain atmospheric detail, while nearby platforms, props, and collision edges stay clearer and less textured. Preserve strong separation around actors and hazards on bright and dark backgrounds. Towns should feel welcoming; dangerous regions can become mysterious or threatening while remaining readable.

### Map Background Contract

- Runtime path: `img/project-starfall/maps/<map-id>.webp`
- Source path: `asset-sources/project-starfall/overhaul-v1/scenery/backgrounds/<map-id>.png`, with exact prompt and ledger entry
- Runtime size: `1280x640`
- Format: WebP, no alpha
- Rendering: 44 complete authored paintings, including four shop interiors shared by 24 regional shops, cover-scaled as bounded panoramas in Canvas and Pixi. They are not edge-periodic textures and must not be tiled by blending unrelated scene edges.
- No text, labels, UI, characters, monsters, or foreground clutter that hides combat.

Map prompt:

```text
Create a 1280x640 Project Starfall side-scroller panoramic background for [MAP_NAME].
Style: Starlit Frontier Fantasy, atmospheric 2D fantasy background, soft detailed distance, simpler clearer nearby surfaces, readable side-scroller depth, clear gameplay lanes, layered parallax feel, no characters, no monsters, no UI, no text. Warm adventure with mystery and credible danger appropriate to the region.
Region: [REGION DESCRIPTION].
Palette: [PALETTE].
Required landmarks: [LANDMARKS].
Context: [OUTDOOR BIOME OR ENCLOSED ROOM PURPOSE]. Playable floor, portal positions and actor clear areas: [POSITIONS AND MATERIALS]. For interiors, keep room displays against the back wall and the lower actor lane clear; no outdoor buildings, sky or meadow dressing.
Composition: foreground gameplay platform areas have clear contact edges and restrained texture, midground supports the theme, background has softer atmospheric depth. Preserve space and contrast for actors and hazard cues. Compose one complete bounded panorama; do not duplicate or blend the scene edges.
Lighting: [TIME/WEATHER/MOOD].
```

Map negative prompt:

```text
text, labels, signs with words, UI, characters, enemies, watermark, logo, photorealism, 3D render, isometric map, top-down map, cluttered foreground, huge object blocking combat lane, hard vertical seam, modern city, sci-fi chrome
```

### Required Map Themes

Generate and maintain backgrounds for these IDs:

`starfall-crossing`, `greenroot-meadow`, `thornpath-thicket`, `bramble-depths`, `rustcoil-ruins`, `gearworks-vault`, `cinder-hollow`, `emberjaw-lair`, `bandit-ridge-camp`, `oreback-quarry`, `ashglass-pass`, `frostfen-outskirts`, `glacier-spine`, `rimewarden-sanctum`, `stormbreak-cliffs`, `astral-archive`, `eclipse-frontier`, `endless-rift`, `rustcoil-outpost`, `cinder-refuge`, `frostfen-camp`, `stormbreak-haven`, `astral-observatory`, `brambleking-court`, `titan-foundry`, `deepcore-core`, `emberjaw-furnace`, `rimewarden-vault`, `stormbreak-aerie`, `astral-stacks`, `eclipse-throne`, plus class trial maps as needed.

Shop interiors use `shop-weapon-interior`, `shop-armor-interior`, `shop-supply-interior` and `shop-special-interior`, selected by vendor type. Shared room art must still suit the shop's function and leave its actual vendor and return portal readable.

### Terrain Atlas Contract

- Runtime path: `img/project-starfall/environment/terrain/<theme-id>.png`
- Size: `512x256`
- Format: PNG alpha
- Layout: 8 columns x 4 rows
- Cell size: `64x64`
- Source authority: `overhaul-v1/scenery/ledger.json` and biome-kit alpha masters; repeatable cells receive recorded horizontal edge treatment during export.

Terrain cell semantics:

| Cells | Meaning |
| --- | --- |
| 0 | ground left cap |
| 1-2 | ground middle variants |
| 3 | ground right cap |
| 4 | platform left cap |
| 5-6 | platform middle variants |
| 7 | platform right cap |
| 8-11 | body fill variants |
| 12-15 | deep body fill variants |
| 16-19 | underside variants |
| 20 | left wall edge |
| 21 | right wall edge |
| 22 | top cap |
| 23-26 | detail overlays |
| 27-30 | long underside variants |
| 31 | shadow/edge support |

Terrain prompt:

```text
Create a Project Starfall terrain atlas for [THEME_ID], 512x256 PNG with 8 columns x 4 rows of 64x64 transparent tiles.
Style: Starlit Frontier Fantasy side-scroller terrain, clean illustrated forms, collision-friendly platform edges, crisp readable top surfaces, restrained material texture and controlled shading that keep actors and hazards distinct, no characters, no enemies, no UI, no text.
Theme materials: [DIRT/STONE/WOOD/BRASS/ICE/VOLCANIC/ASTRAL].
Cell plan: ground caps, ground middles, platform caps, platform middles, body fills, deep body fills, undersides, wall edges, detail overlays, and shadow support.
Top edges must be visually clear and easy to read during combat. Tiles must connect seamlessly horizontally.
```

### Prop Atlas Contract

- Runtime path: `img/project-starfall/environment/props/<theme-id>.png`
- Size: `384x128`
- Format: PNG alpha
- Layout: 6 columns x 2 rows
- Cell size: `64x64`

Prop cell semantics:

| Cell | Prop |
| ---: | --- |
| 0 | grass |
| 1 | bush |
| 2 | tree |
| 3 | rock |
| 4 | flower |
| 5 | small accent |
| 6 | tall accent |
| 7 | crate |
| 8 | crystal |
| 9 | vine |
| 10 | sign |
| 11 | glow |

Prop prompt:

```text
Create a Project Starfall prop atlas for [THEME_ID], 384x128 PNG with 6 columns x 2 rows of 64x64 transparent cells.
Style: Starlit Frontier Fantasy side-scroller props, clean silhouettes, low-profile gameplay-safe details, regional materials and colors, no characters, no enemies, no UI, no text.
Cells in order: grass, bush, tree, rock, flower, small accent, tall accent, crate, crystal, vine, sign without readable words, glow.
Keep each prop centered in its cell with padding. Do not let props cross cell borders.
```

### Ramp Atlas Contract

- Runtime path: `img/project-starfall/environment/ramps/<theme-id>.png`
- Size: `512x128`
- Format: PNG alpha
- Schema: `ramps-v1`
- Use 128px-wide ramp pieces that match the terrain atlas materials.

Ramp prompt:

```text
Create a Project Starfall ramps-v1 atlas for [THEME_ID], 512x128 transparent PNG.
Style: clean side-scroller slope and ramp terrain pieces matching [THEME_ID] terrain materials.
Use collision-friendly silhouettes, clear top edges, seamless side connections, no characters, no text, no UI.
```

### Structure Atlas Contract

- Runtime path: `img/project-starfall/environment/structures/town-landmarks.png`
- Size: `1024x512`
- Format: PNG alpha
- Layout: 4 columns x 2 rows
- Cell size: `256x256`

Cells:

1. `starfallGuildHall`
2. `rustcoilWorkshop`
3. `cinderForge`
4. `frostfenLodge`
5. `stormbreakGate`
6. `astralObservatory`
7. `marketAwning`
8. `lanternArch`

Structure prompt:

```text
Create a Project Starfall town landmark atlas, 1024x512 transparent PNG, 4 columns x 2 rows of 256x256 cells.
Style: Starlit Frontier Fantasy side-scroller landmark props, warm readable fantasy architecture, blue-and-gold Starfall guild motifs, crisp silhouettes, no text labels, no characters, no UI.
Cells in order: Starfall Guild Hall, Rustcoil Workshop, Cinder Forge, Frostfen Lodge, Stormbreak Gate, Astral Observatory, Market Awning, Lantern Arch.
Each landmark must fit inside its cell with padding and read clearly at gameplay scale.
```

### Collision-Friendly Environment Rules

- Platform top edges must contrast with the background.
- Decorative props should not hide enemies, pickups, or the player.
- Hazard visuals must look dangerous before contact.
- Do not bake collision ambiguity into the art. The playable surface should be obvious.
- Repeating terrain cells must share a level contact edge and compatible bevel thickness, with authored variation inside the material. Matching edge colors does not fix height steps; record and review any measured contact-row crop during export. Contact-normalized atlases must opt into `terrainStyle.contactAligned` so both renderers place their top at the collision surface rather than applying the older grass/lip allowance. Verify ground and elevated lanes with visible feet and collision coordinates; Ashglass uses this registration.
- Painted background shelves, stairs and rock ledges must read as depth scenery, without promising nonexistent footholds or exits. Check them against the real platform layout at gameplay scale.
- Use foreground detail sparingly; avoid large dark foreground objects that cover combat lanes.
- Backgrounds can be painterly, but terrain and props must remain clean.

## 7. VFX and Feedback Asset Instructions

### Combat Visual Language v1

This section is authoritative for combat meaning, cues, and asset review, taking precedence over conflicting class/region color or timing examples in older prompt notes. The illustrated migration connects these requirements to shared contact events; the [migration report](ASSET_OVERHAUL_V1.md) distinguishes implemented hooks from validation still pending. Requirements are not a blanket claim of compliance. The production player/enemy contracts above were explicitly migrated; the unchanged review studies remain references.

#### Color, Shape, and Meaning

| Meaning | Main color | Required supporting cue |
| --- | --- | --- |
| Healing | Mint `#62D995` | Rising plus marks and a restorative pulse |
| Damage / danger | Coral `#F06A60` | Sharp contact burst; defined boundaries for dangerous areas |
| Buff / enhancement | Gold `#F2C45E` | Upward chevrons or ascending sparks |
| Shield / protection | Cyan `#63D7E8` | Enclosing shell, shield, or closed arc |
| Debuff / impairment | Violet `#B88AF3` | Downward marks, broken rings, or recognizable status symbols |
| Resource recovery | Blue `#668FFF` | Droplets or particles flowing into the recipient |

- Meaning comes first: use the listed color for an effect's readable core, symbols, and event feedback. Tints and shading can support volume without changing its apparent category. Elemental and character colors remain secondary accents; costumes and scenery do not need recoloring.
- The Oracle's healing uses mint plus marks and a restorative pulse with icy accents. Enemy healing also uses mint: color describes the effect, not whether the player benefits.
- Friendly effects use smooth outer contours; enemy effects use segmented outer contours. Keep the source and recipients identifiable through origin, travel, or a recipient-centered effect. Segment the outer treatment without making an attack's actual boundary ambiguous.
- Poison, slow, and stun share violet but use distinct symbols, such as a droplet, slowing marks, and stars or chains. Their damaging ticks use coral contact feedback. Critical hits increase impact emphasis without switching to a buff or recovery color.
- Safe zones use cyan protection cues. Movement and summons use neutral pearl with character accents unless they also apply a gameplay effect; they do not acquire a healing or buff cue just because they look magical.
- Combined abilities retain every relevant meaning: for example, a mint healing pulse followed by a cyan shield. Match the cue sequence to the actual events; do not imply a delay between effects that resolve together.
- Use shape and motion as well as hue. Give cores and boundaries enough edge contrast to read over bright and dark scenes, and keep effects translucent enough to see actors, feet, ledges, and warnings. Brightness or glow must not erase the category symbol.
- Keep frequent combat effects brief and contained. Give important heals, heavy attacks, and boss abilities stronger emphasis through deliberate size, shape, and timing while preserving the same semantic meanings and warning visibility. Persistent effects retain their required active cue without a continuous large burst.
- Semantic plus marks, shields, chevrons, and status glyphs are intentional effect art. Text labels, numbers, and UI callouts remain runtime-rendered rather than baked into sprite sheets.

#### Preparation, Commitment, Contact, and Recovery

Every action brief states the preparation cue, commitment cue, release/contact event, active duration, and recovery. Timings are measured from action start in milliseconds; frame counts alone are not timing. A windup includes its final commitment window, rather than adding that window afterward.

| Action | Initial windup / event baseline | Minimum final commitment window |
| --- | --- | --- |
| Ordinary enemy melee | About 420 ms before contact | 200 ms |
| Ordinary enemy projectile | About 540 ms before release; contact occurs on collision | 200 ms before release |
| Enemy charge | About 750 ms before movement becomes dangerous | 200 ms |
| Major incoming threat | About 1,000 ms before activation; review authored exceptions individually | 300 ms |
| Normal healing cast | 350 ms of preparation before the restorative pulse applies healing | Specify its recipient/aim behavior in the brief |
| Ordinary player attack | Keep the existing quick response; synchronize the contact cue with the gameplay event | Do not impose enemy windup windows |
| Reactive defense | Protection can apply immediately and must be visible on the same frame | No mandatory windup |

- Ordinary melee and projectile attacks use readable poses and gathering effects. Area attacks, charges, and delayed hazards also show accurate footprints or paths. Avoid filling routine combat with unnecessary range overlays.
- Enemies may adjust aim early, then visibly commit the final direction or area for the window above. A pose settling into its strike or a marker becoming steady must communicate that commitment. Markers cannot silently move after commitment. Homing attacks require an explicit tracking cue rather than pretending their aim is fixed.
- Damage occurs at the visible strike or projectile collision. Healing occurs at the restorative pulse. Gameplay resolution and VFX must share the same event; starting an animation after changing HP does not satisfy this rule. Projectile release is not projectile contact.
- Warning geometry must match the actual affected area or path. On activation, the cue changes clearly from preparation to active danger. An active hazard keeps its danger cue until its hit area is disabled; fading decoration must not imply that an active hazard has become safe.
- Persistent hazards and damage-over-time effects keep recognizable active/status cues. Each tick does not restart a full windup. Newly placed hazards still need their initial warning before becoming dangerous.
- Cancelled actions remove their warning and pending gameplay effect together. Cancellation after release follows the actual projectile/hazard lifecycle: do not hide a warning for an effect that remains active.
- Apply the same event synchronization to player skills, party AI, and enemy support actions. Preserve responsive player controls; document and review any proposed balance change separately from asset generation.

#### Actor and Effect Consistency

- Preserve the approved actor reference, silhouette, face, costume, props, palette, and proportions. Generate actor motion and transparent FX as separate coordinated layers, with explicit anchors and front/behind placement.
- Use one shared scale and a stable foot, hover, hand, or effect-origin anchor as appropriate. Intentional squash, stretch, recoil, and travel are authored motion; per-frame resizing or bounding-box recentering must not introduce accidental motion.
- Require clean, distinct poses with intentional holds and transitions. Reject accidental shape drift, changing accessories, registration jitter, smeared in-between frames, and nearly identical distorted poses used to imitate motion.
- Keep required gameplay animation and effects together in the review. A stable character alone is not a complete healing cast. Preserve approved poses when revising only the effect layer.
- Loops must return seamlessly to their initial pose and effect state. One-shot attacks and casts must recover cleanly to idle; they do not need to loop their impact.
- Choose transparent output or a processor-safe chroma color that does not remove the semantic core or symbol. Keep all cues within their export cells and validate at the runtime display size.

#### First Review Cases and Acceptance

Review the Oracle heal, one melee attack, and one area attack before a broader migration:

| Case | Required evidence |
| --- | --- |
| Oracle heal | Approved actor identity and poses, separate mint healing FX with icy accents, clear enemy ownership and recipient, visible gathering followed by the healing pulse at 350 ms |
| Melee attack | Readable windup pose, visible final aim commitment, sharp coral contact feedback at the hit event, and recovery without registration jitter |
| Area attack | Accurate coral footprint before activation, stable committed area, damage beginning with its active cue, and a boundary that remains visible until danger ends |

- **Meaning:** review at normal gameplay size and without relying on hue; distinguish healing, damage, buffs, protection, impairment, and resource recovery by symbols and motion.
- **Timing:** frame-step preparation through recovery alongside event timestamps or HP state. Verify no damage or healing before its intended contact event; for projectiles, check collision rather than just release. Art-only previews cannot prove runtime synchronization.
- **Prediction:** move the target during preparation and after commitment; check direction, area, cancellation, and explicit homing behavior against the displayed warning.
- **Consistency:** inspect identity, anchors, shared scale, deliberate poses, and applicable loop seams at 100 percent zoom and gameplay size.
- **Readability:** compare bright and dark backgrounds, overlapping abilities, desktop and mobile gameplay scale, and Canvas/Pixi rendering. Critical warnings and category symbols must survive reduced-effects settings and renderer fallbacks.

### Global FX Contract

Global FX sheets:

- Path: `img/project-starfall/animations/fx/<fx-id>-sheet.png`
- Size: `960x160`
- Frame size: `160x160`
- Layout: 6 columns x 1 row
- Format: PNG alpha

Current global FX IDs:

- `slash`
- `cast`
- `arrow-release`
- `party-buff`
- `impact`
- `defeat-burst`

Global FX prompt:

```text
Create a Project Starfall 6-frame transparent VFX sprite strip for [FX_NAME], 960x160 PNG, 6 columns x 1 row, 160x160 per frame.
Style: clean 2D fantasy action VFX, luminous Starfall particles, crisp readable shape, no character body, no text, no UI, no background.
Timing: [PER-FRAME HOLDS IN MS AND PREPARATION, COMMITMENT, RELEASE/CONTACT, ACTIVE, RECOVERY MARKERS]. Match the six cells to this timing; do not infer the gameplay event from a fixed peak-frame number.
Meaning: [EFFECT CATEGORY, SEMANTIC HEX, REQUIRED SYMBOL/MOTION]. Ownership and recipients: [FRIENDLY/ENEMY, SOURCE, RECIPIENTS, OUTER CONTOUR]. Secondary accents only: [CLASS/REGION/ELEMENT MOTIF].
Use the required combat generation brief and Combat Visual Language v1. Preserve the approved actor and generate only the separate transparent FX layer at [ANCHOR, SHARED SCALE, FRONT/BEHIND PLACEMENT].
Keep the effect centered, contained inside each frame, and readable over dark or bright backgrounds.
```

### Skill FX Contract

- Path: `img/project-starfall/animations/combat-fx/skills/<skill-file-id>-sheet.png`
- Source authority: native editable `build/lib/starfall-combat-language-art.js`, generated through `build/generate-project-starfall-combat-fx.js`; `overhaul-v1/fx/ledger.json` records output hashes
- Size: `960x640`
- Frame size: `160x160`
- Layout: 6 columns x 4 rows
- Format: PNG alpha

Rows:

| Row | Action |
| --- | --- |
| 0 | cast |
| 1 | projectile |
| 2 | impact |
| 3 | area |

Skill FX prompt:

```text
Create a Project Starfall skill VFX source sheet for [SKILL_NAME], 960x640 target, 6 columns x 4 rows, 160x160 frames.
Style: clean 2D fantasy combat VFX, Starlit Frontier Fantasy, luminous but readable, no character body, no UI, no text, no background clutter.
Rows:
1 cast: hand/focus origin burst or magic circle.
2 projectile: right-moving projectile or travel trail, horizontally level.
3 impact: hit spark, burst, slash, or explosion.
4 area: lingering zone, aura, trap, rune field, or splash effect.
Meaning: [EFFECT CATEGORY, SEMANTIC HEX, REQUIRED SYMBOL/MOTION]. Secondary accents only: [CLASS COLOR AND ELEMENTAL MOTIF].
Include the required combat generation brief: [OWNERSHIP, RECIPIENTS, PHASE TIMINGS, CONTACT EVENT, ACTIVE DURATION, RECOVERY, REFERENCE LOCKS, ANCHOR, SHARED SCALE, LAYERING]. Travel and contact must be separate events for projectiles.
Keep every frame centered and inside cell bounds. Use genuine alpha and empty gutters, preserving all semantic colors; new raster replacements require an explicit source-owner update.
```

### Basic Attack FX Contract

- Path: `img/project-starfall/animations/combat-fx/basic/basic-<class>-sheet.png`
- Size: `960x640`
- Frame size: `160x160`
- Layout: 6 columns x 4 rows
- Rows: `cast`, `projectile`, `impact`, `trail`

Generate for `fighter`, `mage`, and `archer` first. Advanced classes may reuse or extend their base class motifs.

### Enemy Combat FX Contract

- Path: `img/project-starfall/animations/combat-fx/enemies/<enemy-file-id>-sheet.png`
- Size: `960x800`
- Frame size: `160x160`
- Layout: 6 columns x 5 rows
- Rows: `telegraph`, `melee`, `projectile`, `buff`, `impact`

Enemy FX prompt:

```text
Create a Project Starfall enemy combat FX sheet for [ENEMY_NAME], 960x800 target, 6 columns x 5 rows, 160x160 frames.
Style: clean readable 2D fantasy monster attack VFX, separate transparent effects with no character body, no UI, no text, transparent-ready.
Rows: telegraph warning, melee contact effect, projectile travel, buff/support effect, impact effect.
Meaning per row: [EFFECT CATEGORY, SEMANTIC HEX, REQUIRED SYMBOL/MOTION]. Identify the actual support effect (heal, shield, enhancement, impairment, or summon); the runtime row name buff does not determine its meaning or color.
Ownership: enemy segmented outer contour, clear [SOURCE AND RECIPIENTS]. Secondary accents only: [ENEMY REGION AND ELEMENT].
Include the required combat generation brief with phase timing, release/contact event, active duration, recovery, approved references, anchor, shared scale, and layer placement.
```

### Enemy Projectile Contract

- Path: `img/project-starfall/animations/enemy-projectiles/<projectile-id>-sheet.png`
- Current example: `bandit-knife-sheet.png`
- Size: `192x64`
- Frame size: `64x64`
- Layout: 3 columns x 1 row
- Format: PNG alpha

Projectile prompt:

```text
Create a 3-frame Project Starfall enemy projectile strip for [PROJECTILE_NAME], 192x64 PNG, 3 columns x 1 row, 64x64 frames.
Style: clean 2D side-scroller projectile, right-moving, centered, transparent background, readable silhouette, no text, no UI.
Frames: launch/travel spin, mid travel, bright leading frame or trailing motion cue.
Apply the required combat generation brief: [MEANING, SEMANTIC HEX/SYMBOL, OWNERSHIP CONTOUR, SECONDARY ELEMENT, ORIGIN, SHARED SCALE]. Separate release from collision/contact; show [TRAVEL/TRACKING CUE] and generate impact FX separately.
```

### Portal Contract

- Path: `img/project-starfall/animations/portals/<portal-id>-sheet.png`
- IDs: `standard`, `boss`, `locked`
- Frame size: `160x160`
- Layout: 6 columns x 1 row
- Format: PNG alpha

Portal prompt:

```text
Create a 6-frame Project Starfall animated portal strip for [STANDARD/BOSS/LOCKED] portal, 960x160 PNG, 160x160 frames.
Style: luminous Starfall fantasy portal, clean circular/arched silhouette, cyan/gold for standard, stronger red/gold or violet for boss, sealed dim glyphs for locked.
No characters, no text, no UI, transparent background. Keep base centered and frame-to-frame shape stable.
```

### Pet Contract

- Path: `img/project-starfall/animations/pets/starfall-fox-sheet.png`
- Layout: 6 columns x 6 rows, 160px frames, `960x960`; authored source and output hashes live in `overhaul-v1/players/ledger.json`.
- Rows: `idle`, `run`, `jump`, `fall`, `loot`, `teleport`

Pet prompt:

```text
Create a Project Starfall pet animation sheet for a starfall fox, 6 columns x 6 rows, 160px frames, transparent PNG.
Style: cute compact 2D side-scroller companion, fox silhouette with starry cyan/gold accents, readable at small scale, clean outline.
Rows: idle, run, jump, fall, loot pickup, teleport blink.
No text, no UI, no scenery, no cast shadow.
```

### VFX Timing Notes

These are visual envelopes within the named export contract, not a replacement for the brief's event times or frame holds. Contact FX start at the gameplay contact event; later brightness peaks must not postpone the first visible contact cue.

- Hit sparks: 4 to 6 source poses where the export contract allows, contact visible immediately, then a brief brightness peak and dissipation.
- Dust puffs: 6 frames, expand then fade.
- Landing effects: 6 frames, horizontal dust squash outward.
- Jump effects: 6 frames, small burst under feet.
- Attack slashes: 6 frames, anticipation arc then bright contact frame.
- Projectile impacts: 6 frames, readable semantic contact burst with secondary elemental accents.
- Enemy death effects: 6 frames, collapse burst, no gore.
- Pickup glows: 6 frames if animated, gentle pulse only.
- UI feedback effects: 6 frames, restrained sparkle or pulse, no excessive bloom.

## 8. UI Asset Instructions

### UI Style

Project Starfall UI should feel like a practical fantasy RPG client:

- Dark navy frames.
- Parchment/cream panels.
- Gold trim.
- Cyan Starfall highlights.
- Dense but readable controls.
- Illustrated fantasy icons.
- No modern sci-fi chrome.
- No generic web-app card visuals.
- No oversized marketing hero art inside gameplay UI.

### Existing UI Dimensions

- `img/project-starfall/ui/splash-screen.png` - `1672x941`, no alpha
- `img/project-starfall/ui/start-screen.png` - `1672x941`, no alpha
- `img/project-starfall/ui/character-select-screen.png` - `1672x941`, no alpha
- `img/project-starfall/ui/character-slot-pedestal.png` - `512x160`, PNG alpha
- `img/project-starfall/ui/menu-icons/<icon-id>.png` - `64x64`, PNG alpha

### Menu Icon IDs and Filenames

Generate menu icons for these runtime IDs and registry filenames:

| Runtime ID | Filename |
| --- | --- |
| `character` | `character.png` |
| `equipment` | `equipment.png` |
| `partyPanel` | `party-panel.png` |
| `inventory` | `inventory.png` |
| `skills` | `skills.png` |
| `quests` | `quests.png` |
| `worldmap` | `worldmap.png` |
| `monsters` | `monsters.png` |
| `shop` | `shop.png` |
| `upgrade` | `upgrade.png` |
| `daily` | `beta.png` |
| `cashShop` | `cash-shop.png` |
| `beta` | `beta.png` |
| `guide` | `guide.png` |
| `log` | `log.png` |
| `settings` | `settings.png` |
| `keybinds` | `keybinds.png` |
| `admin` | `admin.png` |
| `logout` | `logout.png` |

Menu icon prompt:

```text
Create a 64x64 transparent Project Starfall menu icon for [ICON_ID].
Style: clean fantasy RPG UI icon, dark navy/gold/cyan Starfall palette, readable silhouette, simple symbol, subtle bevel, no text, no numbers, no background panel unless the symbol requires a small internal shape.
The icon must be clear at 32px and 64px.
```

### Skill Icon Contract

- Base skill path: `img/project-starfall/skills/base/<skill-id>.png`
- Advanced skill path: `img/project-starfall/skills/advanced/<class-id>/<skill-id>.png`
- Source authority: `overhaul-v1/icons/catalog.json`, class-group batches and exact raw sources; 85 active skill icons
- Size: `256x256`
- Format: PNG alpha

Skill icon prompt:

```text
Create a 256x256 transparent Project Starfall skill icon for [SKILL_NAME].
Style: clean fantasy RPG ability icon, Starlit Frontier Fantasy, one main symbol plus one supporting effect, crisp silhouette, high contrast, readable at 32px, no text, no numbers, no UI frame.
Class motif: [CLASS].
Gameplay intent: [MOBBING/BOSSING/SUPPORT/MOBILITY/CONTROL].
Apply the required combat generation brief, marking animation timing not applicable for this static icon. Use [FUNCTIONAL CATEGORY HEX AND REQUIRED SYMBOL], with [CLASS COLORS AND WEAPON/MAGIC/ELEMENT MOTIF] as secondary accents. Movement or summons without an additional gameplay effect use neutral pearl with character accents.
```

Skill icon visual rules:

- Mobbing skills: wide arcs, multiple sparks, area cues.
- Bossing skills: narrow focused strike, precision mark, single bright impact.
- Support skills: use the actual category's symbol (healing plus, protection shield, or enhancement chevrons), with a secondary aura/circle/pulse if needed.
- Mobility skills: arrow, dash trail, step burst.
- Control skills: trap, rune, snare, barrier, lock shape.

### Large UI Screen Prompt

Use for splash, start, and character select screen art:

```text
Create a 1672x941 Project Starfall UI background screen for [SCREEN_NAME].
Style: Starlit Frontier Fantasy, painterly but clean fantasy RPG interface background, dark navy frame areas, warm parchment/gold/cyan accents, fallen-star motif, no readable text baked into the image, no logos, no watermarks.
Composition: leave clear areas for runtime UI buttons, character slots, titles, and panels. Do not include fake buttons or fake text.
```

### HUD and Panel Asset Rules

If generating standalone HUD or panel art:

- Health indicators: red heart/bar with gold/dark frame.
- Resource indicators: class-specific color, with Mage energy cyan, Fighter momentum gold/red, Archer focus green/gold.
- Buff/debuff icons: `64x64`, transparent, with semantic color and symbol from Combat Visual Language v1. Resource restoration feedback uses blue even when a persistent resource meter has a class-specific color.
- Inventory slots: dark navy inset, gold/parchment rim, no item baked in.
- Dialogue boxes: parchment center, dark navy/gold frame, no text baked in.
- Buttons: dark navy or parchment base, gold trim, hover/active variants if coded.
- Fonts/text: do not generate raster text unless the UI asset is explicitly a title image. Runtime text should be HTML/CSS.

## 9. Prompt Engineering Rules

### Required Combat Generation Brief

Before generating or revising any combat actor, skill icon, or FX asset, fill this brief using [Combat visual language v1](#combat-visual-language-v1). Keep metadata outside the generated image. For non-combat assets, retain the relevant existing template. Reusable prompt forms live in [prompts/README.md](../../asset-sources/project-starfall/prompts/README.md).

- **Meaning:** action and effect category (or all categories for a combined ability), semantic hex, required symbol, and motion cue. Name the actual gameplay effect rather than just the runtime animation row.
- **Ownership and targets:** friendly/enemy contour, source, recipients, origin/travel/recipient placement, and secondary class/region/element accents.
- **Timing:** preparation pose; commitment cue and aim-lock time; release and contact event; active duration; recovery; loop/one-shot behavior; per-frame holds in milliseconds. State explicitly when a static icon or purely cosmetic motion has no contact event.
- **Geometry and lifecycle:** affected area/path, anchor, tracking or homing behavior, cancellation behavior, and what remains visible while danger/status is active.
- **Reference locks:** approved image references and identity details, proportions, fixed palette, one shared scale, registration anchor, and separately generated actor/FX layer placement.
- **Style and movement:** clean illustrated finish, character personality, anatomy-appropriate articulation/weight, nearby scenery contrast, and everyday versus major-action effect emphasis. Use the approved samples for both appearance and motion without inheriting historical pixel-grid or palette-count restrictions.
- **Export and review:** exact existing runtime layout and target size, requested layer/row, and applicable acceptance evidence. A review study with a different pose count is not an import-ready replacement.

### Master Style Prompt

Append this to every non-UI asset prompt unless it conflicts with the asset type:

```text
Project Starfall Starlit Frontier Fantasy, original 2D side-scroller fantasy RPG asset, clean illustrated finish, crisp readable contours, controlled shading, clear features, little visual noise, expressive characters with varied personalities and preserved compact identities, warm adventure with mystery and credible danger, luminous fallen-star magic accents, blue-and-gold guild motifs, practical crafted fantasy materials, readable at gameplay scale, consistent upper-left/front lighting, no copied IP.
```

For backgrounds, replace character and sprite-contour language with:

```text
soft detailed atmospheric distance, simpler crisp nearby platforms and props, clear gameplay lanes and contact edges, strong separation around characters and hazards, readable foreground/midground/background depth
```

### Master Negative Prompt

Use this for all generated assets:

```text
text, labels, numbers, watermark, signature, logo, fake UI, unrelated background clutter, photorealism, 3D render, isometric gameplay view, top-down gameplay view, copied IP, copied MapleStory sprite, inconsistent style, warped anatomy, extra limbs, missing limbs, broken hands, broken feet, melted outlines, noisy unreadable details, over-bloom, muddy colors, cropped subject, cell overflow, inconsistent lighting, inconsistent proportions, changing costume, changing weapon, changing face
```

Add these negatives for source sheets:

```text
guide grid color inside artwork, chroma key color inside artwork, row labels, column labels, frame numbers, character crossing grid lines
```

### Variables To Change By Asset Type

Change only these variables between prompts:

- Asset name.
- Asset file ID.
- Class/enemy/item/world theme.
- Palette.
- Gameplay role.
- Animation row descriptions.
- Required dimensions and sheet layout.
- Chroma background color when required.
- Weapon, material, element, or region motif.
- Completed combat brief fields for the requested action; semantic colors and symbols are selected from the standard, not invented per class.

### Details To Lock Across Prompts

Keep these locked:

- Project Starfall style phrase.
- Side-scroller perspective for gameplay assets.
- Right-facing orientation for sprites.
- Transparent runtime backgrounds.
- No text/logos/watermarks.
- Starfall cyan/gold identity accents, subordinate to functional colors in combat effects.
- Clean silhouette and outline rules.
- Upper-left/front lighting for sprites.
- Exact sheet row order.
- Exact file path and naming.

### Consistent Characters Across Animations

For each player class:

1. Generate or choose one approved base reference.
2. Write a short character lock: hair/helmet, face visibility, outfit, weapon, colors, accessory.
3. Use that exact lock in every prompt.
4. Generate coherent eight-pose action strips or action-pair masters; pack the 8x10 runtime sheet through the player importer.
5. If fixing a row, use the approved sheet as image reference and regenerate only that row with the same lock.
6. Do not accept costume drift between rows.

### Consistent Enemies Across Animations

For each enemy:

1. Lock silhouette, number of limbs, head shape, horns/antlers/core, palette, and material.
2. Generate six distinct poses for each of the eight actions, then measure and pack the 6x8 runtime sheet through the enemy importer.
3. Keep idle, move, telegraph, attack, projectile, buff, hit, and defeat as the same creature.
4. Reject sheets where the enemy changes species, grows extra parts, or changes armor.

### Prompting Sprite Sheets

Always state:

- Exact canvas size.
- Exact columns and rows.
- Exact frame size.
- Exact row order.
- One subject per cell.
- No labels.
- Consistent baseline or hover center.
- Empty transparent gutters and no drawn grid for illustrated actors, scenery, and standalone FX.
- An extraction key only for an icon batch whose owner explicitly supports and records it.

### Prompting Transparent Backgrounds

Use:

```text
transparent background, alpha PNG, no scenery, no floor, no shadow, no UI, no text
```

For an icon batch explicitly configured for green-key extraction, use:

```text
flat solid #00ff00 background outside the artwork for chroma key removal, with no #00ff00 pixels inside the artwork
```

For green actors, plant/ooze subjects, or semantic effects that conflict with a key, require genuine alpha and validate it. Do not switch to a historical magenta/cyan-grid pipeline unless that source-owner change is explicitly implemented and reviewed.

### Avoiding Frame Flicker

- Keep the same seed/reference for all frames.
- Generate sheets rather than separate frames.
- Avoid prompts that describe a new costume in each row.
- Avoid too many tiny accessories.
- Generate combat VFX separately from the character body and coordinate anchors and timing using the required brief.
- Use simple strong shapes for weapons and class props.
- Run visual review before importing.

## 10. Asset Review Checklist

Use this checklist before importing any generated asset.

### Style

- Matches the [confirmed Starlit Frontier Fantasy direction](#confirmed-art-direction-and-references): clean illustrated finish, expressive varied characters, warm adventure, and readable danger.
- Does not look like copied third-party IP.
- Does not look photorealistic or 3D-rendered.
- Uses the correct regional/class palette for identity and the functional palette for combat meaning.
- Uses clean outlines for sprites/icons.
- Has no obvious AI artifacts.

### Technical

- Correct file format.
- Correct dimensions.
- Correct frame count.
- Correct sheet layout.
- Correct row order.
- Correct transparent background or processor-safe chroma background.
- No cyan guide pixels in runtime output.
- No chroma pixels in runtime output.
- No text, labels, signatures, logos, watermarks, or frame numbers.
- Correct filename and folder placement.

### Animation

- Baseline or hover center stays consistent.
- Horizontal body registration uses reviewed anatomical anchors, including when source gutters are uneven; grid centers or bounds are not sufficient proof.
- Independent packed-pixel or landmark checks cover every accepted action and its loop seam or recovery transition, with intended movement preserved and review scope recorded.
- Scale does not flicker.
- Costume, weapon, face, and silhouette stay consistent.
- Attack tells are readable.
- Hit and defeat frames are distinct.
- Looping rows loop cleanly.
- Non-looping rows have clear preparation, commitment, contact, and recovery with explicit event timing.
- No limb, weapon, VFX, or prop crosses cell boundaries.
- Motion fits the character's anatomy and weight while preserving identity, rather than applying the same squash and bounce to every actor.

### Gameplay Readability

- Silhouette reads at in-game size.
- Player class identity is clear.
- Enemy role is clear.
- Hazard or attack direction is clear.
- Projectile direction is clear.
- UI icons read at 32px and 64px.
- Terrain top edges are collision-readable.
- Background does not hide combat.
- Combat assets pass the meaning, timing, prediction, consistency, and readability checks in [First review cases and acceptance](#first-review-cases-and-acceptance), including reduced-effects and both-renderer checks during integration.

### Cohesion Review Before Wider Production

Place a compact player, an expressive creature, and their coordinated effects against representative gameplay scenery before extending the direction across the game. Review at actual gameplay size, not only on isolated enlarged sheets.

- Check a coherent finish across player, creature, props, and effects, with atmospheric background depth and clearer playable surfaces.
- Confirm recognizable identities and movement that fits each character's anatomy and weight.
- Compare bright and dark scenes for terrain contact edges, actor silhouettes, and accurate danger cues.
- Play complete actions and applicable loops to catch registration jitter, accidental shape changes, and imperfect transitions; clean up drifting landmarks even on approved direction samples.
- Overlap ordinary and major effects to verify restrained routine feedback and proportionate emphasis for important actions, without hiding warnings or feet.

This review complements combat-event and renderer validation. The guide records requirements; it does not certify existing assets or untested future samples as compliant.

### Documentation Match

- Asset matches this guide.
- Asset matches the GDD and Starfall asset prompt notes.
- Asset matches `js/games/project-starfall/data/assets.js`.
- Asset matches the relevant runtime contract in `js/games/project-starfall/data/animations.js`.
- New folders, if any, are included in `build/copy-to-public.js`.

## 11. Import and Validation

After generating assets, import through the domain owner in [Illustrated production ownership](#illustrated-production-ownership). Preserve its exact prompt, source hash, measured extraction and output hash. Then run validation and the relevant tests:

```bash
npm run validate:project-starfall-assets
node build/verify-project-starfall-overhaul-icons.js
node build/process-project-starfall-overhaul-scenery.js --validate
node build/audit-project-starfall-overhaul.js --complete
npm run test:starfall:assets
npm test
```

For player sheets, the compatibility validator checks the current source-owned format; use the new player importer only when intentionally rebuilding its output and attachment metadata:

```bash
node build/process-project-starfall-player-ai-assets.js --validate
```

For enemy actors, run the new processor without `--import` to produce measured review output, inspect it, then use `--import` only for an accepted source. The legacy compact command is retained for compatibility and must defer to the illustrated owner:

```bash
node build/process-project-starfall-overhaul-enemies.js --enemy <file-id>
node build/process-project-starfall-compact-bandits.js --validate
```

Enemy activation automatically regenerates [combat body masks](#enemy-combat-body-masks); for another accepted import workflow use `node build/generate-project-starfall-enemy-hurtboxes.js` explicitly. Require `--check` and `npm run test:starfall:hitboxes` either way. The freshness check compares production hashes and deterministically derived mask data; passing visual registration checks alone is insufficient.

If a generated asset fails validation, fix the source image rather than weakening the runtime contract.
