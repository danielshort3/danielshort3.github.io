# Project Starfall Asset Generation Guide

This guide is the asset-generation contract for Project Starfall. It is based on the current GDD, Starfall asset prompt notes, runtime data modules, build processors, CSS/UI tokens, and the existing asset folders.

Assumptions are marked with `[Assumption]`. Everything else should be treated as a current project requirement.

## 1. Project-Specific Visual Style Rules

### Core Style

Use the Project Starfall style: **Starlit Frontier Fantasy**.

All generated assets must feel like they belong in a charming 2D side-scrolling browser action RPG with:

- Clean fantasy silhouettes.
- Compact readable gameplay forms.
- Warm guild-town frontier materials.
- Luminous fallen-star magic.
- Blue-and-gold Starfall motifs.
- Crafted wood, stone, cloth, brass, crystal, lanterns, runes, portals, and practical magic devices.
- Painterly background depth, but sprite assets with crisp cel-shaded readability.

Do not copy MapleStory or any other existing game. MapleStory is only a high-level reference for side-scroller readability and action coverage.

### Camera and Perspective

- Gameplay sprites: side-scroller camera, right-facing by default.
- Characters and enemies: slight readable 3/4 side-view is allowed, but the pose must still read as side-scroller gameplay.
- UI portraits: chest-up or full-body presentation inside a square frame, 3/4 fantasy portrait angle.
- Maps/backgrounds: side-scroller panoramic view, not isometric or top-down unless generating the world-map atlas.
- World-map atlas: painterly top-down fantasy map, no labels baked into the image.

### Character Scale and Proportions

- Player classes: compact heroic proportions, about 3.5 to 4 heads tall.
- Faces: minimal detail at gameplay scale, readable hair/helmet/costume shape.
- Hands, weapons, shields, hats, and class props can be slightly oversized for readability.
- Enemies: compact silhouettes normalized by the enemy processor. Bosses should feel larger through shape, posture, horns, shoulders, wings, or aura, not by exceeding the cell.
- Items/icons: object fills most of the icon space but leaves transparent padding.

### Line Weight and Outlines

- Gameplay sprites: crisp dark contour outline, visually around 2 to 4 px at 160px frame scale.
- Enemy sprites: crisp outline, readable at 128px frame scale.
- Item icons: thin dark rim or painted edge separation, no heavy UI border unless the asset is itself a UI frame.
- Backgrounds: no hard sprite outline; use painterly edge control and depth separation instead.

### Palette

Use the existing Starfall UI and world palette as the base:

- Ink/dark frame: `#1c2631`, `#102033`, `#071323`
- Parchment panels: `#f2dfbf`, `#f7ebd2`
- Gold accents: `#d8a531`, `#f5cf72`
- Starfall cyan/blue: `#2aaad3`, `#7bdff2`, `#9be7ff`
- Green/nature: `#4e9d61`
- Red/damage/fire: `#d9584a`
- Teal/magic: `#2aa79a`

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

- Sprites: soft cel shading with a controlled painterly finish.
- Backgrounds: painterly fantasy lighting with depth gradients and atmospheric perspective.
- UI: clean illustrated fantasy UI, not noisy or over-rendered.
- `[Assumption]` Use upper-left/front lighting for gameplay sprites unless a specific VFX or map calls for a different light source.
- Do not let rim lights or glow effects erase the silhouette.

### Pixel Density and Resolution Style

Project Starfall is not strict low-resolution pixel art. Generate high-quality 2D painted/cel-shaded source art that remains readable when processed into the current runtime sizes.

- Player and FX frame size: 160px.
- Enemy compact frame size: 128px.
- Item/menu icons: 64px.
- Skill icons: 256px.
- Map backgrounds: 1280x640 WebP.

Avoid tiny noisy detail that collapses at these sizes.

### Transparency and Background Rules

- Runtime sprites, icons, FX, props, stations, portraits, and UI cutouts must have transparent backgrounds.
- Source sheets may use the project processors' chroma backgrounds:
  - Flat `#00ff00` green for most source sheets.
  - Flat `#ff00ff` magenta when green subjects would collide with chroma key, especially ooze, slime, plant, moss, thorn, vine, briar, bloom, or cap assets.
  - `#00ffff` cyan guide grid lines where required by the processor.
- Runtime files must not contain chroma pixels or cyan guide pixels.
- Map backgrounds and splash/start screens are full rectangular images and should not have alpha.

### Animation Readability

- Every animation must read clearly in silhouette before color is considered.
- Attack tells must have obvious windup frames.
- Player attack and skill rows must show anticipation, action, and recovery.
- Enemy attacks must make the danger type obvious before contact or projectile release.
- Keep feet, baselines, and body scale stable across frames.
- Do not add camera shake, motion blur, UI labels, text, hit numbers, or particles that cross cell borders.

### How To Avoid AI-Generated Artifacts

- Generate complete sprite sheets in one pass whenever possible.
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
- Use the regional palette for biome assets and the class palette for class assets.
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
- For processor source sheets, use cyan `#00ffff` guide lines only for grid lines, never inside the artwork.
- Do not place action labels or row labels in the image.

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
asset-sources/project-starfall/players/classes/
asset-sources/project-starfall/players/equipment/
asset-sources/project-starfall/enemies/compact/
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

## 3. Character Asset Instructions

### Current Player Runtime Contract

All player class animation sheets use:

- Runtime path: `img/project-starfall/animations/players/<file-id>-sheet.png`
- Source path: `asset-sources/project-starfall/players/classes/<file-id>-source.png`
- Portrait path: `img/project-starfall/characters/<file-id>.png`
- Sheet size: `960x1600`
- Frame size: `160x160`
- Layout: 6 columns x 10 rows
- Background: transparent in runtime PNG
- Facing: right-facing
- Portrait size: `320x320`, transparent PNG

Rows must be exactly:

| Row | Action | Frames | FPS | Loop |
| --- | --- | ---: | ---: | --- |
| 0 | idle | 6 | 6 | yes |
| 1 | run | 6 | 12 | yes |
| 2 | jump | 6 | 10 | no |
| 3 | fall | 6 | 8 | no |
| 4 | climb | 6 | 8 | yes |
| 5 | basic | 6 | 14 | no |
| 6 | skill | 6 | 12 | no |
| 7 | party | 6 | 12 | no |
| 8 | hit | 6 | 16 | no |
| 9 | defeat | 6 | 9 | no |

Do not add a new runtime row unless `js/games/project-starfall/data/animations.js` and the processing scripts are updated.

`[Assumption]` Interaction animation currently maps best to the `party` row for friendly interactions or the `idle` row for neutral interact prompts. If a true `interact` animation is required later, generate it as a separate 6-frame source strip first and update the runtime contract before import.

### Required Player Class Files

Generate or maintain sheets and portraits for these file IDs:

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

Style: Starlit Frontier Fantasy, clean cel-shaded 2D game sprite, crisp dark contour outline, compact 3.5 to 4 heads tall proportions, readable silhouette, subtle painterly color, luminous fallen-star magic accents, no copied IP, no MapleStory costume copying.

Character lock: [EXACT HAIR/HELMET], [EXACT FACE VISIBILITY], [EXACT OUTFIT], [EXACT WEAPON OR FOCUS], [CLASS COLORS], [SIGNATURE ACCESSORY]. Keep these identical in every frame.

Sheet layout: 6 columns x 10 rows, 160px frame target, one centered character per cell, right-facing, transparent-ready, consistent baseline, generous padding, no frame labels.

Rows in order:
1 idle: subtle breathing and cloth/hair motion.
2 run: alternating stride cycle with clear contact and passing poses.
3 jump: crouch takeoff, upward lift, airborne stretch.
4 fall: downward bracing, cloak/hair rising, landing-ready posture.
5 climb: hands reaching and pulling on a ladder or ledge without drawing the ladder.
6 basic: [CLASS BASIC ATTACK], windup, strike/release, recovery.
7 skill: [CLASS SIGNATURE SKILL], charge, cast/release, follow-through.
8 party: supportive buff or rally gesture with small Starfall aura.
9 hit: recoil, flinch, stagger, recover.
10 defeat: collapse/downed pose, readable but non-gory.

Background: transparent or processor-safe flat background only. No text, no labels, no numbers, no watermark, no UI, no scenery, no cast shadow, no guide color inside art.
```

### Master Player Negative Prompt

```text
text, labels, numbers, watermark, signature, logo, UI, speech bubble, background scenery, floor shadow, frame labels, isometric view, top-down view, front-facing idle only, photorealism, 3D render, low-quality pixel mush, noisy texture, inconsistent costume, changing hair, changing weapon, changing face, extra limbs, missing limbs, warped hands, broken feet, melted outline, flickering proportions, inconsistent lighting, copied MapleStory sprite, copyrighted character, cell overflow, cropped weapon, cyan guide color inside artwork, chroma key color inside artwork
```

### Player Animation Rows

| Action | Pose Requirements | Motion Principles | Review Checklist | Filename |
| --- | --- | --- | --- | --- |
| Idle | Stable stance, subtle breathing, weapon held naturally | Small loop, no foot sliding | Same baseline, no costume drift, frame 6 loops to frame 1 | Row 0 in `<file-id>-source.png` |
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
| Party | Buff, rally, shield, or supportive pulse | Clear supportive intent | Aura stays inside cell | Row 7 |

### Class-Specific Attack/Skill Prompt Inserts

Use these inserts in the master prompt:

- Fighter basic: `short sword slash with grounded stance and momentum trail`.
- Fighter skill: `wide guarded power strike with gold impact arc`.
- Mage basic: `small cyan arcane bolt cast from hand or focus`.
- Mage skill: `larger star-rune burst with a bright casting circle`.
- Archer basic: `quick bow shot with clear release pose`.
- Archer skill: `focused multi-arrow or piercing shot with gold-green aim line`.
- Guardian basic: `shield bash or guarded weapon strike`.
- Guardian skill: `oath barrier pulse and shield-wall stance`.
- Berserker basic: `heavy cleave with red momentum arc`.
- Berserker skill: `rage surge, two-handed smash, crimson impact`.
- Duelist basic: `fast precise cut with silver slash arc`.
- Duelist skill: `flash-step flourish with repeated blade afterimage`.
- Fire Mage basic: `small firebolt with orange ember trail`.
- Fire Mage skill: `inferno burst or wildfire cast with heat aura`.
- Rune Mage basic: `rune-marked arcane shot`.
- Rune Mage skill: `ground glyph detonation with cyan runes`.
- Storm Mage basic: `chain-bolt hand cast`.
- Storm Mage skill: `stormfront charge with lightning arcs`.
- Sniper basic: `aimed arrow shot with steady posture`.
- Sniper skill: `one perfect shot, narrow gold aim flash`.
- Trapper basic: `quick shot or snare toss`.
- Trapper skill: `trap deployment with field kit and green snare cue`.
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

All compact enemy sheets use:

- Runtime path: `img/project-starfall/animations/enemies/<file-id>-compact-sheet.png`
- Source path: `asset-sources/project-starfall/enemies/compact/<file-id>-compact-source.png`
- Portrait path: `img/project-starfall/enemies/<file-id>.png`
- Sheet size: `384x1024`
- Frame size: `128x128`
- Layout: 3 columns x 8 rows
- Facing: right-facing
- Runtime background: transparent
- Source guide: cyan `#00ffff` grid lines
- Source chroma: `#00ff00` for most enemies, `#ff00ff` for green/plant/ooze enemies

Rows must be exactly:

| Row | Action | Source Frames | Runtime Notes |
| --- | --- | ---: | --- |
| 0 | idle | 3 | Breathing, bobbing, stance |
| 1 | move | 3 | Hop, walk, crawl, fly, or charge preparation |
| 2 | telegraph | 3 | Clear attack warning |
| 3 | attack | 3 | Melee, bite, slash, slam, charge contact |
| 4 | projectile | 3 | Throw, spit, cast, shoot, or no-op pose if enemy has no projectile |
| 5 | buff | 3 | Heal, shield, enrage, phase, summon, or special |
| 6 | hit | 3 | Runtime may use one frame, but source still uses 3 cells |
| 7 | defeat | 3 | Collapse, dissolve, shatter, burn out, non-gory |

### Master Enemy Sheet Prompt

```text
Create one original Project Starfall compact enemy animation source sheet for [ENEMY_NAME].

Style: Starlit Frontier Fantasy, charming 2D side-scroller RPG monster, clean cel-shaded sprite, crisp dark contour outline, readable silhouette at 128px, compact proportions, no copied IP, no text.

Enemy design: [VISUAL DESCRIPTION]. Gameplay role: [ROLE]. Attack tell: [ATTACK TELL]. Regional palette: [REGION PALETTE].

Sheet layout: 3 columns x 8 rows, 128px frame target, right-facing, one enemy centered per cell, consistent baseline or hover center, generous padding. Use cyan #00ffff guide grid lines. Use flat [#00ff00 or #ff00ff] processor-safe background outside the art.

Rows in order:
1 idle: 3 readable idle/breath/bob frames.
2 move: 3 movement frames.
3 telegraph: 3 warning frames before the main attack.
4 attack: 3 strike/contact frames.
5 projectile: 3 projectile launch/cast frames or a clear no-projectile special pose.
6 buff: 3 support/enrage/shield/heal/special frames.
7 hit: 3 recoil frames.
8 defeat: 3 collapse/dissolve/shatter frames, non-gory.

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

Admin-only bandit comparison variants may exist as `bandit-cutter-direct`, `bandit-cutter-reference`, `bandit-cutter-hybrid`, and `bandit-cutter-puppet`. Generate those only for pipeline tests, not for the production enemy library.

### Enemy Archetype Readability Rules

- Hoppers: show squash before hop and stretch at launch.
- Bruisers/tanks: show weight through low stance and delayed impact.
- Turrets: keep base fixed; telegraph through glowing pod/barrel/lance.
- Skirmishers: use lean, crouch, and quick recoil poses.
- Chargers: telegraph must be long and unmistakable.
- Blockers: shield/guard must be readable in idle and telegraph.
- Flyers: maintain hover center instead of ground baseline.
- Healers/support: buff row must show outward aura, spores, cloud ring, or runes.
- Elites: silhouette can be more complex but must remain readable at 128px.
- Bosses: use crown/core/armor/wing/aura to imply status, not oversized cells.

## 5. Item, Pickup, and Interactable Asset Instructions

### Item Icon Runtime Contract

- Final icon path: `img/project-starfall/items/icons/<item-id>.png`
- Size: `64x64`
- Format: PNG with alpha
- Source path: `img/project-starfall/items/source/<sheet-name>.png`
- Processed sheet path: `img/project-starfall/items/sheets/<sheet-name>-sheet.png`
- Style: clean fantasy item icon, centered, readable, no UI frame unless the item is itself a coupon/card.

### Item Icon Global Prompt

```text
Create a Project Starfall 64x64 transparent fantasy RPG item icon for [ITEM_NAME].
Style: Starlit Frontier Fantasy, clean painted 2D icon, crisp silhouette, controlled cel-shaded detail, dark edge separation, readable at 64px, [TIER MATERIALS], [REGION OR CLASS MOTIF].
Composition: one centered item, no background, no UI border, no rarity frame, no quantity number, no label, no watermark.
Use palette: [PALETTE]. Leave transparent padding around the item.
```

### Item Icon Negative Prompt

```text
text, numbers, label, watermark, signature, UI frame, inventory slot, rarity border, background, scenery, character hand, photorealism, 3D render, clutter, multiple unrelated items, cropped item, drop shadow crossing icon edge, copied IP, unreadable tiny detail
```

### Required Item Source Sheets

Generate sheets with one item per cell and no labels:

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

Star Card exception:

- `ai-items-star-cards-sheet.png` is currently an external processed sheet under `img/project-starfall/items/sheets/`.
- It is sliced by `build/process-project-starfall-ai-item-icons.js` into the five 64px Star Card item icons.
- If Star Cards are regenerated from a source sheet later, update the processor, manifest, and this guide in the same change.

### Card Icon Contract

Monster/Star card icons are a separate asset family from item icons.

- Source path: `img/project-starfall/cards/source/ai-card-icons.png`
- Runtime icon path: `img/project-starfall/cards/icons/<card-id>.png`
- Runtime size: `64x64`
- Format: PNG with alpha
- Source layout: 7 columns x 3 rows
- Processor: `build/process-project-starfall-card-icons.js`
- Runtime IDs: every entry in `CARD_DEFINITIONS` in `js/games/project-starfall/data/cards.js`

Card icon prompt:

```text
Create a Project Starfall card icon source sheet, 7 columns x 3 rows, covering every CARD_DEFINITIONS entry in data order.
Style: Starlit Frontier Fantasy item-symbol art, clean 2D fantasy card motifs, one centered readable symbol per cell, transparent-ready flat #ff00ff background, cyan #00ffff guide grid, no text, no labels, no card UI frame, no numerals, no watermark.
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

### Map Background Contract

- Runtime path: `img/project-starfall/maps/<map-id>.webp`
- Source field path: `img/project-starfall/maps/source/field/<map-id>.png`
- Source safe-zone path: `img/project-starfall/maps/source/safe-zones/<map-id>.png`
- Runtime size: `1280x640`
- Format: WebP, no alpha
- Edges: left and right edges must wrap cleanly for side-scroller looping/blending.
- No text, labels, UI, characters, monsters, or foreground clutter that hides combat.

Map prompt:

```text
Create a 1280x640 Project Starfall side-scroller panoramic background for [MAP_NAME].
Style: Starlit Frontier Fantasy, painterly 2D fantasy background, readable side-scroller depth, clear gameplay lanes, layered parallax feel, no characters, no monsters, no UI, no text.
Region: [REGION DESCRIPTION].
Palette: [PALETTE].
Required landmarks: [LANDMARKS].
Composition: foreground gameplay platform areas must remain visually clear, midground supports the theme, background has atmospheric depth. Left and right edges must tile or blend seamlessly.
Lighting: [TIME/WEATHER/MOOD].
```

Map negative prompt:

```text
text, labels, signs with words, UI, characters, enemies, watermark, logo, photorealism, 3D render, isometric map, top-down map, cluttered foreground, huge object blocking combat lane, hard vertical seam, modern city, sci-fi chrome
```

### Required Map Themes

Generate and maintain backgrounds for these IDs:

`starfall-crossing`, `greenroot-meadow`, `thornpath-thicket`, `bramble-depths`, `rustcoil-ruins`, `gearworks-vault`, `cinder-hollow`, `emberjaw-lair`, `bandit-ridge-camp`, `oreback-quarry`, `ashglass-pass`, `frostfen-outskirts`, `glacier-spine`, `rimewarden-sanctum`, `stormbreak-cliffs`, `astral-archive`, `eclipse-frontier`, `endless-rift`, `rustcoil-outpost`, `cinder-refuge`, `frostfen-camp`, `stormbreak-haven`, `astral-observatory`, `brambleking-court`, `titan-foundry`, `deepcore-core`, `emberjaw-furnace`, `rimewarden-vault`, `stormbreak-aerie`, `astral-stacks`, `eclipse-throne`, plus class trial maps as needed.

### Terrain Atlas Contract

- Runtime path: `img/project-starfall/environment/terrain/<theme-id>.png`
- Size: `512x256`
- Format: PNG alpha
- Layout: 8 columns x 4 rows
- Cell size: `64x64`
- Source style: isolated side-scroller terrain parts on transparent or processor-safe chroma background.

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
Style: Starlit Frontier Fantasy side-scroller terrain, clean collision-friendly platform edges, crisp readable top surfaces, painterly material detail, no characters, no enemies, no UI, no text.
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
- Use foreground detail sparingly; avoid large dark foreground objects that cover combat lanes.
- Backgrounds can be painterly, but terrain and props must remain clean.

## 7. VFX and Feedback Asset Instructions

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
Timing: frame 1 anticipation/start, frames 2-3 build, frame 4 peak, frames 5-6 dissipate.
Color: [CLASS/REGION COLOR].
Keep the effect centered, contained inside each frame, and readable over dark or bright backgrounds.
```

### Skill FX Contract

- Path: `img/project-starfall/animations/combat-fx/skills/<skill-file-id>-sheet.png`
- Source path: `img/project-starfall/animations/combat-fx/skills/source/<skill-file-id>-source.png`
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
Palette and motif: [CLASS COLOR AND MOTIF].
Keep every frame centered and inside cell bounds. Use transparent-ready art or processor-safe chroma source.
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
Style: clean readable 2D fantasy monster attack VFX, no character body except small abstract attack shapes if necessary, no UI, no text, transparent-ready.
Rows: telegraph warning, melee contact effect, projectile travel, buff/support effect, impact effect.
Motif: [ENEMY REGION AND ELEMENT].
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
- `[Assumption]` Current intended layout is 6 columns x 6 rows, 160px frames, `960x960`.
- Rows: `idle`, `run`, `jump`, `fall`, `loot`, `teleport`

Pet prompt:

```text
Create a Project Starfall pet animation sheet for a starfall fox, 6 columns x 6 rows, 160px frames, transparent PNG.
Style: cute compact 2D side-scroller companion, fox silhouette with starry cyan/gold accents, readable at small scale, clean outline.
Rows: idle, run, jump, fall, loot pickup, teleport blink.
No text, no UI, no scenery, no cast shadow.
```

### VFX Timing Notes

- Hit sparks: 4 to 6 frames, peak by frame 2 or 3.
- Dust puffs: 6 frames, expand then fade.
- Landing effects: 6 frames, horizontal dust squash outward.
- Jump effects: 6 frames, small burst under feet.
- Attack slashes: 6 frames, anticipation arc then bright contact frame.
- Projectile impacts: 6 frames, readable element burst.
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
- Source sheets: `img/project-starfall/skills/source/<sheet-name>.png`
- Size: `256x256`
- Format: PNG alpha

Skill icon prompt:

```text
Create a 256x256 transparent Project Starfall skill icon for [SKILL_NAME].
Style: clean fantasy RPG ability icon, Starlit Frontier Fantasy, one main symbol plus one supporting effect, crisp silhouette, high contrast, readable at 32px, no text, no numbers, no UI frame.
Class motif: [CLASS].
Gameplay intent: [MOBBING/BOSSING/SUPPORT/MOBILITY/CONTROL].
Use [CLASS COLORS] and a clear [WEAPON/MAGIC/ELEMENT] symbol.
```

Skill icon visual rules:

- Mobbing skills: wide arcs, multiple sparks, area cues.
- Bossing skills: narrow focused strike, precision mark, single bright impact.
- Support skills: aura, shield, circle, radiating pulse.
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
- Buff/debuff icons: `64x64`, transparent, clear symbol and color.
- Inventory slots: dark navy inset, gold/parchment rim, no item baked in.
- Dialogue boxes: parchment center, dark navy/gold frame, no text baked in.
- Buttons: dark navy or parchment base, gold trim, hover/active variants if coded.
- Fonts/text: do not generate raster text unless the UI asset is explicitly a title image. Runtime text should be HTML/CSS.

## 9. Prompt Engineering Rules

### Master Style Prompt

Append this to every non-UI asset prompt unless it conflicts with the asset type:

```text
Project Starfall Starlit Frontier Fantasy, original 2D side-scroller fantasy RPG asset, clean readable silhouette, crisp dark contour outline where appropriate, soft cel-shaded sprite rendering, subtle painterly color, luminous fallen-star magic accents, blue-and-gold guild motifs, practical crafted fantasy materials, readable at gameplay scale, consistent upper-left/front lighting, no copied IP.
```

For backgrounds, replace "crisp dark contour outline" with:

```text
painterly side-scroller background depth, clear gameplay lanes, atmospheric perspective, readable foreground/midground/background separation
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

### Details To Lock Across Prompts

Keep these locked:

- Project Starfall style phrase.
- Side-scroller perspective for gameplay assets.
- Right-facing orientation for sprites.
- Transparent runtime backgrounds.
- No text/logos/watermarks.
- Starfall cyan/gold magic language.
- Clean silhouette and outline rules.
- Upper-left/front lighting for sprites.
- Exact sheet row order.
- Exact file path and naming.

### Consistent Characters Across Animations

For each player class:

1. Generate or choose one approved base reference.
2. Write a short character lock: hair/helmet, face visibility, outfit, weapon, colors, accessory.
3. Use that exact lock in every prompt.
4. Generate the full 6x10 sheet in one pass when possible.
5. If fixing a row, use the approved sheet as image reference and regenerate only that row with the same lock.
6. Do not accept costume drift between rows.

### Consistent Enemies Across Animations

For each enemy:

1. Lock silhouette, number of limbs, head shape, horns/antlers/core, palette, and material.
2. Generate the complete 3x8 sheet in one pass.
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
- Cyan `#00ffff` guide grid only if the processor expects it.
- Chroma key color only if using a processor source sheet.

### Prompting Transparent Backgrounds

Use:

```text
transparent background, alpha PNG, no scenery, no floor, no shadow, no UI, no text
```

If the generator cannot output true alpha, use:

```text
flat solid #00ff00 background outside the artwork for chroma key removal, with no #00ff00 pixels inside the artwork
```

For green/plant/ooze subjects, use:

```text
flat solid #ff00ff background outside the artwork for chroma key removal, with no #ff00ff pixels inside the artwork
```

### Avoiding Frame Flicker

- Keep the same seed/reference for all frames.
- Generate sheets rather than separate frames.
- Avoid prompts that describe a new costume in each row.
- Avoid too many tiny accessories.
- Keep VFX separate from character body when possible.
- Use simple strong shapes for weapons and class props.
- Run visual review before importing.

## 10. Asset Review Checklist

Use this checklist before importing any generated asset.

### Style

- Matches Starlit Frontier Fantasy.
- Does not look like copied third-party IP.
- Does not look photorealistic or 3D-rendered.
- Uses the correct regional/class palette.
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
- Scale does not flicker.
- Costume, weapon, face, and silhouette stay consistent.
- Attack tells are readable.
- Hit and defeat frames are distinct.
- Looping rows loop cleanly.
- Non-looping rows have clear anticipation, peak, and recovery.
- No limb, weapon, VFX, or prop crosses cell boundaries.

### Gameplay Readability

- Silhouette reads at in-game size.
- Player class identity is clear.
- Enemy role is clear.
- Hazard or attack direction is clear.
- Projectile direction is clear.
- UI icons read at 32px and 64px.
- Terrain top edges are collision-readable.
- Background does not hide combat.

### Documentation Match

- Asset matches this guide.
- Asset matches the GDD and Starfall asset prompt notes.
- Asset matches `js/games/project-starfall/data/assets.js`.
- Asset matches the relevant runtime contract in `js/games/project-starfall/data/animations.js`.
- New folders, if any, are included in `build/copy-to-public.js`.

## 11. Import and Validation

After generating assets, run the project asset pipeline and tests:

```bash
npm run validate:project-starfall-assets
npm run build:project-starfall-assets
npm run test:starfall:assets
npm test
```

For player sheets, also validate with the player processor if the source file changed:

```bash
node build/process-project-starfall-player-ai-assets.js --validate
```

For compact enemies, validate with the compact enemy processor if compact enemy source sheets changed:

```bash
node build/process-project-starfall-compact-bandits.js --validate
```

If a generated asset fails validation, fix the source image rather than weakening the runtime contract.
