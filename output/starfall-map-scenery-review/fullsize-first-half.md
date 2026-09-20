# Full-size scenery review: first 28 maps

Reviewed on 2026-09-19. Scope follows entries 1–28 of `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/review-manifest.json`.

## Coverage and method

All 112 primary desktop Pixi captures were opened individually at their original 1440 × 1000 resolution: arrival, middle, upper, and exit for every map below. This was a gameplay-size inspection, not an inference from thumbnail contact sheets. No one-lane shop duplicates occur in this half of the manifest.

The common capture root is `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/`. Primary files use `verified-clear/`, except the five repaired Warden maps use `npc-final/` and Ashglass uses `contact-final/`. Exact primary filenames are listed below. Four-view recaptures of affected maps were inspected after repairs, not merely assumed to match. Twelve additional Canvas images were inspected to investigate suspicious details and verify corrections.

Initial captures with large transient notification banners were superseded by the clean captures before this audit was closed. The clean fixture represents those notifications having faded; it does not remove a persistent gameplay interface.

These stills support judgments about material fit, visible ground contact, actor/background separation, composition, and missing artwork. They do not establish animation smoothness, attack readability through every frame, hitbox precision, traversal reachability, population balance, or reduced-effects/mobile acceptance. Those require the separately recorded runtime, training, and deep-renderer checks.

## Findings and disposition

Two concrete visual defects found here were corrected and re-inspected:

- Generated hunt Wardens in Bramble Depths, Gearworks Vault, Emberjaw Lair, Bandit Animation Lab, and Rimewarden Sanctum used the geometric actor fallback. Explicit existing-art mapping now supplies the illustrated actor. All five corrected Pixi four-view sets and all five Canvas arrivals were inspected. This restores intended artwork without changing NPC identities or bodies.
- Ashglass's crisp basalt top sat above visible player/NPC feet, making them appear roughly 26 pixels inside the front face in the middle capture. The surface-contact correction was inspected in all four final Pixi views and final Canvas middle/upper views. Visible feet now meet the basalt lip.

No additional definite foreground obstruction or missing actor artwork was found in these final views. This is not an unqualified statement that every visual acceptance criterion passes. The following presentation issues remain:

- Some town rear buildings read too broad/low compared with neighboring service facades, especially Rustcoil and Cinder; Astral's rear dome also differs strongly in scale.
- Screen-edge portal/vendor labels sometimes overlap or clip. This appears in several towns and Cinder's exit.
- Frostfen's shared shop facades lack the snow dressing of surrounding architecture.
- Large Orebacks and Lava Ticks can visually fill much of the gap beneath thin elevated shelves. Their silhouettes remain visible, but shelf clearance and apparent weight deserve contextual polish.
- Meadow's broken background aqueduct and Titan Foundry's hanging press tops retain relatively strong horizontal edges. Perspective, material, and outlining distinguish them from playable ground, but a first-time-player check is warranted before declaring that all false-foothold ambiguity is gone.
- Some procedural ladders have much simpler treatment than the painted setting. Rungs remain readable.
- Glacier's snow crown extends above visible feet; soft snow overlap alone is not evidence of the same hard-slab defect found in Ashglass. This review does not certify its exact collision/art registration.

## Map-by-map assessment

| # | Map ID | Final primary source | Assessment |
|---|---|---|---|
| 1 | `starfallCrossing` | verified-clear | Warm stone, blue/gold fixtures and distinct service silhouettes read coherently. Grass caps, ladders and ramp joins clearly identify playable ground. Busy upper service cluster still leaves the player clear; no blocking visual issue. |
| 2 | `rustcoilOutpost` | verified-clear | Brass, gears and stone match the industrial region. Foreground terrain and rungs remain clear. Arrival rear workshop is conspicuously broad/low relative to small shop facades; exit portal labels overlap at left. Both are polish findings, with no ground obstruction. |
| 3 | `cinderRefuge` | verified-clear | Basalt faces, ochre edges and warm cave architecture establish the volcanic refuge. Large rear forge remains noticeably stretched next to smaller service buildings. Upper vendor arrows and exit portal labels overlap at the left screen edge; playable ramps and floors remain readable. |
| 4 | `frostfenCamp` | verified-clear | Snow caps and blue-gray stones clearly separate terrain from pink winter distance. Dark actor outlines remain strong. Shared service buildings lack snow dressing compared with the large rear cabin. Exit portal labels overlap at left; no scenery blocks the route. |
| 5 | `stormbreakHaven` | verified-clear | Pale masonry, blue banners and wind devices fit the sky region. Background bridges remain softer and narrower than outlined playable slabs. Vendor arrows and exit labels overlap at screen edges; no route obstruction. |
| 6 | `astralObservatory` | verified-clear | Ivory/gold surfaces and violet distance separate well; crystal props remain sparse. The rear arrival dome is much broader than nearby service buildings and merits proportion polish. Ramps, player and useful surfaces remain clear. |
| 7 | `greenrootMeadow` | verified-clear | Crisp brown/green playable surfaces and strongly outlined creatures read clearly against atmospheric valley. The broken gray aqueduct has a fairly crisp horizontal edge between lanes: a mild false-foothold ambiguity to test with new players, although its color and masonry differ from playable ground. |
| 8 | `thornpathThicket` | verified-clear | Roots, moss, thorns and native creatures make a coherent forest. Detailed near tree competes somewhat with dark terrain and green ladders, but cap edges, rungs and actors remain visible. No scenery obstruction; middle view is visually sparse in combat activity. |
| 9 | `brambleDepths` | npc-final | Rooted stone, dim teal depth and bright forest enemies are coherent. The geometric Warden placeholder found in the initial capture is repaired: all four final Pixi views and the final Canvas arrival show the intended illustrated actor. Ground and ramps remain legible; the background door is clearly distant scenery. |
| 10 | `rustcoilRuins` | verified-clear | Brass-banded stone, mechanical creatures and monumental machinery fit the region. Bright background stays lighter than the outlined actors; foreground edges are strong. Background water pipes have different perspective/material treatment from the terrain. No blocking issue. |
| 11 | `gearworksVault` | npc-final | Large brass door and machine architecture frame clear brass/stone lanes; tiny crystals do not obstruct actors. The initial geometric Warden placeholder is repaired in the four final Pixi views and final Canvas arrival. Artwork remains coherent with the mechanical region. |
| 12 | `cinderHollow` | verified-clear | Dark basalt and ochre caps separate from the softer volcanic distance. Fire creatures retain strong silhouettes despite warm shared hues. Ring-style climbables are clear; tall Lava Tick silhouettes crowd the gap under some small platforms but the player lane remains clear. Exit NPC label clips at the right screen edge. |
| 13 | `emberjawLair` | npc-final | The dragon-shaped volcanic landmark gives a distinct dungeon identity; dark rock and warm edges leave actors readable. The initial generated-Warden placeholder is repaired in the four final Pixi views and final Canvas arrival. The distant bridge is perspective-scaled and subdued compared with foreground slabs. |
| 14 | `banditRidgeCamp` | verified-clear | Wood, red fabric and rocky autumn camp fit bandit silhouettes. Full-size enemy outlines stay clear; ladders have stronger structural simplicity than distant rope bridges. Some background tower decking sits near playable edges, but visible slab caps distinguish the route. No blocking issue. |
| 15 | `banditAnimationLab` | npc-final | Simple repeated practice ledges keep isolated bandit silhouettes visible. Regional camp art matches the enemy; the map remains appropriately sparse for an animation lab. The initial Warden placeholder is repaired in all four final Pixi views and the final Canvas arrival. |
| 16 | `orebackQuarry` | verified-clear | Veined sandstone, timber bracing, cranes and mineral-backed creatures fit the quarry. Bright distance and thick outlines give strong contrast. Large beetles nearly fill some vertical gaps and look heavy on thin upper shelves: nonblocking scale/clearance polish, with floor and ladders still visible. No large prop placed unsupported on the steep ramps. |
| 17 | `ashglassPass` | contact-final | Dark basalt with violet glass distinctly replaces Quarry materials and suits the volcanic landscape. The initial player/NPC contact defect placed feet about 26 px down the slab face; the final contact repair was inspected in all four Pixi views and middle/upper Canvas. Visible feet now meet the basalt lip. Foreground paths remain distinct from the distant arched bridge. |
| 18 | `frostfenOutskirts` | verified-clear | Snow/stone slabs and sparse ice props fit the frozen marsh. Cool enemies are strongly outlined against pale terrain; no false background platforms apparent. Arrival Tracker is temporarily covered by an overlapping enemy in this snapshot, but its label remains clear. |
| 19 | `glacierSpine` | verified-clear | Sweeping ice spines, pale snow caps and dark character contours remain readable. Small props do not interrupt route edges; the distant rope bridge is thin and perspective-scaled. The soft snow crown extends above visible feet on ground; that alone does not establish a collision mismatch, and this still-image review does not certify exact snow-surface registration. |
| 20 | `rimewardenSanctum` | npc-final | Reviewed final repaired captures: illustrated Warden now matches other NPCs. Cold cathedral, bells and recessed staircase create a clear dungeon landmark. Foreground slab edges remain distinct from perspective stairs; bright enemies retain dark silhouettes. |
| 21 | `stormbreakCliffs` | verified-clear | Wind-battered masonry and cloud-spires fit the storm roster. Warm platform caps distinguish paths from pale distant rock tops; dark bow users and bright floating sprites stay distinct. No prop obstruction or convincing false floor. |
| 22 | `astralArchive` | verified-clear | Library architecture, blue/gold motifs and book-bearing enemies form a coherent scene. Strong foreground caps separate slabs from the recessed decorative floor. Back staircase/bridge perspective remains different from gameplay geometry. No blocking issue. |
| 23 | `eclipseFrontier` | verified-clear | Violet stone, pale caps and floating ruins establish a darker continuation of Astral. Black/gold enemy outlines remain readable on pink sky, though some dark lower details merge at foot level. No blocking prop or convincing false floor; background islands are clearly atmospheric. |
| 24 | `endlessRift` | verified-clear | Broken circular celestial landmark and scattered islands fit the endless encounter. Repeated dark-violet lanes keep a consistent edge treatment; small teal crystals stay secondary. Upper ledge bodies remain clear and no decoration masks ladder access. |
| 25 | `bramblekingCourt` | verified-clear | Golden wooden throne, green banners, roots, and foliage creatures form a coherent woodland court. Playable turf and stone caps remain distinct from the broad perspective floor and recessed throne stairs; no obstruction found in the four views. |
| 26 | `titanFoundry` | verified-clear | Molten channels, industrial masonry, hanging presses, and the giant statue establish a coherent foundry. Foreground ledges and enemy outlines remain clear. The broad flat tops of hanging background press weights could invite a mistaken jump, although their chains, perspective, and darker material distinguish them; this needs a player-readability check rather than a collision claim from stills. |
| 27 | `deepcoreCore` | verified-clear | Large teal crystal, cranes, suspended crates, and veined quarry masonry establish a clear underground mine. Foreground cap edges are much warmer and sharper than the receding bridges. The large Oreback overlaps the player in the upper still, but their outlines and silhouettes remain separable; no foreground prop hides a route. |
| 28 | `emberjawFurnace` | verified-clear | The monumental dragon furnace, molten channels, basalt ledges, and fire creatures are coherent. Foreground lanes remain distinct from the distant bridges despite shared warm colors; stronger actor outlines preserve readability. Thin ring ladders are less substantial than the painted environments but visible. No obstruction or missing actor artwork found. |

## Exact primary files inspected

All paths in this inventory are relative to the common capture root above. Each listed file was visually opened at original resolution.

### starfallCrossing

Source: `verified-clear/`.

- `starfallCrossing-arrival-pixi.png`
- `starfallCrossing-middle-pixi.png`
- `starfallCrossing-upper-pixi.png`
- `starfallCrossing-exit-pixi.png`

### rustcoilOutpost

Source: `verified-clear/`.

- `rustcoilOutpost-arrival-pixi.png`
- `rustcoilOutpost-middle-pixi.png`
- `rustcoilOutpost-upper-pixi.png`
- `rustcoilOutpost-exit-pixi.png`

### cinderRefuge

Source: `verified-clear/`.

- `cinderRefuge-arrival-pixi.png`
- `cinderRefuge-middle-pixi.png`
- `cinderRefuge-upper-pixi.png`
- `cinderRefuge-exit-pixi.png`

### frostfenCamp

Source: `verified-clear/`.

- `frostfenCamp-arrival-pixi.png`
- `frostfenCamp-middle-pixi.png`
- `frostfenCamp-upper-pixi.png`
- `frostfenCamp-exit-pixi.png`

### stormbreakHaven

Source: `verified-clear/`.

- `stormbreakHaven-arrival-pixi.png`
- `stormbreakHaven-middle-pixi.png`
- `stormbreakHaven-upper-pixi.png`
- `stormbreakHaven-exit-pixi.png`

### astralObservatory

Source: `verified-clear/`.

- `astralObservatory-arrival-pixi.png`
- `astralObservatory-middle-pixi.png`
- `astralObservatory-upper-pixi.png`
- `astralObservatory-exit-pixi.png`

### greenrootMeadow

Source: `verified-clear/`.

- `greenrootMeadow-arrival-pixi.png`
- `greenrootMeadow-middle-pixi.png`
- `greenrootMeadow-upper-pixi.png`
- `greenrootMeadow-exit-pixi.png`

### thornpathThicket

Source: `verified-clear/`.

- `thornpathThicket-arrival-pixi.png`
- `thornpathThicket-middle-pixi.png`
- `thornpathThicket-upper-pixi.png`
- `thornpathThicket-exit-pixi.png`

### brambleDepths

Source: `npc-final/`.

- `brambleDepths-arrival-pixi.png`
- `brambleDepths-middle-pixi.png`
- `brambleDepths-upper-pixi.png`
- `brambleDepths-exit-pixi.png`

### rustcoilRuins

Source: `verified-clear/`.

- `rustcoilRuins-arrival-pixi.png`
- `rustcoilRuins-middle-pixi.png`
- `rustcoilRuins-upper-pixi.png`
- `rustcoilRuins-exit-pixi.png`

### gearworksVault

Source: `npc-final/`.

- `gearworksVault-arrival-pixi.png`
- `gearworksVault-middle-pixi.png`
- `gearworksVault-upper-pixi.png`
- `gearworksVault-exit-pixi.png`

### cinderHollow

Source: `verified-clear/`.

- `cinderHollow-arrival-pixi.png`
- `cinderHollow-middle-pixi.png`
- `cinderHollow-upper-pixi.png`
- `cinderHollow-exit-pixi.png`

### emberjawLair

Source: `npc-final/`.

- `emberjawLair-arrival-pixi.png`
- `emberjawLair-middle-pixi.png`
- `emberjawLair-upper-pixi.png`
- `emberjawLair-exit-pixi.png`

### banditRidgeCamp

Source: `verified-clear/`.

- `banditRidgeCamp-arrival-pixi.png`
- `banditRidgeCamp-middle-pixi.png`
- `banditRidgeCamp-upper-pixi.png`
- `banditRidgeCamp-exit-pixi.png`

### banditAnimationLab

Source: `npc-final/`.

- `banditAnimationLab-arrival-pixi.png`
- `banditAnimationLab-middle-pixi.png`
- `banditAnimationLab-upper-pixi.png`
- `banditAnimationLab-exit-pixi.png`

### orebackQuarry

Source: `verified-clear/`.

- `orebackQuarry-arrival-pixi.png`
- `orebackQuarry-middle-pixi.png`
- `orebackQuarry-upper-pixi.png`
- `orebackQuarry-exit-pixi.png`

### ashglassPass

Source: `contact-final/`.

- `ashglassPass-arrival-pixi.png`
- `ashglassPass-middle-pixi.png`
- `ashglassPass-upper-pixi.png`
- `ashglassPass-exit-pixi.png`

### frostfenOutskirts

Source: `verified-clear/`.

- `frostfenOutskirts-arrival-pixi.png`
- `frostfenOutskirts-middle-pixi.png`
- `frostfenOutskirts-upper-pixi.png`
- `frostfenOutskirts-exit-pixi.png`

### glacierSpine

Source: `verified-clear/`.

- `glacierSpine-arrival-pixi.png`
- `glacierSpine-middle-pixi.png`
- `glacierSpine-upper-pixi.png`
- `glacierSpine-exit-pixi.png`

### rimewardenSanctum

Source: `npc-final/`.

- `rimewardenSanctum-arrival-pixi.png`
- `rimewardenSanctum-middle-pixi.png`
- `rimewardenSanctum-upper-pixi.png`
- `rimewardenSanctum-exit-pixi.png`

### stormbreakCliffs

Source: `verified-clear/`.

- `stormbreakCliffs-arrival-pixi.png`
- `stormbreakCliffs-middle-pixi.png`
- `stormbreakCliffs-upper-pixi.png`
- `stormbreakCliffs-exit-pixi.png`

### astralArchive

Source: `verified-clear/`.

- `astralArchive-arrival-pixi.png`
- `astralArchive-middle-pixi.png`
- `astralArchive-upper-pixi.png`
- `astralArchive-exit-pixi.png`

### eclipseFrontier

Source: `verified-clear/`.

- `eclipseFrontier-arrival-pixi.png`
- `eclipseFrontier-middle-pixi.png`
- `eclipseFrontier-upper-pixi.png`
- `eclipseFrontier-exit-pixi.png`

### endlessRift

Source: `verified-clear/`.

- `endlessRift-arrival-pixi.png`
- `endlessRift-middle-pixi.png`
- `endlessRift-upper-pixi.png`
- `endlessRift-exit-pixi.png`

### bramblekingCourt

Source: `verified-clear/`.

- `bramblekingCourt-arrival-pixi.png`
- `bramblekingCourt-middle-pixi.png`
- `bramblekingCourt-upper-pixi.png`
- `bramblekingCourt-exit-pixi.png`

### titanFoundry

Source: `verified-clear/`.

- `titanFoundry-arrival-pixi.png`
- `titanFoundry-middle-pixi.png`
- `titanFoundry-upper-pixi.png`
- `titanFoundry-exit-pixi.png`

### deepcoreCore

Source: `verified-clear/`.

- `deepcoreCore-arrival-pixi.png`
- `deepcoreCore-middle-pixi.png`
- `deepcoreCore-upper-pixi.png`
- `deepcoreCore-exit-pixi.png`

### emberjawFurnace

Source: `verified-clear/`.

- `emberjawFurnace-arrival-pixi.png`
- `emberjawFurnace-middle-pixi.png`
- `emberjawFurnace-upper-pixi.png`
- `emberjawFurnace-exit-pixi.png`

## Additional Canvas files inspected

- `verified-clear/brambleDepths-arrival-canvas.png`
- `verified-clear/gearworksVault-arrival-canvas.png`
- `verified-clear/ashglassPass-middle-canvas.png`
- `verified-clear/greenrootMeadow-middle-canvas.png`
- `verified-clear/greenrootMeadow-upper-canvas.png`
- `npc-final/brambleDepths-arrival-canvas.png`
- `npc-final/gearworksVault-arrival-canvas.png`
- `npc-final/emberjawLair-arrival-canvas.png`
- `npc-final/banditAnimationLab-arrival-canvas.png`
- `npc-final/rimewardenSanctum-arrival-canvas.png`
- `contact-final/ashglassPass-middle-canvas.png`
- `contact-final/ashglassPass-upper-canvas.png`

## Earlier versions inspected before correction

The four Pixi views of Bramble Depths, Gearworks Vault, Emberjaw Lair, and Bandit Animation Lab were also inspected under `verified-clear/` before their `npc-final/` replacements. Their arrival views exposed the placeholder defect. Rimewarden Sanctum's primary review began with its repaired `npc-final/` images.

The four Ashglass Pixi views and Canvas middle were first inspected under `verified-clear/`, then the corrected files listed above under `contact-final/`. The initial contact finding is retained here as provenance, not as an unresolved claim about the final repaired images.

