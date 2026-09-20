# Full-size visual review: maps 28–55

This review inspected **88 distinct Pixi camera views across all 28 maps** in entries 28–55 of the review manifest: all four views for the four late-game combat maps, and three distinct views for each of the 24 shops. Every listed image was opened individually at its original 1440×1000 resolution. This is additional manual evidence beyond the earlier contact sheets.

The clean final screenshots are in `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/verified/` (also retained in `verified-clear/`). They show the final local built route, `http://127.0.0.1:4174/games/project-starfall`. The capture report has an empty errors array. The capture fixture clears transient map-loading toasts to reveal the scene as it appears after those notices fade; this does not change production code. The clean files were inspected after the refresh, including rechecking the arrival/exit views first reviewed before it.

## Findings

No new blocking scenery defect was identified in this subset.

- **Rimewarden Vault:** snow-capped stone surfaces and brown/gold ladders remain legible against the bright ice-vault illustration. The ramp and raised-lane edges remain identifiable around the player and enemy silhouettes.
- **Stormbreak Aerie:** pale wind-weathered platforms, warm edge vegetation and cyan ladders fit the region. The distant nest, cloud banks and architecture read behind the actor lane. The exit view has an uninterrupted standing surface.
- **Astral Stacks:** ivory/gold foreground and cyan/violet traversal ladders remain distinct from the book-tower scenery. The illustration contains pale decorative library ladders, so their resemblance to usable ladders is a minor remaining clarity concern. The foreground ladders are brighter and terminate at the actual platform edges; Canvas checks show the same distinction. Further atmospheric separation would be a possible future refinement.
- **Eclipse Throne:** the dark violet foreground separates from the distant golden throne and eclipse. Actual platform ends, ramp surfaces and bright traversal ladders stay readable.
- **All 24 shops:** the four vendor room types consistently use their intended workshop, armory, apothecary/supply or star-themed interior. The neutral gray-taupe masonry and timber support fits all four rooms. The continuous floor top stays aligned with the compact actors. Background furniture does not obstruct the standing lane, vendors or arrival portals. No outdoor fallback buildings, violet-glass floor, missing textures or apparent floor gaps remain.
- **Labels and scale:** white vendor text on dark nameplates is legible at this desktop gameplay size. Rear-wall weapons, armor and shelving are larger than the compact characters but read as display scenery. Player and vendor currently share the compact actor appearance; this review does not imply distinct vendor identities were authored.

These are static presentation checks. Capturing an upper view may place the stopped player directly over an existing enemy; that alone is not evidence of a spawn or collision defect. The screenshots do not establish animation smoothness, hitbox accuracy, combat timing, dynamic camera behavior or traversal success. Those require the separate runtime and measurement evidence. No runtime or asset files were changed during this review.

## Exact Pixi coverage

Each filename below was individually inspected in the clean capture set. Shop `upper` selects the same ground-center position as `middle` because these rooms have a single platform. It does not add another camera region. The earlier `starfallCrossingWeaponShop-upper-pixi.png` and `starfallCrossingArmorShop-upper-pixi.png` were opened to check this equivalence; they are not counted among the 88 distinct views. The remaining 22 duplicate shop upper captures were not individually reopened.

| Map ID | Exact inspected filenames |
| --- | --- |
| `rimewardenVault` | `rimewardenVault-arrival-pixi.png`<br>`rimewardenVault-middle-pixi.png`<br>`rimewardenVault-upper-pixi.png`<br>`rimewardenVault-exit-pixi.png` |
| `stormbreakAerie` | `stormbreakAerie-arrival-pixi.png`<br>`stormbreakAerie-middle-pixi.png`<br>`stormbreakAerie-upper-pixi.png`<br>`stormbreakAerie-exit-pixi.png` |
| `astralStacks` | `astralStacks-arrival-pixi.png`<br>`astralStacks-middle-pixi.png`<br>`astralStacks-upper-pixi.png`<br>`astralStacks-exit-pixi.png` |
| `eclipseThrone` | `eclipseThrone-arrival-pixi.png`<br>`eclipseThrone-middle-pixi.png`<br>`eclipseThrone-upper-pixi.png`<br>`eclipseThrone-exit-pixi.png` |
| `starfallCrossingWeaponShop` | `starfallCrossingWeaponShop-arrival-pixi.png`<br>`starfallCrossingWeaponShop-middle-pixi.png`<br>`starfallCrossingWeaponShop-exit-pixi.png` |
| `starfallCrossingArmorShop` | `starfallCrossingArmorShop-arrival-pixi.png`<br>`starfallCrossingArmorShop-middle-pixi.png`<br>`starfallCrossingArmorShop-exit-pixi.png` |
| `starfallCrossingSupplyShop` | `starfallCrossingSupplyShop-arrival-pixi.png`<br>`starfallCrossingSupplyShop-middle-pixi.png`<br>`starfallCrossingSupplyShop-exit-pixi.png` |
| `starfallCrossingSpecialShop` | `starfallCrossingSpecialShop-arrival-pixi.png`<br>`starfallCrossingSpecialShop-middle-pixi.png`<br>`starfallCrossingSpecialShop-exit-pixi.png` |
| `rustcoilOutpostWeaponShop` | `rustcoilOutpostWeaponShop-arrival-pixi.png`<br>`rustcoilOutpostWeaponShop-middle-pixi.png`<br>`rustcoilOutpostWeaponShop-exit-pixi.png` |
| `rustcoilOutpostArmorShop` | `rustcoilOutpostArmorShop-arrival-pixi.png`<br>`rustcoilOutpostArmorShop-middle-pixi.png`<br>`rustcoilOutpostArmorShop-exit-pixi.png` |
| `rustcoilOutpostSupplyShop` | `rustcoilOutpostSupplyShop-arrival-pixi.png`<br>`rustcoilOutpostSupplyShop-middle-pixi.png`<br>`rustcoilOutpostSupplyShop-exit-pixi.png` |
| `rustcoilOutpostSpecialShop` | `rustcoilOutpostSpecialShop-arrival-pixi.png`<br>`rustcoilOutpostSpecialShop-middle-pixi.png`<br>`rustcoilOutpostSpecialShop-exit-pixi.png` |
| `cinderRefugeWeaponShop` | `cinderRefugeWeaponShop-arrival-pixi.png`<br>`cinderRefugeWeaponShop-middle-pixi.png`<br>`cinderRefugeWeaponShop-exit-pixi.png` |
| `cinderRefugeArmorShop` | `cinderRefugeArmorShop-arrival-pixi.png`<br>`cinderRefugeArmorShop-middle-pixi.png`<br>`cinderRefugeArmorShop-exit-pixi.png` |
| `cinderRefugeSupplyShop` | `cinderRefugeSupplyShop-arrival-pixi.png`<br>`cinderRefugeSupplyShop-middle-pixi.png`<br>`cinderRefugeSupplyShop-exit-pixi.png` |
| `cinderRefugeSpecialShop` | `cinderRefugeSpecialShop-arrival-pixi.png`<br>`cinderRefugeSpecialShop-middle-pixi.png`<br>`cinderRefugeSpecialShop-exit-pixi.png` |
| `frostfenCampWeaponShop` | `frostfenCampWeaponShop-arrival-pixi.png`<br>`frostfenCampWeaponShop-middle-pixi.png`<br>`frostfenCampWeaponShop-exit-pixi.png` |
| `frostfenCampArmorShop` | `frostfenCampArmorShop-arrival-pixi.png`<br>`frostfenCampArmorShop-middle-pixi.png`<br>`frostfenCampArmorShop-exit-pixi.png` |
| `frostfenCampSupplyShop` | `frostfenCampSupplyShop-arrival-pixi.png`<br>`frostfenCampSupplyShop-middle-pixi.png`<br>`frostfenCampSupplyShop-exit-pixi.png` |
| `frostfenCampSpecialShop` | `frostfenCampSpecialShop-arrival-pixi.png`<br>`frostfenCampSpecialShop-middle-pixi.png`<br>`frostfenCampSpecialShop-exit-pixi.png` |
| `stormbreakHavenWeaponShop` | `stormbreakHavenWeaponShop-arrival-pixi.png`<br>`stormbreakHavenWeaponShop-middle-pixi.png`<br>`stormbreakHavenWeaponShop-exit-pixi.png` |
| `stormbreakHavenArmorShop` | `stormbreakHavenArmorShop-arrival-pixi.png`<br>`stormbreakHavenArmorShop-middle-pixi.png`<br>`stormbreakHavenArmorShop-exit-pixi.png` |
| `stormbreakHavenSupplyShop` | `stormbreakHavenSupplyShop-arrival-pixi.png`<br>`stormbreakHavenSupplyShop-middle-pixi.png`<br>`stormbreakHavenSupplyShop-exit-pixi.png` |
| `stormbreakHavenSpecialShop` | `stormbreakHavenSpecialShop-arrival-pixi.png`<br>`stormbreakHavenSpecialShop-middle-pixi.png`<br>`stormbreakHavenSpecialShop-exit-pixi.png` |
| `astralObservatoryWeaponShop` | `astralObservatoryWeaponShop-arrival-pixi.png`<br>`astralObservatoryWeaponShop-middle-pixi.png`<br>`astralObservatoryWeaponShop-exit-pixi.png` |
| `astralObservatoryArmorShop` | `astralObservatoryArmorShop-arrival-pixi.png`<br>`astralObservatoryArmorShop-middle-pixi.png`<br>`astralObservatoryArmorShop-exit-pixi.png` |
| `astralObservatorySupplyShop` | `astralObservatorySupplyShop-arrival-pixi.png`<br>`astralObservatorySupplyShop-middle-pixi.png`<br>`astralObservatorySupplyShop-exit-pixi.png` |
| `astralObservatorySpecialShop` | `astralObservatorySpecialShop-arrival-pixi.png`<br>`astralObservatorySpecialShop-middle-pixi.png`<br>`astralObservatorySpecialShop-exit-pixi.png` |

## Additional Canvas checks

These three original-resolution clean images were inspected to check the pale ice scene and the library-ladder ambiguity:

- `rimewardenVault-middle-canvas.png`
- `astralStacks-middle-canvas.png`
- `astralStacks-arrival-canvas.png`

They preserve the same foreground material, standing edges and ladder placement as the corresponding Pixi views. Random enemy selection and render-time decorative particles differ between the captures. This scoped comparison does not claim individual full-size inspection of all 224 Canvas captures.

The separately documented [connector review](visual-review.md#final-hunting-connector-checks) covers all eight new connector locations in both renderers at desktop and mobile/reduced-effects sizes.

The subsequent [deep renderer and portrait review](deep-review.md) adds 20 desktop Canvas views and 40 portrait/reduced-effects views across ten representative maps, with exact filenames and the remaining text-readability caveats.
