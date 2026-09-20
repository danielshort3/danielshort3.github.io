# Starfall contextual scenery visual review

The subsequent [full-size review of maps 28–55](fullsize-second-half.md) records 88 distinct clean Pixi views across all 24 shops and four late-game maps, plus three Canvas detail checks. It supersedes the limited full-size coverage described in the earlier review below. All eight new hunting connectors also received desktop and mobile/reduced-effects checks as recorded at the end of this note.

Reviewed desktop captures at 1440×1000. The final built-output review inspected contact sheets 4–7: 28 maps across arrival, middle, upper and exit views (112 panels), covering all 24 shop interiors and four late-game combat maps. Four full-size final captures received additional detail checks. Earlier review inspected 16 full-size shop captures across two regions and both renderers, plus two field captures. These are the manually inspected subsets; this note does not claim individual full-size inspection of all 448 captured views or mobile behavior.

## Combined review coverage

The primary implementation review also inspected contact sheets 0–3, covering the other 28 maps at arrival, middle, elevated and exit positions. Together the two reviews cover all 56 maps at contact-sheet scale. The all-map capture set has 448 views across Canvas and Pixi; separate deep mobile/reduced-effects captures contain 128 views. Eight representative original-resolution before/after pairs are preserved in [scene-review.html](scene-review.html); the complete local viewer is available at http://127.0.0.1:4187 while its review server runs.

## Result

The new neutral gray-taupe masonry and walnut floor is acceptable in all four shop rooms. It fits the warm walls and room floor, keeps a continuous level top edge, and gives the compact characters a clear standing surface. No violet glass, vegetation, glow, missing images or new foreground occlusion is visible. Weapon, armor, supply and star-room identities remain distinct. Back-wall displays are larger than the compact characters but read as scenery; the lower actor lane remains open. The Cinder arrival and exit views keep the vendor and return route visible.

Canvas and Pixi agree on the room, floor material, floor height and character placement. The earlier vendor-label contrast issue is fixed in the final built output: white text on dark nameplates is readable in Crossing Special (Canvas) and Cinder Armor (Pixi). The HUD also uses a consistent dark treatment. Player and vendor use the same compact actor appearance in these captures, preserving current game behavior.

## Final built-output checks

Directory: `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/verified/`

| Contact sheet | Maps inspected in four views per row |
| --- | --- |
| contact-4.png | Rimewarden Vault, Stormbreak Aerie, Astral Stacks, Eclipse Throne, Crossing Weapon, Crossing Armor, Crossing Supply |
| contact-5.png | Crossing Special, all four Rustcoil shops, Cinder Weapon, Cinder Armor |
| contact-6.png | Cinder Supply, Cinder Special, all four Frostfen shops, Stormbreak Weapon |
| contact-7.png | Stormbreak Armor, Stormbreak Supply, Stormbreak Special, all four Astral shops |

All 24 interiors consistently use the correct purpose-specific room and neutral floor. Furniture remains behind the actor lane, arrival portals are visible, and no outdoor building leftovers, missing textures, new foreground obstruction or material discontinuity were apparent. The middle and upper views repeat naturally in these one-lane rooms. At contact-sheet scale, all four late-game maps have coherent regional materials and readable platform silhouettes and ladders.

Additional full-size files inspected:

- `starfallCrossingSpecialShop-middle-canvas.png`: corrected vendor nameplate and HUD contrast; room, actor scale and floor accepted.
- `cinderRefugeArmorShop-middle-pixi.png`: corrected vendor nameplate; warm interior and neutral foreground remain coherent.
- `astralStacks-upper-pixi.png`: celestial masonry and book towers remain visually distinct from the platform edges; ladder and compact actors are readable.
- `eclipseThrone-upper-pixi.png`: dark violet foreground separates from the distant gold throne; platform and ladder connections remain clear.

No additional blocking scenery or overlap defect was identified in this final manual subset. Static screenshots establish presentation only; traversal, spawning, combat, reduced-effects behavior and mobile layouts require their separate checks. No asset or runtime edits were made during this final visual QA pass.

## Earlier full-size shop captures

Directory: `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/final-interiors/`

| Room / view | Pixi | Canvas |
| --- | --- | --- |
| Crossing weapon / middle | starfallCrossingWeaponShop-middle-pixi.png | starfallCrossingWeaponShop-middle-canvas.png |
| Crossing armor / middle | starfallCrossingArmorShop-middle-pixi.png | starfallCrossingArmorShop-middle-canvas.png |
| Crossing supply / middle | starfallCrossingSupplyShop-middle-pixi.png | starfallCrossingSupplyShop-middle-canvas.png |
| Crossing special / middle | starfallCrossingSpecialShop-middle-pixi.png | starfallCrossingSpecialShop-middle-canvas.png |
| Cinder weapon / arrival | cinderRefugeWeaponShop-arrival-pixi.png | cinderRefugeWeaponShop-arrival-canvas.png |
| Cinder armor / middle | cinderRefugeArmorShop-middle-pixi.png | cinderRefugeArmorShop-middle-canvas.png |
| Cinder supply / middle | cinderRefugeSupplyShop-middle-pixi.png | cinderRefugeSupplyShop-middle-canvas.png |
| Cinder special / exit | cinderRefugeSpecialShop-exit-pixi.png | cinderRefugeSpecialShop-exit-canvas.png |

## Earlier field checks

Directory: `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/final/`

- `ashglassPass-middle-pixi.png`: the revised dark narrow bevel gives the ground a continuous top; the previous bright gray slab steps are gone. Rock-cell repetition remains visible as authored material variation, without a new contact-height discontinuity. Glass/basalt materials fit the surrounding region.
- `cinderHollow-middle-pixi.png`: volcanic spire and lava remain in atmospheric distance. Removed foreground shelves no longer compete with the actual lane edges. The player and enemy silhouettes remain distinguishable from the background.

These field files precede the final Pixi fade integration, so this review does not assess that later world-bottom treatment. No asset or runtime edits were made during this visual QA pass.

## Final hunting-connector checks

Inspected all 32 original PNGs in `C:/Users/clopt/AppData/Local/Temp/starfall-map-pass/connector-final/` and `connector-final-mobile-reduced/`: the eight filename stems below, each with both `-canvas.png` and `-pixi.png` in both directories. The capture manifests identify the final local built route, `http://127.0.0.1:4174/games/project-starfall`, and report no capture errors.

| Map | Exact filename stems |
| --- | --- |
| Cinder | `cinderHollow-cinder_hollow_hunting_bridge_1`, `cinderHollow-cinder_hollow_hunting_ramp_1` |
| Quarry | `orebackQuarry-oreback_quarry_hunting_bridge_1`, `orebackQuarry-oreback_quarry_hunting_bridge_2` |
| Ashglass | `ashglassPass-ashglass_pass_hunting_bridge_1`, `ashglassPass-ashglass_pass_hunting_bridge_2` |
| Stormbreak | `stormbreakCliffs-stormbreak_cliffs_hunting_bridge_1`, `stormbreakCliffs-stormbreak_cliffs_hunting_bridge_2` |

No new blocking visual defect was found. The connector tops form readable continuous standing surfaces; their ends meet the adjacent platforms, and actors remain clear of foreground scenery. Quarry retains warm masonry, Ashglass dark volcanic glass, Cinder orange volcanic ledges, and Stormbreak pale wind-weathered stone. Canvas and Pixi agree on support height and material placement. Cinder's shallow slope has a visible texture-density change at the flat transition, but no apparent gap or foreign material. The mobile/reduced-effects views preserve the actor lane and touch controls; canvas HUD text remains small at this inherited portrait scale.

These stationary captures cover presentation, not smooth traversal or danger-marker behavior during attacks. The separately measured Cinder player descent can briefly lose its grounded flag within about four pixels of the slope; these screenshots do not establish continuous grounding. Loading toasts and randomly chosen enemies differ between captures. No runtime or asset edits were made for this review.

## Final overlapping-effect fixtures

The [16 controlled charge/heal cases](warning-verification.json) cover a bright Frostfen background and a dark Eclipse background, Canvas/Pixi, normal/reduced effects, and desktop/portrait viewports. Preparation leaves HP unchanged; a deliberately resolved healing contact changes recipient HP and displays the mint pulse while the coral charge footprint remains present. These fixtures deliberately place the same actors in both regions to test readability; they are not evidence of authored biome rosters or a complete timed attack playtest. Fresh captures clear expired-style notification banners and use the player's actual maximum HP. Examples: [desktop bright/reduced](scenes/frostfenOutskirts-pixi-reduced-desktop.png), [portrait dark/reduced](scenes/eclipseFrontier-canvas-reduced-mobile.png). Portrait silhouettes and warning colors remain identifiable, but the small world/HUD text limitation recorded in the deep review remains.
