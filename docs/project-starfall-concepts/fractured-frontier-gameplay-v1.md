# Project Starfall: Fractured Frontier vertical slice

This document turns `fractured-frontier-gameplay-v1.png` into the production target for Project Starfall's opening 90 seconds. The goal is to replace the current Maple-like starter impression with a coherent identity that can later scale across the rest of the game.

## Creative thesis

Project Starfall is a frontier action RPG built in the aftermath of a fallen celestial structure. Every opening-screen decision should reinforce expedition life, unstable starstone, improvised repair craft, and dangerous meteor ecology.

The opening should not use bright generic meadow art, cute slime silhouettes, stacked grass-block lanes, chibi anime proportions, blue-gold filigree, parchment windows, or a portfolio-style marketing wrapper.

## Primary screen composition

- Full 1280 x 806 game surface whenever the viewport can support it.
- One safe arrival shelf with a warm repair camp.
- One low combat basin that teaches the Fighter's contact rhythm.
- One elevated shortcut/reward pocket with a clear return path.
- One fractured starstone bridge that reads as a traversal landmark.
- One distant cyan exit beacon that gives the route a visible goal.
- Asymmetrical terrain silhouettes; no repeated three-lane cluster template.

## Visual system

| Role | Treatment |
| --- | --- |
| Background | Deep indigo sky, charcoal star ribs, cool atmospheric depth |
| Traversable terrain | Weathered umber and pale stone with crisp top edges |
| Starfall material | Cyan-white fractured seams; never generic neon outlines |
| Wayfinding | Restrained amber lamps, objective marks, and repair cloth |
| Danger | Desaturated red only for telegraphs and enemy health |
| UI surfaces | Translucent charcoal/navy with thin neutral borders |
| Typography | Star-white, compact, sans-serif, minimum 12 CSS-equivalent pixels |

## HUD hierarchy

- Upper left: one objective line only.
- Upper right: compact minimap with current route and destination.
- Lower left: level, HP, class resource, and one compact utility slot.
- Lower center: five core actions with clear cooldown and input states.
- Lower right: two consumables.
- Secondary MMO systems live behind deliberate panels, not on the default combat screen.

## Opening enemies

- **Glassback:** angular ground scavenger with a star-glass mantle, a low charge tell, and a readable recoil/break state.
- **Rift Lantern:** hovering void fauna with a luminous core, a short aim tell, and a slow projectile that teaches vertical repositioning.

Neither enemy should share a slime, mushroom, blob, mascot, or generic chibi silhouette.

## Player and combat motion

- Grounded frontier armor proportions with readable weapon reach.
- Basic Fighter attack uses anticipation, contact, and recovery phases.
- Damage lands on the contact frame.
- Standard hit uses 45 ms hitstop, restrained camera impulse, enemy recoil/squash, a contact-synchronized sound, and one bright directional spark.
- Reduced-effects mode removes camera impulse and shortens the flash without changing timing or damage.

## Asset inventory

- Gameplay concept: `fractured-frontier-gameplay-v1.png`.
- Three parallax layers: far sky/ribs, middle canyon, near frontier silhouettes.
- Seam-safe ground top, body, ledge, support, ramp, and fractured-bridge pieces.
- Arrival camp props, amber lamps, repair cloth, beacon, and return gate.
- Glassback idle/move/telegraph/attack/hit/defeat states.
- Rift Lantern idle/move/telegraph/attack/hit/defeat states.
- Fighter idle/run/jump/attack/hit/defeat states.

## Intentional implementation sequence

1. Fix the embedded game shell and readable scale.
2. Author the opening route geometry and its landmark beats.
3. Replace the background, terrain, return gate, two starter enemies, and Fighter presentation.
4. Add contact-frame combat feedback.
5. Compact the default HUD.
6. Validate the first 90 seconds before propagating the system to later regions.

