# Starfall enemy sprite and hitbox audit

The old combat rectangle did include substantial empty space. The independent [pixel audit](../../output/starfall-hitbox-review/enemy-alpha-audit.md) found **13 enemy IDs with active poses containing hittable points at least 20 gameplay pixels from visible anatomy**. Lava Tick reached **26.6 px** in its first telegraph pose; Stormbreak Roc reached **55.16 px** in its third projectile pose. These are measured lower bounds on the largest empty radius, at actual gameplay scale. Glassback and Shardling share a sheet, so enemy IDs are not a count of unique drawings.

The rectangle came from the stable terrain body, while sprites used different visual sizes and per-enemy registration. Checking sprite alignment did not establish correct combat collision. A bounding rectangle around visible pixels would still count empty corners and holes, so replacing one rectangle with another would not resolve the full problem.

## Correction

[Generated combat masks](../../js/games/project-starfall/data/enemy-hurtboxes.js) now cover **48 production sheets, all 2,304 frame slots, and all 51 runtime enemy IDs**. Each mask includes exactly the actor's source pixels with alpha at least 64/255. Lossless scanline encoding preserves transparent gaps; decoding is cached per frame. The [runtime helper](../../js/games/project-starfall/engine/enemy-hurtboxes.js) applies the current frame's registered drawing transform, facing and hit-reaction placement. Rectangle and circle queries reject empty mask space, and targeting aims at a visible body pixel.

Melee, projectiles, charge contact, target clicks, areas, traps and enemy field checks use this combat geometry. Spatial candidate searches include the drawn extents so a visible extremity outside the old terrain box can still be hit. Ground/lane constraints remain where the ability requires them. Terrain `x/y/w/h`, navigation and enemy separation retain their existing bodies. Enemy attack reach and declared hazards remain separate from the actor's hurtbox.

Pixi now tries the production sprite before its older procedural fallback for Glassback, Rift Lantern and Fault Skitter. This removes a renderer-specific drawing override that otherwise would not match the production masks. Canvas and normal loaded-sprite Pixi rendering use the same registered geometry.

## Evidence and review

Open the [interactive hitbox review](../../output/starfall-hitbox-review/index.html) to compare the old body rectangle with the current mask, step poses, change facing and inspect empty regions. [Its builder](../../build/generate-project-starfall-hitbox-review.js) owns the local review output. The [before overlays](../../output/starfall-hitbox-review/old-body-overlays.png) retain the original mismatch evidence.

[Independent current-mask verification](../../output/starfall-hitbox-review/current-mask-verification.json) compared **62,668,800 source pixels** with decoded occupancy across every registered pose. It also checked **5,508 frame/facing/reaction cases**, **220,320 visible** and **220,320 transparent** rectangle/circle queries, and rejected **5,446 applicable old empty-space witnesses**. Separate geometry tests cover hollow/disconnected shapes, codec edge cases, visible aiming, both facings and recoil-sized transforms; all 2,304 source frames and 2,448 enemy/frame mappings resolve to masks.

Alpha 64 is a deliberate raster-edge threshold: fainter antialiasing is not hittable. Scaling/filtering can soften the displayed edge at subpixel positions, so this is an exact source-mask guarantee, not a claim that every screen-space antialiasing sample is a separate collision pixel. Cosmetic FX, shadows and warnings are separate layers and do not enlarge the enemy body. This audit does not certify the drawing quality or seamlessness of every animation, nor redefine player hurtboxes or enemy weapon/hazard reach.

Exact masks occupy approximately **340 KB gzip** in a separate content-hashed script. The start screen does not request it; selecting Start loads it before character selection can open, with a disabled action, loading status and explicit retry after failure. The initial game bundle is about **1.08 MB gzip**, while both scripts together remain within the existing **1.5 MiB gzip** and **5 MiB raw** limits. The complete initial route is about **1.48 MB gzip** against its unchanged **1.7 MB** delivery budget. The same exact masks preserve pixel-level transparent gaps without runtime image readback.

## Regeneration

[The generator](../../build/generate-project-starfall-enemy-hurtboxes.js) records each production sheet's SHA-256, frame dimensions and threshold. [Enemy activation](../../build/integrate-project-starfall-overhaul-enemies.js) regenerates masks automatically after applying accepted imports. Other accepted repaint/repack workflows must regenerate explicitly. Require both freshness and collision checks:

```bash
node build/generate-project-starfall-enemy-hurtboxes.js
node build/generate-project-starfall-enemy-hurtboxes.js --check
npm run test:starfall:hitboxes
```

`--check` exits with an error for stale production inventory hashes or generated masks, and [the JavaScript build](../../build/build-js.js) runs it before bundling. `test:starfall:hitboxes` combines freshness, geometry, source-pixel and combat-path tests and is included in `test:starfall:assets`. Registration-only changes also require overlay and collision verification because they change the world transform. Keep these requirements with the [asset guide](ASSET_GENERATION_GUIDE.md#enemy-combat-body-masks) and [generation briefs](../../asset-sources/project-starfall/prompts/README.md).
