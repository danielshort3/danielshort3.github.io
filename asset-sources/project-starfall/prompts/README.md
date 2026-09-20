# Project Starfall Prompt Templates

The illustrated-v1 production sources live under `asset-sources/project-starfall/overhaul-v1/{players,enemies,fx,scenery,icons}`. Use the [ownership table](../../../docs/project-starfall/ASSET_GENERATION_GUIDE.md#illustrated-production-ownership) and [migration report](../../../docs/project-starfall/ASSET_OVERHAUL_V1.md) to choose the importer. Classic, Fracture Runner, cyan-grid sources, and earlier generation briefs remain historical. Approved session study images and their original prompts are immutable references.

Use these templates with `docs/project-starfall/ASSET_GENERATION_GUIDE.md`. The guide remains authoritative for full prompts, frame counts, dimensions, and review rules.

Start with the [confirmed art direction and references](../../../docs/project-starfall/ASSET_GENERATION_GUIDE.md#confirmed-art-direction-and-references). The approved Oracle and Glowcap studies guide both appearance and motion: clean illustrated fantasy, crisp contours, controlled shading, clear features, and minimal visual noise. Transfer their clarity and deliberate articulation while preserving each subject's identity and anatomy. Their historical generation prompts remain provenance; strict pixel-grid, fixed palette-count, and fixed shading-tone instructions from those prompts are not universal style requirements.

## Required Combat Brief

Prepend a completed copy of this brief to every combat actor, skill icon, or effect prompt. Follow [Combat Visual Language v1](../../../docs/project-starfall/ASSET_GENERATION_GUIDE.md#combat-visual-language-v1) for the category symbols, contour language, timing, and review criteria. Functional meaning comes first; class, region, and element are secondary accents. A class palette must not replace the functional color or symbol.

Choose the functional category and its exact hex together: healing `#62D995`, damage/danger `#F06A60`, buff `#F2C45E`, shield `#63D7E8`, debuff `#B88AF3`, or resource `#668FFF`. Use the canonical symbol as well as color. For an action with multiple functions, specify each cue and its event separately instead of blending their meanings into one aura.

For movement, summons, or cosmetic actions with no functional effect, use neutral pearl with character accents as specified by the canonical standard; mark the gameplay category and category symbol as not applicable. If the action also deals damage, heals, protects, or changes status, include each actual functional cue. Do not assign a healing or buff color solely because an action looks magical.

```text
Canonical standard: Project Starfall Combat Visual Language v1, supplied with this prompt.
Art direction: Project Starfall Confirmed Art Direction and References, supplied with this prompt. Clean illustrated fantasy with crisp contours, controlled shading, clear features and minimal noise; warm adventure with real threat where the subject calls for it.
Asset and action: [ASSET_ID], [ACTION], [ACTOR POSES / STANDALONE FX / STATIC SKILL ICON]. Gameplay purpose: [WHAT THE VIEWER MUST UNDERSTAND].
Functional cue: [CATEGORY or NO FUNCTIONAL EFFECT], [EXACT CATEGORY HEX or NEUTRAL PEARL], [CANONICAL SYMBOL AND SHAPE or N/A]. Keep applicable symbols recognizable at gameplay size.
Ownership and recipients: [FRIENDLY or ENEMY OWNER], [SELF / ALLIES / ENEMIES / GROUND AREA], [ACTUAL RECIPIENT OR TARGET]. Use the canonical smooth outer contour for friendly effects or segmented outer contour for enemy effects; retain the category color and symbol in both cases.
Secondary accents: [CLASS / REGION / ELEMENT, ACCENT COLORS AND MOTIFS]. These support the functional cue without replacing or obscuring it. Preserve the approved actor's costume palette.
Gameplay timing: [TOTAL ACTION DURATION IN MS], [EXACT COMMIT / CONTACT / APPLICATION EVENTS]. Do not invent a gameplay delay to fit a decorative effect.
Preparation: [START-END MS, READABLE WINDUP POSE OR WARNING CUE].
Commit: [START-END MS, FINAL AIM-LOCK CUE AND WINDOW, LOCKED DIRECTION OR AREA]. The commitment window is part of the windup, not an additional delay.
Release: [PROJECTILE RELEASE TIMESTAMP IN MS AND VISIBLE RELEASE CUE, or N/A]. Keep release separate from the commitment window and later projectile contact.
Contact: [START-END MS, EXACT HIT / HEAL / PROTECTION / STATUS APPLICATION MOMENT AND LOCATION].
Active: [START-END MS, EFFECT OR STATUS DURATION, WHAT PERSISTS].
Recovery: [START-END MS, SETTLE / DISSIPATION / RETURN TO READY].
Phase-to-frame map: [ORDERED FRAME INDICES AND DURATIONS, INCLUDING HOLDS]. If phases coincide or do not apply, state that explicitly and explain why; do not add empty timing beats. For static icons, mark animation timing fields N/A. For purely cosmetic motion, specify its animation timing but mark gameplay release/contact events N/A.
Geometry and tracking: [ACTUAL AFFECTED FOOTPRINT OR PATH], [EARLY TRACKING / FIXED AIM / EXPLICIT HOMING CUE], [WHEN THE AREA OR DIRECTION LOCKS]. Warning geometry must match the gameplay area; do not silently move it after commitment. Mark fields that do not apply N/A.
Lifecycle: [CANCELLATION BEFORE AND AFTER RELEASE], [WHAT REMAINS ACTIVE], [PERSISTENT DANGER OR STATUS CUE], [EXACT CUE END EVENT]. Remove a warning when its pending effect is cancelled, but preserve cues for released projectiles or hazards that remain dangerous. Do not restart a full windup for every periodic tick.
Reference roles and locks: [SUBJECT IDENTITY REFERENCE], [APPROVED APPEARANCE AND MOTION REFERENCES], [APPROVED POSE SHEET IF PRESENT]. Use the approved Oracle and Glowcap studies for both appearance and motion quality without transferring their creature identity. Lock the subject's face, anatomy and limb count, costume, landmarks, palette, materials, contours, detail scale and lighting. State the intended role of any additional reference.
Anatomy and motion: [BODY STRUCTURE, WEIGHT, MATERIAL FLEXIBILITY, CONTACTS AND SECONDARY MOTION]. Mushrooms may spring and squash, plants may flex through stems, petals and vines, and humanoids need solid weight, joint articulation and convincing foot contact. Adapt the motion to the subject; do not give every creature the same elastic bounce.
Anchor and scale: [FIXED LOCAL PIVOT X/Y], [FOOT CONTACT Y OR HOVER CENTER], [ACTOR DRAW HEIGHT AND ONE SHARED SCALE], [CAMERA / FACING], [INTENDED ROOT MOTION IF ANY]. Keep the camera and scale fixed; do not normalize each pose independently by its bounding box.
Registration evidence: [ANATOMICAL X LANDMARK PER POSE], [INDEPENDENT SECOND LANDMARK OR RIGID PIXEL PATCH], [EXACT ACTIONS/AXES TO REVIEW], [LOOP SEAM AND ACTION TRANSITIONS], [KNOWN LIMITATIONS]. Uneven gutters require measured source-space anchors; nominal grid centers and bounding-box centers do not prove alignment. Preserve intended lean, travel, squash and articulation. Confirm the packed pixels independently rather than checking the same anchor values used to place them.
Layering and attachment: [BEHIND ACTOR / IN FRONT / GROUND], [CASTER OR RECIPIENT ATTACHMENT], [FX ORIGIN X/Y RELATIVE TO ACTOR PIVOT], [FACE / FEET / HAZARD CLEAR AREAS]. Preserve visibility of the actor and gameplay cues.
Enemy combat body: [VISIBLE ANATOMY AND INTENTIONAL TRANSPARENT GAPS]. Keep shadows, aura and effects out of the actor sheet. Production combat masks follow actor pixels at alpha >= 64/255 in the current frame with the same registration, facing, scale and recoil as the drawing; empty cell padding is never hittable. Keep terrain collision and declared attack/hazard reach separate.
Effect emphasis: [EVERYDAY ACTION or MAJOR MOVE], [INTENDED VISUAL EMPHASIS]. Keep everyday FX compact and readable; give major moves greater emphasis while preserving category symbols, actor visibility and hazard clarity.
Output contract: [PRODUCTION CONTRACT ID or REVIEW STUDY], [CANVAS WIDTH/HEIGHT], [COLUMNS/ROWS], [CELL WIDTH/HEIGHT], [GUTTER], [ROW-MAJOR POSE ORDER], [LOOP or ONE-SHOT], [RUNTIME ALPHA or EXISTING PROCESSOR SOURCE BACKGROUND]. No labels, baked checkerboard, or cell overflow.
Scope lock: [EXACT NEW OUTPUTS]. Preserve approved actor poses when requesting FX. Generate actors and effects as separate aligned layers. Do not revise identity, poses, export contracts, or gameplay timing unless explicitly requested.
```

The prompt must include the relevant canonical rules in readable form when the generator cannot open the linked guide. Fill every field; do not leave placeholders for the generator to guess. Current player production is 8x10 cells of 160px; enemy production is 6x8 cells of 160px. A differently sized review study still does not change the production contract. Use measured source cells, the existing shared-scale contract, explicit anatomical registration, and the existing importer. Follow [anatomical registration and scoped motion review](../../../docs/project-starfall/ASSET_GENERATION_GUIDE.md#anatomical-registration-and-scoped-motion-review): independently inspect the packed pixels for each accepted action and seam, and record the precise scope in `source.json` and the import report. An idle-horizontal check does not certify other rows, vertical motion, anatomy or runtime timing.

A clearly labeled pose-only study may be reviewed as an intermediate step. To review an action as complete, show the approved actor and all required FX together with their shared timing, anchors, and intended gameplay event. An art-only preview does not prove runtime event synchronization.

For every accepted enemy repaint or repack, follow [Enemy combat body masks](../../../docs/project-starfall/ASSET_GENERATION_GUIDE.md#enemy-combat-body-masks): enemy activation regenerates masks automatically; other import workflows must run `node build/generate-project-starfall-enemy-hurtboxes.js`. Require its `--check` mode and `npm run test:starfall:hitboxes`, and inspect overlays through every changed action in both facings and recoil. The JavaScript build rejects stale masks. Registration-only changes also require this review. Do not substitute a broad bounding rectangle for the visible body or count separate FX as anatomy.

## Master Style Prompt

```text
Project Starfall Starlit Frontier Fantasy, original 2D side-scroller fantasy RPG asset, clean illustrated fantasy, crisp readable contours, controlled shading, clear facial and material features, minimal visual noise, warm adventure with real threat where appropriate, luminous fallen-star magic accents, blue-and-gold guild motifs, practical crafted fantasy materials, readable at gameplay scale, consistent upper-left/front lighting, no copied IP.
```

Use the approved Oracle and Glowcap studies as appearance and motion references under the canonical art direction. Preserve the subject's approved identity, proportions, and readable finish without imposing a universal pixel grid, palette count, or number of shading tones. Decorative blue/gold motifs remain secondary to functional combat cues.

For map backgrounds, replace sprite outline language with:

```text
illustrated side-scroller depth, softly detailed distant scenery with atmospheric perspective, crisp nearby surfaces and gameplay edges, clear gameplay lanes, readable foreground/midground/background separation
```

## Master Negative Prompt

```text
text, labels, numbers, watermark, signature, logo, fake UI, unrelated background clutter, photorealism, 3D render, isometric gameplay view, top-down gameplay view, copied IP, copied MapleStory sprite, inconsistent style, warped anatomy, extra limbs, missing limbs, broken hands, broken feet, melted outlines, noisy unreadable details, over-bloom, muddy colors, cropped subject, cell overflow, inconsistent lighting, inconsistent proportions, changing costume, changing weapon, changing face
```

For source sheets, also add:

```text
guide grid color inside artwork, chroma key color inside artwork, row labels, column labels, frame numbers, character crossing grid lines
```

## Player Sheet Template

```text
Create an original Project Starfall player class animation source sheet for [CLASS_NAME], a compact heroic 2D side-scroller fantasy RPG character.
Apply the completed Required Combat Brief and Combat Visual Language v1 to every combat action row; specify each action's function, ownership, recipients and timing separately.
Style: Starlit Frontier Fantasy, clean illustrated fantasy with crisp contours, controlled shading, clear features and minimal noise. Use the approved appearance references and preserve the current compact player silhouette and proportions; do not redesign the body around a numeric head-height rule. No copied IP.
Character lock: [HAIR/HELMET], [FACE VISIBILITY], [OUTFIT], [WEAPON OR FOCUS], [CLASS COLORS], [SIGNATURE ACCESSORY]. Keep these identical in every frame.
Runtime layout: 8 columns x 10 rows, 160px cells, 1280x1600. Generate [REQUESTED ACTION ROWS] as eight-pose strips or action-pair masters with genuine alpha, empty gutters, no drawn grid, one complete actor per cell, right-facing, one shared scale and a declared foot/hover anchor. The importer assembles the full runtime sheet.
Rows: idle, run, jump, fall, climb, basic, skill, party, hit, defeat.
Draw articulated poses with intentional preparation, commit, contact, active and recovery beats mapped to the existing rows. Give the humanoid solid weight, believable joint movement and clear foot contact; use the approved studies' motion clarity without borrowing mushroom spring or plant flex. Keep actor artwork separate from functional FX; class colors describe identity, not the effect's function.
```

## Enemy Compact Sheet Template

```text
Create one original Project Starfall compact enemy animation source sheet for [ENEMY_NAME].
Apply the completed Required Combat Brief and Combat Visual Language v1 to each combat action. For the buff row, name the actual function, such as healing or shield, rather than treating every support or special action as the same visual effect.
Style: Starlit Frontier Fantasy, expressive illustrated 2D side-scroller RPG monster, crisp contours, controlled shading, clear features, minimal noise, readable silhouette at 160px and gameplay size, approved proportions and detail scale, no copied IP. Keep the world's warm adventure tone while making dangerous creatures feel threatening through their design and poses.
Enemy design: [VISUAL DESCRIPTION AND LOCKED ANATOMY / LANDMARK COUNTS]. Gameplay role: [ROLE]. Attack tell: [PREPARATION CUE]. Regional palette: [REGION PALETTE, SECONDARY TO FUNCTIONAL FX CUES]. Preserve the actor's established palette and identity across all rows.
Runtime layout: 6 columns x 8 rows, 160px cells, 960x1280. Author six distinct poses per row, right-facing, one complete enemy per cell, one shared scale and a declared foot/hover anchor, generous empty gutters, genuine alpha. No cyan guide lines, chroma background, labels, or manufactured motion from resized duplicates.
Rows: idle, move, telegraph, attack, projectile, buff, hit, defeat.
Motion: [ANATOMY, WEIGHT AND MATERIAL BEHAVIOR]. Use the approved Oracle and Glowcap references for appearance and articulated motion quality. Mushrooms can spring, plants can flex, and humanoids should carry solid weight; preserve this enemy's own anatomy and identity.
Keep actor poses separate from functional FX. Map the brief's timing to the existing row contract; do not add cells or replace a missing anticipation pose with a color flash.
Preserve transparent spaces between limbs and around the silhouette. Do not bake shadows, aura or particles into the actor layer: production alpha defines the enemy's hittable body, and cell padding must remain empty.
```

## Actor Action Study Template

Use this with the completed common brief for a focused pose review before a production import.

```text
Create [FRAME COUNT] deliberately drawn poses of [APPROVED ACTOR] performing [ACTION], following the completed Required Combat Brief and Combat Visual Language v1.
Use [SUBJECT IDENTITY REFERENCE] to preserve the same face, anatomy, landmark counts, costume, materials, contours, palette and lighting in every pose. Use the approved Oracle and Glowcap studies for both clean illustrated appearance and deliberate articulated motion, as specified in the brief. Adapt spring, flex or solid weight to this actor's anatomy; keep its identity locked.
Show distinct articulated preparation, commit, contact, active and recovery poses using the supplied phase-to-frame map. Do not substitute translated, stretched, recolored or near-identical copies of one pose for actual articulation.
Keep the brief's pivot, contact baseline, shared scale, camera and facing consistent, with only declared root motion. Draw the complete actor inside every cell with the specified padding.
Output one actor-only sheet with genuine transparent alpha and the exact study grid specified in the brief. Omit aura, particles, symbols, ground shadows and other FX; list the separate FX layer required to complete the action in the accompanying delivery notes. Label this a pose-only intermediate study until the actor and all required FX are shown together in the combined action review.
If approved poses are already supplied and only effects are requested, preserve the pose sheet unchanged and use the Standalone Combat FX Template instead.
This is a review study, not a change to production sheet layout or gameplay timing.
```

## Item Icon Template

```text
Create a Project Starfall 64x64 transparent fantasy RPG item icon for [ITEM_NAME].
Style: Starlit Frontier Fantasy, clean illustrated 2D icon, crisp silhouette, controlled shading, clear material features, minimal noise, dark edge separation, readable at 64px, [TIER MATERIALS], [REGION OR CLASS MOTIF].
Composition: one centered item, no background, no UI border, no rarity frame, no quantity number, no label, no watermark.
Use palette: [PALETTE]. Leave transparent padding around the item.
```

## Map Background Template

```text
Create a 1280x640 Project Starfall side-scroller panoramic background for [MAP_NAME].
Style: Starlit Frontier Fantasy, clean illustrated fantasy environment, warm adventure with real threat where appropriate, readable side-scroller depth, clear gameplay lanes, layered parallax feel, no characters, no monsters, no UI, no text.
Region: [REGION DESCRIPTION]. Palette: [PALETTE]. Required landmarks: [LANDMARKS].
Context: [OUTDOOR BIOME OR ENCLOSED ROOM PURPOSE]. Actual playable floor, portal and actor clear areas: [POSITIONS AND MATERIALS]. For interiors, show purposeful back-wall displays and a clear lower actor lane; exclude outdoor buildings, sky and meadow dressing.
Composition: distant scenery is softly detailed with atmospheric depth; nearby surfaces, platforms, hazards and gameplay edges are crisp and easy to read. Keep detail away from the main action's visual focus, and let the midground support the regional theme. Compose a complete bounded panorama for cover scaling, not a periodic texture or a scene with blended duplicate edges.
Keep painted shelves, stairs and ledges behind the gameplay plane. Do not imply extra footholds or exits. Actual traversable surfaces and danger boundaries are separate runtime layers.
```

For terrain briefs, specify a shared level contact edge and bevel thickness across repeated cells, compatible joining materials and distinct authored interior variations. Avoid broad raised caps that make each repeat read as a separate step. Record measured contact-row crops in the scenery ledger and review floor/ramp joins with actors in both renderers; matching edge colors alone does not prove a continuous surface.

## Skill FX Template

```text
Create a Project Starfall skill VFX source sheet for [SKILL_NAME], 960x640 target, 6 columns x 4 rows, 160x160 frames.
Apply the completed Required Combat Brief and Combat Visual Language v1, including the functional category, exact hex, canonical symbol, ownership, recipients, and all five timing stages.
Style: clean illustrated 2D fantasy combat VFX, Starlit Frontier Fantasy, [APPROVED ACTOR-COMPATIBLE FX STYLE], crisp readable shapes, controlled shading and luminosity, minimal noise, no character body, no UI, no text, no background clutter.
Rows: cast, projectile, impact, area.
Primary palette and shape: [FUNCTIONAL CATEGORY HEX AND CANONICAL SYMBOL]. Secondary accent: [CLASS / ELEMENT COLOR AND MOTIF].
Use the brief's friendly smooth or enemy segmented outer contour, attachment point and layer order. Keep the fixed local FX origin and cell bounds; leave the specified actor and gameplay clear areas visible.
Keep everyday FX compact and give major moves greater emphasis as specified in the brief, without obscuring actors, category symbols or incoming danger.
Map action phases to these production rows and frames without changing their meanings. Keep all actors and approved poses outside this FX sheet.
```

## Standalone Combat FX Template

Use this with the completed common brief to add or revise effects while preserving an approved actor sheet. A separately generated layer is still required to align with the actor's action and contact/application event.

```text
Create a standalone transparent combat FX atlas for [ACTION / EFFECT], following the completed Required Combat Brief and Combat Visual Language v1.
Functional meaning: [CATEGORY], exact [HEX], recognizable [CANONICAL SYMBOL AND SHAPE]. Owner: [FRIENDLY or ENEMY]. Recipients: [WHO OR WHAT RECEIVES THE EFFECT]. Use the required ownership contour without changing the functional meaning.
Use [APPROVED ACTOR SHEET] only for visual style, scale, attachment and timing alignment. Preserve it unchanged. Draw no actor, face, body, mannequin, humanoid silhouette, or replacement pose inside the FX atlas.
Draw distinct FX stages using the supplied preparation, commit, contact, active and recovery timings and phase-to-frame map. Make the contact/application cue occur at [SPECIFIED GAMEPLAY EVENT]. Let secondary [ELEMENT / CLASS] accents support the function without replacing its symbol or color.
Keep [FIXED FX ORIGIN] in every cell. Use [BEHIND / FRONT / GROUND LAYER] with [ATTACHMENT OFFSET], and keep [FACE / FEET / HAZARDS / CENTRAL ACTOR AREA] clear as specified. Do not recenter individual frames by their visible bounds.
Match the approved clean illustrated finish, contours, controlled shading and palette treatment. Keep everyday FX compact; give major moves greater emphasis while preserving readable symbols and gameplay visibility. Avoid unrelated projectiles, slashes or decorative effects.
Output exactly [CANVAS], [COLUMNS x ROWS], [CELL SIZE], [GUTTER] in chronological row-major order with genuine transparent alpha, including transparent empty areas. No grid, labels, text, background, baked checkerboard or cropped particles.
Respect [LOOP OR ONE-SHOT] and the supplied final recovery state. This output is [REVIEW STUDY OR NAMED PRODUCTION CONTRACT]; do not infer a runtime contract change from a review grid.
```

## UI Icon Template

```text
Create a 64x64 transparent Project Starfall menu icon for [ICON_ID].
Style: clean fantasy RPG UI icon, dark navy/gold/cyan Starfall palette, readable silhouette, simple symbol, subtle bevel, no text, no numbers, no background panel unless the symbol requires a small internal shape.
The icon must be clear at 32px and 64px.
```
