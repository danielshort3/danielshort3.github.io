# Starfall animation samples

Review-only output for the September 19, 2026 before/after request. No production game assets, animation definitions, or renderers were edited.

`starfall-before-after.html` is the self-contained inline comparison. It contains original PNG row pixels and applies the proposed source-pixel drawing offsets only to the corrected side. Both sides use one clock, identical frame order and holds, and the same scale. No crossfade, morphing, or frame deletion hides pose differences.

Included samples:

- Icebloom Oracle idle: x offsets `[0, -2, -10]`, all y offsets zero. Ground-support drift reduced from 10 to 0.25 source pixels. Original 2-second loop with holds `[4, 2, 4]` at 5 fps.
- Fox idle: x offsets `[0, -2, 2, 1, 5, 12]`, all y offsets zero. Paw-support drift reduced from 14.5 to 0.5 source pixels. Original 1-second loop at 6 fps.

`registration.json` records source hashes, offsets, landmark methods, and diagnostics. These are alignment corrections; existing sparse poses, changing contours and breathing motion remain visible. They do not establish that every animation is seamless.

`build-review.cjs` crops only the relevant original atlas rows into lossless PNG data and checks exact decoded RGBA equality. It builds the inline fragment from `review-template.html`. `review-integrity.json` contains those checks. `verify-review.cjs` verifies synchronized frames, weighted timing, play/pause, scrubbing, frame stepping, facing/scale controls and horizontal fit at 736 and 390 pixels. All four sample/viewport combinations passed.

The generated player redraw is **rejected and excluded**. Both attempts failed to establish a convincing alternating gait, and the selected raw candidate also crosses nominal row boundaries. `player-run-generation-prompt.md` and `player-run-candidate-metadata.json` record the built-in image-generation prompts, source, and rejection. The PNG is retained only as an unmodified rejected study; do not import it into the game.

Rebuild the comparison with `node output/starfall-animation-samples/build-review.cjs`. Review corrections with the user before applying them to authoritative source assets or runtime metadata.

## Grounded spring study after animated reference feedback

The actual linked Tenor GIF is preserved as `mushroom-motion-reference.gif`. It contains 41 display frames over 5.39 seconds, with three recurring broad pose families: upright, compressed/tilted, and low/compressed. This is a planted squash-and-rise with cap wobble. The initial clipboard PNG did not preserve animation. `mushroom-reference-motion.json` and the contact sheets document the decoded reference.

`glowcap-spring-study.png` contains eight newly generated pose drawings preserving the green Glowcap identity. It replaces the earlier airborne hop concept for this review. `glowcap-spring-generation-prompt.md` and `glowcap-spring-study-metadata.json` retain generation details. The raw PNG remains unmodified.

`build-glowcap-spring-review.cjs` builds the current idle / grounded spring comparison in the thread's visualization directory. The current side uses the active compact sheet's original three idle drawings with holds `[4, 2, 4]` at 5 fps (2 seconds). The proposed side uses eight drawings, registered by anatomical foot contact, one common scale, and a 1.2-second loop with holds `[3, 1, 2, 1, 1, 1, 1, 2]`. Each side retains its own timing. No crossfade, morph, per-pose rescaling, or generated interpolation occurs. The inline copy uses quality-96 WebP with alpha quality 100 to fit the display size limit; the raw PNG is retained for inspection.

`glowcap-spring-registration.json` records the foot anchors. `glowcap-review-meta.json` records source hash, rectangles, anchors, common scale and timing. `glowcap-spring-check.json` confirms all eight poses are unique and fit measured frame rectangles without substantial-alpha gutter clipping. `verify-glowcap-review.cjs` checks the rendered comparison and controls at 736, 390 and 320 pixels.

This is a direction sample, not an approved production atlas. The drawings now have clear squash/stretch extremes, but small cap-spot, leaf and contour differences still need model-sheet cleanup before a full rollout. A production pass should lock the palette, outlines and recurring landmarks, then verify the last-to-first transition, action poses, roots and timing inside both game renderers. The previous `glowcap-hop-study.png` and `build-glowcap-review.cjs` are superseded experiments and must not be imported. No production game assets or behavior changed.

## Icebloom Oracle healing cast

Following approval of the new pose format, `oracle-cast-study.png` applies it to another character and action: ready, draw inward, gather, unfurl, bloom, follow-through, recover, ready again. The eight distinct drawings use vine-arm articulation, a lowered gathering pose, a broad bloom and facial expression changes. Built-in image generation produced the raw sheet; the prompt and measured export details are in `oracle-cast-generation-prompt.md` and `oracle-cast-study-metadata.json`.

The before side uses the **active compact sheet's buff row 5**, plus the original idle frame as a bookend. Oracle is a healer: `healNearby` locks the buff state for 0.58 seconds. Its three source poses last 1/12 second, 1/12 second, and the remaining 0.413333 seconds. The preview preserves that timing. Both sides repeat on a staged 1.2-second review loop, with 0.35 seconds before the cast and 0.27 seconds afterward; this is not the native 2.8-second healing cooldown. Separate runtime world effects are omitted. Existing effects baked into the before sheet remain visible.

The new side has the same 0.58-second action window and one scale throughout. Pose registration uses foot support, not the crown or effect bounds. The neutral height is matched to the current actor; no pose is independently resized or morphed. This isolates the new drawings and pose timing for feedback. The original PNG is unchanged; the inline preview uses quality-96 WebP with full-quality alpha to fit its size limit.

Build with `node output/starfall-animation-samples/build-oracle-cast-review.cjs`. The thread comparison is `oracle-cast-study.html`; `oracle-review-meta.json` records timing, source hash and registration. `verify-oracle-review.cjs` checks the rendered comparison at 736, 390 and 320 pixels, including all poses, clipping, exact scrubbing, state restoration, speed and frame stepping. Screenshots and results are saved with the `oracle-review-` prefix.

Review status: sample only. The gathering pose makes the crown and face somewhat smaller; the loop endpoints and fine crystal highlights still need final model-sheet cleanup if this direction is selected. Current poses are deliberately redrawn; no claim of production-perfect consistency is made. The prior Glowcap comparison and all production assets remain untouched.

## Oracle revision: retain the healing effect

The next review revision includes the move's healing cue as part of the animation. `oracle-healing-fx-study.png` is a separate eight-frame effect: forming halo, expanding bloom, prominent healing plus symbols, an outward pulse and dissipating particles. Its raw image and generation prompt are saved alongside the previous Oracle poses. The approved `oracle-cast-study.png` remains byte-identical, and its registration, common scale and pose durations remain identical.

`build-oracle-heal-review.cjs` creates `oracle-healing-cast.html` in the thread visualization directory. It draws the healing effect behind the character to keep the face and pose readable. Its clock starts at the cast event (0.35 seconds into the staged review loop), runs for the game's 0.7-second healing-effect lifetime, and fades before the loop restarts. The eight body poses retain the 0.58-second casting window. The effect has its own frame clock, so it continues during a held body pose, and it follows pause, speed, stepping and scrubbing.

The current side still shows the original actor sheet with its baked healing aura. In actual Canvas gameplay, `healNearby` also triggers a procedural green ring/column/white cross via `drawPartyBuffEffect`; the generic and enemy FX atlases are not used by this heal event. This comparison remains a staged sprite review rather than a full game-renderer capture. The new icy plus-marked aura is a proposed replacement visual for review, not a statement that the native effect is already replaced.

`oracle-heal-review-meta.json` records both raw image hashes, effect geometry, timing, encoding and four checks proving preservation of the approved actor. `verify-oracle-heal-review.cjs` checks the composite at desktop and narrow widths, including every effect frame, held-pose effect motion, fade completion, precise scrubbing and restored state. Future samples must include the move-specific visual cue and its timing together with character poses. No production assets or gameplay were changed.
