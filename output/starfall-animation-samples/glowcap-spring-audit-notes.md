# Independent spring-study check

Candidate: `glowcap-spring-study.png`, SHA256 `6a5466fefd53d0b308b9ef071674fefe72a1ee9584849c76cb72c1024d3826e2`.

- Eight source rectangles preserve the full source with no overlap or omitted pixels. All eight contain nonempty, unique drawings.
- At alpha >= 16, no pose touches its source rectangle boundary or its four-pixel gutter. The source dimensions are 1774 x 887, so the measured 443/444-pixel rectangles are appropriate; the nominal grid has fractional cell dimensions.
- Visible silhouette widths are 352-390 source pixels; heights are 257-395. These differences are authored squat/stretch poses. The preview does not independently resize them.
- Visual inspection shows planted brown foot/leaf pads in every frame. The sampled bottom band contains those supports, not cap pixels. The action is now a grounded spring/wobble sequence, with drawn body compression/extension and cap follow-through.
- The preview uses one scale, `0.22262612910332671`, for both axes in every frame. Its `drawImage` call applies per-frame root translation and that shared scale only; there are no generated in-between frames, shape warps, or per-frame scale adjustments. Preview WebP encoding is separate; the original PNG remains intact.
- An independent alpha >= 128 contact check compared the first preview's root X with the midpoint of the lowest ten pixels of foot-support silhouette. Seven frames agreed within 0.12 displayed game pixels. Frame 2 (deep squash) differed by 1.11 displayed pixels: initial X=216 versus lower-contact-span X=221. The broader foot band included the flared left leaf in this pose. The final `glowcap-spring-registration.json` adopts the measured ten-pixel contact-band centers for all eight frames, including X=221 for frame 2. The contact-check JSON retains the initial measurements; the final anchors and repeated browser checks are in `glowcap-review-meta.json` and `glowcap-review-verification.json`.
- The exported root Y is the pixel boundary immediately below the lowest main opaque foot pixel. The top and bottom rows have different packed baselines; registration corrects that packing difference while retaining the drawn deformation above the feet.
- A one-pixel-deep contact test is unstable because some soles end a pixel earlier than others (especially frame 1); the thicker support band is preferable. For a final atlas, semantic left/right sole-contact points supplied during authoring remain preferable to this measured approximation.

Evidence is in `glowcap-spring-check.json` and `glowcap-spring-foot-contact-check.json`. Structural containment and registration checks do not certify seamless animation. The final decision still requires playback and user feedback on the actual poses and timing.
