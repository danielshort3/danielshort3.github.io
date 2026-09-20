# Glowcap hop study - preview only

Generated with the built-in image_gen tool. The selected raw PNG is copied unchanged from the second, layout-focused attempt. No game assets or runtime code were replaced.

Inputs: current green Glowcap Healer portrait as character identity; the supplied static orange mushroom PNG as a clean-shape reference. The actual animated GIF was supplied after this generation had finished, so this study has not yet been matched to that motion.

Observed: sixteen distinguishable drawings include preparation crouch, stretched push-off, folded airborne feet, extended landing feet, impact squash and settling. The source is actually 1254 x 1254 with transparent alpha, rather than the requested dimensions. Exact equal-cell grid and requested vertical anchor fractions were not obeyed; metadata includes measured padded source rectangles that avoid cutting through first-row feet. Tiny stray pixels remain outside some silhouettes. Playback and actual-reference review are required before any smoothness or seamlessness claim.

## Initial generation prompt

+Use case: identity-preserve.
Asset type: ONE sixteen-frame enemy-animation sprite-sheet STUDY, a clear complete mushroom hop loop.
Input image1 is the exact character identity reference: the Project Starfall Glowcap Healer. Input image2 (orange mushroom) is ONLY a shape-clarity and clean-outline reference. Keep our character GREEN; do not copy the orange character, its hat, expression, checkerboard or background.
Redraw our squat friendly mushroom as a clean crisp2D sprite illustration: short warm-cream stem-body, large green cap with exactly four easily tracked large luminous pale-yellow round spots in consistent places on its surface, exactly two small glowing cap nubs, two expressive black oval eyes with the same simple happy mouth, two tiny hands, and small paired leaf-shaped feet. Constant palette, outline thickness, cap-spot design and lighting. Simplify the busy texture of reference1 into solid readable shaded shapes. Friendly expressive squash-and-stretch animation, not photorealism.
Canvas/export: exact1024x1024 square PNG with actual transparent alpha. EXACTLY FOUR columns by FOUR rows, sixteen equal256x256 cells on an invisible mathematically regular grid. One sprite inside EACH cell; no missing/extra sprites. Row-major reading order0..15 is ONE chronological continuous hop, not separate animation rows. The sprite's root is always centered horizontally at x128 of its cell. Neutral full character height about133px (52%cell); maximum cap width166px (65%cell). Planted feet touch y217 (85%cell). At apex feet rise to y153 (60%cell) and top remains at or below y20. Keep every pose fully inside its own256square with ample transparent margin, no overflow. Cell origins are x=0,256,512,768 and y=0,256,512,768. SAME body/head volume and outline across frames; shape changes must depict real squash, bent limbs, cap lag and recovery, not accidental scale drift.
Sixteen DISTINCT AUTHORED poses with obvious internal articulation, in row-major order:
0: neutral cheerful squat mushroom standing, feet together-ish, hands relaxed.
1: anticipation: knees/leaf feet bend, body starts crouching, cap tilts down, hands begin to draw back.
2: deepest preparation crouch: body visibly compresses, cap settles and widens elastically, feet spread and fold, hands pulled backward.
3: forceful push-off: body elongates, feet extend on toes, arms swing upward, cap still lags low relative to lifted face.
4: early ascent: both feet leave ground and BEGIN TUCKING under belly, body extended, hands lift, cap trailing slightly.
5: ascending: both feet clearly fold inward, body shortens from stretched push-off, cap begins catching up, hands bent.
6: near apex: feet tucked tight, cap and nubs lift with delayed spring, hands spread for balance.
7: apex: compact airborne pose, face tucked gently into cap, feet folded, hands balanced, cap relaxed high.
8: first descent: face/body start dropping below cap, one leaf foot begins unfolding, hands turn downward.
9: descending: both feet extend toward ground, torso lengthens, cap lifts relative to body due drag, hands lower.
10: prelanding: feet spread forward ready to catch body, arms out, body downward braced, cap still high.
11: first contact: toes/feet touch shared ground baseline, body still fairly tall, arms spread wider, cap catching up downward.
12: deepest impact squash: body visibly flattened and broad, leaf feet splay, cap presses down around face, hands out and slightly up. Distinct from preparation pose2.
13: recoil rebound: body rises and uncompresses, feet fold back to normalstance, cap springs upward after impact, arms drop halfway.
14: settle: body almost neutral, cap slightly above resting position, hands relax.
15: approaching neutral: small settling change in feet/hands/cap; leads smoothly into0. Do NOT duplicate frame0.
Each frame must be a real newly drawn in-between with consistent anatomy, changing relationships among cap/body/hands/feet. Don't reproduce one pose shifted, scaled, recolored or slightly warped. Do not use a static mushroom floating up and down. Crisp readable pose differences are the central requirement.
No scenery, floor, groundline, castshadow, dust, particles, aura, motionblur, motionlines, caption, number, letter, gridline, border, UI, watermark or baked checkerboard. Only16 transparent character cutouts in exactgrid. Full16frames mandatory; make the final-to-first change small and natural. Preview study only.

## Targeted layout revision prompt

+Edit target: this16-pose green Glowcap Healer sheet.
ONE TARGETED CORRECTION: fix export layout and per-frame root registration ONLY while preserving the16 distinctly drawn chronological hop poses and their internal anatomy, shape, colors, consistentface, greencapspots and two glowingnubs.
Current image packs characters far too tightly. Redraw this as a spacious FOUR-column FOUR-row mathematically regular sprite sheet on actual transparent alpha, preferably1536x1536 or1024x1024. Sixteen cells remain ordered row-major0..15. No cropping and no neighboring cell contamination.
Make EVERY mushroom only55% as large as currently shown, applying exactly the same global scale factor to all16 poses. In the finished sheet, a neutral mushroom including nubs should occupy only about50% of ONE CELL HEIGHT; cap width at most60%ofCELLWIDTH. There must be enormous clear transparent margin compared to the presentimage. Do not fill blankspace with enlargedart or extraelements.
Eachcell has identical fixed centerline at50%cellwidth. Assign the bodycenter/root to thatline, not the changing silhouetteboundingbox. Cell-origin grid spacing must be absolutely uniform.
Now place the bottom of the FEET in eachcell at these exact fractions ofcellheight to depict hopposition without alteringpose:
frame0 .85
frame1 .85
frame2 .85
frame3 .82
frame4 .74
frame5 .67
frame6 .62
frame7 .60
frame8 .62
frame9 .68
frame10 .77
frame11 .85
frame12 .85
frame13 .81
frame14 .85
frame15 .85.
With mushrooms small enough, even theapex remains insidecell with at least8%topmargin. Fixedhorizontalroot, no sidewayspositiondrift. Preserve crouch bodycompression, push-off bodyextension, the actual bent/tuckedfeet inascent, extendedfeet beforelanding, and deepimpact squash, not rigid translation ofneutral. Keep consistenthead/face/capdesign. Preserve16 distinctdrawings, no duplicatedposes orendpoint.
Absolutely no text, labels, frame numbers, visiblegridlines, shadow, groundplane, particles, motionlines, FX, bakedcheckerboard or coloredbackground. Actual transparentbackground. This is one continuous16frame hop that settles smoothly into firstframe.
