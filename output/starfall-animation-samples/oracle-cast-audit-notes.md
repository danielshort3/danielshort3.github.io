# Independent Oracle healing-cast review

Source SHA256: `062807836c25b7ccdfcb9e65081285589a85a15dcc64473578fb4b88468a51e8`.

## Decision

Keep the existing preview root metadata. No registration override is warranted, so `oracle-cast-registration.json` was not created.

## Evidence

- Eight unique, nonempty measured rectangles preserve all source pixels with no overlap or omission. No alpha >= 16 pose touches a rectangle edge.
- Frames 0-2 enter the four-pixel bottom gutter, retaining approximately three transparent rows below their lowest visible pixels. This is sufficient for this independent-frame review; add consistent export padding in a final atlas rather than claiming the current packing is production-ready.
- The largest opaque component bounds match the bounds of all alpha >= 128 pixels in every frame. The crown is connected to the measured actor; the neutral scale was not accidentally calculated from a crownless body.
- Independently measured horizontal foot-support centers in the lowest ten opaque rows differ from the current roots by `[-0.123, -0.123, 0.123, -0.247, -0.123, 0, -0.247, -0.493]` displayed game pixels. The maximum difference is below half a pixel. The current wider foot band is reasonable and excludes the crown, arms and detached effects.
- Every exported root Y equals the pixel boundary immediately below the opaque sole bottom. Keep those Y values.
- One uniform scale, `0.24671345811051693`, applies to all eight drawings. Existing translations register source packing and the supporting leaf feet; they do not create the drawn arm curl, anticipation, blink, body release or bloom poses.

## Visual limits

- The leaf feet have lobed outlines; counting their tips as extra legs would be misleading. The same primary crown and paired curled appendages remain recognizable throughout the sheet.
- The last pose is a near-ready drawing, not a duplicate of frame 0. At the common root, its opaque silhouette is about 3.7 displayed pixels shorter than frame 0. That remaining endpoint change belongs to the authored pose transition, not a registration failure. Review it in playback; do not conceal it through individual scaling or shape warping.
- This source check supports the feedback preview. It does not certify a final seamless loop or final gameplay timing.

Detailed reports: `oracle-cast-check.json` and `oracle-foot-contact-check.json`.
