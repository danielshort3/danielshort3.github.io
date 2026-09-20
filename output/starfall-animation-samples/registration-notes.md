# Registration sample notes

These offsets are deliberately limited to idle registration. They change no artwork, timeline, pose order, scale, alpha, or gameplay data. Keep both comparison panels on the exact same playback clock. Apply offsets in actor-local source coordinates before facing reflection.

## icebloom-oracle-idle

Median horizontal midpoint of the grounded lower-petal support, rows y=108..115, alpha >= 100; reference frame 0. This region excludes raised arms, curls, face and loose particles. The lowest lone petal pixels at y117 are not a reliable horizontal root.

Offsets (x, y): [(0, 0), (-2, 0), (-10, 0)]. Ground offset stays zero to preserve breathing/blinking and the existing baseline.
Support-anchor x spread: 10.0 px before; 0.25 px after. Alpha-centroid x spread: 9.1102 px before; 0.9283 px after. Remaining centroid movement is expected as the pose and tail change.
Native frame size 128 x 128; 5 fps; holds [4, 2, 4]; cycle 2.000 s. All original visible alpha survives the proposed offsets within the existing cell.

## starfall-fox-idle

Midpoint between rearmost and foremost visible paw contact pixels on y142, alpha >= 100; reference frame 0. At this row the tail is absent in all six frames. This measures the four-paw support span, not the whole sprite bounds or head/tail center.

Offsets (x, y): [(0, 0), (-2, 0), (2, 0), (1, 0), (5, 0), (12, 0)]. Ground offset stays zero to preserve breathing/blinking and the existing baseline.
Support-anchor x spread: 14.5 px before; 0.5 px after. Alpha-centroid x spread: 17.0543 px before; 3.6222 px after. Remaining centroid movement is expected as the pose and tail change.
Native frame size 160 x 160; 6 fps; holds [1, 1, 1, 1, 1, 1]; cycle 1.000 s. All original visible alpha survives the proposed offsets within the existing cell.

The Oracle root comes from its supporting petal base. The Fox root comes from its paw-support span, with the tail excluded. Neither is aligned by full alpha bounding boxes. These are candidate offsets for user review; contour redrawing, additional poses and lifecycle changes are outside this sample.
