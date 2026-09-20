# Visual review of enemy idle registration

This is a read-only follow-up to the Lava Tick report. It screens the 44 canonical production identities, excluding four bandit aliases. It does not modify production assets, import configs, or protected reference images.

The screening script fits core RGB/alpha under translation, then compares that estimate to a whole-silhouette fit. Both are measured at the 160px runtime-sheet scale. The fit is a review signal: squash, articulation, expression changes, and turns can bias it. Bounding-box stability alone cannot establish alignment, and these metrics do not establish alignment for the other seven action rows.

## Confirmed visual-review priorities

The ranked contact sheets show fixed cell-center guides. Inspecting all four reveals repeated progressive horizontal relocation of essentially the same standing or hovering body, including these especially clear cases:

| Identity | Core offsets from idle frame 1 across frames 1–6 | Supporting observation |
|---|---|---|
| Eclipse Sovereign | 0, -11, -14, -19, -20, -24 | Crown, face, and torso all migrate left across the row. Full silhouette agrees within 1px. |
| Cinder Spitter | 0, -1, -6, -10, -10, -13 | Head/body root moves left while feet stay at nearly the same height. Core and full fits agree exactly. |
| Cracked Mimic | 0, -2, -4, -10, -11, -12 | Chest lock and rigid chest body drift left together. Final pose translation reduces core error by 95%. |
| Bandit Cutter | 0, -2, -4, -5, -10, -10 | Hood, torso and legs shift left. The four canonical aliases inherit its sheet. |
| Clockbug | 0, +4, +2, -2, -6, -8 | Clock body is progressively misplaced relative to the anchor after the second pose. Core/full fits agree exactly. |
| Briar Stag | 0, +4, +2, +1, -7, -8 | Torso and legs shift together; the antlers are not the sole source of the bounds change. |
| Clockwork Titan | 0, -3, -6, -8, -9, -10 | Chest and feet progressively shift left. |
| Rimewarden | 0, -2, -3, -5, -5, -8 | Rigid torso/crown and feet progressively shift left. |
| Dew Slime | 0, -2, -3, -5, -7, -8 | Legitimate squash remains, but the body base also translates left. |
| Brambleking | 0, -2, -4, -8, -8, -9 | Torso/face shifts left with little corresponding body turn. |
| Index Scribe | 0, -1, -1, -2, -4, -8 | Hood/face/book group shifts left near loop closure. |
| Lava Tick | 0, -8, -7, -12, -8, -9 | Head/shell relocation agrees with the reported preview defect. Root task is correcting it. |

Additional high-scoring candidates requiring anatomy-aware review are Rust Ratchet (16px span; crouch contributes), Snowglare Wisp (15px; hover motion contributes), Void Mote (14px; changing tendrils and facial proportions produce method disagreement), Ember Wisp (12px; flame deformation contributes), and Stormbreak Roc (10px; wing/neck articulation contributes). Slimelet, Rift Aberration, Glacier Sentinel, Thorn Sprout, Dust Imp, Emberjaw Golem, and Astral Archivist also exceed the 7px core-span screening threshold.

There are 24 canonical idle rows with core spans at least 7px, including Lava Tick. This is not equivalent to 24 confirmed defects, and lower scores do not certify the remaining rows. The screen does establish that Lava Tick is not an isolated reason to revisit the blanket alignment claim.

## Recommended acceptance change

Use measured anatomical anchors or reviewed stable landmarks per authored pose, instead of nominal equal source-column centers. Do not automatically center silhouettes by their bounding boxes: a tail, weapon extension, or lean can change bounds without moving the body root. Preserve authored squash and action travel. Verify corrected candidates in animated loops and frame stepping with a fixed anchor guide, including the last-to-first transition. Then verify transitions between semantic action rows; the idle-only screen does not cover them.

Artifacts: `enemy-idle-drift-screen.json`, `enemy-idle-drift-screen.md`, `enemy-idle-drift-contact-1.png` through `enemy-idle-drift-contact-4.png`, and the rerunnable `screen-enemy-idle-drift.cjs`.
