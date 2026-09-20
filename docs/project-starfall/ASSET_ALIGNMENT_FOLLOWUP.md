# Starfall horizontal alignment follow-up

The Lava Tick report was correct: its body shifted horizontally in the idle preview. The earlier migration completed asset replacement and checked dimensions, alpha, scale and coverage, but those checks did not establish consistent anatomical registration. Review the correction in the [before/after animation viewer](../../output/starfall-alignment-review/index.html).

## Cause and correction

The importer detected complete actors from transparent source gutters, then used equal column centers as horizontal roots. Generated bodies were not evenly centered in those columns. That mismatch moved otherwise similar bodies between runtime cells. The correction records explicit source-space anatomical X anchors and preserves the existing shared identity scale, vertical placement and authored pixels. It does not redraw the art or manufacture motion through per-frame resizing.

| Identity | Corrected scope |
| --- | --- |
| Lava Tick | All 48 horizontal anchors across idle, move, telegraph, attack, projectile, buff, hit and defeat |
| Eclipse Sovereign | Six idle horizontal anchors |
| Cinder Spitter | Six idle horizontal anchors |
| Cracked Mimic | Six idle horizontal anchors |
| Bandit Cutter | Six idle horizontal anchors; its four compatibility copies inherit the same sheet |
| Clockbug | Six idle horizontal anchors |
| Briar Stag | Six idle horizontal anchors |
| Clockwork Titan | Six idle horizontal anchors |
| Rimewarden | Six idle horizontal anchors |
| Dew Slime | Six idle horizontal anchors |
| Brambleking | Six idle horizontal anchors |
| Index Scribe | Six idle horizontal anchors |

These are 12 canonical identities, not 12 fully re-reviewed animation sets. The other 42 frame slots remain pixel-identical for each of the eleven idle-only corrections. Before sheets/configs/reports are retained under [enemy-idle-alignment](../../output/enemy-idle-alignment/); Lava Tick has its own [evidence folder](../../output/lava-tick-alignment/). Raw sources and approved session references remain unchanged.

## Evidence and limits

Lava Tick's independently measured rigid-shell horizontal range fell from **12 px to 1 px**, and its last-to-first horizontal jump fell from **9 px to 0 px**, measured at the 160px runtime-cell scale. This RGB/alpha patch check is separate from the shell-contour landmark used to choose its anchors. [The evidence](../../output/lava-tick-alignment/evidence.json) also verifies all 48 output poses differ only by their intended horizontal translations, with the original scale and vertical placement intact.

The [44-identity idle screen](../../output/starfall-overhaul-validation/enemy-idle-drift-review.md) found the other eleven visually confirmed relocation cases. Their [six-identity evidence](../../output/enemy-idle-alignment/confirmed-idle-corrections.json) and [five-identity evidence](../../output/enemy-idle-alignment/five-verification-summary.json) record preservation checks and before/after comparisons. Correlation scores are review signals: a turning body, flexible plant, deforming slime, weapon or wing can bias a translation fit. They are not a reason to erase intended motion.

Remaining limits are explicit. Lava Tick's original 1–2 px vertical idle differences, fissure/highlight changes and shell/head shape variations remain. Its posture-changing actions were reviewed for horizontal carapace placement, not certified as anatomically identical. The other eleven corrections cover horizontal idle registration only; their other actions and action transitions need separate review. The remaining cast is not certified by this pass, including candidates whose apparent translation may be intended articulation. A stable root does not guarantee consistent faces, proportions or seamless drawing transitions.

The [asset guide](ASSET_GENERATION_GUIDE.md#anatomical-registration-and-scoped-motion-review) and [reusable briefs](../../asset-sources/project-starfall/prompts/README.md) now require anatomical X anchors, independent packed-pixel/landmark checks for each accepted action and seam, and a precise review scope. Production output hashes, transparent-pixel fractions and cell-edge counts are refreshed in the [enemy validation report](../../asset-sources/project-starfall/overhaul-v1/enemies/validation-report.json). These measurements establish file/packing integrity; they do not replace animation review.

After a reviewed import, run `node build/refresh-project-starfall-enemy-validation.js` to refresh that report. The command rejects production/candidate mismatches and changed source hashes before recording imported evidence; it does not import or alter artwork.
