# Tableau dashboard redesign

Both dashboards are published in Tableau Public with the requested feedback revision complete. Native desktop/Phone interactions, final workbook structure, official exports, refreshed previews, and local website checks passed. **The website changes are local; no website deployment is claimed.** This final record supersedes the earlier checkpoint and concept-fidelity acceptance notes.

## Published behavior

| Dashboard | Navigation on every desktop/Phone page | Persistent controls |
|---|---|---|
| Pizza | Overview, Timing, Trends, Weekday, Tip estimate, About | Date / City / Housing retain selection and identical positions across pages. Order cost alone drives Tip estimate. |
| UFO | Overview, Places, Timing, Shapes, About | Year / State / Reported shape retain selection and identical positions across pages. |

Navigation, filters and charts remain native Tableau objects. White rounded backing panels, navy headings, blue marks, concise labels and visible summary values follow the AI concepts. Source, definitions and caveats are consolidated in About. Visible tooltips use friendly labels; redundant KPI/model tooltips are disabled. Controls keep their positions between pages; they are not sticky while scrolling. Tableau's accessibility chart description can still name the underlying Pizza count calculation, separately from the cleaned visible tooltip.

Pizza retains all **1,251 delivery records**, **$6.00 median tip** and **$8,933.49 total tips**. Overview includes aligned city average/count/recorded-minute columns, small-sample styling, the complete $2-bin tip distribution with dynamic median, and all 15 monthly averages/counts. Timing retains elapsed order-to-delivery minutes and all 34 zero-minute records; the interpretation is in About. Trends supplies the complete monthly table. Weekday compares average, median, total tips and record count in Monday–Sunday order. An explicit City-only Select/clear-all action carries Overview city focus to the four Weekday charts, while model cards remain excluded.

UFO uses the 48-contiguous-state whitelist and defaults to **6,334 reports / July 810 (12.8%) / 71.9% evening**. It includes the map, Top 5 states/shapes, recorded month/hour heatmap, dynamic notes, full state/shape count-share rankings and monthly/hourly details. Unknown, Other and Not recorded remain distinct. Its five All-fields Select/auto-clear actions use controlled source charts; ordinary clicks replace mark focus, while dropdowns combine with it. No unused legacy worksheet is exposed. Reporting/coverage/denominator limitations are in About.

## Final verification

[Native Pizza QA][pizza-ui] passed the complete six-page desktop and true-mobile 390px Phone cycles with **January 1–September 23, 2018 / Frisco / Hotel** retained: **21 records / $5.00 median / $136.25 total**. Friday, Saturday and Sunday each have seven records. Timing shows **40.3 mean / 41 median / 24–55 minutes**, with the redundant caveat removed. Every About section, including the final paragraph, is readable by scrolling. Revert restores full dates / All / All and is then disabled.

All default weekday values matched source data. Native Order cost entries at **$6.17, $80 and $243.02** gave the expected two-decimal outputs. A Lewisville + Hotel selection with no historical rows leaves the $40 estimator at **$6.94 / $3.43 to $10.45**. [Independent acceptance][pizza-independent] passed **265 Phone structural assertions**, all 24,937 interval-format cases, model isolation and original-data equality. A fresh public download has byte-identical uncompressed TWB/Hyper members; its ZIP metadata explains a different outer package hash.

[Final package checks][structure] passed **126 Pizza + 74 UFO assertions**. Pizza has six 1200 x 850 desktop pages and Phone scroll heights **2100 / 1500 / 1200 / 2100 / 1240 / 1550** in navigation order. UFO has five 1200 x 840 pages and Phone heights **2350 / 1550 / 1780 / 1450 / 1550**. All 22 shown Pizza and 15 shown UFO worksheets were covered by the tooltip/structure review; Pizza's two model cards remain outside history filters. The tiny retained Trends backing worksheet keeps shared filter cards available; it is not a visible analysis panel.

[UFO public/local website QA][ufo-web] confirmed California through Timing and About at **653 / December 92 (14.1%) / 68.3% evening**, then Reset to 2013 / All / All. Public Phone review verified all five destinations, readable controls/charts/About, combined filtering, clear and Revert. [Pizza website QA][pizza-web] verified six-tab navigation, native $80 entry and website Reset back to $40. Each wide iframe is centered at 1200px with equal 123.667px gutters at the professional 1920px route; Pizza/UFO heights are 880/870px, including the complete 27px toolbar.

The original preview-only phone integration was superseded by the September 13 local mobile-embed update. Narrow project panels now load the interactive native Phone view directly; the full-aspect preview remains available when site JavaScript is disabled. Both Open dashboard links still allow Tableau to select its standalone device layout.

The [both-preview build][build] passed on an unchanged retry after a Windows `UNKNOWN` write error on an unrelated generated page. The [full test chain][tests] passed; there were no later runtime changes. [Final source/public parity][parity] passed **158 assertions with zero issues at 2026-09-13 00:18:40 UTC**, covering 16 image pairs, eight generated page pairs, six helpers, two manifest contracts and 17 bundles. The public manifest intentionally excludes the retired contributions entry/bundle; the verifier checks that deployment transform and exact parity for the remaining entries.

## Published artifacts and provenance

| Artifact | Pizza | UFO |
|---|---|---|
| Canonical manifest | [Pizza](published/pizza-delivery-published-manifest.json) | [UFO](published/ufo-sightings-published-manifest.json) |
| Capture UTC | 2026-09-13 00:13:33 | 2026-09-12 22:06:58 |
| TWBX bytes | 147,859 | 5,182,165 |
| TWBX SHA-256 | `f679d226b2096fb3b09c7c0c9c1e8018cade063c0a96f68a316ddb0304e19919` | `650249a2a3e6a82ada2d8b4c3780ccca61011951d7887f67a84b09a00c9b0ae8` |
| Hyper SHA-256, identical to original | `614f454d2f64b3de28ee7fae5d18eed3e712c9c474cf21a64e9323b72fbc5f52` | `c0ddd072e24b69b66e02d5ec594708bbb1ac60bfcb117a8d0d776da593b6b16a` |
| Official SVG bytes / text | 342,118 / 120 nonempty Arial texts, zero ellipses | 580,106 / 93 nonempty Arial texts, zero ellipses |
| Official SVG SHA-256 | `4972a2572397aaefe1855d119a6ef17e0c932f1e3bde870c547432f78bf8de7c` | `8239b0afd4751855ca2b1a438e07e5c9def536573d592a58a6df18862cf1e2a9` |
| Full raster / preview | 1600 x 1133 / 1280 x 906 | 1600 x 1120 / 1280 x 896 |

Both complete official rasters passed visual review. The final Pizza SVG's faithful density-96 raster has **zero differing channels out of 7,251,200** versus the already-built preview source; [pixel comparison][pixel] confirms all eight Pizza variants remain current without another build. [Pizza SVG inspection][pizza-svg] and [UFO SVG inspection][ufo-svg] verify default text and metrics, supplementing visual review rather than proving clipping on their own.

`sources/` preserves the original workbooks. `published/` holds actual public snapshots and official exports; its manifests are the authority for capture times and hashes. [pre-fidelity-revision-20260912-074558](published/pre-fidelity-revision-20260912-074558/) preserves an earlier seven-file publication with verified hashes; do not overwrite it. Earlier research/QA remains historical evidence, not the current acceptance result.

Top-level `*-revamped.twbx` / `*-revamped.twb` files generated by [rebuild-pizza.py](../../build/tableau/rebuild-pizza.py) and [rebuild-ufo.py](../../build/tableau/rebuild-ufo.py) are unvalidated alternatives. They were not uploaded, are not live copies, and are **not approved for publication**. Open TWBX files to include their extracts; standalone TWB files are not self-contained.

## Reproducible analysis

The [analysis package](analysis/pizza-weekday-estimate/README.md) preserves the original 1,251-row CSV, portable analysis script, weekday metrics, frozen model/validation outputs and [native implementation notes](analysis/pizza-weekday-estimate/native-implementation.md). Seven compact outputs reproduced exactly. Scored-row CSVs and the notebook are not duplicated. The original handoff's field captions/cent step differ from actual native captions/continuous input as documented; frozen math/spec outputs are unchanged.

The cost-only ridge model uses 751 fit/model-selection rows, 249 calibration rows and 251 later holdout rows, with no date split between roles. Holdout MAE is $2.62 versus $3.06 for a $6 baseline; interval coverage is 203/251. Published-formula parity differs by at most **3.91e-14** over all source rows. About explains historical support, sparse high-cost examples, individual variation and why the approximate 80% interval is not a future guarantee. See [concept acceptance and numerical authority](concept-acceptance.md).

## Website integration

Authoritative content is [Pizza JSON](../../content/projects/pizzaDashboard.json) / [UFO JSON](../../content/projects/ufoDashboard.json), rendered by [generate-project-pages.js](../../build/generate-project-pages.js). Edit source, then rebuild; do not patch generated HTML or `public/` directly.

Native iframes request `:device=desktop` or `:device=phone` according to the project panel width, with `:tabs=no` in both modes. Open dashboard/modal URLs hide legacy sheet tabs and allow Phone selection. [tableau-controls.js](../../js/portfolio/tableau-controls.js) resets the saved Overview URL with `:revert=all`, removes `:iid`, and retains the active device layout. Standalone Tableau provides its native Revert control.

[project-page.css](../../css/components/project-page.css) expands only dashboard detail routes to an 1800px stage. Containers at least 1226px wide show centered 1200px desktop iframes. Narrower panels embed the fit-width Phone layout in a centered frame up to 720px wide, with a scrollable height of `clamp(600px, 85svh, 900px)`. Reset and Open dashboard remain available. At widths up to 480px, the dashboard uses the full panel width so native navigation stays readable. The shared loader reads the CSS container mode. Changing device mode or the actual Phone canvas width starts a fresh native view; width changes are debounced for 200ms because the published floating Phone canvas does not reflow inside a raw iframe. Height-only changes and unchanged canvas widths preserve the current dashboard and filters. The preview/action remains the no-JavaScript fallback. Other pages retain their existing stage widths and navigation rails.

## Refresh and verify

Use an official Tableau PNG or faithful official-SVG raster. Direct PNG exports in this session had colored edge artifacts, so current previews use Sharp at `density: 96`. Archive the untouched SVG and inspect the full raster. Do not crop charts, change SVG fonts/geometry or equate raster dimensions with the logical Tableau canvas.

```powershell
node -e 'require("sharp")("design/tableau/published/ufo-sightings-published.svg", { density: 96 }).png().toFile("design/tableau/published/ufo-sightings-published.png").catch(error => { console.error(error); process.exitCode = 1; });'
node build/tableau/update-dashboard-previews.js ufoDashboard "design/tableau/published/ufo-sightings-published.png"
python build/tableau/snapshot-public-workbook.py ufoDashboard
python build/tableau/inspect-published-workbook.py ufoDashboard
```

Substitute `pizzaDashboard` and matching filenames for Pizza. The preview helper writes eight full-aspect variants and rereads metadata to preserve unrelated edits. The snapshot helper accepts only the two fixed public URLs and verifies original Hyper hashes before replacement; `--check-only` performs no writes. The inspector is local/read-only; raw XML geometry does not prove rendering.

After future changes, run the build, relevant tests, `git diff --check` and [source/public verifier][parity-tool]. The full suite can rebuild local JS, so check parity afterward. Archive/SVG-only changes need no website rebuild when the retained preview raster is pixel-identical. [Final acceptance](concept-acceptance.md#final-feedback-acceptance) is complete for this revision. Local validation and Tableau publication do not establish website deployment.

[pizza-ui]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-native-interaction-qa.json
[pizza-independent]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-final-independent-acceptance.json
[structure]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-final-structural-verification.json
[pizza-web]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-website-qa.json
[ufo-web]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/ufo-feedback-website-qa.json
[build]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-both-preview-build-retry.log
[tests]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-full-test.log
[parity]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-final-parity.json
[pixel]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-final-export-pixel-parity.json
[pizza-svg]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-svg-inspection.json
[ufo-svg]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/ufo-feedback-svg-inspection.json
[parity-tool]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/verification/verify-dashboard-build.cjs
