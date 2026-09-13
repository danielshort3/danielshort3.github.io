# Tableau concept fidelity acceptance

**Final feedback revision accepted, September 12 local / September 13, 2026 UTC.** Both Tableau Public dashboards passed native desktop/Phone interaction and readability checks, final archive inspection, official-export review and local website validation. F1–F10 below supersede all earlier checkpoint/baseline completion language. **No website deployment is claimed.**

## Final feedback acceptance

| ID | Accepted requirement | Final evidence |
|---|---|---|
| F1 | [x] Full navigation on every desktop and Phone page | Pizza: Overview / Timing / Trends / Weekday / Tip estimate / About. UFO: Overview / Places / Timing / Shapes / About. Every other destination is directly reachable; active-page labels and complete native cycles passed. |
| F2 | [x] Consistent filter positions and selections across tabs | Date/City/Housing and Year/State/Reported shape retain their positions and scope on every destination. Pizza's complete Phone cycle retained 2018 + Frisco + Hotel. Filters are stable between pages, not sticky while scrolling. |
| F3 | [x] Source and caveat prose consolidated into About | Source/scope, metric/reading guidance and model methods remain readable in About. Timing's zero-entry caveat was removed from analysis. Final Pizza About Phone review includes the complete last paragraph. |
| F4 | [x] Friendly or intentionally disabled visible tooltips | All 22 shown Pizza and 15 shown UFO worksheets were covered structurally, with native hover examples including City/Deliveries and Weekday/Average tip. Redundant KPI/model tooltips are disabled. No formula/CSV/style field appears in these visible tooltips. |
| F5 | [x] Useful native Weekday page | Four AVG/MEDIAN/SUM/COUNT charts use manual Monday–Sunday ordering. All seven default rows match the source. Historical filters and the explicit Overview City-only action update all four charts consistently. |
| F6 | [x] Approved cost-only estimate and native parameter | Continuous Order cost FLOAT $6.17–$243.02, default $40, currency cents; native minimum/$80/maximum examples passed. Default $6.94 and $3.43 to $10.45 Approx. 80% range. 24,937 interval-format cases passed. |
| F7 | [x] Estimator independent of history filters/actions | Two MIN cards use only the parameter/model fields; the unused upper-range helper is not shown. Lewisville + Hotel gives no historical records but retains the default estimate/range. Revert restores cost $40. |
| F8 | [x] Complete readable desktop/Phone flows | All six Pizza and five UFO destinations, combined filtering, clear, details, tooltips, About and Revert passed. True-mobile 390px Phone reviews have no horizontal overflow. Both website Open URLs select Phone automatically. |
| F9 | [x] Final published packages preserve original data | Final local checks: Pizza 126 / UFO 74, zero failures. Independent Pizza acceptance: 265 Phone assertions, zero failures; a fresh public download has identical uncompressed TWB/Hyper members. Original extract hashes match. |
| F10 | [x] Official exports and local website artifacts match publication | Complete native rasters reviewed, all 16 image variants current, both-preview build and full tests passed. Final source/public parity: 158 assertions, zero issues. Final Pizza export is pixel-identical to the built preview source. |

Evidence: [final structural checks][structure], [Pizza native interaction][pizza-ui], [independent Pizza acceptance][pizza-independent], [Pizza website QA][pizza-web], [UFO website QA][ufo-web], and [final source/public parity][parity]. XML/text checks supplement actual native rendering and interaction review. Tableau's accessibility chart description still names the underlying Pizza count calculation; this is distinct from the cleaned visible tooltip.

## Final native interaction evidence

Pizza's full desktop and Phone sequence is Overview → Timing → Trends → Weekday → Tip estimate → About → Overview. **January 1–September 23, 2018 / Frisco / Hotel** persists throughout: **21 records / $5.00 median / $136.25 total**. Weekday shows Friday/Saturday/Sunday at seven records each and totals $47.00/$52.95/$36.30. Timing shows **40.3 mean / 41 median / 24–55 minutes**, with no repeated zero-entry caveat. The three About sections are complete and readable by scrolling. Revert restores July 5, 2017–September 23, 2018 / All / All and is disabled afterward.

Desktop Date/City/Housing bounds are `(40,100,420,42)` / `(500,100,280,42)` / `(816,100,344,42)` on every page. At true-mobile 390px, City/Housing/Date are `(8,124,374,50)` / `(8,182,374,50)` / `(8,240,374,60)` on every page; client and scroll widths both remain 390px. Phone document heights including the 27px toolbar are **2127 / 1527 / 1227 / 2127 / 1267 / 1577** in navigation order. The 1px Trends backing worksheet retains shared filter controls without presenting another visible chart.

Native cost **$6.17 / $80 / $243.02** produces **$3.38 / $11.15 / $28.29** and ranges **$0.00 to $6.89 / $7.64 to $14.66 / $24.78 to $31.81**. The no-history Lewisville + Hotel test retains **$6.94 / $3.43 to $10.45** at cost $40. Actual Frisco tooltip text is `City: Frisco / Deliveries: 794`; a weekday example is `Weekday: Wednesday / Average tip: $8.07`. All four Weekday charts remain consistent with Overview city selection and clear.

UFO's shared filters and five controlled-source All-fields Select/auto-clear actions retain the tested combined behavior. California alone gives **653 / December 92 (14.1%) / 68.3% evening**. California + Light gives **150 / December 27 (18.0%) / 71.3% evening**; clearing Light restores 653. In the phone heatmap, that combined scope gives legend 1–7 and December 20:00 count 7; clearing gives 1–22. Full state/shape rankings and monthly/hourly details remain accessible. All five Phone destinations, About, returns and Revert passed; defaults are 2013 / All / All, with no marks and Revert disabled. No legacy worksheet participates in exposed navigation.

Both public Overview deliverables were left at defaults, temporary viewport overrides cleared, and authoring/recovery tabs closed. The native task is complete.

## Final artifacts and local website

| Item | Pizza | UFO |
|---|---|---|
| Canonical capture UTC | 2026-09-13 00:13:33 | 2026-09-12 22:06:58 |
| Package bytes | 147,859 | 5,182,165 |
| Package SHA-256 | `f679d226b2096fb3b09c7c0c9c1e8018cade063c0a96f68a316ddb0304e19919` | `650249a2a3e6a82ada2d8b4c3780ccca61011951d7887f67a84b09a00c9b0ae8` |
| Embedded TWB bytes | 500,289 | 381,600 |
| Embedded TWB SHA-256 | `a7e2e652e1285b0418b1d503a653e0d6e3faf5eedfa2e35de2225468c772738a` | `f45da24d288de153783ecc5fdf1814b337dcc88d9dca624080b803b371b8ee46` |
| Native desktop canvas | Six pages, 1200 x 850 | Five pages, 1200 x 840 |
| Native Phone canvas heights, navigation order | 2100 / 1500 / 1200 / 2100 / 1240 / 1550 | 2350 / 1550 / 1780 / 1450 / 1550 |
| Official SVG text | 120 nonempty Arial texts, zero ellipses | 93 nonempty Arial texts, zero ellipses |
| Full raster / preview | 1600 x 1133 / 1280 x 906 | 1600 x 1120 / 1280 x 896 |

The current [Pizza manifest](published/pizza-delivery-published-manifest.json) and [UFO manifest](published/ufo-sightings-published-manifest.json) are canonical. Their original Hyper hashes remain `614f454d2f64b3de28ee7fae5d18eed3e712c9c474cf21a64e9323b72fbc5f52` and `c0ddd072e24b69b66e02d5ec594708bbb1ac60bfcb117a8d0d776da593b6b16a`. Independent redownload ZIP hashes can differ because of container metadata; all uncompressed members were checked for equality.

Final Pizza SVG: **342,118 bytes**, SHA-256 `4972a2572397aaefe1855d119a6ef17e0c932f1e3bde870c547432f78bf8de7c`. Final UFO SVG: **580,106 bytes**, SHA-256 `8239b0afd4751855ca2b1a438e07e5c9def536573d592a58a6df18862cf1e2a9`. Complete native rasters passed visual review. The final Pizza density-96 raster has **zero differing channels out of 7,251,200** against the already-built preview source; all eight Pizza variants and image metadata remain current without a redundant build. See [pixel parity][pixel], [Pizza SVG check][pizza-svg] and [UFO SVG check][ufo-svg].

| Local website component | Verified result |
|---|---|
| Wide embeds | At professional 1920px: centered 1200 x 880 Pizza / 1200 x 870 UFO, equal 123.667px gutters, no overflow, full native toolbar. |
| Responsive scope | Dashboard-only 1800px stage; other pages 1500px. Containers below 1226px show previews. Personal one-rail 1440px fits; professional four-rail 1440px intentionally falls back. Navigation rails unchanged. |
| Mobile previews | True-mobile 390px Pizza: 312.667 x 221.698 from 1280 x 906. Recorded UFO route: 302.667 x 212.260 from 1280 x 896. Complete previews/Open actions, no overflow or loaded iframe. |
| Open dashboard | Both exact links select Phone; Pizza launch DOM shows 390 x 2127, all six destinations and defaults. Its independent reader supplies visual proof because the launch-tab screenshot was unreliable. UFO launch shows 390 x 2377. |
| Reset and lifecycle | Saved Overview URL with `:revert=all`, no `:iid`; direct load, soft entry/re-entry, cleanup and keyboard/pointer use passed. Pizza $80 input resets to $40; UFO resets to 2013 / All / All. |
| Build | [Both-preview build][build] passed unchanged retry after an unrelated Windows `UNKNOWN` file-write error. |
| Full tests | [Full test chain][tests] passed; no later runtime changes. |
| Final parity | [158 assertions][parity], zero issues at 2026-09-13 00:18:40 UTC, covering 16 images, eight page pairs, six helpers, two manifest contracts and 17 bundles. Public contributions exclusion is intentionally checked rather than claiming identical manifest bytes. |

## Numerical authority retained

The [Pizza concept][pizza-concept] and [UFO concept][ufo-concept] supplied visual direction. Native charts, audited extracts and actual interaction/export evidence determine accepted behavior. The [Pizza source audit][pizza-audit] and [UFO source audit][ufo-audit] remain numerical authority. Source/caveat explanations below belong in native About, not repeated analytical footers.

### Pizza source and comparisons

Preserve all **1,251 fact rows**; one OrderNum is reused, so COUNTD(OrderNum) is 1,250. The 11,286 weather rows are not deliveries, and weather/ZIP relationships must not multiply the facts. Coverage is July 5, 2017–September 23, 2018; the first/last months are partial. Tips total $8,933.49, average $7.14, median $6.00. Recorded minutes are elapsed order-to-delivery time, not driver work hours, and include 34 zeros. Housing counts are Residential 998, Apartment 162, Hotel 67, Business 24. Hotel gives 67 / $6.00 / $461.52; Lewisville gives 12 / $10.00 / $140.00 and ten populated months.

| City | Average tip | Records | Mean recorded minutes |
|---|---:|---:|---:|
| Frisco | $7.28 | 794 | 41.9 |
| Plano | $6.63 | 348 | 36.5 |
| The Colony | $7.09 | 69 | 40.2 |
| Carrollton | $7.36 | 16* | 45.6 |
| Lewisville | $11.67 | 12* | 39.5 |
| McKinney | $8.91 | 10* | 52.5 |
| Allen | $6.50 | 2* | 48.0 |

Asterisks/pale bars identify n < 30 as a display heuristic. July 2017–September 2018 monthly average tips are 6.86, 6.87, 7.32, 6.83, 8.23, 7.52, 6.48, 7.74, 6.84, 6.72, 7.17, 7.34, 6.98, 6.67, 7.56. Corresponding counts are 93, 109, 94, 75, 77, 97, 82, 71, 86, 73, 78, 82, 69, 75, 90.

| Weekday | Records | Average tip | Median tip | Total tips |
|---|---:|---:|---:|---:|
| Monday | 85 | $7.83 | $6.00 | $665.82 |
| Tuesday | 17 | $7.50 | $6.60 | $127.46 |
| Wednesday | 45 | $8.07 | $6.00 | $363.04 |
| Thursday | 6 | $6.75 | $5.00 | $40.47 |
| Friday | 367 | $7.47 | $6.00 | $2,741.04 |
| Saturday | 334 | $7.18 | $6.00 | $2,398.14 |
| Sunday | 397 | $6.54 | $5.12 | $2,597.52 |

The [reproducible package](analysis/pizza-weekday-estimate/README.md) preserves frozen source/math/spec outputs; seven compact outputs reproduced exactly. [Native implementation](analysis/pizza-weekday-estimate/native-implementation.md) documents actual captions and the continuous cost parameter without a cent-step constraint. The cost-only ridge model has 751 fit/model-selection rows, 249 calibration rows and 251 later holdout rows, with whole dates kept together. Holdout MAE is $2.61692664 versus $3.05944223 for the $6 baseline; 203/251 tips fall in the approximate 80% interval. Published formula precision differs from the approved model by at most **3.91e-14** across all source rows; all **24,937** interval-format cases pass. Counts reflect recorded schedules, not demand; the data does not support hourly-earnings, causal or future-coverage guarantees.

### UFO source and comparisons

The archive has **88,875 reports**; **1,262** lack a usable sighting date. Valid dates extend from 1906 through May 8, 2014. The overview includes the 48 contiguous U.S. states, excludes Alaska/Hawaii/D.C./territories/unknown locations and defaults to 2013. There is no population denominator, current reporting feed, verified-event classification or timezone conversion.

| Default metric | Audited value |
|---|---|
| Reports / peak month | 6,334 / July 810 (12.8%) |
| Evening, recorded 18:00–23:59 | 4,551 reports (71.9%) |
| Top states | California 653; Florida 483; Ohio 335; Washington 300; Pennsylvania 280 |
| Leading state share | California 10.3% |
| Top shapes | Light 1,368; Fireball 855; Circle 837; Triangle 500; Sphere 495 |
| Unknown or missing shape | Unknown 399 + missing 80 = 479 (7.6%); Other 320 is separate |
| Peak recorded hour | 21:00–21:59: 1,223 reports (19.3%) |
| California selection | 653; December 92 (14.1%); 68.3% evening |
| Light selection | 1,368; August 179; 76.7% evening |
| 2012 selection | 6,626; July 775; 68.1% evening |

## Historical provenance and completion boundary

Original `sources/` extracts remain untouched. [Earlier published backup](published/pre-fidelity-revision-20260912-074558/) and prior research/verification artifacts preserve the historical design iterations. Their earlier four-page Pizza navigation, footer placements, archive hashes and viewport measurements are superseded by this final record. Unvalidated `*-revamped` candidates are not published artifacts; [README provenance](README.md#published-artifacts-and-provenance) identifies the distinction.

The [snapshot/preview workflow](README.md#refresh-and-verify) documents fixed-ID helpers, original-data checks, faithful official-SVG rasterization and source/public verification. **F1–F10 are complete for this revision. Tableau Public is updated; the website build and QA are local, with no website deployment claimed.**

[structure]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-final-structural-verification.json
[pizza-ui]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-native-interaction-qa.json
[pizza-independent]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-final-independent-acceptance.json
[pizza-web]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-website-qa.json
[ufo-web]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/ufo-feedback-website-qa.json
[parity]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-final-parity.json
[pixel]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-final-export-pixel-parity.json
[pizza-svg]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-svg-inspection.json
[ufo-svg]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/ufo-feedback-svg-inspection.json
[build]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-both-preview-build-retry.log
[tests]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/feedback-full-test.log
[pizza-concept]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/pizza-dashboard-concept.png
[ufo-concept]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/ufo-dashboard-concept.png
[pizza-audit]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/research/pizza/pizza-audit.md
[ufo-audit]: C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/research/ufo/audit.md
