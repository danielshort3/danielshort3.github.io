# UFO concept fidelity revision

Implementation plan, 2026-09-12. This describes the next native web-authoring revision; it is not a claim that these changes are published. Preserve the existing `UFO Sighting Dashboard - 2013` dashboard and its Public identity. The archived published workbook under `published/pre-fidelity-revision-20260912-074558/` is the rollback baseline. The XML-generated `ufo-sightings-revamped.twbx` is a separate, unrendered candidate and is not the implementation baseline.

## Acceptance target and scope

Match the approved `ufo-dashboard-concept.png` hierarchy at approximately 1200 × 840: compact header and controls, a single three-part KPI strip, a combined map/ranking panel, and a heatmap/shape row. Restore the concept's supporting facts, legends, explanations, and working destinations. Do not reproduce invented AI heatmap cell colors or add decorative controls that do nothing.

All analytical views use the same historical report records, the explicit 48-state geography, and common Year/State/Shape controls. Keep Other, Unknown, and Not recorded distinct. Do not remove valid low-count places, deduplicate records silently, or substitute a different public dataset.

The source is the preserved Hyper extract from [the exact original Public workbook](https://public.tableau.com/workbooks/UFO_Sightings_16769494135040.twb). `research/ufo/data-profile.json`, `quality-profile.json`, and `filter-qa-expectations.json` in the external research folder contain the calculations behind the figures below. The displayed observation year is selected, not hard-coded into headings. Year has one selected, non-null year; remove its optional All entry and exclude Null, so all panels describe the same dated records without introducing a different all-years count definition.

## Reusable calculated fields

These are **web-editor display names** and formulas using the original source field captions. Create missing reusable fields rather than repeatedly entering long ad hoc pills. Reuse the existing `Contiguous United States` and `Reported shape` fields unchanged. Retire an ad hoc equivalent only after the replacement renders the same values.

```tableau
// Existing: Contiguous United States — filter True, in context
[Country] = 'us' AND [State] IN (
  'al','ar','az','ca','co','ct','de','fl','ga','ia','id','il',
  'in','ks','ky','la','ma','md','me','mi','mn','mo','ms','mt',
  'nc','nd','ne','nh','nj','nm','nv','ny','oh','ok','or','pa',
  'ri','sc','sd','tn','tx','ut','va','vt','wa','wi','wv','wy'
)

// Existing: Reported shape
IF ISNULL([Shape]) THEN 'Not recorded'
ELSE UPPER(LEFT([Shape], 1)) + MID([Shape], 2) END

// Report records — aggregate measure; format #,##0
COUNT([Datetime])

// Recorded month — discrete integer, ascending in chronological charts
DATEPART('month', [Datetime])

// Recorded month name
DATENAME('month', [Datetime])

// Recorded hour — discrete integer, ascending in chronological charts
DATEPART('hour', [Datetime])

// Recorded hour label
RIGHT('0' + STR([Recorded hour]), 2) + ':00'

// Recorded hour range
[Recorded hour label] + '–' + RIGHT('0' + STR([Recorded hour]), 2) + ':59'

// Evening share — aggregate measure; format 0.0%
SUM(IIF([Recorded hour] >= 18, 1, 0)) / [Report records]

// Unknown or missing reports — aggregate measure; format #,##0
ZN(SUM(IIF(ISNULL([Shape]) OR LOWER([Shape]) = 'unknown', 1, 0)))

// Unknown or missing share — aggregate measure; format 0.0%
[Unknown or missing reports] / [Report records]

// Share of selected reports — table calculation; format 0.0%
[Report records] / WINDOW_SUM([Report records])

// Overview rank — table calculation; integer
INDEX()

// Selected year caption — aggregate string; also safe if a multiyear view is added later
IF MIN(YEAR([Datetime])) = MAX(YEAR([Datetime])) THEN
  STR(MIN(YEAR([Datetime])))
ELSE STR(MIN(YEAR([Datetime]))) + '–' + STR(MAX(YEAR([Datetime]))) END

// Scope caption — aggregate string; describes the reports actually represented
IF COUNTD([State]) = 48 THEN 'Contiguous U.S.'
ELSEIF COUNTD([State]) = 1 THEN MIN([State name])
ELSE STR(COUNTD([State])) + ' states with reports' END
+ ' · ' + [Selected year caption]

```

Create `State name` once. Use this field for rankings, captions, and the visible State filter; keep raw State on geographic Detail and for selected-field action mapping. Apply the visible State name filter to all worksheets using this source, then remove the redundant old raw-State quick-filter control after verifying its All selection imposes no restriction.

```tableau
CASE [State]
WHEN 'al' THEN 'Alabama' WHEN 'ar' THEN 'Arkansas' WHEN 'az' THEN 'Arizona'
WHEN 'ca' THEN 'California' WHEN 'co' THEN 'Colorado' WHEN 'ct' THEN 'Connecticut'
WHEN 'de' THEN 'Delaware' WHEN 'fl' THEN 'Florida' WHEN 'ga' THEN 'Georgia'
WHEN 'ia' THEN 'Iowa' WHEN 'id' THEN 'Idaho' WHEN 'il' THEN 'Illinois'
WHEN 'in' THEN 'Indiana' WHEN 'ks' THEN 'Kansas' WHEN 'ky' THEN 'Kentucky'
WHEN 'la' THEN 'Louisiana' WHEN 'ma' THEN 'Massachusetts' WHEN 'md' THEN 'Maryland'
WHEN 'me' THEN 'Maine' WHEN 'mi' THEN 'Michigan' WHEN 'mn' THEN 'Minnesota'
WHEN 'mo' THEN 'Missouri' WHEN 'ms' THEN 'Mississippi' WHEN 'mt' THEN 'Montana'
WHEN 'nc' THEN 'North Carolina' WHEN 'nd' THEN 'North Dakota' WHEN 'ne' THEN 'Nebraska'
WHEN 'nh' THEN 'New Hampshire' WHEN 'nj' THEN 'New Jersey' WHEN 'nm' THEN 'New Mexico'
WHEN 'nv' THEN 'Nevada' WHEN 'ny' THEN 'New York' WHEN 'oh' THEN 'Ohio'
WHEN 'ok' THEN 'Oklahoma' WHEN 'or' THEN 'Oregon' WHEN 'pa' THEN 'Pennsylvania'
WHEN 'ri' THEN 'Rhode Island' WHEN 'sc' THEN 'South Carolina' WHEN 'sd' THEN 'South Dakota'
WHEN 'tn' THEN 'Tennessee' WHEN 'tx' THEN 'Texas' WHEN 'ut' THEN 'Utah'
WHEN 'va' THEN 'Virginia' WHEN 'vt' THEN 'Vermont' WHEN 'wa' THEN 'Washington'
WHEN 'wi' THEN 'Wisconsin' WHEN 'wv' THEN 'West Virginia' WHEN 'wy' THEN 'Wyoming'
END
```

`Report records` agrees with the internal object count for every selected valid year. The 1,262 null timestamps in the full source cannot be assigned to a year; explain this on About. Do not change the geographic scope calculation to conceal that distinction.

## Worksheet recipe and dynamic supporting facts

Reuse the seven currently displayed, verified sheets. Duplicate them for new notes or detail pages. Remove inherited INDEX filters before building a total or completeness note. Create all new worksheets before arranging dashboards.

| Sheet | Native construction and displayed information |
|---|---|
| Reports | No Rows/Columns; `Report records` on Text, then `Scope caption`. Left-align the full mark. Title Reports. |
| Peak month | Recorded month on Rows; sort count descending; Overview rank ≤ 1. Text: month name, count, Share of selected reports. Compute rank and share over the month dimension, with share's window seeing all months before the rank filter. Title Peak month. The default caption is `810 reports · 12.8% of total`. |
| Evening reports | No dimensions or INDEX. Evening share on Text; static caption `Recorded hour 18:00–23:59`. |
| Count by State Map | Country and raw State on Detail, Report records on Color. No City, city-count calculation, or minimum-count filter. Filled map; fixed U.S. extent; no unnecessary basemap labels; 100% opacity, pale-to-#005FED sequential color, thin white state boundaries. |
| Count by State | State name Rows, Report records Columns/Text, raw State Detail; descending count, Overview rank ≤ 5. Entire View with a visible count axis and enough row height for the five marks. |
| Leading state note | Duplicate state ranking; rank ≤ 1; hidden headers. Text: State name, Report records, Share of selected reports. Default `California: 653 reports (10.3%)`. Share is across all represented states, not only the five visible bars. |
| Month/Hour (Military Time) Heatmap | Recorded month Rows, Recorded hour Columns; chronological order on both. Abbreviated month labels; discrete hour labels that fit at desktop width; Count Color; square marks, thin white cell boundaries. Tooltip: month, hour range, count. Include a real quantitative color legend. |
| Peak hour note | Duplicate an hourly aggregation; Recorded hour Rows; descending count; rank ≤ 1. Text: Recorded hour range and Report records. Default `Peak recorded hour: 21:00–21:59 · 1,223 reports`. |
| Count by Shape | Reported shape Rows; Report records Columns/Text; descending count; rank ≤ 5. No original Other/Unknown/null exclusions. Visible count axis and five readable bars. |
| Shape completeness note | Duplicate Reports, retaining no dimensions or rank. Unknown or missing reports and Unknown or missing share on Text. Default `Unknown or missing shape: 479 reports (7.6%)`. |
| Phone month/hour heatmap | Duplicate the corrected desktop heatmap, then swap axes: 24 hours down, 12 abbreviated months across. Entire View. This is a separate sheet, not a phone-specific fit override on the desktop sheet. |

Use native inserted field tokens in label editors; do not type a rendered default value as a static caption. For the peak notes, use a consistent, tested tie order, and do not imply that a tied maximum is unique. Test the percentage table calculation after applying the late INDEX filter: displaying 100% for the peak month or leading state fails acceptance.

Keep real zero-report heatmap combinations blank unless Tableau domain completion is explicitly configured and verified. A small caption says `Blank cells: no reports in the selection`. The extract supplies 283 nonzero cells and five empty combinations in the default 12 × 24 grid; never invent records to fill them.

## Overview geometry, desktop 1200 × 840

Coordinates are X, Y, width, height in native layout pixels. White panels on #F5F8FC; #10254F text, #526586 secondary text, #005FED emphasis. Arial throughout, including explicitly set mark-label runs. Native Layout has a verified Corner Radius control: use white backing Blank objects with radius10, a restrained 1px pale border, and consistent padding. Remove inherited inner frames, pane rules, and excess worksheet shading.

| Object | Geometry / content |
|---|---|
| Brand | 26, 12, 850, 16 — `DANIEL SHORT / DATA EXPLORER`, 9–10pt |
| Title | 26, 31, 950, 34 — `UFO sighting reports`, 25–28pt bold |
| Subtitle | 26, 66, 950, 19 — `Historical reported observations across the contiguous United States` |
| About navigation | 1060, 14, 114, 25 — working About the data destination |
| Common controls | Inline labels plus single-value dropdowns: Year label26, 96, 30, 20 and control60, 90, 108, 30; State label178, 96, 33, 20 and control215, 90, 153, 30; Shape label380, 96, 38, 20 and control422, 90, 148, 30. Hide native stacked titles if supported; otherwise allow a36–40px control row and adjust the strip without clipping. Native All is acceptable for State/Shape; do not pretend its UI label says All 48. |
| Reset guidance | 580, 93, 216, 23 — brief Tableau Revert guidance if necessary. The website's actual Reset filters button reloads the saved native view. Do not create a nonfunctional native Reset button. |
| Navigation strip | 820, 89, 354, 31 — Overview / Places / Timing / Shapes as working native Navigation objects; selected item styled blue with a thin underline. |
| KPI strip | 16, 130, 1168, 90 — one white panel. Reports 40, 139, 350, 72; Peak 422, 139, 350, 72; Evening 806, 139, 350, 72. Thin separators x408/x792. Titles 11–12pt, values 28–30pt, captions 10–11pt. All content left-aligned. |
| Geography panel | 16, 231, 1168, 277 — one white panel, divider at x752. Map title/subtitle 28, 239, 712, 36. Map view 28, 278, 712, 217. State heading 778, 243, 386, 24; bar sheet 778, 274, 386, 186; leading-state note 778, 469, 270, 23. Working View all states link 1048, 474, 118, 20. |
| Map legend/caveat | Native color legend placed inside map panel's lower-right free space, approximately 550, 410, 180, 30; `Raw counts; not population-adjusted` beneath. Adjust after rendered map extent, never cover states. |
| Timing panel | 16, 519, 728, 265 — title/subtitle 28, 528, 700, 37; heatmap 28, 568, 604, 183; count legend 642, 698, 84, 43; dynamic peak-hour note 28, 756, 700, 18. Put the blank-cell meaning in the subtitle or tooltip if the caption would collide. |
| Shapes panel | 754, 519, 430, 265 — title/subtitle 768, 528, 400, 37; bars 768, 569, 400, 179; completeness note 768, 756, 280, 18; working Explore all shapes link 1053, 756, 117, 18. |
| Footer | 16, 797, 1168, 31 — `How to read this: Counts describe submitted reports, not verified events or incidence rates.` Source identified at right as the historical NUFORC-derived workbook, extract through May 2014. |

Chart subtitles restore `Report counts · select a state to focus the views`, `Recorded observation time · hours run from midnight to late evening`, and `Top 5 · all shapes included in total`. Do not use an interaction instruction until its native action passes QA. The map's color legend should retain truthful numeric endpoints; literal Fewer/More labels are optional. The verified exact-blue route is categorical field on Color → Edit Colors → member option Control+Space → Hex textbox → #005FED → Enter. A constant categorical calculation such as `Brand color = 'Reports'` can give every bar the same color; hide its redundant legend and verify the actual mark color.

## Real native destinations

Create four additional dashboards with working navigation back to Overview and among the detail pages. Use the same 1200 × 840 header/control/nav structure. The destinations must work from the published view with sheet tabs hidden; do not rely on the author's worksheet tabs.

| Destination | Worksheets and complete scope |
|---|---|
| Places | Duplicate corrected state ranking as `All states`, remove its INDEX filter, retain full State name/count/share. Use 1148 × 610 at 24, 190 with readable row height and vertical scrolling when needed. This is the complete ranking of represented states, up to48, rather than another unexplained top subset. Keep the raw-count/population caveat. Do not expose the legacy city worksheets as part of this revision. |
| Timing | `Monthly reports`: numeric month ascending, count bars, 560 × 580 at 24, 190. `Hourly reports`: numeric hour0–23, count bars, 560 × 580 at 612, 190. Include recorded-time meaning and dynamic peak notes. These are report-time distributions; Date Posted is not substituted for Datetime. The Overview already supplies the month/hour grid. |
| Shapes | `All reported shapes`: duplicate corrected shape ranking, remove INDEX, show count and share, 1148 × 610 at 24, 190. No old shape-line/table exclusion filters. Scroll instead of shrinking labels below readable size. Include Other, Unknown, and Not recorded, with the completeness note. No new shape-month matrix is needed. |
| About the data | Readable native text cards: source and historical coverage; selected scope and metric definitions; shape/missing-data meaning; limitations and how to interact/reset. Identify the exact workbook and NUFORC-derived report archive. Include the source link through a supported text hyperlink/URL action, with the website source link as an independently working route. |

About must state: source contains 88,875 records, valid observation timestamps from 1906-11-11 through 2014-05-08; 2014 is incomplete; no current feed is implied. Defaults select 2013 and the 48 contiguous states. Count is one report record, not a verified event/person. Raw counts have no population adjustment or verified timezone-normalization policy. Unknown and absent shape are combined only for the completeness statistic; Other is separate. In the full original extract 1,262 dates, 12,561 country values, 7,519 state values, and 3,118 shapes are null; these are source-level counts, not current-filter counts. There are 34 surplus exact duplicate rows across the substantive columns, retained; none occur in the default 2013 contiguous-state slice. Duration values include nonpositive values and extreme outliers, so duration is not a headline metric. Detailed raw records remain available through Tableau View Data/download; this redesign does not silently clean the source.

## Shared filters and actions

1. Keep Contiguous United States=True in context on every analytic sheet. Apply Year, State name, and Reported shape to **All Using This Data Source**, including detail pages and both heatmap copies. Default Year=2013, State/Shape=All. Year excludes Null and has no All option. Use All Values in Context for State/Shape choices so an empty combination does not remove the All recovery choice.
2. Continue post-filter INDEX limits: five for overview rankings, one for peak/leading notes, none on totals, completeness, full listings, or chronological distributions. Do not use native dimension Top N filters that rank before ordinary dimension filters. Configure Compute Using explicitly for each relevant dimension and verify the visible rank and percentage.
3. Add named native dashboard filter actions `Focus state from map` and `Focus state from ranking`, running on Select, mapping raw State to raw State on the intended other analytic sheets. Clearing selection shows all values. Keep raw State on Detail in ranking sheets. Verify the action does not accidentally transmit Count, City, or other fields. Add `Focus reported shape` only with the same explicit Reported shape mapping and clearing behavior.
4. Quick-filter selections and mark selections are distinct in Tableau. Explain `Select a mark to focus; click it again or use Revert to clear`. A raw-State action does not rewrite the State name dropdown. Dynamic captions must describe the resulting data. The website Reset filters control restores the saved default iframe; native toolbar Revert is the native reset route.
5. Native Navigation objects implement Overview/Places/Timing/Shapes/About and the two detail links. Use clear labels and verify filter state persists when navigating. A highlighted active label must not masquerade as a separate toggle. No new parameter is needed for navigation or reset.

## Phone and tablet

Phone uses a custom, vertically stacked layout with the same working Year/State/Shape controls, navigation, three KPIs, map/ranking, all three supporting facts, legends, and About access. Use 350–375px available content width, 12–16px outer padding, readable text, and normal vertical scrolling. KPI blocks are left-aligned. The separate phone heatmap has **24 hour rows × 12 month columns**, approximately 500–540px high; show all month abbreviations and complete hour labels, plus count tooltips and legend. Do not crop the hour axis, hide half the hours, or replace the grid with an aggregate that loses month/hour information.

All device-specific sheets must first exist on Tableau's Default layout. Create the phone heatmap on Default, then make Desktop and Tablet explicit custom layouts containing only the desktop version, and Phone custom containing only the transposed version. **Keep the visible Default and Desktop composition at concept proportions.** Verify the exact public desktop view and a native Image/SVG export immediately after adding the phone sheet; do not leave an extra sheet or a giant blank area below the intended canvas. Do not assume an off-canvas sheet or a larger Default superset will be excluded from export. Adjust the Default superset only as needed after actual native proof, and restore the baseline geometry if the duplicate affects export. Detail destinations also require phone layouts: readable scrolling state/shape lists and stacked timing charts, with working navigation back.

This follows Tableau's [device-layout workflow](https://help.tableau.com/current/pro/desktop/en-us/dashboards_dsd_create.htm). Worksheet fit is shared: changing a sheet's phone fit can regress desktop. Separate sheets are essential for the transposed grid. Test phone at the actual deployed iframe width, not only the author's device preview.

## Expanded native QA expectations

Recomputed locally on 2026-09-12 by `research/ufo/revision_qa.py`, using a disposable copy of the preserved Hyper. Full SQL, tied maxima, counts, and checks are in `research/ufo/revision-qa-expectations.json` in the external research folder. Original Hyper SHA-256 remained `c0ddd072e24b69b66e02d5ec594708bbb1ac60bfcb117a8d0d776da593b6b16a`. For each slice, month/state/hour group totals equal both COUNT(*) and COUNT(Datetime). Percentages below are count ÷ all report records in the **current selection**, rounded to one decimal.

| Year / State / Shape | Reports | Peak month: count / share | Leading state: count / share | Peak recorded hour: count / share | Unknown + missing: count / share | Evening: count / share |
|---|---:|---|---|---|---|---|
| 2013 / All / All | 6,334 | July: 810 / 12.8% | California: 653 / 10.3% | 21:00–21:59: 1,223 / 19.3% | 479 / 7.6% | 4,551 / 71.9% |
| 2013 / California / All | 653 | December: 92 / 14.1% | California: 653 / 100.0% | 21:00–21:59: 111 / 17.0% | 42 / 6.4% | 446 / 68.3% |
| 2013 / All / Light | 1,368 | August: 179 / 13.1% | California: 150 / 11.0% | 21:00–21:59: 273 / 20.0% | 0 / 0.0% | 1,049 / 76.7% |
| 2012 / All / All | 6,626 | July: 775 / 11.7% | California: 692 / 10.4% | 21:00–21:59: 1,205 / 18.2% | 534 / 8.1% | 4,514 / 68.1% |
| 2013 / California / Light | 150 | December: 27 / 18.0% | California: 150 / 100.0% | 21:00–21:59: 26 / 17.3% | 0 / 0.0% | 107 / 71.3% |
| 1910 / All / All | 2 | May and June tied: 1 each / 50.0% each | Maine and Texas tied: 1 each / 50.0% each | 15:00–15:59 and 21:00–21:59 tied: 1 each / 50.0% each | 1 / 50.0% | 1 / 50.0% |
| 1910 / California / Light | 0 | No peak | No leading state | No peak | 0; share undefined | 0; share undefined |

The 1910 combination is an extract-backed empty case, not a promise that context-restricted dropdowns offer every empty combination. Test it only if the native controls permit that selection; otherwise use a permitted empty intersection through existing selection actions. Never fabricate a nonexistent year filter option. The earliest represented year in the 48-state scope is1910, although the full international archive has an earlier1906 observation.

Shape completeness decomposition: default399 Unknown +80 Not recorded=479; California36+6=42; 2012all447+87=534; 1910all1+0=1. Other is excluded from this statistic: its corresponding counts are320,30,333,0. Filtering Shape=Light makes the selected-scope unknown/missing count zero; it must not keep displaying the default479.

**Calculation risks to resolve in the actual renderer:**

- `WINDOW_SUM([Report records])` must cover all months/states/hours **before** the late INDEX filter. A default peak share of100% or state share of100% is wrong; California-only state share100% is correct. FIXED totals that ignore State/Shape filters are not substitutes.
- Tableau partitions can change when another dimension is added to Detail or Text. For state bars with both State name and raw State, explicitly address both in Specific Dimensions, or remove unneeded duplicate dimensions from a caption sheet. For month/hour note labels, use ATTR(label) where possible so the label does not create a new partition. Verify the full-window denominator on every caption sheet after adding label fields.
- 1910 proves that INDEX≤1 chooses one of several equal maxima. Prefer displaying all tied maxima when the native fit permits; otherwise retain neutral `Peak month` wording and a short tie indicator or an About note explaining the verified deterministic ordering. Do not add a long overview explanation. A reusable candidate `Peak tie count` is `WINDOW_SUM(IIF([Report records] = WINDOW_MAX([Report records]), 1, 0))`, with the same addressing as the peak dimension. Its actual native execution still requires validation. If selecting one tied time value, verify a deterministic order; alphabetical month order selects June before May, while chronological order selects May.
- Empty selections have count0 and no valid percentage denominator. Display `No reports`/an em dash for percentages and peaks, with a working clear/reset route; do not turn undefined shares into0.0%. `ZN` is appropriate for a missing count, not for a divide-by-zero percentage. Verify Tableau's empty-sheet behavior rather than assuming a text calculation always produces a mark.
- Keep the selected non-null year and explicit48-state filter common to every view. Source-wide1906 coverage, excluded DC/PR, missing timestamps, hidden shape exclusions, or a leftover city minimum filter must not change the denominator. Top-five lists may legitimately contain fewer than five categories in a small selection.

## Fast native sequence and proof

1. Finish fields and new worksheets first. Duplicate current, correctly scoped sheets; never expose the old city grouping or inconsistent shape views. Reuse label styles by duplicating the closest sheet. Format the entire workbook Arial and remove inner pane rules once, then adjust only relevant titles/marks.
2. Build real detail destinations and Navigation objects; then compose the compact Overview. Floating Layout controls give deterministic geometry. Commit each numeric field with Tab, read its rendered value, and verify the actual zone bounds before editing the next object. Do not batch a fourth unverified dimension into an already pending update. Close color/padding dialogs with their Close buttons.
3. Arrange desktop and phone variants independently. Inspect the complete desktop after every shared worksheet change, especially map extent, fit, heatmap headers, font, and count axes.
4. Default acceptance: **6,334; July 810 and 12.8%; evening 71.9%; California 653 and 10.3%; peak hour 21:00–21:59 and 1,223; unknown/missing 479 and 7.6%**. Top states California653/Florida483/Ohio335/Washington300/Pennsylvania280. Top shapes Light1368/Fireball855/Circle837/Triangle500/Sphere495.
5. Meaningful nondefault checks: California+2013=653 reports, 68.3% evening, December92; shapes Light150/Circle83/Fireball72/Sphere62/Triangle45. Light+2013=1,368 reports, 76.7%, August179; states California150/Florida100/Washington75/Pennsylvania61/Ohio53. 2012 with all48states=6,626, 68.1%, July775. Confirm new share/peak-hour/completeness captions recompute, rather than retaining default strings.
6. Check real navigation in both directions, every detail page's matching scope, action selection/clearing, filter recovery from an empty combination, and website Reset/native Revert. Full state/shape lists must have no hidden top/minimum filter; shape lists must include all categories. Legacy city and inconsistent shape worksheets remain outside all user-facing destinations.
7. Restore 2013/All/All and clear all mark selections. Inspect desktop at1200×840 and phone at375px in a fresh public view after Publish. Capture actual native Image/SVG exports and download the actual published TWBX. Replace preview/archive files only from this verified publication, preserving the prior backup. Renderer evidence, not XML schema validity or source presence, determines completion.

Useful native object references: [Navigation objects](https://help.tableau.com/current/pro/desktop/en-us/dashboards_create.htm), [precise floating layout](https://help.tableau.com/current/pro/desktop/en-us/dashboards_organize_floatingandtiled.htm). No supported Public web TWBX import route was found; this revision is authored in the existing authenticated editor.
