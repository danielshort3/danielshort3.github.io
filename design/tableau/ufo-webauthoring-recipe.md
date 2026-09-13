# UFO native Tableau redesign

This folder contains a native Tableau implementation candidate and its original source. Website integration and the Tableau Public web-authoring draft are separate work products. The generated package has not been imported into Tableau Public or rendered by Tableau Desktop.

## Rebuild

```powershell
python build/tableau/rebuild-ufo.py
```

The source is `design/tableau/sources/UFO_Sightings.twbx`, downloaded from [the exact published workbook](https://public.tableau.com/workbooks/UFO_Sightings_16769494135040.twb). Its SHA-256 is pinned in the script. The builder refuses a different snapshot, so later publication cannot silently change the baseline.

Outputs:

- `ufo-sightings-revamped.twbx`: native Tableau package.
- `ufo-sightings-revamped.twb`: XML for review and inspection.
- `ufo-sightings-revamped.manifest.json`: source identity, scope, extract hashes, and validation limits.

The original Hyper extract, all packaged supporting files, and the original published workbook/dashboard repository identifiers remain unchanged. The rebuilt ZIP is deterministic.

Optional structural validation uses `--schema PATH_TO_OFFICIAL_TWB_XSD` and requires `lxml`. The [official Tableau schema](https://github.com/tableau/tableau-document-schemas/blob/main/schemas/2026_1/twb_2026.1.0.xsd) omits imported user/XML namespace definitions. The builder supplies those definitions and reports deviations already present in the original workbook separately from newly introduced errors. This checks XML structure, not calculation execution or rendered behavior.

## Exact metric scope

All default views describe report records with country `us`, sighting year 2013, and state in this explicit list of the 48 contiguous states:

```text
al ar az ca co ct de fl ga ia id il in ks ky la ma md me mi mn mo ms mt
nc nd ne nh nj nm nv ny oh ok or pa ri sc sd tn tx ut va vt wa wi wv wy
```

A common Boolean geography filter remains in context so choosing All in a visible State control retains this scope. The historical source also contains DC and Puerto Rico records; excluding only Alaska and Hawaii is insufficient when changing the year. The browser implementation uses the same state list, but its layout and validation remain separate from this generated candidate.

| Metric | Verified value |
|---|---:|
| Report records | 6,334 |
| Peak month | July, 810 |
| Recorded from 18:00 through 23:59 | 4,551 / 71.9% |
| Recorded from 20:00 through 23:59 | 3,679 / 58.1% |
| Peak recorded hour | 21:00–21:59, 1,223 |
| Unknown or missing shape | 479 |

Top states: California 653; Florida 483; Ohio 335; Washington 300; Pennsylvania 280.

Top shapes: Light 1,368; Fireball 855; Circle 837; Triangle 500; Sphere 495.

## Main dashboard structure

- Preserve the dashboard name `UFO Sighting Dashboard - 2013` and published identity.
- Default fixed canvas: 1,400 × 1,000; white panels over a pale background; blue marks and dark text.
- Clear title, historical scope, common Year/State/Shape controls, and a visible reset instruction using Tableau Revert.
- Three responsive KPI worksheets: reports, peak month, and evening share.
- Geographic panel: state map and directly labeled horizontal top-five state bars.
- Time panel: month-by-hour heatmap ordered 0 through 23.
- Shape panel: directly labeled horizontal top-five bars plus unknown/missing count.
- Phone layout retains visible controls, stacks KPI and ranking views, and uses an hourly bar chart in place of a compressed 24-column heatmap.

Existing city sheets are retained for detailed exploration. The city label combines city and state; the city map no longer removes locations with fewer than five reports. Existing shape sheets use one common shape-label field; Other, Unknown, and Not recorded remain distinct.

Native generated filter actions and their explicit receiving filters/slices are authored for state-map, state-bar, and shape-bar selections. Shared native filter groups apply the same Year, State, and Shape scope to all sheets.

## Native web-authoring equivalent

Tableau Public's browser editor has no supported TWBX import path. Reconstruct the design in its authenticated editor when Desktop Public Edition is unavailable. Browser edits do not prove that the generated package renders identically.

1. Work inside the existing dashboard. Preserve its published identity.
2. Duplicate an existing correctly scoped worksheet to create KPIs. Remove its top-N filter and dimensions before displaying report count; verify 6,334. Keep country and contiguous-state constraints.
3. Evening share calculation: `SUM(IIF(DATEPART('hour', [Datetime]) >= 18, 1, 0)) / COUNT([Datetime])`. Format as percentage, one decimal place.
4. For peak month, group by `DATENAME('month', [Datetime])`, sort by report count descending, retain a top-one INDEX table-calculation filter computed across that month dimension, and display both the month and count.
5. Swap the state and shape chart axes to horizontal bars, remove redundant Country headers and redundant continuous count coloring, and limit overview rankings to five after filtering.
6. Heatmap Hour sort must be chronological 0–23; the original explicitly used 12–23 then 0–11. Give every hour label sufficient width.
7. Remove the old city map/ranking and mismatched shape-line/table pair from the overview layout while retaining their worksheets.
8. Add the three KPI sheets, corrected shape bars, concise explanations, and visible common controls. Tableau's native Layout panel provides precise floating item dimensions and positions.
9. Verify dynamic filtering before publication. California should produce 653 reports, and returning to the default selection should restore 6,334. Verify that Unknown and Other are distinct and that KPI totals include missing shapes.
10. Change Year to 2012 while retaining the 48-state scope: expect 6,626 reports, 68.1% evening share, and July as the peak month with 775 reports. Restore 2013 afterward.

## Required publication checks

- The actual Tableau renderer opens the final workbook with no missing-field/calculation warnings.
- Default rendered values match the table above; bar counts, widths, and order agree.
- Year/State/Shape selections affect every intended chart and KPI; reset restores the verified default.
- Heatmap order and labels are readable at the deployed embed width.
- Phone layout is readable and retains working controls.
- Save/publish the tested draft, then inspect the live embed separately from local package checks.

Counts are reports, not verified events or incidence rates. The source contains no population denominator or explicit timezone-normalization policy, and the historical extract stops in May 2014.
