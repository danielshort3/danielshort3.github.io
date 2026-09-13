# Pizza Delivery: native Tableau build and browser recipe

**Published implementation:** the web rebuild is live as of September 11, 2026.
See [the final implementation notes](README.md) and the actual downloaded
[published workbook](published/pizza-delivery-published.twbx). The live overview
is 1200 × 1050 with seven views and shared City, Housing, and Date filters.
The richer layout, timing view, reference line, and click-action steps below
remain design alternatives; they are not a checklist of published features.

This package is a prepared native workbook, not a published dashboard. The browser
workbook is `Pizza_Delivery` and the embedded dashboard must remain named
`Pizza Delivery Dashboard` so its existing public URL continues to work.

## Prepared workbook

- Source snapshot: `sources/Pizza_Delivery.twbx`, downloaded from
  <https://public.tableau.com/workbooks/Pizza_Delivery.twb?showVizHome=no>.
- Builder: `build/tableau/rebuild-pizza.py` from the repository root.
- Output: `pizza-delivery-revamped.twbx`.
- `pizza-delivery-revamped.twb` is inspectable XML; open the packaged `.twbx` to
  include the extract. The standalone XML is not a self-contained data package.
- `pizza-delivery-revamped.validation.json` records the checks and pending work.

Build command from the repository root:

```powershell
python build/tableau/rebuild-pizza.py
```

An optional `--schema <path>` validates against the official Tableau TWB XSD at
<https://github.com/tableau/tableau-document-schemas/blob/main/schemas/2026_2/twb_2026.2.0.xsd>.
The checked output introduces no schema errors. The downloaded original and the
output share one legacy schema deviation: no trailing explain-data section.
Supplemental namespace declarations resolve the official XSD's unlocated XML/user
imports; user metadata is validated laxly. This does not prove calculation
semantics or native render behavior.

The Hyper extract and datasource connection/relationship XML remain unchanged.
The builder never edits the original package and never publishes anything.

## Browser authoring: smallest useful sequence

The signed-in authoring destination is
<https://public.tableau.com/authoring/Pizza_Delivery/PizzaDeliveryDashboard>.

1. Duplicate **Tip Forecast**. Turn Forecast and Trend Lines off. Replace
   `SUM(Tip)` on Rows with `AVG(Tip)`; keep continuous `MONTH(Date)` on Columns.
   Change Color to one blue (`#005FED`), retain `AVG(Tip)` on Label, format as
   `$0.00`, and include zero on the value axis. Rename to **Monthly average tip**.
   This restores the final observed month, September 2018, which the forecast
   previously ignored.
2. Duplicate that sheet. Change Rows and Label to `COUNT(Tip)`; change marks to
   Bar, color `#BBC7D2`. Rename **Monthly deliveries**. Place it below the average
   chart with the same date-axis width, as a shallow strip.
3. Duplicate **Tip $ Histogram**. Remove its `Tip (bin)` exclusion filter so all
   tips, including the $30, $32, and $40 outliers, remain included. Keep the native
   $2 bins. Remove count from Color and the bin value from Label. Use uniform blue,
   show the horizontal dollar axis, and retain a zero count baseline.
4. Create three Text sheets for **Deliveries**, **Median tip**, and **Total tips**.
   Put `COUNT(Tip)`, `MEDIAN(Tip)`, and `SUM(Tip)` on Text respectively, with no
   dimension on Rows/Columns/Detail. Format count `#,##0`, currency `$#,##0.00`.
   Use roughly 28 pt navy values and 11-12 pt labels. All-data values must be
   **1,251**, **$6.00**, and **$8,933.49**.
5. Duplicate **Order Quantity by City** and switch its Pie marks to Bar.
   Put `City` on Rows and `AVG(Tip)` on Columns, remove count from Size/Color,
   put `AVG(Tip)` on Label, and sort City by descending `COUNT(Tip)`.
   Use a zero dollar baseline and uniform blue. Add `COUNT(Tip)` and
   `AVG(Delivery_Length)` to Tooltip. Add the small-sample calculation below.
6. Place the three KPIs across the top, city bars left and tip distribution right,
   and the monthly average/volume pair across the bottom. Remove the duplicate
   maps and pies from this overview; timing/housing views can remain on a
   secondary dashboard.
7. Add Date, City, and Housing to a worksheet's Filters, initially including all.
   Show their filter controls. On each filter card use **Apply to Worksheets →
   All Using This Data Source**, then verify that every KPI and chart responds.
   Prefer a date-range control and compact City/Housing dropdowns.
8. Enable city selection as a dashboard filter. Clearing the selection must show
   all rows. Tableau's native **Revert** toolbar action is the reliable reset;
   do not draw a text-only reset button that does not actually reset state.

## Calculations and inclusion rules

**Delivery records** is `COUNT([Tip])`, which counts all 1,251 non-null delivery
facts. Do not use `COUNTD([Order Num])`: one order ID is repeated for distinct rows
and that would produce 1,250.

**Sample size** (aggregate calculation, add to Color or Tooltip):

```text
IF COUNT([Tip]) < 30 THEN "Fewer than 30"
ELSE "30 or more"
END
```

**Deliveries label with asterisk** (for an adjacent city text column):

```text
STR(COUNT([Tip])) +
IF COUNT([Tip]) < 30 THEN "*" ELSE "" END
```

**Histogram median reference**, recalculated after dimension filters:

```text
{ EXCLUDE [Tip (bin)] : MEDIAN([Tip]) }
```

Use the average of this LOD field as the reference-line value on the dollar
axis, scoped to the entire table. This avoids using the median of bin midpoints.
If this is awkward in the browser, show median in its KPI and tooltip first.

**Recorded minutes** already exists as `Delivery_Length` in the workbook:

```text
INT([Total Delivery Time - Split 1] * 60 + [Total Delivery Time - Split 2])
```

Keep all records. The 34 zero-minute records and the 135-minute record are
source observations, not hidden exclusions. Label this measure **Recorded min**
and explain it as order-to-delivery elapsed time. It is not driver labor time.

## Expected results and presentation

| City, sorted by deliveries | Average tip | Deliveries | Recorded min |
|---|---:|---:|---:|
| Frisco | $7.28 | 794 | 41.9 |
| Plano | $6.63 | 348 | 36.5 |
| The Colony | $7.09 | 69 | 40.2 |
| Carrollton | $7.36 | 16* | 45.6 |
| Lewisville | $11.67 | 12* | 39.5 |
| McKinney | $8.91 | 10* | 52.5 |
| Allen | $6.50 | 2* | 48.0 |

Use blue `#005FED`, navy `#091F3B`, slate `#475569`, white panels, background
`#F9F9FA`, and fine rules `#EEF2F7`. Small-sample cities can use pale blue
`#D8E9FF`, with the asterisk/tooltip carrying the meaning without color reliance.
Do not add a city benchmark line; the approved concept removed it.

Suggested desktop overview dimensions: 1500 × 1060. Header about 110 px, filters
72 px, KPI row 130 px, main comparison row 385 px, trend row 275 px. A phone view
needs a separate stacked layout rather than shrinking the desktop canvas.

Source footer: **Historical delivery records: Jul 5, 2017–Sep 23, 2018. First and
last months are partial. Recorded time includes 34 zero-minute entries.**

## Native validation still required

- Open the packaged workbook and resolve any calculation or rendering errors.
- Confirm all-data KPI values and every city row against the table above.
- City = Frisco should show 794 records and $5,777.68 total tips.
- Housing = Hotel should show 67 records; no housing category is missing.
- Clear filters/revert and return to 1,251 records.
- Confirm all 15 actual months are included, especially September 2018.
- Confirm the $40 tip and 135-minute duration are retained in their histograms.
- Verify date, city, housing, and city-click filtering together; each control
  must update all the views it claims to control.
- Verify desktop, narrow browser, and phone layouts before overwriting the live
  published workbook.

Tableau Public's documented browser file import accepts data, not TWB/TWBX
workbooks. Cross-workbook copy/import is Desktop-only. The supported packaged-file
publishing route is Tableau Desktop Public Edition → Server → Tableau Public →
Save to Tableau Public. The existing browser workbook can instead be rebuilt
through its authoring interface. See the official
[copy/import](https://help.tableau.com/current/pro/desktop/en-us/copy_b_wkbks.htm)
and [Public publishing](https://help.tableau.com/current/pro/desktop/en-us/publish_workbooks_tableaupublic.htm)
documentation.
