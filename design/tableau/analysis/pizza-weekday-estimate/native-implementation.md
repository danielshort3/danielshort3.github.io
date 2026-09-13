# Published native implementation

The final public workbook captured on **September 13, 2026 at 00:13:33 UTC** passed source/model checks and native desktop/Phone interaction review. All six destinations retain historical selections and consistent filter positions. This note records the implemented behavior separately from the original [handoff specification](tableau-weekday-estimator-spec.md); the reproducible JSON, CSV, and analysis script are unchanged.

## Parameter and calculations

| Native caption | Stored implementation |
|---|---|
| Order cost | FLOAT (`real`), range **6.17–243.02**, default **40**, currency with two decimals. A continuous Type In control appears in both device layouts. No step/granularity constraint is stored. |
| Tip estimate | `MAX(0, 2.732885000663216 + 0.105182445989069 * [Order cost])` |
| Tip range low | `MAX(0, [Tip estimate] - 3.512406537346251)` |
| Tip range high | `[Tip estimate] + 3.512406537346251` |
| Tip interval label | A string calculation formats both bounds to exactly two decimal places, joined by ` to `. At order cost $40 it displays **$3.43 to $10.45**. |

The formulas above use friendly captions. The stored XML refers to the Order cost parameter and calculation IDs. Tableau's serialized decimal coefficients differ slightly from the original full-precision handoff. Comparing the actual published formula with the approved model across all **1,251 source rows** gives a maximum absolute difference of **3.907985046680551e-14** (about 3.91e-14).

The handoff called the parameter `Order cost ($)` and proposed a $0.01 step. The implemented caption is `Order cost`; its currency format shows cents, while the continuous input has no enforced cent step. The handoff's `Estimate range low/high` calculations are named `Tip range low/high` in Tableau.

The exact interval string calculation, expressed with friendly captions, is:

```text
"$"+STR(INT(ROUND([Tip range low]*100,0)/100))
+"."+RIGHT("0"+STR(INT(ROUND([Tip range low]*100,0)) % 100),2)
+" to $"+STR(INT(ROUND([Tip range high]*100,0)/100))
+"."+RIGHT("0"+STR(INT(ROUND([Tip range high]*100,0)) % 100),2)
```

The independent public-package audit checked **24,937 cost cases** and confirmed two-decimal interval formatting in every case. At the default $40, the estimate is 6.940182840225976 and displays **$6.94**. At the parameter bounds, the range labels are **$0.00 to $6.89** and **$24.78 to $31.81**. These are computation checks, not proof of rendered clipping or future prediction coverage.

## Presented worksheets

| Visible model card | Native text encoding |
|---|---|
| Estimated tip | `MIN([Tip estimate])`, numeric, currency with two decimals |
| Approx. 80% range | `MIN([Tip interval label])`, string |

The unused `Estimate upper range` helper remains in the workbook but is not presented. Both visible cards have tooltips disabled, contain no historical filters, and are outside the published selection-action targets. Their prediction uses order cost alone. Native interaction confirmed that a Lewisville + Hotel selection with no historical rows leaves the $40 estimate at **$6.94 / $3.43 to $10.45**. Native minimum, maximum and $80 entries produced the expected two-decimal results; Revert restored cost $40. The complete six-tab Phone cycle retained January 1–September 23, 2018 / Frisco / Hotel while preserving the estimate. The compact estimator Phone canvas is 1240px high, with a 390 x 1267 document including Tableau controls.

Weekday is `DATENAME("weekday", [Date])`, explicitly sorted **Monday → Tuesday → Wednesday → Thursday → Friday → Saturday → Sunday**. The four charts use:

| Worksheet | Aggregation |
|---|---|
| Weekday average tip | `AVG([Tip])` |
| Weekday median tip | `MEDIAN([Tip])` |
| Weekday total tips | `SUM([Tip])` |
| Weekday deliveries | `COUNT([Tip])` |

All four weekday worksheets share the historical Date/City/Housing filter groups. An explicit City-only Select/clear-all action also carries Overview city-mark focus to all four charts; model cards remain excluded. Their custom tooltip definitions use friendly labels and valid field references. All seven default rows matched source values in the native view. January 1–September 23, 2018 / Frisco / Hotel yielded seven records each on Friday, Saturday and Sunday, totaling 21 / $136.25. The Phone charts passed complete scrolling/readability review without horizontal overflow. The source extract is unchanged: SHA-256 `614f454d2f64b3de28ee7fae5d18eed3e712c9c474cf21a64e9323b72fbc5f52`.

Evidence: [final independent acceptance](C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-final-independent-acceptance.json), [native interaction QA](C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-feedback-native-interaction-qa.json), [public model/weekday audit](C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/pizza-public-model-layout-audit.json), [audit script](C:/Users/clopt/.codex/visualizations/2026/09/11/01a08eca-052b-70a3-96da-6b1df4d11ada/tableau-redesign/feedback-revision/audit-pizza-public-model-layout.py).

The [canonical manifest](../../published/pizza-delivery-published-manifest.json) records package SHA-256 `f679d226b2096fb3b09c7c0c9c1e8018cade063c0a96f68a316ddb0304e19919`; the embedded TWB is `a7e2e652e1285b0418b1d503a653e0d6e3faf5eedfa2e35de2225468c772738a`. A fresh independent download has identical uncompressed TWB/Hyper members, though ZIP metadata changes its package hash. All **265 Phone structural assertions** passed. Structural checks supplement the successful native interaction/readability review; they do not replace it. **No website deployment is claimed.**
