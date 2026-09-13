# Pizza Weekday and Estimate implementation specification

Validated September 12, 2026. This file is a handoff specification; it does not modify the native workbook.

## Weekday

Preserve the existing fact table and all 1,251 rows. Do not join to weather or ZIP tables. OrderNum is not unique, so use COUNT([Tip]) or SUM([Record count]) with Record count = 1, not COUNTD([OrderNum]). All source tips are populated. Keep all seven weekdays, ordered Monday to Sunday, rather than sorting by the performance measure.

Calculated fields (using the source field captions):

```tableau
// Weekday order
DATEPART('weekday', [Date], 'monday')

// Weekday name (sort ascending by Weekday order)
DATENAME('weekday', [Date])

// Delivery records
COUNT([Tip])

// Average tip
AVG([Tip])

// Median tip
MEDIAN([Tip])

// Total tips
SUM([Tip])

// Weighted tip rate: use this wording, not an average of per-order percentages
SUM([Tip]) / SUM([Cost])

// Small sample
COUNT([Tip]) < 30
```

Existing source elapsed-minute calculations can supply average recorded minutes. The measure is order-to-delivery elapsed time, not driver work time. The all-data definition includes 34 zero-minute records. No tips-per-hour, demand, profit, or shift-efficiency claim is supported by this extract.

The essential main chart is weekday average tip with visibly aligned delivery count and median; totals or weighted rate can be secondary measures. Avoid calling Wednesday the best shift: its larger average order cost and only 45 records are important context. About-only caveat for this revision: **Counts reflect recorded workdays, not demand. Fewer than 30 deliveries: compare cautiously.**

| Weekday | Records | Mean tip | Median tip | Total tips | Weighted tip rate | Avg recorded min | Recorded dates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Monday | 85 | $7.83 | $6.00 | $665.82 | 19.00% | 42.2 | 13 |
| Tuesday | 17* | $7.50 | $6.60 | $127.46 | 18.87% | 41.6 | 3 |
| Wednesday | 45 | $8.07 | $6.00 | $363.04 | 15.63% | 35.6 | 8 |
| Thursday | 6* | $6.75 | $5.00 | $40.47 | 12.65% | 52.2 | 2 |
| Friday | 367 | $7.47 | $6.00 | $2,741.04 | 17.39% | 41.2 | 53 |
| Saturday | 334 | $7.18 | $6.00 | $2,398.14 | 16.82% | 39.5 | 53 |
| Sunday | 397 | $6.54 | $5.12 | $2,597.52 | 16.86% | 40.4 | 54 |

Exact unrounded values: `weekday-performance.csv`. The weekday record and tip totals reconcile to the existing overview.

## Estimate

### Selected model and criteria

Use the exact published Tableau extract, not the repository's separate geocoded estimator dataset. A regularized linear model using **order cost alone** had the lowest expanding-validation mean absolute error. Adding city, housing, weekday, and cyclic hour terms did not improve that validation result. This is a model-selection result, not a claim that those factors never matter.

Do not add controls that imply they alter the prediction unless using a separately specified and validated model. City and housing may appear only as explicitly labeled historical comparison criteria, separate from the prediction. Delivery Time and Total Delivery Time are unavailable before delivery; Tip Percentage directly contains the outcome; neither belongs in the estimate model. OrderNum is also excluded.

Native parameter:

- Internal/display name: **Order cost ($)**; float; default **40.00**.
- Range **6.17 to 243.02**, step **0.01**. This is the training range, not a guarantee of dense coverage. Only 22 training orders cost more than $100.
- Currency format, two decimal places. Keep input cost distinct from tip-inclusive total; source Cost is the order amount used in Tip/Cost.

Native calculated fields:

```tableau
// Tip estimate
MAX(0, 2.7328850006632157 + 0.10518244598906885 * [Order cost ($)])

// Estimate range low
MAX(0, [Tip estimate] - 3.5124065373462514)

// Estimate range high
[Tip estimate] + 3.5124065373462514
```

Render these parameter-only fields as **MIN**, never SUM. Keep the estimator sheets outside date/city/housing filters and dashboard selection actions. A parameter-only estimate must not disappear when a historical comparison has no rows.

At the default $40 cost: **$6.94** estimate; **$3.43–$10.45** range. Label the range **Approx. 80% range**, not a confidence interval, guarantee, or the full observed range. Point estimates and bounds use full precision internally and round only for display. Numeric parity against the trained model was verified on every source record.

Main tab copy can stay minimal:

- Title: **Estimate a tip**
- Input: **Order cost ($)**
- Output: **Estimated tip**
- Interval: **Approx. 80% range**
- Put the context **Based on historical deliveries; individual tips vary** in About, per the user's latest request to consolidate caveats there.

### About text

**The estimator uses order cost in a simple regression, trained on 751 early deliveries and checked on later records. Its average absolute error on the latest 251 deliveries was $2.62, compared with $3.06 for always estimating $6. A separate 249-record period set the range; it contained 203 of 251 later tips (80.9%). Other recorded criteria did not improve model-selection results. These personal records are historical, and the range is not guaranteed for future orders.**

Do not imply that the model was fitted on all 1,251 rows. All rows remain in dashboard statistics; 751 fit the frozen model, 249 calibrate its interval, and 251 validate it independently.

### Validation detail

Chronological partitions use whole dates, with no same-date mixing:

| Role | Records | Period |
|---|---:|---|
| Fit and internal model selection | 751 | Jul 5, 2017–Mar 18, 2018 |
| Range calibration | 249 | Mar 23–Jun 24, 2018 |
| Final holdout | 251 | Jun 25–Sep 23, 2018 |

Eight pre-tip feature sets and six ridge strengths were compared only within the first partition using three expanding date-blocked folds. The selection rule prefers fewer criteria within 1% of the best MAE; order cost was both simplest and numerically best. Cost is standardized using the fitting partition's mean and population standard deviation; the intercept is unpenalized. Selected ridge alpha = 0.1. Raw-scale coefficients are provided above.

The frozen interval half-width is the 200th smallest absolute residual among 249 calibration records, corresponding to an approximately 80% split-calibration target. With changing historical schedules, this is an empirical range, not a coverage guarantee.

| Final holdout measure | Value |
|---|---:|
| Mean absolute error | $2.61692664 |
| Median absolute error | $1.85785547 |
| RMSE | $4.09670626 |
| R-squared | 0.29316142 |
| Actual minus prediction, mean | +$0.21003199 |
| Actual minus prediction, min/max | −$7.05310154 / +$20.95759355 |
| Interval coverage | 203/251 = 80.8765% |
| Baseline always $6, MAE | $3.05944223 |
| Baseline training mean $7.125486, MAE | $3.29173829 |
| MAE reduction versus $6 baseline | 14.4639% |

Holdout results are descriptive for the recorded later period. They are not proof of current or universal prediction accuracy. Subgroup errors and small supports are preserved in `estimator-test-subgroups.csv`; no category-specific accuracy claim is warranted for one to five holdout rows.

### References and reproducibility

- Authoritative extract: `source-tips.csv`, extracted from `Pizza_Delivery.twbx` downloaded for the original audit; source URL: https://public.tableau.com/workbooks/Pizza_Delivery.twb?showVizHome=no.
- `analyze-weekday-estimator.py`: complete reproducible computation and model selection.
- `weekday-estimator-analysis.json`: source hash, quality checks, fold dates, partition support, metrics, and results.
- `estimator-model.json`: native coefficients and interval constants.
- `candidate-validation.csv`: all candidate model-selection results.
- `estimator-*-scored.csv`: optional exports from `--include-scored`; not duplicated in this package.
- `estimator-parameter-examples.csv`: reproducible native input/output examples and local cost support.
- `README.md`: portable reproduction command and artifact provenance; the external notebook is not duplicated here.
- [Tableau date-function documentation](https://help.tableau.com/current/pro/desktop/en-us/functions_functions_date.htm) documents explicit week starts.
- [Official Ridge reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) defines the squared-error plus L2-penalty objective; the implementation here uses NumPy, not an installed scikit-learn dependency.

Overall assessment: **Share with caveats**. Weekday arithmetic and model implementation are verified. The main limitations are historical schedule bias, sparse subgroup support, wide individual-tip errors, and unguaranteed future interval coverage.
