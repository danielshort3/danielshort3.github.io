# Pizza Weekday and Tip estimate proof

This package preserves the approved analysis. The computation and the public checkpoint's source, model fields, and weekday definitions/tooltips are verified; final navigation, persistent filter placement, and desktop/phone layout checks remain pending. See [native implementation](native-implementation.md) for the actual field captions and parameter behavior. This is separate from the website's geocoded tip-predictor project.

`source-tips.csv` contains the 1,251 original Tableau delivery rows, with no weather/ZIP join or OrderNum deduplication. Its SHA-256 is `c09bec697c296e0c0f34f160902fe10521972968dfa6e7296648695e1ae0e753`. It was exported from the preserved original `../../sources/Pizza_Delivery.twbx`; the unchanged Hyper is `614f454d2f64b3de28ee7fae5d18eed3e712c9c474cf21a64e9323b72fbc5f52`.

## Contents

- `analyze-weekday-estimator.py`: original NumPy/pandas analysis with portable arguments, source-hash guard, separate output directory, and optional scored-row exports. Model mathematics is unchanged.
- `weekday-performance.csv`: seven Monday-to-Sunday rows; counts and tips reconcile to 1,251 and $8,933.49.
- `estimator-model.json`, `tableau-estimator-spec.json`: frozen model coefficients, interval constants, native parameter range, and examples.
- `candidate-validation.csv`, `estimator-test-subgroups.csv`, `estimator-city-housing-support.csv`, `estimator-parameter-examples.csv`: compact validation/support evidence.
- `weekday-estimator-analysis.json`: exact source quality, chronology, folds, partitions, selected model, and metrics.
- `tableau-weekday-estimator-spec.md`: native formulas and interpretation. Caveats are assigned to About under the current user request.
- [native-implementation.md](native-implementation.md): verified published captions, continuous parameter behavior, two model cards, weekday ordering, and precision/format checks. Original handoff and reproducible outputs remain intact.
- `provenance.json`: source artifact hashes and the portable packaging changes. Row-level scored CSVs and the notebook remain in the external research artifact; they are not duplicated here.

## Reproduce

Use Python with the NumPy/pandas versions in `requirements.txt` (the bundled runtime already has them). From the repository root:

```powershell
python design/tableau/analysis/pizza-weekday-estimate/analyze-weekday-estimator.py --output "C:/path/to/scratch/pizza-weekday-estimate"
```

The script reads the bundled input by default, requires a separate output directory, and rejects a changed source hash. `--source` accepts another byte-identical copy; `--include-scored` optionally writes every scored row. No workbook, website, or publication is modified. Reproduction results should match the reviewed model within floating-point tolerance; output source paths can differ.

## Validated method

Weekday is the recorded delivery date's weekday, Monday first. Essential comparisons are average tip, count, and median; weighted tip rate is SUM(Tip)/SUM(Cost). Counts reflect the recorded schedule, not market demand. The 34 zero-minute elapsed-time entries remain in time statistics.

The selected ridge model uses order cost alone, fitted on 751 early rows. A separate 249-row period calibrates an approximately 80% range, followed by a 251-row chronological holdout. No date is split between roles. City, housing, weekday, and cyclic hour candidates did not improve internal validation. Outcome-derived or post-delivery fields were excluded.

Native estimate: `MAX(0, 2.7328850006632157 + 0.10518244598906885 * cost)`. Range half-width: `3.5124065373462514`, lower bound clipped at zero. Display parameter-only fields with MIN, never SUM, outside historical filters and selection actions. At $40, the display is $6.94 with $3.43–$10.45 Approx. 80% range. Parameter bounds are $6.17–$243.02; support is sparse above $100.

Holdout MAE is $2.61692664 versus $3.05944223 for always estimating $6. The interval contains 203/251 holdout tips (80.8765%). These are historical results, not future guarantees. About must explain fit/calibration/test roles, individual variation, schedule bias, sparse support, and no causal/hourly-earnings interpretation.
