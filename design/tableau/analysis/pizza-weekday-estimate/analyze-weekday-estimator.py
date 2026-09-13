"""Reproducible weekday metrics and leak-free native Tableau estimator research.

Run with the bundled Python (numpy and pandas). No source files are modified.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).with_name('source-tips.csv'))
parser.add_argument('--output', type=Path, required=True, help='Directory for regenerated proof outputs; use a separate scratch directory.')
parser.add_argument('--include-scored', action='store_true', help='Also export row-level scored CSVs; omitted by default.')
args = parser.parse_args()
SOURCE = args.source.resolve()
OUT = args.output.resolve()
if OUT == Path(__file__).resolve().parent:
  parser.error('--output must be separate from the reviewed artifact directory')
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != 'c09bec697c296e0c0f34f160902fe10521972968dfa6e7296648695e1ae0e753':
  parser.error('Source CSV hash differs from the audited Tableau extract')
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SOURCE)
original = df.copy()
df['source_row'] = np.arange(1, len(df) + 1)
df['date'] = pd.to_datetime(df['Date'])
df['weekday_number'] = df['date'].dt.dayofweek + 1
df['weekday'] = df['date'].dt.day_name()
df['hour'] = pd.to_datetime(df['Order Time']).dt.hour
df['minutes'] = df['Total Delivery Time'].map(lambda t: int(t.split(':')[0]) * 60 + int(t.split(':')[1]))
df['timestamp'] = df['date'] + pd.to_timedelta(pd.to_datetime(df['Order Time']).dt.strftime('%H:%M:%S'))
df = df.sort_values(['timestamp', 'source_row'], kind='stable').reset_index(drop=True)
assert len(df) == 1251 and not original.isna().any().any()
assert not original.duplicated().any()
assert abs(df['Tip'].sum() - 8933.49) < 1e-8

weekday = df.groupby(['weekday_number', 'weekday']).agg(
  records=('Tip', 'size'), mean_tip=('Tip', 'mean'), median_tip=('Tip', 'median'),
  total_tips=('Tip', 'sum'), mean_order_cost=('Cost', 'mean'), total_order_cost=('Cost', 'sum'),
  mean_recorded_minutes=('minutes', 'mean'), median_recorded_minutes=('minutes', 'median'),
  recorded_dates=('date', 'nunique'), zero_tip_records=('Tip', lambda s: int(s.eq(0).sum())),
  zero_duration_records=('minutes', lambda s: int(s.eq(0).sum()))).reset_index()
weekday['weighted_tip_rate'] = weekday['total_tips'] / weekday['total_order_cost']
weekday['share_of_records'] = weekday['records'] / len(df)
weekday['fewer_than_30_records'] = weekday['records'] < 30
weekday.to_csv(OUT / 'weekday-performance.csv', index=False, float_format='%.10f')
assert weekday['records'].sum() == 1251
assert abs(weekday['total_tips'].sum() - 8933.49) < 1e-8

def split_at_fraction(data, fraction):
  # Select a date boundary close to desired row fraction, never split a date.
  counts = data.groupby('date').size().cumsum()
  eligible = counts.iloc[:-1]
  cutoff = (eligible - len(data) * fraction).abs().idxmin()
  return data[data['date'] <= cutoff].copy(), data[data['date'] > cutoff].copy()

train, future = split_at_fraction(df, .60)
calibration, test = split_at_fraction(future, .5)
assert train.date.max() < calibration.date.min() <= calibration.date.max() < test.date.min()

FEATURE_SETS = {
  'cost': [],
  'cost_housing': ['Housing'],
  'cost_city_housing': ['City', 'Housing'],
  'cost_housing_weekday': ['Housing', 'weekday'],
  'cost_city_housing_weekday': ['City', 'Housing', 'weekday'],
  'cost_housing_hour': ['Housing', 'hour_cycle'],
  'cost_city_housing_hour': ['City', 'Housing', 'hour_cycle'],
  'cost_city_housing_weekday_hour': ['City', 'Housing', 'weekday', 'hour_cycle'],
}

def design(data, spec=None, columns=None):
  if spec is None:
    spec = {'cost_mean': float(data.Cost.mean()), 'cost_sd': float(data.Cost.std(ddof=0)), 'categories': {}}
    for column in columns:
      if column != 'hour_cycle':
        spec['categories'][column] = sorted(data[column].unique())
    spec['hour_cycle'] = 'hour_cycle' in columns
  parts = [np.ones(len(data)), (data.Cost.to_numpy() - spec['cost_mean']) / spec['cost_sd']]
  names = ['intercept', 'cost_standardized']
  for column, categories in spec['categories'].items():
    for category in categories:
      parts.append(data[column].eq(category).to_numpy(dtype=float))
      names.append(f'{column}={category}')
  if spec['hour_cycle']:
    parts.extend([np.sin(2*np.pi*data.hour.to_numpy()/24), np.cos(2*np.pi*data.hour.to_numpy()/24)])
    names.extend(['hour_sin', 'hour_cos'])
  return np.column_stack(parts), names, spec

def fit(data, feature_set, alpha):
  X, names, spec = design(data, columns=FEATURE_SETS[feature_set])
  penalty = np.eye(X.shape[1]) * alpha
  penalty[0, 0] = 0
  beta = np.linalg.solve(X.T @ X + penalty, X.T @ data.Tip.to_numpy())
  return {'feature_set': feature_set, 'alpha': alpha, 'spec': spec, 'names': names, 'beta': beta.tolist()}

def predict(data, model):
  X, _, _ = design(data, spec=model['spec'])
  return np.maximum(0., X @ np.asarray(model['beta']))

def metrics(truth, pred):
  truth = np.asarray(truth)
  resid = truth - pred
  total_squared_deviation = np.sum((truth-truth.mean())**2)
  return {'n': len(truth), 'mae': float(np.mean(np.abs(resid))),
      'median_absolute_error': float(np.median(np.abs(resid))),
      'rmse': float(np.sqrt(np.mean(resid**2))),
      'mean_residual_actual_minus_predicted': float(np.mean(resid)),
      'r2': float(1-np.sum(resid**2)/total_squared_deviation) if total_squared_deviation > 0 else None,
      'residual_min': float(resid.min()), 'residual_max': float(resid.max())}

# Expanding, date-blocked validation entirely inside the first 60% training block.
cv_folds = []
for start, end in [(.4, .6), (.6, .8), (.8, 1.)]:
  fold_train, tail = split_at_fraction(train, start)
  if end < 1.:
    through_end, _ = split_at_fraction(train, end)
    fold_valid = through_end[through_end.date > fold_train.date.max()].copy()
  else:
    fold_valid = tail
  cv_folds.append((fold_train, fold_valid))

scores = []
for feature_set in FEATURE_SETS:
  for alpha in [0.1, 1., 10., 30., 100., 300.]:
    truth, pred = [], []
    for a, b in cv_folds:
      model = fit(a, feature_set, alpha)
      truth.extend(b.Tip)
      pred.extend(predict(b, model))
    score = metrics(truth, np.array(pred))
    score.update(feature_set=feature_set, alpha=alpha, criteria_count=1+len(FEATURE_SETS[feature_set]))
    scores.append(score)
scores.sort(key=lambda row: row['mae'])
pd.DataFrame(scores).to_csv(OUT / 'candidate-validation.csv', index=False)
# Choose the least-complex candidate within 1% of the best CV MAE.
tolerance = scores[0]['mae'] * 1.01
eligible = [s for s in scores if s['mae'] <= tolerance]
chosen = sorted(eligible, key=lambda s: (s['criteria_count'], s['mae']))[0]
model = fit(train, chosen['feature_set'], chosen['alpha'])
train_prediction = predict(train, model)
cal_prediction = predict(calibration, model)
test_prediction = predict(test, model)

# A frozen split-calibration range. Quantile is chosen before examining test outcomes.
# Historical time drift precludes a guaranteed coverage claim.
cal_abs = np.sort(np.abs(calibration.Tip.to_numpy() - cal_prediction))
rank = min(len(cal_abs), math.ceil((len(cal_abs) + 1) * .80))
half_width = float(cal_abs[rank - 1])
test_lower = np.maximum(0, test_prediction - half_width)
test_upper = test_prediction + half_width
coverage = float(np.mean((test.Tip >= test_lower) & (test.Tip <= test_upper)))

baselines = {}
for name, value in [('training_median', float(train.Tip.median())), ('training_mean', float(train.Tip.mean()))]:
  baselines[name] = {'value': value, **metrics(test.Tip, np.full(len(test), value))}

coefficient = dict(zip(model['names'], model['beta']))
cost_slope = coefficient.pop('cost_standardized') / model['spec']['cost_sd']
coefficient['intercept'] -= cost_slope * model['spec']['cost_mean']
coefficient['Cost'] = cost_slope
model['raw_scale_coefficients'] = coefficient
model['range_half_width'] = half_width
model['calibration_target'] = .80
model['calibration_rank'] = rank
model['fit_rows'] = len(train)
model['calibration_rows'] = len(calibration)
model['test_rows'] = len(test)
model['test_metrics'] = metrics(test.Tip, test_prediction)
model['test_range_coverage'] = coverage
model['baselines'] = baselines
model['mae_reduction_vs_median_baseline'] = 1-model['test_metrics']['mae']/baselines['training_median']['mae']
model['test_range_inside_count'] = int(np.sum((test.Tip >= test_lower) & (test.Tip <= test_upper)))
model['test_range_outside_count'] = len(test)-model['test_range_inside_count']

# Native-formula parity on every row, not rounded display values.
native_prediction = np.maximum(0, coefficient['intercept'] + coefficient['Cost'] * df.Cost.to_numpy())
assert np.max(np.abs(native_prediction-predict(df,model))) < 1e-10
model['native_formula_max_absolute_difference'] = float(np.max(np.abs(native_prediction-predict(df,model))))
parameter_examples = []
for cost in [6.17, 10, 20, 30, 40, 50, 75, 100, 150, 200, 243.02]:
  tip = max(0, coefficient['intercept'] + coefficient['Cost']*cost)
  parameter_examples.append({'order_cost':cost,'estimate':tip,'range_low':max(0,tip-half_width),
                               'range_high':tip+half_width,'training_records_within_10_dollars':int(train.Cost.between(cost-10,cost+10).sum()),
                               'full_records_within_10_dollars':int(df.Cost.between(cost-10,cost+10).sum())})
pd.DataFrame(parameter_examples).to_csv(OUT/'estimator-parameter-examples.csv',index=False,float_format='%.10f')

def profile(data):
  return {'rows': len(data), 'first_date': str(data.date.min().date()), 'last_date': str(data.date.max().date()),
      'unique_dates': data.date.nunique(), 'cost_min': data.Cost.min(), 'cost_max': data.Cost.max(),
      'tip_min': data.Tip.min(), 'tip_max': data.Tip.max(), 'mean_tip': data.Tip.mean(),
      'city_counts': data.City.value_counts().to_dict(), 'housing_counts': data.Housing.value_counts().to_dict(),
      'weekday_counts': data.weekday.value_counts().to_dict(), 'hour_counts': data.hour.value_counts().to_dict()}

profiles = {k: profile(v) for k,v in [('full',df), ('train',train), ('calibration',calibration), ('test',test)]}
if args.include_scored:
  for split, data, predictions in [('train',train,train_prediction),('calibration',calibration,cal_prediction),('test',test,test_prediction)]:
    scored = data.copy()
    scored['split'] = split
    scored['predicted_tip'] = predictions
    scored['residual_actual_minus_predicted'] = scored.Tip - predictions
    scored['range_low'] = np.maximum(0, predictions-half_width)
    scored['range_high'] = predictions+half_width
    scored.to_csv(OUT / f'estimator-{split}-scored.csv', index=False, float_format='%.10f')

subgroups = []
for column in ['City','Housing','weekday']:
  for value, g in test.groupby(column):
    subgroups.append({'field':column,'value':value, **metrics(g.Tip,predict(g,model))})
pd.DataFrame(subgroups).to_csv(OUT/'estimator-test-subgroups.csv',index=False)
support = df.groupby(['City','Housing']).agg(n=('Tip','size'),cost_min=('Cost','min'),cost_max=('Cost','max')).reset_index()
support.to_csv(OUT/'estimator-city-housing-support.csv',index=False)
report = {'source': SOURCE.name, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'grain':'One source delivery row; do not deduplicate OrderNum.',
          'source_quality': {'rows':len(df),'complete_duplicate_rows':int(original.duplicated().sum()),
          'duplicate_order_ids': int(original.OrderNum.duplicated().sum()), 'missing_fields':original.isna().sum().to_dict(),
          'zero_duration_records':int(df.minutes.eq(0).sum()),'join_policy':'No weather or ZIP join'},
          'split_profiles':profiles, 'cv_folds':[{'train':profile(a),'valid':profile(b)} for a,b in cv_folds],
          'selection_policy':'Minimum criterion count within 1% of lowest expanding-validation MAE; then lowest MAE.',
          'selected_cv_score':chosen, 'best_cv_score':scores[0], 'model':model,
          'weekday':weekday.to_dict(orient='records'),
          'limitations':['Historical single-driver schedule, not market demand or a current tipping model.',
                         'Prediction is associational; tip outcomes vary widely.',
                         '80% calibration target is not guaranteed coverage under chronological drift.',
                         'Sparse cities/housing categories may not support detailed subgroup predictions.',
                         'Elapsed order-to-delivery minutes are not labor hours; no hourly earnings metric.']}

def json_default(value):
  if isinstance(value, np.generic): return value.item()
  raise TypeError(type(value).__name__)
(OUT/'weekday-estimator-analysis.json').write_text(json.dumps(report,indent=2,default=json_default),encoding='utf-8')
(OUT/'estimator-model.json').write_text(json.dumps(model,indent=2,default=json_default),encoding='utf-8')
print(json.dumps({'output':str(OUT),'source_sha256':report['source_sha256'],'rows':len(df),'selected_feature_set':model['feature_set'],'ridge_alpha':model['alpha'],'fit_calibration_test_rows':[len(train),len(calibration),len(test)],'holdout_mae':model['test_metrics']['mae'],'range_coverage':coverage,'native_formula_max_absolute_difference':model['native_formula_max_absolute_difference']},indent=2,default=json_default))
