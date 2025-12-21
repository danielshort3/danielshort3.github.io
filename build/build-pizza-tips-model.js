const fs = require('fs');
const path = require('path');

const INPUT_PATH = path.join(__dirname, '..', 'tips_data_geocoded.csv');
const OUTPUT_DIR = path.join(__dirname, '..', 'aws', 'pizza-tips-predict');
const OUTPUT_PATH = path.join(OUTPUT_DIR, 'model.json');
const BOUNDARY_PATH = path.join(OUTPUT_DIR, 'city-boundaries.json');
const META_PATH = path.join(__dirname, '..', 'js', 'demos', 'pizza-tips-meta.js');

const BASE_FEATURES = [
  { key: 'cost', label: 'Order Cost ($)' },
  { key: 'orderHour', label: 'Order Hour (24h)' },
  { key: 'deliveryMinutes', label: 'Delivery Wait (Minutes)' },
  { key: 'rain', label: 'Rain (Inches)' },
  { key: 'maxTemp', label: 'Max Temp (F)' },
  { key: 'minTemp', label: 'Min Temp (F)' }
];
const SIGNIFICANCE_LEVEL = Number(process.env.SIGNIFICANCE_LEVEL || '0.05');
const TARGET_TRANSFORMS = {
  tip: 'log1p',
  tipPercent: 'log1p'
};

function transformTarget(value, transform) {
  if (!Number.isFinite(value)) return null;
  if (transform === 'log1p') {
    return Math.log1p(value);
  }
  return value;
}

function parseCSV(text) {
  const rows = [];
  let row = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      continue;
    }
    if (ch === ',') {
      row.push(cur);
      cur = '';
      continue;
    }
    if (ch === '\n') {
      row.push(cur);
      rows.push(row);
      row = [];
      cur = '';
      continue;
    }
    if (ch === '\r') continue;
    cur += ch;
  }
  if (cur.length || row.length) {
    row.push(cur);
    rows.push(row);
  }
  return rows;
}

function toNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function parseHour(value) {
  if (!value) return null;
  const parts = String(value).trim().split(':');
  if (!parts.length) return null;
  const h = Number(parts[0]);
  return Number.isFinite(h) ? h : null;
}

function parseDurationMinutes(value) {
  if (!value) return null;
  const parts = String(value).trim().split(':');
  if (parts.length < 2) return null;
  const h = Number(parts[0]);
  const m = Number(parts[1]);
  const s = Number(parts[2] || 0);
  if (![h, m, s].every(n => Number.isFinite(n))) return null;
  return h * 60 + m + s / 60;
}

function normalizeCategory(value) {
  return String(value || '').trim();
}

function transpose(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const out = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j][i] = matrix[i][j];
    }
  }
  return out;
}

function matMul(A, B) {
  const rows = A.length;
  const cols = B[0].length;
  const mid = B.length;
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < mid; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cols; j++) {
        out[i][j] += aik * B[k][j];
      }
    }
  }
  return out;
}

function matVecMul(A, v) {
  return A.map(row => row.reduce((sum, val, idx) => sum + val * v[idx], 0));
}

function solveLinear(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) {
        maxRow = k;
      }
    }
    if (maxRow !== i) {
      [M[i], M[maxRow]] = [M[maxRow], M[i]];
    }
    const pivot = M[i][i];
    if (pivot === 0) throw new Error('Singular matrix');
    for (let j = i; j < n + 1; j++) {
      M[i][j] /= pivot;
    }
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = M[k][i];
      for (let j = i; j < n + 1; j++) {
        M[k][j] -= factor * M[i][j];
      }
    }
  }
  return M.map(row => row[n]);
}

function invertMatrix(A) {
  const n = A.length;
  const inv = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const e = Array(n).fill(0);
    e[i] = 1;
    const col = solveLinear(A, e);
    for (let r = 0; r < n; r++) {
      inv[r][i] = col[r];
    }
  }
  return inv;
}

function logGamma(z) {
  const cof = [
    57.1562356658629235,
    -59.5979603554754912,
    14.1360979747417471,
    -0.491913816097620199,
    0.339946499848118887e-4,
    0.465236289270485756e-4,
    -0.983744753048795646e-4,
    0.158088703224912494e-3,
    -0.210264441724104883e-3,
    0.217439618115212643e-3,
    -0.16431810653676389e-3,
    0.844182239838527433e-4,
    -0.261908384015814087e-4,
    0.368991826595316234e-5
  ];
  let x = 0.999999999999997092;
  let y = z;
  for (let j = 0; j < cof.length; j++) {
    y += 1;
    x += cof[j] / y;
  }
  const t = z + 5.2421875;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x / z);
}

function betacf(a, b, x) {
  const MAX_ITER = 200;
  const EPS = 3e-7;
  const FPMIN = 1e-30;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAX_ITER; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function betaIncomplete(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) {
    return bt * betacf(a, b, x) / a;
  }
  return 1 - bt * betacf(b, a, 1 - x) / b;
}

function tCdf(t, df) {
  if (!Number.isFinite(t)) return 1;
  if (df <= 0) return 0.5;
  const x = df / (df + t * t);
  const ib = betaIncomplete(df / 2, 0.5, x);
  if (t >= 0) {
    return 1 - 0.5 * ib;
  }
  return 0.5 * ib;
}

function buildMatrix(data, featureKeys) {
  return data.map((row) => {
    const values = [1];
    featureKeys.forEach((key) => {
      const val = row.values[key];
      values.push(Number.isFinite(val) ? val : 0);
    });
    return values;
  });
}

function fitModel(data, featureKeys, targetKey) {
  const X = buildMatrix(data, featureKeys);
  const y = data.map(d => d[targetKey]);
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  const Xty = matVecMul(Xt, y);
  const coeffs = solveLinear(XtX, Xty);
  const preds = X.map(row => row.reduce((sum, val, idx) => sum + val * coeffs[idx], 0));
  const meanY = y.reduce((a, b) => a + b, 0) / y.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < y.length; i++) {
    const err = y[i] - preds[i];
    ssRes += err * err;
    const dev = y[i] - meanY;
    ssTot += dev * dev;
  }
  const rmse = Math.sqrt(ssRes / y.length);
  const r2 = ssTot ? 1 - ssRes / ssTot : 0;
  const df = y.length - coeffs.length;
  const pValues = Array(coeffs.length).fill(1);
  if (df > 0) {
    const invXtX = invertMatrix(XtX);
    const sigma2 = ssRes / df;
    for (let i = 0; i < coeffs.length; i++) {
      const se = Math.sqrt(sigma2 * invXtX[i][i]);
      const t = se > 0 ? coeffs[i] / se : 0;
      const p = 2 * (1 - tCdf(Math.abs(t), df));
      pValues[i] = Number.isFinite(p) ? p : 1;
    }
  }
  return { coeffs, rmse, r2, n: y.length, df, pValues };
}

function selectFeatures(data, featureKeys, targetKey, alpha) {
  let current = [...featureKeys];
  const removed = [];
  while (true) {
    const stats = fitModel(data, current, targetKey);
    if (!current.length || stats.df <= 0) {
      return { featureKeys: current, model: stats, removed };
    }
    let worstKey = null;
    let worstP = -1;
    current.forEach((key, idx) => {
      const p = stats.pValues[idx + 1] ?? 1;
      if (p > worstP) {
        worstP = p;
        worstKey = key;
      }
    });
    if (worstP <= alpha) {
      return { featureKeys: current, model: stats, removed };
    }
    current = current.filter(key => key !== worstKey);
    removed.push({ key: worstKey, p: worstP });
  }
}

function summarizeStats(values) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const val of values) {
    min = Math.min(min, val);
    max = Math.max(max, val);
    sum += val;
  }
  return {
    min,
    max,
    mean: values.length ? sum / values.length : 0
  };
}

const hasInput = fs.existsSync(INPUT_PATH);
const hasBoundary = fs.existsSync(BOUNDARY_PATH);
const hasOutputs = fs.existsSync(OUTPUT_PATH) && fs.existsSync(META_PATH);

if (!hasInput) {
  if (hasBoundary && hasOutputs) {
    console.warn(`Missing input CSV at ${INPUT_PATH}. Using existing model artifacts.`);
    process.exit(0);
  }
  if (!hasBoundary) {
    throw new Error(`Missing city boundaries at ${BOUNDARY_PATH}`);
  }
  throw new Error(`Missing input CSV at ${INPUT_PATH}`);
}
if (!hasBoundary) {
  throw new Error(`Missing city boundaries at ${BOUNDARY_PATH}`);
}

const csvText = fs.readFileSync(INPUT_PATH, 'utf8');
const rows = parseCSV(csvText);
const header = rows[0].map(h => h.replace(/^\ufeff/, '').trim());
const idx = Object.fromEntries(header.map((h, i) => [h, i]));
const cityBoundaries = JSON.parse(fs.readFileSync(BOUNDARY_PATH, 'utf8'));

const latValues = [];
const lonValues = [];
const costValues = [];
const tipValues = [];
const tipPctValues = [];
const hourValues = [];
const deliveryValues = [];
const rainValues = [];
const maxTempValues = [];
const minTempValues = [];
const cityStats = {};
const housingCounts = {};
const rowsData = [];

for (let i = 1; i < rows.length; i++) {
  const r = rows[i];
  const cost = toNumber(r[idx['Cost']]);
  const tip = toNumber(r[idx['Tip']]);
  const tipPct = toNumber(r[idx['Tip Percentage']]);
  const hour = parseHour(r[idx['Order Time']]);
  const deliveryMin = parseDurationMinutes(r[idx['Total Delivery Time']]);
  const lat = toNumber(r[idx['Latitude']]);
  const lon = toNumber(r[idx['Longitude']]);
  const rain = toNumber(r[idx['Rain (Inches)']]);
  const tmax = toNumber(r[idx['Max Temp']]);
  const tmin = toNumber(r[idx['Min Temp']]);
  const city = normalizeCategory(r[idx['City']]);
  const housing = normalizeCategory(r[idx['Housing']]);
  if ([cost, tip, tipPct, hour, deliveryMin, lat, lon, rain, tmax, tmin].some(v => v === null)) {
    continue;
  }
  const tipTransformed = transformTarget(tip, TARGET_TRANSFORMS.tip);
  const tipPctTransformed = transformTarget(tipPct, TARGET_TRANSFORMS.tipPercent);
  if (!Number.isFinite(tipTransformed) || !Number.isFinite(tipPctTransformed)) {
    continue;
  }
  if (!city || !housing) continue;
  costValues.push(cost);
  tipValues.push(tip);
  tipPctValues.push(tipPct);
  hourValues.push(hour);
  deliveryValues.push(deliveryMin);
  latValues.push(lat);
  lonValues.push(lon);
  rainValues.push(rain);
  maxTempValues.push(tmax);
  minTempValues.push(tmin);

  if (!cityStats[city]) {
    cityStats[city] = {
      count: 0,
      latSum: 0,
      lonSum: 0,
      minLat: lat,
      maxLat: lat,
      minLon: lon,
      maxLon: lon
    };
  }
  cityStats[city].count += 1;
  cityStats[city].latSum += lat;
  cityStats[city].lonSum += lon;
  cityStats[city].minLat = Math.min(cityStats[city].minLat, lat);
  cityStats[city].maxLat = Math.max(cityStats[city].maxLat, lat);
  cityStats[city].minLon = Math.min(cityStats[city].minLon, lon);
  cityStats[city].maxLon = Math.max(cityStats[city].maxLon, lon);

  housingCounts[housing] = (housingCounts[housing] || 0) + 1;

  rowsData.push({
    cost,
    tip: tipTransformed,
    tipPct: tipPctTransformed,
    hour,
    deliveryMin,
    rain,
    tmax,
    tmin,
    city,
    housing
  });
}

const latStats = summarizeStats(latValues);
const lonStats = summarizeStats(lonValues);

const cityOrder = Object.keys(cityStats).sort((a, b) => {
  const diff = cityStats[b].count - cityStats[a].count;
  return diff !== 0 ? diff : a.localeCompare(b);
});
const cityBaseline = cityOrder[0] || '';

const cityCenters = {};
const cityBounds = {};
cityOrder.forEach((city) => {
  const stats = cityStats[city];
  cityCenters[city] = {
    latitude: stats.latSum / stats.count,
    longitude: stats.lonSum / stats.count,
    count: stats.count
  };
  cityBounds[city] = {
    latitude: { min: stats.minLat, max: stats.maxLat },
    longitude: { min: stats.minLon, max: stats.maxLon }
  };
});

const housingOrder = Object.keys(housingCounts).sort((a, b) => {
  const diff = housingCounts[b] - housingCounts[a];
  return diff !== 0 ? diff : a.localeCompare(b);
});
const housingBaseline = housingOrder[0] || '';

const cityFeatureKeys = cityOrder.filter(city => city !== cityBaseline).map(city => `city:${city}`);
const housingFeatureKeys = housingOrder.filter(h => h !== housingBaseline).map(h => `housing:${h}`);

const features = [
  ...BASE_FEATURES,
  ...cityFeatureKeys.map(key => ({ key, label: `City: ${key.split(':')[1]}` })),
  ...housingFeatureKeys.map(key => ({ key, label: `Housing: ${key.split(':')[1]}` }))
];

const featureKeys = features.map(feature => feature.key);
const data = rowsData.map((row) => {
  const values = {
    cost: row.cost,
    orderHour: row.hour,
    deliveryMinutes: row.deliveryMin,
    rain: row.rain,
    maxTemp: row.tmax,
    minTemp: row.tmin
  };
  cityFeatureKeys.forEach((key) => {
    const cityName = key.split(':')[1];
    values[key] = row.city === cityName ? 1 : 0;
  });
  housingFeatureKeys.forEach((key) => {
    const housingName = key.split(':')[1];
    values[key] = row.housing === housingName ? 1 : 0;
  });
  return {
    values,
    tip: row.tip,
    tipPct: row.tipPct
  };
});

const tipSelection = selectFeatures(data, featureKeys, 'tip', SIGNIFICANCE_LEVEL);
const pctSelection = selectFeatures(data, featureKeys, 'tipPct', SIGNIFICANCE_LEVEL);

const activeFeatureSet = new Set([
  ...tipSelection.featureKeys,
  ...pctSelection.featureKeys
]);
const activeFeatures = features.filter(feature => activeFeatureSet.has(feature.key));
const inputFeatures = BASE_FEATURES.map(feature => feature.key).filter(key => activeFeatureSet.has(key));
const usesHousing = housingFeatureKeys.some(key => activeFeatureSet.has(key));

const tipCoeffValues = {};
const pctCoeffValues = {};
tipSelection.featureKeys.forEach((key, idx) => {
  tipCoeffValues[key] = tipSelection.model.coeffs[idx + 1];
});
pctSelection.featureKeys.forEach((key, idx) => {
  pctCoeffValues[key] = pctSelection.model.coeffs[idx + 1];
});

const model = {
  version: 4,
  generatedAt: new Date().toISOString(),
  features: activeFeatures,
  inputFeatures,
  targets: {
    tip: { transform: TARGET_TRANSFORMS.tip },
    tipPercent: { transform: TARGET_TRANSFORMS.tipPercent }
  },
  featureSelection: {
    alpha: SIGNIFICANCE_LEVEL,
    tip: tipSelection.featureKeys,
    tipPercent: pctSelection.featureKeys
  },
  categories: {
    city: {
      baseline: cityBaseline,
      values: cityOrder,
      centers: cityCenters,
      bounds: cityBounds,
      boundaries: cityBoundaries
    },
    housing: {
      baseline: housingBaseline,
      values: housingOrder
    }
  },
  bounds: {
    latitude: { min: latStats.min, max: latStats.max },
    longitude: { min: lonStats.min, max: lonStats.max }
  },
  ranges: {
    cost: summarizeStats(costValues),
    tip: summarizeStats(tipValues),
    tipPercent: summarizeStats(tipPctValues),
    orderHour: summarizeStats(hourValues),
    deliveryMinutes: summarizeStats(deliveryValues),
    rain: summarizeStats(rainValues),
    maxTemp: summarizeStats(maxTempValues),
    minTemp: summarizeStats(minTempValues)
  },
  metrics: {
    tip: { rmse: tipSelection.model.rmse, r2: tipSelection.model.r2, n: tipSelection.model.n },
    tipPercent: { rmse: pctSelection.model.rmse, r2: pctSelection.model.r2, n: pctSelection.model.n }
  },
  coefficients: {
    tip: {
      intercept: tipSelection.model.coeffs[0],
      values: tipCoeffValues
    },
    tipPercent: {
      intercept: pctSelection.model.coeffs[0],
      values: pctCoeffValues
    }
  }
};

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.writeFileSync(OUTPUT_PATH, JSON.stringify(model, null, 2));

const meta = {
  cityCenters,
  cityBaseline,
  cityOrder,
  cityBoundaries,
  housingOptions: housingOrder,
  housingBaseline,
  bounds: model.bounds,
  inputFeatures,
  useHousing: usesHousing,
  targetTransforms: {
    tip: TARGET_TRANSFORMS.tip,
    tipPercent: TARGET_TRANSFORMS.tipPercent
  }
};

fs.mkdirSync(path.dirname(META_PATH), { recursive: true });
fs.writeFileSync(META_PATH, `window.PizzaTipsMeta = ${JSON.stringify(meta, null, 2)};`);

console.log(`Model written to ${OUTPUT_PATH}`);
