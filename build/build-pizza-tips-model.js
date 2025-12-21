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

function fitModel(data, targetKey) {
  const X = data.map(d => d.x);
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
  const r2 = 1 - ssRes / ssTot;
  return { coeffs, rmse, r2, n: y.length };
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

if (!fs.existsSync(INPUT_PATH)) {
  throw new Error(`Missing input CSV at ${INPUT_PATH}`);
}
if (!fs.existsSync(BOUNDARY_PATH)) {
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
    tip,
    tipPct,
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

const data = rowsData.map((row) => {
  const cityIndicators = cityFeatureKeys.map((key) => {
    const cityName = key.split(':')[1];
    return row.city === cityName ? 1 : 0;
  });
  const housingIndicators = housingFeatureKeys.map((key) => {
    const housingName = key.split(':')[1];
    return row.housing === housingName ? 1 : 0;
  });
  return {
    x: [
      1,
      row.cost,
      row.hour,
      row.deliveryMin,
      row.rain,
      row.tmax,
      row.tmin,
      ...cityIndicators,
      ...housingIndicators
    ],
    tip: row.tip,
    tipPct: row.tipPct
  };
});

const tipModel = fitModel(data, 'tip');
const pctModel = fitModel(data, 'tipPct');

const featureKeys = features.map(feature => feature.key);
const tipCoeffValues = {};
const pctCoeffValues = {};
featureKeys.forEach((key, idx) => {
  tipCoeffValues[key] = tipModel.coeffs[idx + 1];
  pctCoeffValues[key] = pctModel.coeffs[idx + 1];
});

const model = {
  version: 2,
  generatedAt: new Date().toISOString(),
  features,
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
    tip: { rmse: tipModel.rmse, r2: tipModel.r2, n: tipModel.n },
    tipPercent: { rmse: pctModel.rmse, r2: pctModel.r2, n: pctModel.n }
  },
  coefficients: {
    tip: {
      intercept: tipModel.coeffs[0],
      values: tipCoeffValues
    },
    tipPercent: {
      intercept: pctModel.coeffs[0],
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
  bounds: model.bounds
};

fs.mkdirSync(path.dirname(META_PATH), { recursive: true });
fs.writeFileSync(META_PATH, `window.PizzaTipsMeta = ${JSON.stringify(meta, null, 2)};`);

console.log(`Model written to ${OUTPUT_PATH}`);
