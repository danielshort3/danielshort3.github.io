const model = require('./model.json');

const { CONFIDENCE_LEVEL = '0.8' } = process.env;

const Z_BY_LEVEL = {
  0.8: 1.282,
  0.85: 1.44,
  0.9: 1.645,
  0.95: 1.96
};
const normalizeConfidenceLevel = (value) => {
  if (value === null || value === undefined || value === '') return null;
  let num = Number(value);
  if (!Number.isFinite(num)) {
    const cleaned = String(value).replace('%', '').trim();
    num = Number(cleaned);
  }
  if (!Number.isFinite(num)) return null;
  if (num > 1) num /= 100;
  const levels = Object.keys(Z_BY_LEVEL).map(Number);
  if (!levels.length) return null;
  let closest = levels[0];
  let minDiff = Math.abs(num - closest);
  levels.forEach((level) => {
    const diff = Math.abs(num - level);
    if (diff < minDiff) {
      minDiff = diff;
      closest = level;
    }
  });
  return closest;
};
const DEFAULT_CONF_LEVEL = normalizeConfidenceLevel(CONFIDENCE_LEVEL) || 0.8;
const getZScore = (level) => Z_BY_LEVEL[level] || Z_BY_LEVEL[DEFAULT_CONF_LEVEL] || 1.282;

const buildHeaders = () => ({
  'Content-Type': 'application/json'
});

const parseBody = (event = {}) => {
  const rawBody = event.isBase64Encoded
    ? Buffer.from(event.body || '', 'base64').toString('utf8')
    : (event.body || '');
  const header = event.headers || {};
  const contentType = (header['content-type'] || header['Content-Type'] || '').toLowerCase();
  if (!rawBody) return {};
  if (contentType.includes('application/x-www-form-urlencoded')) {
    const params = new URLSearchParams(rawBody);
    return Object.fromEntries(params.entries());
  }
  try {
    return JSON.parse(rawBody);
  } catch {
    return {};
  }
};

const parseNumber = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const parseHour = (value) => {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string' && value.includes(':')) {
    const parts = value.split(':');
    const h = Number(parts[0]);
    return Number.isFinite(h) ? h : null;
  }
  const h = Number(value);
  return Number.isFinite(h) ? h : null;
};

const parseDurationMinutes = (value) => {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string' && value.includes(':')) {
    const parts = value.split(':');
    if (parts.length < 2) return null;
    const h = Number(parts[0]);
    const m = Number(parts[1]);
    const s = Number(parts[2] || 0);
    if (![h, m, s].every(n => Number.isFinite(n))) return null;
    return h * 60 + m + s / 60;
  }
  const m = Number(value);
  return Number.isFinite(m) ? m : null;
};

const BASE_KEYS = ['cost', 'orderHour', 'deliveryMinutes', 'rain', 'maxTemp', 'minTemp'];

const normalizeOption = (value, options = []) => {
  if (value === null || value === undefined) return null;
  const cleaned = String(value).trim();
  if (!cleaned) return null;
  const lower = cleaned.toLowerCase();
  const match = options.find(opt => String(opt).toLowerCase() === lower);
  return match || null;
};

const pointInRing = (point, ring) => {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const intersect = ((yi > point[1]) !== (yj > point[1]))
      && (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
};

const pointInPolygon = (point, rings) => {
  if (!rings.length || !pointInRing(point, rings[0])) return false;
  for (let i = 1; i < rings.length; i++) {
    if (pointInRing(point, rings[i])) return false;
  }
  return true;
};

const pointInGeometry = (point, geometry) => {
  if (!geometry) return false;
  if (geometry.type === 'Polygon') {
    return pointInPolygon(point, geometry.coordinates || []);
  }
  if (geometry.type === 'MultiPolygon') {
    return (geometry.coordinates || []).some((rings) => pointInPolygon(point, rings));
  }
  return false;
};

const resolveCity = (lat, lon) => {
  const features = model.categories?.city?.boundaries?.features || [];
  const point = [lon, lat];
  for (const feature of features) {
    if (pointInGeometry(point, feature.geometry)) {
      return { city: feature.properties?.city || null };
    }
  }
  return { city: null };
};
const clampMin = (value, min) => (value < min ? min : value);

const predictValue = (coeffs, inputs) => {
  let total = coeffs.intercept;
  Object.keys(coeffs.values).forEach((key) => {
    total += coeffs.values[key] * inputs[key];
  });
  return total;
};

const buildBreakdown = (coeffs, inputs) => {
  const items = [
    { key: 'intercept', value: 1, contribution: coeffs.intercept }
  ];
  Object.keys(coeffs.values).forEach((key) => {
    const value = inputs[key];
    const contribution = coeffs.values[key] * value;
    items.push({ key, value, contribution });
  });
  const total = items.reduce((sum, entry) => sum + entry.contribution, 0);
  return { total, items };
};

const buildInterval = (prediction, rmse, level, zScore) => {
  const delta = zScore * rmse;
  return {
    level,
    low: prediction - delta,
    high: prediction + delta,
    rmse
  };
};

const getTransform = (target) => model.targets?.[target]?.transform || 'none';

const inverseTransform = (value, transform) => {
  if (transform === 'log1p') {
    return Math.expm1(value);
  }
  return value;
};

const buildIntervalTransformed = (prediction, rmse, transform, level, zScore) => {
  if (transform === 'log1p') {
    const delta = zScore * rmse;
    return {
      level,
      low: Math.expm1(prediction - delta),
      high: Math.expm1(prediction + delta),
      rmse,
      scale: 'log1p'
    };
  }
  return buildInterval(prediction, rmse, level, zScore);
};

const addRangeWarning = (warnings, label, value, range) => {
  if (!range || typeof range.min !== 'number' || typeof range.max !== 'number') return;
  if (value < range.min || value > range.max) {
    warnings.push(`${label} ${value} is outside the training range (${range.min} to ${range.max}).`);
  }
};

const clampInt = (value, min, max, fallback) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  const rounded = Math.round(num);
  return Math.min(max, Math.max(min, rounded));
};

const normalizeBounds = (bounds) => {
  if (!bounds || typeof bounds !== 'object') return null;
  const latMin = parseNumber(bounds.latMin);
  const latMax = parseNumber(bounds.latMax);
  const lonMin = parseNumber(bounds.lonMin);
  const lonMax = parseNumber(bounds.lonMax);
  if ([latMin, latMax, lonMin, lonMax].some(v => v === null)) return null;
  return {
    latMin: Math.min(latMin, latMax),
    latMax: Math.max(latMin, latMax),
    lonMin: Math.min(lonMin, lonMax),
    lonMax: Math.max(lonMin, lonMax)
  };
};

exports.handler = async (event) => {
  if (event.requestContext?.http?.method === 'OPTIONS') {
    return { statusCode: 204, headers: buildHeaders() };
  }

  const payload = parseBody(event);
  if (!payload || typeof payload !== 'object') {
    return {
      statusCode: 400,
      headers: buildHeaders(),
      body: JSON.stringify({ error: 'Invalid payload.' })
    };
  }

  const confidenceLevel = normalizeConfidenceLevel(
    payload.confidenceLevel ?? payload.confidence ?? payload.interval
  ) || DEFAULT_CONF_LEVEL;
  const zScore = getZScore(confidenceLevel);

  const latitude = parseNumber(payload.latitude ?? payload.lat);
  const longitude = parseNumber(payload.longitude ?? payload.lon ?? payload.lng);
  const coeffKeys = new Set([
    ...Object.keys(model.coefficients?.tip?.values || {}),
    ...Object.keys(model.coefficients?.tipPercent?.values || {})
  ]);
  const activeBase = new Set(model.inputFeatures || BASE_KEYS);
  const requireBase = (key) => coeffKeys.has(key) || activeBase.has(key);
  const cost = requireBase('cost') ? parseNumber(payload.cost ?? payload.orderCost) : null;
  const orderHourRaw = requireBase('orderHour') ? parseHour(payload.orderHour ?? payload.hour ?? payload.orderTime) : null;
  const deliveryMinutesRaw = requireBase('deliveryMinutes')
    ? parseDurationMinutes(payload.deliveryMinutes ?? payload.totalDeliveryMinutes ?? payload.deliveryTime)
    : null;
  const rain = requireBase('rain') ? parseNumber(payload.rain ?? payload.rainInches) : null;
  const maxTemp = requireBase('maxTemp') ? parseNumber(payload.maxTemp ?? payload.tmax) : null;
  const minTemp = requireBase('minTemp') ? parseNumber(payload.minTemp ?? payload.tmin) : null;
  const housingRaw = payload.housing ?? payload.housingType ?? payload.housingCategory;
  const housing = normalizeOption(housingRaw, model.categories?.housing?.values || []);
  const usesHousing = Array.from(coeffKeys).some((key) => key.startsWith('housing:'));

  const errors = [];
  if (latitude === null || longitude === null) errors.push('Latitude and longitude are required.');
  if (requireBase('cost') && cost === null) errors.push('Cost is required.');
  if (requireBase('orderHour') && orderHourRaw === null) errors.push('Order hour is required.');
  if (requireBase('deliveryMinutes') && deliveryMinutesRaw === null) errors.push('Delivery minutes are required.');
  if (requireBase('rain') && rain === null) errors.push('Rain (inches) is required.');
  if (requireBase('maxTemp') && maxTemp === null) errors.push('Max temp is required.');
  if (requireBase('minTemp') && minTemp === null) errors.push('Min temp is required.');
  if (usesHousing && !housing) errors.push('Housing type is required.');

  if (errors.length) {
    return {
      statusCode: 400,
      headers: buildHeaders(),
      body: JSON.stringify({ error: 'Missing or invalid inputs.', details: errors })
    };
  }

  const orderHour = orderHourRaw === null ? null : Math.floor(orderHourRaw);
  const deliveryMinutes = deliveryMinutesRaw;
  const { city } = resolveCity(latitude, longitude);
  if (!city) {
    return {
      statusCode: 400,
      headers: buildHeaders(),
      body: JSON.stringify({ error: 'Location is outside supported city boundaries.' })
    };
  }

  const cityValues = model.categories?.city?.values || [];
  const cityBaseline = model.categories?.city?.baseline || '';
  const cityKeyList = cityValues.filter((cityName) => {
    if (cityName === cityBaseline) return false;
    return coeffKeys.has(`city:${cityName}`);
  });
  const housingValues = model.categories?.housing?.values || [];
  const housingBaseline = model.categories?.housing?.baseline || '';
  const housingKeyList = housingValues.filter((housingName) => {
    if (housingName === housingBaseline) return false;
    return coeffKeys.has(`housing:${housingName}`);
  });

  const baseInputs = {};
  if (coeffKeys.has('cost') && cost !== null) baseInputs.cost = cost;
  if (coeffKeys.has('orderHour') && orderHour !== null) baseInputs.orderHour = orderHour;
  if (coeffKeys.has('deliveryMinutes') && deliveryMinutes !== null) baseInputs.deliveryMinutes = deliveryMinutes;
  if (coeffKeys.has('rain') && rain !== null) baseInputs.rain = rain;
  if (coeffKeys.has('maxTemp') && maxTemp !== null) baseInputs.maxTemp = maxTemp;
  if (coeffKeys.has('minTemp') && minTemp !== null) baseInputs.minTemp = minTemp;

  const buildInputsForLocation = (cityName, housingName) => {
    const inputs = { ...baseInputs };
    cityKeyList.forEach((name) => {
      inputs[`city:${name}`] = name === cityName ? 1 : 0;
    });
    housingKeyList.forEach((name) => {
      inputs[`housing:${name}`] = name === housingName ? 1 : 0;
    });
    return inputs;
  };

  const inputs = buildInputsForLocation(city, housing);

  const tipTransform = getTransform('tip');
  const tipPctTransform = getTransform('tipPercent');
  const tipPredictionTransformed = predictValue(model.coefficients.tip, inputs);
  const tipPctPredictionTransformed = predictValue(model.coefficients.tipPercent, inputs);
  const tipPrediction = clampMin(inverseTransform(tipPredictionTransformed, tipTransform), 0);
  const tipPctPrediction = clampMin(inverseTransform(tipPctPredictionTransformed, tipPctTransform), 0);
  const predictForInputs = (inputValues) => {
    const tipTransformed = predictValue(model.coefficients.tip, inputValues);
    const tipValue = clampMin(inverseTransform(tipTransformed, tipTransform), 0);
    const tipPctTransformed = predictValue(model.coefficients.tipPercent, inputValues);
    const tipPctValue = clampMin(inverseTransform(tipPctTransformed, tipPctTransform), 0);
    return { tip: tipValue, tipPercent: tipPctValue };
  };

  const tipInterval = buildIntervalTransformed(
    tipPredictionTransformed,
    model.metrics.tip.rmse,
    tipTransform,
    confidenceLevel,
    zScore
  );
  const tipPctInterval = buildIntervalTransformed(
    tipPctPredictionTransformed,
    model.metrics.tipPercent.rmse,
    tipPctTransform,
    confidenceLevel,
    zScore
  );

  tipInterval.low = clampMin(tipInterval.low, 0);
  tipPctInterval.low = clampMin(tipPctInterval.low, 0);

  let heatmap = null;
  if (payload.grid) {
    const gridBounds = normalizeBounds(payload.grid.bounds) || {
      latMin: model.bounds.latitude.min,
      latMax: model.bounds.latitude.max,
      lonMin: model.bounds.longitude.min,
      lonMax: model.bounds.longitude.max
    };
    const rows = clampInt(payload.grid.rows, 10, 40, 24);
    const cols = clampInt(payload.grid.cols, 10, 40, 24);
    const latSpan = gridBounds.latMax - gridBounds.latMin;
    const lonSpan = gridBounds.lonMax - gridBounds.lonMin;
    const latStep = rows > 1 ? latSpan / (rows - 1) : 0;
    const lonStep = cols > 1 ? lonSpan / (cols - 1) : 0;
    const points = [];
    let minTip = Infinity;
    let maxTip = -Infinity;
    for (let r = 0; r < rows; r++) {
      const lat = gridBounds.latMin + latStep * r;
      for (let c = 0; c < cols; c++) {
        const lon = gridBounds.lonMin + lonStep * c;
        const resolved = resolveCity(lat, lon);
        if (!resolved.city) continue;
        const gridInputs = buildInputsForLocation(resolved.city, housing);
        const prediction = predictForInputs(gridInputs);
        if (!Number.isFinite(prediction.tip)) continue;
        points.push({ lat, lon, tip: prediction.tip });
        minTip = Math.min(minTip, prediction.tip);
        maxTip = Math.max(maxTip, prediction.tip);
      }
    }
    if (!Number.isFinite(minTip) || !Number.isFinite(maxTip)) {
      minTip = 0;
      maxTip = 0;
    }
    heatmap = {
      bounds: gridBounds,
      rows,
      cols,
      min: minTip,
      max: maxTip,
      points
    };
  }

  const warnings = [];
  addRangeWarning(warnings, 'Latitude', latitude, model.bounds.latitude);
  addRangeWarning(warnings, 'Longitude', longitude, model.bounds.longitude);
  if (requireBase('cost') && cost !== null) {
    addRangeWarning(warnings, 'Cost', cost, model.ranges.cost);
  }
  if (requireBase('orderHour') && orderHour !== null) {
    addRangeWarning(warnings, 'Order hour', orderHour, model.ranges.orderHour);
  }
  if (requireBase('deliveryMinutes') && deliveryMinutes !== null) {
    addRangeWarning(warnings, 'Delivery minutes', deliveryMinutes, model.ranges.deliveryMinutes);
  }
  if (requireBase('rain') && rain !== null) {
    addRangeWarning(warnings, 'Rain', rain, model.ranges.rain);
  }
  if (requireBase('maxTemp') && maxTemp !== null) {
    addRangeWarning(warnings, 'Max temp', maxTemp, model.ranges.maxTemp);
  }
  if (requireBase('minTemp') && minTemp !== null) {
    addRangeWarning(warnings, 'Min temp', minTemp, model.ranges.minTemp);
  }

  const responseInputs = {
    latitude,
    longitude
  };
  if (cost !== null) responseInputs.cost = cost;
  if (orderHour !== null) responseInputs.orderHour = orderHour;
  if (deliveryMinutes !== null) responseInputs.deliveryMinutes = deliveryMinutes;
  if (rain !== null) responseInputs.rain = rain;
  if (maxTemp !== null) responseInputs.maxTemp = maxTemp;
  if (minTemp !== null) responseInputs.minTemp = minTemp;
  if (housing) responseInputs.housing = housing;

  const response = {
    ok: true,
    inputs: responseInputs,
    bucket: {
      city,
      housing: housing || null
    },
    location: {
      latitude,
      longitude
    },
    predictions: {
      tip: {
        value: tipPrediction,
        interval: tipInterval
      },
      tipPercent: {
        value: tipPctPrediction,
        interval: tipPctInterval
      }
    },
    heatmap,
    breakdown: {
      tip: { ...buildBreakdown(model.coefficients.tip, inputs), scale: tipTransform },
      tipPercent: { ...buildBreakdown(model.coefficients.tipPercent, inputs), scale: tipPctTransform }
    },
    warnings
  };

  return {
    statusCode: 200,
    headers: buildHeaders(),
    body: JSON.stringify(response)
  };
};
