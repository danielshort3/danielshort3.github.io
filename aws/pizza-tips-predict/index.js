const model = require('./model.json');

const { CONFIDENCE_LEVEL = '0.9' } = process.env;

const CONF_LEVEL = Number(CONFIDENCE_LEVEL) || 0.9;
const Z_BY_LEVEL = {
  0.8: 1.282,
  0.85: 1.44,
  0.9: 1.645,
  0.95: 1.96
};
const Z_SCORE = Z_BY_LEVEL[CONF_LEVEL] || 1.645;

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

const buildInterval = (prediction, rmse) => {
  const delta = Z_SCORE * rmse;
  return {
    level: CONF_LEVEL,
    low: prediction - delta,
    high: prediction + delta,
    rmse
  };
};

const addRangeWarning = (warnings, label, value, range) => {
  if (!range || typeof range.min !== 'number' || typeof range.max !== 'number') return;
  if (value < range.min || value > range.max) {
    warnings.push(`${label} ${value} is outside the training range (${range.min} to ${range.max}).`);
  }
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

  const latitude = parseNumber(payload.latitude ?? payload.lat);
  const longitude = parseNumber(payload.longitude ?? payload.lon ?? payload.lng);
  const cost = parseNumber(payload.cost ?? payload.orderCost);
  const orderHourRaw = parseHour(payload.orderHour ?? payload.hour ?? payload.orderTime);
  const deliveryMinutesRaw = parseDurationMinutes(payload.deliveryMinutes ?? payload.totalDeliveryMinutes ?? payload.deliveryTime);
  const rain = parseNumber(payload.rain ?? payload.rainInches);
  const maxTemp = parseNumber(payload.maxTemp ?? payload.tmax);
  const minTemp = parseNumber(payload.minTemp ?? payload.tmin);
  const housingRaw = payload.housing ?? payload.housingType ?? payload.housingCategory;
  const housing = normalizeOption(housingRaw, model.categories?.housing?.values || []);

  const errors = [];
  if (latitude === null || longitude === null) errors.push('Latitude and longitude are required.');
  if (cost === null) errors.push('Cost is required.');
  if (orderHourRaw === null) errors.push('Order hour is required.');
  if (deliveryMinutesRaw === null) errors.push('Delivery minutes are required.');
  if (rain === null) errors.push('Rain (inches) is required.');
  if (maxTemp === null) errors.push('Max temp is required.');
  if (minTemp === null) errors.push('Min temp is required.');
  if (!housing) errors.push('Housing type is required.');

  if (errors.length) {
    return {
      statusCode: 400,
      headers: buildHeaders(),
      body: JSON.stringify({ error: 'Missing or invalid inputs.', details: errors })
    };
  }

  const orderHour = Math.floor(orderHourRaw);
  const deliveryMinutes = deliveryMinutesRaw;
  const { city } = resolveCity(latitude, longitude);
  if (!city) {
    return {
      statusCode: 400,
      headers: buildHeaders(),
      body: JSON.stringify({ error: 'Location is outside supported city boundaries.' })
    };
  }

  const inputs = {
    cost,
    orderHour,
    deliveryMinutes,
    rain,
    maxTemp,
    minTemp
  };
  const cityValues = model.categories?.city?.values || [];
  const cityBaseline = model.categories?.city?.baseline || '';
  cityValues.forEach((cityName) => {
    if (cityName !== cityBaseline) {
      inputs[`city:${cityName}`] = cityName === city ? 1 : 0;
    }
  });
  const housingValues = model.categories?.housing?.values || [];
  const housingBaseline = model.categories?.housing?.baseline || '';
  housingValues.forEach((housingName) => {
    if (housingName !== housingBaseline) {
      inputs[`housing:${housingName}`] = housingName === housing ? 1 : 0;
    }
  });

  const tipPrediction = predictValue(model.coefficients.tip, inputs);
  const tipPctPrediction = predictValue(model.coefficients.tipPercent, inputs);

  const tipInterval = buildInterval(tipPrediction, model.metrics.tip.rmse);
  const tipPctInterval = buildInterval(tipPctPrediction, model.metrics.tipPercent.rmse);

  tipInterval.low = clampMin(tipInterval.low, 0);
  tipPctInterval.low = clampMin(tipPctInterval.low, 0);

  const warnings = [];
  addRangeWarning(warnings, 'Latitude', latitude, model.bounds.latitude);
  addRangeWarning(warnings, 'Longitude', longitude, model.bounds.longitude);
  addRangeWarning(warnings, 'Cost', cost, model.ranges.cost);
  addRangeWarning(warnings, 'Order hour', orderHour, model.ranges.orderHour);
  addRangeWarning(warnings, 'Delivery minutes', deliveryMinutes, model.ranges.deliveryMinutes);
  addRangeWarning(warnings, 'Rain', rain, model.ranges.rain);
  addRangeWarning(warnings, 'Max temp', maxTemp, model.ranges.maxTemp);
  addRangeWarning(warnings, 'Min temp', minTemp, model.ranges.minTemp);

  const response = {
    ok: true,
    inputs: {
      latitude,
      longitude,
      cost,
      orderHour,
      deliveryMinutes,
      rain,
      maxTemp,
      minTemp,
      housing
    },
    bucket: {
      city,
      housing
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
    breakdown: {
      tip: buildBreakdown(model.coefficients.tip, inputs),
      tipPercent: buildBreakdown(model.coefficients.tipPercent, inputs)
    },
    warnings
  };

  return {
    statusCode: 200,
    headers: buildHeaders(),
    body: JSON.stringify(response)
  };
};
