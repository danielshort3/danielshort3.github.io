/*
  Shared helpers for short-link set templates and generated batches.
*/
'use strict';

const {
  DEFAULT_RANDOM_LENGTH,
  buildSetRecordKey,
  createSetId,
  normalizeDestination,
  normalizeRandomLength,
  normalizeSetId
} = require('./short-links');

const DEFAULT_DURATION_VALUE = 7;
const DEFAULT_DURATION_UNIT = 'days';
const MAX_SET_ENTRIES = 40;

function sanitizeText(value, maxLen){
  const raw = typeof value === 'string' ? value : '';
  if (!raw) return '';
  const cleaned = raw.replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  return cleaned.length <= maxLen ? cleaned : cleaned.slice(0, maxLen);
}

function normalizeDurationUnit(value){
  const raw = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (raw === 'hours' || raw === 'weeks') return raw;
  return DEFAULT_DURATION_UNIT;
}

function normalizeDurationValue(value, fallback = DEFAULT_DURATION_VALUE){
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(1, Math.min(365, Math.floor(numeric)));
}

function normalizeExpirationMode(value){
  return String(value || '').trim().toLowerCase() === 'temporary' ? 'temporary' : 'permanent';
}

function normalizeTemplateEntries(entries){
  return (Array.isArray(entries) ? entries : [])
    .map((entry, index) => {
      const label = sanitizeText(entry && entry.label, 120);
      const destination = normalizeDestination(entry && entry.destination);
      const rowId = sanitizeText(entry && (entry.rowId || entry.id), 64) || `entry-${index + 1}`;
      if (!label || !destination) return null;
      return {
        rowId,
        label,
        destination,
        enabled: entry && entry.enabled !== false
      };
    })
    .filter(Boolean)
    .slice(0, MAX_SET_ENTRIES);
}

function normalizeGenerationContext(input){
  const source = input && typeof input === 'object' ? input : {};
  return {
    type: sanitizeText(source.type, 32),
    entryId: sanitizeText(source.entryId, 96),
    company: sanitizeText(source.company, 160),
    title: sanitizeText(source.title, 160)
  };
}

function serializeSetTemplate(record){
  const setId = normalizeSetId(record && record.setId);
  return {
    setId,
    title: sanitizeText(record && record.title, 160),
    defaultRandomLength: normalizeRandomLength(record && record.defaultRandomLength, DEFAULT_RANDOM_LENGTH),
    defaultExpirationMode: normalizeExpirationMode(record && record.defaultExpirationMode),
    defaultDurationValue: normalizeDurationValue(record && record.defaultDurationValue, DEFAULT_DURATION_VALUE),
    defaultDurationUnit: normalizeDurationUnit(record && record.defaultDurationUnit),
    entries: normalizeTemplateEntries(record && record.entries),
    createdAt: typeof record?.createdAt === 'string' ? record.createdAt : '',
    updatedAt: typeof record?.updatedAt === 'string' ? record.updatedAt : ''
  };
}

function buildSetTemplateRecord(body, existing){
  const base = existing && typeof existing === 'object' ? serializeSetTemplate(existing) : null;
  const setId = normalizeSetId(body && body.setId) || base?.setId || createSetId();
  const hasTitle = !!(body && Object.prototype.hasOwnProperty.call(body, 'title'));
  const hasEntries = !!(body && Object.prototype.hasOwnProperty.call(body, 'entries'));
  const title = hasTitle ? sanitizeText(body && body.title, 160) : (base?.title || '');
  const entries = hasEntries ? normalizeTemplateEntries(body && body.entries) : (base?.entries || []);
  const updatedAt = new Date().toISOString();
  return {
    slug: buildSetRecordKey(setId),
    entityType: 'setTemplate',
    setId,
    title,
    defaultRandomLength: normalizeRandomLength(body && body.defaultRandomLength, base?.defaultRandomLength || DEFAULT_RANDOM_LENGTH),
    defaultExpirationMode: normalizeExpirationMode(body && body.defaultExpirationMode ? body.defaultExpirationMode : base?.defaultExpirationMode),
    defaultDurationValue: normalizeDurationValue(body && body.defaultDurationValue, base?.defaultDurationValue || DEFAULT_DURATION_VALUE),
    defaultDurationUnit: normalizeDurationUnit(body && body.defaultDurationUnit ? body.defaultDurationUnit : base?.defaultDurationUnit),
    entries,
    createdAt: base?.createdAt || updatedAt,
    updatedAt
  };
}

function buildBatchTitle(providedTitle, templateTitle, context){
  const explicit = sanitizeText(providedTitle, 160);
  if (explicit) return explicit;
  const ctx = normalizeGenerationContext(context);
  return [sanitizeText(templateTitle, 160), ctx.company, ctx.title].filter(Boolean).join(' · ');
}

function resolveBatchTiming(body, template){
  const nowSeconds = Math.floor(Date.now() / 1000);
  const hasExplicitExpiresAt = !!(body && Object.prototype.hasOwnProperty.call(body, 'expiresAt'));
  if (hasExplicitExpiresAt) {
    const numericExpiresAt = Number(body.expiresAt);
    if (!Number.isFinite(numericExpiresAt) || numericExpiresAt <= nowSeconds) {
      return { ok: false, error: 'Invalid expiresAt (must be a future Unix timestamp in seconds)' };
    }
    return {
      ok: true,
      permanent: false,
      expiresAt: Math.floor(numericExpiresAt),
      expirationMode: 'temporary',
      durationValue: 0,
      durationUnit: ''
    };
  }

  const templateDefaults = serializeSetTemplate(template);
  const mode = normalizeExpirationMode(body && body.expirationMode ? body.expirationMode : templateDefaults.defaultExpirationMode);
  if (mode !== 'temporary') {
    return {
      ok: true,
      permanent: true,
      expiresAt: 0,
      expirationMode: 'permanent',
      durationValue: 0,
      durationUnit: ''
    };
  }

  const durationValue = normalizeDurationValue(body && body.durationValue, templateDefaults.defaultDurationValue || DEFAULT_DURATION_VALUE);
  const durationUnit = normalizeDurationUnit(body && body.durationUnit ? body.durationUnit : templateDefaults.defaultDurationUnit);
  const multiplier = durationUnit === 'hours'
    ? 60 * 60
    : durationUnit === 'weeks'
      ? 7 * 24 * 60 * 60
      : 24 * 60 * 60;
  const expiresAt = nowSeconds + (durationValue * multiplier);
  return {
    ok: true,
    permanent: false,
    expiresAt,
    expirationMode: 'temporary',
    durationValue,
    durationUnit
  };
}

module.exports = {
  DEFAULT_DURATION_VALUE,
  DEFAULT_DURATION_UNIT,
  MAX_SET_ENTRIES,
  sanitizeText,
  normalizeDurationUnit,
  normalizeDurationValue,
  normalizeExpirationMode,
  normalizeTemplateEntries,
  normalizeGenerationContext,
  serializeSetTemplate,
  buildSetTemplateRecord,
  buildBatchTitle,
  resolveBatchTiming
};
