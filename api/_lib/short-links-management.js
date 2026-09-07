'use strict';

const { normalizeDestination } = require('./short-links');

const QR_DESIGN_MAX_BYTES = 128 * 1024;
const TEXT_FIELDS = Object.freeze({
  label: 160, templateId: 64, templateTitle: 160, batchId: 64, batchTitle: 160,
  contextType: 32, contextEntryId: 96, contextCompany: 160, contextTitle: 160
});
const QR_ENUMS = Object.freeze({
  uiMode: ['basic', 'advanced'], dotStyle: ['square', 'rounded', 'dots'],
  cornerStyle: ['square', 'rounded', 'extra-rounded'], ecc: ['L', 'M', 'Q', 'H'],
  centerMode: ['none', 'image', 'text'], logoShape: ['rounded', 'square', 'circle'],
  logoPlateStyle: ['auto', 'white', 'custom'], logoBorderStyle: ['auto', 'custom', 'none'],
  centerTextWeight: ['500', '600', '700'], centerTextColorStyle: ['auto', 'custom'],
  captionAlign: ['left', 'center', 'right'], captionColorStyle: ['auto', 'custom'],
  captionBgStyle: ['auto', 'white', 'custom', 'none'], imageSize: ['512', '1024', '2048', '4096'],
  exportPreset: ['sticker', 'web', 'print', 'poster', 'custom']
});
const QR_RANGES = Object.freeze({
  marginModules: [2, 10], logoSizePct: [10, 30], logoPaddingPct: [6, 22],
  logoBorderPct: [0, 12], captionSizePct: [3, 10]
});
const QR_COLORS = ['fg', 'bg', 'logoPlateColor', 'logoBorderColor', 'centerTextColor', 'captionColor', 'captionBgColor'];
const own = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);

function invalid(message){
  const error = new Error(message);
  error.statusCode = 400;
  return error;
}

function normalizeText(value, limit){
  return typeof value === 'string' ? value.replace(/\s+/g, ' ').trim().slice(0, limit) : '';
}

function normalizeTags(value){
  if (!Array.isArray(value) || value.length > 20 || value.some(tag => typeof tag !== 'string')) {
    throw invalid('Tags must be a list of up to 20 strings');
  }
  return [...new Set(value.map(tag => normalizeText(tag, 48)).filter(Boolean))];
}

function normalizeQrDesign(value){
  if (value === null) return null;
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw invalid('QR design must be an object');
  if (Buffer.byteLength(JSON.stringify(value), 'utf8') > QR_DESIGN_MAX_BYTES) throw invalid('QR design exceeds 128 KB; use a smaller logo');
  const design = { schemaVersion: 1 };
  Object.entries(QR_ENUMS).forEach(([key, allowed]) => {
    if (!own(value, key)) return;
    if (!allowed.includes(value[key])) throw invalid(`Invalid QR design ${key}`);
    design[key] = value[key];
  });
  Object.entries(QR_RANGES).forEach(([key, [min, max]]) => {
    if (!own(value, key)) return;
    const number = Number(value[key]);
    if (!Number.isFinite(number) || number < min || number > max) throw invalid(`Invalid QR design ${key}`);
    design[key] = number;
  });
  QR_COLORS.forEach(key => {
    if (!own(value, key)) return;
    if (typeof value[key] !== 'string' || !/^#[a-f\d]{6}$/i.test(value[key])) throw invalid(`Invalid QR design ${key}`);
    design[key] = value[key];
  });
  ['transparent', 'captionEnabled'].forEach(key => {
    if (!own(value, key)) return;
    if (typeof value[key] !== 'boolean') throw invalid(`Invalid QR design ${key}`);
    design[key] = value[key];
  });
  Object.entries({ centerText: 160, captionText: 400, filename: 160 }).forEach(([key, max]) => {
    if (!own(value, key)) return;
    if (typeof value[key] !== 'string') throw invalid(`Invalid QR design ${key}`);
    design[key] = normalizeText(value[key], max);
  });
  if (own(value, 'logoDataUrl')) {
    const logo = typeof value.logoDataUrl === 'string' ? value.logoDataUrl.trim() : '';
    if (logo && !/^data:image\/(?:png|jpeg|webp|gif);base64,[a-z\d+/=\s]+$/i.test(logo) &&
      !/^https?:\/\//i.test(logo) && !/^\/(?!\/)/.test(logo)) throw invalid('Invalid QR logo image');
    if (logo) design.logoDataUrl = logo;
  }
  if (own(value, 'previewDataUrl')) {
    const preview = value.previewDataUrl;
    if (typeof preview !== 'string' || !/^data:image\/png;base64,[a-z\d+/=]+$/i.test(preview)) throw invalid('Invalid QR preview image');
    design.previewDataUrl = preview;
  }
  return design;
}

function buildLinkPatch(body, current){
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw invalid('Invalid link changes');
  if (own(body, 'slug') && body.slug !== current.slug) throw invalid('Link endings cannot be renamed');
  const patch = {};
  if (own(body, 'destination')) {
    patch.destination = normalizeDestination(body.destination, { absolutizeInternalPath: true });
    if (!patch.destination) throw invalid('Enter a valid destination URL');
  }
  if (own(body, 'label')) {
    if (typeof body.label !== 'string') throw invalid('Title must be text');
    patch.label = normalizeText(body.label, TEXT_FIELDS.label);
  }
  if (own(body, 'tags')) patch.tags = normalizeTags(body.tags);
  if (own(body, 'qrDesign')) patch.qrDesign = normalizeQrDesign(body.qrDesign);
  ['disabled', 'permanent'].forEach(key => {
    if (!own(body, key)) return;
    if (typeof body[key] !== 'boolean') throw invalid(`Invalid ${key} value`);
    patch[key] = body[key];
  });
  if (own(body, 'expiresAt')) {
    const number = Number(body.expiresAt);
    if (!Number.isFinite(number) || number < 0) throw invalid('Expiration must be a Unix timestamp in seconds');
    patch.expiresAt = Math.floor(number);
    if (patch.expiresAt && patch.expiresAt <= Math.floor(Date.now() / 1000)) throw invalid('Expiration must be in the future');
  }
  const permanent = own(patch, 'permanent') ? patch.permanent : Boolean(current.permanent);
  const expiration = own(patch, 'expiresAt') ? patch.expiresAt : Number(current.expiresAt || 0);
  if (permanent && expiration && (own(patch, 'permanent') || own(patch, 'expiresAt'))) {
    throw invalid('Choose a temporary redirect before setting an expiration');
  }
  if (!Object.keys(patch).length) throw invalid('No supported link changes supplied');
  return patch;
}

function serializeLink(record, fallbackSlug = '', fallbackUpdatedAt = ''){
  const link = {
    slug: typeof record?.slug === 'string' ? record.slug : fallbackSlug,
    destination: typeof record?.destination === 'string' ? record.destination : '',
    permanent: Boolean(record?.permanent), disabled: Boolean(record?.disabled),
    expiresAt: Number.isFinite(Number(record?.expiresAt)) ? Number(record.expiresAt) : 0,
    createdAt: typeof record?.createdAt === 'string' ? record.createdAt : fallbackUpdatedAt,
    updatedAt: typeof record?.updatedAt === 'string' ? record.updatedAt : fallbackUpdatedAt,
    clicks: Number.isFinite(Number(record?.clicks)) ? Number(record.clicks) : 0,
    tags: Array.isArray(record?.tags) ? record.tags.filter(tag => typeof tag === 'string') : [],
    qrDesign: null
  };
  Object.keys(TEXT_FIELDS).forEach(key => { link[key] = typeof record?.[key] === 'string' ? record[key] : ''; });
  if (record?.qrDesign) {
    try { link.qrDesign = normalizeQrDesign(record.qrDesign); } catch {}
  }
  return link;
}

function buildAnalyticsReport({ links, items, days = '30', slug = '', configured = true, truncated = false, now = Date.now() }){
  const totals = { clicks: 0, linkClicks: 0, qrScans: 0, unknownClicks: 0, lifetimeClicks: 0 };
  const byDate = new Map();
  const linkSlugs = new Set(links.map(link => link.slug));
  const start = days === 'all' ? 0 : Date.parse(new Date(now).toISOString().slice(0, 10)) - (Number(days) - 1) * 86400000;
  totals.lifetimeClicks = links.reduce((sum, link) => sum + Math.max(0, Number(link.clicks) || 0), 0);
  let detailedEvents = 0;
  let invalidDateEvents = 0;
  if (days !== 'all') {
    for (let day = start; day <= now; day += 86400000) {
      const date = new Date(day).toISOString().slice(0, 10);
      byDate.set(date, { date, clicks: 0, linkClicks: 0, qrScans: 0, unknownClicks: 0 });
    }
  }
  items.forEach(item => {
    if (!linkSlugs.has(item?.slug) || item.clickId === '__historical_baseline__' || item.entityType === 'clickBaseline') return;
    detailedEvents += 1;
    const timestamp = Date.parse(item.clickedAt);
    if (!Number.isFinite(timestamp) || timestamp > now) { invalidDateEvents += 1; return; }
    if (timestamp < start) return;
    const date = new Date(timestamp).toISOString().slice(0, 10);
    if (!byDate.has(date)) byDate.set(date, { date, clicks: 0, linkClicks: 0, qrScans: 0, unknownClicks: 0 });
    const row = byDate.get(date);
    const channel = item.channel === 'qr' ? 'qrScans' : item.channel === 'link' ? 'linkClicks' : 'unknownClicks';
    totals.clicks += 1;
    totals[channel] += 1;
    row.clicks += 1;
    row[channel] += 1;
  });
  const unattributedHistoricalClicks = Math.max(0, totals.lifetimeClicks - detailedEvents) + invalidDateEvents;
  if (days === 'all') {
    totals.unknownClicks += unattributedHistoricalClicks;
    totals.clicks += unattributedHistoricalClicks;
  }
  const complete = configured && !truncated && unattributedHistoricalClicks === 0;
  let note = 'Link and QR activity reflects recorded requests, including automated visits. Older unmarked requests remain combined activity.';
  if (!configured) note = 'Detailed tracking is not configured. Lifetime totals are available; dates and channels are unknown.';
  else if (truncated) note = 'Detailed history reached the query limit. Date and channel totals are partial; choose a single link for a smaller report.';
  else if (unattributedHistoricalClicks) note = 'Some lifetime activity has no dated event history and cannot be assigned to a date or channel. Date filters show recorded activity only.';
  return {
    days, slug, totals,
    daily: [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date)),
    completeness: {
      complete, truncated, detailedTrackingConfigured: configured, unattributedHistoricalClicks,
      detailedEvents, retentionVerified: false, note
    }
  };
}

module.exports = { QR_DESIGN_MAX_BYTES, TEXT_FIELDS, buildAnalyticsReport, buildLinkPatch, normalizeQrDesign, normalizeTags, serializeLink };
