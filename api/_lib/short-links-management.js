'use strict';

const { normalizeDestination } = require('./short-links');
const { randomBytes } = require('node:crypto');

const QR_DESIGN_MAX_BYTES = 128 * 1024;
const DESTINATION_HISTORY_MAX_BYTES = 128 * 1024;
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

function nextLinkTimestamp(current, now = Date.now()){
  const previous = Date.parse(current?.updatedAt);
  return new Date(Number.isFinite(previous) ? Math.max(now, previous + 1) : now).toISOString();
}

// Bound the history so even long destinations can coexist with a saved QR design in one DynamoDB item.
function revisedDestinations(current, changedAt){
  const history = Array.isArray(current.destinationHistory) ? current.destinationHistory : [];
  const retained = [];
  let bytes = 2;
  for (const entry of [{ destination: current.destination, changedAt }, ...history.slice(0, 24)]) {
    bytes += Buffer.byteLength(JSON.stringify(entry)) + 1;
    if (bytes > DESTINATION_HISTORY_MAX_BYTES) break;
    retained.push(entry);
  }
  return retained;
}

function appendPlacement(current, input){
  const name = typeof input?.name === 'string' ? input.name.trim().replace(/\s+/g, ' ') : '';
  if (!name || name.length > 96) throw invalid('Enter a placement name of up to 96 characters');
  if (!['qr', 'link'].includes(input.channel)) throw invalid('Choose QR code or short link for the placement');
  const placements = Array.isArray(current.placements) ? current.placements : [];
  if (placements.length >= 64) throw invalid('This link already has 64 placements');
  if (placements.some(item => item.channel === input.channel && item.name.toLowerCase() === name.toLowerCase())) {
    throw invalid('This placement already exists');
  }
  // The identifier goes into every printed code. Keep it short and unique within this link.
  let id;
  do { id = randomBytes(6).toString('base64url'); } while (placements.some(item => item.id === id));
  return [...placements, { id, name, channel: input.channel }];
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

function buildLinkPatch(body, current, updatedAt = nextLinkTimestamp(current)){
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw invalid('Invalid link changes');
  if (own(body, 'expectedUpdatedAt') && body.expectedUpdatedAt !== (current.updatedAt || '')) {
    const error = new Error('This link changed. Reload it and review your changes.');
    error.statusCode = 409;
    throw error;
  }
  if (own(body, 'slug') && body.slug !== current.slug) throw invalid('Link endings cannot be renamed');
  const patch = {};
  if (own(body, 'destination')) {
    // Older international paths can exceed the input limit once URL-encoded by the server.
    const saved = current.destination === body.destination ||
      current.destinationHistory?.some(entry => entry.destination === body.destination);
    patch.destination = normalizeDestination(body.destination, { absolutizeInternalPath: true, allowSavedUrl: saved });
    if (!patch.destination) throw invalid('Enter a valid destination URL');
    if (current.destination && current.destination !== patch.destination) {
      patch.destinationHistory = revisedDestinations(current, updatedAt);
    }
  }
  if (own(body, 'label')) {
    if (typeof body.label !== 'string') throw invalid('Title must be text');
    patch.label = normalizeText(body.label, TEXT_FIELDS.label);
  }
  if (own(body, 'tags')) patch.tags = normalizeTags(body.tags);
  if (own(body, 'addPlacement')) patch.placements = appendPlacement(current, body.addPlacement);
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
    destinationHistory: Array.isArray(record?.destinationHistory) ? record.destinationHistory : [],
    placements: Array.isArray(record?.placements) ? record.placements : [],
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
  const byPlacement = new Map();
  const linkSlugs = new Set(links.map(link => link.slug));
  const start = days === 'all' ? 0 : Date.parse(new Date(now).toISOString().slice(0, 10)) - (Number(days) - 1) * 86400000;
  const previousStart = days === 'all' ? 0 : start - Number(days) * 86400000;
  let previousClicks = 0;
  for (const link of links) {
    for (const placement of link.placements || []) {
      byPlacement.set(`${link.slug}/${placement.id}`, { ...placement, slug: link.slug, clicks: 0 });
    }
  }
  function addUnassigned(channel, clicks){
    const key = `/${channel}`;
    if (!byPlacement.has(key)) byPlacement.set(key, { id: '', slug, name: 'Unassigned', channel, clicks: 0 });
    byPlacement.get(key).clicks += clicks;
  }
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
    if (timestamp < start) {
      if (timestamp >= previousStart) previousClicks += 1;
      return;
    }
    const date = new Date(timestamp).toISOString().slice(0, 10);
    if (!byDate.has(date)) byDate.set(date, { date, clicks: 0, linkClicks: 0, qrScans: 0, unknownClicks: 0 });
    const row = byDate.get(date);
    const channel = item.channel === 'qr' ? 'qrScans' : item.channel === 'link' ? 'linkClicks' : 'unknownClicks';
    const placement = item.placementId && byPlacement.get(`${item.slug}/${item.placementId}`);
    if (placement && placement.channel === item.channel) placement.clicks += 1;
    else addUnassigned(channel === 'unknownClicks' ? 'unknown' : item.channel, 1);
    totals.clicks += 1;
    totals[channel] += 1;
    row.clicks += 1;
    row[channel] += 1;
  });
  const unattributedHistoricalClicks = Math.max(0, totals.lifetimeClicks - detailedEvents) + invalidDateEvents;
  if (days === 'all') {
    totals.unknownClicks += unattributedHistoricalClicks;
    totals.clicks += unattributedHistoricalClicks;
    if (unattributedHistoricalClicks) addUnassigned('unknown', unattributedHistoricalClicks);
  }
  const complete = configured && !truncated && unattributedHistoricalClicks === 0;
  let note = 'Link and QR activity reflects recorded requests, including automated visits. Older unmarked requests remain combined activity.';
  if (!configured) note = 'Detailed tracking is not configured. Lifetime totals are available; dates and channels are unknown.';
  else if (truncated) note = 'Detailed history reached the query limit. Date and channel totals are partial; choose a single link for a smaller report.';
  else if (unattributedHistoricalClicks) note = 'Some lifetime activity has no dated event history and cannot be assigned to a date or channel. Date filters show recorded activity only.';
  return {
    days, slug, totals,
    placements: [...byPlacement.values()].sort((a, b) => b.clicks - a.clicks || a.name.localeCompare(b.name)),
    previous: days === 'all' ? null : {
      clicks: previousClicks, start: new Date(previousStart).toISOString().slice(0, 10),
      end: new Date(start - 86400000).toISOString().slice(0, 10), complete
    },
    daily: [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date)),
    completeness: {
      complete, truncated, detailedTrackingConfigured: configured, unattributedHistoricalClicks,
      detailedEvents, retentionVerified: false, note
    }
  };
}

module.exports = { QR_DESIGN_MAX_BYTES, TEXT_FIELDS, buildAnalyticsReport, buildLinkPatch, nextLinkTimestamp, normalizeQrDesign, normalizeTags, revisedDestinations, serializeLink };
