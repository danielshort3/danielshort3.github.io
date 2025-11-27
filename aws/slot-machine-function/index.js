const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  GetCommand,
  PutCommand,
  UpdateCommand,
  QueryCommand,
  DeleteCommand,
  BatchWriteCommand
} = require('@aws-sdk/lib-dynamodb');
const { randomUUID, randomBytes, createHmac, timingSafeEqual, createHash } = require('crypto');
const bcrypt = require('bcryptjs');
const { spin: runSlotSpin, machineMetadata } = require('./slot-engine');
const {
  rollDrops,
  applyDrops,
  getDropTable,
  getTierWeights,
  constants: DROP_CONSTANTS
} = require('./drop-engine');
let UPGRADE_DEFINITIONS;
try {
  UPGRADE_DEFINITIONS = require('../../slot-config/upgrade-definitions.json');
} catch (error) {
  if (error.code === 'MODULE_NOT_FOUND') {
    UPGRADE_DEFINITIONS = require('./upgrade-definitions.js');
  } else {
    throw error;
  }
}

const dynamo = DynamoDBDocumentClient.from(new DynamoDBClient({}), {
  marshallOptions: { removeUndefinedValues: true }
});

const {
  TABLE_NAME,
  USERS_TABLE,
  HISTORY_TABLE,
  STARTING_CREDITS = '1000',
  MAX_BET = '100',
  ALLOWED_ORIGINS = '',
  SESSION_TTL_MINUTES = '4320',
  PASSWORD_SALT_ROUNDS = '12'
} = process.env;

const STARTING_BALANCE = Math.max(parseInt(STARTING_CREDITS, 10) || 1000, 1);
const BET_LIMIT = Math.max(parseInt(MAX_BET, 10) || 100, 1);
const SESSION_TTL_MIN = Math.max(parseInt(SESSION_TTL_MINUTES, 10) || 4320, 5);
const PASSWORD_COST = Math.max(parseInt(PASSWORD_SALT_ROUNDS, 10) || 12, 4);
const allowedOrigins = ALLOWED_ORIGINS.split(',').map(origin => origin.trim()).filter(Boolean);
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MACHINE_META = machineMetadata();
const UPGRADE_LIMITS = MACHINE_META.upgrades || {
  baseRows: MACHINE_META.rows || 3,
  maxRows: MACHINE_META.rows || 3,
  baseReels: MACHINE_META.reels || 3,
  maxReels: MACHINE_META.reels || 3,
  costs: { rows: [], reels: [], lines: [] }
};
const MAX_LINE_TIER = MACHINE_META.lineTier || 3;
const BASE_SYMBOL_COUNT = MACHINE_META.baseSymbolCount || 5;
const DAY_MS = 24 * 60 * 60 * 1000;
const DAILY_CALENDAR_LENGTH = 7;
const DAILY_VIP_BASE = 35;
const DAILY_VIP_STEP = 22;
const DAILY_CREDIT_BASE = 250;
const DEBUG_EMAIL = 'danielshort3@gmail.com';
const DEBUG_COIN_AMOUNT = 100000;
const UPGRADE_INDEX = new Map();
const DEFAULT_UPGRADES = {};
const UPGRADE_COST_GROWTH = 1.85;
UPGRADE_DEFINITIONS.forEach(def => {
  if (!def?.key) return;
  const defaultLevel = Number.isFinite(def.defaultLevel) ? def.defaultLevel : 0;
  DEFAULT_UPGRADES[def.key] = defaultLevel;
  UPGRADE_INDEX.set(def.key, def);
  UPGRADE_INDEX.set(def.key.toLowerCase(), def);
});

const SYMBOL_LABELS = Object.fromEntries(
  (MACHINE_META.symbols || []).map(symbol => [symbol.key, symbol.label || symbol.key])
);

const nowIso = () => new Date().toISOString();

function getUpgradeDefinition(type = '') {
  if (typeof type !== 'string') return null;
  const key = type.trim();
  if (!key) return null;
  return UPGRADE_INDEX.get(key) || UPGRADE_INDEX.get(key.toLowerCase()) || null;
}

function resolveUpgradeMax(def) {
  if (!def) return 0;
  if (def.clientOnly) return Number.isFinite(def.max) ? def.max : 0;
  if (def.dynamicMax === 'premiumSymbols') {
    const symbolCount = (MACHINE_META.symbols || []).filter(entry => entry.key !== 'wild').length;
    return Math.max(0, symbolCount - BASE_SYMBOL_COUNT);
  }
  if (def.dynamicMax === 'cullSymbols') {
    return Math.max(0, BASE_SYMBOL_COUNT - 1);
  }
  return Number.isFinite(def.max) ? def.max : 0;
}

function normalizeUpgrades(source = {}) {
  const normalized = {};
  UPGRADE_DEFINITIONS.forEach(def => {
    if (def.clientOnly) return;
    const hasValue = source && Number.isFinite(source[def.key]);
    const raw = hasValue ? source[def.key] : (DEFAULT_UPGRADES[def.key] ?? 0);
    const safe = Math.max(0, Math.floor(raw));
    const max = resolveUpgradeMax(def);
    normalized[def.key] = Math.min(safe, max);
  });
  return normalized;
}

function computeUpgradeCost(def, level = 0) {
  if (!def) return 0;
  const base = def.cost || 0;
  const growth = Number.isFinite(def.costGrowth) ? def.costGrowth : UPGRADE_COST_GROWTH;
  const step = Math.max(0, Math.floor(level));
  return Math.round(base * Math.pow(Math.max(1, growth), step));
}

function computeActiveSymbols(upgrades = {}) {
  const list = (MACHINE_META.symbols || []).filter(symbol => symbol.key !== 'wild');
  const disable = Math.max(0, Math.floor(upgrades.disable || 0));
  const premium = Math.max(0, Math.floor(upgrades.premium || 0));
  const start = Math.min(disable, list.length - 1);
  const count = Math.min(list.length - start, BASE_SYMBOL_COUNT + premium);
  const pool = list.slice(start, start + count).map(entry => entry.key);
  if (upgrades.wildUnlock) pool.push('wild');
  return pool;
}

function computeMaxBet(upgrades = {}) {
  const level = Math.max(0, Math.floor(upgrades.betMultiplier || 0));
  return BET_LIMIT * Math.max(1, 10 ** level);
}

function dropSkillSpec(upgrades = {}) {
  const duration = Math.max(0, Math.floor(upgrades.dropRateDuration || 0));
  const durationMs = 60000 * (1 + duration);
  return {
    durationMs,
    cooldownMs: durationMs * 2,
    multiplier: 1 + DROP_CONSTANTS.DROP_BOOST_SCALE * (1 + Math.max(0, upgrades.dropRateEffect || 0))
  };
}

function normalizeSkillState(skillState = {}, nowMs = Date.now()) {
  const next = {
    dropRate: {
      activeUntil: 0,
      cooldownUntil: 0
    }
  };
  if (skillState.dropRate) {
    const active = Number(skillState.dropRate.activeUntil) || 0;
    const cooldown = Number(skillState.dropRate.cooldownUntil) || 0;
    next.dropRate.activeUntil = active > nowMs ? active : 0;
    next.dropRate.cooldownUntil = cooldown > nowMs ? cooldown : 0;
  }
  return next;
}

function evaluateSkills({ payload = {}, upgrades = {}, skillState = {}, nowMs = Date.now() }) {
  const normalized = normalizeSkillState(skillState, nowMs);
  const next = { ...normalized };
  const dropSpec = dropSkillSpec(upgrades);
  let dropRateActive = next.dropRate.activeUntil > nowMs;

  if (next.dropRate.cooldownUntil && next.dropRate.cooldownUntil <= nowMs) {
    next.dropRate.cooldownUntil = 0;
  }
  const canActivate = payload?.activeSkills?.dropRate && (upgrades.dropBoostUnlock || 0) > 0;
  const cooldownActive = next.dropRate.cooldownUntil && next.dropRate.cooldownUntil > nowMs;
  if (canActivate && !dropRateActive && !cooldownActive) {
    next.dropRate.activeUntil = nowMs + dropSpec.durationMs;
    next.dropRate.cooldownUntil = next.dropRate.activeUntil + dropSpec.cooldownMs;
    dropRateActive = true;
  }
  if (next.dropRate.activeUntil <= nowMs) {
    dropRateActive = false;
  }

  return {
    skillState: next,
    dropRateActive
  };
}

const startOfDayMs = (ms = Date.now()) => {
  const date = new Date(ms);
  date.setUTCHours(0, 0, 0, 0);
  return date.getTime();
};

function normalizeDaily(daily = {}, nowMs = Date.now()) {
  const today = startOfDayMs(nowMs);
  const rawStreak = Number.isFinite(daily.streak) ? daily.streak : 1;
  const streak = Math.max(1, Math.min(DAILY_CALENDAR_LENGTH, Math.floor(rawStreak)));
  const last = Number.isFinite(daily.lastClaimMs) ? daily.lastClaimMs : 0;
  const lastClaimMs = last ? startOfDayMs(last) : 0;
  const claimedToday = lastClaimMs === today && !!daily.claimedToday;
  return { streak, lastClaimMs, claimedToday };
}

function advanceDaily(daily = {}, nowMs = Date.now()) {
  const today = startOfDayMs(nowMs);
  const normalized = normalizeDaily(daily, nowMs);
  let { streak } = normalized;
  let { claimedToday } = normalized;
  let lastClaimMs = normalized.lastClaimMs;
  let changed = false;

  if (!lastClaimMs) {
    streak = 1;
    claimedToday = false;
    changed = true;
  } else if (lastClaimMs > today) {
    lastClaimMs = today;
    claimedToday = false;
    changed = true;
  } else if (today > lastClaimMs) {
    const diff = Math.max(1, Math.round((today - lastClaimMs) / DAY_MS));
    streak = diff === 1 ? Math.min(DAILY_CALENDAR_LENGTH, streak + 1) : 1;
    claimedToday = false;
    changed = true;
  }

  return {
    streak,
    lastClaimMs,
    claimedToday,
    changed,
    nextResetAt: today + DAY_MS
  };
}

function dailyRewardFor(streak = 1) {
  const dayIndex = Math.max(
    0,
    Math.min(DAILY_CALENDAR_LENGTH - 1, (Math.floor(streak) - 1) % DAILY_CALENDAR_LENGTH)
  );
  const vipMarks = DAILY_VIP_BASE + DAILY_VIP_STEP * dayIndex;
  const credits = DAILY_CREDIT_BASE + (dayIndex * 50);
  const drops = [
    { type: 'vipMarks', amount: vipMarks, name: 'VIP Mark' }
  ];
  if (dayIndex === DAILY_CALENDAR_LENGTH - 1) {
    drops.push({ type: 'reelMod', tier: 2, amount: 1, name: 'Tier 2 Reel Mod' });
  } else if (dayIndex === 2 || dayIndex === 4) {
    drops.push({ type: 'spinBooster', tier: 1, amount: 1, name: 'Spin Booster' });
  }
  return {
    day: dayIndex + 1,
    vipMarks,
    credits,
    drops
  };
}

function formatDailyPayload(daily = {}, nowMs = Date.now()) {
  const state = advanceDaily(daily, nowMs);
  return {
    streak: state.streak,
    claimedToday: state.claimedToday,
    ready: !state.claimedToday,
    lastClaimMs: state.lastClaimMs || null,
    nextResetAt: state.nextResetAt,
    todayReward: dailyRewardFor(state.streak)
  };
}

async function syncDailyState(player, nowMs = Date.now()) {
  const state = advanceDaily(player.daily || {}, nowMs);
  const daily = {
    streak: state.streak,
    lastClaimMs: state.lastClaimMs,
    claimedToday: state.claimedToday
  };
  const needsPersist = !player.daily
    || player.daily.streak !== daily.streak
    || player.daily.lastClaimMs !== daily.lastClaimMs
    || player.daily.claimedToday !== daily.claimedToday;

  player.daily = daily;
  if (!needsPersist) return player;

  const updated = await dynamo.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { playerId: player.playerId },
    UpdateExpression: 'SET daily = :daily, updatedAt = :now',
    ExpressionAttributeValues: {
      ':daily': daily,
      ':now': nowIso()
    },
    ReturnValues: 'ALL_NEW'
  }));
  const attrs = updated.Attributes || player;
  attrs.daily = daily;
  return attrs;
}

async function settleIdleIncome(player, { maxCoins = null } = {}) {
  if (!player || !player.playerId) return { player, gained: 0 };
  const upgrades = normalizeUpgrades(player.upgrades || DEFAULT_UPGRADES);
  const idleLevel = upgrades.idle || 0;
  const nowMs = Date.now();
  const fallbackStamp = player.idleCheckpoint || player.updatedAt || player.createdAt || nowIso();
  let checkpointMs = Date.parse(fallbackStamp);
  if (!Number.isFinite(checkpointMs)) checkpointMs = nowMs;
  const stampNow = new Date(nowMs).toISOString();
  if (!idleLevel) {
    if (!player.idleCheckpoint) {
      await dynamo.send(new UpdateCommand({
        TableName: TABLE_NAME,
        Key: { playerId: player.playerId },
        UpdateExpression: 'SET idleCheckpoint = :checkpoint',
        ExpressionAttributeValues: { ':checkpoint': stampNow }
      })).catch(() => {});
      player.idleCheckpoint = stampNow;
    }
    return { player, gained: 0 };
  }
  const elapsedSeconds = Math.max(0, Math.floor((nowMs - checkpointMs) / 1000));
  if (elapsedSeconds <= 0) {
    return { player, gained: 0 };
  }
  const availableCoins = idleLevel * elapsedSeconds;
  let grantable = availableCoins;
  if (Number.isFinite(maxCoins)) {
    const allowed = idleLevel * Math.floor(Math.max(0, Math.floor(maxCoins)) / idleLevel);
    grantable = Math.min(grantable, allowed);
  }
  if (!grantable) {
    return { player, gained: 0 };
  }
  const secondsConsumed = Math.max(1, Math.floor(grantable / idleLevel));
  const remainderSeconds = Math.max(0, elapsedSeconds - secondsConsumed);
  const checkpointStamp = new Date(nowMs - remainderSeconds * 1000).toISOString();
  const updated = await dynamo.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { playerId: player.playerId },
    UpdateExpression: 'SET credits = :credits, idleCheckpoint = :checkpoint, updatedAt = :now',
    ExpressionAttributeValues: {
      ':credits': (player.credits || 0) + grantable,
      ':checkpoint': checkpointStamp,
      ':now': stampNow
    },
    ReturnValues: 'ALL_NEW'
  }));
  const attrs = updated.Attributes || player;
  attrs.idleCheckpoint = checkpointStamp;
  attrs.idleGained = grantable;
  if (!attrs.inventory) attrs.inventory = player.inventory || {};
  if (!attrs.lastDrops) attrs.lastDrops = player.lastDrops || [];
  return { player: attrs, gained: grantable };
}

function computeDimensions(upgrades = DEFAULT_UPGRADES) {
  const rows = Math.min(
    UPGRADE_LIMITS.maxRows || UPGRADE_LIMITS.baseRows || MACHINE_META.rows || 3,
    (UPGRADE_LIMITS.baseRows || MACHINE_META.rows || 3) + (upgrades.rows || 0)
  );
  const reels = Math.min(
    UPGRADE_LIMITS.maxReels || UPGRADE_LIMITS.baseReels || MACHINE_META.reels || 3,
    (UPGRADE_LIMITS.baseReels || MACHINE_META.reels || 3) + (upgrades.reels || 0)
  );
  const lineTier = Math.min(MAX_LINE_TIER, (upgrades.lines || 0));
  return { rows, reels, lineTier };
}

function httpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  error.expose = true;
  return error;
}

function base64UrlDecode(input = '') {
  const normalized = input.replace(/-/g, '+').replace(/_/g, '/');
  return Buffer.from(normalized, 'base64');
}

function deriveSyncHmacKey(token = '', salt = Buffer.alloc(0)) {
  return createHash('sha256').update(Buffer.concat([Buffer.from(token), salt])).digest();
}

function verifySyncSignature(token, snapshot, signature = {}) {
  if (!signature || !signature.salt || !signature.value) {
    throw httpError(401, 'Missing sync signature.');
  }
  const salt = base64UrlDecode(signature.salt);
  const provided = base64UrlDecode(signature.value);
  const data = Buffer.from(JSON.stringify(snapshot || {}));
  const key = deriveSyncHmacKey(token, salt);
  const hmac = createHmac('sha256', key).update(data).digest();
  if (hmac.length !== provided.length || !timingSafeEqual(hmac, provided)) {
    throw httpError(401, 'Invalid sync signature.');
  }
}

function resolveCorsOrigin(requestOrigin = '') {
  if (allowedOrigins.length === 0 || allowedOrigins.includes('*')) {
    return requestOrigin || '*';
  }
  return allowedOrigins.includes(requestOrigin) ? requestOrigin : allowedOrigins[0];
}

function buildHeaders(origin) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'OPTIONS,POST',
    'Access-Control-Max-Age': '86400',
    'Content-Type': 'application/json'
  };
}

const respond = (statusCode, payload, origin) => ({
  statusCode,
  headers: buildHeaders(origin),
  body: JSON.stringify(payload)
});

function parseBody(event = {}) {
  if (!event.body) return {};
  const rawBody = event.isBase64Encoded ? Buffer.from(event.body, 'base64').toString('utf8') : event.body;
  try {
    return JSON.parse(rawBody);
  } catch {
    return {};
  }
}

function sanitizePlayerId(value = '') {
  const trimmed = value.toString().trim();
  return /^[A-Za-z0-9_-]{4,64}$/.test(trimmed) ? trimmed : '';
}

function sanitizeUsername(value = '') {
  return value.toString().trim().toLowerCase();
}

function validatePassword(password = '') {
  return typeof password === 'string' && password.length >= 8;
}

function summarizeGroups(groups = []) {
  if (!groups.length) return 'No match';
  const sorted = [...groups].sort((a, b) => (b.payout || 0) - (a.payout || 0));
  const best = sorted[0];
  if (!best) return 'No match';
  if (best.symbol === 'bonus') {
    return `Bonus ${best.count}×`;
  }
  const label = SYMBOL_LABELS[best.symbol] || best.symbol;
  return `${label} ${best.count}×`;
}

function buildSpinPayload(engineResult, bet, timestamp) {
  const { outcome, payout, groups, metadata } = engineResult;
  return {
    grid: outcome,
    rows: metadata.rows,
    reels: metadata.reels,
    lineTier: metadata.lineTier,
    lines: metadata.lines,
    activeSymbols: metadata.activeSymbols,
    winGroups: groups,
    bet,
    winAmount: payout,
    multiplier: null,
    outcome: summarizeGroups(groups),
    timestamp
  };
}

async function fetchPlayer(playerId) {
  const result = await dynamo.send(new GetCommand({
    TableName: TABLE_NAME,
    Key: { playerId }
  }));
  return result.Item || null;
}

async function getOrCreatePlayer(playerId) {
  let player = await fetchPlayer(playerId);
  if (player) {
    if (!player.upgrades) {
      player.upgrades = { ...DEFAULT_UPGRADES };
    }
    if (!player.inventory) {
      player.inventory = {};
    }
    if (!player.lastDrops) {
      player.lastDrops = [];
    }
    if (!player.daily) {
      player.daily = { streak: 1, lastClaimMs: 0, claimedToday: false };
    }
    return player;
  }

  const now = nowIso();
  const newPlayer = {
    playerId,
    credits: STARTING_BALANCE,
    spins: 0,
    upgrades: { ...DEFAULT_UPGRADES },
    inventory: {},
    lastDrops: [],
    daily: { streak: 1, lastClaimMs: 0, claimedToday: false },
    snapshotRev: 0,
    snapshotData: null,
    createdAt: now,
    updatedAt: now
  };

  try {
    await dynamo.send(new PutCommand({
      TableName: TABLE_NAME,
      Item: newPlayer,
      ConditionExpression: 'attribute_not_exists(playerId)'
    }));
    return newPlayer;
  } catch (error) {
    if (error.code === 'ConditionalCheckFailedException') {
      player = await fetchPlayer(playerId);
      if (player) return player;
    }
    throw error;
  }
}

async function applySpin(player, bet, spinPayload, extra = {}) {
  const now = spinPayload?.timestamp || nowIso();
  const lastSpin = { ...spinPayload, timestamp: now };
  const winnings = Number(lastSpin.winAmount) || 0;
  const nextCredits = player.credits - bet + winnings;

  let current = player;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const setParts = [
        'credits = :credits',
        'spins = if_not_exists(spins, :zero) + :one',
        'lastSpin = :spin',
        'updatedAt = :now'
      ];
      const values = {
        ':credits': nextCredits,
        ':expected': current.credits,
        ':spin': lastSpin,
        ':one': 1,
        ':zero': 0,
        ':now': now
      };
      if (extra.inventory !== undefined) {
        setParts.push('inventory = :inventory');
        values[':inventory'] = extra.inventory;
      }
      if (extra.lastDrops !== undefined) {
        setParts.push('lastDrops = :lastDrops');
        values[':lastDrops'] = extra.lastDrops;
      }
      if (extra.skillState !== undefined) {
        setParts.push('skillState = :skillState');
        values[':skillState'] = extra.skillState;
      }
      const updated = await dynamo.send(new UpdateCommand({
        TableName: TABLE_NAME,
        Key: { playerId: current.playerId },
        UpdateExpression: `SET ${setParts.join(', ')}`,
        ConditionExpression: 'credits = :expected',
        ExpressionAttributeValues: values,
        ReturnValues: 'ALL_NEW'
      }));
      return updated.Attributes;
    } catch (error) {
      if (error.code === 'ConditionalCheckFailedException') {
        current = await fetchPlayer(current.playerId);
        if (!current) throw new Error('Player missing');
        continue;
      }
      throw error;
    }
  }
  throw new Error('Balance update conflict');
}

function formatPlayerPayload(player, extra = {}) {
  const nowMs = Date.now();
  const upgrades = normalizeUpgrades(player?.upgrades || DEFAULT_UPGRADES);
  const dims = computeDimensions(upgrades);
  const activeSymbols = computeActiveSymbols(upgrades);
  const skillState = normalizeSkillState(player?.skillState || {}, nowMs);
  const daily = formatDailyPayload(player?.daily || {}, nowMs);
  let snapshot = null;
  if (player?.snapshotData) {
    try {
      snapshot = JSON.parse(player.snapshotData);
      if (typeof player.snapshotRev === 'number') {
        snapshot.rev = player.snapshotRev;
      }
    } catch {
      snapshot = null;
    }
  }
  return {
    playerId: player.playerId,
    balance: player.credits,
    spins: player.spins || 0,
    lastSpin: player.lastSpin || null,
    startingBalance: STARTING_BALANCE,
    maxBet: computeMaxBet(upgrades),
    upgrades,
    currentRows: dims.rows,
    currentReels: dims.reels,
    currentLineTier: dims.lineTier,
    machine: {
      ...MACHINE_META,
      currentRows: dims.rows,
      currentReels: dims.reels,
      currentLineTier: dims.lineTier,
      activeSymbols
    },
    dropState: {
      inventory: player.inventory || {},
      lastDrops: player.lastDrops || [],
      tableKey: MACHINE_META.id,
      table: getDropTable(MACHINE_META.id),
      tierWeights: getTierWeights(),
      constants: DROP_CONSTANTS
    },
    daily,
    skillState,
    snapshot,
    ...extra
  };
}

function createPlayerId(prefix = 'slot') {
  return `${prefix}_${(randomUUID && randomUUID()) || randomBytes(8).toString('hex')}`;
}

async function deleteHistoryEntries(playerId) {
  if (!HISTORY_TABLE || !playerId) return;
  let lastKey;
  do {
    const res = await dynamo.send(new QueryCommand({
      TableName: HISTORY_TABLE,
      KeyConditionExpression: 'playerId = :pid',
      ExpressionAttributeValues: { ':pid': playerId },
      ProjectionExpression: 'playerId, spinTime',
      ExclusiveStartKey: lastKey
    }));
    lastKey = res.LastEvaluatedKey;
    const items = res.Items || [];
    if (!items.length) continue;
    for (let i = 0; i < items.length; i += 25) {
      const chunk = items.slice(i, i + 25);
      const requestItems = chunk.map(item => ({
        DeleteRequest: { Key: { playerId: item.playerId, spinTime: item.spinTime } }
      }));
      await dynamo.send(new BatchWriteCommand({
        RequestItems: { [HISTORY_TABLE]: requestItems }
      }));
    }
  } while (lastKey);
}

async function getUser(username) {
  if (!username) return null;
  const result = await dynamo.send(new GetCommand({
    TableName: USERS_TABLE,
    Key: { username }
  }));
  return result.Item || null;
}

async function ensureUserPlayer(user) {
  if (user.playerId) return user.playerId;
  const playerId = createPlayerId('user');
  const updated = await dynamo.send(new UpdateCommand({
    TableName: USERS_TABLE,
    Key: { username: user.username },
    UpdateExpression: 'SET playerId = if_not_exists(playerId, :playerId)',
    ExpressionAttributeValues: { ':playerId': playerId },
    ReturnValues: 'ALL_NEW'
  }));
  await getOrCreatePlayer(updated.Attributes.playerId);
  return updated.Attributes.playerId;
}

async function issueSession(username) {
  const token = randomBytes(32).toString('hex');
  const expiresAt = new Date(Date.now() + SESSION_TTL_MIN * 60 * 1000).toISOString();
  await dynamo.send(new UpdateCommand({
    TableName: USERS_TABLE,
    Key: { username },
    UpdateExpression: 'SET sessionToken = :token, sessionExpiresAt = :expires',
    ExpressionAttributeValues: {
      ':token': token,
      ':expires': expiresAt
    }
  }));
  return { token, expiresAt };
}

async function revokeSession(username) {
  await dynamo.send(new UpdateCommand({
    TableName: USERS_TABLE,
    Key: { username },
    UpdateExpression: 'REMOVE sessionToken, sessionExpiresAt'
  }));
}

async function resolveAuth(token, { required = false } = {}) {
  if (!token) {
    if (required) throw httpError(401, 'Missing auth token.');
    return null;
  }

  const query = await dynamo.send(new QueryCommand({
    TableName: USERS_TABLE,
    IndexName: 'sessionToken-index',
    KeyConditionExpression: 'sessionToken = :token',
    ExpressionAttributeValues: { ':token': token }
  }));
  const user = query.Items?.[0];
  if (!user) {
    if (required) throw httpError(401, 'Invalid or expired session.');
    return null;
  }

  if (user.sessionExpiresAt && new Date(user.sessionExpiresAt).getTime() < Date.now()) {
    await revokeSession(user.username);
    if (required) throw httpError(401, 'Session expired. Please log in again.');
    return null;
  }

  const playerId = await ensureUserPlayer(user);
  const player = await getOrCreatePlayer(playerId);
  return { username: user.username, token, player };
}

async function logSpinHistory(entry = {}) {
  if (!HISTORY_TABLE || !entry.playerId) return;
  const item = {
    playerId: entry.playerId,
    spinTime: entry.spinTime || nowIso(),
    username: entry.username || null,
    bet: entry.bet ?? null,
    reels: entry.reels || [],
    rows: entry.rows ?? null,
    outcome: entry.outcome || null,
    multiplier: entry.multiplier ?? null,
    winAmount: entry.winAmount ?? 0,
    balanceBefore: entry.balanceBefore ?? null,
    balanceAfter: entry.balanceAfter ?? null,
    winGroups: entry.winGroups || null,
    errorCode: entry.errorCode || null
  };
  try {
    await dynamo.send(new PutCommand({
      TableName: HISTORY_TABLE,
      Item: item
    }));
  } catch (error) {
    console.error('History log error', error);
  }
}

async function handleSession(payload = {}) {
  if ((payload.action === 'debugCoins') && payload.token) {
    return handleDebugCoins(payload);
  }
  if ((payload.action === 'upgrade' || payload.upgradeType || payload.type) && payload.token) {
    return handleUpgrade({
      token: payload.token,
      type: payload.type || payload.upgradeType || payload.actionType
    });
  }
  if (payload.token) {
    const auth = await resolveAuth(payload.token, { required: true });
    const idleResult = await settleIdleIncome(auth.player);
    const playerWithDaily = await syncDailyState(idleResult.player);
    return formatPlayerPayload(playerWithDaily, {
      username: auth.username,
      token: auth.token,
      idleGained: idleResult.gained || 0
    });
  }

  let playerId = sanitizePlayerId(payload.playerId);
  if (!playerId) {
    playerId = createPlayerId('slot');
  }
  const player = await getOrCreatePlayer(playerId);
  const idleResult = await settleIdleIncome(player);
  const playerWithDaily = await syncDailyState(idleResult.player);
  return formatPlayerPayload(playerWithDaily, { idleGained: idleResult.gained || 0 });
}

async function handleDaily(payload = {}) {
  const auth = await resolveAuth(payload.token, { required: true });
  let player = await syncDailyState(auth.player);
  const daily = player.daily || { streak: 1, lastClaimMs: 0, claimedToday: false };
  if (payload.action === 'claim' || payload.claim === true) {
    if (daily.claimedToday) {
      return {
        errorCode: 'DAILY_CLAIMED',
        message: 'Daily reward already claimed.',
        daily: formatDailyPayload(daily)
      };
    }
    const reward = dailyRewardFor(daily.streak);
    const inventory = applyDrops(player.inventory || {}, reward.drops || []);
    const dailyRecord = {
      streak: daily.streak,
      lastClaimMs: Date.now(),
      claimedToday: true
    };
    const updated = await dynamo.send(new UpdateCommand({
      TableName: TABLE_NAME,
      Key: { playerId: player.playerId },
      UpdateExpression: 'SET credits = :credits, inventory = :inventory, daily = :daily, lastDrops = :drops, updatedAt = :now',
      ExpressionAttributeValues: {
        ':credits': (player.credits || 0) + (reward.credits || 0),
        ':inventory': inventory,
        ':daily': dailyRecord,
        ':drops': reward.drops || [],
        ':now': nowIso()
      },
      ReturnValues: 'ALL_NEW'
    }));
    const attrs = updated.Attributes || player;
    attrs.daily = dailyRecord;
    attrs.lastDrops = reward.drops || [];
    return formatPlayerPayload(
      attrs,
      { username: auth.username, token: auth.token, reward }
    );
  }
  return formatPlayerPayload(player, { username: auth.username, token: auth.token });
}

async function handleSync(payload = {}) {
  const auth = await resolveAuth(payload.token, { required: true });
  const player = auth.player;
  const currentRev = Number.isFinite(player.snapshotRev) ? player.snapshotRev : 0;
  let storedSnapshot = null;
  if (player.snapshotData) {
    try {
      storedSnapshot = JSON.parse(player.snapshotData);
      if (typeof storedSnapshot.rev !== 'number') storedSnapshot.rev = currentRev;
    } catch {
      storedSnapshot = null;
    }
  }
  const incoming = payload.snapshot;
  if (!incoming || typeof incoming !== 'object') {
    return { rev: currentRev, snapshot: storedSnapshot };
  }
  verifySyncSignature(auth.token, incoming, payload.signature);
  const incomingRev = Number.isFinite(incoming.rev) ? incoming.rev : 0;
  if (incomingRev < currentRev && storedSnapshot) {
    return { rev: currentRev, snapshot: storedSnapshot, conflict: true };
  }
  const nextRev = Math.max(currentRev, incomingRev) + 1;
  const toStore = { ...incoming, rev: nextRev };
  await dynamo.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { playerId: player.playerId },
    UpdateExpression: 'SET snapshotRev = :rev, snapshotData = :data, updatedAt = :now',
    ExpressionAttributeValues: {
      ':rev': nextRev,
      ':data': JSON.stringify(toStore),
      ':now': nowIso()
    }
  }));
  return { rev: nextRev, snapshot: toStore };
}

async function handleSpin(payload = {}) {
  const auth = payload.token ? await resolveAuth(payload.token, { required: true }) : null;
  const playerId = auth?.player?.playerId || sanitizePlayerId(payload.playerId);
  const bet = Math.trunc(Number(payload.bet));

  if (!playerId) {
    return { errorCode: 'BAD_REQUEST', message: 'Missing playerId.', maxBet: BET_LIMIT };
  }
  if (!Number.isFinite(bet) || bet <= 0) {
    if (playerId) {
      await logSpinHistory({ playerId, username: auth?.username || null, bet, errorCode: 'BAD_REQUEST' });
    }
    return { errorCode: 'BAD_REQUEST', message: 'Bet must be a positive integer.', maxBet: BET_LIMIT };
  }
  let player = await getOrCreatePlayer(playerId);
  const claimedCoins = Number.isFinite(Number(payload.clientIdleCoins))
    ? Math.max(0, Math.floor(Number(payload.clientIdleCoins)))
    : null;
  const idleResult = await settleIdleIncome(player, { maxCoins: claimedCoins });
  player = await syncDailyState(idleResult.player);
  if (!player.inventory) {
    player.inventory = {};
  }
  if (!player.lastDrops) {
    player.lastDrops = [];
  }
  if (payload.pendingUpgrades) {
    player = await applyPendingUpgrades(player, payload.pendingUpgrades);
  }
  const upgrades = normalizeUpgrades(player.upgrades || auth?.player?.upgrades || DEFAULT_UPGRADES);
  const maxBet = computeMaxBet(upgrades);
  if (bet > maxBet) {
    if (playerId) {
      await logSpinHistory({ playerId, username: auth?.username || null, bet, errorCode: 'LIMIT_EXCEEDED' });
    }
    return { errorCode: 'LIMIT_EXCEEDED', message: `Bet cannot exceed ${maxBet}.`, maxBet };
  }
  if (player.credits < bet) {
    await logSpinHistory({
      playerId,
      username: auth?.username || null,
      bet,
      balanceBefore: player.credits,
      balanceAfter: player.credits,
      errorCode: 'INSUFFICIENT_CREDITS'
    });
    return {
      errorCode: 'INSUFFICIENT_CREDITS',
      message: 'You do not have enough credits.',
      balance: player.credits,
      maxBet
    };
  }

  const dimensions = computeDimensions(upgrades);
  const engineResult = runSlotSpin(bet, {
    rows: dimensions.rows,
    reels: dimensions.reels,
    lineTier: dimensions.lineTier,
    upgrades
  });
  let finalResult = engineResult;
  let retriggered = false;
  if ((upgrades.retrigger || upgrades.retriggerQuality) && engineResult.payout === 0) {
    const base = 0.05 * Math.max(0, upgrades.retrigger || 0);
    const bonus = 0.05 * Math.max(0, upgrades.retriggerQuality || 0);
    const chance = Math.min(0.95, base + bonus);
    if (chance > 0 && Math.random() < chance) {
      retriggered = true;
      finalResult = runSlotSpin(bet, {
        rows: dimensions.rows,
        reels: dimensions.reels,
        lineTier: dimensions.lineTier,
        upgrades
      });
    }
  }
  const nowMs = Date.now();
  const timestamp = new Date(nowMs).toISOString();
  const spinPayload = buildSpinPayload(finalResult, bet, timestamp);
  spinPayload.rows = dimensions.rows;
  spinPayload.reels = dimensions.reels;
  spinPayload.lineTier = dimensions.lineTier;
  spinPayload.retriggered = retriggered;
  const skillEval = evaluateSkills({
    payload,
    upgrades,
    skillState: player.skillState,
    nowMs
  });
  const dropResult = rollDrops({
    bet,
    payout: finalResult.payout,
    upgrades,
    skillActive: skillEval.dropRateActive,
    tableKey: MACHINE_META.id
  });
  const inventory = applyDrops(player.inventory, dropResult.drops);
  spinPayload.drops = dropResult.drops;
  spinPayload.dropMultiplier = dropResult.multiplier;
  const updatedPlayer = await applySpin(player, bet, spinPayload, {
    inventory,
    lastDrops: dropResult.drops,
    skillState: skillEval.skillState
  });
  await logSpinHistory({
    playerId,
    username: auth?.username || null,
    bet,
    reels: spinPayload.grid,
    rows: spinPayload.rows,
    outcome: spinPayload.outcome,
    winGroups: spinPayload.winGroups,
    winAmount: spinPayload.winAmount,
    drops: dropResult.drops?.length || 0,
    balanceBefore: player.credits,
    balanceAfter: updatedPlayer.credits
  });

  return formatPlayerPayload(
    updatedPlayer,
    auth ? { username: auth.username, token: auth.token, idleGained: idleResult.gained || 0 } : { idleGained: idleResult.gained || 0 }
  );
}

async function handleRegister(payload = {}) {
  if (!USERS_TABLE) {
    throw httpError(500, 'User registration unavailable.');
  }

  const username = sanitizeUsername(payload.username);
  const password = (payload.password || '').toString();

  if (!EMAIL_REGEX.test(username)) {
    throw httpError(400, 'Please provide a valid email address.');
  }
  if (!validatePassword(password)) {
    throw httpError(400, 'Password must be at least 8 characters.');
  }

  const existing = await getUser(username);
  if (existing) {
    throw httpError(409, 'An account with this email already exists.');
  }

  const passwordHash = await bcrypt.hash(password, PASSWORD_COST);
  const playerId = createPlayerId('user');
  await dynamo.send(new PutCommand({
    TableName: USERS_TABLE,
    Item: {
      username,
      passwordHash,
      playerId,
      createdAt: nowIso()
    },
    ConditionExpression: 'attribute_not_exists(username)'
  }));

  let player = await getOrCreatePlayer(playerId);
  player = await syncDailyState(player);
  const session = await issueSession(username);
  return formatPlayerPayload(player, {
    username,
    token: session.token
  });
}

async function handleLogin(payload = {}) {
  if (!USERS_TABLE) {
    throw httpError(500, 'User login unavailable.');
  }

  const username = sanitizeUsername(payload.username);
  const password = (payload.password || '').toString();
  const user = await getUser(username);

  if (!user || !user.passwordHash) {
    throw httpError(401, 'Invalid email or password.');
  }

  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) {
    throw httpError(401, 'Invalid email or password.');
  }

  const playerId = await ensureUserPlayer({ ...user, username });
  let player = await getOrCreatePlayer(playerId);
  player = await syncDailyState(player);
  const session = await issueSession(username);
  return formatPlayerPayload(player, {
    username,
    token: session.token
  });
}

async function handleLogout(payload = {}) {
  const auth = await resolveAuth(payload.token, { required: true });
  await revokeSession(auth.username);
  return { ok: true };
}

async function handleUpgrade(payload = {}) {
  const def = getUpgradeDefinition(payload.type || '');
  if (!def) {
    throw httpError(400, 'Unknown upgrade type.');
  }
  if (def.clientOnly) {
    return { errorCode: 'CLIENT_ONLY', message: 'Upgrade is client-managed.', upgrades: normalizeUpgrades() };
  }
  const type = def.key;
  const auth = await resolveAuth(payload.token, { required: true });
  let player = await syncDailyState(auth.player);
  const upgrades = normalizeUpgrades(player.upgrades || DEFAULT_UPGRADES);
  const currentLevel = upgrades[type] || 0;
  if (def.requires) {
    const requirements = Array.isArray(def.requires) ? def.requires : [def.requires];
    const unlocked = requirements.every(req => (upgrades[req] || 0) > 0);
    if (!unlocked) {
      return { errorCode: 'UPGRADE_LOCKED', message: 'Upgrade locked.', upgrades };
    }
  }
  const maxLevels = resolveUpgradeMax(def);
  if (currentLevel >= maxLevels) {
    return { errorCode: 'UPGRADE_MAX', message: 'Upgrade maxed out.', upgrades };
  }
  const cost = computeUpgradeCost(def, currentLevel);
  if (player.credits < cost) {
    return {
      errorCode: 'INSUFFICIENT_CREDITS',
      message: 'Not enough credits to purchase upgrade.',
      balance: player.credits
    };
  }
  const nextLevel = currentLevel + 1;
  const nextCredits = player.credits - cost;
  const now = nowIso();
  const updatedUpgrades = { ...upgrades, [type]: nextLevel };
  console.log('upgrade:start', {
    type,
    currentLevel,
    nextLevel,
    cost,
    credits: player.credits
  });
  const updated = await dynamo.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { playerId: player.playerId },
    UpdateExpression: 'SET credits = :credits, upgrades = :upgrades, updatedAt = :now',
    ExpressionAttributeValues: {
      ':credits': nextCredits,
      ':upgrades': updatedUpgrades,
      ':now': now
    },
    ReturnValues: 'ALL_NEW'
  }));
  console.log('upgrade:complete', {
    type,
    credits: updated.Attributes?.credits,
    upgrades: updated.Attributes?.upgrades
  });
  return formatPlayerPayload(
    updated.Attributes,
    { username: auth.username, token: auth.token }
  );
}

async function applyPendingUpgrades(player, pending = {}) {
  const entries = Object.entries(pending || {}).filter(([, val]) => Number.isFinite(val) && val > 0);
  if (!entries.length) return player;
  const upgrades = normalizeUpgrades(player.upgrades || DEFAULT_UPGRADES);
  let credits = player.credits || 0;
  for (const [key, count] of entries) {
    const def = getUpgradeDefinition(key);
    if (!def) continue;
    for (let i = 0; i < count; i += 1) {
      const maxLevels = resolveUpgradeMax(def);
      const currentLevel = upgrades[key] || 0;
      if (currentLevel >= maxLevels) {
        throw httpError(400, `Upgrade ${key} exceeds maximum level.`);
      }
      if (def.requires) {
        const reqs = Array.isArray(def.requires) ? def.requires : [def.requires];
        const unlocked = reqs.every(req => (upgrades[req] || 0) > 0);
        if (!unlocked) {
          throw httpError(400, `Prerequisite missing for upgrade ${key}.`);
        }
      }
      const cost = computeUpgradeCost(def, currentLevel);
      if (credits < cost) {
        throw httpError(400, 'Not enough credits to sync upgrades.');
      }
      credits -= cost;
      upgrades[key] = currentLevel + 1;
    }
  }
  const now = nowIso();
  const updated = await dynamo.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { playerId: player.playerId },
    UpdateExpression: 'SET credits = :credits, upgrades = :upgrades, updatedAt = :now',
    ExpressionAttributeValues: {
      ':credits': credits,
      ':upgrades': upgrades,
      ':now': now
    },
    ReturnValues: 'ALL_NEW'
  }));
  return updated.Attributes || player;
}

async function handleDebugCoins(payload = {}) {
  const auth = await resolveAuth(payload.token, { required: true });
  const email = (auth.username || '').toLowerCase();
  if (email !== DEBUG_EMAIL) {
    throw httpError(403, 'Not authorized.');
  }
  const bonus = DEBUG_COIN_AMOUNT;
  const player = auth.player;
  const nextCredits = (player.credits || 0) + bonus;
  const now = nowIso();
  const updated = await dynamo.send(new UpdateCommand({
    TableName: TABLE_NAME,
    Key: { playerId: player.playerId },
    UpdateExpression: 'SET credits = :credits, updatedAt = :now',
    ExpressionAttributeValues: {
      ':credits': nextCredits,
      ':now': now
    },
    ReturnValues: 'ALL_NEW'
  }));
  console.log('debug:coins', {
    username: auth.username,
    playerId: player.playerId,
    amount: bonus,
    credits: updated.Attributes?.credits
  });
  return formatPlayerPayload(
    updated.Attributes,
    { username: auth.username, token: auth.token }
  );
}

async function handleDeleteAccount(payload = {}) {
  if (!USERS_TABLE) {
    throw httpError(500, 'Account deletion unavailable.');
  }
  const auth = await resolveAuth(payload.token, { required: true });
  const username = auth.username;
  const playerId = auth.player?.playerId;
  await deleteHistoryEntries(playerId);
  if (playerId) {
    await dynamo.send(new DeleteCommand({
      TableName: TABLE_NAME,
      Key: { playerId }
    })).catch(() => {});
  }
  await dynamo.send(new DeleteCommand({
    TableName: USERS_TABLE,
    Key: { username }
  })).catch(() => {});
  await revokeSession(username);
  return { ok: true };
}

exports.handler = async (event = {}) => {
  if (!TABLE_NAME) {
    throw new Error('TABLE_NAME environment variable is required.');
  }

  const requestOrigin = event?.headers?.origin || event?.headers?.Origin || '';
  const corsOrigin = resolveCorsOrigin(requestOrigin);

  if (event.requestContext?.http?.method === 'OPTIONS') {
    return { statusCode: 204, headers: buildHeaders(corsOrigin) };
  }

  const routeKey = event.routeKey || `${event.requestContext?.http?.method || ''} ${event.requestContext?.http?.path || ''}`.trim();
  const payload = parseBody(event);

  try {
    if (routeKey.endsWith('/auth/register')) {
      const register = await handleRegister(payload);
      return respond(200, register, corsOrigin);
    }
    if (routeKey.endsWith('/auth/login')) {
      const login = await handleLogin(payload);
      return respond(200, login, corsOrigin);
    }
    if (routeKey.endsWith('/auth/logout')) {
      const logout = await handleLogout(payload);
      return respond(200, logout, corsOrigin);
    }
    if (routeKey.endsWith('/auth/delete')) {
      const deleted = await handleDeleteAccount(payload);
      return respond(200, deleted, corsOrigin);
    }
    if (routeKey.endsWith('/upgrade')) {
      const upgraded = await handleUpgrade(payload);
      if (upgraded.errorCode) {
        const status = upgraded.errorCode === 'INSUFFICIENT_CREDITS' ? 400 : 422;
        return respond(status, upgraded, corsOrigin);
      }
      return respond(200, upgraded, corsOrigin);
    }
    if (routeKey.endsWith('/session')) {
      const session = await handleSession(payload);
      return respond(200, session, corsOrigin);
    }
    if (routeKey.endsWith('/daily')) {
      const daily = await handleDaily(payload);
      if (daily.errorCode) {
        return respond(400, daily, corsOrigin);
      }
      return respond(200, daily, corsOrigin);
    }
    if (routeKey.endsWith('/sync')) {
      const sync = await handleSync(payload);
      return respond(200, sync, corsOrigin);
    }
    if (routeKey.endsWith('/spin')) {
      const spin = await handleSpin(payload);
      if (spin.errorCode) {
        const status = spin.errorCode === 'INSUFFICIENT_CREDITS' ? 400 : 422;
        return respond(status, spin, corsOrigin);
      }
      return respond(200, spin, corsOrigin);
    }
    return respond(404, { error: 'Not found' }, corsOrigin);
  } catch (error) {
    console.error('Slot machine error', error);
    if (error.statusCode) {
      const payload = error.expose ? { error: error.message } : { error: 'Server error' };
      return respond(error.statusCode, payload, corsOrigin);
    }
    return respond(500, { error: 'Server error' }, corsOrigin);
  }
};

exports._getUpgradeDefinition = getUpgradeDefinition;
exports._evaluateSkills = evaluateSkills;
exports._dailyRewardFor = dailyRewardFor;
exports._formatDailyPayload = formatDailyPayload;
exports._advanceDaily = advanceDaily;
exports._handleSync = handleSync;
