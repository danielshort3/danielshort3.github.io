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
const { randomUUID, randomBytes } = require('crypto');
const bcrypt = require('bcryptjs');
const { spin: runSlotSpin, machineMetadata } = require('./slot-engine');
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
const DEBUG_EMAIL = 'danielshort3@gmail.com';
const DEBUG_COIN_AMOUNT = 100000;
const UPGRADE_INDEX = new Map();
const DEFAULT_UPGRADES = {};
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
    const hasValue = source && Number.isFinite(source[def.key]);
    const raw = hasValue ? source[def.key] : (DEFAULT_UPGRADES[def.key] ?? 0);
    const safe = Math.max(0, Math.floor(raw));
    const max = resolveUpgradeMax(def);
    normalized[def.key] = Math.min(safe, max);
  });
  return normalized;
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
    return player;
  }

  const now = nowIso();
  const newPlayer = {
    playerId,
    credits: STARTING_BALANCE,
    spins: 0,
    upgrades: { ...DEFAULT_UPGRADES },
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

async function applySpin(player, bet, spinPayload) {
  const now = spinPayload?.timestamp || nowIso();
  const lastSpin = { ...spinPayload, timestamp: now };
  const winnings = Number(lastSpin.winAmount) || 0;
  const nextCredits = player.credits - bet + winnings;

  let current = player;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const updated = await dynamo.send(new UpdateCommand({
        TableName: TABLE_NAME,
        Key: { playerId: current.playerId },
        UpdateExpression: 'SET credits = :credits, spins = if_not_exists(spins, :zero) + :one, lastSpin = :spin, updatedAt = :now',
        ConditionExpression: 'credits = :expected',
        ExpressionAttributeValues: {
          ':credits': nextCredits,
          ':expected': current.credits,
          ':spin': lastSpin,
          ':one': 1,
          ':zero': 0,
          ':now': now
        },
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
  const upgrades = normalizeUpgrades(player?.upgrades || DEFAULT_UPGRADES);
  const dims = computeDimensions(upgrades);
  const activeSymbols = computeActiveSymbols(upgrades);
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
    return formatPlayerPayload(idleResult.player, {
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
  return formatPlayerPayload(idleResult.player, { idleGained: idleResult.gained || 0 });
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
  player = idleResult.player;
  if (payload.pendingUpgrades) {
    player = await applyPendingUpgrades(player, payload.pendingUpgrades);
  }
  const upgrades = normalizeUpgrades((auth?.player || player).upgrades || DEFAULT_UPGRADES);
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
  const timestamp = nowIso();
  const spinPayload = buildSpinPayload(finalResult, bet, timestamp);
  spinPayload.rows = dimensions.rows;
  spinPayload.reels = dimensions.reels;
  spinPayload.lineTier = dimensions.lineTier;
  spinPayload.retriggered = retriggered;
  const updatedPlayer = await applySpin(player, bet, spinPayload);
  await logSpinHistory({
    playerId,
    username: auth?.username || null,
    bet,
    reels: spinPayload.grid,
    rows: spinPayload.rows,
    outcome: spinPayload.outcome,
    winGroups: spinPayload.winGroups,
    winAmount: spinPayload.winAmount,
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

  const player = await getOrCreatePlayer(playerId);
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
  const player = await getOrCreatePlayer(playerId);
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
  const type = def.key;
  const auth = await resolveAuth(payload.token, { required: true });
  const player = auth.player;
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
  const cost = Math.round((def.cost || 0) * (currentLevel + 1));
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
      const cost = def.cost * (currentLevel + 1);
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
