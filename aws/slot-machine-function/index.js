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
const SYMBOL_LABELS = Object.fromEntries(
  (MACHINE_META.symbols || []).map(symbol => [symbol.key, symbol.label || symbol.key])
);

const nowIso = () => new Date().toISOString();

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
  if (player) return player;

  const now = nowIso();
  const newPlayer = {
    playerId,
    credits: STARTING_BALANCE,
    spins: 0,
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
  return {
    playerId: player.playerId,
    balance: player.credits,
    spins: player.spins || 0,
    lastSpin: player.lastSpin || null,
    startingBalance: STARTING_BALANCE,
    maxBet: BET_LIMIT,
    machine: MACHINE_META,
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
  if (payload.token) {
    const auth = await resolveAuth(payload.token, { required: true });
    return formatPlayerPayload(auth.player, {
      username: auth.username,
      token: auth.token
    });
  }

  let playerId = sanitizePlayerId(payload.playerId);
  if (!playerId) {
    playerId = createPlayerId('slot');
  }
  const player = await getOrCreatePlayer(playerId);
  return formatPlayerPayload(player);
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
  if (bet > BET_LIMIT) {
    if (playerId) {
      await logSpinHistory({ playerId, username: auth?.username || null, bet, errorCode: 'LIMIT_EXCEEDED' });
    }
    return { errorCode: 'LIMIT_EXCEEDED', message: `Bet cannot exceed ${BET_LIMIT}.`, maxBet: BET_LIMIT };
  }

  const player = auth?.player || await getOrCreatePlayer(playerId);
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
      maxBet: BET_LIMIT
    };
  }

  const engineResult = runSlotSpin(bet);
  const timestamp = nowIso();
  const spinPayload = buildSpinPayload(engineResult, bet, timestamp);
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
    auth ? { username: auth.username, token: auth.token } : {}
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
