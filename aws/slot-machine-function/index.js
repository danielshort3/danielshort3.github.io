const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { randomUUID } = require('crypto');

const dynamo = DynamoDBDocumentClient.from(new DynamoDBClient({}), {
  marshallOptions: { removeUndefinedValues: true }
});

const {
  TABLE_NAME,
  STARTING_CREDITS = '1000',
  MAX_BET = '100',
  ALLOWED_ORIGINS = ''
} = process.env;

const STARTING_BALANCE = Math.max(parseInt(STARTING_CREDITS, 10) || 1000, 1);
const BET_LIMIT = Math.max(parseInt(MAX_BET, 10) || 100, 1);
const allowedOrigins = ALLOWED_ORIGINS.split(',').map(origin => origin.trim()).filter(Boolean);

const SYMBOLS = [
  { key: 'cherry', icon: '🍒', label: 'Cherry', weight: 35, pairMultiplier: 2, tripleMultiplier: 8 },
  { key: 'lemon', icon: '🍋', label: 'Lemon', weight: 27, pairMultiplier: 3, tripleMultiplier: 12 },
  { key: 'grape', icon: '🍇', label: 'Grapes', weight: 20, pairMultiplier: 5, tripleMultiplier: 18 },
  { key: 'star', icon: '⭐', label: 'Star', weight: 12, pairMultiplier: 8, tripleMultiplier: 28 },
  { key: 'diamond', icon: '💎', label: 'Diamond', weight: 6, pairMultiplier: 15, tripleMultiplier: 50 }
];

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

const totalWeight = SYMBOLS.reduce((sum, symbol) => sum + symbol.weight, 0);

function drawSymbol() {
  const target = Math.random() * totalWeight;
  let running = 0;
  for (const symbol of SYMBOLS) {
    running += symbol.weight;
    if (target <= running) return symbol;
  }
  return SYMBOLS[SYMBOLS.length - 1];
}

function spinReels() {
  return Array.from({ length: 3 }, () => drawSymbol());
}

function evaluateSpin(picks = [], bet = 1) {
  const [first, second, third] = picks;
  const names = picks.map(pick => pick.key);
  const reels = picks.map(pick => ({ key: pick.key, icon: pick.icon, label: pick.label }));
  let multiplier = 0;
  let outcome = 'No match';

  if (names[0] === names[1] && names[1] === names[2]) {
    multiplier = first.tripleMultiplier;
    outcome = `Triple ${first.label}!`;
  } else if (names[0] === names[1] || names[0] === names[2]) {
    multiplier = first.pairMultiplier;
    outcome = `Pair of ${first.label}`;
  } else if (names[1] === names[2]) {
    multiplier = second.pairMultiplier;
    outcome = `Pair of ${second.label}`;
  }

  const winAmount = multiplier > 0 ? bet * multiplier : 0;
  return { reels, multiplier, winAmount, outcome };
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

  const now = new Date().toISOString();
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

async function applySpin(player, bet, spinResult) {
  const now = new Date().toISOString();
  const winnings = spinResult.winAmount;
  const nextCredits = player.credits - bet + winnings;
  const lastSpin = {
    reels: spinResult.reels,
    bet,
    winAmount: winnings,
    multiplier: spinResult.multiplier,
    outcome: spinResult.outcome,
    timestamp: now
  };

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

async function handleSession(payload = {}) {
  let playerId = sanitizePlayerId(payload.playerId);
  if (!playerId) {
    playerId = `slot_${(randomUUID && randomUUID()) || Math.random().toString(36).slice(2, 10)}`;
  }
  const player = await getOrCreatePlayer(playerId);
  return {
    playerId,
    balance: player.credits,
    spins: player.spins || 0,
    lastSpin: player.lastSpin || null,
    startingBalance: STARTING_BALANCE,
    maxBet: BET_LIMIT
  };
}

async function handleSpin(payload = {}) {
  const playerId = sanitizePlayerId(payload.playerId);
  const bet = Math.trunc(Number(payload.bet));

  if (!playerId) {
    return { errorCode: 'BAD_REQUEST', message: 'Missing playerId.', maxBet: BET_LIMIT };
  }
  if (!Number.isFinite(bet) || bet <= 0) {
    return { errorCode: 'BAD_REQUEST', message: 'Bet must be a positive integer.', maxBet: BET_LIMIT };
  }
  if (bet > BET_LIMIT) {
    return { errorCode: 'LIMIT_EXCEEDED', message: `Bet cannot exceed ${BET_LIMIT}.`, maxBet: BET_LIMIT };
  }

  const player = await getOrCreatePlayer(playerId);
  if (player.credits < bet) {
    return {
      errorCode: 'INSUFFICIENT_CREDITS',
      message: 'You do not have enough credits.',
      balance: player.credits,
      maxBet: BET_LIMIT
    };
  }

  const picks = spinReels();
  const spinResult = evaluateSpin(picks, bet);
  const updatedPlayer = await applySpin(player, bet, spinResult);

  return {
    playerId,
    reels: spinResult.reels,
    bet,
    winAmount: spinResult.winAmount,
    multiplier: spinResult.multiplier,
    outcome: spinResult.outcome,
    balance: updatedPlayer.credits,
    spins: updatedPlayer.spins,
    lastSpin: updatedPlayer.lastSpin,
    maxBet: BET_LIMIT
  };
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
    return respond(500, { error: 'Server error' }, corsOrigin);
  }
};
