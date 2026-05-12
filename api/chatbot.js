'use strict';

const crypto = require('crypto');
const { BedrockRuntimeClient, ConverseCommand } = require('@aws-sdk/client-bedrock-runtime');
const { publicSources, retrieveKnowledge } = require('./_lib/chatbot-knowledge');
const { recordChatbotLog } = require('./_lib/chatbot-logs');
const {
  checkChatbotRateLimit,
  getClientIp,
  getLimitConfig,
  isProductionRuntime
} = require('./_lib/chatbot-rate-limit');

const SITE_ORIGIN = 'https://www.danielshort.me';
const DEFAULT_MODEL_ID = 'amazon.nova-lite-v1:0';
const MAX_CONTEXT_CHARS = 7000;
const DEFAULT_SUGGESTED_LINKS = [
  { title: 'Analytics Portfolio', url: '/portfolio?audience=analytics', reason: 'See analytics projects and case studies.' },
  { title: 'Resume', url: '/resume-analytics', reason: 'Open the resume page.' },
  { title: 'Contact Daniel', url: '/contact', reason: 'Start a direct message or email.' }
];

let cachedBedrockClient = null;
let cachedBedrockKey = '';

function pickEnv(keys) {
  for (const key of keys) {
    const raw = process.env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function getRegion() {
  return pickEnv(['CHATBOT_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
}

function getAwsCredentialsFromEnv() {
  const accessKeyId = pickEnv(['CHATBOT_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID']);
  const secretAccessKey = pickEnv(['CHATBOT_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY']);
  const sessionToken = pickEnv(['CHATBOT_AWS_SESSION_TOKEN', 'AWS_SESSION_TOKEN']);
  if (!accessKeyId || !secretAccessKey) return null;
  return {
    accessKeyId,
    secretAccessKey,
    ...(sessionToken ? { sessionToken } : {})
  };
}

function getBedrockClient() {
  const region = getRegion();
  const credentials = getAwsCredentialsFromEnv();
  const key = `${region}:${credentials ? credentials.accessKeyId : 'default'}`;
  if (cachedBedrockClient && cachedBedrockKey === key) return cachedBedrockClient;

  cachedBedrockClient = new BedrockRuntimeClient({
    region,
    credentials: credentials || undefined
  });
  cachedBedrockKey = key;
  return cachedBedrockClient;
}

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.end(JSON.stringify(payload));
}

async function readJson(req, maxBytes = 24_000) {
  if (req.body && typeof req.body === 'object') return req.body;
  if (typeof req.body === 'string' && req.body.trim()) return JSON.parse(req.body);

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > maxBytes) {
      const err = new Error('Request body too large');
      err.code = 'BODY_TOO_LARGE';
      throw err;
    }
    chunks.push(buf);
  }

  const raw = Buffer.concat(chunks).toString('utf8').trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

function isEnabled() {
  return String(process.env.CHATBOT_ENABLED || '').trim().toLowerCase() === 'true';
}

function getConfigPayload() {
  const limits = getLimitConfig();
  return {
    ok: true,
    enabled: isEnabled(),
    model: 'Amazon Nova Lite',
    turnstileSiteKey: pickEnv(['CHATBOT_TURNSTILE_SITE_KEY']),
    limits: {
      minSecondsBetweenQueries: limits.minSecondsBetweenQueries,
      windowSeconds: limits.windowSeconds,
      windowLimit: limits.windowLimit,
      dailyLimit: limits.dailyLimit
    }
  };
}

function allowedOrigins() {
  const configured = String(process.env.CHATBOT_ALLOWED_ORIGINS || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  const vercelUrl = process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : '';
  return new Set([
    SITE_ORIGIN,
    'https://danielshort.me',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    vercelUrl,
    ...configured
  ].filter(Boolean));
}

function isAllowedOrigin(req) {
  const origin = String(req.headers.origin || '').trim();
  if (!origin) return true;
  if (allowedOrigins().has(origin)) return true;
  try {
    const url = new URL(origin);
    if (!isProductionRuntime() && ['localhost', '127.0.0.1', '::1'].includes(url.hostname)) return true;
    return url.hostname.endsWith('.vercel.app') && !isProductionRuntime();
  } catch {
    return false;
  }
}

function normalizeMessage(value) {
  const message = String(value || '').replace(/\s+/g, ' ').trim();
  const maxChars = Math.max(200, Number(process.env.CHATBOT_MAX_MESSAGE_CHARS) || 1000);
  if (!message) {
    const err = new Error('Message is required.');
    err.code = 'MESSAGE_REQUIRED';
    throw err;
  }
  if (message.length > maxChars) {
    const err = new Error(`Message is too long. Please keep it under ${maxChars} characters.`);
    err.code = 'MESSAGE_TOO_LONG';
    throw err;
  }
  return message;
}

function normalizePageContext(value) {
  const input = value && typeof value === 'object' ? value : {};
  return {
    url: String(input.url || '').slice(0, 240),
    title: String(input.title || '').slice(0, 180)
  };
}

function makeConversationId(value) {
  const raw = String(value || '').trim();
  if (/^[a-zA-Z0-9_-]{12,80}$/.test(raw)) return raw;
  return crypto.randomBytes(14).toString('base64url');
}

async function verifyTurnstile(token, req) {
  const secret = pickEnv(['CHATBOT_TURNSTILE_SECRET_KEY']);
  const response = String(token || '').trim();
  if (!response) return false;
  if (!secret) return !isProductionRuntime() && response === 'dev-pass';

  const params = new URLSearchParams();
  params.set('secret', secret);
  params.set('response', response);
  params.set('remoteip', getClientIp(req));

  try {
    const verifyRes = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
      method: 'POST',
      headers: { 'content-type': 'application/x-www-form-urlencoded' },
      body: params
    });
    const data = await verifyRes.json().catch(() => null);
    return Boolean(verifyRes.ok && data && data.success);
  } catch {
    return false;
  }
}

function lowConfidenceAnswer(message, retrieval) {
  const fallbackSources = publicSources(retrieval.chunks, 3);
  const sourceHint = fallbackSources.length
    ? 'The closest related pages I found are linked below.'
    : 'Try asking about Daniel Short, his portfolio projects, resume, analytics work, tourism work, data science background, or contact options.';
  return {
    ok: true,
    answer: `I could not find enough support in the website content to answer that confidently. ${sourceHint}`,
    sources: fallbackSources,
    suggestedLinks: suggestedLinksFromRetrieval(message, retrieval, { includeDefaults: true }),
    confidence: retrieval.bestScore,
    skippedModel: true,
    messageEcho: message.slice(0, 120)
  };
}

function normalizePath(value) {
  try {
    const url = new URL(String(value || ''), SITE_ORIGIN);
    return `${url.pathname || '/'}${url.search || ''}`.replace(/\/+$/, '') || '/';
  } catch {
    return String(value || '').trim();
  }
}

function addSuggestedLink(links, seen, item) {
  const url = normalizePath(item && item.url);
  const title = String(item && item.title ? item.title : url).trim();
  if (!url || seen.has(url)) return;
  seen.add(url);
  links.push({
    title: title.slice(0, 120),
    url,
    reason: String(item && item.reason ? item.reason : 'Relevant page on this website.').trim().slice(0, 180)
  });
}

function suggestedLinksFromRetrieval(message, retrieval, options = {}) {
  const query = String(message || '').toLowerCase();
  const links = [];
  const seen = new Set();

  if (/\b(contact|email|hire|reach|linkedin|github|message)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Contact Daniel', url: '/contact', reason: 'Use the contact page for email, LinkedIn, GitHub, and direct messages.' });
  }
  if (/\bresume|cv|experience|background|work history|qualification|qualified\b/.test(query)) {
    if (/\btourism|destination|travel\b/.test(query)) {
      addSuggestedLink(links, seen, { title: 'Tourism Resume', url: '/resume-tourism', reason: 'Resume tailored to tourism analytics and destination work.' });
    } else if (/\bdata science|machine learning|ml|model|python\b/.test(query)) {
      addSuggestedLink(links, seen, { title: 'Data Science Resume', url: '/resume-data-science', reason: 'Resume tailored to data science and ML work.' });
    } else {
      addSuggestedLink(links, seen, { title: 'Analytics Resume', url: '/resume-analytics', reason: 'Resume tailored to analytics, BI, SQL, and reporting.' });
    }
  }
  if (/\bportfolio|project|case stud|dashboard|example|work sample\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Portfolio', url: '/portfolio', reason: 'Browse projects and examples.' });
  }
  if (/\banalytics|bi|tableau|sql|reporting|dashboard\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Analytics Home', url: '/analytics', reason: 'Start with the analytics-focused overview.' });
    addSuggestedLink(links, seen, { title: 'Analytics Portfolio', url: '/portfolio?audience=analytics', reason: 'Filter portfolio work to analytics projects.' });
  }
  if (/\bdata science|machine learning|ml|model|python|nlp\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Data Science Home', url: '/data-science', reason: 'Start with the data-science-focused overview.' });
    addSuggestedLink(links, seen, { title: 'Data Science Portfolio', url: '/portfolio?audience=data-science', reason: 'Filter portfolio work to data science projects.' });
  }
  if (/\btourism|destination|travel|visitor|grand junction\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Tourism Analytics Home', url: '/tourism', reason: 'Start with the tourism analytics overview.' });
    addSuggestedLink(links, seen, { title: 'Tourism Portfolio', url: '/portfolio?audience=tourism', reason: 'Filter portfolio work to tourism and destination projects.' });
  }

  publicSources(retrieval && retrieval.chunks ? retrieval.chunks : [], 5).forEach((source) => {
    addSuggestedLink(links, seen, {
      title: source.title,
      url: source.url,
      reason: 'Related source used for this answer.'
    });
  });

  if (!links.length && options.includeDefaults) {
    DEFAULT_SUGGESTED_LINKS.forEach((item) => addSuggestedLink(links, seen, item));
  }

  return links.slice(0, 5);
}

function retrievalLogChunks(retrieval) {
  return (retrieval && Array.isArray(retrieval.chunks) ? retrieval.chunks : [])
    .map((chunk) => ({
      id: chunk.id,
      title: chunk.title,
      url: chunk.url,
      category: chunk.category,
      score: chunk.score
    }))
    .slice(0, 12);
}

async function safeRecordChatbotLog(req, input) {
  try {
    const result = await recordChatbotLog(input);
    return result && result.logId ? result.logId : '';
  } catch (err) {
    try {
      console.warn('Chatbot log failed', {
        name: err && err.name ? err.name : '',
        message: err && err.message ? err.message : ''
      });
    } catch {}
    return '';
  }
}

function buildContext(chunks) {
  const lines = [];
  let used = 0;
  chunks.forEach((chunk, index) => {
    const sourceUrl = `${SITE_ORIGIN}${chunk.url}`;
    const block = [
      `[${index + 1}] ${chunk.title} (${sourceUrl})`,
      `Category: ${chunk.category || 'Website'}`,
      chunk.text
    ].join('\n');
    if (used + block.length > MAX_CONTEXT_CHARS) return;
    used += block.length;
    lines.push(block);
  });
  return lines.join('\n\n');
}

function extractAnswer(response) {
  const parts = response && response.output && response.output.message && Array.isArray(response.output.message.content)
    ? response.output.message.content
    : [];
  return parts.map((part) => part && part.text ? String(part.text) : '').join('').trim();
}

async function callBedrock(message, retrieval, pageContext) {
  const modelId = pickEnv(['CHATBOT_BEDROCK_MODEL_ID']) || DEFAULT_MODEL_ID;
  const maxTokens = Math.max(160, Math.min(900, Number(process.env.CHATBOT_MAX_OUTPUT_TOKENS) || 420));
  const context = buildContext(retrieval.chunks);
  const system = [
    'You are a concise website chatbot and navigation assistant for Daniel Short.',
    'Answer only using the provided website sources.',
    'If the sources do not support an answer, say you do not have enough information.',
    'Do not invent credentials, experience, prices, contact details, or claims.',
    'When helpful, guide the visitor to the most relevant page, resume, project, portfolio view, or contact path from the sources.',
    'Use a helpful, direct tone. Keep answers short unless the visitor asks for detail.',
    'Mention source numbers like [1] when a claim depends on a source.'
  ].join(' ');

  const userText = [
    `Current page: ${pageContext.title || 'Unknown'} ${pageContext.url || ''}`.trim(),
    '',
    'Website sources:',
    context,
    '',
    `Visitor question: ${message}`
  ].join('\n');

  const command = new ConverseCommand({
    modelId,
    system: [{ text: system }],
    messages: [
      {
        role: 'user',
        content: [{ text: userText }]
      }
    ],
    inferenceConfig: {
      maxTokens,
      temperature: 0.2,
      topP: 0.9
    }
  });

  const response = await getBedrockClient().send(command);
  return {
    answer: extractAnswer(response),
    usage: response.usage || null,
    modelId
  };
}

module.exports = async (req, res) => {
  const startedAt = Date.now();

  if (req.method === 'GET') {
    sendJson(res, 200, getConfigPayload());
    return;
  }

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'GET, POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  if (!isAllowedOrigin(req)) {
    sendJson(res, 403, { ok: false, error: 'Origin is not allowed.' });
    return;
  }

  if (!isEnabled()) {
    sendJson(res, 503, { ok: false, error: 'Chatbot is not enabled.' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch (err) {
    sendJson(res, err && err.code === 'BODY_TOO_LARGE' ? 413 : 400, { ok: false, error: 'Invalid JSON body.' });
    return;
  }

  if (body && typeof body.website === 'string' && body.website.trim()) {
    sendJson(res, 200, { ok: false, error: 'Request ignored.' });
    return;
  }

  let message;
  try {
    message = normalizeMessage(body.message);
  } catch (err) {
    sendJson(res, err && err.code === 'MESSAGE_TOO_LONG' ? 413 : 400, { ok: false, error: err.message || 'Invalid message.' });
    return;
  }

  const pageContext = normalizePageContext(body.pageContext);
  const conversationId = makeConversationId(body.conversationId);
  const challengePassed = await verifyTurnstile(body.challengeToken, req);

  let rateLimit;
  try {
    rateLimit = await checkChatbotRateLimit(req, { ...body, conversationId }, { challengePassed });
  } catch (err) {
    const statusCode = err && /MISSING|configured/i.test(String(err.code || err.message)) ? 503 : 500;
    sendJson(res, statusCode, { ok: false, error: 'Chatbot protection is not configured.' });
    return;
  }

  if (!rateLimit.allowed) {
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'rate_limited',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer: '',
      error: rateLimit.payload && rateLimit.payload.error,
      pageContext,
      rateLimit: rateLimit.payload && rateLimit.payload.limits,
      latencyMs: Date.now() - startedAt
    });
    sendJson(res, rateLimit.statusCode || 429, {
      ...rateLimit.payload,
      logId,
      turnstileSiteKey: pickEnv(['CHATBOT_TURNSTILE_SITE_KEY']) || ''
    });
    return;
  }

  let retrieval;
  try {
    retrieval = retrieveKnowledge(message, pageContext);
  } catch {
    sendJson(res, 503, { ok: false, error: 'Chatbot knowledge database is unavailable.' });
    return;
  }

  if (!retrieval.confident) {
    const fallback = lowConfidenceAnswer(message, retrieval);
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'low_confidence',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer: fallback.answer,
      pageContext,
      sources: fallback.sources,
      suggestedLinks: fallback.suggestedLinks,
      confidence: retrieval.bestScore,
      skippedModel: true,
      rateLimit: rateLimit.counts || null,
      retrievalConfident: retrieval.confident,
      retrievalBestScore: retrieval.bestScore,
      queryTerms: retrieval.queryTerms,
      retrievalChunks: retrievalLogChunks(retrieval),
      latencyMs: Date.now() - startedAt
    });
    sendJson(res, 200, {
      ...fallback,
      logId,
      conversationId,
      limits: rateLimit.counts || null
    });
    return;
  }

  try {
    const modelResponse = await callBedrock(message, retrieval, pageContext);
    const answer = modelResponse.answer || 'I found relevant site content, but I could not generate a useful answer.';
    const sources = publicSources(retrieval.chunks, 5);
    const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval);
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'answered',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer,
      pageContext,
      sources,
      suggestedLinks,
      confidence: retrieval.bestScore,
      skippedModel: false,
      model: 'Amazon Nova Lite',
      usage: modelResponse.usage,
      rateLimit: rateLimit.counts || null,
      retrievalConfident: retrieval.confident,
      retrievalBestScore: retrieval.bestScore,
      queryTerms: retrieval.queryTerms,
      retrievalChunks: retrievalLogChunks(retrieval),
      latencyMs: Date.now() - startedAt
    });
    sendJson(res, 200, {
      ok: true,
      answer,
      sources,
      suggestedLinks,
      logId,
      conversationId,
      usage: modelResponse.usage,
      model: 'Amazon Nova Lite',
      limits: rateLimit.counts || null
    });
  } catch (err) {
    const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval, { includeDefaults: true });
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'model_error',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer: '',
      error: err && err.message ? err.message : String(err || ''),
      pageContext,
      sources: publicSources(retrieval.chunks, 5),
      suggestedLinks,
      confidence: retrieval.bestScore,
      rateLimit: rateLimit.counts || null,
      retrievalConfident: retrieval.confident,
      retrievalBestScore: retrieval.bestScore,
      queryTerms: retrieval.queryTerms,
      retrievalChunks: retrievalLogChunks(retrieval),
      latencyMs: Date.now() - startedAt
    });
    sendJson(res, 502, {
      ok: false,
      error: 'Chatbot model service is unavailable.',
      suggestedLinks,
      logId,
      detail: !isProductionRuntime() ? String(err && err.message ? err.message : err) : undefined
    });
  }
};

module.exports._private = {
  buildContext,
  getConfigPayload,
  isAllowedOrigin,
  lowConfidenceAnswer,
  normalizeMessage,
  suggestedLinksFromRetrieval,
  verifyTurnstile
};
