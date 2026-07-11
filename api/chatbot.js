'use strict';

const crypto = require('crypto');
const {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand
} = require('@aws-sdk/client-bedrock-runtime');
const { loadKnowledge, publicSources, retrieveKnowledge } = require('./_lib/chatbot-knowledge');
const chatbotLogsApi = require('./_lib/chatbot-logs-api');
const { recordChatbotLog } = require('./_lib/chatbot-logs');
const {
  AWS_WORKLOADS,
  getAwsClientConfig,
  hasAwsAuthConfiguration
} = require('./_lib/aws-credentials');
const {
  checkChatbotRateLimit,
  getClientIp,
  getLimitConfig,
  isProductionRuntime
} = require('./_lib/chatbot-rate-limit');

const SITE_ORIGIN = 'https://www.danielshort.me';
const DEFAULT_MODEL_ID = 'us.amazon.nova-lite-v1:0';
const DEFAULT_EMBED_MODEL_ID = 'amazon.titan-embed-text-v2:0';
const DEFAULT_EMBED_DIMENSIONS = 512;
const MAX_CONTEXT_CHARS = 7000;
const FOLLOWUP_MAX_CHARS = 96;
const FOLLOWUP_STOPWORDS = new Set([
  'a', 'about', 'an', 'and', 'are', 'as', 'at', 'best', 'can', 'could', 'daniel',
  'does', 'for', 'from', 'has', 'he', 'his', 'how', 'i', 'in', 'is', 'it', 'me',
  'of', 'on', 'or', 'should', 'show', 'that', 'the', 'this', 'to', 'what',
  'which', 'why', 'with', 'would'
]);
const AUDIENCE_TERM_GROUPS = {
  analytics: ['analytics', 'bi', 'sql', 'tableau', 'dashboard', 'dashboards', 'reporting', 'forecasting'],
  'data-science': ['data science', 'machine learning', 'ml', 'model', 'models', 'python', 'nlp', 'rag', 'lora'],
  tourism: ['tourism', 'destination', 'visitor', 'visitors', 'lodging', 'travel', 'grand junction', 'dmo']
};
const DEFAULT_SUGGESTED_LINKS = [
  { title: 'Home', url: '/', reason: 'Start from the personal site homepage.' },
  { title: 'Projects', url: '/portfolio', reason: 'Browse projects, demos, and case studies.' },
  { title: 'Tools', url: '/tools', reason: 'Open the public tool directory.' },
  { title: 'Games', url: '/games', reason: 'Browse browser games and simulations.' },
  { title: 'Contact Daniel', url: '/contact', reason: 'Send a note or find public contact links.' }
];
const AUDIENCE_PROFILES = {};

let cachedBedrockClient = null;
let cachedBedrockKey = '';

function pickEnv(keys) {
  for (const key of keys) {
    const raw = process.env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function boolEnv(key, fallback = false) {
  const raw = String(process.env[key] || '').trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(raw)) return true;
  if (['0', 'false', 'no', 'off'].includes(raw)) return false;
  return fallback;
}

function numberEnv(key, fallback) {
  const value = Number(process.env[key]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function getRegion() {
  return pickEnv(['CHATBOT_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
}

function hasBedrockConfiguration() {
  return hasAwsAuthConfiguration(AWS_WORKLOADS.CHATBOT_BEDROCK, { region: getRegion() });
}

function getBedrockClient() {
  const region = getRegion();
  const aws = getAwsClientConfig(AWS_WORKLOADS.CHATBOT_BEDROCK, { region });
  const key = `${region}:${aws.cacheKey}`;
  if (cachedBedrockClient && cachedBedrockKey === key) return cachedBedrockClient;

  cachedBedrockClient = new BedrockRuntimeClient(aws.clientConfig);
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
  return boolEnv('CHATBOT_ENABLED', true);
}

function isLogsRoute(req) {
  try {
    const url = new URL(req.url || '', SITE_ORIGIN);
    const pathname = (url.pathname || '').replace(/\/+$/, '') || '/';
    return pathname === '/api/chatbot/logs' || url.searchParams.get('__route') === 'logs';
  } catch {
    return false;
  }
}

function getConfigPayload() {
  const limits = getLimitConfig();
  return {
    ok: true,
    enabled: isEnabled(),
    model: 'Amazon Nova Lite',
    modelId: pickEnv(['CHATBOT_BEDROCK_MODEL_ID']) || DEFAULT_MODEL_ID,
    streaming: true,
    embeddings: {
      enabled: boolEnv('CHATBOT_EMBEDDINGS_ENABLED', true),
      modelId: pickEnv(['CHATBOT_BEDROCK_EMBED_MODEL_ID']) || DEFAULT_EMBED_MODEL_ID,
      dimensions: numberEnv('CHATBOT_BEDROCK_EMBED_DIMENSIONS', DEFAULT_EMBED_DIMENSIONS)
    },
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

function normalizeAudience(value) {
  const normalized = String(value || '').trim().toLowerCase().replace(/_/g, '-');
  if (normalized === 'data science' || normalized === 'datascience') return 'data-science';
  return Object.prototype.hasOwnProperty.call(AUDIENCE_PROFILES, normalized) ? normalized : '';
}

function audienceProfile(audience) {
  return AUDIENCE_PROFILES[normalizeAudience(audience)] || null;
}

function inferAudienceFromUrl(value) {
  try {
    const url = new URL(String(value || ''), SITE_ORIGIN);
    const queryAudience = normalizeAudience(url.searchParams.get('audience'));
    if (queryAudience) return queryAudience;
    return '';
  } catch {
    return '';
  }
}

function normalizePageContext(value) {
  const input = value && typeof value === 'object' ? value : {};
  const url = String(input.url || '').slice(0, 240);
  return {
    url,
    title: String(input.title || '').slice(0, 180),
    audience: normalizeAudience(input.audience) || inferAudienceFromUrl(url)
  };
}

function normalizeHistory(value) {
  return (Array.isArray(value) ? value : [])
    .map((turn) => ({
      role: String(turn && turn.role || '').trim() === 'assistant' ? 'assistant' : 'user',
      text: String(turn && turn.text || '').replace(/\s+/g, ' ').trim().slice(0, 700)
    }))
    .filter((turn) => turn.text)
    .slice(-8);
}

function normalizeFollowupContext(value) {
  const input = value && typeof value === 'object' ? value : {};
  const sourceLabels = Array.isArray(input.source_labels) ? input.source_labels : [];
  const sourceUrls = Array.isArray(input.source_urls) ? input.source_urls : [];
  return {
    source: String(input.source || '').slice(0, 80),
    prompt: String(input.prompt || '').replace(/\s+/g, ' ').trim().slice(0, 240),
    previousQuestion: String(input.previous_question || '').replace(/\s+/g, ' ').trim().slice(0, 300),
    previousAnswer: String(input.previous_answer || '').replace(/\s+/g, ' ').trim().slice(0, 800),
    previousIntent: String(input.previous_intent || '').slice(0, 80),
    previousRoute: String(input.previous_route || '').slice(0, 120),
    sourceLabels: sourceLabels.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 8),
    sourceUrls: sourceUrls.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 8)
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

function lowConfidenceAnswer(message, retrieval, pageContext = {}) {
  const profile = audienceProfile(pageContext.audience || retrieval.audience);
  const fallbackSources = publicSourcesForAudience(retrieval.chunks, 3, pageContext.audience || retrieval.audience);
  const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval, { includeDefaults: true, pageContext });
  const linkText = inlineLinkSentence(suggestedLinks, 'Best next step:');
  const fallbackHint = profile
    ? `Try asking about Daniel Short's ${profile.roleLabel} fit, relevant projects, or contact options.`
    : 'Try asking about Daniel Short, his projects, tools, experiments, or contact options.';
  return {
    ok: true,
    answer: `I could not find enough support in the site content to answer that confidently. ${linkText || fallbackHint}`,
    sources: fallbackSources,
    suggestedLinks,
    confidence: retrieval.bestScore,
    retrievalMode: retrieval.retrievalMode || 'lexical',
    skippedModel: true,
    messageEcho: message.slice(0, 120)
  };
}

function summarizeChunk(chunk) {
  const text = String(chunk && chunk.text ? chunk.text : '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!text) return '';
  const sentence = text.match(/^.{60,150}?(?:[.!?](?:\s|$)|$)/);
  return String(sentence ? sentence[0] : text.slice(0, 150)).trim();
}

function retrievalOnlyAnswer(message, retrieval, pageContext = {}) {
  const profile = audienceProfile(pageContext.audience || retrieval.audience);
  const sources = publicSourcesForAudience(retrieval.chunks, 5, pageContext.audience || retrieval.audience);
  const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval, { includeDefaults: true, pageContext });
  const highlights = (retrieval.chunks || [])
    .slice(0, 2)
    .map((chunk) => ({
      title: markdownLink({ title: chunk.title || 'Relevant site page', url: chunk.url }) || String(chunk.title || 'Relevant site page').trim(),
      summary: summarizeChunk(chunk)
    }))
    .filter((item) => item.summary);
  const highlightsText = highlights.length
    ? highlights.map((item) => `- **${item.title}**: ${item.summary}`).join('\n')
    : inlineLinkSentence(suggestedLinks, 'Closest pages:');
  return {
    ok: true,
    answer: profile
      ? `Closest supported content for Daniel's ${profile.roleLabel} fit:\n\n${highlightsText}`
      : `Closest supported content:\n\n${highlightsText}`,
    sources,
    suggestedLinks,
    confidence: retrieval.bestScore,
    retrievalMode: retrieval.retrievalMode || 'lexical',
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

function markdownLink(item) {
  const title = String(item && item.title ? item.title : '').trim();
  const url = normalizePath(item && item.url);
  if (!title || !url) return '';
  return `[${title}](${url})`;
}

function normalizeLinkText(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function linkLabelMatches(label, expectedTitle) {
  const labelText = normalizeLinkText(label);
  const titleText = normalizeLinkText(expectedTitle);
  if (!labelText || !titleText) return false;
  if (labelText.includes(titleText) || titleText.includes(labelText)) return true;
  const genericTerms = ['portfolio', 'projects', 'tools', 'games', 'contact'];
  return genericTerms.some((term) => labelText.includes(term) && titleText.includes(term));
}

function sanitizeInlineLinks(answer, allowedLinks) {
  const links = Array.isArray(allowedLinks) ? allowedLinks : [];
  if (!links.length) return String(answer || '').trim();
  return String(answer || '').replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, rawUrl) => {
    const normalizedUrl = normalizePath(rawUrl);
    const matched = links.find((link) => normalizePath(link && link.url) === normalizedUrl);
    if (!matched) return label;
    return linkLabelMatches(label, matched.title) ? match : label;
  }).trim();
}

function inlineLinkSentence(links, prefix = 'See:') {
  const items = (Array.isArray(links) ? links : [])
    .map(markdownLink)
    .filter(Boolean)
    .slice(0, 2);
  if (!items.length) return '';
  return `${prefix} ${items.join(' and ')}.`;
}

function answerHasInlineLink(value) {
  return /\[[^\]]+\]\((?:https?:\/\/|\/)[^)]+\)|https?:\/\//i.test(String(value || ''));
}

function ensureInlineLinks(answer, suggestedLinks) {
  const cleanAnswer = sanitizeInlineLinks(answer, suggestedLinks);
  if (!cleanAnswer || answerHasInlineLink(cleanAnswer)) return cleanAnswer;
  const linkText = inlineLinkSentence(suggestedLinks);
  return linkText ? `${cleanAnswer}\n\n${linkText}` : cleanAnswer;
}

function pathOnly(value) {
  return normalizePath(value).split('?')[0] || '/';
}

function defaultSuggestedLinksForAudience(audience) {
  const profile = audienceProfile(audience);
  return profile ? profile.defaults : DEFAULT_SUGGESTED_LINKS;
}

function sourceAllowedForAudience(source, audience) {
  const profile = audienceProfile(audience);
  if (!profile) return true;
  const path = pathOnly(source && source.url);
  const fullPath = normalizePath(source && source.url);
  if (path === '/contact') return true;
  if (path === profile.homeUrl) return true;
  if (fullPath.startsWith('/portfolio?audience=')) return fullPath === profile.portfolioUrl;
  if (fullPath === profile.portfolioUrl || path === '/portfolio' || path.startsWith('/portfolio/')) return true;
  return true;
}

function publicSourcesForAudience(chunks, maxSources, audience) {
  return publicSources(chunks, maxSources).filter((source) => sourceAllowedForAudience(source, audience));
}

function suggestedLinksFromRetrieval(message, retrieval, options = {}) {
  const query = String(message || '').toLowerCase();
  const audience = normalizeAudience(options.audience || options.pageContext && options.pageContext.audience || retrieval && retrieval.audience);
  const profile = audienceProfile(audience);
  const links = [];
  const seen = new Set();

  if (/\b(contact|email|reach|linkedin|github|message|note)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Contact Daniel', url: '/contact', reason: 'Use the contact page for email, LinkedIn, GitHub, and direct messages.' });
  }
  if (/\bportfolio|project|case stud|dashboard|example|work sample\b/.test(query)) {
    addSuggestedLink(links, seen, profile
      ? { title: profile.portfolioTitle, url: profile.portfolioUrl, reason: `Browse projects most relevant to ${profile.roleLabel} roles.` }
      : { title: 'Projects', url: '/portfolio', reason: 'Browse projects and examples.' });
  }
  if (/\b(tool|tools|utility|utilities|editor|cleanup|qr|utm)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Tools', url: '/tools', reason: 'Browse public tools and utilities.' });
  }
  if (/\b(game|games|simulation|experiment|experiments)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Games', url: '/games', reason: 'Browse games, simulations, and experiments.' });
  }
  if (profile && /\b(home|overview|start)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: `${profile.label} Home`, url: profile.homeUrl, reason: `Start with Daniel's ${profile.roleLabel} overview.` });
  }

  publicSources(retrieval && retrieval.chunks ? retrieval.chunks : [], 5).forEach((source) => {
    if (!sourceAllowedForAudience(source, audience)) return;
    addSuggestedLink(links, seen, {
      title: source.title,
      url: source.url,
      reason: 'Related source used for this answer.'
    });
  });

  if (!links.length && options.includeDefaults) {
    defaultSuggestedLinksForAudience(audience).forEach((item) => addSuggestedLink(links, seen, item));
  }
  if (profile && links.length < 3) {
    defaultSuggestedLinksForAudience(audience).forEach((item) => addSuggestedLink(links, seen, item));
  }

  return links.slice(0, 5);
}

function navigationAnswer(message, retrieval, pageContext = {}) {
  const query = String(message || '').toLowerCase();
  const profile = audienceProfile(pageContext.audience || retrieval && retrieval.audience);
  const sources = publicSourcesForAudience(retrieval && retrieval.chunks ? retrieval.chunks : [], 5, pageContext.audience || retrieval && retrieval.audience);
  const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval, { includeDefaults: true, pageContext });
  const base = {
    ok: true,
    sources,
    suggestedLinks,
    confidence: retrieval && Number.isFinite(retrieval.bestScore) ? retrieval.bestScore : 0,
    skippedModel: true,
    messageEcho: String(message || '').slice(0, 120)
  };

  if (/\b(contact|email|reach|linkedin|github|message|note)\b/.test(query)) {
    return {
      ...base,
      answer: 'Use [Contact Daniel](/contact) for email, LinkedIn, GitHub, and direct message options.'
    };
  }

  if (/\b(tool|tools|utility|utilities|editor|cleanup|qr|utm|recorder|optimizer|generator)\b/.test(query)) {
    return {
      ...base,
      answer: 'Start with the [Tools](/tools) directory, then open the specific utility that matches the task.'
    };
  }

  if (/\b(game|games|roulette|dogfight|probability engine|ocean wave|simulation|simulator)\b/.test(query)) {
    return {
      ...base,
      answer: 'Start with the [Games](/games) directory for Daniel\'s browser games and simulations.'
    };
  }

  const wantsPortfolioProof = /\bportfolio|project|projects|case stud|dashboard|work sample\b/.test(query) ||
    (/\bproof|evidence|examples?|support|back up|prove\b/.test(query) &&
      /\banalytics|bi|sql|reporting|dashboard|data science|machine learning|ml|python|tourism|destination|visitor\b/.test(query));
  if (wantsPortfolioProof) {
    return {
      ...base,
      answer: profile
        ? `Start with the [${profile.portfolioTitle}](${profile.portfolioUrl}); it focuses the project view on ${profile.focus}.`
        : 'Start with the [Projects](/portfolio), then browse the project pages that match your interest.'
    };
  }

  if (/\bportfolio\b/.test(query) || /\b(show|see|browse|open|view|find)\b.*\b(project|projects|case stud|work sample)\b/.test(query)) {
    return {
      ...base,
      answer: profile
        ? `Start with the [${profile.portfolioTitle}](${profile.portfolioUrl}); it focuses the project view on ${profile.focus}.`
        : 'Start with the [Projects](/portfolio), then browse the project pages that match your interest.'
    };
  }

  return null;
}

function retrievalLogChunks(retrieval) {
  return (retrieval && Array.isArray(retrieval.chunks) ? retrieval.chunks : [])
    .map((chunk) => ({
      id: chunk.id,
      title: chunk.title,
      url: chunk.url,
      audience: chunk.audience,
      category: chunk.category,
      score: chunk.score,
      lexicalScore: chunk.lexicalScore,
      embeddingScore: chunk.embeddingScore
    }))
    .slice(0, 12);
}

function sendStreamHeaders(res) {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Accel-Buffering', 'no');
  if (typeof res.flushHeaders === 'function') res.flushHeaders();
}

function writeStreamEvent(res, type, payload = {}) {
  res.write(`${JSON.stringify({ type, ...payload })}\n`);
}

function sendStreamDone(res, payload) {
  sendStreamHeaders(res);
  writeStreamEvent(res, 'done', { data: payload });
  res.end();
}

function vectorFromValue(value, dimensions) {
  const vector = Array.isArray(value) ? value : [];
  if (vector.length !== dimensions) return null;
  const normalized = vector.map((item) => Number(item));
  if (normalized.some((item) => !Number.isFinite(item))) return null;
  return normalized;
}

function embeddingConfigForKnowledge(knowledge) {
  const metadata = knowledge && knowledge.embeddings ? knowledge.embeddings : {};
  return {
    enabled: boolEnv('CHATBOT_EMBEDDINGS_ENABLED', true),
    required: boolEnv('CHATBOT_EMBEDDINGS_REQUIRED', false),
    modelId: pickEnv(['CHATBOT_BEDROCK_EMBED_MODEL_ID']) || metadata.modelId || DEFAULT_EMBED_MODEL_ID,
    dimensions: numberEnv('CHATBOT_BEDROCK_EMBED_DIMENSIONS', Number(metadata.dimensions) || DEFAULT_EMBED_DIMENSIONS),
    metadata
  };
}

async function embedQuery(message, knowledge) {
  const config = embeddingConfigForKnowledge(knowledge);
  const metadata = config.metadata || {};
  if (!config.enabled || !metadata.chunkCount || !['ready', 'partial'].includes(String(metadata.status || ''))) return null;
  if (metadata.modelId && metadata.modelId !== config.modelId) return null;
  if (metadata.dimensions && Number(metadata.dimensions) !== config.dimensions) return null;

  try {
    const command = new InvokeModelCommand({
      modelId: config.modelId,
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify({
        inputText: String(message || '').slice(0, 50_000),
        dimensions: config.dimensions,
        normalize: true
      })
    });
    const response = await getBedrockClient().send(command);
    const parsed = JSON.parse(Buffer.from(response.body || []).toString('utf8') || '{}');
    const vector = vectorFromValue(parsed.embedding, config.dimensions);
    if (!vector) throw new Error('Bedrock returned an invalid query embedding');
    return {
      vector,
      modelId: config.modelId,
      dimensions: config.dimensions
    };
  } catch (err) {
    if (config.required) throw err;
    return null;
  }
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
  chunks.forEach((chunk) => {
    const sourceUrl = `${SITE_ORIGIN}${chunk.url}`;
    const block = [
      `Source: ${chunk.title}`,
      `URL: ${sourceUrl}`,
      `Category: ${chunk.category || 'Website'}`,
      chunk.text
    ].join('\n');
    if (used + block.length > MAX_CONTEXT_CHARS) return;
    used += block.length;
    lines.push(block);
  });
  return lines.join('\n\n');
}

function formatHistory(history) {
  if (!history.length) return 'No prior turns.';
  return history.map((turn) => `${turn.role === 'assistant' ? 'Assistant' : 'Visitor'}: ${turn.text}`).join('\n');
}

function formatFollowupContext(followupContext) {
  if (!followupContext || !followupContext.source) return 'No follow-up context.';
  return [
    `Source: ${followupContext.source}`,
    followupContext.previousQuestion ? `Previous question: ${followupContext.previousQuestion}` : '',
    followupContext.previousAnswer ? `Previous answer: ${followupContext.previousAnswer}` : '',
    followupContext.sourceLabels.length ? `Previous sources: ${followupContext.sourceLabels.join(', ')}` : '',
    'Instruction: answer the current visitor question as a fresh follow-up. Do not repeat the previous answer; use it only to add a new angle, new evidence, or a clearer next step.'
  ].filter(Boolean).join('\n');
}

function wantsFreshFollowupAnswer(followupContext) {
  return Boolean(followupContext && followupContext.source === 'recommended_followup');
}

function buildUserText(message, retrieval, pageContext, history = [], followupContext = null) {
  const profile = audienceProfile(pageContext && pageContext.audience);
  const audienceLines = profile
    ? [
      `Active audience lens: ${profile.label}`,
      `Topic focus: ${profile.focus}`,
      `Preferred links: ${profile.homeUrl}, ${profile.portfolioUrl}, /contact`
    ]
    : ['Active audience lens: General website'];
  return [
    `Current page: ${pageContext.title || 'Unknown'} ${pageContext.url || ''}`.trim(),
    ...audienceLines,
    '',
    'Recent conversation:',
    formatHistory(history),
    '',
    'Follow-up context:',
    formatFollowupContext(followupContext),
    '',
    'Website sources:',
    buildContext(retrieval.chunks),
    '',
    `Visitor question: ${message}`
  ].join('\n');
}

function extractAnswer(response) {
  const parts = response && response.output && response.output.message && Array.isArray(response.output.message.content)
    ? response.output.message.content
    : [];
  return stripSourceCitations(parts.map((part) => part && part.text ? String(part.text) : '').join(''));
}

function stripSourceCitations(value) {
  return String(value || '')
    .replace(/[ \t]*\[(?:\d{1,2})(?:\s*,\s*\d{1,2})*\]/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function bedrockSystemPrompt(pageContext = {}) {
  const profile = audienceProfile(pageContext.audience);
  const audienceInstruction = profile
    ? `Active audience: ${profile.label}. Focus on ${profile.focus}. Only recommend links relevant to this audience: ${profile.homeUrl}, ${profile.portfolioUrl}, /contact, and project pages that support that topic.`
    : 'No audience lens is active, so use the broad website context and recommend the most relevant public page.';
  return [
    'You are Daniel Short\'s personal website chatbot and navigation assistant.',
    audienceInstruction,
    'Answer only using the provided website sources and recent conversation.',
    'Explain Daniel\'s projects, tools, experiments, and site pages with evidence from the provided sources.',
    'Be concise by default: answer with one short paragraph or at most two bullets unless the visitor explicitly asks for depth.',
    'When a relevant website page supports the answer, include one or two inline markdown links to that page instead of separate citation text.',
    'If the sources do not support an answer, say you do not have enough information.',
    'Do not invent credentials, experience, prices, contact details, or claims.',
    'Do not include bracketed numeric citations like [1], [2], or [3] in the visible answer.'
  ].join(' ');
}

async function callBedrock(message, retrieval, pageContext, history = [], followupContext = null) {
  if (!hasBedrockConfiguration()) {
    const err = new Error('Bedrock credentials are not configured.');
    err.code = 'CHATBOT_BEDROCK_CREDENTIALS_MISSING';
    throw err;
  }

  const modelId = pickEnv(['CHATBOT_BEDROCK_MODEL_ID']) || DEFAULT_MODEL_ID;
  const maxTokens = Math.max(120, Math.min(900, Number(process.env.CHATBOT_MAX_OUTPUT_TOKENS) || 180));
  const userText = buildUserText(message, retrieval, pageContext, history, followupContext);

  const command = new ConverseCommand({
    modelId,
    system: [{ text: bedrockSystemPrompt(pageContext) }],
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

async function callBedrockStream(message, retrieval, pageContext, history, followupContext, onToken) {
  if (!hasBedrockConfiguration()) {
    const err = new Error('Bedrock credentials are not configured.');
    err.code = 'CHATBOT_BEDROCK_CREDENTIALS_MISSING';
    throw err;
  }

  const modelId = pickEnv(['CHATBOT_BEDROCK_MODEL_ID']) || DEFAULT_MODEL_ID;
  const maxTokens = Math.max(120, Math.min(900, Number(process.env.CHATBOT_MAX_OUTPUT_TOKENS) || 180));
  const command = new ConverseStreamCommand({
    modelId,
    system: [{ text: bedrockSystemPrompt(pageContext) }],
    messages: [
      {
        role: 'user',
        content: [{ text: buildUserText(message, retrieval, pageContext, history, followupContext) }]
      }
    ],
    inferenceConfig: {
      maxTokens,
      temperature: 0.2,
      topP: 0.9
    }
  });

  const response = await getBedrockClient().send(command);
  let answer = '';
  let usage = null;
  for await (const event of response.stream || []) {
    const text = event && event.contentBlockDelta && event.contentBlockDelta.delta
      ? String(event.contentBlockDelta.delta.text || '')
      : '';
    if (text) {
      answer += text;
      onToken(text);
    }
    if (event && event.metadata && event.metadata.usage) usage = event.metadata.usage;
  }

  return { answer: stripSourceCitations(answer), usage, modelId };
}

function followupSystemPrompt() {
  return [
    'You write suggested follow-up question chips for Daniel Short\'s website chatbot.',
    'The goal is to help a visitor explore Daniel\'s projects, tools, experiments, and public site content.',
    'Return only valid compact JSON with this shape: {"followups":["question 1","question 2","question 3"]}.',
    'Each chip must be a concise question, grounded in the supplied evidence, and useful as the next user message.',
    'Use one detail question, one comparison or context question, and one next-step question.',
    'Do not repeat or paraphrase blocked prompts, the current question, or recent user questions.',
    'Do not invent metrics, credentials, employers, links, or claims.'
  ].join(' ');
}

function sourceSummariesForFollowups(retrieval) {
  return (retrieval && Array.isArray(retrieval.chunks) ? retrieval.chunks : [])
    .slice(0, 6)
    .map((chunk) => ({
      title: String(chunk && chunk.title || '').slice(0, 120),
      url: normalizePath(chunk && chunk.url),
      audience: String(chunk && chunk.audience || '').slice(0, 40),
      category: String(chunk && chunk.category || '').slice(0, 60),
      summary: summarizeChunk(chunk)
    }))
    .filter((item) => item.title || item.summary);
}

function evidencePaletteForAudience(audience) {
  const profile = audienceProfile(audience);
  if (profile && Array.isArray(profile.evidence)) return profile.evidence.slice(0, 16);
  return Object.keys(AUDIENCE_PROFILES)
    .flatMap((key) => AUDIENCE_PROFILES[key].evidence || [])
    .filter(Boolean)
    .slice(0, 24);
}

function collectPriorFollowupPrompts(message, history = [], followupContext = null) {
  const items = [
    message,
    followupContext && followupContext.prompt,
    followupContext && followupContext.previousQuestion
  ];
  (Array.isArray(history) ? history : [])
    .filter((turn) => turn && turn.role === 'user')
    .forEach((turn) => items.push(turn.text));
  return items
    .map((item) => String(item || '').replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .slice(-10);
}

function buildModelFollowupPrompt(message, answer, retrieval, pageContext, history = [], suggestedLinks = [], followupContext = null) {
  const profile = audienceProfile(pageContext && pageContext.audience || retrieval && retrieval.audience);
  const payload = {
    activeAudience: profile ? profile.label : 'General website',
    roleFocus: profile ? profile.focus : 'Daniel Short\'s projects, tools, experiments, and contact paths',
    preferredLinks: profile ? [profile.homeUrl, profile.portfolioUrl, '/contact'] : ['/', '/portfolio', '/tools', '/games', '/contact'],
    evidencePalette: evidencePaletteForAudience(profile ? (pageContext && pageContext.audience || retrieval && retrieval.audience) : ''),
    currentPage: {
      title: String(pageContext && pageContext.title || '').slice(0, 140),
      url: String(pageContext && pageContext.url || '').slice(0, 180)
    },
    currentQuestion: String(message || '').slice(0, 300),
    answer: String(answer || '').replace(/\s+/g, ' ').trim().slice(0, 900),
    recentUserQuestions: (Array.isArray(history) ? history : [])
      .filter((turn) => turn && turn.role === 'user' && turn.text)
      .map((turn) => String(turn.text).replace(/\s+/g, ' ').trim())
      .slice(-5),
    blockedPrompts: collectPriorFollowupPrompts(message, history, followupContext),
    sourceEvidence: sourceSummariesForFollowups(retrieval),
    suggestedLinks: (Array.isArray(suggestedLinks) ? suggestedLinks : [])
      .slice(0, 4)
      .map((link) => ({
        title: String(link && link.title || '').slice(0, 90),
        url: normalizePath(link && link.url)
      })),
    rules: [
      `Return exactly 3 follow-up questions, each ${FOLLOWUP_MAX_CHARS} characters or fewer.`,
      'Every item must end with a question mark.',
      'Question 1 should ask for concrete project, tool, or experiment detail.',
      'Question 2 should connect that evidence to the visitor\'s stated interest.',
      'Question 3 should move the visitor to a stronger next step, such as a project, tool, game, or contact path.',
      'Use the visitor\'s language and the active audience lens.',
      'Do not ask the same kind of question three times.',
      'Do not include unsupported metrics or claims.'
    ]
  };
  return JSON.stringify(payload, null, 2);
}

function parseFollowupJson(value) {
  const text = String(value || '').trim();
  if (!text) return [];
  const candidates = [text];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced && fenced[1]) candidates.unshift(fenced[1].trim());
  const objectStart = text.indexOf('{');
  const objectEnd = text.lastIndexOf('}');
  if (objectStart >= 0 && objectEnd > objectStart) candidates.push(text.slice(objectStart, objectEnd + 1));
  const arrayStart = text.indexOf('[');
  const arrayEnd = text.lastIndexOf(']');
  if (arrayStart >= 0 && arrayEnd > arrayStart) candidates.push(text.slice(arrayStart, arrayEnd + 1));

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (Array.isArray(parsed)) return parsed;
      if (parsed && Array.isArray(parsed.followups)) return parsed.followups;
      if (parsed && parsed.followups && typeof parsed.followups === 'object') return Object.values(parsed.followups);
    } catch {}
  }
  return [];
}

function normalizeFollowupQuestion(value) {
  let text = String(value || '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[-*\d.)\s]+/, '')
    .replace(/^["']+|["']+$/g, '')
    .trim();
  if (!text) return '';
  text = text.charAt(0).toUpperCase() + text.slice(1);
  if (/[.!]$/.test(text) && /^(what|which|how|why|where|when|who|can|could|does|do|is|are|should|would)\b/i.test(text)) {
    text = text.slice(0, -1);
  }
  if (!/\?$/.test(text) && /^(what|which|how|why|where|when|who|can|could|does|do|is|are|should|would)\b/i.test(text)) {
    text += '?';
  }
  return text;
}

function followupToken(value) {
  let token = String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
  if (token.length > 5 && token.endsWith('ing')) token = token.slice(0, -3);
  if (token.length > 4 && token.endsWith('ed')) token = token.slice(0, -2);
  if (token.length > 3 && token.endsWith('s')) token = token.slice(0, -1);
  return token;
}

function followupFingerprint(value) {
  const tokens = normalizePrompt(value)
    .split(/\s+/)
    .map(followupToken)
    .filter((token) => token.length > 2 && !FOLLOWUP_STOPWORDS.has(token));
  return Array.from(new Set(tokens)).sort();
}

function followupSimilarity(a, b) {
  const left = followupFingerprint(a);
  const right = followupFingerprint(b);
  if (!left.length || !right.length) return 0;
  const rightSet = new Set(right);
  const overlap = left.filter((token) => rightSet.has(token)).length;
  return overlap / Math.max(left.length, right.length);
}

function isDuplicateFollowup(text, existing) {
  return existing.some((item) => normalizePrompt(item) === normalizePrompt(text) || followupSimilarity(item, text) >= 0.74);
}

function metricTokens(value) {
  return new Set((String(value || '').match(/[+-]?\d+(?:\.\d+)?(?:%|\+|x)?/gi) || [])
    .map((item) => item.toLowerCase()));
}

function hasUnsupportedMetric(text, context = {}) {
  const tokens = Array.from(metricTokens(text));
  if (!tokens.length) return false;
  const profile = audienceProfile(context.pageContext && context.pageContext.audience || context.retrieval && context.retrieval.audience);
  const evidenceText = [
    context.answer,
    profile && profile.evidence ? profile.evidence.join(' ') : '',
    ...(context.retrieval && Array.isArray(context.retrieval.chunks)
      ? context.retrieval.chunks.map((chunk) => `${chunk.title || ''} ${chunk.text || ''}`)
      : [])
  ].join(' ');
  const allowed = metricTokens(evidenceText);
  return tokens.some((token) => !allowed.has(token));
}

function mentionsOffAudience(text, context = {}) {
  const audience = normalizeAudience(context.pageContext && context.pageContext.audience || context.retrieval && context.retrieval.audience);
  if (!audience) return false;
  const value = normalizePrompt(text);
  const contextText = normalizePrompt([context.message, context.answer].join(' '));
  return Object.keys(AUDIENCE_TERM_GROUPS)
    .filter((key) => key !== audience)
    .some((key) => AUDIENCE_TERM_GROUPS[key].some((term) => value.includes(normalizePrompt(term)) && !contextText.includes(normalizePrompt(term))));
}

function followupIntent(text) {
  const value = normalizePrompt(text);
  if (/\b(contact|email|reach|message|linkedin|github|ask)\b/.test(value)) return 'next-step';
  if (/\b(experience|background|qualified|qualification)\b/.test(value)) return 'background';
  if (/\b(project|portfolio|case|example|dashboard|workflow|model|work)\b/.test(value)) return 'project-proof';
  if (/\b(skill|skills|sql|python|tableau|machine|tourism|stakeholder|reporting)\b/.test(value)) return 'skills';
  if (/\b(fit|team|help|support|valuable)\b/.test(value)) return 'fit';
  return 'general';
}

function validateModelFollowups(rawItems, context = {}) {
  const prior = collectPriorFollowupPrompts(context.message, context.history, context.followupContext);
  const accepted = [];
  const usedIntents = new Set();
  (Array.isArray(rawItems) ? rawItems : []).forEach((item) => {
    const text = normalizeFollowupQuestion(item);
    if (!text || text.length > FOLLOWUP_MAX_CHARS || !/\?$/.test(text)) return;
    if (!/^(what|which|how|why|where|when|who|can|could|does|do|is|are|should|would)\b/i.test(text)) return;
    if (hasUnsupportedMetric(text, context) || mentionsOffAudience(text, context)) return;
    if (isDuplicateFollowup(text, prior) || isDuplicateFollowup(text, accepted)) return;
    const intent = followupIntent(text);
    if (usedIntents.has(intent) && intent !== 'general' && accepted.length < 2) return;
    usedIntents.add(intent);
    accepted.push(text);
  });
  return accepted.slice(0, 3);
}

async function generateModelFollowups(message, answer, retrieval, pageContext, history = [], suggestedLinks = [], followupContext = null) {
  if (!boolEnv('CHATBOT_MODEL_FOLLOWUPS_ENABLED', true) || !hasBedrockConfiguration()) return [];
  try {
    const modelId = pickEnv(['CHATBOT_BEDROCK_FOLLOWUP_MODEL_ID', 'CHATBOT_BEDROCK_MODEL_ID']) || DEFAULT_MODEL_ID;
    const command = new ConverseCommand({
      modelId,
      system: [{ text: followupSystemPrompt() }],
      messages: [
        {
          role: 'user',
          content: [{ text: buildModelFollowupPrompt(message, answer, retrieval, pageContext, history, suggestedLinks, followupContext) }]
        }
      ],
      inferenceConfig: {
        maxTokens: 260,
        temperature: 0.35,
        topP: 0.9
      }
    });
    const response = await getBedrockClient().send(command);
    return validateModelFollowups(parseFollowupJson(extractAnswer(response)), {
      message,
      answer,
      retrieval,
      pageContext,
      history,
      suggestedLinks,
      followupContext
    });
  } catch {
    return [];
  }
}

function normalizePrompt(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function makeFollowups(message, answer, retrieval, pageContext, history = [], suggestedLinks = [], followupContext = null) {
  const combined = [message, answer, pageContext && pageContext.title, ...(retrieval.queryTerms || [])].join(' ').toLowerCase();
  const profile = audienceProfile(pageContext && pageContext.audience || retrieval && retrieval.audience);
  const roleLabel = profile ? profile.roleLabel : 'professional';
  const prior = new Set(history.filter((turn) => turn.role === 'user').map((turn) => normalizePrompt(turn.text)));
  [message, followupContext && followupContext.prompt, followupContext && followupContext.previousQuestion]
    .forEach((item) => {
      const normalized = normalizePrompt(item);
      if (normalized) prior.add(normalized);
    });
  const seen = new Set();
  const items = [];
  const add = (text) => {
    const normalized = normalizePrompt(text);
    if (!normalized || seen.has(normalized) || prior.has(normalized)) return;
    seen.add(normalized);
    items.push(text);
  };
  const addProfileDefaults = () => {
    if (profile) profile.followups.forEach(add);
  };
  const normalizedMessage = normalizePrompt(message);

  if (/\b(contact|email|reach|linkedin|github|message|note)\b/.test(combined)) {
    add('Which project should I look at first?');
    add('Which tools has Daniel built?');
    add('Which games has Daniel built?');
  }
  if (/\bexperience|background|qualified|work history\b/.test(combined)) {
    add('Which projects show Daniel\'s background best?');
    add('Which tools show Daniel\'s practical skills?');
    add('How do I contact Daniel?');
  }
  if (/\bproject|portfolio|case study|dashboard|work sample\b/.test(combined)) {
    add('What skills does Daniel demonstrate in this project?');
    add('What implementation choices matter in this project?');
    add('Which other project should I review next?');
  }
  if (/\bgame|games|roulette|dogfight|probability engine|ocean wave|simulation|simulator\b/.test(combined)) {
    add('Which game should I try first?');
    add('What does each game demonstrate?');
  }
  if (/\btourism|destination|visitor|grand junction|lodging|travel\b/.test(combined)) {
    add('What tourism analytics skills does Daniel demonstrate?');
    add('How does this work support destination decisions?');
  }
  if (/\bdata science|machine learning|ml|python|nlp|rag|lora|model\b/.test(combined)) {
    add('What data science skills does Daniel demonstrate?');
    add('How does Daniel turn models into useful products?');
  }
  if (/\banalytics|bi|tableau|sql|reporting|dashboard|forecast\b/.test(combined)) {
    add('What analytics skills does Daniel demonstrate?');
    add('How does this show reporting or dashboard impact?');
  }
  if (followupContext && followupContext.source === 'recommended_followup') {
    if (/\bproof|review|project|portfolio|example|work sample\b/.test(normalizedMessage)) {
      add('What skills does Daniel demonstrate in this project?');
      add("How would Daniel's work help a team?");
    }
    if (/\bsection|experience|qualified\b/.test(normalizedMessage)) {
      add('Which projects best show Daniel\'s background?');
      add('What should I ask Daniel about this project?');
    }
    if (/\bwhy|strong|fit\b/.test(normalizedMessage)) {
      add('Which project evidence supports that?');
      add('What should I ask Daniel next?');
    }
  }

  suggestedLinks.slice(0, 2).forEach((link) => {
    const title = String(link && link.title || '').trim();
    if (!title) return;
    if (/\bcontact\b/i.test(title)) {
      add('How can I contact Daniel?');
      return;
    }
    add(`What should I review in ${title}?`);
  });
  addProfileDefaults();
  add('Which project should I review first?');
  add('How do I contact Daniel?');
  add('Which tools has Daniel built?');
  add('Which games has Daniel built?');
  return items.slice(0, 3);
}

function withFollowups(payload, message, answer, retrieval, pageContext, history, followupContext = null) {
  return {
    ...payload,
    retrievalMode: payload.retrievalMode || retrieval.retrievalMode || 'lexical',
    embeddingModel: retrieval.embeddingModel || undefined,
    followups: makeFollowups(message, answer || payload.answer, retrieval, pageContext, history, payload.suggestedLinks || [], followupContext)
  };
}

async function withSmartFollowups(payload, message, answer, retrieval, pageContext, history, followupContext = null) {
  const suggestedLinks = payload.suggestedLinks || [];
  const fallbackFollowups = makeFollowups(message, answer || payload.answer, retrieval, pageContext, history, suggestedLinks, followupContext);
  const context = {
    message,
    answer: answer || payload.answer,
    retrieval,
    pageContext,
    history,
    suggestedLinks,
    followupContext
  };
  const modelFollowups = await generateModelFollowups(message, answer || payload.answer, retrieval, pageContext, history, suggestedLinks, followupContext);
  const followups = validateModelFollowups([...modelFollowups, ...fallbackFollowups], context);
  return {
    ...payload,
    retrievalMode: payload.retrievalMode || retrieval.retrievalMode || 'lexical',
    embeddingModel: retrieval.embeddingModel || undefined,
    followups: followups.length ? followups : fallbackFollowups.slice(0, 3)
  };
}

async function streamModelAnswer(req, res, input) {
  const {
    message,
    retrieval,
    pageContext,
    history,
    followupContext,
    rateLimit,
    startedAt,
    conversationId
  } = input;
  const sources = publicSourcesForAudience(retrieval.chunks, 5, pageContext.audience || retrieval.audience);
  const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval, { pageContext });
  sendStreamHeaders(res);
  writeStreamEvent(res, 'meta', {
    sources,
    suggestedLinks,
    retrievalMode: retrieval.retrievalMode || 'lexical',
    embeddingModel: retrieval.embeddingModel || ''
  });

  let streamedAnswer = '';
  try {
    const modelResponse = await callBedrockStream(message, retrieval, pageContext, history, followupContext, (token) => {
      streamedAnswer += token;
      writeStreamEvent(res, 'token', { text: token });
    });
    const answer = ensureInlineLinks(modelResponse.answer || streamedAnswer.trim() || 'I found relevant site content, but I could not generate a useful answer.', suggestedLinks);
    const payload = await withSmartFollowups({
      ok: true,
      answer,
      sources,
      suggestedLinks,
      conversationId,
      usage: modelResponse.usage,
      model: modelResponse.modelId,
      limits: rateLimit.counts || null
    }, message, answer, retrieval, pageContext, history, followupContext);
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
      model: modelResponse.modelId,
      usage: modelResponse.usage,
      rateLimit: rateLimit.counts || null,
      retrievalConfident: retrieval.confident,
      retrievalBestScore: retrieval.bestScore,
      queryTerms: retrieval.queryTerms,
      retrievalChunks: retrievalLogChunks(retrieval),
      latencyMs: Date.now() - startedAt
    });
    writeStreamEvent(res, 'done', { data: { ...payload, logId } });
  } catch (err) {
    const fallback = withFollowups(retrievalOnlyAnswer(message, retrieval, pageContext), message, '', retrieval, pageContext, history, followupContext);
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'model_fallback',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer: fallback.answer,
      error: err && err.message ? err.message : String(err || ''),
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
    if (streamedAnswer.trim()) {
      writeStreamEvent(res, 'error', { error: 'The streamed model response stopped before the final answer.' });
    }
    writeStreamEvent(res, 'done', {
      data: {
        ...fallback,
        logId,
        conversationId,
        modelUnavailable: true,
        limits: rateLimit.counts || null
      }
    });
  } finally {
    res.end();
  }
}

module.exports = async (req, res) => {
  const startedAt = Date.now();

  if (isLogsRoute(req)) {
    await chatbotLogsApi(req, res);
    return;
  }

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
  const stream = body.stream === true;
  const history = normalizeHistory(body.history);
  const followupContext = normalizeFollowupContext(body.followupContext || body.followup_context);
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
    const knowledge = loadKnowledge();
    const queryEmbedding = await embedQuery(message, knowledge);
    retrieval = retrieveKnowledge(message, pageContext, {
      audience: pageContext.audience,
      queryEmbedding: queryEmbedding && queryEmbedding.vector,
      embeddingModel: queryEmbedding && queryEmbedding.modelId,
      embeddingDimensions: queryEmbedding && queryEmbedding.dimensions
    });
  } catch {
    sendJson(res, 503, { ok: false, error: 'Chatbot knowledge database is unavailable.' });
    return;
  }

  const freshFollowupAnswer = wantsFreshFollowupAnswer(followupContext);
  const navigation = freshFollowupAnswer ? null : navigationAnswer(message, retrieval, pageContext);
  if (navigation) {
    const payload = await withSmartFollowups(navigation, message, navigation.answer, retrieval, pageContext, history, followupContext);
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'navigation_answer',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer: navigation.answer,
      pageContext,
      sources: payload.sources,
      suggestedLinks: payload.suggestedLinks,
      confidence: retrieval.bestScore,
      skippedModel: true,
      rateLimit: rateLimit.counts || null,
      retrievalConfident: retrieval.confident,
      retrievalBestScore: retrieval.bestScore,
      queryTerms: retrieval.queryTerms,
      retrievalChunks: retrievalLogChunks(retrieval),
      latencyMs: Date.now() - startedAt
    });
    if (stream) {
      sendStreamDone(res, {
        ...payload,
        logId,
        conversationId,
        limits: rateLimit.counts || null
      });
      return;
    }
    sendJson(res, 200, {
      ...payload,
      logId,
      conversationId,
      limits: rateLimit.counts || null
    });
    return;
  }

  if (!retrieval.confident && !freshFollowupAnswer) {
    const fallback = await withSmartFollowups(lowConfidenceAnswer(message, retrieval, pageContext), message, '', retrieval, pageContext, history, followupContext);
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
    if (stream) {
      sendStreamDone(res, {
        ...fallback,
        logId,
        conversationId,
        limits: rateLimit.counts || null
      });
      return;
    }
    sendJson(res, 200, {
      ...fallback,
      logId,
      conversationId,
      limits: rateLimit.counts || null
    });
    return;
  }

  if (stream) {
    await streamModelAnswer(req, res, {
      message,
      retrieval,
      pageContext,
      history,
      followupContext,
      rateLimit,
      startedAt,
      conversationId
    });
    return;
  }

  try {
    const modelResponse = await callBedrock(message, retrieval, pageContext, history, followupContext);
    const sources = publicSourcesForAudience(retrieval.chunks, 5, pageContext.audience || retrieval.audience);
    const suggestedLinks = suggestedLinksFromRetrieval(message, retrieval, { pageContext });
    const answer = ensureInlineLinks(modelResponse.answer || 'I found relevant site content, but I could not generate a useful answer.', suggestedLinks);
    const payload = await withSmartFollowups({
      ok: true,
      answer,
      sources,
      suggestedLinks,
      usage: modelResponse.usage,
      model: modelResponse.modelId,
      limits: rateLimit.counts || null
    }, message, answer, retrieval, pageContext, history, followupContext);
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
      model: modelResponse.modelId,
      usage: modelResponse.usage,
      rateLimit: rateLimit.counts || null,
      retrievalConfident: retrieval.confident,
      retrievalBestScore: retrieval.bestScore,
      queryTerms: retrieval.queryTerms,
      retrievalChunks: retrievalLogChunks(retrieval),
      latencyMs: Date.now() - startedAt
    });
    sendJson(res, 200, {
      ...payload,
      logId,
      conversationId,
      limits: rateLimit.counts || null
    });
  } catch (err) {
    const fallback = withFollowups(retrievalOnlyAnswer(message, retrieval, pageContext), message, '', retrieval, pageContext, history, followupContext);
    const logId = await safeRecordChatbotLog(req, {
      req,
      status: 'model_fallback',
      actorHash: rateLimit.actorHash,
      conversationId,
      question: message,
      answer: fallback.answer,
      error: err && err.message ? err.message : String(err || ''),
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
      modelUnavailable: true,
      limits: rateLimit.counts || null
    });
  }
};

module.exports._private = {
  audienceProfile,
  buildContext,
  bedrockSystemPrompt,
  getConfigPayload,
  isAllowedOrigin,
  isLogsRoute,
  lowConfidenceAnswer,
  inlineLinkSentence,
  ensureInlineLinks,
  buildModelFollowupPrompt,
  parseFollowupJson,
  makeFollowups,
  navigationAnswer,
  normalizeAudience,
  normalizeMessage,
  normalizePageContext,
  retrievalOnlyAnswer,
  stripSourceCitations,
  suggestedLinksFromRetrieval,
  validateModelFollowups,
  verifyTurnstile
};
