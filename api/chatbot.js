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
const DEFAULT_SUGGESTED_LINKS = [
  { title: 'Analytics Portfolio', url: '/portfolio?audience=analytics', reason: 'See analytics projects and case studies.' },
  { title: 'Resume', url: '/resume-analytics', reason: 'Open the resume page.' },
  { title: 'Contact Daniel', url: '/contact', reason: 'Start a direct message or email.' }
];
const AUDIENCE_PROFILES = {
  analytics: {
    label: 'Analytics',
    roleLabel: 'analytics',
    homeUrl: '/analytics',
    portfolioUrl: '/portfolio?audience=analytics',
    resumeUrl: '/resume-analytics',
    resumeTitle: 'Analytics Resume',
    portfolioTitle: 'Analytics Portfolio',
    focus: 'SQL workflows, BI dashboards, reporting automation, forecasting, and stakeholder-ready business analysis',
    defaults: [
      { title: 'Analytics Portfolio', url: '/portfolio?audience=analytics', reason: 'Review analytics projects, dashboards, SQL workflows, and reporting examples.' },
      { title: 'Analytics Resume', url: '/resume-analytics', reason: 'Open the resume tailored to analytics, BI, SQL, and reporting work.' },
      { title: 'Contact Daniel', url: '/contact', reason: 'Start a direct message or email.' }
    ],
    followups: [
      'Show analytics proof a hiring manager should review',
      'Which analytics resume sections matter most?',
      'Why is Daniel a strong analytics hire?'
    ]
  },
  'data-science': {
    label: 'Data Science',
    roleLabel: 'data science',
    homeUrl: '/data-science',
    portfolioUrl: '/portfolio?audience=data-science',
    resumeUrl: '/resume-data-science',
    resumeTitle: 'Data Science Resume',
    portfolioTitle: 'Data Science Portfolio',
    focus: 'machine learning, Python, NLP, evaluation, and deployment-minded data products',
    defaults: [
      { title: 'Data Science Portfolio', url: '/portfolio?audience=data-science', reason: 'Review machine learning, NLP, Python, and applied model projects.' },
      { title: 'Data Science Resume', url: '/resume-data-science', reason: 'Open the resume tailored to data science and ML work.' },
      { title: 'Contact Daniel', url: '/contact', reason: 'Start a direct message or email.' }
    ],
    followups: [
      'Show data science proof a hiring manager should review',
      'Which projects show applied ML depth?',
      'Why is Daniel a strong data science hire?'
    ]
  },
  tourism: {
    label: 'Tourism Analytics',
    roleLabel: 'tourism analytics',
    homeUrl: '/tourism',
    portfolioUrl: '/portfolio?audience=tourism',
    resumeUrl: '/resume-tourism',
    resumeTitle: 'Tourism Resume',
    portfolioTitle: 'Tourism Portfolio',
    focus: 'destination reporting, visitor demand analysis, stakeholder communication, and public-sector decision support',
    defaults: [
      { title: 'Tourism Portfolio', url: '/portfolio?audience=tourism', reason: 'Review destination analytics, visitor data, and stakeholder reporting examples.' },
      { title: 'Tourism Resume', url: '/resume-tourism', reason: 'Open the resume tailored to tourism analytics and destination work.' },
      { title: 'Contact Daniel', url: '/contact', reason: 'Start a direct message or email.' }
    ],
    followups: [
      'Show tourism proof a hiring manager should review',
      'Which projects connect to destination analytics?',
      'Why is Daniel a strong tourism analytics hire?'
    ]
  }
};

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

function hasBedrockConfiguration() {
  if (!isProductionRuntime()) return true;
  return Boolean(getAwsCredentialsFromEnv()) || Boolean(pickEnv([
    'AWS_PROFILE',
    'AWS_SHARED_CREDENTIALS_FILE',
    'AWS_WEB_IDENTITY_TOKEN_FILE',
    'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
    'AWS_CONTAINER_CREDENTIALS_FULL_URI'
  ]));
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
    const path = (url.pathname || '').replace(/\/+$/, '') || '/';
    const queryAudience = normalizeAudience(url.searchParams.get('audience'));
    if (queryAudience) return queryAudience;
    if (path === '/analytics' || path === '/resume-analytics') return 'analytics';
    if (path === '/data-science' || path === '/resume-data-science') return 'data-science';
    if (path === '/tourism' || path === '/resume-tourism') return 'tourism';
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
    ? `Try asking about Daniel Short's ${profile.roleLabel} fit, relevant projects, resume, or contact options.`
    : 'Try asking about Daniel Short, his projects, resume, or contact options.';
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
  const genericTerms = ['portfolio', 'resume', 'contact'];
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
  if (path === profile.homeUrl || path === profile.resumeUrl) return true;
  if (fullPath.startsWith('/portfolio?audience=')) return fullPath === profile.portfolioUrl;
  if (fullPath === profile.portfolioUrl || path === '/portfolio' || path.startsWith('/portfolio/')) return true;
  if (path === '/analytics' || path === '/data-science' || path === '/tourism') return false;
  if (path === '/resume-analytics' || path === '/resume-data-science' || path === '/resume-tourism') return false;
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

  if (/\b(contact|email|hire|reach|linkedin|github|message)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Contact Daniel', url: '/contact', reason: 'Use the contact page for email, LinkedIn, GitHub, and direct messages.' });
  }
  if (/\bresume|cv|experience|background|work history|qualification|qualified\b/.test(query)) {
    if (profile) {
      addSuggestedLink(links, seen, { title: profile.resumeTitle, url: profile.resumeUrl, reason: `Resume tailored to ${profile.focus}.` });
    } else if (/\btourism|destination|travel\b/.test(query)) {
      addSuggestedLink(links, seen, { title: 'Tourism Resume', url: '/resume-tourism', reason: 'Resume tailored to tourism analytics and destination work.' });
    } else if (/\bdata science|machine learning|ml|model|python\b/.test(query)) {
      addSuggestedLink(links, seen, { title: 'Data Science Resume', url: '/resume-data-science', reason: 'Resume tailored to data science and ML work.' });
    } else {
      addSuggestedLink(links, seen, { title: 'Analytics Resume', url: '/resume-analytics', reason: 'Resume tailored to analytics, BI, SQL, and reporting.' });
    }
  }
  if (/\bportfolio|project|case stud|dashboard|example|work sample\b/.test(query)) {
    addSuggestedLink(links, seen, profile
      ? { title: profile.portfolioTitle, url: profile.portfolioUrl, reason: `Browse projects most relevant to ${profile.roleLabel} roles.` }
      : { title: 'Portfolio', url: '/portfolio', reason: 'Browse projects and examples.' });
  }
  if (profile && /\b(home|overview|start|role|fit|candidate)\b/.test(query)) {
    addSuggestedLink(links, seen, { title: `${profile.label} Home`, url: profile.homeUrl, reason: `Start with Daniel's ${profile.roleLabel} overview.` });
  }
  if (!profile && /\banalytics|bi|tableau|sql|reporting|dashboard\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Analytics Home', url: '/analytics', reason: 'Start with the analytics-focused overview.' });
    addSuggestedLink(links, seen, { title: 'Analytics Portfolio', url: '/portfolio?audience=analytics', reason: 'Filter portfolio work to analytics projects.' });
  }
  if (!profile && /\bdata science|machine learning|ml|model|python|nlp\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Data Science Home', url: '/data-science', reason: 'Start with the data-science-focused overview.' });
    addSuggestedLink(links, seen, { title: 'Data Science Portfolio', url: '/portfolio?audience=data-science', reason: 'Filter portfolio work to data science projects.' });
  }
  if (!profile && /\btourism|destination|travel|visitor|grand junction\b/.test(query)) {
    addSuggestedLink(links, seen, { title: 'Tourism Analytics Home', url: '/tourism', reason: 'Start with the tourism analytics overview.' });
    addSuggestedLink(links, seen, { title: 'Tourism Portfolio', url: '/portfolio?audience=tourism', reason: 'Filter portfolio work to tourism and destination projects.' });
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

  if (/\b(contact|email|hire|reach|linkedin|github|message)\b/.test(query)) {
    return {
      ...base,
      answer: profile
        ? `Use [Contact Daniel](/contact) for the fastest path. For a ${profile.roleLabel} hiring conversation, review the [${profile.resumeTitle}](${profile.resumeUrl}) and [${profile.portfolioTitle}](${profile.portfolioUrl}).`
        : 'Use [Contact Daniel](/contact) for email, LinkedIn, GitHub, and direct message options.'
    };
  }

  if (/\b(resume|cv|work history|qualification|qualified)\b/.test(query)) {
    let answer = profile
      ? `Use the [${profile.resumeTitle}](${profile.resumeUrl}); it frames Daniel around ${profile.focus}.`
      : 'Use the [Analytics Resume](/resume-analytics) for the broadest current resume view.';
    if (!profile && /\b(tourism|destination|travel|visitor|grand junction)\b/.test(query)) {
      answer = 'Use the [Tourism Resume](/resume-tourism) for destination analytics and visitor-focused work.';
    } else if (!profile && /\b(data science|machine learning|ml|model|python|nlp)\b/.test(query)) {
      answer = 'Use the [Data Science Resume](/resume-data-science) for machine learning, Python, and modeling-focused work.';
    }
    return {
      ...base,
      answer
    };
  }

  if (/\bportfolio\b/.test(query) || /\b(show|see|browse|open|view|find)\b.*\b(project|projects|case stud|work sample)\b/.test(query)) {
    return {
      ...base,
      answer: profile
        ? `Start with the [${profile.portfolioTitle}](${profile.portfolioUrl}); it focuses the project view on ${profile.focus}.`
        : 'Start with the [Portfolio](/portfolio), then filter by analytics, data science, or tourism work.'
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
    followupContext.sourceLabels.length ? `Previous sources: ${followupContext.sourceLabels.join(', ')}` : ''
  ].filter(Boolean).join('\n');
}

function buildUserText(message, retrieval, pageContext, history = [], followupContext = null) {
  const profile = audienceProfile(pageContext && pageContext.audience);
  const audienceLines = profile
    ? [
      `Active audience lens: ${profile.label}`,
      `Role focus: ${profile.focus}`,
      `Preferred links: ${profile.homeUrl}, ${profile.portfolioUrl}, ${profile.resumeUrl}, /contact`
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
    ? `Active audience: ${profile.label}. Advocate for Daniel as a ${profile.roleLabel} candidate, focusing on ${profile.focus}. Only recommend links relevant to this audience: ${profile.homeUrl}, ${profile.portfolioUrl}, ${profile.resumeUrl}, /contact, and project pages that support the ${profile.roleLabel} case. Shared projects may be discussed, but frame them through this role lens.`
    : 'No audience lens is active, so use the broad website context and recommend the most relevant public page.';
  return [
    'You are Daniel Short\'s website chatbot and recruiting navigation assistant.',
    audienceInstruction,
    'Answer only using the provided website sources and recent conversation.',
    'Advocate for Daniel with evidence: connect projects, resume details, outcomes, and site pages to why he is a strong candidate.',
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

function normalizePrompt(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

function makeFollowups(message, answer, retrieval, pageContext, history = [], suggestedLinks = []) {
  const combined = [message, answer, pageContext && pageContext.title, ...(retrieval.queryTerms || [])].join(' ').toLowerCase();
  const profile = audienceProfile(pageContext && pageContext.audience || retrieval && retrieval.audience);
  const prior = new Set(history.filter((turn) => turn.role === 'user').map((turn) => normalizePrompt(turn.text)));
  const seen = new Set();
  const items = [];
  const add = (text) => {
    const normalized = normalizePrompt(text);
    if (!normalized || seen.has(normalized) || prior.has(normalized)) return;
    seen.add(normalized);
    items.push(text);
  };

  if (profile) {
    profile.followups.forEach(add);
  }
  if (/\b(contact|email|hire|reach|linkedin|github|message)\b/.test(combined)) {
    add(profile ? `Which ${profile.roleLabel} projects should I review first?` : 'Which projects should I review first?');
    add(profile ? `Which resume sections support ${profile.roleLabel} roles?` : 'Which resume is most relevant?');
  }
  if (/\bresume|cv|experience|qualified|work history\b/.test(combined)) {
    add(profile ? `Show ${profile.roleLabel} portfolio proof for this resume` : 'Show portfolio proof for this resume');
    add(profile ? `Summarize Daniel's strongest ${profile.roleLabel} fit` : "Summarize Daniel's strongest fit");
    add('How do I contact Daniel?');
  }
  if (/\bproject|portfolio|case study|dashboard|work sample\b/.test(combined)) {
    add('Show similar projects');
    add('Summarize the strongest project');
    add('Which resume matches this work?');
  }
  if (/\btourism|destination|visitor|grand junction|lodging|travel\b/.test(combined)) {
    add('Show tourism analytics examples');
    add('Summarize destination analytics experience');
  }
  if (/\bdata science|machine learning|ml|python|nlp|rag|lora|model\b/.test(combined)) {
    add('Show data science projects');
    add('Explain the RAG chatbot project');
  }
  if (/\banalytics|bi|tableau|sql|reporting|dashboard|forecast\b/.test(combined)) {
    add('Show analytics portfolio examples');
    add('Summarize dashboard experience');
  }

  suggestedLinks.slice(0, 2).forEach((link) => {
    if (link && link.title) add(`Open ${link.title}`);
  });
  add('What should I look at next?');
  add('How do I contact Daniel?');
  add('Which resume fits this role?');
  return items.slice(0, 3);
}

function withFollowups(payload, message, answer, retrieval, pageContext, history) {
  return {
    ...payload,
    retrievalMode: payload.retrievalMode || retrieval.retrievalMode || 'lexical',
    embeddingModel: retrieval.embeddingModel || undefined,
    followups: makeFollowups(message, answer || payload.answer, retrieval, pageContext, history, payload.suggestedLinks || [])
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
    const payload = withFollowups({
      ok: true,
      answer,
      sources,
      suggestedLinks,
      conversationId,
      usage: modelResponse.usage,
      model: modelResponse.modelId,
      limits: rateLimit.counts || null
    }, message, answer, retrieval, pageContext, history);
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
    const fallback = withFollowups(retrievalOnlyAnswer(message, retrieval, pageContext), message, '', retrieval, pageContext, history);
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

  const navigation = navigationAnswer(message, retrieval, pageContext);
  if (navigation) {
    const payload = withFollowups(navigation, message, navigation.answer, retrieval, pageContext, history);
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

  if (!retrieval.confident) {
    const fallback = withFollowups(lowConfidenceAnswer(message, retrieval, pageContext), message, '', retrieval, pageContext, history);
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
    const payload = withFollowups({
      ok: true,
      answer,
      sources,
      suggestedLinks,
      usage: modelResponse.usage,
      model: modelResponse.modelId,
      limits: rateLimit.counts || null
    }, message, answer, retrieval, pageContext, history);
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
    const fallback = withFollowups(retrievalOnlyAnswer(message, retrieval, pageContext), message, '', retrieval, pageContext, history);
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
  navigationAnswer,
  normalizeAudience,
  normalizeMessage,
  normalizePageContext,
  retrievalOnlyAnswer,
  stripSourceCitations,
  suggestedLinksFromRetrieval,
  verifyTurnstile
};
