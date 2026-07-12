#!/usr/bin/env node
'use strict';

/*
  Generate a citeable, dependency-free knowledge database for the embedded site
  chatbot. The output is intentionally simple JSON so the serverless API can
  retrieve context cheaply without a hosted vector database.
*/

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { BedrockRuntimeClient, InvokeModelCommand } = require('@aws-sdk/client-bedrock-runtime');
const { resolveAwsCredentials } = require('../api/_lib/aws-credentials');
const { normalizePathname, loadNoindexPathnamesFromVercel } = require('./lib/seo-routing');

const root = path.resolve(__dirname, '..');
const outPath = path.join(root, 'dist', 'chatbot-knowledge.json');
const SITE_ORIGIN = 'https://www.danielshort.me';
const MAX_CHUNK_CHARS = 900;
const MIN_CHUNK_CHARS = 260;
const MAX_PAGE_CHUNKS = 8;
const DEFAULT_EMBED_MODEL_ID = 'amazon.titan-embed-text-v2:0';
const DEFAULT_EMBED_DIMENSIONS = 512;

const excludedPathPatterns = [
  /^\/(?:chatbot-demo|.*-demo)(?:\/|$)/i,
  /^\/privacy$/i,
  /^\/sitemap(?:-pretty)?$/i,
  /^\/project-starfall$/i,
  /^\/(?:analytics|data-science|tourism|destination-analytics|contributions)$/i,
  /^\/tools\/(?:dashboard|short-links|ga4-utm-performance|job-application-tracker|transcribe|whisper-transcribe-monitor)$/i,
  /^\/resume(?:-[a-z-]+)?(?:-pdf)?$/i
];

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
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

function pickEnv(keys) {
  for (const key of keys) {
    const raw = process.env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function getAwsCredentialConfig() {
  return resolveAwsCredentials({
    service: 'chatbot-build-embeddings',
    region: getRegion(),
    roleArnEnvKeys: ['CHATBOT_BEDROCK_AWS_ROLE_ARN'],
    staticCredentialSets: [
      {
        name: 'chatbot',
        accessKeyId: 'CHATBOT_AWS_ACCESS_KEY_ID',
        secretAccessKey: 'CHATBOT_AWS_SECRET_ACCESS_KEY',
        sessionToken: 'CHATBOT_AWS_SESSION_TOKEN'
      },
      {
        name: 'default',
        accessKeyId: 'AWS_ACCESS_KEY_ID',
        secretAccessKey: 'AWS_SECRET_ACCESS_KEY',
        sessionToken: 'AWS_SESSION_TOKEN'
      }
    ]
  });
}

function hasBedrockBuildConfig() {
  let auth;
  try {
    auth = getAwsCredentialConfig();
  } catch {
    return false;
  }
  return auth.source !== 'default' || Boolean(pickEnv([
    'AWS_PROFILE',
    'AWS_SHARED_CREDENTIALS_FILE',
    'AWS_WEB_IDENTITY_TOKEN_FILE',
    'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
    'AWS_CONTAINER_CREDENTIALS_FULL_URI'
  ]));
}

function getRegion() {
  return pickEnv(['CHATBOT_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
}

function getEmbeddingConfig() {
  return {
    enabled: boolEnv('CHATBOT_EMBEDDINGS_ENABLED', true),
    required: boolEnv('CHATBOT_EMBEDDINGS_REQUIRED', false),
    modelId: pickEnv(['CHATBOT_BEDROCK_EMBED_MODEL_ID']) || DEFAULT_EMBED_MODEL_ID,
    dimensions: numberEnv('CHATBOT_BEDROCK_EMBED_DIMENSIONS', DEFAULT_EMBED_DIMENSIONS)
  };
}

function decodeHtml(value) {
  return String(value ?? '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => {
      const code = Number.parseInt(hex, 16);
      if (!Number.isFinite(code)) return '';
      try { return String.fromCodePoint(code); } catch { return ''; }
    })
    .replace(/&#(\d+);/g, (_, dec) => {
      const code = Number.parseInt(dec, 10);
      if (!Number.isFinite(code)) return '';
      try { return String.fromCodePoint(code); } catch { return ''; }
    });
}

function stripSiteSuffix(title) {
  return String(title || '')
    .replace(/\s*\|\s*Daniel Short\s*$/i, '')
    .replace(/\s*-\s*Daniel Short\s*$/i, '')
    .trim();
}

function extractTitle(html) {
  const match = /<title>([^<]+)<\/title>/i.exec(String(html || ''));
  return stripSiteSuffix(decodeHtml(match ? match[1] : ''));
}

function extractMeta(html, attr, key) {
  const safeAttr = String(attr || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const safeKey = String(key || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const re = new RegExp(`<meta\\b[^>]*\\b${safeAttr}="${safeKey}"[^>]*\\bcontent="([^"]*)"[^>]*>`, 'i');
  const match = re.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function extractDescription(html) {
  return extractMeta(html, 'name', 'description') || extractMeta(html, 'property', 'og:description');
}

function extractCanonical(html) {
  const match = /<link\s+[^>]*rel="canonical"[^>]*href="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function extractRobots(html) {
  const match = /<meta\s+[^>]*name="robots"[^>]*content="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function isNoindex(html) {
  return extractRobots(html).toLowerCase().includes('noindex');
}

function walkHtmlFiles(dirRelPath) {
  const start = path.join(root, dirRelPath);
  if (!fs.existsSync(start)) return [];
  const files = [];
  const stack = [start];
  while (stack.length) {
    const current = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    entries.forEach((entry) => {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        return;
      }
      if (entry.isFile() && entry.name.endsWith('.html')) files.push(full);
    });
  }
  return files.sort();
}

function listRootHtmlFiles() {
  return fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.html'))
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function relFromRoot(absPath) {
  return path.relative(root, absPath).replace(/\\/g, '/');
}

function toPathFromCanonical(canonical) {
  const raw = String(canonical || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (url.origin !== SITE_ORIGIN) return '';
    return normalizePathname(url.pathname || '/');
  } catch {
    return '';
  }
}

function toPathFromRelFile(relPath) {
  const safe = String(relPath || '').replace(/\\/g, '/');
  if (safe === 'index.html') return '/';
  if (safe.startsWith('pages/portfolio/') && safe.endsWith('.html')) {
    const id = safe.replace(/^pages\/portfolio\//, '').replace(/\.html$/, '');
    return id ? `/portfolio/${encodeURIComponent(id)}` : '';
  }
  if (safe.startsWith('pages/') && safe.endsWith('.html')) {
    const slug = safe.replace(/^pages\//, '').replace(/\.html$/, '');
    return slug ? `/${slug}` : '';
  }
  if (safe.endsWith('.html') && !safe.includes('/')) {
    const slug = safe.replace(/\.html$/, '');
    return slug ? `/${slug}` : '';
  }
  return '';
}

function categoryForPath(urlPath) {
  if (urlPath === '/portfolio' || urlPath.startsWith('/portfolio/')) return 'Portfolio';
  if (urlPath === '/tools' || urlPath.startsWith('/tools/')) return 'Tools';
  if (urlPath === '/games' || urlPath.startsWith('/games/')) return 'Games';
  if (['/', '/contact'].includes(urlPath)) return 'Core';
  return 'Pages';
}

function audienceForPath(urlPath) {
  void urlPath;
  return '';
}

function extractKeywords(html) {
  const keywords = [];
  const tagRe = /<span\b[^>]*\bclass="[^"]*\b(?:project-tag|tool-pill|game-pill|resume-chip)\b[^"]*"[^>]*>([\s\S]*?)<\/span>/gi;
  let match;
  while ((match = tagRe.exec(String(html || '')))) {
    const text = cleanText(match[1]);
    if (text) keywords.push(text);
  }
  return [...new Set(keywords)].slice(0, 30);
}

function collectProjectText(value, out = [], key = '') {
  if (value === null || value === undefined) return out;
  if (typeof value === 'string') {
    const text = value.trim();
    if (!text || /^https?:\/\//i.test(text) || /\.(?:png|jpe?g|webp|webm|mp4|svg|gif)$/i.test(text)) return out;
    out.push(text);
    return out;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectProjectText(item, out, key));
    return out;
  }
  if (typeof value === 'object') {
    Object.entries(value).forEach(([childKey, childValue]) => {
      if (/^(image|imageWidth|imageHeight|videoWebm|videoMp4|icon|url|order)$/i.test(childKey)) return;
      collectProjectText(childValue, out, childKey);
    });
  }
  return out;
}

function compactUnique(values, maxItems = 80) {
  const seen = new Set();
  return values
    .map((item) => cleanText(item))
    .filter(Boolean)
    .filter((item) => {
      const key = item.toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, maxItems);
}

function projectKeywords(project) {
  return compactUnique([
    project.id,
    project.title,
    project.subtitle,
    ...(Array.isArray(project.tools) ? project.tools : []),
    ...(Array.isArray(project.concepts) ? project.concepts : []),
    ...(Array.isArray(project.audiences) ? project.audiences : [])
  ], 40);
}

function loadProjectMetadata() {
  const dirPath = path.join(root, 'content', 'projects');
  if (!fs.existsSync(dirPath)) return new Map();

  const metadata = new Map();
  fs.readdirSync(dirPath)
    .filter((fileName) => fileName.endsWith('.json'))
    .sort()
    .forEach((fileName) => {
      let project;
      try {
        project = JSON.parse(fs.readFileSync(path.join(dirPath, fileName), 'utf8'));
      } catch {
        return;
      }
      const id = String(project.id || fileName.replace(/\.json$/, '')).trim();
      if (!id) return;
      const url = `/portfolio/${encodeURIComponent(id)}`;
      const text = compactUnique(collectProjectText(project), 140).join(' ');
      metadata.set(url, {
        description: cleanText(project.subtitle || project.notes || project.problem || ''),
        keywords: projectKeywords(project),
        text
      });
    });
  return metadata;
}

function cleanText(value) {
  return decodeHtml(String(value || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, ' ')
    .replace(/<[^>]+>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

function extractMainText(html) {
  const stripped = String(html || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ');
  const mainMatch = /<main\b[^>]*>([\s\S]*?)<\/main>/i.exec(stripped);
  let region = mainMatch ? mainMatch[1] : stripped;
  region = region
    .replace(/<header\b[^>]*>[\s\S]*?<\/header>/gi, ' ')
    .replace(/<nav\b[^>]*>[\s\S]*?<\/nav>/gi, ' ')
    .replace(/<footer\b[^>]*>[\s\S]*?<\/footer>/gi, ' ')
    .replace(/<article\b[^>]*\bdata-tools-visibility="[^"]+"[^>]*>[\s\S]*?<\/article>/gi, ' ')
    .replace(/<[^>]+\bhidden\b[^>]*>[\s\S]*?<\/[^>]+>/gi, ' ')
    .replace(/<button\b[^>]*>[\s\S]*?<\/button>/gi, ' ')
    .replace(/<form\b[^>]*>[\s\S]*?<\/form>/gi, ' ');
  return cleanText(region);
}

function splitIntoSentences(text) {
  return String(text || '')
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/g)
    .map((part) => part.trim())
    .filter(Boolean);
}

function chunkText(text) {
  const sentences = splitIntoSentences(text);
  const chunks = [];
  let current = '';

  sentences.forEach((sentence) => {
    const candidate = current ? `${current} ${sentence}` : sentence;
    if (candidate.length <= MAX_CHUNK_CHARS) {
      current = candidate;
      return;
    }
    if (current.length >= MIN_CHUNK_CHARS) chunks.push(current);
    current = sentence.length > MAX_CHUNK_CHARS ? sentence.slice(0, MAX_CHUNK_CHARS) : sentence;
  });

  if (current.length >= 120) chunks.push(current);
  if (!chunks.length && text) chunks.push(text.slice(0, MAX_CHUNK_CHARS));
  return chunks.slice(0, MAX_PAGE_CHUNKS);
}

function fallbackTextForPath(urlPath) {
  const normalized = normalizePathname(urlPath);
  if (normalized !== '/') return '';
  return [
    'Daniel Short is a data science and analytics professional who publishes portfolio projects, practical browser tools, and interactive games on this site.',
    'The homepage routes visitors into the project library, tools directory, games directory, resume, and contact page.',
    'Featured work includes analytics dashboards, SQL and ETL projects, machine learning experiments, data visualization, and utility tools that run in the browser.'
  ].join(' ');
}

function makeId(input) {
  return crypto.createHash('sha256').update(String(input || '')).digest('hex').slice(0, 16);
}

function vectorFromValue(value, dimensions) {
  const vector = Array.isArray(value) ? value : [];
  if (vector.length !== dimensions) return null;
  const normalized = vector.map((item) => Number(item));
  if (normalized.some((item) => !Number.isFinite(item))) return null;
  return normalized.map((item) => Number(item.toFixed(6)));
}

function loadExistingEmbeddingCache(modelId, dimensions) {
  if (!fs.existsSync(outPath)) return new Map();
  try {
    const previous = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    if (!previous || !previous.embeddings) return new Map();
    if (previous.embeddings.modelId !== modelId || Number(previous.embeddings.dimensions) !== dimensions) return new Map();
    const cache = new Map();
    (Array.isArray(previous.chunks) ? previous.chunks : []).forEach((chunk) => {
      const vector = vectorFromValue(chunk && chunk.embedding, dimensions);
      if (chunk && chunk.id && vector) cache.set(chunk.id, vector);
    });
    return cache;
  } catch {
    return new Map();
  }
}

function embeddingText(chunk) {
  return [
    chunk.title,
    chunk.url,
    chunk.category,
    chunk.audience,
    ...(Array.isArray(chunk.keywords) ? chunk.keywords : []),
    chunk.text
  ].filter(Boolean).join('\n').slice(0, 50_000);
}

async function embedText(client, modelId, dimensions, text) {
  const command = new InvokeModelCommand({
    modelId,
    contentType: 'application/json',
    accept: 'application/json',
    body: JSON.stringify({
      inputText: text,
      dimensions,
      normalize: true
    })
  });
  const response = await client.send(command);
  const raw = Buffer.from(response.body || []).toString('utf8');
  const parsed = JSON.parse(raw || '{}');
  const vector = vectorFromValue(parsed.embedding, dimensions);
  if (!vector) throw new Error('Bedrock returned an invalid embedding vector');
  return vector;
}

async function applyEmbeddings(knowledge) {
  const config = getEmbeddingConfig();
  const startedAt = Date.now();
  knowledge.embeddings = {
    enabled: config.enabled,
    status: config.enabled ? 'pending' : 'disabled',
    modelId: config.modelId,
    dimensions: config.dimensions,
    normalize: true,
    chunkCount: 0,
    generatedCount: 0,
    reusedCount: 0,
    failedCount: 0
  };

  if (!config.enabled) return knowledge;

  if (!hasBedrockBuildConfig()) {
    knowledge.embeddings.status = 'skipped';
    knowledge.embeddings.reason = 'Bedrock credentials are not configured for build-time embeddings.';
    if (config.required) throw new Error(knowledge.embeddings.reason);
    return knowledge;
  }

  const auth = getAwsCredentialConfig();
  const client = new BedrockRuntimeClient({
    region: getRegion(),
    credentials: auth.credentials
  });
  const cache = loadExistingEmbeddingCache(config.modelId, config.dimensions);

  for (const chunk of knowledge.chunks) {
    const cached = cache.get(chunk.id);
    if (cached) {
      chunk.embedding = cached;
      knowledge.embeddings.reusedCount += 1;
      continue;
    }

    try {
      chunk.embedding = await embedText(client, config.modelId, config.dimensions, embeddingText(chunk));
      knowledge.embeddings.generatedCount += 1;
    } catch (err) {
      knowledge.embeddings.failedCount += 1;
      knowledge.embeddings.reason = err && err.message ? err.message : String(err || 'Embedding generation failed');
      if (config.required) throw err;
      break;
    }
  }

  knowledge.embeddings.chunkCount = knowledge.chunks.filter((chunk) => Array.isArray(chunk.embedding)).length;
  knowledge.embeddings.status = knowledge.embeddings.failedCount
    ? (knowledge.embeddings.chunkCount ? 'partial' : 'skipped')
    : 'ready';
  knowledge.embeddings.generatedAt = new Date().toISOString();
  knowledge.embeddings.durationMs = Date.now() - startedAt;
  return knowledge;
}

function isExcludedUrl(urlPath, noindexPathnames) {
  const normalized = normalizePathname(urlPath);
  if (!normalized) return true;
  if (noindexPathnames.has(normalized)) return true;
  return excludedPathPatterns.some((pattern) => pattern.test(normalized));
}

function buildKnowledge() {
  const noindexPathnames = loadNoindexPathnamesFromVercel(root);
  const projectMetadata = loadProjectMetadata();
  const files = [...listRootHtmlFiles(), ...walkHtmlFiles('pages'), ...walkHtmlFiles('demos')];
  const pagesByUrl = new Map();

  files.forEach((absPath) => {
    const sourcePath = relFromRoot(absPath);
    let html = '';
    try {
      html = fs.readFileSync(absPath, 'utf8');
    } catch {
      return;
    }
    if (isNoindex(html)) return;

    const url = toPathFromCanonical(extractCanonical(html)) || toPathFromRelFile(sourcePath);
    if (isExcludedUrl(url, noindexPathnames)) return;

    const title = extractTitle(html) || url;
    const metadata = projectMetadata.get(url) || null;
    const description = metadata && metadata.description ? metadata.description : extractDescription(html);
    let text = [extractMainText(html), metadata && metadata.text].filter(Boolean).join(' ');
    if (url === '/' && text.length < 160) {
      text = [text, fallbackTextForPath(url)].filter(Boolean).join(' ');
    }
    const searchableText = [description, text].filter(Boolean).join(' ');
    if (!searchableText || searchableText.length < 160) return;

    const entry = {
      url,
      title,
      description,
      sourcePath,
      category: categoryForPath(url),
      audience: audienceForPath(url),
      keywords: compactUnique([...extractKeywords(html), ...((metadata && metadata.keywords) || [])], 60),
      text: text || description
    };

    const previous = pagesByUrl.get(url);
    if (!previous || entry.text.length > previous.text.length) pagesByUrl.set(url, entry);
  });

  const chunks = [];
  [...pagesByUrl.values()]
    .sort((a, b) => a.url.localeCompare(b.url))
    .forEach((page) => {
      chunkText([page.description, page.text].filter(Boolean).join(' ')).forEach((text, index) => {
        chunks.push({
          id: makeId(`${page.url}:${index}:${text}`),
          pageId: makeId(page.url),
          url: page.url,
          title: page.title,
          category: page.category,
          audience: page.audience,
          sourcePath: page.sourcePath,
          keywords: page.keywords,
          text
        });
      });
    });

  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    origin: SITE_ORIGIN,
    pages: [...pagesByUrl.values()].map(({ text, ...page }) => page),
    chunks
  };
}

async function main() {
  ensureDir(path.dirname(outPath));
  const knowledge = await applyEmbeddings(buildKnowledge());
  fs.writeFileSync(outPath, JSON.stringify(knowledge, null, 2) + '\n', 'utf8');
  const embeddingStatus = knowledge.embeddings && knowledge.embeddings.enabled
    ? `, embeddings ${knowledge.embeddings.status} (${knowledge.embeddings.chunkCount || 0}/${knowledge.chunks.length})`
    : '';
  process.stdout.write(`[chatbot-knowledge] Wrote dist/chatbot-knowledge.json (${knowledge.pages.length} pages, ${knowledge.chunks.length} chunks${embeddingStatus})\n`);
}

main().catch((err) => {
  console.error('[chatbot-knowledge] Failed:', err && err.stack ? err.stack : err);
  process.exitCode = 1;
});
