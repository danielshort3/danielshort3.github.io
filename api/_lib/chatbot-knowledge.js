'use strict';

const fs = require('fs');
const path = require('path');

const STOPWORDS = new Set([
  'a', 'about', 'after', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
  'be', 'been', 'but', 'by', 'can', 'could', 'did', 'do', 'does', 'for', 'from',
  'had', 'has', 'have', 'he', 'her', 'him', 'his', 'how', 'i', 'if', 'in', 'into',
  'is', 'it', 'me', 'my', 'of', 'on', 'or', 'our', 'she', 'should', 'so', 'that',
  'the', 'their', 'them', 'there', 'they', 'this', 'to', 'was', 'we', 'were',
  'what', 'when', 'where', 'which', 'who', 'why', 'with', 'would', 'you', 'your',
  'daniel', 'short', 'website', 'site', 'page', 'pages', 'please', 'show', 'tell',
  'find', 'looking', 'look', 'learn', 'use', 'using', 'want', 'wants', 'need',
  'needs', 'help', 'helps', 'thing', 'things', 'info', 'information'
]);

let cachedKnowledge = null;
let cachedMtime = 0;

function normalizeText(value) {
  return String(value || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9+#./-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function termCount(text, term) {
  if (!text || !term) return 0;
  const pattern = new RegExp(`(?:^|[^a-z0-9+#.-])${escapeRegExp(term)}`, 'g');
  const matches = String(text).match(pattern);
  return matches ? matches.length : 0;
}

function vectorFromValue(value) {
  const vector = Array.isArray(value) ? value.map((item) => Number(item)) : [];
  if (!vector.length || vector.some((item) => !Number.isFinite(item))) return null;
  return vector;
}

function dotProduct(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return 0;
  let score = 0;
  for (let index = 0; index < a.length; index += 1) score += a[index] * b[index];
  return score;
}

function detectQueryIntent(message) {
  const text = normalizeText(message);
  const has = (pattern) => pattern.test(text);
  const grandJunction = text.includes('grand junction') || text.includes('visit grand junction') || has(/\bvgj\b/);
  return {
    contact: has(/\b(contact|email|hire|reach|linkedin|github|message)\b/),
    resume: has(/\b(resume|cv|work history|qualification|qualified)\b/),
    portfolio: has(/\b(portfolio|project|projects|case study|case studies|dashboard|example|sample|work sample)\b/),
    analytics: has(/\b(analytics|bi|tableau|sql|reporting|dashboard|kpi|forecast|forecasting)\b/),
    dataScience: has(/\b(data science|machine learning|ml|model|models|python|nlp|rag|lora)\b/),
    tourism: grandJunction || has(/\b(tourism|destination|travel|visitor|visitors|lodging|hotel|attraction)\b/),
    chatbotProject: has(/\b(chatbot|rag|lora|bedrock|sagemaker|qwen|mistral)\b/) &&
      (grandJunction || has(/\b(tourism|visitor|destination|project|demo)\b/))
  };
}

function candidatePaths() {
  const cwd = process.cwd();
  return [
    path.join(cwd, 'dist', 'chatbot-knowledge.json'),
    path.join(cwd, 'public', 'dist', 'chatbot-knowledge.json'),
    path.join(__dirname, '..', '..', 'dist', 'chatbot-knowledge.json'),
    path.join(__dirname, '..', '..', 'public', 'dist', 'chatbot-knowledge.json')
  ];
}

function loadKnowledge() {
  const knowledgePath = candidatePaths().find((filePath) => fs.existsSync(filePath));
  if (!knowledgePath) {
    const err = new Error('Chatbot knowledge database is missing. Run npm run build.');
    err.code = 'CHATBOT_KNOWLEDGE_MISSING';
    throw err;
  }

  const stat = fs.statSync(knowledgePath);
  if (cachedKnowledge && cachedMtime === stat.mtimeMs) return cachedKnowledge;

  const parsed = JSON.parse(fs.readFileSync(knowledgePath, 'utf8'));
  const chunks = Array.isArray(parsed.chunks) ? parsed.chunks : [];
  cachedKnowledge = {
    ...parsed,
    chunks: chunks.map((chunk) => ({
      ...chunk,
      _titleText: normalizeText(chunk.title),
      _urlText: normalizeText(chunk.url),
      _keywordText: normalizeText(Array.isArray(chunk.keywords) ? chunk.keywords.join(' ') : ''),
      _metaText: normalizeText([chunk.category, chunk.audience].filter(Boolean).join(' ')),
      _bodyText: normalizeText(chunk.text),
      _embedding: vectorFromValue(chunk.embedding),
      _searchText: [
        chunk.title,
        chunk.url,
        chunk.category,
        chunk.audience,
        ...(Array.isArray(chunk.keywords) ? chunk.keywords : []),
        chunk.text
      ].filter(Boolean).join(' ').toLowerCase()
    }))
  };
  cachedMtime = stat.mtimeMs;
  return cachedKnowledge;
}

function tokenize(value) {
  const terms = normalizeText(value).match(/[a-z0-9][a-z0-9+#.-]{1,}/g);
  if (!terms) return [];
  return [...new Set(terms.filter((term) => !STOPWORDS.has(term) && term.length > 1))].slice(0, 40);
}

function normalizePath(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw, 'https://www.danielshort.me');
    return (url.pathname || '').replace(/\/+$/, '') || '/';
  } catch {
    return raw.replace(/[?#].*$/, '').replace(/\/+$/, '') || '/';
  }
}

function scoreIntentBoost(chunk, intent) {
  const url = normalizePath(chunk.url);
  let score = 0;

  if (intent.contact && url === '/contact') score += 60;

  if (intent.resume) {
    const analyticsResumeMatch = intent.analytics || (!intent.dataScience && !intent.tourism);
    if (url === '/resume-analytics') score += analyticsResumeMatch ? 42 : 24;
    if (url === '/resume-data-science') score += intent.dataScience ? 42 : 18;
    if (url === '/resume-tourism') score += intent.tourism ? 42 : 18;
    if (url.includes('resume')) score += 8;
  }

  if (intent.chatbotProject && url === '/portfolio/chatbotLora') score += 80;
  if (intent.portfolio && (url === '/portfolio' || url.startsWith('/portfolio/'))) score += url === '/portfolio' ? 12 : 18;
  if (intent.analytics && (url === '/analytics' || chunk.audience === 'analytics')) score += 14;
  if (intent.dataScience && (url === '/data-science' || chunk.audience === 'data-science')) score += 14;
  if (intent.tourism && (url === '/tourism' || chunk.audience === 'tourism')) score += 14;

  return score;
}

function scoreChunk(chunk, terms, pageContext, message = '') {
  if (!terms.length) return 0;
  const title = chunk._titleText || normalizeText(chunk.title);
  const url = chunk._urlText || normalizeText(chunk.url);
  const keywords = chunk._keywordText || normalizeText(Array.isArray(chunk.keywords) ? chunk.keywords.join(' ') : '');
  const meta = chunk._metaText || normalizeText([chunk.category, chunk.audience].filter(Boolean).join(' '));
  const body = chunk._bodyText || normalizeText(chunk.text);
  const intent = detectQueryIntent(message);
  let score = 0;

  terms.forEach((term) => {
    if (!term) return;
    score += Math.min(5, termCount(body, term));
    if (termCount(title, term)) score += 8;
    if (termCount(keywords, term)) score += 7;
    if (termCount(url, term)) score += 5;
    if (termCount(meta, term)) score += 3;
  });

  score += scoreIntentBoost(chunk, intent);

  const contextPath = normalizePath(pageContext && pageContext.url);
  if (contextPath && normalizePath(chunk.url) === contextPath) score += 1;
  if (contextPath && normalizePath(chunk.url).startsWith(contextPath + '/')) score += 0.5;

  return score;
}

function retrieveKnowledge(message, pageContext = {}, options = {}) {
  const knowledge = loadKnowledge();
  const terms = tokenize(message);
  const maxChunks = Math.max(1, Math.min(8, Number(options.maxChunks) || 5));
  const minScore = Math.max(1, Number(options.minScore) || 4);
  const queryEmbedding = vectorFromValue(options.queryEmbedding);
  const canUseEmbedding = Boolean(queryEmbedding && knowledge.chunks.some((chunk) => Array.isArray(chunk._embedding)));

  const ranked = knowledge.chunks
    .map((chunk) => {
      const lexicalScore = scoreChunk(chunk, terms, pageContext, message);
      const embeddingScore = canUseEmbedding && chunk._embedding
        ? Math.max(0, dotProduct(queryEmbedding, chunk._embedding))
        : 0;
      const score = canUseEmbedding
        ? lexicalScore + (embeddingScore * 85)
        : lexicalScore;
      return { chunk, score, lexicalScore, embeddingScore };
    })
    .filter((entry) => entry.score > 0 && (!canUseEmbedding || entry.lexicalScore > 0 || entry.embeddingScore >= 0.12))
    .sort((a, b) => b.score - a.score || String(a.chunk.url).localeCompare(String(b.chunk.url)))
    .slice(0, maxChunks);

  const bestScore = ranked.length ? ranked[0].score : 0;
  return {
    knowledge,
    queryTerms: terms,
    bestScore,
    bestEmbeddingScore: ranked.length ? ranked[0].embeddingScore : 0,
    confident: bestScore >= minScore,
    retrievalMode: canUseEmbedding ? 'embedding' : 'lexical',
    embeddingModel: canUseEmbedding ? String(options.embeddingModel || knowledge.embeddings && knowledge.embeddings.modelId || '') : '',
    embeddingDimensions: canUseEmbedding ? Number(options.embeddingDimensions || knowledge.embeddings && knowledge.embeddings.dimensions || 0) : 0,
    chunks: ranked.map(({ chunk, score, lexicalScore, embeddingScore }) => ({
      ...chunk,
      score,
      lexicalScore,
      embeddingScore
    }))
  };
}

function publicSources(chunks, maxSources = 5) {
  const byUrl = new Map();
  chunks.forEach((chunk) => {
    const url = String(chunk.url || '').trim();
    if (!url || byUrl.has(url)) return;
    byUrl.set(url, {
      id: chunk.id,
      title: chunk.title || url,
      url,
      category: chunk.category || '',
      sourcePath: chunk.sourcePath || ''
    });
  });
  return [...byUrl.values()].slice(0, maxSources);
}

module.exports = {
  detectQueryIntent,
  loadKnowledge,
  publicSources,
  retrieveKnowledge,
  scoreChunk,
  tokenize
};
