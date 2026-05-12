'use strict';

const fs = require('fs');
const path = require('path');

const STOPWORDS = new Set([
  'a', 'about', 'after', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
  'be', 'been', 'but', 'by', 'can', 'could', 'did', 'do', 'does', 'for', 'from',
  'had', 'has', 'have', 'he', 'her', 'him', 'his', 'how', 'i', 'if', 'in', 'into',
  'is', 'it', 'me', 'my', 'of', 'on', 'or', 'our', 'she', 'should', 'so', 'that',
  'the', 'their', 'them', 'there', 'they', 'this', 'to', 'was', 'we', 'were',
  'what', 'when', 'where', 'which', 'who', 'why', 'with', 'would', 'you', 'your'
]);

let cachedKnowledge = null;
let cachedMtime = 0;

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
  const terms = String(value || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .match(/[a-z0-9][a-z0-9+#.-]{1,}/g);
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

function scoreChunk(chunk, terms, pageContext) {
  if (!terms.length) return 0;
  const text = chunk._searchText || '';
  let score = 0;

  terms.forEach((term) => {
    if (!term) return;
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const count = (text.match(new RegExp(`\\b${escaped}`, 'g')) || []).length;
    if (count) score += Math.min(6, count);

    const title = String(chunk.title || '').toLowerCase();
    if (title.includes(term)) score += 3;
    const keywords = Array.isArray(chunk.keywords) ? chunk.keywords.join(' ').toLowerCase() : '';
    if (keywords.includes(term)) score += 2;
    const url = String(chunk.url || '').toLowerCase();
    if (url.includes(term)) score += 2;
  });

  const contextPath = normalizePath(pageContext && pageContext.url);
  if (contextPath && normalizePath(chunk.url) === contextPath) score += 5;
  if (contextPath && normalizePath(chunk.url).startsWith(contextPath + '/')) score += 2;

  return score;
}

function retrieveKnowledge(message, pageContext = {}, options = {}) {
  const knowledge = loadKnowledge();
  const terms = tokenize([message, pageContext && pageContext.title].filter(Boolean).join(' '));
  const maxChunks = Math.max(1, Math.min(8, Number(options.maxChunks) || 5));
  const minScore = Math.max(1, Number(options.minScore) || 4);

  const ranked = knowledge.chunks
    .map((chunk) => ({ chunk, score: scoreChunk(chunk, terms, pageContext) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || String(a.chunk.url).localeCompare(String(b.chunk.url)))
    .slice(0, maxChunks);

  const bestScore = ranked.length ? ranked[0].score : 0;
  return {
    knowledge,
    queryTerms: terms,
    bestScore,
    confident: bestScore >= minScore,
    chunks: ranked.map(({ chunk, score }) => ({ ...chunk, score }))
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
  loadKnowledge,
  publicSources,
  retrieveKnowledge,
  tokenize
};
