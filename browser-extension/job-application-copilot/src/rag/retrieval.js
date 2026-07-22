import { SOURCE_ROLES, VALIDATION_LIMITS } from '../shared/schemas.js';
import { tokenizeForRetrieval } from './chunker.js';

const DEFAULT_ROLE_PRIORITY = Object.freeze({
  [SOURCE_ROLES.USER_VERIFIED]: 1,
  [SOURCE_ROLES.CANDIDATE_EVIDENCE]: 0.95,
  [SOURCE_ROLES.JOB_REQUIREMENT]: 0.85,
  [SOURCE_ROLES.COMPANY_CONTEXT]: 0.8,
  [SOURCE_ROLES.STYLE_EXAMPLE]: 0.5
});

export const cosineSimilarity = (left, right) => {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length || !left.length) return 0;
  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = Number(left[index]);
    const rightValue = Number(right[index]);
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) return 0;
    dot += leftValue * rightValue;
    leftMagnitude += leftValue * leftValue;
    rightMagnitude += rightValue * rightValue;
  }
  if (!leftMagnitude || !rightMagnitude) return 0;
  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude));
};

export const lexicalSimilarity = (query, chunk) => {
  const queryTerms = [...new Set(tokenizeForRetrieval(query))];
  if (!queryTerms.length) return 0;
  const chunkTerms = chunk.terms?.length ? chunk.terms : tokenizeForRetrieval(chunk.text);
  const frequencies = new Map();
  chunkTerms.forEach(term => frequencies.set(term, (frequencies.get(term) || 0) + 1));
  const matched = queryTerms.reduce((total, term) => total + Math.min(1, frequencies.get(term) || 0), 0);
  const phraseBonus = String(chunk.text || '').toLocaleLowerCase('en-US')
    .includes(queryTerms.join(' ')) ? 0.15 : 0;
  return Math.min(1, (matched / queryTerms.length) + phraseBonus);
};

const matchesFilter = (chunk, { applicationId, documentIds, sourceRoles } = {}) => {
  if (applicationId !== undefined && chunk.applicationId !== applicationId) return false;
  if (documentIds?.length && !documentIds.includes(chunk.documentId)) return false;
  if (sourceRoles?.length && !sourceRoles.includes(chunk.sourceRole)) return false;
  return true;
};

export const hybridRetrieve = ({
  queryText,
  queryEmbedding = null,
  chunks,
  limit = 6,
  filters = {},
  vectorWeight = queryEmbedding ? 0.65 : 0,
  lexicalWeight = queryEmbedding ? 0.25 : 0.9,
  roleWeight = 0.1,
  minScore = 0.05,
  minVectorSimilarity = 0.08,
  rolePriority = DEFAULT_ROLE_PRIORITY
}) => {
  if (!Array.isArray(chunks)) throw new Error('chunks must be an array.');
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > VALIDATION_LIMITS.maxEvidenceChunksPerPage) {
    throw new Error('limit is outside the supported retrieval range.');
  }
  return chunks
    .filter(chunk => matchesFilter(chunk, filters))
    .map(chunk => {
      const vectorScore = queryEmbedding ? Math.max(0, cosineSimilarity(queryEmbedding, chunk.embedding)) : 0;
      const lexicalScore = lexicalSimilarity(queryText, chunk);
      const roleScore = rolePriority[chunk.sourceRole] ?? 0.5;
      return {
        ...chunk,
        retrieval: {
          score: (vectorScore * vectorWeight) + (lexicalScore * lexicalWeight) + (roleScore * roleWeight),
          vectorScore,
          lexicalScore,
          roleScore
        }
      };
    })
    .filter(chunk => chunk.retrieval.lexicalScore > 0
      || chunk.retrieval.vectorScore >= minVectorSimilarity)
    .filter(chunk => chunk.retrieval.score >= minScore)
    .sort((left, right) => right.retrieval.score - left.retrieval.score || left.id.localeCompare(right.id))
    .slice(0, limit);
};

export const buildEvidencePack = (fieldResults, {
  maxChunks = VALIDATION_LIMITS.maxEvidenceChunksPerPage
} = {}) => {
  if (!Array.isArray(fieldResults)) throw new Error('fieldResults must be an array.');
  const bestByChunk = new Map();
  const fieldChunkIds = new Map();
  for (const entry of fieldResults) {
    if (!entry?.fieldId || !Array.isArray(entry.results)) continue;
    const ids = [];
    for (const result of entry.results) {
      ids.push(result.id);
      const existing = bestByChunk.get(result.id);
      if (!existing || (result.retrieval?.score || 0) > (existing.retrieval?.score || 0)) {
        bestByChunk.set(result.id, result);
      }
    }
    fieldChunkIds.set(entry.fieldId, ids);
  }
  const selected = [];
  const selectedIds = new Set();
  for (const entry of fieldResults) {
    const first = entry?.results?.find(result => !selectedIds.has(result.id));
    if (!first || selected.length >= maxChunks) continue;
    selected.push(bestByChunk.get(first.id) || first);
    selectedIds.add(first.id);
  }
  const ranked = [...bestByChunk.values()]
    .sort((left, right) => (right.retrieval?.score || 0) - (left.retrieval?.score || 0) || left.id.localeCompare(right.id));
  for (const result of ranked) {
    if (selected.length >= maxChunks) break;
    if (selectedIds.has(result.id)) continue;
    selected.push(result);
    selectedIds.add(result.id);
  }
  const citationByChunk = new Map(selected.map((chunk, index) => [chunk.id, `c${index + 1}`]));
  const citations = selected.map(chunk => ({
    citationId: citationByChunk.get(chunk.id),
    documentId: chunk.documentId,
    documentVersion: chunk.documentVersion,
    chunkId: chunk.id,
    sourceRole: chunk.sourceRole,
    locator: chunk.locator,
    quoteHash: chunk.quoteHash,
    text: chunk.text
  }));
  const byField = {};
  for (const [fieldId, ids] of fieldChunkIds) {
    byField[fieldId] = ids.map(id => citationByChunk.get(id)).filter(Boolean);
  }
  return { citations, byField };
};

