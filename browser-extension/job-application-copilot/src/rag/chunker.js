import { SOURCE_ROLES } from '../shared/schemas.js';
import { validateSourceRole } from '../shared/validators.js';
import { sha256Base64Url } from '../vault/crypto.js';

const normalizeWhitespace = (value) => String(value || '').replace(/\s+/gu, ' ').trim();

export const tokenizeForRetrieval = (value) => normalizeWhitespace(value)
  .toLocaleLowerCase('en-US')
  .match(/[\p{L}\p{N}][\p{L}\p{N}+#._/-]*/gu) || [];

const fnv1a = (value) => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
};

const splitLongBlock = (block, maxTokens) => {
  const words = normalizeWhitespace(block.text).split(' ').filter(Boolean);
  if (words.length <= maxTokens) return [{ ...block, text: words.join(' '), tokenCount: words.length }];
  const segments = [];
  for (let start = 0; start < words.length; start += maxTokens) {
    const text = words.slice(start, start + maxTokens).join(' ');
    segments.push({
      ...block,
      text,
      tokenCount: Math.min(maxTokens, words.length - start),
      segmentStartToken: start,
      segmentEndToken: Math.min(words.length, start + maxTokens)
    });
  }
  return segments;
};

const compatibleLocation = (left, right) => {
  if (!left) return true;
  if (left.section && right.section && left.section !== right.section) return false;
  if (left.page !== null && right.page !== null && left.page !== right.page) return false;
  return true;
};

export const chunkDocument = async (documentRecord, {
  maxTokens = 500,
  overlapTokens = 80,
  sourceRole = documentRecord?.document?.sourceRole || SOURCE_ROLES.CANDIDATE_EVIDENCE
} = {}) => {
  if (!documentRecord?.document?.id || !Array.isArray(documentRecord.blocks)) {
    throw new Error('A document vault record with blocks is required.');
  }
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 50) throw new Error('maxTokens must be at least 50.');
  if (!Number.isSafeInteger(overlapTokens) || overlapTokens < 0 || overlapTokens >= maxTokens) {
    throw new Error('overlapTokens must be non-negative and smaller than maxTokens.');
  }
  validateSourceRole(sourceRole);

  const segments = documentRecord.blocks
    .flatMap(block => splitLongBlock(block, maxTokens))
    .filter(segment => segment.text);
  const groups = [];
  let current = [];
  let currentTokenCount = 0;

  const emit = () => {
    if (!current.length) return;
    groups.push(current);
    const overlap = overlapTokens
      ? current.map(segment => segment.text).join(' ').split(' ').slice(-overlapTokens).join(' ')
      : '';
    const last = current[current.length - 1];
    current = overlap ? [{ ...last, text: overlap, tokenCount: tokenizeForRetrieval(overlap).length, overlap: true }] : [];
    currentTokenCount = current.reduce((total, segment) => total + segment.tokenCount, 0);
  };

  for (const segment of segments) {
    if (current.length && !compatibleLocation(current[0], segment)) {
      groups.push(current);
      current = [];
      currentTokenCount = 0;
    } else if (current.length && currentTokenCount + segment.tokenCount > maxTokens) {
      emit();
    }
    current.push(segment);
    currentTokenCount += segment.tokenCount;
  }
  if (current.length) groups.push(current);

  return Promise.all(groups.map(async (group, index) => {
    const first = group[0];
    const last = group[group.length - 1];
    const text = group.map(segment => segment.text).join('\n').trim();
    const quoteHash = await sha256Base64Url(text);
    const idSeed = `${documentRecord.document.id}|${documentRecord.document.version || 1}|${index}|${quoteHash}`;
    return {
      id: `chunk:${fnv1a(idSeed)}:${index}`,
      documentId: documentRecord.document.id,
      documentVersion: documentRecord.document.version || 1,
      applicationId: documentRecord.document.applicationId || null,
      sourceRole,
      text,
      tokenCount: tokenizeForRetrieval(text).length,
      terms: tokenizeForRetrieval(text),
      quoteHash,
      locator: {
        pageStart: first.page ?? null,
        pageEnd: last.page ?? first.page ?? null,
        section: first.section || last.section || '',
        paragraphStart: first.paragraph ?? null,
        paragraphEnd: last.paragraph ?? first.paragraph ?? null,
        blockStart: first.blockIndex ?? null,
        blockEnd: last.blockIndex ?? first.blockIndex ?? null
      }
    };
  }));
};
