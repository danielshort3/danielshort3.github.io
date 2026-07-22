import { sha256Base64Url } from '../vault/crypto.js';

const MIME_ALIASES = new Map([
  ['application/pdf', 'pdf'],
  ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'docx'],
  ['text/html', 'html'],
  ['application/xhtml+xml', 'html'],
  ['text/plain', 'text'],
  ['text/markdown', 'text']
]);

const extensionKind = (name) => {
  const extension = String(name || '').toLowerCase().split('.').pop();
  if (extension === 'pdf') return 'pdf';
  if (extension === 'doc') return 'legacy-doc';
  if (extension === 'docx') return 'docx';
  if (extension === 'html' || extension === 'htm') return 'html';
  if (extension === 'txt' || extension === 'md') return 'text';
  return '';
};

const CANONICAL_MIME_TYPES = Object.freeze({
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  html: 'text/html',
  text: 'text/plain'
});

const normalizeInput = ({ name, type, data, lastModified }) => {
  if (typeof name !== 'string' || !name.trim()) throw new Error('Document name is required.');
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data || 0);
  if (!bytes.byteLength) throw new Error('Document is empty.');
  return {
    name: name.trim().slice(0, 500),
    type: String(type || 'application/octet-stream').toLowerCase(),
    data: bytes,
    lastModified: Number.isFinite(lastModified) ? Number(lastModified) : null
  };
};

export const parseDocument = async (input) => {
  const normalized = normalizeInput(input);
  if (normalized.type === 'application/msword' || extensionKind(normalized.name) === 'legacy-doc') {
    throw new Error('Legacy .doc files are not supported; convert the document to DOCX or PDF first.');
  }
  const kind = MIME_ALIASES.get(normalized.type) || extensionKind(normalized.name);
  let parser;
  if (kind === 'pdf') parser = (await import('./pdf.js')).parsePdfDocument;
  else if (kind === 'docx') parser = (await import('./docx.js')).parseDocxDocument;
  else if (kind === 'html') parser = (await import('./html.js')).parseHtmlDocument;
  else if (kind === 'text') parser = (await import('./text.js')).parseTextDocument;
  else throw new Error(`Unsupported document type: ${normalized.type}`);

  // PDF.js transfers (and therefore detaches) the Uint8Array it receives. Capture
  // immutable source metadata first and give format parsers their own copy so a
  // successful PDF import cannot be mislabeled as an empty file.
  const sourceSize = normalized.data.byteLength;
  const sha256 = await sha256Base64Url(normalized.data);
  const parsed = await parser({ ...normalized, data: normalized.data.slice() });
  const extractedText = typeof parsed.text === 'string' ? parsed.text.trim() : '';
  const extractedBlocks = Array.isArray(parsed.blocks)
    ? parsed.blocks.filter(block => typeof block?.text === 'string' && block.text.trim())
    : [];
  if (!extractedText || !extractedBlocks.length) {
    const label = kind === 'text' ? 'Text file'
      : kind === 'html' ? 'HTML document'
        : kind === 'docx' ? 'DOCX document'
          : 'PDF document';
    throw new Error(`${label} has no extractable text; use OCR, convert the file, or paste the text.`);
  }
  return {
    document: {
      id: `doc:${sha256.slice(0, 24)}`,
      filename: normalized.name,
      mimeType: CANONICAL_MIME_TYPES[kind],
      size: sourceSize,
      lastModified: normalized.lastModified,
      sha256,
      version: 1
    },
    text: extractedText,
    blocks: extractedBlocks,
    warnings: parsed.warnings || []
  };
};
