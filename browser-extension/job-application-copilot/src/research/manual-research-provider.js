import { SOURCE_ROLES } from '../shared/schemas.js';
import { sha256Base64Url } from '../vault/crypto.js';

const normalizeUrl = (value) => {
  if (!String(value || '').trim()) return null;
  const url = new URL(String(value || ''));
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Manual research URL must use HTTP or HTTPS.');
  url.hash = '';
  return url.toString();
};

const normalizeText = (value, label) => {
  const text = String(value || '').replace(/\s+/gu, ' ').trim();
  if (!text) throw new Error(`${label} is required.`);
  return text;
};

export class ManualResearchProvider {
  get type() {
    return 'manual';
  }

  async createSnapshot({
    title,
    url,
    text,
    retrievedAt = new Date().toISOString(),
    applicationId = null,
    notes = ''
  }) {
    const normalizedTitle = String(title || '').replace(/\s+/gu, ' ').trim().slice(0, 500)
      || 'Pasted research note';
    const normalizedUrl = normalizeUrl(url);
    const normalizedText = normalizeText(text, 'Research text');
    const timestamp = new Date(retrievedAt);
    if (Number.isNaN(timestamp.getTime())) throw new Error('retrievedAt must be a valid date.');
    const sha256 = await sha256Base64Url(`${normalizedTitle}\n${normalizedUrl || ''}\n${normalizedText}`);
    const paragraphs = String(text)
      .replaceAll('\r\n', '\n')
      .replaceAll('\r', '\n')
      .split(/\n{2,}/u)
      .map(value => value.replace(/\s+/gu, ' ').trim())
      .filter(Boolean);
    return {
      provider: 'manual',
      document: {
        id: `research:${sha256.slice(0, 24)}`,
        filename: normalizedTitle,
        mimeType: 'text/plain',
        size: new TextEncoder().encode(normalizedText).byteLength,
        sha256,
        version: 1,
        sourceRole: SOURCE_ROLES.COMPANY_CONTEXT,
        sourceUrl: normalizedUrl,
        retrievedAt: timestamp.toISOString(),
        applicationId: applicationId || null,
        notes: String(notes || '').trim().slice(0, 2000)
      },
      text: normalizedText,
      blocks: paragraphs.map((paragraph, index) => ({
        blockIndex: index,
        paragraph: index + 1,
        section: normalizedTitle,
        page: null,
        text: paragraph
      })),
      warnings: []
    };
  }
}
