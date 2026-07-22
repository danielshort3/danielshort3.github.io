import { SOURCE_ROLES } from '../shared/schemas.js';
import { validateSourceRole } from '../shared/validators.js';
import { bytesToBase64 } from '../vault/crypto.js';

export const createDocumentVaultRecord = (parsedDocument, {
  sourceRole = SOURCE_ROLES.CANDIDATE_EVIDENCE,
  applicationId = null,
  importedAt = new Date().toISOString(),
  retainOriginal = false,
  originalBytes = null
} = {}) => {
  validateSourceRole(sourceRole);
  if (!parsedDocument?.document?.id || !Array.isArray(parsedDocument.blocks)) {
    throw new Error('A parsed document result is required.');
  }
  if (retainOriginal && !originalBytes) throw new Error('originalBytes is required when retainOriginal is true.');
  return {
    document: {
      ...parsedDocument.document,
      sourceRole,
      applicationId: applicationId || null,
      importedAt
    },
    blocks: parsedDocument.blocks,
    text: parsedDocument.text,
    warnings: parsedDocument.warnings || [],
    originalBytesBase64: retainOriginal ? bytesToBase64(originalBytes) : null
  };
};

