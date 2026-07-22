import { parseDocument } from '../parsers/document-parser.js';

self.addEventListener('message', async (event) => {
  const requestId = event.data?.requestId;
  try {
    const result = await parseDocument(event.data?.document || {});
    self.postMessage({ requestId, ok: true, result });
  } catch (error) {
    self.postMessage({
      requestId,
      ok: false,
      error: {
        name: error?.name || 'Error',
        message: error?.message || 'Unable to parse document.'
      }
    });
  }
});

