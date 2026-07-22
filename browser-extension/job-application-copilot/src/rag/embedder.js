import { DEFAULT_LOCAL_MODEL_CONFIG } from '../shared/schemas.js';

export const embedChunks = async ({
  client,
  chunks,
  model = DEFAULT_LOCAL_MODEL_CONFIG.embeddingModel,
  batchSize = 32,
  signal,
  onProgress
}) => {
  if (!client?.embed) throw new Error('An OllamaClient instance is required.');
  if (!Array.isArray(chunks)) throw new Error('chunks must be an array.');
  if (!Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 128) {
    throw new Error('batchSize must be between 1 and 128.');
  }
  const embedded = [];
  for (let offset = 0; offset < chunks.length; offset += batchSize) {
    if (signal?.aborted) throw signal.reason || new DOMException('Aborted', 'AbortError');
    const batch = chunks.slice(offset, offset + batchSize);
    const response = await client.embed({
      model,
      input: batch.map(chunk => chunk.text),
      keepAlive: DEFAULT_LOCAL_MODEL_CONFIG.keepAlive,
      signal
    });
    response.embeddings.forEach((embedding, index) => {
      embedded.push({ ...batch[index], embedding });
    });
    onProgress?.({ completed: embedded.length, total: chunks.length });
  }
  return embedded;
};

