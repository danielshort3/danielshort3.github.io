const ALLOWED_ORIGIN = 'http://127.0.0.1:11434';
const ALLOWED_PATHS = new Set(['/api/version', '/api/tags', '/api/show', '/api/embed', '/api/chat']);
const CLOUD_MODEL_NAME_PATTERN = /(?:^|[:/-])cloud(?:$|[:/-])/iu;
const REMOTE_METADATA_KEYS = new Set(['remote_host', 'remote_model', 'remote_url', 'cloud_model']);

export const isCloudModelName = value => CLOUD_MODEL_NAME_PATTERN.test(String(value || '').trim());

export const hasRemoteModelMetadata = (value, seen = new Set()) => {
  if (!value || typeof value !== 'object' || seen.has(value)) return false;
  seen.add(value);
  for (const [key, entry] of Object.entries(value)) {
    const normalizedKey = key.toLocaleLowerCase('en-US').replaceAll('-', '_');
    if (REMOTE_METADATA_KEYS.has(normalizedKey)) {
      if (typeof entry === 'string' && entry.trim()) return true;
      if (typeof entry === 'boolean' && entry) return true;
      if (typeof entry === 'number' && entry !== 0) return true;
      if (entry && typeof entry === 'object' && Object.keys(entry).length) return true;
    }
    if (hasRemoteModelMetadata(entry, seen)) return true;
  }
  return false;
};

export const assertLocalModelName = (value) => {
  const model = String(value || '').trim();
  if (!model) throw new Error('An Ollama model name is required.');
  if (isCloudModelName(model)) {
    throw new OllamaError('Cloud or remote Ollama models are not allowed. Choose a model stored on this device.', 'REMOTE_MODEL_NOT_ALLOWED');
  }
  return model;
};

const createAbortContext = (externalSignal, timeoutMs) => {
  const controller = new AbortController();
  const abort = () => controller.abort(externalSignal?.reason || new DOMException('Aborted', 'AbortError'));
  if (externalSignal?.aborted) abort();
  else externalSignal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => controller.abort(new DOMException('Ollama request timed out', 'TimeoutError')), timeoutMs);
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer);
      externalSignal?.removeEventListener('abort', abort);
    }
  };
};

const ensureAllowedBaseUrl = (value) => {
  const url = new URL(value);
  if (url.origin !== ALLOWED_ORIGIN || (url.pathname !== '/' && url.pathname !== '')) {
    throw new Error(`Ollama base URL must be exactly ${ALLOWED_ORIGIN}.`);
  }
  return url.origin;
};

const readBoundedText = async (response, maxBytes = 4 * 1024 * 1024) => {
  const text = await response.text();
  if (new TextEncoder().encode(text).byteLength > maxBytes) {
    throw new OllamaError('Ollama response exceeded the local size limit.', 'RESPONSE_TOO_LARGE', response.status);
  }
  return text;
};

export class OllamaError extends Error {
  constructor(message, code = 'OLLAMA_ERROR', status = 0, details = null) {
    super(message);
    this.name = 'OllamaError';
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

export class OllamaClient {
  constructor({
    baseUrl = ALLOWED_ORIGIN,
    fetchImpl = globalThis.fetch,
    timeoutMs = 180000
  } = {}) {
    if (typeof fetchImpl !== 'function') throw new Error('A fetch implementation is required.');
    const runtimeFetch = globalThis.fetch;
    this.baseUrl = ensureAllowedBaseUrl(baseUrl);
    this.fetchImpl = fetchImpl === runtimeFetch
      ? runtimeFetch.bind(globalThis)
      : fetchImpl;
    this.timeoutMs = timeoutMs;
  }

  async getVersion(options = {}) {
    return this.#requestJson('/api/version', { method: 'GET', ...options });
  }

  async listModels(options = {}) {
    return this.#requestJson('/api/tags', { method: 'GET', ...options });
  }

  async showModel(model, { signal } = {}) {
    const localModel = assertLocalModelName(model);
    return this.#requestJson('/api/show', {
      method: 'POST',
      body: { model: localModel },
      signal
    });
  }

  async assertModelIsLocal(model, { signal } = {}) {
    const localModel = assertLocalModelName(model);
    const metadata = await this.showModel(localModel, { signal });
    if (hasRemoteModelMetadata(metadata)) {
      throw new OllamaError(
        `Cloud or remote Ollama model "${localModel}" is not allowed. Set OLLAMA_NO_CLOUD=1 and choose a model stored on this device.`,
        'REMOTE_MODEL_NOT_ALLOWED'
      );
    }
    return metadata;
  }

  async embed({
    model,
    input,
    truncate = true,
    keepAlive = '10m',
    signal
  }) {
    const localModel = assertLocalModelName(model);
    const values = Array.isArray(input) ? input : [input];
    if (!values.length || values.some(value => typeof value !== 'string' || !value.trim())) {
      throw new Error('Embedding input must contain non-empty strings.');
    }
    await this.assertModelIsLocal(localModel, { signal });
    const response = await this.#requestJson('/api/embed', {
      method: 'POST',
      body: { model: localModel, input: values, truncate, keep_alive: keepAlive },
      signal
    });
    if (!Array.isArray(response.embeddings) || response.embeddings.length !== values.length) {
      throw new OllamaError('Ollama returned an invalid embedding response.', 'INVALID_EMBEDDING_RESPONSE', 200);
    }
    return response;
  }

  async preload({ model, keepAlive = '10m', signal }) {
    const localModel = assertLocalModelName(model);
    await this.assertModelIsLocal(localModel, { signal });
    return this.#requestJson('/api/chat', {
      method: 'POST',
      body: { model: localModel, messages: [], stream: false, keep_alive: keepAlive },
      signal
    });
  }

  async chatStructured({
    model,
    messages,
    format,
    options = {},
    think = false,
    keepAlive = '10m',
    signal,
    onProgress
  }) {
    const localModel = assertLocalModelName(model);
    if (!Array.isArray(messages) || !messages.length) throw new Error('Chat messages are required.');
    if (!format || typeof format !== 'object') throw new Error('A structured-output schema is required.');
    await this.assertModelIsLocal(localModel, { signal });
    const path = '/api/chat';
    const url = this.#url(path);
    const abortContext = createAbortContext(signal, this.timeoutMs);
    let response;
    try {
      response = await this.fetchImpl(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: localModel,
          messages,
          format,
          options,
          think,
          keep_alive: keepAlive,
          stream: true
        }),
        signal: abortContext.signal
      });
      if (!response.ok) await this.#throwResponseError(response);
      if (!response.body?.getReader) {
        throw new OllamaError('Streaming Ollama response is unavailable.', 'STREAM_UNAVAILABLE', response.status);
      }
      return await this.#readChatStream(response, onProgress);
    } catch (error) {
      if (error instanceof OllamaError) throw error;
      if (abortContext.signal.aborted) {
        throw new OllamaError(
          abortContext.signal.reason?.message || 'Ollama request was cancelled.',
          abortContext.signal.reason?.name === 'TimeoutError' ? 'TIMEOUT' : 'ABORTED'
        );
      }
      throw new OllamaError(error?.message || 'Unable to reach local Ollama.', 'CONNECTION_FAILED');
    } finally {
      abortContext.cleanup();
    }
  }

  async #requestJson(path, { method = 'GET', body, signal } = {}) {
    const abortContext = createAbortContext(signal, this.timeoutMs);
    try {
      const response = await this.fetchImpl(this.#url(path), {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: abortContext.signal
      });
      if (!response.ok) await this.#throwResponseError(response);
      const text = await readBoundedText(response);
      try {
        return text ? JSON.parse(text) : {};
      } catch {
        throw new OllamaError('Ollama returned malformed JSON.', 'MALFORMED_JSON', response.status);
      }
    } catch (error) {
      if (error instanceof OllamaError) throw error;
      if (abortContext.signal.aborted) {
        throw new OllamaError(
          abortContext.signal.reason?.message || 'Ollama request was cancelled.',
          abortContext.signal.reason?.name === 'TimeoutError' ? 'TIMEOUT' : 'ABORTED'
        );
      }
      throw new OllamaError(error?.message || 'Unable to reach local Ollama.', 'CONNECTION_FAILED');
    } finally {
      abortContext.cleanup();
    }
  }

  #url(path) {
    if (!ALLOWED_PATHS.has(path)) throw new Error(`Unsupported Ollama API path: ${path}`);
    return `${this.baseUrl}${path}`;
  }

  async #throwResponseError(response) {
    const text = await readBoundedText(response, 256 * 1024);
    let details = null;
    try {
      details = text ? JSON.parse(text) : null;
    } catch {
      details = text;
    }
    throw new OllamaError(
      details?.error || `Ollama request failed with HTTP ${response.status}.`,
      response.status === 503 ? 'OVERLOADED' : 'HTTP_ERROR',
      response.status,
      details
    );
  }

  async #readChatStream(response, onProgress) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let content = '';
    let thinking = '';
    let finalEvent = null;
    let bytesRead = 0;

    const handleLine = (line) => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        throw new OllamaError('Ollama returned malformed streaming JSON.', 'MALFORMED_STREAM');
      }
      if (event.error) throw new OllamaError(String(event.error), 'STREAM_ERROR');
      const contentDelta = event.message?.content || '';
      const thinkingDelta = event.message?.thinking || '';
      content += contentDelta;
      thinking += thinkingDelta;
      if (event.done) finalEvent = event;
      onProgress?.({ contentDelta, thinkingDelta, content, thinking, done: Boolean(event.done) });
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytesRead += value.byteLength;
      if (bytesRead > 4 * 1024 * 1024) {
        await reader.cancel();
        throw new OllamaError('Ollama stream exceeded the local size limit.', 'RESPONSE_TOO_LARGE');
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach(handleLine);
    }
    buffer += decoder.decode();
    if (buffer.trim()) handleLine(buffer);
    if (!finalEvent) throw new OllamaError('Ollama stream ended before completion.', 'INCOMPLETE_STREAM');
    return {
      content,
      thinking,
      metrics: {
        totalDuration: finalEvent.total_duration || 0,
        loadDuration: finalEvent.load_duration || 0,
        promptEvalCount: finalEvent.prompt_eval_count || 0,
        promptEvalDuration: finalEvent.prompt_eval_duration || 0,
        evalCount: finalEvent.eval_count || 0,
        evalDuration: finalEvent.eval_duration || 0,
        doneReason: finalEvent.done_reason || ''
      }
    };
  }
}

export const OLLAMA_LOOPBACK_ORIGIN = ALLOWED_ORIGIN;
