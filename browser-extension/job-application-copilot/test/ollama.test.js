import assert from 'node:assert/strict';
import test from 'node:test';
import {
  assertLocalModelName,
  hasRemoteModelMetadata,
  isCloudModelName,
  OllamaClient,
  OllamaError
} from '../src/ollama/client.js';

test('Ollama client is pinned to the exact loopback origin', async () => {
  assert.throws(() => new OllamaClient({ baseUrl: 'http://localhost:11434' }), /exactly http:\/\/127\.0\.0\.1:11434/u);
  assert.throws(() => new OllamaClient({ baseUrl: 'http://127.0.0.1:11434/api' }), /exactly/u);
  let requestedUrl;
  const client = new OllamaClient({
    fetchImpl: async url => {
      requestedUrl = url;
      return new Response('{"version":"0.32.0"}');
    }
  });
  assert.deepEqual(await client.getVersion(), { version: '0.32.0' });
  assert.equal(requestedUrl, 'http://127.0.0.1:11434/api/version');
});

test('browser fetch keeps its global receiver while custom fetch injection is preserved', async () => {
  const originalFetch = globalThis.fetch;
  const requestedUrls = [];

  globalThis.fetch = async function (url) {
    if (this !== globalThis) throw new TypeError('Illegal invocation');
    requestedUrls.push(url);
    return new Response('{"version":"0.32.0"}');
  };

  try {
    const browserClient = new OllamaClient();
    assert.deepEqual(await browserClient.getVersion(), { version: '0.32.0' });
    assert.deepEqual(requestedUrls, ['http://127.0.0.1:11434/api/version']);
  } finally {
    globalThis.fetch = originalFetch;
  }

  const customFetch = async () => new Response('{"version":"custom"}');
  const customClient = new OllamaClient({ fetchImpl: customFetch });

  assert.equal(customClient.fetchImpl, customFetch);
  assert.deepEqual(await customClient.getVersion(), { version: 'custom' });
});

test('Ollama client preserves HTTP 403 for exact-origin diagnostics', async () => {
  const client = new OllamaClient({
    fetchImpl: async () => new Response('', { status: 403 })
  });
  await assert.rejects(
    client.getVersion(),
    error => error instanceof OllamaError
      && error.code === 'HTTP_ERROR'
      && error.status === 403
  );
});

test('Ollama client parses bounded NDJSON structured chat streams', async () => {
  const events = [
    { message: { content: '{"ok":' }, done: false },
    { message: { content: 'true}' }, done: true, eval_count: 3, eval_duration: 100 }
  ];
  const client = new OllamaClient({
    fetchImpl: async url => url.endsWith('/api/show')
      ? new Response('{"model_info":{"general.architecture":"qwen3"}}')
      : new Response(`${events.map(value => JSON.stringify(value)).join('\n')}\n`)
  });
  const progress = [];
  const response = await client.chatStructured({
    model: 'qwen3.5:27b',
    messages: [{ role: 'user', content: 'Return JSON.' }],
    format: { type: 'object' },
    onProgress: update => progress.push(update)
  });
  assert.equal(response.content, '{"ok":true}');
  assert.equal(response.metrics.evalCount, 3);
  assert.equal(progress.at(-1).done, true);
});

test('Ollama client rejects malformed and incomplete streams', async () => {
  const malformed = new OllamaClient({
    fetchImpl: async url => url.endsWith('/api/show') ? new Response('{}') : new Response('not-json\n')
  });
  await assert.rejects(malformed.chatStructured({
    model: 'qwen3:8b',
    messages: [{ role: 'user', content: 'x' }],
    format: { type: 'object' }
  }), error => error instanceof OllamaError && error.code === 'MALFORMED_STREAM');

  const incomplete = new OllamaClient({
    fetchImpl: async url => url.endsWith('/api/show') ? new Response('{}') : new Response('{"message":{"content":"x"}}\n')
  });
  await assert.rejects(incomplete.chatStructured({
    model: 'qwen3:8b',
    messages: [{ role: 'user', content: 'x' }],
    format: { type: 'object' }
  }), error => error instanceof OllamaError && error.code === 'INCOMPLETE_STREAM');
});

test('Ollama model guards reject explicit cloud names and remote show metadata', async () => {
  assert.equal(isCloudModelName('qwen3:cloud'), true);
  assert.equal(isCloudModelName('team/model-cloud'), true);
  assert.equal(isCloudModelName('cloudy-local:latest'), false);
  assert.throws(() => assertLocalModelName('qwen3:cloud'), error => error.code === 'REMOTE_MODEL_NOT_ALLOWED');
  assert.equal(hasRemoteModelMetadata({ details: { remote_host: 'https://ollama.com' } }), true);
  assert.equal(hasRemoteModelMetadata({ remote_model: 'qwen3:cloud' }), true);
  assert.equal(hasRemoteModelMetadata({ remote_host: '', remote_model: '', model_info: { format: 'gguf' } }), false);

  const requestedPaths = [];
  const remoteClient = new OllamaClient({
    fetchImpl: async (url) => {
      requestedPaths.push(new URL(url).pathname);
      return new Response('{"remote_host":"https://ollama.com","remote_model":"qwen3:cloud"}');
    }
  });
  await assert.rejects(remoteClient.embed({
    model: 'qwen3:8b',
    input: ['local evidence']
  }), error => error instanceof OllamaError && error.code === 'REMOTE_MODEL_NOT_ALLOWED');
  assert.deepEqual(requestedPaths, ['/api/show']);
});

test('Ollama validates local model metadata before embed and chat endpoints', async () => {
  const requestedPaths = [];
  const events = [{ message: { content: '{"ok":true}' }, done: true }];
  const client = new OllamaClient({
    fetchImpl: async (url) => {
      const pathname = new URL(url).pathname;
      requestedPaths.push(pathname);
      if (pathname === '/api/show') return new Response('{"model_info":{"general.file_type":2}}');
      if (pathname === '/api/embed') return new Response('{"embeddings":[[0.1,0.2]]}');
      return new Response(`${events.map(value => JSON.stringify(value)).join('\n')}\n`);
    }
  });
  await client.embed({ model: 'nomic-embed-text', input: ['evidence'] });
  await client.chatStructured({
    model: 'qwen3:8b',
    messages: [{ role: 'user', content: 'Return JSON.' }],
    format: { type: 'object' }
  });
  assert.deepEqual(requestedPaths, ['/api/show', '/api/embed', '/api/show', '/api/chat']);
});
