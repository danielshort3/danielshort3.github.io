'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const os = require('os');
const vm = require('vm');
const Module = require('module');
const { once } = require('events');
const { createLocalServer } = require('../../build/dev');
const proxy = require('../../api/_lib/demo-proxy');

let checks = 0;
const check = (value, message) => { assert(value, message); checks += 1; };
const unavailable = { ok: false, error: 'Demo proxy is unavailable.', code: 'DEMO_PROXY_CONFIGURATION_UNAVAILABLE' };
const fastRetry = { retries: 12, baseDelayMs: 0, maxDelayMs: 0 };

function createClient(respond) {
  const requests = [];
  const env = {
    window: { location: { search: '' } },
    localStorage: { getItem() { return null; }, setItem() {} },
    URLSearchParams,
    setTimeout: (callback) => queueMicrotask(callback),
    fetch: async (url, options) => {
      requests.push({ url, options });
      const response = respond(url, options, requests.length);
      return {
        ok: response.status < 400,
        status: response.status,
        statusText: '',
        headers: { get() { return null; } },
        text: async () => JSON.stringify(response.body)
      };
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/demos/aws-client.js'), 'utf8'), env);
  return { api: env.window.DemoAws, requests };
}

async function testConfigurationRetries() {
  let configured = false;
  const { api, requests } = createClient((url) => ({
    status: configured || url.includes('/other/') ? 200 : 503,
    body: configured || url.includes('/other/') ? { ok: true } : unavailable
  }));
  await assert.rejects(api.retryRequest(() => api.healthJson('/api/demos/shape/'), fastRetry), { code: unavailable.code });
  check(requests.length === 1, 'definitive configuration failures should make only one health request');
  await assert.rejects(api.warmupJson('/api/demos/shape/', {}), { code: unavailable.code });
  await assert.rejects(api.retryRequest(() => api.postWithFallback('/api/demos/shape/', ['predict'], {}), fastRetry), { code: unavailable.code });
  check(requests.length === 1, 'configuration failures should block warmup and inference fallback requests');
  await api.healthJson('/api/demos/other/');
  check(requests.length === 2, 'a failed demo configuration must not disable other demos');
  configured = true;
  await api.healthJson('/api/demos/shape/');
  await api.postWithFallback('/api/demos/shape/', ['predict'], {});
  check(requests.length === 4, 'a fresh health check should allow reconnect and inference after configuration is repaired');
  configured = false;
  await assert.rejects(api.healthJson('/api/demos/shape/'));
  await assert.rejects(api.postWithFallback('/api/demos/shape/', ['predict'], {}));
  configured = true;
  await api.postWithFallback('/api/demos/shape/', ['predict'], {});
  check(requests.length === 6, 'suppression should end after the initialization fallback so a later user retry can recover');

  const transient = createClient((url, options, attempt) => ({
    status: attempt < 3 ? 503 : 200,
    body: attempt < 3 ? { error: 'Cold start in progress.' } : { ok: true }
  }));
  await transient.api.retryRequest(() => transient.api.healthJson('/api/demos/shape/'), fastRetry);
  check(transient.requests.length === 3, 'transient cold-start failures should continue to retry');
  check(transient.api.isRetryableError({ status: 503, code: 'UNKNOWN_PROVIDER_FAILURE' }), 'unknown provider failures should remain retryable');
  check(transient.api.isRetryableError({ message: 'Failed to fetch' }), 'transient network failures should remain retryable');
  check(!transient.api.isRetryableError({ status: 429 }), 'deliberate rate limits should remain nonretryable');

  const warmupFailure = createClient((url) => ({
    status: url.endsWith('/health') ? 200 : 503,
    body: url.endsWith('/health') ? { ok: true } : unavailable
  }));
  await warmupFailure.api.healthJson('/api/demos/shape/');
  await assert.rejects(warmupFailure.api.retryRequest(() => warmupFailure.api.warmupJson('/api/demos/shape/'), fastRetry));
  await assert.rejects(warmupFailure.api.postWithFallback('/api/demos/shape/', ['predict'], {}));
  check(warmupFailure.requests.length === 2, 'configuration failure discovered at warmup should also prevent inference fallback');
}

function captureResponse() {
  return {
    statusCode: 200,
    setHeader() {},
    end(body) { this.body = JSON.parse(body); }
  };
}

async function testProxyConfigurationCodes() {
  const values = {
    DEMO_PROXY_MODE: 'off', DEMO_SHAPE_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:demo:live',
    DEMO_REQUIRE_DDB_RATE_LIMIT: 'false', AWS_AUTH_MODE: 'auto', VERCEL_ENV: 'development',
    DEMO_INVOKE_AWS_ROLE_ARN: '', DEMO_AWS_ACCESS_KEY_ID: '', DEMO_AWS_SECRET_ACCESS_KEY: '',
    AWS_ACCESS_KEY_ID: '', AWS_SECRET_ACCESS_KEY: ''
  };
  const previous = Object.fromEntries(Object.keys(values).map(key => [key, process.env[key]]));
  Object.assign(process.env, values);
  const req = { method: 'GET', url: '/api/demos/shape/health', headers: { host: 'localhost' } };
  try {
    const missing = captureResponse();
    await proxy.handleDemoRequest(req, missing, ['shape', 'health']);
    check(missing.statusCode === 503 && missing.body.code === unavailable.code,
      'known configuration failures should expose a stable nonsecret code');
    check(JSON.stringify(missing.body) === JSON.stringify(unavailable), 'configuration response should not expose settings or credential details');
    process.env.DEMO_PROXY_MODE = 'iam';
    proxy._internal.setClientFactoryForTests(() => { throw Object.assign(new Error('private provider details'), { code: 'UNKNOWN_PROVIDER_FAILURE' }); });
    const unknown = captureResponse();
    await proxy.handleDemoRequest(req, unknown, ['shape', 'health']);
    check(unknown.statusCode === 503 && !unknown.body.code && !JSON.stringify(unknown.body).includes('private'),
      'unknown client failures should stay retryable and retain sanitized diagnostics');
  } finally {
    proxy._internal.setClientFactoryForTests(null);
    for (const [key, value] of Object.entries(previous)) {
      if (typeof value === 'undefined') delete process.env[key];
      else process.env[key] = value;
    }
  }
}

async function testShapeExportRecovery() {
  const html = fs.readFileSync(path.join(__dirname, '../../demos/shape-demo.html'), 'utf8');
  const source = html.slice(html.indexOf('async function classify()'), html.indexOf("classifyBtn.addEventListener('click', classify);"));
  for (const exportImage of [
    (callback) => callback(null),
    () => { throw new Error('Canvas export failed'); },
    (callback) => callback({ arrayBuffer: async () => { throw new Error('Image buffer failed'); } })
  ]) {
    const predictions = [];
    let restored = false;
    let requested = false;
    const env = {
      serverReady: true, hasDrawn: true, classifying: false,
      canvas: { toBlob: exportImage },
      setStep() {}, setPredictionUI: (message) => predictions.push(message),
      renderShapeScores() {}, updateClassifyState() {},
      resultBadge: { classList: { add() {}, remove() { restored = true; } } },
      postToEndpoint() { requested = true; }
    };
    vm.runInNewContext(source, env);
    await env.classify();
    check(!env.classifying && restored && !requested && predictions.at(-1).startsWith('Error:'),
      'canvas export failures should be handled and restore classification controls without requesting inference');
  }
}

function testChatbotSyncedControls() {
  const html = fs.readFileSync(path.join(__dirname, '../../demos/chatbot-demo.html'), 'utf8');
  const functionRange = (start, end) => html.slice(html.indexOf('    function ' + start + '('), html.indexOf('    function ' + end + '('));
  class Element {
    constructor() {
      this.dataset = {};
      this.children = [];
      this.listeners = [];
      this.className = '';
      this.textContent = '';
      this.scrollTop = 0;
    }
    appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
    append(...children) { children.forEach(child => this.appendChild(child)); }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    remove() { this.parentNode.children = this.parentNode.children.filter(child => child !== this); }
    matches(selector) {
      if (selector.startsWith('.')) return selector.slice(1).split('.').every(name => this.className.split(' ').includes(name));
      const key = selector.slice(6, -1).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      return Object.hasOwn(this.dataset, key);
    }
    querySelectorAll(selector) {
      return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]);
    }
    closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector); }
    addEventListener(event, callback) { if (event === 'click') this.listeners.push(callback); }
    click() { this.listeners.forEach(callback => callback()); }
    serialize() {
      return { dataset: this.dataset, className: this.className, textContent: this.textContent, children: this.children.map(child => child.serialize()) };
    }
    get innerHTML() { return JSON.stringify(this.children.map(child => child.serialize())); }
    set innerHTML(value) {
      // Like innerHTML in a browser, copying markup preserves data attributes, but never event listeners.
      const restore = ({ children, ...attributes }) => {
        const node = Object.assign(new Element(), attributes);
        node.append(...children.map(restore));
        return node;
      };
      this.replaceChildren(...JSON.parse(value).map(restore));
    }
  }
  const submissions = [];
  const retries = [];
  const makeContext = (id) => {
    const ctx = { id, messages: new Element(), prompt: { value: '' }, followupHistory: [], controller: null };
    ctx.form = { requestSubmit: () => submissions.push({ id, prompt: ctx.prompt.value, context: ctx.pendingFollowupContext }) };
    return ctx;
  };
  const regular = makeContext('regular');
  const popup = makeContext('popup');
  const followupContext = { previous_question: 'Plan a trip.', source_urls: ['https://example.com/travel'] };
  const env = {
    contexts: [regular, popup], sharedConversation: { draft: '', activeContextId: regular.id }, serverReady: true,
    document: { createElement: () => new Element() },
    autoResize() {}, updateSendButtons() {}, notifyResize() {},
    followupCandidates: () => ['Find a lunch stop'], followupContext: () => followupContext,
    normalizePrompt: text => text.toLowerCase(),
    handleSubmit: (ctx, event, options) => retries.push({ ctx, options })
  };
  const declarations = html.match(/const bound\w+ = new WeakSet\(\);/g) || [];
  vm.runInNewContext([
    ...declarations,
    functionRange('clearFollowups', 'scrollMessages'),
    functionRange('retryFailedResponse', 'emptyState'),
    functionRange('emptyState', 'clearContext')
  ].join('\n'), env);

  env.emptyState(regular);
  check(regular.suggestions.length === 3 && popup.suggestions.length === 3, 'chatbot presets should be created and copied into both chat views');
  for (const ctx of [regular, popup]) {
    const before = submissions.length;
    env.bindSyncedMessageControls(ctx);
    env.bindSyncedMessageControls(ctx);
    ctx.suggestions[0].click();
    check(submissions.length === before + 1 && submissions.at(-1).id === ctx.id &&
      submissions.at(-1).prompt === ctx.suggestions[0].dataset.suggestionPrompt,
    'each original or copied preset should submit its own chat form exactly once after repeated binding');
  }
  const originalPreset = regular.suggestions[0];
  originalPreset.dataset.syncedChatBound = 'yes';
  env.syncMessagesFrom(regular);
  check(popup.suggestions[0] !== originalPreset && popup.suggestions[0].dataset.syncedChatBound === 'yes',
    'message synchronization should reproduce serialized markers on newly created button objects');
  const beforeClonedPreset = submissions.length;
  popup.suggestions[0].click();
  check(submissions.length === beforeClonedPreset + 1 && submissions.at(-1).id === popup.id,
    'a copied preset with a serialized binding marker should still submit from the popup demo');
  env.syncMessagesFrom(popup);
  const beforeReturn = submissions.length;
  regular.suggestions[0].click();
  check(regular.suggestions[0] !== originalPreset && submissions.length === beforeReturn + 1 && submissions.at(-1).id === regular.id,
    'presets should keep working after messages are synchronized back to the regular demo');

  const answer = new Element();
  answer.className = 'message assistant';
  regular.messages.appendChild(answer);
  env.addFollowups(answer, regular, {}, [], 'Plan a trip.');
  for (const ctx of [regular, popup]) {
    const before = submissions.length;
    env.bindSyncedMessageControls(ctx);
    env.bindSyncedMessageControls(ctx);
    ctx.followups[0].click();
    check(submissions.length === before + 1 && submissions.at(-1).id === ctx.id && submissions.at(-1).prompt === 'Find a lunch stop' &&
      JSON.stringify(submissions.at(-1).context) === JSON.stringify(followupContext),
    'original and copied follow-ups should submit exactly once with their decoded context');
  }
  regular.followups[0].dataset.syncedChatBound = 'yes';
  env.syncMessagesFrom(regular);
  const beforeClonedFollowup = submissions.length;
  popup.followups[0].click();
  check(submissions.length === beforeClonedFollowup + 1 && JSON.stringify(submissions.at(-1).context) === JSON.stringify(followupContext),
    'copied follow-ups should remain clickable when serialized markup retains an old binding marker');

  const retry = new Element();
  retry.dataset.retryPrompt = 'Retry this trip';
  retry.dataset.followupContext = JSON.stringify(followupContext);
  answer.appendChild(retry);
  env.syncMessagesFrom(regular);
  for (const ctx of [regular, popup]) {
    env.bindSyncedMessageControls(ctx);
    env.bindSyncedMessageControls(ctx);
    const before = retries.length;
    const button = ctx.messages.querySelectorAll('[data-retry-prompt]')[0];
    button.click();
    check(retries.length === before + 1 && retries.at(-1).ctx === ctx && retries.at(-1).options.message === button.closest('.message.assistant') &&
      retries.at(-1).options.prompt === retry.dataset.retryPrompt && JSON.stringify(retries.at(-1).options.followupContext) === JSON.stringify(followupContext),
    'retry controls should submit once in either view with the correct message and decoded context');
  }
  const beforeUnavailable = submissions.length;
  env.serverReady = false;
  popup.suggestions[0].click();
  popup.followups[0].click();
  check(submissions.length === beforeUnavailable, 'synced presets and follow-ups should still wait until the demo is ready');
}

async function testChatbotCompletedResponseScroll() {
  const html = fs.readFileSync(path.join(__dirname, '../../demos/chatbot-demo.html'), 'utf8');
  const source = html.slice(html.indexOf('    async function handleSubmit('), html.indexOf('    function bindContext('));
  const scrollSource = html.slice(html.indexOf('    function scrollMessages('), html.indexOf('    function addMessage('));
  for (const activeId of ['regular', 'popup']) {
    const focused = [];
    const frames = [];
    const contexts = ['regular', 'popup'].map(id => ({
      id,
      controller: null,
      transcript: [],
      prompt: { value: 'Plan a trip.', focus: options => focused.push({ id, options }) },
      messages: { scrollTop: 400, scrollHeight: 1000, querySelector: () => null }
    }));
    const env = {
      contexts, serverReady: true,
      window: { requestAnimationFrame: callback => frames.push(callback) },
      activeChatContext: () => contexts.find(ctx => ctx.id === activeId),
      clearFollowups() {}, addMessage() {}, autoResize() {}, syncPromptFrom() {},
      addWaitingMessage: () => ({}), syncMessagesFrom() {}, updateSendButtons() {},
      submitPrompt: async () => ({ answer: 'Your itinerary.' }),
      answerText: data => data.answer, sourceItems: () => [],
      renderAssistantAnswer() {},
      addFollowups: () => contexts.forEach(ctx => { ctx.messages.scrollHeight = 1240; }),
      // Iframe sizing can settle after the response has been rendered.
      notifyResize: () => frames.push(() => contexts.forEach(ctx => { ctx.messages.scrollHeight = 1300; }))
    };
    vm.runInNewContext(scrollSource + source, env);
    await env.handleSubmit(contexts[0], { preventDefault() {} });
    check(contexts.every(ctx => ctx.messages.scrollTop === 1240),
      'completed responses should scroll both views after follow-ups are appended');
    frames.forEach(callback => callback());
    check(contexts.every(ctx => ctx.messages.scrollTop === 1300),
      'response scrolling should settle at the bottom after iframe layout updates');
    check(focused.length === 1 && focused[0].id === activeId && focused[0].options?.preventScroll === true,
      'completion should focus the active composer without hiding the final follow-ups');
  }
}

async function testLocalStreamRoute() {
  const previousArn = process.env.CHATBOT_STREAM_FUNCTION_ARN;
  delete process.env.CHATBOT_STREAM_FUNCTION_ARN;
  // This case deliberately exercises missing configuration, independent of a developer's local files.
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'demo-runtime-empty-env-'));
  const server = createLocalServer({ envDir });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const url = 'http://127.0.0.1:' + server.address().port + '/api/chatbot-stream';
  const options = { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: new URL(url).origin, 'Sec-Fetch-Site': 'same-origin' }, body: JSON.stringify({ prompt: 'Plan a trip.' }) };
  const originalLoad = Module._load;
  const originalError = console.error;
  try {
    const unavailableResponse = await fetch(url, options);
    check(unavailableResponse.status === 503 && unavailableResponse.headers.get('content-type').includes('application/json'),
      'unconfigured local streaming should return structured unavailable status instead of a static HTML 404');
    check((await unavailableResponse.json()).error === 'Chatbot stream is not configured.', 'local streaming should preserve the real handler configuration checks');
    const methodResponse = await fetch(url, { headers: options.headers });
    check(methodResponse.status === 405 && methodResponse.headers.get('allow') === 'POST', 'local streaming should delegate method validation to the real handler');
    await methodResponse.text();

    const streamPath = path.join(__dirname, '../../api/chatbot-stream.js');
    let delegatedMethod = '';
    let delegatedBody = '';
    Module._load = function(request, parent, isMain) {
      if (request === streamPath) {
        return async (req, res) => {
          delegatedMethod = req.method;
          for await (const chunk of req) delegatedBody += chunk;
          res.setHeader('Content-Type', 'application/x-ndjson');
          res.write(JSON.stringify({ type: 'token', text: 'Hello' }) + '\n');
          res.end(JSON.stringify({ type: 'done' }) + '\n');
        };
      }
      return originalLoad.call(this, request, parent, isMain);
    };
    const streamed = await fetch(url, options);
    const streamedText = await streamed.text();
    check(streamed.status === 200 && streamed.headers.get('content-type') === 'application/x-ndjson' &&
      streamedText.includes('"type":"token"') && streamedText.includes('"type":"done"') &&
      delegatedMethod === 'POST' && JSON.parse(delegatedBody).prompt === 'Plan a trip.',
      'local route should delegate incoming requests and preserve streamed response chunks');
    let errorsLogged = 0;
    console.error = () => { errorsLogged += 1; };
    Module._load = function(request, parent, isMain) {
      if (request === streamPath) return async () => { throw new Error('private handler failure'); };
      return originalLoad.call(this, request, parent, isMain);
    };
    const rejected = await fetch(url, options);
    const rejection = await rejected.json();
    check(rejected.status === 500 && rejection.error === 'Local chatbot stream handler failed.' && errorsLogged === 1,
      'local route should handle unexpected promise rejections, log them, and return safe JSON');
  } finally {
    Module._load = originalLoad;
    console.error = originalError;
    if (typeof previousArn === 'undefined') delete process.env.CHATBOT_STREAM_FUNCTION_ARN;
    else process.env.CHATBOT_STREAM_FUNCTION_ARN = previousArn;
    fs.rmdirSync(envDir);
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}

(async () => {
  await testConfigurationRetries();
  await testProxyConfigurationCodes();
  await testShapeExportRecovery();
  testChatbotSyncedControls();
  await testChatbotCompletedResponseScroll();
  await testLocalStreamRoute();
  console.log('demo-runtime-resilience: ' + checks + ' checks passed');
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
