#!/usr/bin/env node
'use strict';

/*
  Regenerates project-card preview videos from the current light-theme demo UI.

  The script avoids browser automation dependencies by talking directly to
  Chrome's DevTools Protocol and then encoding captured PNG frames with ffmpeg.
*/

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const outputDir = path.join(root, 'img', 'projects');
const tmpRoot = path.join(root, 'tmp', 'project-preview-recordings');
const DEFAULT_PORT = 3317;
const DEFAULT_CHROME_PORT = 9227;
const FPS = 12;
const DEFAULT_DURATION_MS = 7600;

const PREVIEWS = [
  {
    id: 'smartSentence',
    url: '/sentence-demo',
    width: 952,
    height: 952,
    actions: [
      [900, `setInput('#query', 'Alice follows the White Rabbit into a strange room.');`],
      [1300, `setInput('#top', '8');`],
      [2300, `window.renderResults?.({ top: [
        { score: 0.92, sentence: 'Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it.' },
        { score: 0.84, sentence: 'The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down.' },
        { score: 0.78, sentence: 'Down, down, down. Would the fall never come to an end?' }
      ] });`],
      [4400, `window.scrollTo({ top: Math.min(260, document.documentElement.scrollHeight), behavior: 'auto' });`]
    ]
  },
  {
    id: 'chatbotLora',
    url: '/chatbot-demo',
    width: 1280,
    height: 720,
    actions: [
      [900, `setInput('#regular-prompt', 'Plan a Grand Junction first day with citations.');`],
      [2800, `document.querySelector('[data-view="popup"]')?.click();`],
      [3500, `document.querySelector('#popup-launcher')?.click();`],
      [4400, `setInput('#popup-prompt', 'Compare outdoor options for a weekend trip.');`]
    ]
  },
  {
    id: 'shapeClassifier',
    url: '/shape-demo',
    width: 926,
    height: 926,
    actions: [
      [900, `drawShapeDemo(); renderShapePreviewResult();`],
      [3200, `renderShapePreviewResult();`]
    ]
  },
  {
    id: 'covidAnalysis',
    url: '/covid-outbreak-demo',
    width: 1446,
    height: 1446,
    actions: [
      [1500, `setRangeProgress('#date-slider', 0.72);`],
      [3100, `document.querySelector('.state-shape[data-state="CA"], .state-shape[data-state="TX"], .state-shape')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));`],
      [5200, `setRangeProgress('#date-slider', 0.88);`]
    ]
  },
  {
    id: 'targetEmptyPackage',
    url: '/target-empty-package-demo',
    width: 1460,
    height: 1460,
    actions: [
      [1600, `document.querySelector('[data-breakdown="department"]')?.click();`],
      [3100, `chooseSelectIndex('#filter-location', 1);`],
      [4700, `document.querySelector('[data-metric="count"]')?.click();`]
    ]
  },
  {
    id: 'handwritingRating',
    url: '/handwriting-rating-demo',
    width: 1138,
    height: 1138,
    actions: [
      [1100, `chooseSelectIndex('#sample-select', 2); const sampleStatus = document.querySelector('#sample-status'); if (sampleStatus) sampleStatus.textContent = 'Preview sample loaded.';`],
      [2500, `drawHandwritingDemo();`],
      [4400, `renderHandwritingPreviewResult();`]
    ]
  },
  {
    id: 'digitGenerator',
    url: '/digit-generator-demo',
    width: 1470,
    height: 1470,
    actions: [
      [1200, `setInput('#cluster-select', '7'); renderDigitGridPreview(7, 0);`],
      [1700, `renderDigitGridPreview(7, 1);`],
      [3900, `setRangeProgress('#value-slider', 0.75); renderDigitGridPreview(7, 2);`]
    ]
  },
  {
    id: 'sheetMusicUpscale',
    url: '/portfolio/sheetMusicUpscale',
    width: 1604,
    height: 1230,
    durationMs: 6200,
    scroll: { from: 0, to: 520, startMs: 1200, endMs: 5600 }
  },
  {
    id: 'retailStore',
    url: '/retail-loss-sales-demo',
    width: 1410,
    height: 1410,
    actions: [
      [1500, `document.querySelector('[data-sales-metric="online"]')?.click();`],
      [3100, `document.querySelector('[data-incident-metric="proven"]')?.click();`],
      [4700, `document.querySelector('[data-empty-metric="areas"]')?.click();`]
    ]
  },
  {
    id: 'pizza',
    url: '/pizza-tips-demo',
    width: 1398,
    height: 1398,
    actions: [
      [1200, `setInput('#cost', '42.75'); setInput('#delivery-number', '38'); setInput('#delivery', '38');`],
      [2600, `chooseSelectIndex('#housing', 2); chooseSelectIndex('#order-hour', 18);`],
      [4200, `document.querySelector('#scenario-form')?.requestSubmit?.();`]
    ]
  },
  {
    id: 'babynames',
    url: '/baby-names-demo',
    width: 1238,
    height: 1238,
    actions: [
      [1200, `document.querySelector('[data-sex="M"]')?.click();`],
      [2600, `setInput('#name-search', 'Liam');`],
      [4300, `document.querySelector('[data-tab="ratings"]')?.click();`]
    ]
  },
  {
    id: 'nonogram',
    url: '/nonogram-demo',
    width: 1228,
    height: 1228,
    actions: [
      [1000, `renderNonogramPreview(8);`],
      [3600, `renderNonogramPreview(17); document.querySelector('#solution-btn')?.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));`],
      [5600, `renderNonogramPreview(25); document.querySelector('#solution-btn')?.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));`]
    ]
  },
  {
    id: 'website',
    url: '/analytics',
    width: 1464,
    height: 1464,
    durationMs: 7000,
    scroll: { from: 0, to: 900, startMs: 1300, endMs: 6400 }
  }
];

function log(message) {
  process.stdout.write(`[record-previews] ${message}\n`);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function consentRecordSource() {
  return `{
    version: 1,
    timestamp: Date.now(),
    region: 'US',
    gpc: false,
    categories: {
      necessary: true,
      analytics: false,
      functional: true,
      advertising: false
    },
    tcString: ''
  }`;
}

function consentStorageSource() {
  return `
  try {
    localStorage.setItem('pcz_consent_v1', JSON.stringify(${consentRecordSource()}));
  } catch (_) {}
`;
}

function parseFlag(name, fallback = null) {
  const args = process.argv.slice(2);
  const index = args.indexOf(name);
  if (index >= 0 && args[index + 1]) return args[index + 1];
  return fallback;
}

function hasFlag(name) {
  return process.argv.slice(2).includes(name);
}

function ensureTool(command, label) {
  const result = spawnSync(command, ['-version'], {
    cwd: root,
    stdio: ['ignore', 'ignore', 'ignore']
  });
  if (result.error || result.status !== 0) {
    throw new Error(`${label || command} is required but was not found.`);
  }
}

function findChrome() {
  const explicit = process.env.CHROME_PATH || parseFlag('--chrome');
  const candidates = [
    explicit,
    'chromium-browser',
    'chromium',
    'google-chrome',
    'google-chrome-stable',
    '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
    '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe',
    '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
  ].filter(Boolean);

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['--version'], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'ignore']
    });
    if (!result.error && result.status === 0) return candidate;
  }
  throw new Error('Could not find Chromium, Chrome, or Edge for recording.');
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: options.stdio || 'inherit',
    env: { ...process.env, ...(options.env || {}) }
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} exited with status ${result.status}`);
  }
  return result;
}

function startDevServer() {
  if (hasFlag('--no-server')) {
    return { baseUrl: parseFlag('--base-url', 'http://127.0.0.1:3000'), close: async () => {} };
  }

  const port = Number(parseFlag('--port', String(DEFAULT_PORT)));
  const child = spawn(process.execPath, ['build/dev.js', '--no-watch', '--port', String(port)], {
    cwd: root,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  let ready = false;
  let selectedUrl = null;
  const onData = (buffer) => {
    const text = buffer.toString();
    process.stdout.write(text);
    const match = text.match(/Serving local site on (http:\/\/[^\s]+)/);
    if (match) {
      ready = true;
      selectedUrl = match[1];
    }
  };
  child.stdout.on('data', onData);
  child.stderr.on('data', (buffer) => process.stderr.write(buffer));

  const waitForReady = async () => {
    const start = Date.now();
    while (!ready && Date.now() - start < 90000) {
      if (child.exitCode !== null) break;
      await wait(250);
    }
    if (!ready || !selectedUrl) {
      child.kill('SIGTERM');
      throw new Error('Local dev server did not become ready.');
    }
    return selectedUrl;
  };

  return {
    get baseUrl() {
      return selectedUrl;
    },
    waitForReady,
    close: async () => {
      if (child.exitCode !== null) return;
      child.kill('SIGTERM');
      await wait(500);
    }
  };
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.json();
}

class Cdp {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.nextId = 1;
    this.pending = new Map();
    this.waiters = [];
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Timed out connecting to Chrome DevTools')), 10000);
      this.ws.addEventListener('open', () => {
        clearTimeout(timeout);
        resolve();
      }, { once: true });
      this.ws.addEventListener('error', (event) => {
        clearTimeout(timeout);
        reject(new Error(`DevTools WebSocket error: ${event.message || 'unknown error'}`));
      }, { once: true });
    });
    this.ws.addEventListener('message', (event) => this.handleMessage(event.data));
  }

  handleMessage(raw) {
    const message = JSON.parse(raw);
    if (message.id && this.pending.has(message.id)) {
      const { resolve, reject } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message || JSON.stringify(message.error)));
      else resolve(message.result || {});
      return;
    }

    if (!message.method) return;
    this.waiters = this.waiters.filter((waiter) => {
      if (waiter.method !== message.method) return true;
      if (waiter.sessionId && waiter.sessionId !== message.sessionId) return true;
      clearTimeout(waiter.timeout);
      waiter.resolve(message.params || {});
      return false;
    });
  }

  send(method, params = {}, sessionId = null) {
    const id = this.nextId++;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.ws.send(JSON.stringify(payload));
    return promise;
  }

  waitFor(method, sessionId, timeoutMs = 15000) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.waiters = this.waiters.filter((waiter) => waiter.resolve !== resolve);
        reject(new Error(`Timed out waiting for ${method}`));
      }, timeoutMs);
      this.waiters.push({ method, sessionId, resolve, reject, timeout });
    });
  }

  close() {
    try {
      this.ws.close();
    } catch (_) {}
  }
}

function startChrome(chromePath) {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'preview-chrome-'));
  const port = Number(parseFlag('--chrome-port', String(DEFAULT_CHROME_PORT)));
  const child = spawn(chromePath, [
    '--headless=new',
    '--disable-gpu',
    '--disable-dev-shm-usage',
    '--disable-extensions',
    '--disable-background-networking',
    '--disable-sync',
    '--hide-scrollbars',
    '--mute-audio',
    '--no-first-run',
    '--no-default-browser-check',
    '--no-sandbox',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    'about:blank'
  ], {
    cwd: root,
    stdio: ['ignore', 'ignore', 'pipe']
  });
  child.stderr.on('data', (buffer) => {
    const text = buffer.toString();
    if (!/DevTools listening/.test(text)) return;
    process.stderr.write(text);
  });

  return {
    port,
    close: async () => {
      if (child.exitCode === null) child.kill('SIGTERM');
      await wait(500);
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }
  };
}

async function waitForChrome(port) {
  const versionUrl = `http://127.0.0.1:${port}/json/version`;
  const start = Date.now();
  while (Date.now() - start < 30000) {
    try {
      return await fetchJson(versionUrl);
    } catch (_) {
      await wait(250);
    }
  }
  throw new Error('Chrome DevTools endpoint did not become ready.');
}

async function waitForHttpOk(baseUrl, urlPath) {
  const url = new URL(urlPath, baseUrl).href;
  const start = Date.now();
  let lastStatus = 'not requested';

  while (Date.now() - start < 30000) {
    try {
      const response = await fetch(url, { redirect: 'follow' });
      lastStatus = `HTTP ${response.status}`;
      if (response.ok) {
        const html = await response.text();
        const isNotFound = /<title>\s*(?:404|page not found)\b/i.test(html)
          || /\b404\s*[–-]\s*page not found\b/i.test(html)
          || /^\s*not found\s*$/i.test(html);
        if (!isNotFound) return;
        lastStatus = 'ready check received a not-found page';
      }
    } catch (error) {
      lastStatus = error && error.message ? error.message : 'request failed';
    }
    await wait(300);
  }

  throw new Error(`Preview URL did not become ready: ${url} (${lastStatus})`);
}

function recordingBootstrap() {
  return `
(() => {
  ${consentStorageSource()}
  const mark = () => {
    if (!document.documentElement) return;
    document.documentElement.setAttribute('data-embedded', 'true');
    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.style.colorScheme = 'light';
  };
  mark();
  document.addEventListener('readystatechange', mark, true);
  document.addEventListener('DOMContentLoaded', mark, true);
})();
`;
}

function runtimeHelpers() {
  return `
(() => {
  ${consentStorageSource()}
  document.documentElement.setAttribute('data-embedded', 'true');
  document.documentElement.setAttribute('data-theme', 'light');
  document.documentElement.style.colorScheme = 'light';
  document.body?.classList.add('is-preview-recording');
  document.body?.removeAttribute('data-consent-banner');
  document.body?.classList.remove('consent-blocked');
  document.querySelectorAll('#pcz-banner, #pcz-modal').forEach((el) => el.remove());

  const style = document.createElement('style');
  style.textContent = [
    'html,body{scroll-behavior:auto!important;background:#F9F9FA!important}',
    '.is-preview-recording *{scroll-behavior:auto!important}',
    '.is-preview-recording #pcz-banner,.is-preview-recording #pcz-modal{display:none!important}',
    '.is-preview-recording #pad{background:#fff!important}',
    '.is-preview-recording .cell--filled{background:#091F3B!important;border-color:#005FED!important}',
    '.is-preview-recording .clue{background:#fff!important;color:#091F3B!important}',
    '.is-preview-recording #refresh-seed-btn,.is-preview-recording #action-btn,.is-preview-recording #classify,.is-preview-recording #rate{background:#005FED!important;color:#fff!important;border:1px solid #005FED!important;border-radius:10px!important;padding:12px 22px!important;font-weight:700!important;box-shadow:0 12px 24px rgba(0,95,237,.18)!important}'
  ].join('');
  document.head.appendChild(style);

  window.polishPreviewState = () => {
    document.body?.removeAttribute('data-consent-banner');
    document.body?.classList.remove('consent-blocked');
    document.querySelectorAll('#pcz-banner, #pcz-modal').forEach((el) => el.remove());
    document.querySelectorAll('.health-pill, .status-pill').forEach((el) => {
      el.dataset.state = 'ok';
      if (/warming|loading|checking|error|offline|off/i.test(el.textContent || '')) {
        el.textContent = 'Ready';
      }
    });
    document.querySelectorAll('button, input, select, textarea').forEach((el) => {
      el.disabled = false;
      el.removeAttribute('disabled');
    });
    ['#health-text', '#status-meta', '#connection-meta', '#sample-status'].forEach((selector) => {
      const el = document.querySelector(selector);
      if (el && /warming|loading|checking|connecting|error|unavailable/i.test(el.textContent || '')) {
        el.textContent = 'Ready for preview.';
      }
    });
    const status = document.querySelector('#status');
    if (status && /warming|loading|connecting|error|unavailable/i.test(status.textContent || '')) {
      status.textContent = 'Ready for preview.';
    }
  };

  window.setInput = (selector, value) => {
    const el = document.querySelector(selector);
    if (!el) return false;
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  };

  window.chooseSelectIndex = (selector, index) => {
    const el = document.querySelector(selector);
    if (!el || !el.options || !el.options.length) return false;
    const next = Math.max(0, Math.min(el.options.length - 1, Number(index) || 0));
    el.selectedIndex = next;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  };

  window.setRangeProgress = (selector, progress) => {
    const el = document.querySelector(selector);
    if (!el) return false;
    const min = Number(el.min || 0);
    const max = Number(el.max || 100);
    const pct = Math.max(0, Math.min(1, Number(progress) || 0));
    el.value = String(Math.round(min + (max - min) * pct));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  };

  window.drawShapeDemo = () => {
    const canvas = document.querySelector('#pad');
    if (!canvas) return false;
    canvas.style.background = '#fff';
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = Math.max(12, Math.round(canvas.width * 0.055));
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#091F3B';
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.5, canvas.height * 0.13);
    ctx.lineTo(canvas.width * 0.86, canvas.height * 0.84);
    ctx.lineTo(canvas.width * 0.14, canvas.height * 0.84);
    ctx.closePath();
    ctx.stroke();
    canvas.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  };

  window.drawHandwritingDemo = () => {
    const canvas = document.querySelector('#pad');
    if (!canvas) return false;
    canvas.style.background = '#fff';
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = Math.max(11, Math.round(canvas.width * 0.05));
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#091F3B';
    ctx.beginPath();
    ctx.moveTo(canvas.width * 0.25, canvas.height * 0.26);
    ctx.bezierCurveTo(canvas.width * 0.43, canvas.height * 0.08, canvas.width * 0.72, canvas.height * 0.19, canvas.width * 0.71, canvas.height * 0.43);
    ctx.bezierCurveTo(canvas.width * 0.69, canvas.height * 0.72, canvas.width * 0.29, canvas.height * 0.78, canvas.width * 0.28, canvas.height * 0.52);
    ctx.bezierCurveTo(canvas.width * 0.27, canvas.height * 0.31, canvas.width * 0.58, canvas.height * 0.37, canvas.width * 0.72, canvas.height * 0.48);
    ctx.stroke();
    canvas.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  };

  window.renderShapePreviewResult = () => {
    const shape = document.querySelector('#result-shape');
    const confidenceBar = document.querySelector('#confidence-bar');
    const confidenceText = document.querySelector('#confidence-text');
    const list = document.querySelector('#shape-score-list');
    if (shape) shape.textContent = 'Triangle classified';
    if (confidenceBar) confidenceBar.style.width = '72%';
    if (confidenceText) confidenceText.textContent = 'Confidence: 72%';
    if (list) {
      list.innerHTML = [
        '<div class="shape-row is-top"><span class="shape-name">Triangle</span><span class="shape-track"><span class="shape-fill" style="width:72%"></span></span><span class="shape-pct">72%</span></div>',
        '<div class="shape-row"><span class="shape-name">Hexagon</span><span class="shape-track"><span class="shape-fill" style="width:11%"></span></span><span class="shape-pct">11%</span></div>',
        '<div class="shape-row"><span class="shape-name">Square</span><span class="shape-track"><span class="shape-fill" style="width:8%"></span></span><span class="shape-pct">8%</span></div>'
      ].join('');
    }
  };

  window.renderHandwritingPreviewResult = () => {
    const fields = {
      '#result-digit': 'Prediction: 2',
      '#result-actual': 'Actual: 2',
      '#confidence-text': 'Confidence: 86%',
      '#legibility-value': '82',
      '#sample-status': 'Preview sample loaded.'
    };
    Object.keys(fields).forEach((selector) => {
      const el = document.querySelector(selector);
      if (el) el.textContent = fields[selector];
    });
    const confidenceBar = document.querySelector('#confidence-bar');
    const legibilityBar = document.querySelector('#legibility-bar');
    const list = document.querySelector('#confidence-list');
    if (confidenceBar) confidenceBar.style.width = '86%';
    if (legibilityBar) legibilityBar.style.width = '82%';
    if (list) {
      list.innerHTML = [
        '<div class="confidence-row is-actual"><span class="confidence-digit">2</span><span class="confidence-track"><span class="confidence" style="width:86%"></span></span><span class="confidence-pct">86%</span></div>',
        '<div class="confidence-row"><span class="confidence-digit">3</span><span class="confidence-track"><span class="confidence" style="width:7%"></span></span><span class="confidence-pct">7%</span></div>',
        '<div class="confidence-row"><span class="confidence-digit">8</span><span class="confidence-track"><span class="confidence" style="width:4%"></span></span><span class="confidence-pct">4%</span></div>'
      ].join('');
    }
  };

  window.renderDigitGridPreview = (digit, variant) => {
    const grid = document.querySelector('#grid');
    if (!grid) return false;
    const status = document.querySelector('#status');
    const pill = document.querySelector('#grid-number-pill');
    const selected = String(Number.isFinite(Number(digit)) ? digit : 7);
    if (pill) pill.textContent = selected;
    if (status) status.textContent = 'Generated 16 preview digits.';
    grid.style.setProperty('--grid-cols', '4');
    grid.innerHTML = '';
    const strokes = [
      'M33 21 C47 14 66 15 81 24 L48 105',
      'M29 24 L86 24 L52 108',
      'M38 18 C57 14 76 18 86 30 C72 52 60 78 51 108',
      'M27 25 C48 20 70 20 89 27 L56 111'
    ];
    for (let index = 0; index < 16; index += 1) {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'digit-cell';
      cell.setAttribute('aria-label', 'Generated preview digit ' + (index + 1));
      const path = strokes[(index + (Number(variant) || 0)) % strokes.length];
      const opacity = 0.72 + ((index % 4) * 0.06);
      const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><rect width="120" height="120" rx="18" fill="#fff"/><path d="' + path + '" fill="none" stroke="#091F3B" stroke-width="' + (12 + (index % 3)) + '" stroke-linecap="round" stroke-linejoin="round" opacity="' + opacity.toFixed(2) + '"/></svg>';
      const img = document.createElement('img');
      img.alt = 'Generated digit ' + selected;
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
      cell.appendChild(img);
      grid.appendChild(cell);
    }
    return true;
  };

  window.renderNonogramPreview = (count) => {
    const grid = document.querySelector('#grid');
    const top = document.querySelector('#clues-top');
    const left = document.querySelector('#clues-left');
    const status = document.querySelector('#status');
    const accuracy = document.querySelector('#stat-accuracy');
    const log = document.querySelector('#log');
    const action = document.querySelector('#action-btn');
    const solution = document.querySelector('#solution-btn');
    if (!grid || !top || !left) return false;
    const puzzle = [
      [1, 0, 1, 1, 0],
      [1, 1, 1, 0, 0],
      [0, 1, 0, 1, 1],
      [1, 1, 0, 1, 0],
      [0, 0, 1, 1, 1]
    ];
    const topClues = ['2', '3', '2 1', '1 3', '1 1'];
    const leftClues = ['1 2', '3', '1 2', '2 1', '3'];
    const revealed = Math.max(0, Math.min(25, Number(count) || 0));
    top.innerHTML = topClues.map((text) => '<div class="clue">' + text + '</div>').join('');
    left.innerHTML = leftClues.map((text) => '<div class="clue">' + text + '</div>').join('');
    grid.innerHTML = '';
    for (let row = 0; row < 5; row += 1) {
      for (let col = 0; col < 5; col += 1) {
        const index = row * 5 + col;
        const cell = document.createElement('div');
        cell.className = 'cell';
        if (index < revealed) cell.classList.add(puzzle[row][col] ? 'cell--filled' : 'cell--empty');
        grid.appendChild(cell);
      }
    }
    if (status) status.textContent = revealed >= 25 ? 'Solved preview puzzle.' : 'AI solving preview puzzle...';
    if (accuracy) accuracy.textContent = revealed >= 25 ? '100%' : '96%';
    if (action) {
      action.textContent = revealed >= 25 ? 'New Puzzle' : 'Solve With AI';
      action.disabled = false;
    }
    if (solution) solution.disabled = false;
    if (log) {
      log.innerHTML = [
        '<div class="log-entry">Loaded 5x5 puzzle with row and column clues.</div>',
        '<div class="log-entry">Agent selected high-confidence cells first.</div>',
        '<div class="log-entry">Current accuracy: ' + (revealed >= 25 ? '100%' : '96%') + '.</div>'
      ].join('');
    }
    return true;
  };

  window.polishPreviewState();
})();
`;
}

async function openPage(cdp, baseUrl, preview) {
  const target = await cdp.send('Target.createTarget', { url: 'about:blank' });
  const attached = await cdp.send('Target.attachToTarget', {
    targetId: target.targetId,
    flatten: true
  });
  const sessionId = attached.sessionId;
  await cdp.send('Page.enable', {}, sessionId);
  await cdp.send('Runtime.enable', {}, sessionId);
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: preview.width,
    height: preview.height,
    deviceScaleFactor: 1,
    mobile: false,
    screenWidth: preview.width,
    screenHeight: preview.height
  }, sessionId);
  await cdp.send('Emulation.setEmulatedMedia', {
    media: 'screen',
    features: [{ name: 'prefers-color-scheme', value: 'light' }]
  }, sessionId);
  await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
    source: recordingBootstrap()
  }, sessionId);

  const loadPromise = cdp.waitFor('Page.loadEventFired', sessionId, 30000).catch(() => null);
  await cdp.send('Page.navigate', { url: new URL(preview.url, baseUrl).href }, sessionId);
  await loadPromise;
  await wait(preview.waitMs || 2200);
  await cdp.send('Runtime.evaluate', {
    expression: runtimeHelpers(),
    awaitPromise: true,
    returnByValue: true
  }, sessionId);
  await wait(500);
  return { targetId: target.targetId, sessionId };
}

async function evalPage(cdp, sessionId, source) {
  await cdp.send('Runtime.evaluate', {
    expression: `(() => { try { ${source} } catch (error) { console.warn(error && error.message ? error.message : error); } })()`,
    awaitPromise: true,
    returnByValue: true
  }, sessionId);
}

async function captureFrame(cdp, sessionId, filePath) {
  const shot = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    fromSurface: true
  }, sessionId);
  fs.writeFileSync(filePath, Buffer.from(shot.data, 'base64'));
}

function encodeVideo(preview, framesDir) {
  const framePattern = path.join(framesDir, 'frame_%04d.png');
  const mp4 = path.join(outputDir, `${preview.id}.mp4`);
  const webm = path.join(outputDir, `${preview.id}.webm`);
  const evenScale = 'pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p';

  run('ffmpeg', [
    '-hide_banner',
    '-loglevel', 'error',
    '-y',
    '-framerate', String(FPS),
    '-i', framePattern,
    '-an',
    '-vf', evenScale,
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '27',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    mp4
  ]);

  run('ffmpeg', [
    '-hide_banner',
    '-loglevel', 'error',
    '-y',
    '-framerate', String(FPS),
    '-i', framePattern,
    '-an',
    '-vf', evenScale,
    '-c:v', 'libvpx-vp9',
    '-deadline', 'realtime',
    '-cpu-used', '8',
    '-row-mt', '1',
    '-b:v', '0',
    '-crf', '40',
    webm
  ]);
}

async function recordPreview(cdp, baseUrl, preview) {
  log(`Recording ${preview.id} (${preview.width}x${preview.height})`);
  const framesDir = path.join(tmpRoot, preview.id);
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  await waitForHttpOk(baseUrl, preview.url);
  const { targetId, sessionId } = await openPage(cdp, baseUrl, preview);
  const durationMs = preview.durationMs || DEFAULT_DURATION_MS;
  const frameCount = Math.max(1, Math.round((durationMs / 1000) * FPS));
  const actions = [...(preview.actions || [])].sort((a, b) => a[0] - b[0]);
  let actionIndex = 0;

  for (let frame = 0; frame < frameCount; frame += 1) {
    const elapsed = Math.round((frame / FPS) * 1000);
    while (actionIndex < actions.length && actions[actionIndex][0] <= elapsed) {
      await evalPage(cdp, sessionId, actions[actionIndex][1]);
      actionIndex += 1;
      await wait(160);
    }

    if (preview.scroll) {
      const { from, to, startMs, endMs } = preview.scroll;
      const pct = Math.max(0, Math.min(1, (elapsed - startMs) / Math.max(1, endMs - startMs)));
      const eased = pct * pct * (3 - (2 * pct));
      const y = Math.round(from + (to - from) * eased);
      await evalPage(cdp, sessionId, `window.scrollTo(0, ${y});`);
    }

    await evalPage(cdp, sessionId, `window.polishPreviewState?.();`);
    const frameName = `frame_${String(frame + 1).padStart(4, '0')}.png`;
    await captureFrame(cdp, sessionId, path.join(framesDir, frameName));
    await wait(Math.max(8, Math.round(1000 / FPS) - 18));
  }

  await cdp.send('Target.closeTarget', { targetId });
  encodeVideo(preview, framesDir);
  log(`Wrote img/projects/${preview.id}.mp4 and .webm`);
}

async function main() {
  ensureTool('ffmpeg', 'ffmpeg');
  fs.mkdirSync(outputDir, { recursive: true });
  fs.rmSync(tmpRoot, { recursive: true, force: true });
  fs.mkdirSync(tmpRoot, { recursive: true });

  const server = startDevServer();
  const baseUrl = hasFlag('--no-server') ? server.baseUrl : await server.waitForReady();
  await wait(1000);
  const chromePath = findChrome();
  const chrome = startChrome(chromePath);
  let cdp;

  try {
    const version = await waitForChrome(chrome.port);
    cdp = new Cdp(version.webSocketDebuggerUrl);
    await cdp.connect();

    const only = parseFlag('--only');
    const selected = only
      ? PREVIEWS.filter((preview) => only.split(',').map((id) => id.trim()).includes(preview.id))
      : PREVIEWS;
    if (!selected.length) throw new Error(`No previews matched --only ${only}`);

    for (const preview of selected) {
      await recordPreview(cdp, baseUrl, preview);
    }
  } finally {
    if (cdp) cdp.close();
    await chrome.close();
    await server.close();
  }

  log('Done.');
}

main().catch((error) => {
  console.error(`[record-previews] ${error.stack || error.message || error}`);
  process.exit(1);
});
