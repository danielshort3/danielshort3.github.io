'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/tools/qr-code-generator.js'), 'utf8');
const sourceBlock = (start, end) => {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert(from >= 0 && to > from, `Missing QR controller: ${start}`);
  return source.slice(from, to);
};

function createHarness() {
  const elements = new Map();
  const element = (key) => {
    if (!elements.has(key)) elements.set(key, { hidden: false, textContent: '', value: '', disabled: false, readOnly: false });
    return elements.get(key);
  };
  const calls = { downloads: 0, renders: [], configs: [], tab: '' };
  let getLink = async (slug) => ({ link: { slug, label: 'Campaign', destination: 'https://example.com/campaign' } });
  const context = vm.createContext({
    URL, URLSearchParams, TextEncoder, Object, JSON, Date,
    window: {
      location: { href: 'https://www.danielshort.me/tools/qr-code-generator?link=campaign&download=png' },
      history: { replaceState: (_state, _title, url) => {
        context.window.location.href = new URL(url, context.window.location.href).href;
      } },
      ShortLinksClient: {
        get: (slug) => getLink(slug),
        publicUrl: (slug, options) => `https://dshort.me/${slug}${options?.qr ? '?__qr=1' : ''}`,
      },
    },
    $: element,
    dataInput: element('data'),
    payloadModeSelect: element('payload-mode'),
    linkModeSelect: element('link-mode'),
    linkedContext: element('linked-context'),
    managedCreate: element('managed-create'),
    designSave: element('design-save'),
    saveDesignButton: element('save-design'),
    createManagedLinkButton: element('create-link'),
    linkHelp: element('help'),
    destinationPickerOpen: element('destinations'),
    exampleBtn: element('example'),
    clearBtn: element('clear'),
    emptyOverlay: element('empty'),
    linkStatus: element('status'),
    linkAccess: element('access'),
    designStatus: element('design-status'),
    downloadPngBtn: { click: () => { calls.downloads += 1; } },
    state: { data: '', centerMode: 'none' },
    setInlineStatus: (target, message) => { target.textContent = message; },
    setPayloadMode: (mode) => { context.payloadModeSelect.value = mode; },
    getPayloadMode: () => context.payloadModeSelect.value || 'url',
    applyConfigSnapshot: (config) => {
      calls.configs.push(config);
      Object.assign(context.state, config);
      context.dataInput.value = config.data;
    },
    buildConfigSnapshot: () => ({ ...context.state }),
    readStateFromControls: () => { context.state.data = context.api.getPayload(); },
    render: () => { calls.renders.push(context.state.data); },
    loadLogoFromUrl: async () => {},
    schedulePersistLastConfig: () => {},
    activateTab: (name) => { calls.tab = name; },
    buildPreviewDataUrl: () => ({ dataUrl: 'data:image/png;base64,cHJldmlldw==' }),
  });
  vm.runInContext(`
    let linkedLink = null;
    let pendingLinkedSlug = '';
    let managedMode = false;
    let linkedLoading = false;
    let linkedRequestId = 0;
    let linkedDesignSaved = '';
    ${sourceBlock('  const getPayloadFromControls =', '  const applyDataValue =')}
    ${sourceBlock('  const QR_DESIGN_KEYS =', '  const loadStoredPresets =')}
    globalThis.api = {
      getPayload: getPayloadFromControls,
      selectDesign: selectQrDesign,
      load: loadLinkedLink,
      inspect: () => ({ linkedLink, pendingLinkedSlug, managedMode, linkedLoading }),
    };
  `, context, { filename: 'qr-code-generator.js' });
  return { context, api: context.api, calls, elements, setGet: (handler) => { getLink = handler; } };
}

(async () => {
  const accessListeners = new Map();
  let accessRefreshes = 0;
  const accessContext = vm.createContext({
    window: { addEventListener: (name, callback) => accessListeners.set(`window:${name}`, callback) },
    document: { addEventListener: (name, callback) => accessListeners.set(`document:${name}`, callback) },
    updateShortlinksPickerVisibility: () => { accessRefreshes += 1; },
  });
  vm.runInContext(`let shortlinksManifest = {}; ${sourceBlock('  const handleLinkAccessChanged =', "  dataInput.addEventListener('input'")}`, accessContext);
  assert.equal(typeof accessListeners.get('document:tools:auth-changed'), 'function');
  accessListeners.get('document:tools:auth-changed')();
  assert.equal(accessRefreshes, 1, 'Non-bubbling account events must refresh the QR link picker.');

  const qr = createHarness();
  qr.context.dataInput.value = 'https://direct.example/';
  assert.equal(qr.api.getPayload(), 'https://direct.example/');
  let resolveLoad;
  qr.setGet(() => new Promise((resolve) => { resolveLoad = resolve; }));
  const loading = qr.api.load('campaign', { download: true });
  assert.equal(qr.api.getPayload(), '', 'Pending managed links must not render the previous direct destination.');
  assert.equal(qr.api.inspect().linkedLoading, true);
  resolveLoad({ link: {
    slug: 'campaign', destination: 'https://example.com/summer',
    qrDesign: { fg: '#123456', centerMode: 'none', data: 'https://wrong.example/', wifiPassword: 'private' },
  } });
  await loading;
  assert.equal(qr.api.getPayload(), 'https://dshort.me/campaign?__qr=1');
  assert.equal(qr.context.dataInput.value, 'https://example.com/summer');
  assert.equal(qr.context.dataInput.readOnly, true);
  qr.context.dataInput.value = 'https://tampered.example/';
  assert.equal(qr.api.getPayload(), 'https://dshort.me/campaign?__qr=1', 'The saved identity governs managed QR encoding.');
  assert.equal(qr.calls.configs[0].data, 'https://example.com/summer');
  assert.equal(qr.calls.configs[0].wifiPassword, undefined);
  assert.equal(qr.calls.configs[0].fg, '#123456', 'Saved designs keep their stored foreground color.');
  assert.equal(qr.calls.downloads, 1);
  assert.equal(qr.calls.tab, 'export');
  assert.equal(new URL(qr.context.window.location.href).searchParams.has('download'), false);

  qr.setGet(async () => { throw Object.assign(new Error('Sign in required'), { status: 401 }); });
  await qr.api.load('missing');
  assert.equal(qr.api.inspect().managedMode, true);
  assert.equal(qr.api.inspect().pendingLinkedSlug, 'missing');
  assert.equal(qr.api.getPayload(), '');
  assert.equal(qr.context.linkAccess.hidden, false);
  assert.equal(qr.context.linkStatus.textContent, 'Sign in required');
  assert.equal(qr.elements.get('[data-qrtool-retry-link]').hidden, false);
  assert.equal(qr.calls.downloads, 1, 'Failed loads must not download any previous QR.');

  const missingLogo = createHarness();
  missingLogo.setGet(async (slug) => ({ link: {
    slug, destination: 'https://example.com/',
    qrDesign: { centerMode: 'image', logoDataUrl: 'https://example.com/missing.png' },
  } }));
  missingLogo.context.loadLogoFromUrl = async () => { throw new Error('Saved logo unavailable'); };
  await missingLogo.api.load('saved-design', { download: true });
  assert.equal(missingLogo.api.inspect().pendingLinkedSlug, 'saved-design');
  assert.equal(missingLogo.api.getPayload(), '');
  assert.equal(missingLogo.calls.downloads, 0, 'A missing logo must not silently download an altered saved design.');

  const race = createHarness();
  const pending = {};
  race.setGet((slug) => new Promise((resolve) => { pending[slug] = resolve; }));
  const first = race.api.load('old');
  const second = race.api.load('new');
  pending.new({ link: { slug: 'new', destination: 'https://new.example/' } });
  await second;
  pending.old({ link: { slug: 'old', destination: 'https://old.example/' } });
  await first;
  assert.equal(race.api.getPayload(), 'https://dshort.me/new?__qr=1', 'Late requests cannot replace a more recently selected link.');
  assert.equal(race.calls.configs[0].fg, '#102c46');
  assert.equal(race.calls.configs[0].bg, '#FFFFFF');
  assert.equal(race.calls.configs[0].dotStyle, 'square');
  assert.equal(race.calls.configs[0].cornerStyle, 'square');
  assert.equal(race.calls.configs[0].marginModules, 4);
  assert.equal(race.calls.configs[0].centerMode, 'none');

  const design = qr.api.selectDesign({
    fg: '#123456', logoDataUrl: 'data:image/png;base64,bG9nbw==', captionText: 'Visit us',
    data: 'https://example.com/', linkedSlug: 'identity', wifiPassword: 'secret', vcardEmail: 'private@example.com',
  });
  assert.deepEqual(JSON.parse(JSON.stringify(design)), {
    fg: '#123456', captionText: 'Visit us', logoDataUrl: 'data:image/png;base64,bG9nbw==',
  });
  console.log('QR link integration passed: stable tracked payloads, saved designs, download recovery, failed loads, and request races.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
