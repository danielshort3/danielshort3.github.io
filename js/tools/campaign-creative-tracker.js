(() => {
  'use strict';

  const TOOL_ID = 'campaign-creative-tracker';
  const STORAGE_KEY = 'campaignCreativeTracker:v1';
  const ASSET_ROOT = 'img/tools/campaign-creative-tracker/';
  const root = document.querySelector('[data-ctc-root]');
  const core = window.CampaignCreativeTrackerCore;

  if (!root || !core) {
    if (root) {
      root.innerHTML = '<div class="ctc-empty"><h2>Workspace unavailable</h2><p>The campaign tracker could not load its local data engine.</p></div>';
    }
    return;
  }

  const UTM_LABELS = {
    utm_id: 'utm_id',
    utm_source: 'utm_source',
    utm_medium: 'utm_medium',
    utm_campaign: 'utm_campaign',
    utm_content: 'utm_content',
    utm_term: 'utm_term',
  };

  const FORMAT_LABELS = {
    static: 'Static',
    animated: 'Animated',
    interactive: 'Interactive',
    video: 'Video',
    ctv: 'CTV',
  };

  const TEST_MODE_LABELS = {
    ab: 'A/B',
    single: 'Single link',
    na: 'Not applicable',
  };

  const DEFAULT_DICTIONARIES = {
    utm_id: [
      { value: 'b', label: 'b — Basis' },
      { value: 'c', label: 'c — Cadent' },
      { value: 'e', label: 'e — Epsilon' },
      { value: 'v', label: 'v — Viant' },
    ],
    utm_source: [
      { value: 'basis', label: 'Basis' },
      { value: 'cadent', label: 'Cadent' },
      { value: 'epsilon', label: 'Epsilon' },
      { value: 'viant', label: 'Viant' },
    ],
    utm_medium: [
      { value: 'display', label: 'Display' },
      { value: 'rich_media', label: 'Rich media' },
      { value: 'olv', label: 'Online video' },
      { value: 'ctv', label: 'Connected TV' },
      { value: 'paid_social', label: 'Paid social' },
    ],
    utm_campaign: [
      { value: '2026_q2_summer', label: '2026 Q2 Summer' },
      { value: '2026_always_on', label: '2026 Always-on' },
      { value: '2026_lodging', label: '2026 Lodging' },
    ],
    utm_content: [
      { value: 'summer_things_to_do', label: 'Summer Things to Do' },
      { value: 'year_round_fun', label: 'Year-round Fun' },
      { value: 'natures_trifecta', label: 'Nature’s Trifecta' },
      { value: 'affordable_lodging', label: 'Affordable Lodging' },
    ],
    utm_term: [
      { value: 'prospecting', label: 'Prospecting' },
      { value: 'retargeting', label: 'Retargeting' },
      { value: 'travel_intenders', label: 'Travel intenders' },
      { value: 'not_set', label: 'Not set' },
    ],
  };

  const deepClone = (value) => JSON.parse(JSON.stringify(value));

  const rendition = ({
    id,
    name,
    width,
    height,
    format,
    asset,
    duration = '',
    previewUrl = '',
    clickTagValid = false,
    qrAttached = false,
    overrideEnabled = false,
    override = {},
  }) => ({
    id,
    name,
    width,
    height,
    format,
    duration,
    assetSrc: asset ? `${ASSET_ROOT}${asset}` : '',
    previewUrl,
    clickTagValid,
    qrAttached,
    overrideEnabled,
    override: {
      testMode: '',
      destinationA: '',
      destinationB: '',
      utms: {},
      ...override,
      utms: { ...(override.utms || {}) },
    },
  });

  const makeFamily = ({
    id,
    name,
    partner,
    destination,
    testMode,
    content,
    campaign = '2026_q2_summer',
    renditions,
  }) => ({
    id,
    name,
    partner,
    destinationA: destination,
    destinationB: `${destination}${destination.includes('?') ? '&' : '?'}experience=alternate`,
    testMode,
    utms: {
      utm_id: partner.slice(0, 1),
      utm_source: partner,
      utm_medium: 'display',
      utm_campaign: campaign,
      utm_content: content,
      utm_term: 'prospecting',
    },
    renditions,
  });

  const buildSeedState = () => {
    const destination = 'https://www.visitgrandjunction.com/things-to-do/';
    const familyOne = makeFamily({
      id: 'GJ-SUMMER-01',
      name: 'Summer Things to Do',
      partner: 'basis',
      destination,
      testMode: 'ab',
      content: 'summer_things_to_do',
      renditions: [
        rendition({ id: 'SUM-300X250', name: '300×250 Static', width: 300, height: 250, format: 'static', asset: 'basis-square-display.webp' }),
        rendition({ id: 'SUM-728X90', name: '728×90 Static', width: 728, height: 90, format: 'static', asset: 'epsilon-banner.webp' }),
        rendition({ id: 'SUM-160X600', name: '160×600 Animated', width: 160, height: 600, format: 'animated', asset: 'cadent-vertical-interactive.webp' }),
        rendition({
          id: 'SUM-970X250',
          name: '970×250 HTML5 Interactive',
          width: 970,
          height: 250,
          format: 'interactive',
          asset: 'epsilon-rich-media-wide.webp',
          previewUrl: 'https://creative.example.com/summer-970x250/',
          clickTagValid: true,
        }),
        rendition({ id: 'SUM-320X50', name: '320×50 Mobile', width: 320, height: 50, format: 'static', asset: 'epsilon-banner.webp' }),
        rendition({
          id: 'SUM-OLV15',
          name: 'OLV 15s',
          width: 1920,
          height: 1080,
          format: 'video',
          duration: '15s',
          asset: 'cadent-video-still.webp',
          overrideEnabled: true,
          override: { testMode: 'single', utms: { utm_medium: 'olv' } },
        }),
        rendition({
          id: 'SUM-CTV30',
          name: 'CTV 30s',
          width: 1920,
          height: 1080,
          format: 'ctv',
          duration: '30s',
          asset: 'epsilon-video-still.webp',
          qrAttached: true,
          overrideEnabled: true,
          override: { testMode: 'na', utms: { utm_medium: 'ctv' } },
        }),
        rendition({
          id: 'SUM-300X600',
          name: '300×600 Interactive',
          width: 300,
          height: 600,
          format: 'interactive',
          asset: 'cadent-vertical-interactive.webp',
          previewUrl: 'https://creative.example.com/summer-300x600/',
          clickTagValid: true,
        }),
        rendition({ id: 'SUM-970X90', name: '970×90 Static', width: 970, height: 90, format: 'static', asset: 'epsilon-banner.webp' }),
        rendition({
          id: 'SUM-OLV06',
          name: 'OLV 6s',
          width: 1920,
          height: 1080,
          format: 'video',
          duration: '6s',
          asset: 'basis-display-qr.webp',
          overrideEnabled: true,
          override: { testMode: 'single', utms: { utm_medium: 'olv' } },
        }),
      ],
    });

    const familyTwo = makeFamily({
      id: 'GJ-YRF-02',
      name: 'Year-round Fun',
      partner: 'epsilon',
      destination: 'https://www.visitgrandjunction.com/plan-your-visit/',
      testMode: 'single',
      content: 'year_round_fun',
      campaign: '2026_always_on',
      renditions: [
        rendition({ id: 'YRF-300X250', name: '300×250 Static', width: 300, height: 250, format: 'static', asset: 'basis-display-qr.webp', overrideEnabled: true, override: { testMode: 'ab' } }),
        rendition({ id: 'YRF-728X90', name: '728×90 Static', width: 728, height: 90, format: 'static', asset: 'epsilon-banner.webp' }),
        rendition({ id: 'YRF-970X250', name: '970×250 Interactive', width: 970, height: 250, format: 'interactive', asset: 'epsilon-rich-media-wide.webp', previewUrl: 'https://creative.example.com/year-round/', clickTagValid: true }),
        rendition({ id: 'YRF-160X600', name: '160×600 Animated', width: 160, height: 600, format: 'animated', asset: 'cadent-vertical-interactive.webp' }),
        rendition({ id: 'YRF-OLV15', name: 'OLV 15s', width: 1920, height: 1080, format: 'video', duration: '15s', asset: 'epsilon-video-still.webp', overrideEnabled: true, override: { testMode: 'single', utms: { utm_medium: 'olv' } } }),
        rendition({ id: 'YRF-CTV30', name: 'CTV 30s', width: 1920, height: 1080, format: 'ctv', duration: '30s', asset: 'cadent-video-still.webp', qrAttached: true, overrideEnabled: true, override: { testMode: 'na', utms: { utm_medium: 'ctv' } } }),
      ],
    });

    const familyThree = makeFamily({
      id: 'GJ-NATURE-03',
      name: 'Nature’s Trifecta',
      partner: 'cadent',
      destination: 'https://www.visitgrandjunction.com/outdoors/',
      testMode: 'single',
      content: 'natures_trifecta',
      campaign: '2026_always_on',
      renditions: [
        rendition({ id: 'NAT-300X250', name: '300×250 Static', width: 300, height: 250, format: 'static', asset: 'basis-square-display.webp' }),
        rendition({ id: 'NAT-728X90', name: '728×90 Static', width: 728, height: 90, format: 'static', asset: 'epsilon-banner.webp' }),
        rendition({ id: 'NAT-970X250', name: '970×250 Interactive', width: 970, height: 250, format: 'interactive', asset: 'epsilon-rich-media-wide.webp', previewUrl: 'https://creative.example.com/trifecta/', clickTagValid: true }),
        rendition({ id: 'NAT-OLV15', name: 'OLV 15s', width: 1920, height: 1080, format: 'video', duration: '15s', asset: 'cadent-video-still.webp', overrideEnabled: true, override: { testMode: 'single', utms: { utm_medium: 'olv' } } }),
        rendition({ id: 'NAT-300X600', name: '300×600 Interactive', width: 300, height: 600, format: 'interactive', asset: 'cadent-vertical-interactive.webp', previewUrl: 'https://creative.example.com/trifecta-vertical/', clickTagValid: true }),
      ],
    });

    const familyFour = makeFamily({
      id: 'GJ-LODGE-04',
      name: 'Affordable Lodging',
      partner: 'viant',
      destination: 'https://www.visitgrandjunction.com/places-to-stay/',
      testMode: 'single',
      content: 'affordable_lodging',
      campaign: '2026_lodging',
      renditions: [
        rendition({ id: 'LODGE-300X250', name: '300×250 Static', width: 300, height: 250, format: 'static', asset: 'basis-display-qr.webp' }),
        rendition({ id: 'LODGE-728X90', name: '728×90 Static', width: 728, height: 90, format: 'static', asset: 'epsilon-banner.webp' }),
        rendition({ id: 'LODGE-160X600', name: '160×600 Static', width: 160, height: 600, format: 'static', asset: 'cadent-vertical-interactive.webp' }),
        rendition({ id: 'LODGE-970X250', name: '970×250 Interactive', width: 970, height: 250, format: 'interactive', asset: 'epsilon-rich-media-wide.webp', previewUrl: 'https://creative.example.com/lodging/', clickTagValid: true }),
        rendition({ id: 'LODGE-OLV15', name: 'OLV 15s', width: 1920, height: 1080, format: 'video', duration: '15s', asset: 'epsilon-video-still.webp', overrideEnabled: true, override: { testMode: 'single', utms: { utm_medium: 'olv' } } }),
        rendition({ id: 'LODGE-CTV30', name: 'CTV 30s', width: 1920, height: 1080, format: 'ctv', duration: '30s', asset: 'cadent-video-still.webp', qrAttached: true, overrideEnabled: true, override: { testMode: 'na', utms: { utm_medium: 'ctv' } } }),
      ],
    });

    return {
      version: 1,
      campaign: {
        id: 'GJ-SUMMER-2026',
        name: 'Grand Junction Summer 2026',
        version: 4,
        updatedAt: new Date().toISOString(),
        families: [familyOne, familyTwo, familyThree, familyFour],
      },
      dictionaries: deepClone(DEFAULT_DICTIONARIES),
      ui: {
        view: 'library',
        selectedFamilyId: familyOne.id,
        selectedRenditionId: familyOne.renditions[6].id,
        expandedFamilyIds: [familyOne.id],
        exportFormats: ['partner-csv', 'package'],
      },
    };
  };

  const runtimeAssetUrls = new Map();
  let toastTimer = 0;

  const escapeHtml = (value) => String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const safeAssetSrc = (value) => {
    const src = String(value || '').trim();
    if (/^(?:blob:|img\/tools\/campaign-creative-tracker\/)/.test(src)) return src;
    return '';
  };
  const safeHttpHref = (value) => {
    const href = String(value || '').trim();
    if (!href) return '';
    try {
      const parsed = new URL(href);
      return ['http:', 'https:'].includes(parsed.protocol) ? parsed.toString() : '';
    } catch {
      return '';
    }
  };

  const icon = (name) => {
    const paths = {
      library: '<rect x="4" y="4" width="6" height="6" rx="1"></rect><rect x="14" y="4" width="6" height="6" rx="1"></rect><rect x="4" y="14" width="6" height="6" rx="1"></rect><rect x="14" y="14" width="6" height="6" rx="1"></rect>',
      dashboard: '<path d="M4 19V9M10 19V5M16 19v-7M22 19V3"></path>',
      export: '<path d="M12 15V3M7 8l5-5 5 5"></path><path d="M5 13v6h14v-6"></path>',
      upload: '<path d="M12 16V4M7 9l5-5 5 5"></path><path d="M5 14v5h14v-5"></path>',
      plus: '<path d="M12 5v14M5 12h14"></path>',
      settings: '<circle cx="12" cy="12" r="3"></circle><path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1a8 8 0 0 0-1.7-1L14.5 3h-5l-.4 3.1a8 8 0 0 0-1.7 1l-2.4-1-2 3.4L5.1 11a7 7 0 0 0 0 2L3 14.5l2 3.4 2.4-1a8 8 0 0 0 1.7 1l.4 3.1h5l.4-3.1a8 8 0 0 0 1.7-1l2.4 1 2-3.4-2.1-1.5a7 7 0 0 0 .1-1z"></path>',
      check: '<path d="M5 12l4 4L19 6"></path>',
      warning: '<path d="M12 3l10 18H2L12 3z"></path><path d="M12 9v5M12 18h.01"></path>',
      chevron: '<path d="M9 6l6 6-6 6"></path>',
      copy: '<rect x="8" y="8" width="11" height="11" rx="2"></rect><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"></path>',
      play: '<path d="M8 5l11 7-11 7V5z"></path>',
      link: '<path d="M10 13a5 5 0 0 0 7.1.1l2-2a5 5 0 0 0-7.1-7.1l-1.1 1.1"></path><path d="M14 11a5 5 0 0 0-7.1-.1l-2 2A5 5 0 0 0 12 20l1.1-1.1"></path>',
      folder: '<path d="M3 6h7l2 2h9v11H3z"></path>',
      file: '<path d="M6 3h8l4 4v14H6z"></path><path d="M14 3v5h5"></path>',
      eye: '<path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12z"></path><circle cx="12" cy="12" r="2.5"></circle>',
      reset: '<path d="M4 4v6h6"></path><path d="M5.5 16a8 8 0 1 0 .5-9l-2 3"></path>',
    };
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.file}</svg>`;
  };

  const sanitizeStateForStorage = (value) => {
    const copy = deepClone(value);
    (copy.campaign?.families || []).forEach((family) => {
      (family.renditions || []).forEach((item) => {
        if (String(item.assetSrc || '').startsWith('blob:')) item.assetSrc = '';
      });
    });
    return copy;
  };

  const normalizeLoadedStateLegacy = (candidate) => {
    const fallback = buildSeedState();
    if (!candidate || candidate.version !== 1 || !Array.isArray(candidate.campaign?.families)) return fallback;
    return {
      ...fallback,
      ...candidate,
      campaign: { ...fallback.campaign, ...candidate.campaign },
      dictionaries: { ...deepClone(DEFAULT_DICTIONARIES), ...(candidate.dictionaries || {}) },
      ui: { ...fallback.ui, ...(candidate.ui || {}) },
    };
  };
  const normalizeLoadedState = (candidate) => {
    const fallback = buildSeedState();
    if (!candidate || candidate.version !== 1 || !Array.isArray(candidate.campaign?.families)) return fallback;

    const asSafeText = (value) => String(value ?? '').trim();
    const asPositiveNumber = (value) => {
      const number = Number(value);
      return Number.isFinite(number) && number > 0 ? number : 0;
    };
    const cleanUtms = (value) => Object.fromEntries(core.UTM_KEYS.map((key) => [
      key,
      asSafeText(value && typeof value === 'object' ? value[key] : ''),
    ]));
    const dictionaries = Object.fromEntries(core.UTM_KEYS.map((key) => {
      const source = Array.isArray(candidate.dictionaries?.[key])
        ? candidate.dictionaries[key]
        : DEFAULT_DICTIONARIES[key];
      const seen = new Set();
      const options = source
        .filter((option) => option && typeof option === 'object')
        .map((option) => ({
          value: asSafeText(option.value),
          label: asSafeText(option.label || option.value),
        }))
        .filter((option) => option.value && !seen.has(option.value) && seen.add(option.value));
      return [key, options.length ? options : deepClone(DEFAULT_DICTIONARIES[key])];
    }));
    const families = candidate.campaign.families
      .filter((family) => family && typeof family === 'object')
      .map((family, familyIndex) => ({
        id: asSafeText(family.id) || `FAMILY-${familyIndex + 1}`,
        name: asSafeText(family.name) || `Creative family ${familyIndex + 1}`,
        partner: asSafeText(family.partner),
        destinationA: asSafeText(family.destinationA),
        destinationB: asSafeText(family.destinationB),
        testMode: core.TEST_MODES.includes(asSafeText(family.testMode)) ? asSafeText(family.testMode) : 'single',
        utms: cleanUtms(family.utms),
        renditions: (Array.isArray(family.renditions) ? family.renditions : [])
          .filter((item) => item && typeof item === 'object')
          .map((item, itemIndex) => ({
            id: asSafeText(item.id) || `${asSafeText(family.id) || 'FAMILY'}-RENDITION-${itemIndex + 1}`,
            name: asSafeText(item.name),
            width: asPositiveNumber(item.width),
            height: asPositiveNumber(item.height),
            format: asSafeText(item.format).toLocaleLowerCase('en-US'),
            duration: asSafeText(item.duration),
            assetSrc: safeAssetSrc(item.assetSrc),
            previewUrl: asSafeText(item.previewUrl),
            clickTagValid: item.clickTagValid === true,
            qrAttached: item.qrAttached === true,
            fileName: asSafeText(item.fileName),
            fileType: asSafeText(item.fileType),
            fileSize: Math.max(0, Number(item.fileSize) || 0),
            overrideEnabled: item.overrideEnabled === true,
            override: {
              testMode: core.TEST_MODES.includes(asSafeText(item.override?.testMode))
                ? asSafeText(item.override.testMode)
                : '',
              destinationA: asSafeText(item.override?.destinationA),
              destinationB: asSafeText(item.override?.destinationB),
              utms: cleanUtms(item.override?.utms),
            },
          })),
      }));

    return {
      version: 1,
      campaign: {
        id: asSafeText(candidate.campaign?.id) || fallback.campaign.id,
        name: asSafeText(candidate.campaign?.name) || fallback.campaign.name,
        version: asPositiveNumber(candidate.campaign?.version) || fallback.campaign.version,
        updatedAt: asSafeText(candidate.campaign?.updatedAt) || fallback.campaign.updatedAt,
        families,
      },
      dictionaries,
      ui: {
        view: ['library', 'dashboard', 'export'].includes(candidate.ui?.view) ? candidate.ui.view : 'library',
        selectedFamilyId: asSafeText(candidate.ui?.selectedFamilyId),
        selectedRenditionId: asSafeText(candidate.ui?.selectedRenditionId),
        expandedFamilyIds: Array.isArray(candidate.ui?.expandedFamilyIds)
          ? candidate.ui.expandedFamilyIds.map(asSafeText).filter(Boolean)
          : [],
        exportFormats: Array.isArray(candidate.ui?.exportFormats)
          ? candidate.ui.exportFormats.filter((value) => ['partner-csv', 'package'].includes(value))
          : ['partner-csv', 'package'],
      },
    };
  };

  const loadState = () => {
    try {
      return normalizeLoadedState(JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'));
    } catch {
      return buildSeedState();
    }
  };

  let state = loadState();

  const persistState = () => {
    state.campaign.updatedAt = new Date().toISOString();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeStateForStorage(state)));
    } catch {}
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const showToast = (message, tone = 'info') => {
    const toast = document.querySelector('[data-ctc-toast]');
    if (!toast) return;
    window.clearTimeout(toastTimer);
    toast.textContent = message;
    toast.dataset.tone = tone;
    toast.hidden = false;
    toastTimer = window.setTimeout(() => {
      toast.hidden = true;
    }, 3200);
  };

  const selectedFamily = () => {
    const families = state.campaign.families;
    return families.find((family) => family.id === state.ui.selectedFamilyId) || families[0] || null;
  };

  const selectedRendition = () => {
    const family = selectedFamily();
    if (!family) return null;
    return family.renditions.find((item) => item.id === state.ui.selectedRenditionId) || family.renditions[0] || null;
  };

  const normalizeSelection = () => {
    const family = selectedFamily();
    if (!family) {
      state.ui.selectedFamilyId = '';
      state.ui.selectedRenditionId = '';
      return;
    }
    state.ui.selectedFamilyId = family.id;
    const item = selectedRendition();
    state.ui.selectedRenditionId = item?.id || '';
  };

  const commit = (dirty = true) => {
    normalizeSelection();
    persistState();
    render();
    if (dirty) markSessionDirty();
  };

  const optionLabel = (key, value) => {
    const option = (state.dictionaries[key] || []).find((item) => item.value === value);
    return option?.label || value || 'Not set';
  };

  const renderOptions = (key, value, includeInherited = false) => {
    const options = state.dictionaries[key] || [];
    const inherited = includeInherited ? '<option value="">Inherit family value</option>' : '';
    return inherited + options.map((option) => (
      `<option value="${escapeHtml(option.value)}" ${option.value === value ? 'selected' : ''}>${escapeHtml(option.label)}</option>`
    )).join('');
  };

  const renderUtmField = (family, key) => `
    <label class="ctc-rule-field">
      <span class="ctc-rule-label">${escapeHtml(UTM_LABELS[key])}</span>
      <select data-ctc-family-utm="${escapeHtml(key)}" aria-label="${escapeHtml(UTM_LABELS[key])}">
        ${renderOptions(key, family.utms[key])}
      </select>
      <small>${state.dictionaries[key]?.length || 0} allowed values</small>
    </label>
  `;

  const renderTestMode = (current, scope) => `
    <div class="ctc-segmented" role="group" aria-label="Test mode">
      ${Object.entries(TEST_MODE_LABELS).map(([mode, label]) => `
        <button type="button" class="${mode === current ? 'is-active' : ''}" data-ctc-action="set-test-mode" data-ctc-test-scope="${scope}" data-ctc-test-mode="${mode}" aria-pressed="${mode === current}">${escapeHtml(label)}</button>
      `).join('')}
    </div>
  `;

  const displayAssetSrc = (item) => runtimeAssetUrls.get(item.id) || safeAssetSrc(item.assetSrc);

  const renderAssetLegacy = (family, item, compact = false) => {
    const src = displayAssetSrc(item);
    const play = ['video', 'ctv'].includes(item.format)
      ? `<span class="ctc-play">${icon('play')}</span>`
      : '';
    if (!src) {
      return `<div class="ctc-asset ctc-asset-empty ${compact ? 'is-compact' : ''}">${icon('file')}<span>${escapeHtml(item.fileName || item.format || 'Asset')}</span></div>`;
    }
    return `<div class="ctc-asset ${compact ? 'is-compact' : ''}"><img src="${escapeHtml(src)}" alt="${escapeHtml(family.name)} — ${escapeHtml(item.name)} preview" loading="lazy" decoding="async">${play}</div>`;
  };

  const renderAsset = (family, item = {}, compact = false) => {
    const src = displayAssetSrc(item);
    const isRuntimeAsset = runtimeAssetUrls.has(item.id);
    const isUploadedImage = isRuntimeAsset && String(item.fileType || '').startsWith('image/');
    const isUploadedVideo = isRuntimeAsset && String(item.fileType || '').startsWith('video/');
    const play = ['video', 'ctv'].includes(item.format)
      ? `<span class="ctc-play">${icon('play')}</span>`
      : '';
    if (!src || (isRuntimeAsset && !isUploadedImage && !isUploadedVideo)) {
      return `<div class="ctc-asset ctc-asset-empty ${compact ? 'is-compact' : ''}">${icon('file')}<span>${escapeHtml(item.fileName || item.format || 'Asset')}</span></div>`;
    }
    const label = `${escapeHtml(family.name)} - ${escapeHtml(item.name)} preview`;
    const media = isUploadedVideo
      ? `<video src="${escapeHtml(src)}" aria-label="${label}" muted playsinline preload="metadata"></video>`
      : `<img src="${escapeHtml(src)}" alt="${label}" loading="lazy" decoding="async">`;
    return `<div class="ctc-asset ${compact ? 'is-compact' : ''}">${media}${play}</div>`;
  };
  const renderFamilyRail = () => `
    <aside class="ctc-family-pane" aria-label="Creative families">
      <div class="ctc-pane-heading">
        <div>
          <h3>Families</h3>
          <p>${state.campaign.families.length} campaign concepts</p>
        </div>
        <button type="button" class="ctc-icon-button" data-ctc-action="new-family" aria-label="Create creative family">${icon('plus')}</button>
      </div>
      <div class="ctc-family-list">
        ${state.campaign.families.map((family) => {
          const cover = family.renditions[0] || {};
          const selected = family.id === state.ui.selectedFamilyId;
          return `
            <button type="button" class="ctc-family-card ${selected ? 'is-selected' : ''}" data-ctc-action="select-family" data-family-id="${escapeHtml(family.id)}" aria-pressed="${selected}">
              ${renderAsset(family, cover, true)}
              <span class="ctc-family-card-copy">
                <strong>${escapeHtml(family.name)}</strong>
                <span>${family.renditions.length} renditions</span>
                <small>${escapeHtml(family.partner)}</small>
              </span>
              ${icon('chevron')}
            </button>
          `;
        }).join('')}
      </div>
      <button type="button" class="ctc-import-button" data-ctc-action="import">${icon('upload')}<span>Import creative</span></button>
    </aside>
  `;

  const renderRenditionCard = (family, item) => {
    const selected = item.id === state.ui.selectedRenditionId;
    const validation = core.validateRendition(family, item, state.dictionaries);
    const ratio = Number(item.width) && Number(item.height) ? Number(item.width) / Number(item.height) : 1;
    const wide = ratio > 2.25;
    const tall = ratio < 0.6;
    return `
      <button type="button" class="ctc-rendition-card ${selected ? 'is-selected' : ''} ${wide ? 'is-wide' : ''} ${tall ? 'is-tall' : ''}" data-ctc-action="select-rendition" data-rendition-id="${escapeHtml(item.id)}" aria-pressed="${selected}">
        ${renderAsset(family, item)}
        <span class="ctc-rendition-copy">
          <span>
            <strong>${escapeHtml(item.name)}</strong>
            <small>${item.width && item.height ? `${item.width}×${item.height}` : 'Dimensions pending'}${item.duration ? ` · ${escapeHtml(item.duration)}` : ''}</small>
          </span>
          <span class="ctc-rendition-tags">
            <span class="ctc-format ctc-format-${escapeHtml(item.format)}">${escapeHtml(FORMAT_LABELS[item.format] || item.format)}</span>
            ${item.overrideEnabled ? '<span class="ctc-status ctc-status-override">Override</span>' : '<span class="ctc-status">Inherited</span>'}
            <span class="ctc-validation ${validation.valid ? 'is-valid' : 'is-error'}" title="${escapeHtml(validation.issues.map((issue) => issue.message).join(' '))}">${validation.valid ? icon('check') : icon('warning')}</span>
          </span>
        </span>
      </button>
    `;
  };

  const renderRenditionPane = (family) => `
    <section class="ctc-rendition-pane" aria-labelledby="ctc-renditions-title">
      <div class="ctc-pane-heading ctc-pane-heading-main">
        <div>
          <p class="ctc-overline">Selected family</p>
          <h3 id="ctc-renditions-title">${escapeHtml(family.name)}</h3>
          <p>${family.renditions.length} renditions across static, animated, interactive, OLV, and CTV formats</p>
        </div>
        <div class="ctc-family-id">
          <span>Family ID</span>
          <strong>${escapeHtml(family.id)}</strong>
        </div>
      </div>
      <div class="ctc-rendition-grid">
        ${family.renditions.map((item) => renderRenditionCard(family, item)).join('')}
      </div>
    </section>
  `;

  const renderLinkPreview = (family, item) => {
    if (!item) return '';
    let variants = [];
    let error = '';
    try {
      variants = core.buildLinkVariants(family, item);
    } catch (caught) {
      error = caught?.message || String(caught);
    }
    if (error) return `<div class="ctc-link-preview is-error">${icon('warning')}<span>${escapeHtml(error)}</span></div>`;
    if (!variants.length) {
      return `<div class="ctc-link-preview">${icon('link')}<span>No clickable URL for this rendition. Delivery uses the attached QR asset.</span></div>`;
    }
    return `
      <div class="ctc-link-list">
        ${variants.map((variant) => `
          <div class="ctc-link-preview">
            <span class="ctc-link-variant">${escapeHtml(variant.label)}</span>
            <code title="${escapeHtml(variant.url)}">${escapeHtml(variant.url)}</code>
            <button type="button" class="ctc-icon-button" data-ctc-action="copy-url" data-url="${escapeHtml(variant.url)}" aria-label="Copy ${escapeHtml(variant.label)} URL">${icon('copy')}</button>
          </div>
        `).join('')}
      </div>
    `;
  };

  const renderInspector = (family, item) => {
    const effective = item ? core.getEffectiveConfig(family, item) : null;
    const validation = item ? core.validateRendition(family, item, state.dictionaries) : { valid: true, issues: [] };
    const overridden = family.renditions.filter((renditionItem) => renditionItem.overrideEnabled).length;
    return `
      <aside class="ctc-inspector" aria-label="Family link and UTM rules">
        <div class="ctc-inspector-header">
          <div>
            <h3>Family link &amp; UTM rules</h3>
            <p>Applies to ${escapeHtml(family.name)}</p>
          </div>
          <button type="button" class="ctc-text-button" data-ctc-action="apply-to-all">Apply to all ${family.renditions.length}</button>
        </div>
        <div class="ctc-inheritance-summary">${icon('link')}<span>${family.renditions.length - overridden} inherited · ${overridden} overridden</span></div>
        <div class="ctc-rule-grid">
          ${core.UTM_KEYS.map((key) => renderUtmField(family, key)).join('')}
        </div>
        <div class="ctc-rule-section">
          <div class="ctc-section-heading">
            <div>
              <h4>Link setup</h4>
              <p>Family destination and default test scope</p>
            </div>
          </div>
          <div class="ctc-destination-grid">
            <label class="ctc-field">
              <span>Destination A</span>
              <input type="url" value="${escapeHtml(family.destinationA)}" data-ctc-family-field="destinationA">
            </label>
            <label class="ctc-field">
              <span>Destination B</span>
              <input type="url" value="${escapeHtml(family.destinationB)}" data-ctc-family-field="destinationB" ${family.testMode !== 'ab' ? 'disabled' : ''}>
            </label>
          </div>
          ${renderTestMode(family.testMode, 'family')}
        </div>
        ${item ? `
          <div class="ctc-rule-section ctc-override-section ${item.overrideEnabled ? 'is-enabled' : ''}">
            <div class="ctc-section-heading">
              <div>
                <h4>${escapeHtml(item.name)} override</h4>
                <p>Change only this rendition when partner requirements differ.</p>
              </div>
              <label class="ctc-switch">
                <input type="checkbox" data-ctc-rendition-field="overrideEnabled" ${item.overrideEnabled ? 'checked' : ''}>
                <span aria-hidden="true"></span>
                <em>${item.overrideEnabled ? 'On' : 'Off'}</em>
              </label>
            </div>
            ${item.overrideEnabled ? `
              ${renderTestMode(effective.testMode, 'rendition')}
              <div class="ctc-override-grid">
                <label class="ctc-field">
                  <span>utm_medium override</span>
                  <select data-ctc-rendition-utm="utm_medium">${renderOptions('utm_medium', item.override?.utms?.utm_medium || '', true)}</select>
                </label>
                <label class="ctc-field">
                  <span>utm_content override</span>
                  <select data-ctc-rendition-utm="utm_content">${renderOptions('utm_content', item.override?.utms?.utm_content || '', true)}</select>
                </label>
              </div>
              <div class="ctc-destination-grid ctc-override-destinations">
                <label class="ctc-field">
                  <span>Destination A override</span>
                  <input type="url" value="${escapeHtml(item.override?.destinationA || '')}" placeholder="${escapeHtml(family.destinationA)}" data-ctc-rendition-override-field="destinationA">
                </label>
                <label class="ctc-field">
                  <span>Destination B override</span>
                  <input type="url" value="${escapeHtml(item.override?.destinationB || '')}" placeholder="${escapeHtml(family.destinationB)}" data-ctc-rendition-override-field="destinationB" ${effective.testMode !== 'ab' ? 'disabled' : ''}>
                </label>
              </div>
              <div class="ctc-override-grid">
                ${core.UTM_KEYS.filter((key) => !['utm_medium', 'utm_content'].includes(key)).map((key) => `
                  <label class="ctc-field">
                    <span>${escapeHtml(key)} override</span>
                    <select data-ctc-rendition-utm="${escapeHtml(key)}">${renderOptions(key, item.override?.utms?.[key] || '', true)}</select>
                  </label>
                `).join('')}
              </div>
            ` : '<p class="ctc-inherit-note">This rendition uses the family destinations, UTM selections, and testing rule.</p>'}
            <div class="ctc-rendition-metadata">
              <div class="ctc-section-heading">
                <div><h4>Rendition QA metadata</h4><p>Confirm the delivery format and requirements for this asset.</p></div>
              </div>
              <div class="ctc-override-grid">
                <label class="ctc-field">
                  <span>Format</span>
                  <select data-ctc-rendition-meta="format">
                    ${Object.entries(FORMAT_LABELS).map(([value, label]) => `<option value="${escapeHtml(value)}" ${item.format === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('')}
                  </select>
                </label>
                <label class="ctc-field">
                  <span>Width (px)</span>
                  <input type="number" min="1" step="1" value="${escapeHtml(item.width || '')}" data-ctc-rendition-meta="width">
                </label>
                <label class="ctc-field">
                  <span>Height (px)</span>
                  <input type="number" min="1" step="1" value="${escapeHtml(item.height || '')}" data-ctc-rendition-meta="height">
                </label>
                ${['video', 'ctv'].includes(item.format) ? `
                  <label class="ctc-field">
                    <span>Duration</span>
                    <input type="text" value="${escapeHtml(item.duration || '')}" placeholder="15s" data-ctc-rendition-meta="duration">
                  </label>
                ` : ''}
              </div>
              ${item.format === 'interactive' ? `
                <div class="ctc-qa-controls">
                  <label class="ctc-field">
                    <span>Interactive preview URL</span>
                    <input type="url" value="${escapeHtml(item.previewUrl || '')}" placeholder="https://creative.example.com/preview/" data-ctc-rendition-meta="previewUrl">
                  </label>
                  <label class="ctc-check-field">
                    <input type="checkbox" data-ctc-rendition-meta="clickTagValid" ${item.clickTagValid ? 'checked' : ''}>
                    <span>Click tag tested and valid</span>
                  </label>
                </div>
              ` : ''}
              ${item.format === 'ctv' ? `
                <label class="ctc-check-field">
                  <input type="checkbox" data-ctc-rendition-meta="qrAttached" ${item.qrAttached ? 'checked' : ''}>
                  <span>QR destination asset attached and reviewed</span>
                </label>
              ` : ''}
            </div>
            <div class="ctc-qa-row ${validation.valid ? 'is-valid' : 'is-error'}">
              ${validation.valid ? icon('check') : icon('warning')}
              <span>${validation.valid ? 'Rendition QA passed' : escapeHtml(validation.issues[0]?.message || 'Rendition needs review')}</span>
              ${item.format === 'interactive' && safeHttpHref(item.previewUrl) ? `<a href="${escapeHtml(safeHttpHref(item.previewUrl))}" target="_blank" rel="noopener noreferrer">Open preview</a>` : ''}
            </div>
          </div>
          <div class="ctc-rule-section ctc-generated-links">
            <div class="ctc-section-heading">
              <div>
                <h4>Generated links</h4>
                <p>Effective values for the selected rendition</p>
              </div>
            </div>
            ${renderLinkPreview(family, item)}
          </div>
        ` : ''}
      </aside>
    `;
  };

  const renderLibrary = () => {
    const family = selectedFamily();
    if (!family) return '<div class="ctc-empty"><h2>No creative families yet</h2><p>Create a family, then import its renditions.</p><button class="ctc-button ctc-button-primary" data-ctc-action="new-family">Create family</button></div>';
    return `<div class="ctc-library-layout">${renderFamilyRail()}${renderRenditionPane(family)}${renderInspector(family, selectedRendition())}</div>`;
  };

  const renderMetric = (value, label, iconName, tone = '') => `
    <div class="ctc-metric ${tone}">
      <span class="ctc-metric-icon">${icon(iconName)}</span>
      <strong>${value}</strong>
      <span>${escapeHtml(label)}</span>
    </div>
  `;

  const renderDashboardRendition = (family, item) => {
    const effective = core.getEffectiveConfig(family, item);
    const validation = core.validateRendition(family, item, state.dictionaries);
    return `
      <div class="ctc-dashboard-child">
        <span class="ctc-tree-line" aria-hidden="true"></span>
        ${renderAsset(family, item, true)}
        <span class="ctc-dashboard-name"><strong>${escapeHtml(item.name)}</strong><small>${item.overrideEnabled ? 'Overridden' : 'Inherited'} · ${escapeHtml(TEST_MODE_LABELS[effective.testMode] || effective.testMode)}</small></span>
        <span class="ctc-dashboard-utm">${escapeHtml(optionLabel('utm_medium', effective.utms.utm_medium))} · ${escapeHtml(optionLabel('utm_content', effective.utms.utm_content))}</span>
        <span class="ctc-dashboard-test">${escapeHtml(TEST_MODE_LABELS[effective.testMode] || effective.testMode)}</span>
        <span class="ctc-dashboard-qa ${validation.valid ? 'is-valid' : 'is-error'}">${validation.valid ? icon('check') : icon('warning')}<span>${validation.valid ? 'Passed' : 'Review'}</span></span>
      </div>
    `;
  };

  const renderDashboardFamily = (family) => {
    const expanded = state.ui.expandedFamilyIds.includes(family.id);
    const familyValid = family.renditions.every((item) => core.validateRendition(family, item, state.dictionaries).valid);
    return `
      <section class="ctc-dashboard-family ${expanded ? 'is-expanded' : ''}">
        <button type="button" class="ctc-dashboard-family-row" data-ctc-action="toggle-dashboard-family" data-family-id="${escapeHtml(family.id)}" aria-expanded="${expanded}">
          <span class="ctc-dashboard-chevron">${icon('chevron')}</span>
          ${renderAsset(family, family.renditions[0] || {}, true)}
          <span class="ctc-dashboard-name"><strong>${escapeHtml(family.name)}</strong><small>${escapeHtml(family.id)}</small></span>
          <span class="ctc-dashboard-partner">${escapeHtml(family.partner)}</span>
          <span class="ctc-dashboard-utm">${escapeHtml(optionLabel('utm_id', family.utms.utm_id))} · ${escapeHtml(optionLabel('utm_campaign', family.utms.utm_campaign))}</span>
          <span class="ctc-dashboard-test">${escapeHtml(TEST_MODE_LABELS[family.testMode])}</span>
          <span class="ctc-dashboard-qa ${familyValid ? 'is-valid' : 'is-error'}">${familyValid ? icon('check') : icon('warning')}<span>${familyValid ? 'QA passed' : 'Review'}</span></span>
          <span class="ctc-dashboard-count">${family.renditions.length}</span>
        </button>
        ${expanded ? `<div class="ctc-dashboard-children">${family.renditions.map((item) => renderDashboardRendition(family, item)).join('')}</div>` : ''}
      </section>
    `;
  };

  const renderDashboard = () => {
    const summary = core.campaignSummary(state.campaign, state.dictionaries);
    const overrides = state.campaign.families.reduce((sum, family) => sum + family.renditions.filter((item) => item.overrideEnabled).length, 0);
    return `
      <div class="ctc-dashboard">
        <div class="ctc-dashboard-toolbar">
          <div>
            <h2>Campaign delivery</h2>
            <p>${escapeHtml(state.campaign.name)}</p>
          </div>
          <div class="ctc-toolbar-actions">
            <button type="button" class="ctc-button ctc-button-secondary" data-ctc-action="manage-dictionaries">${icon('settings')}Manage UTM options</button>
            <button type="button" class="ctc-button ctc-button-primary" data-ctc-action="view-export">${icon('export')}Export</button>
          </div>
        </div>
        <div class="ctc-metrics">
          ${renderMetric(summary.familyCount, 'creative families', 'library')}
          ${renderMetric(summary.renditionCount, 'renditions', 'play')}
          ${renderMetric(summary.linkCount, 'tagged links', 'link')}
          ${renderMetric(summary.abPairCount, 'A/B pairs', 'dashboard')}
          ${renderMetric(summary.qrCount, 'QR assets', 'file')}
          ${renderMetric(summary.errorRenditionCount, 'errors', summary.errorRenditionCount ? 'warning' : 'check', summary.errorRenditionCount ? 'is-error' : 'is-success')}
        </div>
        <div class="ctc-dashboard-layout">
          <section class="ctc-dashboard-table" aria-labelledby="ctc-family-dashboard-title">
            <div class="ctc-dashboard-table-header">
              <h3 id="ctc-family-dashboard-title">Creative families</h3>
              <div class="ctc-dashboard-columns" aria-hidden="true"><span>Family</span><span>Partner</span><span>Controlled UTM rules</span><span>Test scope</span><span>QA</span><span>Renditions</span></div>
            </div>
            <div class="ctc-dashboard-list">${state.campaign.families.map(renderDashboardFamily).join('')}</div>
          </section>
          <aside class="ctc-activity-rail">
            <section>
              <h3>Activity</h3>
              <div class="ctc-activity-item"><span class="ctc-activity-icon">${icon('settings')}</span><div><strong>UTM taxonomy</strong><span>Version 3 · Active</span><small>${Object.keys(state.dictionaries).length} controlled dictionaries</small></div></div>
            </section>
            <section>
              <h3>Approval history</h3>
              <div class="ctc-timeline-item">${icon('check')}<div><strong>Basis family rules reviewed</strong><span>Prototype seed · Marketing team</span></div></div>
              <div class="ctc-timeline-item">${icon('check')}<div><strong>CTV test scope reviewed</strong><span>N/A with QR delivery</span></div></div>
            </section>
            <div class="ctc-attention">${icon('warning')}<span>${overrides} rendition overrides in this campaign</span></div>
          </aside>
        </div>
      </div>
    `;
  };

  const EXPORT_FORMATS = [
    { id: 'excel', title: 'Excel workbook', extension: '.xlsx', available: false, description: 'Output concept: Family Summary, Rendition Index, URL Audit, and UTM Dictionary sheets.' },
    { id: 'powerpoint', title: 'PowerPoint handoff', extension: '.pptx', available: false, description: 'Output concept: family overview slides plus rendition and link details.' },
    { id: 'pdf', title: 'PDF approval report', extension: '.pdf', available: false, description: 'Output concept: review-ready UTM, testing, and sign-off proof.' },
    { id: 'partner-csv', title: 'Excel-ready partner CSV', extension: '.csv', available: true, description: 'Downloadable now: one row per rendition and link variant.' },
    { id: 'package', title: 'Campaign manifest', extension: '.json', available: true, description: 'Downloadable now: creative metadata, dictionary, validation, and inheritance state.' },
  ];

  const renderExportFormat = (format) => {
    const available = format.available !== false;
    const checked = available && state.ui.exportFormats.includes(format.id);
    return `
      <div class="ctc-export-row ${available ? '' : 'is-preview-only'}">
        <label class="ctc-export-check">
          <input type="checkbox" data-ctc-export-format="${format.id}" ${checked ? 'checked' : ''} ${available ? '' : 'disabled'} aria-label="${available ? 'Select' : 'Preview'} ${escapeHtml(format.title)}">
          <span aria-hidden="true">${checked ? icon('check') : ''}</span>
          <em>${format.id === 'package' ? icon('folder') : icon('file')}</em>
        </label>
        <div class="ctc-export-copy"><h3>${escapeHtml(format.title)} <small>${escapeHtml(format.extension)}</small>${available ? '' : '<span class="ctc-preview-badge">Preview only</span>'}</h3><p>${escapeHtml(format.description)}</p></div>
        <button type="button" class="ctc-button ctc-button-secondary" data-ctc-action="preview-export" data-export-id="${format.id}">${icon('eye')}Preview</button>
      </div>
    `;
  };

  const renderPackageTree = () => `
    <div class="ctc-package-tree">
      <div class="ctc-tree-row is-root">${icon('folder')}<strong>Creative families</strong></div>
      ${state.campaign.families.map((family, index) => `
        <div class="ctc-tree-row is-family" style="--tree-index:${index}">${icon('folder')}<span>${escapeHtml(family.name)}</span><small>${family.renditions.length}</small></div>
        ${index === 0 ? family.renditions.slice(0, 4).map((item) => `<div class="ctc-tree-row is-rendition">${icon('file')}<span>${escapeHtml(item.name)}</span></div>`).join('') : ''}
      `).join('')}
    </div>
  `;

  const countUnknownControlledValues = () => {
    const unknown = new Set();
    state.campaign.families.forEach((family) => {
      const renditions = family.renditions.length ? family.renditions : [{}];
      renditions.forEach((item) => {
        const config = core.getEffectiveConfig(family, item);
        core.UTM_KEYS.forEach((key) => {
          const value = String(config.utms?.[key] || '');
          const allowed = (state.dictionaries[key] || []).some((option) => option.value === value);
          if (value && !allowed) unknown.add(`${key}:${value}`);
        });
      });
    });
    return unknown.size;
  };
  const renderExport = () => {
    const summary = core.campaignSummary(state.campaign, state.dictionaries);
    const unknownCount = countUnknownControlledValues();
    return `
      <div class="ctc-export">
        <header class="ctc-export-header">
          <div><h2>Export campaign package</h2><p>Generate handoff files from the current governed campaign state.</p></div>
          <dl><div><dt>Campaign</dt><dd>${escapeHtml(state.campaign.name)}</dd></div><div><dt>Version</dt><dd>v${escapeHtml(state.campaign.version)}</dd></div><div><dt>Status</dt><dd class="${summary.errorRenditionCount ? 'is-error' : 'is-ready'}">${summary.errorRenditionCount ? 'Needs review' : 'Ready to deliver'}</dd></div></dl>
        </header>
        <div class="ctc-export-metrics">
          <span><strong>${summary.familyCount}</strong> families</span>
          <span><strong>${summary.renditionCount}</strong> renditions</span>
          <span><strong>${summary.linkCount}</strong> tagged links</span>
          <span><strong>${summary.abPairCount}</strong> A/B pairs</span>
          <span><strong>${summary.qrCount}</strong> QR assets</span>
          <span class="${summary.errorRenditionCount ? 'is-error' : 'is-success'}"><strong>${summary.errorRenditionCount}</strong> errors</span>
        </div>
        <div class="ctc-export-layout">
          <section class="ctc-export-formats" aria-label="Export formats">${EXPORT_FORMATS.map(renderExportFormat).join('')}</section>
          <aside class="ctc-package-contents">
            <h3>Package contents</h3>
            ${renderPackageTree()}
            <div class="ctc-package-meta"><strong>Controlled UTM dictionary v3</strong><span>${Object.keys(state.dictionaries).length} parameter sets</span><small>${unknownCount ? icon('warning') : icon('check')}${unknownCount} unknown value${unknownCount === 1 ? '' : 's'}</small></div>
            <div class="ctc-manifest-row">${icon('file')}<span>Inheritance / override manifest</span>${icon('chevron')}</div>
            <div class="ctc-version-row"><span>Version history</span><div><button type="button">v2</button><button type="button">v3</button><button type="button" class="is-current">v4</button></div></div>
          </aside>
        </div>
        <div class="ctc-export-actions">
          <button type="button" class="ctc-button ctc-button-secondary" data-ctc-action="save-export-preset">Save export preset</button>
          <button type="button" class="ctc-button ctc-button-primary" data-ctc-action="generate-package" ${state.ui.exportFormats.length ? '' : 'disabled'}>Download selected files</button>
        </div>
      </div>
    `;
  };

  const renderAppHeader = () => `
    <header class="ctc-app-header">
      <div class="ctc-app-title"><h2>Creative families</h2><p>${escapeHtml(state.campaign.name)}</p></div>
      <nav class="ctc-view-tabs" role="tablist" aria-label="Campaign tracker views">
        ${[
          ['library', 'Creative library', 'library'],
          ['dashboard', 'Delivery dashboard', 'dashboard'],
          ['export', 'Export hub', 'export'],
        ].map(([id, label, iconName]) => `
          <button type="button" role="tab" id="ctc-tab-${id}" aria-controls="ctc-panel-${id}" data-ctc-action="set-view" data-view="${id}" aria-selected="${state.ui.view === id}" tabindex="${state.ui.view === id ? '0' : '-1'}" class="${state.ui.view === id ? 'is-active' : ''}">${icon(iconName)}<span>${escapeHtml(label)}</span></button>
        `).join('')}
      </nav>
      <div class="ctc-app-actions">
        <button type="button" class="ctc-button ctc-button-secondary" data-ctc-action="import" aria-label="Import creative renditions">${icon('upload')}<span class="ctc-action-label">Import</span></button>
        <button type="button" class="ctc-button ctc-button-primary" data-ctc-action="new-family" aria-label="Create creative family">${icon('plus')}<span class="ctc-action-label">New family</span></button>
        <button type="button" class="ctc-icon-button" data-ctc-action="reset-demo" aria-label="Reset prototype data" title="Reset prototype data">${icon('reset')}</button>
      </div>
    </header>
  `;

  function render() {
    const views = {
      library: renderLibrary,
      dashboard: renderDashboard,
      export: renderExport,
    };
    const view = views[state.ui.view] ? state.ui.view : 'library';
    root.innerHTML = `${renderAppHeader()}<div class="ctc-view" id="ctc-panel-${view}" role="tabpanel" aria-labelledby="ctc-tab-${view}" tabindex="0">${views[view]()}</div>`;
    renderImportFamilyOptions();
  }

  const openDialog = (name) => {
    const dialog = document.querySelector(`[data-ctc-dialog="${name}"]`);
    if (!dialog) return;
    if (name === 'dictionary') renderDictionaryDialog();
    if (name === 'import') renderImportFamilyOptions();
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  };

  const closeDialog = (button) => {
    const dialog = button.closest('dialog');
    if (!dialog) return;
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
  };

  const renderImportFamilyOptions = () => {
    const select = document.querySelector('[data-ctc-import-family]');
    if (!select) return;
    select.innerHTML = state.campaign.families.map((family) => `<option value="${escapeHtml(family.id)}" ${family.id === state.ui.selectedFamilyId ? 'selected' : ''}>${escapeHtml(family.name)}</option>`).join('');
  };

  const renderDictionaryDialog = () => {
    const list = document.querySelector('[data-ctc-dictionary-list]');
    const parameter = document.querySelector('[data-ctc-form="dictionary"] select[name="parameter"]');
    if (!list || !parameter) return;
    list.innerHTML = core.UTM_KEYS.map((key) => `
      <section class="ctc-dictionary-group">
        <div><h3>${escapeHtml(key)}</h3><span>${state.dictionaries[key]?.length || 0} values</span></div>
        <div class="ctc-option-list">
          ${(state.dictionaries[key] || []).map((option) => `
            <span class="ctc-option-chip"><strong>${escapeHtml(option.label)}</strong><code>${escapeHtml(option.value)}</code><button type="button" data-ctc-action="remove-option" data-parameter="${key}" data-value="${escapeHtml(option.value)}" aria-label="Remove ${escapeHtml(option.label)}">&times;</button></span>
          `).join('')}
        </div>
      </section>
    `).join('');
    parameter.innerHTML = core.UTM_KEYS.map((key) => `<option value="${key}">${escapeHtml(key)}</option>`).join('');
  };

  const setFormStatus = (name, message, tone = '') => {
    const target = document.querySelector(`[data-ctc-form-status="${name}"]`);
    if (!target) return;
    target.textContent = message;
    target.dataset.tone = tone;
  };

  const createId = (value, fallback) => {
    const normalized = String(value || '').trim().toUpperCase().replace(/[^A-Z0-9]+/g, '-').replace(/^-|-$/g, '');
    return normalized || `${fallback}-${Date.now().toString(36).toUpperCase()}`;
  };

  const createUniqueRenditionId = (family, value) => {
    const base = createId(value, 'RENDITION');
    const ids = new Set((family.renditions || []).map((item) => item.id));
    let candidate = base;
    let suffix = 2;
    while (ids.has(candidate)) {
      candidate = `${base}-${suffix}`;
      suffix += 1;
    }
    return candidate;
  };
  const handleFamilyForm = (form) => {
    const data = new FormData(form);
    const id = createId(data.get('familyId'), 'FAMILY');
    if (state.campaign.families.some((family) => family.id === id)) {
      setFormStatus('family', 'That family ID already exists.', 'error');
      return;
    }
    const partner = String(data.get('partner') || 'basis');
    const name = String(data.get('familyName') || '').trim();
    const destinationA = String(data.get('destinationA') || '').trim();
    if (!safeHttpHref(destinationA)) {
      setFormStatus('family', 'Enter a valid HTTP or HTTPS destination.', 'error');
      return;
    }
    const contentValue = core.normalizeValue(name);
    if (!(state.dictionaries.utm_content || []).some((option) => option.value === contentValue)) {
      state.dictionaries.utm_content.push({ value: contentValue, label: name });
    }
    const family = makeFamily({
      id,
      name,
      partner,
      destination: destinationA,
      testMode: 'single',
      content: contentValue,
      campaign: state.dictionaries.utm_campaign[0]?.value || 'campaign',
      renditions: [],
    });
    state.campaign.families.push(family);
    state.ui.selectedFamilyId = family.id;
    state.ui.selectedRenditionId = '';
    state.ui.view = 'library';
    form.reset();
    form.querySelector('[name="destinationA"]').value = 'https://www.visitgrandjunction.com/';
    form.closest('dialog')?.close();
    commit();
    showToast('Creative family created.', 'success');
  };

  const getImageDimensions = (file, url) => new Promise((resolve) => {
    if (!file.type.startsWith('image/')) {
      resolve({ width: 0, height: 0 });
      return;
    }
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => resolve({ width: 0, height: 0 });
    image.src = url;
  });
  const getMediaMetadata = (file, url) => new Promise((resolve) => {
    const empty = { width: 0, height: 0, duration: '' };
    if (file.type.startsWith('image/')) {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight, duration: '' });
      image.onerror = () => resolve(empty);
      image.src = url;
      return;
    }
    if (file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.onloadedmetadata = () => resolve({
        width: video.videoWidth,
        height: video.videoHeight,
        duration: Number.isFinite(video.duration) ? `${Math.max(1, Math.round(video.duration))}s` : '',
      });
      video.onerror = () => resolve(empty);
      video.preload = 'metadata';
      video.src = url;
      return;
    }
    resolve(empty);
  });

  const inferFormat = (file) => {
    const name = file.name.toLowerCase();
    if (name.includes('ctv')) return 'ctv';
    if (file.type.startsWith('video/')) return 'video';
    if (name.endsWith('.zip')) return 'interactive';
    if (file.type === 'image/gif') return 'animated';
    return 'static';
  };

  const handleImportForm = async (form) => {
    const data = new FormData(form);
    const family = state.campaign.families.find((item) => item.id === data.get('familyId'));
    const files = Array.from(form.elements.creativeFiles?.files || []);
    if (!family || !files.length) {
      setFormStatus('import', 'Choose a family and at least one creative file.', 'error');
      return;
    }
    setFormStatus('import', 'Reading rendition metadata…');
    const created = [];
    for (const file of files) {
      const id = createUniqueRenditionId(family, `${family.id}-${file.name.replace(/\.[^.]+$/, '')}`);
      const objectUrl = URL.createObjectURL(file);
      const metadata = await getMediaMetadata(file, objectUrl);
      const format = inferFormat(file);
      const item = rendition({
        id,
        name: file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' '),
        width: metadata.width,
        height: metadata.height,
        duration: metadata.duration,
        format,
        asset: '',
        previewUrl: format === 'interactive' ? '' : '',
        clickTagValid: false,
        qrAttached: false,
        overrideEnabled: format === 'ctv',
        override: format === 'ctv' ? { testMode: 'na', utms: { utm_medium: 'ctv' } } : {},
      });
      item.fileName = file.name;
      item.fileType = file.type || 'application/octet-stream';
      item.fileSize = file.size;
      runtimeAssetUrls.set(id, objectUrl);
      family.renditions.push(item);
      created.push(item);
    }
    state.ui.selectedFamilyId = family.id;
    state.ui.selectedRenditionId = created[0]?.id || '';
    state.ui.view = 'library';
    form.reset();
    form.closest('dialog')?.close();
    commit();
    showToast(`${created.length} rendition${created.length === 1 ? '' : 's'} imported for review.`, 'success');
  };

  const handleDictionaryForm = (form) => {
    const data = new FormData(form);
    const key = String(data.get('parameter') || '');
    const value = core.normalizeValue(data.get('value'));
    const label = String(data.get('label') || '').trim();
    if (!core.UTM_KEYS.includes(key) || !value || !label) {
      setFormStatus('dictionary', 'Complete all three fields.', 'error');
      return;
    }
    if ((state.dictionaries[key] || []).some((option) => option.value === value)) {
      setFormStatus('dictionary', 'That approved value already exists.', 'error');
      return;
    }
    state.dictionaries[key].push({ value, label });
    form.reset();
    setFormStatus('dictionary', `${label} added to ${key}.`, 'success');
    persistState();
    markSessionDirty();
    renderDictionaryDialog();
    render();
  };

  const optionIsUsed = (key, value) => state.campaign.families.some((family) => (
    family.utms?.[key] === value
    || family.renditions.some((item) => item.override?.utms?.[key] === value)
  ));

  const downloadBlob = (filename, content, type) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const generatePackageLegacy = () => {
    const rows = core.campaignToExportRows(state.campaign, state.dictionaries);
    const summary = core.campaignSummary(state.campaign, state.dictionaries);
    const packageData = {
      schema: 'campaign-creative-tracker.package.v1',
      generatedAt: new Date().toISOString(),
      selectedFormats: state.ui.exportFormats,
      campaign: sanitizeStateForStorage(state).campaign,
      dictionaries: state.dictionaries,
      summary,
      renditionRows: rows,
    };
    downloadBlob('grand-junction-summer-2026-campaign-package.json', JSON.stringify(packageData, null, 2), 'application/json;charset=utf-8');
    if (state.ui.exportFormats.includes('partner-csv')) {
      window.setTimeout(() => downloadBlob('grand-junction-summer-2026-partner-links.csv', core.exportRowsToCsv(rows), 'text/csv;charset=utf-8'), 180);
    }
    showToast('Campaign manifest generated. Selected document formats are represented in the package plan.', 'success');
    try {
      document.dispatchEvent(new CustomEvent('tools:run-complete', { detail: { toolId: TOOL_ID, itemCount: rows.length } }));
    } catch {}
  };
  const generatePackage = () => {
    const rows = core.campaignToExportRows(state.campaign, state.dictionaries);
    const summary = core.campaignSummary(state.campaign, state.dictionaries);
    if (summary.errorRenditionCount > 0) {
      showToast(`Resolve ${summary.errorRenditionCount} rendition QA error${summary.errorRenditionCount === 1 ? '' : 's'} before export.`, 'error');
      try {
        document.dispatchEvent(new CustomEvent('tools:run-error', { detail: { toolId: TOOL_ID, errorCount: summary.errorRenditionCount } }));
      } catch {}
      return;
    }
    if (!state.ui.exportFormats.length) {
      showToast('Select CSV or JSON before downloading.', 'error');
      return;
    }

    const packageData = {
      schema: 'campaign-creative-tracker.package.v1',
      generatedAt: new Date().toISOString(),
      selectedFormats: state.ui.exportFormats,
      campaign: sanitizeStateForStorage(state).campaign,
      dictionaries: state.dictionaries,
      summary,
      renditionRows: rows,
      assetPolicy: 'Uploaded creative bytes remain browser-session only; filenames and QA metadata are included.',
    };
    const fileStem = core.normalizeValue(state.campaign.id || state.campaign.name) || 'campaign';
    let downloadCount = 0;
    if (state.ui.exportFormats.includes('package')) {
      downloadBlob(`${fileStem}-campaign-manifest.json`, JSON.stringify(packageData, null, 2), 'application/json;charset=utf-8');
      downloadCount += 1;
    }
    if (state.ui.exportFormats.includes('partner-csv')) {
      const delay = downloadCount ? 180 : 0;
      window.setTimeout(() => downloadBlob(`${fileStem}-partner-links.csv`, core.exportRowsToCsv(rows), 'text/csv;charset=utf-8'), delay);
      downloadCount += 1;
    }
    showToast(`${downloadCount} export file${downloadCount === 1 ? '' : 's'} prepared.`, 'success');
    try {
      document.dispatchEvent(new CustomEvent('tools:run-complete', { detail: { toolId: TOOL_ID, itemCount: rows.length, downloadCount } }));
    } catch {}
  };

  const previewExport = (id) => {
    const format = EXPORT_FORMATS.find((item) => item.id === id) || EXPORT_FORMATS[0];
    const rows = core.campaignToExportRows(state.campaign, state.dictionaries);
    const dialog = document.querySelector('[data-ctc-dialog="preview"]');
    const title = dialog?.querySelector('[data-ctc-preview-title]');
    const summary = dialog?.querySelector('[data-ctc-preview-summary]');
    const body = dialog?.querySelector('[data-ctc-preview-body]');
    if (!dialog || !title || !summary || !body) return;
    title.textContent = `${format.title} preview`;
    summary.textContent = format.description;
    body.innerHTML = `
      <div class="ctc-preview-tabs"><span class="is-active">Family Summary</span><span>Rendition Index</span><span>URL Audit</span><span>UTM Dictionary</span></div>
      <div class="ctc-preview-table-wrap">
        <table class="ctc-preview-table">
          <thead><tr><th>Family</th><th>Rendition</th><th>Format</th><th>Test</th><th>Variant</th><th>Status</th></tr></thead>
          <tbody>${rows.slice(0, 12).map((row) => `<tr><td>${escapeHtml(row.family_name)}</td><td>${escapeHtml(row.rendition_name)}</td><td>${escapeHtml(row.format)}</td><td>${escapeHtml(TEST_MODE_LABELS[row.test_mode] || row.test_mode)}</td><td>${escapeHtml(row.variant.toUpperCase())}</td><td class="${row.validation_status === 'valid' ? 'is-valid' : 'is-error'}">${escapeHtml(row.validation_status)}</td></tr>`).join('')}</tbody>
        </table>
      </div>
      <p class="ctc-preview-note">Prototype preview: the test page generates a real JSON campaign manifest and partner CSV. Native XLSX, PPTX, and PDF renderers are the next implementation phase.</p>
    `;
    dialog.showModal();
  };

  root.addEventListener('keydown', (event) => {
    const tab = event.target.closest('[role="tab"][data-view]');
    if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const tabs = Array.from(root.querySelectorAll('[role="tab"][data-view]'));
    const currentIndex = tabs.indexOf(tab);
    if (currentIndex < 0) return;
    event.preventDefault();
    const nextIndex = event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? tabs.length - 1
        : (currentIndex + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[nextIndex]?.click();
  });
  root.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-ctc-action]');
    if (!button) return;
    const action = button.dataset.ctcAction;

    if (action === 'set-view') {
      const nextView = button.dataset.view || 'library';
      state.ui.view = nextView;
      commit(false);
      window.requestAnimationFrame(() => {
        root.querySelector(`[role="tab"][data-view="${nextView}"]`)?.focus();
      });
      return;
    }
    if (action === 'select-family') {
      const family = state.campaign.families.find((item) => item.id === button.dataset.familyId);
      if (!family) return;
      state.ui.selectedFamilyId = family.id;
      state.ui.selectedRenditionId = family.renditions[0]?.id || '';
      commit(false);
      return;
    }
    if (action === 'select-rendition') {
      state.ui.selectedRenditionId = button.dataset.renditionId || '';
      commit(false);
      return;
    }
    if (action === 'new-family') {
      openDialog('family');
      return;
    }
    if (action === 'import') {
      openDialog('import');
      return;
    }
    if (action === 'manage-dictionaries') {
      openDialog('dictionary');
      return;
    }
    if (action === 'view-export') {
      state.ui.view = 'export';
      commit(false);
      return;
    }
    if (action === 'set-test-mode') {
      const family = selectedFamily();
      const item = selectedRendition();
      const mode = button.dataset.ctcTestMode;
      if (!family || !core.TEST_MODES.includes(mode)) return;
      if (button.dataset.ctcTestScope === 'rendition' && item) {
        item.overrideEnabled = true;
        item.override.testMode = mode;
      } else {
        family.testMode = mode;
      }
      commit();
      return;
    }
    if (action === 'apply-to-all') {
      const family = selectedFamily();
      if (!family) return;
      family.renditions.forEach((item) => {
        item.overrideEnabled = false;
      });
      commit();
      showToast('All renditions now inherit the family rules.', 'success');
      return;
    }
    if (action === 'copy-url') {
      try {
        await navigator.clipboard.writeText(button.dataset.url || '');
        showToast('Tagged URL copied.', 'success');
      } catch {
        showToast('Copy failed. Select the URL manually.', 'error');
      }
      return;
    }
    if (action === 'toggle-dashboard-family') {
      const id = button.dataset.familyId;
      state.ui.expandedFamilyIds = state.ui.expandedFamilyIds.includes(id)
        ? state.ui.expandedFamilyIds.filter((item) => item !== id)
        : [...state.ui.expandedFamilyIds, id];
      commit(false);
      return;
    }
    if (action === 'preview-export') {
      previewExport(button.dataset.exportId);
      return;
    }
    if (action === 'generate-package') {
      try {
        document.dispatchEvent(new CustomEvent('tools:run-start', { detail: { toolId: TOOL_ID } }));
      } catch {}
      generatePackage();
      return;
    }
    if (action === 'save-export-preset') {
      persistState();
      markSessionDirty();
      showToast('Export preset saved locally.', 'success');
      return;
    }
    if (action === 'remove-option') {
      const key = button.dataset.parameter;
      const value = button.dataset.value;
      if (optionIsUsed(key, value)) {
        setFormStatus('dictionary', 'This value is in use. Reassign affected records before removing it.', 'error');
        return;
      }
      state.dictionaries[key] = (state.dictionaries[key] || []).filter((option) => option.value !== value);
      persistState();
      markSessionDirty();
      renderDictionaryDialog();
      render();
      return;
    }
    if (action === 'reset-dictionaries') {
      const previous = state.dictionaries;
      const next = deepClone(DEFAULT_DICTIONARIES);
      state.campaign.families.forEach((family) => {
        const utmSets = [
          family.utms,
          ...family.renditions.map((item) => item.override?.utms || {}),
        ];
        utmSets.forEach((utms) => {
          core.UTM_KEYS.forEach((key) => {
            const value = String(utms?.[key] || '');
            if (!value || next[key].some((option) => option.value === value)) return;
            const prior = (previous[key] || []).find((option) => option.value === value);
            next[key].push({ value, label: prior?.label || value });
          });
        });
      });
      state.dictionaries = next;
      persistState();
      markSessionDirty();
      renderDictionaryDialog();
      render();
      setFormStatus('dictionary', 'Default controlled values restored.', 'success');
      return;
    }
    if (action === 'reset-demo') {
      if (!window.confirm('Reset the local prototype to its original sample campaign?')) return;
      runtimeAssetUrls.forEach((url) => URL.revokeObjectURL(url));
      runtimeAssetUrls.clear();
      state = buildSeedState();
      persistState();
      render();
      markSessionDirty();
      showToast('Prototype data reset.', 'success');
    }
  });

  root.addEventListener('change', (event) => {
    const target = event.target;
    const family = selectedFamily();
    const item = selectedRendition();
    if (!family) return;

    if (target.matches('[data-ctc-family-utm]')) {
      family.utms[target.dataset.ctcFamilyUtm] = target.value;
      commit();
      return;
    }
    if (target.matches('[data-ctc-family-field]')) {
      family[target.dataset.ctcFamilyField] = target.value;
      commit();
      return;
    }
    if (target.matches('[data-ctc-rendition-field="overrideEnabled"]') && item) {
      item.overrideEnabled = target.checked;
      commit();
      return;
    }
    if (target.matches('[data-ctc-rendition-override-field]') && item) {
      const field = target.dataset.ctcRenditionOverrideField;
      if (['destinationA', 'destinationB'].includes(field)) {
        item.overrideEnabled = true;
        item.override[field] = target.value;
        commit();
      }
      return;
    }
    if (target.matches('[data-ctc-rendition-meta]') && item) {
      const field = target.dataset.ctcRenditionMeta;
      if (['clickTagValid', 'qrAttached'].includes(field)) {
        item[field] = target.checked;
      } else if (['width', 'height'].includes(field)) {
        item[field] = Math.max(0, Number(target.value) || 0);
      } else if (['format', 'duration', 'previewUrl'].includes(field)) {
        item[field] = target.value;
      }
      commit();
      return;
    }
    if (target.matches('[data-ctc-rendition-utm]') && item) {
      const key = target.dataset.ctcRenditionUtm;
      if (target.value) item.override.utms[key] = target.value;
      else delete item.override.utms[key];
      commit();
      return;
    }
    if (target.matches('[data-ctc-export-format]')) {
      const id = target.dataset.ctcExportFormat;
      state.ui.exportFormats = target.checked
        ? Array.from(new Set([...state.ui.exportFormats, id]))
        : state.ui.exportFormats.filter((value) => value !== id);
      commit();
    }
  });

  document.addEventListener('click', (event) => {
    const closeButton = event.target.closest('[data-ctc-close-dialog]');
    if (closeButton) closeDialog(closeButton);
  });

  document.addEventListener('submit', (event) => {
    const form = event.target.closest('[data-ctc-form]');
    if (!form) return;
    event.preventDefault();
    const type = form.dataset.ctcForm;
    if (type === 'family') handleFamilyForm(form);
    if (type === 'import') handleImportForm(form);
    if (type === 'dictionary') handleDictionaryForm(form);
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail || {};
    if (detail.toolId !== TOOL_ID) return;
    const snapshotState = sanitizeStateForStorage(state);
    const summary = core.campaignSummary(state.campaign, state.dictionaries);
    const output = {
      kind: 'campaign-creative-tracker',
      summary: `${summary.familyCount} families, ${summary.renditionCount} renditions, ${summary.linkCount} tagged links`,
      campaignId: state.campaign.id,
      version: state.campaign.version,
    };
    if (detail.payload && typeof detail.payload === 'object') {
      detail.payload.outputSummary = output.summary;
      detail.payload.inputs = { state: snapshotState };
      detail.payload.output = output;
    }
    if (detail.snapshot && typeof detail.snapshot === 'object') {
      detail.snapshot.inputs = { state: snapshotState };
      detail.snapshot.output = output;
    }
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail || {};
    if (detail.toolId !== TOOL_ID) return;
    const restored = detail.snapshot?.inputs?.state;
    if (!restored) return;
    runtimeAssetUrls.forEach((url) => URL.revokeObjectURL(url));
    runtimeAssetUrls.clear();
    state = normalizeLoadedState(restored);
    persistState();
    render();
  });

  window.addEventListener('beforeunload', () => {
    runtimeAssetUrls.forEach((url) => URL.revokeObjectURL(url));
  });

  normalizeSelection();
  persistState();
  render();
})();
