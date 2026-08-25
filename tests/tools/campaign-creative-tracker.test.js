const core = require('../../js/tools/campaign-creative-tracker-core.js');

const makeRendition = (overrides = {}) => ({
  id: 'rendition-display',
  name: 'Display 300x250',
  width: 300,
  height: 250,
  format: 'static',
  duration: '',
  assetSrc: 'display-300x250.jpg',
  previewUrl: '',
  clickTagValid: false,
  qrAttached: false,
  overrideEnabled: false,
  override: {
    testMode: '',
    destinationA: '',
    destinationB: '',
    utms: {},
  },
  ...overrides,
});

const makeFamily = (overrides = {}) => ({
  id: 'family-summer',
  name: 'Summer Escape',
  partner: 'Basis',
  destinationA: 'https://example.com/landing-a?ref=nav#offers',
  destinationB: 'https://example.com/landing-b',
  testMode: 'ab',
  utms: {
    utm_id: 'b',
    utm_source: 'Basis DSP',
    utm_medium: 'Display',
    utm_campaign: 'Summer Escape',
    utm_content: 'Hero',
    utm_term: 'Prospecting',
  },
  renditions: [makeRendition()],
  ...overrides,
});

module.exports = function runCampaignCreativeTrackerTests({ assert }) {
  assert(core && typeof core === 'object', 'campaign creative tracker core missing');
  [
    'normalizeValue',
    'buildTaggedUrl',
    'getEffectiveConfig',
    'buildLinkVariants',
    'validateRendition',
    'campaignToExportRows',
    'campaignSummary',
  ].forEach((name) => {
    assert(typeof core[name] === 'function', `${name} export missing`);
  });

  assert(
    core.normalizeValue('  Summer / Caf\u00e9 & Tea  ') === 'summer_caf\u00e9_tea',
    'normalizeValue should emit lowercase Unicode-safe snake case',
  );

  const encoded = core.buildTaggedUrl(
    'https://example.com/\u00fcber uns?ref=hello%20world&UTM_Source=old#r\u00e9sum\u00e9',
    {
      utm_source: 'Partner & Co.',
      utm_medium: 'Display / Native',
      utm_campaign: 'Caf\u00e9 Launch',
    },
  );
  const encodedUrl = new URL(encoded);
  assert(encodedUrl.pathname === '/%C3%BCber%20uns', 'URL path should be strictly encoded');
  assert(encodedUrl.searchParams.get('ref') === 'hello world', 'existing query params should be preserved');
  assert(encodedUrl.searchParams.get('utm_source') === 'partner_co', 'UTM source should be normalized and encoded');
  assert(encodedUrl.searchParams.get('utm_medium') === 'display_native', 'UTM medium should be normalized');
  assert(encodedUrl.searchParams.get('utm_campaign') === 'caf\u00e9_launch', 'Unicode UTM values should round-trip');
  assert(!encodedUrl.searchParams.has('UTM_Source'), 'case-insensitive duplicate UTM keys should be replaced');
  assert(encodedUrl.hash === '#r%C3%A9sum%C3%A9', 'URL fragment should be preserved and encoded');

  const ignoredOverrideRendition = makeRendition({
    overrideEnabled: false,
    override: {
      testMode: 'single',
      destinationA: 'https://override.example/single',
      destinationB: '',
      utms: { utm_content: 'Override Hero' },
    },
  });
  const inherited = core.getEffectiveConfig(makeFamily(), ignoredOverrideRendition);
  assert(inherited.testMode === 'ab', 'disabled override should inherit family test mode');
  assert(inherited.destinationA.includes('landing-a'), 'disabled override should inherit family destination');
  assert(inherited.utms.utm_content === 'Hero', 'disabled override should inherit family UTMs');

  const activeOverrideRendition = makeRendition({
    overrideEnabled: true,
    override: {
      testMode: 'single',
      destinationA: 'https://override.example/single',
      destinationB: '',
      utms: { utm_content: 'Override Hero' },
    },
  });
  const overridden = core.getEffectiveConfig(makeFamily(), activeOverrideRendition);
  assert(overridden.testMode === 'single', 'rendition test mode should override the family');
  assert(overridden.destinationA === 'https://override.example/single', 'rendition destination should win');
  assert(overridden.destinationB === 'https://example.com/landing-b', 'blank override should inherit destination B');
  assert(overridden.utms.utm_content === 'Override Hero', 'rendition UTM should win');
  assert(overridden.utms.utm_source === 'Basis DSP', 'non-overridden UTM should remain inherited');

  const abVariants = core.buildLinkVariants(makeFamily(), makeRendition());
  assert(abVariants.length === 2, 'A/B mode should create two links');
  assert(abVariants[0].variantId === 'family-summer:rendition-display:a', 'A link should have stable identity');
  assert(abVariants[1].variantId === 'family-summer:rendition-display:b', 'B link should have stable identity');

  const singleVariants = core.buildLinkVariants(makeFamily(), activeOverrideRendition);
  assert(singleVariants.length === 1, 'Single mode should create one link');
  assert(singleVariants[0].variant === 'single', 'Single link should be labeled as a single variant');
  assert(singleVariants[0].url.startsWith('https://override.example/single?'), 'Single override should use its destination');

  const naRendition = makeRendition({
    id: 'rendition-ctv',
    name: 'CTV 30s',
    format: 'ctv',
    duration: '30s',
    qrAttached: true,
    overrideEnabled: true,
    override: {
      testMode: 'na',
      destinationA: '',
      destinationB: '',
      utms: {},
    },
  });
  assert(core.buildLinkVariants(makeFamily(), naRendition).length === 0, 'N/A mode should create no links');

  let invalidDestinationRejected = false;
  try {
    core.buildLinkVariants(
      makeFamily({ destinationB: 'javascript:alert(1)' }),
      makeRendition(),
    );
  } catch {
    invalidDestinationRejected = true;
  }
  assert(invalidDestinationRejected, 'link generation should reject non-HTTP destinations');

  const invalidInteractive = makeRendition({
    format: 'interactive',
    previewUrl: '',
    clickTagValid: false,
  });
  const invalidInteractiveResult = core.validateRendition(makeFamily(), invalidInteractive);
  assert(!invalidInteractiveResult.valid, 'interactive rendition should require QA metadata');
  assert(
    invalidInteractiveResult.issues.some((issue) => issue.code === 'interactive_preview_required'),
    'interactive rendition should require a preview URL',
  );
  assert(
    invalidInteractiveResult.issues.some((issue) => issue.code === 'interactive_click_tag_required'),
    'interactive rendition should require a valid click tag',
  );

  const validInteractive = makeRendition({
    format: 'interactive',
    previewUrl: 'https://preview.example/creative',
    clickTagValid: true,
  });
  assert(core.validateRendition(makeFamily(), validInteractive).valid, 'validated interactive rendition should pass');
  const missingUtmResult = core.validateRendition(
    makeFamily({ utms: { ...makeFamily().utms, utm_content: '' } }),
    makeRendition(),
  );
  assert(
    missingUtmResult.issues.some((issue) => issue.code === 'missing_utm_content'),
    'rendition QA should require every governed UTM selection',
  );

  const controlledDictionaries = Object.fromEntries(core.UTM_KEYS.map((key) => [
    key,
    [{ value: makeFamily().utms[key] }],
  ]));
  const unapprovedUtmResult = core.validateRendition(
    makeFamily({ utms: { ...makeFamily().utms, utm_source: 'Unknown Partner' } }),
    makeRendition(),
    controlledDictionaries,
  );
  assert(
    unapprovedUtmResult.issues.some((issue) => issue.code === 'unapproved_utm_source'),
    'rendition QA should reject values outside the controlled dictionary',
  );

  const missingAssetResult = core.validateRendition(
    makeFamily(),
    makeRendition({ assetSrc: '', fileName: '' }),
  );
  assert(
    missingAssetResult.issues.some((issue) => issue.code === 'rendition_asset_required'),
    'rendition QA should require linked creative metadata',
  );


  const missingQrResult = core.validateRendition(
    makeFamily(),
    { ...naRendition, qrAttached: false },
  );
  assert(
    missingQrResult.issues.some((issue) => issue.code === 'ctv_qr_required'),
    'CTV N/A rendition should require a QR attachment',
  );
  assert(core.validateRendition(makeFamily(), naRendition).valid, 'CTV N/A rendition with QR should pass');

  const exportFamily = makeFamily({
    renditions: [makeRendition(), naRendition],
  });
  const rows = core.campaignToExportRows(exportFamily);
  assert(rows.length === 3, 'export should include A, B, and an N/A rendition identity row');
  assert(rows[0].family_id === 'family-summer', 'export should retain family identity');
  assert(rows[0].rendition_id === 'rendition-display', 'export should retain rendition identity');
  assert(rows[0].variant_id === 'family-summer:rendition-display:a', 'export should retain variant identity');
  assert(rows[0].utm_source === 'basis_dsp', 'export should contain normalized effective UTMs');
  assert(rows[2].variant_id === 'family-summer:rendition-ctv:na', 'N/A export row should have stable identity');
  assert(rows[2].qr_attached === true, 'N/A CTV export should retain QR state');

  assert(
    core.protectSpreadsheetValue('=HYPERLINK("https://bad.example")').startsWith("'="),
    'spreadsheet formula cells should be neutralized',
  );
  const csv = core.exportRowsToCsv([{ family_name: '=2+2', rendition_id: 'safe' }]);
  assert(csv.includes("'=2+2"), 'CSV export should neutralize formula-like cells');

  const summary = core.campaignSummary(exportFamily);
  assert(summary.familyCount === 1, 'summary should count families');
  assert(summary.renditionCount === 2, 'summary should count renditions');
  assert(summary.linkCount === 2, 'summary should count generated links');
  assert(summary.abPairCount === 1, 'summary should count A/B pairs');
  assert(summary.naCount === 1, 'summary should count N/A renditions');
  assert(summary.qrCount === 1, 'summary should count QR attachments');
  assert(summary.errorRenditionCount === 0, 'valid sample should have no rendition errors');
};
