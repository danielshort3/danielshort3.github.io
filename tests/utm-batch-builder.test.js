const core = require('../src/utm-batch-builder/core.js');

const makeField = (overrides = {}) => ({
  mode: 'single',
  single: '',
  list: '',
  csvColumnIndex: null,
  ...overrides,
});

module.exports = function runUtmBatchBuilderTests({ assert }) {
  // URL builder
  {
    const url = core.buildUrl('https://example.com?x=1#frag', {
      utm_source: 'Google Search',
      utm_medium: 'cpc',
    }, { overrideExisting: true });
    assert(url.startsWith('https://example.com?'), 'buildUrl should preserve base (no extra slash)');
    assert(url.includes('x=1'), 'buildUrl should preserve existing query params');
    assert(url.includes('utm_source=Google+Search'), 'buildUrl should URL-encode values');
    assert(url.includes('#frag'), 'buildUrl should preserve fragments');
  }

  {
    const url = core.buildUrl('https://example.com?utm_source=keep&utm_medium=cpc', {
      utm_source: 'new',
    }, { overrideExisting: false });
    assert(url.includes('utm_source=keep'), 'overrideExisting=false should preserve existing params');
    assert(!url.includes('utm_source=new'), 'overrideExisting=false should not overwrite existing params');
  }

  // Combination generator: cartesian order + exclude rules
  {
    const config = {
      mode: core.COMBINATION_MODE.CARTESIAN,
      overrideExisting: true,
      normalization: { lowercase: true, spaces: 'underscore', stripSpecial: false, slugify: false },
      excludeRulesText: 'utm_source=meta & utm_medium=cpc',
      csvText: '',
      baseUrl: makeField({ mode: 'list', list: 'https://a.example\nhttps://b.example' }),
      utm: {
        source: makeField({ mode: 'list', list: 'google\nmeta' }),
        medium: makeField({ mode: 'single', single: 'cpc' }),
        campaign: makeField({ mode: 'single', single: 'sale' }),
        content: makeField(),
        term: makeField(),
      },
      customParams: [],
    };

    const { errors, resolved } = core.resolveAndValidateConfig(config);
    assert(errors.length === 0 && resolved, 'cartesian config should validate');
    assert(core.estimateTotalRows(resolved) === 4, 'estimateTotalRows should include all combinations (pre-exclude)');

    const rows = Array.from(core.generateRows(resolved));
    assert(rows.length === 2, 'exclude rule should filter out meta+cpc rows');
    assert(rows[0].finalUrl.includes('utm_source=google'), 'first row should use first utm_source value');
    assert(rows[0].finalUrl.includes('utm_campaign=sale'), 'required fields should be present');
  }

  // Zip mode: mismatched lengths should error
  {
    const config = {
      mode: core.COMBINATION_MODE.ZIP,
      overrideExisting: true,
      normalization: { lowercase: true, spaces: 'underscore', stripSpecial: false, slugify: false },
      excludeRulesText: '',
      csvText: '',
      baseUrl: makeField({ mode: 'single', single: 'https://example.com/landing' }),
      utm: {
        source: makeField({ mode: 'list', list: 'google\nmeta' }),
        medium: makeField({ mode: 'single', single: 'cpc' }),
        campaign: makeField({ mode: 'list', list: 'c1\nc2\nc3' }),
        content: makeField(),
        term: makeField(),
      },
      customParams: [],
    };

    const { errors } = core.resolveAndValidateConfig(config);
    assert(errors.length > 0, 'zip config should error on mismatched list lengths');
  }

  // Template+Rows: blank CSV cells fall back to defaults
  {
    const csvText = [
      'base_url,source,campaign',
      'https://example.com/a,google,sale_a',
      'https://example.com/b,,sale_b',
    ].join('\n');

    const config = {
      mode: core.COMBINATION_MODE.TEMPLATE_ROWS,
      overrideExisting: true,
      normalization: { lowercase: true, spaces: 'underscore', stripSpecial: false, slugify: false },
      excludeRulesText: '',
      csvText,
      baseUrl: makeField({ mode: 'csvColumn', single: '', csvColumnIndex: 0 }),
      utm: {
        source: makeField({ mode: 'csvColumn', single: 'google', csvColumnIndex: 1 }),
        medium: makeField({ mode: 'single', single: 'cpc' }),
        campaign: makeField({ mode: 'csvColumn', single: 'default', csvColumnIndex: 2 }),
        content: makeField(),
        term: makeField(),
      },
      customParams: [],
    };

    const { errors, resolved } = core.resolveAndValidateConfig(config);
    assert(errors.length === 0 && resolved, 'templateRows config should validate');
    assert(core.estimateTotalRows(resolved) === 2, 'templateRows estimate should equal rowCount');

    const rows = Array.from(core.generateRows(resolved));
    assert(rows.length === 2, 'templateRows should generate one URL per row');
    assert(rows[1].baseUrl === 'https://example.com/b', 'base URL should come from CSV');
    assert(rows[1].finalUrl.includes('utm_source=google'), 'blank override should fall back to default');
    assert(rows[1].finalUrl.includes('utm_campaign=sale_b'), 'campaign should map from CSV');
  }
};

