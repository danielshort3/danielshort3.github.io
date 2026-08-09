(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
    return;
  }
  root.CampaignCreativeTrackerCore = factory();
}(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  'use strict';

  const TEST_MODES = Object.freeze(['ab', 'single', 'na']);
  const RENDITION_FORMATS = Object.freeze([
    'static',
    'animated',
    'interactive',
    'video',
    'ctv',
  ]);
  const UTM_KEYS = Object.freeze([
    'utm_id',
    'utm_source',
    'utm_medium',
    'utm_campaign',
    'utm_content',
    'utm_term',
  ]);
  const FORMULA_PREFIX = /^[\t\r\n ]*[=+\-@]/;

  const own = (value, key) => Object.prototype.hasOwnProperty.call(value || {}, key);

  const asText = (value) => String(value ?? '').trim();

  const hasOverrideValue = (value) => (
    value !== null
    && value !== undefined
    && asText(value) !== ''
  );

  const normalizeValue = (value) => asText(value)
    .normalize('NFKC')
    .toLocaleLowerCase('en-US')
    .replace(/[^\p{L}\p{N}]+/gu, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');

  const orderedUtmKeys = (utms) => {
    const supplied = Object.keys(utms || {})
      .filter((key) => /^utm_[a-z0-9_-]+$/i.test(key));
    const extras = supplied
      .filter((key) => !UTM_KEYS.includes(key.toLocaleLowerCase('en-US')))
      .sort((left, right) => left.localeCompare(right));
    return [...UTM_KEYS, ...extras];
  };

  const normalizeUtms = (utms) => {
    const normalized = {};
    orderedUtmKeys(utms).forEach((key) => {
      if (!own(utms, key)) return;
      const value = normalizeValue(utms[key]);
      if (!value) return;
      normalized[key.toLocaleLowerCase('en-US')] = value;
    });
    return normalized;
  };

  const parseHttpUrl = (value, label = 'Destination URL') => {
    const text = asText(value);
    if (!text) throw new Error(`${label} is required.`);
    let parsed;
    try {
      parsed = new URL(text);
    } catch {
      throw new Error(`${label} is invalid.`);
    }
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      throw new Error(`${label} must use HTTP or HTTPS.`);
    }
    return parsed;
  };

  const buildTaggedUrl = (destination, utms = {}) => {
    const url = parseHttpUrl(destination);
    const normalized = normalizeUtms(utms);

    Object.entries(normalized).forEach(([key, value]) => {
      const duplicates = [];
      url.searchParams.forEach((_currentValue, currentKey) => {
        if (currentKey.toLocaleLowerCase('en-US') === key) duplicates.push(currentKey);
      });
      duplicates.forEach((duplicate) => url.searchParams.delete(duplicate));
      url.searchParams.set(key, value);
    });

    return url.toString();
  };

  const getEffectiveConfig = (family = {}, rendition = {}) => {
    const familyUtms = family.utms && typeof family.utms === 'object'
      ? family.utms
      : {};
    const inherited = {
      testMode: asText(family.testMode || 'single').toLocaleLowerCase('en-US'),
      destinationA: asText(family.destinationA),
      destinationB: asText(family.destinationB),
      utms: { ...familyUtms },
    };

    if (!rendition.overrideEnabled) return inherited;

    const override = rendition.override && typeof rendition.override === 'object'
      ? rendition.override
      : {};
    const overrideUtms = override.utms && typeof override.utms === 'object'
      ? override.utms
      : {};
    const utms = { ...familyUtms };

    Object.keys(overrideUtms).forEach((key) => {
      if (hasOverrideValue(overrideUtms[key])) utms[key] = overrideUtms[key];
    });

    return {
      testMode: hasOverrideValue(override.testMode)
        ? asText(override.testMode).toLocaleLowerCase('en-US')
        : inherited.testMode,
      destinationA: hasOverrideValue(override.destinationA)
        ? asText(override.destinationA)
        : inherited.destinationA,
      destinationB: hasOverrideValue(override.destinationB)
        ? asText(override.destinationB)
        : inherited.destinationB,
      utms,
    };
  };

  const variantIdentity = (family, rendition, variant) => {
    const familyId = asText(family?.id) || 'family';
    const renditionId = asText(rendition?.id) || 'rendition';
    return `${familyId}:${renditionId}:${variant}`;
  };

  const buildLinkVariants = (family, rendition) => {
    const config = getEffectiveConfig(family, rendition);
    if (!TEST_MODES.includes(config.testMode)) {
      throw new Error(`Unsupported test mode "${config.testMode}".`);
    }
    if (config.testMode === 'na') return [];

    const variants = config.testMode === 'ab'
      ? [
          { key: 'a', label: 'A', destination: config.destinationA },
          { key: 'b', label: 'B', destination: config.destinationB },
        ]
      : [{ key: 'single', label: 'Single', destination: config.destinationA }];

    return variants.map((variant) => {
      const destination = parseHttpUrl(
        variant.destination,
        config.testMode === 'ab' ? `Destination ${variant.label}` : 'Destination URL',
      ).toString();
      return {
        variantId: variantIdentity(family, rendition, variant.key),
        variant: variant.key,
        label: variant.label,
        destination,
        url: buildTaggedUrl(destination, config.utms),
      };
    });
  };

  const makeIssue = (code, message, field) => ({
    code,
    severity: 'error',
    message,
    ...(field ? { field } : {}),
  });

  const dictionaryHasValue = (dictionaries, key, value) => {
    if (!dictionaries || !Array.isArray(dictionaries[key])) return true;
    return dictionaries[key].some((option) => asText(option?.value) === value);
  };

  const validateRendition = (family, rendition = {}, dictionaries = null) => {
    const issues = [];
    const config = getEffectiveConfig(family, rendition);
    const format = asText(rendition.format).toLocaleLowerCase('en-US');
    if (!asText(rendition.id)) {
      issues.push(makeIssue(
        'rendition_id_required',
        'Rendition ID is required.',
        'id',
      ));
    }
    if (!asText(rendition.name)) {
      issues.push(makeIssue(
        'rendition_name_required',
        'Rendition name is required.',
        'name',
      ));
    }
    if (!format) {
      issues.push(makeIssue(
        'rendition_format_required',
        'Choose a rendition format.',
        'format',
      ));
    } else if (!RENDITION_FORMATS.includes(format)) {
      issues.push(makeIssue(
        'invalid_rendition_format',
        `Unsupported rendition format "${format}".`,
        'format',
      ));
    }
    if (!asText(rendition.assetSrc) && !asText(rendition.fileName)) {
      issues.push(makeIssue(
        'rendition_asset_required',
        'Attach a creative asset or retain its source filename.',
        'assetSrc',
      ));
    }
    if (!(Number(rendition.width) > 0) || !(Number(rendition.height) > 0)) {
      issues.push(makeIssue(
        'rendition_dimensions_required',
        'Enter positive width and height values.',
        'dimensions',
      ));
    }
    if (['video', 'ctv'].includes(format) && !asText(rendition.duration)) {
      issues.push(makeIssue(
        'rendition_duration_required',
        'Enter the video duration.',
        'duration',
      ));
    }
    UTM_KEYS.forEach((key) => {
      const value = asText(config.utms?.[key]);
      if (!value) {
        issues.push(makeIssue(`missing_${key}`, `${key} must use an approved value.`, key));
      } else if (!dictionaryHasValue(dictionaries, key, value)) {
        issues.push(makeIssue(`unapproved_${key}`, `${key} is not in the controlled dictionary.`, key));
      }
    });

    if (!TEST_MODES.includes(config.testMode)) {
      issues.push(makeIssue(
        'invalid_test_mode',
        'Choose A/B, Single link, or Not applicable.',
        'testMode',
      ));
    } else if (config.testMode !== 'na') {
      try {
        parseHttpUrl(
          config.destinationA,
          config.testMode === 'ab' ? 'Destination A' : 'Destination URL',
        );
      } catch (error) {
        issues.push(makeIssue(
          'invalid_destination_a',
          error instanceof Error ? error.message : 'Destination A is invalid.',
          'destinationA',
        ));
      }

      if (config.testMode === 'ab') {
        try {
          parseHttpUrl(config.destinationB, 'Destination B');
        } catch (error) {
          issues.push(makeIssue(
            'invalid_destination_b',
            error instanceof Error ? error.message : 'Destination B is invalid.',
            'destinationB',
          ));
        }
      }
    }

    if (format === 'interactive') {
      try {
        parseHttpUrl(rendition.previewUrl, 'Interactive preview URL');
      } catch (error) {
        issues.push(makeIssue(
          'interactive_preview_required',
          error instanceof Error ? error.message : 'Interactive preview URL is required.',
          'previewUrl',
        ));
      }
      if (rendition.clickTagValid !== true) {
        issues.push(makeIssue(
          'interactive_click_tag_required',
          'Confirm that the interactive rendition has a valid click tag.',
          'clickTagValid',
        ));
      }
    }

    if (format === 'ctv' && config.testMode === 'na' && rendition.qrAttached !== true) {
      issues.push(makeIssue(
        'ctv_qr_required',
        'Attach a QR code when a CTV rendition does not use a clickable link.',
        'qrAttached',
      ));
    }

    return {
      valid: issues.every((issue) => issue.severity !== 'error'),
      issues,
      errors: issues.filter((issue) => issue.severity === 'error'),
      warnings: issues.filter((issue) => issue.severity === 'warning'),
    };
  };

  const expectedVariantDescriptors = (family, rendition, config) => {
    if (config.testMode === 'ab') {
      return [
        { key: 'a', label: 'A', destination: config.destinationA },
        { key: 'b', label: 'B', destination: config.destinationB },
      ];
    }
    if (config.testMode === 'single') {
      return [{ key: 'single', label: 'Single', destination: config.destinationA }];
    }
    return [{ key: 'na', label: 'N/A', destination: '' }];
  };

  const inputFamilies = (input) => {
    if (Array.isArray(input)) return input;
    if (Array.isArray(input?.families)) return input.families;
    return input && typeof input === 'object' ? [input] : [];
  };

  const campaignToExportRows = (input, dictionaries = null) => inputFamilies(input).flatMap((family) => {
    const renditions = Array.isArray(family.renditions) ? family.renditions : [];
    return renditions.flatMap((rendition) => {
      const config = getEffectiveConfig(family, rendition);
      const utms = normalizeUtms(config.utms);
      const validation = validateRendition(family, rendition, dictionaries);
      let generated = [];
      try {
        generated = buildLinkVariants(family, rendition);
      } catch {
        generated = [];
      }
      const generatedByVariant = new Map(generated.map((variant) => [variant.variant, variant]));

      return expectedVariantDescriptors(family, rendition, config).map((descriptor) => {
        const variant = generatedByVariant.get(descriptor.key);
        return {
          family_id: asText(family.id),
          family_name: asText(family.name),
          partner: asText(family.partner),
          rendition_id: asText(rendition.id),
          rendition_name: asText(rendition.name),
          width: Number(rendition.width) || '',
          height: Number(rendition.height) || '',
          format: asText(rendition.format).toLocaleLowerCase('en-US'),
          duration: asText(rendition.duration),
          asset_src: asText(rendition.assetSrc),
          preview_url: asText(rendition.previewUrl),
          click_tag_valid: rendition.clickTagValid === true,
          qr_attached: rendition.qrAttached === true,
          override_enabled: rendition.overrideEnabled === true,
          test_mode: config.testMode,
          variant_id: variantIdentity(family, rendition, descriptor.key),
          variant: descriptor.key,
          destination_url: variant?.destination || asText(descriptor.destination),
          generated_url: variant?.url || '',
          ...Object.fromEntries(UTM_KEYS.map((key) => [key, utms[key] || ''])),
          validation_status: validation.valid ? 'valid' : 'error',
          validation_issues: validation.issues.map((issue) => issue.message).join(' | '),
        };
      });
    });
  });

  const protectSpreadsheetValue = (value) => {
    const text = value === null || value === undefined ? '' : String(value);
    return FORMULA_PREFIX.test(text) ? `'${text}` : text;
  };

  const escapeCsvCell = (value) => {
    const text = protectSpreadsheetValue(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };

  const exportRowsToCsv = (rows) => {
    const values = Array.isArray(rows) ? rows : [];
    if (!values.length) return '';
    const headers = Array.from(values.reduce((keys, row) => {
      Object.keys(row || {}).forEach((key) => keys.add(key));
      return keys;
    }, new Set()));
    const lines = [headers.map(escapeCsvCell).join(',')];
    values.forEach((row) => {
      lines.push(headers.map((header) => escapeCsvCell(row?.[header])).join(','));
    });
    return `\uFEFF${lines.join('\r\n')}`;
  };

  const campaignSummary = (input, dictionaries = null) => {
    const families = inputFamilies(input);
    const summary = {
      familyCount: families.length,
      renditionCount: 0,
      linkCount: 0,
      abPairCount: 0,
      abTestCount: 0,
      singleLinkCount: 0,
      naCount: 0,
      qrCount: 0,
      validRenditionCount: 0,
      errorRenditionCount: 0,
      warningRenditionCount: 0,
    };

    families.forEach((family) => {
      const renditions = Array.isArray(family.renditions) ? family.renditions : [];
      renditions.forEach((rendition) => {
        summary.renditionCount += 1;
        if (rendition.qrAttached === true) summary.qrCount += 1;

        const config = getEffectiveConfig(family, rendition);
        if (config.testMode === 'ab') {
          summary.abPairCount += 1;
          summary.abTestCount += 1;
        } else if (config.testMode === 'single') {
          summary.singleLinkCount += 1;
        } else if (config.testMode === 'na') {
          summary.naCount += 1;
        }

        const validation = validateRendition(family, rendition, dictionaries);
        if (validation.errors.length) summary.errorRenditionCount += 1;
        else if (validation.warnings.length) summary.warningRenditionCount += 1;
        else summary.validRenditionCount += 1;

        try {
          summary.linkCount += buildLinkVariants(family, rendition).length;
        } catch {
          // Invalid destinations are represented by validation counts, not links.
        }
      });
    });

    return summary;
  };

  return Object.freeze({
    TEST_MODES,
    RENDITION_FORMATS,
    UTM_KEYS,
    normalizeValue,
    buildTaggedUrl,
    getEffectiveConfig,
    buildLinkVariants,
    validateRendition,
    campaignToExportRows,
    campaignSummary,
    protectSpreadsheetValue,
    exportRowsToCsv,
  });
}));
