/* eslint-disable no-control-regex */
/* Core, dependency-free UTM generation helpers (Node + browser compatible). */

'use strict';

const INPUT_MODE = {
  SINGLE: 'single',
  LIST: 'list',
  CSV_COLUMN: 'csvColumn',
};

const COMBINATION_MODE = {
  CARTESIAN: 'cartesian',
  ZIP: 'zip',
  TEMPLATE_ROWS: 'templateRows',
  GROUPS: 'groups',
};

const DEFAULT_NORMALIZATION = {
  lowercase: true,
  spaces: 'underscore', // underscore | dash | none
  stripSpecial: false,
  slugify: false,
};

const stripBom = (value) => (value || '').replace(/^\uFEFF/, '');

const parseMultiline = (text) => stripBom(String(text || ''))
  .split(/\r?\n/)
  .map(line => line.trim())
  .filter(Boolean);

const parseMultilinePreserveEmpty = (text) => {
  const lines = stripBom(String(text || ''))
    .split(/\r?\n/)
    .map(line => line.trim());

  let start = 0;
  while (start < lines.length && lines[start] === '') start += 1;
  let end = lines.length;
  while (end > start && lines[end - 1] === '') end -= 1;

  return lines.slice(start, end);
};

const parseCsv = (text) => {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  const input = stripBom(String(text || ''));

  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    if (char === '"') {
      if (inQuotes && input[i + 1] === '"') {
        field += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      row.push(field);
      field = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && input[i + 1] === '\n') i += 1;
      row.push(field);
      if (row.some(cell => String(cell || '').trim() !== '')) rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }

  if (field.length || row.length) {
    row.push(field);
    if (row.some(cell => String(cell || '').trim() !== '')) rows.push(row);
  }

  return rows;
};

const getCsvHeaders = (csvRows) => (Array.isArray(csvRows) && csvRows.length ? csvRows[0] : []);
const getCsvDataRows = (csvRows) => (Array.isArray(csvRows) && csvRows.length ? csvRows.slice(1) : []);

const getCsvColumnValues = (csvRows, columnIndex) => {
  const idx = typeof columnIndex === 'number' ? columnIndex : -1;
  if (!Array.isArray(csvRows) || idx < 0) return [];
  return getCsvDataRows(csvRows).map(row => (row && row[idx] !== undefined ? String(row[idx]) : '').trim());
};

const isValidParamKey = (key) => /^[a-zA-Z0-9_-]+$/.test(String(key || '').trim());

const normalizeValue = (raw, opts = {}) => {
  const options = { ...DEFAULT_NORMALIZATION, ...(opts || {}) };
  const separator = options.spaces === 'dash' ? '-' : (options.spaces === 'underscore' ? '_' : '');
  let value = String(raw ?? '').trim();
  if (!value) return '';

  if (options.lowercase) value = value.toLowerCase();

  // Normalize accented chars for slugify/special stripping.
  if (options.slugify || options.stripSpecial) {
    try {
      value = value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
    } catch (_) {}
  }

  if (options.slugify) {
    value = value
      .replace(/[^a-z0-9]+/gi, separator || ' ')
      .replace(new RegExp(`${separator || ' '}+`, 'g'), separator || ' ')
      .replace(new RegExp(`^${separator}|${separator}$`, 'g'), '');
    return value;
  }

  if (options.stripSpecial) {
    value = value.replace(/[^a-z0-9 _-]+/gi, '');
  }

  if (separator) {
    value = value.replace(/\s+/g, separator);
    value = value.replace(new RegExp(`${separator}{2,}`, 'g'), separator);
    value = value.replace(new RegExp(`^${separator}|${separator}$`, 'g'), '');
  }

  return value;
};

const splitUrlParts = (url) => {
  const raw = String(url || '');
  const hashIndex = raw.indexOf('#');
  const beforeHash = hashIndex >= 0 ? raw.slice(0, hashIndex) : raw;
  const hash = hashIndex >= 0 ? raw.slice(hashIndex) : '';
  const queryIndex = beforeHash.indexOf('?');
  const base = queryIndex >= 0 ? beforeHash.slice(0, queryIndex) : beforeHash;
  const query = queryIndex >= 0 ? beforeHash.slice(queryIndex + 1) : '';
  return { base, query, hash };
};

const buildUrl = (baseUrl, params, options = {}) => {
  const overrideExisting = options.overrideExisting !== false;
  const { base, query, hash } = splitUrlParts(baseUrl);

  // Validate URL upfront; preserve the raw base/query/hash formatting in output.
  let parsed;
  try {
    parsed = new URL(baseUrl);
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    throw new Error(`Invalid URL: ${baseUrl} (${message})`);
  }
  if (!/^https?:$/.test(parsed.protocol)) {
    throw new Error(`Invalid URL protocol (must be http/https): ${baseUrl}`);
  }

  const searchParams = new URLSearchParams(query);
  const entries = Array.isArray(params)
    ? params
    : Object.entries(params || {}).map(([key, value]) => ({ key, value }));

  entries.forEach(({ key, value }) => {
    const k = String(key || '').trim();
    const v = String(value ?? '').trim();
    if (!k) return;
    if (v === '') return;
    if (overrideExisting) {
      searchParams.set(k, v);
    } else if (!searchParams.has(k)) {
      searchParams.append(k, v);
    }
  });

  const qs = searchParams.toString();
  return `${base}${qs ? `?${qs}` : ''}${hash}`;
};

const parseExcludeRules = (text) => {
  const rules = [];
  const lines = stripBom(String(text || '')).split(/\r?\n/);
  lines.forEach((raw, index) => {
    const line = raw.trim();
    if (!line) return;
    const parts = line.split(/\s*(?:&|,)\s*/g).filter(Boolean);
    const pairs = [];
    parts.forEach((part) => {
      const m = part.match(/^([^=]+)=(.*)$/);
      if (!m) {
        throw new Error(`Exclude rule line ${index + 1} is invalid (expected key=value): ${line}`);
      }
      const key = m[1].trim();
      const value = m[2].trim();
      if (!key) {
        throw new Error(`Exclude rule line ${index + 1} has an empty key: ${line}`);
      }
      pairs.push({ key, value });
    });
    if (pairs.length) rules.push(pairs);
  });
  return rules;
};

const matchesExcludeRules = (params, rules) => {
  if (!rules || !rules.length) return false;
  return rules.some(rule => rule.every(({ key, value }) => String(params[key] ?? '') === value));
};

const resolveAndValidateConfig = (config) => {
  const errors = [];
  const warnings = [];
  const mode = config?.mode || COMBINATION_MODE.CARTESIAN;
  const overrideExisting = config?.overrideExisting !== false;
  const normalization = { ...DEFAULT_NORMALIZATION, ...(config?.normalization || {}) };
  let excludeRules = [];

  const csvText = String(config?.csvText || '');
  const csvRows = csvText ? parseCsv(csvText) : null;
  const csvHeaders = csvRows ? getCsvHeaders(csvRows) : [];
  const csvRowCount = csvRows ? Math.max(0, csvRows.length - 1) : 0;
  const maxErrors = 25;
  const addError = (message) => {
    if (errors.length >= maxErrors) return;
    errors.push(message);
  };
  const addWarning = (message) => {
    warnings.push(message);
  };

  try {
    excludeRules = parseExcludeRules(config?.excludeRulesText || '');
  } catch (err) {
    addError(err.message || String(err));
  }

  const requireCsv = (label, columnIndex) => {
    if (!csvRows) {
      addError(`${label} is set to "From CSV column" but no CSV is loaded.`);
      return [];
    }
    const idx = typeof columnIndex === 'number' ? columnIndex : -1;
    if (idx < 0 || idx >= csvHeaders.length) {
      addError(`${label} has an invalid CSV column selection.`);
      return [];
    }
    return getCsvColumnValues(csvRows, idx);
  };

  const validateUrl = (url) => {
    const value = String(url || '').trim();
    if (!value) return { ok: false, message: 'Empty URL' };
    try {
      const parsed = new URL(value);
      if (!/^https?:$/.test(parsed.protocol)) return { ok: false, message: 'Must be http/https' };
      return { ok: true };
    } catch (_) {
      return { ok: false, message: 'Invalid URL' };
    }
  };

  const baseInput = config?.baseUrl || {};
  const utm = config?.utm || {};
  const customParams = Array.isArray(config?.customParams) ? config.customParams : [];

  const getValueAt = (values, index) => {
    if (!Array.isArray(values) || values.length === 0) return '';
    if (values.length === 1) return values[0];
    return values[index] ?? '';
  };

  if (mode === COMBINATION_MODE.CARTESIAN) {
    // Base URLs
    const baseMode = baseInput?.mode || INPUT_MODE.SINGLE;
    const baseUrls = (() => {
      if (baseMode === INPUT_MODE.LIST) return parseMultiline(baseInput?.list || '');
      if (baseMode === INPUT_MODE.CSV_COLUMN) return requireCsv('Base URL', baseInput?.csvColumnIndex).filter(Boolean);
      return [String(baseInput?.single ?? '').trim()].filter(Boolean);
    })();

    if (!baseUrls.length) addError('Base URL is required (provide at least one).');
    baseUrls.forEach((url, idx) => {
      const result = validateUrl(url);
      if (!result.ok) addError(`Base URL ${idx + 1} is invalid (${result.message}): ${url}`);
    });

    const resolveParamValues = (fieldInput, label) => {
      const inputMode = fieldInput?.mode || INPUT_MODE.SINGLE;
      if (inputMode === INPUT_MODE.LIST) {
        return parseMultiline(fieldInput?.list || '')
          .map(v => normalizeValue(v, normalization))
          .filter(Boolean);
      }
      if (inputMode === INPUT_MODE.CSV_COLUMN) {
        return requireCsv(label, fieldInput?.csvColumnIndex)
          .map(v => normalizeValue(v, normalization))
          .filter(Boolean);
      }
      const single = normalizeValue(fieldInput?.single ?? '', normalization);
      return single ? [single] : [];
    };

    const fields = [];
    const requiredFields = [
      { key: 'utm_source', input: utm.source },
      { key: 'utm_medium', input: utm.medium },
      { key: 'utm_campaign', input: utm.campaign },
    ];
    const optionalFields = [
      { key: 'utm_content', input: utm.content },
      { key: 'utm_term', input: utm.term },
    ];

    requiredFields.forEach(({ key, input }) => {
      const values = resolveParamValues(input, key);
      if (!values.length) addError(`${key} is required (provide at least one value).`);
      fields.push({ key, required: true, values });
    });
    optionalFields.forEach(({ key, input }) => {
      const values = resolveParamValues(input, key);
      fields.push({ key, required: false, values });
    });

    const customKeys = new Set();
    const resolvedCustomParams = [];
    customParams.forEach((entry) => {
      const key = String(entry?.key || '').trim();
      if (!key) return;
      if (!isValidParamKey(key)) {
        addError(`Custom param key "${key}" is invalid (use letters, numbers, _ or -).`);
        return;
      }
      if (customKeys.has(key)) {
        addError(`Custom param key "${key}" is duplicated.`);
        return;
      }
      if (['utm_source','utm_medium','utm_campaign','utm_content','utm_term'].includes(key)) {
        addError(`Custom param key "${key}" conflicts with a UTM field.`);
        return;
      }
      customKeys.add(key);
      const values = resolveParamValues(entry?.value, `Custom param "${key}"`);
      if (!values.length) addWarning(`Custom param "${key}" has no values (it will be omitted).`);
      resolvedCustomParams.push({ key, values });
    });

    return {
      errors,
      warnings,
      resolved: errors.length ? null : {
        mode,
        overrideExisting,
        excludeRules,
        baseUrls,
        fields,
        customParams: resolvedCustomParams,
      },
    };
  }

  if (mode === COMBINATION_MODE.ZIP) {
    const baseMode = baseInput?.mode || INPUT_MODE.SINGLE;
    const baseUrls = (() => {
      if (baseMode === INPUT_MODE.LIST) return parseMultilinePreserveEmpty(baseInput?.list || '');
      if (baseMode === INPUT_MODE.CSV_COLUMN) return requireCsv('Base URL', baseInput?.csvColumnIndex);
      return [String(baseInput?.single ?? '').trim()];
    })();

    const resolveParamValues = (fieldInput, label) => {
      const inputMode = fieldInput?.mode || INPUT_MODE.SINGLE;
      if (inputMode === INPUT_MODE.LIST) {
        return parseMultilinePreserveEmpty(fieldInput?.list || '').map(v => normalizeValue(v, normalization));
      }
      if (inputMode === INPUT_MODE.CSV_COLUMN) {
        return requireCsv(label, fieldInput?.csvColumnIndex).map(v => normalizeValue(v, normalization));
      }
      return [normalizeValue(fieldInput?.single ?? '', normalization)];
    };

    const fields = [];
    const requiredFields = [
      { key: 'utm_source', input: utm.source },
      { key: 'utm_medium', input: utm.medium },
      { key: 'utm_campaign', input: utm.campaign },
    ];
    const optionalFields = [
      { key: 'utm_content', input: utm.content },
      { key: 'utm_term', input: utm.term },
    ];

    requiredFields.forEach(({ key, input }) => {
      const values = resolveParamValues(input, key);
      fields.push({ key, required: true, values });
    });
    optionalFields.forEach(({ key, input }) => {
      const values = resolveParamValues(input, key);
      fields.push({ key, required: false, values });
    });

    const customKeys = new Set();
    const resolvedCustomParams = [];
    customParams.forEach((entry) => {
      const key = String(entry?.key || '').trim();
      if (!key) return;
      if (!isValidParamKey(key)) {
        addError(`Custom param key "${key}" is invalid (use letters, numbers, _ or -).`);
        return;
      }
      if (customKeys.has(key)) {
        addError(`Custom param key "${key}" is duplicated.`);
        return;
      }
      if (['utm_source','utm_medium','utm_campaign','utm_content','utm_term'].includes(key)) {
        addError(`Custom param key "${key}" conflicts with a UTM field.`);
        return;
      }
      customKeys.add(key);
      const values = resolveParamValues(entry?.value, `Custom param "${key}"`);
      resolvedCustomParams.push({ key, values });
    });

    const allLengths = [
      baseUrls.length,
      ...fields.map(f => f.values.length),
      ...resolvedCustomParams.map(p => p.values.length),
    ].map(len => (len > 1 ? len : 1));
    const rowCount = Math.max(1, ...allLengths);

    const checkLengths = (label, values, required = false) => {
      if (!Array.isArray(values) || values.length === 0) {
        if (required) addError(`${label} is required (provide at least one value).`);
        return;
      }
      if (values.length !== 1 && values.length !== rowCount) {
        addError(`${label} has ${values.length} values but must be length 1 or ${rowCount} in ZIP mode.`);
      }
    };

    checkLengths('Base URL', baseUrls, true);
    fields.forEach(f => checkLengths(f.key, f.values, f.required));
    resolvedCustomParams.forEach(p => checkLengths(`Custom param "${p.key}"`, p.values, false));

    // Row-level checks for required fields and URLs (cap errors).
    for (let i = 0; i < rowCount && errors.length < maxErrors; i++) {
      const baseUrl = String(getValueAt(baseUrls, i) || '').trim();
      const baseCheck = validateUrl(baseUrl);
      if (!baseCheck.ok) {
        addError(`Base URL row ${i + 1} is invalid (${baseCheck.message}): ${baseUrl || '(empty)'}`);
      }
      fields.filter(f => f.required).forEach((field) => {
        const value = String(getValueAt(field.values, i) ?? '').trim();
        if (!value) addError(`${field.key} row ${i + 1} is empty.`);
      });
    }

    return {
      errors,
      warnings,
      resolved: errors.length ? null : {
        mode,
        overrideExisting,
        excludeRules,
        rowCount,
        baseUrls,
        fields,
        customParams: resolvedCustomParams,
      },
    };
  }

  if (mode === COMBINATION_MODE.GROUPS) {
    const relationshipGroups = Array.isArray(config?.relationshipGroups) ? config.relationshipGroups : [];

    const fields = [];
    const requiredFields = [
      { key: 'utm_source', input: utm.source },
      { key: 'utm_medium', input: utm.medium },
      { key: 'utm_campaign', input: utm.campaign },
    ];
    const optionalFields = [
      { key: 'utm_content', input: utm.content },
      { key: 'utm_term', input: utm.term },
    ];

    requiredFields.forEach(({ key, input }) => fields.push({ key, required: true, input }));
    optionalFields.forEach(({ key, input }) => fields.push({ key, required: false, input }));

    const customKeys = new Set();
    const customInputsByKey = Object.create(null);
    const customParamsResolved = [];
    customParams.forEach((entry) => {
      const key = String(entry?.key || '').trim();
      if (!key) return;
      if (!isValidParamKey(key)) {
        addError(`Custom param key "${key}" is invalid (use letters, numbers, _ or -).`);
        return;
      }
      if (customKeys.has(key)) {
        addError(`Custom param key "${key}" is duplicated.`);
        return;
      }
      if (['utm_source','utm_medium','utm_campaign','utm_content','utm_term'].includes(key)) {
        addError(`Custom param key "${key}" conflicts with a UTM field.`);
        return;
      }
      customKeys.add(key);
      customInputsByKey[key] = entry?.value || {};
      customParamsResolved.push({ key });
    });

    const requiredKeys = new Set(['base_url', 'utm_source', 'utm_medium', 'utm_campaign']);
    const allowedKeys = new Set([
      'base_url',
      ...fields.map(f => f.key),
      ...customParamsResolved.map(p => p.key),
    ]);

    const zipGroups = [];
    const assigned = new Set();
    relationshipGroups.forEach((group, groupIndex) => {
      const rawKeys = Array.isArray(group?.keys) ? group.keys : [];
      const keys = [];
      rawKeys.forEach((raw) => {
        const key = String(raw || '').trim();
        if (!key) return;
        if (!allowedKeys.has(key)) return;
        if (assigned.has(key)) {
          addError(`Relationship groups: "${key}" is assigned to multiple zip groups.`);
          return;
        }
        assigned.add(key);
        keys.push(key);
      });
      if (!keys.length) return;
      const name = String(group?.name || '').trim() || `Zip group ${zipGroups.length + 1}`;
      zipGroups.push({ id: String(group?.id || groupIndex), name, keys });
    });

    const isRowAlignedKey = (key) => assigned.has(key);

    const baseMode = baseInput?.mode || INPUT_MODE.SINGLE;
    const baseUrls = (() => {
      if (baseMode === INPUT_MODE.LIST) {
        const list = baseInput?.list || '';
        return (isRowAlignedKey('base_url') ? parseMultilinePreserveEmpty(list) : parseMultiline(list))
          .map(v => String(v ?? '').trim());
      }
      if (baseMode === INPUT_MODE.CSV_COLUMN) {
        const values = requireCsv('Base URL', baseInput?.csvColumnIndex).map(v => String(v ?? '').trim());
        return isRowAlignedKey('base_url') ? values : values.filter(Boolean);
      }
      return [String(baseInput?.single ?? '').trim()];
    })();

    const resolveParamValues = (fieldInput, label, rowAligned) => {
      const inputMode = fieldInput?.mode || INPUT_MODE.SINGLE;
      if (inputMode === INPUT_MODE.LIST) {
        const rawValues = rowAligned
          ? parseMultilinePreserveEmpty(fieldInput?.list || '')
          : parseMultiline(fieldInput?.list || '');
        const normalized = rawValues.map(v => normalizeValue(v, normalization));
        return rowAligned ? normalized : normalized.filter(Boolean);
      }
      if (inputMode === INPUT_MODE.CSV_COLUMN) {
        const rawValues = requireCsv(label, fieldInput?.csvColumnIndex);
        const normalized = rawValues.map(v => normalizeValue(v, normalization));
        return rowAligned ? normalized : normalized.filter(Boolean);
      }
      const single = normalizeValue(fieldInput?.single ?? '', normalization);
      if (rowAligned) return [single];
      return single ? [single] : [];
    };

    const resolvedFields = fields.map((field) => {
      const values = resolveParamValues(field.input, field.key, isRowAlignedKey(field.key));
      if (field.required && !values.length) addError(`${field.key} is required (provide at least one value).`);
      return { key: field.key, required: field.required, values };
    });

    const resolvedCustomParams = customParamsResolved.map((entry) => {
      const values = resolveParamValues(customInputsByKey[entry.key], `Custom param "${entry.key}"`, isRowAlignedKey(entry.key));
      return { key: entry.key, values };
    });

    const valuesByKey = Object.create(null);
    valuesByKey.base_url = baseUrls;
    resolvedFields.forEach((f) => { valuesByKey[f.key] = f.values; });
    resolvedCustomParams.forEach((p) => { valuesByKey[p.key] = p.values; });

    const allKeysInOrder = [
      'base_url',
      ...resolvedFields.map(f => f.key),
      ...resolvedCustomParams.map(p => p.key),
    ];
    const independentKeys = allKeysInOrder.filter(k => k !== 'base_url' && !assigned.has(k));

    const baseZipGroupIndex = zipGroups.findIndex(g => g.keys.includes('base_url'));
    const baseZipGroup = baseZipGroupIndex >= 0 ? zipGroups[baseZipGroupIndex] : null;
    const otherZipGroups = baseZipGroup
      ? zipGroups.filter((_g, idx) => idx !== baseZipGroupIndex)
      : zipGroups.slice();

    const groupsPlan = [];
    if (baseZipGroup) groupsPlan.push(baseZipGroup);
    else groupsPlan.push({ id: 'base_url', name: 'Base URL', keys: ['base_url'] });
    otherZipGroups.forEach(g => groupsPlan.push(g));
    independentKeys.forEach(k => groupsPlan.push({ id: `ind:${k}`, name: k, keys: [k] }));

    const groups = groupsPlan.map((g) => {
      const lens = (g.keys || []).map((key) => {
        const values = valuesByKey[key] || [];
        const len = Array.isArray(values) ? values.length : 0;
        return len > 1 ? len : 1;
      });
      const rowCount = Math.max(1, ...lens);
      return { id: g.id, name: g.name, keys: g.keys, rowCount };
    });

    const checkGroupLengths = (group) => {
      const groupLabel = String(group?.name || '').trim() || 'a zip group';
      (group.keys || []).forEach((key) => {
        const values = valuesByKey[key] || [];
        const required = requiredKeys.has(key);
        if (!Array.isArray(values) || values.length === 0) {
          if (required && errors.length < maxErrors) {
            if (key === 'base_url') addError('Base URL is required (provide at least one).');
            else addError(`${key} is required (provide at least one value).`);
          }
          return;
        }
        if (values.length !== 1 && values.length !== group.rowCount) {
          const label = key === 'base_url' ? 'Base URL' : key;
          addError(`${label} has ${values.length} values but must be length 1 or ${group.rowCount} within "${groupLabel}" in Grouped mode.`);
        }
      });
    };

    groups.forEach(checkGroupLengths);

    // Row-level checks (cap errors).
    groups.forEach((group) => {
      if (errors.length >= maxErrors) return;
      const containsBaseUrl = (group.keys || []).includes('base_url');
      const requiredParamKeys = (group.keys || []).filter(k => k !== 'base_url' && requiredKeys.has(k));
      for (let i = 0; i < group.rowCount && errors.length < maxErrors; i++) {
        if (containsBaseUrl) {
          const baseUrl = String(getValueAt(baseUrls, i) || '').trim();
          const baseCheck = validateUrl(baseUrl);
          if (!baseCheck.ok) {
            addError(`Base URL row ${i + 1} is invalid (${baseCheck.message}): ${baseUrl || '(empty)'}`);
          }
        }

        requiredParamKeys.forEach((key) => {
          if (errors.length >= maxErrors) return;
          const values = valuesByKey[key] || [];
          const value = String(getValueAt(values, i) ?? '').trim();
          if (!value) addError(`${key} row ${i + 1} is empty.`);
        });
      }
    });

    if (csvRows && csvRowCount) {
      const anyCsvField = [
        baseInput,
        utm.source,
        utm.medium,
        utm.campaign,
        utm.content,
        utm.term,
        ...customParams.map(p => p?.value),
      ].some(f => f?.mode === INPUT_MODE.CSV_COLUMN);
      if (!anyCsvField) addWarning('CSV uploaded but no fields are mapped to CSV columns.');
    }

    return {
      errors,
      warnings,
      resolved: errors.length ? null : {
        mode,
        overrideExisting,
        excludeRules,
        groups,
        baseUrls,
        fields: resolvedFields,
        customParams: resolvedCustomParams,
        valuesByKey,
      },
    };
  }

  // TEMPLATE_ROWS (defaults + per-row overrides)
  {
    const baseDefault = String(baseInput?.single ?? '').trim();
    const baseMode = baseInput?.mode || INPUT_MODE.SINGLE;
    const baseOverrides = (() => {
      if (baseMode === INPUT_MODE.LIST) return parseMultilinePreserveEmpty(baseInput?.list || '');
      if (baseMode === INPUT_MODE.CSV_COLUMN) return requireCsv('Base URL', baseInput?.csvColumnIndex);
      return [];
    })().map(v => String(v ?? '').trim());

    const resolveTemplateField = (fieldInput, label) => {
      const inputMode = fieldInput?.mode || INPUT_MODE.SINGLE;
      const defaultValue = normalizeValue(fieldInput?.single ?? '', normalization);
      if (inputMode === INPUT_MODE.LIST) {
        return { defaultValue, overrides: parseMultilinePreserveEmpty(fieldInput?.list || '').map(v => normalizeValue(v, normalization)) };
      }
      if (inputMode === INPUT_MODE.CSV_COLUMN) {
        return { defaultValue, overrides: requireCsv(label, fieldInput?.csvColumnIndex).map(v => normalizeValue(v, normalization)) };
      }
      return { defaultValue, overrides: [] };
    };

    const fields = [];
    const requiredFields = [
      { key: 'utm_source', input: utm.source },
      { key: 'utm_medium', input: utm.medium },
      { key: 'utm_campaign', input: utm.campaign },
    ];
    const optionalFields = [
      { key: 'utm_content', input: utm.content },
      { key: 'utm_term', input: utm.term },
    ];

    requiredFields.forEach(({ key, input }) => {
      fields.push({ key, required: true, ...resolveTemplateField(input, key) });
    });
    optionalFields.forEach(({ key, input }) => {
      fields.push({ key, required: false, ...resolveTemplateField(input, key) });
    });

    const customKeys = new Set();
    const resolvedCustomParams = [];
    customParams.forEach((entry) => {
      const key = String(entry?.key || '').trim();
      if (!key) return;
      if (!isValidParamKey(key)) {
        addError(`Custom param key "${key}" is invalid (use letters, numbers, _ or -).`);
        return;
      }
      if (customKeys.has(key)) {
        addError(`Custom param key "${key}" is duplicated.`);
        return;
      }
      if (['utm_source','utm_medium','utm_campaign','utm_content','utm_term'].includes(key)) {
        addError(`Custom param key "${key}" conflicts with a UTM field.`);
        return;
      }
      customKeys.add(key);
      const resolved = resolveTemplateField(entry?.value, `Custom param "${key}"`);
      resolvedCustomParams.push({ key, ...resolved });
    });

    const allOverrideLengths = [
      baseOverrides.length,
      ...fields.map(f => f.overrides.length),
      ...resolvedCustomParams.map(p => p.overrides.length),
    ].map(len => (len > 1 ? len : 1));
    const rowCount = Math.max(1, ...allOverrideLengths);

    const checkOverrideLengths = (label, overrides) => {
      const len = Array.isArray(overrides) ? overrides.length : 0;
      if (len > 1 && len !== rowCount) {
        addError(`${label} overrides has ${len} rows but must be length 1 or ${rowCount} in Template+Rows mode.`);
      }
    };

    checkOverrideLengths('Base URL', baseOverrides);
    fields.forEach(f => checkOverrideLengths(f.key, f.overrides));
    resolvedCustomParams.forEach(p => checkOverrideLengths(`Custom param "${p.key}"`, p.overrides));

    if (!baseDefault && baseOverrides.length === 0) addError('Base URL is required (provide a default or overrides).');
    const baseDefaultCheck = baseDefault ? validateUrl(baseDefault) : null;
    if (baseDefault && baseDefaultCheck && !baseDefaultCheck.ok) {
      addError(`Base URL default is invalid (${baseDefaultCheck.message}): ${baseDefault}`);
    }

    // Validate row-wise: resolved base URL and required UTM fields must be present.
    for (let i = 0; i < rowCount && errors.length < maxErrors; i++) {
      const baseOverride = String(getValueAt(baseOverrides, i) || '').trim();
      const baseUrl = baseOverride || baseDefault;
      const baseCheck = validateUrl(baseUrl);
      if (!baseCheck.ok) {
        addError(`Base URL row ${i + 1} is invalid (${baseCheck.message}): ${baseUrl || '(empty)'}`);
      }

      fields.filter(f => f.required).forEach((field) => {
        const override = String(getValueAt(field.overrides, i) ?? '').trim();
        const value = override || String(field.defaultValue || '').trim();
        if (!value) addError(`${field.key} row ${i + 1} is empty (no override and no default).`);
      });
    }

    if (csvRows && csvRowCount) {
      const anyCsvField = [
        baseInput,
        utm.source,
        utm.medium,
        utm.campaign,
        utm.content,
        utm.term,
        ...customParams.map(p => p?.value),
      ].some(f => f?.mode === INPUT_MODE.CSV_COLUMN);
      if (!anyCsvField) addWarning('CSV uploaded but no fields are mapped to CSV columns.');
    }

    return {
      errors,
      warnings,
      resolved: errors.length ? null : {
        mode,
        overrideExisting,
        excludeRules,
        rowCount,
        baseUrl: { defaultValue: baseDefault, overrides: baseOverrides },
        fields,
        customParams: resolvedCustomParams,
      },
    };
  }
};

function* generateRows(resolved, options = {}) {
  const limit = typeof options.limit === 'number' && options.limit >= 0 ? options.limit : Infinity;
  const mode = resolved?.mode || COMBINATION_MODE.CARTESIAN;
  const overrideExisting = resolved?.overrideExisting !== false;
  const excludeRules = resolved?.excludeRules || [];
  const rowCount = resolved?.rowCount || 1;

  let yielded = 0;

  const buildParamsForValues = (valuesByKey) => {
    const params = Object.create(null);
    Object.entries(valuesByKey).forEach(([key, value]) => {
      const v = String(value ?? '').trim();
      if (v === '') return;
      params[key] = v;
    });
    return params;
  };

  const getValueAt = (values, index) => {
    if (!Array.isArray(values) || values.length === 0) return '';
    if (values.length === 1) return values[0];
    return values[index] ?? '';
  };

  if (mode === COMBINATION_MODE.CARTESIAN) {
    const baseUrls = resolved?.baseUrls || [];
    const fields = resolved?.fields || [];
    const customParams = resolved?.customParams || [];
    const allFields = [...fields, ...customParams.map(p => ({ key: p.key, required: false, values: p.values }))];

    const normalizedFields = allFields.map((field) => {
      const values = Array.isArray(field.values) && field.values.length ? field.values : [''];
      const trimmed = values.map(v => String(v ?? '').trim()).filter(v => v !== '');
      if (!field.required) return { ...field, values: trimmed.length ? trimmed : [''] };
      return { ...field, values: trimmed };
    });

    for (const baseUrl of baseUrls) {
      const indices = new Array(normalizedFields.length).fill(0);
      if (!normalizedFields.length) {
        const finalUrl = buildUrl(baseUrl, [], { overrideExisting });
        yield { baseUrl, params: {}, finalUrl };
        yielded += 1;
        if (yielded >= limit) return;
        continue;
      }

      while (true) {
        const valuesByKey = Object.create(null);
        normalizedFields.forEach((field, idx) => {
          valuesByKey[field.key] = field.values[indices[idx]];
        });

        const params = buildParamsForValues(valuesByKey);
        if (!matchesExcludeRules(params, excludeRules)) {
          const finalUrl = buildUrl(baseUrl, params, { overrideExisting });
          yield { baseUrl, params, finalUrl };
          yielded += 1;
          if (yielded >= limit) return;
        }

        let pos = indices.length - 1;
        while (pos >= 0) {
          indices[pos] += 1;
          if (indices[pos] < normalizedFields[pos].values.length) break;
          indices[pos] = 0;
          pos -= 1;
        }
        if (pos < 0) break;
      }
    }
    return;
  }

  if (mode === COMBINATION_MODE.GROUPS) {
    const groups = Array.isArray(resolved?.groups) ? resolved.groups : [];
    const baseUrls = Array.isArray(resolved?.baseUrls) ? resolved.baseUrls : [];
    const fields = Array.isArray(resolved?.fields) ? resolved.fields : [];
    const customParams = Array.isArray(resolved?.customParams) ? resolved.customParams : [];
    const paramKeys = [
      ...fields.map(f => f.key),
      ...customParams.map(p => p.key),
    ];

    const valuesByKey = resolved?.valuesByKey || Object.create(null);
    const fallbackValuesByKey = Object.create(null);
    fallbackValuesByKey.base_url = baseUrls;
    fields.forEach((f) => { fallbackValuesByKey[f.key] = f.values; });
    customParams.forEach((p) => { fallbackValuesByKey[p.key] = p.values; });
    const effectiveValuesByKey = Object.assign(Object.create(null), fallbackValuesByKey, valuesByKey || {});

    const groupList = groups.map((g) => {
      const rowCount = typeof g?.rowCount === 'number' && g.rowCount > 0 ? g.rowCount : 1;
      return { keys: g.keys || [], rowCount };
    });
    if (!groupList.length) return;

    const indices = new Array(groupList.length).fill(0);
    while (true) {
      const valueMap = Object.create(null);
      groupList.forEach((group, groupIndex) => {
        const idx = indices[groupIndex];
        (group.keys || []).forEach((key) => {
          const values = effectiveValuesByKey[key] || [];
          valueMap[key] = getValueAt(values, idx);
        });
      });

      const baseUrl = String(valueMap.base_url || '').trim();
      const entries = [];
      const params = Object.create(null);
      paramKeys.forEach((key) => {
        const v = String(valueMap[key] ?? '').trim();
        if (v === '') return;
        params[key] = v;
        entries.push({ key, value: v });
      });

      if (!matchesExcludeRules(params, excludeRules)) {
        const finalUrl = buildUrl(baseUrl, entries, { overrideExisting });
        yield { baseUrl, params, finalUrl };
        yielded += 1;
        if (yielded >= limit) return;
      }

      let pos = indices.length - 1;
      while (pos >= 0) {
        indices[pos] += 1;
        if (indices[pos] < groupList[pos].rowCount) break;
        indices[pos] = 0;
        pos -= 1;
      }
      if (pos < 0) break;
    }
    return;
  }

  if (mode === COMBINATION_MODE.ZIP) {
    const baseUrls = resolved?.baseUrls || [];
    const fields = resolved?.fields || [];
    const customParams = resolved?.customParams || [];
    const allFields = [...fields, ...customParams.map(p => ({ key: p.key, required: false, values: p.values }))];

    for (let i = 0; i < rowCount; i++) {
      const baseUrl = String(getValueAt(baseUrls, i) ?? '').trim();
      const valuesByKey = Object.create(null);
      allFields.forEach((field) => {
        valuesByKey[field.key] = getValueAt(field.values, i);
      });

      const params = buildParamsForValues(valuesByKey);
      if (!matchesExcludeRules(params, excludeRules)) {
        const finalUrl = buildUrl(baseUrl, params, { overrideExisting });
        yield { baseUrl, params, finalUrl };
        yielded += 1;
        if (yielded >= limit) return;
      }
    }
    return;
  }

  // TEMPLATE_ROWS (defaults + per-row overrides)
  const base = resolved?.baseUrl || { defaultValue: '', overrides: [] };
  const fields = resolved?.fields || [];
  const customParams = resolved?.customParams || [];

  for (let i = 0; i < rowCount; i++) {
    const baseOverride = String(getValueAt(base.overrides, i) ?? '').trim();
    const baseUrl = baseOverride || String(base.defaultValue || '').trim();
    const valuesByKey = Object.create(null);

    fields.forEach((field) => {
      const override = String(getValueAt(field.overrides, i) ?? '').trim();
      valuesByKey[field.key] = override || String(field.defaultValue || '').trim();
    });
    customParams.forEach((field) => {
      const override = String(getValueAt(field.overrides, i) ?? '').trim();
      valuesByKey[field.key] = override || String(field.defaultValue || '').trim();
    });

    const params = buildParamsForValues(valuesByKey);
    if (!matchesExcludeRules(params, excludeRules)) {
      const finalUrl = buildUrl(baseUrl, params, { overrideExisting });
      yield { baseUrl, params, finalUrl };
      yielded += 1;
      if (yielded >= limit) return;
    }
  }
}

const estimateTotalRows = (resolved) => {
  const mode = resolved?.mode || COMBINATION_MODE.CARTESIAN;
  const cap = Number.MAX_SAFE_INTEGER;
  const multiplyCapped = (a, b) => {
    const left = Number(a);
    const right = Number(b);
    if (!Number.isFinite(left) || !Number.isFinite(right)) return cap;
    if (left <= 0 || right <= 0) return 0;
    const next = left * right;
    if (!Number.isFinite(next) || next > cap) return cap;
    return next;
  };

  if (mode === COMBINATION_MODE.CARTESIAN) {
    const baseCount = Array.isArray(resolved?.baseUrls) ? resolved.baseUrls.length : 0;
    if (!baseCount) return 0;
    const fields = Array.isArray(resolved?.fields) ? resolved.fields : [];
    const customParams = Array.isArray(resolved?.customParams) ? resolved.customParams : [];
    const lengths = [
      ...fields.map(f => (Array.isArray(f?.values) && f.values.length ? f.values.length : 1)),
      ...customParams.map(f => (Array.isArray(f?.values) && f.values.length ? f.values.length : 1)),
    ];
    return lengths.reduce((acc, len) => multiplyCapped(acc, len), baseCount);
  }
  if (mode === COMBINATION_MODE.GROUPS) {
    const groups = Array.isArray(resolved?.groups) ? resolved.groups : [];
    if (!groups.length) return 0;
    return groups.reduce((acc, g) => multiplyCapped(acc, g?.rowCount || 1), 1);
  }
  if (mode === COMBINATION_MODE.ZIP || mode === COMBINATION_MODE.TEMPLATE_ROWS) {
    return typeof resolved?.rowCount === 'number' ? resolved.rowCount : 0;
  }
  return 0;
};

module.exports = {
  INPUT_MODE,
  COMBINATION_MODE,
  DEFAULT_NORMALIZATION,
  parseMultiline,
  parseMultilinePreserveEmpty,
  parseCsv,
  getCsvHeaders,
  getCsvColumnValues,
  isValidParamKey,
  normalizeValue,
  buildUrl,
  parseExcludeRules,
  resolveAndValidateConfig,
  generateRows,
  estimateTotalRows,
};
