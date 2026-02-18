(() => {
  'use strict';

  const TOOL_ID = 'button-css-lab';
  const MAX_SESSION_CSS_CHARS = 120_000;
  const PREVIEW_STATE_STYLE_ID = 'buttonlab-preview-state-style';

  const BASE_PRESET = {
    backgroundColor: '#2563eb',
    backgroundAlpha: 1,
    textColor: '#ffffff',
    borderColor: '#1d4ed8',
    borderWidth: 1,
    borderStyle: 'solid',
    radius: 12,
    fontSize: 16,
    fontWeight: '500',
    fontFamily: '"Poppins", "Segoe UI", sans-serif',
    lineHeight: 1.2,
    paddingY: 12,
    paddingX: 20,
    letterSpacing: 0.1,
    minWidth: 160,
    textTransform: 'none',
    justifyContent: 'center',
    shadowX: 0,
    shadowY: 10,
    shadowBlur: 22,
    shadowSpread: 0,
    shadowColor: '#2563eb',
    shadowAlpha: 0.28,
    shadowInset: false,
    opacity: 1,
    transform: '',
    cursor: 'pointer',
    transitionProperty: 'all',
    transitionMs: 180,
    transitionEasing: 'ease',
    includeHover: true,
    hoverBackgroundColor: '#1d4ed8',
    hoverBackgroundAlpha: 1,
    hoverTextColor: '#ffffff',
    hoverBorderColor: '#1e40af',
    hoverShadowX: 0,
    hoverShadowY: 12,
    hoverShadowBlur: 26,
    hoverShadowSpread: 0,
    hoverShadowColor: '#1d4ed8',
    hoverShadowAlpha: 0.34,
    hoverShadowInset: false,
    hoverLift: 1,
    includeActive: true,
    activeLift: 1,
    includeFocus: true,
    focusRingColor: '#93c5fd',
    focusRingWidth: 3
  };

  const PRESETS = {
    starter: {
      ...BASE_PRESET
    },
    'pill-pop': {
      ...BASE_PRESET,
      backgroundColor: '#0f172a',
      borderColor: '#0f172a',
      radius: 999,
      fontSize: 15,
      fontWeight: '600',
      paddingY: 11,
      paddingX: 22,
      letterSpacing: 0.2,
      shadowY: 8,
      shadowBlur: 18,
      shadowColor: '#0f172a',
      shadowAlpha: 0.32,
      hoverBackgroundColor: '#1e293b',
      hoverBorderColor: '#1e293b',
      hoverShadowY: 10,
      hoverShadowBlur: 20,
      hoverShadowColor: '#0f172a',
      hoverShadowAlpha: 0.38,
      transitionMs: 160,
      focusRingColor: '#94a3b8'
    },
    'outline-clean': {
      ...BASE_PRESET,
      backgroundColor: '#f8fafc',
      backgroundAlpha: 0,
      textColor: '#0f172a',
      borderColor: '#334155',
      borderWidth: 2,
      radius: 10,
      fontWeight: '600',
      shadowY: 0,
      shadowBlur: 0,
      shadowSpread: 0,
      shadowAlpha: 0,
      hoverBackgroundColor: '#f8fafc',
      hoverBackgroundAlpha: 1,
      hoverTextColor: '#0f172a',
      hoverBorderColor: '#0f172a',
      hoverShadowY: 8,
      hoverShadowBlur: 18,
      hoverShadowColor: '#0f172a',
      hoverShadowAlpha: 0.16,
      transitionMs: 200,
      focusRingColor: '#64748b'
    },
    'dark-solid': {
      ...BASE_PRESET,
      backgroundColor: '#111827',
      textColor: '#f9fafb',
      borderColor: '#0f172a',
      radius: 8,
      fontSize: 15,
      fontWeight: '500',
      paddingY: 10,
      paddingX: 18,
      letterSpacing: 0.25,
      shadowY: 6,
      shadowBlur: 14,
      shadowColor: '#111827',
      shadowAlpha: 0.34,
      hoverBackgroundColor: '#1f2937',
      hoverBorderColor: '#1f2937',
      hoverShadowY: 9,
      hoverShadowBlur: 18,
      hoverShadowColor: '#111827',
      hoverShadowAlpha: 0.38,
      transitionMs: 180,
      focusRingColor: '#9ca3af'
    },
    'sunset-gradient': {
      ...BASE_PRESET,
      backgroundColor: '#f97316',
      borderColor: '#fb7185',
      radius: 14,
      fontWeight: '600',
      paddingX: 22,
      letterSpacing: 0.2,
      shadowY: 12,
      shadowBlur: 26,
      shadowColor: '#ec4899',
      shadowAlpha: 0.3,
      hoverBackgroundColor: '#ea580c',
      hoverBorderColor: '#f43f5e',
      hoverShadowY: 14,
      hoverShadowBlur: 28,
      hoverShadowColor: '#db2777',
      hoverShadowAlpha: 0.34,
      hoverLift: 1.5,
      activeLift: 1.2,
      transitionMs: 220,
      focusRingColor: '#fda4af'
    }
  };

  const DEFAULT_PRESET_ID = 'starter';

  const getEl = (id) => document.getElementById(id);

  const els = {
    form: getEl('buttonlab-form'),
    preset: getEl('buttonlab-preset'),
    applyPresetBtn: getEl('buttonlab-apply-preset'),
    resetDefaultsBtn: getEl('buttonlab-reset-defaults'),

    label: getEl('buttonlab-label'),
    className: getEl('buttonlab-class'),

    background: getEl('buttonlab-background'),
    backgroundAlpha: getEl('buttonlab-background-alpha'),
    textColor: getEl('buttonlab-text-color'),
    borderColor: getEl('buttonlab-border-color'),
    borderWidth: getEl('buttonlab-border-width'),
    borderStyle: getEl('buttonlab-border-style'),
    radius: getEl('buttonlab-radius'),
    fontSize: getEl('buttonlab-font-size'),
    fontWeight: getEl('buttonlab-font-weight'),
    fontFamily: getEl('buttonlab-font-family'),
    lineHeight: getEl('buttonlab-line-height'),
    paddingY: getEl('buttonlab-padding-y'),
    paddingX: getEl('buttonlab-padding-x'),
    letterSpacing: getEl('buttonlab-letter-spacing'),
    minWidth: getEl('buttonlab-min-width'),
    textTransform: getEl('buttonlab-text-transform'),
    justifyContent: getEl('buttonlab-justify-content'),

    shadowX: getEl('buttonlab-shadow-x'),
    shadowY: getEl('buttonlab-shadow-y'),
    shadowBlur: getEl('buttonlab-shadow-blur'),
    shadowSpread: getEl('buttonlab-shadow-spread'),
    shadowColor: getEl('buttonlab-shadow-color'),
    shadowAlpha: getEl('buttonlab-shadow-alpha'),
    shadowInset: getEl('buttonlab-shadow-inset'),

    opacity: getEl('buttonlab-opacity'),
    transform: getEl('buttonlab-transform'),

    includeHover: getEl('buttonlab-include-hover'),
    hoverBackground: getEl('buttonlab-hover-background'),
    hoverBackgroundAlpha: getEl('buttonlab-hover-background-alpha'),
    hoverTextColor: getEl('buttonlab-hover-text-color'),
    hoverBorderColor: getEl('buttonlab-hover-border-color'),

    hoverShadowX: getEl('buttonlab-hover-shadow-x'),
    hoverShadowY: getEl('buttonlab-hover-shadow-y'),
    hoverShadowBlur: getEl('buttonlab-hover-shadow-blur'),
    hoverShadowSpread: getEl('buttonlab-hover-shadow-spread'),
    hoverShadowColor: getEl('buttonlab-hover-shadow-color'),
    hoverShadowAlpha: getEl('buttonlab-hover-shadow-alpha'),
    hoverShadowInset: getEl('buttonlab-hover-shadow-inset'),

    hoverLift: getEl('buttonlab-hover-lift'),

    includeActive: getEl('buttonlab-include-active'),
    activeLift: getEl('buttonlab-active-lift'),

    includeFocus: getEl('buttonlab-include-focus'),
    focusRingColor: getEl('buttonlab-focus-ring-color'),
    focusRingWidth: getEl('buttonlab-focus-ring-width'),

    cursor: getEl('buttonlab-cursor'),
    transitionProperty: getEl('buttonlab-transition-property'),
    transitionMs: getEl('buttonlab-transition'),
    transitionEasing: getEl('buttonlab-transition-easing'),

    customCss: getEl('buttonlab-custom-css'),
    customStatus: getEl('buttonlab-custom-status'),

    previewButton: getEl('buttonlab-preview-button'),
    previewDock: getEl('buttonlab-preview-dock'),
    previewDockButton: getEl('buttonlab-preview-button-dock'),
    previewDockSummary: getEl('buttonlab-summary-dock'),
    summary: getEl('buttonlab-summary'),
    exportOutput: getEl('buttonlab-export'),
    exportStatus: getEl('buttonlab-export-status'),
    copyCssBtn: getEl('buttonlab-copy-css'),
    downloadCssBtn: getEl('buttonlab-download-css')
  };

  const requiredEls = [
    'form', 'preset', 'applyPresetBtn', 'resetDefaultsBtn', 'label', 'className',
    'background', 'backgroundAlpha', 'textColor', 'borderColor', 'borderWidth', 'borderStyle',
    'radius', 'fontSize', 'fontWeight', 'fontFamily', 'lineHeight', 'paddingY', 'paddingX',
    'letterSpacing', 'minWidth', 'textTransform', 'justifyContent', 'shadowX', 'shadowY',
    'shadowBlur', 'shadowSpread', 'shadowColor', 'shadowAlpha', 'shadowInset', 'opacity', 'transform',
    'includeHover', 'hoverBackground', 'hoverBackgroundAlpha', 'hoverTextColor', 'hoverBorderColor',
    'hoverShadowX', 'hoverShadowY', 'hoverShadowBlur', 'hoverShadowSpread', 'hoverShadowColor',
    'hoverShadowAlpha', 'hoverShadowInset', 'hoverLift',
    'includeActive', 'activeLift', 'includeFocus', 'focusRingColor', 'focusRingWidth',
    'cursor', 'transitionProperty', 'transitionMs', 'transitionEasing',
    'customCss', 'customStatus', 'previewButton', 'summary', 'exportOutput', 'exportStatus',
    'copyCssBtn', 'downloadCssBtn'
  ];

  const createFallbackControl = ({ value = '', checked = false } = {}) => ({
    value: String(value),
    checked: Boolean(checked),
    disabled: false,
    textContent: '',
    dataset: {},
    style: {
      cssText: '',
      setProperty() {}
    },
    options: [],
    addEventListener() {},
    removeEventListener() {},
    focus() {}
  });

  const criticalEls = ['form', 'previewButton', 'summary', 'exportOutput'];
  if (!criticalEls.every((key) => Boolean(els[key]))) return;

  const fallbackValues = {
    preset: DEFAULT_PRESET_ID,
    label: 'Click me',
    className: 'my-button',
    background: BASE_PRESET.backgroundColor,
    backgroundAlpha: BASE_PRESET.backgroundAlpha,
    textColor: BASE_PRESET.textColor,
    borderColor: BASE_PRESET.borderColor,
    borderWidth: BASE_PRESET.borderWidth,
    borderStyle: BASE_PRESET.borderStyle,
    radius: BASE_PRESET.radius,
    fontSize: BASE_PRESET.fontSize,
    fontWeight: BASE_PRESET.fontWeight,
    fontFamily: BASE_PRESET.fontFamily,
    lineHeight: BASE_PRESET.lineHeight,
    paddingY: BASE_PRESET.paddingY,
    paddingX: BASE_PRESET.paddingX,
    letterSpacing: BASE_PRESET.letterSpacing,
    minWidth: BASE_PRESET.minWidth,
    textTransform: BASE_PRESET.textTransform,
    justifyContent: BASE_PRESET.justifyContent,
    shadowX: BASE_PRESET.shadowX,
    shadowY: BASE_PRESET.shadowY,
    shadowBlur: BASE_PRESET.shadowBlur,
    shadowSpread: BASE_PRESET.shadowSpread,
    shadowColor: BASE_PRESET.shadowColor,
    shadowAlpha: BASE_PRESET.shadowAlpha,
    opacity: BASE_PRESET.opacity,
    transform: BASE_PRESET.transform,
    hoverBackground: BASE_PRESET.hoverBackgroundColor,
    hoverBackgroundAlpha: BASE_PRESET.hoverBackgroundAlpha,
    hoverTextColor: BASE_PRESET.hoverTextColor,
    hoverBorderColor: BASE_PRESET.hoverBorderColor,
    hoverShadowX: BASE_PRESET.hoverShadowX,
    hoverShadowY: BASE_PRESET.hoverShadowY,
    hoverShadowBlur: BASE_PRESET.hoverShadowBlur,
    hoverShadowSpread: BASE_PRESET.hoverShadowSpread,
    hoverShadowColor: BASE_PRESET.hoverShadowColor,
    hoverShadowAlpha: BASE_PRESET.hoverShadowAlpha,
    hoverLift: BASE_PRESET.hoverLift,
    activeLift: BASE_PRESET.activeLift,
    focusRingColor: BASE_PRESET.focusRingColor,
    focusRingWidth: BASE_PRESET.focusRingWidth,
    cursor: BASE_PRESET.cursor,
    transitionProperty: BASE_PRESET.transitionProperty,
    transitionMs: BASE_PRESET.transitionMs,
    transitionEasing: BASE_PRESET.transitionEasing,
    customCss: '',
    exportOutput: '',
    exportStatus: '',
    customStatus: '',
    previewDockSummary: ''
  };

  const fallbackChecked = {
    shadowInset: BASE_PRESET.shadowInset,
    includeHover: BASE_PRESET.includeHover,
    hoverShadowInset: BASE_PRESET.hoverShadowInset,
    includeActive: BASE_PRESET.includeActive,
    includeFocus: BASE_PRESET.includeFocus
  };

  requiredEls.forEach((key) => {
    if (els[key]) return;
    els[key] = createFallbackControl({
      value: Object.prototype.hasOwnProperty.call(fallbackValues, key) ? fallbackValues[key] : '',
      checked: Object.prototype.hasOwnProperty.call(fallbackChecked, key) ? fallbackChecked[key] : false
    });

    if (key === 'preset') {
      els[key].options = Object.keys(PRESETS).map((value) => ({
        value,
        textContent: value
      }));
    }
  });

  let applyingSession = false;

  const markSessionDirty = () => {
    if (applyingSession) return;
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const clampNumber = (value, min, max, fallback) => {
    const parsed = Number.parseFloat(String(value ?? '').trim());
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const toStringValue = (value, fallback = '') => {
    const next = String(value ?? '').trim();
    return next || fallback;
  };

  const formatDecimal = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return '0';
    if (Math.abs(n - Math.round(n)) < 0.001) return String(Math.round(n));
    return String(n.toFixed(3)).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
  };

  const sanitizeClassName = (value) => {
    const raw = String(value || '').trim();
    const fallback = 'my-button';
    if (!raw) return fallback;

    const token = raw.split(/\s+/).find(Boolean) || '';
    const cleaned = token.replace(/[^A-Za-z0-9_-]/g, '');
    if (!cleaned) return fallback;
    if (!/^[A-Za-z_]/.test(cleaned)) return `btn-${cleaned}`;
    return cleaned;
  };

  const normalizeHexColor = (value, fallback = '#000000') => {
    const read = (input) => {
      const raw = String(input || '').trim();
      const match = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(raw);
      if (!match) return '';
      const body = match[1].length === 3
        ? match[1].split('').map((ch) => ch + ch).join('')
        : match[1];
      return `#${body.toLowerCase()}`;
    };

    return read(value) || read(fallback) || '#000000';
  };

  const hexToRgb = (hexValue) => {
    const hex = normalizeHexColor(hexValue, '#000000').slice(1);
    return {
      r: Number.parseInt(hex.slice(0, 2), 16),
      g: Number.parseInt(hex.slice(2, 4), 16),
      b: Number.parseInt(hex.slice(4, 6), 16)
    };
  };

  const rgbaFromHex = (hexValue, alphaValue) => {
    const { r, g, b } = hexToRgb(hexValue);
    const alpha = clampNumber(alphaValue, 0, 1, 1);
    return `rgba(${r}, ${g}, ${b}, ${formatDecimal(alpha)})`;
  };

  const buildBoxShadowValue = ({ x, y, blur, spread, color, alpha, inset }) => {
    const normalizedAlpha = clampNumber(alpha, 0, 1, 0);
    if (normalizedAlpha <= 0) return 'none';

    const prefix = inset ? 'inset ' : '';
    return `${prefix}${formatDecimal(x)}px ${formatDecimal(y)}px ${formatDecimal(blur)}px ${formatDecimal(spread)}px ${rgbaFromHex(color, normalizedAlpha)}`;
  };

  const parseCustomDeclarations = (raw) => {
    const textValue = String(raw || '').trim();
    if (!textValue) return { declarations: [], invalid: [] };

    let body = textValue;
    const openBrace = body.indexOf('{');
    const closeBrace = body.lastIndexOf('}');
    if (openBrace >= 0 && closeBrace > openBrace) {
      body = body.slice(openBrace + 1, closeBrace);
    }

    const declarations = [];
    const invalid = [];

    body
      .split(';')
      .map((segment) => segment.trim())
      .filter(Boolean)
      .forEach((segment) => {
        const colonIndex = segment.indexOf(':');
        if (colonIndex <= 0 || colonIndex >= segment.length - 1) {
          invalid.push(segment);
          return;
        }

        const property = segment.slice(0, colonIndex).trim();
        const value = segment.slice(colonIndex + 1).trim();
        const validProperty = /^--[A-Za-z0-9_-]+$/.test(property) || /^[A-Za-z-]+$/.test(property);

        if (!validProperty || !value) {
          invalid.push(segment);
          return;
        }

        declarations.push([property, value]);
      });

    return { declarations, invalid };
  };

  const setExportStatus = (message, tone = '') => {
    els.exportStatus.textContent = String(message || '');
    els.exportStatus.dataset.tone = tone;
  };

  const setCustomStatus = (invalidCount) => {
    if (!invalidCount) {
      els.customStatus.textContent = 'Optional: add extra CSS lines like transform: translateY(-1px);';
      els.customStatus.dataset.tone = '';
      return;
    }

    els.customStatus.textContent = `${invalidCount} line${invalidCount === 1 ? '' : 's'} skipped. Use "property: value;" format.`;
    els.customStatus.dataset.tone = 'warn';
  };

  const setStateControlGroupEnabled = (name, enabled) => {
    const groups = [...document.querySelectorAll(`[data-state-controls="${name}"]`)];
    groups.forEach((group) => {
      group.classList.toggle('is-disabled', !enabled);
      [...group.querySelectorAll('input, select, textarea, button')].forEach((control) => {
        control.disabled = !enabled;
      });
    });
  };

  const syncStateControlAvailability = () => {
    setStateControlGroupEnabled('hover', Boolean(els.includeHover.checked));
    setStateControlGroupEnabled('active', Boolean(els.includeActive.checked));
    setStateControlGroupEnabled('focus', Boolean(els.includeFocus.checked));
  };

  const getDefaultPreset = () => PRESETS[DEFAULT_PRESET_ID] || BASE_PRESET;

  const readStateFromInputs = () => {
    const defaults = getDefaultPreset();

    return {
      presetId: toStringValue(els.preset.value, DEFAULT_PRESET_ID),
      label: toStringValue(els.label.value, 'Click me'),
      className: sanitizeClassName(els.className.value),

      backgroundColor: normalizeHexColor(els.background.value, defaults.backgroundColor),
      backgroundAlpha: clampNumber(els.backgroundAlpha.value, 0, 1, defaults.backgroundAlpha),
      textColor: normalizeHexColor(els.textColor.value, defaults.textColor),
      borderColor: normalizeHexColor(els.borderColor.value, defaults.borderColor),
      borderWidth: clampNumber(els.borderWidth.value, 0, 12, defaults.borderWidth),
      borderStyle: toStringValue(els.borderStyle.value, defaults.borderStyle),
      radius: clampNumber(els.radius.value, 0, 80, defaults.radius),

      fontSize: clampNumber(els.fontSize.value, 10, 42, defaults.fontSize),
      fontWeight: toStringValue(els.fontWeight.value, defaults.fontWeight),
      fontFamily: toStringValue(els.fontFamily.value, defaults.fontFamily),
      lineHeight: clampNumber(els.lineHeight.value, 0.8, 3, defaults.lineHeight),
      paddingY: clampNumber(els.paddingY.value, 0, 48, defaults.paddingY),
      paddingX: clampNumber(els.paddingX.value, 0, 80, defaults.paddingX),
      letterSpacing: clampNumber(els.letterSpacing.value, -2, 6, defaults.letterSpacing),
      minWidth: clampNumber(els.minWidth.value, 0, 480, defaults.minWidth),
      textTransform: toStringValue(els.textTransform.value, defaults.textTransform),
      justifyContent: toStringValue(els.justifyContent.value, defaults.justifyContent),

      shadowX: clampNumber(els.shadowX.value, -80, 80, defaults.shadowX),
      shadowY: clampNumber(els.shadowY.value, -80, 80, defaults.shadowY),
      shadowBlur: clampNumber(els.shadowBlur.value, 0, 160, defaults.shadowBlur),
      shadowSpread: clampNumber(els.shadowSpread.value, -80, 80, defaults.shadowSpread),
      shadowColor: normalizeHexColor(els.shadowColor.value, defaults.shadowColor),
      shadowAlpha: clampNumber(els.shadowAlpha.value, 0, 1, defaults.shadowAlpha),
      shadowInset: Boolean(els.shadowInset.checked),

      opacity: clampNumber(els.opacity.value, 0, 1, defaults.opacity),
      transform: String(els.transform.value || '').trim(),

      includeHover: Boolean(els.includeHover.checked),
      hoverBackgroundColor: normalizeHexColor(els.hoverBackground.value, defaults.hoverBackgroundColor),
      hoverBackgroundAlpha: clampNumber(els.hoverBackgroundAlpha.value, 0, 1, defaults.hoverBackgroundAlpha),
      hoverTextColor: normalizeHexColor(els.hoverTextColor.value, defaults.hoverTextColor),
      hoverBorderColor: normalizeHexColor(els.hoverBorderColor.value, defaults.hoverBorderColor),

      hoverShadowX: clampNumber(els.hoverShadowX.value, -80, 80, defaults.hoverShadowX),
      hoverShadowY: clampNumber(els.hoverShadowY.value, -80, 80, defaults.hoverShadowY),
      hoverShadowBlur: clampNumber(els.hoverShadowBlur.value, 0, 160, defaults.hoverShadowBlur),
      hoverShadowSpread: clampNumber(els.hoverShadowSpread.value, -80, 80, defaults.hoverShadowSpread),
      hoverShadowColor: normalizeHexColor(els.hoverShadowColor.value, defaults.hoverShadowColor),
      hoverShadowAlpha: clampNumber(els.hoverShadowAlpha.value, 0, 1, defaults.hoverShadowAlpha),
      hoverShadowInset: Boolean(els.hoverShadowInset.checked),

      hoverLift: clampNumber(els.hoverLift.value, -20, 20, defaults.hoverLift),

      includeActive: Boolean(els.includeActive.checked),
      activeLift: clampNumber(els.activeLift.value, -20, 20, defaults.activeLift),

      includeFocus: Boolean(els.includeFocus.checked),
      focusRingColor: normalizeHexColor(els.focusRingColor.value, defaults.focusRingColor),
      focusRingWidth: clampNumber(els.focusRingWidth.value, 0, 16, defaults.focusRingWidth),

      cursor: toStringValue(els.cursor.value, defaults.cursor),
      transitionProperty: toStringValue(els.transitionProperty.value, defaults.transitionProperty),
      transitionMs: clampNumber(els.transitionMs.value, 0, 2000, defaults.transitionMs),
      transitionEasing: toStringValue(els.transitionEasing.value, defaults.transitionEasing),

      customCss: String(els.customCss.value || '')
    };
  };

  const syncInputsFromPreset = (presetId) => {
    const preset = PRESETS[presetId] || getDefaultPreset();

    els.background.value = normalizeHexColor(preset.backgroundColor, BASE_PRESET.backgroundColor);
    els.backgroundAlpha.value = String(preset.backgroundAlpha);
    els.textColor.value = normalizeHexColor(preset.textColor, BASE_PRESET.textColor);
    els.borderColor.value = normalizeHexColor(preset.borderColor, BASE_PRESET.borderColor);
    els.borderWidth.value = String(preset.borderWidth);
    els.borderStyle.value = String(preset.borderStyle);
    els.radius.value = String(preset.radius);

    els.fontSize.value = String(preset.fontSize);
    els.fontWeight.value = String(preset.fontWeight);
    els.fontFamily.value = preset.fontFamily;
    els.lineHeight.value = String(preset.lineHeight);
    els.paddingY.value = String(preset.paddingY);
    els.paddingX.value = String(preset.paddingX);
    els.letterSpacing.value = String(preset.letterSpacing);
    els.minWidth.value = String(preset.minWidth);
    els.textTransform.value = String(preset.textTransform);
    els.justifyContent.value = String(preset.justifyContent);

    els.shadowX.value = String(preset.shadowX);
    els.shadowY.value = String(preset.shadowY);
    els.shadowBlur.value = String(preset.shadowBlur);
    els.shadowSpread.value = String(preset.shadowSpread);
    els.shadowColor.value = normalizeHexColor(preset.shadowColor, BASE_PRESET.shadowColor);
    els.shadowAlpha.value = String(preset.shadowAlpha);
    els.shadowInset.checked = Boolean(preset.shadowInset);

    els.opacity.value = String(preset.opacity);
    els.transform.value = preset.transform || '';

    els.includeHover.checked = Boolean(preset.includeHover);
    els.hoverBackground.value = normalizeHexColor(preset.hoverBackgroundColor, BASE_PRESET.hoverBackgroundColor);
    els.hoverBackgroundAlpha.value = String(preset.hoverBackgroundAlpha);
    els.hoverTextColor.value = normalizeHexColor(preset.hoverTextColor, BASE_PRESET.hoverTextColor);
    els.hoverBorderColor.value = normalizeHexColor(preset.hoverBorderColor, BASE_PRESET.hoverBorderColor);

    els.hoverShadowX.value = String(preset.hoverShadowX);
    els.hoverShadowY.value = String(preset.hoverShadowY);
    els.hoverShadowBlur.value = String(preset.hoverShadowBlur);
    els.hoverShadowSpread.value = String(preset.hoverShadowSpread);
    els.hoverShadowColor.value = normalizeHexColor(preset.hoverShadowColor, BASE_PRESET.hoverShadowColor);
    els.hoverShadowAlpha.value = String(preset.hoverShadowAlpha);
    els.hoverShadowInset.checked = Boolean(preset.hoverShadowInset);

    els.hoverLift.value = String(preset.hoverLift);

    els.includeActive.checked = Boolean(preset.includeActive);
    els.activeLift.value = String(preset.activeLift);

    els.includeFocus.checked = Boolean(preset.includeFocus);
    els.focusRingColor.value = normalizeHexColor(preset.focusRingColor, BASE_PRESET.focusRingColor);
    els.focusRingWidth.value = String(preset.focusRingWidth);

    els.cursor.value = String(preset.cursor);
    els.transitionProperty.value = String(preset.transitionProperty);
    els.transitionMs.value = String(preset.transitionMs);
    els.transitionEasing.value = String(preset.transitionEasing);

    syncStateControlAvailability();
  };

  const buildTransitionValue = (state) => {
    const ms = clampNumber(state.transitionMs, 0, 2000, 180);
    if (state.transitionProperty === 'none' || ms <= 0) return 'none';
    return `${state.transitionProperty} ${formatDecimal(ms)}ms ${state.transitionEasing}`;
  };

  const buildBaseDeclarations = (state) => {
    const declarations = [
      ['display', 'inline-flex'],
      ['align-items', 'center'],
      ['justify-content', state.justifyContent],
      ['min-width', `${formatDecimal(state.minWidth)}px`],
      ['gap', '0.5rem'],
      ['font-family', state.fontFamily],
      ['font-size', `${formatDecimal(state.fontSize)}px`],
      ['font-weight', String(state.fontWeight)],
      ['line-height', formatDecimal(state.lineHeight)],
      ['letter-spacing', `${formatDecimal(state.letterSpacing)}px`],
      ['text-transform', state.textTransform],
      ['padding', `${formatDecimal(state.paddingY)}px ${formatDecimal(state.paddingX)}px`],
      ['color', state.textColor],
      ['background-color', rgbaFromHex(state.backgroundColor, state.backgroundAlpha)],
      ['border-width', `${formatDecimal(state.borderWidth)}px`],
      ['border-style', state.borderStyle],
      ['border-color', state.borderColor],
      ['border-radius', `${formatDecimal(state.radius)}px`],
      ['box-shadow', buildBoxShadowValue({
        x: state.shadowX,
        y: state.shadowY,
        blur: state.shadowBlur,
        spread: state.shadowSpread,
        color: state.shadowColor,
        alpha: state.shadowAlpha,
        inset: state.shadowInset
      })],
      ['opacity', formatDecimal(state.opacity)],
      ['cursor', state.cursor],
      ['transition', buildTransitionValue(state)]
    ];

    const transformValue = String(state.transform || '').trim();
    if (transformValue && transformValue.toLowerCase() !== 'none') {
      declarations.push(['transform', transformValue]);
    }

    return declarations;
  };

  const buildHoverDeclarations = (state) => {
    if (!state.includeHover) return [];

    const declarations = [
      ['background-color', rgbaFromHex(state.hoverBackgroundColor, state.hoverBackgroundAlpha)],
      ['color', state.hoverTextColor],
      ['border-color', state.hoverBorderColor],
      ['box-shadow', buildBoxShadowValue({
        x: state.hoverShadowX,
        y: state.hoverShadowY,
        blur: state.hoverShadowBlur,
        spread: state.hoverShadowSpread,
        color: state.hoverShadowColor,
        alpha: state.hoverShadowAlpha,
        inset: state.hoverShadowInset
      })]
    ];

    if (Math.abs(state.hoverLift) > 0.0001) {
      declarations.push(['transform', `translateY(${formatDecimal(-state.hoverLift)}px)`]);
    }

    return declarations;
  };

  const buildActiveDeclarations = (state) => {
    if (!state.includeActive) return [];
    if (Math.abs(state.activeLift) <= 0.0001) return [];
    return [['transform', `translateY(${formatDecimal(state.activeLift)}px)`]];
  };

  const buildFocusDeclarations = (state) => {
    if (!state.includeFocus) return [];
    const width = clampNumber(state.focusRingWidth, 0, 16, 0);
    if (width <= 0) return [];

    return [
      ['outline', `${formatDecimal(width)}px solid ${state.focusRingColor}`],
      ['outline-offset', '2px']
    ];
  };

  const toCssBlock = (selector, declarations) => {
    if (!selector || !Array.isArray(declarations) || !declarations.length) return '';
    const lines = declarations
      .filter(([property, value]) => String(property || '').trim() && String(value || '').trim())
      .map(([property, value]) => `  ${property}: ${value};`);
    if (!lines.length) return '';
    return `${selector} {\n${lines.join('\n')}\n}`;
  };

  const buildCssOutput = ({ className, baseDeclarations, customDeclarations, hoverDeclarations, activeDeclarations, focusDeclarations }) => {
    const blocks = [];

    const baseBlock = toCssBlock(`.${className}`, [...baseDeclarations, ...customDeclarations]);
    if (baseBlock) blocks.push(baseBlock);

    const hoverBlock = toCssBlock(`.${className}:hover`, hoverDeclarations);
    if (hoverBlock) blocks.push(hoverBlock);

    const activeBlock = toCssBlock(`.${className}:active`, activeDeclarations);
    if (activeBlock) blocks.push(activeBlock);

    const focusBlock = toCssBlock(`.${className}:focus-visible`, focusDeclarations);
    if (focusBlock) blocks.push(focusBlock);

    return blocks.join('\n\n').trim();
  };

  const getPresetLabel = (presetId) => {
    const option = [...els.preset.options].find((entry) => entry.value === presetId);
    return String(option?.textContent || presetId || DEFAULT_PRESET_ID).trim();
  };

  const ensurePreviewStateStyleEl = () => {
    let el = document.getElementById(PREVIEW_STATE_STYLE_ID);
    if (el) return el;

    el = document.createElement('style');
    el.id = PREVIEW_STATE_STYLE_ID;
    document.head.appendChild(el);
    return el;
  };

  const updatePreviewStateStyles = ({ baseDeclarations, customDeclarations, hoverDeclarations, activeDeclarations, focusDeclarations }) => {
    const styleEl = ensurePreviewStateStyleEl();
    const blocks = [];

    const previewSelectors = ['#buttonlab-preview-button'];
    if (els.previewDockButton && typeof els.previewDockButton === 'object') {
      previewSelectors.push('#buttonlab-preview-button-dock');
    }

    const selectorList = (suffix = '') => previewSelectors.map((selector) => `${selector}${suffix}`).join(', ');

    const baseBlock = toCssBlock(selectorList(), [...baseDeclarations, ...customDeclarations]);
    if (baseBlock) blocks.push(baseBlock);

    const hoverBlock = toCssBlock(selectorList(':hover'), hoverDeclarations);
    if (hoverBlock) blocks.push(hoverBlock);

    const activeBlock = toCssBlock(selectorList(':active'), activeDeclarations);
    if (activeBlock) blocks.push(activeBlock);

    const focusBlock = toCssBlock(selectorList(':focus-visible'), focusDeclarations);
    if (focusBlock) blocks.push(focusBlock);

    styleEl.textContent = blocks.join('\n\n');
  };

  const setPreviewDockVisible = (visible) => {
    if (!els.previewDock || !els.previewDock.classList) return;
    els.previewDock.classList.toggle('is-visible', Boolean(visible));
  };

  const initPreviewDockVisibility = () => {
    if (!els.previewDock) return;

    const section = document.querySelector('.buttonlab-section');
    if (!section) {
      setPreviewDockVisible(true);
      return;
    }

    if (!('IntersectionObserver' in window)) {
      setPreviewDockVisible(true);
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      const entry = entries?.[0];
      setPreviewDockVisible(Boolean(entry && entry.isIntersecting));
    }, {
      threshold: 0.03
    });

    setPreviewDockVisible(true);
    observer.observe(section);
  };

  const getRenderedState = () => {
    const state = readStateFromInputs();
    const baseDeclarations = buildBaseDeclarations(state);
    const customResult = parseCustomDeclarations(state.customCss);

    const hoverDeclarations = buildHoverDeclarations(state);
    const activeDeclarations = buildActiveDeclarations(state);
    const focusDeclarations = buildFocusDeclarations(state);

    const cssOutput = buildCssOutput({
      className: state.className,
      baseDeclarations,
      customDeclarations: customResult.declarations,
      hoverDeclarations,
      activeDeclarations,
      focusDeclarations
    });

    const summaryParts = [
      `Preset: ${getPresetLabel(state.presetId)}`,
      `${formatDecimal(state.radius)}px radius`,
      `${formatDecimal(state.fontSize)}px type`,
      state.includeHover ? 'hover on' : 'hover off'
    ];

    return {
      state,
      baseDeclarations,
      customDeclarations: customResult.declarations,
      invalidCustomLines: customResult.invalid,
      hoverDeclarations,
      activeDeclarations,
      focusDeclarations,
      cssOutput,
      summary: summaryParts.join(' · ')
    };
  };

  const render = ({ dirty = false } = {}) => {
    syncStateControlAvailability();
    const rendered = getRenderedState();

    els.className.value = rendered.state.className;
    els.label.value = rendered.state.label;
    els.previewButton.textContent = rendered.state.label;
    if (els.previewDockButton) els.previewDockButton.textContent = rendered.state.label;
    els.summary.textContent = rendered.summary;
    if (els.previewDockSummary) els.previewDockSummary.textContent = rendered.summary;

    els.previewButton.style.cssText = '';
    updatePreviewStateStyles({
      baseDeclarations: rendered.baseDeclarations,
      customDeclarations: rendered.customDeclarations,
      hoverDeclarations: rendered.hoverDeclarations,
      activeDeclarations: rendered.activeDeclarations,
      focusDeclarations: rendered.focusDeclarations
    });

    els.exportOutput.value = rendered.cssOutput;
    setCustomStatus(rendered.invalidCustomLines.length);

    if (dirty) markSessionDirty();
    return rendered;
  };

  const applyPresetSelection = ({ presetId, clearCustomCss = false, dirty = false } = {}) => {
    const nextPresetId = PRESETS[presetId] ? presetId : DEFAULT_PRESET_ID;
    els.preset.value = nextPresetId;
    syncInputsFromPreset(nextPresetId);
    if (clearCustomCss) els.customCss.value = '';
    render({ dirty });
  };

  const copyCss = async () => {
    const css = String(els.exportOutput.value || '').trim();
    if (!css) {
      setExportStatus('No CSS to copy yet.', 'warn');
      return;
    }

    try {
      await navigator.clipboard.writeText(css);
      setExportStatus('CSS copied to clipboard.', 'success');
    } catch {
      setExportStatus('Clipboard access failed. Copy manually from the export box.', 'warn');
    }
  };

  const downloadCss = () => {
    const css = String(els.exportOutput.value || '').trim();
    if (!css) {
      setExportStatus('No CSS to download yet.', 'warn');
      return;
    }

    const filename = `${sanitizeClassName(els.className.value || 'my-button')}.css`;

    let blob;
    try {
      blob = new Blob([`${css}\n`], { type: 'text/css;charset=utf-8' });
    } catch {
      setExportStatus('Could not create CSS file in this browser.', 'warn');
      return;
    }

    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.rel = 'noopener';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setExportStatus(`Downloaded ${filename}.`, 'success');
  };

  const getToolInputs = () => {
    return {
      ...readStateFromInputs()
    };
  };

  const getToolSnapshotOutput = () => {
    const rendered = getRenderedState();
    const cssText = String(rendered.cssOutput || '').trim();
    const truncated = cssText.length > MAX_SESSION_CSS_CHARS;

    return {
      kind: 'text',
      text: truncated ? cssText.slice(0, MAX_SESSION_CSS_CHARS) : cssText,
      summary: rendered.summary,
      truncated,
      className: rendered.state.className
    };
  };

  const bindEvents = () => {
    els.form.addEventListener('input', (event) => {
      setExportStatus('');
      render({ dirty: Boolean(event?.isTrusted) });
    });

    els.form.addEventListener('change', (event) => {
      setExportStatus('');
      render({ dirty: Boolean(event?.isTrusted) });
    });

    els.form.addEventListener('submit', (event) => {
      event.preventDefault();
      render({ dirty: true });
    });

    els.preset.addEventListener('change', () => {
      applyPresetSelection({ presetId: els.preset.value, dirty: true });
    });

    els.applyPresetBtn.addEventListener('click', () => {
      applyPresetSelection({ presetId: els.preset.value, dirty: true });
    });

    els.resetDefaultsBtn.addEventListener('click', () => {
      els.preset.value = DEFAULT_PRESET_ID;
      applyPresetSelection({ presetId: DEFAULT_PRESET_ID, clearCustomCss: true, dirty: true });
    });

    els.copyCssBtn.addEventListener('click', () => {
      void copyCss();
    });

    els.downloadCssBtn.addEventListener('click', downloadCss);

    const root = document.getElementById('main');
    if (!root) return;

    root.addEventListener('tools:session-capture', (event) => {
      const detail = event?.detail || {};
      if (detail.toolId !== TOOL_ID) return;

      const payload = detail.payload || {};
      const output = getToolSnapshotOutput();
      const inputs = getToolInputs();

      payload.outputSummary = String(output.summary || payload.outputSummary || '').trim();
      payload.inputs = inputs;
      payload.output = output;

      if (detail.snapshot && typeof detail.snapshot === 'object') {
        detail.snapshot.output = output;
        detail.snapshot.inputs = inputs;
      }
    });

    root.addEventListener('tools:session-applied', (event) => {
      const detail = event?.detail || {};
      if (detail.toolId !== TOOL_ID) return;

      applyingSession = true;
      try {
        render({ dirty: false });
      } finally {
        applyingSession = false;
      }

      const output = detail?.snapshot?.output;
      const hasText = output && typeof output === 'object' && String(output.text || '').trim();
      if (hasText) setExportStatus('Session restored.', 'success');
    });
  };

  applyPresetSelection({ presetId: DEFAULT_PRESET_ID, dirty: false });
  setExportStatus('');
  bindEvents();
  initPreviewDockVisibility();
})();
