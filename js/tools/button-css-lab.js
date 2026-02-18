(() => {
  'use strict';

  const TOOL_ID = 'button-css-lab';
  const MAX_SESSION_CSS_CHARS = 120_000;
  const PREVIEW_STATE_STYLE_ID = 'buttonlab-preview-state-style';

  const BASE_PRESET = {
    background: '#2563eb',
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
    shadow: '0 10px 22px rgba(37, 99, 235, 0.28)',
    opacity: 1,
    transform: '',
    cursor: 'pointer',
    transitionProperty: 'all',
    transitionMs: 180,
    transitionEasing: 'ease',
    includeHover: true,
    hoverBackground: '#1d4ed8',
    hoverTextColor: '#ffffff',
    hoverBorderColor: '#1e40af',
    hoverShadow: '0 12px 26px rgba(29, 78, 216, 0.34)',
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
      background: '#0f172a',
      borderColor: '#0f172a',
      radius: 999,
      fontSize: 15,
      fontWeight: '600',
      paddingY: 11,
      paddingX: 22,
      letterSpacing: 0.2,
      shadow: '0 8px 18px rgba(15, 23, 42, 0.32)',
      hoverBackground: '#1e293b',
      hoverBorderColor: '#1e293b',
      hoverShadow: '0 10px 20px rgba(15, 23, 42, 0.38)',
      transitionMs: 160,
      focusRingColor: '#94a3b8'
    },
    'outline-clean': {
      ...BASE_PRESET,
      background: 'transparent',
      textColor: '#0f172a',
      borderColor: '#334155',
      borderWidth: 2,
      radius: 10,
      fontWeight: '600',
      shadow: 'none',
      hoverBackground: '#f8fafc',
      hoverTextColor: '#0f172a',
      hoverBorderColor: '#0f172a',
      hoverShadow: '0 8px 18px rgba(15, 23, 42, 0.16)',
      transitionMs: 200,
      focusRingColor: '#64748b'
    },
    'dark-solid': {
      ...BASE_PRESET,
      background: '#111827',
      textColor: '#f9fafb',
      borderColor: '#0f172a',
      radius: 8,
      fontSize: 15,
      fontWeight: '500',
      paddingY: 10,
      paddingX: 18,
      letterSpacing: 0.25,
      shadow: '0 6px 14px rgba(17, 24, 39, 0.34)',
      hoverBackground: '#1f2937',
      hoverBorderColor: '#1f2937',
      hoverShadow: '0 9px 18px rgba(17, 24, 39, 0.38)',
      transitionMs: 180,
      focusRingColor: '#9ca3af'
    },
    'sunset-gradient': {
      ...BASE_PRESET,
      background: 'linear-gradient(135deg, #f97316 0%, #ec4899 100%)',
      borderColor: '#fb7185',
      radius: 14,
      fontWeight: '600',
      paddingX: 22,
      letterSpacing: 0.2,
      shadow: '0 12px 26px rgba(236, 72, 153, 0.3)',
      hoverBackground: 'linear-gradient(135deg, #ea580c 0%, #db2777 100%)',
      hoverBorderColor: '#f43f5e',
      hoverShadow: '0 14px 28px rgba(219, 39, 119, 0.34)',
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
    shadow: getEl('buttonlab-shadow'),
    opacity: getEl('buttonlab-opacity'),
    transform: getEl('buttonlab-transform'),

    includeHover: getEl('buttonlab-include-hover'),
    hoverBackground: getEl('buttonlab-hover-background'),
    hoverTextColor: getEl('buttonlab-hover-text-color'),
    hoverBorderColor: getEl('buttonlab-hover-border-color'),
    hoverShadow: getEl('buttonlab-hover-shadow'),
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
    summary: getEl('buttonlab-summary'),
    exportOutput: getEl('buttonlab-export'),
    exportStatus: getEl('buttonlab-export-status'),
    copyCssBtn: getEl('buttonlab-copy-css'),
    downloadCssBtn: getEl('buttonlab-download-css')
  };

  const requiredEls = [
    'form', 'preset', 'applyPresetBtn', 'resetDefaultsBtn', 'label', 'className',
    'background', 'textColor', 'borderColor', 'borderWidth', 'borderStyle', 'radius', 'fontSize',
    'fontWeight', 'fontFamily', 'lineHeight', 'paddingY', 'paddingX', 'letterSpacing', 'minWidth',
    'textTransform', 'justifyContent', 'shadow', 'opacity', 'transform', 'includeHover',
    'hoverBackground', 'hoverTextColor', 'hoverBorderColor', 'hoverShadow', 'hoverLift',
    'includeActive', 'activeLift', 'includeFocus', 'focusRingColor', 'focusRingWidth',
    'cursor', 'transitionProperty', 'transitionMs', 'transitionEasing', 'customCss', 'customStatus',
    'previewButton', 'summary', 'exportOutput', 'exportStatus', 'copyCssBtn', 'downloadCssBtn'
  ];

  if (!requiredEls.every((key) => Boolean(els[key]))) return;

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
      els.customStatus.textContent = 'Add declarations like transform: translateY(-1px); Property-value pairs are applied in order.';
      els.customStatus.dataset.tone = '';
      return;
    }

    els.customStatus.textContent = `${invalidCount} custom line${invalidCount === 1 ? '' : 's'} skipped. Use "property: value;" format.`;
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

      background: toStringValue(els.background.value, defaults.background),
      textColor: toStringValue(els.textColor.value, defaults.textColor),
      borderColor: toStringValue(els.borderColor.value, defaults.borderColor),
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

      shadow: toStringValue(els.shadow.value, defaults.shadow),
      opacity: clampNumber(els.opacity.value, 0, 1, defaults.opacity),
      transform: String(els.transform.value || '').trim(),

      includeHover: Boolean(els.includeHover.checked),
      hoverBackground: toStringValue(els.hoverBackground.value, defaults.hoverBackground),
      hoverTextColor: toStringValue(els.hoverTextColor.value, defaults.hoverTextColor),
      hoverBorderColor: toStringValue(els.hoverBorderColor.value, defaults.hoverBorderColor),
      hoverShadow: toStringValue(els.hoverShadow.value, defaults.hoverShadow),
      hoverLift: clampNumber(els.hoverLift.value, -20, 20, defaults.hoverLift),

      includeActive: Boolean(els.includeActive.checked),
      activeLift: clampNumber(els.activeLift.value, -20, 20, defaults.activeLift),

      includeFocus: Boolean(els.includeFocus.checked),
      focusRingColor: toStringValue(els.focusRingColor.value, defaults.focusRingColor),
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

    els.background.value = preset.background;
    els.textColor.value = preset.textColor;
    els.borderColor.value = preset.borderColor;
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

    els.shadow.value = preset.shadow;
    els.opacity.value = String(preset.opacity);
    els.transform.value = preset.transform || '';

    els.includeHover.checked = Boolean(preset.includeHover);
    els.hoverBackground.value = preset.hoverBackground;
    els.hoverTextColor.value = preset.hoverTextColor;
    els.hoverBorderColor.value = preset.hoverBorderColor;
    els.hoverShadow.value = preset.hoverShadow;
    els.hoverLift.value = String(preset.hoverLift);

    els.includeActive.checked = Boolean(preset.includeActive);
    els.activeLift.value = String(preset.activeLift);

    els.includeFocus.checked = Boolean(preset.includeFocus);
    els.focusRingColor.value = preset.focusRingColor;
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
      ['background', state.background],
      ['border-width', `${formatDecimal(state.borderWidth)}px`],
      ['border-style', state.borderStyle],
      ['border-color', state.borderColor],
      ['border-radius', `${formatDecimal(state.radius)}px`],
      ['box-shadow', state.shadow],
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
      ['background', state.hoverBackground],
      ['color', state.hoverTextColor],
      ['border-color', state.hoverBorderColor],
      ['box-shadow', state.hoverShadow]
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

  const applyDeclarationsToPreview = (declarations) => {
    els.previewButton.style.cssText = '';
    declarations.forEach(([property, value]) => {
      els.previewButton.style.setProperty(property, value);
    });
  };

  const updatePreviewStateStyles = ({ hoverDeclarations, activeDeclarations, focusDeclarations }) => {
    const styleEl = ensurePreviewStateStyleEl();
    const blocks = [];

    const hoverBlock = toCssBlock('#buttonlab-preview-button:hover', hoverDeclarations);
    if (hoverBlock) blocks.push(hoverBlock);

    const activeBlock = toCssBlock('#buttonlab-preview-button:active', activeDeclarations);
    if (activeBlock) blocks.push(activeBlock);

    const focusBlock = toCssBlock('#buttonlab-preview-button:focus-visible', focusDeclarations);
    if (focusBlock) blocks.push(focusBlock);

    styleEl.textContent = blocks.join('\n\n');
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
    els.summary.textContent = rendered.summary;

    applyDeclarationsToPreview([...rendered.baseDeclarations, ...rendered.customDeclarations]);
    updatePreviewStateStyles({
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
    const state = readStateFromInputs();
    return {
      ...state
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
})();
