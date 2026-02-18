(() => {
  'use strict';

  const TOOL_ID = 'button-css-lab';
  const MAX_SESSION_CSS_CHARS = 120_000;

  const PRESETS = {
    starter: {
      background: '#2563eb',
      textColor: '#ffffff',
      borderColor: '#1d4ed8',
      borderWidth: 1,
      radius: 12,
      fontSize: 16,
      fontWeight: '500',
      paddingY: 12,
      paddingX: 20,
      letterSpacing: 0.1,
      shadow: '0 10px 22px rgba(37, 99, 235, 0.28)',
      transitionMs: 180
    },
    'pill-pop': {
      background: '#0f172a',
      textColor: '#f8fafc',
      borderColor: '#0f172a',
      borderWidth: 1,
      radius: 999,
      fontSize: 15,
      fontWeight: '600',
      paddingY: 11,
      paddingX: 22,
      letterSpacing: 0.2,
      shadow: '0 8px 18px rgba(15, 23, 42, 0.32)',
      transitionMs: 160
    },
    'outline-clean': {
      background: 'transparent',
      textColor: '#0f172a',
      borderColor: '#334155',
      borderWidth: 2,
      radius: 10,
      fontSize: 16,
      fontWeight: '600',
      paddingY: 11,
      paddingX: 20,
      letterSpacing: 0.15,
      shadow: 'none',
      transitionMs: 200
    },
    'dark-solid': {
      background: '#111827',
      textColor: '#f9fafb',
      borderColor: '#0f172a',
      borderWidth: 1,
      radius: 8,
      fontSize: 15,
      fontWeight: '500',
      paddingY: 10,
      paddingX: 18,
      letterSpacing: 0.25,
      shadow: '0 6px 14px rgba(17, 24, 39, 0.34)',
      transitionMs: 180
    },
    'sunset-gradient': {
      background: 'linear-gradient(135deg, #f97316 0%, #ec4899 100%)',
      textColor: '#ffffff',
      borderColor: '#fb7185',
      borderWidth: 1,
      radius: 14,
      fontSize: 16,
      fontWeight: '600',
      paddingY: 12,
      paddingX: 22,
      letterSpacing: 0.2,
      shadow: '0 12px 26px rgba(236, 72, 153, 0.3)',
      transitionMs: 220
    }
  };

  const DEFAULT_PRESET_ID = 'starter';

  const form = document.getElementById('buttonlab-form');
  const presetInput = document.getElementById('buttonlab-preset');
  const applyPresetBtn = document.getElementById('buttonlab-apply-preset');
  const resetDefaultsBtn = document.getElementById('buttonlab-reset-defaults');

  const labelInput = document.getElementById('buttonlab-label');
  const classInput = document.getElementById('buttonlab-class');

  const backgroundInput = document.getElementById('buttonlab-background');
  const textColorInput = document.getElementById('buttonlab-text-color');
  const borderColorInput = document.getElementById('buttonlab-border-color');
  const borderWidthInput = document.getElementById('buttonlab-border-width');
  const radiusInput = document.getElementById('buttonlab-radius');
  const fontSizeInput = document.getElementById('buttonlab-font-size');
  const fontWeightInput = document.getElementById('buttonlab-font-weight');
  const paddingYInput = document.getElementById('buttonlab-padding-y');
  const paddingXInput = document.getElementById('buttonlab-padding-x');
  const letterSpacingInput = document.getElementById('buttonlab-letter-spacing');
  const shadowInput = document.getElementById('buttonlab-shadow');
  const transitionInput = document.getElementById('buttonlab-transition');

  const customCssInput = document.getElementById('buttonlab-custom-css');
  const customStatusEl = document.getElementById('buttonlab-custom-status');

  const previewButton = document.getElementById('buttonlab-preview-button');
  const summaryEl = document.getElementById('buttonlab-summary');

  const exportOutput = document.getElementById('buttonlab-export');
  const exportStatusEl = document.getElementById('buttonlab-export-status');
  const copyCssBtn = document.getElementById('buttonlab-copy-css');
  const downloadCssBtn = document.getElementById('buttonlab-download-css');

  if (
    !form || !presetInput || !applyPresetBtn || !resetDefaultsBtn ||
    !labelInput || !classInput || !backgroundInput || !textColorInput || !borderColorInput ||
    !borderWidthInput || !radiusInput || !fontSizeInput || !fontWeightInput ||
    !paddingYInput || !paddingXInput || !letterSpacingInput || !shadowInput || !transitionInput ||
    !customCssInput || !customStatusEl || !previewButton || !summaryEl || !exportOutput ||
    !exportStatusEl || !copyCssBtn || !downloadCssBtn
  ) {
    return;
  }

  let applyingSession = false;

  const markSessionDirty = () => {
    if (applyingSession) return;
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const clampNumber = (value, min, max, fallback) => {
    const n = Number.parseFloat(String(value ?? '').trim());
    if (!Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
  };

  const toStringValue = (value, fallback = '') => {
    const next = String(value ?? '').trim();
    return next || fallback;
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

  const formatDecimal = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return '0';
    if (Math.abs(n - Math.round(n)) < 0.001) return String(Math.round(n));
    return String(n.toFixed(3)).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
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
    exportStatusEl.textContent = String(message || '');
    exportStatusEl.dataset.tone = tone;
  };

  const setCustomStatus = ({ invalidCount }) => {
    if (!invalidCount) {
      customStatusEl.textContent = 'Add declarations like transform: translateY(-1px); Property-value pairs are applied in order.';
      customStatusEl.dataset.tone = '';
      return;
    }
    customStatusEl.textContent = `${invalidCount} custom line${invalidCount === 1 ? '' : 's'} skipped. Use "property: value;" format.`;
    customStatusEl.dataset.tone = 'warn';
  };

  const buildBaseDeclarations = (state) => {
    const transitionMs = clampNumber(state.transitionMs, 0, 2000, 180);
    const declarations = [
      ['display', 'inline-flex'],
      ['align-items', 'center'],
      ['justify-content', 'center'],
      ['gap', '0.5rem'],
      ['font-family', '"Poppins", "Segoe UI", sans-serif'],
      ['font-size', `${formatDecimal(state.fontSize)}px`],
      ['font-weight', String(state.fontWeight)],
      ['line-height', '1.2'],
      ['letter-spacing', `${formatDecimal(state.letterSpacing)}px`],
      ['padding', `${formatDecimal(state.paddingY)}px ${formatDecimal(state.paddingX)}px`],
      ['color', state.textColor],
      ['background', state.background],
      ['border-width', `${formatDecimal(state.borderWidth)}px`],
      ['border-style', 'solid'],
      ['border-color', state.borderColor],
      ['border-radius', `${formatDecimal(state.radius)}px`],
      ['box-shadow', state.shadow],
      ['cursor', 'pointer'],
      ['transition', `all ${formatDecimal(transitionMs)}ms ease`]
    ];

    return declarations;
  };

  const readStateFromInputs = () => ({
    presetId: toStringValue(presetInput.value, DEFAULT_PRESET_ID),
    label: toStringValue(labelInput.value, 'Click me'),
    className: sanitizeClassName(classInput.value),
    background: toStringValue(backgroundInput.value, PRESETS.starter.background),
    textColor: toStringValue(textColorInput.value, PRESETS.starter.textColor),
    borderColor: toStringValue(borderColorInput.value, PRESETS.starter.borderColor),
    borderWidth: clampNumber(borderWidthInput.value, 0, 12, PRESETS.starter.borderWidth),
    radius: clampNumber(radiusInput.value, 0, 80, PRESETS.starter.radius),
    fontSize: clampNumber(fontSizeInput.value, 10, 42, PRESETS.starter.fontSize),
    fontWeight: toStringValue(fontWeightInput.value, PRESETS.starter.fontWeight),
    paddingY: clampNumber(paddingYInput.value, 0, 48, PRESETS.starter.paddingY),
    paddingX: clampNumber(paddingXInput.value, 0, 80, PRESETS.starter.paddingX),
    letterSpacing: clampNumber(letterSpacingInput.value, -2, 6, PRESETS.starter.letterSpacing),
    shadow: toStringValue(shadowInput.value, PRESETS.starter.shadow),
    transitionMs: clampNumber(transitionInput.value, 0, 2000, PRESETS.starter.transitionMs),
    customCss: String(customCssInput.value || '')
  });

  const syncInputsFromPreset = (presetId) => {
    const preset = PRESETS[presetId] || PRESETS[DEFAULT_PRESET_ID];

    backgroundInput.value = preset.background;
    textColorInput.value = preset.textColor;
    borderColorInput.value = preset.borderColor;
    borderWidthInput.value = String(preset.borderWidth);
    radiusInput.value = String(preset.radius);
    fontSizeInput.value = String(preset.fontSize);
    fontWeightInput.value = String(preset.fontWeight);
    paddingYInput.value = String(preset.paddingY);
    paddingXInput.value = String(preset.paddingX);
    letterSpacingInput.value = String(preset.letterSpacing);
    shadowInput.value = preset.shadow;
    transitionInput.value = String(preset.transitionMs);
  };

  const applyDeclarationsToPreview = (declarations) => {
    previewButton.style.cssText = '';
    declarations.forEach(([property, value]) => {
      previewButton.style.setProperty(property, value);
    });
  };

  const buildCssOutput = ({ className, declarations }) => {
    const lines = declarations.map(([property, value]) => `  ${property}: ${value};`);
    return `.${className} {\n${lines.join('\n')}\n}`;
  };

  const presetLabelForId = (presetId) => {
    const option = [...presetInput.options].find((entry) => entry.value === presetId);
    if (option) return String(option.textContent || presetId).trim();
    return presetId;
  };

  const getRenderedState = () => {
    const state = readStateFromInputs();
    const baseDeclarations = buildBaseDeclarations(state);
    const custom = parseCustomDeclarations(state.customCss);
    const combinedDeclarations = [...baseDeclarations, ...custom.declarations];
    const cssOutput = buildCssOutput({ className: state.className, declarations: combinedDeclarations });
    const summary = `Preset: ${presetLabelForId(state.presetId)} · ${formatDecimal(state.radius)}px radius · ${formatDecimal(state.fontSize)}px type`;

    return {
      state,
      baseDeclarations,
      customDeclarations: custom.declarations,
      invalidCustomLines: custom.invalid,
      combinedDeclarations,
      cssOutput,
      summary
    };
  };

  const render = ({ dirty = false } = {}) => {
    const rendered = getRenderedState();

    classInput.value = rendered.state.className;
    labelInput.value = rendered.state.label;
    previewButton.textContent = rendered.state.label;
    summaryEl.textContent = rendered.summary;

    applyDeclarationsToPreview(rendered.combinedDeclarations);
    exportOutput.value = rendered.cssOutput;

    setCustomStatus({ invalidCount: rendered.invalidCustomLines.length });
    if (dirty) markSessionDirty();

    return rendered;
  };

  const applyPresetSelection = ({ presetId, clearCustomCss = false, dirty = false } = {}) => {
    const safePresetId = PRESETS[presetId] ? presetId : DEFAULT_PRESET_ID;
    presetInput.value = safePresetId;
    syncInputsFromPreset(safePresetId);
    if (clearCustomCss) customCssInput.value = '';
    render({ dirty });
  };

  const copyCss = async () => {
    const css = String(exportOutput.value || '').trim();
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
    const css = String(exportOutput.value || '').trim();
    if (!css) {
      setExportStatus('No CSS to download yet.', 'warn');
      return;
    }

    const name = sanitizeClassName(classInput.value || 'my-button');
    const filename = `${name}.css`;

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
      presetId: state.presetId,
      label: state.label,
      className: state.className,
      background: state.background,
      textColor: state.textColor,
      borderColor: state.borderColor,
      borderWidth: state.borderWidth,
      radius: state.radius,
      fontSize: state.fontSize,
      fontWeight: state.fontWeight,
      paddingY: state.paddingY,
      paddingX: state.paddingX,
      letterSpacing: state.letterSpacing,
      shadow: state.shadow,
      transitionMs: state.transitionMs,
      customCss: state.customCss
    };
  };

  const getToolSnapshotOutput = () => {
    const rendered = getRenderedState();
    const cssText = String(rendered.cssOutput || '').trim();
    const truncated = cssText.length > MAX_SESSION_CSS_CHARS;
    const text = truncated ? cssText.slice(0, MAX_SESSION_CSS_CHARS) : cssText;

    return {
      kind: 'text',
      text,
      summary: rendered.summary,
      truncated,
      className: rendered.state.className
    };
  };

  const bindEvents = () => {
    form.addEventListener('input', (event) => {
      setExportStatus('');
      render({ dirty: Boolean(event?.isTrusted) });
    });

    form.addEventListener('change', (event) => {
      setExportStatus('');
      render({ dirty: Boolean(event?.isTrusted) });
    });

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      render({ dirty: true });
    });

    presetInput.addEventListener('change', () => {
      applyPresetSelection({ presetId: presetInput.value, dirty: true });
    });

    applyPresetBtn.addEventListener('click', () => {
      applyPresetSelection({ presetId: presetInput.value, dirty: true });
    });

    resetDefaultsBtn.addEventListener('click', () => {
      presetInput.value = DEFAULT_PRESET_ID;
      applyPresetSelection({ presetId: DEFAULT_PRESET_ID, clearCustomCss: true, dirty: true });
    });

    copyCssBtn.addEventListener('click', () => {
      void copyCss();
    });

    downloadCssBtn.addEventListener('click', downloadCss);

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
      if (hasText) {
        setExportStatus('Session restored.', 'success');
      }
    });
  };

  applyPresetSelection({ presetId: DEFAULT_PRESET_ID, dirty: false });
  setExportStatus('');
  bindEvents();
})();
