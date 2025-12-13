(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  const form = $('#qrtool-form');
  const dataInput = $('#qrtool-data');
  const exampleBtn = $('#qrtool-example');
  const clearBtn = $('#qrtool-clear');

  const canvas = $('#qrtool-canvas');
  const stage = $('#qrtool-stage');
  const emptyOverlay = $('#qrtool-empty');
  const metaEl = $('#qrtool-meta');
  const qualityEl = $('#qrtool-quality');
  const warningEl = $('#qrtool-warning');

  const dotStyleSelect = $('#qrtool-dot-style');
  const cornerStyleSelect = $('#qrtool-corner-style');
  const eccSelect = $('#qrtool-ecc');
  const marginInput = $('#qrtool-margin');
  const marginValue = $('#qrtool-margin-value');

  const fgInput = $('#qrtool-fg');
  const bgInput = $('#qrtool-bg');
  const fgLabel = $('#qrtool-fg-label');
  const bgLabel = $('#qrtool-bg-label');
  const transparentInput = $('#qrtool-transparent');

  const logoInput = $('#qrtool-logo');
  const logoRemoveBtn = $('#qrtool-logo-remove');
  const logoPreviewWrap = $('#qrtool-logo-preview-wrap');
  const logoPreview = $('#qrtool-logo-preview');
  const logoSizeInput = $('#qrtool-logo-size');
  const logoSizeValue = $('#qrtool-logo-size-value');
  const logoPaddingInput = $('#qrtool-logo-padding');
  const logoPaddingValue = $('#qrtool-logo-padding-value');

  const filenameInput = $('#qrtool-filename');
  const pngSizeSelect = $('#qrtool-png-size');
  const pdfSizeInput = $('#qrtool-pdf-size');
  const pdfUnitSelect = $('#qrtool-pdf-unit');

  const downloadPngBtn = $('#qrtool-download-png');
  const downloadSvgBtn = $('#qrtool-download-svg');
  const downloadPdfBtn = $('#qrtool-download-pdf');

  const templateButtons = Array.from(document.querySelectorAll('.qrtool-template'));

  if (!form || !dataInput || !canvas || !stage) return;

  const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
  if (!ctx) return;

  const DEFAULTS = Object.freeze({
    data: '',
    dotStyle: 'square',
    cornerStyle: 'square',
    ecc: 'M',
    marginModules: 4,
    fg: '#0B0F14',
    bg: '#FFFFFF',
    transparent: false,
    logoDataUrl: null,
    logoSizePct: 20,
    logoPaddingPct: 12,
  });

  const TEMPLATES = Object.freeze({
    'minimal-light': {
      dotStyle: 'square',
      cornerStyle: 'square',
      ecc: 'M',
      fg: '#0B0F14',
      bg: '#FFFFFF',
      transparent: false,
    },
    'modern-navy': {
      dotStyle: 'rounded',
      cornerStyle: 'rounded',
      ecc: 'M',
      fg: '#0B2D4A',
      bg: '#F5F7FA',
      transparent: false,
    },
    'rounded-teal': {
      dotStyle: 'rounded',
      cornerStyle: 'extra-rounded',
      ecc: 'M',
      fg: '#0F6D77',
      bg: '#FFFFFF',
      transparent: false,
    },
    'bold-black': {
      dotStyle: 'square',
      cornerStyle: 'square',
      ecc: 'Q',
      fg: '#000000',
      bg: '#FFFFFF',
      transparent: false,
    },
    'high-contrast': {
      dotStyle: 'square',
      cornerStyle: 'square',
      ecc: 'H',
      fg: '#000000',
      bg: '#FFFFFF',
      transparent: false,
    },
    'dark-premium': {
      dotStyle: 'rounded',
      cornerStyle: 'rounded',
      ecc: 'H',
      fg: '#EAF2FF',
      bg: '#0D1117',
      transparent: false,
    },
  });

  const state = {
    ...DEFAULTS,
    logoImage: null,
    qr: null,
    moduleCount: 0,
    darkModules: null,
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const toInt = (value, fallback) => {
    const parsed = parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const toFloat = (value, fallback) => {
    const parsed = parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const normalizeHex = (hex) => {
    if (!hex) return '#000000';
    const v = hex.trim().toUpperCase();
    if (/^#[0-9A-F]{6}$/.test(v)) return v;
    if (/^#[0-9A-F]{3}$/.test(v)) {
      return `#${v[1]}${v[1]}${v[2]}${v[2]}${v[3]}${v[3]}`;
    }
    return '#000000';
  };

  const hexToRgb = (hex) => {
    const v = normalizeHex(hex).slice(1);
    return {
      r: parseInt(v.slice(0, 2), 16),
      g: parseInt(v.slice(2, 4), 16),
      b: parseInt(v.slice(4, 6), 16),
    };
  };

  const rgbToCss = ({ r, g, b }) => `rgb(${r} ${g} ${b})`;

  const srgbToLinear = (c) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };

  const relativeLuminance = (rgb) => {
    const r = srgbToLinear(rgb.r);
    const g = srgbToLinear(rgb.g);
    const b = srgbToLinear(rgb.b);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const contrastRatio = (fg, bg) => {
    const l1 = relativeLuminance(fg);
    const l2 = relativeLuminance(bg);
    const hi = Math.max(l1, l2);
    const lo = Math.min(l1, l2);
    return (hi + 0.05) / (lo + 0.05);
  };

  const debounce = (fn, waitMs) => {
    let t = 0;
    return (...args) => {
      window.clearTimeout(t);
      t = window.setTimeout(() => fn(...args), waitMs);
    };
  };

  const fileNameSafe = (name) => (name || 'qr-code')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9._-]+/gi, '')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'qr-code';

  const downloadBlob = (blob, filename) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1500);
  };

  const setWarning = (text) => {
    if (!warningEl) return;
    if (!text) {
      warningEl.hidden = true;
      warningEl.textContent = '';
      return;
    }
    warningEl.hidden = false;
    warningEl.textContent = text;
  };

  const setQuality = (text) => {
    if (!qualityEl) return;
    if (!text) {
      qualityEl.hidden = true;
      qualityEl.textContent = '';
      return;
    }
    qualityEl.hidden = false;
    qualityEl.textContent = text;
  };

  const isFinderArea = (row, col, size) => {
    const top = row < 7;
    const left = col < 7;
    const right = col >= size - 7;
    const bottom = row >= size - 7;
    return (top && left) || (top && right) || (bottom && left);
  };

  const pathRoundedRect = (targetCtx, x, y, w, h, r) => {
    const radius = clamp(r, 0, Math.min(w, h) / 2);
    if (!radius) {
      targetCtx.rect(x, y, w, h);
      return;
    }
    const xr = x + w;
    const yb = y + h;
    targetCtx.moveTo(x + radius, y);
    targetCtx.arcTo(xr, y, xr, yb, radius);
    targetCtx.arcTo(xr, yb, x, yb, radius);
    targetCtx.arcTo(x, yb, x, y, radius);
    targetCtx.arcTo(x, y, xr, y, radius);
    targetCtx.closePath();
  };

  const drawFinderPattern = (targetCtx, x, y, moduleSize, cornerStyle, fgCss, bgCss) => {
    const outer = moduleSize * 7;
    const mid = moduleSize * 5;
    const inner = moduleSize * 3;

    let outerRadius = 0;
    if (cornerStyle === 'rounded') outerRadius = moduleSize * 2.1;
    if (cornerStyle === 'extra-rounded') outerRadius = moduleSize * 3;
    outerRadius = clamp(outerRadius, 0, outer / 2);

    const innerRadius = clamp(outerRadius * 0.65, 0, inner / 2);
    const midRadius = clamp(outerRadius * 0.75, 0, mid / 2);

    targetCtx.fillStyle = fgCss;
    targetCtx.beginPath();
    pathRoundedRect(targetCtx, x, y, outer, outer, outerRadius);
    targetCtx.fill();

    targetCtx.fillStyle = bgCss;
    targetCtx.beginPath();
    pathRoundedRect(targetCtx, x + moduleSize, y + moduleSize, mid, mid, midRadius);
    targetCtx.fill();

    targetCtx.fillStyle = fgCss;
    targetCtx.beginPath();
    pathRoundedRect(targetCtx, x + moduleSize * 2, y + moduleSize * 2, inner, inner, innerRadius);
    targetCtx.fill();
  };

  const computeLogoBox = (moduleCount, logoSizePct, logoPaddingPct) => {
    const sizeModules = clamp(Math.round((moduleCount * logoSizePct) / 100), 6, Math.floor(moduleCount * 0.3));
    const sizeAligned = (sizeModules % 2 === moduleCount % 2) ? sizeModules : sizeModules + 1;
    const paddingModules = clamp(Math.round((sizeAligned * logoPaddingPct) / 100), 1, 10);
    const boxModules = sizeAligned + paddingModules * 2;
    return { sizeModules: sizeAligned, paddingModules, boxModules };
  };

  const renderQrToCanvas = (targetCanvas, qrData, options) => {
    const { darkModules, moduleCount } = qrData;
    const stageRect = stage.getBoundingClientRect();
    const targetRect = targetCanvas === canvas ? stageRect : { width: targetCanvas.width, height: targetCanvas.height };
    const dpr = targetCanvas === canvas ? Math.min(2, window.devicePixelRatio || 1) : 1;

    const sizePx = Math.max(240, Math.round(Math.min(targetRect.width, targetRect.height) * dpr));
    if (targetCanvas.width !== sizePx || targetCanvas.height !== sizePx) {
      targetCanvas.width = sizePx;
      targetCanvas.height = sizePx;
    }

    const targetCtx = targetCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!targetCtx) return;

    const quiet = clamp(options.marginModules, 2, 10);
    const totalModules = moduleCount + quiet * 2;
    const moduleSize = Math.max(1, Math.floor(sizePx / totalModules));
    const codeSize = moduleSize * totalModules;
    const offset = Math.floor((sizePx - codeSize) / 2);
    const qrOrigin = offset + quiet * moduleSize;

    const fgRgb = hexToRgb(options.fg);
    const bgRgb = hexToRgb(options.bg);
    const fgCss = rgbToCss(fgRgb);
    const bgCss = rgbToCss(bgRgb);
    const transparent = !!options.transparent;

    targetCtx.setTransform(1, 0, 0, 1, 0, 0);
    targetCtx.clearRect(0, 0, sizePx, sizePx);
    if (!transparent) {
      targetCtx.fillStyle = bgCss;
      targetCtx.fillRect(0, 0, sizePx, sizePx);
    }

    const drawBackground = transparent ? '#FFFFFF' : bgCss;

    targetCtx.imageSmoothingEnabled = false;

    const drawSquareModules = () => {
      const useRuns = options.dotStyle === 'square';
      if (!useRuns) return;

      targetCtx.fillStyle = fgCss;
      for (let row = 0; row < moduleCount; row++) {
        let runStart = -1;
        for (let col = 0; col <= moduleCount; col++) {
          const idx = row * moduleCount + col;
          const isDark = col < moduleCount && darkModules[idx] === 1 && !isFinderArea(row, col, moduleCount);
          if (isDark) {
            if (runStart === -1) runStart = col;
            continue;
          }
          if (runStart !== -1) {
            const runLen = col - runStart;
            const x = qrOrigin + runStart * moduleSize;
            const y = qrOrigin + row * moduleSize;
            targetCtx.fillRect(x, y, runLen * moduleSize, moduleSize);
            runStart = -1;
          }
        }
      }
    };

    const drawStyledModules = () => {
      const style = options.dotStyle;
      if (style === 'square') {
        drawSquareModules();
        return;
      }

      const radius = style === 'rounded' ? moduleSize * 0.38 : 0;
      const dotRadius = moduleSize * 0.46;

      targetCtx.fillStyle = fgCss;
      targetCtx.beginPath();
      for (let row = 0; row < moduleCount; row++) {
        for (let col = 0; col < moduleCount; col++) {
          if (isFinderArea(row, col, moduleCount)) continue;
          const idx = row * moduleCount + col;
          if (darkModules[idx] !== 1) continue;
          const x = qrOrigin + col * moduleSize;
          const y = qrOrigin + row * moduleSize;
          if (style === 'dots') {
            targetCtx.moveTo(x + moduleSize / 2 + dotRadius, y + moduleSize / 2);
            targetCtx.arc(x + moduleSize / 2, y + moduleSize / 2, dotRadius, 0, Math.PI * 2);
          } else {
            pathRoundedRect(targetCtx, x, y, moduleSize, moduleSize, radius);
          }
        }
      }
      targetCtx.fill();
    };

    drawStyledModules();

    const finders = [
      { col: 0, row: 0 },
      { col: moduleCount - 7, row: 0 },
      { col: 0, row: moduleCount - 7 },
    ];
    finders.forEach(({ col, row }) => {
      const x = qrOrigin + col * moduleSize;
      const y = qrOrigin + row * moduleSize;
      drawFinderPattern(targetCtx, x, y, moduleSize, options.cornerStyle, fgCss, drawBackground);
    });

    if (options.logoImage) {
      const { sizeModules, paddingModules } = computeLogoBox(moduleCount, options.logoSizePct, options.logoPaddingPct);
      const logoPlateModules = sizeModules + paddingModules * 2;
      const logoStart = Math.floor((moduleCount - logoPlateModules) / 2);
      const plateX = qrOrigin + logoStart * moduleSize;
      const plateY = qrOrigin + logoStart * moduleSize;
      const plateSize = logoPlateModules * moduleSize;

      const plateRadius = moduleSize * 2.1;
      targetCtx.fillStyle = drawBackground;
      targetCtx.beginPath();
      pathRoundedRect(targetCtx, plateX, plateY, plateSize, plateSize, plateRadius);
      targetCtx.fill();

      const logoSizePx = sizeModules * moduleSize;
      const logoX = plateX + paddingModules * moduleSize;
      const logoY = plateY + paddingModules * moduleSize;

      targetCtx.imageSmoothingEnabled = true;
      targetCtx.imageSmoothingQuality = 'high';

      const img = options.logoImage;
      const scale = Math.min(logoSizePx / img.naturalWidth, logoSizePx / img.naturalHeight);
      const drawW = img.naturalWidth * scale;
      const drawH = img.naturalHeight * scale;
      const dx = logoX + (logoSizePx - drawW) / 2;
      const dy = logoY + (logoSizePx - drawH) / 2;
      targetCtx.drawImage(img, dx, dy, drawW, drawH);
    }
  };

  const buildDarkModules = (qrInstance) => {
    const moduleCount = qrInstance.getModuleCount();
    const dark = new Uint8Array(moduleCount * moduleCount);
    for (let r = 0; r < moduleCount; r++) {
      for (let c = 0; c < moduleCount; c++) {
        dark[r * moduleCount + c] = qrInstance.isDark(r, c) ? 1 : 0;
      }
    }
    return { moduleCount, darkModules: dark };
  };

  const svgEsc = (s) => String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');

  const svgRoundedRect = (x, y, w, h, r) => {
    const rr = clamp(r, 0, Math.min(w, h) / 2);
    if (!rr) return `M${x} ${y}h${w}v${h}h${-w}Z`;
    const xr = x + w;
    const yb = y + h;
    return [
      `M${x + rr} ${y}`,
      `L${xr - rr} ${y}`,
      `Q${xr} ${y} ${xr} ${y + rr}`,
      `L${xr} ${yb - rr}`,
      `Q${xr} ${yb} ${xr - rr} ${yb}`,
      `L${x + rr} ${yb}`,
      `Q${x} ${yb} ${x} ${yb - rr}`,
      `L${x} ${y + rr}`,
      `Q${x} ${y} ${x + rr} ${y}`,
      'Z',
    ].join('');
  };

  const buildSvg = (qrData, options, exportPx) => {
    const { darkModules, moduleCount } = qrData;
    const quiet = clamp(options.marginModules, 2, 10);
    const totalModules = moduleCount + quiet * 2;
    const sizePx = clamp(exportPx, 128, 8192);

    const fg = normalizeHex(options.fg);
    const bg = normalizeHex(options.bg);
    const transparent = !!options.transparent;
    const bgFill = transparent ? 'none' : bg;
    const plateBg = transparent ? '#FFFFFF' : bg;

    const parts = [];
    parts.push('<?xml version="1.0" encoding="UTF-8"?>');
    parts.push(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${sizePx}" height="${sizePx}" viewBox="0 0 ${totalModules} ${totalModules}" role="img" aria-label="QR code">`
    );
    parts.push(`<rect width="100%" height="100%" fill="${bgFill}"/>`);

    const finderPositions = [
      { x: quiet, y: quiet },
      { x: quiet + moduleCount - 7, y: quiet },
      { x: quiet, y: quiet + moduleCount - 7 },
    ];

    const isFinderModule = (r, c) => isFinderArea(r, c, moduleCount);

    const dotStyle = options.dotStyle;
    if (dotStyle === 'square') {
      const path = [];
      for (let row = 0; row < moduleCount; row++) {
        let runStart = -1;
        for (let col = 0; col <= moduleCount; col++) {
          const idx = row * moduleCount + col;
          const draw = col < moduleCount && darkModules[idx] === 1 && !isFinderModule(row, col);
          if (draw) {
            if (runStart === -1) runStart = col;
            continue;
          }
          if (runStart !== -1) {
            const runLen = col - runStart;
            const x = quiet + runStart;
            const y = quiet + row;
            path.push(`M${x} ${y}h${runLen}v1h${-runLen}Z`);
            runStart = -1;
          }
        }
      }
      parts.push(`<path d="${path.join('')}" fill="${fg}" shape-rendering="crispEdges"/>`);
    } else if (dotStyle === 'dots') {
      const circles = [];
      const r = 0.46;
      for (let row = 0; row < moduleCount; row++) {
        for (let col = 0; col < moduleCount; col++) {
          if (isFinderModule(row, col)) continue;
          const idx = row * moduleCount + col;
          if (darkModules[idx] !== 1) continue;
          const cx = quiet + col + 0.5;
          const cy = quiet + row + 0.5;
          circles.push(`<circle cx="${cx}" cy="${cy}" r="${r}"/>`);
        }
      }
      parts.push(`<g fill="${fg}">${circles.join('')}</g>`);
    } else {
      const rects = [];
      const rr = 0.38;
      for (let row = 0; row < moduleCount; row++) {
        for (let col = 0; col < moduleCount; col++) {
          if (isFinderModule(row, col)) continue;
          const idx = row * moduleCount + col;
          if (darkModules[idx] !== 1) continue;
          rects.push(`<rect x="${quiet + col}" y="${quiet + row}" width="1" height="1" rx="${rr}" ry="${rr}"/>`);
        }
      }
      parts.push(`<g fill="${fg}">${rects.join('')}</g>`);
    }

    let outerR = 0;
    if (options.cornerStyle === 'rounded') outerR = 2.1;
    if (options.cornerStyle === 'extra-rounded') outerR = 3;
    outerR = clamp(outerR, 0, 3.5);
    const midR = clamp(outerR * 0.75, 0, 2.5);
    const innerR = clamp(outerR * 0.65, 0, 1.5);

    finderPositions.forEach(({ x, y }) => {
      parts.push(`<path d="${svgRoundedRect(x, y, 7, 7, outerR)}" fill="${fg}"/>`);
      parts.push(`<path d="${svgRoundedRect(x + 1, y + 1, 5, 5, midR)}" fill="${plateBg}"/>`);
      parts.push(`<path d="${svgRoundedRect(x + 2, y + 2, 3, 3, innerR)}" fill="${fg}"/>`);
    });

    if (options.logoDataUrl) {
      const { sizeModules, paddingModules } = computeLogoBox(moduleCount, options.logoSizePct, options.logoPaddingPct);
      const plateModules = sizeModules + paddingModules * 2;
      const start = quiet + Math.floor((moduleCount - plateModules) / 2);
      const plateR = 2.1;
      parts.push(`<path d="${svgRoundedRect(start, start, plateModules, plateModules, plateR)}" fill="${plateBg}"/>`);

      const logoX = start + paddingModules;
      const logoSize = sizeModules;
      const href = svgEsc(options.logoDataUrl);
      parts.push(
        `<image href="${href}" x="${logoX}" y="${logoX}" width="${logoSize}" height="${logoSize}" preserveAspectRatio="xMidYMid meet"/>`
      );
    }

    parts.push('</svg>');
    return parts.join('\n');
  };

  const pdfColor = (hex) => {
    const rgb = hexToRgb(hex);
    return {
      r: (rgb.r / 255).toFixed(4),
      g: (rgb.g / 255).toFixed(4),
      b: (rgb.b / 255).toFixed(4),
    };
  };

  const pdfCirclePath = (cx, cy, r) => {
    const k = 0.552284749831;
    const ox = r * k;
    const oy = r * k;
    const x0 = cx - r;
    const x1 = cx - r + ox;
    const x2 = cx - ox;
    const x3 = cx;
    const x4 = cx + ox;
    const x5 = cx + r - ox;
    const x6 = cx + r;
    const y0 = cy - r;
    const y1 = cy - r + oy;
    const y2 = cy - oy;
    const y3 = cy;
    const y4 = cy + oy;
    const y5 = cy + r - oy;
    const y6 = cy + r;
    return [
      `${x3} ${y6} m`,
      `${x4} ${y6} ${x6} ${y4} ${x6} ${y3} c`,
      `${x6} ${y2} ${x4} ${y0} ${x3} ${y0} c`,
      `${x2} ${y0} ${x0} ${y2} ${x0} ${y3} c`,
      `${x0} ${y4} ${x2} ${y6} ${x3} ${y6} c`,
      'h',
    ].join('\n');
  };

  const pdfRoundedRectPath = (x, y, w, h, r) => {
    const rr = clamp(r, 0, Math.min(w, h) / 2);
    if (!rr) return `${x} ${y} ${w} ${h} re`;
    const k = 0.552284749831;
    const ox = rr * k;
    const oy = rr * k;
    const x0 = x;
    const x1 = x + rr;
    const x2 = x + w - rr;
    const x3 = x + w;
    const y0 = y;
    const y1 = y + rr;
    const y2 = y + h - rr;
    const y3 = y + h;
    return [
      `${x1} ${y0} m`,
      `${x2} ${y0} l`,
      `${x2 + ox} ${y0} ${x3} ${y1 - oy} ${x3} ${y1} c`,
      `${x3} ${y2} l`,
      `${x3} ${y2 + oy} ${x2 + ox} ${y3} ${x2} ${y3} c`,
      `${x1} ${y3} l`,
      `${x1 - ox} ${y3} ${x0} ${y2 + oy} ${x0} ${y2} c`,
      `${x0} ${y1} l`,
      `${x0} ${y1 - oy} ${x1 - ox} ${y0} ${x1} ${y0} c`,
      'h',
    ].join('\n');
  };

  const base64ToBytes = (b64) => {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  };

  const buildPdf = (qrData, options, pdfSizePt) => {
    const { darkModules, moduleCount } = qrData;
    const quiet = clamp(options.marginModules, 2, 10);
    const totalModules = moduleCount + quiet * 2;
    const pagePt = clamp(pdfSizePt, 72 * 0.5, 72 * 20);
    const cell = pagePt / totalModules;

    const fg = pdfColor(options.fg);
    const bg = pdfColor(options.bg);
    const transparent = !!options.transparent;
    const plateBgHex = transparent ? '#FFFFFF' : options.bg;
    const plateBg = pdfColor(plateBgHex);

    const resources = [];
    let imgObj = null;
    let imgWidth = 0;
    let imgHeight = 0;
    let imgBytes = null;

    if (options.logoImage) {
      const tmp = document.createElement('canvas');
      tmp.width = 512;
      tmp.height = 512;
      const tctx = tmp.getContext('2d');
      if (tctx) {
        tctx.fillStyle = normalizeHex(plateBgHex);
        tctx.fillRect(0, 0, tmp.width, tmp.height);
        tctx.imageSmoothingEnabled = true;
        tctx.imageSmoothingQuality = 'high';
        const img = options.logoImage;
        const scale = Math.min(tmp.width / img.naturalWidth, tmp.height / img.naturalHeight);
        const dw = img.naturalWidth * scale;
        const dh = img.naturalHeight * scale;
        tctx.drawImage(img, (tmp.width - dw) / 2, (tmp.height - dh) / 2, dw, dh);
        const dataUrl = tmp.toDataURL('image/jpeg', 0.92);
        const b64 = dataUrl.split(',')[1] || '';
        imgBytes = base64ToBytes(b64);
        imgWidth = tmp.width;
        imgHeight = tmp.height;
      }
    }

    const content = [];
    content.push('q');
    content.push('1 0 0 1 0 0 cm');
    if (!transparent) {
      content.push(`${bg.r} ${bg.g} ${bg.b} rg`);
      content.push(`0 0 ${pagePt.toFixed(3)} ${pagePt.toFixed(3)} re f`);
    }

    content.push(`${fg.r} ${fg.g} ${fg.b} rg`);

    const finderOrigins = [
      { col: 0, row: 0 },
      { col: moduleCount - 7, row: 0 },
      { col: 0, row: moduleCount - 7 },
    ];

    const isFinderModule = (r, c) => isFinderArea(r, c, moduleCount);

    const moduleToPdfX = (col) => (col + quiet) * cell;
    const moduleToPdfY = (row) => pagePt - (row + quiet + 1) * cell;

    const dotStyle = options.dotStyle;
    if (dotStyle === 'square') {
      for (let row = 0; row < moduleCount; row++) {
        let runStart = -1;
        for (let col = 0; col <= moduleCount; col++) {
          const idx = row * moduleCount + col;
          const draw = col < moduleCount && darkModules[idx] === 1 && !isFinderModule(row, col);
          if (draw) {
            if (runStart === -1) runStart = col;
            continue;
          }
          if (runStart !== -1) {
            const runLen = col - runStart;
            const x = moduleToPdfX(runStart);
            const y = moduleToPdfY(row);
            const w = runLen * cell;
            content.push(`${x.toFixed(3)} ${y.toFixed(3)} ${w.toFixed(3)} ${cell.toFixed(3)} re`);
            runStart = -1;
          }
        }
      }
      content.push('f');
    } else if (dotStyle === 'dots') {
      const r = cell * 0.46;
      for (let row = 0; row < moduleCount; row++) {
        for (let col = 0; col < moduleCount; col++) {
          if (isFinderModule(row, col)) continue;
          const idx = row * moduleCount + col;
          if (darkModules[idx] !== 1) continue;
          const cx = moduleToPdfX(col) + cell / 2;
          const cy = moduleToPdfY(row) + cell / 2;
          content.push(pdfCirclePath(cx, cy, r));
          content.push('f');
        }
      }
    } else {
      const rr = cell * 0.38;
      for (let row = 0; row < moduleCount; row++) {
        for (let col = 0; col < moduleCount; col++) {
          if (isFinderModule(row, col)) continue;
          const idx = row * moduleCount + col;
          if (darkModules[idx] !== 1) continue;
          const x = moduleToPdfX(col);
          const y = moduleToPdfY(row);
          content.push(pdfRoundedRectPath(x, y, cell, cell, rr));
          content.push('f');
        }
      }
    }

    const cornerStyle = options.cornerStyle;
    let outerR = 0;
    if (cornerStyle === 'rounded') outerR = cell * 2.1;
    if (cornerStyle === 'extra-rounded') outerR = cell * 3;
    outerR = clamp(outerR, 0, cell * 3.5);
    const midR = clamp(outerR * 0.75, 0, cell * 2.5);
    const innerR = clamp(outerR * 0.65, 0, cell * 1.5);

    finderOrigins.forEach(({ row, col }) => {
      const x = moduleToPdfX(col);
      const y = moduleToPdfY(row + 6);
      const outerSize = cell * 7;
      const midSize = cell * 5;
      const innerSize = cell * 3;

      content.push(`${fg.r} ${fg.g} ${fg.b} rg`);
      content.push(pdfRoundedRectPath(x, y, outerSize, outerSize, outerR));
      content.push('f');

      content.push(`${plateBg.r} ${plateBg.g} ${plateBg.b} rg`);
      content.push(pdfRoundedRectPath(x + cell, y + cell, midSize, midSize, midR));
      content.push('f');

      content.push(`${fg.r} ${fg.g} ${fg.b} rg`);
      content.push(pdfRoundedRectPath(x + cell * 2, y + cell * 2, innerSize, innerSize, innerR));
      content.push('f');
    });

    if (options.logoImage) {
      const { sizeModules, paddingModules } = computeLogoBox(moduleCount, options.logoSizePct, options.logoPaddingPct);
      const plateModules = sizeModules + paddingModules * 2;
      const start = Math.floor((moduleCount - plateModules) / 2);
      const x = moduleToPdfX(start);
      const y = moduleToPdfY(start + plateModules - 1);
      const plateSize = plateModules * cell;
      const plateRadius = cell * 2.1;

      content.push(`${plateBg.r} ${plateBg.g} ${plateBg.b} rg`);
      content.push(pdfRoundedRectPath(x, y, plateSize, plateSize, plateRadius));
      content.push('f');

      if (imgBytes && imgBytes.length) {
        const logoX = moduleToPdfX(start + paddingModules);
        const logoY = moduleToPdfY(start + paddingModules + sizeModules - 1);
        const logoSize = sizeModules * cell;
        imgObj = 'Im0';
        resources.push(`/XObject << /${imgObj} 5 0 R >>`);
        content.push('q');
        content.push(`${logoSize.toFixed(3)} 0 0 ${logoSize.toFixed(3)} ${logoX.toFixed(3)} ${logoY.toFixed(3)} cm`);
        content.push(`/${imgObj} Do`);
        content.push('Q');
      }
    }

    content.push('Q');
    const contentStr = content.join('\n') + '\n';

    const encoder = new TextEncoder();
    const parts = [];
    let offset = 0;
    const offsets = [0];

    const pushBytes = (bytes) => {
      parts.push(bytes);
      offset += bytes.length;
    };

    const pushStr = (str) => pushBytes(encoder.encode(str));

    const pushHeader = () => {
      pushStr('%PDF-1.4\n');
      pushBytes(new Uint8Array([0x25, 0xE2, 0xE3, 0xCF, 0xD3, 0x0A]));
    };

    const startObj = (id) => {
      offsets[id] = offset;
      pushStr(`${id} 0 obj\n`);
    };

    const endObj = () => pushStr('endobj\n');

    pushHeader();

    startObj(1);
    pushStr('<< /Type /Catalog /Pages 2 0 R >>\n');
    endObj();

    startObj(2);
    pushStr('<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n');
    endObj();

    startObj(3);
    const resourceDict = resources.length ? `<< ${resources.join(' ')} >>` : '<< >>';
    pushStr(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pagePt.toFixed(3)} ${pagePt.toFixed(3)}] /Resources ${resourceDict} /Contents 4 0 R >>\n`);
    endObj();

    startObj(4);
    const contentBytes = encoder.encode(contentStr);
    pushStr(`<< /Length ${contentBytes.length} >>\nstream\n`);
    pushBytes(contentBytes);
    pushStr('endstream\n');
    endObj();

    if (imgBytes && imgBytes.length) {
      startObj(5);
      pushStr(`<< /Type /XObject /Subtype /Image /Width ${imgWidth} /Height ${imgHeight} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${imgBytes.length} >>\nstream\n`);
      pushBytes(imgBytes);
      pushStr('\nendstream\n');
      endObj();
    }

    const objCount = imgBytes && imgBytes.length ? 5 : 4;
    const xrefStart = offset;
    pushStr(`xref\n0 ${objCount + 1}\n`);
    pushStr('0000000000 65535 f \n');
    for (let i = 1; i <= objCount; i++) {
      const off = String(offsets[i] || 0).padStart(10, '0');
      pushStr(`${off} 00000 n \n`);
    }
    pushStr(`trailer\n<< /Size ${objCount + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`);

    const totalLen = parts.reduce((sum, p) => sum + p.length, 0);
    const out = new Uint8Array(totalLen);
    let ptr = 0;
    parts.forEach((p) => {
      out.set(p, ptr);
      ptr += p.length;
    });
    return out;
  };

  const updateControlsFromState = () => {
    dotStyleSelect.value = state.dotStyle;
    cornerStyleSelect.value = state.cornerStyle;
    eccSelect.value = state.ecc;
    marginInput.value = state.marginModules;
    marginValue.textContent = String(state.marginModules);

    fgInput.value = state.fg.toLowerCase();
    bgInput.value = state.bg.toLowerCase();
    fgLabel.textContent = normalizeHex(state.fg);
    bgLabel.textContent = normalizeHex(state.bg);
    transparentInput.checked = state.transparent;

    logoSizeInput.value = String(state.logoSizePct);
    logoSizeValue.textContent = `${state.logoSizePct}%`;
    logoPaddingInput.value = String(state.logoPaddingPct);
    logoPaddingValue.textContent = `${state.logoPaddingPct}%`;

    if (state.logoDataUrl && state.logoImage && logoPreview) {
      logoPreview.src = state.logoDataUrl;
      logoPreviewWrap?.classList.remove('hide');
      logoPreviewWrap?.setAttribute('aria-hidden', 'false');
      logoRemoveBtn.disabled = false;
      logoSizeInput.disabled = false;
      logoPaddingInput.disabled = false;
    } else {
      logoPreviewWrap?.classList.add('hide');
      logoPreviewWrap?.setAttribute('aria-hidden', 'true');
      logoRemoveBtn.disabled = true;
      logoSizeInput.disabled = true;
      logoPaddingInput.disabled = true;
    }
  };

  const render = () => {
    const data = (state.data || '').trim();
    if (!data) {
      emptyOverlay?.classList.remove('hide');
      metaEl.textContent = 'Enter a URL to generate a QR code.';
      setQuality('');
      setWarning('');
      downloadPngBtn.disabled = true;
      downloadSvgBtn.disabled = true;
      downloadPdfBtn.disabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    let qrInstance;
    try {
      const encoder = window.qrcode;
      if (typeof encoder !== 'function') throw new Error('QR encoder missing');
      if (encoder.stringToBytesFuncs && encoder.stringToBytesFuncs['UTF-8']) {
        encoder.stringToBytes = encoder.stringToBytesFuncs['UTF-8'];
      }
      qrInstance = encoder(0, state.ecc);
      qrInstance.addData(data);
      qrInstance.make();
    } catch (err) {
      const isMissing = String(err && (err.message || err)).includes('QR encoder missing');
      emptyOverlay?.classList.remove('hide');
      metaEl.textContent = isMissing
        ? 'QR encoder failed to load.'
        : 'That URL is too long for a QR code at the selected settings.';
      setQuality(isMissing ? 'Unavailable' : 'Too long');
      setWarning(
        isMissing
          ? 'Refresh the page. If the issue persists, your browser may be blocking required scripts.'
          : 'Try shortening the URL or lowering the error correction level.'
      );
      downloadPngBtn.disabled = true;
      downloadSvgBtn.disabled = true;
      downloadPdfBtn.disabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const { moduleCount, darkModules } = buildDarkModules(qrInstance);
    state.qr = qrInstance;
    state.moduleCount = moduleCount;
    state.darkModules = darkModules;

    emptyOverlay?.classList.add('hide');
    const version = qrInstance.typeNumber;
    metaEl.textContent = `Version ${version} • ${moduleCount}×${moduleCount} modules • ECC ${state.ecc} • Quiet zone ${state.marginModules}`;

    const fgRgb = hexToRgb(state.fg);
    const bgRgb = hexToRgb(state.transparent ? '#FFFFFF' : state.bg);
    const ratio = contrastRatio(fgRgb, bgRgb);
    const contrastOk = ratio >= 4.5;

    const warnings = [];
    if (!contrastOk) warnings.push(`Low contrast (≈${ratio.toFixed(1)}:1) can reduce scan reliability.`);
    if (state.transparent) warnings.push('Transparent backgrounds are best for digital; use a solid background for print.');

    if (state.logoImage) {
      const { boxModules } = computeLogoBox(moduleCount, state.logoSizePct, state.logoPaddingPct);
      const cover = (boxModules * boxModules) / (moduleCount * moduleCount);
      const coverPct = Math.round(cover * 100);
      if (state.ecc !== 'H' && coverPct >= 12) warnings.push(`Logo coverage is ~${coverPct}%. Consider ECC H for best reliability.`);
      if (coverPct >= 18) warnings.push(`Logo coverage is ~${coverPct}%. Reduce logo size for safer scanning.`);
    }

    setWarning(warnings[0] || '');
    setQuality(contrastOk ? 'Business-ready' : 'Check contrast');

    renderQrToCanvas(canvas, { moduleCount, darkModules }, {
      dotStyle: state.dotStyle,
      cornerStyle: state.cornerStyle,
      ecc: state.ecc,
      marginModules: state.marginModules,
      fg: state.fg,
      bg: state.bg,
      transparent: state.transparent,
      logoImage: state.logoImage,
      logoDataUrl: state.logoDataUrl,
      logoSizePct: state.logoSizePct,
      logoPaddingPct: state.logoPaddingPct,
    });

    downloadPngBtn.disabled = false;
    downloadSvgBtn.disabled = false;
    downloadPdfBtn.disabled = false;
  };

  const scheduleRender = debounce(render, 140);

  const applyTemplate = (name) => {
    const preset = TEMPLATES[name];
    if (!preset) return;
    state.dotStyle = preset.dotStyle;
    state.cornerStyle = preset.cornerStyle;
    state.ecc = preset.ecc;
    state.fg = preset.fg;
    state.bg = preset.bg;
    state.transparent = preset.transparent;
    updateControlsFromState();
    scheduleRender();
  };

  templateButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const name = btn.getAttribute('data-template');
      if (!name) return;
      templateButtons.forEach((b) => b.classList.toggle('is-active', b === btn));
      applyTemplate(name);
    });
  });

  const readStateFromControls = () => {
    state.data = dataInput.value;
    state.dotStyle = dotStyleSelect.value;
    state.cornerStyle = cornerStyleSelect.value;
    state.ecc = eccSelect.value;
    state.marginModules = clamp(toInt(marginInput.value, DEFAULTS.marginModules), 2, 10);
    marginValue.textContent = String(state.marginModules);

    state.fg = normalizeHex(fgInput.value);
    state.bg = normalizeHex(bgInput.value);
    state.transparent = !!transparentInput.checked;
    fgLabel.textContent = normalizeHex(state.fg);
    bgLabel.textContent = normalizeHex(state.bg);

    state.logoSizePct = clamp(toInt(logoSizeInput.value, DEFAULTS.logoSizePct), 10, 30);
    logoSizeValue.textContent = `${state.logoSizePct}%`;
    state.logoPaddingPct = clamp(toInt(logoPaddingInput.value, DEFAULTS.logoPaddingPct), 6, 22);
    logoPaddingValue.textContent = `${state.logoPaddingPct}%`;
  };

  const clearLogo = () => {
    state.logoDataUrl = null;
    state.logoImage = null;
    if (logoInput) logoInput.value = '';
    updateControlsFromState();
  };

  const loadLogoFromFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result || '');
      const img = new Image();
      img.onload = () => {
        state.logoDataUrl = url;
        state.logoImage = img;
        if (state.ecc !== 'H') state.ecc = 'H';
        updateControlsFromState();
        readStateFromControls();
        scheduleRender();
      };
      img.onerror = () => {
        clearLogo();
        setWarning('Logo could not be loaded. Try a different file.');
      };
      img.src = url;
    };
    reader.readAsDataURL(file);
  };

  dataInput.addEventListener('input', () => {
    readStateFromControls();
    scheduleRender();
  });

  [
    dotStyleSelect,
    cornerStyleSelect,
    eccSelect,
    marginInput,
    fgInput,
    bgInput,
    transparentInput,
    logoSizeInput,
    logoPaddingInput,
  ].forEach((el) => {
    el.addEventListener('input', () => {
      readStateFromControls();
      scheduleRender();
    });
    el.addEventListener('change', () => {
      readStateFromControls();
      scheduleRender();
    });
  });

  exampleBtn.addEventListener('click', () => {
    dataInput.value = 'https://example.com/campaign?utm_source=brochure&utm_medium=qr';
    readStateFromControls();
    scheduleRender();
    dataInput.focus();
    dataInput.setSelectionRange(dataInput.value.length, dataInput.value.length);
  });

  clearBtn.addEventListener('click', () => {
    dataInput.value = '';
    readStateFromControls();
    scheduleRender();
    dataInput.focus();
  });

  logoInput.addEventListener('change', () => {
    const file = logoInput.files && logoInput.files[0];
    loadLogoFromFile(file);
  });

  logoRemoveBtn.addEventListener('click', () => {
    clearLogo();
    scheduleRender();
  });

  window.addEventListener('resize', debounce(() => {
    if (!state.darkModules) return;
    render();
  }, 160));

  downloadPngBtn.addEventListener('click', async () => {
    if (!state.darkModules) return;
    const size = clamp(toInt(pngSizeSelect.value, 1024), 128, 8192);
    const outCanvas = document.createElement('canvas');
    outCanvas.width = size;
    outCanvas.height = size;
    renderQrToCanvas(outCanvas, { moduleCount: state.moduleCount, darkModules: state.darkModules }, {
      dotStyle: state.dotStyle,
      cornerStyle: state.cornerStyle,
      marginModules: state.marginModules,
      fg: state.fg,
      bg: state.bg,
      transparent: state.transparent,
      logoImage: state.logoImage,
      logoDataUrl: state.logoDataUrl,
      logoSizePct: state.logoSizePct,
      logoPaddingPct: state.logoPaddingPct,
    });
    outCanvas.toBlob((blob) => {
      if (!blob) return;
      const filename = `${fileNameSafe(filenameInput.value)}.png`;
      downloadBlob(blob, filename);
    }, 'image/png');
  });

  downloadSvgBtn.addEventListener('click', () => {
    if (!state.darkModules) return;
    const size = clamp(toInt(pngSizeSelect.value, 1024), 128, 8192);
    const svg = buildSvg({ moduleCount: state.moduleCount, darkModules: state.darkModules }, {
      dotStyle: state.dotStyle,
      cornerStyle: state.cornerStyle,
      marginModules: state.marginModules,
      fg: state.fg,
      bg: state.bg,
      transparent: state.transparent,
      logoDataUrl: state.logoDataUrl,
      logoImage: state.logoImage,
      logoSizePct: state.logoSizePct,
      logoPaddingPct: state.logoPaddingPct,
    }, size);
    downloadBlob(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), `${fileNameSafe(filenameInput.value)}.svg`);
  });

  downloadPdfBtn.addEventListener('click', () => {
    if (!state.darkModules) return;
    const sizeVal = clamp(toFloat(pdfSizeInput.value, 2), 0.5, 20);
    const unit = pdfUnitSelect.value === 'mm' ? 'mm' : 'in';
    const pt = unit === 'mm' ? (sizeVal * 72) / 25.4 : sizeVal * 72;
    const pdfBytes = buildPdf({ moduleCount: state.moduleCount, darkModules: state.darkModules }, {
      dotStyle: state.dotStyle,
      cornerStyle: state.cornerStyle,
      marginModules: state.marginModules,
      fg: state.fg,
      bg: state.bg,
      transparent: false,
      logoImage: state.logoImage,
      logoDataUrl: state.logoDataUrl,
      logoSizePct: state.logoSizePct,
      logoPaddingPct: state.logoPaddingPct,
    }, pt);
    downloadBlob(new Blob([pdfBytes], { type: 'application/pdf' }), `${fileNameSafe(filenameInput.value)}.pdf`);
  });

  readStateFromControls();
  updateControlsFromState();
  scheduleRender();
})();
