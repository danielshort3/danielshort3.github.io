(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const form = $('#textcompare-form');
  const originalEl = $('#textcompare-original');
  const revisedEl = $('#textcompare-revised');
  const outputEl = $('#textcompare-output');
  const summaryEl = $('#textcompare-summary');
  const clearBtn = $('#textcompare-clear');
  const swapBtn = $('#textcompare-swap');
  const copyBtn = $('#textcompare-copy');
  const copyStatus = $('#textcompare-copy-status');
  const originalPasteBtn = $('#textcompare-original-paste');
  const revisedPasteBtn = $('#textcompare-revised-paste');
  const originalImportBtn = $('#textcompare-original-import');
  const revisedImportBtn = $('#textcompare-revised-import');
  const originalFileInput = $('#textcompare-original-file');
  const revisedFileInput = $('#textcompare-revised-file');
  const originalStatusEl = $('#textcompare-original-status');
  const revisedStatusEl = $('#textcompare-revised-status');
  const warningEl = $('#textcompare-warning');
  const insBgEl = $('#textcompare-ins-bg');
  const insTextEl = $('#textcompare-ins-text');
  const delBgEl = $('#textcompare-del-bg');
  const delTextEl = $('#textcompare-del-text');
  const delStrikeEl = $('#textcompare-del-strike');
  const modeInputs = Array.from(document.querySelectorAll('input[name="textcompare-mode"]'));

  if (!form || !originalEl || !revisedEl || !outputEl || !summaryEl) return;

  const compareCore = window.TextCompareCore;
  if (!compareCore || typeof compareCore.compareText !== 'function') return;

  const TOOL_ID = 'text-compare';
  const MAX_CHARS = 600_000;
  const MAX_TOKENS = 200_000;
  const MAX_IMPORT_BYTES = 24 * 1024 * 1024;
  const COMPARE_WORKER_PATH = '/js/tools/text-compare-worker.js';
  const PDF_WORKER_PATH = '/js/vendor/pdfjs/pdf.worker.min.js';
  const PDFJS_SRC = '/js/vendor/pdfjs/pdf.min.js';
  const FFLATE_SRC = '/js/vendor/fflate/fflate.min.js';
  const vendorScriptPromises = new Map();
  const compareRequests = new Map();
  let lastRuns = null;
  let lastRevisedText = '';
  let compareWorker = null;
  let workerUnavailable = false;
  let latestCompareRequestId = 0;
  const fields = {
    original: {
      textarea: originalEl,
      pasteBtn: originalPasteBtn,
      importBtn: originalImportBtn,
      fileInput: originalFileInput,
      statusEl: originalStatusEl,
      sourceKind: 'text'
    },
    revised: {
      textarea: revisedEl,
      pasteBtn: revisedPasteBtn,
      importBtn: revisedImportBtn,
      fileInput: revisedFileInput,
      statusEl: revisedStatusEl,
      sourceKind: 'text'
    }
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const escapeHtml = (s) => String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const setCopyStatus = (msg, tone) => {
    if (!copyStatus) return;
    copyStatus.textContent = msg;
    copyStatus.dataset.tone = tone || '';
  };

  const setWarningStatus = (msg, tone) => {
    if (!warningEl) return;
    warningEl.textContent = String(msg || '');
    warningEl.dataset.tone = tone || '';
  };

  const sharedEdgeScore = (a, b) => {
    const left = String(a || '');
    const right = String(b || '');
    const maxCheck = Math.min(left.length, right.length);
    let prefix = 0;
    while (prefix < maxCheck && left[prefix] === right[prefix]) prefix += 1;
    let suffix = 0;
    while (
      suffix < maxCheck - prefix &&
      left[left.length - 1 - suffix] === right[right.length - 1 - suffix]
    ) {
      suffix += 1;
    }
    return prefix + suffix;
  };

  const splitWhitespace = (text) => {
    const s = String(text || '');
    if (!s) return { leading: '', core: '', trailing: '' };
    if (/^\s+$/.test(s)) return { leading: s, core: '', trailing: '' };
    const leading = (s.match(/^\s+/) || [''])[0];
    const trailing = (s.match(/\s+$/) || [''])[0];
    const core = s.slice(leading.length, s.length - trailing.length);
    return { leading, core, trailing };
  };

  const renderCharDiff = (delCore, insCore, kind) => {
    const segments = compareCore.diffChars(delCore, insCore);
    return segments.map((seg) => {
      if (seg.type === 'equal') return escapeHtml(seg.text);
      if (kind === 'del' && seg.type === 'delete') return `<span class="diff-char-del">${escapeHtml(seg.text)}</span>`;
      if (kind === 'ins' && seg.type === 'insert') return `<span class="diff-char-ins">${escapeHtml(seg.text)}</span>`;
      return '';
    }).join('');
  };

  const renderReplace = (delText, insText) => {
    const delParts = splitWhitespace(delText);
    const insParts = splitWhitespace(insText);
    const leading = insParts.leading;
    const trailing = insParts.trailing;
    const delCoreText = delParts.core;
    const insCoreText = insParts.core;
    const delTrim = delCoreText.trim();
    const insTrim = insCoreText.trim();
    const singleToken = delTrim && insTrim && !/\s/.test(delTrim) && !/\s/.test(insTrim);
    const smallEnough = delTrim.length <= 42 && insTrim.length <= 42;
    const similarEnough = sharedEdgeScore(delTrim, insTrim) >= 2;

    if (!singleToken || !smallEnough || !similarEnough) {
      return `${escapeHtml(leading)}<del class="diff-del">${escapeHtml(delCoreText)}</del><ins class="diff-ins">${escapeHtml(insCoreText)}</ins>${escapeHtml(trailing)}`;
    }

    const delInner = renderCharDiff(delCoreText, insCoreText, 'del');
    const insInner = renderCharDiff(delCoreText, insCoreText, 'ins');
    return `${escapeHtml(leading)}<del class="diff-del">${delInner}</del><ins class="diff-ins">${insInner}</ins>${escapeHtml(trailing)}`;
  };

  const buildMoveAttrs = (run) => {
    if (!run?.moveId) return { className: '', attrs: '' };
    const role = run.moveRole === 'from' ? 'from' : 'to';
    return {
      className: ' diff-move',
      attrs: ` data-move-role="${role}" data-move-id="${run.moveId}"`
    };
  };

  const renderOutput = (runs) => runs.map((run) => {
    if (run.type === 'equal') return escapeHtml(run.tokens.join(''));
    if (run.type === 'insert') {
      const move = buildMoveAttrs(run);
      return `<ins class="diff-ins${move.className}"${move.attrs}>${escapeHtml(run.tokens.join(''))}</ins>`;
    }
    if (run.type === 'delete') {
      const move = buildMoveAttrs(run);
      return `<del class="diff-del${move.className}"${move.attrs}>${escapeHtml(run.tokens.join(''))}</del>`;
    }
    if (run.type === 'replace') {
      const delText = run.delTokens.join('');
      const insText = run.insTokens.join('');
      return renderReplace(delText, insText);
    }
    return '';
  }).join('');

  const setEmpty = (msg) => {
    outputEl.innerHTML = `<p class="textcompare-empty">${escapeHtml(msg)}</p>`;
  };

  const escapeHtmlWithBreaks = (text) => escapeHtml(text).replace(/\r\n|\r|\n/g, '<br>');

  const getCopyStyle = () => ({
    insBg: insBgEl?.value || '#00FF00',
    insColor: insTextEl?.value || '#000000',
    delBg: delBgEl?.value || '#FF0000',
    delColor: delTextEl?.value || '#000000',
    delStrike: delStrikeEl?.value || '#000000'
  });

  const normalizeHexColor = (value, fallback) => {
    const s = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s.toUpperCase();
    return fallback;
  };

  const applyPreviewStyle = () => {
    const style = getCopyStyle();
    document.body.style.setProperty('--textcompare-ins-bg', normalizeHexColor(style.insBg, '#00FF00'));
    document.body.style.setProperty('--textcompare-ins-text', normalizeHexColor(style.insColor, '#000000'));
    document.body.style.setProperty('--textcompare-del-bg', normalizeHexColor(style.delBg, '#FF0000'));
    document.body.style.setProperty('--textcompare-del-text', normalizeHexColor(style.delColor, '#000000'));
    document.body.style.setProperty('--textcompare-del-strike', normalizeHexColor(style.delStrike, '#000000'));
  };

  const hexToRgb = (hex) => {
    const h = String(hex || '').replace('#', '');
    const r = parseInt(h.slice(0, 2), 16) || 0;
    const g = parseInt(h.slice(2, 4), 16) || 0;
    const b = parseInt(h.slice(4, 6), 16) || 0;
    return { r, g, b };
  };

  const escapeRtf = (text) => {
    const s = String(text || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    let out = '';
    for (let i = 0; i < s.length; i += 1) {
      const code = s.codePointAt(i);
      const ch = String.fromCodePoint(code);
      if (code > 0xFFFF) i += 1;
      if (ch === '\\') out += '\\\\';
      else if (ch === '{') out += '\\{';
      else if (ch === '}') out += '\\}';
      else if (ch === '\n') out += '\\line\n';
      else if (code <= 0x7F) out += ch;
      else if (code <= 0xFFFF) {
        const signed = code > 0x7FFF ? code - 0x10000 : code;
        out += `\\u${signed}?`;
      } else {
        const cp = code - 0x10000;
        const hi = 0xD800 + (cp >> 10);
        const lo = 0xDC00 + (cp & 0x3FF);
        const hiSigned = hi > 0x7FFF ? hi - 0x10000 : hi;
        const loSigned = lo > 0x7FFF ? lo - 0x10000 : lo;
        out += `\\u${hiSigned}?\\u${loSigned}?`;
      }
    }
    return out;
  };

  const buildClipboardFragment = (runs, style) => {
    const insBg = normalizeHexColor(style.insBg, '#00FF00');
    const insColor = normalizeHexColor(style.insColor, '#000000');
    const delBg = normalizeHexColor(style.delBg, '#FF0000');
    const delColor = normalizeHexColor(style.delColor, '#000000');
    const delStrike = normalizeHexColor(style.delStrike, '#000000');

    const insStyle = `background:${insBg};background-color:${insBg};color:${insColor};mso-highlight:${insBg};`;
    const delWrapStyle = `background:${delBg};background-color:${delBg};mso-highlight:${delBg};`;
    const delInnerStyle = `color:${delColor};text-decoration:line-through;text-decoration-color:${delStrike};mso-text-decoration:line-through;`;
    return runs.map((run) => {
      if (run.type === 'equal') return escapeHtmlWithBreaks(run.tokens.join(''));
      if (run.type === 'insert') return `<span style="${insStyle}">${escapeHtmlWithBreaks(run.tokens.join(''))}</span>`;
      if (run.type === 'delete') return `<span style="${delWrapStyle}"><s style="${delInnerStyle}">${escapeHtmlWithBreaks(run.tokens.join(''))}</s></span>`;
      if (run.type === 'replace') {
        const delText = run.delTokens.join('');
        const insText = run.insTokens.join('');
        const delParts = splitWhitespace(delText);
        const insParts = splitWhitespace(insText);
        const leading = insParts.leading;
        const trailing = insParts.trailing;
        return `${escapeHtmlWithBreaks(leading)}<span style="${delWrapStyle}"><s style="${delInnerStyle}">${escapeHtmlWithBreaks(delParts.core)}</s></span><span style="${insStyle}">${escapeHtmlWithBreaks(insParts.core)}</span>${escapeHtmlWithBreaks(trailing)}`;
      }
      return '';
    }).join('');
  };

  const buildClipboardHtml = (runs, style) => {
    const fragment = buildClipboardFragment(runs, style);
    const bodyStyle = [
      'font-family:Calibri, Arial, sans-serif',
      'font-size:11pt',
      'line-height:1.5',
      'color:#000',
      'background:#fff'
    ].join(';');
    return `<div style="${bodyStyle}"><!--StartFragment-->${fragment}<!--EndFragment--></div>`;
  };

  const buildClipboardRtf = (runs, style) => {
    const insBg = normalizeHexColor(style.insBg, '#00FF00');
    const insColor = normalizeHexColor(style.insColor, '#000000');
    const delBg = normalizeHexColor(style.delBg, '#FF0000');
    const delColor = normalizeHexColor(style.delColor, '#000000');

    const insBgRgb = hexToRgb(insBg);
    const insColorRgb = hexToRgb(insColor);
    const delBgRgb = hexToRgb(delBg);
    const delColorRgb = hexToRgb(delColor);

    const colors = [
      { r: 0, g: 0, b: 0 }, // index 1: black (fallback)
      insBgRgb,            // index 2: inserted highlight
      insColorRgb,         // index 3: inserted text color
      delBgRgb,            // index 4: deleted highlight
      delColorRgb          // index 5: deleted text color
    ];
    const colorTable = `{\n\\colortbl ;${colors.map(c => `\\red${c.r}\\green${c.g}\\blue${c.b};`).join('')}\n}\n`;

    const normalPrefix = '\\highlight0\\cf1\\strike0 ';
    const insertPrefix = '\\highlight2\\cf3\\strike0 ';
    const deletePrefix = '\\highlight4\\cf5\\strike ';

    const body = runs.map((run) => {
      if (run.type === 'equal') return escapeRtf(run.tokens.join(''));
      if (run.type === 'insert') return `${insertPrefix}${escapeRtf(run.tokens.join(''))}${normalPrefix}`;
      if (run.type === 'delete') return `${deletePrefix}${escapeRtf(run.tokens.join(''))}${normalPrefix}`;
      if (run.type === 'replace') {
        const delText = run.delTokens.join('');
        const insText = run.insTokens.join('');
        const delParts = splitWhitespace(delText);
        const insParts = splitWhitespace(insText);
        const leading = insParts.leading;
        const trailing = insParts.trailing;
        return `${escapeRtf(leading)}${deletePrefix}${escapeRtf(delParts.core)}${normalPrefix}${insertPrefix}${escapeRtf(insParts.core)}${normalPrefix}${escapeRtf(trailing)}`;
      }
      return '';
    }).join('');

    return `{\\rtf1\\ansi\\deff0\n{\\fonttbl{\\f0 Calibri;}}\n${colorTable}\\viewkind4\\uc1\\pard\\f0\\fs22 ${normalPrefix}${body}\\par\n}`;
  };

  const copyFormatted = async () => {
    if (!lastRuns || !lastRuns.length) {
      setCopyStatus('Nothing to copy yet.', 'error');
      return;
    }

    setCopyStatus('Copying…');
    const style = getCopyStyle();
    const html = buildClipboardHtml(lastRuns, style);
    const rtf = buildClipboardRtf(lastRuns, style);
    const plainText = lastRevisedText || '';

    try {
      if (navigator.clipboard && window.ClipboardItem) {
        const item = new ClipboardItem({
          'text/html': new Blob([html], { type: 'text/html' }),
          'text/plain': new Blob([plainText], { type: 'text/plain' }),
          'text/rtf': new Blob([rtf], { type: 'text/rtf' })
        });
        await navigator.clipboard.write([item]);
        setCopyStatus('Copied with formatting (Outlook-friendly).', 'success');
        return;
      }
    } catch {
      // fall through to selection-based copy
    }

    try {
      const temp = document.createElement('div');
      temp.style.position = 'fixed';
      temp.style.left = '-9999px';
      temp.style.top = '0';
      temp.style.whiteSpace = 'normal';
      temp.contentEditable = 'true';
      temp.innerHTML = html;
      document.body.appendChild(temp);

      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(temp);
      selection?.removeAllRanges();
      selection?.addRange(range);
      const handleCopy = (event) => {
        if (!event.clipboardData) return;
        event.clipboardData.setData('text/plain', plainText);
        event.clipboardData.setData('text/html', html);
        try {
          event.clipboardData.setData('text/rtf', rtf);
        } catch {
          // ignore if the browser blocks RTF
        }
        event.preventDefault();
      };
      document.addEventListener('copy', handleCopy);
      let ok = false;
      try {
        ok = document.execCommand('copy');
      } finally {
        document.removeEventListener('copy', handleCopy);
        selection?.removeAllRanges();
        temp.remove();
      }

      setCopyStatus(ok ? 'Copied with formatting.' : 'Copy failed.', ok ? 'success' : 'error');
    } catch {
      setCopyStatus('Copy failed. Try selecting the output and copying manually.', 'error');
    }
  };

  const setFieldStatus = (fieldKey, message, tone) => {
    const field = fields[fieldKey];
    if (!field?.statusEl) return;
    field.statusEl.textContent = String(message || '');
    field.statusEl.dataset.tone = tone || '';
  };

  const setFieldBusy = (fieldKey, busy) => {
    const field = fields[fieldKey];
    if (!field) return;
    if (field.pasteBtn) field.pasteBtn.disabled = Boolean(busy);
    if (field.importBtn) field.importBtn.disabled = Boolean(busy);
    if (field.fileInput) field.fileInput.disabled = Boolean(busy);
  };

  const clearFieldStatuses = () => {
    setFieldStatus('original', '', '');
    setFieldStatus('revised', '', '');
  };

  const normalizeInputText = (text) => String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\u00A0/g, ' ');

  const normalizeImportedText = (text) => normalizeInputText(text)
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{4,}/g, '\n\n\n')
    .trim();

  const applyTextToField = (fieldKey, text, options) => {
    const field = fields[fieldKey];
    if (!field?.textarea) return;
    field.sourceKind = String(options?.sourceKind || field.sourceKind || 'text');
    field.textarea.value = String(text || '');
    field.textarea.dispatchEvent(new Event('input', { bubbles: true }));
    field.textarea.dispatchEvent(new Event('change', { bubbles: true }));
    markSessionDirty();
    field.textarea.focus();
  };

  const decodeXmlEntities = (text) => String(text || '')
    .replace(/&#x([0-9a-f]+);/gi, (_match, hex) => {
      const code = Number.parseInt(hex, 16);
      if (Number.isNaN(code)) return '';
      try {
        return String.fromCodePoint(code);
      } catch {
        return '';
      }
    })
    .replace(/&#([0-9]+);/g, (_match, dec) => {
      const code = Number.parseInt(dec, 10);
      if (Number.isNaN(code)) return '';
      try {
        return String.fromCodePoint(code);
      } catch {
        return '';
      }
    })
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');

  const extractHtmlText = (html) => {
    const raw = String(html || '');
    const withBreakHints = raw
      .replace(/<\s*br\s*\/?>/gi, '\n')
      .replace(/<\/(p|div|li|tr|h[1-6])>/gi, '\n')
      .replace(/<li\b[^>]*>/gi, '- ');
    const doc = new DOMParser().parseFromString(withBreakHints, 'text/html');
    doc.querySelectorAll('script, style, noscript').forEach((node) => node.remove());
    return doc.body?.textContent || '';
  };

  const extractRtfText = (rtf) => String(rtf || '')
    .replace(/\\par[d]?/gi, '\n')
    .replace(/\\line\b/gi, '\n')
    .replace(/\\tab\b/gi, '\t')
    .replace(/\\'([0-9a-f]{2})/gi, (_match, hex) => String.fromCharCode(Number.parseInt(hex, 16) || 32))
    .replace(/\\u(-?\d+)\??/g, (_match, num) => {
      let code = Number.parseInt(num, 10);
      if (Number.isNaN(code)) return '';
      if (code < 0) code += 65536;
      try {
        return String.fromCharCode(code);
      } catch {
        return '';
      }
    })
    .replace(/\\[a-z]+-?\d* ?/gi, '')
    .replace(/[{}]/g, '');

  const extractDocxXmlText = (xmlText) => {
    const text = String(xmlText || '')
      .replace(/<w:tab[^>]*\/>/gi, '\t')
      .replace(/<w:(br|cr)[^>]*\/>/gi, '\n')
      .replace(/<\/w:tc>/gi, '\t')
      .replace(/<\/w:(p|tr)>/gi, '\n')
      .replace(/<[^>]+>/g, '');
    return decodeXmlEntities(text);
  };

  const extractLegacyDocText = (bytes) => {
    const clean = (text) => String(text || '')
      .replace(/[^ -~\n\r\t]/g, ' ')
      .replace(/[ \t]{2,}/g, ' ')
      .replace(/\n{4,}/g, '\n\n\n');

    const latin = clean(new TextDecoder('latin1').decode(bytes));
    const utf16 = clean(new TextDecoder('utf-16le').decode(bytes));
    return utf16.length > latin.length ? utf16 : latin;
  };

  const loadVendorScript = (src) => {
    const safeSrc = String(src || '').trim();
    if (!safeSrc) return Promise.reject(new Error('Vendor script source is missing.'));
    if (vendorScriptPromises.has(safeSrc)) return vendorScriptPromises.get(safeSrc);

    const existing = document.querySelector(`script[src="${safeSrc}"]`);
    if (existing && existing.dataset.vendorLoaded === 'true') {
      return Promise.resolve();
    }

    const pending = new Promise((resolve, reject) => {
      const script = existing || document.createElement('script');

      const finish = (error) => {
        script.removeEventListener('load', onLoad);
        script.removeEventListener('error', onError);
        if (error) {
          vendorScriptPromises.delete(safeSrc);
          reject(error);
          return;
        }
        script.dataset.vendorLoaded = 'true';
        resolve();
      };
      const onLoad = () => finish();
      const onError = () => finish(new Error(`Unable to load ${safeSrc}.`));

      script.addEventListener('load', onLoad);
      script.addEventListener('error', onError);

      if (!existing) {
        script.src = safeSrc;
        script.async = true;
        document.head.appendChild(script);
      }
    });

    vendorScriptPromises.set(safeSrc, pending);
    return pending;
  };

  const ensureFflate = async () => {
    const api = window.fflate;
    if (api && typeof api.unzipSync === 'function') return api;
    await loadVendorScript(FFLATE_SRC);
    const loaded = window.fflate;
    if (!loaded || typeof loaded.unzipSync !== 'function') {
      throw new Error('DOCX import is unavailable: zip parser failed to load.');
    }
    return loaded;
  };

  const ensurePdfjs = async () => {
    const api = window.pdfjsLib;
    if (api && typeof api.getDocument === 'function') return api;
    await loadVendorScript(PDFJS_SRC);
    const loaded = window.pdfjsLib;
    if (!loaded || typeof loaded.getDocument !== 'function') {
      throw new Error('PDF import is unavailable: parser failed to load.');
    }
    return loaded;
  };

  const parseDocxFile = async (file) => {
    const api = await ensureFflate();
    const bytes = new Uint8Array(await file.arrayBuffer());
    const files = api.unzipSync(bytes);
    const decoder = new TextDecoder('utf-8');
    const xmlPaths = Object.keys(files)
      .filter((name) => /^word\/(document|header\d+|footer\d+|footnotes|endnotes)\.xml$/i.test(name))
      .sort((a, b) => a.localeCompare(b));

    if (!xmlPaths.length) {
      throw new Error('Unable to find readable text in this DOCX file.');
    }

    const chunks = [];
    xmlPaths.forEach((path) => {
      try {
        const xml = decoder.decode(files[path]);
        const text = normalizeImportedText(extractDocxXmlText(xml));
        if (text) chunks.push(text);
      } catch {
        // Skip malformed sections and keep best-effort extraction.
      }
    });
    return chunks.join('\n\n');
  };

  const parsePdfFile = async (file) => {
    const pdfjs = await ensurePdfjs();
    if (pdfjs.GlobalWorkerOptions && !pdfjs.GlobalWorkerOptions.workerSrc) {
      pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER_PATH;
    }

    const loadingTask = pdfjs.getDocument({ data: await file.arrayBuffer() });
    let pdf = null;

    try {
      pdf = await loadingTask.promise;
      const pages = [];
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const textContent = await page.getTextContent();
        let pageText = '';
        textContent.items.forEach((item) => {
          const value = String(item?.str || '');
          if (!value) return;
          pageText += value;
          pageText += item?.hasEOL ? '\n' : ' ';
        });
        const normalized = normalizeImportedText(pageText);
        if (normalized) pages.push(normalized);
      }
      return pages.join('\n\n');
    } finally {
      try {
        await loadingTask.destroy();
      } catch {}
      try {
        pdf?.cleanup?.();
      } catch {}
    }
  };

  const detectImportType = (file) => {
    const name = String(file?.name || '').toLowerCase();
    const type = String(file?.type || '').toLowerCase();

    if (name.endsWith('.csv') || type.includes('csv')) return 'csv';
    if (name.endsWith('.tsv') || type.includes('tab-separated-values')) return 'tsv';
    if (name.endsWith('.json') || type.includes('json')) return 'json';
    if (name.endsWith('.xml') || type.includes('xml')) return 'xml';
    if (name.endsWith('.pdf') || type.includes('pdf')) return 'pdf';
    if (name.endsWith('.docx') || type.includes('officedocument.wordprocessingml.document')) return 'docx';
    if (name.endsWith('.doc') || type === 'application/msword') return 'doc';
    if (name.endsWith('.rtf') || type.includes('rtf')) return 'rtf';
    if (name.endsWith('.html') || name.endsWith('.htm') || type.includes('html')) return 'html';
    if (name.endsWith('.md') || name.endsWith('.markdown')) return 'markdown';
    return 'text';
  };

  const parseImportedFile = async (file) => {
    const importType = detectImportType(file);
    if (importType === 'pdf') {
      return { text: await parsePdfFile(file), warning: '', sourceKind: importType };
    }
    if (importType === 'docx') {
      return { text: await parseDocxFile(file), warning: '', sourceKind: importType };
    }
    if (importType === 'doc') {
      const bytes = new Uint8Array(await file.arrayBuffer());
      return {
        text: extractLegacyDocText(bytes),
        warning: 'Legacy .doc extraction is best-effort. Convert to .docx for highest fidelity.',
        sourceKind: importType
      };
    }
    if (importType === 'rtf') {
      return { text: extractRtfText(await file.text()), warning: '', sourceKind: importType };
    }
    if (importType === 'html') {
      return { text: extractHtmlText(await file.text()), warning: '', sourceKind: importType };
    }
    return { text: await file.text(), warning: '', sourceKind: importType };
  };

  const pasteIntoField = async (fieldKey) => {
    const field = fields[fieldKey];
    if (!field?.textarea) return;

    setFieldBusy(fieldKey, true);
    setFieldStatus(fieldKey, 'Reading clipboard…', 'info');

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.readText !== 'function') {
        throw new Error('Clipboard read is unavailable.');
      }
      const clipboardText = normalizeInputText(await navigator.clipboard.readText());
      if (!clipboardText.trim()) {
        setFieldStatus(fieldKey, 'Clipboard is empty.', 'error');
        return;
      }
      applyTextToField(fieldKey, clipboardText, { sourceKind: 'text' });
      setFieldStatus(fieldKey, `Pasted ${clipboardText.length.toLocaleString('en-US')} characters.`, 'success');
    } catch {
      setFieldStatus(fieldKey, 'Clipboard access blocked. Use Ctrl/Cmd+V in the text box.', 'error');
    } finally {
      setFieldBusy(fieldKey, false);
    }
  };

  const importIntoField = async (fieldKey) => {
    const field = fields[fieldKey];
    const file = field?.fileInput?.files?.[0];
    if (!field?.textarea || !file) return;

    setFieldBusy(fieldKey, true);
    setFieldStatus(fieldKey, `Importing ${file.name}…`, 'info');

    try {
      if (file.size > MAX_IMPORT_BYTES) {
        const maxMb = Math.round(MAX_IMPORT_BYTES / (1024 * 1024));
        throw new Error(`File is too large. Max supported size is ${maxMb} MB.`);
      }

      const parsed = await parseImportedFile(file);
      const importedText = normalizeImportedText(parsed.text);
      if (!importedText.trim()) {
        throw new Error('No readable text was found in this file.');
      }

      applyTextToField(fieldKey, importedText, { sourceKind: parsed.sourceKind || 'text' });
      const summary = `${file.name} imported (${importedText.length.toLocaleString('en-US')} characters).`;
      const statusText = parsed.warning ? `${summary} ${parsed.warning}` : summary;
      setFieldStatus(fieldKey, statusText, parsed.warning ? 'info' : 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to import this file.';
      setFieldStatus(fieldKey, message, 'error');
    } finally {
      if (field.fileInput) field.fileInput.value = '';
      setFieldBusy(fieldKey, false);
    }
  };

  const getSelectedMode = () => {
    const selected = modeInputs.find((input) => input.checked);
    return String(selected?.value || compareCore.MODES.AUTO);
  };

  const formatCompareSummary = (counts) => {
    if (!counts?.hasChanges) return 'No differences found.';
    const parts = [];
    if (counts.insertedWords) parts.push(`${counts.insertedWords.toLocaleString('en-US')} inserted words`);
    if (counts.deletedWords) parts.push(`${counts.deletedWords.toLocaleString('en-US')} deleted words`);
    if (counts.replacements) parts.push(`${counts.replacements.toLocaleString('en-US')} replacements`);
    if (counts.movedBlocks) parts.push(`${counts.movedBlocks.toLocaleString('en-US')} moved blocks`);
    return `Changes: ${parts.join(' · ')}.`;
  };

  const getAutoModeNotice = (modeOverride, inferredMode) => {
    if (modeOverride !== compareCore.MODES.AUTO) return '';
    if (inferredMode === compareCore.MODES.STRUCTURED) return 'Auto mode used structured comparison.';
    if (inferredMode === compareCore.MODES.DOCUMENT) return 'Auto mode used document comparison.';
    return '';
  };

  const ensureCompareWorker = () => {
    if (workerUnavailable || typeof Worker !== 'function') return null;
    if (compareWorker) return compareWorker;
    try {
      compareWorker = new Worker(COMPARE_WORKER_PATH);
      compareWorker.addEventListener('message', (event) => {
        const data = event?.data || {};
        const pending = compareRequests.get(data.requestId);
        if (!pending) return;
        compareRequests.delete(data.requestId);
        if (!data.ok) {
          pending.reject(new Error(data.error || 'Background compare failed.'));
          return;
        }
        pending.resolve(data);
      });
      compareWorker.addEventListener('error', (event) => {
        workerUnavailable = true;
        const error = new Error(event?.message || 'Background compare failed.');
        compareRequests.forEach((pending) => pending.reject(error));
        compareRequests.clear();
        try {
          compareWorker?.terminate();
        } catch {}
        compareWorker = null;
      });
      return compareWorker;
    } catch {
      workerUnavailable = true;
      compareWorker = null;
      return null;
    }
  };

  const requestWorkerCompare = (payload) => {
    const worker = ensureCompareWorker();
    if (!worker) {
      return Promise.reject(new Error('Background compare worker is unavailable.'));
    }
    return new Promise((resolve, reject) => {
      compareRequests.set(payload.requestId, { resolve, reject });
      try {
        worker.postMessage(payload);
      } catch (error) {
        compareRequests.delete(payload.requestId);
        reject(error);
      }
    });
  };

  const runCompareOnMainThread = (payload) => compareCore.compareText({
    leftText: payload.leftText,
    rightText: payload.rightText,
    modeOverride: payload.modeOverride,
    sourceHints: payload.sourceHints
  });

  const renderCompareResult = (result, revisedText, modeOverride, fallbackWarning) => {
    const warnings = [];
    if (Array.isArray(result?.warnings)) warnings.push(...result.warnings);
    const autoNotice = getAutoModeNotice(modeOverride, result?.inferredMode);
    if (autoNotice) warnings.unshift(autoNotice);
    if (fallbackWarning) warnings.push(fallbackWarning);

    lastRuns = result?.runs || [];
    lastRevisedText = revisedText;
    outputEl.innerHTML = renderOutput(lastRuns) || '<p class="textcompare-empty">No output.</p>';
    summaryEl.textContent = formatCompareSummary(result?.counts);
    setWarningStatus(warnings.join(' '), warnings.length ? 'info' : '');
    markSessionDirty();
  };

  const runCompare = () => {
    setCopyStatus('');
    setWarningStatus('', '');
    markSessionDirty();
    const originalInput = originalEl.value || '';
    const revisedInput = revisedEl.value || '';
    const originalHasUser = Boolean(originalInput.trim());
    const revisedHasUser = Boolean(revisedInput.trim());
    const original = (!originalHasUser && !revisedHasUser) ? (originalEl.placeholder || '') : originalInput;
    const revised = (!originalHasUser && !revisedHasUser) ? (revisedEl.placeholder || '') : revisedInput;

    if ((!originalHasUser && revisedHasUser) || (originalHasUser && !revisedHasUser)) {
      summaryEl.textContent = 'Paste both versions to compare.';
      setEmpty('Paste text in both boxes, then click Compare.');
      lastRuns = null;
      lastRevisedText = '';
      latestCompareRequestId += 1;
      setWarningStatus('', '');
      markSessionDirty();
      return;
    }

    if (!original.trim() && !revised.trim()) {
      summaryEl.textContent = 'Click Compare to run the built-in example, or paste your own drafts.';
      setEmpty('Waiting for input.');
      lastRuns = null;
      lastRevisedText = '';
      latestCompareRequestId += 1;
      setWarningStatus('', '');
      markSessionDirty();
      return;
    }

    if (original.length + revised.length > MAX_CHARS) {
      summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
      setEmpty('Input too large.');
      lastRuns = null;
      lastRevisedText = '';
      latestCompareRequestId += 1;
      setWarningStatus('', '');
      markSessionDirty();
      return;
    }

    summaryEl.textContent = 'Comparing…';
    setEmpty('Comparing…');
    const requestId = latestCompareRequestId + 1;
    latestCompareRequestId = requestId;
    const payload = {
      requestId,
      leftText: original,
      rightText: revised,
      modeOverride: getSelectedMode(),
      sourceHints: {
        leftKind: fields.original.sourceKind || 'text',
        rightKind: fields.revised.sourceKind || 'text'
      }
    };

    requestAnimationFrame(() => {
      void (async () => {
        let result = null;
        let fallbackWarning = '';

        try {
          if (typeof compareCore.tokenize === 'function') {
            const tokenCount = compareCore.tokenize(payload.leftText).length + compareCore.tokenize(payload.rightText).length;
            if (tokenCount > MAX_TOKENS) {
              summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
              setEmpty('Input too large.');
              lastRuns = null;
              lastRevisedText = '';
              setWarningStatus('', '');
              markSessionDirty();
              return;
            }
          }

          result = await requestWorkerCompare(payload);
        } catch {
          workerUnavailable = true;
          try {
            compareWorker?.terminate();
          } catch {}
          compareWorker = null;
          result = runCompareOnMainThread(payload);
          fallbackWarning = 'Compared on the main thread because the background worker was unavailable.';
        }

        if (requestId !== latestCompareRequestId) return;
        renderCompareResult(result, revised, payload.modeOverride, fallbackWarning);
      })();
    });
  };

  [insBgEl, insTextEl, delBgEl, delTextEl, delStrikeEl].forEach((el) => {
    el?.addEventListener('input', applyPreviewStyle);
  });
  applyPreviewStyle();

  originalPasteBtn?.addEventListener('click', () => {
    void pasteIntoField('original');
  });
  revisedPasteBtn?.addEventListener('click', () => {
    void pasteIntoField('revised');
  });
  originalImportBtn?.addEventListener('click', () => {
    originalFileInput?.click();
  });
  revisedImportBtn?.addEventListener('click', () => {
    revisedFileInput?.click();
  });
  originalFileInput?.addEventListener('change', () => {
    void importIntoField('original');
  });
  revisedFileInput?.addEventListener('change', () => {
    void importIntoField('revised');
  });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCompare();
  });

  clearBtn?.addEventListener('click', () => {
    latestCompareRequestId += 1;
    originalEl.value = '';
    revisedEl.value = '';
    fields.original.sourceKind = 'text';
    fields.revised.sourceKind = 'text';
    summaryEl.textContent = 'Click Compare to run the built-in example, or paste your own drafts.';
    setEmpty('Waiting for input.');
    lastRuns = null;
    lastRevisedText = '';
    setCopyStatus('');
    setWarningStatus('', '');
    clearFieldStatuses();
    markSessionDirty();
    originalEl.focus();
  });

  swapBtn?.addEventListener('click', () => {
    const a = originalEl.value;
    const sourceKind = fields.original.sourceKind;
    originalEl.value = revisedEl.value;
    revisedEl.value = a;
    fields.original.sourceKind = fields.revised.sourceKind;
    fields.revised.sourceKind = sourceKind;
    runCompare();
  });

  copyBtn?.addEventListener('click', copyFormatted);
  const MAX_SAVED_OUTPUT_HTML_CHARS = 120_000;
  const MAX_SAVED_OUTPUT_TEXT_CHARS = 120_000;

  const clampText = (value, maxChars) => {
    const text = String(value || '');
    if (text.length <= maxChars) return { text, truncated: false };
    return { text: text.slice(0, maxChars), truncated: true };
  };

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const summary = String(summaryEl?.textContent || '').trim();
    payload.outputSummary = summary;

    const html = String(outputEl?.innerHTML || '').trim();
    if (html && html.length <= MAX_SAVED_OUTPUT_HTML_CHARS) {
      payload.output = { kind: 'html', html, summary };
      return;
    }

    const content = String(outputEl?.textContent || '').trim();
    const { text, truncated } = clampText(content, MAX_SAVED_OUTPUT_TEXT_CHARS);
    payload.output = { kind: 'text', text, summary, truncated };
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const snapshot = detail?.snapshot;
    const output = snapshot?.output;
    if (!output || typeof output !== 'object') return;

    const summary = String(output.summary || '').trim();
    if (summary) summaryEl.textContent = summary;
    setWarningStatus('', '');

    const kind = String(output.kind || '').trim();
    if (kind === 'html') {
      outputEl.innerHTML = String(output.html || '').trim() || '<p class="textcompare-empty">No output.</p>';
      return;
    }

    if (kind === 'text') {
      const raw = String(output.text || '').trim();
      outputEl.innerHTML = raw
        ? `<pre>${escapeHtml(raw)}</pre>`
        : '<p class="textcompare-empty">No output.</p>';
    }
  });
})();
