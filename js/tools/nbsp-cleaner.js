(() => {
  'use strict';
  const $ = (sel) => document.querySelector(sel);
  const form = $('#nbsp-form');
  const input = $('#nbsp-input');
  const output = $('#nbsp-output');
  const summary = $('#nbsp-summary');
  const countsList = $('#nbsp-counts');
  const copyBtn = $('#nbsp-copy');
  const copyStatus = $('#nbsp-copy-status');
  const clearBtn = $('#nbsp-clear');
  const preview = $('#nbsp-preview');
  const fixHardToggle = $('#nbsp-fix-hard');
  const stripNonAsciiToggle = $('#nbsp-strip-nonascii');
  const pasteBtn = $('#nbsp-paste');
  const importBtn = $('#nbsp-import');
  const fileInput = $('#nbsp-file');
  const inputStatus = $('#nbsp-input-status');

  if (!form || !input || !output || !summary || !countsList || !preview) return;

  const TOOL_ID = 'nbsp-cleaner';
  const MAX_IMPORT_BYTES = 24 * 1024 * 1024;
  const PDF_WORKER_PATH = '/js/vendor/pdfjs/pdf.worker.min.js';
  const PDFJS_SRC = '/js/vendor/pdfjs/pdf.min.js';
  const FFLATE_SRC = '/js/vendor/fflate/fflate.min.js';
  const vendorScriptPromises = new Map();

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const HARD_SPACES = [
    { char: '\u00A0', label: 'Non-breaking space', code: 'U+00A0' },
    { char: '\u202F', label: 'Narrow no-break space', code: 'U+202F' },
    { char: '\u2007', label: 'Figure space', code: 'U+2007' },
    { char: '\u2009', label: 'Thin space', code: 'U+2009' },
  ];
  const HARD_SET = new Set(HARD_SPACES.map((h) => h.char));

  const escapeForSet = (str) => str.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const hardChars = HARD_SPACES.map((h) => h.char).join('');
  const hardRegex = new RegExp(`[${escapeForSet(hardChars)}]`, 'g');
  const NON_ASCII_REGEX = /[^\x00-\x7F]/g;

  const formatNumber = (n) => n.toLocaleString('en-US');
  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const normalizeInputText = (text) => String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');

  const setInputStatus = (msg, tone) => {
    if (!inputStatus) return;
    inputStatus.textContent = String(msg || '');
    inputStatus.dataset.tone = tone || '';
  };

  const setInputBusy = (busy) => {
    const state = Boolean(busy);
    if (pasteBtn) pasteBtn.disabled = state;
    if (importBtn) importBtn.disabled = state;
    if (fileInput) fileInput.disabled = state;
  };

  const codePointLabel = (ch) => {
    const cp = ch.codePointAt(0).toString(16).toUpperCase().padStart(4, '0');
    return `U+${cp}`;
  };

  const analyze = (text) => {
    const perType = HARD_SPACES.map((h) => ({ ...h, count: 0 }));
    let total = 0;
    for (const ch of text) {
      const idx = HARD_SPACES.findIndex((h) => h.char === ch);
      if (idx !== -1) {
        perType[idx].count += 1;
        total += 1;
      }
    }
    return { perType, total };
  };

  const countNonAscii = (text) => {
    const counts = new Map();
    for (const ch of text) {
      if (HARD_SET.has(ch)) continue;
      if (ch.charCodeAt(0) > 127) {
        counts.set(ch, (counts.get(ch) || 0) + 1);
      }
    }
    return counts;
  };

  const renderCounts = (perType, nonAsciiCounts) => {
    countsList.innerHTML = '';
    const active = perType.filter((p) => p.count > 0);
    active.forEach((entry) => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${formatNumber(entry.count)}×</strong>
        <div>
          <div>${entry.label}</div>
          <small>${entry.code}</small>
        </div>`;
      countsList.appendChild(li);
    });
    const nonAsciiTotal = [...nonAsciiCounts.values()].reduce((sum, v) => sum + v, 0);
    if (nonAsciiTotal > 0) {
      const li = document.createElement('li');
      const samples = [...nonAsciiCounts.entries()].slice(0, 3)
        .map(([ch, c]) => `${codePointLabel(ch)} (${formatNumber(c)}×)`).join(', ');
      li.innerHTML = `<strong>${formatNumber(nonAsciiTotal)}×</strong>
        <div>
          <div>Other non-ASCII characters</div>
          <small>${samples}</small>
        </div>`;
      countsList.appendChild(li);
    }
  };

  const renderPreview = (text) => {
    const parts = [];
    for (const ch of text) {
      if (HARD_SET.has(ch)) {
        parts.push(`<span class="nbsp-chip nbsp-hard" title="${codePointLabel(ch)} hard space">␠</span>`);
      } else if (ch.charCodeAt(0) > 127) {
        parts.push(`<span class="nbsp-chip nbsp-nonascii" title="Non-ASCII ${codePointLabel(ch)}">${escapeHtml(ch)}</span>`);
      } else {
        parts.push(escapeHtml(ch));
      }
    }
    preview.innerHTML = parts.join('') || '<span class="nbsp-status">Preview will appear after you paste text.</span>';
  };

  const buildCleaned = (text) => {
    let cleaned = text;
    const replaceHard = !fixHardToggle || fixHardToggle.checked;
    const stripNonAscii = Boolean(stripNonAsciiToggle?.checked);
    if (replaceHard) {
      cleaned = cleaned.replace(hardRegex, ' ');
    }
    if (stripNonAscii) {
      cleaned = cleaned.replace(NON_ASCII_REGEX, '');
    }
    return { cleaned, replacedHard: replaceHard, strippedNonAscii: stripNonAscii };
  };

  const setCopyStatus = (msg, tone) => {
    if (!copyStatus) return;
    copyStatus.textContent = msg;
    copyStatus.dataset.tone = tone || '';
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
        const text = normalizeInputText(extractDocxXmlText(xml));
        if (text.trim()) chunks.push(text);
      } catch {}
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
        const normalized = normalizeInputText(pageText);
        if (normalized.trim()) pages.push(normalized);
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

    if (name.endsWith('.pdf') || type.includes('pdf')) return 'pdf';
    if (name.endsWith('.docx') || type.includes('officedocument.wordprocessingml.document')) return 'docx';
    if (name.endsWith('.doc') || type === 'application/msword') return 'doc';
    if (name.endsWith('.rtf') || type.includes('rtf')) return 'rtf';
    if (name.endsWith('.html') || name.endsWith('.htm') || type.includes('html') || type.includes('xml')) return 'html';
    return 'text';
  };

  const parseImportedFile = async (file) => {
    const importType = detectImportType(file);
    if (importType === 'pdf') return { text: await parsePdfFile(file), warning: '' };
    if (importType === 'docx') return { text: await parseDocxFile(file), warning: '' };
    if (importType === 'doc') {
      const bytes = new Uint8Array(await file.arrayBuffer());
      return {
        text: extractLegacyDocText(bytes),
        warning: 'Legacy .doc extraction is best-effort. Convert to .docx for highest fidelity.'
      };
    }
    if (importType === 'rtf') return { text: extractRtfText(await file.text()), warning: '' };
    if (importType === 'html') return { text: extractHtmlText(await file.text()), warning: '' };
    return { text: await file.text(), warning: '' };
  };

  const runCleaner = () => {
    const text = normalizeInputText(input.value || '');
    input.value = text;
    if (!text.trim()) {
      summary.textContent = 'Paste text above, then run the cleaner.';
      countsList.innerHTML = '';
      output.value = '';
      setCopyStatus('');
      preview.innerHTML = '<span class="nbsp-status">Preview will appear after you paste text.</span>';
      return;
    }
    const { perType, total } = analyze(text);
    const nonAsciiCounts = countNonAscii(text);
    renderCounts(perType, nonAsciiCounts);
    const { cleaned, replacedHard, strippedNonAscii } = buildCleaned(text);
    output.value = cleaned;
    const nonAsciiTotal = [...nonAsciiCounts.values()].reduce((sum, v) => sum + v, 0);
    const findings = [];
    if (total > 0) findings.push(`${formatNumber(total)} hard spaces${replacedHard ? ' replaced' : ' detected (kept)'}`);
    if (nonAsciiTotal > 0) findings.push(`${formatNumber(nonAsciiTotal)} other non-ASCII ${strippedNonAscii ? 'removed' : 'detected (kept)'}`);
    if (findings.length) {
      summary.innerHTML = `Found ${findings.join(' and ')}.${strippedNonAscii ? ' Output is ASCII-only.' : ''}`;
    } else {
      summary.textContent = 'No hard spaces or non-ASCII characters detected. Output matches your input.';
    }
    renderPreview(text);
    setCopyStatus('');
    markSessionDirty();
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCleaner();
  });

  clearBtn?.addEventListener('click', () => {
    input.value = '';
    output.value = '';
    countsList.innerHTML = '';
    summary.textContent = 'Paste text and run the cleaner to see findings.';
    setCopyStatus('');
    setInputStatus('');
    preview.innerHTML = '<span class="nbsp-status">Preview will appear after you paste text.</span>';
    markSessionDirty();
    input.focus();
  });

  copyBtn?.addEventListener('click', async () => {
    if (!output.value) {
      setCopyStatus('Nothing to copy yet.', 'error');
      return;
    }
    try {
      await navigator.clipboard.writeText(output.value);
      setCopyStatus('Cleaned text copied.', 'success');
    } catch {
      setCopyStatus('Copy failed. Please copy manually.', 'error');
    }
  });

  const handlePasteInput = async () => {
    setInputBusy(true);
    setInputStatus('Reading clipboard…', 'info');

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.readText !== 'function') {
        throw new Error('Clipboard read is unavailable.');
      }
      const clipboardText = normalizeInputText(await navigator.clipboard.readText());
      if (!clipboardText.trim()) {
        setInputStatus('Clipboard is empty.', 'error');
        return;
      }
      input.value = clipboardText;
      setInputStatus(`Pasted ${clipboardText.length.toLocaleString('en-US')} characters.`, 'success');
      runCleaner();
      input.focus();
    } catch {
      setInputStatus('Clipboard access blocked. Use Ctrl/Cmd+V in the text box.', 'error');
    } finally {
      setInputBusy(false);
    }
  };

  const handleImportInput = async () => {
    const file = fileInput?.files?.[0];
    if (!file) return;

    setInputBusy(true);
    setInputStatus(`Importing ${file.name}…`, 'info');

    try {
      if (file.size > MAX_IMPORT_BYTES) {
        const maxMb = Math.round(MAX_IMPORT_BYTES / (1024 * 1024));
        throw new Error(`File is too large. Max supported size is ${maxMb} MB.`);
      }

      const parsed = await parseImportedFile(file);
      const text = normalizeInputText(parsed.text);
      if (!text.trim()) {
        throw new Error('No readable text was found in this file.');
      }

      input.value = text;
      const status = parsed.warning
        ? `${file.name} imported. ${parsed.warning}`
        : `${file.name} imported (${text.length.toLocaleString('en-US')} characters).`;
      setInputStatus(status, parsed.warning ? 'info' : 'success');
      runCleaner();
      input.focus();
    } catch (error) {
      setInputStatus(error instanceof Error ? error.message : 'Unable to import this file.', 'error');
    } finally {
      if (fileInput) fileInput.value = '';
      setInputBusy(false);
    }
  };

  pasteBtn?.addEventListener('click', () => {
    void handlePasteInput();
  });

  importBtn?.addEventListener('click', () => {
    fileInput?.click();
  });

  fileInput?.addEventListener('change', () => {
    void handleImportInput();
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    payload.outputSummary = String(summary?.textContent || '').replace(/\s+/g, ' ').trim();
    payload.inputs = { Input: input.value || '' };
  });
})();
