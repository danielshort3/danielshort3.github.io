(() => {
  'use strict';
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const form = $('[data-oxford-form]');
  const input = $('[data-oxford-input]');
  const summary = $('[data-oxford-summary]');
  const counts = $('[data-oxford-counts]');
  const resultsList = $('[data-oxford-results]');
  const empty = $('[data-oxford-empty]');
  const output = $('[data-oxford-output]');
  const clearBtn = $('[data-oxford-clear]');
  const conjInputs = $$('[data-oxford-conjunction]');
  const presentColorInput = $('[data-oxford-color="present"]');
  const absentColorInput = $('[data-oxford-color="absent"]');
  const resetColorsBtn = $('[data-oxford-reset-colors]');
  const pasteBtn = $('#oxford-paste');
  const importBtn = $('#oxford-import');
  const fileInput = $('#oxford-file');
  const inputStatus = $('#oxford-input-status');
  let hasRun = false;

  if (!form || !input || !summary || !counts || !output) return;

  const TOOL_ID = 'oxford-comma-checker';
  const MAX_SAVED_OUTPUT_HTML_CHARS = 120_000;
  const MAX_IMPORT_BYTES = 24 * 1024 * 1024;
  const PDF_WORKER_PATH = '/js/vendor/pdfjs/pdf.worker.min.js';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapeHtml = (value) => value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const normalizeInputText = (value) => String(value || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');
  const formatNumber = (value) => value.toLocaleString('en-US');
  const truncate = (value, max = 180) => (value.length <= max ? value : `${value.slice(0, max - 3)}...`);

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

  const defaultColors = {
    present: presentColorInput?.value || '',
    absent: absentColorInput?.value || '',
  };

  const applyColorVars = () => {
    if (presentColorInput?.value) {
      document.body.style.setProperty('--oxford-present-color', presentColorInput.value);
    }
    if (absentColorInput?.value) {
      document.body.style.setProperty('--oxford-absent-color', absentColorInput.value);
    }
  };

  applyColorVars();

  const getTextForAnalysis = () => {
    const raw = normalizeInputText(input.value || '');
    input.value = raw;
    if (raw.trim()) {
      return { text: raw, isPlaceholder: false };
    }
    const placeholder = input.placeholder || '';
    if (placeholder.trim()) {
      return { text: placeholder, isPlaceholder: true };
    }
    return { text: '', isPlaceholder: false };
  };

  const getConjunctions = () => conjInputs
    .filter((item) => item.checked)
    .map((item) => (item.dataset.oxfordConjunction || '').toLowerCase())
    .filter(Boolean);

  const buildRegex = (conjunctions) => {
    if (!conjunctions.length) return null;
    const conjPattern = conjunctions.map((c) => escapeRegExp(c)).join('|');
    return new RegExp(
      `([^.!?\\n;:]+(?:,[^.!?\\n;:]+)+\\s+\\b(${conjPattern})\\b\\s+[^.!?\\n;:]+)(?=[.!?\\n;:]|$)`,
      'gi'
    );
  };

  const findLastConjunctionIndex = (text, conj) => {
    const regex = new RegExp(`\\b${escapeRegExp(conj)}\\b`, 'gi');
    let idx = -1;
    let match;
    while ((match = regex.exec(text)) !== null) {
      idx = match.index;
    }
    return idx;
  };

  const findMatches = (text, conjunctions) => {
    const regex = buildRegex(conjunctions);
    if (!regex) return [];
    const matches = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
      const raw = match[0];
      const conj = (match[2] || '').toLowerCase();
      if (!conj) continue;
      const conjIndex = findLastConjunctionIndex(raw, conj);
      if (conjIndex < 0) continue;
      const before = raw.slice(0, conjIndex).trimEnd();
      const hasOxford = /,\s*$/.test(before);
      matches.push({
        start: match.index,
        end: match.index + raw.length,
        text: raw,
        conj,
        hasOxford,
      });
    }
    return matches;
  };

  const renderCounts = (total, present, missing) => {
    counts.innerHTML = `
      <li>
        <strong>${formatNumber(total)}</strong>
        <span>List candidates</span>
      </li>
      <li>
        <strong>${formatNumber(present)}</strong>
        <span>Oxford comma present</span>
      </li>
      <li>
        <strong>${formatNumber(missing)}</strong>
        <span>Oxford comma absent</span>
      </li>
    `;
  };

  const renderResults = (matches) => {
    if (!resultsList) return;
    resultsList.innerHTML = '';
    if (!matches.length) {
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'No list candidates detected yet.';
      }
      return;
    }
    if (empty) empty.hidden = true;
    matches.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'oxford-result';
      li.dataset.status = item.hasOxford ? 'present' : 'missing';
      const snippet = truncate(item.text.trim());
      li.innerHTML = `
        <div class="oxford-result-header">
          <span class="oxford-result-status">${item.hasOxford ? 'Oxford comma present' : 'Oxford comma absent'}</span>
          <span class="oxford-result-tag">${item.conj}</span>
        </div>
        <p class="oxford-result-text">${escapeHtml(snippet)}</p>
      `;
      resultsList.appendChild(li);
    });
  };

  const renderOutput = (text, highlights) => {
    if (!text.trim()) {
      output.innerHTML = '<p class="oxford-empty">Paste text and click Check.</p>';
      return;
    }
    if (!highlights.length) {
      output.textContent = text;
      return;
    }
    const sorted = highlights.slice().sort((a, b) => a.start - b.start);
    let cursor = 0;
    let html = '';
    sorted.forEach((range) => {
      if (range.start < cursor) return;
      const tone = range.status === 'present' ? 'present' : 'missing';
      html += escapeHtml(text.slice(cursor, range.start));
      html += `<mark class="oxford-mark oxford-mark-${tone}">${escapeHtml(text.slice(range.start, range.end))}</mark>`;
      cursor = range.end;
    });
    html += escapeHtml(text.slice(cursor));
    output.innerHTML = html;
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

  const parseDocxFile = async (file) => {
    const api = window.fflate;
    if (!api || typeof api.unzipSync !== 'function') {
      throw new Error('DOCX import is unavailable: zip parser failed to load.');
    }
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
    const pdfjs = window.pdfjsLib;
    if (!pdfjs || typeof pdfjs.getDocument !== 'function') {
      throw new Error('PDF import is unavailable: parser failed to load.');
    }
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

  const runAnalysis = () => {
    const { text } = getTextForAnalysis();
    if (!text.trim()) {
      hasRun = false;
      summary.textContent = 'Paste text and click Check.';
      counts.innerHTML = '';
      if (resultsList) resultsList.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Waiting for input.';
      }
      renderOutput('', []);
      return;
    }
    const conjunctions = getConjunctions();
    if (!conjunctions.length) {
      hasRun = true;
      summary.textContent = 'Select at least one conjunction to scan.';
      counts.innerHTML = '';
      if (resultsList) resultsList.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Choose "and", "or", or "nor" to scan lists.';
      }
      renderOutput(text, []);
      return;
    }
    const matches = findMatches(text, conjunctions);
    const missing = matches.filter((item) => !item.hasOxford);
    const missingCount = missing.length;
    const present = matches.length - missingCount;
    renderCounts(matches.length, present, missingCount);
    hasRun = true;
    if (matches.length) {
      const majorityLabel = present === missingCount
        ? 'Tie'
        : present > missingCount
          ? 'Oxford comma present'
          : 'Oxford comma absent';
      summary.textContent = `${formatNumber(matches.length)} list candidate${matches.length === 1 ? '' : 's'} found. Oxford comma present in ${formatNumber(present)}, absent in ${formatNumber(missingCount)}. Majority: ${majorityLabel}.`;
    } else {
      summary.textContent = 'No list candidates detected. Try a longer sample.';
    }
    const highlights = matches.map((item) => ({
      start: item.start,
      end: item.end,
      status: item.hasOxford ? 'present' : 'missing',
    }));
    renderResults(matches);
    renderOutput(text, highlights);
    markSessionDirty();
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  const maybeRerun = () => {
    if (!hasRun) return;
    runAnalysis();
  };

  conjInputs.forEach((item) => {
    item.addEventListener('change', maybeRerun);
  });
  presentColorInput?.addEventListener('input', applyColorVars);
  absentColorInput?.addEventListener('input', applyColorVars);
  resetColorsBtn?.addEventListener('click', () => {
    if (presentColorInput && defaultColors.present) {
      presentColorInput.value = defaultColors.present;
    }
    if (absentColorInput && defaultColors.absent) {
      absentColorInput.value = defaultColors.absent;
    }
    applyColorVars();
    markSessionDirty();
  });
  clearBtn?.addEventListener('click', () => {
    input.value = '';
    hasRun = false;
    counts.innerHTML = '';
    if (resultsList) resultsList.innerHTML = '';
    summary.textContent = 'Paste text and click Check.';
    setInputStatus('');
    if (empty) {
      empty.hidden = false;
      empty.textContent = 'Waiting for input.';
    }
    output.innerHTML = '<p class="oxford-empty">Paste text and click Check.</p>';
    markSessionDirty();
    input.focus();
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
      runAnalysis();
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
      runAnalysis();
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

  const clampText = (value, maxChars) => {
    const text = String(value || '');
    if (text.length <= maxChars) return { text, truncated: false };
    return { text: text.slice(0, maxChars), truncated: true };
  };

  const captureSummary = () => String(summary?.textContent || '').replace(/\s+/g, ' ').trim();

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const outSummary = captureSummary();
    payload.outputSummary = outSummary;
    payload.inputs = { Text: input.value || '' };

    const html = String(output?.innerHTML || '').trim();
    if (html && html.length <= MAX_SAVED_OUTPUT_HTML_CHARS) {
      payload.output = { kind: 'html', html, summary: outSummary };
      return;
    }

    const rawText = String(output?.textContent || '').trim();
    const { text, truncated } = clampText(rawText, 120_000);
    if (text) payload.output = { kind: 'text', text, summary: outSummary, truncated };
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    requestAnimationFrame(() => {
      try {
        runAnalysis();
      } catch {}
    });
  });
})();
