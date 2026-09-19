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
  const COMPARE_WORKER_PATH = '/js/tools/text-compare-worker.js';
  const ORIGINAL_EXAMPLE = 'Product analytics should be easy to act on. The next report arrives on Monday.';
  const REVISED_EXAMPLE = 'Product analytics should be easy to use. The next report arrives on Friday.';
  const compareRequests = new Map();
  let lastRuns = null;
  let lastRevisedText = '';
  let compareWorker = null;
  let workerUnavailable = false;
  let latestCompareRequestId = 0;
  let refreshTimer = 0;
  let comparisonStarted = false;
  const legendEl = $('.textcompare-legend');
  const hasUserText = () => originalEl.value.length > 0 || revisedEl.value.length > 0;
  const updateExamplePreview = () => {
    const showingExample = !hasUserText();
    originalEl.placeholder = showingExample ? ORIGINAL_EXAMPLE : 'Paste the text before changes';
    revisedEl.placeholder = showingExample ? REVISED_EXAMPLE : 'Paste the text after changes';
  };
  updateExamplePreview();

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const reportRunComplete = (resultBucket) => {
    try {
      document.dispatchEvent(new CustomEvent('tools:run-complete', {
        detail: { toolId: TOOL_ID, resultBucket }
      }));
    } catch {}
  };

  const reportRunError = (errorType) => {
    try {
      document.dispatchEvent(new CustomEvent('tools:run-error', {
        detail: { toolId: TOOL_ID, errorType }
      }));
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
    insBg: insBgEl?.value || '#DDF2EC',
    insColor: insTextEl?.value || '#091F3B',
    delBg: delBgEl?.value || '#FBE4E8',
    delColor: delTextEl?.value || '#091F3B',
    delStrike: delStrikeEl?.value || '#091F3B'
  });

  const normalizeHexColor = (value, fallback) => {
    const s = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s.toUpperCase();
    return fallback;
  };

  const applyPreviewStyle = () => {
    const style = getCopyStyle();
    document.body.style.setProperty('--textcompare-ins-bg', normalizeHexColor(style.insBg, '#DDF2EC'));
    document.body.style.setProperty('--textcompare-ins-text', normalizeHexColor(style.insColor, '#091F3B'));
    document.body.style.setProperty('--textcompare-del-bg', normalizeHexColor(style.delBg, '#FBE4E8'));
    document.body.style.setProperty('--textcompare-del-text', normalizeHexColor(style.delColor, '#091F3B'));
    document.body.style.setProperty('--textcompare-del-strike', normalizeHexColor(style.delStrike, '#091F3B'));
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
    const insBg = normalizeHexColor(style.insBg, '#DDF2EC');
    const insColor = normalizeHexColor(style.insColor, '#091F3B');
    const delBg = normalizeHexColor(style.delBg, '#FBE4E8');
    const delColor = normalizeHexColor(style.delColor, '#091F3B');
    const delStrike = normalizeHexColor(style.delStrike, '#091F3B');

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
    const insBg = normalizeHexColor(style.insBg, '#DDF2EC');
    const insColor = normalizeHexColor(style.insColor, '#091F3B');
    const delBg = normalizeHexColor(style.delBg, '#FBE4E8');
    const delColor = normalizeHexColor(style.delColor, '#091F3B');

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

  window.SiteRoutes?.addCleanup?.(() => {
    latestCompareRequestId += 1;
    window.clearTimeout(refreshTimer);
  });

  const getSelectedMode = () => {
    const selected = modeInputs.find((input) => input.checked);
    return String(selected?.value || compareCore.MODES.AUTO);
  };

  const formatSummaryCount = (count, singular, plural = `${singular}s`) =>
    `${count.toLocaleString('en-US')} ${count === 1 ? singular : plural}`;

  const formatCompareSummary = (counts) => {
    if (!counts?.hasChanges) return 'No differences found.';
    const parts = [];
    if (counts.insertedWords) parts.push(formatSummaryCount(counts.insertedWords, 'inserted word'));
    if (counts.deletedWords) parts.push(formatSummaryCount(counts.deletedWords, 'deleted word'));
    if (counts.replacements) parts.push(formatSummaryCount(counts.replacements, 'replacement'));
    if (counts.movedBlocks) parts.push(formatSummaryCount(counts.movedBlocks, 'moved block'));
    return `Changes: ${parts.join(' · ')}.`;
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
    if (fallbackWarning) warnings.push(fallbackWarning);

    lastRuns = result?.runs || [];
    lastRevisedText = revisedText;
    if (copyBtn) copyBtn.disabled = !lastRuns.length;
    if (legendEl) legendEl.hidden = !result?.counts?.hasChanges;
    outputEl.innerHTML = renderOutput(lastRuns) || '<p class="textcompare-empty">No output.</p>';
    summaryEl.textContent = formatCompareSummary(result?.counts);
    setWarningStatus(warnings.join(' '), warnings.length ? 'info' : '');
    markSessionDirty();
  };

  const runCompare = ({ reportOutcome = false } = {}) => {
    window.clearTimeout(refreshTimer);
    comparisonStarted = true;
    lastRuns = null;
    if (copyBtn) copyBtn.disabled = true;
    if (legendEl) legendEl.hidden = true;
    setCopyStatus('');
    setWarningStatus('', '');
    markSessionDirty();
    // The example stays out of saved editor values and never fills a blank side of a user's comparison.
    const showingExample = !hasUserText();
    const original = showingExample ? ORIGINAL_EXAMPLE : originalEl.value;
    const revised = showingExample ? REVISED_EXAMPLE : revisedEl.value;

    if (original.length + revised.length > MAX_CHARS) {
      summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
      setEmpty('Input too large.');
      lastRuns = null;
      lastRevisedText = '';
      latestCompareRequestId += 1;
      setWarningStatus('', '');
      markSessionDirty();
      if (reportOutcome) reportRunError('validation');
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
        leftKind: 'text',
        rightKind: 'text'
      }
    };

    requestAnimationFrame(() => {
      void (async () => {
        if (requestId !== latestCompareRequestId) return;
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
              if (reportOutcome) reportRunError('validation');
              return;
            }
          }

          try {
            result = await requestWorkerCompare(payload);
          } catch {
            if (requestId !== latestCompareRequestId) return;
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
          if (reportOutcome) {
            reportRunComplete(result?.counts?.hasChanges ? 'with_changes' : 'no_changes');
          }
        } catch {
          if (requestId === latestCompareRequestId) {
            summaryEl.textContent = 'Unable to compare these drafts.';
            setEmpty('Comparison failed. Please try smaller sections.');
            setWarningStatus('The comparison could not be completed.', 'error');
            markSessionDirty();
          }
          if (reportOutcome) reportRunError('processing');
        }
      })();
    });
  };

  [insBgEl, insTextEl, delBgEl, delTextEl, delStrikeEl].forEach((el) => {
    el?.addEventListener('input', applyPreviewStyle);
  });
  applyPreviewStyle();

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    try {
      runCompare({ reportOutcome: true });
    } catch (error) {
      reportRunError('processing');
      throw error;
    }
  });

  clearBtn?.addEventListener('click', () => {
    window.clearTimeout(refreshTimer);
    comparisonStarted = false;
    if (copyBtn) copyBtn.disabled = true;
    if (legendEl) legendEl.hidden = true;
    latestCompareRequestId += 1;
    originalEl.value = '';
    revisedEl.value = '';
    updateExamplePreview();
    summaryEl.textContent = 'Changes appear here.';
    setEmpty('Compare the example, or enter your own text.');
    lastRuns = null;
    lastRevisedText = '';
    setCopyStatus('');
    setWarningStatus('', '');
    markSessionDirty();
    originalEl.focus();
  });

  swapBtn?.addEventListener('click', () => {
    const a = originalEl.value;
    originalEl.value = revisedEl.value;
    revisedEl.value = a;
    updateExamplePreview();
    runCompare();
  });

  copyBtn?.addEventListener('click', copyFormatted);
  const queueComparison = () => {
    latestCompareRequestId += 1;
    window.clearTimeout(refreshTimer);
    lastRuns = null;
    lastRevisedText = '';
    if (copyBtn) copyBtn.disabled = true;
    if (legendEl) legendEl.hidden = true;
    setCopyStatus('');
    setWarningStatus('', '');
    if (!comparisonStarted) {
      setEmpty(hasUserText() ? 'Click Compare to see the changes.' : 'Compare the example, or enter your own text.');
      return;
    }
    summaryEl.textContent = 'Updating comparison…';
    setEmpty('Updating…');
    refreshTimer = window.setTimeout(() => runCompare(), 450);
  };
  [originalEl, revisedEl].forEach(field => field.addEventListener('input', () => {
    updateExamplePreview();
    queueComparison();
  }));
  modeInputs.forEach((input) => input.addEventListener('change', queueComparison));
  const MAX_SAVED_OUTPUT_HTML_CHARS = 120_000;
  const MAX_SAVED_OUTPUT_TEXT_CHARS = 120_000;

  const clampText = (value, maxChars) => {
    const text = String(value || '');
    if (text.length <= maxChars) return { text, truncated: false };
    return { text: text.slice(0, maxChars), truncated: true };
  };

  const hasSavedComparison = (output) => {
    if (output?.kind === 'html') {
      const html = String(output.html || '').trim();
      return Boolean(html) && !/class\s*=\s*["'][^"']*\btextcompare-empty\b/.test(html);
    }
    if (output?.kind !== 'text') return false;
    const text = String(output.text || '').trim();
    return Boolean(text) && ![
      'Waiting for input.', 'Ready to compare.', 'No output.', 'Add both drafts, then compare.', 'Updating…',
      'Click Compare to see the changes.', 'Compare the example, or enter your own text.',
      'Paste text in both boxes, then click Compare.', 'Comparing…',
      'Input too large.', 'Comparison failed. Please try smaller sections.'
    ].includes(text);
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
    // Legacy view preferences no longer hide drafts or results.
    updateExamplePreview();
    window.clearTimeout(refreshTimer);
    setCopyStatus('');
    setWarningStatus('', '');
    latestCompareRequestId += 1;
    lastRuns = null;
    lastRevisedText = '';
    if (copyBtn) copyBtn.disabled = true;
    if (legendEl) legendEl.hidden = true;
    comparisonStarted = hasSavedComparison(output);
    if (comparisonStarted) {
      runCompare();
      return;
    }
    summaryEl.textContent = 'Changes appear here.';
    setEmpty(hasUserText() ? 'Click Compare to see the changes.' : 'Compare the example, or enter your own text.');
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
