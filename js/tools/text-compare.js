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
  const insBgEl = $('#textcompare-ins-bg');
  const insTextEl = $('#textcompare-ins-text');
  const delBgEl = $('#textcompare-del-bg');
  const delTextEl = $('#textcompare-del-text');
  const delStrikeEl = $('#textcompare-del-strike');

  if (!form || !originalEl || !revisedEl || !outputEl || !summaryEl) return;

  const MAX_CHARS = 200_000;
  const MAX_TOKENS = 20_000;
  let lastRuns = null;
  let lastRevisedText = '';

  const escapeHtml = (s) => String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const countWords = (s) => (String(s || '').match(/\S+/g) || []).length;

  const setCopyStatus = (msg, tone) => {
    if (!copyStatus) return;
    copyStatus.textContent = msg;
    copyStatus.dataset.tone = tone || '';
  };

  const tokenize = (text) => {
    const s = String(text || '');
    const tokens = [];
    const len = s.length;
    let i = 0;

    if (i < len && /\s/.test(s[i])) {
      let j = i;
      while (j < len && /\s/.test(s[j])) j += 1;
      tokens.push(s.slice(i, j));
      i = j;
    }

    while (i < len) {
      let wordEnd = i;
      while (wordEnd < len && !/\s/.test(s[wordEnd])) wordEnd += 1;
      let spaceEnd = wordEnd;
      while (spaceEnd < len && /\s/.test(s[spaceEnd])) spaceEnd += 1;
      tokens.push(s.slice(i, spaceEnd));
      i = spaceEnd;
    }
    return tokens;
  };

  const myersEdits = (a, b) => {
    const n = a.length;
    const m = b.length;
    const max = n + m;
    const offset = max;
    let v = new Array(2 * max + 1).fill(0);
    const trace = [];

    const backtrack = () => {
      let x = n;
      let y = m;
      const edits = [];
      const D = trace.length - 1;

      for (let d = D; d > 0; d -= 1) {
        const vPrev = trace[d - 1];
        const k = x - y;
        const kIdx = k + offset;
        const down = k === -d || (k !== d && vPrev[kIdx - 1] < vPrev[kIdx + 1]);
        const prevK = down ? k + 1 : k - 1;
        const prevX = vPrev[prevK + offset];
        const prevY = prevX - prevK;

        while (x > prevX && y > prevY) {
          edits.push({ type: 'equal', value: a[x - 1] });
          x -= 1;
          y -= 1;
        }

        if (down) {
          edits.push({ type: 'insert', value: b[prevY] });
        } else {
          edits.push({ type: 'delete', value: a[prevX] });
        }

        x = prevX;
        y = prevY;
      }

      while (x > 0 && y > 0) {
        edits.push({ type: 'equal', value: a[x - 1] });
        x -= 1;
        y -= 1;
      }
      while (x > 0) {
        edits.push({ type: 'delete', value: a[x - 1] });
        x -= 1;
      }
      while (y > 0) {
        edits.push({ type: 'insert', value: b[y - 1] });
        y -= 1;
      }

      edits.reverse();
      return edits;
    };

    for (let d = 0; d <= max; d += 1) {
      for (let k = -d; k <= d; k += 2) {
        const kIdx = k + offset;
        let x;
        const down = k === -d || (k !== d && v[kIdx - 1] < v[kIdx + 1]);
        if (down) {
          x = v[kIdx + 1];
        } else {
          x = v[kIdx - 1] + 1;
        }
        let y = x - k;
        while (x < n && y < m && a[x] === b[y]) {
          x += 1;
          y += 1;
        }
        v[kIdx] = x;
        if (x >= n && y >= m) {
          trace.push(v.slice());
          return backtrack();
        }
      }
      trace.push(v.slice());
    }
    return [];
  };

  const groupRuns = (edits) => {
    const runs = [];
    let current = null;
    edits.forEach((edit) => {
      if (!current || current.type !== edit.type) {
        current = { type: edit.type, tokens: [edit.value] };
        runs.push(current);
      } else {
        current.tokens.push(edit.value);
      }
    });
    return runs;
  };

  const normalizeRuns = (runs) => {
    const out = [];
    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      const next = runs[i + 1];
      if (run.type === 'delete' && next && next.type === 'insert') {
        out.push({ type: 'replace', delTokens: run.tokens, insTokens: next.tokens });
        i += 1;
        continue;
      }
      if (run.type === 'insert' && next && next.type === 'delete') {
        out.push({ type: 'replace', delTokens: next.tokens, insTokens: run.tokens });
        i += 1;
        continue;
      }
      out.push(run);
    }
    return out;
  };

  const mergeEdits = (edits) => {
    const merged = [];
    let current = null;
    edits.forEach((edit) => {
      if (!current || current.type !== edit.type) {
        current = { type: edit.type, text: edit.value };
        merged.push(current);
      } else {
        current.text += edit.value;
      }
    });
    return merged;
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
    const edits = myersEdits(Array.from(delCore), Array.from(insCore));
    const segments = mergeEdits(edits);
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

  const renderOutput = (runs) => runs.map((run) => {
    if (run.type === 'equal') return escapeHtml(run.tokens.join(''));
    if (run.type === 'insert') return `<ins class="diff-ins">${escapeHtml(run.tokens.join(''))}</ins>`;
    if (run.type === 'delete') return `<del class="diff-del">${escapeHtml(run.tokens.join(''))}</del>`;
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
    insBg: insBgEl?.value || '#C6EFCE',
    insColor: insTextEl?.value || '#000000',
    delBg: delBgEl?.value || '#FFC7CE',
    delColor: delTextEl?.value || '#000000',
    delStrike: delStrikeEl?.value || '#9C0006'
  });

  const normalizeHexColor = (value, fallback) => {
    const s = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s.toUpperCase();
    return fallback;
  };

  const applyPreviewStyle = () => {
    const style = getCopyStyle();
    document.body.style.setProperty('--textcompare-ins-bg', normalizeHexColor(style.insBg, '#C6EFCE'));
    document.body.style.setProperty('--textcompare-ins-text', normalizeHexColor(style.insColor, '#000000'));
    document.body.style.setProperty('--textcompare-del-bg', normalizeHexColor(style.delBg, '#FFC7CE'));
    document.body.style.setProperty('--textcompare-del-text', normalizeHexColor(style.delColor, '#000000'));
    document.body.style.setProperty('--textcompare-del-strike', normalizeHexColor(style.delStrike, '#9C0006'));
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
    const insBg = normalizeHexColor(style.insBg, '#C6EFCE');
    const insColor = normalizeHexColor(style.insColor, '#000000');
    const delBg = normalizeHexColor(style.delBg, '#FFC7CE');
    const delColor = normalizeHexColor(style.delColor, '#000000');
    const delStrike = normalizeHexColor(style.delStrike, '#9C0006');

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
    return `<!doctype html><html><head><meta charset="utf-8"></head><body><div style="${bodyStyle}"><!--StartFragment-->${fragment}<!--EndFragment--></div></body></html>`;
  };

  const buildClipboardRtf = (runs, style) => {
    const insBg = normalizeHexColor(style.insBg, '#C6EFCE');
    const insColor = normalizeHexColor(style.insColor, '#000000');
    const delBg = normalizeHexColor(style.delBg, '#FFC7CE');
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

    try {
      if (navigator.clipboard && window.ClipboardItem) {
        const item = new ClipboardItem({
          'text/html': new Blob([html], { type: 'text/html' }),
          'text/plain': new Blob([lastRevisedText || ''], { type: 'text/plain' }),
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
      temp.innerHTML = buildClipboardFragment(lastRuns, style);
      document.body.appendChild(temp);

      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(temp);
      selection?.removeAllRanges();
      selection?.addRange(range);
      const ok = document.execCommand('copy');
      selection?.removeAllRanges();
      temp.remove();

      setCopyStatus(ok ? 'Copied with formatting.' : 'Copy failed.', ok ? 'success' : 'error');
    } catch {
      setCopyStatus('Copy failed. Try selecting the output and copying manually.', 'error');
    }
  };

  const runCompare = () => {
    setCopyStatus('');
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
      return;
    }

    if (!original.trim() && !revised.trim()) {
      summaryEl.textContent = 'Paste text above and click Compare.';
      setEmpty('Waiting for input.');
      lastRuns = null;
      lastRevisedText = '';
      return;
    }

    if (original.length + revised.length > MAX_CHARS) {
      summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
      setEmpty('Input too large.');
      lastRuns = null;
      lastRevisedText = '';
      return;
    }

    summaryEl.textContent = 'Comparing…';
    setEmpty('Comparing…');

    requestAnimationFrame(() => {
      const aTokens = tokenize(original);
      const bTokens = tokenize(revised);
      if (aTokens.length + bTokens.length > MAX_TOKENS) {
        summaryEl.textContent = 'Text is too large to compare quickly. Please compare smaller sections.';
        setEmpty('Input too large.');
        lastRuns = null;
        lastRevisedText = '';
        return;
      }

      const edits = myersEdits(aTokens, bTokens);
      const runs = normalizeRuns(groupRuns(edits));
      lastRuns = runs;
      lastRevisedText = revised;
      const html = renderOutput(runs);
      outputEl.innerHTML = html || '<p class="textcompare-empty">No output.</p>';

      let insertedWords = 0;
      let deletedWords = 0;
      let replacements = 0;
      runs.forEach((run) => {
        if (run.type === 'insert') insertedWords += countWords(run.tokens.join(''));
        if (run.type === 'delete') deletedWords += countWords(run.tokens.join(''));
        if (run.type === 'replace') {
          replacements += 1;
          insertedWords += countWords(run.insTokens.join(''));
          deletedWords += countWords(run.delTokens.join(''));
        }
      });

      if (!insertedWords && !deletedWords) {
        summaryEl.textContent = 'No differences found.';
        return;
      }
      const parts = [];
      if (insertedWords) parts.push(`${insertedWords.toLocaleString('en-US')} inserted words`);
      if (deletedWords) parts.push(`${deletedWords.toLocaleString('en-US')} deleted words`);
      if (replacements) parts.push(`${replacements.toLocaleString('en-US')} replacements`);
      summaryEl.textContent = `Changes: ${parts.join(' · ')}.`;
    });
  };

  [insBgEl, insTextEl, delBgEl, delTextEl, delStrikeEl].forEach((el) => {
    el?.addEventListener('input', applyPreviewStyle);
  });
  applyPreviewStyle();

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCompare();
  });

  clearBtn?.addEventListener('click', () => {
    originalEl.value = '';
    revisedEl.value = '';
    summaryEl.textContent = 'Paste text above and click Compare.';
    setEmpty('Waiting for input.');
    lastRuns = null;
    lastRevisedText = '';
    setCopyStatus('');
    originalEl.focus();
  });

  swapBtn?.addEventListener('click', () => {
    const a = originalEl.value;
    originalEl.value = revisedEl.value;
    revisedEl.value = a;
    runCompare();
  });

  copyBtn?.addEventListener('click', copyFormatted);
})();
