(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
    return;
  }
  root.TextCompareCore = factory();
}(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  'use strict';

  const MODES = {
    AUTO: 'auto',
    DOCUMENT: 'document',
    STRUCTURED: 'structured',
    PROSE: 'document'
  };

  const MAX_TRACE_CELLS = 16_000_000;
  const MAX_TOKEN_DIFF_DEPTH = 2;
  const MAX_DOCUMENT_DIFF_DEPTH = 4;
  const MAX_ANCHOR_CANDIDATES = 6_000;
  const MAX_SIMILARITY_CANDIDATES = 7_500;
  const AUTO_COMPARE_BASE_SETTINGS = {
    mergeMinWords: 8,
    mergeMinChars: 40,
    mergeMinRuns: 4,
    softEqualMaxWords: 2,
    softEqualMaxWordLength: 2,
    commonWordMinCount: 2
  };

  const buildTokenRegexes = () => {
    const unicodeWord = "[\\p{L}\\p{N}_]+(?:['-][\\p{L}\\p{N}_]+)*";
    try {
      return {
        word: new RegExp(unicodeWord, 'gu'),
        wordToken: new RegExp(`^${unicodeWord}$`, 'u'),
        wordChar: new RegExp('[\\p{L}\\p{N}_]', 'u'),
        token: new RegExp(`${unicodeWord}|\\s+|[^\\p{L}\\p{N}_\\s]+`, 'gu')
      };
    } catch {
      const asciiWord = "[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*";
      return {
        word: new RegExp(asciiWord, 'g'),
        wordToken: new RegExp(`^${asciiWord}$`),
        wordChar: new RegExp('[A-Za-z0-9_]'),
        token: new RegExp(`${asciiWord}|\\s+|[^A-Za-z0-9_\\s]+`, 'g')
      };
    }
  };

  const {
    word: WORD_RE,
    token: TOKEN_RE,
    wordToken: WORD_TOKEN_RE,
    wordChar: WORD_CHAR_RE
  } = buildTokenRegexes();

  const sentenceSegmenter = (() => {
    try {
      if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
        return new Intl.Segmenter(undefined, { granularity: 'sentence' });
      }
    } catch {}
    return null;
  })();

  const STRUCTURED_SOURCE_KINDS = new Set(['csv', 'tsv', 'json', 'xml']);
  const DOCUMENT_SOURCE_KINDS = new Set(['doc', 'docx', 'pdf', 'rtf', 'markdown', 'md', 'text']);
  const UNORDERED_LIST_MARKER_RE = /^\s*(?:[-*+•◦▪‣])\s+/;
  const ORDERED_LIST_MARKER_RE = /^\s*(?:(?:\d{1,4}|[A-Za-z])[.)])\s+/;
  const QUOTE_LINE_RE = /^\s*>+/;

  const matchAll = (re, text) => {
    re.lastIndex = 0;
    return text.match(re) || [];
  };

  const clampNumber = (value, min, max) => Math.min(max, Math.max(min, value));

  const normalizeMode = (value) => {
    const mode = String(value || '').trim().toLowerCase();
    if (mode === 'prose' || mode === MODES.DOCUMENT) return MODES.DOCUMENT;
    if (mode === MODES.STRUCTURED) return MODES.STRUCTURED;
    return MODES.AUTO;
  };

  const normalizeSourceKind = (value) => {
    const kind = String(value || '').trim().toLowerCase();
    if (!kind) return 'text';
    if (kind === 'md') return 'markdown';
    if (kind === 'htm') return 'html';
    return kind;
  };

  const countWords = (value) => matchAll(WORD_RE, String(value || '')).length;

  const tokenize = (value) => matchAll(TOKEN_RE, String(value || ''));

  const isWordToken = (token) => WORD_TOKEN_RE.test(String(token || ''));

  const hasWordChar = (text) => WORD_CHAR_RE.test(String(text || ''));

  const countWordTokens = (tokens) => (tokens || []).reduce(
    (sum, token) => sum + (isWordToken(token) ? 1 : 0),
    0
  );

  const collectLowerWords = (text) => matchAll(WORD_RE, String(text || '')).map((word) => word.toLowerCase());

  const countMatches = (text, re) => matchAll(re, String(text || '')).length;

  const wordOverlapScore = (aWords, bWords) => {
    if (!aWords.length || !bWords.length) return 0;
    const counts = new Map();
    aWords.forEach((word) => counts.set(word, (counts.get(word) || 0) + 1));
    let overlap = 0;
    bWords.forEach((word) => {
      const available = counts.get(word) || 0;
      if (available <= 0) return;
      overlap += 1;
      counts.set(word, available - 1);
    });
    return (2 * overlap) / (aWords.length + bWords.length);
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

  const textSimilarityScore = (leftText, rightText) => {
    const left = String(leftText || '').trim();
    const right = String(rightText || '').trim();
    if (!left && !right) return 1;
    if (!left || !right) return 0;
    if (left === right) return 1;

    const aWords = collectLowerWords(left);
    const bWords = collectLowerWords(right);
    const lexical = wordOverlapScore(aWords, bWords);
    const maxLength = Math.max(left.length, right.length) || 1;
    const lengthRatio = Math.min(left.length, right.length) / maxLength;
    const edgeRatio = sharedEdgeScore(left, right) / maxLength;
    return (lexical * 0.7) + (lengthRatio * 0.2) + (edgeRatio * 0.1);
  };

  const shouldPairAsReplace = (delTokens, insTokens) => {
    const delText = (delTokens || []).join('');
    const insText = (insTokens || []).join('');
    if (!delText && !insText) return false;

    const delTrim = delText.trim();
    const insTrim = insText.trim();
    if (!delTrim || !insTrim) return false;
    if (delTrim === insTrim) return true;

    const delWords = countWords(delTrim);
    const insWords = countWords(insTrim);
    const longEnough = delWords >= 3 || insWords >= 3 || delTrim.length >= 28 || insTrim.length >= 28;
    const score = textSimilarityScore(delTrim, insTrim);
    if (!longEnough) return score >= 0.18;
    return score >= 0.3;
  };

  const getAutoCompareSettings = (aTokens, bTokens) => {
    const totalTokenCount = aTokens.length + bTokens.length;
    const totalWordCount = countWordTokens(aTokens) + countWordTokens(bTokens);
    const meanWords = totalWordCount / 2;
    const wordDensity = totalTokenCount ? totalWordCount / totalTokenCount : 0;
    const sizeFactor = clampNumber(meanWords / 1_800, 0, 1);
    const punctuationHeavy = wordDensity < 0.42;
    const denseLargeDoc = meanWords > 2_800;

    return {
      mergeMinWords: clampNumber(
        Math.round(AUTO_COMPARE_BASE_SETTINGS.mergeMinWords + (sizeFactor * 8) + (punctuationHeavy ? 2 : 0)),
        4,
        24
      ),
      mergeMinChars: clampNumber(
        Math.round(AUTO_COMPARE_BASE_SETTINGS.mergeMinChars + (sizeFactor * 72) + (punctuationHeavy ? 10 : 0)),
        20,
        160
      ),
      mergeMinRuns: clampNumber(
        Math.round(AUTO_COMPARE_BASE_SETTINGS.mergeMinRuns + (sizeFactor * 4)),
        2,
        10
      ),
      softEqualMaxWords: denseLargeDoc ? 1 : AUTO_COMPARE_BASE_SETTINGS.softEqualMaxWords,
      softEqualMaxWordLength: meanWords > 2_000 ? 1 : AUTO_COMPARE_BASE_SETTINGS.softEqualMaxWordLength,
      commonWordMinCount: meanWords > 1_200 ? 3 : AUTO_COMPARE_BASE_SETTINGS.commonWordMinCount
    };
  };

  const buildWordCounts = (tokens) => {
    const counts = new Map();
    (tokens || []).forEach((token) => {
      if (!isWordToken(token)) return;
      const key = token.toLowerCase();
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    return counts;
  };

  const isCommonWord = (token, countsA, countsB, settings) => {
    const key = token.toLowerCase();
    const threshold = settings.commonWordMinCount;
    return (countsA.get(key) || 0) >= threshold || (countsB.get(key) || 0) >= threshold;
  };

  const isSoftEqual = (run, countsA, countsB, settings) => {
    if (run.type !== 'equal') return false;
    const text = run.tokens.join('');
    if (!text) return true;
    if (!hasWordChar(text)) return true;
    const words = run.tokens.filter(isWordToken);
    if (!words.length) return true;
    if (settings.softEqualMaxWords <= 0) return false;
    if (words.length <= settings.softEqualMaxWords && words.every((word) => isCommonWord(word, countsA, countsB, settings))) return true;
    if (
      settings.softEqualMaxWordLength > 0 &&
      words.length <= settings.softEqualMaxWords &&
      words.every((word) => word.length <= settings.softEqualMaxWordLength)
    ) {
      return true;
    }
    return false;
  };

  const coalesceRuns = (runs, countsA, countsB, settings) => {
    const merged = [];
    const minMergeWords = settings.mergeMinWords;
    const minMergeChars = settings.mergeMinChars;
    const minMergeRuns = settings.mergeMinRuns;

    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      if (run.type === 'equal') {
        merged.push(run);
        continue;
      }

      const segment = [];
      let hasInsert = false;
      let hasDelete = false;
      let softEqualCount = 0;
      let editRunCount = 0;
      let editWordCount = 0;
      let editCharCount = 0;
      let j = i;

      while (j < runs.length) {
        const current = runs[j];
        if (current.type === 'equal' && !isSoftEqual(current, countsA, countsB, settings)) break;
        segment.push(current);
        if (current.type === 'equal') {
          softEqualCount += 1;
        } else {
          editRunCount += 1;
          if (current.type === 'insert') {
            hasInsert = true;
            editWordCount += countWordTokens(current.tokens);
            editCharCount += current.tokens.join('').length;
          } else if (current.type === 'delete') {
            hasDelete = true;
            editWordCount += countWordTokens(current.tokens);
            editCharCount += current.tokens.join('').length;
          } else if (current.type === 'replace') {
            hasInsert = true;
            hasDelete = true;
            editWordCount += countWordTokens(current.delTokens) + countWordTokens(current.insTokens);
            editCharCount += current.delTokens.join('').length + current.insTokens.join('').length;
          }
        }
        j += 1;
      }

      const shouldMerge = hasInsert && hasDelete && softEqualCount > 0 && (
        editWordCount >= minMergeWords ||
        editCharCount >= minMergeChars ||
        editRunCount >= minMergeRuns
      );

      if (shouldMerge) {
        const delTokens = [];
        const insTokens = [];
        segment.forEach((seg) => {
          if (seg.type === 'equal') {
            delTokens.push(...seg.tokens);
            insTokens.push(...seg.tokens);
          } else if (seg.type === 'insert') {
            insTokens.push(...seg.tokens);
          } else if (seg.type === 'delete') {
            delTokens.push(...seg.tokens);
          } else if (seg.type === 'replace') {
            delTokens.push(...seg.delTokens);
            insTokens.push(...seg.insTokens);
          }
        });
        merged.push({ type: 'replace', delTokens, insTokens });
      } else {
        merged.push(...segment);
      }

      i = j - 1;
    }
    return merged;
  };

  const myersEdits = (a, b, options) => {
    const n = a.length;
    const m = b.length;
    if (!n && !m) return [];
    if (!n) return b.map((value) => ({ type: 'insert', value }));
    if (!m) return a.map((value) => ({ type: 'delete', value }));

    const max = n + m;
    const maxDFromTrace = Math.max(0, Math.floor(Math.sqrt(MAX_TRACE_CELLS)) - 1);
    const maxD = Math.min(
      max,
      Math.max(0, options?.maxD ?? maxDFromTrace)
    );

    const trace = [];
    let vPrev = new Int32Array(1);
    let prefix = 0;
    while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix += 1;
    vPrev[0] = prefix;
    trace.push(vPrev);
    if (prefix >= n && prefix >= m) {
      return a.map((value) => ({ type: 'equal', value }));
    }

    const backtrack = (D) => {
      let x = n;
      let y = m;
      const edits = [];

      for (let d = D; d > 0; d -= 1) {
        const v = trace[d - 1];
        const offset = d - 1;
        const k = x - y;
        const down = k === -d || (k !== d && v[k - 1 + offset] < v[k + 1 + offset]);
        const prevK = down ? k + 1 : k - 1;
        const prevX = v[prevK + offset];
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

    for (let d = 1; d <= maxD; d += 1) {
      const offsetPrev = d - 1;
      const vNext = new Int32Array(2 * d + 1);
      const offset = d;

      for (let k = -d; k <= d; k += 2) {
        const kIdx = k + offset;
        const down = k === -d || (k !== d && vPrev[k - 1 + offsetPrev] < vPrev[k + 1 + offsetPrev]);
        let x;
        if (down) {
          x = vPrev[k + 1 + offsetPrev];
        } else {
          x = vPrev[k - 1 + offsetPrev] + 1;
        }
        let y = x - k;
        while (x < n && y < m && a[x] === b[y]) {
          x += 1;
          y += 1;
        }
        vNext[kIdx] = x;

        if (x >= n && y >= m) {
          trace.push(vNext);
          return backtrack(d);
        }
      }

      trace.push(vNext);
      vPrev = vNext;
    }

    return null;
  };

  const groupRuns = (edits) => {
    const runs = [];
    let current = null;
    (edits || []).forEach((edit) => {
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
    const pushRun = (run) => {
      if (!run) return;
      const last = out[out.length - 1];
      if (!last || last.type !== run.type) {
        out.push(run);
        return;
      }
      if (run.type === 'replace') {
        last.delTokens.push(...run.delTokens);
        last.insTokens.push(...run.insTokens);
        return;
      }
      last.tokens.push(...run.tokens);
    };

    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      const next = runs[i + 1];
      if (next && ((run.type === 'delete' && next.type === 'insert') || (run.type === 'insert' && next.type === 'delete'))) {
        const delTokens = run.type === 'delete' ? run.tokens : next.tokens;
        const insTokens = run.type === 'insert' ? run.tokens : next.tokens;
        if (shouldPairAsReplace(delTokens, insTokens)) {
          pushRun({ type: 'replace', delTokens: [...delTokens], insTokens: [...insTokens] });
          i += 1;
          continue;
        }
      }
      if (run.type === 'replace') {
        pushRun({ type: 'replace', delTokens: [...run.delTokens], insTokens: [...run.insTokens] });
        continue;
      }
      pushRun({ type: run.type, tokens: [...run.tokens] });
    }
    return out;
  };

  const cloneRun = (run) => {
    if (!run || !run.type) return null;
    if (run.type === 'replace') {
      return {
        type: 'replace',
        delTokens: [...(run.delTokens || [])],
        insTokens: [...(run.insTokens || [])],
        moveId: run.moveId,
        moveRole: run.moveRole
      };
    }
    return {
      type: run.type,
      tokens: [...(run.tokens || [])],
      moveId: run.moveId,
      moveRole: run.moveRole
    };
  };

  const pushMergedRun = (target, run) => {
    const next = cloneRun(run);
    if (!next) return;

    if (next.type === 'replace') {
      if (!next.delTokens.length && !next.insTokens.length) return;
    } else if (!next.tokens.length) {
      return;
    }

    const last = target[target.length - 1];
    const canMerge = Boolean(last) &&
      last.type === next.type &&
      !last.moveId &&
      !next.moveId;

    if (!canMerge) {
      target.push(next);
      return;
    }

    if (next.type === 'replace') {
      last.delTokens.push(...next.delTokens);
      last.insTokens.push(...next.insTokens);
      return;
    }
    last.tokens.push(...next.tokens);
  };

  const appendRuns = (target, runs) => {
    (runs || []).forEach((run) => pushMergedRun(target, run));
  };

  const makeSingleRun = (type, text) => {
    const tokens = tokenize(text);
    if (!tokens.length) return [];
    return [{ type, tokens }];
  };

  const makeReplaceRun = (delText, insText) => {
    const delTokens = tokenize(delText);
    const insTokens = tokenize(insText);
    if (!delTokens.length && !insTokens.length) return [];
    return [{ type: 'replace', delTokens, insTokens }];
  };

  const createWarningBag = () => new Set();

  const pushWarning = (warnings, message) => {
    if (!warnings || !message) return;
    warnings.add(String(message));
  };

  const splitParagraphBlocks = (text) => {
    const source = String(text || '');
    if (!source) return [];
    const parts = source.split(/(\n{2,})/);
    const blocks = [];
    for (let i = 0; i < parts.length; i += 2) {
      const body = parts[i] || '';
      const separator = parts[i + 1] || '';
      const block = `${body}${separator}`;
      if (block) blocks.push(block);
    }
    return blocks.length ? blocks : [source];
  };

  const splitSentenceBlocksFallback = (text) => {
    const source = String(text || '');
    if (!source) return [];
    const chunks = source.match(/[^.!?\n]+(?:[.!?]+)?(?:\s+|$)|\n+/g);
    if (!chunks || !chunks.length) return [source];
    return chunks.filter(Boolean);
  };

  const splitSentenceBlocks = (text) => {
    const source = String(text || '');
    if (!source) return [];
    if (!sentenceSegmenter) return splitSentenceBlocksFallback(source);
    try {
      const segments = Array.from(sentenceSegmenter.segment(source), (part) => part?.segment || '').filter(Boolean);
      return segments.length ? segments : splitSentenceBlocksFallback(source);
    } catch {
      return splitSentenceBlocksFallback(source);
    }
  };

  const splitStructuredLines = (text) => {
    const source = String(text || '');
    if (!source) return [];
    return source.match(/[^\n]*\n|[^\n]+$/g) || [source];
  };

  const stripTrailingNewline = (text) => String(text || '').replace(/\n$/, '');

  const isBlankLine = (line) => !stripTrailingNewline(line).trim();

  const isListStartLine = (line) => {
    const source = stripTrailingNewline(line);
    return UNORDERED_LIST_MARKER_RE.test(source) || ORDERED_LIST_MARKER_RE.test(source);
  };

  const isQuoteLine = (line) => QUOTE_LINE_RE.test(stripTrailingNewline(line));

  const isPseudoRowLine = (line) => {
    const source = stripTrailingNewline(line);
    const trimmed = source.trim();
    if (!trimmed) return false;
    const pipeCount = countMatches(source, /\|/g);
    const commaCount = countMatches(source, /,/g);
    return /\t/.test(source) || pipeCount >= 2 || commaCount >= 2;
  };

  const normalizeDocumentSpacing = (text) => String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');

  const stripListMarker = (text) => String(text || '')
    .replace(UNORDERED_LIST_MARKER_RE, '')
    .replace(ORDERED_LIST_MARKER_RE, '');

  const detectDocumentBlockKind = (text) => {
    const lines = splitStructuredLines(text);
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (isBlankLine(line)) continue;
      if (isListStartLine(line)) return 'list-item';
      if (isQuoteLine(line)) return 'quote';
      if (isPseudoRowLine(line)) return 'row';
      return 'paragraph';
    }
    return 'blank';
  };

  const normalizeDocumentBlockText = (text, kind) => {
    let source = normalizeDocumentSpacing(text);
    if (kind === 'list-item') {
      source = source.split('\n').map((line, index) => (
        index === 0 ? stripListMarker(line) : line
      )).join('\n');
    } else if (kind === 'quote') {
      source = source.replace(/^\s*>+\s?/gm, '');
    }
    return source
      .replace(/[ \t]+/g, ' ')
      .replace(/\s*\n\s*/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
  };

  const createDocumentSignature = (text) => {
    const kind = detectDocumentBlockKind(text);
    return normalizeDocumentBlockText(text, kind);
  };

  const buildDocumentUnits = (text) => (splitDocumentBlocks(text) || []).map((chunk) => {
    const value = String(chunk || '');
    const kind = detectDocumentBlockKind(value);
    const normalized = normalizeDocumentBlockText(value, kind);
    return {
      text: value,
      kind,
      normalized,
      signature: normalized,
      words: countWords(normalized || value),
      chars: normalized.length || value.trim().length
    };
  });

  const createSentenceSignature = (text) => normalizeDocumentSpacing(text)
    .replace(/[ \t]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();

  const buildSentenceUnits = (text) => (splitSentenceBlocks(text) || []).map((chunk) => {
    const value = String(chunk || '');
    const normalized = createSentenceSignature(value);
    return {
      text: value,
      kind: 'sentence',
      normalized,
      signature: normalized,
      words: countWords(normalized || value),
      chars: normalized.length || value.trim().length
    };
  });

  const splitDocumentBlocks = (text) => {
    const source = String(text || '');
    if (!source) return [];

    const lines = splitStructuredLines(source);
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];

      if (isBlankLine(line)) {
        let blank = '';
        while (index < lines.length && isBlankLine(lines[index])) {
          blank += lines[index];
          index += 1;
        }
        if (blank) blocks.push(blank);
        continue;
      }

      if (isListStartLine(line)) {
        let item = lines[index];
        index += 1;
        while (
          index < lines.length &&
          !isBlankLine(lines[index]) &&
          !isListStartLine(lines[index]) &&
          !isQuoteLine(lines[index]) &&
          !isPseudoRowLine(lines[index])
        ) {
          item += lines[index];
          index += 1;
        }
        blocks.push(item);
        continue;
      }

      if (isQuoteLine(line)) {
        let quote = '';
        while (index < lines.length && !isBlankLine(lines[index]) && isQuoteLine(lines[index])) {
          quote += lines[index];
          index += 1;
        }
        blocks.push(quote);
        continue;
      }

      if (isPseudoRowLine(line)) {
        blocks.push(lines[index]);
        index += 1;
        continue;
      }

      let paragraph = '';
      while (
        index < lines.length &&
        !isBlankLine(lines[index]) &&
        !isListStartLine(lines[index]) &&
        !isQuoteLine(lines[index]) &&
        !isPseudoRowLine(lines[index])
      ) {
        paragraph += lines[index];
        index += 1;
      }
      if (paragraph) blocks.push(paragraph);
    }

    return blocks.length ? blocks : [source];
  };

  const createAnchorSignature = (text) => String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const createStructuredSignature = (text) => String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');

  const buildTextUnits = (text, splitter, options) => {
    const createSignature = typeof options?.createSignature === 'function'
      ? options.createSignature
      : createAnchorSignature;
    return (splitter(text) || []).map((chunk) => {
      const value = String(chunk || '');
      return {
        text: value,
        signature: createSignature(value),
        words: countWords(value),
        chars: value.trim().length
      };
    });
  };

  const joinUnitRange = (units, from, to) => units.slice(from, to).map((unit) => unit.text).join('');

  const selectIncreasingPairs = (pairs, maxPairs) => {
    if (!pairs.length) return [];
    if (pairs.length > maxPairs) {
      const greedy = [];
      let lastB = -1;
      pairs.forEach((pair) => {
        if (pair.bIndex <= lastB) return;
        greedy.push(pair);
        lastB = pair.bIndex;
      });
      return greedy;
    }

    const count = pairs.length;
    const best = new Array(count).fill(0);
    const prev = new Array(count).fill(-1);

    for (let i = 0; i < count; i += 1) {
      best[i] = pairs[i].score;
      for (let j = 0; j < i; j += 1) {
        if (pairs[j].bIndex >= pairs[i].bIndex) continue;
        const candidate = best[j] + pairs[i].score;
        if (candidate > best[i]) {
          best[i] = candidate;
          prev[i] = j;
        }
      }
    }

    let bestIndex = 0;
    for (let i = 1; i < count; i += 1) {
      if (best[i] > best[bestIndex]) bestIndex = i;
    }

    const selected = [];
    let cursor = bestIndex;
    while (cursor !== -1) {
      selected.push(pairs[cursor]);
      cursor = prev[cursor];
    }
    selected.reverse();
    return selected;
  };

  const buildOccurrenceIndex = (units, options) => {
    const map = new Map();
    const minWords = Math.max(0, options?.minWords || 0);
    const minChars = Math.max(1, options?.minChars || 1);
    units.forEach((unit, index) => {
      if (!unit.signature) return;
      if (unit.words < minWords && unit.chars < minChars) return;
      const existing = map.get(unit.signature) || [];
      existing.push(index);
      map.set(unit.signature, existing);
    });
    return map;
  };

  const findUnitAnchors = (aUnits, bUnits, options) => {
    if (!aUnits.length || !bUnits.length) return [];
    const maxPairs = Math.max(80, options?.maxPairs || 420);
    const maxOccurrencePerSide = Math.max(1, options?.maxOccurrencePerSide || 3);
    const maxFrequency = Math.max(2, options?.maxFrequency || 6);

    const aIndex = buildOccurrenceIndex(aUnits, options);
    const bIndex = buildOccurrenceIndex(bUnits, options);
    if (!aIndex.size || !bIndex.size) return [];

    const candidates = [];
    aIndex.forEach((aIndexes, signature) => {
      const bIndexes = bIndex.get(signature);
      if (!bIndexes || !bIndexes.length) return;
      if (aIndexes.length > maxOccurrencePerSide || bIndexes.length > maxOccurrencePerSide) return;
      const frequency = aIndexes.length + bIndexes.length;
      if (frequency > maxFrequency) return;
      const rarityWeight = Math.max(1, (maxFrequency + 2) - frequency);

      aIndexes.forEach((aIndexValue) => {
        bIndexes.forEach((bIndexValue) => {
          const unit = aUnits[aIndexValue];
          const sizeWeight = Math.max(1, (unit.words * 2) + Math.floor(unit.chars / 24));
          const score = (rarityWeight * rarityWeight * 3) + sizeWeight;
          candidates.push({ aIndex: aIndexValue, bIndex: bIndexValue, score });
        });
      });
    });

    if (!candidates.length) return [];

    let usable = candidates;
    if (usable.length > MAX_ANCHOR_CANDIDATES) {
      usable = [...usable]
        .sort((left, right) => right.score - left.score)
        .slice(0, MAX_ANCHOR_CANDIDATES);
    }

    usable.sort((left, right) => {
      if (left.aIndex !== right.aIndex) return left.aIndex - right.aIndex;
      return left.bIndex - right.bIndex;
    });

    return selectIncreasingPairs(usable, maxPairs);
  };

  const buildSignatureCounts = (units) => {
    const counts = new Map();
    (units || []).forEach((unit) => {
      if (!unit?.signature) return;
      counts.set(unit.signature, (counts.get(unit.signature) || 0) + 1);
    });
    return counts;
  };

  const areDocumentKindsCompatible = (leftKind, rightKind) => {
    if (leftKind === rightKind) return true;
    if (leftKind === 'blank' || rightKind === 'blank') return false;
    if (leftKind === 'paragraph' && rightKind === 'quote') return true;
    if (leftKind === 'quote' && rightKind === 'paragraph') return true;
    return false;
  };

  const scoreSentenceUnitPair = (leftUnit, rightUnit) => {
    if (!leftUnit?.normalized || !rightUnit?.normalized) return 0;
    const exact = leftUnit.signature && leftUnit.signature === rightUnit.signature;
    if (exact) {
      return 250 + Math.min(28, (Math.min(leftUnit.words, rightUnit.words) * 4));
    }

    const similarity = textSimilarityScore(leftUnit.normalized, rightUnit.normalized);
    const maxWords = Math.max(leftUnit.words, rightUnit.words, 1);
    const wordRatio = Math.min(leftUnit.words, rightUnit.words) / maxWords;
    const minScore = maxWords >= 16 || Math.max(leftUnit.chars, rightUnit.chars) >= 120 ? 0.42 : 0.52;
    if (similarity < minScore || wordRatio < 0.34) return 0;

    let score = similarity * 100;
    if (wordRatio >= 0.7) score += 10;
    score += Math.min(24, Math.floor(Math.min(leftUnit.words, rightUnit.words) * 2));
    return score;
  };

  const scoreDocumentUnitPair = (leftUnit, rightUnit, context) => {
    if (!leftUnit?.normalized || !rightUnit?.normalized) return 0;
    if (!areDocumentKindsCompatible(leftUnit.kind, rightUnit.kind)) return 0;

    const exact = leftUnit.signature && leftUnit.signature === rightUnit.signature;
    const similarity = exact ? 1 : textSimilarityScore(leftUnit.normalized, rightUnit.normalized);
    const maxWords = Math.max(leftUnit.words, rightUnit.words, 1);
    const minWords = Math.min(leftUnit.words, rightUnit.words);
    const wordRatio = minWords / maxWords;
    const maxChars = Math.max(leftUnit.chars, rightUnit.chars, 1);
    const charRatio = Math.min(leftUnit.chars, rightUnit.chars) / maxChars;

    let minScore = 0.54;
    if (leftUnit.kind === 'row' || rightUnit.kind === 'row') {
      minScore = 0.74;
    } else if (leftUnit.kind === 'list-item' && rightUnit.kind === 'list-item') {
      minScore = 0.48;
    } else if (maxWords >= 18 || maxChars >= 160) {
      minScore = 0.4;
    }
    if (!exact && (similarity < minScore || wordRatio < 0.26 || charRatio < 0.22)) return 0;

    let score = similarity * 100;
    if (exact) {
      const frequency = (context?.aCounts?.get(leftUnit.signature) || 0) + (context?.bCounts?.get(rightUnit.signature) || 0);
      score += 220 + Math.max(0, 32 - (frequency * 7));
    }
    if (leftUnit.kind === rightUnit.kind) score += 12;
    if (wordRatio >= 0.75) score += 10;
    else if (wordRatio >= 0.55) score += 4;
    if (charRatio >= 0.75) score += 8;
    score += Math.min(34, Math.floor((minWords * 2) + (Math.min(leftUnit.chars, rightUnit.chars) / 90)));
    return score;
  };

  const findSimilarUnitPairs = (aUnits, bUnits, options) => {
    if (!aUnits.length || !bUnits.length) return [];

    const maxPairs = Math.max(24, options?.maxPairs || 180);
    const maxCandidates = Math.max(300, options?.maxCandidates || MAX_SIMILARITY_CANDIDATES);
    const scorer = options?.scorer;
    if (typeof scorer !== 'function') return [];

    const aCounts = buildSignatureCounts(aUnits);
    const bCounts = buildSignatureCounts(bUnits);
    const candidates = [];

    for (let aIndex = 0; aIndex < aUnits.length; aIndex += 1) {
      const leftUnit = aUnits[aIndex];
      if (!leftUnit?.normalized) continue;
      for (let bIndex = 0; bIndex < bUnits.length; bIndex += 1) {
        const rightUnit = bUnits[bIndex];
        if (!rightUnit?.normalized) continue;
        const score = scorer(leftUnit, rightUnit, { aCounts, bCounts });
        if (!score) continue;
        candidates.push({ aIndex, bIndex, score });
      }
    }

    if (!candidates.length) return [];

    let usable = candidates;
    if (usable.length > maxCandidates) {
      usable = [...usable]
        .sort((left, right) => right.score - left.score)
        .slice(0, maxCandidates);
    }

    usable.sort((left, right) => {
      if (left.aIndex !== right.aIndex) return left.aIndex - right.aIndex;
      return left.bIndex - right.bIndex;
    });

    return selectIncreasingPairs(usable, maxPairs);
  };

  const buildRunsFromUnitPairs = (aUnits, bUnits, pairs, options) => {
    const handleGap = options?.handleGap || (() => []);
    const handlePair = options?.handlePair || (() => []);
    const runs = [];
    let aCursor = 0;
    let bCursor = 0;

    (pairs || []).forEach((pair) => {
      appendRuns(runs, handleGap(
        joinUnitRange(aUnits, aCursor, pair.aIndex),
        joinUnitRange(bUnits, bCursor, pair.bIndex)
      ));
      appendRuns(runs, handlePair(aUnits[pair.aIndex], bUnits[pair.bIndex]));
      aCursor = pair.aIndex + 1;
      bCursor = pair.bIndex + 1;
    });

    appendRuns(runs, handleGap(
      joinUnitRange(aUnits, aCursor, aUnits.length),
      joinUnitRange(bUnits, bCursor, bUnits.length)
    ));
    return runs;
  };

  const diffWithAnchors = (aUnits, bUnits, options) => {
    const fallback = options?.fallback || (() => []);
    const anchors = findUnitAnchors(aUnits, bUnits, options);
    if (!anchors.length) {
      return fallback(joinUnitRange(aUnits, 0, aUnits.length), joinUnitRange(bUnits, 0, bUnits.length));
    }

    const runs = [];
    let aCursor = 0;
    let bCursor = 0;

    anchors.forEach((anchor) => {
      const leftGap = joinUnitRange(aUnits, aCursor, anchor.aIndex);
      const rightGap = joinUnitRange(bUnits, bCursor, anchor.bIndex);
      appendRuns(runs, fallback(leftGap, rightGap));

      const aAnchorText = aUnits[anchor.aIndex]?.text || '';
      const bAnchorText = bUnits[anchor.bIndex]?.text || '';
      if (aAnchorText === bAnchorText) {
        appendRuns(runs, makeSingleRun('equal', aAnchorText));
      } else {
        appendRuns(runs, fallback(aAnchorText, bAnchorText));
      }

      aCursor = anchor.aIndex + 1;
      bCursor = anchor.bIndex + 1;
    });

    const leftTail = joinUnitRange(aUnits, aCursor, aUnits.length);
    const rightTail = joinUnitRange(bUnits, bCursor, bUnits.length);
    appendRuns(runs, fallback(leftTail, rightTail));
    return runs;
  };

  const buildTokenChunkUnits = (text, options) => {
    const source = String(text || '');
    if (!source) return [];
    const tokens = tokenize(source);
    if (!tokens.length) return [];

    const targetWords = Math.max(4, options?.targetWords || 24);
    const targetChars = Math.max(80, options?.targetChars || 220);
    const maxTokens = Math.max(16, options?.maxTokens || 64);
    const chunks = [];
    let current = [];
    let wordCount = 0;
    let charCount = 0;

    const flush = () => {
      if (!current.length) return;
      const chunkText = current.join('');
      chunks.push({
        text: chunkText,
        signature: createAnchorSignature(chunkText),
        words: countWordTokens(current),
        chars: chunkText.trim().length
      });
      current = [];
      wordCount = 0;
      charCount = 0;
    };

    tokens.forEach((token, index) => {
      current.push(token);
      charCount += token.length;
      if (isWordToken(token)) wordCount += 1;

      const isBoundary = /\n/.test(token) || /[.!?;:]/.test(token);
      const atEnd = index === tokens.length - 1;
      const largeEnough = wordCount >= targetWords || charCount >= targetChars || current.length >= maxTokens;
      if ((largeEnough && isBoundary) || charCount >= targetChars * 1.4 || current.length >= maxTokens || atEnd) {
        flush();
      }
    });

    return chunks;
  };

  const buildRunsFromGroupedArrayEdits = (groupedRuns, refinePair) => {
    const out = [];
    for (let i = 0; i < groupedRuns.length; i += 1) {
      const run = groupedRuns[i];
      const next = groupedRuns[i + 1];
      if (run.type === 'equal') {
        appendRuns(out, makeSingleRun('equal', run.tokens.join('')));
        continue;
      }
      if (next && ((run.type === 'delete' && next.type === 'insert') || (run.type === 'insert' && next.type === 'delete'))) {
        const delText = run.type === 'delete' ? run.tokens.join('') : next.tokens.join('');
        const insText = run.type === 'insert' ? run.tokens.join('') : next.tokens.join('');
        appendRuns(out, refinePair(delText, insText));
        i += 1;
        continue;
      }
      appendRuns(out, makeSingleRun(run.type, run.tokens.join('')));
    }
    return out;
  };

  const chunkDiffRuns = (leftText, rightText, warnings, options) => {
    const depth = options?.depth || 0;
    const structured = Boolean(options?.structured);
    const leftUnits = buildTokenChunkUnits(leftText, structured
      ? { targetWords: 12, targetChars: 140, maxTokens: 36 }
      : { targetWords: 24, targetChars: 240, maxTokens: 72 });
    const rightUnits = buildTokenChunkUnits(rightText, structured
      ? { targetWords: 12, targetChars: 140, maxTokens: 36 }
      : { targetWords: 24, targetChars: 240, maxTokens: 72 });

    if (!leftUnits.length && !rightUnits.length) return [];
    if (!leftUnits.length) return makeSingleRun('insert', rightText);
    if (!rightUnits.length) return makeSingleRun('delete', leftText);

    const edits = myersEdits(
      leftUnits.map((unit) => unit.signature || unit.text),
      rightUnits.map((unit) => unit.signature || unit.text),
      { maxD: Math.max(leftUnits.length, rightUnits.length) * 4 }
    );

    if (!edits) {
      pushWarning(warnings, 'Large sections were compared with coarse segmentation to stay responsive.');
      return makeReplaceRun(leftText, rightText);
    }

    const leftTexts = leftUnits.map((unit) => unit.text);
    const rightTexts = rightUnits.map((unit) => unit.text);
    let leftCursor = 0;
    let rightCursor = 0;
    const mappedEdits = edits.map((edit) => {
      if (edit.type === 'equal') {
        const value = leftTexts[leftCursor];
        leftCursor += 1;
        rightCursor += 1;
        return { type: 'equal', value };
      }
      if (edit.type === 'delete') {
        const value = leftTexts[leftCursor];
        leftCursor += 1;
        return { type: 'delete', value };
      }
      const value = rightTexts[rightCursor];
      rightCursor += 1;
      return { type: 'insert', value };
    });

    return buildRunsFromGroupedArrayEdits(groupRuns(mappedEdits), (delText, insText) => {
      const totalTokens = tokenize(delText).length + tokenize(insText).length;
      if (!totalTokens) return [];
      if (depth >= MAX_TOKEN_DIFF_DEPTH || totalTokens > 4_000) {
        pushWarning(warnings, 'Large sections were compared with coarse segmentation to stay responsive.');
        return makeReplaceRun(delText, insText);
      }
      return tokenDiffRuns(delText, insText, warnings, {
        depth: depth + 1,
        structured
      });
    });
  };

  const tokenDiffRuns = (leftText, rightText, warnings, options) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    const depth = options?.depth || 0;
    const structured = Boolean(options?.structured);
    if (!left && !right) return [];
    if (!left) return makeSingleRun('insert', right);
    if (!right) return makeSingleRun('delete', left);

    const aTokens = tokenize(left);
    const bTokens = tokenize(right);
    const edits = myersEdits(aTokens, bTokens);
    if (edits) return normalizeRuns(groupRuns(edits));

    if (depth < MAX_TOKEN_DIFF_DEPTH) {
      pushWarning(warnings, 'Large sections were compared with coarse segmentation to stay responsive.');
      return chunkDiffRuns(left, right, warnings, {
        depth: depth + 1,
        structured
      });
    }
    return makeReplaceRun(left, right);
  };

  const structuredPairShouldRefine = (delText, insText) => {
    const totalChars = String(delText || '').length + String(insText || '').length;
    const totalLines = splitStructuredLines(delText).length + splitStructuredLines(insText).length;
    if (totalChars <= 6_000 && totalLines <= 32) return true;
    return textSimilarityScore(delText, insText) >= 0.45;
  };

  const lineDiffRuns = (leftText, rightText, warnings, options) => {
    const leftLines = splitStructuredLines(leftText);
    const rightLines = splitStructuredLines(rightText);
    if (!leftLines.length && !rightLines.length) return [];
    if (!leftLines.length) return makeSingleRun('insert', rightText);
    if (!rightLines.length) return makeSingleRun('delete', leftText);

    const edits = myersEdits(leftLines, rightLines, {
      maxD: Math.max(leftLines.length, rightLines.length) * 4
    });

    if (!edits) {
      return chunkDiffRuns(leftText, rightText, warnings, {
        depth: options?.depth || 0,
        structured: true
      });
    }

    return buildRunsFromGroupedArrayEdits(groupRuns(edits), (delText, insText) => {
      if (!structuredPairShouldRefine(delText, insText)) {
        return makeReplaceRun(delText, insText);
      }
      return tokenDiffRuns(delText, insText, warnings, {
        depth: (options?.depth || 0) + 1,
        structured: true
      });
    });
  };

  const diffStructuredGap = (leftText, rightText, warnings, options) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];
    if (!left) return makeSingleRun('insert', right);
    if (!right) return makeSingleRun('delete', left);
    if (left.includes('\n') || right.includes('\n')) {
      return lineDiffRuns(left, right, warnings, options);
    }
    return tokenDiffRuns(left, right, warnings, {
      depth: options?.depth || 0,
      structured: true
    });
  };

  const diffStructured = (leftText, rightText, warnings) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];

    const aUnits = buildTextUnits(left, splitStructuredLines, {
      createSignature: createStructuredSignature
    });
    const bUnits = buildTextUnits(right, splitStructuredLines, {
      createSignature: createStructuredSignature
    });

    if (aUnits.length < 3 && bUnits.length < 3) {
      return diffStructuredGap(left, right, warnings, { depth: 0 });
    }

    return normalizeRuns(diffWithAnchors(aUnits, bUnits, {
      minWords: 0,
      minChars: 1,
      maxPairs: 900,
      maxOccurrencePerSide: 3,
      maxFrequency: 6,
      fallback: (leftGap, rightGap) => diffStructuredGap(leftGap, rightGap, warnings, { depth: 0 })
    }));
  };

  const diffBySentenceAnchors = (leftText, rightText, warnings) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];

    const aUnits = buildTextUnits(left, splitSentenceBlocks);
    const bUnits = buildTextUnits(right, splitSentenceBlocks);
    if (aUnits.length < 3 && bUnits.length < 3) {
      return tokenDiffRuns(left, right, warnings, { depth: 0, structured: false });
    }

    return diffWithAnchors(aUnits, bUnits, {
      minWords: 2,
      minChars: 22,
      maxPairs: 520,
      maxOccurrencePerSide: 3,
      maxFrequency: 6,
      fallback: (leftGap, rightGap) => tokenDiffRuns(leftGap, rightGap, warnings, {
        depth: 0,
        structured: false
      })
    });
  };

  const diffByParagraphAnchors = (leftText, rightText, warnings) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];

    const aUnits = buildTextUnits(left, splitParagraphBlocks);
    const bUnits = buildTextUnits(right, splitParagraphBlocks);
    if (aUnits.length < 2 && bUnits.length < 2) {
      return diffBySentenceAnchors(left, right, warnings);
    }

    return diffWithAnchors(aUnits, bUnits, {
      minWords: 4,
      minChars: 36,
      maxPairs: 260,
      maxOccurrencePerSide: 4,
      maxFrequency: 8,
      fallback: (leftGap, rightGap) => diffBySentenceAnchors(leftGap, rightGap, warnings)
    });
  };

  const diffSentenceGap = (leftText, rightText, warnings, options) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    const depth = options?.depth || 0;
    if (!left && !right) return [];
    if (!left) return makeSingleRun('insert', right);
    if (!right) return makeSingleRun('delete', left);
    if (depth >= MAX_DOCUMENT_DIFF_DEPTH) {
      return tokenDiffRuns(left, right, warnings, { depth: 0, structured: false });
    }

    const aUnits = buildSentenceUnits(left);
    const bUnits = buildSentenceUnits(right);
    if (aUnits.length <= 1 && bUnits.length <= 1) {
      return tokenDiffRuns(left, right, warnings, { depth: 0, structured: false });
    }

    const anchors = findUnitAnchors(aUnits, bUnits, {
      minWords: 2,
      minChars: 18,
      maxPairs: 220,
      maxOccurrencePerSide: 3,
      maxFrequency: 6
    });
    if (anchors.length) {
      return normalizeRuns(buildRunsFromUnitPairs(aUnits, bUnits, anchors, {
        handleGap: (leftGap, rightGap) => diffSentenceGap(leftGap, rightGap, warnings, { depth: depth + 1 }),
        handlePair: (leftUnit, rightUnit) => {
          if ((leftUnit?.text || '') === (rightUnit?.text || '')) {
            return makeSingleRun('equal', leftUnit.text);
          }
          return tokenDiffRuns(leftUnit?.text || '', rightUnit?.text || '', warnings, { depth: 0, structured: false });
        }
      }));
    }

    const pairs = findSimilarUnitPairs(aUnits, bUnits, {
      maxPairs: 140,
      maxCandidates: 2_400,
      scorer: scoreSentenceUnitPair
    });
    if (pairs.length) {
      return normalizeRuns(buildRunsFromUnitPairs(aUnits, bUnits, pairs, {
        handleGap: (leftGap, rightGap) => tokenDiffRuns(leftGap, rightGap, warnings, { depth: 0, structured: false }),
        handlePair: (leftUnit, rightUnit) => {
          if ((leftUnit?.text || '') === (rightUnit?.text || '')) {
            return makeSingleRun('equal', leftUnit.text);
          }
          return tokenDiffRuns(leftUnit?.text || '', rightUnit?.text || '', warnings, { depth: 0, structured: false });
        }
      }));
    }

    return diffBySentenceAnchors(left, right, warnings);
  };

  const diffDocumentPair = (leftUnit, rightUnit, warnings, options) => {
    const leftText = typeof leftUnit === 'string' ? leftUnit : (leftUnit?.text || '');
    const rightText = typeof rightUnit === 'string' ? rightUnit : (rightUnit?.text || '');
    const leftKind = typeof leftUnit === 'string' ? detectDocumentBlockKind(leftUnit) : (leftUnit?.kind || detectDocumentBlockKind(leftText));
    const rightKind = typeof rightUnit === 'string' ? detectDocumentBlockKind(rightUnit) : (rightUnit?.kind || detectDocumentBlockKind(rightText));
    const depth = options?.depth || 0;

    if (!leftText && !rightText) return [];
    if (!leftText) return makeSingleRun('insert', rightText);
    if (!rightText) return makeSingleRun('delete', leftText);
    if (leftText === rightText) return makeSingleRun('equal', leftText);

    if (leftKind === 'row' || rightKind === 'row') {
      return lineDiffRuns(leftText, rightText, warnings, { depth: 0, structured: true });
    }

    if (depth >= MAX_DOCUMENT_DIFF_DEPTH) {
      return tokenDiffRuns(leftText, rightText, warnings, { depth: 0, structured: false });
    }

    const leftSentences = splitSentenceBlocks(leftText);
    const rightSentences = splitSentenceBlocks(rightText);
    if (leftSentences.length > 1 || rightSentences.length > 1) {
      return diffSentenceGap(leftText, rightText, warnings, { depth: depth + 1 });
    }

    return tokenDiffRuns(leftText, rightText, warnings, { depth: 0, structured: false });
  };

  const diffDocumentGap = (leftText, rightText, warnings, options) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    const depth = options?.depth || 0;
    if (!left && !right) return [];
    if (!left) return makeSingleRun('insert', right);
    if (!right) return makeSingleRun('delete', left);

    const aUnits = buildDocumentUnits(left);
    const bUnits = buildDocumentUnits(right);
    if (!aUnits.length && !bUnits.length) return [];
    if (!aUnits.length) return makeSingleRun('insert', right);
    if (!bUnits.length) return makeSingleRun('delete', left);

    if (depth >= MAX_DOCUMENT_DIFF_DEPTH) {
      pushWarning(warnings, 'Large sections were compared with coarse segmentation to stay responsive.');
      return diffBySentenceAnchors(left, right, warnings);
    }

    if (aUnits.length === 1 && bUnits.length === 1) {
      return diffDocumentPair(aUnits[0], bUnits[0], warnings, { depth: depth + 1 });
    }

    const anchors = findUnitAnchors(aUnits, bUnits, {
      minWords: 0,
      minChars: 1,
      maxPairs: 260,
      maxOccurrencePerSide: 4,
      maxFrequency: 8
    });
    if (anchors.length) {
      return normalizeRuns(buildRunsFromUnitPairs(aUnits, bUnits, anchors, {
        handleGap: (leftGap, rightGap) => diffDocumentGap(leftGap, rightGap, warnings, { depth: depth + 1 }),
        handlePair: (leftDocUnit, rightDocUnit) => {
          if ((leftDocUnit?.text || '') === (rightDocUnit?.text || '')) {
            return makeSingleRun('equal', leftDocUnit.text);
          }
          return diffDocumentPair(leftDocUnit, rightDocUnit, warnings, { depth: depth + 1 });
        }
      }));
    }

    const pairs = findSimilarUnitPairs(aUnits, bUnits, {
      maxPairs: 180,
      maxCandidates: 3_200,
      scorer: scoreDocumentUnitPair
    });
    if (pairs.length) {
      return normalizeRuns(buildRunsFromUnitPairs(aUnits, bUnits, pairs, {
        handleGap: (leftGap, rightGap) => diffDocumentGap(leftGap, rightGap, warnings, { depth: depth + 1 }),
        handlePair: (leftDocUnit, rightDocUnit) => diffDocumentPair(leftDocUnit, rightDocUnit, warnings, { depth: depth + 1 })
      }));
    }

    if (aUnits.length + bUnits.length >= 14) {
      pushWarning(warnings, 'Large sections were compared with coarse segmentation to stay responsive.');
      return diffBySentenceAnchors(left, right, warnings);
    }

    return tokenDiffRuns(left, right, warnings, { depth: 0, structured: false });
  };

  const diffDocument = (leftText, rightText, warnings) => diffDocumentGap(leftText, rightText, warnings, { depth: 0 });

  const normalizeMoveSignature = (text) => String(text || '')
    .replace(/\s+/g, ' ')
    .trim();

  const markMovedRuns = (runs) => {
    const moveCandidates = { insert: [], delete: [] };
    const usedInsertIndexes = new Set();
    const scoredPairs = [];

    const collectCandidate = (run, index) => {
      if (!run || (run.type !== 'insert' && run.type !== 'delete')) return;
      const rawText = run.tokens.join('');
      const text = normalizeMoveSignature(rawText);
      if (!text) return;
      const words = countWords(text);
      if (words < 4 && text.length < 25) return;
      moveCandidates[run.type].push({
        index,
        text,
        words,
        chars: text.length
      });
    };

    runs.forEach(collectCandidate);

    moveCandidates.delete.forEach((leftCandidate) => {
      moveCandidates.insert.forEach((rightCandidate) => {
        if (Math.abs(rightCandidate.index - leftCandidate.index) <= 1) return;

        let score = 0;
        if (leftCandidate.text === rightCandidate.text) {
          score = 1.35;
        } else {
          const similarity = textSimilarityScore(leftCandidate.text, rightCandidate.text);
          const maxWords = Math.max(leftCandidate.words, rightCandidate.words, 1);
          const wordRatio = Math.min(leftCandidate.words, rightCandidate.words) / maxWords;
          if ((maxWords >= 18 || Math.max(leftCandidate.chars, rightCandidate.chars) >= 160) && similarity >= 0.46 && wordRatio >= 0.32) {
            score = similarity + 0.24;
          } else if (similarity >= 0.62 && wordRatio >= 0.42) {
            score = similarity;
          }
        }

        if (!score) return;
        scoredPairs.push({
          deleteIndex: leftCandidate.index,
          insertIndex: rightCandidate.index,
          score
        });
      });
    });

    scoredPairs.sort((left, right) => right.score - left.score);

    let moveCount = 0;
    let moveId = 1;
    const usedDeleteIndexes = new Set();

    scoredPairs.forEach((pair) => {
      if (usedDeleteIndexes.has(pair.deleteIndex) || usedInsertIndexes.has(pair.insertIndex)) return;
      const deleteRun = runs[pair.deleteIndex];
      const insertRun = runs[pair.insertIndex];
      if (!deleteRun || !insertRun || deleteRun.type !== 'delete' || insertRun.type !== 'insert') return;

      usedDeleteIndexes.add(pair.deleteIndex);
      usedInsertIndexes.add(pair.insertIndex);
      deleteRun.moveId = moveId;
      deleteRun.moveRole = 'from';
      insertRun.moveId = moveId;
      insertRun.moveRole = 'to';
      moveCount += 1;
      moveId += 1;
    });

    return moveCount;
  };

  const scoreStructuredHint = (kind) => {
    const normalized = normalizeSourceKind(kind);
    if (normalized === 'csv' || normalized === 'tsv') return 2.5;
    if (normalized === 'json') return 2.2;
    if (normalized === 'xml') return 2.0;
    if (normalized === 'html') return 0.6;
    return 0;
  };

  const scoreDocumentHint = (kind) => {
    const normalized = normalizeSourceKind(kind);
    if (DOCUMENT_SOURCE_KINDS.has(normalized)) return 1.2;
    if (normalized === 'html') return 0.2;
    return 0;
  };

  const analyzeTextShape = (text) => {
    const source = String(text || '').trim();
    if (!source) {
      return { structuredScore: 0, documentScore: 0 };
    }

    const lineBlocks = splitStructuredLines(source)
      .map((line) => line.replace(/\n$/, ''))
      .filter((line) => line.trim());

    if (!lineBlocks.length) {
      return { structuredScore: 0, documentScore: 0 };
    }

    const lineCount = lineBlocks.length;
    const blankBlocks = (source.match(/\n\s*\n/g) || []).length;
    const averageWords = lineBlocks.reduce((sum, line) => sum + countWords(line), 0) / lineCount;
    const averageChars = lineBlocks.reduce((sum, line) => sum + line.trim().length, 0) / lineCount;
    const delimiterHeavy = lineBlocks.filter((line) => /[\t,:;{}[\]<>|]/.test(line)).length;
    const jsonXmlLike = lineBlocks.filter((line) => /":\s|<[^>]+>|^\s*[\[{]/.test(line)).length;

    const delimiterPatterns = new Map();
    let delimitedRows = 0;
    lineBlocks.forEach((line) => {
      const tabCount = (line.match(/\t/g) || []).length;
      const commaCount = (line.match(/,/g) || []).length;
      const pipeCount = (line.match(/\|/g) || []).length;
      const delimiterCount = Math.max(tabCount, commaCount, pipeCount);
      if (delimiterCount >= 1) delimitedRows += 1;
      if (tabCount >= 1) {
        const key = `t${tabCount}`;
        delimiterPatterns.set(key, (delimiterPatterns.get(key) || 0) + 1);
      } else if (commaCount >= 2) {
        const key = `c${commaCount}`;
        delimiterPatterns.set(key, (delimiterPatterns.get(key) || 0) + 1);
      } else if (pipeCount >= 2) {
        const key = `p${pipeCount}`;
        delimiterPatterns.set(key, (delimiterPatterns.get(key) || 0) + 1);
      }
    });
    const repeatedDelimiterPattern = Math.max(0, ...delimiterPatterns.values());

    let structuredScore = 0;
    let documentScore = 0;

    if (/^\s*[\[{<]/.test(source)) structuredScore += 1.1;
    if (lineCount >= 3 && delimiterHeavy / lineCount >= 0.5) structuredScore += 1.0;
    if (lineCount >= 3 && jsonXmlLike / lineCount >= 0.4) structuredScore += 0.8;
    if (lineCount >= 4 && delimitedRows / lineCount >= 0.6) structuredScore += 1.1;
    if (repeatedDelimiterPattern >= Math.max(2, Math.floor(lineCount * 0.45))) structuredScore += 1.3;
    if (lineCount >= 4 && averageWords <= 10 && averageChars <= 90) structuredScore += 0.7;
    if (blankBlocks === 0 && lineCount >= 5 && averageWords <= 14) structuredScore += 0.4;

    if (blankBlocks >= 1) documentScore += 0.8;
    if (averageWords >= 12) documentScore += 0.8;
    if (countWords(source) >= 80 && averageWords >= 10) documentScore += 0.6;
    if (/[.!?][)"'\]]*\s+[A-Z]/.test(source)) documentScore += 0.5;
    if (blankBlocks >= 2) documentScore += 0.4;

    return { structuredScore, documentScore };
  };

  const detectComparisonMode = (options) => {
    const modeOverride = normalizeMode(options?.modeOverride);
    if (modeOverride !== MODES.AUTO) return modeOverride;

    const sourceHints = options?.sourceHints || {};
    const leftKind = normalizeSourceKind(sourceHints.leftKind || sourceHints.originalKind || sourceHints.left);
    const rightKind = normalizeSourceKind(sourceHints.rightKind || sourceHints.revisedKind || sourceHints.right);

    let structuredScore = scoreStructuredHint(leftKind) + scoreStructuredHint(rightKind);
    let documentScore = scoreDocumentHint(leftKind) + scoreDocumentHint(rightKind);

    const leftShape = analyzeTextShape(options?.leftText);
    const rightShape = analyzeTextShape(options?.rightText);
    structuredScore += leftShape.structuredScore + rightShape.structuredScore;
    documentScore += leftShape.documentScore + rightShape.documentScore;

    if (STRUCTURED_SOURCE_KINDS.has(leftKind) && STRUCTURED_SOURCE_KINDS.has(rightKind)) {
      structuredScore += 1.0;
    }

    return structuredScore > documentScore + 0.75 ? MODES.STRUCTURED : MODES.DOCUMENT;
  };

  const summarizeRuns = (runs, movedBlocks) => {
    let insertedWords = 0;
    let deletedWords = 0;
    let replacements = 0;

    (runs || []).forEach((run) => {
      if (run.type === 'insert') insertedWords += countWords(run.tokens.join(''));
      if (run.type === 'delete') deletedWords += countWords(run.tokens.join(''));
      if (run.type === 'replace') {
        replacements += 1;
        insertedWords += countWords(run.insTokens.join(''));
        deletedWords += countWords(run.delTokens.join(''));
      }
    });

    return {
      insertedWords,
      deletedWords,
      replacements,
      movedBlocks: movedBlocks || 0,
      hasChanges: Boolean(insertedWords || deletedWords)
    };
  };

  const diffChars = (leftText, rightText) => {
    const edits = myersEdits(Array.from(String(leftText || '')), Array.from(String(rightText || ''))) || [];
    const merged = [];
    let current = null;
    edits.forEach((edit) => {
      if (!current || current.type !== edit.type) {
        current = { type: edit.type, text: edit.value };
        merged.push(current);
        return;
      }
      current.text += edit.value;
    });
    return merged;
  };

  const compareDocument = (leftText, rightText, warnings) => {
    const aTokens = tokenize(leftText);
    const bTokens = tokenize(rightText);
    const settings = getAutoCompareSettings(aTokens, bTokens);
    const countsA = buildWordCounts(aTokens);
    const countsB = buildWordCounts(bTokens);
    let runs = normalizeRuns(diffDocument(leftText, rightText, warnings));
    runs = coalesceRuns(runs, countsA, countsB, settings);
    const movedBlocks = markMovedRuns(runs);
    return {
      runs,
      counts: summarizeRuns(runs, movedBlocks)
    };
  };

  const compareStructuredText = (leftText, rightText, warnings) => {
    const runs = diffStructured(leftText, rightText, warnings);
    return {
      runs,
      counts: summarizeRuns(runs, 0)
    };
  };

  const compareText = (options) => {
    const leftText = String(options?.leftText || '');
    const rightText = String(options?.rightText || '');
    const inferredMode = detectComparisonMode(options);
    const warnings = createWarningBag();

    const result = inferredMode === MODES.STRUCTURED
      ? compareStructuredText(leftText, rightText, warnings)
      : compareDocument(leftText, rightText, warnings);

    return {
      runs: result.runs,
      counts: result.counts,
      inferredMode,
      warnings: Array.from(warnings)
    };
  };

  return {
    MODES,
    compareText,
    detectComparisonMode,
    diffChars,
    tokenize
  };
}));
