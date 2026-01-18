(() => {
  'use strict';

  const $ = (s, c = document) => c.querySelector(s);

  const headerForm = $('.nav-search');
  const headerInput = $('.nav-search-input');
  const results = $('#search-results');
  const status = $('#search-status');

  if (!headerForm || !headerInput || !results || !status) return;

  const SITE_ORIGIN = 'https://danielshort.me';
  const INDEX_URL = 'dist/search-index.json';
  const MAX_RESULTS = 50;
  const CATEGORY_ORDER = ['Tools', 'Portfolio', 'Pages'];

  const escapeHtml = (value) => String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const escapeRegExp = (value) => String(value ?? '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const normalizeText = (value) => String(value ?? '')
    .toLowerCase()
    .replace(/[\u2019\u2018]/g, "'")
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

  const tokenize = (query) => {
    const normalized = normalizeText(query);
    if (!normalized) return [];
    return normalized.split(/\s+/g).filter(Boolean);
  };

  const highlight = (text, tokens) => {
    const raw = String(text ?? '');
    if (!raw || !tokens.length) return escapeHtml(raw);
    let out = escapeHtml(raw);
    tokens.forEach((token) => {
      if (!token) return;
      const re = new RegExp(`(${escapeRegExp(token)})`, 'ig');
      out = out.replace(re, '<mark class="search-highlight">$1</mark>');
    });
    return out;
  };

  const toAbsoluteUrl = (relativeOrAbsolute) => {
    const raw = String(relativeOrAbsolute || '').trim();
    if (!raw) return '';
    if (/^https?:\/\//i.test(raw)) return raw;
    if (raw.startsWith('/')) return `${SITE_ORIGIN}${raw}`;
    return `${SITE_ORIGIN}/${raw}`;
  };

  const toDisplayUrl = (relativeOrAbsolute) => {
    const abs = toAbsoluteUrl(relativeOrAbsolute);
    if (!abs) return '';
    return abs.replace(/^https?:\/\//i, '');
  };

  let indexPromise = null;
  const loadIndex = () => {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(INDEX_URL, { cache: 'force-cache' })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const pages = data && Array.isArray(data.pages) ? data.pages : [];
        return pages
          .filter((p) => p && p.url && p.title)
          .map((p) => {
            const keywords = Array.isArray(p.keywords) ? p.keywords : [];
            const haystack = normalizeText([
              p.title,
              p.description,
              p.url,
              ...keywords
            ].join(' '));
            return { ...p, keywords, haystack };
          });
      })
      .catch((err) => {
        status.textContent = `Search is unavailable right now. (${String(err && err.message ? err.message : err)})`;
        return [];
      });
    return indexPromise;
  };

  const scoreEntry = (entry, tokens) => {
    if (!entry || !tokens.length) return 0;
    const title = normalizeText(entry.title);
    const desc = normalizeText(entry.description);
    const url = normalizeText(entry.url);
    const keywords = normalizeText((entry.keywords || []).join(' '));

    let score = 0;
    for (const token of tokens) {
      const inTitle = title.includes(token);
      const inDesc = desc.includes(token);
      const inKeywords = keywords.includes(token);
      const inUrl = url.includes(token);
      if (!(inTitle || inDesc || inKeywords || inUrl)) return 0;
      if (inTitle) score += 6;
      else if (inKeywords) score += 5;
      else if (inDesc) score += 3;
      else if (inUrl) score += 1;
    }
    return score;
  };

  const slugify = (value) => String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .trim();

  const normalizeCategory = (value) => {
    const raw = String(value || '').trim();
    return raw || 'Pages';
  };

  const groupByCategory = (entries) => {
    const groups = new Map();
    entries.forEach((entry) => {
      const category = normalizeCategory(entry && entry.category);
      if (!groups.has(category)) groups.set(category, []);
      groups.get(category).push(entry);
    });

    const ordered = [
      ...CATEGORY_ORDER.filter((c) => groups.has(c)),
      ...[...groups.keys()].filter((c) => !CATEGORY_ORDER.includes(c)).sort((a, b) => a.localeCompare(b))
    ];

    return ordered.map((category) => ({ category, entries: groups.get(category) || [] }));
  };

  const renderEntry = (entry, tokens) => {
    const url = String(entry.url || '').trim();
    const title = String(entry.title || '').trim();
    const desc = String(entry.description || '').trim();

    const keywordHtml = (entry.keywords || []).length
      ? `<div class="search-keywords" aria-label="Keywords">${(entry.keywords || [])
          .slice(0, 10)
          .map((k) => `<span class="search-keyword">${escapeHtml(k)}</span>`)
          .join('')}</div>`
      : '';

    return `
      <article class="search-result">
        <div class="search-result-head">
          <a class="search-result-title" href="${escapeHtml(url)}">${highlight(title, tokens)}</a>
        </div>
        <div class="search-result-url">${escapeHtml(toDisplayUrl(url))}</div>
        ${desc ? `<p class="search-result-desc">${highlight(desc, tokens)}</p>` : ''}
        ${keywordHtml}
      </article>
    `;
  };

  const renderResults = (entries, query, tokens, totalMatches) => {
    if (!query) {
      status.textContent = 'Search using the bar in the header.';
      results.innerHTML = '';
      return;
    }

    if (!entries.length) {
      status.textContent = 'No results found.';
      results.innerHTML = `
        <div class="search-result" role="note">
          <div class="search-result-head">
            <span class="search-result-title">No matches</span>
          </div>
          <p class="search-result-desc">Try fewer words, or search for a tool/project name (like “UTM”, “Nonogram”, or “Oxford comma”).</p>
          <p class="search-result-desc">You can also browse the full <a class="search-result-title" href="sitemap">sitemap</a>.</p>
        </div>
      `;
      return;
    }

    const groups = groupByCategory(entries);
    const countByCategory = groups.map((g) => `${g.category}: ${g.entries.length}`).join(' · ');
    const shown = entries.length;
    const isTruncated = Number.isFinite(totalMatches) && totalMatches > shown;

    status.textContent = isTruncated
      ? `Showing ${shown} of ${totalMatches} matches for “${query}”. ${countByCategory}`
      : `${shown} result${shown === 1 ? '' : 's'} for “${query}”. ${countByCategory}`;

    results.innerHTML = groups
      .filter((g) => g.entries.length)
      .map((group) => {
        const id = `search-cat-${slugify(group.category) || 'other'}`;
        return `
          <section class="search-category" aria-labelledby="${escapeHtml(id)}">
            <div class="search-category-head">
              <h2 class="search-category-title" id="${escapeHtml(id)}">${escapeHtml(group.category)}</h2>
              <span class="search-category-count" aria-label="${escapeHtml(group.category)} results">${group.entries.length}</span>
            </div>
            <div class="search-category-list">
              ${group.entries.map((entry) => renderEntry(entry, tokens)).join('')}
            </div>
          </section>
        `;
      })
      .join('');
  };

  const setQueryInUrl = (query) => {
    const url = new URL(location.href);
    if (query) url.searchParams.set('q', query);
    else url.searchParams.delete('q');
    history.replaceState({}, '', url.pathname + url.search);
  };

  let updateTimer = null;
  const scheduleUpdate = (fn) => {
    if (updateTimer) window.clearTimeout(updateTimer);
    updateTimer = window.setTimeout(fn, 80);
  };

  const runSearch = async (query) => {
    const trimmed = String(query || '').trim();
    headerInput.value = trimmed;
    setQueryInUrl(trimmed);

    const tokens = tokenize(trimmed);
    if (!tokens.length) {
      renderResults([], '', tokens, 0);
      return;
    }

    status.textContent = 'Searching…';
    const pages = await loadIndex();

    const scored = pages
      .map((entry) => ({ entry, score: scoreEntry(entry, tokens) }))
      .filter((m) => m.score > 0)
      .sort((a, b) => b.score - a.score || String(a.entry.title).localeCompare(String(b.entry.title)));

    const totalMatches = scored.length;
    const matches = scored.slice(0, MAX_RESULTS).map((m) => m.entry);

    renderResults(matches, trimmed, tokens, totalMatches);
  };

  const initialQuery = new URLSearchParams(location.search).get('q') || '';
  headerInput.value = initialQuery;

  headerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runSearch(headerInput.value);
  });

  headerInput.addEventListener('input', () => {
    scheduleUpdate(() => runSearch(headerInput.value));
  });

  headerInput.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!headerInput.value) return;
    e.preventDefault();
    headerInput.value = '';
    runSearch('');
    try { headerInput.blur(); } catch (_) {}
  });

  // Initial run
  runSearch(headerInput.value);
})();
