(() => {
  'use strict';
  if (window.SiteSearch) return;
  const controllers = new WeakMap();
  const querySnapshots = new Map();
  let preparedIndex = null;
  const preload = async (context = {}) => {
    if (preparedIndex) return;
    const response = await fetch('dist/search-index.json', { cache: 'force-cache', signal: context.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!context.signal?.aborted) preparedIndex = data;
  };
  const mount = (context = {}) => {
    const root = context.root || document;
    if (controllers.has(root)) return controllers.get(root);
    let disposed = false;
    let searchRevision = 0;
    const abortController = new AbortController();
    const cleanups = [];
    const listen = (target, type, listener) => {
      target.addEventListener(type, listener);
      cleanups.push(() => target.removeEventListener(type, listener));
    };
    const routeUrl = new URL(context.url || window.location.href, window.location.href);
    const stateKey = routeUrl.pathname + '?audience=' + (routeUrl.searchParams.get('audience') || 'personal');

    const $ = (s, c = root) => c.querySelector(s);
    const initialQuery = (() => {
      try {
        const url = new URL(routeUrl.href);
        const query = url.searchParams.has('q') ? url.searchParams.get('q') : querySnapshots.get(stateKey)?.query || '';
        if (url.searchParams.has('q')) {
          url.searchParams.delete('q');
          window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
        }
        return query;
      } catch (_) {
        return '';
      }
    })();

    const headerForm = $('.nav-search', document);
    const headerInput = $('.nav-search-input', document);
    const pageForm = $('#search-page-form');
    const pageInput = $('#search-page-q');
    const results = $('#search-results');
    const status = $('#search-status');
    const filters = $('#search-filters');
    const filterButtons = Array.from(filters?.querySelectorAll('[data-search-category]') || []);
    let selectedCategory = routeUrl.searchParams.has('q') ? 'All' : querySnapshots.get(stateKey)?.category || 'All';
    let currentSearch = { entries: [], query: '', tokens: [] };
    const activeForm = pageForm || headerForm;
    const activeInput = pageInput || headerInput;

    if (!activeForm || !activeInput || !results || !status) return;

    const SITE_ORIGIN = 'https://www.danielshort.me';
    const INDEX_URL = 'dist/search-index.json';
    const MAX_RESULTS = 50;

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
      const pattern = [...new Set(tokens)].sort((a, b) => b.length - a.length).map(escapeRegExp).join('|');
      return raw.split(new RegExp(`(${pattern})`, 'ig')).map((part, index) => index % 2
        ? `<mark class="search-highlight">${escapeHtml(part)}</mark>` : escapeHtml(part)).join('');
    };

    const buildSnippet = (text, tokens) => {
      const raw = String(text ?? '').replace(/\s+/g, ' ').trim();
      if (!raw) return '';
      const maxLen = 220;
      if (!tokens.length) {
        return raw.length > maxLen ? raw.slice(0, maxLen).replace(/\s+\S*$/, '') + '…' : raw;
      }

      const lower = raw.toLowerCase();
      let firstIdx = -1;
      for (const token of tokens) {
        if (!token) continue;
        const idx = lower.indexOf(token.toLowerCase());
        if (idx === -1) continue;
        if (firstIdx === -1 || idx < firstIdx) firstIdx = idx;
      }

      if (firstIdx === -1) {
        return raw.length > maxLen ? raw.slice(0, maxLen).replace(/\s+\S*$/, '') + '…' : raw;
      }

      let start = Math.max(0, firstIdx - 60);
      let end = Math.min(raw.length, start + maxLen);
      if (start > 0) {
        const nextSpace = raw.indexOf(' ', start);
        if (nextSpace !== -1 && nextSpace < firstIdx) start = nextSpace + 1;
      }
      if (end < raw.length) {
        const prevSpace = raw.lastIndexOf(' ', end);
        if (prevSpace !== -1 && prevSpace > firstIdx) end = prevSpace;
      }

      let snippet = raw.slice(start, end).trim();
      if (start > 0) snippet = `…${snippet}`;
      if (end < raw.length) snippet = `${snippet}…`;
      return snippet;
    };

    const categoryLabel = (value) => {
      const raw = String(value || '').trim().toLowerCase();
      if (raw === 'portfolio' || raw === 'projects') return 'Projects';
      return raw === 'tools' ? 'Tools' : 'Pages';
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
      indexPromise = (preparedIndex ? Promise.resolve(preparedIndex) : fetch(INDEX_URL, { cache: 'force-cache', signal: abortController.signal })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        }))
        .then((data) => {
          const pages = data && Array.isArray(data.pages) ? data.pages : [];
          return pages
            .filter((p) => p && p.url && p.title)
            .map((p) => {
              const keywords = Array.isArray(p.keywords) ? p.keywords : [];
              const content = String(p.content || '').trim();
              const haystack = normalizeText([
                p.title,
                p.description,
                content,
                p.url,
                ...keywords
              ].join(' '));
              return { ...p, keywords, content, haystack };
            });
        })
        .catch((err) => {
          indexPromise = null;
          throw err;
        });
      return indexPromise;
    };

    const scoreEntry = (entry, tokens) => {
      if (!entry || !tokens.length) return 0;
      const title = normalizeText(entry.title);
      const desc = normalizeText(entry.description);
      const content = normalizeText(entry.content);
      const url = normalizeText(entry.url);
      const keywords = normalizeText((entry.keywords || []).join(' '));

      let score = 0;
      for (const token of tokens) {
        const inTitle = title.includes(token);
        const inDesc = desc.includes(token);
        const inKeywords = keywords.includes(token);
        const inContent = content.includes(token);
        const inUrl = url.includes(token);
        if (!(inTitle || inDesc || inKeywords || inContent || inUrl)) return 0;
        if (inTitle) score += 6;
        else if (inKeywords) score += 5;
        else if (inDesc) score += 3;
        else if (inContent) score += 2;
        else if (inUrl) score += 1;
        if (title.split(' ').includes(token)) score += 2;
      }
      const phrase = tokens.join(' ');
      if (title === phrase) score += 12;
      else if (title.startsWith(`${phrase} `)) score += 4;
      return score;
    };

    const renderEntry = (entry, tokens) => {
      const url = String(entry.url || '').trim();
      const title = String(entry.title || '').trim();
      const desc = String(entry.description || '').trim();
      const content = String(entry.content || '').trim();
      const badge = categoryLabel(entry.category);
      const snippet = desc ? '' : buildSnippet(content, tokens);

      return `
        <a class="search-result" href="${escapeHtml(url)}">
          <div class="search-result-head">
            <span class="search-result-title">${highlight(title, tokens)}</span>
            <span class="search-badge">${escapeHtml(badge)}</span>
          </div>
          <div class="search-result-url">${escapeHtml(toDisplayUrl(url))}</div>
          ${desc ? `<p class="search-result-desc">${highlight(desc, tokens)}</p>` : (snippet ? `<p class="search-result-desc">${highlight(snippet, tokens)}</p>` : '')}
        </a>
      `;
    };

    const renderResults = () => {
      const { entries: allEntries, query, tokens } = currentSearch;
      if (filters) filters.hidden = !query || !allEntries.length;
      filterButtons.forEach((button) => {
        const category = button.dataset.searchCategory;
        const count = category === 'All' ? allEntries.length : allEntries.filter((entry) => categoryLabel(entry.category) === category).length;
        button.setAttribute('aria-pressed', String(category === selectedCategory));
        button.setAttribute('aria-label', `${category}, ${count} result${count === 1 ? '' : 's'}`);
        button.innerHTML = `${escapeHtml(category)} <span class="search-filter-count" aria-hidden="true">${count}</span>`;
      });
      if (!query) {
        status.textContent = 'Try a tool name, project title, or a topic like SQL, OCR, or Tableau.';
        results.innerHTML = '';
        return;
      }

      const matches = selectedCategory === 'All' ? allEntries : allEntries.filter((entry) => categoryLabel(entry.category) === selectedCategory);
      const entries = matches.slice(0, MAX_RESULTS);
      const totalMatches = matches.length;
      const scope = selectedCategory === 'All' ? '' : ` in ${selectedCategory}`;

      if (!entries.length) {
        status.textContent = `No results${scope} for “${query}”.`;
        results.innerHTML = `
          <div class="search-result" role="note">
            <div class="search-result-head">
              <span class="search-result-title">No matches</span>
            </div>
            ${allEntries.length
              ? '<p class="search-result-desc">There are matches in other categories.</p><button class="btn-secondary" type="button" data-search-reset-category>Show all results</button>'
              : '<p class="search-result-desc">Try fewer words, or search for a tool or project name.</p><p class="search-result-desc">You can also browse the full <a class="search-result-title" href="sitemap">sitemap</a>.</p>'}
          </div>
        `;
        return;
      }

      const shown = entries.length;
      const isTruncated = Number.isFinite(totalMatches) && totalMatches > shown;

      status.textContent = isTruncated
        ? `Showing ${shown} of ${totalMatches} results${scope} for “${query}”.`
        : `${shown} result${shown === 1 ? '' : 's'}${scope} for “${query}”.`;

      results.innerHTML = entries.map((entry) => renderEntry(entry, tokens)).join('');
    };

    const syncInputs = (value, source) => {
      [headerInput, pageInput].forEach((input) => {
        if (!input || input === source) return;
        input.value = value;
      });
    };

    let updateTimer = null;
    const scheduleUpdate = (fn) => {
      if (updateTimer) window.clearTimeout(updateTimer);
      updateTimer = window.setTimeout(fn, 80);
    };

    const lengthBucket = (value) => {
      const length = String(value || '').trim().length;
      if (length === 0) return '0';
      if (length <= 10) return '1-10';
      if (length <= 25) return '11-25';
      if (length <= 50) return '26-50';
      return '51-plus';
    };

    const tokenBucket = (count) => {
      const numeric = Math.max(0, Number(count) || 0);
      if (numeric === 0) return '0';
      if (numeric === 1) return '1';
      if (numeric <= 3) return '2-3';
      if (numeric <= 6) return '4-6';
      return '7-plus';
    };

    const trackSearchEvent = (name, params) => {
      if (typeof window.gaEvent !== 'function') return;
      window.gaEvent(name, params);
    };

    const runSearch = async (query) => {
      const revision = ++searchRevision;
      if (disposed) return null;
      const trimmed = String(query || '').trim();
      activeInput.value = trimmed;
      syncInputs(trimmed, activeInput);

      const tokens = tokenize(trimmed);
      if (!tokens.length) {
        currentSearch = { entries: [], query: '', tokens: [] };
        results.setAttribute('aria-busy', 'false');
        renderResults();
        return { query: trimmed, tokenCount: 0, totalMatches: 0 };
      }

      status.textContent = 'Searching…';
      results.setAttribute('aria-busy', 'true');
      if (filters) filters.hidden = true;
      let pages;
      try {
        pages = await loadIndex();
      } catch (_) {
        if (disposed || revision !== searchRevision) return null;
        results.setAttribute('aria-busy', 'false');
        status.textContent = 'Search is unavailable right now. Please try again.';
        results.innerHTML = '<div class="search-result" role="note"><button class="btn-secondary" type="button" data-search-retry>Try again</button></div>';
        return null;
      }
      if (disposed || revision !== searchRevision) return null;

      const scored = pages
        .map((entry) => ({ entry, score: scoreEntry(entry, tokens) }))
        .filter((m) => m.score > 0)
        .sort((a, b) => b.score - a.score || String(a.entry.title).localeCompare(String(b.entry.title)));

      const totalMatches = scored.length;
      currentSearch = { entries: scored.map((match) => match.entry), query: trimmed, tokens };
      results.setAttribute('aria-busy', 'false');
      renderResults();
      return { query: trimmed, tokenCount: tokens.length, totalMatches };
    };

    syncInputs(initialQuery);
    activeInput.value = initialQuery;

    const bindForm = (form, input) => {
      if (!form || !input || form.dataset.searchBound === 'yes') return;
      form.dataset.searchBound = 'yes';
      cleanups.push(() => { delete form.dataset.searchBound; });
      listen(form, 'submit', (e) => {
        e.preventDefault();
        const searchSurface = form === pageForm ? 'search_page' : 'header';
        void runSearch(input.value).then((metrics) => {
          if (!metrics || !metrics.tokenCount) return;
          trackSearchEvent('site_search', {
            query_length_bucket: lengthBucket(metrics.query),
            query_token_bucket: tokenBucket(metrics.tokenCount),
            result_count: metrics.totalMatches,
            search_surface: searchSurface
          });
        });
      });
    };

    const bindInput = (input) => {
      if (!input || input.dataset.searchBound === 'yes') return;
      input.dataset.searchBound = 'yes';
      cleanups.push(() => { delete input.dataset.searchBound; });
      listen(input, 'input', () => {
        syncInputs(input.value, input);
        scheduleUpdate(() => runSearch(input.value));
      });

      listen(input, 'keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (!input.value) return;
        e.preventDefault();
        input.value = '';
        syncInputs('', input);
        runSearch('');
        try { input.blur(); } catch (_) {}
      });
    };

    bindForm(headerForm, headerInput);
    bindForm(pageForm, pageInput);
    bindInput(headerInput);
    bindInput(pageInput);

    filterButtons.forEach((button) => listen(button, 'click', () => {
      selectedCategory = button.dataset.searchCategory;
      renderResults();
    }));

    listen(results, 'click', (event) => {
      if (event.target.closest('[data-search-retry]')) {
        void runSearch(activeInput.value);
        return;
      }
      if (event.target.closest('[data-search-reset-category]')) {
        selectedCategory = 'All';
        renderResults();
        filterButtons[0]?.focus();
        return;
      }
      const result = event.target.closest('a.search-result[href]');
      if (!result || !results.contains(result)) return;
      const resultLinks = Array.from(results.querySelectorAll('a.search-result[href]'));
      const position = resultLinks.indexOf(result) + 1;
      if (position <= 0) return;
      const category = String(result.querySelector('.search-badge')?.textContent || 'Pages')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 40) || 'pages';
      trackSearchEvent('site_search_result_click', {
        result_category: category,
        result_position: position
      });
    });

    // Initial run
    const ready = runSearch(activeInput.value).then((metrics) => {
      if (!metrics || !metrics.tokenCount) return;
      trackSearchEvent('site_search', {
        query_length_bucket: lengthBucket(metrics.query),
        query_token_bucket: tokenBucket(metrics.tokenCount),
        result_count: metrics.totalMatches,
        search_surface: 'landing'
      });
    });
    const controller = {
      ready,
      dispose() {
        if (disposed) return;
        querySnapshots.set(stateKey, { query: activeInput.value, category: selectedCategory });
        while (querySnapshots.size > 4) querySnapshots.delete(querySnapshots.keys().next().value);
        disposed = true;
        searchRevision += 1;
        abortController.abort();
        if (updateTimer) window.clearTimeout(updateTimer);
        cleanups.splice(0).reverse().forEach((cleanup) => cleanup());
        controllers.delete(root);
      }
    };
    controllers.set(root, controller);
    context.cleanup?.(() => controller.dispose());
    return controller;
  };
  window.SiteSearch = Object.freeze({ mount, preload });
  if (window.SiteRoutes?.register) {
    window.SiteRoutes.register('search:search', {
      preload,
      async mount(context) {
        window.SiteContent?.mount(context);
        await mount(context)?.ready;
      }
    });
  } else {
    mount();
  }
})();
