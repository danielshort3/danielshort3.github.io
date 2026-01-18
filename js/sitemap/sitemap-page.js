(() => {
  'use strict';

  const SITEMAP_XML_PATH = '/sitemap.xml';
  const SEARCH_INDEX_PATH = '/dist/search-index.json';
  const SITEMAP_NS = 'http://www.sitemaps.org/schemas/sitemap/0.9';

  function $(selector) {
    return document.querySelector(selector);
  }

  function normalizeText(value) {
    return String(value || '').trim().toLowerCase();
  }

  function tokenize(query) {
    return normalizeText(query).split(/\s+/).filter(Boolean);
  }

  function parseLastmod(value) {
    const raw = String(value || '').trim();
    if (!raw) return null;
    const m = raw.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (!m) return null;
    const date = new Date(`${m[1]}-${m[2]}-${m[3]}T00:00:00Z`);
    if (Number.isNaN(date.getTime())) return null;
    return date;
  }

  function formatLastmod(value) {
    const date = parseLastmod(value);
    if (!date) return '—';
    try {
      return new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: '2-digit' }).format(date);
    } catch (_) {
      return String(value || '').trim() || '—';
    }
  }

  function readText(node) {
    if (!node) return '';
    return String(node.textContent || '').trim();
  }

  function toPathname(loc) {
    const raw = String(loc || '').trim();
    if (!raw) return '';
    try {
      return new URL(raw).pathname || '';
    } catch (_) {
      return raw.startsWith('/') ? raw : '';
    }
  }

  function readSitemapLastmods(xmlDoc) {
    const out = new Map();
    if (!xmlDoc) return out;
    const urls = Array.from(xmlDoc.getElementsByTagNameNS(SITEMAP_NS, 'url'));
    urls.forEach((urlEl) => {
      const locEl = urlEl.getElementsByTagNameNS(SITEMAP_NS, 'loc')[0];
      const lastmodEl = urlEl.getElementsByTagNameNS(SITEMAP_NS, 'lastmod')[0];
      const loc = readText(locEl);
      if (!loc) return;
      const path = toPathname(loc);
      if (!path) return;
      out.set(path, readText(lastmodEl));
    });
    return out;
  }

  function createSection(category, subtitle) {
    const section = document.createElement('section');
    section.className = 'sitemap-section';
    section.dataset.sitemapSection = category;

    const head = document.createElement('div');
    head.className = 'sitemap-section-head';

    const title = document.createElement('h2');
    title.className = 'sitemap-section-title';
    title.textContent = category;

    const count = document.createElement('span');
    count.className = 'sitemap-section-count';
    count.dataset.sitemapCount = category;
    count.textContent = '0';

    const titleRow = document.createElement('div');
    titleRow.className = 'sitemap-section-title-row';
    titleRow.appendChild(title);
    titleRow.appendChild(count);

    head.appendChild(titleRow);

    if (subtitle) {
      const sub = document.createElement('p');
      sub.className = 'sitemap-section-subtitle';
      sub.textContent = subtitle;
      head.appendChild(sub);
    }

    const items = document.createElement('div');
    items.className = 'sitemap-items';
    items.dataset.sitemapItems = category;

    section.appendChild(head);
    section.appendChild(items);
    return { section, items, count };
  }

  function createItem(entry) {
    const link = document.createElement('a');
    link.className = 'sitemap-item';
    link.href = entry.url;
    link.dataset.url = entry.url;

    const title = document.createElement('div');
    title.className = 'sitemap-item-title';
    title.textContent = entry.title || entry.url;

    const url = document.createElement('div');
    url.className = 'sitemap-item-url';
    url.textContent = entry.url;

    link.appendChild(title);
    link.appendChild(url);

    if (entry.description) {
      const desc = document.createElement('div');
      desc.className = 'sitemap-item-desc';
      desc.textContent = entry.description;
      link.appendChild(desc);
    }

    const meta = document.createElement('div');
    meta.className = 'sitemap-item-meta';

    const updated = document.createElement('span');
    updated.className = 'sitemap-item-updated';
    updated.textContent = `Updated: ${formatLastmod(entry.lastmod)}`;
    meta.appendChild(updated);

    link.appendChild(meta);

    const keywords = Array.isArray(entry.keywords) ? entry.keywords : [];
    const searchText = [
      entry.title,
      entry.description,
      entry.url,
      entry.category,
      ...keywords
    ]
      .map((v) => normalizeText(v))
      .filter(Boolean)
      .join(' ');
    link.dataset.searchText = searchText;
    return link;
  }

  async function fetchText(url) {
    const res = await fetch(url, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(`Failed to fetch ${url} (${res.status})`);
    return await res.text();
  }

  async function fetchJson(url) {
    const res = await fetch(url, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(`Failed to fetch ${url} (${res.status})`);
    return await res.json();
  }

  async function loadData() {
    const [xmlText, indexJson] = await Promise.all([
      fetchText(SITEMAP_XML_PATH),
      fetchJson(SEARCH_INDEX_PATH)
    ]);

    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
    const parseError = xmlDoc.getElementsByTagName('parsererror')[0];
    if (parseError) throw new Error('Could not parse sitemap.xml');

    const lastmods = readSitemapLastmods(xmlDoc);
    const pages = indexJson && Array.isArray(indexJson.pages) ? indexJson.pages : [];

    const seen = new Set();
    const merged = pages
      .map((page) => {
        const url = String(page && page.url ? page.url : '').trim();
        if (!url || !url.startsWith('/')) return null;
        if (seen.has(url)) return null;
        seen.add(url);
        return {
          url,
          title: String(page.title || '').trim(),
          description: String(page.description || '').trim(),
          category: String(page.category || '').trim() || 'Pages',
          keywords: Array.isArray(page.keywords) ? page.keywords : [],
          lastmod: lastmods.get(url) || ''
        };
      })
      .filter(Boolean);

    return {
      generatedAt: String(indexJson && indexJson.generatedAt ? indexJson.generatedAt : '').trim(),
      pages: merged
    };
  }

  function updateFilterState(options) {
    const input = options.input;
    const clear = options.clear;
    const statusEl = options.statusEl;
    const shownEl = options.shownEl;
    const totalEl = options.totalEl;
    const sections = options.sections;
    const items = options.items;

    const tokens = tokenize(input && input.value ? input.value : '');
    const query = tokens.join(' ');
    const isFiltering = tokens.length > 0;

    let shown = 0;

    items.forEach((item) => {
      const text = String(item.dataset.searchText || '');
      const match = !isFiltering || tokens.every((t) => text.includes(t));
      item.hidden = !match;
      if (match) shown += 1;
    });

    sections.forEach((section) => {
      const visible = section.items.querySelectorAll('.sitemap-item:not([hidden])').length;
      section.section.hidden = visible === 0;
      section.count.textContent = String(visible);
    });

    if (shownEl) shownEl.textContent = String(shown);
    if (totalEl) totalEl.textContent = String(items.length);
    if (clear) clear.disabled = !isFiltering;
    if (statusEl) {
      statusEl.textContent = isFiltering ? `Filtering by: “${query}”` : '';
    }

    try {
      const url = new URL(window.location.href);
      if (isFiltering) url.searchParams.set('q', query);
      else url.searchParams.delete('q');
      window.history.replaceState(null, '', url.toString());
    } catch (_) {}
  }

  async function main() {
    const sectionsEl = $('#sitemap-sections');
    if (!sectionsEl) return;

    const input = $('#sitemap-filter-input');
    const clear = $('#sitemap-filter-clear');
    const statusEl = $('#sitemap-status');
    const shownEl = document.querySelector('[data-sitemap-shown]');
    const totalEl = document.querySelector('[data-sitemap-total]');

    sectionsEl.innerHTML = '';

    const status = document.createElement('p');
    status.className = 'sitemap-loading';
    status.textContent = 'Loading sitemap…';
    sectionsEl.appendChild(status);

    let data;
    try {
      data = await loadData();
    } catch (err) {
      sectionsEl.innerHTML = '';
      const fallback = document.createElement('div');
      fallback.className = 'sitemap-empty';
      fallback.innerHTML = 'Could not load the sitemap data. You can still view the <a href="/sitemap.xml">XML sitemap</a>.';
      sectionsEl.appendChild(fallback);
      if (statusEl) statusEl.textContent = '';
      return;
    }

    const categoryOrder = [
      { name: 'Pages', subtitle: 'Core pages across the site.' },
      { name: 'Tools', subtitle: 'Privacy-first tools you can run in the browser.' },
      { name: 'Portfolio', subtitle: 'Shareable project pages under /portfolio/<id>.' }
    ];

    const grouped = new Map();
    data.pages.forEach((page) => {
      const category = String(page.category || '').trim() || 'Pages';
      if (!grouped.has(category)) grouped.set(category, []);
      grouped.get(category).push(page);
    });

    sectionsEl.innerHTML = '';

    const renderedSections = [];
    const allItems = [];

    categoryOrder.forEach((cat) => {
      const list = grouped.get(cat.name);
      if (!list || list.length === 0) return;

      list.sort((a, b) => String(a.title || a.url).localeCompare(String(b.title || b.url)));
      const section = createSection(cat.name, cat.subtitle);

      list.forEach((entry) => {
        const item = createItem(entry);
        section.items.appendChild(item);
        allItems.push(item);
      });

      renderedSections.push(section);
      sectionsEl.appendChild(section.section);
    });

    const extras = Array.from(grouped.keys()).filter((k) => !categoryOrder.some((c) => c.name === k));
    extras.sort().forEach((category) => {
      const list = grouped.get(category);
      if (!list || list.length === 0) return;

      list.sort((a, b) => String(a.title || a.url).localeCompare(String(b.title || b.url)));
      const section = createSection(category, '');
      list.forEach((entry) => {
        const item = createItem(entry);
        section.items.appendChild(item);
        allItems.push(item);
      });
      renderedSections.push(section);
      sectionsEl.appendChild(section.section);
    });

    if (totalEl) totalEl.textContent = String(allItems.length);

    const options = {
      input,
      clear,
      statusEl,
      shownEl,
      totalEl,
      sections: renderedSections,
      items: allItems
    };

    const urlQ = (() => {
      try {
        return new URLSearchParams(window.location.search || '').get('q') || '';
      } catch (_) {
        return '';
      }
    })();

    if (input && urlQ) input.value = urlQ;

    const runUpdate = () => updateFilterState(options);

    if (input) {
      input.addEventListener('input', runUpdate);
      input.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        input.value = '';
        runUpdate();
        input.blur();
      });
    }

    if (clear) {
      clear.addEventListener('click', () => {
        if (!input) return;
        input.value = '';
        runUpdate();
        input.focus();
      });
    }

    runUpdate();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
  } else {
    main();
  }
})();

