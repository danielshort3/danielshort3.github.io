(() => {
  'use strict';

  const MONTH_PATTERN = '(January|February|March|April|May|June|July|August|September|October|November|December)';
  const MONTH_DAY_YEAR_RE = new RegExp(`${MONTH_PATTERN}\\s+\\d{1,2},\\s+\\d{4}`);
  const MONTH_YEAR_RE = new RegExp(`${MONTH_PATTERN}\\s+\\d{4}`);

  const isHome = () => document?.body?.dataset?.page === 'home';

  const formatMonthYear = (date) => {
    if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '';
    return date.toLocaleString('en-US', { month: 'short', year: 'numeric' });
  };

  const parseItemDate = (item) => {
    if (!item) return null;
    if (item.date) {
      const explicit = new Date(item.date);
      if (!Number.isNaN(explicit.getTime())) return explicit;
    }

    const haystack = `${item.title ?? ''} ${item.role ?? ''}`;
    const match = haystack.match(MONTH_DAY_YEAR_RE);
    if (match) {
      const inferred = new Date(`${match[0]} 00:00:00 UTC`);
      if (!Number.isNaN(inferred.getTime())) return inferred;
    }

    const monthYear = haystack.match(MONTH_YEAR_RE);
    if (monthYear) {
      const [month, year] = monthYear[0].split(/\s+/);
      const inferred = new Date(`${month} 1, ${year} 00:00:00 UTC`);
      if (!Number.isNaN(inferred.getTime())) return inferred;
    }

    return null;
  };

  const findLatestContribution = (slug) => {
    const sections = window?.contributions;
    if (!Array.isArray(sections)) return null;
    const section = sections.find(s => s?.slug === slug);
    const items = section?.items;
    if (!Array.isArray(items) || !items.length) return null;

    return items.reduce((latest, item) => {
      if (!latest) return item;
      const latestDate = parseItemDate(latest);
      const nextDate = parseItemDate(item);
      if (!latestDate && nextDate) return item;
      if (!latestDate && !nextDate) return latest;
      if (latestDate && !nextDate) return latest;
      return nextDate.getTime() > latestDate.getTime() ? item : latest;
    }, null);
  };

  const ensureContributionsData = () => {
    return new Promise((resolve) => {
      if (Array.isArray(window?.contributions)) {
        resolve(true);
        return;
      }

      const existing = document.querySelector('script[data-contributions-data="true"]');
      if (existing) {
        existing.addEventListener('load', () => resolve(true), { once: true });
        existing.addEventListener('error', () => resolve(false), { once: true });
        return;
      }

      const script = document.createElement('script');
      script.src = 'js/contributions/contributions-data.js';
      script.defer = true;
      script.dataset.contributionsData = 'true';
      script.onload = () => resolve(true);
      script.onerror = () => resolve(false);
      document.head.appendChild(script);
    });
  };

  const updateLatestCouncilBriefingCard = () => {
    const card = document.getElementById('latest-council-briefing');
    if (!card) return;

    const latest = findLatestContribution('council-briefings');
    if (!latest) return;

    const titleEl = card.querySelector('.doc-title');
    if (titleEl && latest.title) titleEl.textContent = latest.title;

    const roleEl = card.querySelector('.doc-role');
    if (roleEl && latest.role) roleEl.textContent = latest.role;

    const date = parseItemDate(latest);
    const labelEl = card.querySelector('.doc-label');
    if (labelEl && date) {
      labelEl.textContent = `Latest council briefing · ${formatMonthYear(date)}`;
    }

    const linkEl = card.querySelector('.doc-links .doc-link');
    if (linkEl && latest.link) {
      linkEl.href = latest.link;
      linkEl.target = '_blank';
      linkEl.rel = 'noopener noreferrer';
      linkEl.setAttribute('aria-label', 'Open latest council briefing');
    }
  };

  const initHome = () => {
    if (!isHome()) return;
    ensureContributionsData().then(() => updateLatestCouncilBriefingCard());
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHome);
  } else {
    initHome();
  }
})();

