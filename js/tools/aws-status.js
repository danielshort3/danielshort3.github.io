(() => {
  'use strict';

  const REGION = 'us-east-2';
  const STATUS_URL = '/api/tools/aws-status';
  const HISTORY_DAYS = 90;
  const HISTORY_ZONE = 'America/Denver';
  const DAY_MS = 24 * 60 * 60 * 1000;
  const STALE_MS = 2 * 60 * 60 * 1000;
  const REFRESH_MS = 5 * 60 * 1000;
  const REQUEST_TIMEOUT_MS = 18 * 1000;
  const STATES = { operational: 'Operational', degraded: 'Degraded', outage: 'Outage', unknown: 'Unknown' };
  const GROUPS = ['Tools', 'Projects', 'Games'];
  const SOURCES = { production: 'Production check', browser: 'Browser check', local: 'Local check' };
  const FIXTURE_STATES = { operational: 'passed', degraded: 'had issues', outage: 'failed' };
  const DATE_KEY = /^\d{4}-\d{2}-\d{2}$/;

  const civilDateFormat = new Intl.DateTimeFormat('en-CA', {
    timeZone: HISTORY_ZONE, year: 'numeric', month: '2-digit', day: '2-digit'
  });
  const dayFormat = new Intl.DateTimeFormat(undefined, { timeZone: 'UTC', dateStyle: 'medium' });
  const clockFormat = new Intl.DateTimeFormat(undefined, {
    timeZone: HISTORY_ZONE, hour: 'numeric', minute: '2-digit', timeZoneName: 'short'
  });
  const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
    timeZone: HISTORY_ZONE, month: 'short', day: 'numeric', year: 'numeric',
    hour: 'numeric', minute: '2-digit', timeZoneName: 'short'
  });

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function statusBadge(state) {
    const badge = element('span', STATES[state], 'aws-status__badge');
    badge.dataset.state = state;
    return badge;
  }

  function hasTimestamp(value) {
    return typeof value === 'string' && Number.isFinite(Date.parse(value));
  }

  function mountainDay(time) {
    const parts = civilDateFormat.formatToParts(time);
    const date = Object.fromEntries(parts.filter((part) => part.type !== 'literal').map((part) => [part.type, part.value]));
    return date.year + '-' + date.month + '-' + date.day;
  }

  function recentDates(time = Date.now()) {
    // Walk civil date keys, since Mountain days can be 23 or 25 hours long.
    const end = Date.parse(mountainDay(time) + 'T00:00:00Z');
    return Array.from({ length: HISTORY_DAYS }, (_, index) =>
      new Date(end - (HISTORY_DAYS - 1 - index) * DAY_MS).toISOString().slice(0, 10));
  }

  function dateLabel(date) {
    return dayFormat.format(Date.parse(date + 'T00:00:00Z'));
  }

  function timeLabel(value) {
    const label = element('time', dateTimeFormat.format(Date.parse(value)));
    label.dateTime = value;
    return label;
  }

  function describeWindow(day) {
    const from = Date.parse(day.from);
    const to = Date.parse(day.to);
    if (!Number.isFinite(from) || !Number.isFinite(to)) return '';
    const format = mountainDay(from) === day.date && mountainDay(to) === day.date
      ? clockFormat : dateTimeFormat;
    if (from === to) return format.format(from);
    return format.format(from) + ' – ' + format.format(to);
  }

  function validStatus(data) {
    return data?.ok === true && data.region === REGION && data.days === HISTORY_DAYS &&
      (data.checkedAt === null || hasTimestamp(data.checkedAt)) &&
      Array.isArray(data.connections) && data.connections.every((connection) =>
        typeof connection?.label === 'string' && Array.isArray(connection.days) &&
        connection.days.length <= HISTORY_DAYS && connection.days.every((day) =>
          DATE_KEY.test(day?.date) && Number.isFinite(Date.parse(day.date)) &&
          Object.hasOwn(STATES, day.state) && hasTimestamp(day.from) && hasTimestamp(day.to) &&
          Date.parse(day.from) <= Date.parse(day.to) && typeof day.issue === 'string')) &&
      Array.isArray(data.catalog) && data.catalog.every((item) =>
        typeof item?.label === 'string' && GROUPS.includes(item.group) && Object.hasOwn(STATES, item.state) &&
        (item.lastChecked === null || hasTimestamp(item.lastChecked)) && typeof item.issue === 'string');
  }

  function getAccess() {
    const api = window.ToolsAuth;
    const auth = api?.getAuth?.();
    if (!api?.authIsValid?.(auth)) return { signedIn: false, owner: '' };
    const owner = api.isAdmin?.(auth) ? (api.getUser?.(auth)?.sub || '') : '';
    return { signedIn: true, owner, expiresAt: Number(auth.expiresAt) };
  }

  function init() {
    const root = document.querySelector('[data-aws-status]');
    if (!root) return;
    const part = (name) => root.querySelector('[data-aws-status-part="' + name + '"]');
    const dashboard = part('dashboard');
    const access = part('access');
    const refresh = part('refresh');
    const freshness = part('freshness');
    const notice = part('notice');
    const readings = part('readings');
    const history = part('history');
    const historyGrid = part('history-grid');
    const historyScroll = part('history-scroll');
    const directory = part('directory');
    const directoryTable = directory.querySelector('table');
    const tooltip = root.querySelector('#aws-status-day-details');
    const tabList = root.querySelector('[role="tablist"]');
    const tabs = [...tabList.querySelectorAll('[role="tab"]')];
    const panels = tabs.map((tab) => document.getElementById(tab.getAttribute('aria-controls')));
    const detailsByTick = new WeakMap();
    let owner = '';
    let snapshot = null;
    let failed = false;
    let request = null;
    let staleTimer;
    let accessTimer;
    let refreshTimer;
    let nextRefreshAt = 0;
    let tooltipTimer;
    let activeTick;
    let pinnedTick;
    let historyPositioned = false;

    function setNotice(message) {
      notice.textContent = message;
      notice.hidden = !message;
    }

    function setBusy(busy) {
      refresh.disabled = busy;
      refresh.textContent = busy ? 'Refreshing…' : 'Refresh';
      readings.setAttribute('aria-busy', String(busy));
    }

    function alignHistory() {
      if (!historyPositioned && !panels[0].hidden && !history.hidden && !readings.hidden) {
        historyScroll.scrollLeft = historyScroll.scrollWidth;
        historyPositioned = true;
      }
    }

    function hideTooltip() {
      clearTimeout(tooltipTimer);
      activeTick?.removeAttribute('aria-describedby');
      activeTick = null;
      pinnedTick = null;
      tooltip.hidden = true;
      tooltip.replaceChildren();
    }

    function selectTab(index, moveFocus = false) {
      hideTooltip();
      tabs.forEach((tab, i) => {
        const selected = i === index;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
        panels[i].hidden = !selected;
      });
      if (moveFocus) tabs[index].focus();
      alignHistory();
    }

    function clearStatus() {
      request?.abort();
      request = null;
      snapshot = null;
      failed = false;
      clearTimeout(staleTimer);
      clearTimeout(refreshTimer);
      nextRefreshAt = 0;
      readings.hidden = true;
      freshness.textContent = '';
      historyGrid.replaceChildren();
      for (const group of [...directoryTable.tBodies]) group.remove();
      historyPositioned = false;
      selectTab(0);
      setNotice('');
      setBusy(false);
    }

    function showAccess(message) {
      clearTimeout(accessTimer);
      clearStatus();
      dashboard.hidden = true;
      access.textContent = message;
      access.hidden = false;
    }

    function renderFreshness() {
      clearTimeout(staleTimer);
      if (!snapshot?.checkedAt) {
        freshness.textContent = 'Awaiting the first status check.';
        return;
      }
      const age = Math.max(0, Date.now() - Date.parse(snapshot.checkedAt));
      const stale = failed || age >= STALE_MS;
      freshness.replaceChildren('Last checked ', timeLabel(snapshot.checkedAt), stale ? ' · Stale' : '');
      if (!stale) staleTimer = setTimeout(renderFreshness, STALE_MS - age);
    }

    function renderHistory() {
      const dates = recentDates();
      const rows = snapshot.connections.map((connection) => {
        const byDate = new Map(connection.days.map((day) => [day.date, day]));
        const row = element('div', '', 'aws-status__history-row');
        const label = element('h3', connection.label, 'aws-status__service');
        const ticks = element('div', '', 'aws-status__ticks');
        ticks.setAttribute('role', 'group');
        ticks.setAttribute('aria-label', connection.label + ' daily history. Use left and right arrows to choose a day.');
        dates.forEach((date, index) => {
          const day = byDate.get(date) || { date, state: 'unknown' };
          const tick = element('button', '', 'aws-status__tick');
          tick.type = 'button';
          tick.tabIndex = index === dates.length - 1 ? 0 : -1;
          tick.dataset.state = day.state;
          tick.setAttribute('aria-label', dateLabel(date) + ', ' + STATES[day.state]);
          detailsByTick.set(tick, day);
          ticks.append(tick);
        });
        row.append(label, ticks);
        return row;
      });
      historyGrid.replaceChildren(...rows);
      history.hidden = rows.length === 0;
      part('connections-empty').hidden = rows.length > 0;
    }

    function renderDirectory() {
      for (const body of [...directoryTable.tBodies]) body.remove();
      for (const group of GROUPS) {
        const items = snapshot.catalog.filter((item) => item.group === group);
        if (!items.length) continue;
        const body = element('tbody');
        const header = element('tr', '', 'aws-status__directory-group');
        const heading = element('th', group);
        heading.scope = 'rowgroup';
        heading.colSpan = 2;
        header.append(heading);
        body.append(header);
        for (const item of items) {
          const row = element('tr');
          const name = element('th', item.label);
          const status = element('td');
          const source = Object.hasOwn(SOURCES, item.source) ? SOURCES[item.source] : '';
          const badge = statusBadge(item.state);
          if (source && item.source !== 'production' && Object.hasOwn(FIXTURE_STATES, item.state)) {
            badge.textContent = source + ' ' + FIXTURE_STATES[item.state];
          }
          const detail = [
            source,
            item.lastChecked && 'Last checked ' + timeLabel(item.lastChecked).textContent,
            item.issue
          ].filter(Boolean).join(' · ');
          if (detail) badge.title = detail;
          name.scope = 'row';
          status.append(badge);
          if (detail) status.append(element('span', ' ' + detail, 'visually-hidden'));
          row.append(name, status);
          body.append(row);
        }
        directoryTable.append(body);
      }
      directory.hidden = snapshot.catalog.length === 0;
      part('catalog-empty').hidden = snapshot.catalog.length > 0;
    }

    function renderStatus() {
      hideTooltip();
      renderFreshness();
      renderHistory();
      renderDirectory();
      readings.hidden = false;
      alignHistory();
    }

    function positionTooltip() {
      if (!activeTick) return;
      const tick = activeTick.getBoundingClientRect();
      const clip = historyScroll.getBoundingClientRect();
      if (tick.right <= clip.left || tick.left >= clip.right ||
        tick.bottom <= 0 || tick.top >= window.innerHeight) {
        hideTooltip();
        return;
      }
      const margin = 12;
      const gap = 8;
      const { width, height } = tooltip.getBoundingClientRect();
      const center = (tick.left + tick.right) / 2;
      const left = Math.max(margin, Math.min(window.innerWidth - width - margin, center - width / 2));
      const above = tick.top - height - gap >= margin || tick.top > window.innerHeight - tick.bottom;
      const top = above ? tick.top - height - gap : tick.bottom + gap;
      tooltip.style.left = left + 'px';
      tooltip.style.top = Math.max(margin, Math.min(window.innerHeight - height - margin, top)) + 'px';
      tooltip.style.setProperty('--aws-status-tip', Math.max(12, Math.min(width - 12, center - left)) + 'px');
      tooltip.dataset.side = above ? 'above' : 'below';
    }

    function showTooltip(tick) {
      clearTimeout(tooltipTimer);
      if (activeTick === tick) {
        positionTooltip();
        return;
      }
      activeTick?.removeAttribute('aria-describedby');
      activeTick = tick;
      const day = detailsByTick.get(tick);
      const title = element('strong', dateLabel(day.date));
      const windowLabel = describeWindow(day);
      if (windowLabel) title.append(' · ' + windowLabel);
      tooltip.replaceChildren(title, statusBadge(day.state));
      if (day.issue) tooltip.append(element('p', day.issue));
      if (hasTimestamp(day.observedAt)) {
        const observed = element('p', 'Observed ', 'aws-status__observed');
        observed.append(timeLabel(day.observedAt));
        tooltip.append(observed);
      }
      tooltip.hidden = false;
      tick.setAttribute('aria-describedby', tooltip.id);
      positionTooltip();
    }

    function queueTooltipClose(tick) {
      if (activeTick !== tick || pinnedTick === tick || document.activeElement === tick) return;
      clearTimeout(tooltipTimer);
      tooltipTimer = setTimeout(hideTooltip, 150);
    }

    function tickFrom(event) {
      const tick = event.target.closest?.('.aws-status__tick');
      return detailsByTick.has(tick) ? tick : null;
    }

    function setCurrentTick(tick) {
      const current = tick.parentElement.querySelector('[tabindex="0"]');
      if (current) current.tabIndex = -1;
      tick.tabIndex = 0;
    }

    function installHistoryControls() {
      historyGrid.addEventListener('pointerover', (event) => {
        const tick = tickFrom(event);
        if (tick && event.pointerType !== 'touch' && !pinnedTick) showTooltip(tick);
      });
      historyGrid.addEventListener('pointerout', (event) => {
        const tick = tickFrom(event);
        if (tick && !tooltip.contains(event.relatedTarget)) queueTooltipClose(tick);
      });
      historyGrid.addEventListener('focusin', (event) => {
        const tick = tickFrom(event);
        if (!tick) return;
        setCurrentTick(tick);
        if (pinnedTick !== tick) pinnedTick = null;
        showTooltip(tick);
      });
      historyGrid.addEventListener('focusout', (event) => {
        const tick = tickFrom(event);
        if (tick && tick === activeTick && !tooltip.contains(event.relatedTarget)) hideTooltip();
      });
      historyGrid.addEventListener('click', (event) => {
        const tick = tickFrom(event);
        if (!tick) return;
        if (pinnedTick === tick) {
          hideTooltip();
          return;
        }
        pinnedTick = tick;
        setCurrentTick(tick);
        showTooltip(tick);
      });
      historyGrid.addEventListener('keydown', (event) => {
        const tick = tickFrom(event);
        if (!tick || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        const siblings = [...tick.parentElement.children];
        const index = siblings.indexOf(tick);
        const next = {
          ArrowLeft: siblings[Math.max(0, index - 1)],
          ArrowRight: siblings[Math.min(siblings.length - 1, index + 1)],
          Home: siblings[0],
          End: siblings.at(-1)
        }[event.key];
        event.preventDefault();
        next.focus();
        if (next === tick) showTooltip(tick);
      });
      tooltip.addEventListener('pointerenter', () => clearTimeout(tooltipTimer));
      tooltip.addEventListener('pointerleave', () => queueTooltipClose(activeTick));
      document.addEventListener('pointerdown', (event) => {
        if (activeTick && !tickFrom(event) && !tooltip.contains(event.target)) hideTooltip();
      });
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && activeTick) {
          hideTooltip();
          event.stopPropagation();
        }
      }, true);
      window.addEventListener('resize', positionTooltip);
      window.addEventListener('scroll', positionTooltip, { capture: true, passive: true });
    }

    function scheduleRefresh() {
      clearTimeout(refreshTimer);
      if (!owner || request || document.visibilityState !== 'visible') return;
      const delay = nextRefreshAt - Date.now();
      if (delay > 0) refreshTimer = setTimeout(scheduleRefresh, delay);
      else loadStatus();
    }

    async function loadStatus() {
      const { owner: currentOwner } = getAccess();
      if (currentOwner !== owner) {
        syncAccess();
        return;
      }
      if (!currentOwner || request || document.visibilityState !== 'visible') return;
      clearTimeout(refreshTimer);
      nextRefreshAt = Date.now() + REFRESH_MS;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const isCurrent = () => request === controller && getAccess().owner === currentOwner;
      request = controller;
      setBusy(true);
      setNotice(snapshot ? 'Refreshing AWS status…' : 'Checking AWS status…');
      try {
        const response = await window.ToolsAuth.fetchWithAuth(STATUS_URL, {
          cache: 'no-store',
          signal: controller.signal,
          headers: { Accept: 'application/json' }
        });
        if (!isCurrent()) return;
        if (response.status === 401 || response.status === 403) {
          owner = '';
          showAccess(response.status === 401
            ? 'Sign in again to view AWS status.'
            : 'This account does not have access to AWS status.');
          return;
        }
        if (!response.ok) throw new Error('Status unavailable');
        const data = await response.json();
        if (!isCurrent()) return;
        if (!validStatus(data)) throw new Error('Invalid status response');
        snapshot = data;
        failed = false;
        renderStatus();
        setNotice('');
      } catch {
        if (!isCurrent()) return;
        failed = true;
        if (snapshot) renderFreshness();
        setNotice(snapshot
          ? 'Could not refresh AWS status. Showing the last result.'
          : 'AWS status is unavailable. Try refreshing.');
      } finally {
        clearTimeout(timeout);
        if (request === controller) {
          request = null;
          setBusy(false);
          scheduleRefresh();
        }
      }
    }

    function syncAccess() {
      clearTimeout(accessTimer);
      const next = getAccess();
      if (!next.owner) {
        owner = '';
        showAccess(next.signedIn
          ? 'This account does not have access to AWS status.'
          : 'Sign in with an administrator account to view AWS status.');
        return;
      }
      if (Number.isFinite(next.expiresAt)) {
        const delay = Math.max(1000, next.expiresAt - Date.now() - 60_000 + 1);
        accessTimer = setTimeout(syncAccess, delay);
      }
      if (next.owner === owner) return;
      owner = next.owner;
      clearStatus();
      access.hidden = true;
      dashboard.hidden = false;
      loadStatus();
    }

    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => selectTab(index));
      tab.addEventListener('keydown', (event) => {
        const next = {
          ArrowLeft: (index - 1 + tabs.length) % tabs.length,
          ArrowRight: (index + 1) % tabs.length,
          Home: 0,
          End: tabs.length - 1
        }[event.key];
        if (!Number.isInteger(next)) return;
        event.preventDefault();
        selectTab(next, true);
      });
    });
    installHistoryControls();
    refresh.addEventListener('click', () => {
      syncAccess();
      loadStatus();
    });
    document.addEventListener('tools:auth-changed', syncAccess);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') syncAccess();
      scheduleRefresh();
    });
    window.addEventListener('pagehide', () => {
      owner = '';
      showAccess('Sign in with an administrator account to view AWS status.');
    });
    window.addEventListener('pageshow', (event) => {
      if (event.persisted) syncAccess();
    });
    syncAccess();
  }

  if (document.readyState === 'complete') init();
  else document.addEventListener('DOMContentLoaded', init, { once: true });
})();
