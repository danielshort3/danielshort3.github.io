(() => {
  'use strict';

  const REGION = 'us-east-2';
  const STATUS_URL = '/api/tools/aws-status';
  const HISTORY_DAYS = 90;
  const HISTORY_HOURS = 72;
  const HISTORY_ZONE = 'America/Denver';
  const HOUR_MS = 60 * 60 * 1000;
  const DAY_MS = 24 * HOUR_MS;
  const STALE_MS = 2 * 60 * 60 * 1000;
  const HOME_GPU_STALE_MS = 75 * 60 * 1000;
  const REFRESH_MS = 5 * 60 * 1000;
  const REQUEST_TIMEOUT_MS = 18 * 1000;
  const STATES = { operational: 'Operational', degraded: 'Degraded', outage: 'Outage', unknown: 'Unknown' };
  const HOME_GPU_STATES = { operational: 'Online', degraded: 'Worker unavailable', outage: 'Offline', unknown: 'Unknown' };
  const GROUPS = ['Tools', 'Projects', 'Games'];
  const DATE_KEY = /^\d{4}-\d{2}-\d{2}$/;
  const HOUR_KEY = /^\d{4}-\d{2}-\d{2}T\d{2}:00:00\.000Z$/;
  const CHECK_RESULTS = {
    '-': ['unknown', 'No check recorded.'],
    u: ['unknown', 'Inconclusive check.'],
    o: ['operational', ''],
    d: ['degraded', 'Check reported an issue.'],
    x: ['outage', 'Check failed.']
  };
  const HISTORY_VIEWS = {
    hourly: { range: 'Last 72 hours', start: '72 hours ago', end: 'Latest shown hour', period: 'an hour' },
    daily: { range: 'Last 90 days', start: '90 days ago', end: 'Today', period: 'a day' },
    weekly: { range: 'Last 90 days', start: '90 days ago', end: 'This week', period: 'a week' }
  };

  const civilDateFormat = new Intl.DateTimeFormat('en-CA', {
    timeZone: HISTORY_ZONE, year: 'numeric', month: '2-digit', day: '2-digit'
  });
  const dayFormat = new Intl.DateTimeFormat(undefined, { timeZone: 'UTC', dateStyle: 'medium' });
  const shortDayFormat = new Intl.DateTimeFormat(undefined, {
    timeZone: HISTORY_ZONE, month: 'short', day: 'numeric'
  });
  const hourFormat = new Intl.DateTimeFormat(undefined, {
    timeZone: HISTORY_ZONE, hour: 'numeric', minute: '2-digit'
  });
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

  function recentHours(time) {
    const end = Math.floor(time / HOUR_MS) * HOUR_MS;
    return Array.from({ length: HISTORY_HOURS }, (_, index) => {
      const from = end - (HISTORY_HOURS - 1 - index) * HOUR_MS;
      const to = from + HOUR_MS;
      return { start: new Date(from).toISOString(), label: hourLabel(from, to), from, to };
    });
  }

  function calendarWeeks(dates) {
    const weeks = [];
    for (const date of dates) {
      const weekday = new Date(date + 'T00:00:00Z').getUTCDay();
      if (weekday === 1 || weeks.length === 0) weeks.push([]);
      weeks.at(-1).push(date);
    }
    return weeks.map((week) => {
      const first = dateLabel(week[0]);
      const last = dateLabel(week.at(-1));
      return { dates: week, label: first === last ? first : first + ' – ' + last };
    });
  }

  function weekResult({ dates, label }, byDate) {
    const days = dates.map((date) => byDate.get(date));
    const incident = days.findLast((day) => day?.state === 'outage') ||
      days.findLast((day) => day?.state === 'degraded');
    if (incident) return { ...incident, label, issue: dateLabel(incident.date) + ' · ' + incident.issue };
    const measured = days.filter((day) => day?.state === 'operational');
    const complete = measured.length === days.length;
    return {
      label,
      state: complete ? 'operational' : 'unknown',
      issue: complete ? '' : measured.length ? 'Some checks missing.' : 'No check recorded.'
    };
  }

  function dateLabel(date) {
    return dayFormat.format(Date.parse(date + 'T00:00:00Z'));
  }

  function hourLabel(from, to) {
    const startDay = shortDayFormat.format(from);
    const endDay = mountainDay(from) === mountainDay(to) ? '' : shortDayFormat.format(to) + ', ';
    const [startClock, endClock] = [from, to].map((time) =>
      clockFormat.formatToParts(time).find((part) => part.type === 'timeZoneName')?.value);
    const endTime = startClock === endClock ? hourFormat : clockFormat;
    return startDay + ', ' + clockFormat.format(from) + ' – ' + endDay + endTime.format(to);
  }

  function timeLabel(value) {
    const label = element('time', dateTimeFormat.format(Date.parse(value)));
    label.dateTime = value;
    return label;
  }

  function homeGpuLastCheck(hours) {
    const now = Date.now();
    const latest = hours?.findLast((hour) => {
      const observed = Date.parse(hour.observedAt);
      const start = Date.parse(hour.start);
      return hour.complete === true && hasTimestamp(hour.observedAt) && observed <= now &&
        observed >= start && observed < start + HOUR_MS;
    });
    const status = ['Home GPU last check: '];
    if (latest) {
      const stale = now - Date.parse(latest.observedAt) >= HOME_GPU_STALE_MS;
      status.push(element('strong', HOME_GPU_STATES[latest.state]), ' · ', timeLabel(latest.observedAt));
      if (stale) status.push(' · Stale');
    } else {
      status.push('Unknown');
    }
    return status;
  }

  function validHistoryTile(tile) {
    return Object.hasOwn(STATES, tile?.state) && hasTimestamp(tile.from) && hasTimestamp(tile.to) &&
      Date.parse(tile.from) <= Date.parse(tile.to) && typeof tile.issue === 'string';
  }

  function validConnection(connection) {
    return typeof connection?.label === 'string' && Array.isArray(connection.days) &&
      connection.days.length <= HISTORY_DAYS && connection.days.every((day) =>
        DATE_KEY.test(day?.date) && Number.isFinite(Date.parse(day.date)) && validHistoryTile(day)) &&
      (connection.hours === undefined || Array.isArray(connection.hours) && connection.hours.length <= HISTORY_HOURS &&
        connection.hours.every((hour) => HOUR_KEY.test(hour?.start) && hasTimestamp(hour.start) && validHistoryTile(hour)));
  }

  function catalogHistory(data) {
    const history = data.catalogHistory;
    if (!HOUR_KEY.test(history?.hourStart) || !hasTimestamp(history.hourStart) ||
        !DATE_KEY.test(history.dayStart) || !hasTimestamp(history.dayStart) ||
        !Array.isArray(history.observedAt) || history.observedAt.length !== HISTORY_HOURS ||
        !history.results || typeof history.results !== 'object') return null;
    return history;
  }

  function catalogObservation(code) {
    const [state, issue] = Object.hasOwn(CHECK_RESULTS, code) ? CHECK_RESULTS[code] : CHECK_RESULTS['-'];
    return { state, issue };
  }

  function validStatus(data) {
    return data?.ok === true && data.region === REGION && data.days === HISTORY_DAYS &&
      (data.checkedAt === null || hasTimestamp(data.checkedAt)) &&
      Array.isArray(data.connections) && data.connections.every(validConnection) &&
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
    const part = (name, scope = root) => scope.querySelector('[data-aws-status-part="' + name + '"]');
    const dashboard = part('dashboard');
    const access = part('access');
    const refresh = part('refresh');
    const freshness = part('freshness');
    const notice = part('notice');
    const readings = part('readings');
    const tooltip = root.querySelector('#aws-status-history-details');
    const granularity = part('granularity');
    const history = part('history');
    const grid = part('history-grid');
    const scroll = part('history-scroll');
    const rangeLabel = part('history-range');
    const startLabel = part('history-start');
    const endLabel = part('history-end');
    const empty = part('history-empty');
    const homeGpuDetails = part('home-gpu-details');
    const homeGpuStatus = part('home-gpu-last-check');
    const detailsByTick = new WeakMap();
    const expandedGroups = new Set();
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

    function setNotice(message) {
      notice.textContent = message;
      notice.hidden = !message;
    }

    function setBusy(busy) {
      refresh.disabled = busy;
      refresh.textContent = busy ? 'Refreshing…' : 'Refresh';
      readings.setAttribute('aria-busy', String(busy));
    }

    function alignHistory(left) {
      scroll.scrollLeft = left ?? scroll.scrollWidth;
    }

    function hideTooltip() {
      clearTimeout(tooltipTimer);
      activeTick?.removeAttribute('aria-describedby');
      activeTick = null;
      pinnedTick = null;
      tooltip.hidden = true;
      tooltip.replaceChildren();
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
      hideTooltip();
      grid.replaceChildren();
      if (homeGpuDetails) homeGpuDetails.hidden = true;
      homeGpuStatus?.replaceChildren();
      granularity.value = 'hourly';
      expandedGroups.clear();
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

    function historyPeriods(item, timeline, catalog) {
      const { dates, labels, weeks, hours, view } = timeline;
      if (hours && catalog) {
        const results = catalog.history?.results[item.id]?.hours;
        const checked = Date.parse(item.lastChecked);
        return hours.map(({ from, to, label }) => {
          const index = (from - catalog.hourStart) / HOUR_MS;
          const code = results?.[index];
          const observation = catalogObservation(code);
          if (['u', 'd', 'x'].includes(code) && observation.state === item.state && checked >= from && checked < to) {
            observation.issue = item.issue || observation.issue;
          }
          return { label, ...observation };
        });
      }
      if (hours) {
        const byHour = new Map((item.hours || []).map((hour) => [hour.start, hour]));
        return hours.map(({ start, label }) => {
          const hour = byHour.get(start);
          return {
            ...hour,
            label,
            state: hour?.state || 'unknown',
            issue: hour?.issue || 'No check recorded.'
          };
        });
      }
      const byDate = catalog ? new Map(dates.map((date) => {
        const index = (Date.parse(date) - catalog.dayStart) / DAY_MS;
        const code = catalog.history?.results[item.id]?.days?.[index];
        const day = catalogObservation(code);
        if (code === 'u') day.issue = 'Missing or inconclusive checks.';
        return [date, { date, ...day }];
      })) : new Map(item.days.map((day) => [day.date, day]));
      if (view === 'weekly') return weeks.map((week) => weekResult(week, byDate));
      return dates.map((date) => ({
        state: 'unknown', issue: 'No check recorded.', ...byDate.get(date), label: labels.get(date)
      }));
    }

    function renderTicks(label, periods, timeline, stateLabels) {
      const ticks = element('div', '', 'aws-status__ticks');
      ticks.setAttribute('role', 'group');
      ticks.setAttribute('aria-label', label + ' ' + timeline.view +
        ' history. Use left and right arrows to choose ' + timeline.period + '.');
      periods.forEach((detail, index) => {
        const stateLabel = stateLabels[detail.state];
        const tick = element('button', '', 'aws-status__tick');
        tick.type = 'button';
        tick.tabIndex = index === periods.length - 1 ? 0 : -1;
        tick.dataset.state = detail.state;
        tick.setAttribute('aria-label', detail.label + ', ' + stateLabel);
        detailsByTick.set(tick, { ...detail, stateLabel });
        ticks.append(tick);
      });
      return ticks;
    }

    function groupResult(rows, index) {
      const incident = rows.find((row) => row.periods[index].state === 'outage') ||
        rows.find((row) => row.periods[index].state === 'degraded');
      if (incident) return { ...incident.periods[index], issue: incident.label };
      const unknown = rows.some((row) => row.periods[index].state === 'unknown');
      return {
        label: rows[0].periods[index].label,
        state: unknown ? 'unknown' : 'operational',
        issue: unknown ? 'Missing or inconclusive checks.' : ''
      };
    }

    function renderGroup(group, items, timeline, catalog) {
      const rows = items.map((item) => ({ label: item.label, periods: historyPeriods(item, timeline, catalog) }));
      const homeGpu = group === 'Local services';
      const stateLabels = homeGpu ? HOME_GPU_STATES : STATES;
      const section = element('section', '', 'aws-status__group');
      section.dataset.group = group;
      const summary = element('div', '', 'aws-status__group-summary');
      const heading = element('h3', '', 'aws-status__group-heading');
      const toggle = element('button', group, 'aws-status__group-toggle');
      toggle.type = 'button';
      const count = element('span', String(rows.length), 'aws-status__group-count');
      count.setAttribute('aria-hidden', 'true');
      toggle.append(count);
      heading.append(toggle);
      const summaryPeriods = homeGpu ? rows[0].periods : rows[0].periods.map((_, index) => groupResult(rows, index));
      summary.append(heading, renderTicks(group, summaryPeriods, timeline, stateLabels));
      const body = element('div', '', 'aws-status__group-items');
      body.id = 'aws-status-group-' + group.toLowerCase().replace(/ /g, '-');
      toggle.setAttribute('aria-controls', homeGpu && homeGpuDetails ? body.id + ' ' + homeGpuDetails.id : body.id);
      section.append(summary, body);
      function renderItems() {
        const open = expandedGroups.has(group);
        toggle.setAttribute('aria-expanded', String(open));
        body.hidden = !open;
        body.replaceChildren(...(open ? rows.map(({ label, periods }) => {
          const row = element('div', '', 'aws-status__history-row');
          row.append(element('h4', label, 'aws-status__service'), renderTicks(label, periods, timeline, stateLabels));
          return row;
        }) : []));
        if (homeGpu && homeGpuDetails && homeGpuStatus) {
          homeGpuDetails.hidden = !open;
          homeGpuStatus.replaceChildren(...(open ? homeGpuLastCheck(items[0].hours) : []));
        }
      }
      toggle.addEventListener('click', () => {
        hideTooltip();
        if (expandedGroups.has(group)) expandedGroups.delete(group);
        else expandedGroups.add(group);
        renderItems();
      });
      renderItems();
      return section;
    }

    function renderHistory(left) {
      const view = Object.hasOwn(HISTORY_VIEWS, granularity.value) ? granularity.value : 'hourly';
      const { range, start, end, period } = HISTORY_VIEWS[view];
      const now = Date.now();
      const cutoff = HOUR_KEY.test(snapshot.hourlyReadyThrough) && hasTimestamp(snapshot.hourlyReadyThrough)
        ? Date.parse(snapshot.hourlyReadyThrough) : null;
      const firstUnsettledDay = cutoff === null ? null : mountainDay(cutoff + HOUR_MS);
      const dates = view === 'hourly' ? [] : recentDates(now).filter((date) => !firstUnsettledDay || date < firstUnsettledDay);
      const timeline = {
        view, period, dates,
        labels: view === 'daily' ? new Map(dates.map((date) => [date, dateLabel(date)])) : null,
        weeks: view === 'weekly' ? calendarWeeks(dates) : null,
        hours: view === 'hourly' ? recentHours(now).filter((hour) => cutoff === null || hour.from <= cutoff) : null
      };
      rangeLabel.textContent = range;
      startLabel.textContent = start;
      endLabel.textContent = cutoff === null ? end : {
        hourly: 'Latest completed hour', daily: 'Latest completed day', weekly: 'Latest shown week'
      }[view];
      grid.dataset.granularity = view;
      const catalogData = catalogHistory(snapshot);
      const catalog = {
        history: catalogData, hourStart: Date.parse(catalogData?.hourStart), dayStart: Date.parse(catalogData?.dayStart)
      };
      const groups = [
        ['AWS Connections', snapshot.connections],
        ['Local services', snapshot.homeGpu?.id === 'home-gpu' && validConnection(snapshot.homeGpu) ? [snapshot.homeGpu] : []],
        ...GROUPS.map((group) => [group, snapshot.catalog.filter((item) => item.group === group), catalog])
      ];
      const periodCount = (timeline.hours || timeline.weeks || dates).length;
      if (homeGpuDetails) homeGpuDetails.hidden = true;
      const sections = groups.filter(([, items]) => items.length > 0 && periodCount > 0)
        .map(([group, items, data]) => renderGroup(group, items, timeline, data));
      const focus = document.activeElement?.closest('.aws-status__group');
      const focusToggle = focus?.querySelector('.aws-status__group-toggle') === document.activeElement;
      const focusHistory = document.activeElement?.parentElement?.getAttribute('aria-label');
      const focusDate = document.activeElement?.getAttribute('aria-label');
      grid.replaceChildren(...sections);
      history.hidden = sections.length === 0;
      empty.hidden = sections.length > 0;
      alignHistory(left);
      if (!focus) return;
      const restored = sections.find((section) => section.dataset.group === focus.dataset.group);
      const toggle = restored?.querySelector('.aws-status__group-toggle');
      const row = [...(restored?.querySelectorAll('.aws-status__ticks') || [])]
        .find((ticks) => ticks.getAttribute('aria-label') === focusHistory);
      const tick = [...(row?.children || [])].find((candidate) => candidate.getAttribute('aria-label') === focusDate);
      (focusToggle ? toggle : tick || row?.lastElementChild || toggle)?.focus({ preventScroll: true });
    }

    function renderStatus() {
      hideTooltip();
      renderFreshness();
      const atEnd = scroll.scrollWidth - scroll.clientWidth - scroll.scrollLeft <= 2;
      const left = atEnd || readings.hidden ? null : scroll.scrollLeft;
      readings.hidden = false;
      renderHistory(left);
    }

    function positionTooltip() {
      if (!activeTick) return;
      const tick = activeTick.getBoundingClientRect();
      const clip = scroll.getBoundingClientRect();
      const stickyLabel = activeTick.parentElement.previousElementSibling?.getBoundingClientRect();
      if (tick.right <= (stickyLabel?.right ?? clip.left) || tick.left >= clip.right ||
        tick.bottom <= Math.max(0, clip.top) || tick.top >= Math.min(window.innerHeight, clip.bottom)) {
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
      const detail = detailsByTick.get(tick);
      const title = element('strong', detail.label);
      const status = element('p', detail.stateLabel, 'aws-status__tooltip-status');
      status.dataset.state = detail.state;
      const issue = detail.issue?.trim() || '';
      if (detail.state !== 'operational' && issue) status.append(' · ' + (issue.length > 100 ? issue.slice(0, 99) + '…' : issue));
      tooltip.replaceChildren(title, status);
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
      readings.addEventListener('pointerover', (event) => {
        const tick = tickFrom(event);
        if (tick && event.pointerType !== 'touch' && !pinnedTick) showTooltip(tick);
      });
      readings.addEventListener('pointerout', (event) => {
        const tick = tickFrom(event);
        if (tick && !tooltip.contains(event.relatedTarget)) queueTooltipClose(tick);
      });
      readings.addEventListener('focusin', (event) => {
        const tick = tickFrom(event);
        if (!tick) return;
        setCurrentTick(tick);
        if (pinnedTick !== tick) pinnedTick = null;
        showTooltip(tick);
      });
      readings.addEventListener('focusout', (event) => {
        const tick = tickFrom(event);
        if (tick && tick === activeTick && !tooltip.contains(event.relatedTarget)) hideTooltip();
      });
      readings.addEventListener('click', (event) => {
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
      readings.addEventListener('keydown', (event) => {
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

    installHistoryControls();
    granularity.addEventListener('change', () => {
      if (!snapshot) return;
      hideTooltip();
      renderHistory();
    });
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
