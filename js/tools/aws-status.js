(() => {
  'use strict';

  const REGION = 'us-east-2';
  const STATUS_URL = '/api/tools/aws-status';
  const HISTORY_DAYS = 90;
  const HISTORY_HOURS = 72;
  const HOUR_MS = 60 * 60 * 1000;
  const CYCLE_HOURS = 6;
  const CYCLE_MS = CYCLE_HOURS * HOUR_MS;
  const PERIOD_COUNT = 6;
  const STALE_MS = 2 * 60 * 60 * 1000;
  const REFRESH_OFFSET_MS = 21 * 60 * 1000;
  const CYCLE_GRACE_MS = 80 * 60_000;
  const REQUEST_TIMEOUT_MS = 18 * 1000;
  const GROUP_MOTION_MS = 220;
  const STATES = { operational: 'Operational', degraded: 'Degraded', outage: 'Outage', unknown: 'Unknown' };
  const CANARY_LABELS = {
    passed: 'Test passed', failed: 'Test failed', mismatch: 'Transcript mismatch',
    timeout: 'Test timed out', unavailable: 'Test unverified', running: 'Test in progress'
  };
  const HOME_GPU_STATES = { operational: 'Online', degraded: 'Worker unavailable', outage: 'Offline', unknown: 'Unknown' };
  const CATALOG_GROUPS = new Map([
    ['Tools', { prefix: 'tool:', path: '/tools/' }],
    ['Projects', { prefix: 'project:', path: '/portfolio/' }],
    ['Games', { prefix: 'game:', path: '/games/' }]
  ]);
  const SHARED_GROUP = 'Site & shared services';
  const SERVICE_CHECKS = new Map([
    ['cognito', { label: 'Cognito discovery', route: '/tools' }],
    ['tracker', { owner: 'tool:job-application-tracker', label: 'Tracker alarms', route: '/tools/job-application-tracker' }],
    ['short-links', { owner: 'tool:short-links', label: 'Short-link redirect', route: '/tools/short-links' }],
    ['contact', { label: 'Contact email alarm', route: '/contact#contact-modal' }],
    ['transcribe', { owner: 'tool:transcribe', label: 'AWS transcription', route: '/tools/transcribe' }],
    ['home-gpu', { owner: 'tool:transcribe', label: 'Home GPU readiness', route: '/tools/transcribe' }],
    ['chatbot', { owner: 'project:chatbotLora', label: 'Chatbot endpoint', route: '/chatbot-demo' }],
    ['markdown', { label: 'Markdown endpoint', route: 'https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/endpoints/VGJ-Markdown-Async' }]
  ]);
  const BROWSER_CHECK_NAMES = new Map([
    ['tool:transcribe', 'Browser file queue'],
    ['tool:short-links', 'Links workspace'],
    ['tool:job-application-tracker', 'Save and reload an application'],
    ['project:chatbotLora', 'Browser chat workflow']
  ]);
  const DATE_KEY = /^\d{4}-\d{2}-\d{2}$/;
  const HOUR_KEY = /^\d{4}-\d{2}-\d{2}T\d{2}:00:00\.000Z$/;
  const CHECK_RESULTS = {
    '-': ['unknown', 'No check recorded.'],
    u: ['unknown', 'Inconclusive check.'],
    o: ['operational', ''],
    d: ['degraded', 'Check reported an issue.'],
    x: ['outage', 'Check failed.']
  };
  const CHECK_DETAILS = {
    r: 'The built page did not load.',
    t: 'The browser check timed out waiting for the expected result.',
    m: 'The result differed from what the browser check expected.',
    e: 'The browser check could not complete.',
    n: 'The browser check could not verify this item.'
  };
  const axisDayFormat = new Intl.DateTimeFormat(undefined, { timeZone: 'UTC', month: 'short', day: 'numeric' });
  const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
    timeZone: 'UTC', month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short'
  });

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function itemRoute(group, id) {
    const catalogGroup = CATALOG_GROUPS.get(group);
    if (!catalogGroup || typeof id !== 'string' || !id.startsWith(catalogGroup.prefix)) return;
    if (id === 'tool:aws-status') return '/tools/system-status';
    const slug = id.slice(catalogGroup.prefix.length);
    if (/^[A-Za-z0-9][A-Za-z0-9-]{0,70}$/.test(slug)) return catalogGroup.path + slug;
  }

  function hasTimestamp(value) {
    return typeof value === 'string' && Number.isFinite(Date.parse(value));
  }

  function cadenceStart(data) {
    const time = Date.parse(data?.cadenceStart);
    return data?.cadenceHours === 6 && time % CYCLE_MS === 0 ? time : Infinity;
  }

  function closedBoundary(now) {
    return Math.floor(now / CYCLE_MS) * CYCLE_MS;
  }

  function completedPeriods(boundary) {
    return Array.from({ length: PERIOD_COUNT }, (_, index) => {
      const from = boundary - (PERIOD_COUNT - index) * CYCLE_MS;
      const to = from + CYCLE_MS;
      const start = new Date(from).toISOString();
      const end = new Date(to).toISOString();
      const date = start.slice(0, 10);
      const clock = start.slice(11, 16);
      const endDate = date === end.slice(0, 10) ? '' : axisDayFormat.format(to) + ', ';
      const label = axisDayFormat.format(from) + ', ' + clock + ' – ' + endDate + end.slice(11, 16) + ' UTC';
      return { from, start, date, clock, label };
    });
  }

  function renderHistoryAxis(periods) {
    const axis = element('div', '', 'aws-status__axis');
    axis.setAttribute('aria-hidden', 'true');
    const row = element('div', '', 'aws-status__axis-row');
    const columns = element('div', '', 'aws-status__axis-columns');
    const bands = element('div', '', 'aws-status__axis-bands');
    const times = element('div', '', 'aws-status__axis-times');
    periods.forEach((period, index) => {
      if (period.date !== periods[index - 1]?.date) {
        const count = periods.filter((entry) => entry.date === period.date).length;
        const band = element('div', '', 'aws-status__axis-band');
        band.style.gridColumn = 'span ' + count;
        band.dataset.start = period.date;
        band.dataset.periodCount = String(count);
        const label = element('time', axisDayFormat.format(period.from), 'aws-status__axis-label');
        label.dateTime = period.date;
        band.append(label);
        bands.append(band);
      }
      const cell = element('div', '', 'aws-status__axis-time');
      if (index === periods.length - 1) {
        cell.dataset.latest = '';
        cell.append(element('span', 'Latest', 'aws-status__axis-latest'));
      }
      const clock = element('time', period.clock);
      clock.dateTime = period.start;
      cell.append(clock);
      times.append(cell);
    });
    columns.append(bands, times);
    row.append(element('div', '', 'aws-status__axis-spacer'), columns);
    axis.append(row);
    return axis;
  }

  function legacyPeriod(hours, transcriber) {
    const outage = hours.findLast((hour) => hour?.state === 'outage');
    const degraded = hours.findLast((hour) => hour?.state === 'degraded');
    if (outage) return outage;
    if (transcriber) {
      const measured = hours.filter((hour) => hour?.transcription);
      if (measured.length) {
        // Only terminal counts are additive. A running job can cross several hour boundaries.
        const counts = { completed: 0, failed: 0, inProgress: hours.at(-1)?.transcription?.inProgress };
        for (const { transcription } of measured) {
          counts.completed += transcription.completed;
          counts.failed += transcription.failed;
        }
        if (degraded && !counts.failed) return degraded;
        const partial = measured.length < hours.length;
        const state = counts.failed ? 'degraded' : !partial && counts.completed ? 'operational' : 'unknown';
        const hadActivity = measured.some((hour) => hour.transcription.inProgress > 0);
        return { state, transcription: counts, partial, hadActivity, observedAt: measured.at(-1).observedAt };
      }
    }
    if (degraded) return degraded;
    if (hours.every((hour) => hour?.state === 'operational')) return hours.at(-1);
    const unknown = hours.findLast((hour) => hour?.state === 'unknown' && hour.issue !== 'No check recorded.');
    if (unknown) return unknown;
    const partial = hours.some((hour) => hour?.state === 'operational');
    return { state: 'unknown', issue: partial ? 'Some hourly checks are missing.' : 'No check recorded.' };
  }

  function timeLabel(value, format = dateTimeFormat) {
    const label = element('time', format.format(Date.parse(value)));
    label.dateTime = value;
    return label;
  }

  function periodPublished(start, firstCycle, publishedThrough) {
    const lastSlot = start >= firstCycle ? start : start + CYCLE_MS - HOUR_MS;
    return lastSlot <= publishedThrough;
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

  function catalogObservation(code, detail) {
    const [state, issue] = Object.hasOwn(CHECK_RESULTS, code) ? CHECK_RESULTS[code] : CHECK_RESULTS['-'];
    return { state, issue: Object.hasOwn(CHECK_DETAILS, detail) ? CHECK_DETAILS[detail] : issue };
  }

  function validStatus(data) {
    return data?.ok === true && data.region === REGION && data.days === HISTORY_DAYS &&
      (data.checkedAt === null || hasTimestamp(data.checkedAt)) &&
      Array.isArray(data.connections) && data.connections.every(validConnection) &&
      Array.isArray(data.catalog) && data.catalog.every((item) =>
        typeof item?.label === 'string' && CATALOG_GROUPS.has(item.group) && Object.hasOwn(STATES, item.state) &&
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
    const historyAnnouncement = part('history-announcement');
    const history = part('history');
    const grid = part('history-grid');
    const empty = part('history-empty');
    const detailsByTick = new WeakMap();
    const expandedGroups = new Set();
    let owner = '';
    let snapshot = null;
    let displayedBoundary = null;
    let failed = false;
    let request = null;
    let staleTimer;
    let accessTimer;
    let refreshTimer;
    let nextRefreshAt = 0;
    let selectedTick = null;

    function setNotice(message) {
      notice.textContent = message;
      notice.hidden = !message;
    }

    function setBusy(busy) {
      refresh.disabled = busy;
      refresh.textContent = busy ? 'Refreshing…' : 'Refresh';
      readings.setAttribute('aria-busy', String(busy));
    }

    function closeDetail() {
      if (!selectedTick) return;
      selectedTick.setAttribute('aria-expanded', 'false');
      const { panel } = detailsByTick.get(selectedTick);
      panel.hidden = true;
      panel.replaceChildren();
      selectedTick = null;
      if (historyAnnouncement) historyAnnouncement.textContent = '';
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
      closeDetail();
      grid.replaceChildren();
      displayedBoundary = null;
      expandedGroups.clear();
      expandedGroups.add('Tools');
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
      const allowedAge = Date.now() >= cadenceStart(snapshot) ? CYCLE_MS + CYCLE_GRACE_MS : STALE_MS;
      const stale = failed || age >= allowedAge;
      freshness.replaceChildren('Last checked ', timeLabel(snapshot.checkedAt), stale ? ' · Stale' : '');
      if (!stale) staleTimer = setTimeout(renderFreshness, allowedAge - age);
    }

    function historyPeriods(item, periods, firstCycle, publishedThrough, catalog) {
      const byHour = new Map((item.hours || []).map((hour) => [Date.parse(hour.start), hour]));
      const results = catalog?.history?.results[item.id];
      const checked = Date.parse(item.lastChecked);

      function observationAt(start) {
        if (!catalog) return byHour.get(start);
        const index = (start - catalog.hourStart) / HOUR_MS;
        const code = results?.hours[index];
        if (!code || code === '-') return;
        const detail = results?.details?.hours?.[index];
        const observation = catalogObservation(code, detail);
        observation.observedAt = catalog.history.observedAt[index];
        if (!Object.hasOwn(CHECK_DETAILS, detail) && ['u', 'd', 'x'].includes(code) && item.source === 'browser' &&
            observation.state === item.state && checked === Date.parse(observation.observedAt)) {
          observation.issue = item.issue || observation.issue;
        }
        return observation;
      }

      return periods.map(({ from, label }) => {
        if (!periodPublished(from, firstCycle, publishedThrough)) {
          return { state: 'unknown', issue: 'Awaiting the published period.', label };
        }
        const result = from >= firstCycle
          ? observationAt(from)
          : legacyPeriod(Array.from({ length: CYCLE_HOURS }, (_, hour) => observationAt(from + hour * HOUR_MS)), item.id === 'transcribe');
        return { state: 'unknown', issue: 'No check recorded.', ...result, label };
      });
    }

    function checkOutcome(id, result) {
      if (Object.hasOwn(CANARY_LABELS, result.canary)) return CANARY_LABELS[result.canary];
      if (result.transcription) {
        const { completed, failed, inProgress } = result.transcription;
        if (failed) return 'Failed';
        if (result.partial) return 'Partial data';
        if (completed) return 'Completed';
        if (inProgress) return 'In progress';
        return result.hadActivity ? 'Activity observed' : 'No activity';
      }
      if (result.state === 'unknown') {
        if (result.issue === 'Awaiting the published period.') return 'Awaiting publication';
        if (result.issue === 'No check recorded.') return 'No report';
        if (result.issue === 'Some hourly checks are missing.') return 'Partial data';
        return 'Inconclusive';
      }
      if (id === 'home-gpu') return HOME_GPU_STATES[result.state];
      return result.state === 'operational' ? 'Passed' : STATES[result.state];
    }

    function featureRows(periods, firstCycle, publishedThrough, catalog) {
      const groups = new Map([...CATALOG_GROUPS.keys(), SHARED_GROUP].map((name) => [name, []]));
      const owners = new Map();
      function check(item, label, history) {
        return { id: item.id, label, results: historyPeriods(item, periods, firstCycle, publishedThrough, history) };
      }
      for (const item of snapshot.catalog) {
        const row = {
          id: item.id, label: item.label, route: itemRoute(item.group, item.id),
          checks: [check(item, BROWSER_CHECK_NAMES.get(item.id) || 'Browser workflow', catalog)]
        };
        groups.get(item.group).push(row);
        owners.set(item.id, row);
      }
      const services = [...snapshot.connections];
      if (snapshot.homeGpu?.id === 'home-gpu' && validConnection(snapshot.homeGpu)) services.push(snapshot.homeGpu);
      for (const item of services) {
        const config = SERVICE_CHECKS.get(item.id);
        const source = check(item, config?.label || item.label);
        const owner = owners.get(config?.owner);
        if (owner) {
          if (config.route !== owner.route) source.route = config.route;
          owner.checks.push(source);
        } else {
          groups.get(SHARED_GROUP).push({
            id: item.id || item.label, label: item.label, route: config?.route, checks: [source]
          });
        }
      }
      groups.get(SHARED_GROUP).sort((a, b) => a.label.localeCompare(b.label));
      return groups;
    }

    function renderRow(row, periods, firstCycle) {
      const node = element('div', '', 'aws-status__history-row');
      node.dataset.itemId = row.id;
      const heading = element('h4', '', 'aws-status__service');
      if (row.route) {
        const link = element('a', row.label, 'aws-status__item-link');
        link.href = row.route;
        heading.append(link);
      } else {
        heading.textContent = row.label;
      }
      const panel = element('section', '', 'aws-status__period-detail');
      panel.id = 'aws-status-detail-' + encodeURIComponent(row.id);
      panel.hidden = true;
      const ticks = element('div', '', 'aws-status__ticks');
      ticks.setAttribute('role', 'group');
      ticks.setAttribute('aria-label', row.label + ' six-hour history. Use left and right arrows to choose a period. Press Enter to inspect its checks.');
      periods.forEach((period, index) => {
        const checks = row.checks.map(({ id, label, route, results }) => {
          const result = results[index];
          const checkLabel = id === 'transcribe' && period.from < firstCycle ? 'AWS transcription activity' : label;
          const activityOnly = result.state === 'unknown' && result.transcription && !result.partial;
          const visualState = activityOnly || result.canary === 'running' ? 'neutral' : result.state;
          return { label: checkLabel, route, result, visualState, outcome: checkOutcome(id, result) };
        });
        // Legacy job counts with no activity do not signal that a scheduled check was missed.
        const representative = checks.find(({ result }) => result.state === 'outage') ||
          checks.find(({ result }) => result.state === 'degraded') ||
          checks.find(({ result }) => result.state === 'unknown' && (!result.transcription || result.partial)) ||
          checks.find(({ result }) => result.state === 'operational') || checks[0];
        const { result, outcome } = representative;
        const summary = checks.length > 1 && result.state !== 'operational' && outcome !== 'Awaiting publication'
          ? representative.label + ': ' + outcome : outcome;
        const tick = element('button', '', 'aws-status__tick');
        tick.type = 'button';
        tick.tabIndex = index === periods.length - 1 ? 0 : -1;
        tick.dataset.state = representative.visualState;
        tick.dataset.period = period.start;
        if (index === periods.length - 1) tick.dataset.latest = '';
        tick.setAttribute('aria-expanded', 'false');
        tick.setAttribute('aria-controls', panel.id);
        tick.setAttribute('aria-label', row.label + ', ' + period.label + ', ' + summary);
        tick.title = period.label + ' · ' + summary;
        detailsByTick.set(tick, { panel, label: row.label, period, checks });
        ticks.append(tick);
      });
      node.append(heading, ticks, panel);
      return node;
    }

    function renderGroup(group, rows, periods, firstCycle) {
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
      summary.append(heading);
      const body = element('div', '', 'aws-status__group-items');
      body.id = 'aws-status-group-' + group.toLowerCase().replace(/[^a-z]+/g, '-');
      toggle.setAttribute('aria-controls', body.id);
      section.append(summary, body);
      let motion = null;

      function settleItems(open) {
        body.hidden = !open;
        if (!open) body.replaceChildren();
        delete body.dataset.awsStatusMoving;
      }

      function renderItems(animate = false) {
        const open = expandedGroups.has(group);
        // Read the height before canceling so reversing a toggle starts where it is now.
        const from = animate ? body.getBoundingClientRect().height : 0;
        motion?.cancel();
        motion = null;
        toggle.setAttribute('aria-expanded', String(open));
        if (!open && body.contains(document.activeElement)) toggle.focus({ preventScroll: true });
        if (!open && body.contains(selectedTick)) closeDetail();
        body.inert = !open;
        if (open) {
          body.hidden = false;
          if (!body.firstChild) body.replaceChildren(...rows.map((row) => renderRow(row, periods, firstCycle)));
        }
        const to = animate && open ? body.getBoundingClientRect().height : 0;
        if (!animate || from === to) {
          settleItems(open);
          return;
        }
        body.dataset.awsStatusMoving = '';
        const animation = body.animate([{ height: from + 'px' }, { height: to + 'px' }], {
          duration: GROUP_MOTION_MS,
          easing: 'cubic-bezier(.2, .7, .2, 1)',
          fill: 'both'
        });
        motion = animation;
        animation.onfinish = () => {
          if (motion !== animation) return;
          settleItems(open);
          animation.cancel();
          motion = null;
        };
      }
      toggle.addEventListener('click', () => {
        if (expandedGroups.has(group)) expandedGroups.delete(group);
        else expandedGroups.add(group);
        renderItems(typeof body.animate === 'function' && window.matchMedia('(prefers-reduced-motion: no-preference)').matches);
      });
      renderItems();
      return section;
    }

    function renderHistory() {
      displayedBoundary = closedBoundary(Date.now());
      const periods = completedPeriods(displayedBoundary);
      const firstCycle = cadenceStart(snapshot);
      const publishedThrough = HOUR_KEY.test(snapshot.hourlyReadyThrough) && hasTimestamp(snapshot.hourlyReadyThrough)
        ? Date.parse(snapshot.hourlyReadyThrough) : Infinity;
      grid.dataset.granularity = 'six-hour';
      const archive = catalogHistory(snapshot);
      const catalog = { history: archive, hourStart: Date.parse(archive?.hourStart) };
      const groups = featureRows(periods, firstCycle, publishedThrough, catalog);
      const sections = [...groups].filter(([, rows]) => rows.length)
        .map(([group, rows]) => renderGroup(group, rows, periods, firstCycle));

      const active = document.activeElement;
      const focusedGroup = active?.closest('.aws-status__group')?.dataset.group;
      const focusedRow = active?.closest('.aws-status__history-row')?.dataset.itemId;
      const selectedRow = selectedTick?.closest('.aws-status__history-row')?.dataset.itemId;
      const selectedPeriod = selectedTick?.dataset.period;
      closeDetail();
      grid.replaceChildren(...(sections.length ? [renderHistoryAxis(periods), ...sections] : []));
      history.hidden = sections.length === 0;
      empty.hidden = sections.length > 0;
      const rows = [...grid.querySelectorAll('.aws-status__history-row')];
      const findTick = (row, period) => [...(row?.querySelectorAll('.aws-status__tick') || [])]
        .find((tick) => tick.dataset.period === period);
      if (selectedRow) {
        const tick = findTick(rows.find((row) => row.dataset.itemId === selectedRow), selectedPeriod);
        if (tick) openDetail(tick, false);
      }
      if (!focusedGroup) return;
      const row = rows.find((candidate) => candidate.dataset.itemId === focusedRow);
      const restored = sections.find((section) => section.dataset.group === focusedGroup);
      const nextFocus = active.matches('a') ? [...(row?.querySelectorAll('a') || [])].find((link) => link.href === active.href) :
        findTick(row, active.dataset.period) || row?.querySelector('.aws-status__tick[data-latest]');
      (nextFocus || restored?.querySelector('.aws-status__group-toggle'))?.focus({ preventScroll: true });
    }

    function renderStatus() {
      renderFreshness();
      readings.hidden = false;
      renderHistory();
    }

    function selectedDetails(detail) {
      let issue = detail.issue?.trim();
      if (detail.transcription) {
        const { completed, failed, inProgress } = detail.transcription;
        issue = `Website jobs in the measured hours: ${completed} completed, ${failed} failed. `;
        issue += inProgress === undefined ? 'The queue at the end of this period is unknown.'
          : `${inProgress} queued or running at the end of this period.`;
        if (detail.partial) issue += ' Some hourly observations are missing.';
      }
      if (detail.canary && detail.previousSixHourActivity) {
        const { completed, failed, inProgress } = detail.previousSixHourActivity;
        const activity = [completed && `${completed} completed`, failed && `${failed} failed`,
          inProgress && `${inProgress} queued or running`].filter(Boolean).join(', ') || 'none';
        issue = [issue, 'Website jobs in the preceding six hours: ' + activity + '.'].filter(Boolean).join(' ');
      }
      const notes = [];
      if (issue && !['No check recorded.', 'Awaiting the published period.', 'The scheduled check passed.', 'Inconclusive check.'].includes(issue)) {
        notes.push(element('p', issue, 'aws-status__check-note'));
      }
      const observation = element('p', '', 'aws-status__check-time');
      const incident = !detail.transcription && !detail.canary && (detail.state === 'degraded' || detail.state === 'outage');
      if (incident && hasTimestamp(detail.from) && hasTimestamp(detail.to)) {
        const span = Date.parse(detail.from) !== Date.parse(detail.to);
        observation.append(span ? 'Observed bounds ' : 'Observed ', timeLabel(detail.from));
        if (span) {
          observation.append(' – ', timeLabel(detail.to));
        }
      } else if (hasTimestamp(detail.observedAt)) {
        observation.append('Last check ', timeLabel(detail.observedAt));
      }
      if (observation.hasChildNodes()) notes.push(observation);
      return notes;
    }

    function openDetail(tick, userActivated = true) {
      const { panel, label, period, checks } = detailsByTick.get(tick);
      closeDetail();
      selectedTick = tick;
      setCurrentTick(tick);
      tick.setAttribute('aria-expanded', 'true');
      const heading = element('h5', period.label, 'aws-status__period-heading');
      heading.id = panel.id + '-heading';
      panel.setAttribute('aria-labelledby', heading.id);
      const list = element('ul', '', 'aws-status__check-list');
      for (const check of checks) {
        const item = element('li');
        const name = element(check.route ? 'a' : 'span', check.label, 'aws-status__check-name');
        if (check.route) name.href = check.route;
        const outcome = element('span', check.outcome, 'aws-status__check-result');
        outcome.dataset.state = check.visualState;
        item.append(name, outcome, ...selectedDetails(check.result));
        list.append(item);
      }
      panel.replaceChildren(heading, list);
      panel.hidden = false;
      if (!userActivated) return;
      // Reveal the first outcome with the smallest scroll in the page or desktop frame.
      list.querySelector('.aws-status__check-result').scrollIntoView({
        block: 'nearest', inline: 'nearest',
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'
      });
      if (historyAnnouncement) {
        const outcomes = checks.map((check) => check.label + ', ' + check.outcome);
        historyAnnouncement.textContent = label + ', ' + period.label + '. ' + outcomes.join('. ');
      }
    }

    function setCurrentTick(tick) {
      const current = tick.parentElement.querySelector('[tabindex="0"]');
      if (current) current.tabIndex = -1;
      tick.tabIndex = 0;
    }

    function installHistoryControls() {
      grid.addEventListener('focusin', (event) => {
        if (detailsByTick.has(event.target)) setCurrentTick(event.target);
      });
      grid.addEventListener('click', (event) => {
        const tick = event.target.closest('.aws-status__tick');
        if (!detailsByTick.has(tick)) return;
        if (selectedTick === tick) closeDetail();
        else openDetail(tick);
      });
      grid.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && selectedTick) {
          if (detailsByTick.get(selectedTick).panel.contains(event.target)) selectedTick.focus({ preventScroll: true });
          closeDetail();
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        const tick = event.target;
        if (!detailsByTick.has(tick) || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
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
      });
    }

    function scheduleRefresh() {
      clearTimeout(refreshTimer);
      if (!owner || request || document.visibilityState !== 'visible') return;
      if (snapshot && displayedBoundary !== closedBoundary(Date.now())) renderStatus();
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
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const isCurrent = () => request === controller && getAccess().owner === currentOwner;
      request = controller;
      setBusy(true);
      setNotice(snapshot ? 'Refreshing system status…' : 'Checking system status…');
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
            ? 'Sign in again to view system status.'
            : 'This account does not have access to system status.');
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
          ? 'Could not refresh system status. Showing the last result.'
          : 'System status is unavailable. Try refreshing.');
      } finally {
        clearTimeout(timeout);
        if (request === controller) {
          request = null;
          setBusy(false);
          const sixHourly = Date.now() >= cadenceStart(snapshot);
          const interval = sixHourly ? CYCLE_MS : HOUR_MS;
          const offset = sixHourly ? CYCLE_GRACE_MS + 60_000 : REFRESH_OFFSET_MS;
          const now = Date.now();
          const nextReport = (Math.floor((now - offset) / interval) + 1) * interval + offset;
          nextRefreshAt = Math.min(nextReport, closedBoundary(now) + CYCLE_MS);
          if (failed) nextRefreshAt = Math.min(nextRefreshAt, now + 5 * 60_000);
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
          ? 'This account does not have access to system status.'
          : 'Sign in with an administrator account to view system status.');
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
      showAccess('Sign in with an administrator account to view system status.');
    });
    window.addEventListener('pageshow', (event) => {
      if (event.persisted) syncAccess();
    });
    syncAccess();
  }

  if (document.readyState === 'complete') init();
  else document.addEventListener('DOMContentLoaded', init, { once: true });
})();
