/* Shared report for the workspace and an individual short link. */
(() => {
  'use strict';

  const DAY = 86400000;
  const ranges = [['7', 'Last 7 days'], ['30', 'Last 30 days'], ['90', 'Last 90 days'], ['all', 'All time']];
  const dateFormat = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' });
  const fullDate = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeZone: 'UTC' });
  const numberFormat = new Intl.NumberFormat();
  const channels = [
    { key: 'qrScans', label: 'QR visits', color: '#087d87' },
    { key: 'linkClicks', label: 'Link visits', color: '#233e59' },
    { key: 'unknownClicks', label: 'Unclassified visits', color: '#856022' }
  ];

  function element(tag, className, text){
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatDate(iso, formatter = dateFormat){
    return formatter.format(new Date(iso));
  }

  function makeTable(headings, caption){
    const table = element('table', 'shortlinks-report-table');
    table.createCaption().className = 'visually-hidden';
    table.caption.textContent = caption;
    const row = table.createTHead().insertRow();
    headings.forEach(label => {
      const th = element('th', '', label);
      th.scope = 'col';
      row.append(th);
    });
    table.createTBody();
    return table;
  }

  function addPlacementTable(body, report, slug){
    const placements = (report.placements || []).slice().sort((a, b) => b.clicks - a.clicks || a.name.localeCompare(b.name));
    if (!placements.length) {
      body.append(element('p', 'shortlinks-report-note', 'No placement activity in this period.'));
      return;
    }
    const table = makeTable(['Placement', 'Channel', 'Visits'], 'Visits by placement, highest first');
    for (const placement of placements) {
      const row = table.tBodies[0].insertRow();
      const name = element('th', '', placement.name);
      name.scope = 'row';
      if (!slug && placement.slug) name.append(element('small', '', placement.slug));
      row.append(name);
      row.insertCell().textContent = ({ qr: 'QR code', link: 'Short link' })[placement.channel] || 'Unclassified';
      row.insertCell().textContent = numberFormat.format(placement.clicks);
    }
    body.append(table);
  }

  function svgElement(tag, attributes){
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function addDailyPlot(details, daily){
    const series = channels.filter(channel => channel.key !== 'unknownClicks' || daily.some(day => day.unknownClicks));
    const legend = element('div', 'shortlinks-report-legend');
    series.forEach(channel => {
      const label = element('span', '', channel.label);
      label.style.setProperty('--series-color', channel.color);
      legend.append(label);
    });

    const first = Date.parse(daily[0].date);
    const last = Date.parse(daily[daily.length - 1].date);
    const peak = daily.reduce((max, day) => Math.max(max, ...series.map(channel => day[channel.key] || 0)), 1);
    const plot = element('div', 'shortlinks-report-plot');
    const ticks = element('div', 'shortlinks-report-ticks');
    ticks.append(element('span', '', numberFormat.format(peak)), element('span', '', '0'));
    const canvas = element('div');
    const svg = svgElement('svg', { viewBox: '0 0 800 96', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    svg.append(svgElement('path', { d: 'M0 4H800 M0 48H800 M0 92H800', stroke: '#dce4e8', fill: 'none' }));
    const x = time => last === first ? 400 : 2 + (time - first) * 796 / (last - first);
    const y = visits => 92 - 88 * visits / peak;
    for (const channel of series) {
      const points = [];
      let previous;
      for (const day of daily) {
        const time = Date.parse(day.date);
        // All-time data omits zero days. Bring each series to zero across gaps.
        if (previous !== undefined && time - previous > DAY) {
          points.push(x(previous + DAY) + ',92', x(time - DAY) + ',92');
        }
        points.push(x(time) + ',' + y(day[channel.key] || 0));
        previous = time;
      }
      if (first === last) {
        svg.append(svgElement('circle', { cx: 400, cy: y(daily[0][channel.key] || 0), r: 3, fill: channel.color }));
      } else {
        svg.append(svgElement('polyline', { points: points.join(' '), stroke: channel.color, fill: 'none', 'stroke-width': 2.25, 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke' }));
      }
    }
    const dates = element('div', 'shortlinks-report-dates');
    dates.append(element('span', '', formatDate(first)));
    if (last !== first) dates.append(element('span', '', formatDate(last)));
    canvas.append(svg, dates);
    plot.append(ticks, canvas);
    details.append(legend, plot);
  }

  function addTrend(body, report, expanded){
    const details = element('details', 'shortlinks-report-trend');
    details.open = expanded;
    details.append(element('summary', '', 'Daily trend'));
    const previous = report.previous;
    if (previous) {
      const period = formatDate(previous.start) + ' – ' + formatDate(previous.end);
      const coverage = previous.complete === false ? ' (partial history)' : '';
      details.append(element('p', 'shortlinks-report-note', 'Previous ' + report.days + ' days (' + period + ') · ' + numberFormat.format(previous.clicks) + ' recorded visits' + coverage));
    }
    const daily = report.daily || [];
    if (daily.some(day => day.clicks)) {
      addDailyPlot(details, daily);
      const disclosure = element('details', 'shortlinks-report-breakdown');
      disclosure.append(element('summary', '', 'Daily breakdown'));
      const table = makeTable(['Date', 'QR', 'Link', 'Other', 'Visits'], 'Recorded days, most recent first');
      for (const day of daily.filter(day => day.clicks).reverse()) {
        const row = table.tBodies[0].insertRow();
        const label = element('th', '', formatDate(day.date, fullDate));
        label.scope = 'row';
        row.append(label);
        [day.qrScans, day.linkClicks, day.unknownClicks, day.clicks].forEach(value => {
          row.insertCell().textContent = numberFormat.format(value);
        });
      }
      disclosure.append(table);
      details.append(disclosure);
    } else {
      details.append(element('p', 'shortlinks-report-note', 'No dated visits in this period.'));
    }
    body.append(details);
  }

  function mount(host, { slug = '', load }){
    const previousRange = host.querySelector('select[data-activity-range]')?.value || '30';
    const root = element('div', 'shortlinks-report');
    const top = element('div', 'shortlinks-report-top');
    const figures = element('div', 'shortlinks-report-figures');
    const picker = element('select', 'shortlinks-report-range');
    picker.dataset.activityRange = '';
    picker.setAttribute('aria-label', 'Activity date range');
    ranges.forEach(([value, label]) => picker.add(new Option(label, value)));
    picker.value = previousRange;
    top.append(figures, picker);
    const status = element('p', 'shortlinks-report-status');
    status.setAttribute('role', 'status');
    const body = element('div', 'shortlinks-report-body');
    root.append(top, status, body);
    host.replaceChildren(root);
    let sequence = 0;

    async function refresh(){
      const request = ++sequence;
      const range = picker.value;
      const expanded = !!body.querySelector('.shortlinks-report-trend')?.open;
      body.replaceChildren();
      figures.replaceChildren();
      status.textContent = 'Loading activity…';
      try {
        const report = await load(range);
        if (!report?.totals) throw new Error('Activity is unavailable.');
        if (request !== sequence || !root.isConnected) return;
        const totals = report.totals;
        const total = element('p', 'shortlinks-report-total');
        total.append(element('strong', '', numberFormat.format(totals.clicks)), document.createTextNode(totals.clicks === 1 ? ' visit' : ' visits'));
        const breakdown = [numberFormat.format(totals.qrScans) + ' QR visits', numberFormat.format(totals.linkClicks) + ' link visits'];
        if (totals.unknownClicks) breakdown.push(numberFormat.format(totals.unknownClicks) + ' unclassified');
        figures.append(total, element('p', 'shortlinks-report-split', breakdown.join(' · ')));
        // Retain the range name in the menu and expose the actual dates in its closed state.
        for (const option of picker.options) option.text = ranges.find(([value]) => value === option.value)[1];
        if (range !== 'all' && report.daily?.length) {
          const daily = report.daily;
          picker.selectedOptions[0].text = formatDate(daily[0].date) + ' – ' + formatDate(daily[daily.length - 1].date) + ' (' + range + ' days)';
        }
        addPlacementTable(body, report, slug);
        addTrend(body, report, expanded);
        if (report.completeness?.note) {
          if (report.completeness.complete) {
            body.querySelector('.shortlinks-report-trend').append(element('p', 'shortlinks-report-note', report.completeness.note));
          } else {
            body.prepend(element('p', 'shortlinks-report-note shortlinks-report-coverage', report.completeness.note));
          }
        }
        status.textContent = '';
      } catch (error) {
        if (request !== sequence || !root.isConnected) return;
        status.textContent = error.message || 'Activity could not be loaded.';
        const retry = element('button', 'btn-secondary', 'Retry');
        retry.type = 'button';
        retry.addEventListener('click', refresh);
        body.replaceChildren(retry);
      }
    }
    picker.addEventListener('change', refresh);
    return refresh();
  }

  window.ShortLinksActivity = Object.freeze({ mount });
})();
