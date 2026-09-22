/* Quiet, persistent traveler lanes. One inline evidence view; totals from accepted records. */
(function () {
  'use strict';
  const main = document.querySelector('#main[data-count]');
  if (!main) return;
  const one = (selector) => document.querySelector(selector);
  const all = (selector) => [...document.querySelectorAll(selector)];
  const core = window.AdVerificationCore;
  const api = window.AdVerificationPlayer;
  const narrow = window.matchMedia('(max-width:760px)');
  const reduced = window.matchMedia('(prefers-reduced-motion:reduce)');
  const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  const icons = {
    ad: '<path d="M3 9h5l11-5v16L8 15H3Zm5 6 2 6h4l-2-4M22 9v6"/>',
    website: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18M5 6h14M5 18h14"/>',
    destination: '<path d="M12 22S4 14 4 9a8 8 0 0 1 16 0c0 5-8 13-8 13Z" fill="currentColor" stroke="none"/><circle cx="12" cy="9" r="2.5" fill="white" stroke="none"/>',
    people: '<circle cx="8" cy="7" r="3"/><path d="M2 21v-3a6 6 0 0 1 12 0v3Zm13-17a3 3 0 0 1 0 6m2 3a5 5 0 0 1 5 5v3"/>',
    chain: '<path d="m10 14 4-4m-7 6-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m4 1 2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0" transform="translate(1 -1)"/>',
    check: '<circle cx="12" cy="12" r="10" fill="currentColor" stroke="none"/><path d="m7 12 3 3 7-7" stroke="white" stroke-width="2"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 3"/>',
    document: '<path d="M6 2h8l5 5v15H6Zm8 0v6h5M9 12h7m-7 4h7"/>',
    play: '<path d="m8 4 12 8-12 8Z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M8 5v14m8-14v14" stroke-width="5"/>',
    arrow: '<path d="M3 12h17m-5-5 5 5-5 5"/>'
  };
  const svg = (type) => `<svg class="av-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[type] || icons.document}</svg>`;
  all('[data-icon]').forEach((node) => { node.innerHTML = svg(node.dataset.icon); });
  one('[data-play-icon]').innerHTML = svg('play');
  const types = ['ad', 'website', 'destination'];
  const labels = ['Outdoor enthusiast', 'Weekend traveler', 'Couple getaway', 'Event attendee', 'Casual browser'];
  const personName = (id) => id ? 'Traveler ' + Number(id.slice(1)) : 'Shared campaign';
  const eventOf = (block) => block.transactions[0].event;
  const titleOf = (event) => event.type === 'attribution' ? event.data.credited ? 'Visit attributed' : 'Visit not attributed' : core.TYPES[event.type].title;
  const iconOf = (type) => ['destination', 'attribution'].includes(type) ? 'destination' : types.includes(type) ? type : 'document';
  const dialog = one('[data-dialog]');
  const historyDialog = one('[data-history-dialog]');
  let player;
  let state;
  let proof = null;
  let working = [];
  let report = null;
  let selected = null;
  let inspected = null;
  let pinned = false;
  let highlight = null;
  let checking = false;
  let tampered = false;
  let generation = 0;
  let announcedFirst = false;
  const announce = (text) => { one('[data-announcement]').textContent = text; };
  const statusOf = (index) => report?.blocks[index]?.state || 'verified';
  one('[data-travelers]').innerHTML = Array.from({ length: 5 }, (_, slot) => `<article class="av-traveler" data-slot="${slot}"><button type="button" class="av-person" data-person disabled><span class="av-avatar" aria-hidden="true"></span><span><strong data-name>Traveler ${slot + 1}</strong><small>${labels[slot]}</small></span></button><div class="av-path">${types.map((type) => `<button type="button" class="av-step" data-step="${type}" data-state="waiting" disabled><span class="av-stop">${svg(type)}</span><span class="av-sr-only" data-step-label>Waiting</span></button>`).join('')}<i class="av-dot" aria-hidden="true"></i></div><button type="button" class="av-outcome" data-outcome disabled><span data-outcome-icon>${svg('clock')}</span><span><strong data-outcome-title>Waiting</strong><small data-lane-status>Campaign not started</small></span></button></article>`).join('');
  const rows = all('[data-slot]').map((row) => ({ row, person: row.querySelector('[data-person]'), name: row.querySelector('[data-name]'), avatar: row.querySelector('.av-avatar'), outcome: row.querySelector('[data-outcome]'), status: row.querySelector('[data-lane-status]'), outcomeTitle: row.querySelector('[data-outcome-title]'), outcomeIcon: row.querySelector('[data-outcome-icon]'), steps: types.map((type) => row.querySelector(`[data-step="${type}"]`)) }));
  if (!core || !api) {
    main.dataset.error = 'true'; one('[data-status]').textContent = 'Unavailable';
    one('[data-message]').textContent = 'The verification engine did not load. Reload this page over HTTPS to retry.';
    return;
  }
  function paintLanes() {
    for (const lane of state.lanes) {
      const view = rows[lane.slot];
      const row = view.row;
      row.dataset.travelerId = lane.id || '';
      row.dataset.summaryHeight = String(lane.records.summary?.height || '');
      row.dataset.summaryKey = lane.records.summary?.key || '';
      row.dataset.attributionHeight = String(lane.records.attribution?.height || '');
      row.dataset.attributionKey = lane.records.attribution?.key || '';
      row.dataset.phase = lane.phase; row.dataset.key = lane.id ? lane.key : '';
      row.dataset.progress = lane.progress.toFixed(5);
      row.style.setProperty('--progress', lane.progress.toFixed(5));
      row.style.setProperty('--opacity', reduced.matches ? '1' : String(lane.opacity ?? 1));
      row.dataset.highlight = String(Boolean(lane.id && lane.id === highlight));
      row.dataset.featured = String(Boolean(state.writer?.travelerId && state.writer.travelerId === lane.id));
      view.name.textContent = personName(lane.id || 'T' + (lane.slot + 1));
      view.person.disabled = !lane.records.ad;
      view.person.setAttribute('aria-label', `Show records for ${view.name.textContent}`);
      view.avatar.style.setProperty('--avatar', String(lane.id ? (lane.number - 1) % 5 : lane.slot));
      for (let index = 0; index < types.length; index += 1) {
        const type = types[index]; const node = view.steps[index]; const record = lane.records[type];
        const active = type === lane.type && ['recording', 'queued', 'verifying'].includes(lane.phase);
        const status = record ? statusOf(record.height - 1) : active ? lane.phase : lane.phase === 'done' ? 'skipped' : 'waiting';
        node.dataset.state = status; node.dataset.key = record ? record.key : active ? lane.key : '';
        node.dataset.height = record ? String(record.height) : ''; node.dataset.phase = record ? 'committed' : active ? lane.phase : status;
        node.disabled = !record;
        const label = record ? status === 'verified' ? 'recorded in block ' + record.height : status : status === 'skipped' ? 'not recorded' : active ? 'in progress' : 'waiting';
        node.querySelector('[data-step-label]').textContent = label;
        node.setAttribute('aria-label', `${view.name.textContent}, ${core.TYPES[type].title}, ${label}`);
        row.style.setProperty('--path-' + type, record ? '1' : '0');
      }
      const attributionIndex = lane.records.attribution ? lane.records.attribution.height - 1 : -1;
      const attribution = attributionIndex >= 0 ? eventOf(working[attributionIndex]).data : null;
      const changed = Object.values(lane.records).some((record) => statusOf(record.height - 1) !== 'verified');
      let title = lane.id ? 'In progress' : 'Waiting';
      let detail = lane.records.website ? 'Website recorded' : lane.records.ad ? 'Ad recorded' : 'Waiting for ad';
      let outcome = 'neutral';
      if (lane.type === 'attribution' && !attribution) { title = 'Matching visit'; detail = 'Checking earlier ad'; outcome = 'active'; }
      else if (lane.records.destination && !attribution) { title = 'Visit reported'; detail = 'Awaiting attribution'; }
      if (attribution) { title = attribution.credited ? 'Attributed' : 'Not attributed'; detail = attribution.credited ? `${attribution.elapsedDays} days after ad` : 'Outside demo window'; outcome = attribution.credited ? 'credited' : 'neutral'; }
      else if (lane.phase === 'done') { title = lane.records.website ? 'Website only' : 'No visit recorded'; detail = 'Measurement ended'; }
      if (changed) { title = 'Record changed'; detail = 'Edited copy rejected'; outcome = 'changed'; }
      view.outcome.dataset.state = outcome;
      view.outcome.dataset.key = lane.records.attribution?.key || '';
      view.outcome.dataset.height = String(lane.records.attribution?.height || '');
      view.outcomeTitle.textContent = title; view.status.textContent = detail;
      const desiredIcon = outcome === 'credited' ? 'check' : 'clock';
      if (view.outcomeIcon.dataset.icon !== desiredIcon) { view.outcomeIcon.innerHTML = svg(desiredIcon); view.outcomeIcon.dataset.icon = desiredIcon; }
      view.outcome.disabled = !attribution;
      view.outcome.setAttribute('aria-label', `${view.name.textContent}: ${title}. ${attribution ? 'Show attribution evidence.' : detail}`);
      // The lanes still overlap, but only the writer's lane has a prominent recording accent.
      const target = types.indexOf(lane.type);
      row.dataset.moving = String(lane.phase === 'recording' && target >= 0);
      if (lane.phase === 'recording' && target >= 0) {
        const from = target === 0 ? -.4 : target === 1 || !lane.records.website ? 0 : 1;
        row.style.setProperty('--runner', `${16.6667 + (from + (target - from) * lane.progress) * 33.3333}%`);
      }
    }
  }
  function evidenceMarkup(block, index) {
    const event = eventOf(block); const status = statusOf(index);
    if (status !== 'verified') return `<div class="av-inline-evidence"><h3>Edited history does not verify</h3><p>The original evidence and campaign totals are unchanged.</p><button class="av-link" type="button" data-inspect="${index}">Inspect the mismatch</button></div>`;
    if (event.type !== 'attribution') return `<div class="av-inline-evidence"><h3>${escape(core.TYPES[event.type].source)} reported this event</h3><p>This record is signed and links to ${index ? 'block #' + index : 'the start of the campaign'}. Verification checks its recorded contents, not the truth of the claim.</p><button type="button" class="av-link" data-inspect="${index}">Inspect or try an edit</button></div>`;
    const data = event.data;
    const refs = `data-support="${index}"`;
    return `<div class="av-inline-evidence" data-evidence-for="${index + 1}" data-credited="${data.credited}"><h3>${data.credited ? 'Why this visit was counted' : 'Why this visit was not counted'}</h3><p class="av-check-line">${svg('check')}<span>Earlier ad matched to the same example traveler.</span></p><p class="av-check-line" data-pass="${data.withinWindow}">${svg(data.withinWindow ? 'check' : 'clock')}<span>${data.elapsedDays} days after the ad · ${data.withinWindow ? 'within' : 'outside'} the 30-day demo rule.</span></p><p class="av-check-line">${svg('check')}<span>This visit was not previously counted.</span></p><div class="av-evidence-actions"><button type="button" class="av-link" ${refs}>Supporting records ${svg('arrow')}</button><button type="button" class="av-link" data-inspect="${index}">Try an edit</button></div><span class="av-evidence-note">A website visit is optional. Attribution is not proof of causation.</span></div>`;
  }
  function blockMarkup(block, index, archive = false) {
    const event = eventOf(block); const status = statusOf(index);
    const isSelected = !archive && index === selected;
    const credited = event.type === 'attribution' && event.data.credited && status === 'verified';
    const detail = event.type === 'website' ? 'Viewed: ' + event.data.page : event.type === 'destination' ? 'Location: ' + event.data.place : event.type === 'attribution' ? event.data.credited ? 'Matched to campaign' : 'Outside attribution window' : 'Source: ' + core.TYPES[event.type].source;
    return `<li class="av-block" data-height="${index + 1}" data-key="${event.key}" data-traveler-id="${event.travelerId || ''}" data-type="${event.type}" data-state="${status}" data-credited="${credited}" data-selected="${isSelected}" data-highlight="${Boolean(highlight && highlight === event.travelerId)}"><span class="av-chain-marker">${svg(iconOf(event.type))}</span><div class="av-record"><button type="button" class="av-block-button" ${archive ? 'data-inspect' : 'data-select'}="${index}" ${archive ? '' : `aria-expanded="${isSelected}"`} aria-label="${archive ? 'Inspect' : 'Expand'} block ${index + 1}, ${escape(titleOf(event))}, ${personName(event.travelerId)}"><span class="av-block-number">#${index + 1}</span><span class="av-block-description"><strong>${escape(titleOf(event))}</strong><small>${event.data.exampleDay !== undefined ? 'Example day ' + event.data.exampleDay : core.TYPES[event.type].source}</small></span><span class="av-block-person"><strong>${personName(event.travelerId)}</strong><small>${escape(detail)}</small></span><span class="av-record-status">${status === 'verified' ? svg('check') + '<span>Verified</span>' : status === 'changed' ? '× Changed' : '! Prior change'}</span></button>${isSelected ? evidenceMarkup(block, index) : ''}</div></li>`;
  }
  function renderChain() {
    const indexes = Array.from({ length: Math.min(4, working.length) }, (_, offset) => working.length - Math.min(4, working.length) + offset);
    if (selected !== null && working[selected] && !indexes.includes(selected)) { indexes.shift(); indexes.push(selected); indexes.sort((a, b) => a - b); }
    let previous = null;
    one('[data-blocks]').innerHTML = indexes.map((index) => {
      const gap = previous !== null && index > previous + 1 ? `<li class="av-gap">${index - previous - 1} intervening blocks in full history</li>` : '';
      previous = index;
      return gap + blockMarkup(working[index], index);
    }).join('');
    one('[data-empty]').hidden = working.length > 0;
    one('[data-count-label]').textContent = working.length + ' blocks';
    one('[data-history-note]').textContent = working.length > 4 ? `${working.length} total · recent and selected records` : 'Every block links to the one before it.';
    one('[data-warning]').hidden = !tampered;
  }
  function paintWriter() {
    const writer = state.writer; const node = one('[data-writer]');
    node.dataset.key = writer?.key || ''; node.dataset.phase = writer ? 'verifying' : 'idle';
    node.dataset.progress = (writer?.progress || 0).toFixed(5); node.dataset.height = writer ? String(writer.height) : '';
    node.style.setProperty('--progress', String(writer?.progress || 0));
    one('[data-writer-title]').textContent = writer ? `${personName(writer.travelerId)} · ${core.TYPES[writer.type].title} → verifying #${writer.height}` : state.started ? 'Every new block references the previous block.' : 'Ready to record the campaign';
  }
  function renderTotals() {
    // Authentic totals never read from the editable working copy.
    const totals = proof ? core.campaignTotals(proof.blocks) : { exposures: 0, websiteVisits: 0, attributedVisits: 0 };
    for (const key of ['exposures', 'websiteVisits', 'attributedVisits']) one(`[data-metric="${key}"]`).textContent = String(totals[key]);
    one('[data-results-note]').innerHTML = tampered ? 'Original results retained.<br>Your edited copy was rejected.' : 'Simulated results.<br>Calculated from accepted records.';
    one('[data-results-state]').dataset.resultsState = tampered ? 'edited-copy-rejected' : 'original';
    one('[data-result-evidence]').disabled = !proof?.blocks.some((block) => eventOf(block).type === 'attribution') || checking;
  }
  function controls() {
    const drained = state.capacityClosed && state.lanes.every((lane) => !lane.id || lane.phase === 'done');
    const disabled = !state.ready || checking || tampered || Boolean(state.error) || drained;
    one('[data-play]').disabled = disabled;
    one('[data-play-label]').textContent = state.running ? 'Pause campaign' : state.started ? 'Resume campaign' : 'Start campaign';
    const desiredIcon = state.running ? 'pause' : 'play';
    if (one('[data-play-icon]').dataset.icon !== desiredIcon) { one('[data-play-icon]').innerHTML = svg(desiredIcon); one('[data-play-icon]').dataset.icon = desiredIcon; }
    all('[data-history],[data-verify],[data-export]').forEach((node) => { node.disabled = !working.length || checking; });
    all('[data-restore]').forEach((node) => { node.disabled = checking; });
    one('[data-edit-form] button[type="submit"]').disabled = checking;
    one('[data-status]').textContent = state.error ? 'Unavailable' : tampered ? 'Edited copy · paused' : state.running ? 'Live simulation' : !state.ready ? 'Preparing…' : state.started ? drained ? 'Complete' : 'Paused' : 'Ready';
    one('[data-status]').dataset.running = String(state.running);
    one('[data-metric="active"]').textContent = String(state.lanes.filter((lane) => lane.id && lane.phase !== 'done').length);
    main.dataset.running = String(state.running); main.dataset.count = String(state.count); main.dataset.time = state.time.toFixed(3);
    main.dataset.measuring = String(state.lanes.filter((lane) => lane.phase === 'recording').length);
    main.dataset.admitted = String(state.admitted); main.dataset.completed = String(state.completed); main.dataset.queued = String(state.queue.length);
  }
  function update(change, next) {
    state = next;
    if (change.kind === 'reset') {
      generation += 1; proof = null; working = []; report = null; checking = false; tampered = false;
      selected = null; inspected = null; pinned = false; highlight = null; announcedFirst = false;
      if (dialog.open) dialog.close(); if (historyDialog.open) historyDialog.close();
      one('[data-message]').textContent = 'Accelerated simulation. A website visit is optional; attribution is not proof that an ad caused a trip.';
    }
    if (state.proof) proof = state.proof;
    if (change.block) {
      working.push(core.clone(change.block)); report = null;
      const event = eventOf(change.block); main.dataset.lastKey = event.key;
      if (event.type === 'attribution' && !pinned) selected = working.length - 1;
      if (event.type === 'attribution') announce(`${personName(event.travelerId)}: ${event.data.credited ? 'visit attributed' : 'visit outside the demo window, not credited'}. Block ${state.count} recorded.`);
      else if (!announcedFirst) { announce('The first signed block is recorded. Travelers now join the campaign.'); announcedFirst = true; }
    }
    if (['reset', 'ready', 'commit'].includes(change.kind)) { renderChain(); renderTotals(); }
    paintLanes(); paintWriter(); controls();
    if (state.error) { main.dataset.error = 'true'; one('[data-message]').textContent = 'Verification unavailable: ' + state.error + ' Reset to retry.'; }
    else { delete main.dataset.error; if (state.capacityClosed) one('[data-message]').textContent = 'Demo limit: new arrivals stopped. Current travelers finish before the chain stops. Reset for a new campaign.'; }
  }
  player = api.createPlayer({ onChange: update });
  one('[data-play]').addEventListener('click', () => {
    if (checking || tampered) return;
    if (state.running) player.pause();
    else { pinned = false; player.play(); if (narrow.matches) one('.av-workspace').scrollIntoView({ block: 'start', behavior: 'instant' }); }
  });
  one('[data-reset]').addEventListener('click', () => player.reset());
  one('[data-scenario]').addEventListener('change', (event) => { player.setScenario(event.target.value); one('[data-message]').textContent = 'New arrivals use ' + core.SCENARIOS[event.target.value].toLowerCase() + '. Current travelers keep their own paths.'; });
  one('[data-speed]').addEventListener('change', (event) => player.setSpeed(Number(event.target.value)));
  one('[data-continuous]').addEventListener('change', (event) => player.setContinuous(event.target.checked));
  document.addEventListener('visibilitychange', () => { if (document.hidden) player.pause(); });
  window.addEventListener('pagehide', () => player.pause());
  main.addEventListener('focusin', (event) => { if (event.target.closest('[data-person],[data-step],[data-outcome],[data-select],[data-inspect],[data-support]')) player.pause(); });
  function selectBlock(index) {
    if (checking || !working[index]) return;
    player.pause(); selected = index; pinned = true; highlight = eventOf(working[index]).travelerId;
    renderChain(); paintLanes();
    // Keep keyboard focus on the replacement selector, without stealing it on automatic updates.
    one(`[data-blocks] [data-select="${index}"]`)?.focus({ preventScroll: true });
    if (narrow.matches) one('.av-ledger').scrollIntoView({ block: 'start', behavior: 'instant' });
  }
  one('[data-travelers]').addEventListener('click', (event) => {
    const record = event.target.closest('[data-step],[data-outcome]');
    if (record?.dataset.height) { selectBlock(Number(record.dataset.height) - 1); return; }
    const person = event.target.closest('[data-person]');
    if (!person) return;
    const id = person.closest('[data-slot]').dataset.travelerId;
    const index = working.findLastIndex((block) => eventOf(block).travelerId === id && eventOf(block).type === 'attribution');
    selectBlock(index >= 0 ? index : working.findLastIndex((block) => eventOf(block).travelerId === id));
  });
  one('[data-blocks]').addEventListener('click', (event) => {
    const select = event.target.closest('[data-select]');
    if (select) { selectBlock(Number(select.dataset.select)); return; }
    const edit = event.target.closest('[data-inspect]'); if (edit) { inspect(Number(edit.dataset.inspect)); return; }
    const support = event.target.closest('[data-support]'); if (support) showSupporting(Number(support.dataset.support));
  });
  one('[data-result-evidence]').addEventListener('click', () => {
    const last = working.findLastIndex((block) => eventOf(block).type === 'attribution' && eventOf(block).data.credited);
    const fallback = working.findLastIndex((block) => eventOf(block).type === 'attribution');
    selectBlock(last >= 0 ? last : fallback);
  });
  async function recheck() {
    if (!proof || checking) return false;
    player.pause(); checking = true; controls();
    const token = generation;
    try {
      const result = await core.verifyResults(working, proof.trust);
      if (token !== generation) return false;
      report = result.report; tampered = !report.valid;
      paintLanes(); renderChain(); renderTotals();
      announce(report.valid ? 'All recorded evidence and attribution decisions passed verification.' : 'The edited copy failed verification. Original totals are retained.');
      return true;
    } catch (_) {
      if (token === generation) { tampered = true; renderTotals(); renderChain(); one('[data-message]').textContent = 'Verification could not complete. Restore the original records or reset to retry.'; }
      return false;
    } finally { if (token === generation) { checking = false; controls(); renderTotals(); } }
  }
  one('[data-verify]').addEventListener('click', async () => { if (await recheck()) one('[data-message]').textContent = report.valid ? `All ${working.length} records checked. Attribution decisions and campaign totals match the evidence.` : 'Your edited copy was rejected. The original results are unchanged.'; });
  function showHistory() {
    if (checking || !working.length) return;
    player.pause(); one('[data-archive-caption]').textContent = 'One campaign, all traveler paths, in recorded order. No records are removed from this history.';
    one('[data-archive]').innerHTML = working.map((block, index) => blockMarkup(block, index, true)).join('');
    historyDialog.showModal();
  }
  function showSupporting(index) {
    if (checking || !working[index] || eventOf(working[index]).type !== 'attribution') return;
    player.pause(); const event = eventOf(working[index]);
    const indexes = [event.data.exposure.blockHeight - 1, event.data.visit.blockHeight - 1, index];
    one('[data-archive-caption]').textContent = 'Selected evidence from the same campaign chain. Other travelers’ records can sit between these block numbers. Website activity is not required.';
    one('[data-archive]').innerHTML = indexes.map((i) => blockMarkup(working[i], i, true)).join(''); historyDialog.showModal();
  }
  one('[data-history]').addEventListener('click', showHistory);
  one('[data-close-history]').addEventListener('click', () => historyDialog.close());
  one('[data-archive]').addEventListener('click', (event) => { const node = event.target.closest('[data-inspect]'); if (node) inspect(Number(node.dataset.inspect)); });
  const fieldLabel = (key) => ({ credited: 'Credit this campaign', elapsedDays: 'Days after ad', withinWindow: 'Inside matching window', notPreviouslyCounted: 'Not previously counted', websiteRecorded: 'Website recorded', destinationRecorded: 'Destination recorded', impressions: 'Impressions', page: 'Website page', name: 'Campaign name', exampleDay: 'Example occurrence day', place: 'Destination place' })[key] || key;
  function editField() {
    if (inspected === null || !working[inspected]) return;
    const value = eventOf(working[inspected]).data[one('[data-edit-field]').value]; const boolean = typeof value === 'boolean';
    one('[data-edit-value]').hidden = boolean; one('[data-edit-value]').required = !boolean; one('[data-edit-boolean]').hidden = !boolean;
    one('[data-edit-value]').value = String(value); one('[data-edit-boolean]').value = String(value);
  }
  function fillInspector() {
    const block = working[inspected]; if (!block || !report) return;
    const event = eventOf(block); const row = report.blocks[inspected];
    one('#av-dialog-title').textContent = `Block #${inspected + 1} · ${titleOf(event)}`;
    one('[data-dialog-subtitle]').textContent = personName(event.travelerId) + ' · ' + event.campaignId;
    const printable = (value) => value && typeof value === 'object' ? 'Block #' + value.blockHeight + ' · ' + value.eventId : String(value);
    const fields = [['Previous block', inspected ? '#' + inspected : 'None (first block)'], ['Source', core.TYPES[event.type].source], ...Object.entries(event.data).filter(([key]) => !['synthetic', 'note'].includes(key)).map(([key, value]) => [fieldLabel(key), printable(value)])];
    one('[data-fields]').innerHTML = fields.map(([label, value]) => `<dt>${escape(label)}</dt><dd>${escape(value)}</dd>`).join('');
    const keys = Object.keys(event.data).filter((key) => !['synthetic', 'note'].includes(key) && ['string', 'number', 'boolean'].includes(typeof event.data[key]));
    one('[data-edit-field]').innerHTML = keys.map((key) => `<option value="${escape(key)}">${escape(fieldLabel(key))}</option>`).join('');
    const preferred = keys.find((key) => ['credited', 'impressions', 'page', 'place', 'name', 'websiteRecorded'].includes(key)); if (preferred) one('[data-edit-field]').value = preferred;
    editField();
    one('[data-verdict]').dataset.valid = String(row.state === 'verified');
    one('[data-verdict]').textContent = row.state === 'verified' ? 'This record and its applicable rule checks match.' : row.state === 'changed' ? 'Verification failed. Your edited copy no longer matches its original evidence.' : 'This record is unchanged, but depends on earlier changed history.';
    one('[data-previous-hash]').textContent = block.header.previousHash; one('[data-recorded-hash]').textContent = block.hash;
    one('[data-computed-hash]').textContent = row.computedHash || 'Unable to calculate';
    const names = { structure: 'Record structure', journey: 'Traveler path', attribution: 'Attribution rule and evidence', eventSignature: 'Event signature', merkleRoot: 'Record fingerprint', blockHash: 'Recorded block hash', previousHash: 'Previous block link', approvals: 'Three local approvals' };
    one('[data-checks]').innerHTML = Object.entries(row.checks).map(([key, valid]) => `<li>${valid ? '✓' : '×'} ${names[key]}: ${valid ? 'passes' : 'fails'}</li>`).join('');
    one('[data-json]').textContent = JSON.stringify(block, null, 2);
  }
  async function inspect(index) {
    if (checking || !working[index]) return;
    player.pause(); inspected = index; if (historyDialog.open) historyDialog.close();
    if (await recheck()) { fillInspector(); dialog.showModal(); }
  }
  one('[data-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    const target = one(`[data-select="${inspected}"]`) || one('[data-history]');
    if (!target.disabled) target.focus({ preventScroll: true });
  });
  one('[data-edit-field]').addEventListener('change', editField);
  one('[data-edit-form]').addEventListener('submit', async (event) => {
    event.preventDefault(); if (checking || inspected === null) return;
    const data = eventOf(working[inspected]).data; const key = one('[data-edit-field]').value; const prior = data[key];
    let value = typeof prior === 'boolean' ? one('[data-edit-boolean]').value === 'true' : one('[data-edit-value]').value;
    if (typeof prior === 'number') {
      value = Number(value);
      if (!Number.isFinite(value) || Math.abs(value) > 1e12) { one('[data-verdict]').textContent = 'Enter a finite number between -1,000,000,000,000 and 1,000,000,000,000.'; return; }
    }
    data[key] = value; if (await recheck()) fillInspector();
  });
  all('[data-restore]').forEach((button) => button.addEventListener('click', async () => {
    if (!proof || checking) return; working = core.clone(proof.blocks);
    if (await recheck()) { if (dialog.open) fillInspector(); one('[data-message]').textContent = 'Exact original signed records restored. Resume to continue this campaign.'; }
  }));
  one('[data-export]').addEventListener('click', () => {
    if (!proof || checking) return;
    player.pause();
    const url = URL.createObjectURL(new Blob([JSON.stringify({ schema: 'ad-verification-demo/v4', notice: 'Fictional observations, an example attribution rule and same-browser signers. Not independent evidence of visitation or causation.', blocks: working, trust: proof.trust }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'cedar-valley-campaign-proof.json'; document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000); announce('Public proof exported. No private keys are included.');
  });
  player.reset();
})();
