/* Five persistent lane views and a compact view of ONE append-only chain. */
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
    ad: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 8h18m-12 5 5-2v6l-5-2Z"/>',
    website: '<rect x="3" y="3" width="18" height="14" rx="2"/><path d="M1 21h22m-16-4-1 4m11-4 1 4"/>',
    destination: '<path d="M12 22S4 14 4 9a8 8 0 0 1 16 0c0 5-8 13-8 13Z"/><circle cx="12" cy="9" r="3"/>'
  };
  const types = ['ad', 'website', 'destination'];
  const shortTitle = { campaign: 'Campaign started', purchase: 'Ad space purchased', ad: 'Ad served', website: 'Website visit', destination: 'Visit attributed', summary: 'Measurement ended' };
  const personName = (id) => id ? 'Traveler ' + Number(id.slice(1)) : 'Shared campaign';
  const phaseLabels = { waiting: 'Watching', recording: 'Measuring', queued: 'Waiting to record', verifying: 'Verifying record', done: 'Measurement ended' };
  const svg = (type) => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[type]}</svg>`;
  const dialog = one('[data-dialog]');
  const historyDialog = one('[data-history-dialog]');
  let player;
  let state;
  let proof = null;
  let working = [];
  let report = null;
  let selected = null;
  let highlight = null;
  let checking = false;
  let tampered = false;
  let generation = 0;
  const announce = (text) => { one('[data-announcement]').textContent = text; };
  const eventOf = (block) => block.transactions[0].event;
  const statusOf = (height) => report?.blocks[height - 1]?.state || 'verified';
  one('[data-travelers]').innerHTML = Array.from({ length: 5 }, (_, slot) => `<article class="av-traveler" data-slot="${slot}"><button type="button" class="av-person" data-person disabled><span class="av-avatar" aria-hidden="true"></span><span><strong data-name>Waiting</strong><small data-lane-status>Campaign not started</small><span class="av-person-progress" data-person-progress hidden><i></i></span></span></button><div class="av-path">${types.map((type) => `<button type="button" class="av-step" data-step="${type}" data-state="waiting" disabled><span class="av-stop">${svg(type)}</span><span data-step-label>Waiting</span></button>`).join('')}<i class="av-dot" aria-hidden="true"></i></div></article>`).join('');
  const rows = all('[data-slot]').map((row) => ({ row, person: row.querySelector('[data-person]'), name: row.querySelector('[data-name]'), avatar: row.querySelector('.av-avatar'), status: row.querySelector('[data-lane-status]'), bar: row.querySelector('[data-person-progress]'), steps: types.map((type) => row.querySelector(`[data-step="${type}"]`)) }));
  if (!core || !api) {
    main.dataset.error = 'true';
    one('[data-status]').textContent = 'Unavailable';
    one('[data-message]').textContent = 'The verification engine did not load. Reload this page over HTTPS to retry.';
    return;
  }
  function paintLanes() {
    for (const lane of state.lanes) {
      const view = rows[lane.slot];
      const { row } = view;
      row.dataset.travelerId = lane.id || '';
      row.dataset.summaryHeight = String(lane.records.summary?.height || '');
      row.dataset.summaryKey = lane.records.summary?.key || '';
      row.dataset.phase = lane.phase;
      row.dataset.key = lane.id ? lane.key : '';
      row.dataset.progress = lane.progress.toFixed(5);
      row.style.setProperty('--progress', lane.progress.toFixed(5));
      row.style.setProperty('--opacity', reduced.matches ? '1' : String(lane.opacity ?? 1));
      row.dataset.highlight = String(Boolean(lane.id && lane.id === highlight));
      view.name.textContent = lane.id ? personName(lane.id) : 'Waiting';
      view.person.disabled = !lane.id;
      view.person.setAttribute('aria-label', lane.id ? `Highlight records for ${personName(lane.id)}` : 'Waiting for campaign');
      view.avatar.style.setProperty('--avatar', String(lane.id ? (lane.number - 1) % 4 : lane.slot % 4));
      view.bar.hidden = !['recording', 'verifying'].includes(lane.phase);
      let label = lane.id ? phaseLabels[lane.phase] : 'Campaign not started';
      if (lane.phase === 'recording' && lane.type !== 'summary') label = shortTitle[lane.type] + ' · measuring';
      if (lane.phase === 'waiting' && !lane.records.ad) label = 'Waiting for ad';
      if (lane.phase === 'done') label = lane.records.destination ? (lane.records.website ? 'Website + destination' : 'Destination only') : lane.records.website ? 'Website only' : 'No visits recorded';
      const changed = Object.values(lane.records).some((record) => statusOf(record.height) !== 'verified');
      view.status.dataset.alert = String(changed);
      view.status.textContent = changed ? 'History changed' : label;
      for (let index = 0; index < types.length; index += 1) {
        const type = types[index];
        const node = view.steps[index];
        const record = lane.records[type];
        const active = type === lane.type && ['recording', 'queued', 'verifying'].includes(lane.phase);
        const status = record ? statusOf(record.height) : active ? lane.phase : lane.phase === 'done' ? 'skipped' : 'waiting';
        node.dataset.state = status;
        node.dataset.key = record ? record.key : active ? lane.key : '';
        node.dataset.height = record ? String(record.height) : '';
        node.dataset.phase = record ? 'committed' : active ? lane.phase : status;
        node.disabled = !record;
        node.querySelector('[data-step-label]').textContent = record ? (status === 'verified' ? '✓ #' + record.height : status === 'changed' ? 'Changed' : 'Earlier change') : status === 'skipped' ? 'Not recorded' : active ? (status === 'verifying' ? 'Verifying' : status === 'queued' ? 'Queued' : 'Measuring') : 'Waiting';
        node.setAttribute('aria-label', `${personName(lane.id)}, ${type === 'destination' ? 'destination visit attributed' : shortTitle[type]}, ${record ? 'block ' + record.height : status}`);
      }
      const target = types.indexOf(lane.type);
      const moving = lane.phase === 'recording' && target >= 0;
      row.dataset.moving = String(moving);
      if (moving) {
        const previous = target === 0 ? -.4 : target === 1 || !lane.records.website ? 0 : 1;
        row.style.setProperty('--runner', `${16.6667 + (previous + (target - previous) * lane.progress) * 33.3333}%`);
      }
    }
  }
  function blockMarkup(block, index) {
    const event = eventOf(block);
    const status = statusOf(index + 1);
    return `<li class="av-block" data-height="${index + 1}" data-key="${event.key}" data-traveler-id="${event.travelerId || ''}" data-type="${event.type}" data-state="${status}" data-highlight="${Boolean(highlight && highlight === event.travelerId)}"><button type="button" class="av-block-button" data-inspect="${index}" aria-label="Inspect block ${index + 1}, ${shortTitle[event.type]}, ${personName(event.travelerId)}"><span class="av-block-number">#${index + 1}</span><span><strong class="av-block-title">${shortTitle[event.type]}</strong><span class="av-block-person">${personName(event.travelerId)}</span></span><span class="av-block-valid">${status === 'verified' ? '✓' : status === 'changed' ? '×' : '!'}<span class="av-sr-only"> ${status}</span></span></button></li>`;
  }
  function renderChain() {
    const limit = narrow.matches ? 3 : 6;
    const from = Math.max(0, working.length - limit);
    one('[data-blocks]').innerHTML = working.slice(from).map((block, index) => blockMarkup(block, from + index)).join('');
    one('[data-count-label]').textContent = working.length + ' blocks';
    one('[data-history-note]').textContent = working.length ? (from ? `Latest ${working.length - from} of ${working.length} blocks · earlier records retained` : 'Every block links to the one before it') : 'No records yet';
    one('[data-empty]').hidden = working.length > 0;
    one('[data-history]').textContent = `View full chain${working.length ? ' (' + working.length + ')' : ''}`;
    one('[data-warning]').hidden = !tampered;
  }
  function paintWriter() {
    const writer = state.writer;
    const node = one('[data-writer]');
    node.dataset.key = writer?.key || '';
    node.dataset.phase = writer ? 'verifying' : 'idle';
    node.dataset.progress = (writer?.progress || 0).toFixed(5);
    node.dataset.height = writer ? String(writer.height) : '';
    node.style.setProperty('--progress', String(writer?.progress || 0));
    one('[data-writer-title]').textContent = writer ? `Adding #${writer.height} · ${shortTitle[writer.type]}` : state.started ? 'Waiting for the next event' : 'Waiting for an event';
    one('[data-writer-detail]').textContent = writer ? `${personName(writer.travelerId)} · checking record${state.queue.length ? ' · ' + state.queue.length + ' queued' : ''}` : 'Blocks appear after verification.';
  }
  function controls() {
    const drained = state.capacityClosed && state.lanes.every((lane) => !lane.id || lane.phase === 'done');
    one('[data-play]').disabled = !state.ready || checking || tampered || Boolean(state.error) || drained;
    one('[data-play]').textContent = state.running ? 'Pause campaign' : state.started ? 'Resume campaign' : 'Start campaign';
    for (const node of all('[data-history],[data-verify],[data-export]')) node.disabled = !working.length || checking;
    for (const node of all('[data-restore]')) node.disabled = checking;
    one('[data-edit-form] button[type="submit"]').disabled = checking;
    one('[data-status]').textContent = state.error ? 'Unavailable' : tampered ? 'Edited · paused' : state.running ? 'Live' : !state.ready ? 'Preparing…' : state.started ? drained ? 'Finished' : 'Paused' : 'Ready';
    one('[data-totals]').textContent = state.started ? `${state.lanes.filter((lane) => lane.id && lane.phase !== 'done').length} being measured · ${state.completed} completed` : 'Five lanes · one campaign';
    main.dataset.running = String(state.running);
    main.dataset.count = String(state.count);
    main.dataset.time = state.time.toFixed(3);
    main.dataset.measuring = String(state.lanes.filter((lane) => lane.phase === 'recording').length);
    main.dataset.admitted = String(state.admitted);
    main.dataset.completed = String(state.completed);
    main.dataset.queued = String(state.queue.length);
  }
  function update(change, next) {
    state = next;
    if (change.kind === 'reset') {
      generation += 1; proof = null; working = []; report = null; checking = false; tampered = false; selected = null; highlight = null;
      if (dialog.open) dialog.close(); if (historyDialog.open) historyDialog.close();
      one('[data-message]').textContent = 'When measurement ends, only that traveler is replaced. Earlier records stay in the chain.';
    }
    if (state.proof) proof = state.proof;
    if (change.block) {
      working.push(core.clone(change.block)); report = null;
      const event = eventOf(change.block);
      main.dataset.lastKey = event.key;
      announce(`Block ${state.count} recorded: ${shortTitle[event.type]}, ${personName(event.travelerId)}.`);
    }
    if (['reset', 'ready', 'commit'].includes(change.kind)) renderChain();
    paintLanes(); paintWriter(); controls();
    if (state.error) { main.dataset.error = 'true'; one('[data-message]').textContent = 'Verification unavailable: ' + state.error + ' Reset to retry.'; }
    else { delete main.dataset.error; if (state.capacityClosed) one('[data-message]').textContent = 'Demo limit: no new arrivals. Current travelers finish before the chain stops. Reset to begin again.'; }
  }
  player = api.createPlayer({ onChange: update });
  one('[data-play]').addEventListener('click', () => {
    if (checking || tampered) return;
    if (state.running) player.pause();
    else { player.play(); if (narrow.matches) one('.av-workspace').scrollIntoView({ block: 'start', behavior: 'instant' }); }
  });
  one('[data-reset]').addEventListener('click', () => player.reset());
  one('[data-scenario]').addEventListener('change', (event) => { player.setScenario(event.target.value); one('[data-message]').textContent = 'New arrivals use ' + core.SCENARIOS[event.target.value].toLowerCase() + '. Current travelers keep their own paths.'; });
  one('[data-speed]').addEventListener('change', (event) => player.setSpeed(Number(event.target.value)));
  one('[data-continuous]').addEventListener('change', (event) => player.setContinuous(event.target.checked));
  document.addEventListener('visibilitychange', () => { if (document.hidden) player.pause(); });
  window.addEventListener('pagehide', () => player.pause());
  narrow.addEventListener('change', () => renderChain());
  // Do not replace a focused traveler or trim a focused block during inspection.
  main.addEventListener('focusin', (event) => { if (event.target.closest('[data-person],[data-step],[data-inspect]')) player.pause(); });
  one('[data-travelers]').addEventListener('click', (event) => {
    const step = event.target.closest('[data-step][data-height]');
    if (step?.dataset.height) { inspect(Number(step.dataset.height) - 1); return; }
    const person = event.target.closest('[data-person]');
    if (!person) return;
    player.pause();
    highlight = person.closest('[data-slot]').dataset.travelerId;
    renderChain(); paintLanes();
    one('[data-message]').textContent = `${personName(highlight)} is highlighted. The full chain retains all earlier traveler records.`;
  });
  async function recheck() {
    if (!proof || checking) return false;
    player.pause(); checking = true; controls();
    const token = generation;
    try {
      const result = await core.verifyChain(working, proof.trust);
      if (token !== generation) return false;
      report = result; tampered = !result.valid;
      paintLanes(); renderChain();
      announce(result.valid ? 'All records match their original signed history.' : 'A changed record was detected.');
      return true;
    } catch (error) {
      if (token === generation) { tampered = true; one('[data-message]').textContent = 'Verification could not complete. Reset or restore the original records.'; }
      return false;
    } finally { if (token === generation) { checking = false; controls(); } }
  }
  one('[data-verify]').addEventListener('click', async () => { if (await recheck()) one('[data-message]').textContent = report.valid ? `All ${working.length} records match their signed history.` : 'A changed record was detected. Restore original records to continue.'; });
  one('[data-history]').addEventListener('click', () => {
    player.pause(); one('[data-archive]').innerHTML = working.map(blockMarkup).join(''); historyDialog.showModal();
  });
  one('[data-close-history]').addEventListener('click', () => historyDialog.close());
  for (const parent of [one('[data-blocks]'), one('[data-archive]')]) parent.addEventListener('click', (event) => {
    const node = event.target.closest('[data-inspect]'); if (node) inspect(Number(node.dataset.inspect));
  });
  const fieldLabel = (key) => ({ websiteRecorded: 'Website recorded', destinationRecorded: 'Destination recorded', impressions: 'Impressions', page: 'Website page', name: 'Campaign name', place: 'Destination place' })[key] || key.charAt(0).toUpperCase() + key.slice(1);
  function editField() {
    if (selected === null || !working[selected]) return;
    const value = eventOf(working[selected]).data[one('[data-edit-field]').value];
    const boolean = typeof value === 'boolean';
    one('[data-edit-value]').hidden = boolean; one('[data-edit-value]').required = !boolean;
    one('[data-edit-boolean]').hidden = !boolean;
    one('[data-edit-value]').value = String(value); one('[data-edit-boolean]').value = String(value);
  }
  function fillInspector() {
    const block = working[selected];
    if (!block || !report) return;
    const event = eventOf(block);
    const row = report.blocks[selected];
    one('#av-dialog-title').textContent = `Block #${selected + 1} · ${shortTitle[event.type]}`;
    one('[data-dialog-subtitle]').textContent = personName(event.travelerId) + ' · ' + event.campaignId;
    const fields = [['Previous block', selected ? '#' + selected : 'None (first block)'], ['Traveler event', event.key], ['Prior traveler event', event.previousTravelerEvent || 'None'], ...Object.entries(event.data).filter(([key]) => key !== 'synthetic').map(([key, value]) => [fieldLabel(key), String(value)])];
    one('[data-fields]').innerHTML = fields.map(([label, value]) => `<dt>${escape(label)}</dt><dd>${escape(value)}</dd>`).join('');
    const keys = Object.keys(event.data).filter((key) => !['synthetic', 'note'].includes(key));
    one('[data-edit-field]').innerHTML = keys.map((key) => `<option value="${escape(key)}">${escape(fieldLabel(key))}</option>`).join('');
    const preferred = keys.find((key) => ['impressions', 'page', 'place', 'name', 'websiteRecorded'].includes(key));
    if (preferred) one('[data-edit-field]').value = preferred;
    editField();
    one('[data-verdict]').dataset.valid = String(row.state === 'verified');
    one('[data-verdict]').textContent = row.state === 'verified' ? 'This record matches its signed history.' : row.state === 'changed' ? 'Verification failed. The edited record no longer matches its original proof.' : 'This record is unchanged, but an earlier record failed verification.';
    one('[data-recorded-hash]').textContent = block.hash;
    one('[data-computed-hash]').textContent = row.computedHash || 'Unable to calculate';
    const names = { structure: 'Record structure', journey: 'Traveler path', eventSignature: 'Event signature', merkleRoot: 'Record fingerprint', blockHash: 'Recorded header hash', previousHash: 'Previous block link', approvals: 'Three approvals for recorded hash' };
    one('[data-checks]').innerHTML = Object.entries(row.checks).map(([key, valid]) => `<li>${valid ? '✓' : '×'} ${names[key]}: ${valid ? 'passes' : 'fails'}</li>`).join('');
    one('[data-json]').textContent = JSON.stringify(block, null, 2);
  }
  async function inspect(index) {
    if (checking || !working[index]) return;
    player.pause(); selected = index;
    if (historyDialog.open) historyDialog.close();
    if (await recheck()) { fillInspector(); dialog.showModal(); }
  }
  one('[data-close]').addEventListener('click', () => dialog.close());
  one('[data-edit-field]').addEventListener('change', editField);
  one('[data-edit-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (checking || selected === null) return;
    const data = eventOf(working[selected]).data;
    const key = one('[data-edit-field]').value;
    const prior = data[key];
    let value = typeof prior === 'boolean' ? one('[data-edit-boolean]').value === 'true' : one('[data-edit-value]').value;
    if (typeof prior === 'number') {
      value = Number(value);
      if (!Number.isFinite(value) || value < 0 || value > 1e12) { one('[data-verdict]').textContent = 'Enter a finite number from 0 to 1,000,000,000,000.'; return; }
    }
    data[key] = value;
    if (await recheck()) fillInspector();
  });
  all('[data-restore]').forEach((button) => button.addEventListener('click', async () => {
    if (!proof || checking) return;
    working = core.clone(proof.blocks);
    if (await recheck()) { if (dialog.open) fillInspector(); one('[data-message]').textContent = 'Exact original signed records restored. Resume to continue the same campaign.'; }
  }));
  one('[data-export]').addEventListener('click', () => {
    if (!proof || checking) return;
    player.pause();
    const url = URL.createObjectURL(new Blob([JSON.stringify({ schema: 'ad-verification-demo/v3', notice: 'Fictional observations and local signers. Not independent evidence of visitation.', blocks: working, trust: proof.trust }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'cedar-valley-campaign-proof.json'; document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce('Public proof exported. No private keys are included.');
  });
  player.reset();
})();
