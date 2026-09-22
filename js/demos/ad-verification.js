/* The UI illustrates provider activity. Public exports never contain these traveler annotations. */
(function () {
  'use strict';
  const main = document.querySelector('#main[data-blocks]');
  if (!main) return;
  const one = (selector) => document.querySelector(selector);
  const all = (selector) => [...document.querySelectorAll(selector)];
  const core = window.AdVerificationCore;
  const api = window.AdVerificationPlayer;
  const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const paths = {
    ad: '<path d="M3 9h5l11-5v16L8 15H3Zm5 6 2 6h4l-2-4M22 9v6"/>',
    website: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18M5 6h14M5 18h14"/>',
    visit: '<path d="M12 22S4 14 4 9a8 8 0 0 1 16 0c0 5-8 13-8 13Z"/><circle cx="12" cy="9" r="2.5"/>',
    chain: '<path d="m9 15 6-6m-7 7-2 2a4 4 0 0 1-6-6l6-6a4 4 0 0 1 6 0m0 2 2-2a4 4 0 0 1 6 6l-6 6a4 4 0 0 1-6 0" transform="translate(1 0)"/>'
  };
  const icon = (type) => `<svg class="av-icon" viewBox="0 0 24 24" aria-hidden="true">${paths[type] || paths.chain}</svg>`;
  all('[data-icon]').forEach((node) => { node.innerHTML = icon(node.dataset.icon); });
  const names = ['Outdoor enthusiast', 'Weekend traveler', 'Couple getaway', 'Event attendee', 'Casual browser'];
  const types = ['ad', 'website', 'visit'];
  const travelerName = (id) => id ? 'Traveler ' + Number(id.slice(1)) : 'Campaign';
  const shortHash = (hash) => hash.slice(0, 9) + '…' + hash.slice(-5);
  const dialog = one('[data-dialog]');
  const body = one('[data-dialog-body]');
  let player;
  let state;
  let proof = null;
  let records = [];
  let byId = new Map();
  let selectedBlock = null;
  let uiEpoch = 0;
  let dialogVersion = 0;
  let returnFocus = null;
  let currentView = null;
  let lastReportId = null;
  const announce = (text) => { one('[data-announcement]').textContent = text; };
  const message = (text) => { one('[data-message]').textContent = text; announce(text); };
  const reduce = window.matchMedia('(prefers-reduced-motion:reduce)');
  one('[data-travelers]').innerHTML = Array.from({ length: 5 }, (_, i) => `<article class="av-traveler" data-lane="${i}"><div class="av-person"><span class="av-avatar" style="--avatar:${i}" aria-hidden="true"></span><span><strong data-person-name>Traveler ${i + 1}</strong><small>${names[i]}</small></span></div><div class="av-path">${types.map((type) => `<span class="av-stop" data-stage="${type}" data-state="waiting" role="img" aria-label="Waiting">${icon(type)}<span class="av-receipt-check" hidden aria-hidden="true">✓</span></span>`).join('')}<i class="av-motion" aria-hidden="true"></i></div><button type="button" class="av-outcome" disabled><strong data-outcome>Waiting</strong><small data-outcome-detail>Start the campaign</small></button></article>`).join('');
  const rows = all('[data-lane]').map((row) => ({ row, name: row.querySelector('[data-person-name]'), avatar: row.querySelector('.av-avatar'),
    outcome: row.querySelector('[data-outcome]'), detail: row.querySelector('[data-outcome-detail]'), stages: types.map((type) => row.querySelector(`[data-stage="${type}"]`)) }));
  if (!core || !api) {
    main.dataset.error = 'true'; one('[data-status]').textContent = 'Unavailable';
    message('The verification engine could not load. Open this page over HTTPS and reload to retry.'); return;
  }
  function refreshProof(next) {
    if (!next) return;
    proof = next; records = core.receipts(proof.blocks);
    byId = new Map();
    for (const block of proof.blocks) for (const signed of block.records) byId.set(signed.receipt.id, { signed, block });
  }
  function receiptLabel(r) {
    return r.type === 'attribution' ? r.data.credited ? 'Visit attributed' : 'Visit not attributed' : core.TITLES[r.type];
  }
  function receiptButton(signed) {
    const r = signed.receipt;
    const privateAnnotation = player.describe(r.id)?.traveler;
    return `<button type="button" class="av-receipt" data-receipt="${r.id}" data-type="${r.type}" data-person="${privateAnnotation || ''}" data-credited="${r.data.credited ?? ''}" data-target="${r.refs[0] || ''}"><span><strong>${receiptLabel(r)}</strong><small>${core.SOURCES[r.source]}${privateAnnotation ? ' · ' + travelerName(privateAnnotation) + ' (example)' : ''}</small></span><span aria-hidden="true">→</span></button>`;
  }
  function blockMarkup(block, expanded = false, archive = false) {
    const label = block.records.length + ' signed ' + (block.records.length === 1 ? 'receipt' : 'receipts');
    return `<li class="av-block" data-height="${block.header.height}" data-selected="${expanded}"><${archive ? 'div' : 'button type="button"'} class="av-block-title" ${archive ? '' : `data-block="${block.header.height}" aria-expanded="${expanded}"`}><span><strong>Block #${block.header.height}</strong><small>${label}</small></span><span class="av-valid">✓ Recorded</span></${archive ? 'div' : 'button'}>${expanded ? `<div class="av-records">${block.records.map(receiptButton).join('')}</div><div class="av-fingerprint">${block.header.height > 1 ? 'Previous: ' + shortHash(block.header.previous) : 'Start of the chain'}<br>Fingerprint: ${shortHash(block.hash)}</div>` : ''}</li>`;
  }
  function renderChain() {
    const recent = proof ? proof.blocks.slice(-3) : [];
    const important = recent.findLast((block) => block.records.some((r) => ['attribution', 'correction', 'report'].includes(r.receipt.type)));
    const chosen = selectedBlock || important?.header.height || recent.at(-1)?.header.height;
    one('[data-chain]').innerHTML = recent.map((block) => blockMarkup(block, block.header.height === chosen)).join('');
    one('[data-empty]').hidden = recent.length > 0;
    one('[data-count]').textContent = (proof?.blocks.length || 0) + ' blocks';
    one('[data-retained]').textContent = proof?.blocks.length > 3 ? `${proof.blocks.length} blocks retained · latest three shown` : 'Every block references the one before it.';
  }
  function renderTotals() {
    const totals = core.totals(records);
    for (const key of ['exposures', 'websites', 'attributed']) one(`[data-total="${key}"]`).textContent = String(totals[key]);
    one('[data-total="active"]').textContent = String(state.lanes.filter((lane) => lane.id && lane.phase !== 'done').length);
  }
  function paint() {
    const corrected = new Set(records.filter((r) => r.type === 'correction').map((r) => r.refs[0]));
    for (const lane of state.lanes) {
      const view = rows[lane.slot];
      view.row.dataset.person = lane.id || '';
      view.row.dataset.phase = lane.phase;
      view.row.dataset.observing = lane.active || '';
      view.row.dataset.progress = String(lane.progress || 0);
      view.row.dataset.attribution = lane.recorded.attribution?.id || '';
      view.row.dataset.ended = lane.recorded.end?.id || '';
      view.row.style.setProperty('--opacity', reduce.matches ? '1' : String(lane.opacity));
      view.name.textContent = travelerName(lane.id || 'T' + (lane.slot + 1));
      view.avatar.style.setProperty('--avatar', String(lane.id ? (lane.number - 1) % 5 : lane.slot));
      for (let i = 0; i < types.length; i += 1) {
        const type = types[i]; const node = view.stages[i]; const record = lane.recorded[type];
        const status = record ? 'recorded' : lane.active === type ? 'observing' : lane.observed[type] ? 'reported' : lane.phase === 'done' ? 'not-recorded' : 'waiting';
        node.dataset.state = status; node.dataset.receiptId = record?.id || ''; node.dataset.block = String(record?.block || '');
        node.querySelector('.av-receipt-check').hidden = !record;
        node.setAttribute('aria-label', `${view.name.textContent}: ${type === 'visit' ? 'destination visit' : type}, ${status}${record ? ' in block ' + record.block : ''}`);
      }
      const attr = byId.get(lane.recorded.attribution?.id)?.signed.receipt;
      let title = !lane.id ? 'Waiting' : lane.active ? 'In progress' : 'Watching';
      let detail = lane.active ? ({ ad: 'Ad report', website: 'Website activity', visit: 'Visit report', attribution: 'Matching evidence', end: 'Closing observation' })[lane.active] : 'Independent timeline';
      let credited = false;
      if (lane.observed.visit) { title = 'Visit reported'; detail = 'Awaiting receipt'; }
      if (attr) { credited = attr.data.credited && !corrected.has(attr.id); title = credited ? 'Attributed' : corrected.has(attr.id) ? 'Corrected' : 'Not attributed'; detail = corrected.has(attr.id) ? 'Duplicate removed' : 'Provider decision recorded'; }
      if (lane.phase === 'done' && !attr) { title = lane.observed.website ? 'Website only' : 'No visit recorded'; detail = 'Observation closed'; }
      view.outcome.textContent = title; view.detail.textContent = detail;
      view.outcome.parentElement.dataset.credited = String(credited);
      view.outcome.parentElement.disabled = !attr;
      view.outcome.parentElement.dataset.receipt = attr?.id || '';
      view.outcome.parentElement.setAttribute('aria-label', `${view.name.textContent}: ${title}${attr ? '. Inspect the attribution receipt.' : ''}`);
      view.row.dataset.active = String(Boolean(lane.active && types.includes(lane.active)));
      if (lane.active && types.includes(lane.active)) {
        const target = types.indexOf(lane.active); const from = target === 0 ? -.3 : target === 2 && !lane.observed.website ? 0 : target - 1;
        view.row.style.setProperty('--position', `${16.6667 + (from + (target - from) * lane.progress) * 33.3333}%`);
      }
    }
    const writer = one('[data-writer]');
    writer.dataset.progress = (state.writer?.progress || 0).toFixed(5);
    writer.dataset.ids = state.writer?.ids.join(',') || '';
    writer.style.setProperty('--progress', writer.dataset.progress);
    one('[data-writer-label]').textContent = state.writer ? `Verifying block #${state.writer.height} · ${state.writer.count} signed receipts` : state.queued ? `Preparing next block · ${state.queued} receipts waiting` : state.recordCount ? 'Waiting for the next measurement report' : 'Waiting for the campaign';
    one('[data-play]').textContent = state.running ? 'Pause campaign' : state.admitted ? 'Resume campaign' : 'Start campaign';
    one('[data-play]').disabled = !state.ready || state.busy || Boolean(state.error) || (state.limited && state.completed === state.admitted);
    one('[data-status]').textContent = state.error ? 'Unavailable' : !state.ready ? 'Preparing…' : state.busy ? 'Checking…' : state.running ? 'Live simulation' : state.admitted ? 'Paused' : 'Ready';
    for (const node of all('[data-history],[data-copies],[data-export]')) node.disabled = !state.blockCount || state.busy;
    one('[data-report]').disabled = state.recordCount < 2 || state.busy;
    main.dataset.blocks = String(state.blockCount); main.dataset.receipts = String(state.recordCount);
    main.dataset.running = String(state.running); main.dataset.completed = String(state.completed);
    main.dataset.admitted = String(state.admitted); main.dataset.time = String(state.time);
    main.dataset.queue = String(state.queued);
    one('[data-total="active"]').textContent = String(state.lanes.filter((lane) => lane.id && lane.phase !== 'done').length);
  }
  function renderCopies() {
    one('[data-copy-states]').innerHTML = state.copies.map((copy, i) => `<span class="av-copy-summary" data-status="${copy.status}" data-copy="${i}">${copy.status === 'Up to date' ? '✓' : copy.status === 'Behind' ? '◷' : '×'} ${i === 2 ? 'Measurement' : copy.name}</span>`).join('');
  }
  function update(event, next) {
    state = next;
    if (event.kind === 'reset') {
      uiEpoch += 1; dialogVersion += 1; proof = null; records = []; byId.clear(); selectedBlock = null; lastReportId = null;
      currentView = null; if (dialog.open) dialog.close();
      message('Accelerated example. Website visits are optional. Attribution is not proof that an ad caused a trip.');
    }
    if (next.proof) refreshProof(next.proof);
    if (event.block) {
      selectedBlock = event.block.header.height;
      main.dataset.lastIds = event.block.records.map((r) => r.receipt.id).join(',');
      announce(`Block ${event.block.header.height} appended with ${event.block.records.length} signed receipts.`);
    }
    if (['reset', 'ready', 'block', 'manual', 'action'].includes(event.kind)) { renderChain(); renderTotals(); renderCopies(); }
    paint();
    if (state.error) { main.dataset.error = 'true'; message('Verification unavailable: ' + state.error + ' Reset to retry.'); }
    else delete main.dataset.error;
    if (state.limited) one('[data-message]').textContent = 'The demo is closing admissions. Existing travelers finish; history is retained. Reset to start again.';
  }
  player = api.createPlayer({ onChange: update });
  function openDialog(title, html, view) {
    if (!dialog.open) returnFocus = document.activeElement;
    one('#av-dialog-title').textContent = title; body.innerHTML = html; currentView = view;
    if (!dialog.open) dialog.showModal();
  }
  function reportData(id) { return byId.get(id)?.signed.receipt; }
  function showReport(id, note = '') {
    const r = reportData(id); if (!r) return;
    const latest = core.totals(records);
    const candidates = records.filter((item) => item.type === 'attribution' && item.data.credited && !records.some((old) => old.type === 'correction' && old.refs[0] === item.id));
    openDialog('Can this campaign report be changed quietly?', `<p>A signed report is a snapshot of committed blocks. Pending measurements appear in later reports. Corrections create new records, not hidden rewrites.</p><div class="av-checks"><p><strong>Signed report: ${r.data.totals.attributed} attributed visits</strong></p><p>Through block #${r.data.throughBlock}. Current signed-receipt total: <strong data-current-total>${latest.attributed}</strong>.</p></div><label class="av-edit-label">Try a different reported total<input data-report-value type="number" min="0" max="1000000" step="1" value="${r.data.totals.attributed + 3}" aria-label="Edited attributed-visit total"></label><button type="button" class="av-button av-primary" data-test-report="${id}">Check edited copy</button><p class="av-verdict" data-test-verdict ${note ? '' : 'hidden'}>${escape(note)}</p><h3>What about a real correction?</h3><p>Emulate an attribution service identifying a duplicate. An authorized, signed correction removes that visit from the current total while retaining the old decision.</p><label class="av-edit-label">Credited decision<select data-correction-target aria-label="Decision to correct">${candidates.map((item) => `<option value="${item.id}">${travelerName(player.describe(item.id)?.traveler)} · block ${byId.get(item.id).block.header.height}</option>`).join('') || '<option value="">None available</option>'}</select></label><div class="av-actions"><button type="button" class="av-button av-secondary" data-correct ${candidates.length ? '' : 'disabled'}>Append example correction</button><button type="button" class="av-button av-secondary" data-new-report>Sign updated report</button></div><details><summary>View signed report receipt</summary><pre>${escape(JSON.stringify(byId.get(id).signed, null, 2))}</pre></details><p>These totals describe signed reports. They do not prove the provider observed every event or that advertising caused the visits.</p>`, { kind: 'report', id });
  }
  async function openReceipt(id) {
    if (!byId.has(id) || state.busy) return;
    player.pause();
    const version = ++dialogVersion; const generation = uiEpoch;
    openDialog('Checking the receipt', '<p>Verifying signatures and requesting the separate example evidence…</p>', { kind: 'receipt', id });
    try {
      const audit = await player.audit(id);
      if (version !== dialogVersion || generation !== uiEpoch || !dialog.open) return;
      const entry = byId.get(id); const r = entry.signed.receipt;
      const hidden = player.describe(id).hidden;
      const corrected = records.some((item) => item.type === 'correction' && item.refs[0] === id);
      const evidenceLabel = audit.evidence === 'checked' ? r.type === 'attribution' ? 'Example attribution calculation reproduced' : 'Provider evidence fingerprint checked' : audit.evidence === 'unavailable' ? 'Evidence unavailable — calculation not reproduced' : 'Evidence mismatch';
      const calculation = audit.calculation ? `<p>${audit.calculation.days} example days after the earlier ad; ${audit.calculation.credited ? 'inside' : 'outside'} the ${audit.calculation.window}-day rule. A website visit is not required.</p>` : '';
      const receiptFields = `<dt>Reported by</dt><dd>${core.SOURCES[r.source]}</dd><dt>Stored in</dt><dd>Block #${entry.block.header.height} · ${entry.block.records.length} receipt(s)</dd><dt>Shared receipt</dt><dd>${receiptLabel(r)}</dd>${r.type === 'attribution' ? `<dt>Signed decision</dt><dd>${r.data.credited ? 'Credited' : 'Not credited'}${corrected ? ' · subsequently corrected' : ''}</dd>` : ''}<dt>Evidence fingerprint</dt><dd><code>${r.evidenceDigest}</code></dd>`;
      let editKey = ['credited', 'count', 'window'].find((key) => Object.hasOwn(r.data, key));
      if (!editKey) editKey = Object.keys(r.data).find((key) => typeof r.data[key] === 'boolean' || typeof r.data[key] === 'number');
      const edit = editKey ? `<details><summary>Try changing this shared receipt</summary><label class="av-edit-label">${escape(editKey)}${typeof r.data[editKey] === 'boolean' ? `<select data-edit-value><option value="true" ${r.data[editKey] ? 'selected' : ''}>Yes</option><option value="false" ${!r.data[editKey] ? 'selected' : ''}>No</option></select>` : `<input data-edit-value type="number" value="${r.data[editKey] + 1}" min="0" max="1000000">`}</label><button type="button" class="av-button av-secondary" data-test-receipt="${id}" data-edit-key="${editKey}">Verify edited copy</button><p class="av-verdict" data-test-verdict hidden></p></details>` : '';
      openDialog(receiptLabel(r), `<div class="av-checks" data-record-check="${audit.record}" data-evidence-check="${audit.evidence}"><p><strong>Signed record: ${audit.record ? 'verified against the original checkpoint' : 'did not verify'}</strong></p><p>${evidenceLabel}</p>${calculation}</div><dl class="av-fields">${receiptFields}</dl><h3>Shared receipt, separate evidence</h3><p>Partners receive the signed summary and evidence fingerprint. Traveler associations, occurrence days and page/location details stay in the example provider store.</p><div class="av-actions"><button type="button" class="av-button av-secondary" data-withhold="${id}" data-hidden="${hidden}">${hidden ? 'Restore example evidence' : 'Emulate unavailable evidence'}</button></div>${r.refs.length ? `<h3>Supporting receipts</h3>${r.refs.map((ref) => receiptButton(byId.get(ref).signed)).join('')}` : ''}${audit.evidence === 'checked' ? `<details><summary>View private example evidence (not shared on-chain)</summary><pre>${escape(JSON.stringify(audit.body, null, 2))}</pre></details>` : ''}${edit}<details><summary>Block links and signed receipt</summary><p>Previous fingerprint: <code>${entry.block.header.previous}</code></p><p>This block: <code>${entry.block.hash}</code></p><pre>${escape(JSON.stringify(entry.signed, null, 2))}</pre></details><p>Record verification does not establish the accuracy of the provider's observation or prove causation. The fictional traveler labels are interface annotations, not shared receipt fields.</p>`, { kind: 'receipt', id });
    } catch (error) { if (version === dialogVersion) openDialog('Evidence check unavailable', `<p>${escape(error.message)}</p>`, { kind: 'receipt', id }); }
  }
  function showCopies(note = '') {
    openDialog('Same history, separately checked demo copies', `<p>Each participant holds a separate local copy. These are simulated organizations in one browser, not independent operators or a network consensus protocol.</p>${state.copies.map((copy, index) => `<section class="av-copy-row"><div><strong>${copy.name}</strong><span data-copy-status="${index}">${copy.status} · ${copy.length} blocks${copy.online ? '' : ' · updates paused'}</span></div><div class="av-actions"><button class="av-link" type="button" data-copy-action="pause" data-copy-index="${index}">Pause updates</button><button class="av-link" type="button" data-copy-action="alter" data-copy-index="${index}">Alter this copy</button><button class="av-link" type="button" data-copy-action="restore" data-copy-index="${index}">Restore verified history</button></div></section>`).join('')}<p data-copy-note>${escape(note || 'Pausing delivery leaves a valid older copy. Altering a stored receipt causes a signature or fingerprint mismatch. The other copies and original totals stay intact.')}</p>`, { kind: 'copies' });
  }
  async function manual(name, value, extra) {
    const generation = uiEpoch;
    const buttons = [...body.querySelectorAll('button')]; buttons.forEach((button) => { button.disabled = true; });
    try { const result = await player.action(name, value, extra); return generation === uiEpoch ? result : null; }
    catch (error) { if (generation === uiEpoch) message(error.message); return null; }
    finally { if (generation === uiEpoch) buttons.filter((button) => button.isConnected).forEach((button) => { button.disabled = false; }); }
  }
  async function newReport() {
    const origin = dialog.open ? dialogVersion : null;
    const block = await manual('report');
    if (!block) return;
    lastReportId = block.records[0].receipt.id;
    if (origin === null || (dialog.open && origin === dialogVersion)) showReport(lastReportId);
  }
  async function checkEdited(id, key, raw) {
    const r = byId.get(id)?.signed.receipt; if (!r) return;
    const originalValue = key === 'reportedTotal' ? r.data.totals.attributed : r.data[key];
    const value = typeof originalValue === 'boolean' ? raw === 'true' : Number(raw);
    if (typeof value === 'number' && (!Number.isSafeInteger(value) || value < 0 || value > 1000000)) throw new Error('Enter a whole number from 0 to 1,000,000.');
    const edited = core.clone(proof);
    const target = edited.blocks.flatMap((block) => block.records).find((signed) => signed.receipt.id === id).receipt;
    if (key === 'reportedTotal') target.data.totals.attributed = value; else target.data[key] = value;
    const version = dialogVersion; const generation = uiEpoch;
    const result = await core.verifyProof(edited);
    if (version !== dialogVersion || generation !== uiEpoch) return;
    const verdict = body.querySelector('[data-test-verdict]'); if (!verdict) return;
    verdict.hidden = false; verdict.dataset.valid = String(result.valid);
    verdict.textContent = result.valid ? 'No mismatch: this copy still matches the signed records.' : 'Edited copy rejected. Original signatures, history and campaign totals are unchanged.';
    announce(verdict.textContent);
  }
  one('[data-play]').addEventListener('click', () => {
    if (state.busy) return;
    if (state.running) player.pause(); else { selectedBlock = null; player.play(); if (innerWidth <= 900) one('.av-workspace').scrollIntoView({ block: 'start', behavior: 'instant' }); }
  });
  one('[data-reset]').addEventListener('click', () => player.reset());
  one('[data-scenario]').addEventListener('change', (event) => { player.setScenario(event.target.value); message('This mix applies to new arrivals. Existing travelers keep their current paths.'); });
  one('[data-speed]').addEventListener('change', (event) => player.setSpeed(Number(event.target.value)));
  one('[data-continuous]').addEventListener('change', (event) => player.setContinuous(event.target.checked));
  one('[data-travelers]').addEventListener('click', (event) => {
    const receipt = event.target.closest('[data-receipt]'); if (receipt?.dataset.receipt) openReceipt(receipt.dataset.receipt);
  });
  one('[data-chain]').addEventListener('click', (event) => {
    const receipt = event.target.closest('[data-receipt]'); if (receipt) { openReceipt(receipt.dataset.receipt); return; }
    const block = event.target.closest('[data-block]');
    if (block) { player.pause(); selectedBlock = Number(block.dataset.block); renderChain(); one(`[data-chain] [data-block="${selectedBlock}"]`)?.focus({ preventScroll: true }); }
  });
  main.addEventListener('focusin', (event) => { if (event.target.closest('[data-block],[data-receipt]')) player.pause(); });
  one('[data-history]').addEventListener('click', () => { player.pause(); openDialog('Full campaign blockchain', `<p>${proof.blocks.length} blocks · ${records.length} signed receipts. All earlier records are retained. Traveler labels below are private example annotations.</p><ol class="av-archive">${proof.blocks.map((block) => blockMarkup(block, true, true)).join('')}</ol>`, { kind: 'history' }); });
  one('[data-copies]').addEventListener('click', () => { player.pause(); showCopies(); });
  one('[data-report]').addEventListener('click', newReport);
  one('[data-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { dialogVersion += 1; if (returnFocus?.isConnected && !returnFocus.disabled) returnFocus.focus({ preventScroll: true }); });
  body.addEventListener('click', async (event) => {
    const version = dialogVersion;
    const receipt = event.target.closest('[data-receipt]'); if (receipt) { await openReceipt(receipt.dataset.receipt); return; }
    const withholding = event.target.closest('[data-withhold]');
    if (withholding) { if (await manual('withhold', withholding.dataset.withhold, withholding.dataset.hidden !== 'true')) if (dialog.open && version === dialogVersion) await openReceipt(withholding.dataset.withhold); return; }
    const copy = event.target.closest('[data-copy-action]');
    if (copy) { const result = await manual('copy', Number(copy.dataset.copyIndex), copy.dataset.copyAction); if (result && dialog.open && version === dialogVersion) showCopies(copy.dataset.copyAction === 'pause' ? 'Updates paused. Resume the campaign to see this copy fall behind.' : 'The copied history has been checked. Other participants retain their own records.'); return; }
    const testReport = event.target.closest('[data-test-report]');
    const testReceipt = event.target.closest('[data-test-receipt]');
    if (testReport || testReceipt) {
      const button = testReport || testReceipt; button.disabled = true;
      try { await checkEdited(testReport?.dataset.testReport || testReceipt.dataset.testReceipt,
        testReport ? 'reportedTotal' : testReceipt.dataset.editKey,
        body.querySelector(testReport ? '[data-report-value]' : '[data-edit-value]').value); }
      catch (error) { message(error.message); }
      finally { if (button.isConnected) button.disabled = false; }
      return;
    }
    if (event.target.closest('[data-correct]')) {
      const id = body.querySelector('[data-correction-target]').value;
      const block = await manual('correct', id);
      if (block && dialog.open && version === dialogVersion) showReport(lastReportId, `Correction added in block #${block.header.height}. The old report is preserved; the current total was recalculated once.`);
      return;
    }
    if (event.target.closest('[data-new-report]')) await newReport();
  });
  one('[data-export]').addEventListener('click', () => {
    player.pause();
    const url = URL.createObjectURL(new Blob([JSON.stringify(proof, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'cedar-valley-shared-audit-proof.json'; document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000); message('Shared proof exported: signed receipts and public keys, with no private provider evidence or traveler IDs.');
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) player.pause(); });
  window.addEventListener('pagehide', () => player.pause());
  player.reset();
})();
