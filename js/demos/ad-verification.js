/* DOM views of the same event stream. No ad, analytics, or location requests. */
(function () {
  'use strict';
  const main = document.getElementById('main');
  if (!main || !main.querySelector('[data-travelers]')) return;
  const one = (selector) => document.querySelector(selector);
  const all = (selector) => [...document.querySelectorAll(selector)];
  const core = window.AdVerificationCore;
  const playerAPI = window.AdVerificationPlayer;
  const paths = {
    play: '<path d="m8 4 12 8-12 8Z" fill="currentColor" stroke="none"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
    megaphone: '<path d="M4 9h4l11-5v16L8 15H4Zm4 6 2 6h4l-2-4M22 9v6"/>',
    document: '<path d="M6 2h8l5 5v15H6Zm8 0v6h5M9 12h7m-7 4h7"/>',
    ad: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 8h18m-13 5 5-2v6l-5-2Z"/>',
    website: '<rect x="3" y="3" width="18" height="14" rx="1.5"/><path d="M1 21h22M7 17l-1 4m11-4 1 4"/>',
    pin: '<path d="M12 22S4 14 4 9a8 8 0 0 1 16 0c0 5-8 13-8 13Z" fill="currentColor" stroke="none"/><circle cx="12" cy="9" r="2.8" fill="white" stroke="none"/>',
    cube: '<path d="m12 2 10 5.5v10L12 23 2 17.5v-10Zm0 11L2 7.5m10 5.5 10-5.5M12 13v10M7 4.7l10 5.6"/>',
    arrow: '<path d="M3 12h17m-6-6 6 6-6 6"/>',
    check: '<path d="m5 12 4 4L20 5"/>'
  };
  const icon = (name, className = '') => `<svg class="av-icon ${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.document}</svg>`;
  all('[data-icon]').forEach((node) => { node.innerHTML = icon(node.dataset.icon); });
  const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const applyReduced = () => { document.body.dataset.reducedMotion = String(reduced.matches); };
  applyReduced();
  reduced.addEventListener('change', applyReduced);
  if (!core || !playerAPI) {
    main.dataset.phase = 'error';
    one('[data-current]').textContent = 'The verification engine did not load. Reload this page over HTTPS to try again.';
    one('[data-run-status]').textContent = 'Verification unavailable';
    return;
  }
  const outcomeTitles = { both: 'Website + destination', website: 'Website only', destination: 'Destination without website', none: 'No visits recorded' };
  const descriptions = { both: 'Explores online, then takes a trip.', website: 'Explores the website. No trip recorded.', destination: 'A trip without a website session.', none: 'Ad exposure. No later visits recorded.' };
  const people = ['Outdoor enthusiast', 'Weekend traveler', 'Event goer', 'Casual browser'];
  const typeIcons = { campaign: 'megaphone', purchase: 'document', ad: 'ad', website: 'website', destination: 'pin', summary: 'check' };
  const scroll = one('[data-ledger-scroll]');
  const dialog = one('[data-dialog]');
  let player;
  let state;
  let proof = null;
  let working = [];
  let report = null;
  let tampered = false;
  let checking = false;
  let selectedGroup = 1;
  let highlight = null;
  let follow = true;
  let selectedBlock = null;
  let epoch = 0;
  let activeNodes = [];

  function announce(text) { one('[data-announcement]').textContent = text; }
  function travelerNumber(id) { return Number(String(id).slice(1)); }
  function travelerName(id) { return 'Traveler ' + travelerNumber(id); }
  function blockEvent(block) { return block.transactions[0].event; }
  function recordState(index) { return report ? report.blocks[index]?.state || 'changed' : 'verified'; }
  function blockFor(id, type) { return working.findIndex((block) => blockEvent(block).travelerId === id && blockEvent(block).type === type); }
  function rowFor(id) { return one(`[data-traveler="${id}"]`); }
  function shortHash(hash) { return hash.slice(0, 7) + '…' + hash.slice(-4); }
  function ensureVisible(container, child) {
    if (!child) return;
    const outer = container.getBoundingClientRect();
    const inner = child.getBoundingClientRect();
    if (inner.bottom > outer.bottom) container.scrollTop += inner.bottom - outer.bottom + 8;
    else if (inner.top < outer.top) container.scrollTop -= outer.top - inner.top + 8;
  }
  function followLive() {
    if (!follow) return;
    scroll.scrollTop = scroll.scrollHeight;
    if (state.active?.draft.travelerId) ensureVisible(one('[data-travelers]'), rowFor(state.active.draft.travelerId));
  }
  function setFollow(value) { follow = value; one('[data-follow]').textContent = value ? 'Following live' : 'Follow latest'; if (value) followLive(); }
  function setHighlight(id) {
    highlight = id;
    all('[data-traveler]').forEach((row) => { row.dataset.highlight = String(row.dataset.traveler === id); });
    all('.av-block').forEach((node) => { node.dataset.highlight = !id ? '' : node.dataset.travelerId === id ? 'match' : 'other'; });
    one('[data-highlight-label]').textContent = id ? `${travelerName(id)} · same shared chain` : 'One campaign. All traveler paths.';
    one('[data-clear-highlight]').hidden = !id;
  }
  function renderTravelers() {
    const group = state.groups.find((item) => item.group === selectedGroup) || { travelers: core.createPlan(state.scenario, 1).travelers };
    const container = one('[data-travelers]');
    const previousScroll = container.scrollTop;
    container.innerHTML = group.travelers.map((traveler) => {
      const summary = blockFor(traveler.id, 'summary');
      const closed = summary >= 0;
      const steps = [['ad', 'Ad served'], ['website', 'Website visit'], ['destination', 'Destination visit']].map(([type, label]) => {
        const index = blockFor(traveler.id, type);
        const status = index >= 0 ? recordState(index) : closed ? 'skipped' : 'waiting';
        const eventId = index >= 0 ? blockEvent(working[index]).id : '';
        return `<div class="av-milestone" data-step="${type}" data-state="${status}" data-event-id="${eventId}" data-phase="${index >= 0 ? 'committed' : status}"><div class="av-stop">${icon(typeIcons[type])}<i class="av-moving-dot" aria-hidden="true"></i></div><span>${label}</span><small>${index >= 0 ? (status === 'verified' ? 'Block #' + (index + 1) : status === 'changed' ? 'Record changed' : 'Earlier change') : closed ? 'Not recorded' : 'Waiting'}</small></div>`;
      }).join(icon('arrow', 'av-path-arrow'));
      const summaryStatus = closed ? recordState(summary) : 'waiting';
      const progressTitle = blockFor(traveler.id, 'destination') >= 0 ? 'Visit attributed' : blockFor(traveler.id, 'website') >= 0 ? 'Website visited' : blockFor(traveler.id, 'ad') >= 0 ? 'Ad recorded' : 'Waiting for ad';
      const outcomeTitle = summaryStatus === 'changed' ? 'Summary changed' : summaryStatus === 'dependent' ? 'Earlier record changed' : closed ? outcomeTitles[traveler.path] : progressTitle;
      return `<article class="av-traveler" data-traveler="${traveler.id}" data-group-id="${traveler.group}" data-highlight="${highlight === traveler.id}"><button type="button" class="av-person" data-highlight-person="${traveler.id}" aria-label="Highlight ${travelerName(traveler.id)} path"><span class="av-avatar" style="--avatar:${traveler.avatar}" aria-hidden="true"></span><span><strong>${travelerName(traveler.id)}</strong><small>${people[traveler.avatar]}</small><em>${descriptions[traveler.path]}</em></span></button><div class="av-path">${steps}<i class="av-travel-runner" aria-hidden="true"></i></div><div class="av-outcome" data-summary="${traveler.id}" data-path="${traveler.path}" data-state="${summaryStatus}" data-phase="${closed ? 'committed' : 'waiting'}" data-event-id="${closed ? blockEvent(working[summary]).id : ''}"><strong>${outcomeTitle}</strong><p>${closed ? 'Observation window closed' : 'Events appear as they are recorded.'}</p></div></article>`;
    }).join('');
    container.scrollTop = previousScroll;
    // Completed campaign milestones are also derived from actual committed blocks.
    for (const type of ['campaign', 'purchase']) {
      const node = one(`[data-campaign-step="${type}"]`);
      const index = working.findIndex((block) => blockEvent(block).type === type);
      node.dataset.state = index < 0 ? 'waiting' : recordState(index);
      node.dataset.eventId = index < 0 ? '' : blockEvent(working[index]).id;
      node.dataset.phase = index < 0 ? 'waiting' : 'committed';
    }
  }
  function blockMarkup(block, index) {
    const event = blockEvent(block);
    const status = recordState(index);
    const content = event.type === 'website' ? event.data.page : event.type === 'destination' ? event.data.place :
      event.type === 'ad' ? event.data.publisher : event.type === 'summary' ? 'Observed website: ' + (event.data.websiteRecorded ? 'yes' : 'no') + ' · destination: ' + (event.data.destinationRecorded ? 'yes' : 'no') : event.data.name || event.data.placement;
    return `<li class="av-block" data-block-index="${index}" data-event-id="${event.id}" data-type="${event.type}" data-phase="committed" data-state="${status}" data-traveler-id="${event.travelerId || ''}"><span class="av-block-marker">${icon('cube')}</span><button class="av-block-button" type="button" data-inspect="${index}" aria-label="Inspect block ${index + 1}: ${core.TYPES[event.type].title}${event.travelerId ? ' for ' + travelerName(event.travelerId) : ''}"><span class="av-block-top">Block #${index + 1}<time datetime="${event.reportedAt}">${event.reportedAt.slice(11, 19)}</time><span class="av-record-status">${status === 'verified' ? '✓ Valid' : status === 'changed' ? '× Changed' : '! Earlier change'}</span></span><strong class="av-block-title">${core.TYPES[event.type].title}${event.travelerId ? ' · ' + travelerName(event.travelerId) : ''}</strong><span class="av-block-detail">${escape(content || 'Cedar Valley Tourism')} ${event.travelerId ? '· ' + event.travelerId : ''}</span><span class="av-block-link">${index ? '← #' + index + ' ' + shortHash(block.header.previousHash) : 'First block · start of chain'}</span></button></li>`;
  }
  function renderLedger(appendOnly = false) {
    const list = one('[data-blocks]');
    if (appendOnly && list.children.length === working.length - 1) list.insertAdjacentHTML('beforeend', blockMarkup(working.at(-1), working.length - 1));
    else list.innerHTML = working.map(blockMarkup).join('');
    one('[data-empty]').hidden = working.length > 0 || Boolean(state.active);
    one('[data-count]').textContent = working.length + (working.length === 1 ? ' block' : ' blocks');
    main.dataset.blockCount = String(working.length);
    one('[data-integrity]').textContent = tampered ? 'Changed history detected' : working.length ? 'All recorded blocks valid' : 'Waiting for campaign';
    setHighlight(highlight);
  }
  function renderPending() {
    const current = state.active;
    const host = one('[data-pending-host]');
    if (!current) { host.replaceChildren(); activeNodes = []; return; }
    host.innerHTML = `<div class="av-pending" data-pending data-event-id="${current.id}" data-phase="recording" data-state="active"><strong>Writing block #${current.height}</strong><p>${core.TYPES[current.draft.type].title}${current.draft.travelerId ? ' · ' + travelerName(current.draft.travelerId) : ''}</p><p data-pending-phase>Recording event…</p><span class="av-pending-bar" aria-hidden="true"></span></div>`;
    const draft = current.draft;
    const row = draft.travelerId ? rowFor(draft.travelerId) : null;
    const milestone = draft.type === 'summary' ? row?.querySelector('[data-summary]') : row?.querySelector(`[data-step="${draft.type}"]`);
    activeNodes = [one('[data-pending]'), milestone || one(`[data-campaign-step="${draft.type}"]`)].filter(Boolean);
    activeNodes.forEach((node) => { node.dataset.eventId = current.id; node.dataset.state = 'active'; });
    if (draft.travelerId && row) {
      const outcome = row.querySelector('.av-outcome');
      outcome.querySelector('strong').textContent = core.TYPES[draft.type].title;
      outcome.querySelector('p').textContent = 'Recording → verifying → appending';
    }
    one('[data-empty]').hidden = true;
    paintActive();
    followLive();
  }
  function paintActive() {
    const active = state.active;
    if (!active) return;
    const value = active.progress.toFixed(5);
    const pulse = reduced.matches ? 0 : (Math.sin(active.progress * Math.PI * 4) + 1) / 2;
    main.dataset.phase = active.phase;
    main.dataset.activeEvent = active.id;
    main.dataset.progress = value;
    for (const node of activeNodes) {
      node.dataset.phase = active.phase;
      node.dataset.progress = value;
      node.style.setProperty('--progress', value);
      node.style.setProperty('--pulse', String(pulse));
      const caption = node.querySelector('small');
      if (caption) caption.textContent = active.phase === 'recording' ? 'Recording…' : 'Verifying…';
    }
    const traveler = active.draft.travelerId ? rowFor(active.draft.travelerId) : null;
    if (traveler) {
      const moving = ['ad', 'website', 'destination'].includes(active.draft.type);
      traveler.dataset.moving = String(moving);
      const target = { ad: 0, website: 1, destination: 2 }[active.draft.type];
      const source = active.draft.type === 'ad' ? -.45 : active.draft.type === 'website' ? 0 : blockFor(active.draft.travelerId, 'website') >= 0 ? 1 : 0;
      if (moving) traveler.style.setProperty('--runner-x', String(16.6667 + (source + (target - source) * active.progress) * 33.3333) + '%');
    }
    const phase = one('[data-pending-phase]');
    if (phase) phase.textContent = active.phase === 'recording' ? 'Recording event…' : 'Checking signatures and links…';
    const name = active.draft.travelerId ? travelerName(active.draft.travelerId) + ' · ' : '';
    one('[data-current]').textContent = `${name}${core.TYPES[active.draft.type].title} → ${active.phase === 'recording' ? 'recording' : 'verifying'} block #${active.height}`;
  }
  function renderControls() {
    const blocked = !state.ready || checking || tampered || Boolean(state.error) || state.limit;
    const label = state.running ? 'Pause campaign' : state.hasWork ? 'Resume campaign' : state.group ? 'Add travelers' : 'Start campaign';
    one('[data-play-label]').textContent = label;
    one('[data-play-inline]').textContent = label;
    one('[data-play]').disabled = blocked;
    one('[data-play-inline]').disabled = blocked;
    one('[data-verify]').disabled = !working.length || checking;
    one('[data-export]').disabled = !working.length || checking;
    all('[data-speed]').forEach((button) => button.setAttribute('aria-pressed', String(Number(button.dataset.speed) === state.speed)));
    one('[data-run-status]').textContent = state.error ? 'Verification unavailable' : tampered ? 'Edited copy · paused' : state.running ? 'Campaign running…' : state.limit ? 'Demo limit reached' : state.hasWork ? 'Campaign paused' : state.group ? 'Group complete' : 'Ready to start';
    one('[data-scenario-note]').textContent = (state.group ? 'Next traveler group: ' : '') + core.SCENARIOS[state.scenario] + '. One shared blockchain.';
    one('[data-tamper-warning]').hidden = !tampered;
    main.dataset.running = String(state.running);
  }
  function onChange(change, next) {
    state = next;
    if (change.kind === 'frame') { paintActive(); return; }
    if (next.proof) proof = next.proof;
    if (change.kind === 'reset') {
      epoch += 1;
      proof = null; working = []; report = null; tampered = false; checking = false;
      selectedGroup = 1; highlight = null; selectedBlock = null; follow = true;
      if (dialog.open) dialog.close();
      one('[data-current]').textContent = 'Press Start campaign to begin. The chain is empty until events are recorded.';
      one('[data-group]').innerHTML = '<option value="1">1</option>';
    }
    if (change.kind === 'group' || change.kind === 'play') selectedGroup = next.group || 1;
    if (change.kind === 'group') {
      one('[data-group]').innerHTML = next.groups.map((item) => `<option value="${item.group}">${item.group} · ${escape({ mixed: 'Mixed', none: 'No visits', website: 'Website', destination: 'Destination', both: 'Both' }[item.scenario])}</option>`).join('');
      highlight = null;
    }
    if (change.kind === 'commit') {
      working = core.clone(proof.blocks);
      report = null;
      main.dataset.phase = 'committed';
      main.dataset.activeEvent = change.completed.id;
      main.dataset.progress = '1.00000';
      one('[data-current]').textContent = `${change.completed.draft.travelerId ? travelerName(change.completed.draft.travelerId) + ' · ' : ''}${core.TYPES[change.completed.draft.type].title} → block #${change.completed.height} appended and verified.`;
      announce(one('[data-current]').textContent);
    }
    if (change.kind === 'error') { one('[data-current]').textContent = 'Unable to verify: ' + state.error + ' Reset the campaign to retry.'; main.dataset.phase = 'error'; }
    if (change.kind === 'limit') one('[data-current]').textContent = 'Demo safety limit reached. All recorded blocks are retained. Reset to start a new campaign.';
    if (change.kind === 'complete') one('[data-current]').textContent = 'This group is complete. Choose another scenario and add travelers to the same chain.';
    if (change.kind === 'reset' || change.kind === 'ready') { main.dataset.phase = 'idle'; main.dataset.activeEvent = ''; main.dataset.progress = '0.00000'; }
    one('[data-group]').value = String(selectedGroup);
    renderTravelers();
    if (['commit', 'reset', 'ready'].includes(change.kind)) renderLedger(change.kind === 'commit');
    renderPending();
    renderControls();
    if (['commit', 'group'].includes(change.kind)) followLive();
  }
  player = playerAPI.createPlayer({ onChange });
  function togglePlay() {
    if (checking || tampered) return;
    if (state.running) player.pause();
    else {
      setFollow(true);
      player.play();
      if (window.innerWidth <= 820) one('#simulation').scrollIntoView({ block: 'start', behavior: 'instant' });
    }
  }
  one('[data-play]').addEventListener('click', togglePlay);
  one('[data-play-inline]').addEventListener('click', togglePlay);
  one('[data-reset]').addEventListener('click', () => player.reset());
  all('[data-speed]').forEach((button) => button.addEventListener('click', () => player.setSpeed(Number(button.dataset.speed))));
  one('[data-scenario]').addEventListener('change', (event) => player.setScenario(event.target.value));
  one('[data-continuous]').addEventListener('change', (event) => player.setContinuous(event.target.checked));
  one('[data-group]').addEventListener('change', (event) => {
    player.pause(); selectedGroup = Number(event.target.value); renderTravelers(); renderPending();
  });
  one('[data-travelers]').addEventListener('click', (event) => {
    const button = event.target.closest('[data-highlight-person]');
    if (!button) return;
    player.pause(); setFollow(false); setHighlight(button.dataset.highlightPerson);
    ensureVisible(scroll, one(`.av-block[data-traveler-id="${highlight}"]`));
  });
  one('[data-clear-highlight]').addEventListener('click', () => setHighlight(null));
  one('[data-follow]').addEventListener('click', () => { setHighlight(null); setFollow(!follow); });
  scroll.addEventListener('wheel', () => setFollow(false), { passive: true });
  scroll.addEventListener('touchstart', () => setFollow(false), { passive: true });
  scroll.addEventListener('keydown', (event) => { if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End'].includes(event.key)) setFollow(false); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) player.pause(); });
  window.addEventListener('pagehide', () => player.pause());
  one('.av-learn').addEventListener('click', () => { one('#about-demo').open = true; });

  async function recheck() {
    if (!proof || checking) return false;
    const token = epoch;
    player.pause(); checking = true; renderControls();
    try {
      const next = await core.verifyChain(working, proof.trust);
      if (token !== epoch) return false;
      report = next; tampered = !next.valid;
      renderTravelers(); renderLedger(); renderPending();
      announce(next.valid ? 'All recorded blocks passed verification.' : 'An edit was detected. Restore original records before continuing.');
      return true;
    } finally { if (token === epoch) { checking = false; renderControls(); } }
  }
  one('[data-verify]').addEventListener('click', () => recheck());
  function editField() {
    if (selectedBlock === null) return;
    const value = blockEvent(working[selectedBlock]).data[one('[data-edit-field]').value];
    const boolean = typeof value === 'boolean';
    one('[data-edit-value]').hidden = boolean;
    one('[data-edit-value]').required = !boolean;
    one('[data-edit-boolean]').hidden = !boolean;
    one('[data-edit-value]').value = String(value);
    one('[data-edit-boolean]').value = String(value);
  }
  const fieldLabel = (key) => ({ websiteRecorded: 'Website visit recorded', destinationRecorded: 'Destination visit recorded', budget: 'Budget', impressions: 'Impressions', page: 'Website page', name: 'Campaign name', place: 'Destination place' })[key] || key.charAt(0).toUpperCase() + key.slice(1);
  function fillInspector() {
    if (selectedBlock === null || !working[selectedBlock] || !report) return;
    const block = working[selectedBlock];
    const event = blockEvent(block);
    const row = report.blocks[selectedBlock];
    one('#av-dialog-title').textContent = 'Block #' + (selectedBlock + 1) + ' · ' + core.TYPES[event.type].title;
    one('[data-dialog-subtitle]').textContent = event.travelerId ? `${travelerName(event.travelerId)} · ${event.travelerId} · Group ${event.group}` : 'Shared campaign event · ' + event.campaignId;
    const fields = [['Campaign', event.campaignId], ['Previous block', selectedBlock ? '#' + selectedBlock : 'None (first block)'],
      ['Previous traveler event', event.previousTravelerEvent || 'None'], ['Example time (UTC)', event.reportedAt], ...Object.entries(event.data).map(([key, value]) => [fieldLabel(key), String(value)])];
    one('[data-fields]').innerHTML = fields.map(([label, value]) => `<dt>${escape(label)}</dt><dd>${escape(value)}</dd>`).join('');
    one('[data-recorded-hash]').textContent = block.hash;
    one('[data-computed-hash]').textContent = row.computedHash || 'Unable to calculate';
    const keys = Object.keys(event.data).filter((key) => !['synthetic', 'note'].includes(key));
    one('[data-edit-field]').innerHTML = keys.map((key) => `<option value="${escape(key)}">${escape(fieldLabel(key))}</option>`).join('');
    const preferred = keys.find((key) => ['impressions', 'page', 'place', 'name', 'websiteRecorded'].includes(key));
    if (preferred) one('[data-edit-field]').value = preferred;
    editField();
    const verdict = one('[data-dialog-verdict]');
    verdict.dataset.valid = String(row.state === 'verified');
    verdict.textContent = row.state === 'verified' ? 'This block matches its signed history.' : row.state === 'changed'
      ? 'Verification failed. This edited record no longer matches its original signatures or fingerprint.'
      : 'This record is unchanged, but an earlier block failed verification.';
    const names = { structure: 'Record structure', journey: 'Traveler path links', eventSignature: 'Event signature', merkleRoot: 'Record fingerprint', blockHash: 'Recorded block hash', previousHash: 'Previous block reference', approvals: 'Three approvals for the recorded hash' };
    one('[data-checks]').innerHTML = Object.entries(row.checks).map(([key, valid]) => `<li>${valid ? '✓' : '×'} ${names[key]}: ${valid ? 'passes' : 'fails'}</li>`).join('');
    one('[data-json]').textContent = JSON.stringify(block, null, 2);
  }
  one('[data-edit-field]').addEventListener('change', editField);
  one('[data-blocks]').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-inspect]');
    if (!button || checking) return;
    player.pause(); selectedBlock = Number(button.dataset.inspect);
    const selected = blockEvent(working[selectedBlock]);
    if (selected.group) { selectedGroup = selected.group; one('[data-group]').value = String(selectedGroup); renderTravelers(); renderPending(); }
    setHighlight(selected.travelerId);
    if (await recheck()) { fillInspector(); dialog.showModal(); }
  });
  one('[data-close]').addEventListener('click', () => dialog.close());
  one('[data-edit-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (checking || selectedBlock === null) return;
    const data = blockEvent(working[selectedBlock]).data;
    const key = one('[data-edit-field]').value;
    const prior = data[key];
    let value = typeof prior === 'boolean' ? one('[data-edit-boolean]').value === 'true' : one('[data-edit-value]').value;
    if (typeof prior === 'number') {
      value = Number(value);
      if (!Number.isFinite(value) || value < 0 || value > 1e12) { one('[data-dialog-verdict]').textContent = 'Enter a finite number from 0 to 1,000,000,000,000.'; return; }
    }
    data[key] = value;
    if (await recheck()) fillInspector();
  });
  all('[data-restore]').forEach((button) => button.addEventListener('click', async () => {
    if (checking || !proof) return;
    working = core.clone(proof.blocks);
    if (await recheck()) { if (dialog.open) fillInspector(); announce('Exact original signed records restored. Playback can resume.'); }
  }));
  one('[data-export]').addEventListener('click', () => {
    if (!proof || checking) return;
    const output = { schema: 'ad-verification-demo/v2', notice: 'Fictional records and same-browser signers. Not independent evidence of ad delivery or visitation.', trust: proof.trust, blocks: working };
    const url = URL.createObjectURL(new Blob([JSON.stringify(output, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'cedar-valley-campaign-proof.json';
    document.body.appendChild(link); link.click(); link.remove();
    // Resource cleanup only; simulation timing is exclusively the player's RAF clock.
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce('Public proof exported. No private keys are included.');
  });
  player.reset();
})();
