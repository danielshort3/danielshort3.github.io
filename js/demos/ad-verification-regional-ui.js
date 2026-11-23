/* The regional lab keeps fictional case details in this page and exports the engine's public proof. */
(function () {
  'use strict';

  const root = document.querySelector('[data-region-root]');
  const receipts = document.querySelector('[data-receipt-root]');
  if (!root || !receipts) return;
  if (new URLSearchParams(window.location.search).get('lab') === 'receipts') {
    root.remove();
    receipts.hidden = false;
    return;
  }
  receipts.remove();

  const engine = window.AdVerificationRegional;
  const field = (name) => root.querySelector('[data-region-' + name + ']');
  const run = field('run');
  const next = field('next');
  const reset = field('reset');
  const verify = field('verify');
  const download = field('export');
  const claim = field('claim');
  const claimValue = field('claim-value');
  const claimSubmit = field('claim-submit');
  const example = field('example');
  const claimResult = field('claim-result');
  const verifyResult = field('verify-result');
  const progress = field('progress');
  const ledger = field('ledger');
  const journeyList = field('journeys');
  const inspector = field('inspector');
  const detailBody = field('detail-body');
  const close = field('close');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const journeyRows = new Map();
  let session;
  let snapshot;
  let job = 'loading';
  let revision = 0;
  let inspected;
  let opener;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function count(value, singular, plural = singular + 's') {
    return value + ' ' + (value === 1 ? singular : plural);
  }

  function complete() {
    return snapshot && snapshot.completedBatches === snapshot.totalBatches;
  }

  function controls() {
    const idle = job === 'idle' && Boolean(snapshot);
    const done = complete();
    const proofReady = idle && snapshot.completedBatches > 0;
    run.disabled = !idle;
    run.textContent = job === 'advance' ? 'Running…' : done ? 'Replay campaign' : 'Run campaign';
    next.disabled = !idle || done;
    reset.disabled = job !== 'failed' && job !== 'advance' && !proofReady;
    reset.textContent = job === 'failed' ? 'Retry setup' : 'Reset';
    verify.disabled = !proofReady;
    download.disabled = !proofReady;
    claimValue.disabled = !done || !idle;
    claimSubmit.disabled = !done || !idle;
    example.disabled = !done || !idle;
    root.dataset.completedBatches = String(snapshot?.completedBatches || 0);
    root.dataset.phase = job === 'failed' ? 'error' : job === 'advance' ? 'running' :
      !snapshot ? 'loading' : done ? 'complete' : 'ready';
  }

  function feedback(node, text, verdict) {
    node.textContent = text;
    node.hidden = !text;
    if (verdict) node.dataset.verdict = verdict;
    else delete node.dataset.verdict;
  }

  function clearChecks() {
    claimValue.value = '';
    claimValue.removeAttribute('aria-invalid');
    feedback(verifyResult, '');
    feedback(claimResult, 'Complete the campaign to audit a claim.');
  }

  function makeJourney(journey, index) {
    const button = element('button', 'rg-case');
    button.type = 'button';
    button.dataset.regionJourney = journey.id;
    button.setAttribute('aria-haspopup', 'dialog');
    const avatar = element('span', 'rg-avatar');
    avatar.setAttribute('aria-hidden', 'true');
    if (index < 5) avatar.style.setProperty('--portrait', String(index));
    else avatar.classList.add('rg-pass');
    const name = element('strong', '', journey.label);
    const summary = element('small');
    const caption = element('span', 'rg-case-caption');
    caption.append(name, summary);
    const badge = element('span', 'rg-outcome');
    const arrow = element('span', 'rg-chevron', '›');
    arrow.setAttribute('aria-hidden', 'true');
    button.append(avatar, caption, badge, arrow);
    const item = element('li');
    item.append(button);
    journeyList.append(item);
    return { summary, badge };
  }

  function renderJourneys() {
    for (const [index, journey] of snapshot.journeys.entries()) {
      if (!journeyRows.has(journey.id)) journeyRows.set(journey.id, makeJourney(journey, index));
      const { summary, badge } = journeyRows.get(journey.id);
      summary.textContent = count(journey.views, 'app view') + ' · ' + count(journey.stops.length, 'signed stop');
      badge.textContent = journey.outcome;
      badge.dataset.tone = journey.outcome === 'Matched' ? 'matched' :
        journey.outcome.startsWith('Awaiting') ? 'pending' : 'excluded';
    }
    const featured = snapshot.journeys[0];
    if (!featured) return;
    for (const marker of root.querySelectorAll('[data-region-pin]')) {
      const label = marker.lastElementChild.textContent;
      marker.dataset.observed = String(featured.stops.some((stop) => stop.label === label));
    }
    field('map-caption').textContent = featured.stops.length ?
      count(featured.stops.length, 'signed regional stop') + ' for the three-stop traveler · ' +
        (featured.outcome === 'Matched' ? '1 matched device' : featured.outcome) :
      'One fictional device can visit several stops and count once.';
  }

  function shortHash(hash) {
    return hash.slice(0, 10) + '…' + hash.slice(-6);
  }

  function renderLedger() {
    for (const batch of snapshot.batches.slice(ledger.children.length)) {
      const item = element('li');
      const button = element('button', 'rg-batch');
      button.type = 'button';
      button.dataset.regionBatch = String(batch.number);
      button.setAttribute('aria-haspopup', 'dialog');
      const heading = element('span', 'rg-batch-title');
      heading.append(element('strong', '', 'Block ' + batch.number), element('code', '', shortHash(batch.hash)));
      const totals = batch.totals;
      const tally = count(totals.views, 'app view') + ' · ' + count(totals.observations, 'observation') +
        ' · ' + count(totals.matched, 'matched device');
      button.append(heading, element('span', 'rg-batch-totals', tally));
      item.append(button);
      ledger.append(item);
    }
    field('empty').hidden = snapshot.batches.length > 0;
  }

  function render() {
    snapshot = session.snapshot();
    controls();
    for (const node of root.querySelectorAll('[data-region-total]')) {
      node.textContent = String(snapshot.totals[node.dataset.regionTotal]);
    }
    renderJourneys();
    renderLedger();
    if (inspector.open && inspected) renderInspector();
    const messages = [
      'Ready to run',
      'Receiving apps signed their view receipts',
      'Regional gateways signed local observations',
      'All batches sealed'
    ];
    progress.textContent = 'Browser simulation · ' + snapshot.completedBatches + ' of ' + snapshot.totalBatches +
      ' batches · ' + (messages[snapshot.completedBatches] || 'Batch sealed');
    if (complete() && !claimResult.dataset.verdict) feedback(claimResult, 'Enter an agency claim or try the example.');
  }

  async function newSession(currentRevision) {
    session = null;
    snapshot = null;
    inspected = null;
    if (inspector.open) inspector.close();
    journeyRows.clear();
    journeyList.replaceChildren();
    ledger.replaceChildren();
    field('empty').hidden = false;
    for (const node of root.querySelectorAll('[data-region-total]')) node.textContent = '0';
    for (const marker of root.querySelectorAll('[data-region-pin]')) marker.dataset.observed = 'false';
    field('map-caption').textContent = 'One fictional device can visit several stops and count once.';
    progress.textContent = 'Preparing the local simulation…';
    clearChecks();
    controls();
    if (!engine || !window.crypto?.subtle) throw new Error('Cryptographic signing is unavailable.');
    const fresh = await engine.createSession();
    if (revision !== currentRevision) return false;
    session = fresh;
    render();
    return true;
  }

  function setupFailed() {
    job = 'failed';
    controls();
    progress.textContent = 'The local signing engine is unavailable. Open this demo over HTTPS or localhost, then retry setup.';
  }

  async function resetSession() {
    if (!['idle', 'failed', 'loading', 'advance'].includes(job)) return;
    const currentRevision = ++revision;
    job = 'loading';
    try {
      if (!await newSession(currentRevision)) return;
      job = 'idle';
      controls();
    } catch (_) {
      if (revision === currentRevision) setupFailed();
    }
  }

  async function advance(all) {
    if (job !== 'idle' || !session || (!all && complete())) return;
    const currentRevision = ++revision;
    job = 'advance';
    controls();
    try {
      if (complete() && !await newSession(currentRevision)) return;
      const activeSession = session;
      do {
        if (revision !== currentRevision) return;
        await activeSession.nextBatch();
        if (revision !== currentRevision) return;
        feedback(verifyResult, '');
        render();
        if (all && !complete()) {
          await new Promise((resolve) => window.setTimeout(resolve, reduceMotion.matches ? 140 : 850));
        }
      } while (all && !complete());
      if (revision !== currentRevision) return;
      job = 'idle';
      controls();
    } catch (_) {
      if (revision !== currentRevision) return;
      job = 'failed';
      controls();
      progress.textContent = 'This batch could not be sealed. Retry setup to start a new local campaign.';
    }
  }

  function addFields(parent, entries) {
    const list = element('dl', 'rg-fields');
    for (const [label, value] of entries) list.append(element('dt', '', label), element('dd', '', String(value)));
    parent.append(list);
  }

  function inspectJourney(journey) {
    field('detail-title').textContent = journey.label;
    const decision = element('div', 'rg-decision');
    decision.append(element('strong', '', journey.outcome), element('p', '', journey.reason));
    detailBody.append(decision, element('h3', '', 'Receiving app receipts'));
    if (!journey.viewReceipts.length) detailBody.append(element('p', '', 'No signed display receipt has arrived for this case.'));
    for (const view of journey.viewReceipts) {
      const card = element('section', 'rg-record');
      card.append(element('h4', '', view.app + ' · day ' + view.day));
      addFields(card, [
        ['Measured display', view.visiblePercent + '% visible for ' + count(view.seconds, 'second')],
        ['Creative', view.creative], ['Placement', view.placement],
        ['Campaign token prefix', view.tokenPreview || 'Unavailable'], ['Receipt ID', view.id]
      ]);
      detailBody.append(card);
    }
    detailBody.append(element('h3', '', 'Destination gateway receipts'));
    if (!journey.stops.length) detailBody.append(element('p', '', 'No signed regional encounter has arrived for this case.'));
    for (const stop of journey.stops) {
      const card = element('section', 'rg-record');
      card.append(element('h4', '', stop.label + ' · day ' + stop.day));
      addFields(card, [
        ['Signed by', stop.source], ['Dwell time', count(stop.dwellMinutes, 'minute')],
        ['Campaign token prefix', stop.tokenPreview || 'Unavailable'], ['Receipt ID', stop.id]
      ]);
      detailBody.append(card);
    }
    detailBody.append(element('h3', '', 'Private campaign match'));
    if (journey.tokenPreview) {
      detailBody.append(element('p', '', 'Authorized receipts can share this campaign-scoped token. Only a shortened preview is shown here.'));
      const token = element('code', 'rg-hash', journey.tokenPreview);
      detailBody.append(token);
    } else {
      detailBody.append(element('p', '', 'There is no shared authorized campaign token available for this case.'));
      if (journey.outcome === 'Unlinked') {
        detailBody.append(element('p', '', 'This fictional case illustrates both events. The verifier receives separate records and cannot tell whether they came from the same device.'));
      }
    }
  }

  function inspectBatch(batch) {
    field('detail-title').textContent = 'Regional block ' + batch.number;
    detailBody.append(element('p', '', 'The public block contains cumulative regional totals and cryptographic commitments.'));
    const signed = session.proof().batches.find((block) => block.number === batch.number);
    const roleNames = { network: 'Regional network', reviewer: 'Independent verifier' };
    const signers = Object.keys(signed.approvals).map((role) => roleNames[role] || role);
    addFields(detailBody, [
      ['App-confirmed ad views', batch.totals.views], ['Regional observations', batch.totals.observations],
      ['Matched visiting devices', batch.totals.matched], ['Signed by', signers.join(' and ')]
    ]);
    for (const [label, value] of [
      ['Private evidence fingerprint', batch.fingerprint], ['Previous block hash', batch.previousHash], ['Block hash', batch.hash]
    ]) {
      detailBody.append(element('h3', '', label), element('code', 'rg-hash', value));
    }
    if (batch.number === 1) detailBody.append(element('p', '', 'The first block begins with an all-zero previous hash.'));
  }

  function renderInspector() {
    detailBody.replaceChildren();
    if (inspected.type === 'journey') {
      inspectJourney(snapshot.journeys.find((journey) => journey.id === inspected.id));
    } else {
      inspectBatch(snapshot.batches.find((batch) => String(batch.number) === inspected.id));
    }
  }

  function openInspector(event) {
    const button = event.target.closest('[data-region-journey], [data-region-batch]');
    if (!button || !snapshot) return;
    opener = button;
    inspected = button.hasAttribute('data-region-journey') ? { type: 'journey', id: button.dataset.regionJourney } :
      { type: 'batch', id: button.dataset.regionBatch };
    renderInspector();
    inspector.showModal();
    close.focus();
  }

  async function verifyPublic() {
    if (job !== 'idle' || !snapshot?.batches.length) return;
    job = 'verify';
    controls();
    feedback(verifyResult, 'Checking the public signatures and hash links…');
    try {
      const result = await engine.verifyProof(session.proof());
      if (result.valid) {
        feedback(verifyResult, 'Verified. The signatures match the keys in the export, and the hash links check out across ' +
          count(snapshot.completedBatches, 'public block') + '.', 'verified');
      } else {
        feedback(verifyResult, 'Verification failed. The public signatures or hash links do not check out.', 'invalid');
      }
    } catch (_) {
      feedback(verifyResult, 'The public proof could not be verified. Try again.', 'invalid');
    } finally {
      job = 'idle';
      controls();
    }
  }

  function exportProof() {
    if (job !== 'idle' || !snapshot?.batches.length) return;
    const blob = new Blob([JSON.stringify(session.proof(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = element('a');
    link.href = url;
    link.download = 'regional-campaign-proof.json';
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function checkClaim() {
    if (job !== 'idle' || !complete()) return;
    const value = claimValue.value.trim();
    const reported = Number(value);
    if (!/^\d+$/.test(value) || !Number.isSafeInteger(reported)) {
      claimValue.setAttribute('aria-invalid', 'true');
      feedback(claimResult, 'Enter a whole number of zero or more.', 'invalid');
      claimValue.focus();
      return;
    }
    claimValue.removeAttribute('aria-invalid');
    job = 'claim';
    controls();
    feedback(claimResult, 'Checking source signatures and recounting eligible devices…');
    try {
      const audit = await session.audit();
      if (!audit.valid) {
        feedback(claimResult, 'The private audit failed. The submitted evidence cannot establish a verified count.', 'invalid');
      } else if (reported === audit.matched) {
        feedback(claimResult, 'Supported. The independent audit found ' + count(audit.matched, 'matched visiting device') +
          ', which agrees with the agency’s claim.', 'supported');
      } else {
        const difference = reported - audit.matched;
        const explanation = difference > 0 ? 'The claim exceeds the evidence by ' + count(difference, 'device') + '.' :
          'The claim omits ' + count(-difference, 'qualifying device') + '.';
        feedback(claimResult, 'Unsupported. The agency claimed ' + reported + ' and the independent audit found ' +
          count(audit.matched, 'matched visiting device') + '. ' + explanation, 'unsupported');
      }
    } catch (_) {
      feedback(claimResult, 'The private evidence could not be audited. Try again.', 'invalid');
    } finally {
      job = 'idle';
      controls();
    }
  }

  root.addEventListener('keydown', (event) => {
    if (event.repeat && (event.key === 'Enter' || event.key === ' ') && event.target.closest('button')) event.preventDefault();
  });
  run.addEventListener('click', (event) => { if (event.detail < 2) advance(true); });
  next.addEventListener('click', (event) => { if (event.detail < 2) advance(false); });
  reset.addEventListener('click', (event) => { if (event.detail < 2) resetSession(); });
  verify.addEventListener('click', verifyPublic);
  download.addEventListener('click', (event) => { if (event.detail < 2) exportProof(); });
  claim.addEventListener('submit', (event) => { event.preventDefault(); checkClaim(); });
  claimValue.addEventListener('input', () => {
    claimValue.removeAttribute('aria-invalid');
    feedback(claimResult, '');
  });
  example.addEventListener('click', () => {
    if (job !== 'idle' || !complete()) return;
    claimValue.value = String(snapshot.totals.matched + 3);
    checkClaim();
  });
  journeyList.addEventListener('click', openInspector);
  ledger.addEventListener('click', openInspector);
  close.addEventListener('click', () => inspector.close());
  inspector.addEventListener('close', () => {
    inspected = null;
    if (opener?.isConnected) opener.focus();
  });

  resetSession();
})();
