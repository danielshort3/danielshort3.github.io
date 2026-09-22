/* UI for the unlisted Cedar Valley demo. All state stays in this page. */
(function () {
  'use strict';
  const main = document.getElementById('main');
  if (!main || !main.querySelector('[data-av-demo]')) return;
  const core = window.AdVerificationCore;
  const one = (selector) => document.querySelector(selector);
  const buttons = [...main.querySelectorAll('button')];
  const steps = [...main.querySelectorAll('[data-av-step]')];
  const dialog = one('#av-record-dialog');
  const successIcon = one('[data-av-result-icon]').innerHTML;
  const warningIcon = '<svg class="av-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 2 21h20ZM12 9v5m0 3v1"/></svg>';
  const labels = ['Advertiser', 'Agency / DSP', 'Publisher', 'Measurement'];
  let original = null;
  let blocks = null;
  let trust = null;
  let report = null;
  let changed = false;
  let busy = false;

  function announce(text) { one('[data-av-announcement]').textContent = text; }
  function setBusy(value) {
    busy = value;
    one('[data-av-demo]').setAttribute('aria-busy', String(value));
    buttons.forEach((button) => { button.disabled = value || !blocks; });
  }

  function pending() {
    one('[data-av-result]').dataset.state = 'pending';
    one('[data-av-verdict]').textContent = 'Checking';
    one('[data-av-message]').textContent = 'Creating and checking four signed example records.';
    one('[data-av-overall]').dataset.state = 'pending';
    one('[data-av-overall-label]').textContent = 'Checking records';
    one('[data-av-change-note]').hidden = true;
    one('[data-av-result-icon]').innerHTML = successIcon;
    steps.forEach((step, index) => {
      step.dataset.state = 'pending';
      one(`[data-av-status="${index}"]`).textContent = 'Checking';
      one(`[data-av-symbol="${index}"]`).textContent = '○';
    });
  }

  function renderStep(index) {
    const row = report.blocks[index];
    const state = row ? row.state : 'changed';
    steps[index].dataset.state = state;
    one(`[data-av-status="${index}"]`).textContent = { verified: 'Verified', changed: 'Changed', dependent: 'Earlier change' }[state];
    one(`[data-av-symbol="${index}"]`).textContent = { verified: '✓', changed: '×', dependent: '!' }[state];
    steps[index].setAttribute('aria-label', `Inspect ${labels[index]} record: ${one(`[data-av-status="${index}"]`).textContent}`);
    if (index === 2) one('[data-av-value="2"]').textContent = blocks[2].transactions[0].event.data.impressions.toLocaleString('en-US') + ' impressions';
  }

  function render() {
    steps.forEach((_, index) => renderStep(index));
    const state = report.valid ? 'verified' : 'changed';
    one('[data-av-result]').dataset.state = state;
    one('[data-av-overall]').dataset.state = state;
    one('[data-av-overall-label]').textContent = report.valid ? 'All records verified' : 'A change was detected';
    one('[data-av-verdict]').textContent = report.valid ? 'Verified' : 'Change detected';
    one('[data-av-message]').textContent = report.valid
      ? 'All four records match their signed history for Cedar Valley Tourism.'
      : 'The publisher record no longer matches its original signed history.';
    one('[data-av-result-icon]').innerHTML = report.valid ? successIcon : warningIcon;
    one('[data-av-toggle-label]').textContent = changed ? 'Restore original' : 'Simulate change';
    one('[data-av-change-note]').hidden = !changed;
  }

  function fail(error) {
    report = null;
    blocks = null;
    setBusy(false);
    one('[data-av-run]').disabled = false;
    one('[data-av-run] span').textContent = 'Retry demo';
    one('[data-av-result]').dataset.state = 'error';
    one('[data-av-result-icon]').innerHTML = warningIcon;
    one('[data-av-verdict]').textContent = 'Unable to verify';
    one('[data-av-overall]').dataset.state = 'changed';
    one('[data-av-overall-label]').textContent = 'Verification unavailable';
    const message = !window.crypto || !window.crypto.subtle
      ? 'Open this page over HTTPS in a current browser to use Web Crypto.'
      : 'The demo could not initialize. Use Retry demo to try again.';
    one('[data-av-message]').textContent = message;
    steps.forEach((step, index) => {
      step.dataset.state = 'pending';
      one(`[data-av-status="${index}"]`).textContent = 'Not verified';
      one(`[data-av-symbol="${index}"]`).textContent = '○';
    });
    announce(message);
    console.error('Ad verification demo:', error && error.message ? error.message : 'Initialization failed.');
  }

  async function launch(animate) {
    if (busy) return;
    setBusy(true);
    pending();
    try {
      if (!core) throw new Error('The verification engine did not load.');
      const demo = await core.createDemo();
      original = core.clone(demo.blocks);
      blocks = core.clone(original);
      trust = demo.trust;
      changed = false;
      report = await core.verifyChain(blocks, trust);
      if (animate && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        for (let index = 0; index < steps.length; index += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 260));
          renderStep(index);
        }
      }
      render();
      one('[data-av-run] span').textContent = 'Launch Demo';
      setBusy(false);
      announce('All four example records are verified. Use Simulate change to test the publisher record.');
    } catch (error) { fail(error); }
  }

  async function toggleChange() {
    if (busy || !original) return;
    setBusy(true);
    try {
      changed = !changed;
      // Restore the original signed bytes; never silently re-sign modified data.
      blocks = changed ? core.simulateChange(original) : core.clone(original);
      report = await core.verifyChain(blocks, trust);
      render();
      setBusy(false);
      announce(changed
        ? 'Change detected. The publisher count changed from 10,000 to 12,500. The next record depends on the altered step.'
        : 'Original signed records restored. All four records verify again.');
    } catch (error) { fail(error); }
  }

  const fieldNames = { destination: 'Destination', campaign: 'Campaign', creative: 'Creative', agency: 'Agency', publisher: 'Publisher', impressions: 'Impressions', provider: 'Provider', reportedImpressions: 'Reported count', note: 'Note' };
  function inspect(index) {
    if (busy || !blocks || !report) return;
    const block = blocks[index];
    const event = block.transactions[0].event;
    const row = report.blocks[index];
    one('#av-record-title').textContent = `${labels[index]} · Block ${index + 1}`;
    one('[data-av-record-explanation]').textContent = row.state === 'changed'
      ? 'This record has changed since it was signed.'
      : row.state === 'dependent' ? 'This record is unchanged, but an earlier step failed verification.'
        : 'This example record matches its signing key and the recorded chain.';
    const fields = one('[data-av-record-fields]');
    fields.replaceChildren();
    const entries = [['Event', event.name], ['Example time', event.reportedAt], ...Object.entries(event.data).map(([key, value]) => [fieldNames[key] || key, typeof value === 'number' ? value.toLocaleString('en-US') : String(value)])];
    entries.forEach(([label, value]) => {
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = value;
      fields.append(dt, dd);
    });
    one('[data-av-recorded-hash]').textContent = block.hash;
    one('[data-av-computed-hash]').textContent = row.computedHash || 'Unable to calculate';
    const checks = one('[data-av-record-checks]');
    checks.replaceChildren();
    const checkLabels = { eventSignature: 'Event signature', merkleRoot: 'Record fingerprint', blockHash: 'Recorded header hash', previousHash: 'Previous block reference', approvals: 'All 3 approvals for the recorded hash' };
    [...Object.entries(checkLabels).map(([key, label]) => [label, row.checks[key]]), ['Earlier records intact', row.ancestryValid]].forEach(([label, valid]) => {
      const li = document.createElement('li');
      li.dataset.valid = String(valid);
      li.textContent = `${valid ? '✓' : '×'} ${label}: ${valid ? 'passes' : 'fails'}`;
      checks.append(li);
    });
    one('[data-av-record-json]').textContent = JSON.stringify(block.transactions[0], null, 2);
    dialog.showModal();
  }

  one('[data-av-run]').addEventListener('click', () => {
    if (busy) return;
    one('#demo').scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    launch(true);
  });
  one('[data-av-toggle]').addEventListener('click', toggleChange);
  steps.forEach((step, index) => step.addEventListener('click', () => inspect(index)));
  one('[data-av-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  one('[data-av-check]').addEventListener('click', async () => {
    if (busy || !blocks) return;
    setBusy(true);
    try {
      report = await core.verifyChain(blocks, trust);
      render();
      setBusy(false);
      announce(report.valid ? 'Recheck complete. All four records still match.' : 'Recheck complete. The modified record still fails verification.');
    } catch (error) { fail(error); }
  });
  one('[data-av-export]').addEventListener('click', () => {
    if (busy || !blocks) return;
    const proof = { schema: 'ad-verification-demo/v1', notice: 'Synthetic data and same-browser signers. This is not independent evidence of ad delivery.', trust: core.clone(trust), blocks: core.clone(blocks) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(proof, null, 2) + '\n'], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'cedar-valley-public-proof.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce('Public proof exported. No private keys are included.');
  });
  launch(false);
})();
