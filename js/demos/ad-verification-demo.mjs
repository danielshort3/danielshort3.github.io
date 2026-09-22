import { createDemo, verifyChain, simulateChange, STEPS } from './ad-verification-core.mjs';

const root = document.querySelector('.av-page');
const find = selector => root.querySelector(selector);
const demo = find('#verification-demo');
const replayButton = find('[data-av-action="replay"]');
const changeButton = find('[data-av-action="change"]');
const dialog = find('[data-av-dialog]');
const cards = Array.from(root.querySelectorAll('[data-av-inspect]'));
let original = null;
let chain = null;
let trust = null;
let report = null;
let busy = false;
let changed = false;

function announce(message) {
  find('[data-av-announcement]').textContent = message;
}
function setBusy(value) {
  busy = value;
  demo.setAttribute('aria-busy', String(value));
  replayButton.disabled = value;
  changeButton.disabled = value || !original || !report;
  cards.forEach(card => { card.disabled = value || !report; });
}
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'av-icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(svg.namespaceURI, 'use');
  use.setAttribute('href', `#av-${name}`);
  svg.appendChild(use);
  return svg;
}
function stepState(index, state, label) {
  const card = cards[index];
  card.dataset.state = state;
  const status = card.querySelector('[data-av-status]');
  status.replaceChildren();
  if (state === 'verified') status.appendChild(icon('check'));
  if (state === 'mismatch' || state === 'untrusted') status.appendChild(icon('warning'));
  status.appendChild(document.createTextNode(label));
}
function resultState(result) {
  if (result.trusted) return ['verified', 'Verified'];
  if (!result.hashValid || !result.signatureValid || !result.approvalsValid || !result.sequenceValid) return ['mismatch', 'Changed'];
  return ['untrusted', 'Untrusted'];
}
function verdict(state, title, description) {
  find('[data-av-verdict]').dataset.state = state;
  find('[data-av-verdict-title]').textContent = title;
  find('[data-av-verdict-copy]').textContent = description;
  find('[data-av-verdict-icon]').setAttribute('href', state === 'mismatch' || state === 'error' ? '#av-warning' : '#av-shield');
}
function renderResult() {
  report.blocks.forEach((result, index) => stepState(index, ...resultState(result)));
  const summary = find('[data-av-summary]');
  summary.dataset.state = report.valid ? 'verified' : 'mismatch';
  summary.textContent = report.valid ? 'All steps verified' : 'A change was detected';
  verdict(report.valid ? 'verified' : 'mismatch', report.valid ? 'Verified' : 'Mismatch', report.valid
    ? 'All four example records are signed, linked, and unchanged.'
    : 'The publisher record changed. The next step can no longer be trusted.');
  find('[data-av-change-label]').textContent = changed ? 'Reset demo' : 'Simulate change';
  find('[data-av-change-icon]').setAttribute('href', changed ? '#av-reset' : '#av-bolt');
  const note = find('[data-av-change-note]');
  note.hidden = !changed;
  note.textContent = changed ? 'Publisher report changed: 12,500 → 13,000 impressions. Its fingerprint and signatures no longer match. Reset restores the original signed records.' : '';
  announce(report.valid ? 'Verification complete. All four records match their signatures, block links, and retained checkpoint.' : 'Change detected in the publisher record: 12,500 became 13,000 impressions. The measurement step is now untrusted.');
}
function showError(error) {
  console.error('Ad verification demo:', error);
  report = null;
  cards.forEach((_, index) => stepState(index, 'error', 'Unavailable'));
  verdict('error', 'Unavailable', 'Verification could not run. Open this page over HTTPS in a current browser, then try again.');
  const summary = find('[data-av-summary]');
  summary.dataset.state = 'mismatch';
  summary.textContent = 'Verification unavailable';
  replayButton.querySelector('span').textContent = 'Try again';
  find('[data-av-change-note]').hidden = true;
  announce('Verification unavailable. No records have been marked as verified.');
}
async function prepare() {
  const session = await createDemo();
  original = structuredClone(session.chain);
  chain = session.chain;
  trust = session.trust;
  changed = false;
}
async function run(replay = false) {
  if (busy) return;
  setBusy(true);
  try {
    if (!original || !report) await prepare();
    if (replay) {
      chain = structuredClone(original);
      changed = false;
      find('[data-av-change-note]').hidden = true;
      cards.forEach((_, index) => stepState(index, 'ready', 'Waiting'));
      find('[data-av-summary]').textContent = 'Checking signed records…';
      find('[data-av-summary]').dataset.state = 'checking';
      verdict('checking', 'Checking', 'Checking each fingerprint, signature, and previous-block link.');
      announce('Checking the four example records.');
    }
    report = await verifyChain(chain, trust);
    if (replay) {
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      for (const [index, result] of report.blocks.entries()) {
        stepState(index, 'checking', 'Checking');
        if (!reducedMotion) await new Promise(resolve => window.setTimeout(resolve, 380));
        stepState(index, ...resultState(result));
      }
      replayButton.querySelector('span').textContent = 'Replay Demo';
    }
    renderResult();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}
async function change() {
  if (busy || !original || !report) return;
  setBusy(true);
  try {
    changed = !changed;
    chain = changed ? simulateChange(original) : structuredClone(original);
    report = await verifyChain(chain, trust);
    renderResult();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}
function inspect(index) {
  if (busy || !chain || !report || !Number.isInteger(index) || !STEPS[index]) return;
  const block = chain[index];
  const result = report.blocks[index];
  const payload = block.transactions[0].payload;
  find('[data-av-dialog-title]').textContent = `${STEPS[index].title} · Block ${index + 1}`;
  find('[data-av-dialog-result]').textContent = result.trusted ? 'Fingerprint, event signature, three local approvals, and block links match.' : result.reasons.join(' ');
  find('[data-av-record-description]').textContent = payload.details.impressions !== undefined
    ? `${payload.details.impressions.toLocaleString('en-US')} reported impressions · simulated batch`
    : `${STEPS[index].label} · simulated display campaign`;
  find('[data-av-record-hash]').textContent = block.hash;
  find('[data-av-computed-hash]').textContent = result.computedHash || 'Could not compute';
  find('[data-av-raw-record]').textContent = JSON.stringify(block, null, 2);
  dialog.querySelector('details').open = false;
  dialog.showModal();
}
replayButton.addEventListener('click', () => { void run(true); });
changeButton.addEventListener('click', () => { void change(); });
cards.forEach((card, index) => card.addEventListener('click', () => inspect(index)));
find('[data-av-close]').addEventListener('click', () => dialog.close());
void run();
