import { onCLS, onINP, onLCP } from 'web-vitals';

// One set of document observers. Neither route changes nor consent toggles
// register another set, and denied/withdrawn consent never queues a report.
let started = false;
let consented = false;
let withdrawn = false;
const permitted = () => consented && !withdrawn && window.SiteAnalyticsEnvironment?.enabled !== false && !['1', 'yes'].includes(String(navigator.doNotTrack || window.doNotTrack || '').toLowerCase());
const report = (metric) => {
  if (permitted()) window.sendWebVital?.(metric);
};
const sync = (state) => {
  const categories = state?.categories || state;
  const next = categories?.analytics === true;
  if (started && consented && !next) withdrawn = true;
  consented = next;
  if (started || !permitted()) return;
  started = true;
  // Standard build only: no element attribution, URLs, or DOM contents.
  onLCP(report);
  onINP(report);
  onCLS(report);
};
window.addEventListener('consent-changed', (event) => sync(event.detail));
sync(window.consentAPI?.get?.());
