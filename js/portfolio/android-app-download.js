/* Keep the Android project download aligned with the public review release feed. */
(() => {
  'use strict';

  if (window.AndroidAppDownload) return;

  const FEED_URL = '/app-updates/review/latest.json';
  const RELEASE_PATH = '/danielshort3/danielshort3.github.io/releases/download/';
  const HASH = /^[a-f0-9]{64}$/;
  const mounted = new WeakSet();

  function validateRelease(manifest) {
    const latest = manifest?.latest;
    const apk = latest?.apk;
    if (manifest?.schemaVersion !== 1 || manifest.channel !== 'review' ||
        manifest.packageName !== 'me.danielshort.app.debug' ||
        !Number.isSafeInteger(latest?.versionCode) || latest.versionCode < 1 ||
        typeof latest.versionName !== 'string' || !/^\d+(?:\.\d+)*(?:-debug)?$/.test(latest.versionName) ||
        !Number.isSafeInteger(latest.minSdk) || latest.minSdk < 26 || latest.minSdk > 1000 ||
        !Number.isSafeInteger(apk?.size) || apk.size < 1 || apk.size > 256 * 1024 * 1024 ||
        !HASH.test(apk?.sha256 || '') || !HASH.test(latest.signerSha256 || '')) {
      throw new Error('Invalid Android release metadata');
    }
    const url = new URL(apk.url);
    if (url.origin !== 'https://github.com' || !url.pathname.startsWith(RELEASE_PATH) ||
        !url.pathname.endsWith('.apk') || url.search || url.hash || url.username || url.password ||
        /%2f|%5c|%25/i.test(apk.url) || url.pathname.split('/').some(part => part === '..')) {
      throw new Error('Invalid Android download URL');
    }
    const approved = manifest.releases?.some((release) => release.versionCode === latest.versionCode &&
      release.sha256 === apk.sha256 && release.size === apk.size && release.signerSha256 === latest.signerSha256);
    if (!approved) throw new Error('Android download is absent from the approved release inventory');
    return { latest, url: url.href };
  }

  function description(latest) {
    const version = latest.versionName.replace(/-debug$/i, '');
    const size = (latest.apk.size / (1024 * 1024)).toFixed(1);
    const minimum = latest.minSdk === 26 ? 'Android 8+' : `Android API ${latest.minSdk}+`;
    return `Review build ${version} · ${size} MiB · ${minimum}`;
  }

  function mount(root = document) {
    root.querySelectorAll('[data-android-review-download]').forEach((link) => {
      if (mounted.has(link)) return;
      mounted.add(link);
      const controls = link.closest('.project-intro-actions');
      const status = controls?.querySelector('[data-android-release-status]');
      const retry = controls?.querySelector('[data-android-release-retry]');
      if (!status || !retry) return;

      async function check() {
        retry.hidden = true;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 8000);
        try {
          const response = await fetch(FEED_URL, { cache: 'no-store', signal: controller.signal });
          if (!response.ok) throw new Error('Android release feed unavailable');
          const { latest, url } = validateRelease(await response.json());
          if (latest.versionCode >= Number(link.dataset.releaseVersionCode || 0)) {
            link.href = url;
            link.dataset.releaseVersionCode = String(latest.versionCode);
            status.textContent = description(latest);
          }
        } catch (_) {
          status.textContent = 'Latest release check unavailable; the listed APK is still available.';
          retry.hidden = false;
        } finally {
          clearTimeout(timeout);
        }
      }

      retry.addEventListener('click', check);
      check();
    });
  }

  window.AndroidAppDownload = Object.freeze({ mount });
  document.addEventListener('DOMContentLoaded', () => mount());
  document.addEventListener('site:content-updated', (event) => mount(event.detail?.root || document));
  if (document.readyState !== 'loading') mount();
})();
