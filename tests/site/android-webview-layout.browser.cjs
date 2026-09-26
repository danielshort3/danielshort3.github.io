'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

// Exercise the installed app's real injection, including its important inline
// grid declaration. Ordinary website checks cannot detect this integration bug.
const nativeSource = fs.readFileSync(path.resolve(__dirname, '../../mobile/android/app/src/main/java/me/danielshort/app/ui/WebExperienceScreen.kt'), 'utf8');
const nativeTemplate = nativeSource.match(/view\.evaluateJavascript\("""\s*([\s\S]*?)"""\.trimIndent\(\)\)/)?.[1];
assert(nativeTemplate, 'Android feature injection is available for the browser regression.');
const nativeCss = name => {
  const value = nativeSource.match(new RegExp(`val ${name} =[^\\n]*\\n\\s*"([^"\\n]*)"`))?.[1];
  assert(value, `Native ${name} CSS is available.`);
  return value;
};
const nativeLayout = condition => {
  const value = nativeSource.slice(nativeSource.indexOf('val featureLayout = when')).match(new RegExp(`${condition} -> """([\\s\\S]*?)"""\\.trimIndent\\(\\)`))?.[1];
  assert(value, `Native feature layout is available: ${condition}`);
  return value;
};
const experiences = [
  { name: 'starfall', route: '/games/project-starfall', selector: '[data-starfall-root]', startCss: nativeCss('startCss'), featureLayout: nativeLayout('isStarfall') },
  { name: 'project', route: '/portfolio/website', selector: '.project-main', projectCss: nativeCss('projectCss') },
  { name: 'stellar', route: '/games/stellar-dogfight', selector: 'main', contentSelector: '#main', gameCss: nativeCss('gameCss'), featureLayout: nativeLayout('experience\\.canonicalPath == "/games/stellar-dogfight"') }
];

async function assertPaintedContent(page, selector, label) {
  const geometry = await page.locator(selector).evaluate(node => {
    const box = element => {
      const r = element.getBoundingClientRect();
      return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height };
    };
    const target = box(node);
    const visible = { left: Math.max(0, target.left), top: Math.max(0, target.top), right: Math.min(innerWidth, target.right), bottom: Math.min(innerHeight, target.bottom) };
    const clips = [];
    for (let parent = node.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent);
      const rect = box(parent);
      if (/hidden|clip|auto|scroll/.test(style.overflowX)) {
        visible.left = Math.max(visible.left, rect.left);
        visible.right = Math.min(visible.right, rect.right);
      }
      if (/hidden|clip|auto|scroll/.test(style.overflowY)) {
        visible.top = Math.max(visible.top, rect.top);
        visible.bottom = Math.min(visible.bottom, rect.bottom);
        clips.push({ className: parent.className, height: rect.height });
      }
    }
    visible.width = Math.max(0, visible.right - visible.left);
    visible.height = Math.max(0, visible.bottom - visible.top);
    // A normal mobile browser retains a fixed bottom dock, which can cover the
    // middle sample in a short landscape window. Sample the exposed region.
    const hits = [.15, .5, .85].map(fraction => document.elementFromPoint(
      (visible.left + visible.right) / 2, visible.top + visible.height * fraction));
    return { target, visible, clips, receivesInput: hits.some(hit => hit && node.contains(hit)), hits: hits.map(hit => hit?.className || hit?.tagName), stage: box(document.querySelector('.site-frame__stage')), overflow: document.documentElement.scrollWidth - innerWidth };
  });
  assert(geometry.visible.width >= 200 && geometry.visible.height >= 100,
    `${label}: content has a usable painted area after ancestor clipping: ${JSON.stringify(geometry)}`);
  assert(geometry.receivesInput, `${label}: visible content receives pointer input: ${JSON.stringify(geometry)}`);
  assert(geometry.overflow <= 1, `${label}: no horizontal page overflow.`);
  return geometry;
}

async function runAndroidWebViewLayoutChecks({ browser, base, artifactDir }) {
  const results = [];
  for (const viewport of [{ width: 390, height: 700 }, { width: 844, height: 390 }]) {
    const context = await browser.newContext({ viewport, hasTouch: true, reducedMotion: 'reduce', serviceWorkers: 'block' });
    await context.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    try {
      for (const experience of experiences) {
        const page = await context.newPage();
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(base + experience.route, { waitUntil: 'load' });
        await page.waitForFunction(() => window.SiteFrame?.root() && !window.SiteNavigation?.isNavigating?.());
        if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
        await page.evaluate(() => document.fonts.ready);
        const label = `${experience.name}-${viewport.width}x${viewport.height}`;
        // The same build must still provide normal browser content/navigation.
        await assertPaintedContent(page, experience.contentSelector || experience.selector, `${label} website`);
        assert(await page.locator('[data-site-shell-header], .mobile-site-masthead').evaluateAll(nodes => nodes.some(node => node.checkVisibility())), `${label}: website navigation remains visible.`);
        const injection = nativeTemplate.replace(/\$(startCss|projectCss|gameCss|featureLayout|selector)\b/g, (_, key) => experience[key] || '');
        assert.equal(await page.evaluate(injection), true, `${label}: native injection finds its target.`);
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const geometry = await assertPaintedContent(page, experience.contentSelector || experience.selector, `${label} Android`);
        assert.equal(await page.locator('[data-site-shell-header], .mobile-site-masthead, .site-frame__tab').evaluateAll(nodes => nodes.some(node => node.checkVisibility())), false, `${label}: native chrome hides duplicate website navigation.`);
        if (experience.name === 'stellar') {
          await page.locator('[data-action="launch"]').first().click();
          await page.waitForFunction(() => document.body.classList.contains('is-playing'));
          await assertPaintedContent(page, '[data-role="battlefield"]', `${label} Android playing`);
        }
        assert.deepEqual(errors, [], `${label}: no uncaught page errors.`);
        if (artifactDir) {
          fs.mkdirSync(artifactDir, { recursive: true });
          await page.screenshot({ path: path.join(artifactDir, `android-webview-${label}.png`) });
        }
        results.push({ label, visible: geometry.visible });
        await page.close();
      }
    } finally {
      await context.close();
    }
  }
  console.log(`Android WebView layout passed: ${results.length} route/viewport cases, normal browser content and active Stellar gameplay.`);
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'android-webview-layout-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runAndroidWebViewLayoutChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'android-webview-layout') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runAndroidWebViewLayoutChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
