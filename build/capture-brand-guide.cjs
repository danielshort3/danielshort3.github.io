'use strict';

// Refresh the guide's real website examples from an already-built local preview.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function main() {
  const root = path.resolve(__dirname, '..');
  const base = process.env.BRAND_GUIDE_URL || 'http://127.0.0.1:4173';
  const out = path.join(root, 'docs', 'brand-guide-assets');
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    for (const item of [
      { name: 'home-desktop', route: '/#closed', width: 1440, height: 900 },
      { name: 'projects-desktop', route: '/#projects', width: 1440, height: 900 },
      { name: 'text-compare-desktop', route: '/tools/text-compare', width: 1440, height: 900 },
      { name: 'tools-mobile', route: '/#tools', width: 390, height: 844 }
    ]) {
      const page = await browser.newPage({ viewport: { width: item.width, height: item.height },
        deviceScaleFactor: 1, serviceWorkers: 'block', reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/*', route => {
        const url = new URL(route.request().url());
        if (url.origin === new URL(base).origin && !url.pathname.startsWith('/api/')) return route.continue();
        return route.abort();
      });
      await page.goto(new URL(item.route, base).href, { waitUntil: 'networkidle' });
      if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
      await page.waitForFunction(() => window.SiteFrame?.root() && !document.querySelector('.site-frame--moving,.site-frame--held'));
      await page.evaluate(() => document.fonts.ready);
      await page.screenshot({ path: path.join(out, `${item.name}.png`) });
      if (errors.length) throw new Error(`${item.name}: ${errors.join('; ')}`);
      await page.close();
    }
    console.log(`Captured four current website examples in ${path.relative(root, out)}.`);
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
