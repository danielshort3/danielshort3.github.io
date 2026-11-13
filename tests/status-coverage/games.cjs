'use strict';

const assert = require('node:assert/strict');
const { expect } = require('@playwright/test');
const { local, unknown, visible, text, status, visit } = require('./tools.cjs');

async function starfall({ page, base, item }) {
  await page.setViewportSize({ width: 390, height: 844 });
  await visit(page, base, item.route);
  const start = page.locator('[data-starfall-start-screen] [data-starfall-action="load"]');
  await expect(start).toBeVisible({ timeout: 15_000 });
  await start.click();
  await page.locator('[data-starfall-character-slot]').first().click({ timeout: 15_000 });
  await page.locator('[data-starfall-character-create-open]').click();
  await page.locator('[data-starfall-character-name]').fill('Fixture Pilot');
  await page.locator('[data-starfall-character-create-confirm]').click();
  await page.locator('[data-starfall-character-class="fighter"]').click();
  await page.locator('[data-starfall-character-create-confirm]').click();
  await expect.poll(() => page.evaluate(() =>
    document.querySelector('[data-starfall-root]').ProjectStarfall?.engine?.running)).toBe(true);

  const right = page.locator('[data-starfall-touch-controls] [data-starfall-touch-action="moveRight"]');
  await expect(right).toBeVisible();
  await right.scrollIntoViewIfNeeded();
  const bounds = await right.boundingBox();
  assert(bounds);
  const touch = await page.context().newCDPSession(page);
  await touch.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 2 });
  await touch.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{
    x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2, id: 1
  }] });
  await expect(right).toHaveAttribute('aria-pressed', 'true');
  assert.equal(await page.evaluate(() =>
    document.querySelector('[data-starfall-root]').ProjectStarfall.engine.input.right), true);
  await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect(right).toHaveAttribute('aria-pressed', 'false');
  return local('A new fighter run started and handled a real held and released movement input.');
}

async function dogfight({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('[data-action="launch"]').first().click();
  await expect(page.locator('body')).toHaveClass(/\bis-playing\b/);
  await visible(page, '[data-role="battlefield"]');
  await page.keyboard.press('Escape');
  await text(page, '[data-stat="status"]', 'Paused');
  return local('Launching the campaign created an active flight and Escape paused the game.');
}

async function roulette({ context, page, base, item }) {
  await context.addInitScript(() => {
    Object.defineProperty(window.crypto, 'getRandomValues', {
      value: (values) => { values.fill(0); return values; }
    });
    Math.random = () => 0;
  });
  await visit(page, base, item.route);
  await page.locator('[data-bet-id="straight-0"]').click();
  await page.locator('#roulette-spin').click();
  await text(page, '#roulette-spin-count', '1');
  await text(page, '#roulette-last-pocket', '0');
  return local('A seeded straight-zero spin settled on the expected zero pocket.');
}

async function probability({ context, page, base, item }) {
  await context.addInitScript(() => { Math.random = () => 0.125; });
  await visit(page, base, item.route);
  await page.locator('#spin-button').click();
  await status(page, '#spin-result-status', 'Credits');
  assert.match(await page.locator('#spin-outcome-title').textContent(), /\S/);
  return local('A seeded spin completed and the browser announced its credits result.');
}

async function stormbreak({ page, base, item }) {
  await page.clock.install();
  await visit(page, base, item.route);
  await visible(page, '#stormbreak-canvas');
  await page.clock.runFor(18_000);
  const readGold = () => page.locator('[data-bind="gold"]').evaluate((node) => Number(node.textContent.replace(/,/g, '')));
  assert((await readGold()) > 0);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.clock.runFor(1000);
  assert((await readGold()) > 0);
  return local('The fixture battle earned gold and retained positive progress on reload.');
}

async function ocean({ page, base, item }) {
  await visit(page, base, item.route);
  await visible(page, '#ocean-wave-canvas');
  const before = await page.locator('#ocean-wave-scene-name').textContent();
  await page.locator('#ocean-wave-scene').selectOption('cove');
  await expect(page.locator('#ocean-wave-scene-name')).not.toHaveText(before);
  return local('The canvas scene changed when the user selected Quiet cove.');
}

async function chasing({ page, base, item }) {
  await page.clock.install();
  await visit(page, base, item.route);
  await page.locator('[data-cl-predict="c"]').click();
  await page.locator('[data-cl-launch]').click();
  await page.keyboard.press('Space');
  await page.clock.runFor(1200);
  await text(page, '[data-cl-delta-distance]', '1.00 light-s');
  await text(page, '[data-cl-delta-time]', '1.00 s');
  await text(page, '[data-cl-delta-speed]', '1.000c');
  return local('The guided pulse produced the expected measured distance, time and speed.');
}

async function wayfarers({ page, base, item }) {
  await page.clock.install({ time: new Date('2026-11-11T18:00:00Z') });
  await visit(page, base, item.route);
  const guide = page.locator('.wx-guide[open]');
  await page.clock.runFor(1200);
  await expect(guide).toHaveAttribute('data-step', 'currency:coins');
  await page.locator('[data-guide-next]').click();
  await page.clock.runFor(100);
  await expect(guide).toHaveAttribute('data-step', 'inspect');
  await status(page, '#wx-guide-heading', 'Pathfinding');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.clock.runFor(1200);
  await expect(guide).toHaveAttribute('data-step', 'inspect');
  return local('The first coin lesson advanced to Pathfinding and kept its progress after a reload.');
}

module.exports = {
  'project-starfall': starfall,
  'stellar-dogfight': dogfight,
  roulette,
  'probability-engine': probability,
  stormbreak,
  'ocean-wave-simulation': ocean,
  'chasing-light': chasing,
  'wayfarers-guild': wayfarers,
  fallback: () => unknown('This game has no deterministic local feature fixture yet.')
};
