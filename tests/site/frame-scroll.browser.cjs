/**
 * Optional native-wheel regression; intentionally excluded from npm test.
 * Requires an already built local server and an externally installed Playwright.
 * Does not build, start a server, submit forms, or write account data.
 *
 * PowerShell example:
 *   $env:PLAYWRIGHT_MODULE = '<absolute path to an installed playwright module>'
 *   $env:FRAME_SCROLL_URL = 'http://127.0.0.1:4173'
 *   node tests/site/frame-scroll.browser.cjs
 *
 * FRAME_SCROLL_ENGINES defaults to chromium,firefox,webkit.
 * FRAME_SCROLL_LABEL optionally identifies a run. Logs and failure screenshots
 * always stay under os.tmpdir()/site-frame-scroll. A failure exits nonzero.
 */
"use strict";
const assert = require("assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");
const engines = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.env.FRAME_SCROLL_URL || "http://127.0.0.1:4173";
const label = (process.env.FRAME_SCROLL_LABEL || "local").replace(/[^a-zA-Z0-9_.-]+/g, "-");
const artifactDir = path.join(os.tmpdir(), "site-frame-scroll");
const output = path.join(artifactDir, "frame-scroll-" + label);
const engineNames = (process.env.FRAME_SCROLL_ENGINES || "chromium,firefox,webkit").split(",").map((value) => value.trim());
const cases = [];
const errors = [];
fs.mkdirSync(artifactDir, { recursive: true });

function save() {
  fs.writeFileSync(output + ".json", JSON.stringify({ base, label, cases, errors }, null, 2));
}

async function settle(page) {
  await page.waitForFunction(() => {
    const frame = window.SiteFrame?.root();
    return frame && window.SiteRoutes?.current()?.root?.isConnected
      && !window.SiteNavigation?.isNavigating?.()
      && !frame.classList.contains("site-frame--moving")
      && !frame.classList.contains("site-frame--held")
      && SiteFrame.outlet()?.getAttribute("aria-busy") !== "true"
      && !document.querySelector("[data-site-frame-loading]:not([hidden])")
      && SiteFrame.viewport().getAnimations().every((animation) => animation.playState !== "running");
  });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function resetScroll(page) {
  await page.evaluate(() => {
    scrollTo({ top: 0, behavior: "instant" });
    SiteFrame.viewport().scrollTo({ top: 0, behavior: "instant" });
  });
  await page.waitForTimeout(80);
}

async function ready(page, route) {
  await page.goto(base + route, { waitUntil: "domcontentloaded" });
  await settle(page);
  if (await page.locator("#pcz-reject").isVisible()) {
    await page.locator("#pcz-reject").click();
    await page.waitForFunction(() => document.body.dataset.consentBanner !== "open");
  }
  await resetScroll(page);
}

async function read(page) {
  return page.evaluate(() => {
    const frame = SiteFrame.root();
    const viewport = SiteFrame.viewport();
    return {
      document: scrollY,
      documentRange: document.scrollingElement.scrollHeight - innerHeight,
      viewport: viewport.scrollTop,
      viewportRange: viewport.scrollHeight - viewport.clientHeight,
      fit: frame.dataset.frameFit,
      compact: frame.dataset.frameCompact,
      audience: frame.dataset.frameAudience
    };
  });
}

async function pointOn(page, selector) {
  return page.locator(selector).first().evaluate((node) => {
    const rect = node.getBoundingClientRect();
    const viewport = window.SiteFrame?.viewport();
    const clip = viewport?.contains(node) ? viewport.getBoundingClientRect() : { left: 0, top: 0, right: innerWidth, bottom: innerHeight };
    const left = Math.max(rect.left, clip.left, 0);
    const right = Math.min(rect.right, clip.right, innerWidth);
    const top = Math.max(rect.top, clip.top, 0);
    const bottom = Math.min(rect.bottom, clip.bottom, innerHeight);
    if (right - left < 2 || bottom - top < 2) throw new Error("Wheel target is outside the visible page: " + node.className);
    const point = { x: (left + right) / 2, y: (top + bottom) / 2 };
    const hit = document.elementFromPoint(point.x, point.y);
    if (!node.contains(hit)) throw new Error("Wheel target is covered by " + hit?.tagName + "." + hit?.className);
    return point;
  });
}

async function wheel(page, point, delta = 520) {
  await page.mouse.move(point.x, point.y);
  await page.mouse.wheel(0, delta);
  // Native wheel completion is asynchronous; allow the browser's own scrolling
  // and any smooth scrolling to settle before measuring, without setting scrollTop.
  await page.waitForTimeout(350);
}

async function expectScroll(page, point, owner) {
  const before = await read(page);
  assert.ok(before[owner + "Range"] > 20, "The fixture must have real " + owner + " scroll range.");
  await wheel(page, point);
  const after = await read(page);
  assert.ok(after[owner] - before[owner] > 20, "Native wheel did not move the " + owner + ": " + JSON.stringify({ before, after }));
  const other = owner === "viewport" ? "document" : "viewport";
  assert.ok(Math.abs(after[other] - before[other]) <= 1, "Wheel moved the wrong scroll owner.");
  return { before, after };
}

async function runCase(page, engine, name, callback) {
  try {
    const details = await callback();
    cases.push({ engine, name, pass: true, ...details });
    console.log(JSON.stringify({ engine, name, pass: true }));
  } catch (error) {
    cases.push({ engine, name, pass: false, message: error.message, url: page.url() });
    await page.screenshot({ path: output + "-" + engine + "-" + name.replace(/[^a-zA-Z0-9_-]/g, "-") + ".png" }).catch(() => {});
    console.error(JSON.stringify({ engine, name, pass: false, message: error.message }));
  }
  save();
}

async function runEngine(engine) {
  assert.ok(engines[engine]?.launch, "Unsupported FRAME_SCROLL_ENGINES value: " + engine);
  const browser = await engines[engine].launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  page.on("pageerror", (error) => {
    // Embedded demos may report unavailable local backend/model errors. Keep
    // their errors in the artifact without confusing them with scroll failures.
    errors.push({ engine, message: error.message, url: page.url() });
  });
  try {
    for (const route of ["/tools/text-compare", "/tools/qr-code-generator", "/tools", "/portfolio", "/portfolio/digitGenerator", "/contact", "/privacy"]) {
      await runCase(page, engine, "desktop-content-" + route, async () => {
        await ready(page, route);
        const state = await read(page);
        assert.equal(state.fit, "viewport");
        assert.equal(state.compact, "false");
        return expectScroll(page, await pointOn(page, ".site-frame__body h1"), "viewport");
      });
    }
    await runCase(page, engine, "homepage-timeline", async () => {
      await ready(page, "/");
      return expectScroll(page, await pointOn(page, "[data-home-timeline-scroller]"), "viewport");
    });
    for (const surface of ["gutter", "header", "rail", "toolbar"]) {
      await runCase(page, engine, "desktop-shell-" + surface, async () => {
        await ready(page, "/tools/text-compare");
        const selector = { header: "[data-site-shell-header] .nav", rail: '[data-site-tab="tools"]', toolbar: "[data-site-route-toolbar]" }[surface];
        const point = selector ? await pointOn(page, selector) : { x: 2, y: 450 };
        return expectScroll(page, point, "viewport");
      });
    }
    await runCase(page, engine, "keyboard-page-down", async () => {
      await ready(page, "/tools/text-compare");
      const point = await pointOn(page, ".site-frame__body h1");
      await page.mouse.click(point.x, point.y);
      const before = await read(page);
      await page.keyboard.press("PageDown");
      await page.waitForTimeout(350);
      const after = await read(page);
      assert.ok(after.viewport - before.viewport > 20, "PageDown must scroll the content after clicking inside it.");
      assert.ok(Math.abs(after.document - before.document) <= 1, "PageDown moved the document instead of the frame.");
      return { before, after };
    });
    await runCase(page, engine, "nested-textarea", async () => {
      await ready(page, "/tools/text-compare");
      const textarea = page.locator("#textcompare-original");
      await textarea.fill(Array.from({ length: 100 }, (_, index) => "Local scroll fixture line " + index).join("\n"));
      await textarea.scrollIntoViewIfNeeded();
      await textarea.evaluate((node) => { node.scrollTop = 0; });
      await page.waitForTimeout(100);
      const before = await read(page);
      await wheel(page, await pointOn(page, "#textcompare-original"), 240);
      const textareaTop = await textarea.evaluate((node) => node.scrollTop);
      const after = await read(page);
      assert.ok(textareaTop > 20, "Textarea must retain its native scrolling.");
      assert.ok(Math.abs(after.viewport - before.viewport) <= 1, "Textarea wheel moved the frame.");
      assert.ok(Math.abs(after.document - before.document) <= 1, "Textarea wheel moved the document.");
      return { before, after, textareaTop };
    });
    await runCase(page, engine, "modal-background-isolation", async () => {
      await page.setViewportSize({ width: 1440, height: 660 });
      await ready(page, "/contact");
      await page.locator("#contact-form-toggle").click();
      await page.locator("#contact-modal.active").waitFor({ state: "visible" });
      await page.waitForTimeout(350);
      const before = await read(page);
      await wheel(page, await pointOn(page, "#contact-modal .modal-title-strip"));
      await wheel(page, { x: 2, y: 330 });
      const after = await read(page);
      assert.ok(Math.abs(after.viewport - before.viewport) <= 1, "Modal wheel moved the background frame.");
      assert.ok(Math.abs(after.document - before.document) <= 1, "Modal wheel moved the background document.");
      await page.locator("#contact-modal .modal-close").click();
      await page.locator("#contact-modal.active").waitFor({ state: "hidden" });
      return { before, after };
    });
    await page.setViewportSize({ width: 1440, height: 900 });
    await runCase(page, engine, "soft-navigation-and-back", async () => {
      await ready(page, "/tools");
      await page.evaluate(() => { window.scrollQaIdentity = { frame: SiteFrame.root(), viewport: SiteFrame.viewport(), time: performance.timeOrigin }; });
      await page.locator('.home-library__card[href="/tools/text-compare"]').click();
      await page.waitForURL("**/tools/text-compare");
      await settle(page);
      await resetScroll(page);
      const detail = await expectScroll(page, await pointOn(page, ".site-frame__body h1"), "viewport");
      await page.goBack();
      await page.waitForURL("**/tools");
      await settle(page);
      await resetScroll(page);
      const back = await expectScroll(page, { x: 2, y: 450 }, "viewport");
      const persistent = await page.evaluate(() => scrollQaIdentity.frame === SiteFrame.root() && scrollQaIdentity.viewport === SiteFrame.viewport() && scrollQaIdentity.time === performance.timeOrigin);
      assert.ok(persistent, "Navigation must keep the persistent frame and document.");
      return { detail, back, persistent };
    });
    for (const size of [{ width: 390, height: 844 }, { width: 1440, height: 580 }]) {
      await page.setViewportSize(size);
      for (const route of ["/tools/text-compare", "/privacy", "/"]) {
        await runCase(page, engine, "document-" + size.width + "x" + size.height + "-" + route, async () => {
          await ready(page, route);
          assert.equal((await read(page)).compact, "true");
          const selector = route === "/" ? "[data-home-timeline-scroller]" : ".site-frame__body h1";
          return expectScroll(page, await pointOn(page, selector), "document");
        });
      }
    }
    await page.setViewportSize({ width: 1440, height: 900 });
    for (const route of ["/contact?audience=analytics", "/portfolio?audience=analytics", "/portfolio/digitGenerator?audience=analytics"]) {
      await runCase(page, engine, "professional-document-" + route, async () => {
        await ready(page, route);
        const state = await read(page);
        assert.equal(state.audience, "analytics");
        assert.equal(state.fit, "document");
        return expectScroll(page, await pointOn(page, ".site-frame__body h1"), "document");
      });
    }
  } finally {
    await context.close();
    await browser.close();
  }
}

(async () => {
  for (const engine of engineNames) await runEngine(engine);
  const failed = cases.filter((record) => !record.pass).length;
  save();
  console.log(JSON.stringify({ cases: cases.length, failed, pageErrors: errors.length, artifact: output + ".json" }));
  if (failed) process.exitCode = 1;
})().catch((error) => {
  save();
  console.error(error);
  process.exitCode = 1;
});
