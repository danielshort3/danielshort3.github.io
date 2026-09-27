const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "pages/games/roulette.html"), "utf8");
const source = fs.readFileSync(path.join(root, "js/games/roulette/app.js"), "utf8");
const game = html.match(/<main id="main"[\s\S]*?<\/main>/);
assert(game, "roulette game markup exists");
const mobileBar = '<div class="roulette00-mobile-bar"><button id="roulette-mobile-spin" type="button">Spin</button></div>';

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.addInitScript(() => {
      const nativeMatchMedia = window.matchMedia.bind(window);
      window.matchMedia = (query) => query.includes("prefers-reduced-motion")
        ? { matches: true, addListener() {}, removeListener() {} }
        : nativeMatchMedia(query);
      Object.defineProperty(window, "crypto", {
        configurable: true,
        value: { getRandomValues(values) { values[0] = 0; return values; } }
      });
    });
    await page.route("http://roulette.test/", (route) => route.fulfill({
      status: 200,
      contentType: "text/html",
      body: `<!doctype html><html><body>${game[0]}${mobileBar}</body></html>`
    }));
    await page.goto("http://roulette.test/");
    await page.addScriptTag({ content: source });

    assert.deepEqual(await page.locator(".roulette00-number-row").count(), 3);
    assert.deepEqual(await page.locator(".roulette00-number-row").first()
      .locator("[data-bet-id]").evaluateAll((buttons) => buttons.map((button) => button.dataset.betId)),
    ["straight-3", "straight-6", "straight-9", "straight-12", "straight-15", "straight-18",
      "straight-21", "straight-24", "straight-27", "straight-30", "straight-33", "straight-36", "column-3"]);
    assert.equal(await page.locator('[data-bet-id="straight-1"]').evaluate((button) =>
      button.style.getPropertyValue("--mobile-order")), "1");
    assert.equal(await page.locator('[data-bet-id="column-3"]').evaluate((button) =>
      button.style.getPropertyValue("--mobile-order")), "39");

    assert.equal(await page.locator(".roulette00-line-tab").count(), 4);
    assert.equal(await page.locator(".roulette00-bet.is-line").count(), 110);
    await page.locator("#roulette-inside-tab-0").focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(await page.locator("#roulette-inside-tab-1").getAttribute("aria-selected"), "true");
    await page.locator("#roulette-inside-tab-0").click();
    assert.equal(await page.locator("#roulette-double").isDisabled(), true);
    await page.locator('[data-bet-id="split-0-00"]').hover();
    assert.deepEqual(await page.locator(".roulette00-bet.is-preview-covered")
      .evaluateAll((buttons) => buttons.map((button) => button.dataset.betId).sort()),
    ["straight-0", "straight-00"]);
    await page.locator("#roulette-spin").hover();
    assert.equal(await page.locator(".roulette00-bet.is-preview-covered").count(), 0);
    await page.locator('[data-bet-id="split-0-00"]').focus();
    assert.equal(await page.locator(".roulette00-bet.is-preview-covered").count(), 2);
    await page.locator("#roulette-spin").focus();
    assert.equal(await page.locator(".roulette00-bet.is-preview-covered").count(), 0);
    await page.locator("#roulette-inside-tab-2").click();
    await page.locator('[data-bet-id="corner-1-2-4-5"]').focus();
    assert.deepEqual(await page.locator(".roulette00-bet.is-preview-covered")
      .evaluateAll((buttons) => buttons.map((button) => button.dataset.betId).sort()),
    ["straight-1", "straight-2", "straight-4", "straight-5"]);
    await page.locator("#roulette-inside-tab-0").click();

    await page.locator('[data-chip="25"]').click();
    await page.locator('[data-bet-id="split-0-00"]').click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$25");
    assert.match(await page.locator("#roulette-spin-status").innerText(), /Split 0, 00.*pays 17:1/);
    await page.locator("#roulette-double").click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$50");
    assert.equal(await page.locator("#roulette-bankroll").innerText(), "$1,950");
    await page.locator("#roulette-undo").click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$25");
    assert.equal(await page.locator("#roulette-bankroll").innerText(), "$1,975");

    await page.locator('[data-chip="100"]').click();
    await page.locator('[data-wager-mode="remove"]').click();
    assert.match(await page.locator('[data-bet-id="split-0-00"]').getAttribute("aria-label"),
      /Activate to remove \$25\./);
    await page.locator('[data-bet-id="split-0-00"]').click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$0",
      "remove returns the smaller remaining stake even with a larger chip selected");
    await page.locator("#roulette-undo").click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$25");

    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(() => document.querySelector("#roulette-number-grid [data-bet-id]")
      .dataset.betId === "straight-1");
    assert.equal(await page.locator(".roulette00-number-row").count(), 13);
    assert.deepEqual(await page.locator("#roulette-number-grid [data-bet-id]")
      .evaluateAll((buttons) => buttons.slice(0, 3).map((button) => button.dataset.betId)),
    ["straight-1", "straight-2", "straight-3"]);
    await page.locator('[data-bet-id="straight-1"]').focus();
    await page.keyboard.press("Tab");
    assert.equal(await page.evaluate(() => document.activeElement.dataset.betId), "straight-2");
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$25");

    await page.evaluate(() => {
      const wheelCard = document.querySelector(".roulette00-wheel-card");
      window.__wheelInView = false;
      wheelCard.getBoundingClientRect = () => window.__wheelInView
        ? ({ top: 0, bottom: 240 })
        : ({ top: -240, bottom: -1 });
      wheelCard.scrollIntoView = (options) => {
        window.__wheelScroll = options;
        window.__wheelScrollCount = (window.__wheelScrollCount || 0) + 1;
        window.__wheelInView = true;
        window.dispatchEvent(new Event("scroll"));
      };
      window.dispatchEvent(new Event("scroll"));
    });
    await page.waitForFunction(() => document.querySelector(".roulette00-mobile-bar")
      .classList.contains("is-visible"));
    await page.locator("#roulette-mobile-spin").click();
    await page.waitForFunction(() => document.getElementById("roulette-spin-count").textContent === "1");
    assert.equal(await page.evaluate(() => window.__wheelScroll.block), "start");
    assert.equal(await page.evaluate(() => window.__wheelScroll.behavior), "auto");
    assert.equal(await page.evaluate(() => window.__wheelScrollCount), 2,
      "mobile result returns to the top of the wheel card");
    await page.waitForFunction(() => !document.querySelector(".roulette00-mobile-bar")
      .classList.contains("is-visible"));
    assert.equal(await page.evaluate(() => document.activeElement.classList.contains("roulette00-wheel-card")),
      true, "focus leaves the mobile bar when that bar is hidden");
    assert.equal(await page.locator("#roulette-last-pocket").innerText(), "0");
    assert.equal(await page.locator("#roulette-bankroll").innerText(), "$2,425");
    assert.match(await page.locator("#roulette-payout-breakdown").innerText(),
      /\$25 staked.*\$450 returned.*\+\$425 net/s);
    assert.match(await page.locator("#roulette-payout-breakdown").innerText(),
      /Split 0, 00: \$25 × 18 = \$450/);
    assert.equal(await page.locator("#roulette-round-phase").innerText(), "Result");
    assert.equal(await page.locator(".roulette00-wheel-label.is-winning").innerText(), "0");

    await page.setViewportSize({ width: 1200, height: 800 });
    await page.waitForFunction(() => document.querySelector("#roulette-number-grid [data-bet-id]")
      .dataset.betId === "straight-3");
    assert.equal(await page.locator(".roulette00-number-row").count(), 3);
    assert.equal(await page.locator("#roulette-bankroll").innerText(), "$2,425");

    await page.locator("#roulette-rebet").click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$25");
    await page.locator("#roulette-undo").click();
    assert.equal(await page.locator("#roulette-total-bet").innerText(), "$0");
    assert.equal(await page.locator("#roulette-bankroll").innerText(), "$2,425");

    await page.evaluate(() => window.dispatchEvent(new Event("beforeunload")));
    const saved = await page.evaluate(() => JSON.parse(
      localStorage.getItem("roulette-double-zero-session-v1")));
    assert.equal(saved.bankroll, 2425);
    assert.equal(saved.lastPocket, "0");
    assert.deepEqual(saved.lastBetSnapshot, [["split-0-00", 25]]);
    await page.reload();
    await page.addScriptTag({ content: source });
    assert.equal(await page.locator("#roulette-bankroll").innerText(), "$2,425");
    assert.match(await page.locator("#roulette-payout-breakdown").innerText(),
      /Split 0, 00: \$25 × 18 = \$450/);

    const motionPage = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    await motionPage.addInitScript(() => {
      const nativeMatchMedia = window.matchMedia.bind(window);
      let reduceMotion = false;
      const motionListeners = [];
      const motionQuery = {
        get matches() { return reduceMotion; },
        addListener(callback) { motionListeners.push(callback); },
        removeListener() {},
        addEventListener(type, callback) {
          if (type === "change") {
            motionListeners.push(callback);
          }
        },
        removeEventListener() {}
      };
      window.matchMedia = (query) => query.includes("prefers-reduced-motion")
        ? motionQuery
        : nativeMatchMedia(query);
      window.__enableReducedMotion = () => {
        reduceMotion = true;
        motionListeners.forEach((callback) => callback({ matches: true }));
      };
      Object.defineProperty(window, "crypto", {
        configurable: true,
        value: { getRandomValues(values) { values[0] = 0; return values; } }
      });
      Math.random = () => 0;
      let frameId = 0;
      const frames = new Map();
      window.requestAnimationFrame = (callback) => {
        frameId += 1;
        frames.set(frameId, callback);
        return frameId;
      };
      window.cancelAnimationFrame = (id) => frames.delete(id);
      window.__advanceSpinFrame = (timestamp) => {
        const callbacks = Array.from(frames.values());
        frames.clear();
        callbacks.forEach((callback) => callback(timestamp));
      };
    });
    await motionPage.route("http://roulette-motion.test/", (route) => route.fulfill({
      status: 200,
      contentType: "text/html",
      body: `<!doctype html><html><body>${game[0]}${mobileBar}</body></html>`
    }));
    await motionPage.goto("http://roulette-motion.test/");
    await motionPage.addStyleTag({ content: ".roulette00-wheel { width: 380px; height: 380px; }" });
    await motionPage.addScriptTag({ content: source });
    await motionPage.locator('[data-bet-id="straight-0"]').click();
    await motionPage.locator("#roulette-spin").click();
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "0",
      "the result is not paid before the ball lands");
    assert.equal(await motionPage.locator("#roulette-spin").isDisabled(), true);
    assert.equal(await motionPage.locator("#roulette-round-phase").innerText(), "No more bets");
    await motionPage.evaluate(() => window.__advanceSpinFrame(1000));
    await motionPage.evaluate(() => window.__advanceSpinFrame(3500));
    assert.equal(await motionPage.locator(".roulette00-wheel-wrap").getAttribute("data-phase"), "orbit");
    assert.equal(await motionPage.locator("#roulette-round-phase").innerText(), "Ball circling");
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "0");
    await motionPage.evaluate(() => window.__advanceSpinFrame(6800));
    assert.equal(await motionPage.locator(".roulette00-wheel-wrap").getAttribute("data-phase"), "drop");
    assert.equal(await motionPage.locator("#roulette-round-phase").innerText(), "Ball dropping");
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "0");
    await motionPage.evaluate(() => window.__advanceSpinFrame(9500));
    await motionPage.waitForFunction(() => document.getElementById("roulette-spin-count").textContent === "1");
    assert.equal(await motionPage.locator(".roulette00-wheel-wrap").getAttribute("data-phase"), "locked");
    assert.equal(await motionPage.locator("#roulette-round-phase").innerText(), "Result");
    assert.equal(await motionPage.locator("#roulette-last-pocket").innerText(), "0");
    assert.equal(await motionPage.locator("#roulette-spin").isEnabled(), true);
    const landing = await motionPage.evaluate(() => {
      const wheel = document.getElementById("roulette-wheel");
      const ball = document.getElementById("roulette-ball");
      const wheelAngle = Number(wheel.style.transform.match(/rotate\(([-\d.]+)deg\)/)[1]);
      const ballAngle = Number(ball.style.transform.match(/rotate\(([-\d.]+)deg\)/)[1]);
      return ((ballAngle - wheelAngle) % 360 + 360) % 360;
    });
    assert(Math.abs(landing - (360 / 38 / 2)) < 0.01,
      `the visible ball aligns to the paid pocket (actual ${landing})`);
    assert.equal(await motionPage.locator(".roulette00-wheel-label.is-winning").innerText(), "0");

    await motionPage.locator("#roulette-rebet").click();
    await motionPage.locator("#roulette-spin").click();
    assert.equal(await motionPage.locator(".roulette00-wheel-label.is-winning").count(), 0);
    await motionPage.evaluate(() => window.__advanceSpinFrame(11000));
    const interruptedSave = await motionPage.evaluate(() => localStorage.getItem(
      "roulette-double-zero-session-v1"));
    assert.equal(JSON.parse(interruptedSave).pendingSpin.winningPocket, "0");
    const recoveryPage = await browser.newPage();
    await recoveryPage.addInitScript((saved) => {
      localStorage.setItem("roulette-double-zero-session-v1", saved);
    }, interruptedSave);
    await recoveryPage.route("http://roulette-motion.test/", (route) => route.fulfill({
      status: 200,
      contentType: "text/html",
      body: `<!doctype html><html><body>${game[0]}${mobileBar}</body></html>`
    }));
    await recoveryPage.goto("http://roulette-motion.test/");
    await recoveryPage.addScriptTag({ content: source });
    assert.equal(await recoveryPage.locator("#roulette-spin-count").innerText(), "2");
    assert.equal(await recoveryPage.locator("#roulette-last-pocket").innerText(), "0");
    assert.equal(await recoveryPage.locator("#roulette-bankroll").innerText(), "$2,070");
    assert.equal(await recoveryPage.locator("#roulette-round-phase").innerText(), "Result");
    assert.match(await recoveryPage.locator("#roulette-spin-status").innerText(), /Interrupted spin completed/);
    assert.equal(await recoveryPage.evaluate(() => JSON.parse(localStorage.getItem(
      "roulette-double-zero-session-v1")).pendingSpin), null);
    await motionPage.evaluate(() => window.dispatchEvent(new Event("pagehide")));
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "2");
    assert.equal(await motionPage.locator("#roulette-spin").isEnabled(), true);
    assert.equal(await motionPage.evaluate(() => JSON.parse(
      localStorage.getItem("roulette-double-zero-session-v1")).spins), 2,
    "page exit saves the settled result");
    await motionPage.evaluate(() => window.__advanceSpinFrame(30000));
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "2",
      "page exit and animation completion cannot pay the same spin twice");
    await motionPage.locator("#roulette-rebet").click();
    await motionPage.locator("#roulette-spin").click();
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "2");
    await motionPage.evaluate(() => window.__enableReducedMotion());
    assert.equal(await motionPage.locator("#roulette-spin-count").innerText(), "3",
      "a newly enabled reduced-motion preference completes the active spin");
    await motionPage.locator('[data-chip="100"]').click();
    for (let index = 0; index < 20; index += 1) {
      await motionPage.locator('[data-bet-id="straight-1"]').click();
    }
    const beforeFailedDouble = await motionPage.locator("#roulette-bankroll").innerText();
    await motionPage.locator("#roulette-double").click();
    assert.equal(await motionPage.locator("#roulette-total-bet").innerText(), "$2,000");
    assert.equal(await motionPage.locator("#roulette-bankroll").innerText(), beforeFailedDouble);
    assert.match(await motionPage.locator("#roulette-spin-status").innerText(), /Insufficient bankroll to double/);
    await motionPage.locator("#roulette-new-session").click();
    await motionPage.locator("#roulette-new-session").click();
    assert.equal(await motionPage.locator("#roulette-round-phase").innerText(), "Ready");
    assert.equal(await motionPage.locator(".roulette00-wheel-label.is-winning").count(), 0);
    assert.equal(await motionPage.locator("#roulette-bankroll").innerText(), "$2,000");
    console.log("Roulette browser betting, restore, frame motion, pocket lock, and exit settlement: passed");
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
