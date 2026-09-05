/**
 * Optional rendered geometry regression; intentionally excluded from npm test.
 * Requires an already built local server and an externally installed Playwright.
 * It does not build, start a server, submit a form, or write account data.
 *
 * PowerShell example:
 *   $env:PLAYWRIGHT_MODULE = '<absolute path to an installed playwright module>'
 *   $env:FRAME_SEAM_URL = 'http://127.0.0.1:4173'
 *   $env:FRAME_SEAM_ARTIFACT_DIR = '<temporary output directory>'
 *   node tests/site/frame-geometry.browser.cjs
 *
 * Optional controls:
 *   FRAME_SEAM_ENGINES=chromium,firefox,webkit
 *   FRAME_SEAM_SIZES=[{"width":2304,"height":1296},{"width":390,"height":844}]
 *   FRAME_SEAM_GROUPS=tabs,routes,stress,held,typography
 *   FRAME_SEAM_LABEL=local
 *
 * Defaults: all three engines; 1440x900, 2304x1296, 390x844, 844x390;
 * all groups; artifacts under os.tmpdir()/site-frame-geometry.
 * The tabs group covers every directed pair, including reverse/nonadjacent moves.
 * Stress/held groups deliberately delay responses or mounting, rotate, and
 * change reduced-motion preference while the next route is still mounting.
 * Typography checks actual SVG/label bounds at normal and 200% root font size.
 * Each animation frame records geometry, all four exposed border edges,
 * clipping, footer containment, loader visibility, and persistent node identity.
 * Settled routes must restore the original inline document min-height exactly.
 * Raw traces and compact summaries are checkpointed every ten cases.
 * A failure exits nonzero.
 */
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const { chromium, firefox, webkit } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.env.FRAME_SEAM_URL || "http://127.0.0.1:4173";
const run = (process.env.FRAME_SEAM_LABEL || "local").replace(/[^a-zA-Z0-9_.-]+/g, "-");
const artifactDir = process.env.FRAME_SEAM_ARTIFACT_DIR || path.join(os.tmpdir(), "site-frame-geometry");
fs.mkdirSync(artifactDir, { recursive: true });
const output = path.join(artifactDir, "frame-seam-qa-" + run);
const engineNames = (process.env.FRAME_SEAM_ENGINES || "chromium,firefox,webkit").split(",").map((value) => value.trim());
const groups = new Set((process.env.FRAME_SEAM_GROUPS || "tabs,routes,stress,held,typography").split(",").map((value) => value.trim()));
const sizes = process.env.FRAME_SEAM_SIZES ? JSON.parse(process.env.FRAME_SEAM_SIZES) : [{ width: 1440, height: 900 }, { width: 2304, height: 1296 }, { width: 390, height: 844 }, { width: 844, height: 390 }];
const cases = [], errors = [], typography = [];
const save = (force = false) => {
  if (!force && cases.length % 10 !== 0) return;
  fs.writeFileSync(output + ".json", JSON.stringify({ base, run, cases, errors, typography }));
  const viewports = [...new Set(cases.map((record) => record.prefix))].map((prefix) => {
    const records = cases.filter((record) => record.prefix === prefix);
    const peak = (key) => Math.max(0, ...records.map((record) => record.analysis[key] || 0));
    return {
      prefix,
      cases: records.length,
      samples: records.reduce((sum, record) => sum + record.samples.length, 0),
      failures: records.flatMap((record) => {
        const checks = Object.entries(record.analysis.checks).filter(([, pass]) => !pass).map(([name]) => name);
        return checks.length ? [{ name: record.name, checks }] : [];
      }),
      maxGap: peak("maxGap"),
      maxFixedDrift: peak("fixedDrift"),
      maxEndpointOvershoot: peak("envelope"),
      maxPaintOverflow: peak("overflow"),
      maxContentOverflow: peak("slotOverflow")
    };
  });
  fs.writeFileSync(output + "-summary.json", JSON.stringify({ base, run, cases: cases.length, viewports, typography, errors }, null, 2));
};
function tabTrail() {
  const ids = ["about", "projects", "tools", "games", "contact"];
  const edges = new Map(ids.map((id) => [id, ids.filter((x) => x !== id)]));
  const stack = ["about"], trail = [];
  while (stack.length) {
    const id = stack.at(-1), next = edges.get(id);
    if (next.length) stack.push(next.shift());
    else trail.push(stack.pop());
  }
  return trail.reverse();
}
async function settle(page) {
  const idle = () => window.SiteFrame?.root() && window.SiteRoutes?.current()?.root?.isConnected && !window.SiteNavigation?.isNavigating?.() && !SiteFrame.root().classList.contains("site-frame--moving") && !SiteFrame.root().classList.contains("site-frame--held") && SiteFrame.outlet()?.getAttribute("aria-busy") !== "true" && !document.querySelector("[data-site-frame-loading]:not([hidden])") && SiteFrame.viewport().getAnimations().every((a) => a.playState !== "running");
  await page.waitForFunction(idle);
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  // Back may have scheduled navigation after an earlier idle observation.
  // Never use an in-progress return transition as the next case's endpoint.
  await page.waitForFunction(idle);
}
async function ready(page, path2 = "/") {
  await page.goto(base + path2, { waitUntil: "domcontentloaded" });
  await settle(page);
  if (await page.locator("#pcz-reject").isVisible()) {
    await page.locator("#pcz-reject").click();
    await page.waitForFunction(() => document.body.dataset.consentBanner !== "open");
  }
  await page.evaluate(() => scrollTo(0, 0));
}
async function install(page) {
  await page.evaluate(() => {
    const frame = SiteFrame.root(), stage = frame.querySelector("[data-site-frame-stage]"), panel = frame.querySelector("[data-site-route-panel]"), slot = frame.querySelector("[data-site-frame-slot]");
    window.seamQA = { frame, stage, panel, slot, viewport: SiteFrame.viewport(), toolbar: frame.querySelector("[data-site-route-toolbar]"), tabs: [...frame.querySelectorAll("[data-site-tab]")], time: performance.timeOrigin, active: null };
    seamQA.rootMinHeight = { value: document.documentElement.style.getPropertyValue("min-height"), priority: document.documentElement.style.getPropertyPriority("min-height") };
    const rect = (n) => {
      if (!n) return null;
      const r = n.getBoundingClientRect();
      return { x: r.x, y: r.y + scrollY, w: r.width, h: r.height, right: r.right, bottom: r.bottom + scrollY };
    };
    const css = (n, pseudo) => {
      const s = getComputedStyle(n, pseudo);
      return { opacity: +s.opacity, display: s.display, visibility: s.visibility, background: s.backgroundColor, bt: parseFloat(s.borderTopWidth), bb: parseFloat(s.borderBottomWidth), bl: parseFloat(s.borderLeftWidth), br: parseFloat(s.borderRightWidth), border: s.borderTopColor, overflowX: s.overflowX, overflowY: s.overflowY };
    };
    const gaps = (start2, end, intervals) => {
      const sorted = intervals.map(([a, b]) => [Math.max(start2, a), Math.min(end, b)]).filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
      let covered = start2;
      const uncovered = [];
      for (const [a, b] of sorted) {
        if (a > covered + 0.5) uncovered.push([covered, a]);
        covered = Math.max(covered, b);
      }
      if (end > covered + 0.5) uncovered.push([covered, end]);
      return uncovered;
    };
    const clipRect = (n) => {
      let painted = { ...rect(n) };
      for (let parent = n.parentElement; parent; parent = parent.parentElement) {
        const style = getComputedStyle(parent), r = rect(parent);
        if (/hidden|clip|auto|scroll/.test(style.overflowX)) {
          painted.x = Math.max(painted.x, r.x);
          painted.right = Math.min(painted.right, r.right);
        }
        if (/hidden|clip|auto|scroll/.test(style.overflowY)) {
          painted.y = Math.max(painted.y, r.y);
          painted.bottom = Math.min(painted.bottom, r.bottom);
        }
      }
      return painted;
    };
    window.seamRead = () => {
      const q = seamQA, f = q.frame, s = q.stage, p = q.panel, host = rect(f), stage2 = rect(s), panel2 = rect(p), ps = css(p), ss = css(s), fs2 = css(f);
      const tabs = [...f.querySelectorAll("[data-site-tab]")].filter((n) => !n.hidden && getComputedStyle(n).display !== "none").map((n) => ({ id: n.dataset.siteTab, rect: rect(n), css: css(n) }));
      const paint = [];
      for (const edge of [ps, css(p, "::before"), css(p, "::after")]) {
        if (edge.opacity <= 0 || edge.border === "transparent" || edge.border === "rgba(0, 0, 0, 0)") continue;
        const ps2 = edge;
        if (ps2.bt) paint.push({ x: panel2.x, y: panel2.y, right: panel2.right, bottom: panel2.y + ps2.bt });
        if (ps2.bb) paint.push({ x: panel2.x, y: panel2.bottom - ps2.bb, right: panel2.right, bottom: panel2.bottom });
        if (ps2.bl) paint.push({ x: panel2.x, y: panel2.y, right: panel2.x + ps2.bl, bottom: panel2.bottom });
        if (ps2.br) paint.push({ x: panel2.right - ps2.br, y: panel2.y, right: panel2.right, bottom: panel2.bottom });
      }
      // During rotation the host may paint a replacement border exactly where
      // its clip cuts across the moving stage. Measure its real computed inset
      // and each painted side, rather than treating the whole host as covered.
      const clippedEdge = getComputedStyle(f, "::after");
      if (clippedEdge.content !== "none" && clippedEdge.display !== "none" && clippedEdge.visibility === "visible" && +clippedEdge.opacity > 0) {
        const bounds = {
          x: host.x + (parseFloat(clippedEdge.left) || 0),
          y: host.y + (parseFloat(clippedEdge.top) || 0),
          right: host.right - (parseFloat(clippedEdge.right) || 0),
          bottom: host.bottom - (parseFloat(clippedEdge.bottom) || 0)
        };
        fs2.clippedEdge = { bounds };
        for (const side of ["Top", "Right", "Bottom", "Left"]) {
          const width = parseFloat(clippedEdge["border" + side + "Width"]);
          const color = clippedEdge["border" + side + "Color"];
          if (!width || color === "transparent" || color === "rgba(0, 0, 0, 0)") continue;
          fs2.clippedEdge[side.toLowerCase()] = width;
          paint.push({
            x: side === "Right" ? bounds.right - width : bounds.x,
            y: side === "Bottom" ? bounds.bottom - width : bounds.y,
            right: side === "Left" ? bounds.x + width : bounds.right,
            bottom: side === "Top" ? bounds.y + width : bounds.bottom
          });
        }
      }
      tabs.forEach((t) => {
        if (t.css.opacity > 0 && t.css.background !== "transparent" && t.css.background !== "rgba(0, 0, 0, 0)") paint.push(t.rect);
      });
      const compact = f.dataset.frameCompact === "true";
      // Scan the exposed perimeter after clipping. A translated stage can keep
      // its own border intact while the host clips that border out of view.
      const painted = clipRect(s);
      const scans = painted.right - painted.x > 4 && painted.bottom - painted.y > 4 ? [
        { axis: "x", at: painted.y + 2, start: painted.x + 2, end: painted.right - 2 },
        { axis: "x", at: painted.bottom - 2, start: painted.x + 2, end: painted.right - 2 },
        { axis: "y", at: painted.x + 2, start: painted.y + 2, end: painted.bottom - 2 },
        { axis: "y", at: painted.right - 2, start: painted.y + 2, end: painted.bottom - 2 }
      ] : [];
      // Sticky document rails cover the currently visible page, not thousands
      // of offscreen pixels. Keep the actual frame-edge positions, but only
      // inspect their visible spans; the viewport cutoff is not a new border.
      const visibleScans = scans.filter((line) => line.axis === "x" ? line.at >= scrollY && line.at <= scrollY + innerHeight : line.at >= 0 && line.at <= innerWidth).map((line) => ({
        ...line,
        start: Math.max(line.start, line.axis === "x" ? 0 : scrollY),
        end: Math.min(line.end, line.axis === "x" ? innerWidth : scrollY + innerHeight)
      })).filter((line) => line.end > line.start);
      const coverage = visibleScans.map((line) => ({ ...line, gaps: gaps(line.start, line.end, paint.filter((r) => line.axis === "x" ? line.at >= r.y && line.at <= r.bottom : line.at >= r.x && line.at <= r.right).map((r) => line.axis === "x" ? [r.x, r.right] : [r.y, r.bottom])) }));
      return { t: performance.now(), url: location.pathname + location.search + location.hash, width: innerWidth, height: innerHeight, scrollY, fit: f.dataset.frameFit, view: f.dataset.frameView, compact, category: f.dataset.frameCategory, host, stage: stage2, panel: panel2, slot: rect(f.querySelector("[data-site-frame-slot]") || p), newSlot: !!f.querySelector("[data-site-frame-slot]"), slotPaint: clipRect(f.querySelector("[data-site-frame-slot]") || p), viewport: rect(SiteFrame.viewport()), viewportPaint: clipRect(SiteFrame.viewport()), held: f.classList.contains("site-frame--held"), footer: rect(document.querySelector("[data-site-shell-footer],footer")), tabs, painted, coverage, maxGap: Math.max(0, ...coverage.flatMap((l) => l.gaps.map(([a, b]) => b - a))), styles: { frame: fs2, stage: ss, panel: ps }, moving: f.classList.contains("site-frame--moving"), clip: getComputedStyle(SiteFrame.viewport()).clipPath, loader: !document.querySelector("[data-site-frame-loading]").hidden, loaderBar: rect(document.querySelector(".site-frame__loading-bar")), identity: { frame: q.frame === SiteFrame.root(), panel: q.panel === f.querySelector("[data-site-route-panel]"), stage: q.stage === f.querySelector("[data-site-frame-stage]"), slot: !q.slot || q.slot === f.querySelector("[data-site-frame-slot]"), viewport: q.viewport === SiteFrame.viewport(), toolbar: q.toolbar === f.querySelector("[data-site-route-toolbar]"), tabs: q.tabs.every((n) => f.contains(n)), document: q.time === performance.timeOrigin } };
    };
  });
}
async function start(page, name, options = {}) {
  return page.evaluate(({ name: name2, options: options2 }) => {
    const initial = seamRead();
    const record = { name: name2, options: options2, initial, samples: [initial], running: true };
    seamQA.active = record;
    const sample = () => {
      if (!record.running) return;
      const renderedWidth = innerWidth, renderedHeight = innerHeight;
      // Observe after every callback in this rendering frame. Sampling in an
      // earlier rAF callback can catch WAAPI's new position before the frame's
      // own later callback synchronizes its slot and decorative border.
      setTimeout(() => {
        if (!record.running) return;
        // A driver can resize between rAF and this task, before the next
        // render delivers resize events. That is not the frame we observed.
        if (innerWidth === renderedWidth && innerHeight === renderedHeight) record.samples.push(seamRead());
        requestAnimationFrame(sample);
      }, 0);
    };
    requestAnimationFrame(sample);
    return initial;
  }, { name, options });
}
function analyze(record) {
  const { initial: a, final: b, samples, options } = record;
  const eq = (x, y) => Math.abs(x - y) <= 1;
  const sameViewport = a.width === b.width && a.height === b.height;
  const fixed = sameViewport && a.fit === "viewport" && b.fit === "viewport" && ["x", "y", "right", "bottom"].every((k) => eq(a.stage[k], b.stage[k]));
  const min = {}, max = {};
  for (const k of ["x", "y", "right", "bottom"]) {
    min[k] = Math.min(a.stage[k], b.stage[k]);
    max[k] = Math.max(a.stage[k], b.stage[k]);
  }
  const fixedDrift = fixed ? Math.max(...samples.flatMap((s) => ["x", "y", "right", "bottom"].map((k) => Math.abs(s.stage[k] - a.stage[k])))) : null;
  const envelope = sameViewport ? Math.max(0, ...samples.flatMap((s) => ["x", "y", "right", "bottom"].map((k) => Math.max(min[k] - s.stage[k], s.stage[k] - max[k])))) : null;
  const overflow = Math.max(0, ...samples.flatMap((s) => [s.host.x - s.painted.x, s.painted.right - s.host.right, s.host.y - s.painted.y, s.painted.bottom - s.host.bottom, s.footer ? s.painted.bottom - s.footer.y : 0]));
  const over = (inner, outer) => inner.right <= inner.x || inner.bottom <= inner.y ? 0 : Math.max(0, outer.x - inner.x, inner.right - outer.right, outer.y - inner.y, inner.bottom - outer.bottom);
  const slotOverflow = Math.max(0, ...samples.filter((s) => s.newSlot).flatMap((s) => [over(s.moving || s.held ? s.slotPaint : s.slot, s.panel), over(s.moving || s.held ? s.viewportPaint : s.viewport, s.slot)]));
  const loaderVisible = samples.filter((s) => s.held && s.loader && s.loaderBar).every((s) => {
    const r = s.loaderBar;
    const x = (r.x + r.right) / 2, y = (r.y + r.bottom) / 2 - s.scrollY;
    return x >= 0 && x <= s.width && y >= 0 && y <= s.height;
  });
  const maxGap = Math.max(...samples.map((s) => s.maxGap));
  const continuity = samples.every((s) => Object.values(s.identity).every(Boolean) && s.styles.frame.opacity === 1 && s.styles.panel.opacity === 1 && s.stage.w > 0 && s.stage.h > 0);
  const gapPeak = samples.reduce((p, s) => s.maxGap > p.maxGap ? s : p, samples[0]);
  const preference = record.heldPreference;
  const heldPreferenceApplied = !options.reduced || Boolean(preference?.held && preference.reduced && !preference.moving && preference.audience === "analytics" && [...preference.tabs].sort().join(",") === "about,contact,projects,resume" && preference.runningAnimations === 0);
  const rootFlowRestored = record.rootMinHeight.original.value === record.rootMinHeight.final.value && record.rootMinHeight.original.priority === record.rootMinHeight.final.priority;
  return { fixed, checks: { continuity, stableViewportEdges: !fixed || fixedDrift <= 1, endpointEnvelope: !sameViewport || envelope <= 1, containedPaint: overflow <= 1, borderCoverage: maxGap <= 1, contentSlot: slotOverflow <= 1, heldLoaderVisible: loaderVisible, heldPreferenceApplied, rootFlowRestored }, slotOverflow, fixedDrift, envelope, overflow, maxGap, peakGapAt: gapPeak.t - a.t, stageWidths: [a.stage.w, Math.max(...samples.map((s) => s.stage.w)), b.stage.w], stageHeights: [a.stage.h, Math.max(...samples.map((s) => s.stage.h)), b.stage.h], samples: samples.length };
}
async function finish(page, prefix) {
  await settle(page);
  const record = await page.evaluate(() => {
    const r = seamQA.active;
    r.running = false;
    r.final = seamRead();
    r.samples.push(r.final);
    r.rootMinHeight = { original: seamQA.rootMinHeight, final: { value: document.documentElement.style.getPropertyValue("min-height"), priority: document.documentElement.style.getPropertyPriority("min-height") } };
    return r;
  });
  const result = { prefix, ...record, analysis: analyze(record) };
  cases.push(result);
  save();
  console.log(JSON.stringify({ prefix, name: record.name, ...result.analysis }));
  return result;
}
async function targetClick(page, selector, touch) {
  const n = page.locator(selector).first();
  await n.scrollIntoViewIfNeeded();
  if (touch) await n.tap();
  else await n.click();
}
async function action(page, prefix, name, callback, options = {}) {
  await start(page, name, options);
  await callback();
  if (options.screenshot) {
    await page.waitForFunction(() => SiteFrame.root().classList.contains("site-frame--moving"));
    await page.waitForTimeout(100);
    await page.screenshot({ path: output + "-" + prefix + "-" + name.replace(/[^a-z0-9]+/gi, "-") + "-moving.png" });
  }
  return finish(page, prefix);
}
async function inspectTypography(page, prefix, fontSize) {
  await page.evaluate((fontSize2) => {
    document.documentElement.style.fontSize = fontSize2;
  }, fontSize);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const tabs = await page.locator("[data-site-tab]:visible").evaluateAll((nodes) => nodes.map((node) => {
    const bounds = (element) => {
      const r = element.getBoundingClientRect();
      return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, width: r.width, height: r.height };
    };
    const svg = node.querySelector(".site-frame__tab-icon svg");
    const label = node.querySelector(".site-frame__tab-label");
    return { id: node.dataset.siteTab, tab: bounds(node), svg: svg ? bounds(svg) : null, label: label ? bounds(label) : null };
  }));
  const result = {
    prefix,
    fontSize,
    tabs,
    pass: tabs.length === 4 && tabs.every(({ tab, svg, label }) => svg && label && svg.bottom + 1 <= label.y && svg.x >= tab.x - 1 && svg.right <= tab.right + 1 && svg.y >= tab.y - 1 && label.bottom <= tab.bottom + 1)
  };
  typography.push(result);
  save(true);
  console.log(JSON.stringify({ prefix, typography: fontSize, pass: result.pass }));
  await page.screenshot({ path: output + "-" + prefix + "-font-" + fontSize.replace("%", "") + ".png" });
}
async function runViewport(browser, engine, size) {
  const prefix = engine + "-" + size.width + "x" + size.height;
  const touch = size.width < 960 || size.height < 620;
  const context = await browser.newContext({ viewport: size, ...touch ? { hasTouch: true, ...engine === "firefox" ? {} : { isMobile: true } } : {} });
  const page = await context.newPage();
  page.on("pageerror", (error) => {
    errors.push({ prefix, message: error.message });
    save(true);
  });
  try {
    await ready(page);
    await install(page);
    if (groups.has("tabs")) {
      const trail = tabTrail();
      for (let i = 1; i < trail.length; i += 1) {
        const next = trail[i];
        await action(page, prefix, "tabs-" + trail[i - 1] + "-to-" + next, () => targetClick(page, '[data-site-tab="' + next + '"]', touch), { screenshot: size.width === 2304 && (i === 4 || i === 9) });
      }
    }
    if (groups.has("routes")) {
      await targetClick(page, '[data-site-tab="projects"]', touch);
      await settle(page);
      await action(page, prefix, "featured-direct", () => targetClick(page, '.home-accordion__card[href="/portfolio/babynames"]', touch), { screenshot: size.width === 2304 });
      await action(page, prefix, "featured-back", () => page.goBack(), { screenshot: size.width === 2304 });
      await action(page, prefix, "view-all-library", () => targetClick(page, '[data-home-library-open="projects"]', touch), { screenshot: size.width === 2304 });
      await action(page, prefix, "library-detail", () => targetClick(page, '[data-home-library-list] a[href="/portfolio/babynames"]', touch));
      await action(page, prefix, "library-back", () => page.goBack());
      await action(page, prefix, "library-close", () => targetClick(page, '[data-home-library-close="projects"]', touch));
    }
    if (groups.has("stress")) {
      await action(page, prefix, "rapid-reversals", async () => {
        await page.evaluate(async () => {
          for (const id of ["contact", "about", "games", "projects", "about"]) {
            document.querySelector('[data-site-tab="' + id + '"]').click();
            await new Promise((resolve) => setTimeout(resolve, 65));
          }
        });
      });
      await page.route("**/privacy?qa=frame-seam-delay", async (route) => {
        await new Promise((resolve) => setTimeout(resolve, 700));
        await route.continue();
      });
      await action(page, prefix, "delayed-resources", () => page.evaluate(() => SiteNavigation.navigate(new URL("/privacy?qa=frame-seam-delay", location.href))));
      await action(page, prefix, "delayed-back", () => page.goBack());
      await action(page, prefix, "resize-rotation", async () => {
        await page.evaluate(() => document.querySelector('[data-site-tab="contact"]').click());
        await page.waitForTimeout(70);
        await page.setViewportSize(touch ? { width: size.height, height: size.width } : { width: size.width === 2304 ? 1440 : 2304, height: size.height });
      }, { resize: true });
    }
    if (groups.has("held")) {
      for (const variant of ["rotation", "reduced-motion"]) {
        await page.setViewportSize(size);
        await page.emulateMedia({ reducedMotion: "no-preference" });
        await ready(page);
        await install(page);
        await page.evaluate(() => {
          const lifecycle = SiteRoutes.get("page:content");
          window.frameQaOriginalLifecycle = lifecycle;
          SiteRoutes.register("page:content", {
            ...lifecycle,
            async mount(context2) {
              if (context2.url.searchParams.get("qa") === "frame-seam-mount-delay") {
                window.frameQaMountStarted = true;
                await new Promise((resolve) => setTimeout(resolve, 700));
                if (context2.signal.aborted) return;
              }
              return lifecycle.mount(context2);
            }
          });
        });
        await action(page, prefix, "held-mount-" + variant, async () => {
          await page.evaluate(() => {
            window.frameQaNavigation = SiteNavigation.navigate(new URL("/analytics?qa=frame-seam-mount-delay", location.href));
          });
          await page.waitForFunction(() => window.frameQaMountStarted);
          if (variant === "rotation") {
            await page.setViewportSize(touch ? { width: size.height, height: size.width } : { width: 844, height: 390 });
          } else {
            await page.emulateMedia({ reducedMotion: "reduce" });
            await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            await page.evaluate(() => {
              const frame = SiteFrame.root();
              const nodes = [frame.querySelector("[data-site-frame-stage]"), frame.querySelector("[data-site-route-panel]"), frame.querySelector("[data-site-frame-slot]"), ...frame.querySelectorAll("[data-site-tab]")];
              seamQA.active.heldPreference = {
                held: frame.classList.contains("site-frame--held"),
                reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
                moving: frame.classList.contains("site-frame--moving"),
                audience: frame.dataset.frameAudience,
                tabs: [...frame.querySelectorAll("[data-site-tab]")].filter((node) => !node.hidden && getComputedStyle(node).display !== "none").map((node) => node.dataset.siteTab),
                runningAnimations: nodes.filter(Boolean).flatMap((node) => node.getAnimations()).filter((animation) => animation.playState === "running").length
              };
            });
          }
          await page.evaluate(() => window.frameQaNavigation);
        }, { resize: variant === "rotation", reduced: variant === "reduced-motion" });
        await page.evaluate(() => {
          SiteRoutes.register("page:content", window.frameQaOriginalLifecycle);
          delete window.frameQaOriginalLifecycle;
          delete window.frameQaMountStarted;
          delete window.frameQaNavigation;
        });
      }
      await page.emulateMedia({ reducedMotion: "no-preference" });
    }
    if (groups.has("typography") && size.width === 844 && size.height === 390) {
      await page.setViewportSize(size);
      await ready(page, "/analytics");
      await install(page);
      await targetClick(page, '[data-site-tab="contact"]', touch);
      await settle(page);
      await inspectTypography(page, prefix, "100%");
      await inspectTypography(page, prefix, "200%");
    }
    await page.screenshot({ path: output + "-" + prefix + "-final.png" });
  } catch (error) {
    errors.push({ prefix, flow: true, message: error.message, url: page.url() });
    save(true);
    await page.screenshot({ path: output + "-" + prefix + "-error.png" }).catch(() => {
    });
    console.error("FLOW ERROR", prefix, error.message);
  } finally {
    save(true);
    await context.close();
  }
}
(async () => {
  for (const engine of engineNames) {
    const type = { chromium, firefox, webkit }[engine];
    if (!type) throw new Error("Unsupported FRAME_SEAM_ENGINES value: " + engine);
    const browser = await type.launch();
    try {
      for (const size of sizes) {
        if (!(size.width > 0 && size.height > 0)) throw new Error("Each FRAME_SEAM_SIZES entry needs positive width and height.");
        await runViewport(browser, engine, size);
      }
    } finally {
      await browser.close();
    }
  }
  const failedCases = cases.filter((record) => Object.values(record.analysis.checks).some((value) => !value));
  const failedTypography = typography.filter((record) => !record.pass);
  save(true);
  console.log(JSON.stringify({ cases: cases.length, failed: failedCases.length, typography: typography.length, failedTypography: failedTypography.length, errors, artifact: output + ".json" }));
  if (failedCases.length || failedTypography.length || errors.length) process.exitCode = 1;
})().catch((error) => {
  save(true);
  console.error(error);
  process.exitCode = 1;
});
