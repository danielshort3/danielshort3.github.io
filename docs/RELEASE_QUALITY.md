# Local and CI release checks

Build first with `npm run build`. Tests use the built site and its real clean-URL server. Keep the existing `npm test` and `npm run test:browser` gates; the release checks supplement them.

## Repeatable checks

- `npm run test:release`: pinned Chromium, Firefox and WebKit; WCAG 2 A/AA, 2.1 A/AA and 2.2 AA axe checks; keyboard activation, focus restoration, reduced motion, forced colors where supported, 320px reflow and enlarged text. Checks all homepage panels, the three libraries, a project, Text Compare results, expanded header search, search results, account and contact dialogs.
- `node tests/release/geometry.cjs`: reuse the existing frame route and 100%/200% typography checks in all three engines, including the 844×390 typography fixture.
- `npm run test:recovery`: contact and guest recovery plus analytics-consent browser regressions. Every contact submission and inference request uses a local fixture. No messages are sent and no account data is written.
- `npm run performance:lab`: mobile Lighthouse, three runs per selected route; retain medians and raw reports. Lab Total Blocking Time is a diagnostic, not real-user INP.
- `npm run test:parity`: verify updated sources, generated pages and content-hashed bundles match `public/`, including the copy step's intentional CSS filename resolution and removal of the retired Contributions bundle.

The Lighthouse harness adds gzip delivery for compressible responses to reflect the production CDN, blocks external services and all API calls, and measures `/`, `/tools/text-compare`, `/portfolio/website` and `/minesweeper-demo`. Raw reports include request timings and transferred bytes. The checked baseline gates CLS at 0.1 and LCP/TBT regressions at 20% plus a 250ms/50ms noise floor respectively. The 2.5-second LCP target remains a goal, not a claim that this local site currently meets it. Update a baseline only with explicit `node build/measure-lighthouse.cjs --update-baseline`, inspect all three runs and their medians, and review the diff. Do not update baselines to hide a regression.

Reports, traces, screenshots and diffs are stored under `tmp/`. CI pins Ubuntu 24.04, Node 22, the lockfile and Playwright 1.63.0's browser revisions: Chromium 153.0.8010.12 (1243), Firefox 155.0 (1543), and WebKit 26.6 (2359). Jobs upload evidence even on failure. Test fixtures block external requests and mock APIs; integration with real account providers, maps and Tableau requires separate review.

## Screenshot baseline review

Fourteen baseline screenshots cover closed home, About, Tools, the website project, Digit Generator, Text Compare results and the contact dialog at 1440×900 and 390×844. The mobile Digit Generator follows its masthead launch into the isolated demo; the mobile comparison uses a full-page capture so the result remains included. Capture them on Linux (WSL is supported), with the lockfile's Chromium, loaded local fonts, reduced motion and deterministic service responses. Windows runs explicitly skip screenshot comparisons; use WSL or the Linux CI gate for this check.

`npm run test:visual` compares against the reviewed images in `tests/release/baselines/`. Baselines never update during a normal run or CI. To propose an intentional change, run `npm run test:visual:update` on Linux, inspect every changed image at full size, run the comparison again without the update flag, and include the baseline diff in code review. The comparison permits a 0.2 per-pixel threshold and at most 0.3% different pixels to accommodate rasterization noise, not layout drift.

## Manual assistive-technology review

**Status: not performed in this automated implementation session.** Automated axe and keyboard checks do not establish complete accessibility, nor do they replace this pass. Record browser, screen reader, OS, date, reviewer and observed failures when completing it.

1. Use NVDA with Firefox/Chrome on Windows, then VoiceOver with Safari on iOS or macOS. Navigate headings and landmarks on the homepage, libraries and project pages.
2. Open and close each homepage tab with a keyboard. Confirm its name/expanded state is announced and hidden panels are absent from the reading order.
3. Open search, follow a result, then use Back. Confirm sensible focus and page announcements.
4. Open account and contact dialogs. Verify the name, isolated reading order, reachable actions, Escape/close and return to the trigger. Submit an empty contact form against the local fixture; check field errors and status announcements.
5. Restore/discard a guest draft, perform a Text Compare, and exercise mocked contact rejection/timeout. Confirm announcements are concise, do not steal focus and preserve the entered content.
6. At 200% text size and 400% browser zoom, inspect all controls and reading order. Check actual Windows High Contrast themes and mobile screen-reader touch exploration. Review third-party Tableau/map accessibility separately.

Do not mark this checklist complete using automated test results alone.
