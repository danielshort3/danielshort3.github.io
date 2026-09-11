# Performance checks

After `npm run build`, run `npm run performance:report`. The command writes a
readable report and JSON to `tmp/performance/`, and fails if a budget is exceeded.
CI publishes the same report in its summary and as a downloadable artifact.

These are deterministic delivery-size regression checks. Each route adds its
initial HTML and directly declared first-party CSS/JS, compressed separately with
gzip level 9. Deferred scripts count; `data-*-src` attributes, preloads, repeated
URLs and external scripts do not accidentally count as initial script requests.
External URLs are listed for review. Runtime imports, game/model downloads,
fonts, page images, browser caching and execution time are outside that total.
The complete icon catalogs have separate limits, including a check that every
original PNG has its WebP variant in the published output.

The September 10, 2026 initial local output was about 171 KB for home, 163–179 KB
for libraries, 161 KB for Baby Names, 223 KB for Text Compare, 214 KB for Image
Optimizer and 156 KB for Privacy
using default gzip compression. Project Starfall (1.45 MB), Stellar Dogfight
(313 KB) and Ocean (230 KB) use separate budgets because their document scripts
do materially different work. The new report consistently uses gzip level 9;
it may differ slightly from that initial reference. Limits allow approximately
15–25% headroom, with a little extra room for the actively developed simulator.
Review a growth report before changing a limit; optimize the cause or explicitly
document the feature tradeoff. Do not increase limits automatically in builds.

The icon variants preserve dimensions and alpha and use WebP quality 90. Their
PNG files remain the source artwork and native `<picture>` fallback. Regenerate
only those variants with `node build/optimize-site-images.js --catalog-only`.

This report does **not** establish that Core Web Vitals pass or fail. For actual
experience, check Search Console/CrUX where data is available and run repeatable
mobile browser profiles covering tab navigation, tool input, and page changes.
Field targets are LCP ≤2.5 s, INP ≤200 ms and CLS ≤0.1 at the 75th percentile.
No extra visitor telemetry or consent behavior is introduced by these checks.

References: [performance budgets](https://web.dev/articles/performance-budgets-101),
[Web Vitals](https://web.dev/articles/vitals),
[WebP images](https://web.dev/articles/serve-images-webp).
