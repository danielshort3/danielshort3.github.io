# Ad Verification Demo

A deliberately small, unlisted project using **Cedar Valley Tourism**, a fictional destination. Four advertising handoffs are recorded in a local miniature blockchain. The interface follows the approved version-one concept: a light project page, four connected cards, one status panel, and **Simulate change / Restore original**.

## Open and use

The deployed entry point is `/demos/ad-verification` (source: [`demos/ad-verification.html`](../demos/ad-verification.html)). The normal Vercel `cleanUrls` setting resolves this static file; no extra API, rewrite, account, token, or environment variable is required.

The example initializes automatically. **Launch Demo** creates a fresh chain and replays its verification. **Simulate change** changes the publisher's reported impression count from 10,000 to 12,500 without re-signing anything. That record fails verification; the later measurement is labeled **Earlier change**, not falsely described as itself edited. **Restore original** restores the exact original signed bytes. Select any card to inspect its record, computed hashes, and checks. The collapsed **Behind the demo** section contains rechecking and public-proof export.

## What is implemented

[`ad-verification-core.js`](../js/demos/ad-verification-core.js) is dependency-free, shared by the browser and Node tests. Each example generates four event-signing identities and three simulated validator identities using Web Crypto ECDSA P-256. Private keys are non-extractable, transient, and never returned by the core or exported. Each of four blocks contains one signed event, a SHA-256 Merkle root, a header linked to the previous block, and three signatures approving that recorded block hash.

Canonical JSON sorts object keys. Hash/signature messages use separate domain prefixes. Verification checks event identity and position, schema, signature, Merkle root, header hash, previous-block reference, and three distinct registered approvals. An original public-key registry, expected length, and terminal hash are kept separately from the editable working copy. These checkpoints also detect truncation and complete rehashing relative to the original example. Export includes only public keys, checkpoint, and signed records.

## Important limits

This is an **educational single-browser blockchain**, not a production distributed ledger. All signers are simulated in one browser session; three local approvals are not independent organizations, Byzantine-fault-tolerant consensus, or external notarization. Someone who controls the browser and replaces its code, keys, and trust checkpoint can create another valid-looking example. Reloading intentionally creates a new session; there is no persistent or externally anchored history.

A verified signature authenticates bytes relative to the trusted public key, not a real-world company or the truth of its claim. The demo does **not** prove an ad was served, viewed by a human, or caused a visit. Example timestamps are signed claims, not trusted clock attestations. No advertising or location APIs are called. No wallet, cryptocurrency, transaction fees, cookies, analytics, login, or browser storage is added by this page.

## Unlisted, not access-controlled

[`content/projects/adVerification.json`](../content/projects/adVerification.json) has `published: false`, `hidden: true`, `noindex: true`, and `visibility: "unlisted"`. The public portfolio generator does not create a portfolio case-study page for this record. The standalone demo has a `noindex, nofollow, noarchive` robots meta tag and is not linked from site navigation or the public project list. The test suite checks the built sitemap, search index, and app catalog for accidental inclusion.

Anyone with its URL can still open the demo. The GitHub repository and source are public. These settings are **not a password or confidentiality guarantee**. Before making the project public, deliberately integrate a normal portfolio page/demo route and review its indexing and catalog settings; do not simply assume changing one flag publishes every surface.

## Sources and styling

- Page: [`demos/ad-verification.html`](../demos/ad-verification.html)
- UI controller: [`js/demos/ad-verification.js`](../js/demos/ad-verification.js)
- Cryptographic engine: [`js/demos/ad-verification-core.js`](../js/demos/ad-verification-core.js)
- Isolated CSS: [`css/components/ad-verification.css`](../css/components/ad-verification.css)
- Decorative landscape: [`img/projects/ad-verification-landscape.webp`](../img/projects/ad-verification-landscape.webp), derived from the scenic-only region of the approved AI-generated concept. No screenshot is used as interactive UI.

The existing `build/copy-to-public.js` copies all four asset directories. Shared layout, brand assets, dependencies, authentication, and advertising integrations are unchanged. The page references the site's existing Inter font, rather than adding a font file.

## Checks

From the repository root on Node 22:

```sh
node --test tests/tools/ad-verification.test.js
npm run build
node --test tests/tools/ad-verification.test.js
node tests/tools/ad-verification.browser.cjs
```

The first command tests SHA-256, deterministic serialization, all original signatures, altered records, duplicate/missing approvals, wrong keys, replay, reordered/deleted blocks, truncation, rehashing, malformed input, and restoration. Built-output checks explicitly skip until `public/` exists.

The browser check uses the repository's Playwright dependency and an isolated loopback server. It tests the real browser Web Crypto runtime, initial verification, tampering, exact restoration, card inspection, keyboard dismissal, export, no-JavaScript messaging, unavailable-engine behavior, and desktop/tablet/mobile reflow. Set `BROWSER_ARTIFACT_DIR` to save screenshots outside tracked source. It does not contact production or submit forms. CI runs the engine/publication checks after the normal build and browser checks after Chromium installation.

For manual preview, serve the repository or `public/` over localhost and open `/demos/ad-verification.html`. Web Crypto requires a supported secure context (HTTPS or localhost).

References: [Web Crypto signing](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/sign), [verification](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/verify).
