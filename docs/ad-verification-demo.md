# Ad Verification Demo

An unlisted project for the fictional **Cedar Valley Tourism**, following the approved four-step concept: Advertiser → Agency / DSP → Publisher → Measurement. The interface stays simple; the cryptographic checks are real.

**Page:** https://www.danielshort.me/demos/ad-verification-demo

## Try it

The page creates four signed example records and checks them in your browser. **Launch Demo** replays the checks. **Simulate change** changes the publisher's batch report from 12,500 to 13,000 impressions without replacing the original signatures. The publisher becomes **Changed**; the following measurement step becomes **Untrusted**. **Reset demo** restores the original signed records. Click a step to inspect its actual record and compare stored and recomputed block fingerprints. Technical information is collapsed by default.

The destination, campaign, publisher, timestamps and events are illustrative. No Visit Grand Junction campaign or partner data is used. `cedar-trails.example` is an example identifier, not a contacted publisher.

## What is real, and what is simulated?

Real: Web Crypto SHA-256 fingerprints, ECDSA P-256 signatures, hash-linked blocks, domain-separated Merkle roots, explicit signer-role validation, local approval signatures, and verification against a separately retained session checkpoint. Changing, removing, appending, reordering or rehashing records produces actual verification failures.

Simulated: advertising delivery and measurement, participant identities, and three approval authorities. **All authorities run within one browser/security boundary. This is a local miniature blockchain mechanics demonstration, not a decentralized or independently secured network.** Each block needs all three local approvals; this is not a Byzantine-fault-tolerant consensus protocol. Blocks contain one illustrative batch event each. Merkle grouping supports up to 32 records, but the public demo schema deliberately expects one per step.

The event signature covers its canonical payload. The block fingerprint covers the header, including its Merkle root, height, network, session and previous-block fingerprint. All three local approval signatures cover that block fingerprint. The four-block checkpoint and public verification keys are retained separately from the editable ledger. Verification never accepts keys supplied by the ledger. Only public keys leave the builder. Private keys are generated non-extractable, are never exported or stored, and leave application scope after creation.

**Integrity is not truth.** Valid signatures do not prove a human viewed an ad, that reported counts are accurate, that a person visited, or that an ad caused a visit. An authority can sign false information. A person controlling this browser or its code can also replace the local trust context; production assurances require independent infrastructure, identity governance and external evidence. Timestamps here are signed illustrative claims, not trusted timestamps.

There are no coins, tokens, mining, wallets, payments, accounts, API calls, analytics, persistent storage or personal advertising data in this page. Its additional CSP sets `connect-src 'none'`. Reloading creates a fresh in-memory session. Canonicalization is a bounded serializer for this demo's JSON schema, not a general RFC 8785 implementation. This project is educational code, not a production security product.

## Source ownership

- `demos/ad-verification-demo.html`: authored standalone page, including `noindex, nofollow` and an isolated CSP.
- `css/components/ad-verification-demo.css`: page-only styling, responsive layouts, and reduced-motion treatment. Not imported into global styles.
- `js/demos/ad-verification-core.mjs`: dependency-free cryptographic builder and verifier, shared with Node tests.
- `js/demos/ad-verification-demo.mjs`: replay, tamper/reset, accessible status updates and record inspection.
- `img/demos/cedar-valley-hero.webp`: decorative mountain scenery derived from the unlettered landscape in the approved AI-generated Cedar Valley concept. The screenshot/UI itself is not shipped as the interface.
- `build/inject-script-bundles.js`: explicitly excludes this standalone page from consent/analytics injection, preserving its network-isolated behavior. Other pages keep their existing consent policy.
- `tests/demos/ad-verification.test.cjs`: cryptographic, schema, privacy and discovery regressions.
- `tests/demos/ad-verification.browser.cjs`: real built-route, responsive, accessibility, interaction, failure-state, build-isolation and console checks.
- `.github/workflows/ad-verification-demo.yml`: focused CI with screenshot artifacts; also runs when shared script injection changes.

Existing `build/copy-to-public.js` recursively publishes `demos`, `js`, `css` and `img`. Vercel's existing `cleanUrls` policy serves this authored page at `/demos/ad-verification-demo`; no new rewrite, global navigation, CMS record, native catalog entry or generated shell is needed. The project is intentionally absent from public catalogs, home cards, search, chatbot knowledge, short-link destination suggestions and sitemaps. The browser and discovery tests guard that behavior. The shared site remains unchanged apart from the explicit build exemption for this page.

**Unlisted is not private:** anyone with the direct link can open the page, and source is visible in this public repository. `noindex` is a request to search engines, not access control. No authentication is claimed.

## Validation

```sh
# Focused, dependency-free cryptographic and source checks (Node 22)
node --test tests/demos/ad-verification.test.cjs

# Normal repository build and tests; install documented subproject dependencies too
npm run build
npm test

# Require built assets and verify exclusion from discovery outputs
AV_REQUIRE_BUILD=1 node --test tests/demos/ad-verification.test.cjs

# Browser checks use build/dev.js, with the real rewrites and security headers
npx playwright install chromium
node tests/demos/ad-verification.browser.cjs

# Interactive local preview
npm run dev -- --port 4173
# Open http://localhost:4173/demos/ad-verification-demo
```

Browser evidence is written outside tracked source, under the OS temporary directory or `BROWSER_ARTIFACT_DIR`. Coverage includes 1448, 1024, 768, 390 and 320-pixel widths, no horizontal overflow, WCAG A/AA axe checks, initial verification, tamper/reset/replay, record inspection, Escape/focus restoration, reduced motion, a missing-Web-Crypto failure state, the `.html` alias and no external requests. No production forms or services are exercised.

## References

- [Web Crypto signatures](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/sign)
- [Web Crypto signature verification](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/verify)
- [Web Crypto interface and secure-context requirements](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto)
