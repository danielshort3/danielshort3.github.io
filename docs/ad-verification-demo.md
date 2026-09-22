# From an Ad to a Visit

Unlisted entry: `/demos/ad-verification`. Cedar Valley, its travelers, source systems and occurrence dates are fictional. Native Web Crypto calculates real signatures and hashes locally. No advertising, analytics, device or location service is connected by this demo.

## Approved first concept

The interface follows the selected calm two-panel design: a small destination header, one headline and control row, five visible traveler lanes with an Outcome column, a compact blockchain with one inline expanded record, and one results band. There is no permanent third inspector, extra dashboard, group selector or group schedule.

All five lanes have independent observation timing and cycle individually. Motion is restrained: small progress cues, a single block-writer accent, and a short replacement fade. No spinning status widgets or simultaneous moving connector network is used. Reduced motion disables movement but retains real verification and status updates. On narrow screens the panels stack; all five traveler lanes remain visible without nested list scrolling.

Illustration-only portions of the user-approved AI concept supply `ad-verification-mark.webp`, `ad-verification-mountains.webp` and the five-portrait sprite `ad-verification-portraits.webp`. UI, controls, text, totals and blockchain records are native DOM content, not a screenshot. The existing Inter font URL is reused, with no new font distribution.

## What attribution actually does

The v4 schema separates **Visit reported** from **Attribution recorded**. After an ad and a destination observation exist, the attribution service calculates a decision from those earlier records. It does not accept a scenario's claimed credit boolean.

The illustrative rule is `example-exposure-window-v1`: the same synthetic traveler and campaign have an earlier exposure; the visit's authored example occurrence day is 0–30 days after that exposure, inclusive; the visit has not already been credited. Website activity is optional. A later visit is still recorded but receives no campaign credit. The mixed scenario includes such an example; a dedicated **Outside the window** scenario makes it directly testable.

Each attribution record contains the rule ID, window, elapsed days, decision and exact ad/visit references (event ID, block height and block hash). Verification recomputes the expected decision from the preceding chain and compares all these fields. Cross-traveler references, edited windows, fabricated credit, altered elapsed time and mismatched fingerprints cannot pass that check. The journey checker also rejects a second attribution for the same traveler and requires the decision before closing a reported-visit path.

Example occurrence days are authored fictional dates, not derived from animation speed. `observedAtMs` represents accelerated receipt order; `recordedAt` is the browser's local recording timestamp. These are different concepts, and neither is an independently certified clock. These illustrative outcome patterns are not observed conversion rates.

## Inspectable evidence, not more panels

The newest attribution opens **Why this visit was counted** (or **not counted**) inside its block. A selected earlier block can remain beside three recent records; omitted intervening blocks are labeled and retained in full history. There is always a single chronological chain, not branches per traveler.

Selecting a traveler, its outcome or a block pauses the simulation and pins that record. Supporting-record buttons show the original ad, visit report and attribution decision. No website prerequisite is implied. Technical fingerprints and signatures stay behind an optional disclosure. Every committed block, including older records, remains inspectable through **View full chain**.

## Results and edited copies

The results band recomputes ad exposures, website visits and attributed visits from accepted original records. The active-traveler count reflects the currently measured lanes and is deliberately not labeled as the cumulative number of measured travelers. Counts are not independent animation counters. **Recheck results** verifies the evidence, signatures, links, attribution rule and original checkpoint.

Editing affects only a working copy. Original signatures, keys and checkpoint are not replaced; the authentic ledger and its totals remain unchanged. The UI rejects the edited copy, labels the original totals accordingly and blocks playback until restoration. `verifyResults` returns `totals: null` for invalid evidence rather than calculating an apparently trusted result. **Restore original** restores the exact original signed bytes, without creating new signatures. An exported edited copy fails independent verification too.

## Source ownership

| Responsibility | Source |
| --- | --- |
| Authored layout and copy | `demos/ad-verification.html` |
| v4 schema, attribution rule, signed ledger, verification and totals | `js/demos/ad-verification-core.js` |
| Parallel scheduling, one block writer, capacity reservation and cancellation | `js/demos/ad-verification-player.js` |
| Persistent lane DOM, inline evidence, supporting records, editing and controls | `js/demos/ad-verification.js` |
| Isolated light-only responsive styles | `css/components/ad-verification.css` |
| Visibility and project metadata | `content/projects/adVerification.json` |

The existing static copy step and clean-URL configuration publish this same route. No shared website bundle, dependency, API, credential or deployment setting changes. No session migration is needed: earlier demo versions do not persist browser campaigns. v3 exported examples require the v3 verifier; v4 intentionally uses a new schema and signing domain.

## Clock and blockchain contract

All traveler schedules share one RAF simulation clock but progress independently. Completed observations enter a bounded queue; a traveler has at most one pending record. The single writer prepares a real signed candidate and verifies it before committing. The corresponding lane and block update synchronously in the same frame. Slow cryptography holds the writer at 80%, while other lanes can complete and queue observations. Pause freezes all animation and commits, even if a signature finishes in the background. Reset retires pending work so it cannot enter the next campaign.

Each block uses SHA-256 Merkle hashing, an ECDSA P-256 event signature, the preceding block hash and three distinct local validator approvals. Original public keys, length and final hash are held separately from the editable copy. Non-extractable private keys are never returned or exported. Capacity is reserved for each admitted traveler's entire remaining path, including attribution and the closing summary, so the 256-block limit does not strand people midway.

## Limits and privacy

This is a browser-local educational blockchain, not an independently operated consensus network. All identities and validators are simulated in the same browser. A party controlling the code, keys and trust checkpoint could construct a different valid-looking demonstration. Verification establishes consistency of recorded bytes and the example attribution rule, not that a real human saw an ad or that advertising caused a trip.

No real identifiers, GPS permissions, campaign network requests, persistent campaign storage, wallet, tokens or payments are added. Normal site-wide consent controls injected by the standard build are preserved. No observation is not proof of no visit. One event per block is an educational simplification; Merkle hashing supports multiple events per block.

Metadata remains `published: false`, `hidden: true`, `noindex: true`, `visibility: "unlisted"`; the page remains `noindex, nofollow, noarchive`. Anyone with the URL can open it and repository source is public. These are not access controls.

## Validation

On Node 22 with the existing repository dependencies:

```sh
node --test tests/tools/ad-verification.test.js
npm run build
node --test tests/tools/ad-verification.test.js
npx playwright install chromium
node tests/tools/ad-verification.browser.cjs
```

Node checks cover evidence references, inclusive attribution boundaries, missing/cross-campaign evidence, optional websites, no duplicate credit, every scenario, overlapping stages at all speeds, per-person replacement, pause/reset, delayed signing, capacity, modified rules, edited credit, invalid-result rejection, rehashing, approval requirements and publication exclusions. Built-output checks explicitly skip until the full site is built.

Native Chromium independently audits rendered frames for simultaneous traveler activity, mixed stages, matching writer progress, same-frame committed milestones, individual replacement and totals derived from accepted block events. It exercises all scenarios, evidence links, full history, invalid edited exports, byte-identical restoration, delayed native signatures, error/no-JavaScript states, reduced motion, and 1672/1280/1024/768/390/320px layouts.

The existing scoped workflow runs the full website build and both suites, uploading native screenshots, layout bounds and `animation-sync-evidence.json`. Evidence goes outside tracked source. Tests use loopback and do not submit production forms. Browser-plugin absence permits the Playwright fallback; a blocked local browser environment must not be presented as a successful native browser test.

API references: [Web Crypto ECDSA parameters](https://developer.mozilla.org/en-US/docs/Web/API/EcdsaParams) and [signature verification](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/verify).
