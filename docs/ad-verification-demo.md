# Live Campaign Measurement

Unlisted entry: `/demos/ad-verification`. Cedar Valley Tourism, all travelers and all observations are fictional. Hashes and signatures are real; no advertising or location service is connected.

## One campaign, independent travelers

Five visible lanes each hold one traveler. Each traveler has an independent observation schedule: ad exposure, an optional website event, an optional destination-attribution event, and the end of its observation window. Different travelers and different stages genuinely overlap. Nobody waits for an entire batch to finish.

Only a traveler whose final summary has been committed can leave. After a short hold/fade, a new ID takes that lane while the other four continue. The campaign and all earlier blocks remain intact. There is no group selector, group schedule, or group field in v3 records. The selected scenario applies to the next individual arrival, not travelers already on screen.

All five lanes remain visible, including on phones. The traveler list has no internal scrolling, pagination or hidden active rows. The compact blockchain shows the latest six records on desktop and three on narrow screens; **View full chain** retains access to every earlier record. The full chain is one linear sequence, not a separate chain per traveler. Traveler IDs and prior-traveler-event references connect business journeys across interleaved records.

The mixed outcome pattern and timing are illustrative, not estimates of real conversion rates or travel durations. An absent observation is not proof of no visit. Skipped outcomes create no pretend visit blocks; the signed closing summary records only what was observed in that simulated window.

## Source ownership

- `demos/ad-verification.html`: authored interface and educational copy.
- `js/demos/ad-verification-core.js`: v3 record schema, signing, verification, checkpoint, candidate/commit boundary, and individual traveler plans.
- `js/demos/ad-verification-player.js`: independent lane schedules, bounded observation queue, one block writer, replacement and cancellation on one RAF clock.
- `js/demos/ad-verification.js`: stable lane DOM, compact chain, full history, editing and export.
- `css/components/ad-verification.css`: isolated light layout; all five lanes visible without list scrolling.
- `content/projects/adVerification.json`: unpublished/unlisted metadata.

The existing static copy step and clean-URL handling publish the same entry point. No shared site bundle, dependency, credential, API or deployment setting is changed. The existing approved traveler illustration sprite is reused.

## Timing and synchronization contract

Traveler measurement animations run concurrently on one shared simulation clock. A completed observation enters an ordered queue. A lane can have at most one outstanding record, bounding the queue to the active pool. Subsequent observations for that traveler wait for its preceding record; other travelers continue independently.

The block writer processes one queued record at a time. It uses real Web Crypto to create and verify a signed candidate. Both the writer panel and its matching traveler display the same event key and verification progress. A slow cryptographic operation holds at 80%; it never produces an early recorded check. Other lanes may finish observations and queue behind it.

On completion, one synchronous update commits the verified private candidate, updates the corresponding traveler milestone, and renders the new block. There is at most one commit per frame. Events retain an observation time separately from the block's recording timestamp; neither is an externally certified time.

Pause freezes lane progress, incoming blocks, observation waits, and replacement fades, even if background signing finishes. A hidden tab pauses automatically. Reset retires the signing session and invalidates pending callbacks, so no old candidate or traveler can appear in a new campaign. Inspecting/focusing a record or traveler pauses playback to protect keyboard focus.

The demo caps history at 256 blocks. Before admitting a traveler, it reserves capacity for every remaining event in that traveler's plan. At the cap, new arrivals stop and every admitted traveler can finish; history is never silently truncated or reset. The optional continuous-replacement checkbox can also let current travelers finish without admitting more.

## What the blockchain verifies

Each block contains one signed event, a SHA-256 Merkle root, a previous-hash-linked header and three distinct local validator signatures. The verifier checks schema, campaign identity, traveler ordering, original event signatures, Merkle roots, header hashes, prior-block references, approvals, and the original length/head checkpoint. Original records and the public-key registry remain separate from the editable copy. Non-extractable private keys are never exported.

One event per block is a teaching simplification. All signers and validators are inside the same browser: this is not independently operated consensus or external notarization. Someone controlling the code, keys and checkpoint can construct a different valid-looking example. A matching signature authenticates bytes relative to the session key; it does not prove a human saw an ad, a website session was genuine, or a destination visit occurred.

## Inspect, edit and restore

Select any recent block, a recorded traveler milestone, or an older block from **View full chain**. Change a data field and apply verification. The original signatures and checkpoint are not replaced. An edited record fails; untouched subsequent records show that they depend on changed history, not that their own data was necessarily edited. Playback stays disabled until the exact original signed bytes are restored.

Export contains records, original public keys and the session checkpoint, never private keys. An edited export also fails independent verification. There is no new campaign network traffic, storage, GPS permission or real device identification. Normal site-wide privacy controls injected by the standard build remain intact.

## Visibility

`published: false`, `hidden: true`, `noindex: true`, and `visibility: "unlisted"` are retained. The page uses `noindex, nofollow, noarchive`. Generated search, sitemap and app catalog exclusion is tested after building. Anyone with the URL can still open the page, and the repository is public: unlisted is not access control. Reset/reload intentionally discards the local session.

## Checks

On Node 22 with existing repository dependencies:

```sh
node --test tests/tools/ad-verification.test.js
npm run build
node --test tests/tools/ad-verification.test.js
npx playwright install chromium
node tests/tools/ad-verification.browser.cjs
```

Node checks cover all five scenarios, overlapping stages at 1x/2x/4x, per-lane replacement, new-arrival scenario changes, append-only history, slow signing, queue behavior, pause/reset, capacity reservation, tampering, rehashing, malformed proofs, distinct approvals, original restoration and noindex publication. The built-output test explicitly skips until the full site build exists.

Native Chromium independently samples rendered frames for simultaneous measurements, different overlapping stages, atomic traveler/block commits, matching verification progress, and replacement only after the closing record. It checks full-history access, edited proof failure, byte-identical restoration, native slow signing, missing-engine/no-JavaScript states, reduced motion, and all five visible lanes at 1280/1024/768/390/320px.

The existing scoped GitHub workflow runs the full build and both focused suites. Browser screenshots, viewport bounds and `animation-sync-evidence.json` are stored in workflow artifacts, not source. `BROWSER_ARTIFACT_DIR` can choose another evidence directory. Tests use a loopback server and do not submit production forms or use real advertising services.
