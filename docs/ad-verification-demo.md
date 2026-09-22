# Campaign Results You Can Check

Unlisted entry: `/demos/ad-verification`. This is an educational **campaign-attribution audit layer**, not a visitor-tracking service. All travelers, provider observations, matching arrangements and participants are fictional. The browser performs real ECDSA signature, SHA-256, Merkle-root and checkpoint checks.

## Product story and visual scope

The approved white, navy and blue two-panel layout is retained: five independent traveler lanes, a compact shared blockchain, one results band, and optional details. The headline remains “From an ad to a visit. With a record you can check.” The story is now explicit:

**Partners measure. Attribution assigns credit. Blockchain preserves the record.**

A visitor can inspect a source receipt, check separate example evidence, deliberately make that evidence unavailable, compare participant copies, test an edited campaign report, and append an authorized correction. Only the relevant detail dialog opens; no permanent third dashboard or animated network is added. Existing Cedar Valley illustration assets and the site's Inter font reference are reused unchanged. No new dependency or font file is introduced.

## Parallel people, batched records

Five lanes run on one simulation clock with independent event schedules. People can continue to the website or destination while earlier receipts are still waiting for a block. Provider work is sequenced within each fictional journey so an attribution calculation has its prerequisite observations; it does **not** wait for the blockchain writer. The illustration separates a reported observation (blue) from a recorded signed receipt (green).

Completed source receipts enter a buffer. Up to four are included in a block, or a smaller batch is prepared after the maximum example wait. Each block contains actual signed receipts and a Merkle root; it links to the previous block's hash. A batch can contain different people and different event types. Each matching recorded check changes in the same synchronous update as the block append. The example is not a claim that production systems should publish each individual observation; larger batches or periodic report commitments would also be possible.

All five active lanes remain visible without a nested traveler scroll region. A traveler is replaced only after the closing receipt and every earlier receipt in that lane have been committed. Other lanes continue. Mix changes apply only to new arrivals. Pause freezes the animation clock, queued receipt admission, block commits and replacement fades, even if cryptographic work finishes in the background. Reset retires stale callbacks and signing material. Reduced motion removes the moving indicators and fades.

The session is limited to 192 issued receipts. Admission reserves the entire future receipt cost of each traveler and leaves 12 slots for review/correction exercises. Existing travelers can finish before the simulation stops. Repeated manual exercises can consume the remaining headroom and require a reset; the demo does not silently discard old history.

## Private provider evidence versus shared audit receipts

The core owns a separate, in-memory provider-evidence map. An evidence packet contains a random salt plus an example source record: synthetic traveler association, occurrence day, source-specific page/place detail, or calculation evidence. `evidenceDigest` commits to the canonical salted packet using a separate hashing domain.

The shared receipt contains only:

- Version, campaign and chain identity, and an opaque random receipt ID.
- Record type and registered reporting role.
- The evidence fingerprint and opaque references to prerequisite receipts.
- The small signed summary or decision needed for the audit trail.
- The provider's ECDSA signature, outside the signed receipt payload.

The shared proof export contains the blockchain, signed receipts, public keys and original length/head checkpoint. It does **not** contain the provider map, salts, traveler IDs, page paths, location histories or private keys. Traveler labels displayed beside receipts are local teaching annotations, not shared receipt fields.

This separation is a demonstration, **not an access-control system**. All components still run in the visitor's browser. The unavailability toggle emulates a provider failing to supply evidence; it is not a deletion, retention or compliance implementation. Salted hashes and opaque IDs are not a general privacy guarantee: receipt relationships, timing and results can still be sensitive. A real deployment needs permitted matching, access control, retention decisions and a threat model.

## Two verification levels

`verifyProof` checks the shared history against the supplied original trust registry and checkpoint: provider signatures, registered roles, receipt schemas, unique IDs, prerequisite receipt types, no reuse of a visit for a second attribution, known rule versions, valid correction targets, recomputed report totals, Merkle roots, block links and three distinct local validator signatures. It never claims to have obtained raw evidence merely because a hash matches a receipt.

`session.audit` first verifies the recorded receipt, then requests the separate provider packet and checks its salted fingerprint. For an attribution receipt, it also retrieves the referenced exposure and visit evidence and reproduces the example calculation. Its result distinguishes:

| State | Meaning |
| --- | --- |
| Record verified; evidence checked | The signed receipt matches the original ledger and the accessible packet matches its fingerprint. For attribution, the example rule was reproduced. |
| Record verified; evidence unavailable | The signature/history remain valid, but the independent evidence calculation cannot be completed. |
| Record verified; evidence mismatch | The source claim or accessible evidence does not reproduce the recorded result. |
| Record invalid | The supplied history does not match its original proof. |

The tests intentionally include a correctly signed but false attribution claim: shared-record verification succeeds while an evidence check finds the calculation mismatch. This is a deliberate illustration of the external-data/oracle boundary, not a defect to hide. Neither level proves the underlying observation is accurate, that all relevant observations were submitted, or that an ad caused a visit.

## Attribution and matching

The example rule is `demo-window-v1`: a provider's ad and visit evidence identify the same fictional traveler and campaign; the visit occurs 0–30 example days after exposure, inclusive. A website visit is optional. An out-of-window visit can be faithfully recorded without receiving credit. The scenario chooses observations, but the attribution service computes its result from that evidence rather than trusting a scenario-supplied credit flag.

Public ledger validation can confirm references and signatures without knowing the traveler identity or occurrence days. An authorized evidence check can reproduce the matching calculation. A production system would need a real, permitted matching arrangement or an existing provider; this project does not implement a data clean room, privacy-enhancing attribution protocol or cross-device identity service.

Authored occurrence days are different from accelerated receipt time and the browser's block-recording timestamp. None is independently certified. The mixes are teaching examples, not estimated campaign conversion rates.

## Reports, discrepancies and corrections

**Check a campaign report** pauses playback and safely cancels/requeues an uncommitted batch candidate. It appends a signed report through the latest committed block. Pending source measurements are excluded and may appear in a later report. The receipt records its exact coverage height, previous head hash and totals. Validation recomputes these totals from the historical prefix, not from current animation counters.

The report dialog edits a copy of that reported total. The copied proof is actually reverified; altered totals fail its signature/hash/policy checks. The original report and current totals are not modified. Entering the original value correctly passes instead of producing a scripted failure.

**Append example correction** emulates an attribution service withdrawing a previously credited decision because the provider reported a duplicate visit. A new signed receipt references that exact decision. Current totals are reduced once, while the earlier decision and signed report remain byte-identical. The correction reason is still a provider claim, not a newly proven real-world fact. This limited example supports one withdrawal per credited decision; it is not a general accounting or dispute-resolution system. **Sign updated report** creates a new snapshot of the corrected current total.

## Separately checked local copies

Advertiser, Agency and Measurement partner each hold a separate copy of the block history. Their reporting identities are separate from the three local ledger-signing identities. Before a block commits, the copy's own history is verified against the canonical prefix; a candidate update is checked before being applied. The synchronous commit uses these prechecked copies and a revision guard, so a copy changed during preparation cannot silently receive a stale approval.

- **Up to date** means that copy verifies against the current head.
- **Behind** means it is a valid earlier prefix, checked against the canonical prefix hash—not an arbitrary self-supplied head.
- **Mismatch** means the copied history no longer verifies.

Pausing delivery does not corrupt old records. Altering one local copy does not modify the other copies or the canonical history. Restore verifies the source history before replacing that demo copy. A mismatched copy is never silently healed by the next append.

All three identities, keys and copies are controlled by **one browser**. This is not independently governed distributed consensus or an assertion of Byzantine fault tolerance. A real consortium would need independently controlled participants, trust anchors, membership/key rotation and dispute/update rules. Even a correctly signed self-contained exported proof requires a separately trusted original key registry/checkpoint to establish external provenance.

## Source ownership and deployment

| File | Responsibility |
| --- | --- |
| `demos/ad-verification.html` | Authoritative entry page, controls, explanatory copy and one dialog host. |
| `js/demos/ad-verification-core.js` | v5 schemas, evidence store, signed receipts, batching, report/correction rules, proof/evidence checks and replica states. |
| `js/demos/ad-verification-player.js` | Independent source schedules, queue, shared clock, batch preparation, synchronous publish, cancellation and manual review actions. |
| `js/demos/ad-verification.js` | Persistent traveler DOM, receipt/block inspection, copy exercises, reports and public export. |
| `css/components/ad-verification.css` | Isolated, light-only two-panel styles and responsive/reduced-motion behavior. |
| `content/projects/adVerification.json` | Unpublished/unlisted metadata. |

Existing static copying and clean-URL handling publish the same route. No shared site bundles, route settings, dependencies or persistent data are modified. Schema v5 intentionally does not accept v4 proofs; earlier versions are available in Git history. No browser campaign migration is required because this project has never persisted campaign state.

The page retains `noindex, nofollow, noarchive`; metadata retains `published: false`, `hidden: true`, `noindex: true`, `visibility: "unlisted"`. The URL and GitHub source remain public to anyone who has them. Normal site-wide consent components injected by the website build are preserved. No real advertising, location collection, campaign network request, token, wallet or payment is added.

## Production decision

An ordinary signed, append-only audit log may be sufficient when all parties accept one operator. A jointly maintained blockchain needs an actual shared-governance requirement and partner participation. Production would also require access agreements, data availability/completeness reconciliation, reporting-period definitions, permissioned identity matching, security review and operational ownership. The demo is a candidate audit workflow, not a production deployment or proof of commercial feasibility.

## Verification

Using the existing Node 22 dependencies:

```sh
node --test tests/tools/ad-verification.test.js
npm run build
node --test tests/tools/ad-verification.test.js
npx playwright install chromium
node tests/tools/ad-verification.browser.cjs
```

Node checks cover cryptography, batched receipt ordering, source/evidence separation, unavailable evidence, false signed claims, attribution boundaries, duplicate credit, historical reports, authorized corrections, independent copy corruption/lag, all scenarios/speeds, overlapping activity while crypto waits, pause/reset, requeued manual reviews, reserved capacity, replay/rehash attacks and publication exclusions. The publication check explicitly skips before a full website build.

Native Chromium independently audits rendered frames, checks every newly committed receipt against its matching traveler milestone, reconstructs displayed totals from the accepted record stream and verifies exported proof using Node Web Crypto. It exercises evidence availability, reporting discrepancies, corrections and copy lag/corruption, plus 1500/1280/1024/768/390/320px layouts and native slow-verification/reduced-motion/error states. The existing scoped workflow uploads screenshots, viewport bounds and `animation-sync-evidence.json` as evidence, outside tracked source. Tests run on loopback and do not interact with real campaigns.

References: [Web Crypto verification](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/verify), [Hyperledger Fabric private-data architecture](https://hyperledger-fabric.readthedocs.io/en/latest/private-data-arch.html). This implementation is a dependency-free educational model, not an integration with Fabric.
