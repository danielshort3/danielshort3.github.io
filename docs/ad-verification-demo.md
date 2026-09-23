# Campaign Attribution Lab

Unlisted entry: `/demos/ad-verification`. This is an educational **campaign-attribution audit layer**, not a visitor-tracking service. All travelers, observations and participants are fictional. The campaign ID is `EXAMPLE-CAMPAIGN`; provider evidence uses the literal placeholder `Example destination`. No named tourism destination or destination logo appears in the demo. The browser performs real ECDSA, SHA-256, Merkle-root and checkpoint checks.

## Interface and layout contract

The demo follows the site's simple project format: a Projects return link, one Inter/navy heading, a short description and divider, compact controls, two work panels and a results band. The relevant style references are `css/components/page-masthead.css`, `css/components/project-demo-layout.css`, and the authored `demos/pizza-tips-demo.html` header. This unlisted standalone page does not change the shared project generator or public navigation.

**Mixed outcomes run automatically.** There is no scenario dropdown or client-side scenario listener. The five independent lanes include website-only, destination-only, both, neither and out-of-window paths. Start/Pause/Resume and Speed share one non-wrapping control cluster, with a reserved button width so the longer Resume label cannot push Speed to another row. Reset remains secondary. Individual scenario factories remain in the core/player for focused tests, not as hidden user-facing controls.

The left panel determines the desktop workspace height. Fixed-height traveler rows reserve space for changing outcome labels. The right-hand ledger uses CSS size containment so changing batch content cannot resize that shared grid row. A `minmax(0, 1fr)` record viewport scrolls inside the panel; heading, writer, full-history action and copy statuses retain their space. On stacked layouts, the ledger has an explicit responsive block size. Scrollbar gutters are reserved to avoid width/reflow jumps. Receipt text wraps rather than being truncated or hidden.

New records scroll only the inner ledger, not the document. Selecting an earlier expanded block keeps it accessible inside that viewport. The region is keyboard-focusable for Page Up/Down navigation. All five traveler lanes remain visible without nested traveler scrolling. The metrics row must stay at the same document position during playback, different batch sizes, block selection, copy status changes and report/correction appends. Resizing the viewport or changing text scale may intentionally reflow the layout.

## Parallel people, batched records

Five lanes use one simulation clock with independent event schedules. Source work is ordered within each fictional journey, but does not wait for the blockchain writer: people may continue to a website or destination while earlier receipts await recording. Blue denotes an observed/reported event; green denotes a committed signed receipt.

Completed source receipts enter a buffer. Up to four join a block, or a smaller batch is prepared after the maximum example wait. Each block contains actual signatures, a Merkle root and the previous block's hash. All matching recorded indicators change in the same synchronous update as the append. A traveler is replaced only after every receipt in that lane, including its closing record, is committed. Other lanes continue independently.

Pause freezes visible clocks, queue admission, commits and fades even when cryptography finishes in the background. Reset retires stale callbacks and signing material. Reduced motion removes movement and fades. The optional continuous checkbox lets current travelers finish without replacements. Admission reserves each traveler's entire future receipt cost and leaves 12 of the 192 receipt slots for reporting/correction exercises; old history is never silently discarded.

## Provider evidence versus shared receipts

A separate in-memory provider store contains salted example evidence packets, including synthetic traveler associations, occurrence days and page/place detail. Shared receipts contain version/chain/campaign identity, opaque receipt IDs, registered source roles, evidence fingerprints, prerequisite receipt references, small summaries/decisions and provider signatures. The public export `campaign-audit-proof.json` contains the signed blockchain, public keys and original checkpoint, not the evidence map, salts, traveler IDs, page paths, location histories or private keys. The visible traveler labels are teaching annotations.

This separation is not access control: every component runs in the same browser. The unavailability toggle models a provider failing to supply evidence, not secure deletion or retention compliance. Opaque IDs and salted hashes do not eliminate sensitivity in timing, relationship metadata or reported outcomes. Production would need an authorized matching arrangement, access/retention controls and a threat model.

## Two verification levels

`verifyProof` checks provider signatures, roles, schema, unique IDs, prerequisite receipt types, known rule version, no reuse of a visit for second credit, valid correction targets, report totals, Merkle roots, prior-block links, three distinct local approvals and the original checkpoint.

`session.audit` additionally retrieves a provider evidence packet and checks its fingerprint. For attribution it retrieves the ad and visit evidence and reproduces the calculation. Results distinguish verified record / checked evidence, verified record / unavailable evidence, verified record / evidence mismatch, and invalid record. The tests deliberately include a correctly signed false claim: its shared record verifies, while the evidence calculation does not. Neither level establishes the observation's real-world accuracy, completeness, or causation.

The example `demo-window-v1` matches the same synthetic traveler and campaign within 0–30 example days of ad exposure, inclusive. A website visit is optional. Late visits may be faithfully reported but not credited. Authored occurrence days, accelerated receipt time and the browser's recording timestamp are distinct and none is independently certified. The mix is not an estimate of real conversion rates.

## Reports and corrections

Check a campaign report pauses playback and safely requeues an uncommitted candidate before appending a signed snapshot through the current block. Pending measurements appear only in later reports. Its coverage height, head hash and totals are reproducible from the historical prefix.

The edit exercise changes only a copy. A changed total or receipt fails real proof validation; the original history and totals remain unchanged. Entering the original value correctly passes. An authorized example correction appends a signed withdrawal that references a credited attribution. Current totals change once while prior decisions/reports stay byte-identical. An updated report can then capture the corrected total. The duplicate explanation is itself a provider claim; this is not a general accounting/dispute system.

## Three local ledger copies

Advertiser, Agency and Measurement partner keep distinct in-memory histories. Their copies are verified before updates. A valid prefix is **Behind**, not corrupted or current; changed history is **Mismatch**; a valid current head is **Up to date**. Pause delivery, alter a copy and restore verified history demonstrate these differences. A changed copy is not silently healed by the next append. Revision guards reject candidates prepared before a copy mutation.

All identities, keys, copies and approvals share one browser owner. This is not independently governed distributed consensus or Byzantine-fault tolerance. External provenance would require independently trusted participant identities and checkpoints. A signed append-only log may suffice when everyone trusts one operator; a consortium ledger needs an actual shared-governance requirement.

## Source ownership and publication

- `demos/ad-verification.html`: authoritative page, controls and ledger viewport.
- `js/demos/ad-verification-core.js`: v5 schemas, separate evidence, receipts, batches, reports/corrections and copy checks.
- `js/demos/ad-verification-player.js`: parallel scheduling, batching, synchronous commits and cancellation.
- `js/demos/ad-verification.js`: persistent lanes, inner-ledger scrolling, inspection, reports and export.
- `css/components/ad-verification.css`: isolated project styling and stable-height layout.
- `content/projects/adVerification.json`: unlisted metadata.

Existing static copying and clean URLs publish the same entry point. Shared site bundles, dependencies, settings and original illustration files are untouched; the old destination logo is no longer referenced. Cache-versioned demo URLs avoid mixing the changed markup and script. The v5 proof schema remains unchanged; renaming this fictional campaign does not invalidate older v5 proofs. No migration is needed because sessions are not persisted.

The page remains `noindex, nofollow, noarchive`, with `published: false`, `hidden: true`, `noindex: true` and `visibility: "unlisted"`. Anyone with the URL can access it and the repository is public; these are not access controls. Site-wide consent components are preserved. No advertising/location request, account, wallet, token, payment or persistent campaign storage is added.

## Validation

Using the existing Node 22 dependencies:

```sh
node --test tests/tools/ad-verification.test.js
npm run build
node --test tests/tools/ad-verification.test.js
npx playwright install chromium
node tests/tools/ad-verification.browser.cjs
```

The existing Node suite keeps all individual scenario, batching, cryptography, evidence, false-claim, report/correction, replica, capacity, pause/reset and publication tests. Only mixed outcomes are exposed by the page.

Native Chromium independently samples both panel heights and the results band's document position throughout real playback and audit interactions. The browser suite checks default mixed totals at 1x/2x/4x, no scenario dropdown, adjacent Speed/Start/Resume geometry, generic identity even in private evidence and exports, variable batch sizes, different selected blocks, long wrapped receipts, keyboard ledger scrolling, and eight widths (1500/1280/1024/901/900/768/390/320). Existing evidence unavailability, rejected report edits, corrections, copied-history checks, slow native verification, reset, reduced-motion and no-JavaScript/error flows remain covered.

The scoped CI workflow builds the actual website before testing. Screenshots, bounds and `animation-sync-evidence.json` are artifacts outside tracked source. Browser-plugin absence permits Playwright fallback; blocked local navigation must not be represented as native integration success. Tests use loopback and do not submit production forms.
