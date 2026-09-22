# Live Campaign Blockchain

Unlisted entry: `/demos/ad-verification`. **Cedar Valley Tourism and every traveler are fictional.** Real SHA-256 and ECDSA operations run locally in the browser; there is no real advertising, device tracking, or location collection.

## The experience

Choose **Mixed results**, **No website or destination visit**, **Website only**, **Destination only**, or **Website + destination**. Start the campaign to watch four travelers proceed while one shared ledger grows alongside them. Each group has four distinct synthetic traveler IDs. Mixed results uses one of each outcome; other scenarios give all four travelers the selected outcome. These are illustrative scenarios, not population estimates or conversion rates.

The campaign is created and purchased once. Traveler events then interleave: ad exposures, optional website sessions, optional attributed destination visits, and observation-window summaries. Each actual example event adds one block. A skipped visit does **not** create a placeholder visit block. The summary records which matching events were observed; no observation is not proof that no real-world trip happened.

**Keep adding travelers** appends more groups to the same campaign. Changing the scenario affects the next group, not an in-progress traveler. Earlier blocks and traveler histories remain intact. There are no ledger forks for different traveler paths: all blocks remain in one chronological chain, while `travelerId` and `previousTravelerEvent` identify the separate business-event paths. Playback stops before 256 blocks as a browser-performance safeguard.

Pause/resume and 1×/2×/4× controls share the same clock. A hidden tab pauses instead of racing ahead. Reset explicitly discards the session and generates new local signing keys. Group selection and traveler highlighting let visitors inspect earlier paths without deleting or filtering blocks out of the chain. The ledger follows new events until the visitor scrolls through history or switches off auto-follow.

On narrow screens, the traveler list and blockchain become two compact, scrollable panels. Starting playback scrolls to this paired view so both animations can be seen together. Reduced-motion preferences retain status changes and verification, without the moving traveler indicator or entry motion.

## Source ownership

| Responsibility | Source |
| --- | --- |
| Authored page and explanatory copy | [`demos/ad-verification.html`](../demos/ad-verification.html) |
| Sign, verify, prepare and append blocks; build scenario plans | [`js/demos/ad-verification-core.js`](../js/demos/ad-verification-core.js) |
| Shared playback clock, event queue, cancellation and commit boundary | [`js/demos/ad-verification-player.js`](../js/demos/ad-verification-player.js) |
| Traveler/ledger DOM views, inspection and editing | [`js/demos/ad-verification.js`](../js/demos/ad-verification.js) |
| Isolated light-theme layout and motion geometry | [`css/components/ad-verification.css`](../css/components/ad-verification.css) |
| Visibility metadata | [`content/projects/adVerification.json`](../content/projects/adVerification.json) |

The existing static copy step publishes `demos/`, `js/`, `css/`, and `img/`; the existing `cleanUrls` setting resolves the route. No new dependency, API, environment variable, wallet, or token is required. Shared site navigation and other project pages are unchanged.

The landscape and four-avatar sprite are cropped from the approved AI-generated concepts. They contain illustration pixels only; all UI labels, controls, data, and animation are native HTML/CSS/JavaScript. The existing site Inter font is referenced, not copied into a new asset.

## How animation synchronization works

The player owns **one `requestAnimationFrame` clock**, one queue, and one active event. Both views receive that event's ID, phase and normalized progress. The traveler indicator moves while the corresponding pending block enters and shows the same progress. There are no independent playback timers in the two views.

The stages are:

1. **Record:** show the event in its traveler path and a pending block. Nothing has been appended yet.
2. **Verify:** calculate a real record fingerprint, event signature, block hash, previous-block link, and validator approvals. Verify the proposed history against the trusted registry.
3. **Append:** once real verification and the visual interval have both finished, commit the private verified candidate and synchronously update both DOM views in the same animation frame.

Slow cryptography holds progress at 78% with a verifying status; it cannot produce a premature valid block. Pause freezes the clock even if signing finishes in the background. Speed changes affect the single clock. Reset retires the old signing session and invalidates all pending callbacks, preventing stale events from appearing in a new campaign. The UI does not clone or reverify the entire history on every animation frame.

## Cryptography and trust boundary

Each block holds one ECDSA P-256-signed event, its SHA-256 Merkle root, a previous-hash-linked header, and three distinct local validator signatures approving the recorded block hash. One event per block makes the teaching view straightforward; the Merkle helper also handles multiple transactions. Canonical JSON sorts keys, and hashing/signing messages use separate versioned domain prefixes.

Verification checks schema, sequence, campaign ID, expected actor, traveler-event ordering, event signatures, Merkle roots, header hashes, prior-block references, and all three approvals. A separately retained public-key registry, original length and terminal hash detect changes relative to the original session. An internal verified-ticket map prevents mutation of a public candidate copy from replacing the actual prepared block.

All validators and signing identities are in **one browser**, not independent organizations or Byzantine-fault-tolerant consensus. A person controlling the browser code and replacing its keys and checkpoint can create a different valid-looking demo. A signature authenticates bytes relative to a key; it does not prove an ad was shown to a human, a website session was genuine, or a trip happened. Destination attribution is a simulated measurement-provider claim. Timestamps are compressed example times, not trusted clock attestations or a realistic travel schedule.

Private keys are non-extractable and are not returned, serialized or exported. No real device IDs, raw coordinates, location requests, campaign network calls or campaign storage are added. Normal site-wide privacy/consent components may be injected by the standard website build and are not bypassed by the demo.

## Inspect and edit

Select any committed block to pause and inspect its campaign/traveler references, event data, original hash, recalculated hash, signature checks and raw record. Change a field and select **Apply edit & verify**. Only an editable copy changes; the original signing material and checkpoint are not replaced.

The altered record fails verification, and subsequent blocks show **Earlier change** rather than falsely claiming their own data was edited. Playback is blocked while history is invalid. **Restore original** restores the exact original signed bytes without generating new signatures. Playback can then resume from its paused event.

Public-proof export includes the displayed records, original public keys and checkpoint. An edited export therefore also fails independent verification. It is self-contained educational evidence, not external certification.

## Unlisted, not confidential

The metadata retains `published: false`, `hidden: true`, `noindex: true` and `visibility: "unlisted"`. The standalone page retains `noindex, nofollow, noarchive`. No public navigation entry is added. Tests check generated discovery indexes for accidental inclusion.

Anyone with the URL can still open the demo, and GitHub source is public. Unlisted and noindex settings are **not access controls**. Reloading or resetting discards the local session.

## Verification

On Node 22 with the repository's existing dependencies:

```sh
node --test tests/tools/ad-verification.test.js
npm run build
node --test tests/tools/ad-verification.test.js
npx playwright install chromium
node tests/tools/ad-verification.browser.cjs
```

The focused Node suite tests genuine cryptography, every scenario, append-only history, interleaved traveler links, edit detection, rehash attacks, original restoration, approval requirements, malformed proofs, slow signing, pause/resume, reset cancellation, speed settings and continuous scenario changes. Publication checks explicitly skip until a full build exists.

The native Chromium suite independently samples the rendered DOM **on every animation frame**. It compares event IDs, phases, shared progress values, actual block-progress widths and traveler-indicator positions. At each append, it checks that the matching traveler milestone commits in the same frame, that block count grows by one, and that no nonexistent visit was added. It runs at 1×, 2× and 4×, exercises all five scenarios and multiple groups, exports and independently verifies proofs, tests tamper/restore and reset, and checks six widths (1312, 1024, 820, 768, 390 and 320 pixels). Additional checks cover delayed native signing, reduced motion, missing engine and JavaScript-disabled states.

The scoped workflow builds the actual website before testing and uploads screenshots plus `animation-sync-evidence.json`. Set `BROWSER_ARTIFACT_DIR` to keep evidence outside tracked source. Browser tests use loopback only and do not submit production forms or contact advertising systems. A full build and focused tests are separate from production deployment verification.
