# Tools Audit and Hardening Plan

Date: 2026-07-11  
Scope: the 14 records in `content/tools/`, their browser clients, the shared tools account/session system, and the Short Links, GA4, Job Application Tracker, and File Transcriber backends.

## Executive status

The tools portfolio is in materially better shape than its starting point. The working tree now has explicit browser resource budgets, safer cloud-storage boundaries, bounded API responses, lifecycle metadata, conflict-aware session writes, and more consistent run telemetry. The 10 public tools remain browser-first; the two account tools and two admin tools are still correctly hidden from the public directory and marked `noindex`.

This is a release candidate, not proof of a production deployment. “Implemented” below means the behavior is present in the current source. Cloud-backed items remain “deployment-gated” until their infrastructure, environment variables, data migrations, and live canaries have been verified. Final build, regression, browser, and production smoke results should be recorded in the verification section before release.

Current portfolio:

| Access class | Count | Tools | Current assessment |
| --- | ---: | --- | --- |
| Public, browser-first | 10 | Text Compare, Non-breaking Space Cleaner, Oxford Comma Checker, Point of View Checker, Word Frequency Analyzer, UTM Batch Builder, QR Code Generator, Image Optimizer, Background Remover, Screen Recorder | Functional and substantially hardened. Remaining work is mostly performance benchmarking, aggregate budgets, and cross-browser automation. |
| Authenticated account | 2 | Job Application Tracker, File Transcriber | Core safety changes are implemented. AWS/Vercel deployment, migration, and live failure-path canaries are required. |
| Admin-only | 2 | Short Links, GA4 UTM Performance | API hardening is implemented. Short Links needs a reservation backfill before the new write path is enabled; GA4 still uses per-instance rate/cache state. |

Release recommendation: complete the P0 migration and verification gates, then deploy. Do not treat a credential-related failure during this pass as a tool regression until the credential-standardization work has been reconciled.

## Ownership boundary: authentication and credentials

Another agent owns login credentials and credential/environment-name standardization. This audit intentionally does not replace that work or write secret values.

Before merge or deployment:

1. Compare that agent’s final Cognito, Vercel, AWS, GA4, and admin-token contract with `.env.example`, `js/accounts/tools-config.js`, the Vercel project environment, and the Job Tracker CloudFormation parameters.
2. Preserve least-privilege, tool-specific credential prefixes where the final standard calls for them; do not reintroduce ambiguous shared credentials accidentally.
3. Do not commit access keys, client secrets, service-account JSON, signing secrets, or admin tokens.
4. Re-run authenticated and admin canaries after the credential work lands. A temporary 401, missing-token response, or unavailable signed-in catalog entry during concurrent work is not sufficient evidence that the underlying tool implementation is broken.

## Status definitions

- **Implemented**: present in the current working tree and covered by source-level contracts or direct code inspection.
- **Deployment-gated**: implementation exists, but production infrastructure, configuration, or data state must be changed and verified.
- **Residual**: a worthwhile follow-up that is not required to understand the current release boundary.
- **P0**: release blocker; **P1**: next hardening milestone; **P2**: planned optimization; **P3**: optional enhancement.

## Per-tool working, changed, and remaining matrix

| Tool | Access / execution | What is working | Implemented in this pass | Remaining recommendation |
| --- | --- | --- | --- | --- |
| Text Compare | Public / browser | Two-draft comparison, background-worker execution, main-thread fallback, multiple comparison modes, imports, copy/export, and account-session summaries. | Enforces a 24 MiB import cap, 600,000 combined characters, 200,000 tokens, and 120,000-character saved-output bounds. Explicit runs now emit success, validation, and processing outcomes without logging text. | **P2:** benchmark at every cap on low-memory mobile devices; add an explicit cancel control for long comparisons; retain the main-thread fallback but warn before it processes near-limit input. |
| Non-breaking Space Cleaner | Public / browser | Detects and replaces hard spaces, reports non-ASCII characters, provides cleaned output, supports imports, and analyzes the full input. | Limits imports to 24 MiB, limits rendered preview work to 4,000 characters while clearly stating that the full input was analyzed, and emits bounded run outcomes. | **P2:** consolidate repeated full-text scans into one pass or move analysis to a worker; add a measured paste-size budget if main-thread latency exceeds the target on low-end devices. |
| Oxford Comma Checker | Public / browser | Finds candidate serial-list issues, highlights results, supports custom conjunction settings, imports, and session restore. | Limits imports to 24 MiB and saved highlighted HTML to 120,000 characters; reports validation, findings/no-findings, and processing outcomes. | **P2:** move near-limit parsing/highlighting off the main thread or introduce a character cap; add a curated true-positive/false-positive language corpus so heuristic changes are measurable. |
| Point of View Checker | Public / browser | Analyzes narrator/person usage, highlights evidence, reports drift, supports configuration/imports, and restores useful session state. | Limits imports to 24 MiB, saved highlighted HTML to 120,000 characters, and saved drift rows to 220; adds explicit outcome telemetry. | **P2:** workerize the largest analysis paths or cap pasted characters; add regression corpora for dialogue, quotations, and intentional POV shifts; benchmark DOM rendering at the drift-row limit. |
| Word Frequency Analyzer | Public / browser | Tokenizes text, applies stopword/custom-list controls, ranks terms, shows occurrence context, imports documents, and exports results. | Limits imports to 24 MiB, retained hits to 600 per term, visible occurrence snippets to 12, and full-text preview to 180,000 characters; adds explicit result/validation telemetry. | **P2:** cap or stream the unique-term map for adversarial all-unique input; workerize tokenization if the 24 MiB ceiling causes long tasks; add memory and multilingual-tokenization benchmarks. |
| UTM Batch Builder | Public / browser worker | Builds combinations/template rows, imports CSV, validates relationships, previews, cancels, and exports generated campaign URLs. | Hard-caps CSV input at 5 MiB and generated rows at 50,000, bounds saved-session CSV/output previews, adds accessible names to dynamic controls, improves narrow-screen upload layout, and distinguishes start/complete/error/cancel signals. | **P2:** fuzz CSV quoting/newline cases, measure export memory at 50,000 rows, and add end-to-end worker cancellation/restart tests. Consider streaming export only if measurements show a real memory problem. |
| QR Code Generator | Public / browser | Generates multiple payload types, styling variants, captions, center content, presets, sharing, config import/export, and bitmap/vector downloads. | Removes Wi-Fi passwords and encoded Wi-Fi payloads from persisted presets, last config, share snapshots, and cloud-session captures; clears the password after session restore; derives QR version from actual module count. Existing browser presets are sanitized and rewritten when read. | **P1:** decide whether contact/vCard and other personal payloads should also be local-only or require explicit cloud-save consent. **P2:** add decoder-based round-trip tests across ECC, logo, caption, and export variants. |
| Image Optimizer | Public / browser | Decodes supported raster images, resizes/converts them, generates responsive variants, and downloads individual or aggregate outputs. | Restricts input to PNG/JPEG/WebP/AVIF; pre-parses bounded codec headers before full decode; caps selection at 20 files, 25 MiB each, 150 MiB total, 12,000 px per side, 40 MP per image, and 120 MP total. It preflights 256 MiB of output-canvas memory, caps encoded output at 200 MiB, uses 96 px thumbnails, processes metadata sequentially, releases canvases/bitmaps/URLs, and supports cooperative cancellation. | **P2:** run Safari, Firefox, Android, and low-memory canaries for codec/parser and canvas behavior. OffscreenCanvas/worker encoding is optional after measurement; do not add it without preserving the current deterministic limits and fallback. |
| Background Remover | Public / browser with model assets | Queues files, supports AI and legacy removal, refinement, comparison, and downloads. | Enforces 20 files, 40 MiB per file, 160 MiB aggregate input, 80 MP per image, 160 MP aggregate decoded pixels, and a 16,384 px canvas ceiling. It also repairs cutout object-URL replacement/revocation so displayed previews are not left attached to revoked URLs, and standardizes start/complete/error/cancel outcomes and error classes. | **P1:** add a visible cancel path if current queue cancellation is not discoverable; bound and report model download/startup timeout failures. **P2:** test aggregate boundaries plus CDN/CSP/offline/model-fallback paths in browsers. |
| Screen Recorder | Public / browser media APIs | Captures screen/window/tab, optional audio/camera composition, crop/layout controls, pause/resume, first-frame assets, multiple encodings, and downloads. | Defaults to one automatic encoder, permits at most two explicit recorders, stops after 30 active minutes or 192 MiB aggregate retained chunks, uses 1-second chunks, reports the active limit, skips in-memory ZIP materialization above 64 MiB in favor of separate downloads, handles recorder errors, and emits one terminal run outcome. | **P2:** test permission denial, tab-share end, audio absence, pause-time accounting, multi-download permission behavior, and MP4/WebM behavior across supported browsers. Consider File System Access streaming for long desktop captures, while keeping the bounded in-memory fallback. |
| Job Application Tracker | Authenticated / API Gateway + Lambda + DynamoDB + S3 | Application/prospect CRUD, views, follow-ups, analytics, attachments, short-link set integration, and bounded data export. | Uses exact-size presigned S3 POST policies with random staging keys and a one-day staging tag; verifies S3 metadata before attaching; persists actual size/ETag; enforces attachment count and size on both sides; retags attached files; rejects stale attachment/application edits with an `updatedAt` precondition; adds the real prospect DELETE route; makes application delete record-first with cleanup reporting; scopes pagination cursors; caps list pages at 500 and internal analytics at 10,000 records/4 MiB/25 Dynamo pages; adds a combined analytics dashboard route; and caps synchronous exports at 1,000 applications, 50 attachments, 50 MiB input, and 8 MiB metadata. The frontend remains compatible with the old PUT response during staged rollout. | **P1:** replace large synchronous exports with a `202 Accepted` queue/worker/status flow; use a durable cleanup queue for failed S3 deletes; inventory and remove historical orphan objects; upgrade/archive-test the older `archiver` dependency chain. Live deployment and failure canaries remain mandatory. |
| Short Links | Admin / Vercel API + DynamoDB | Link/set CRUD, redirects, expiration, click history, export, health checks, and destination tests. | Atomically reserves lowercase slug identities, updates the link total, detailed event, and historical baseline in one transaction, retains click history without TTL expiration, canonicalizes click keys, prevents phantom records, clears stale link expiration, bounds JSON bodies, adds opaque storage cursors, minimizes click telemetry, exposes retention status in health, and hardens outbound destination tests against private addresses, redirect pivots, and DNS rebinding by pinning validated addresses. | **P0:** resolve lowercase duplicates and backfill reservation records for every existing link; run the click-history reconciliation to remove legacy `expiresAt` values, create honest historical baselines, and disable clicks-table TTL before relying on permanent retention. **P1:** add a queryable index/materialized view for filtered/sorted listings so large catalogs do not require full scans; audit pre-change click records for fields no longer retained. |
| GA4 UTM Performance | Admin / Vercel API + GA4 Data API | UTM URL reports, grouping, drilldown, Explore, insights, exports, and client-side stale-request guards. | Uses timing-safe admin-token comparison, an explicit property allowlist, real-date validation with a 366-day default cap, bounded row counts and render counts, one shared 25-second upstream deadline across pagination, transient retries, rejected-token refresh, a 3 MiB final response cap, a bounded short response cache, bounded per-instance rate state, aggregate cache-byte limits, and clearer mobile tab overflow. Generic session snapshots remain disabled. | **P1:** move rate limiting to a shared store if the endpoint will have meaningful multi-instance traffic or broader exposure. **P2:** decide whether response caching also needs a shared store; add canaries for GA4 quota exhaustion, token refresh, and maximum-row pagination. |
| File Transcriber | Authenticated / Vercel API + S3 + Amazon Transcribe + DynamoDB ledger | File preflight, cost estimate/approval, direct upload, deterministic job start, adaptive status polling, transcript display/copy, retry recovery, and metadata-only session capture. | Fails closed in production without the usage ledger; atomically reserves per-user daily cost/files/concurrency and global daily cost/files; pessimistically reserves the provider’s 28,800-second maximum cost; uses exact-size/content-type tagged presigned POSTs; makes start idempotent; returns a recovery token for ambiguous starts; safely refunds only confirmed missing jobs; retries transient starts/polls; adapts polling cadence; offers Resume using the existing run/quote token; preserves only expiring, subject-scoped recovery metadata in same-tab session storage across reloads; constrains transcript fetch time, redirect hosts, raw bytes, and final response bytes; and cleans uploaded media on terminal paths. | **P1:** add scheduled delayed deletion of retained Transcribe job records after the run-token retention window, plus durable retries for failed media cleanup. Replace pessimistic maximum-duration reservation with authoritative server-side duration extraction when a dependable parser is available. **P2:** add a deployment-wide concurrency cap and a durable presign/start request limiter if observed demand warrants them. |

## Cross-cutting architecture and status

### Catalog, routes, and visibility

`content/tools/*.json` is the canonical set of 14 catalog records. Generated directory data contains all 14, while public cards contain only the 10 public, non-hidden, non-`noindex` records. Authenticated and admin records remain hidden until the account context allows them. `transcribe` is the canonical tool ID; `whisper-transcribe-monitor` remains a compatibility alias for historical saved data.

The same identifiers are also repeated in `api/_lib/tools-api.js` and `js/accounts/tools-account-ui.js`. They currently agree, but this duplication is a drift risk. A future build step should generate the API allowlist and account display catalog from `content/tools/` while preserving explicit legacy aliases.

### Shared tools account and session lifecycle

Implemented shared protections:

- Writes accept only known tool IDs; reads retain syntactic compatibility for historical IDs.
- Tools JSON bodies are capped at 512 KiB; saved snapshots are capped at 300,000 bytes.
- Session creation and every mutation require an expected version (`0` for create), and DynamoDB transactions enforce optimistic concurrency.
- Default quotas are 50 sessions per tool and 250 per account.
- Default retention is 365 days for sessions and 90 days for activity, using the configured DynamoDB TTL attribute.
- Activity payload data is capped at 32,000 bytes.
- Tool and global history support opaque cursor pagination and “Load more” UI.
- Individual deletion updates session records, index records, and quota counters transactionally, with reconciliation on an ambiguous cancellation.
- Delete-all uses an account-owned, stale-safe lock, ownership checks on batches, and a 10,000-item guard.
- Expired/stale session links recover into a new session rather than repeatedly failing.
- Page-exit saves use browser `keepalive` only below a conservative 60 KiB serialized-body budget; larger valid snapshots fall back to the normal request path and remain covered by the regular autosave interval.
- Production storage fails closed when DynamoDB is unavailable. The reduced KV compatibility path is local-only and requires `TOOLS_ALLOW_KV_COMPAT=true`.

Privacy boundaries:

- Generic cloud sessions are disabled for GA4, Short Links, and Job Application Tracker.
- Transcriber sessions store workflow metadata, not transcript bodies.
- Transcriber reload recovery uses expiring same-tab storage scoped to the authenticated subject and excludes media bytes and transcript content.
- QR Wi-Fi passwords and encoded Wi-Fi payloads are excluded from local/share/cloud snapshots.
- Password, file, hidden, and other secret-like form controls are skipped by generic capture.

Deployment-gated session work:

- Enable DynamoDB TTL on the configured attribute, normally `ttl`.
- Audit legacy records without TTL/version metadata. Backfill TTL for old session/index/activity records if they should expire without being re-saved.
- Dry-run counts for each account/tool before trusting quota metadata; the code reconciles counters during normal creation/deletion, but a one-time report will expose legacy skew early.
- Use a disposable account to test the destructive delete-all path and the 15-minute stale-lock recovery.

### Run telemetry and observability

Run-oriented tools now use `tools:run-start`, `tools:run-complete`, `tools:run-error`, and, where applicable, `tools:run-cancel` with bounded classifications rather than input/output content. This covers text analysis, media processing, UTM generation, GA4 reports, and transcription. CRUD-oriented surfaces and the continuously rendered QR generator do not need to pretend every edit is a “run.”

The next observability step is an operational scorecard that aggregates only coarse counts and latency/error buckets. It should not add document text, URLs with query strings, filenames, transcript content, QR payloads, application details, or credentials to analytics.

### Resource and response budgets

The current implementation establishes useful boundaries at four layers:

1. Browser input: import bytes, row counts, file counts, dimensions, decoded pixels, recorder duration, and retained chunks.
2. Browser output: preview characters/rows, saved-session output, canvas estimates, encoded output, and rendered GA4 rows.
3. API input: bounded JSON bodies, strict fields, allowed properties, dates, cursor sizes, and exact S3 POST policies.
4. API output/cost: paginated lists, capped exports/transcripts/report rows, timeouts/retries, daily spend/file ledgers, and retention tags.

The largest remaining browser-performance gap is worker/cancellation coverage for near-limit text and model workloads. The largest server architecture gaps are synchronous Job Tracker exports/cleanup and per-instance GA4 limiting.

### Data retention and cleanup

Retention is explicit in code but only effective after infrastructure configuration:

- Tools sessions: `ttl`, 365 days by default.
- Tools activity: `ttl`, 90 days by default.
- Short-link click events: retained indefinitely; no TTL attribute is written and clicks-table TTL must remain disabled.
- Transcribe ledger: configured TTL, 45 days by default.
- Transcribe media: `tool=amazon-transcribe` and `retention=temporary`; S3 lifecycle must enforce expiry.
- Job Tracker staging uploads: `purpose=staging`, one day.
- Job Tracker generated exports: `purpose=export`, seven days.
- Job Tracker incomplete multipart uploads: abort after one day.

Application records and attached Job Tracker files are user-owned durable data and should not receive an automatic short TTL. Their deletion path and future cleanup queue are the lifecycle boundary.

## Deployment and migration plan

### Phase 0: coordinate and inventory

1. Reconcile the credential-standardization agent’s final environment contract; do not copy secret values into files.
2. Snapshot the current Vercel environment-variable names, AWS stack outputs, Cognito client IDs, DynamoDB table schemas/TTL status, S3 CORS/lifecycle rules, and relevant IAM policies.
3. Export or back up the Short Links table and the shared tools table before structural backfills.
4. Record lowercase duplicate slugs, legacy session counts, click-table TTL status, untagged Job Tracker objects, and Transcribe bucket lifecycle status.
5. Confirm root and Job Tracker lockfiles include `@aws-sdk/s3-presigned-post`; build deployment bundles from the committed lockfiles.

### Phase 1: infrastructure prerequisites

1. **Shared tools table:** grant the current transactional/query/update/delete permissions; enable TTL on `TOOLS_DDB_TTL_ATTRIBUTE` (normally `ttl`).
2. **Short Links:** grant `dynamodb:TransactWriteItems` for reservation/write/delete/click transactions and `dynamodb:DescribeTimeToLive` for health. Disable TTL on the clicks table and remove legacy `expiresAt` attributes so detailed events remain durable.
3. **Transcriber:** provision/confirm the `pk`/`sk` ledger table, transactional IAM, TTL, private upload bucket, POST CORS, provider read access, and tag-filtered lifecycle expiry before deploying the fail-closed API.
4. **Job Tracker:** prepare the updated CloudFormation stack with POST CORS, staging/export lifecycle rules, delete/tag/list permissions, new prospect/dashboard routes, 512 MiB Lambda memory, and a 28-second timeout. Package the Lambda with its new dependency.
5. **GA4:** set the property allowlist and request/row/cache/rate ceilings. Keep the service-account secret in the approved secret source from the credential-standardization work.

### Phase 2: mandatory data migrations

1. **Short-link reservation backfill (P0):** group every non-internal link by lowercase slug; resolve any collisions explicitly; then conditionally create `__slug_lower__/<lowercase>` reservation records pointing to the chosen canonical slug. Re-run the audit and require one reservation per live link before enabling writes through the new API.
2. **Tools sessions:** produce a dry-run report for legacy session, tool-index, user-index, and activity records. Backfill TTL where retention is intended, and initialize/reconcile account/tool counters without changing snapshots.
3. **Short-link click reconciliation:** preserve unrecoverable older totals as aggregate-only baseline records, remove legacy click-event `expiresAt` attributes, and audit pre-change rows containing full referrers or coordinates before retaining them permanently.
4. **Job Tracker objects:** inventory old untagged staging/export objects and keys that no longer map to an application. Tag or delete them through a reviewed one-time process; do not infer ownership from filenames alone.
5. **QR browser storage:** no centralized migration is possible. The client sanitizes old presets/last-config data when each browser next reads it; document that limitation.

### Phase 3: staged application deployment

1. Run all local static/build/regression gates below from a clean dependency install.
2. Deploy the Vercel frontend/API only after the shared tools, Short Links, Transcriber, and GA4 prerequisites are ready. The Transcriber intentionally reports unavailable when its required ledger is missing.
3. For Job Tracker, deploy the dual-compatible frontend first. It accepts both legacy PUT and new POST presign responses. Then deploy the updated Lambda/CloudFormation stack; an old frontend cannot consume a POST-only response.
4. Keep the old API artifacts available for immediate application rollback, but do not roll back completed data safety migrations such as slug reservations or enabled TTL without a separate data review.

### Phase 4: canary and release

1. Run signed-out, authenticated, and admin catalog-visibility checks.
2. Run the public-tool browser matrix at desktop and 390 x 844 mobile viewports.
3. Use disposable account/admin records for cloud canaries, including intentional limit, conflict, timeout, and cleanup failures.
4. Confirm DynamoDB/S3/GA4/Transcribe logs contain bounded identifiers and error classes, not sensitive content.
5. Watch 4xx/5xx rate, Lambda/API duration, DynamoDB throttles/cancellations, S3 cleanup failures, GA4 429s, Transcribe spend reservations, and browser run-error buckets through the initial release window.

## Verification checklist

Record the command, date, commit, and result next to each item. Do not mark production verified from source inspection alone.

Local verification snapshot: 2026-07-11, current uncommitted working tree. The local shell used Node 24.18.0 while `package.json` targets Node 22.x; npm emitted an engine warning, but clean install, build, and tests passed. Repeat the release gates on Node 22 in CI/deployment.

### Repository and dependency gates

- [x] `npm ci` — passed; 53 root packages audited with zero vulnerabilities.
- [x] `npm run build` — passed; 2,022 files copied to `public/`.
- [x] `npm test` — passed 5,836 checks plus SEO validation.
- [x] `git diff --check` — passed; only existing line-ending conversion warnings were reported.
- [x] `node --check` for every changed server/client JavaScript file outside generated bundles — 44 modified JavaScript files passed.
- [x] `npm audit --omit=dev` — root and Job Tracker reported zero vulnerabilities.
- [x] In `aws/job-application-tracker/`: `npm ci`, `npm ls`, and `npm audit --omit=dev` passed.
- [x] `aws cloudformation validate-template --template-body file://aws/job-application-tracker/template.yaml` passed during the Job Tracker implementation review.
- [x] Generated `public/` assets match source pages/scripts; all 14 clean tool routes returned HTTP 200.

### Public browser-tool checks

- [x] Each of the 10 public tools loads at its clean route with no uncaught console errors; the four hidden account/admin routes also loaded cleanly.
- [x] All 10 public tools fit a 390 x 844 browser viewport without document-level horizontal overflow.
- [ ] Text Compare: changed/no-change inputs, worker fallback, import cap, character/token caps, save/restore.
- [ ] NBSP Cleaner: no-findings/findings, 4,000-character preview notice, full cleaned output, import cap.
- [ ] Oxford and POV: validation, findings/no-findings, highlighted-output save bounds, large-import response.
- [ ] Word Frequency: stopword/custom settings, all-unique input, preview/hit caps, import cap.
- [ ] UTM Builder: CSV quoting, 5 MiB rejection, 50,000-row boundary, cancel/restart, CSV export, mobile labels.
- [ ] QR: URL/Wi-Fi/vCard generation, decoder round trip, config import, preset migration, and proof that Wi-Fi secrets do not appear in local/share/cloud snapshots.
- [ ] Image Optimizer: each supported codec, unsupported input, every file/byte/pixel/output boundary, cancel, URL cleanup, and multi-file download.
- [ ] Background Remover: AI/legacy paths, model/CDN failure, queue cancellation/removal, object-URL replacement, and per-file plus aggregate byte/pixel limits.
- [ ] Screen Recorder: permission denial, capture-ended event, audio/no-audio, pause accounting, one/two encoders, 30-minute/192 MiB simulated boundaries, and final download.
- [ ] Repeat representative flows in Chromium, Firefox, and Safari/WebKit where the APIs are supported; include a low-memory mobile device or emulator.

### Shared account/session checks

- [ ] Unknown tool writes are rejected; the historical Transcribe alias remains readable.
- [ ] Create with `expectedVersion: 0`, update/delete with the current version, and reject a stale concurrent tab with a clear recovery message.
- [ ] Per-tool and per-account quotas reject only new sessions and do not block updates.
- [ ] Tool and global history cursors load every item once; “Load more” preserves selection and ordering.
- [ ] Expired/missing session URLs start a clean session; transient storage failures do not silently discard the active ID.
- [ ] Metadata edits/deletes synchronize the current tool page, account modal, and dashboard.
- [ ] TTL attributes are numeric epoch seconds and DynamoDB reports TTL enabled on the expected attribute.
- [ ] Individual delete decrements both counters; delete-all removes every known-tool partition plus activity and releases its lock.
- [ ] Production fails closed without DynamoDB; local KV compatibility works only when explicitly enabled.

### Short Links canaries

- [ ] Reservation migration reports no lowercase duplicates and one matching reservation per live link.
- [ ] Concurrent attempts to create `Example` and `example` produce one success and one 409 conflict.
- [ ] Changing a link to permanent removes an old expiration; deleting it removes its owned reservation.
- [ ] A missing slug cannot be created by a click increment; mixed-case redirects increment the canonical link.
- [ ] Click telemetry omits query strings, full referrers, latitude, and longitude; each new click updates the aggregate, event, and baseline atomically; retention health reports TTL `DISABLED`.
- [ ] Destination test rejects loopback/private/link-local/IPv4-mapped targets and revalidates every redirect while preserving the TLS hostname.
- [ ] Storage cursor pages do not duplicate/skip links. Filtered/sorted mode remains correct until a query index replaces its scan.

### GA4 canaries

- [ ] Missing/incorrect admin token is rejected before rate state or upstream work.
- [ ] Allowlisted property succeeds; a different numeric property receives 403.
- [ ] Invalid, reversed, future, and over-limit date ranges receive 400.
- [ ] UTM, Explore, and Insights row caps are enforced across pagination; client render caps show truncation notes.
- [ ] A transient 429/5xx is retried within the overall deadline; a 401 clears the cached access token once.
- [ ] Repeated identical success returns a cache hit without exceeding entry/byte ceilings.
- [ ] Multi-instance behavior is measured so the residual risk of per-instance rate/cache state is explicit.

### Job Tracker canaries

- [ ] Create/update/delete one application and one prospect; prospect deletion uses its dedicated route.
- [ ] Upload a valid attachment by POST; a different byte count/content type or unattached key is rejected at policy/verification time.
- [ ] Saved attachment metadata matches live S3 size/ETag and the object changes from staging to attachment retention.
- [ ] Attachment count/size limits apply on create and update without silently dropping old/new files.
- [ ] Application deletion removes the record first; forced S3 failure returns `cleanupPending` and restores short staging retention.
- [ ] Cursor scope prevents reuse across user/type/range/sort; all pages load once.
- [ ] Dashboard route matches legacy aggregate endpoints; chart-load failure has a visible bounded error.
- [ ] Export succeeds below every cap, rejects above every cap, reuses a valid cached export, and expires the result by lifecycle.

### Transcriber canaries

- [ ] Config reports unavailable when required region/bucket/signing secret/ledger is missing.
- [ ] Presigned POST enforces exact quoted bytes, content type, prefix, and temporary tags; upload bucket remains private.
- [ ] A quote replay and start retry reuse one ledger run/job and charge counters once.
- [ ] User daily file/cost, global daily file/cost, and per-user concurrency limits fail closed transactionally.
- [ ] Ambiguous AWS start returns recoverable `PENDING_CONFIRMATION`; Resume uses the existing token without re-uploading/recharging.
- [ ] A confirmed missing job refunds exactly once; completed/failed jobs remain charged and release their slots.
- [ ] Status polling retries transient failures, slows over time, and restores from the same signed run after a network interruption or same-tab reload.
- [ ] Transcript redirect validation rejects untrusted hosts; fetch/output byte caps and timeouts return bounded errors.
- [ ] Terminal paths delete media; lifecycle removes any forced orphan. Future delayed job deletion waits beyond the run-token retention window.

## Prioritized residual roadmap

Risk means the operational/product risk if the item is deferred. Effort is a relative engineering estimate and excludes approval/deployment lead time.

| Priority | Initiative | Tools / layer | Risk if deferred | Effort | Exit criterion |
| --- | --- | --- | --- | --- | --- |
| P0 | Complete Short Links lowercase-reservation migration and TTL/IAM setup | Short Links | High: case-colliding legacy slugs can bypass the new reservation model; click retention may remain unenforced. | M | Zero unresolved lowercase duplicates, one reservation per live link, transaction canary passes, health reports TTL enabled. |
| P0 | Complete cloud infrastructure/config reconciliation and final full verification | Shared sessions, Transcriber, Job Tracker, GA4 | High: fail-closed routes can be unavailable or lifecycle/cost boundaries can exist only on paper. | M | All infrastructure and verification checklists pass on the release commit; production canaries are recorded. |
| P0 | Reconcile credential-standardization output without duplicating ownership | All authenticated/admin tools | High: mismatched issuer/client/env names can look like application failures or expose overly broad credentials. | S | One documented env contract, no secrets in git, signed-in/admin canaries pass. |
| P1 | Add durable asynchronous export and cleanup workflows | Job Tracker | High at scale: synchronous ZIPs can hit gateway limits; best-effort deletes can leave durable objects. | L | `202` job/status/download flow, idempotent worker, retry/DLQ, cleanup metrics, old sync path retained only for small exports or removed safely. |
| P1 | Add delayed Transcribe job cleanup and durable media cleanup | File Transcriber | Medium-high: AWS job records and failed cleanup artifacts accumulate. | M | Scheduled cleanup waits beyond token retention, is idempotent, retries, and reports backlog/age. |
| P1 | Replace pessimistic Transcribe duration reservation with authoritative server duration | File Transcriber | Medium: the safe $11.52-per-file reservation sharply limits legitimate short jobs. | L | Server-derived duration cannot be client-forged, preserves atomic budget enforcement, and has fallback behavior for unsupported containers. |
| P1 | Decide and implement cloud-save privacy policy for personal QR payloads | QR Generator / sessions | Medium: vCard/contact payloads can be intentionally useful but personally sensitive. | S-M | Payload classes have documented local/cloud behavior and an explicit opt-in where required. |
| P1 | Move GA4 rate limiting to shared state if traffic warrants it | GA4 | Medium: per-instance windows do not enforce a deployment-wide request ceiling. | M | Load test proves a single global ceiling and bounded state across instances; admin auth remains checked first. |
| P2 | Generate tool-ID/display manifests from `content/tools/` | Catalog, API, account UI | Medium: duplicated catalogs can drift and break visibility/session labels. | M | Build fails on unknown/duplicate IDs and generates API/UI manifests plus explicit aliases from one source. |
| P2 | Workerize or more tightly bound large text-analysis paths | NBSP, Oxford, POV, Word Frequency | Medium: valid near-limit input can cause main-thread long tasks or memory pressure. | M-L | Low-end benchmark meets responsiveness target; cancellation and deterministic limits have automated tests. |
| P2 | Add query-first Short Links list architecture | Short Links | Medium at catalog scale: filtered/sorted lists still require scans. | L | GSI/materialized index supports the required sort/filter combinations with stable cursors and backfill validation. |
| P2 | Expand browser/API end-to-end automation | All tools | Medium: static contracts catch wiring drift but not permission prompts, rendering, service failures, or cleanup. | L | CI covers public primary/limit flows plus disposable cloud canaries in a protected environment. |
| P2 | Add privacy-safe operational scorecard and alerts | All tools | Medium: regressions can remain anecdotal without run/error/latency and cleanup-backlog visibility. | M | Dashboard and alerts expose only bounded categories, with no payload/filename/URL-query/transcript/application content. |
| P3 | Add streaming/worker acceleration where measurements justify it | Image Optimizer, Screen Recorder, UTM | Low: current bounded paths are safe but may not maximize desktop throughput. | L | Feature-detected fast paths outperform baseline without weakening limits or compatibility. |

## Definition of done

The current hardening effort is complete when:

1. All P0 items are complete and recorded against one release commit.
2. The 14-tool matrix has no unidentified execution, storage, cost, or retention boundary.
3. Public tools pass their primary and limit-state browser flows without uncaught errors at desktop and mobile sizes.
4. Shared session concurrency, pagination, quota, TTL, stale-link, and destructive-delete paths pass against the deployed store.
5. Short Links reservations/TTL, GA4 allowlist/deadlines, Job Tracker POST/cleanup/export caps, and Transcriber ledger/idempotency/resume/cleanup paths pass live canaries.
6. The credential-standardization owner confirms the final auth/env contract, and no secrets or conflicting credential changes are introduced by this work.
7. Every deferred P1/P2/P3 item has a named owner or is explicitly accepted as residual risk.
