# Vercel Hobby downgrade audit and completed migration

**Completed September 6, 2026 MDT / September 7 UTC:** the team is on **Hobby**, a fresh production deployment built successfully on that plan with **11 functions**, and Fluid Compute is enabled in both the repository and project defaults. The final verification record is at the end of this document.

The initial investigation below records the pre-migration account and repository state. Implementation, deployment, downgrade, and testing were subsequently authorized by the owner.

## Initial finding

The relevant resource is **Vercel Functions**, previously called Serverless Functions. Hobby permits **12 functions per deployment** for this site's framework-free `api/` setup. Pro removes that count limit. One deployed function can dispatch many URLs; JavaScript helper functions and imported modules do not each consume a slot. [Vercel runtime documentation](https://vercel.com/docs/functions/runtimes#functions-created-per-deployment)

The connected Vercel account currently has one listed team, **Daniel Short's projects**, on Pro, and one listed project, **website**. The latest production-target deployment reports **14 Node.js functions** in `meta.lambdaRuntimeStats`.

- Project: `prj_N9VRrsglSeevvFNnCD2VPEoZEMBU`.
- Inspected deployment: `dpl_HwmCyVxx9VZy2npCpTzpdSyhSjHf`, READY.
- Deployment metadata commit: `638da70cabbe3e65a5b0e267c9ee5eee4a5d4287`, with `gitDirty: 1`.
- [Deployment details](https://vercel.com/daniel-shorts-projects/website/HwmCyVxx9VZy2npCpTzpdSyhSjHf).

Repository history independently identifies this same issue: January 19, 2026 commit `0f823310dd7ca4599ba62fe47405203d8b09d95e`, titled "Consolidated functions", combined tools endpoints. The current `api/tools/[...slug].js` still explains that it was consolidated to avoid Hobby limits. The exact original upgrade event was not located, so this establishes the resource and current blocker, rather than the precise billing history.

## Initial function inventory

Both the current working tree and local HEAD contain 18 non-helper API entry files. Four are excluded by `.vercelignore`, leaving the same 14 reported by Vercel. Local HEAD and the inspected deployment differ, and the working tree has substantial preexisting changes; this is not a byte-for-byte production source audit.

| Function group | Count | Entrypoints |
| --- | ---: | --- |
| Short-link management | 5 | `api/short-links/index.js`, `[...slug].js`, `health.js`, `clicks/[...slug].js`, `sets/[...setId].js` |
| Public short-link redirects | 1 | `api/go/[...slug].js` |
| Tools accounts and transcription | 1 | `api/tools/[...slug].js` |
| Shared demo proxy | 1 | `api/demos/[...slug].js` |
| Legacy sentence demo proxy | 1 | `api/sentence-demo/[...slug].js` |
| Main chatbot and its logs | 1 | `api/chatbot.js` |
| Streaming travel chatbot | 1 | `api/chatbot-stream.js` |
| Contact form | 1 | `api/contact.js` |
| GA4 reporting | 1 | `api/ga4/report.js` |
| Legacy portfolio redirects | 1 | `api/portfolio-redirect.js` |
| **Total** | **14** | |

The excluded entrypoints are under `api/cms/`, `api/chatbot/logs.js`, `api/short-domain.js`, and `api/short-links/test/`. Modules under `api/_lib/` are shared implementation code. Merely moving code into a helper while retaining the old deployable entrypoint would not save a slot.

## Recommended consolidation: 14 to 11

Extend the existing short-link catchall router. It already delegates `test/...` to a helper, so this follows the current architecture.

1. Move the implementations of health, clicks, and sets into `api/_lib/short-links-endpoints/`.
2. Dispatch their existing URLs from `api/short-links/[...slug].js`.
3. Remove the three former deployable entry files after updating imports and routing.
4. Keep `api/short-links/index.js` for root list/create/analytics requests.

| Existing public API | Handler after consolidation |
| --- | --- |
| `/api/short-links` | Existing root entrypoint |
| `/api/short-links/health` | Existing catchall -> health helper |
| `/api/short-links/clicks/<slug>` | Existing catchall -> clicks helper |
| `/api/short-links/sets` | Existing catchall -> sets collection helper |
| `/api/short-links/sets/<id>` | Existing catchall -> sets item helper |
| `/api/short-links/sets/<id>/generate` | Existing catchall -> sets generation helper |
| `/api/short-links/test/<slug>` | Existing catchall -> existing test helper |
| `/api/short-links/<slug>` | Existing catchall -> individual-link handler |

This turns five short-link management functions into two and yields **11 total**, with one spare slot. It is intended to preserve every feature and URL. The public redirect function remains separate, preserving its public GET/HEAD, query passthrough, click tracking, and custom-host behavior.

For more headroom, consolidate the root management handler too and explicitly rewrite `/api/short-links` into the shared router. This yields **10 functions**, with two spare slots, at the cost of one additional routing change. A further optional reduction can replace the legacy sentence-demo function with a rewrite to the existing demos router; both currently call the same demo helper. Neither a framework migration nor moving additional services to AWS is necessary to solve the count issue.

Implementation must also update `build/dev.js`, which maps routes to the current handler filenames, relevant imports in `test.js`, and focused short-link tests. Preserve encoded multi-part slugs, sets collection handling, POST/PATCH/DELETE bodies, and query parameters. Trusted path-derived routing values must take precedence over hostile `?slug=` or `?setId=` values. Dispatch before consuming a request body and retain each handler's authorization behavior.

## Execution time and Fluid Compute

The live [Functions settings](https://vercel.com/daniel-shorts-projects/website/settings/functions) show **Fluid Compute disabled**, Basic CPU with 1 GB memory, and one selected region, `iad1`.

The current `vercel.json` sets demos and sentence-demo to **150 seconds** and chatbot, chatbot-stream, and tools to **60 seconds**. Some demo upstream waits can reach 120 seconds, so simply shortening all functions could cause failures.

Hobby with Fluid Compute has a documented **300-second default and maximum**, supporting the existing 150-second settings. Enabling Fluid requires a new deployment. Current live documentation emphasizes Fluid; older indexed documentation still shows legacy Hobby's 10-second default / 60-second maximum. Because this project is on the legacy execution model, do not assume the current Fluid allowance applies to it unchanged. The recommended path is to enable and validate Fluid before downgrading. [Fluid Compute](https://vercel.com/docs/fluid-compute), [Function duration configuration](https://vercel.com/docs/functions/configuring-functions/duration)

A bounded static review found no request-time `process.env` assignments or module-global current-user/request/response objects in the inspected active handlers. Shared state is primarily SDK clients and caches. This is encouraging, but does not prove concurrent request safety. Fluid lets requests share a process, so verify simultaneous requests from separate users, streaming cancellation, and backend failures. [Global state under Fluid](https://vercel.com/docs/fluid-compute#isolation-boundaries-and-global-state)

One existing edge case merits a focused check: `api/_lib/chatbot-stream-proxy.js` waits for `drain` after response backpressure. A slow client disconnecting before that event may leave the wait unsettled. Test this alongside concurrent transcription budget updates, short-link conflicts, and warm AWS/OIDC credential refresh. It is a candidate cleanup issue, not a demonstrated Fluid failure.

## Live usage and billing

The [Usage dashboard](https://vercel.com/daniel-shorts-projects/~/usage) was filtered to the current billing cycle, **August 11 to September 11, 2026**, across all projects. These are partial-cycle readings and dashboard data may lag by one hour.

| Resource | Current cycle so far | Hobby allowance with Fluid |
| --- | ---: | ---: |
| Function invocations | 1.32K | 1,000,000 |
| Edge requests | 46.55K | 1,000,000 |
| Fast data transfer | 1 GB | 100 GB |
| Fast origin transfer | 9 MB | 10 GB |

The dashboard also reports 0.3 GB-hours of **legacy** function duration and 8 hours of build CPU time. Neither is a measurement of Fluid active CPU usage. After migration, check the separate 4 CPU-hour and 360 GB-hour Fluid allowances. Current traffic looks compatible; this snapshot is not a guarantee about future traffic. Combining function entrypoints does not inherently reduce invocation or compute usage. [Hobby allowances](https://vercel.com/docs/plans/hobby), [General limits](https://vercel.com/docs/limits)

The billing screen shows one $20 Pro subscription, approximately $1.84 of infrastructure usage fully covered by included credit, and an upcoming invoice of $20. **Observability Plus is enabled**; account for the loss of its enhanced monitoring in the downgrade. No drains were listed. Project Storage showed no connected databases. AWS services and domain registration bills are separate from the Vercel subscription.

## Account and deployment checks

Hobby is for personal, noncommercial use. A personal portfolio and free demos are generally compatible; monetization, ads, or advertising paid products/services need a separate eligibility assessment. [Fair-use policy](https://vercel.com/docs/limits/fair-use-guidelines#commercial-usage)

The project is linked to the personal GitHub namespace `danielshort3/danielshort3.github.io`. Hobby supports personal repositories; private organization repositories and collaboration have additional restrictions. Git deployments must come from the owner's linked Git identity. [Hobby Git deployments](https://vercel.com/docs/git#using-hobby-teams)

Hobby's **100 MB CLI source-upload limit** also deserves a deployment check because this repository has many assets and recent releases used the CLI. This is an uploaded-source limit, not a 100 MB cap on build output. Source upload size was not measured in this audit. Use the linked Git deployment path or prune unnecessary upload sources if a CLI release exceeds it; build-generated assets should remain in the build pipeline. [Upload limits](https://vercel.com/docs/limits#static-file-uploads)

Hobby builds use 2 vCPUs and 8 GB memory with one concurrent deployment. The current pricing page describes Basic build minutes as included; the older 6,000-minute monthly figure was not confirmed in the live documentation. The observed 8 build CPU-hours must not be compared against the separate 4-hour Function runtime quota. [Build machines](https://vercel.com/docs/builds/managing-builds#build-machines), [Current pricing](https://vercel.com/pricing)

Both **danielshort.me** and **dshort.me** are listed on the team. Vercel's current downgrade documentation says connected Stores and/or Domains must be transferred before downgrading, permits one Hobby team per account, and removes members other than the original owner. Resolve the actual account's domain-transfer requirement while preserving website routing before confirming the downgrade. The project storage check found no connected stores. [Official downgrade procedure](https://vercel.com/docs/plans/pro-plan#downgrading-to-hobby)

The downgrade button currently opens a feedback survey. It was inspected and left without submitting feedback or confirming a plan change. The final account-specific effective date, credits, and transfer blockers were therefore not observed; do not assume the September 11 renewal date is the effective downgrade date.

## Implementation and downgrade sequence

1. Select a coherent implementation state that includes the existing uncommitted short-link work, then implement the 11-function consolidation above.
2. Run focused routing/auth/body tests, `npm.cmd run build`, `npm.cmd test`, and `git diff --check`. Validate rendered short-link management and public redirects.
3. Validate a deployment with Fluid enabled. Confirm the actual Vercel build produces at most 12 functions, uses Hobby-compatible duration/memory/region settings, and preserves chat streaming, tools login/session storage, transcription, demos, contact, and redirects. Verify browser behavior against the built output as well.
4. Deploy and verify the compatible version while Pro is still active. Retain a compatible rollback deployment; the existing 14-function version is not a suitable long-term Hobby rollback target.
5. Resolve the domain-transfer requirement and review the enabled monitoring add-on. Complete the account-specific downgrade through Team Settings -> Billing -> Downgrade, reviewing the final effective-date and billing information.
6. Verify custom-domain routing and important features after the plan change, and confirm a subsequent deployment succeeds under Hobby. Check usage in the new compute model.

No application changes were implemented, no deployments or tests were run, and no billing or domain changes were made during this investigation.

## Implementation record (September 6-7, 2026)

The owner subsequently authorized implementation, deployment, downgrading the team, and live verification.

- Consolidated short-link health, click history, and set handlers behind `api/short-links/[...slug].js`; moved the implementations under `api/_lib/short-links-endpoints/`. The root management and public redirect functions remain separate. The deployment source now has **11 function entrypoints**.
- Updated Vercel rewrites, local development routes, and test imports. URL path values take precedence over conflicting query parameters; request bodies and authorization remain owned by the individual handlers.
- Enabled Fluid Compute in `vercel.json`, preserving the existing 60- and 150-second function settings.
- Fixed a reproduced chatbot streaming backpressure/cancellation hang and added concurrent request, disconnect, timeout, and listener cleanup coverage.
- Corrected `.vercelignore` for Vercel CLI archive behavior: directory exclusions omit trailing slashes, and the excluded `api/chatbot/logs.js` directory is excluded as a whole while retaining `api/chatbot.js`. CLI 59.11.7 can otherwise reintroduce an excluded directory into a tar archive. The first archive attempt was stopped before any upload after this was detected.
- The corrected dry deployment manifest contains 3,959 entries / 850,893,408 source bytes and 11 API entrypoints. Excluded local folders and environments are absent. Six remaining directory entries were inspected and are physically empty. Archive chunks support this asset-heavy site's CLI release path.

Local validation passed with Node 22: `npm.cmd run build`, the full `npm.cmd test` suite, focused short-link and streaming tests, and `git diff --check`. The rebuilt short-link/QR interface rendered and loaded its existing links. The live pre-migration stream baseline completed with HTTP 200 and metadata, token, and done events in 3.65 seconds. The old production set-collection route returned 400; the consolidated router explicitly handles that collection and is expected to fix it.

## Final deployment, billing, and live verification

| Check | Verified result |
| --- | --- |
| Team plan | Billing UI shows **Hobby Plan**; the Vercel teams API independently returns `plan: hobby`. |
| Pro cancellation | The final confirmation specified immediate loss of Pro features and stopping Pro payments. Vercel calculated a **$2.58 refund** for unused time; bank settlement was not checked. |
| Final deployment | [`dpl_7vodw2uG4QbU1nrjPz2C7Zwon1ZY`](https://vercel.com/daniel-shorts-projects/website/7vodw2uG4QbU1nrjPz2C7Zwon1ZY), **READY**, created and built on **Hobby**. |
| Runtime | Deployment API confirms `functionType: fluid`, active CPU accounting, a 300-second default, and `lambdaRuntimeStats: {"nodejs":11}`. The repository retains its explicit 60- and 150-second overrides. |
| Build | Successful remote Linux build on Hobby's two-core / 8 GB build machine. No function-count or duration rejection in the final deployment. |
| Canonical alias | Alias API confirms `www.danielshort.me` points to the final deployment above. |
| Domains | `www.danielshort.me` returns 200; `danielshort.me`, `dshort.me`, and `www.dshort.me` return their expected canonical 308 redirects. Both short-link domains also passed nested-link HEAD redirect tests. No domain transfer was required by the actual downgrade workflow. |
| Public/API smoke checks | All **20** checks passed after the final Hobby deployment, including page routes, tools authentication guards, transcription configuration, and short-link authorization/read endpoints. |
| Short-link integration | All **25** final checks passed: nested-link and set create/read/update/delete, path/query isolation, public GET/HEAD redirects, click history, method guards, and generation validation. |
| Streaming | Final canonical request returned HTTP 200 with one metadata event, 29 token events, and one done event across nine chunks, completing in **3.09 seconds**. |
| Browser | The new interface rendered; Text Compare loaded its example, selected Comparison, and displayed 33 inserted words, 28 deleted words, three replacements, and one moved block. |
| Runtime errors | No 5xx logs found for the final deployment from its completion at 01:13:40 UTC through the 01:16:34 UTC log check. |

Two issues were found and resolved during release verification:

1. **Production short-link creation permission:** candidate reads and set writes worked, but link creation returned 502. The exact helper succeeded locally. AWS policy simulation confirmed that the existing production role denied `dynamodb:ConditionCheckItem`, used by the transactional click-baseline guard. A reviewed CloudFormation change set modified only `ShortLinksRole`, with no resource replacement and all 16 existing parameter values preserved. The stack reached `UPDATE_COMPLETE`. A separate policy statement now allows that operation only on the existing click-history table; simulation confirms it remains denied on the main links table. The repository template, README, and a passing infrastructure regression test preserve that scope. Full candidate and final live link operations then passed.
2. **Project runtime default:** the first Hobby redeploy (`dpl_48oawSEvRpqZGnSokg4EUmWcN8W6`) used the project's legacy runtime default and correctly rejected a 150-second function against its 60-second ceiling. It did not replace the working production deployment. Fluid Compute was then enabled and saved in Project Settings -> Functions. The project API confirms `defaultResourceConfig.fluid: true`, 300 seconds, and `resourceConfig.fluid: true`. A second fresh Hobby rebuild succeeded, proving that both uploaded deployments and redeployments now work.

The earlier verified 11-function candidate, `dpl_2Bhzhynr7xeVbBni1mnt7NL2f41m`, remains available as a compatible source/rollback reference. Future CLI source releases can use the corrected archive exclusions with `vercel deploy --archive=tgz --prod`; the authenticated WSL CLI path was used here. `vercel redeploy <deployment-id> --target=production` was also verified to create a fresh build from stored source under Hobby.

Temporary QA links and sets were deleted and checked for 404 after each successful run. Two intentional synthetic QR clicks and their baselines remain in analytics across candidate and final testing. Temporary credential files were removed, and the task's local verification server was stopped. Existing unrelated workspace changes were preserved; there was no Git commit or push.

Verification scope: this tested the changed backend routes, representative pages/browser behavior, runtime configuration, domains, and complete chatbot streaming. It did not run a paid transcription job, send a contact email, exercise every GPU-backed demo, or create a fresh Cognito account. Unit/integration coverage for the shared account and other existing features passed in the full local suite. Successful set generation was not invoked because its internal batch records lack an HTTP delete endpoint; its validation/method routing and mocked generation coverage passed.

The existing service worker briefly paired updated HTML with an older cached Text Compare script in a tab open across the release. The computed result was present but hidden by the old tab behavior; a reload refreshed the script and the complete browser flow passed. Refresh a tab once if it was open during this deployment. AWS services and domain registration remain separate from the canceled Vercel Pro subscription.
