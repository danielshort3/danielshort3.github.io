# Android validation — 2026-09-19

Version: **0.2.0-debug**, version code 2. Application ID: `me.danielshort.app.debug`. Validation used the dedicated `DanielShort_QA` Android 16 / API 36.1 emulator at 1080 × 2400 and 420 dpi. The final APK uses the production HTTPS content feed, not a local development endpoint. Minimum API 26 compatibility is lint-checked, not verified on an older physical device.

## Automated checks

- **79 unit tests passed**, covering catalog validation, text processing, comparison, roulette, game simulation/checkpoints, model/data contracts, and recording file recovery.
- **25 device tests passed in the final combined run**, covering native text workflows, QR/image UI, historical dashboard filters, Settings, game controls/checkpoints/reset, and the offline pizza estimator. The game tests caught and verified the fix for a Probability Engine cooldown that failed to resume after its first spin.
- The focused website feed test passed: 16 projects, 10 public tools, 6 games, public-field filtering, safe links, stable revisions, and image versioning.
- Website build/CI results from the earlier feed release are recorded separately below; they are not a claim that all currently pending website changes were deployed.

- Final `assembleDebug testDebugUnitTest connectedDebugAndroidTest lintDebug`: **BUILD SUCCESSFUL**. Lint: **0 errors, 29 warnings, 10 hints** (advisory dependency, compatibility, style, and resource findings).
- APK Signature Scheme v2 verification passed. The final APK installed successfully, launched, and opened Settings; the Android crash buffer remained empty.
- `git diff --check`: passed. Current Android source remains local; no remote CI or app-store publication is claimed.

## Exercised in the installed app

- Native catalog navigation, project detail → Open demo, and the header Settings button. The light DS branding, section colors, and system-bar insets remain consistent.
- Settings display automatic updates, unmetered-only updates, reduced motion, manual refresh, image-cache clearing, bookmarks, and the app version. Device tests check persistent preferences and destructive-action confirmation.
- Screen Recorder: system consent cancellation; capture of this app through Android's single-app selection; actual 35-second MP4 recording (1,590,063 bytes); native playback; export through the document picker; recovery after forced process restart; and a valid Android share sheet. The share sheet was dismissed without sending the clip to anyone.
- Background Remover: selected the repository's portrait through Android's photo picker, ran real ML Kit subject segmentation, and exported a 384 × 384 RGBA PNG. File inspection found alpha values spanning 0–255 and 68,139 fully transparent pixels. This was actual segmentation, not a substituted sample.
- A preview-spacing issue found during that check was fixed with bounded sizing and `ContentScale.Fit`; the device regression test verifies the height limit and separation from the caption.
- Stormbreak: combat, Bolt ability, rewards, pause, and saved progress. The inspected checkpoint recorded five kills and 74 gold. Instrumentation separately verifies reset and pause behavior.
- Synthetic Digit Generator: reached the existing AWS service, displayed Connected, and rendered all 36 real generated images. Controls, status, grid, and advanced settings were inspected at normal and 130% font size; font scale was restored afterward.
- No AndroidRuntime crash was found during these manual flows. HTTP error states remain visible to the user instead of inventing output.

## Scope and limitations

All ten public tools, six games, and thirteen project demos have native routes. Larger games intentionally provide core gameplay, not full website parity. The two Tableau dashboards use native filters/charts over reproducible historical snapshots. See `NATIVE_FEATURES.md` for the exact coverage and data boundaries.

Independent source reviews rated utilities **8.5/10**, recorder/settings **8.5/10**, demos **8.2/10**, and core games **8/10** after fixes. These ratings supplement, rather than replace, device and automated evidence.

Background segmentation initially failed under this emulator's SwiftShader OpenGL ES 3.0 renderer. Switching the dedicated QA emulator to host GPU rendering (OpenGL ES 3.1) resolved it and produced the verified transparent PNG. The Android app reports segmentation failures; compatibility across physical devices and Google Play services versions has not been established.

Cloud inference needs connectivity and can require a cold-start retry. The current device check exercised Digit Generator. Other service contract/live HTTP checks do not establish that every model flow was exercised end-to-end on this final APK. Smart Sentence cold-start evidence is detailed below.

Native bookmarks/game checkpoints are local; website account sign-in and cloud session synchronization are not implemented. Screen recording supports optional microphone audio, not device playback audio. Completed private clips survive process restarts; an unfinished recording may be lost if the app is forcibly terminated. Six-hour background work is best effort and was not observed across a full six-hour cycle. Minimum-API hardware, physical phones, Play Store distribution, and production signing are not validated.

Content-feed refresh, invalid-feed retention, offline catalog restart, bookmarks, and native share/navigation were verified during the initial app implementation on 2026-09-14. This update adds the native feature coverage above. The debug APK is for local installation/review; current Android changes have not been committed, published to an app store, or deployed as website code.

## Review artifact

- File: `app/build/outputs/apk/debug/app-debug.apk`
- Version: `0.2.0-debug` (2)
- Size: 24869407 bytes
- SHA-256: `44c7c7a6d1c823033ddb163a32b4799a367f9b594dd0f07490a18e13999997aa`
- Signing: local Android debug key; signature verified. Install with Android's normal package installer or `adb install -r`.
- Feed: `https://www.danielshort.me/app-content/v1/catalog.json`

## Previously verified production feed release — 2026-09-14

Website PR #209 merged as `bdb3dbd74b4e899b0b43d67785b1c8d5cad70020`; Vercel production deployment is READY. The isolated feed release passed the full website CI suite, including rendered desktop/mobile checks, before merge. Pending local website design and native app source changes were kept separate from this release.

The public `https://www.danielshort.me/app-content/v1/catalog.json` returns HTTP 200 and schema version 1, with content revision `1fdca3d1693c49cc9b8dad2f26ccb597440e9f28792250c1fb71a1bc3e0c301b` matching the validated release build. All 52 image URLs returned HTTP 200 with image content types. Cache revalidation was verified with an ETag request returning HTTP 304. The homepage, Text Compare page, and website project page remained byte-identical across deployment. Vercel reported no runtime errors in the post-deployment check.

The unchanged review APK then refreshed from production on `emulator-5554`. Its cached revision matched the live feed exactly; native About and All 16 Projects rendered correctly. A second UI refresh showed "Content is up to date", retained the ETag and cached file, and advanced the last-checked timestamp. No AndroidRuntime errors or crashes were reported.

## Smart Sentence backend check — 2026-09-19

Read-only production checks at approximately 14:15–14:18 UTC distinguish cold backend preparation from an Android request problem:

- `GET /api/demos/smart-sentence/health` returned HTTP 200 in 7.49 seconds, reporting `Snowflake/snowflake-arctic-embed-l-v2.0`, 847 sentences, and 1024 dimensions.
- Immediately afterward, `POST /api/demos/smart-sentence/rank` with `{"query":"She wonders about things.","top":3}` returned HTTP 504 in 60.36 seconds: `Demo service timed out.` The Android payload matches the website exactly; requests without an Origin header are supported. This demo has no separate warmup endpoint.
- The proxy caps ranking requests at 60 seconds. The active production Lambda alias points to version 12 with a 120-second timeout and 3008 MB memory. CloudWatch recorded an earlier 120-second timeout, then another invocation completing after 103.55 seconds, beyond the proxy's response deadline.
- Once that preparation finished, the identical rank request returned HTTP 200 in 1.78 seconds with three real ranked sentences and similarity scores. Thus the endpoint works when warm, but a successful health check does not guarantee that the first ranking request will finish in time.

No AWS configuration, infrastructure, or deployment was changed. No native payload workaround is required. The native interface now identifies a Smart Sentence HTTP 504 specifically and displays "AWS is still preparing. Wait about a minute, then tap Search again." Other errors keep their existing descriptions; no automatic repeated requests were added. Initial use after idle periods can still require waiting and retrying. Fixing the cold-start latency or backend/proxy timeout mismatch is separate from the native port. These HTTP and CloudWatch checks do not substitute for exercising the flow in the Android UI.
