# Android adaptive layout and update validation — 2026-09-20

Local review build: **0.4.0-debug**, version code **5**, package `me.danielshort.app.debug`. These changes and the staged APK are not published. The production manifest remains unchanged.

## Checks

- Android build and instrumentation APK build passed. JVM tests: **161 passed**. Lint completed with no errors; existing advisory/style/resource warnings remain.
- Node update protocol/builder tests: **12 passed**.
- All **48 standard device checks** passed across the full run and focused reruns. The initial run exposed an exact-840dp rounding defect and a test that tried to click through the physical keyboard after synthetic resizing. The breakpoint now uses rounded physical pixels; the navigation test closes the keyboard before changing synthetic density. The final two app-flow tests and all four layout-flow tests pass without weakening installation-safety assertions.
- Three explicitly enabled real-APK fixture checks were exercised separately: installed/archive verification and patch reconstruction, permission refusal plus installer cancellation, and the real self-update session. Normal CI skips fixtures requiring local APKs or an actual installation.
- `git diff --check` passed.

Device: the dedicated `DanielShort_QA` emulator, Android 16 / API 36.1. UI checks used a 411dp phone, a 320dp narrow window, a 1280dp landscape tablet, and a 1024dp portrait tablet with 130% font size. Window resizing recreated the real activity and preserved entered Text Compare text in both directions. Synthetic device tests additionally cover the exact 840dp boundary, rail selection/geometry, content composition, search/bookmarks, scroll state, and reduced-motion behavior.

Visual checks confirmed five site-colored vertical tabs on wide windows, two-column tool cards where space permits, constrained reading/settings widths, side-by-side Text Compare editors, readable narrow settings, and appropriate system-bar clearance. The existing phone navigation and scroll behavior remain covered by the device suite. Independent reviews rated the layout **8.7/10**, update behavior **8.8/10**, and Settings **8.7/10** after fixes. This is not a complete screen-reader or physical-device certification.

## Real installation evidence

A temporary, unpublished version-6 APK was built solely to test a strictly increasing update from a verified local version-5 APK. The production package verifier and patch decoder reconstructed the target from a 447,953-byte patch. Android received a real `PackageInstaller` session with `USER_ACTION_NOT_REQUIRED`.

Google Play Protect requested a security scan and final confirmation for this previously unseen APK. Those prompts were completed normally. Thus this device proves the signed patch/install flow and Android confirmation behavior, not a fully unattended installation. Other devices may also require platform interaction.

External checks after installation confirmed version code 6, target SHA-256 `48f3823ed7313b7f6319e70025fb5fafe5745212a56d04660833755feffdd405`, a durable `INSTALLED` callback record, a preserved `smartSentence` bookmark, and a normal app launch. Instrumentation completion alone was not used as proof of installation. The device and source were then restored to version 5; the temporary version 6 and its manifest were not published.

Unit/device coverage also checks persistent manual-install suppression, forged callbacks, restart recovery, active recording protection after leaving the recorder, foreground/workspace guards, opt-out, network restrictions, and manual retry. Automatic installation waits at least 30 seconds after backgrounding and rechecks eligibility while staging and before commit.

## Review artifact

- APK: `app/build/outputs/app-update/review-v5/Daniel-Short-review-v5-67eb0fa49d23e932.apk`
- Size: 24,747,972 bytes.
- SHA-256: `67eb0fa49d23e932ece0bbc9fbbc41a06a8206dcbe161773315f0e144d089409`.
- Signature verified against the preserved review signing identity.
- A patch from the published version-4 APK and an inventory preserving earlier approved hashes are staged alongside it. Publication still requires uploading the immutable assets and deliberately updating the channel manifest as described in `UPDATES.md`.

The local APK is intentionally unrecognized by the unchanged public release inventory. Its startup check reports that condition in Settings until the reviewed release is published; routine startup failures do not interrupt the current screen.

## Combined release candidate — version 6

The release candidate was rebuilt from the original workspace after combining the Android changes with the pending website content. This **0.4.0-debug**, version-code **6**, APK supersedes the version-5 staging artifact above. It is distinct from the temporary version-6 installation fixture used in the earlier device checks.

- `testDebugUnitTest`, `lintDebug`, `assembleDebug`, and `assembleDebugAndroidTest` passed. All **161 JVM tests** passed with no skips or failures. Lint reported **0 errors**, 33 warnings, and 10 hints.
- All **16 Node protocol and release-feed tests** passed.
- APK: `app/build/outputs/app-update/review-v6/Daniel-Short-review-v6-21283723b7cc6aa7.apk` — **24,747,980 bytes**.
- APK SHA-256: `21283723b7cc6aa74200f189926e227eaacdb7389ac7bc5c03e4b5e700a2fbe5`.
- Android metadata confirms package `me.danielshort.app.debug`, version `0.4.0-debug` / `6`, minimum SDK 26, and target SDK 36.
- `apksigner verify --verbose --print-certs` passed with one APK v2 signer. Certificate SHA-256 remains `88cb3cf9ae50dfd77b6d896742d0ec30a30a09a60b3fc99ee8f38bd96751a44e`, matching the preserved review identity.
- The **743,242-byte** patch from published version 4 (`6b250314…`) and **506,397-byte** patch from local version 5 (`67eb0fa4…`) independently reconstructed the complete version-6 APK byte for byte. All staged `SHA256SUMS.txt` entries matched.
- The staged manifest preserves all three previously published APK identities, adds the verified local version-5 base and this version-6 target, and names two patches: **five approved hashes total**.
- The APK's `assets/catalog.json` exactly matches both `dist/app-content/v1/catalog.json` and `public/app-content/v1/catalog.json`: SHA-256 `31fa9ce3da3ee73a490287d4027b996ae5600f7c52af72786313250314296d1b`, revision `d7800351a48eac4504064c686e327e762fbaf28ab0855f1ac950dbe573274ddc`.

These are local build and artifact checks. No version-6 device installation or public runtime verification was performed during this rebuild. At staging, the designated review feed still selected published version 4; uploading release assets and changing that feed remain separate release steps. Build and staging logs are in the workstation temporary directory as `android-review-v6-gradle.log` and `android-review-v6-staging.log`.
