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
