# Native app updates

Settings opens a compact overview with **Updates**, **Reduce motion**, **Storage**, and **App information**. Open **Settings → Updates** for app-update controls and the separate website-content refresh choice. Update notices open this screen directly, including on repeat visits. See [Settings design and validation](SETTINGS.md) for the screen structure and preference mappings.

**Update mode** offers **Check automatically**, **Automatic**, and **Manual**. The existing default is **Check automatically**: check the selected channel's public manifest once per cold app launch, without automatically downloading an APK. **Check now** remains available manually. Choosing **Automatic** enables verified downloads; **Automatic downloads** then selects **Unmetered connections** (the existing default) or **Any connection**. Android determines whether a connection is metered; this is not a literal Wi-Fi-only rule. These choices do not change website-content refresh preferences, and hidden network preferences are retained.

Enabling automatic app updates also enables launch checks. An offline launch waits for a connection while the app is in the foreground; rotation, permission screens, and installer returns do not start additional checks. Download failures and cancellations do not repeatedly retry the same release automatically within the current process. Manual checks and downloads remain available independently of these preferences.

The app verifies that its installed APK exactly matches a recognized release, then downloads a smaller binary patch when one is available. Otherwise it downloads the complete APK. Both paths verify the resulting APK before handing it to Android. Automatic installation requires explicit opt-in and installation permission, and waits until the app is out of use with no active native workspace. Older Android versions and devices that require user action retain the manual installation flow. No downloaded Kotlin, JavaScript, or native library is loaded independently of Android's package installer.

An automatic installation can begin only after at least 30 seconds in the background, on a browse-only screen, with no active workspace or recording service. Eligibility is checked again while preparing the Android installation session. Android confirmation prompts are opened only by an explicit **Install update** action. A manual installation handoff or a request for confirmation keeps that release on the manual path after a process restart, including when the user cancels.

Android 12 and later support automatic self-updates through `PackageInstaller` when the platform's target-SDK, installer, and permission requirements are met. The app must still handle `STATUS_PENDING_USER_ACTION`; opting into automatic updates cannot override Android's decision. See the [Android installer requirements](https://developer.android.com/reference/android/content/pm/PackageInstaller.SessionParams#setRequireUserAction(int)).

The first version containing this updater must be installed manually once. Older version 0.2.0 does not contain an updater: preparing a patch from that version is useful for verification but cannot add the updater to an already installed copy by itself. Thereafter, new native versions require published, signed APK artifacts and an updated manifest. Website CSS and JavaScript are not native app patches.

## Channels and trust

| Channel | Android package | Manifest |
| --- | --- | --- |
| Review | `me.danielshort.app.debug` | `https://www.danielshort.me/app-updates/review/latest.json` |
| Stable | `me.danielshort.app` | `https://www.danielshort.me/app-updates/stable/latest.json` |

Review builds currently use the existing workstation development signing key. Preserve that identity outside source control for review updates. CI's automatically generated debug key is not interchangeable with that key. A stable distribution needs a securely maintained production signing identity; it is a separate package and cannot replace the review app in place. Do not commit signing keys or passwords.

The manifest names exact APK hashes, byte sizes, package, version codes, signer identity, and any available patches. A version code alone is insufficient: several reviewed local APKs may share an old version code while containing different bytes. Recognized historical APK hashes remain in the release inventory even when no patch is retained for them, enabling a verified full download. Target version codes must always increase.

An unknown installed hash is reported as an unrecognized build and cannot be patched. It may be a legitimate local development build; an unknown hash does not prove malicious modification. Approved local builds can be included explicitly as bases after checking their provenance. The checks establish correspondence to an approved APK, not proof that a rooted device or runtime is uncompromised. Split APK installations are not supported by this sideloaded whole-APK updater.

Downloads use HTTPS. Artifact URLs must belong to `danielshort.me/app-updates/`, `www.danielshort.me/app-updates/`, or this repository's GitHub release downloads. The Android network implementation permits the specific GitHub download redirects needed to retrieve release assets. Files are stored privately; only a verified ready-to-install APK is shared with the installer. Installation preserves app data through Android's normal update process.

## Prepare a release locally

Build and test the target APK first. Set a strictly greater `versionCode` and the intended version name in `app/build.gradle.kts`. Use the same signing identity as prior versions of that channel. Preserve previous signed APKs; recreating their source does not guarantee identical bytes.

The builder uses Node's standard library and Android SDK Build Tools **36.1.0**. Set `ANDROID_HOME` and `JAVA_HOME` as described in the [Android build guide](README.md). Run from the website repository root. This example stages the settings redesign as version 0.5.0 (code 7) using an archived copy of the published version-6 APK:

```powershell
.\mobile\android\gradlew.bat -p mobile/android --no-daemon testDebugUnitTest lintDebug assembleDebug
node mobile/android/scripts/prepare-app-update.cjs `
  --apk mobile/android/app/build/outputs/apk/debug/app-debug.apk `
  --base C:/release-archive/Daniel-Short-review-v6-21283723b7cc6aa7.apk `
  --previous-manifest mobile/android/releases/review/latest.json `
  --output C:/release-staging/android-v0.5.0-review `
  --base-url https://github.com/danielshort3/danielshort3.github.io/releases/download/android-v0.5.0-review/ `
  --channel review
```

Run that build only in the environment retaining the review signing key. The example archive path must contain the actual published APK, not a new build from old source. Repeat `--base` for every available base APK that should receive a patch. Supply `--previous-manifest` to retain previously approved release hashes. The archive is trusted release input: preserve it with the same care as published APKs. Use a fresh output directory for each release. Stop on any signer, package, version, or integrity mismatch rather than changing the manifest to bypass it.

The builder:

1. Runs `apksigner verify` and `aapt dump badging` for the target and all supplied bases. It rejects mismatched package/signing identities, split APKs, non-increasing versions, oversized APKs, and files changed during verification.
2. Builds patches using rolling Adler-32 checksums and exact SHA-256 block matches. Every patch is applied again locally and must reproduce the signed target byte for byte.
3. Includes a patch only when it is smaller than the complete APK. The full signed APK is always included.
4. Writes immutable, hash-qualified APK/patch filenames, `latest.json`, and `SHA256SUMS.txt`. Re-running identical input is safe; conflicting output files are never overwritten.

This command only stages files. It does not create a GitHub release, upload artifacts, publish the website, or change an installed app. The output is not available to phones until publication is completed. A green Android CI build, a merged settings PR, or a website deployment alone does not publish a compatible native update.

## Publication order

Publication remains an explicit release step:

1. Publish the staged APK and patch files at their exact versioned URLs. Keep assets immutable, retain the complete APK, and preserve the signing identity.
2. Retrieve every published file and verify its byte length and SHA-256 against the staged manifest. A successful upload alone is insufficient.
3. Copy the reviewed `latest.json` to `mobile/android/releases/<review|stable>/latest.json`. This designated source is the only update manifest the website build publishes; the website build does not automatically select a local APK or copy Android build outputs.
4. Build and deploy the website. Its build copies approved manifests to `public/app-updates/<channel>/latest.json`. The `.vercelignore` allowlist includes only these staged manifests and the two dependency-free release-validation scripts from `mobile/android/`; native source, APKs, signing files, and Gradle output stay outside the deployment build context. Publish this pointer **after** the referenced files are available.
5. Verify the public manifest and exercise **Settings → Updates → Check now → Download update → Install update** from a supported earlier updater-enabled APK. Earlier app versions may label the first action **Check for updates**. Confirm preserved bookmarks/settings, version increment, and normal launch after installation.

If a release must be withdrawn, point the channel back only for clients that have not installed it. The updater does not downgrade installed apps. Fix an installed faulty release with a higher version code. Retain historical approved hashes so users can still update through a full APK download when a delta is unavailable.

## Patch protocol: `dsupd1-gzip`

The patch is a gzip stream. The decompressed bytes are:

| Field | Encoding |
| --- | --- |
| Magic | Eight ASCII bytes `DSUPD001` |
| Base SHA-256 | 32 raw bytes |
| Target SHA-256 | 32 raw bytes |
| Base byte length | Positive signed 64-bit big-endian integer |
| Target byte length | Positive signed 64-bit big-endian integer |
| Operations | Records below, followed by END |

- **COPY:** byte `0`, nonnegative 64-bit big-endian base offset, positive 32-bit big-endian length.
- **LITERAL:** byte `1`, positive 32-bit big-endian length, then that many literal bytes.
- **END:** byte `255`; no decompressed trailing bytes are allowed.

The generator normally matches 32 KiB source blocks and merges contiguous COPY records. The decoder does not depend on a block size. APKs and downloaded patches are capped at 256 MiB, with no more than 100,000 operations. Every source/output range is checked before copying. Base identity, complete output length, final output hash, gzip integrity, and end-of-stream are checked. Inventories contain at most 200 release records and 200 patches; manifests are bounded to 512 KiB. The installed app and builder must retain matching limits.

## Focused validation

```powershell
node --test mobile/android/scripts/app-update-format.test.cjs
```

`scripts/app-update-protocol-fixture.json` supplies small base, target, compressed-patch, and decompressed-protocol bytes for Kotlin/Node interoperability checks. Tests cover inserted/shifted data, corruption, truncation, range and operation limits, wrong identities, downgrade rejection, historical hashes, approved URLs, and immutable artifact staging. Android updater tests additionally cover the real package metadata, download/install flow, cancellation, retry, and Settings UI.

The Android workflow also runs focused settings, navigation, and update UI tests in an isolated Android 16 emulator. Its device artifact includes reports and screenshots captured by the tests. See [Settings validation](SETTINGS.md#validation) for the equivalent local command and remaining release-device checks.

Review the resulting APK with `apksigner verify`, run the Android unit/device/lint checks, and check that a patch round trip equals the exact signed target. Do not claim public updating works solely from a local staging run.
