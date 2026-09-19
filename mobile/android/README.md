# Daniel Short for Android

A Kotlin and Jetpack Compose app that shares the website's public content. Navigation, project details, tools, game adaptations, dashboards, and AI demo interfaces use Android UI and Kotlin code. There is no WebView, embedded website shell, or HTML renderer. Website updates supply content and data rather than executable web code.

## What is native

| Area | Current behavior |
| --- | --- |
| About | Native profile, interests, experience, education, and credentials |
| Projects | Native searchable catalog and project details, local bookmarks, and Android share sheet |
| Tools | Ten public utilities with native text processing, QR generation, image processing, and screen recording |
| Games | Six native experiences: Roulette, Stellar Dogfight, Ocean Wave Simulation, Project Starfall, Probability Engine, and Stormbreak |
| Project demos | Native drawing, digit generation, language inputs, Nonogram replay, historical datasets, and dashboard filters/charts |
| Settings | Header gear; automatic content updates, unmetered updates, reduced motion, refresh, image cache, and bookmark controls |
| Contact | Native contact cards; project questions open an email app with the project in the subject |
| External resources | PDFs, source repositories, credentials, and unsupported future catalog entries are explicitly opened in another app |

The larger games preserve core gameplay rather than every website mechanic or visual effect. Tableau projects use native charts over the published historical records, not Tableau's HTML embed. Website accounts and cloud session saving remain separate; native bookmarks and game checkpoints are local and do not sync with website accounts.

Text utilities and QR generation run on the device. Image optimization uses Android bitmap codecs and preserves EXIF orientation. Background removal uses Google ML Kit subject segmentation locally; it requires Google Play services and a first-use model download. AI demos send only submitted input to the existing website/AWS inference services and show service failures explicitly. The app does not embed server credentials.

Screen recording uses Android MediaProjection with system consent for each recording. Optional microphone audio is requested only when enabled; device playback audio is not captured. Clips are limited to 15 minutes/512 MiB, retained privately across restarts, and can be previewed, exported through Android's document picker, shared, or deleted. Record only content you choose in Android's capture prompt.

## How website updates reach the app

```text
Website content sources
        ↓ website build
build/generate-mobile-content.js
        ↓ public JSON + revision + versioned image URLs
https://www.danielshort.me/app-content/v1/catalog.json
        ↓ HTTPS fetch and validation
Native repository → Compose screens
```

The generator uses the website's content loader and explicitly selects public fields. The normal website build generates `dist/app-content/v1/catalog.json` and copies it into `public/app-content/v1/catalog.json` for deployment. Changes to supported text, lists, project entries, images, and links appear in the app after the updated feed is deployed and refreshed. New entries appear automatically; a new entry does not automatically gain a native implementation.

With automatic updates enabled, the app checks when opened or brought to the foreground, throttled to 15 minutes after the last successful check. The refresh icon and Settings' Refresh now request an immediate check. WorkManager also schedules a network-dependent refresh approximately every six hours. Settings can disable these automatic checks or restrict them to unmetered networks. Manual refresh overrides those preferences. Background refresh is best effort: Android can delay it for battery saving, idle mode, or connectivity.

The repository validates schema version 1, identifiers, HTTPS links, and payload limits before replacing its saved copy. It uses ETag conditional requests when available and writes the cache atomically. An invalid response, network failure, or unsupported schema leaves the last usable content available.

**The production feed is live** at `https://www.danielshort.me/app-content/v1/catalog.json`, deployed through website PR #209. The existing default APK can refresh from it without reinstalling. Future content edits reach phones after the website build is deployed; building only the Android project or running a local preview does not publish those edits. The bundled catalog remains available offline.

Native layouts, Kotlin behavior, dependencies, permissions, and new native features require a rebuilt and installed app update. Website CSS or JavaScript changes do not alter the native UI. No code is fetched and executed to bypass Android app updates.

## Offline behavior

Every Android build runs the feed generator and bundles its latest JSON as `catalog.json`. The app first loads a valid cached remote catalog, or falls back to that bundled snapshot. This supports the first launch without a working production feed and later use without a connection.

Text tools, QR generation, image optimization, recording, and native games work offline. Background removal works after its Google model is downloaded. Tableau dashboards bundle a reproducible historical snapshot from the repository's published sources; use `scripts/extract-tableau-data.py` (requires `tableauhyperapi`) to regenerate those assets for an app update. Other historical dashboards fetch the published website JSON and use cached datasets offline after first loading. AI inference requires a reachable backend.

Images use Coil's cache after being fetched; remote images that have never loaded can show placeholders offline. Browser resources and email delivery depend on their external apps and connectivity. Clearing app data or uninstalling removes private recordings, bookmarks, checkpoints, and cached content; exported files are separate.

## Build locally

Run commands from the website repository root. Requirements:

- Node.js 22 on `PATH` for the website content generator.
- JDK 17 or newer compatible with Gradle 8.13. Android Studio's bundled JBR 21 is used on this workstation.
- Android SDK platform **36.1**, Build Tools **36.1.0**, and platform-tools. Minimum supported device version is Android 8/API 26; the app targets API 36.

Gradle 8.13, Android Gradle Plugin 8.13.2, Kotlin 2.2.10, and the Compose dependencies are pinned. Use the checked-in wrapper. The wrapper distribution has a pinned SHA-256 checksum.

```powershell
$env:JAVA_HOME = 'C:\Program Files\Android\Android Studio\jbr'
$env:ANDROID_HOME = 'C:\Users\clopt\AppData\Local\Android\Sdk'

.\mobile\android\gradlew.bat -p mobile/android assembleDebug testDebugUnitTest
```

Alternatively, configure the SDK through Android Studio or `mobile/android/local.properties`, which is ignored by Git. If Node 22 is not the default executable, pass `'-PnodeExecutable=C:\path\to\node.exe'` to Gradle.

The installable debug APK is:

```text
mobile/android/app/build/outputs/apk/debug/app-debug.apk
```

Debug application ID: `me.danielshort.app.debug`. Release application ID: `me.danielshort.app`.

To install on a connected test device or running emulator:

```powershell
& "$env:ANDROID_HOME\platform-tools\adb.exe" devices
& "$env:ANDROID_HOME\platform-tools\adb.exe" install -r 'mobile/android/app/build/outputs/apk/debug/app-debug.apk'
& "$env:ANDROID_HOME\platform-tools\adb.exe" shell am start -n 'me.danielshort.app.debug/me.danielshort.app.MainActivity'
```

Add `-s <serial>` after `adb.exe` when more than one device is connected. The existing `Medium_Phone` AVD can be used for local checks; the app does not require this particular emulator.

## Preview local content in the emulator

First generate and serve the website through its route-aware development server:

```powershell
npm.cmd run build
npm.cmd run dev
```

If that server is listening on port 4173, build the debug variant with:

```powershell
.\mobile\android\gradlew.bat -p mobile/android assembleDebug '-PcatalogUrl=http://10.0.2.2:4173/app-content/v1/catalog.json'
```

`10.0.2.2` is the Android emulator's route to the host computer. The override applies only to debug builds. Debug HTTP is restricted to `10.0.2.2`, `127.0.0.1`, and `localhost`; release traffic remains HTTPS and the release feed URL cannot be changed with `catalogUrl`.

For a USB-connected phone, use `adb reverse tcp:4173 tcp:4173` and a debug override of `http://127.0.0.1:4173/app-content/v1/catalog.json`. The phone must remain connected for that route to work.

Rebuild without `-PcatalogUrl` to return to the production endpoint before sharing an APK for normal phone use. The override changes the JSON endpoint only; image URLs generated by the feed still reference the website. New images therefore need to be published before they can load from their production URLs.

## Checks and source locations

```powershell
# Public feed generation and focused website-side validation
node build/generate-mobile-content.js
node tests/site/mobile-content.test.js

# Native parser, text comparison, and game logic tests
.\mobile\android\gradlew.bat -p mobile/android testDebugUnitTest

# Android resource/code checks
.\mobile\android\gradlew.bat -p mobile/android lintDebug
```

Important files:

- `app/src/main/java/me/danielshort/app/ui/DanielShortApp.kt`: native screen hierarchy and Android actions.
- `app/src/main/java/me/danielshort/app/data/ContentRepository.kt`: cache, refresh, and bookmarks.
- `app/src/main/java/me/danielshort/app/data/SiteContent.kt`: typed content model and validation.
- `app/src/main/java/me/danielshort/app/SiteApplication.kt`: background refresh scheduling.
- `app/src/main/java/me/danielshort/app/nativefeatures/`: native tools, recorder, game models, dashboards, and inference clients.
- `app/build.gradle.kts`: dependencies, SDK settings, endpoint, and bundled-catalog generation.

The DS header and launcher mark use native vector paths from the website's current master logo. The shared shell deliberately stays light, with the website's navy, blue, teal, orange, and slate section colors.

## Signing and publication

The Android GitHub Actions workflow runs unit tests, lint, and a debug APK build for relevant pull requests and main-branch changes. It uploads a review APK and reports; it does not publish to an app store. Website CI also checks the public content feed.

The debug APK is for local installation and review. It is signed with Android's development key, not a production signing identity. No Play Store listing, production signing key, upload credentials, or automatic app distribution is configured by this project.

For distribution, create and securely retain a release signing/upload key, configure signing outside committed source, increment `versionCode` for app updates, build a signed release APK or App Bundle, and complete the selected distribution channel's setup. Do not commit keystores or passwords. Release artifacts are not ready for public distribution until signing and publication are configured.

The website feed deployment and Android app publication are separate release steps. Once the feed is public, supported content updates can reach installed apps without a new APK. Changes to native app code continue to require a normal signed app update.
