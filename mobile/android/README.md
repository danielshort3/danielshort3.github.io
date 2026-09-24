# Daniel Short for Android

A Kotlin and Jetpack Compose app that shares the website's public content. Its 16 published project cards open first-party website case studies in an in-app WebView, with native offline summaries. Eleven first-party demos and five games also use the website in-app by default, with native alternatives. Nine public tools and Probability Engine open their canonical website routes in the Android browser, with native alternatives; Screen Recorder stays native. Navigation, settings, bookmarks, recording, and the retained adaptations use Android UI and Kotlin code.

## How each area opens

| Area | Current behavior |
| --- | --- |
| About | Native profile, interests, experience, education, and credentials |
| Projects | Native searchable catalog, local bookmarks, and Android share sheet; all published cards open the website case study in-app, with an offline native summary available |
| Tools | Nine website tools open in the Android browser, each with an optional native adaptation. Screen Recorder uses Android screen capture |
| Games | Project Starfall, Stellar Dogfight, Roulette, Stormbreak, and Ocean Wave Simulation open their website games in-app, with native alternatives. Probability Engine opens in the browser, with a native alternative |
| Project demos | Eleven first-party website demos open in-app with native alternatives. The Pizza Delivery and UFO Tableau dashboard adaptations remain native |
| Settings | Header gear; automatic content updates, unmetered updates, reduced motion, refresh, image cache, and bookmark controls |
| App updates | Checks for published releases on launch by default; optional automatic verified downloads and installation while the app is out of use, subject to Android's permissions |
| Contact | Native contact cards; project questions open an email app with the project in the subject |
| External resources | PDFs, source repositories, credentials, and unsupported future catalog entries are explicitly opened in another app |

The native game alternatives preserve core gameplay rather than every website mechanic or visual effect. The optional native Tableau dashboard adaptations use charts over published historical records; the website case studies contain Tableau embeds. Website accounts and cloud session saving remain separate; native bookmarks and game checkpoints are local and do not sync with website accounts. In-app WebView storage, Android browser storage, and native app storage are separate. Opening a website game does not convert a native checkpoint into a website save.

The optional native text utilities and QR generator run on the device. Native image optimization uses Android bitmap codecs and preserves EXIF orientation. Native background removal uses Google ML Kit subject segmentation locally; it requires Google Play services and a first-use model download. Native AI demos send only submitted input to the existing website/AWS inference services and show service failures explicitly. The app does not embed server credentials. Website tools open in the Android browser so their file import/export and account flows remain available.

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

For published project case studies, 11 first-party demos, and five games (Project Starfall, Stellar Dogfight, Roulette, Stormbreak, and Ocean Wave Simulation), the app loads the deployed first-party route in a WebView. The project view retains Android bookmark/share actions and an **Offline summary** option. Nine website tools and Probability Engine open in the Android browser; their cards provide **Open Android version** as an alternative. Screen Recorder stays native. Website HTML, CSS, and JavaScript changes to the web routes reach them after the website is deployed. Installing an app version with this routing is required first; native alternatives continue to use the installed app code and data.

The generator uses the website's content loader and explicitly selects public fields. The normal website build generates `dist/app-content/v1/catalog.json` and copies it into `public/app-content/v1/catalog.json` for deployment. Changes to supported text, lists, project entries, images, and links appear in the app after the updated feed is deployed and refreshed. New entries appear automatically; a new entry does not automatically gain a native implementation.

With automatic updates enabled, the app checks when opened or brought to the foreground, throttled to 15 minutes after the last successful check. The refresh icon and Settings' Refresh now request an immediate check. WorkManager also schedules a network-dependent refresh approximately every six hours. Settings can disable these automatic checks or restrict them to unmetered networks. Manual refresh overrides those preferences. Background refresh is best effort: Android can delay it for battery saving, idle mode, or connectivity.

The repository validates schema version 1, identifiers, HTTPS links, and payload limits before replacing its saved copy. It uses ETag conditional requests when available and writes the cache atomically. An invalid response, network failure, or unsupported schema leaves the last usable content available.

**The production feed is live** at `https://www.danielshort.me/app-content/v1/catalog.json`, deployed through website PR #209. The existing default APK can refresh from it without reinstalling. Future content edits reach phones after the website build is deployed; building only the Android project or running a local preview does not publish those edits. The bundled catalog remains available offline.

Native layouts, Kotlin behavior, dependencies, permissions, and new native features require a rebuilt and installed app update. Website CSS or JavaScript changes affect the in-app website case studies, demos, and games, plus the browser tools and Probability Engine, after deployment. They do not alter native features. The WebView runs deployed website code; the app does not download code to change its native implementation.

### Updating native features from Settings

The app checks for native updates on a cold launch by default, without blocking the current screen. Rotation and returns from permissions or the installer do not repeat the launch check. **Settings → App updates** also provides a manual check and an optional automatic-update setting, independent of website-content refresh. Automatic downloads are restricted to unmetered connections by default; manual actions remain available.

The updater recognizes exact published APK bytes, checks signing identity, and verifies the finished update before handing it to Android. A smaller patch is preferred; a verified complete APK can recover from an unavailable patch. Unknown local builds, modified APKs, split installations, different signing keys, and mismatched downloads cannot bypass verification.

The first updater-enabled APK must be installed manually once. In-app updates then require a published channel manifest and matching APK/patch assets; a local build alone does not publish them. Android controls installation permission and may require confirmation. Automatic installation is opt-in and deferred while a native workspace, game, recording, or project detail is active. See [UPDATES.md](UPDATES.md) for platform requirements, artifact preparation, signing, and the release order.

## Phones, tablets, and resizable windows

The available app window determines navigation. Below 840dp, the app retains its phone layout and bottom navigation. Wider windows use the website's five colored vertical tabs, with the active page expanding between them. The window can change size without resetting the selected section or saved editor state.

Wide libraries use additional columns when cards have enough room; reading and settings panels keep a comfortable maximum width. The optional native Text Compare adaptation places its editors side by side when space and the text-size setting allow it. The tabs remain available on wide screens, while the header can hide as content scrolls. Phone headers and bottom navigation keep their existing scroll behavior.

## Offline behavior

Every Android build runs the feed generator and bundles its latest JSON as `catalog.json`. The app first loads a valid cached remote catalog, or falls back to that bundled snapshot. This supports the first launch without a working production feed and later use without a connection.

The native text tools, QR generator, image optimizer, recorder, and game adaptations work offline; native background removal works after its Google model is downloaded. Native project summaries use the cached catalog offline. Website case studies, demos, and games need connectivity for a reliable first load. Smart Sentence, Travel Chat, Nonogram, and other model-backed website demos also need their remote services. Pizza Tips map tiles need connectivity, while its retained native model works offline. Website caching may make previously visited resources available offline, but it is not a complete offline copy. The nine browser tools and browser Probability Engine depend on the Android browser and its own cache; their native alternatives can be used offline where their functions permit. Tableau dashboards bundle a reproducible historical snapshot from the repository's published sources; use `scripts/extract-tableau-data.py` (requires `tableauhyperapi`) to regenerate those assets for an app update. Other native historical dashboards fetch the published website JSON and use cached datasets offline after first loading. Other native AI inference requires a reachable backend.

Images use Coil's cache after being fetched; remote images that have never loaded can show placeholders offline. Browser resources and email delivery depend on their external apps and connectivity. In-app website experiences share the app's WebView storage; website tools and Probability Engine use Android browser storage. Both are separate from native checkpoints, and no account, demo-state, tool-state, or game-save transfer between them is implemented. Clearing app data or uninstalling removes private recordings, bookmarks, checkpoints, WebView storage, and cached content; browser data and exported files are separate.

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

Rebuild without `-PcatalogUrl` to return to the production endpoint before sharing an APK for normal phone use. The override changes the JSON endpoint only; image URLs generated by the feed, in-app web routes, and browser tool/game links still reference the website. New images and web behavior therefore need to be published before they can load from their production URLs.

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
- `app/src/main/java/me/danielshort/app/ui/WebExperienceScreen.kt` and `WebExperiencePolicy.kt`: first-party WebView presentation and allowed project, game, and demo routes.
- `app/src/main/java/me/danielshort/app/ui/AdaptiveSiteLayout.kt`: window-width policy, native vertical tabs, and stable content placement across resizing.
- `app/src/main/java/me/danielshort/app/updates/AppUpdateCoordinator.kt`: launch checks, network-aware downloads, and background installation eligibility.
- `app/src/main/java/me/danielshort/app/updates/AutomaticInstallEngine.kt` and `AutomaticAppInstaller.kt`: persistent installation state, Android package sessions, and manual-confirmation fallback. See [UPDATES.md](UPDATES.md) for the verification and publication contract.
- `app/src/main/java/me/danielshort/app/ui/ScrollChrome.kt`: shared header/navigation visibility. Consumed downward gestures hide the complete bars; upward gestures reveal them. Lists retain stable geometry, while text focus, the keyboard, and touch exploration keep navigation available. Reduced motion uses immediate changes. Focused project demos reuse the top bar without adding bottom navigation.
- `app/src/main/java/me/danielshort/app/data/ContentRepository.kt`: cache, refresh, and bookmarks.
- `app/src/main/java/me/danielshort/app/data/SiteContent.kt`: typed content model and validation.
- `app/src/main/java/me/danielshort/app/SiteApplication.kt`: background refresh scheduling.
- `app/src/main/java/me/danielshort/app/nativefeatures/`: native tools, recorder, game models, dashboards, and inference clients.
- `app/build.gradle.kts`: dependencies, SDK settings, endpoint, and bundled-catalog generation.

The DS header and launcher mark use native vector paths from the website's current master logo. The shared shell deliberately stays light, with the website's navy, blue, teal, orange, and slate section colors.

## Signing and publication

The Android GitHub Actions workflow runs unit tests, lint, and a debug APK build for relevant pull requests and main-branch changes. It uploads a review APK and reports; it does not publish to an app store. Website CI also checks the public content feed.

The debug APK is for installation and review. Published review APKs and their update manifest use the preserved workstation development signing identity. CI's separately generated debug key cannot replace that identity. No Play Store listing or stable-channel production signing identity is configured.

For stable distribution, create and securely retain a release signing/upload key, configure signing outside committed source, increment `versionCode` for app updates, build a signed release APK or App Bundle, and complete the selected distribution channel's setup. Do not commit keystores or passwords. Review-channel publication follows [UPDATES.md](UPDATES.md).

The website feed deployment and Android app publication are separate release steps. Once the feed is public, supported content updates can reach installed apps without a new APK. After an app version with the current web routing is installed, deployed website changes to the case studies, demos, games, and browser tools can reach the app without another APK. Changes to native app code or route selection continue to require a normal signed app update.
