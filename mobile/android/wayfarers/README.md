# Wayfarers' Guild for Android

`:wayfarers` is the standalone, offline game application. It installs alongside the personal-site application, with its own private progress and settings. It is not an in-place replacement for either `me.danielshort.app` or `me.danielshort.app.debug`.

| Contract | Value |
| --- | --- |
| Android package, both local build types | `me.danielshort.wayfarers` |
| Generated-resource namespace | `me.danielshort.app` |
| Default candidate | `0.8.0`, version code `10` |
| Offline game origin | `https://appassets.androidplatform.net/assets/wayfarers/index.html` |
| Dedicated update manifest | `https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-updates/latest.json` |

The namespace preserves the canonical updater's `BuildConfig` import; this module generates its own `APP_UPDATE_URL`. The Android package controls installation, private data, installer callback actions, and FileProvider authority. Sharing a Kotlin namespace does not share app data.

## Source ownership

The game remains authored in `pages/games/wayfarers-guild.html`, `js/games/wayfarers-guild/`, `css/games/wayfarers-guild.css`, and `img/wayfarers-guild/`. Gradle's `bundleWayfarers` task calls `build/bundle-wayfarers-android.cjs --output <generated-assets-directory>` and packages its `wayfarers/` tree. No website `public/` build or first network load is required. Treat the appassets origin as a storage contract: changing it would isolate existing WebView saves.

The runtime is under `src/main/java/me/danielshort/wayfarers/`. Its activity and application own the WebView, lifecycle, save-file exchange and native update interface. Its explicit, unexported installer callback receiver lives under `me/danielshort/app/updates/` to satisfy the canonical installer's component reference.

`GuildCheckpointStore.kt` adds a private, atomic, fsynced checkpoint of each successful canonical game save. `web/checkpoint.js` validates the envelope with the canonical parser before mirroring it and before bootstrap recovery; the native wrapper verifies its SHA-256 when reading. A newer valid WebView save wins. Recovery across a different guild identity requires the recorded identity from an explicitly reviewed import. `native-checkpoint.js` is an empty generated fallback for browser previews; the Android asset loader supplies the private checkpoint at that exact same-origin script URL without caching it. Checkpoints contain the normal exported game envelope, never a paid wallet. Lifecycle and installer safety checks finish the native write before reporting success.

`shareUpdaterSources` copies the canonical `app/src/main/java/me/danielshort/app/updates/*.kt` and `data/AppSettings.kt` into an ignored generated source set without rewriting them. It excludes only `AutomaticInstallReceiver.kt`, whose application cast is specific to the main app. `shareUpdaterTests` compiles the original updater JVM tests in this module too. Changes to signatures, exact APK identity, patch verification, download bounds or installer recovery remain owned by the original updater sources; do not fork them here. The main application build configuration is unchanged.

The game envelope now uses schema 6, adding an initially inactive card and equipment collection. The checkpoint container remains record version 1 and accepts matching game envelope/state versions 1 through 6, so an older checkpoint can reach the canonical migration code. Retained expedition-version-2 runs keep their released economy until a confirmed Refit or Charter. Saving the migrated game atomically writes a version-6 envelope; the WebView backup retains the previous valid bytes. [Collection contracts](../../../docs/games/wayfarers-collections.md) describe permanent inventory, named decks, scroll outcomes and schema compatibility.

The additive [onboarding metadata](../../../docs/games/wayfarers-onboarding.md) retains first-visit walkthrough steps, discovery notices and once-only learning rewards in the same checkpoint. Existing saves keep their production and ownership. Area guides resume after an interrupted visit; completed guides can be replayed without another reward. Testing reset clears these fields with the guild, while normal Refit and Charter retain them.

## Local builds and signing

Use the SDK and JDK requirements from [the Android build guide](../README.md). From the repository root:

```powershell
.\mobile\android\gradlew.bat -p mobile/android :wayfarers:assembleDebug :wayfarers:testDebugUnitTest :wayfarers:lintDebug --console=plain
```

Output: `mobile/android/wayfarers/build/outputs/apk/debug/wayfarers-debug.apk`.

Both `debug` and `release` are explicitly signed with the existing persistent workstation development key at `~/.android/debug.keystore`. No signing key or password is stored in this module. `verifyPersistentSigningIdentity` fails if that retained file is missing, before a game build can silently generate another identity. Preserve the key outside source control. CI-generated development keys cannot publish compatible upgrades. A different key at the same path also cannot update an existing installation; release preparation must verify the retained certificate against prior signed APKs.

This is the requested sideload preview channel, not a Google Play production signing configuration. The module does not expose payment or advertisement availability merely because the game contains those optional web clients.

For a baseline upgrade fixture, preserve the exact signed baseline bytes outside the repository before building the candidate:

```powershell
.\mobile\android\gradlew.bat -p mobile/android :wayfarers:assembleDebug '-PwayfarersVersionCode=1' '-PwayfarersVersionName=0.1.0' --console=plain
# Archive the signed baseline outside the repository, then build the candidate.
.\mobile\android\gradlew.bat -p mobile/android :wayfarers:assembleDebug --console=plain
```

These properties affect only `:wayfarers`. Version codes must increase for each published target. Debug builds intentionally have no package or version-name suffix, so baseline and candidate share one installation identity.

## Update and save boundaries

The shared updater verifies recognized installed bytes, package, version, signer, download size and hash, then verifies the exact reconstructed APK after a binary patch. Android's installer replaces the app. No downloaded game scripts are executed independently of an APK update. The dedicated manifest and artifacts require separate publication; a successful local build does not make a phone update available. See [the updater protocol](../UPDATES.md) and the dedicated release tooling's product contract.

The native **Automatic updates** option follows the shared updater's eligibility checks, including verified downloads and deferred installation only while the app is out of use and Android permits it. Its unmetered setting applies to automatic downloads. It does not bypass installation permission or the platform's confirmation requirements. Manual checking, downloading and installation remain available separately.

The manifest permits networking and package installation, disables cleartext traffic and Android backup/transfer, and exposes only the private verified `app-updates/ready/` directory through an unexported FileProvider. It does not request microphone, recording or broad storage access. JSON saves may be opened or shared into the game; the runtime must use inspected import and confirmation before replacing a save. Personal-site app, Android browser and standalone game state remain separate; there is no silent migration.

Source/JVM/lint results, installed offline play and a real in-app baseline-to-candidate upgrade are distinct checks. Preserve a game checkpoint and settings through the installed upgrade before claiming continuity.

Focused game checks run from the repository root:

```powershell
npm run test:wayfarers-guild:expeditions
npm run test:wayfarers-guild:progression:browser
npm run test:wayfarers-guild:collections:browser
npm run test:wayfarers-guild:onboarding:browser
node tests/games/wayfarers-guild-expedition.browser.cjs
node --test mobile/android/scripts/wayfarers-bundle.test.cjs
```

These browser checks build the same offline assets as the APK. The expedition suite checks retained published runs; the progression suite checks the new opening, six-area navigation, exact bulk transactions, cap remainders, gestures and bounded layouts. Set `WAYFARERS_QA_DIR` to an external directory to retain screenshots. `GuildOfflineDeviceTest` separately covers real Android text scaling, rotation, offline rendering, first purchase geometry, context-sheet Back navigation, six-area controls and native checkpoint acknowledgement. Run first-boot and opted-in mature fixtures only on a disposable emulator; never clear an existing player's data to make a case run.

`GuildOnboardingDeviceTest` requires `guildOnboardingQa=true` and an explicitly reset disposable guild. It exercises the actual guide controls, saved step resume, Android Back, rotation, enlarged text, native checkpoint acknowledgment and rewardless replay. Legacy interaction tests can explicitly acknowledge guides with `guildGuideAcknowledgementQa=true`; those tests isolate established gameplay controls and do not replace first-visit coverage. Run the fresh-guide class separately so test ordering cannot consume its initial state.

## Testing reset

Open **Settings & saves → Testing → Reset all game progress**, optionally download a backup, and type `RESET` to confirm. This clears the guild's currencies, areas, ranks, resets, discoveries, cards, equipment and local recovery copies. It preserves app preferences, other site data and account-owned purchases; downloaded backup files remain available. It is separate from the in-game Refit and Charter resets.

`persistence.js` writes a fixed fresh snapshot and unique generation to a durable journal before touching saves. Canonical main/backup keys include that generation; fixed keys are compatibility mirrors. Old sessions cannot overwrite the new generation, including purchases saved after reset. A pending journal blocks gameplay and offers Retry using the same snapshot. The Android bridge requires an atomic `GuildCheckpointStore` reset acknowledgment before marking completion; normal and onPause checkpoints carry the same generation. Lost acknowledgments retry idempotently. Never implement this using `localStorage.clear()`, app-data deletion or silent checkpoint fallback.

`npm run test:wayfarers-guild:progression` includes fault-injection reset tests. Run `npm run test:wayfarers-guild:testing-reset:browser` for confirmation, layout, failure/retry, reload and stale tier-popup checks. Verify a populated disposable device resets, makes a first purchase and retains it through an offline cold launch; never use a retained player device for destructive testing.

## Publish a compatible update

1. Build `:wayfarers:assembleRelease` with an increased `wayfarersVersionCode` and version name. Archive the exact APK outside the checkout. Release builds disable WebView debugging; the retained signing certificate must match the previously installed release.
2. Run the canonical release builder with `--channel wayfarers`, the new APK, each supported prior APK as `--base`, the previous dedicated manifest, an empty external output directory, and the immutable GitHub release URL. For example:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Android\Android Studio\jbr'
node mobile/android/scripts/prepare-app-update.cjs `
  --apk C:\Guild-Releases\candidate.apk `
  --base C:\Guild-Releases\previous.apk `
  --previous-manifest C:\Guild-Releases\previous-latest.json `
  --output C:\Guild-Releases\new-release `
  --base-url https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-v0.1.2/ `
  --channel wayfarers
```

3. Publish the immutable APK, generated smaller patches, versioned manifest and SHA256SUMS to that release. Download the public artifacts and compare exact sizes and SHA256 hashes. The builder verifies signatures and round-trips each binary patch to the exact signed target.
4. Only after those checks, replace `latest.json` on the fixed `wayfarers-guild-updates` release. This one manifest is the mutable channel pointer; APK and patch filenames remain immutable. Do not substitute the main application's review/stable manifest.
5. On the previous installed standalone release, open **App options → App updates → Check now → Download update → Install update**. Allow installation from this app when Android asks, then tap Install update again. Verify the installed package/version/hash and retained guild/settings, then check offline launch.

The default **Check automatically** mode checks at launch without downloading. **Automatic updates** opts into downloads on unmetered connections and attempts installation only after leaving the app with App updates open, a successful save flush and a background grace period. Returning to gameplay prevents automatic installation. Android may require explicit confirmation. **Manual** checks only when requested. An unpublished or unreachable feed shows a recoverable status; offline play remains available.

Earned currencies, progression and backups work in the sideload app. Real Play purchases and rewarded advertisements are unavailable until this separate package is registered and its store/advertisement services are configured and verified. The personal-site application's purchase wallet and advertising credentials are not reused implicitly. To bring a guild from the website/main app, export its game save and open that JSON through the standalone app; review it before confirming replacement. Keep any separate paid-wallet recovery information separately.
