# Android settings

The native settings redesign is introduced with version 0.5.0 / versionCode 7. Source changes and CI artifacts are not public releases; publication still follows [UPDATES.md](UPDATES.md).

## Navigation and ownership

`ui/SettingsScreen.kt` owns the overview, Updates, Storage, and App information destinations. The overview has only four rows:

- **Updates** opens the update task and preferences.
- **Reduce motion** is an immediate, whole-row switch.
- **Storage** contains image-cache cleanup, not a general data-reset operation.
- **App information** explains privacy and shows the version. It is distinct from the main About section about Daniel.

`ui/SettingsControls.kt` owns reusable navigation/switch rows and single-choice dialogs. `data/SettingsChoices.kt` maps those choices to the existing `AppSettings` booleans. `ui/SystemMotion.kt` observes Android's disabled-animation preference; effective reduced motion is the logical OR of the system and app preferences. `ui/BrowseMenus.kt` contains manual content refresh and saved-project management menus.

`ui/DanielShortApp.kt` retains the five primary sections. A new Settings entry starts at the overview; a new update-notice entry starts at Updates. State still survives configuration restoration within an entry. Leaving and reopening an update notice must not restore a previous visit's overview.

Bulk bookmark removal belongs to **Projects → Saved → Saved project options → Remove all saved projects**, with confirmation. It removes bookmark references only. The browsing header's **More options** menu contains **Refresh website content**.

## Preference contract

There is no storage migration and no new preference key. Existing choices remain authoritative. Reading an unusual historical combination must not rewrite it merely by opening Settings.

| UI selection | Existing stored fields |
| --- | --- |
| App updates: Check automatically | `checkAppUpdatesOnLaunch=true`, `automaticAppUpdates=false` |
| App updates: Automatic | `checkAppUpdatesOnLaunch=true`, `automaticAppUpdates=true` |
| App updates: Manual | `checkAppUpdatesOnLaunch=false`, `automaticAppUpdates=false` |
| Automatic downloads: Unmetered connections | `appUpdatesUnmeteredOnly=true` |
| Automatic downloads: Any connection | `appUpdatesUnmeteredOnly=false` |
| Content refresh: Automatic | `automaticUpdates=true`, `unmeteredOnly=false` |
| Content refresh: Unmetered connections only | `automaticUpdates=true`, `unmeteredOnly=true` |
| Content refresh: Manual | `automaticUpdates=false`, preserve `unmeteredOnly` |

Changing an app-update mode preserves the content settings, reduced-motion setting, and app-download network restriction. The network row appears only in Automatic mode; hiding it never clears the stored choice. Content refresh remains independent of native app updates. Explicit manual content refresh and app downloads remain available regardless of the automatic network rules.

Default native app behavior remains **Check automatically**, with automatic app downloads off. The existing default content-refresh behavior remains automatic. The labels describe Android metering, not a literal Wi-Fi transport requirement.

Choice dialogs commit only when an option is selected. Cancel, system Back, or outside dismissal makes no change. The Reduce motion row exposes one switch action and a minimum 48dp interaction target; normal rows are at least 72dp high and expand for wrapped text.

## Presentation and safety

The Settings top bar stays visible while the body scrolls. Settings uses a stable brand-blue accent rather than the currently selected main section's color. Text wraps instead of being ellipsized or shrunk. No additional theme picker, search field, advanced catch-all, or primary navigation overhaul is introduced.

`AppUpdateSection.kt` keeps the current status and action ahead of preferences. Routine patch/full-download implementation details and repeated version text are removed from the primary UI. Verification, error messages, cancellation, retry, permissions, and Android installation handoff remain intact. Settings is still a native workspace and never grants automatic-install eligibility.

Successful refresh or cleanup uses a snackbar. Recoverable failures remain beside their action. Cache clearing touches Coil's image caches only and does not advertise a fabricated storage total. AI-demo notices appear beside submission controls; the digit generator explicitly discloses its automatic initial example request.

## Validation

From the repository root, with Node, JDK, and Android SDK configured:

```bash
node tests/site/mobile-content.test.js
node --test mobile/android/scripts/*.test.cjs tests/site/app-update-feeds.test.cjs
bash mobile/android/gradlew -p mobile/android --no-daemon --console=plain testDebugUnitTest lintDebug assembleDebug assembleDebugAndroidTest
```

For the focused device suite, connect exactly one isolated emulator and no personal phone:

```bash
bash mobile/android/scripts/test-settings-device.sh
```

The script rejects physical-device targets. It runs `SettingsFlowTest`, `SettingsMenusTest`, `AppUpdateSectionTest`, `AdaptiveSiteLayoutFlowTest`, and `ScrollChromeFlowTest`; it preserves the test exit status while collecting crash logs and any captured screenshots. CI uses an Android 16 emulator and read-only repository permissions. Inspect the `android-settings-device-<commit>` artifact, not only the build result.

Coverage includes all 64 historical boolean combinations, independent mode changes, persisted hidden network choices, cancellation, switch semantics and touch targets, state restoration, repeat update-notice navigation, system Back, stable accent color, 320dp-wide/200%-text layouts, wide layouts, and bulk-removal confirmation. Offline fixtures do not submit AI requests or install a production update.

Before publishing a correctly signed release, also exercise the actual older-APK upgrade on a designated review device. Test installation-permission denial/return, Android confirmation/cancellation, offline recovery, preservation of bookmarks and preferences, and the displayed version after restart. Review TalkBack reading order and the system reduced-animation behavior on that device. A passing isolated UI suite does not establish that the public APK and feed have been published or that every vendor installer behaves identically.
