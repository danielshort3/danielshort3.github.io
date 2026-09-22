#!/usr/bin/env bash
# Run only against an isolated emulator, never a personal connected phone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mapfile -t devices < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')
if [[ ${#devices[@]} -ne 1 || "${devices[0]}" != emulator-* ]]; then
  echo "Exactly one isolated Android emulator must be connected." >&2
  exit 1
fi
export ANDROID_SERIAL="${devices[0]}"
mkdir -p output/android-settings
finish() {
  result=$?
  trap - EXIT
  adb pull /sdcard/Android/data/me.danielshort.app.debug/files/settings-qa output/android-settings/ || true
  adb logcat -b crash -d > output/android-settings/crash-log.txt || true
  exit "$result"
}
trap finish EXIT
adb logcat -c
bash mobile/android/gradlew -p mobile/android --no-daemon --console=plain connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=me.danielshort.app.ui.SettingsFlowTest,me.danielshort.app.ui.SettingsMenusTest,me.danielshort.app.ui.AppUpdateSectionTest,me.danielshort.app.ui.AdaptiveSiteLayoutFlowTest,me.danielshort.app.ui.ScrollChromeFlowTest
