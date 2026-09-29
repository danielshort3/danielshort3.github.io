package me.danielshort.app

import androidx.compose.runtime.Composable
import me.danielshort.app.data.AppSettings

/** Google Play owns app installation and updates for the release variant. */
object AppUpdateRuntime {
  fun initialize(app: SiteApplication) {}
  fun onForeground() {}
  fun onBackground() {}
  fun setSafeToInstall(safe: Boolean) {}
  fun suppressAutomaticInstall() {}

  @Composable
  fun notice(onOpenSettings: () -> Unit) {}

  @Composable
  fun settingsSummary(options: AppSettings): String = "Refresh website listings"

  @Composable
  fun settingsContent(options: AppSettings, onChange: (AppSettings) -> Unit) {}
}
