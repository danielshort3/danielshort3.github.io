package me.danielshort.app

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.HorizontalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import me.danielshort.app.data.AppSettings
import me.danielshort.app.data.appUpdateMode
import me.danielshort.app.nativefeatures.recording.ScreenRecording
import me.danielshort.app.ui.AppUpdateNotice
import me.danielshort.app.ui.AppUpdatePreferences
import me.danielshort.app.ui.AppUpdateSection
import me.danielshort.app.ui.appUpdateChoices
import me.danielshort.app.updates.AppUpdateCoordinator
import me.danielshort.app.updates.AppUpdateManager
import me.danielshort.app.updates.AppUpdateState
import me.danielshort.app.updates.AutomaticAppInstaller
import me.danielshort.app.updates.UpdateNetworkMonitor

/** The review channel owns the self-updater; this source is absent from Play builds. */
object AppUpdateRuntime {
  private lateinit var application: SiteApplication
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
  val manager by lazy { AppUpdateManager(application) }
  val installer by lazy { AutomaticAppInstaller(application, manager) }
  private val coordinator by lazy {
    AppUpdateCoordinator(manager, application.contentRepository.settings.state,
      UpdateNetworkMonitor(application, scope).state, scope,
      installAutomatically = { eligible -> installer.attemptIfEligible(eligible) },
      protectedWorkChanges = ScreenRecording.state.map { it.active }.distinctUntilChanged(),
      hasProtectedWork = { ScreenRecording.state.value.active })
  }

  fun initialize(app: SiteApplication) { application = app }
  fun onForeground() {
    coordinator.onForeground()
    scope.launch { installer.recover() }
  }
  fun onBackground() { coordinator.onBackground() }
  fun setSafeToInstall(safe: Boolean) { coordinator.setSafeToInstall(safe) }
  fun suppressAutomaticInstall() { coordinator.suppressAutomaticInstall() }

  @Composable
  fun notice(onOpenSettings: () -> Unit) { AppUpdateNotice(manager, onOpenSettings) }

  @Composable
  fun settingsSummary(options: AppSettings): String {
    val state by manager.state.collectAsStateWithLifecycle()
    return when (state) {
      is AppUpdateState.Available -> "Update available"
      is AppUpdateState.Ready -> "Ready to install"
      is AppUpdateState.Checking -> "Checking for updates…"
      is AppUpdateState.Downloading -> "Downloading update…"
      is AppUpdateState.UpToDate -> "You’re up to date"
      is AppUpdateState.Error -> "Update needs attention"
      is AppUpdateState.Idle -> appUpdateChoices.first { it.value == options.appUpdateMode }.title
    }
  }

  @Composable
  fun settingsContent(options: AppSettings, onChange: (AppSettings) -> Unit) {
    AppUpdateSection(manager, installer, options.automaticAppUpdates) {
      AppUpdatePreferences(options, onChange)
    }
    HorizontalDivider(Modifier.padding(vertical = 8.dp))
  }
}
