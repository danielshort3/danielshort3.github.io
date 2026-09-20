package me.danielshort.app.updates

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import me.danielshort.app.data.AppSettings
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class AppUpdateCoordinatorTest {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
  private val options = MutableStateFlow(AppSettings())
  private val network = MutableStateFlow(UpdateNetworkState(connected = true, metered = false))
  private val recordingActive = MutableStateFlow(false)
  private val actions = FakeActions()
  private val offer = UpdateOffer(5, "0.4.0", 1000, true)

  @After fun cleanup() { scope.cancel() }

  @Test fun launchChecksDefaultOnAndDownloadsRequireOptIn() {
    assertTrue(options.value.checkAppUpdatesOnLaunch)
    assertFalse(options.value.automaticAppUpdates)
    assertTrue(options.value.appUpdatesUnmeteredOnly)
    val coordinator = coordinator()
    assertEquals(0, actions.checks)
    coordinator.onForeground()
    assertEquals(1, actions.checks)
    actions.state.value = AppUpdateState.Available(offer)
    assertEquals(0, actions.downloads)
  }

  @Test fun rotationReentryAndInstallerReturnsDoNotCheckAgain() {
    val coordinator = coordinator()
    coordinator.onForeground()
    repeat(3) { coordinator.onBackground(); coordinator.onForeground() }
    actions.state.value = AppUpdateState.Error("Offline", UpdateRetryAction.CHECK)
    coordinator.onBackground()
    coordinator.onForeground()
    assertEquals(1, actions.checks)
  }

  @Test fun offlineLaunchDefersOneCheckUntilOnlineForeground() {
    network.value = UpdateNetworkState()
    val coordinator = coordinator()
    coordinator.onForeground()
    assertEquals(0, actions.checks)
    coordinator.onBackground()
    network.value = UpdateNetworkState(true, false)
    assertEquals(0, actions.checks)
    coordinator.onForeground()
    assertEquals(1, actions.checks)
  }

  @Test fun launchOptOutKeepsManualCheckAndDownloadAvailable() {
    options.value = options.value.copy(checkAppUpdatesOnLaunch = false)
    coordinator().onForeground()
    assertEquals(0, actions.checks)
    actions.check()
    actions.state.value = AppUpdateState.Available(offer)
    actions.download()
    assertEquals(1, actions.checks)
    assertEquals(1, actions.downloads)
    assertFalse(actions.automaticDownload)
  }

  @Test fun manualCheckBeforeOfflineLaunchCompletesIsNotOverwritten() {
    network.value = UpdateNetworkState()
    val coordinator = coordinator()
    coordinator.onForeground()
    actions.check()
    actions.state.value = AppUpdateState.Available(offer)
    network.value = UpdateNetworkState(true, false)
    assertEquals(1, actions.checks)
    assertTrue(actions.state.value is AppUpdateState.Available)
  }

  @Test fun automaticUpdatesCheckEvenWithLegacyInconsistentLaunchPreference() {
    options.value = options.value.copy(automaticAppUpdates = true, checkAppUpdatesOnLaunch = false)
    coordinator().onForeground()
    assertEquals(1, actions.checks)
  }

  @Test fun meteredDownloadWaitsForUnmeteredAndStopsIfNetworkChanges() {
    options.value = options.value.copy(automaticAppUpdates = true)
    network.value = UpdateNetworkState(true, true)
    coordinator().onForeground()
    actions.state.value = AppUpdateState.Available(offer)
    assertEquals(0, actions.downloads)
    network.value = UpdateNetworkState(true, false)
    assertEquals(1, actions.downloads)
    network.value = UpdateNetworkState(true, true)
    assertEquals(1, actions.automaticCancellations)
    network.value = UpdateNetworkState(true, false)
    assertEquals("A canceled download must not enter an automatic retry loop", 1, actions.downloads)
  }

  @Test fun meteredDownloadsCanBeExplicitlyEnabled() {
    options.value = options.value.copy(automaticAppUpdates = true, appUpdatesUnmeteredOnly = false)
    network.value = UpdateNetworkState(true, true)
    coordinator().onForeground()
    actions.state.value = AppUpdateState.Available(offer)
    assertEquals(1, actions.downloads)
  }

  @Test fun optingOutCancelsAutomaticDownloadButNotManualDownload() {
    options.value = options.value.copy(automaticAppUpdates = true)
    coordinator().onForeground()
    actions.state.value = AppUpdateState.Available(offer)
    options.value = options.value.copy(automaticAppUpdates = false)
    assertEquals(1, actions.automaticCancellations)
    actions.download()
    network.value = UpdateNetworkState()
    assertEquals(1, actions.automaticCancellations)
    assertTrue(actions.state.value is AppUpdateState.Downloading)
  }

  @Test fun cancellationAndErrorsDoNotRestartAutomaticDownloads() {
    options.value = options.value.copy(automaticAppUpdates = true)
    val coordinator = coordinator()
    coordinator.onForeground()
    actions.state.value = AppUpdateState.Available(offer)
    actions.state.value = AppUpdateState.Error("Failed verification", UpdateRetryAction.DOWNLOAD)
    coordinator.onBackground()
    coordinator.onForeground()
    actions.state.value = AppUpdateState.Available(offer)
    assertEquals(1, actions.downloads)
    actions.automaticDownloadsBlocked.value = true
    actions.state.value = AppUpdateState.Available(offer.copy(versionCode = 6))
    assertEquals(1, actions.downloads)
    actions.download()
    assertEquals("Explicit retry remains available", 2, actions.downloads)
  }

  @Test fun installRequiresSafeBackgroundGraceAndRechecksLiveEligibility() = runBlocking {
    options.value = options.value.copy(automaticAppUpdates = true)
    actions.state.value = AppUpdateState.Ready(offer)
    var installs = 0
    var eligibility: (() -> Boolean)? = null
    val coordinator = coordinator { check -> installs++; eligibility = check; AutomaticInstallResult.STARTED }
    coordinator.onForeground()
    coordinator.setSafeToInstall(true)
    delay(30)
    assertEquals(0, installs)
    coordinator.onBackground()
    withTimeout(1000) { while (installs == 0) delay(5) }
    assertTrue(eligibility!!.invoke())
    coordinator.onForeground()
    assertFalse(eligibility!!.invoke())
    coordinator.onBackground()
    delay(30)
    assertEquals("Each version is submitted only once per process", 1, installs)
  }

  @Test fun foregroundReturnUnsafeWorkspaceAndOptOutPreventAutomaticInstall() = runBlocking {
    options.value = options.value.copy(automaticAppUpdates = true)
    actions.state.value = AppUpdateState.Ready(offer)
    var installs = 0
    val coordinator = coordinator { installs++; AutomaticInstallResult.STARTED }
    coordinator.onForeground()
    coordinator.onBackground()
    delay(30)
    assertEquals("Active workspace is ineligible", 0, installs)
    coordinator.onForeground()
    coordinator.setSafeToInstall(true)
    coordinator.onBackground()
    coordinator.onForeground()
    delay(30)
    assertEquals("Short activity transitions are ineligible", 0, installs)
    coordinator.onBackground()
    options.value = options.value.copy(automaticAppUpdates = false)
    delay(30)
    assertEquals(0, installs)
  }

  @Test fun explicitInstallerInteractionSuppressesAutomaticInstallAfterCancel() = runBlocking {
    options.value = options.value.copy(automaticAppUpdates = true)
    actions.state.value = AppUpdateState.Ready(offer)
    var installs = 0
    val coordinator = coordinator { installs++; AutomaticInstallResult.STARTED }
    coordinator.onForeground()
    coordinator.setSafeToInstall(true)
    coordinator.suppressAutomaticInstall()
    coordinator.onBackground()
    delay(30)
    assertEquals(0, installs)
  }

  @Test fun deferredInstallRetriesOnlyAfterAnotherBackgroundGrace() = runBlocking {
    options.value = options.value.copy(automaticAppUpdates = true)
    actions.state.value = AppUpdateState.Ready(offer)
    var installs = 0
    val coordinator = coordinator { installs++; AutomaticInstallResult.DEFERRED }
    coordinator.onForeground()
    coordinator.setSafeToInstall(true)
    coordinator.onBackground()
    withTimeout(1000) { while (installs == 0) delay(5) }
    coordinator.setSafeToInstall(false)
    coordinator.setSafeToInstall(true)
    delay(30)
    assertEquals(1, installs)
    coordinator.onForeground()
    coordinator.onBackground()
    withTimeout(1000) { while (installs < 2) delay(5) }
    assertEquals(2, installs)
  }

  @Test fun unexpectedInstallFailureDoesNotKillObservationOrRepeatVersion() = runBlocking {
    options.value = options.value.copy(automaticAppUpdates = true)
    actions.state.value = AppUpdateState.Ready(offer)
    var installs = 0
    val coordinator = coordinator { installs++; throw IllegalStateException("Journal unavailable") }
    coordinator.onForeground()
    coordinator.setSafeToInstall(true)
    coordinator.onBackground()
    withTimeout(1000) { while (installs == 0) delay(5) }
    coordinator.onForeground()
    coordinator.onBackground()
    delay(30)
    assertEquals(1, installs)
    actions.state.value = AppUpdateState.Ready(offer.copy(versionCode = 6))
    withTimeout(1000) { while (installs < 2) delay(5) }
  }

  @Test fun activeRecordingBlocksInstallAfterLeavingRecorderAndIsObservedWhenStopped() = runBlocking {
    options.value = options.value.copy(automaticAppUpdates = true)
    actions.state.value = AppUpdateState.Ready(offer)
    recordingActive.value = true
    var installs = 0
    var eligibility: (() -> Boolean)? = null
    val coordinator = coordinator { check -> installs++; eligibility = check; AutomaticInstallResult.STARTED }
    coordinator.onForeground()
    coordinator.setSafeToInstall(true)
    coordinator.onBackground()
    delay(30)
    assertEquals("A recording service is protected even on a browse-only screen", 0, installs)
    assertFalse(coordinator.isAutomaticInstallEligible())
    recordingActive.value = false
    withTimeout(1000) { while (installs == 0) delay(5) }
    assertTrue(eligibility!!.invoke())
    recordingActive.value = true
    assertFalse("Installer must see a newly started recording immediately", eligibility!!.invoke())
  }

  private fun coordinator(install: suspend (() -> Boolean) -> AutomaticInstallResult = { AutomaticInstallResult.STARTED }) =
    AppUpdateCoordinator(actions, options, network, scope, install, backgroundGraceMillis = 20,
      protectedWorkChanges = recordingActive, hasProtectedWork = { recordingActive.value })

  private inner class FakeActions : AppUpdateActions {
    override val state = MutableStateFlow<AppUpdateState>(AppUpdateState.Idle)
    override val automaticDownloadsBlocked = MutableStateFlow(false)
    var checks = 0
    var downloads = 0
    var automaticDownload = false
    var automaticCancellations = 0
    override fun check(automated: Boolean): Boolean {
      checks++
      state.value = AppUpdateState.Checking
      return true
    }
    override fun download(automated: Boolean): Boolean {
      downloads++
      automaticDownload = automated
      state.value = AppUpdateState.Downloading(offer, 0f, true)
      return true
    }
    override fun cancelAutomaticDownload() {
      if (automaticDownload) {
        automaticCancellations++
        automaticDownload = false
        state.value = AppUpdateState.Available(offer)
      }
    }
  }
}
