package me.danielshort.app.updates

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import me.danielshort.app.data.AppSettings

interface AppUpdateActions {
  val state: StateFlow<AppUpdateState>
  val automaticDownloadsBlocked: StateFlow<Boolean>
  fun check(automated: Boolean = false): Boolean
  fun download(automated: Boolean = false): Boolean
  fun cancelAutomaticDownload()
}

data class UpdateNetworkState(val connected: Boolean = false, val metered: Boolean = true)

/** One instance per process, started only by the visible app, never a content worker. */
class AppUpdateCoordinator(
  private val updates: AppUpdateActions,
  private val settings: StateFlow<AppSettings>,
  network: StateFlow<UpdateNetworkState>,
  private val scope: CoroutineScope,
  private val installAutomatically: suspend (() -> Boolean) -> AutomaticInstallResult = { AutomaticInstallResult.DEFERRED },
  private val backgroundGraceMillis: Long = 30_000L,
  protectedWorkChanges: Flow<Boolean> = flowOf(false),
  private val hasProtectedWork: () -> Boolean = { false }
) {
  private val foreground = MutableStateFlow(false)
  private val safeToInstall = MutableStateFlow(false)
  private val backgroundReady = MutableStateFlow(0L)
  private var backgroundGeneration = 0L
  private var backgroundGrace: Job? = null
  private var launchHandled = false
  @Volatile private var automaticInstallSuppressed = false
  private val attemptedDownloads = mutableSetOf<Long>()
  private val attemptedInstalls = mutableSetOf<Long>()
  private val deferredInstalls = mutableMapOf<Long, Long>()

  init {
    combine(settings, network, updates.state, updates.automaticDownloadsBlocked, foreground) { options, connection, state, blocked, visible ->
      if (!options.automaticAppUpdates || !connection.connected || (options.appUpdatesUnmeteredOnly && connection.metered)) {
        // Ownership is checked by the manager: manual downloads remain independent.
        updates.cancelAutomaticDownload()
      }
      if (visible && !launchHandled) {
        when {
          !options.checkAppUpdatesOnLaunch && !options.automaticAppUpdates -> launchHandled = true
          state !is AppUpdateState.Idle -> launchHandled = true
          connection.connected -> {
            launchHandled = true
            updates.check(automated = true)
          }
        }
      }
      if (state is AppUpdateState.Downloading) attemptedDownloads += state.offer.versionCode
      if (visible && options.automaticAppUpdates && !blocked && connection.connected &&
        (!options.appUpdatesUnmeteredOnly || !connection.metered) && state is AppUpdateState.Available &&
        state.offer.versionCode !in attemptedDownloads) {
        if (updates.download(automated = true)) attemptedDownloads += state.offer.versionCode
      }
    }.launchIn(scope)

    combine(settings, updates.state, foreground, safeToInstall, backgroundReady) { _, state, _, _, _ -> state }
      .combine(protectedWorkChanges) { state, _ -> state }
      .onEach { state ->
        if (state is AppUpdateState.Ready && isAutomaticInstallEligible() &&
          deferredInstalls[state.offer.versionCode] != backgroundReady.value && attemptedInstalls.add(state.offer.versionCode)) {
          // The installer must recheck this live predicate before committing its session.
          val generation = backgroundReady.value
          val result = try {
            installAutomatically(::isAutomaticInstallEligible)
          } catch (cancelled: CancellationException) {
            throw cancelled
          } catch (_: Exception) {
            // A failed journal or platform call must not stop checks/download observation.
            AutomaticInstallResult.FAILED
          }
          if (result == AutomaticInstallResult.DEFERRED) {
            attemptedInstalls.remove(state.offer.versionCode)
            deferredInstalls[state.offer.versionCode] = generation
          }
        }
      }.launchIn(scope)
  }

  fun onForeground() {
    backgroundGrace?.cancel()
    backgroundReady.value = 0L
    foreground.value = true
  }

  fun onBackground() {
    foreground.value = false
    backgroundReady.value = 0L
    backgroundGrace?.cancel()
    backgroundGrace = scope.launch {
      delay(backgroundGraceMillis)
      backgroundReady.value = ++backgroundGeneration
    }
  }

  /** The shell permits this only on browse-only screens, with no active native workspace. */
  fun setSafeToInstall(safe: Boolean) { safeToInstall.value = safe }

  /** An explicit installer interaction takes priority, including the user's Cancel. */
  fun suppressAutomaticInstall() { automaticInstallSuppressed = true }

  fun isAutomaticInstallEligible(): Boolean = settings.value.automaticAppUpdates &&
    !automaticInstallSuppressed && !foreground.value && safeToInstall.value && !hasProtectedWork() &&
    backgroundReady.value != 0L && updates.state.value is AppUpdateState.Ready
}
