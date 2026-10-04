package me.danielshort.wayfarers.content

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface GuildContentUpdateState {
  data object Idle : GuildContentUpdateState
  data object Checking : GuildContentUpdateState
  data class Current(val version: Long, val label: String) : GuildContentUpdateState
  data class Available(val manifest: GuildContentManifest, val autoDownloadAllowed: Boolean = true) : GuildContentUpdateState
  data class Downloading(val manifest: GuildContentManifest, val progress: Float) : GuildContentUpdateState
  data class Ready(val manifest: GuildContentManifest) : GuildContentUpdateState
  data class Error(val message: String) : GuildContentUpdateState
}

/** Downloads never change a running page; only the Activity may start the checkpointed apply transaction. */
class GuildContentUpdateManager(
  private val store: GuildContentStore,
  private val feedUrl: String,
  private val transport: GuildContentTransport = GuildContentHttpsTransport(),
  private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) {
  private val mutableState = MutableStateFlow<GuildContentUpdateState>(GuildContentUpdateState.Idle)
  val state: StateFlow<GuildContentUpdateState> = mutableState.asStateFlow()
  private var operation: Job? = null
  private var offer: GuildContentManifest? = null
  @Volatile private var automaticDownloadAllowed = true
  private val operationLock = Any()

  fun refresh() = start {
    val ready = store.stagedManifest()
    if (ready != null) {
      offer = ready
      GuildContentUpdateState.Ready(ready)
    } else {
      offer = null
      val active = store.session()
      GuildContentUpdateState.Current(active.contentVersion, active.label)
    }
  }

  fun check() = start {
    automaticDownloadAllowed = true
    mutableState.value = GuildContentUpdateState.Checking
    val coroutine = currentCoroutineContext()
    val signed = transport.manifest(feedUrl) { coroutine.ensureActive() }
    val active = store.session()
    val manifest = try { store.verifyManifest(signed) } catch (failure: GuildContentCompatibilityFailure) {
      // A newer APK may supersede the previous signed feed before the next
      // patch is published. Authenticated older content is never downloaded.
      if (failure.contentVersion <= active.contentVersion) {
        offer = null
        return@start GuildContentUpdateState.Current(active.contentVersion, active.label)
      }
      throw failure
    }
    if (manifest.contentVersion <= active.contentVersion) {
      offer = null
      GuildContentUpdateState.Current(active.contentVersion, active.label)
    } else {
      store.accept(manifest)
      offer = manifest
      val staged = store.stagedManifest()
      if (staged?.id == manifest.id) GuildContentUpdateState.Ready(staged) else GuildContentUpdateState.Available(manifest)
    }
  }

  fun download() = start {
    val manifest = offer ?: throw GuildContentFailure("Check for a game update first.")
    store.accept(manifest)
    mutableState.value = GuildContentUpdateState.Downloading(manifest, 0f)
    val coroutine = currentCoroutineContext()
    val cancel = { coroutine.ensureActive() }
    val temporary = store.downloadFile()
    try {
      transport.download(manifest.archive, temporary, { received, total ->
        mutableState.value = GuildContentUpdateState.Downloading(manifest, (received.toFloat() / total).coerceIn(0f, 1f))
      }, cancel)
      store.stage(manifest, temporary, cancel)
      GuildContentUpdateState.Ready(manifest)
    } finally {
      temporary.delete()
    }
  }

  fun cancel() {
    synchronized(operationLock) {
      automaticDownloadAllowed = false
      operation?.cancel()
    }
  }

  private fun start(work: suspend () -> GuildContentUpdateState) {
    synchronized(operationLock) {
      // A cancelled blocking transfer still owns its temporary file until its finally block completes.
      if (operation != null) return
      val job = scope.launch(start = CoroutineStart.LAZY) {
        val running = currentCoroutineContext()[Job]
        try {
          publishAndRelease(running, work())
        } catch (failure: CancellationException) {
          publishAndRelease(running, cancelledState())
          throw failure
        } catch (failure: Exception) {
          publishAndRelease(running, GuildContentUpdateState.Error(if (failure is GuildContentFailure) failure.message.orEmpty()
            else "The game update could not be completed. Your current game is unchanged."))
        } finally {
          synchronized(operationLock) { if (operation === running) operation = null }
        }
      }
      operation = job
      job.start()
    }
  }

  private fun cancelledState(): GuildContentUpdateState = offer?.let {
    GuildContentUpdateState.Available(it, autoDownloadAllowed = automaticDownloadAllowed)
  } ?: GuildContentUpdateState.Idle

  private fun publishAndRelease(running: Job?, terminal: GuildContentUpdateState) {
    synchronized(operationLock) {
      if (operation !== running) return
      val outcome = if (running?.isCancelled == true) cancelledState() else terminal
      // A collector may immediately request the next operation when it sees Available.
      // Release this operation before publishing its terminal state, including on an inline dispatcher.
      operation = null
      mutableState.value = outcome
    }
  }
}
