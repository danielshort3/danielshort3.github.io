package me.danielshort.app.updates

import android.content.Context
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import me.danielshort.app.BuildConfig
import java.io.File

data class UpdateOffer(val versionCode: Long, val versionName: String, val downloadBytes: Long, val usingPatch: Boolean)
enum class UpdateRetryAction { CHECK, DOWNLOAD }
sealed interface AppUpdateState {
  data object Idle : AppUpdateState
  data object Checking : AppUpdateState
  data class UpToDate(val versionName: String) : AppUpdateState
  data class Available(val offer: UpdateOffer) : AppUpdateState
  data class Downloading(val offer: UpdateOffer, val progress: Float?, val usingPatch: Boolean) : AppUpdateState
  data class Ready(val offer: UpdateOffer) : AppUpdateState
  data class Error(val message: String, val retryAction: UpdateRetryAction) : AppUpdateState
}

/** Application-owned state survives navigation and configuration changes; nothing installs automatically. */
class AppUpdateManager(
  private val storageDirectory: File,
  private val feedUrl: String,
  private val verifier: InstalledAppVerifier,
  private val transport: UpdateTransport = HttpsUpdateTransport(),
  private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
) : AppUpdateActions {
  constructor(
    context: Context,
    feedUrl: String = BuildConfig.APP_UPDATE_URL,
    scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO),
  ) : this(File(context.applicationContext.filesDir, "app-updates"), feedUrl, AndroidInstalledAppVerifier(context), HttpsUpdateTransport(), scope)

  private data class Plan(val manifest: UpdateManifest, val installed: VerifiedApk, val patch: UpdatePatch?, val offer: UpdateOffer)
  private val mutableState = MutableStateFlow<AppUpdateState>(AppUpdateState.Idle)
  override val state: StateFlow<AppUpdateState> = mutableState.asStateFlow()
  private val mutableAutomationBlocked = MutableStateFlow(false)
  override val automaticDownloadsBlocked = mutableAutomationBlocked.asStateFlow()
  private val operationLock = Any()
  private val mutex = Mutex()
  private var operation: Job? = null
  private var automaticDownload: Job? = null
  private var plan: Plan? = null
  private val readyDirectory = File(storageDirectory, "ready")

  override fun check(automated: Boolean) = start(UpdateRetryAction.CHECK, automated) {
    mutableState.value = AppUpdateState.Checking
    plan = null
    prepareStorage()
    val manifest = UpdateManifestParser.parse(transport.manifest(feedUrl))
    val coroutine = currentCoroutineContext()
    val installed = verifier.installed { coroutine.ensureActive() }
    UpdatePolicy.recognizeInstalled(manifest, installed)
    pruneInstalledReleases(installed.versionCode)
    if (manifest.latest.versionCode <= installed.versionCode) {
      return@start AppUpdateState.UpToDate(installed.versionName)
    }
    if (manifest.latest.minSdk > verifier.sdkVersion) throw UpdateFailure("This update requires a newer Android version.")
    val patch = manifest.patches.firstOrNull { it.fromSha256 == installed.sha256 && it.toSha256 == manifest.latest.apk.sha256 && it.artifact.size < manifest.latest.apk.size }
    val offer = UpdateOffer(manifest.latest.versionCode, manifest.latest.versionName, patch?.artifact?.size ?: manifest.latest.apk.size, patch != null)
    plan = Plan(manifest, installed, patch, offer)
    AppUpdateState.Available(offer)
  }

  override fun download(automated: Boolean) = start(UpdateRetryAction.DOWNLOAD, automated) {
    val selected = plan ?: throw UpdateFailure("Check for updates before downloading.")
    mutableState.value = AppUpdateState.Downloading(selected.offer, null, selected.offer.usingPatch)
    prepareStorage()
    val coroutine = currentCoroutineContext()
    val cancel = { coroutine.ensureActive() }
    val installed = try {
      verifier.installed(cancel).also {
        UpdatePolicy.sameInstalled(selected.installed, it)
        UpdatePolicy.recognizeInstalled(selected.manifest, it)
      }
    } catch (failure: CancellationException) {
      throw failure
    } catch (failure: Exception) {
      plan = null
      throw failure
    }
    val target = selected.manifest.latest
    val ready = readyFile(target)
    if (ready.isFile) {
      try {
        UpdatePolicy.verifyTarget(verifier.archive(ready, cancel), installed, target, verifier.sdkVersion)
        return@start AppUpdateState.Ready(selected.offer)
      } catch (failure: CancellationException) {
        throw failure
      } catch (_: Exception) {
        ready.delete()
      }
    }
    val staged = File(storageDirectory, "target.part")
    val patchFile = File(storageDirectory, "patch.part")
    var offer = selected.offer
    try {
      var useFullApk = selected.patch == null
      selected.patch?.let { patch ->
        mutableState.value = AppUpdateState.Downloading(offer, 0f, true)
        try {
          transport.download(patch.artifact, patchFile) { received, total ->
            mutableState.value = AppUpdateState.Downloading(offer, received.toFloat() / total, true)
          }
        } catch (_: UpdateTransferFailure) {
          // A missing/interrupted patch may use the complete APK; corrupt patches fail closed below.
          coroutine.ensureActive()
          useFullApk = true
        }
        if (!useFullApk) {
          verifyArtifact(patchFile, patch.artifact, cancel)
          UpdatePolicy.sameInstalled(installed, verifier.installed(cancel))
          mutableState.value = AppUpdateState.Downloading(offer, null, true)
          try {
            UpdatePatchApplier.apply(installed.file, patchFile, staged, installed.sha256, target.apk.sha256, target.apk.size, cancel)
          } catch (failure: CancellationException) {
            throw failure
          } catch (failure: Exception) {
            throw UpdateFailure("The update patch could not be verified. Nothing was changed. Retry the download or check for updates again.", failure)
          }
        }
      }
      if (useFullApk) {
        offer = selected.offer.copy(downloadBytes = target.apk.size, usingPatch = false)
        mutableState.value = AppUpdateState.Downloading(offer, 0f, false)
        transport.download(target.apk, staged) { received, total ->
          mutableState.value = AppUpdateState.Downloading(offer, received.toFloat() / total, false)
        }
        verifyArtifact(staged, target.apk, cancel)
      }
      mutableState.value = AppUpdateState.Downloading(offer, null, offer.usingPatch)
      UpdatePolicy.sameInstalled(installed, verifier.installed(cancel))
      UpdatePolicy.verifyTarget(verifier.archive(staged, cancel), installed, target, verifier.sdkVersion)
      coroutine.ensureActive()
      if (!staged.renameTo(ready)) throw UpdateFailure("There was not enough storage to prepare the update. Free some space and retry.")
      AppUpdateState.Ready(offer)
    } finally {
      staged.delete()
      patchFile.delete()
    }
  }

  fun cancel() {
    // A deliberate cancellation suppresses further automatic downloads this process.
    // Explicit Check/Download actions remain available.
    mutableAutomationBlocked.value = true
    synchronized(operationLock) { operation?.cancel() }
  }

  override fun cancelAutomaticDownload() {
    synchronized(operationLock) {
      if (operation === automaticDownload) automaticDownload?.cancel()
    }
  }

  /** Call immediately before constructing the Android installer intent; the UI must still ask Android to install. */
  suspend fun verifiedApkForInstall(): File = withContext(Dispatchers.IO) {
    mutex.withLock {
      val selected = plan ?: throw UpdateFailure("Check for updates before installing.")
      if (mutableState.value !is AppUpdateState.Ready) throw UpdateFailure("Download and verify the update before installing.")
      try {
        val coroutine = currentCoroutineContext()
        val cancel = { coroutine.ensureActive() }
        val installed = verifier.installed(cancel)
        UpdatePolicy.sameInstalled(selected.installed, installed)
        UpdatePolicy.recognizeInstalled(selected.manifest, installed)
        val apk = readyFile(selected.manifest.latest)
        UpdatePolicy.verifyTarget(verifier.archive(apk, cancel), installed, selected.manifest.latest, verifier.sdkVersion)
        apk
      } catch (failure: CancellationException) {
        throw failure
      } catch (failure: Exception) {
        mutableState.value = AppUpdateState.Error(friendlyMessage(failure), UpdateRetryAction.CHECK)
        throw failure
      }
    }
  }

  private fun start(retry: UpdateRetryAction, automated: Boolean, action: suspend () -> AppUpdateState): Boolean {
    synchronized(operationLock) {
      if (operation?.isActive == true || (automated && retry == UpdateRetryAction.DOWNLOAD && automaticDownloadsBlocked.value)) return false
      val next = scope.launch(Dispatchers.IO, start = CoroutineStart.LAZY) {
        val running = currentCoroutineContext()[Job]
        val result = try {
          mutex.withLock { action() }
        } catch (_: CancellationException) {
          plan?.let { AppUpdateState.Available(it.offer) } ?: AppUpdateState.Idle
        } catch (failure: Exception) {
          if (retry == UpdateRetryAction.DOWNLOAD) mutableAutomationBlocked.value = true
          AppUpdateState.Error(friendlyMessage(failure), if (plan == null) UpdateRetryAction.CHECK else retry)
        }
        synchronized(operationLock) {
          if (operation === running) {
            // Terminal states are actionable immediately, after storage work and ownership finish.
            operation = null
            automaticDownload = null
            mutableState.value = result
          }
        }
      }
      operation = next
      automaticDownload = if (automated && retry == UpdateRetryAction.DOWNLOAD) next else null
      next.start()
      return true
    }
  }

  private fun prepareStorage() {
    if (!readyDirectory.exists() && !readyDirectory.mkdirs()) throw UpdateFailure("The app could not create storage for this update. Free some space and retry.")
    if (!readyDirectory.isDirectory || !readyDirectory.canWrite()) throw UpdateFailure("Update storage is unavailable. Free some space and retry.")
    // Only this updater's temporary files are removed. Ready APKs may still be open in Android's installer.
    listOf("target.part", "patch.part").forEach { File(storageDirectory, it).delete() }
  }

  private fun readyFile(release: LatestRelease) = File(readyDirectory, "${release.versionCode}-${release.apk.sha256}.apk")

  private fun pruneInstalledReleases(installedVersion: Long) {
    val ownedName = Regex("([1-9][0-9]*)-[0-9a-f]{64}\\.apk")
    readyDirectory.listFiles()?.forEach { file ->
      val version = ownedName.matchEntire(file.name)?.groupValues?.get(1)?.toLongOrNull()
      if (version != null && version <= installedVersion && file.isFile) file.delete()
    }
  }

  private fun verifyArtifact(file: File, artifact: UpdateArtifact, cancel: () -> Unit) {
    if (file.length() != artifact.size || sha256(file, cancel) != artifact.sha256) throw UpdateFailure("The downloaded update failed its integrity check. Nothing was changed. Please retry.")
  }

  private fun friendlyMessage(failure: Exception): String = when (failure) {
    is UpdateFailure, is UpdateTransferFailure -> failure.message ?: "The app update could not be verified. Please try again."
    is java.io.IOException -> "The update could not be saved or downloaded. Check your connection and available storage, then retry."
    else -> "The app update could not be verified. Please try again."
  }
}
