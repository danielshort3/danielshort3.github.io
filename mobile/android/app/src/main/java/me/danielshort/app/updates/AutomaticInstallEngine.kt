package me.danielshort.app.updates

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.util.UUID

enum class AutomaticInstallStatus {
  IDLE, DEFERRED, STAGING, INSTALLING, NEEDS_PERMISSION, MANUAL_REQUIRED, FAILED, INSTALLED;

  val inProgress: Boolean get() = this == STAGING || this == INSTALLING
}

enum class AutomaticInstallResult { DEFERRED, STARTED, ALREADY_ATTEMPTED, MANUAL_REQUIRED, FAILED }

internal data class AutomaticInstallRecord(
  val versionCode: Long,
  val status: AutomaticInstallStatus,
  val sessionId: Int = -1,
  val token: String = "",
  val startedAt: Long = 0,
)

internal interface AutomaticInstallStore {
  fun read(): AutomaticInstallRecord?
  /** Must be durable before a session can be committed. */
  fun write(record: AutomaticInstallRecord)
}

internal interface AutomaticInstallPlatform {
  val sdkVersion: Int
  fun canInstall(): Boolean
  fun installedVersion(): Long
  fun targetSdk(apk: File): Int
  fun sessions(): Set<Int>
  fun create(apk: File): Int
  fun stage(sessionId: Int, apk: File, checkEligible: () -> Unit)
  fun commit(sessionId: Int, token: String)
  fun abandon(sessionId: Int)
}

internal enum class AutomaticInstallCallback { PENDING_USER_ACTION, SUCCESS, FAILURE }

internal object AutomaticInstallPolicy {
  // Unknown future releases fall back to a user-initiated install until reviewed.
  fun supportsUnattended(deviceSdk: Int, targetSdk: Int): Boolean = when (deviceSdk) {
    31, 32 -> targetSdk >= 29
    33 -> targetSdk >= 30
    34 -> targetSdk >= 31
    35 -> targetSdk >= 33
    36 -> targetSdk >= 34
    37 -> targetSdk >= 35
    else -> false
  }
}

/** One application-owned engine serializes attempts, callback delivery, and recovery. */
internal class AutomaticInstallEngine(
  private val store: AutomaticInstallStore,
  private val platform: AutomaticInstallPlatform,
  private val readyVersion: () -> Long?,
  private val verifiedApk: suspend () -> File,
  private val now: () -> Long = System::currentTimeMillis,
) {
  private val mutex = Mutex()
  private val mutableStatus = MutableStateFlow(store.read()?.status ?: AutomaticInstallStatus.IDLE)
  val status = mutableStatus.asStateFlow()

  suspend fun attemptIfEligible(isEligible: () -> Boolean): AutomaticInstallResult = mutex.withLock {
    if (!isEligible()) return@withLock AutomaticInstallResult.DEFERRED
    val version = readyVersion() ?: return@withLock AutomaticInstallResult.DEFERRED
    val previous = store.read()
    if (previous?.status?.inProgress == true ||
      (previous != null && previous.versionCode >= version && previous.status != AutomaticInstallStatus.DEFERRED)) {
      return@withLock AutomaticInstallResult.ALREADY_ATTEMPTED
    }
    if (platform.sdkVersion < 31) {
      save(AutomaticInstallRecord(version, AutomaticInstallStatus.MANUAL_REQUIRED))
      return@withLock AutomaticInstallResult.MANUAL_REQUIRED
    }
    if (!platform.canInstall()) {
      save(AutomaticInstallRecord(version, AutomaticInstallStatus.NEEDS_PERMISSION))
      return@withLock AutomaticInstallResult.MANUAL_REQUIRED
    }
    var record = AutomaticInstallRecord(version, AutomaticInstallStatus.STAGING, token = UUID.randomUUID().toString(), startedAt = now())
    var sessionId = -1
    var commitRequested = false
    try {
      // Journal before verification/session allocation so process death cannot loop attempts.
      save(record)
      val apk = verifiedApk()
      if (!AutomaticInstallPolicy.supportsUnattended(platform.sdkVersion, platform.targetSdk(apk))) {
        save(record.copy(status = AutomaticInstallStatus.MANUAL_REQUIRED))
        return@withLock AutomaticInstallResult.MANUAL_REQUIRED
      }
      val coroutine = currentCoroutineContext()
      val checkEligible = {
        coroutine.ensureActive()
        if (!isEligible() || readyVersion() != version) throw InstallDeferred()
        if (!platform.canInstall()) throw InstallPermissionLost()
      }
      checkEligible()
      sessionId = platform.create(apk)
      record = record.copy(sessionId = sessionId)
      save(record)
      platform.stage(sessionId, apk, checkEligible)
      checkEligible()
      // A callback or process replacement can happen as soon as commit returns.
      save(record.copy(status = AutomaticInstallStatus.INSTALLING))
      checkEligible()
      commitRequested = true
      platform.commit(sessionId, record.token)
      AutomaticInstallResult.STARTED
    } catch (cancelled: CancellationException) {
      abandon(sessionId)
      save(record.copy(status = AutomaticInstallStatus.DEFERRED, sessionId = -1, token = ""))
      throw cancelled
    } catch (_: InstallDeferred) {
      abandon(sessionId)
      save(record.copy(status = AutomaticInstallStatus.DEFERRED, sessionId = -1, token = ""))
      AutomaticInstallResult.DEFERRED
    } catch (_: InstallPermissionLost) {
      abandon(sessionId)
      save(record.copy(status = AutomaticInstallStatus.NEEDS_PERMISSION, sessionId = -1, token = ""))
      AutomaticInstallResult.MANUAL_REQUIRED
    } catch (_: Exception) {
      val abandoned = abandonConfirmed(sessionId)
      // A binder error after commit may have reached Android. Do not enable a second
      // installer while that outcome is unknown and its session cannot be abandoned.
      save(if (commitRequested && !abandoned) record.copy(status = AutomaticInstallStatus.INSTALLING)
        else record.copy(status = AutomaticInstallStatus.FAILED, sessionId = -1, token = ""))
      AutomaticInstallResult.FAILED
    }
  }

  suspend fun markManualInstallRequested(): Boolean = mutex.withLock {
    val version = readyVersion() ?: return@withLock false
    val previous = store.read()
    if (previous?.status?.inProgress == true) return@withLock false
    save(AutomaticInstallRecord(maxOf(version, previous?.versionCode ?: 0), AutomaticInstallStatus.MANUAL_REQUIRED))
    true
  }

  suspend fun receive(sessionId: Int, token: String, result: AutomaticInstallCallback): Boolean = mutex.withLock {
    val record = store.read() ?: return@withLock false
    if (record.status != AutomaticInstallStatus.INSTALLING || record.sessionId != sessionId ||
      record.token.isBlank() || record.token != token) return@withLock false
    val next = when (result) {
      AutomaticInstallCallback.PENDING_USER_ACTION -> AutomaticInstallStatus.MANUAL_REQUIRED
      AutomaticInstallCallback.SUCCESS -> AutomaticInstallStatus.INSTALLED
      AutomaticInstallCallback.FAILURE -> AutomaticInstallStatus.FAILED
    }
    // Never launch or persist the supplied confirmation Intent. A fresh verified manual
    // install from Settings is the explicit retry and survives process death safely.
    if (result == AutomaticInstallCallback.PENDING_USER_ACTION && !abandonConfirmed(record.sessionId)) return@withLock false
    save(record.copy(status = next, token = ""))
    if (result == AutomaticInstallCallback.FAILURE) abandon(record.sessionId)
    true
  }

  suspend fun recover() = mutex.withLock {
    try {
      val record = store.read()
      val sessions = platform.sessions()
      // This app's only PackageInstaller sessions are self-update sessions. Clean up a
      // session allocated immediately before death prevented journaling its ID.
      sessions.filter { it != record?.sessionId }.forEach(::abandon)
      if (record == null) return@withLock
      when {
        platform.installedVersion() >= record.versionCode -> {
          abandon(record.sessionId)
          save(record.copy(status = AutomaticInstallStatus.INSTALLED, token = ""))
        }
        record.status == AutomaticInstallStatus.STAGING -> {
          abandon(record.sessionId)
          save(record.copy(status = AutomaticInstallStatus.FAILED, token = ""))
        }
        record.status == AutomaticInstallStatus.INSTALLING &&
          (record.sessionId !in sessions || now() - record.startedAt !in 0..SESSION_TIMEOUT_MS) -> {
          if (record.sessionId !in sessions || abandonConfirmed(record.sessionId)) {
            save(record.copy(status = AutomaticInstallStatus.MANUAL_REQUIRED, token = ""))
          }
        }
        !record.status.inProgress -> abandon(record.sessionId)
      }
    } catch (cancelled: CancellationException) {
      throw cancelled
    } catch (_: Exception) {
      // The next foreground can retry recovery. If a committed session might still
      // exist, keep manual installation disabled instead of masking it as a failure.
      if (!mutableStatus.value.inProgress) mutableStatus.value = AutomaticInstallStatus.FAILED
    }
  }

  private fun save(record: AutomaticInstallRecord) {
    store.write(record)
    mutableStatus.value = record.status
  }

  private fun abandon(sessionId: Int) {
    abandonConfirmed(sessionId)
  }

  private fun abandonConfirmed(sessionId: Int): Boolean = sessionId < 0 || runCatching { platform.abandon(sessionId) }.isSuccess

  private class InstallDeferred : Exception()
  private class InstallPermissionLost : Exception()
  private companion object { const val SESSION_TIMEOUT_MS = 30 * 60 * 1000L }
}
