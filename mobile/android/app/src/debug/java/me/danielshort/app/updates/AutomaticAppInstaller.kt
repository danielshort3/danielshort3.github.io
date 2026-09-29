package me.danielshort.app.updates

import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.net.Uri
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.security.MessageDigest

/** Optional self-updates; Android still decides whether user confirmation is necessary. */
class AutomaticAppInstaller(context: Context, manager: AppUpdateManager) {
  private val application = context.applicationContext
  private val engine = AutomaticInstallEngine(
    AndroidAutomaticInstallStore(application),
    AndroidAutomaticInstallPlatform(application),
    readyVersion = { (manager.state.value as? AppUpdateState.Ready)?.offer?.versionCode },
    verifiedApk = { manager.verifiedApkForInstall() },
  )
  val status = engine.status

  /** The live predicate must include opt-in, foreground state, and protected work. */
  suspend fun attemptIfEligible(isEligible: () -> Boolean): AutomaticInstallResult = withContext(Dispatchers.IO) {
    engine.attemptIfEligible(isEligible)
  }

  suspend fun recover() = withContext(Dispatchers.IO) { engine.recover() }

  /** Persist a deliberate manual handoff before opening Android, including permission UI. */
  suspend fun markManualInstallRequested(): Boolean = withContext(Dispatchers.IO) { engine.markManualInstallRequested() }

  internal suspend fun receive(intent: Intent) = withContext(Dispatchers.IO) {
    if (intent.action != callbackAction(application)) return@withContext
    val uri = intent.data ?: return@withContext
    if (uri.scheme != CALLBACK_SCHEME) return@withContext
    val packageName = intent.getStringExtra(PackageInstaller.EXTRA_PACKAGE_NAME)
    if (packageName != null && packageName != application.packageName) return@withContext
    val result = when (intent.getIntExtra(PackageInstaller.EXTRA_STATUS, Int.MIN_VALUE)) {
      PackageInstaller.STATUS_PENDING_USER_ACTION -> AutomaticInstallCallback.PENDING_USER_ACTION
      PackageInstaller.STATUS_SUCCESS -> AutomaticInstallCallback.SUCCESS
      PackageInstaller.STATUS_FAILURE, PackageInstaller.STATUS_FAILURE_ABORTED,
      PackageInstaller.STATUS_FAILURE_BLOCKED, PackageInstaller.STATUS_FAILURE_CONFLICT,
      PackageInstaller.STATUS_FAILURE_INCOMPATIBLE, PackageInstaller.STATUS_FAILURE_INVALID,
      PackageInstaller.STATUS_FAILURE_STORAGE, PackageInstaller.STATUS_FAILURE_TIMEOUT -> AutomaticInstallCallback.FAILURE
      else -> return@withContext
    }
    engine.receive(intent.getIntExtra(PackageInstaller.EXTRA_SESSION_ID, -1), uri.schemeSpecificPart, result)
  }

  internal companion object {
    const val CALLBACK_SCHEME = "daniel-short-update"
    fun callbackAction(context: Context) = "${context.packageName}.AUTOMATIC_UPDATE_STATUS"
  }
}

internal class AndroidAutomaticInstallStore(context: Context, name: String = "automatic-app-installer-v1") : AutomaticInstallStore {
  private val preferences = context.getSharedPreferences(name, Context.MODE_PRIVATE)

  override fun read(): AutomaticInstallRecord? {
    val version = preferences.getLong("version", 0)
    if (version <= 0) return null
    val status = runCatching { AutomaticInstallStatus.valueOf(preferences.getString("status", "FAILED")!!) }
      .getOrDefault(AutomaticInstallStatus.FAILED)
    return AutomaticInstallRecord(version, status, preferences.getInt("session", -1),
      preferences.getString("token", "").orEmpty(), preferences.getLong("started", 0))
  }

  override fun write(record: AutomaticInstallRecord) {
    check(preferences.edit().putLong("version", record.versionCode).putString("status", record.status.name)
      .putInt("session", record.sessionId).putString("token", record.token).putLong("started", record.startedAt).commit()) {
      "Could not persist the update installation state."
    }
  }
}

internal class AndroidAutomaticInstallPlatform(private val context: Context) : AutomaticInstallPlatform {
  private val installer get() = context.packageManager.packageInstaller
  override val sdkVersion get() = Build.VERSION.SDK_INT
  override fun canInstall() = context.packageManager.canRequestPackageInstalls()
  override fun installedVersion(): Long {
    val info = context.packageManager.getPackageInfo(context.packageName, 0)
    return if (Build.VERSION.SDK_INT >= 28) info.longVersionCode else @Suppress("DEPRECATION") info.versionCode.toLong()
  }
  override fun targetSdk(apk: File) = context.packageManager.getPackageArchiveInfo(apk.absolutePath, 0)?.applicationInfo?.targetSdkVersion ?: 0
  override fun sessions() = installer.mySessions.filter { it.appPackageName == context.packageName }.map { it.sessionId }.toSet()

  override fun create(apk: File): Int {
    if (Build.VERSION.SDK_INT < 31) throw IllegalStateException("Automatic installation requires Android 12.")
    val parameters = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
      setAppPackageName(context.packageName)
      setSize(apk.length())
      setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_NOT_REQUIRED)
      if (Build.VERSION.SDK_INT >= 33) setPackageSource(PackageInstaller.PACKAGE_SOURCE_DOWNLOADED_FILE)
    }
    return installer.createSession(parameters)
  }

  override fun stage(sessionId: Int, apk: File, checkEligible: () -> Unit) {
    val expectedHash = Regex("[1-9][0-9]*-([0-9a-f]{64})\\.apk").matchEntire(apk.name)?.groupValues?.get(1)
      ?: throw UpdateFailure("The verified update filename is invalid.")
    val expectedSize = apk.length()
    val digest = MessageDigest.getInstance("SHA-256")
    var copied = 0L
    installer.openSession(sessionId).use { session ->
      session.openWrite("base.apk", 0, expectedSize).use { output ->
        apk.inputStream().use { input ->
          val buffer = ByteArray(64 * 1024)
          while (true) {
            checkEligible()
            val count = input.read(buffer)
            if (count < 0) break
            copied += count
            if (copied > expectedSize) throw UpdateFailure("The verified update changed while preparing installation.")
            digest.update(buffer, 0, count)
            output.write(buffer, 0, count)
          }
        }
        if (copied != expectedSize || digest.digest().hex() != expectedHash) {
          throw UpdateFailure("The verified update changed while preparing installation.")
        }
        session.fsync(output)
      }
    }
  }

  override fun commit(sessionId: Int, token: String) {
    // PackageInstaller must fill in result extras, so Android 12+ requires a mutable
    // sender. Its private explicit component and random URI prevent outside callbacks.
    val intent = Intent(context, AutomaticInstallReceiver::class.java)
      .setAction(AutomaticAppInstaller.callbackAction(context))
      .setData(Uri.fromParts(AutomaticAppInstaller.CALLBACK_SCHEME, token, null))
    val flags = PendingIntent.FLAG_UPDATE_CURRENT or if (Build.VERSION.SDK_INT >= 31) PendingIntent.FLAG_MUTABLE else 0
    val callback = PendingIntent.getBroadcast(context, sessionId, intent, flags)
    try {
      installer.openSession(sessionId).use { it.commit(callback.intentSender) }
    } catch (error: Exception) {
      callback.cancel()
      throw error
    }
  }

  override fun abandon(sessionId: Int) { installer.abandonSession(sessionId) }
}
