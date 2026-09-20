package me.danielshort.app.updates

import android.content.Context
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.os.Build
import com.android.apksig.ApkVerifier
import java.io.File
import java.security.MessageDigest

data class VerifiedApk(
  val file: File,
  val packageName: String,
  val versionCode: Long,
  val versionName: String,
  val sha256: String,
  val size: Long,
  val signerSha256: String,
  val minSdk: Int,
)

interface InstalledAppVerifier {
  val sdkVersion: Int
  fun installed(checkCancelled: () -> Unit = {}): VerifiedApk
  fun archive(file: File, checkCancelled: () -> Unit = {}): VerifiedApk
}

class AndroidInstalledAppVerifier(context: Context) : InstalledAppVerifier {
  private val application = context.applicationContext
  override val sdkVersion = Build.VERSION.SDK_INT

  override fun installed(checkCancelled: () -> Unit): VerifiedApk {
    val info = application.packageManager.getPackageInfo(application.packageName, signatureFlags())
    val appInfo = info.applicationInfo ?: throw UpdateFailure("The installed app could not be verified.")
    if (!appInfo.splitSourceDirs.isNullOrEmpty() || !info.splitNames.isNullOrEmpty()) {
      throw UpdateFailure("This installation uses split APKs, which this updater does not support. Update it through its original installer.")
    }
    return inspect(File(appInfo.sourceDir), info, checkCancelled)
  }

  override fun archive(file: File, checkCancelled: () -> Unit): VerifiedApk {
    val info = application.packageManager.getPackageArchiveInfo(file.absolutePath, signatureFlags())
      ?: throw UpdateFailure("The downloaded app package could not be read.")
    if (!info.splitNames.isNullOrEmpty()) throw UpdateFailure("This update is not a supported single APK.")
    return inspect(file, info, checkCancelled)
  }

  private fun inspect(file: File, info: PackageInfo, checkCancelled: () -> Unit): VerifiedApk {
    checkCancelled()
    if (!file.isFile || file.length() !in 1..MAX_UPDATE_BYTES) throw UpdateFailure("The app package is missing or has an unsupported size.")
    val verified = try {
      ApkVerifier.Builder(file).setMinCheckedPlatformVersion(sdkVersion).setMaxCheckedPlatformVersion(sdkVersion).build().verify()
    } catch (failure: Exception) {
      throw UpdateFailure("The app's signing certificate could not be verified. No update was applied.", failure)
    }
    checkCancelled()
    if (!verified.isVerified || verified.signerCertificates.size != 1) throw UpdateFailure("The app's signature could not be verified. No update was applied.")
    val signer = MessageDigest.getInstance("SHA-256").digest(verified.signerCertificates.single().encoded).hex()
    val packageSigners = if (Build.VERSION.SDK_INT >= 28) info.signingInfo?.apkContentsSigners?.toList().orEmpty()
      else @Suppress("DEPRECATION") (info.signatures?.toList().orEmpty())
    if (packageSigners.size != 1 || MessageDigest.getInstance("SHA-256").digest(packageSigners.single().toByteArray()).hex() != signer) {
      throw UpdateFailure("The app's signing identity does not match Android's package information.")
    }
    val version = if (Build.VERSION.SDK_INT >= 28) info.longVersionCode else @Suppress("DEPRECATION") info.versionCode.toLong()
    return VerifiedApk(file, info.packageName, version, info.versionName.orEmpty(), sha256(file, checkCancelled), file.length(), signer, info.applicationInfo?.minSdkVersion ?: 1)
  }

  @Suppress("DEPRECATION")
  private fun signatureFlags() = if (Build.VERSION.SDK_INT >= 28) PackageManager.GET_SIGNING_CERTIFICATES else PackageManager.GET_SIGNATURES
}

object UpdatePolicy {
  fun recognizeInstalled(manifest: UpdateManifest, installed: VerifiedApk) {
    if (manifest.packageName != installed.packageName) throw UpdateFailure("The update feed belongs to a different app. No update was applied.")
    if (manifest.releases.none { it.versionCode == installed.versionCode && it.sha256 == installed.sha256 && it.size == installed.size && it.signerSha256 == installed.signerSha256 }) {
      throw UpdateFailure("This installed build is not in the published release list, so it cannot be safely patched. It may be a local or older build. Install an official release to enable updates.")
    }
    if (manifest.latest.signerSha256 != installed.signerSha256) {
      throw UpdateFailure("This update uses a different signing identity. Certificate rotation is not supported by this updater.")
    }
  }

  fun verifyTarget(target: VerifiedApk, installed: VerifiedApk, release: LatestRelease, sdkVersion: Int) {
    if (target.packageName != installed.packageName || target.versionCode != release.versionCode || target.versionCode <= installed.versionCode || target.versionName != release.versionName) {
      throw UpdateFailure("The downloaded app does not match the expected app and newer version.")
    }
    if (target.signerSha256 != installed.signerSha256 || target.signerSha256 != release.signerSha256) throw UpdateFailure("The update was not signed by the installed app's signing identity.")
    if (target.sha256 != release.apk.sha256 || target.size != release.apk.size) throw UpdateFailure("The downloaded app did not match the published release.")
    if (release.minSdk > sdkVersion || target.minSdk > sdkVersion || target.minSdk != release.minSdk) throw UpdateFailure("This update requires a newer Android version.")
  }

  fun sameInstalled(first: VerifiedApk, current: VerifiedApk) {
    if (first.packageName != current.packageName || first.versionCode != current.versionCode || first.sha256 != current.sha256 || first.size != current.size || first.signerSha256 != current.signerSha256) {
      throw UpdateFailure("The installed app changed while preparing this update. Check for updates again.")
    }
  }
}
