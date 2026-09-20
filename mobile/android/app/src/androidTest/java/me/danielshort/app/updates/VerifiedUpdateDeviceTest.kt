package me.danielshort.app.updates

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile

@RunWith(AndroidJUnit4::class)
class VerifiedUpdateDeviceTest {
  private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

  @Test fun verifiesActualInstalledApkAndRejectsModifiedCopy() {
    val verifier = AndroidInstalledAppVerifier(context)
    val installed = verifier.installed()
    assertEquals(context.packageName, installed.packageName)
    assertEquals(64, installed.sha256.length)
    assertEquals(64, installed.signerSha256.length)
    val copy = File(context.cacheDir, "signature-check.apk")
    try {
      installed.file.copyTo(copy, overwrite = true)
      val archive = verifier.archive(copy)
      assertEquals(installed.sha256, archive.sha256)
      assertEquals(installed.signerSha256, archive.signerSha256)
      RandomAccessFile(copy, "rw").use { file ->
        val position = file.length() / 2
        file.seek(position)
        val original = file.readByte().toInt()
        file.seek(position)
        file.writeByte(original xor 1)
      }
      assertThrows(UpdateFailure::class.java) { verifier.archive(copy) }
    } finally {
      copy.delete()
    }
  }

  /** Optional real v2->v3 Node-built fixture; never sends an install intent or downloads from the network. */
  @Test fun reconstructsAndVerifiesPublishedApkPatchOnAndroid() = runBlocking {
    val fixturePath = InstrumentationRegistry.getArguments().getString("updateFixtureDirectory")
    assumeTrue("Provide the local real-APK patch fixture to exercise the complete update pipeline", !fixturePath.isNullOrBlank())
    val fixture = File(fixturePath!!)
    val base = File(fixture, "base.apk")
    val target = File(fixture, "target.apk")
    val patch = File(fixture, "delta.patch.gz")
    val manifestText = File(fixture, "manifest.json").readText()
    val manifest = UpdateManifestParser.parse(manifestText)
    val androidVerifier = AndroidInstalledAppVerifier(context)
    val verifiedBase = androidVerifier.archive(base)
    val verifiedTarget = androidVerifier.archive(target)
    assertTrue(verifiedTarget.versionCode > verifiedBase.versionCode)
    assertEquals(verifiedBase.signerSha256, verifiedTarget.signerSha256)
    val verifier = object : InstalledAppVerifier {
      override val sdkVersion = androidVerifier.sdkVersion
      override fun installed(checkCancelled: () -> Unit) = androidVerifier.archive(base, checkCancelled)
      override fun archive(file: File, checkCancelled: () -> Unit) = androidVerifier.archive(file, checkCancelled)
    }
    val downloads = mutableListOf<String>()
    val transport = object : UpdateTransport {
      override suspend fun manifest(url: String) = manifestText
      override suspend fun download(artifact: UpdateArtifact, destination: File, progress: (Long, Long) -> Unit) {
        downloads += artifact.url
        val source = if (artifact.url == manifest.latest.apk.url) target else patch
        source.copyTo(destination, overwrite = true)
        progress(destination.length(), artifact.size)
      }
    }
    val job = SupervisorJob()
    val storage = File(context.cacheDir, "verified-update-fixture")
    try {
      val manager = AppUpdateManager(storage, "https://www.danielshort.me/app-updates/review/latest.json", verifier, transport, CoroutineScope(job + Dispatchers.IO))
      manager.check()
      val offered = withTimeout(30_000) { manager.state.first { it is AppUpdateState.Available || it is AppUpdateState.Error } }
      assertTrue(offered.toString(), offered is AppUpdateState.Available)
      assertTrue((offered as AppUpdateState.Available).offer.usingPatch)
      manager.download()
      val prepared = withTimeout(90_000) { manager.state.first { it is AppUpdateState.Ready || it is AppUpdateState.Error } }
      assertTrue(prepared.toString(), prepared is AppUpdateState.Ready)
      val installable = manager.verifiedApkForInstall()
      assertEquals(verifiedTarget.sha256, sha256(installable))
      assertEquals(1, downloads.size)
      assertEquals(manifest.patches.single { it.fromSha256 == verifiedBase.sha256 }.artifact.url, downloads.single())
    } finally {
      job.cancelAndJoin()
      storage.deleteRecursively()
    }
  }
}
