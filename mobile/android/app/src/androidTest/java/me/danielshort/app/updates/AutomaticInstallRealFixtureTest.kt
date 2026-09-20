package me.danielshort.app.updates

import android.accessibilityservice.AccessibilityService
import android.os.Build
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.lifecycle.ActivityLifecycleMonitorRegistry
import androidx.test.runner.lifecycle.Stage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Destructive opt-in QA entry point for a disposable device only. Normal suites skip it.
 * A successful self-update can terminate instrumentation; the external harness MUST verify
 * installed version/hash, persisted settings, and relaunch. Process death is not a test pass.
 */
class AutomaticInstallRealFixtureTest {
  @Test fun commitVerifiedSelfUpdateOnlyWithExplicitDeviceFixture(): Unit = runBlocking {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    val path = InstrumentationRegistry.getArguments().getString("automaticInstallFixtureDirectory")
    assumeTrue("Only run when a disposable-device self-update was explicitly requested", !path.isNullOrBlank())
    assumeTrue("Unattended self-update requires Android 12+", Build.VERSION.SDK_INT >= 31)
    assertTrue("The user must enable Install unknown apps through Android before this test", context.packageManager.canRequestPackageInstalls())
    val fixture = File(path!!)
    val manifestText = File(fixture, "manifest.json").readText()
    val manifest = UpdateManifestParser.parse(manifestText)
    val verifier = AndroidInstalledAppVerifier(context)
    val installed = verifier.installed()
    assertTrue("Fixture version must strictly increase", manifest.latest.versionCode > installed.versionCode)
    UpdatePolicy.recognizeInstalled(manifest, installed)
    val transport = object : UpdateTransport {
      override suspend fun manifest(url: String) = manifestText
      override suspend fun download(artifact: UpdateArtifact, destination: File, progress: (Long, Long) -> Unit) {
        File(fixture, if (artifact.url == manifest.latest.apk.url) "target.apk" else "delta.patch.gz").copyTo(destination, overwrite = true)
        progress(destination.length(), artifact.size)
      }
    }
    val supervisor = SupervisorJob()
    val manager = AppUpdateManager(File(context.filesDir, "automatic-install-real-fixture"),
      "https://www.danielshort.me/app-updates/review/latest.json", verifier, transport, CoroutineScope(supervisor + Dispatchers.IO))
    try {
      manager.check()
      assertTrue(withTimeout(60_000) { manager.state.first { it is AppUpdateState.Available || it is AppUpdateState.Error } } is AppUpdateState.Available)
      manager.download()
      assertTrue(withTimeout(120_000) { manager.state.first { it is AppUpdateState.Ready || it is AppUpdateState.Error } } is AppUpdateState.Ready)
      instrumentation.uiAutomation.performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME)
      delay(30_000)
      val eligible = {
        var resumed = true
        instrumentation.runOnMainSync {
          resumed = ActivityLifecycleMonitorRegistry.getInstance().getActivitiesInStage(Stage.RESUMED).isNotEmpty()
        }
        !resumed
      }
      assertTrue("No app activity may remain resumed", eligible())
      File(fixture, "automatic-install-requested.txt").writeText("version=${manifest.latest.versionCode}\nsha256=${manifest.latest.apk.sha256}\n")
      val installer = AutomaticAppInstaller(context, manager)
      installer.recover()
      val result = installer.attemptIfEligible(eligible)
      File(fixture, "automatic-install-returned.txt").writeText("result=$result\nstatus=${installer.status.value}\n")
      assertEquals("External verification is still required after commit", AutomaticInstallResult.STARTED, result)
      // Allow callback/fallback delivery if Android did not replace this process immediately.
      delay(5_000)
    } finally {
      supervisor.cancelAndJoin()
    }
  }
}
