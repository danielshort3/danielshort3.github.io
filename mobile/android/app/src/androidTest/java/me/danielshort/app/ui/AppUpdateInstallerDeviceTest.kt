package me.danielshort.app.ui

import android.os.ParcelFileDescriptor
import android.view.KeyEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.performClick
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import me.danielshort.app.updates.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Exercises the real Android permission/installer boundary, but never confirms an installation. */
class AppUpdateInstallerDeviceTest {
  @get:Rule val compose = createComposeRule()
  private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
  private val context get() = instrumentation.targetContext

  @Test fun permissionDeclineAndInstallerCancelKeepVerifiedUpdateReady(): Unit = runBlocking {
    val fixturePath = InstrumentationRegistry.getArguments().getString("updateFixtureDirectory")
    assumeTrue("Requires the local real-APK update fixture", !fixturePath.isNullOrBlank())
    val fixture = File(fixturePath!!)
    val manifestText = File(fixture, "manifest.json").readText()
    val manifest = UpdateManifestParser.parse(manifestText)
    val androidVerifier = AndroidInstalledAppVerifier(context)
    val verifier = object : InstalledAppVerifier {
      override val sdkVersion = androidVerifier.sdkVersion
      override fun installed(checkCancelled: () -> Unit) = androidVerifier.archive(File(fixture, "base.apk"), checkCancelled)
      override fun archive(file: File, checkCancelled: () -> Unit) = androidVerifier.archive(file, checkCancelled)
    }
    val transport = object : UpdateTransport {
      override suspend fun manifest(url: String) = manifestText
      override suspend fun download(artifact: UpdateArtifact, destination: File, progress: (Long, Long) -> Unit) {
        File(fixture, if (artifact.url == manifest.latest.apk.url) "target.apk" else "delta.patch.gz").copyTo(destination, overwrite = true)
        progress(destination.length(), artifact.size)
      }
    }
    assumeTrue("Run with installation permission disabled; restore it from adb after the test", !context.packageManager.canRequestPackageInstalls())
    val job = SupervisorJob()
    // This directory must be under the production provider's narrow ready path.
    val storage = File(context.filesDir, "app-updates/ready/installer-test")
    val manager = AppUpdateManager(storage, "https://www.danielshort.me/app-updates/review/latest.json", verifier, transport, CoroutineScope(job + Dispatchers.IO))
    try {
      manager.check()
      assertTrue(withTimeout(30_000) { manager.state.first { it is AppUpdateState.Available || it is AppUpdateState.Error } } is AppUpdateState.Available)
      manager.download()
      assertTrue(withTimeout(90_000) { manager.state.first { it is AppUpdateState.Ready || it is AppUpdateState.Error } } is AppUpdateState.Ready)
      compose.setContent { MaterialTheme { AppUpdateSection(manager) } }
      compose.onNodeWithText("Install update").performClick()
      awaitSystemWindow { it.packageName?.toString() == "com.android.settings" }
      instrumentation.sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
      compose.waitUntil(10_000) { compose.onAllNodesWithText("Installation permission wasn’t enabled. Your verified update is still ready.").fetchSemanticsNodes().isNotEmpty() }
      assertTrue(manager.state.value is AppUpdateState.Ready)
      setInstallPermission(true)
      compose.onNodeWithText("Install update").performClick()
      val installer = awaitSystemWindow { root ->
        root.packageName?.toString()?.contains("packageinstaller") == true &&
          findNode(root) { it.text?.toString().let { text -> text.equals("Update", true) || text.equals("Install", true) } } != null
      }
      val cancel = findNode(installer) { it.text?.toString().equals("Cancel", ignoreCase = true) }
      assertNotNull("The Android installation confirmation must offer Cancel", cancel)
      instrumentation.uiAutomation.waitForIdle(300, 5_000)
      val screenshot = instrumentation.uiAutomation.takeScreenshot()
      File(fixture, "installer-confirmation.png").outputStream().use { screenshot.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
      assertTrue(cancel!!.performAction(AccessibilityNodeInfo.ACTION_CLICK))
      compose.waitUntil(10_000) { compose.onAllNodesWithText("If installation wasn’t completed, you can try again.").fetchSemanticsNodes().isNotEmpty() }
      assertTrue(manager.state.value is AppUpdateState.Ready)
      compose.onNodeWithText("Install update").assertExists()
    } finally {
      job.cancelAndJoin()
      // Revoking this permission terminates the app on Android. The adb harness
      // restores it after instrumentation has finished, so teardown can complete.
      storage.deleteRecursively()
    }
  }

  private fun setInstallPermission(allowed: Boolean) {
    ParcelFileDescriptor.AutoCloseInputStream(instrumentation.uiAutomation.executeShellCommand("appops set ${context.packageName} REQUEST_INSTALL_PACKAGES ${if (allowed) "allow" else "deny"}")).use { it.readBytes() }
  }

  private fun awaitSystemWindow(predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo {
    val deadline = android.os.SystemClock.uptimeMillis() + 20_000
    while (android.os.SystemClock.uptimeMillis() < deadline) {
      instrumentation.uiAutomation.rootInActiveWindow?.let { if (predicate(it)) return it }
      Thread.sleep(100)
    }
    throw AssertionError("Expected Android system screen did not appear")
  }

  private fun findNode(root: AccessibilityNodeInfo, predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
    if (predicate(root)) return root
    for (index in 0 until root.childCount) root.getChild(index)?.let { child -> findNode(child, predicate)?.let { return it } }
    return null
  }
}
