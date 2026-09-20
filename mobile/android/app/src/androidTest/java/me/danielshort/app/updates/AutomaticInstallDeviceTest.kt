package me.danielshort.app.updates

import android.content.ComponentName
import android.content.pm.PackageManager
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.UUID

/** Uses Android persistence with an injected installer; never commits a real install. */
class AutomaticInstallDeviceTest {
  private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

  @Test fun privateReceiverIsNotExportedAndUnattendedPermissionIsDeclared() {
    val receiver = context.packageManager.getReceiverInfo(ComponentName(context, AutomaticInstallReceiver::class.java), 0)
    assertFalse(receiver.exported)
    val permissions = context.packageManager.getPackageInfo(context.packageName, PackageManager.GET_PERMISSIONS).requestedPermissions.orEmpty()
    assertTrue(permissions.contains("android.permission.UPDATE_PACKAGES_WITHOUT_USER_ACTION"))
    assertTrue(permissions.contains("android.permission.REQUEST_INSTALL_PACKAGES"))
  }

  @Test fun confirmationAndCancellationStaySuppressedWithFreshAndroidStoreInstances() = runBlocking {
    for (result in listOf(AutomaticInstallCallback.PENDING_USER_ACTION, AutomaticInstallCallback.FAILURE)) {
      val name = "automatic-installer-device-fixture-${UUID.randomUUID()}"
      val active = mutableSetOf<Int>()
      var commits = 0
      val platform = object : AutomaticInstallPlatform {
        override val sdkVersion = 36
        override fun canInstall() = true
        override fun installedVersion() = 4L
        override fun targetSdk(apk: File) = 36
        override fun sessions() = active.toSet()
        override fun create(apk: File) = 42.also { active.add(it) }
        override fun stage(sessionId: Int, apk: File, checkEligible: () -> Unit) { checkEligible() }
        override fun commit(sessionId: Int, token: String) { commits++ }
        override fun abandon(sessionId: Int) { active.remove(sessionId) }
      }
      fun engine() = AutomaticInstallEngine(AndroidAutomaticInstallStore(context, name), platform, { 5L }, { File("fixture.apk") })
      try {
        val first = engine()
        assertEquals(AutomaticInstallResult.STARTED, first.attemptIfEligible { true })
        val record = AndroidAutomaticInstallStore(context, name).read()!!
        assertFalse(first.receive(record.sessionId, "forged", AutomaticInstallCallback.SUCCESS))
        // A fresh engine models callback delivery after the original process disappeared.
        assertTrue(engine().receive(record.sessionId, record.token, result))
        val restarted = engine()
        restarted.recover()
        val expected = if (result == AutomaticInstallCallback.PENDING_USER_ACTION) AutomaticInstallStatus.MANUAL_REQUIRED else AutomaticInstallStatus.FAILED
        assertEquals(expected, restarted.status.value)
        assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, restarted.attemptIfEligible { true })
        assertTrue(active.isEmpty())
        assertEquals(1, commits)
      } finally {
        context.deleteSharedPreferences(name)
      }
    }
  }
}
