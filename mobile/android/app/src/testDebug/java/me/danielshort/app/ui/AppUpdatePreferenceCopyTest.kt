package me.danielshort.app.ui

import me.danielshort.app.data.AppUpdateMode
import org.junit.Assert.*
import org.junit.Test

class AppUpdatePreferenceCopyTest {
  @Test fun olderAndroidExplainsManualInstallAfterAutomaticDownload() {
    val choices = appUpdateChoicesForSdk(30)
    val automatic = choices.first { it.value == AppUpdateMode.AUTOMATIC }
    assertTrue(automatic.description.contains("Download updates automatically"))
    assertTrue(automatic.description.contains("tap Install update"))
    assertFalse(shouldOfferAutomaticInstallPermission(true, false, 30))
  }

  @Test fun supportedAndroidExplainsIdleInstallationAndCanOfferPermission() {
    val automatic = appUpdateChoicesForSdk(31).first { it.value == AppUpdateMode.AUTOMATIC }
    assertTrue(automatic.description.contains("install when the app is idle"))
    assertTrue(shouldOfferAutomaticInstallPermission(true, false, 31))
    assertFalse(shouldOfferAutomaticInstallPermission(false, false, 31))
    assertFalse(shouldOfferAutomaticInstallPermission(true, true, 31))
  }
}
