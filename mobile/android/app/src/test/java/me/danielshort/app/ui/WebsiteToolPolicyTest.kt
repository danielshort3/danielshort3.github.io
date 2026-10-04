package me.danielshort.app.ui

import me.danielshort.app.BuildConfig
import me.danielshort.app.data.CatalogItem
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WebsiteToolPolicyTest {
  private fun entry(id: String) = CatalogItem(id, id, "", "", "https://www.danielshort.me/tools/$id", "Tools")

  @Test fun guestToolsExcludeBrowserOnlyEntriesFromNewFeedRevisions() {
    val nativeIds = setOf("text-compare", "screen-recorder", "qr-code-generator")
    val entries = listOf(entry("text-compare"), entry("job-application-tracker"), entry("screen-recorder"), entry("future-account-tool"), entry("qr-code-generator"))
    assertEquals(listOf("text-compare", "screen-recorder", "qr-code-generator"),
      visibleToolEntries(entries, nativeIds, false).map { it.id })
    assertEquals(entries, visibleToolEntries(entries, nativeIds, true))
  }

  @Test fun guestLinksRejectCatalogAndLegacyWebsiteAccountRoutes() {
    val paths = listOf("/tools", "/tools/", "/tools.html", "/tools/text-compare", "/tools/future-tool",
      "/tools/dashboard.html", "/pages/tools-dashboard.html", "/pages/background-remover.html",
      "/word-frequency.html", "/nbsp-cleaner", "/oxford-comma-checker", "/short-links",
      "/api/tools/state", "/portfolio/../tools/qr-code-generator", "/portfolio/%2e%2e/%74ools/transcribe")
    listOf("www.danielshort.me", "danielshort.me", "www.dshort.me", "dshort.me").forEach { host ->
      paths.forEach { path ->
        val url = "https://$host$path?state=review#account"
        assertFalse(url, websiteAccountLinkAllowed(url, false))
        assertTrue(url, websiteAccountLinkAllowed(url, true))
      }
    }
    assertFalse(websiteAccountLinkAllowed("https://job-tracker-auth-886623862678.auth.us-east-2.amazoncognito.com/signup", false))
    assertTrue(websiteAccountLinkAllowed("https://job-tracker-auth-886623862678.auth.us-east-2.amazoncognito.com/signup", true))
  }

  @Test fun guestLinksPreservePublicExperiencesAndContactResources() {
    listOf("https://www.danielshort.me/portfolio/covidAnalysis", "https://www.danielshort.me/games/project-starfall",
      "https://www.danielshort.me/covid-outbreak-demo.html", "https://www.danielshort.me/privacy#android-app",
      "https://www.danielshort.me/documents/brand_guide.pdf", "https://github.com/danielshort3",
      "mailto:daniel@danielshort.me", "tel:+15555550100").forEach {
      assertTrue(it, websiteAccountLinkAllowed(it, false))
    }
    assertFalse(websiteAccountLinkAllowed("https://[broken", false))
  }

  @Test fun websiteAccountCapabilityMatchesBuildVariant() {
    assertEquals(BuildConfig.BUILD_TYPE == "debug", BuildConfig.ENABLE_WEBSITE_TOOL_ACCOUNTS)
  }
}
