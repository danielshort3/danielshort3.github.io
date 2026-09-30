package me.danielshort.app.ui

import java.net.URI
import me.danielshort.app.data.CatalogItem

private val FIRST_PARTY_WEBSITE_HOSTS = setOf("www.danielshort.me", "danielshort.me", "www.dshort.me", "dshort.me")
private val WEBSITE_ACCOUNT_TOOL_SLUGS = setOf(
  "tools", "tools-dashboard", "dashboard", "text-compare", "nbsp-cleaner", "word-frequency",
  "point-of-view-checker", "oxford-comma-checker", "utm-batch-builder", "qr-code-generator",
  "image-optimizer", "background-remover", "screen-recorder", "short-links", "transcribe",
  "whisper-transcribe-monitor", "job-application-tracker", "campaign-creative-tracker", "ga4-utm-performance"
)

/** Play exposes tools with a complete native guest experience, including future feed revisions. */
internal fun visibleToolEntries(entries: List<CatalogItem>, nativeIds: Set<String>, enableWebsiteAccounts: Boolean): List<CatalogItem> =
  if (enableWebsiteAccounts) entries else entries.filter { it.id in nativeIds }

/** Guest-only builds cannot launch account tools through catalog, project, or WebView links. */
internal fun websiteAccountLinkAllowed(candidate: String, enableWebsiteAccounts: Boolean): Boolean {
  if (enableWebsiteAccounts) return true
  val uri = runCatching { URI(candidate).normalize() }.getOrNull() ?: return false
  val host = uri.host?.lowercase().orEmpty()
  if (host.endsWith(".amazoncognito.com")) return false
  if (host !in FIRST_PARTY_WEBSITE_HOSTS) return true
  val path = URI(null, null, uri.path.orEmpty(), null).normalize().path.replace(Regex("/{2,}"), "/").removePrefix("/pages")
    .trimEnd('/').removeSuffix(".html")
  return path != "/tools" && !path.startsWith("/tools/") &&
    path != "/api/tools" && !path.startsWith("/api/tools/") && path.removePrefix("/") !in WEBSITE_ACCOUNT_TOOL_SLUGS
}
