package me.danielshort.wayfarers

import java.io.File
import java.net.URI
import me.danielshort.app.BuildConfig
import me.danielshort.app.updates.*
import org.junit.Assert.*
import org.junit.Test

class GuildUpdatePolicyTest {
  private val feed = "https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-updates/latest.json"

  @Test fun dedicatedFeedIsPinnedWithoutBreakingTheExistingMainAppFeed() {
    assertEquals(feed, BuildConfig.APP_UPDATE_URL)
    assertEquals(URI(feed), UpdateUrlPolicy.requireFeedUrl(feed))
    UpdateUrlPolicy.requireFeedUrl("https://www.danielshort.me/app-updates/review/latest.json")
  }

  @Test fun feedExceptionDoesNotPermitOtherGithubDocuments() {
    listOf(
      feed + "?download=1", feed + "#fragment", feed.replace("https:", "http:"),
      feed.replace("github.com/", "github.com:443/"), feed.replace("github.com/", "user@github.com/"),
      feed.replace("latest.json", "other.json"), feed.replace("wayfarers-guild-updates", "another-tag"),
      feed.replace("danielshort3/danielshort3.github.io", "another-owner/project"),
      feed.replace("github.com", "github.com.evil.test")
    ).forEach { assertThrows(it, IllegalArgumentException::class.java) { UpdateUrlPolicy.requireFeedUrl(it) } }
  }

  @Test fun signedCdnRedirectIsAllowedOnlyAfterThePinnedFeed() {
    val cdn = "https://release-assets.githubusercontent.com/github-production-release-asset/1/file?signature=opaque"
    assertEquals(URI(cdn), UpdateUrlPolicy.requireRedirect(URI(feed), cdn, URI(feed), true))
    assertThrows(IllegalArgumentException::class.java) { UpdateUrlPolicy.requireFeedUrl(cdn) }
    val sibling = URI(feed.replace("latest.json", "other.json"))
    assertThrows(IllegalArgumentException::class.java) { UpdateUrlPolicy.requireRedirect(sibling, cdn, sibling, true) }
    listOf("http://release-assets.githubusercontent.com/file", "https://evil.test/file", "https://user@release-assets.githubusercontent.com/file")
      .forEach { assertThrows(it, IllegalArgumentException::class.java) { UpdateUrlPolicy.requireRedirect(URI(feed), it, URI(feed), true) } }
  }

  @Test fun mainAppAndDedicatedGameCannotUpgradeEachOtherEvenWithTheSameSigner() {
    val signer = "ab".repeat(32)
    val baseHash = "cd".repeat(32)
    val installed = VerifiedApk(File("baseline.apk"), "me.danielshort.wayfarers", 1, "0.1.0", baseHash, 1234, signer, 26)
    val target = LatestRelease(2, "0.1.1", 26, UpdateArtifact(feed.replace("latest.json", "candidate.apk"), "ef".repeat(32), 2345), signer)
    val releases = listOf(PublishedRelease(1, baseHash, 1234, signer))
    val manifest = UpdateManifest("me.danielshort.wayfarers", target, releases, emptyList())
    UpdatePolicy.recognizeInstalled(manifest, installed)
    assertThrows(UpdateFailure::class.java) { UpdatePolicy.recognizeInstalled(manifest.copy(packageName = "me.danielshort.app.debug"), installed) }
    assertThrows(UpdateFailure::class.java) { UpdatePolicy.recognizeInstalled(manifest, installed.copy(packageName = "me.danielshort.app")) }
  }
}
