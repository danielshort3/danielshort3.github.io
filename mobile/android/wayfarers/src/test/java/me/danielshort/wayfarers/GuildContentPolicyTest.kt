package me.danielshort.wayfarers

import me.danielshort.app.BuildConfig
import org.junit.Assert.*
import org.junit.Test

class GuildContentPolicyTest {
  @Test fun saveOriginAndPackageRemainStableAcrossVersionedApks() {
    assertEquals("me.danielshort.wayfarers", BuildConfig.APPLICATION_ID)
    assertNotEquals("me.danielshort.app", BuildConfig.APPLICATION_ID)
    assertNotEquals("me.danielshort.app.debug", BuildConfig.APPLICATION_ID)
    assertEquals("https://appassets.androidplatform.net/assets/wayfarers/index.html", BuildConfig.GAME_URL)
    assertEquals(BuildConfig.GAME_URL, GuildContentPolicy.GAME_URL)
    assertEquals("https://appassets.androidplatform.net", GuildContentPolicy.ORIGIN)
  }

  @Test fun onlyTheLocalGameDocumentCanReceiveNativeActions() {
    assertTrue(GuildContentPolicy.isGame(GuildContentPolicy.GAME_URL))
    assertTrue(GuildContentPolicy.isGame(GuildContentPolicy.GAME_URL + "#settings"))
    listOf(
      "/assets/wayfarers/index.html", "https://appassets.androidplatform.net/",
      GuildContentPolicy.GAME_URL + "?version=2", GuildContentPolicy.GAME_URL + "/",
      GuildContentPolicy.GAME_URL.replace("index.html", "other.html"),
      "https://www.danielshort.me/games/wayfarers-guild"
    ).forEach { assertFalse("Unexpected game document: $it", GuildContentPolicy.isGame(it)) }
  }

  @Test fun bundledArtAliasesAndVersionedAssetsRemainOffline() {
    listOf(
      "/assets/wayfarers/game.js", "/assets/wayfarers/styles.css",
      "/assets/img/wayfarers-guild/actors.png?v=51f333248522",
      "/img/wayfarers-guild/realms.png?v=5f5dccb03a2d"
    ).forEach { assertTrue(it, GuildContentPolicy.isAsset(GuildContentPolicy.ORIGIN + it)) }
    listOf("/assets/private.json", "/assets/wayfarers-other/game.js", "/img/other.png", "/img/wayfarers-guild-other/actors.png")
      .forEach { assertFalse(it, GuildContentPolicy.isAsset(GuildContentPolicy.ORIGIN + it)) }
  }

  @Test fun foreignAndAmbiguousOriginsNeverReachGameOrAssetLoader() {
    listOf(
      "http://appassets.androidplatform.net/assets/wayfarers/index.html",
      "https://appassets.androidplatform.net:443/assets/wayfarers/index.html",
      "https://appassets.androidplatform.net:444/assets/wayfarers/index.html",
      "https://user@appassets.androidplatform.net/assets/wayfarers/index.html",
      "https://appassets.androidplatform.net.evil.test/assets/wayfarers/index.html",
      "https://evil.test/assets/wayfarers/index.html", "file:///android_asset/wayfarers/index.html",
      "javascript:alert(1)", "data:text/html,game", "not a URI"
    ).forEach { assertFalse(it, GuildContentPolicy.isGame(it)); assertFalse(it, GuildContentPolicy.isAsset(it)) }
  }

  @Test fun traversalAndEncodedSeparatorsCannotEscapeBundledDirectories() {
    listOf(
      "/assets/wayfarers/../private.json", "/assets/wayfarers/./index.html",
      "/assets/wayfarers/%2e%2e/private.json", "/assets/wayfarers/%2E/index.html",
      "/assets/wayfarers/%2fprivate.json", "/assets/wayfarers/%5cprivate.json",
      "/assets/wayfarers/%252e%252e/private.json", "/assets/wayfarers/\\private.json"
    ).forEach {
      val value = GuildContentPolicy.ORIGIN + it
      assertFalse(it, GuildContentPolicy.isGame(value))
      assertFalse(it, GuildContentPolicy.isAsset(value))
    }
  }
}
