package me.danielshort.wayfarers

import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.content.pm.ActivityInfo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Reads the real bundled game and its normal autosave; never seeds or changes progression. */
@RunWith(AndroidJUnit4::class)
class GuildOfflineDeviceTest {
  @Test fun bundledGameAndDurableSaveSurviveActivityRecreation() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario)
      assertEquals(GuildContentPolicy.GAME_URL, before.getString("url"))
      assertEquals("ready", before.getString("scene"))
      assertTrue("The working DOM must also be physically visible and usable", before.getBoolean("rendered"))
      assertTrue("Normal gameplay must have created its own save", before.getBoolean("saved"))
      assertTrue(before.getLong("createdAt") > 0)
      scenario.recreate()
      val after = awaitReady(scenario)
      assertEquals("The version-independent origin preserves the same game", before.getLong("createdAt"), after.getLong("createdAt"))
      assertEquals("Purchased boot level survives recreation", before.getInt("boots"), after.getInt("boots"))
      assertEquals("Quiet preference survives recreation", before.optString("quiet"), after.optString("quiet"))
      assertTrue(after.getLong("lastUpdate") >= before.getLong("lastUpdate"))
    }
  }

  @Test fun appAssetsLoadWithoutGrantingFileOrContentAccessToTheGame() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      awaitReady(scenario)
      scenario.onActivity { activity ->
        val web = findWebView(activity.window.decorView) ?: error("Bundled game WebView missing")
        assertEquals("Wayfarers Guild game", web.contentDescription)
        assertFalse("External filesystem is not the save origin", web.settings.allowFileAccess)
        assertFalse("Native import owns content URI access", web.settings.allowContentAccess)
        assertTrue(web.settings.domStorageEnabled)
      }
    }
  }

  @Test fun portraitAndLandscapeKeepTheRealPurchaseVisible() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      try {
        scenario.onActivity { it.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE }
        val landscape = awaitReady(scenario, landscape = true)
        assertTrue(landscape.getInt("viewportWidth") > landscape.getInt("viewportHeight"))
        scenario.onActivity { it.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT }
        val portrait = awaitReady(scenario, landscape = false)
        assertTrue(portrait.getInt("viewportHeight") > portrait.getInt("viewportWidth"))
        assertEquals(landscape.getLong("createdAt"), portrait.getLong("createdAt"))
      } finally {
        scenario.onActivity { it.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED }
      }
    }
  }

  private fun awaitReady(scenario: ActivityScenario<MainActivity>, landscape: Boolean? = null): JSONObject {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(35)
    var snapshot = JSONObject()
    while (System.nanoTime() < deadline) {
      snapshot = readSnapshot(scenario)
      val orientationMatches = landscape == null || (snapshot.optInt("viewportWidth") > snapshot.optInt("viewportHeight")) == landscape
      if (snapshot.optString("scene") == "ready" && snapshot.optBoolean("saved") && snapshot.optBoolean("rendered") && orientationMatches) return snapshot
      Thread.sleep(150)
    }
    throw AssertionError("Bundled game did not become ready with a durable save: $snapshot")
  }

  private fun readSnapshot(scenario: ActivityScenario<MainActivity>): JSONObject {
    val latch = CountDownLatch(1)
    var result = JSONObject()
    scenario.onActivity { activity ->
      val web = findWebView(activity.window.decorView)
      if (web == null) { latch.countDown(); return@onActivity }
      web.evaluateJavascript("""
        JSON.stringify((function () {
          var envelope = JSON.parse(localStorage.getItem('wayfarers-guild-save-v1') || 'null');
          var state = envelope && envelope.state;
          var scene = document.querySelector('[data-scene]');
          var game = document.querySelector('[data-game]');
          var action = document.querySelector('[data-main-actions] .wg-buy');
          var gameBox = game && game.getBoundingClientRect(), buttonBox = action && action.getBoundingClientRect();
          var hit = buttonBox && document.elementFromPoint(buttonBox.x + buttonBox.width / 2, buttonBox.y + buttonBox.height / 2);
          return {url: location.href, scene: scene && scene.dataset.sceneStatus, saved: !!state,
            createdAt: state && state.createdAt, lastUpdate: state && state.lastUpdate,
            boots: state && state.upgrades.boots, quiet: localStorage.getItem('wayfarers-guild-quiet'),
            viewportWidth: innerWidth, viewportHeight: innerHeight, gameHeight: gameBox && gameBox.height,
            rendered: !!(gameBox && buttonBox && gameBox.height >= innerHeight * .8 &&
              game.scrollWidth <= game.clientWidth + 1 && buttonBox.width >= 47 && buttonBox.height >= 47 &&
              buttonBox.top >= 0 && buttonBox.bottom <= innerHeight && hit && (action === hit || action.contains(hit)))};
        }()))
      """.trimIndent()) { raw ->
        result = runCatching { JSONObject(JSONArray("[$raw]").getString(0)) }.getOrDefault(JSONObject())
        latch.countDown()
      }
    }
    assertTrue("WebView read callback timed out", latch.await(5, TimeUnit.SECONDS))
    return result
  }

  private fun findWebView(view: View): WebView? {
    if (view is WebView) return view
    if (view is ViewGroup) for (index in 0 until view.childCount) findWebView(view.getChildAt(index))?.let { return it }
    return null
  }
}
