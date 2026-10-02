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
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Reads the real bundled game and its normal autosave; never seeds or changes progression. */
@RunWith(AndroidJUnit4::class)
class GuildOfflineDeviceTest {
  @Test fun firstBootAvailabilityAndPurchaseKeepThePlayfieldAnchored() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val opening = awaitReady(scenario)
      // Run on a fresh disposable AVD to cover the real first-run transition.
      // Retained save QA deliberately does not reset somebody else's guild.
      assumeTrue("Opening transition requires a naturally fresh guild", opening.getInt("boots") == 0)
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(90)
      var ready = opening
      while (!ready.optBoolean("canBuyBoots") && System.nanoTime() < deadline) {
        Thread.sleep(200)
        ready = readSnapshot(scenario)
        assertAnchored(opening, ready)
      }
      assertTrue("The first useful upgrade must arrive within 90 seconds", ready.optBoolean("canBuyBoots"))
      assertAnchored(opening, ready)
      evaluate(scenario, "document.querySelector('[data-perform=\"main:boots\"]').click(); true")
      var purchased = readSnapshot(scenario)
      val saveDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
      while ((purchased.optInt("boots") == 0 || !purchased.optBoolean("nativeConfirmed")) && System.nanoTime() < saveDeadline) {
        Thread.sleep(200)
        purchased = readSnapshot(scenario)
      }
      assertEquals("Normal purchase must persist its improvement", 1, purchased.getInt("boots"))
      assertTrue("Native storage acknowledges the durable purchase", purchased.getBoolean("nativeConfirmed"))
      assertAnchored(opening, purchased)
    }
  }

  @Test fun nativeCheckpointAcknowledgesTheCanonicalSaveBeforeLeaving() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      awaitReady(scenario)
      evaluate(scenario, "window.WayfarersAndroidUI.flush(); true")
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
      var saved = readSnapshot(scenario)
      while (!saved.optBoolean("nativeConfirmed") && System.nanoTime() < deadline) {
        Thread.sleep(100)
        saved = readSnapshot(scenario)
      }
      assertTrue("The native checkpoint must finish its fsync before acknowledgement", saved.getBoolean("nativeConfirmed"))
      scenario.onActivity { activity ->
        val checkpoint = GuildCheckpointStore(java.io.File(activity.filesDir, "guild-checkpoint.json")).read()
        assertNotNull("A separate reader verifies the checksum and complete atomic file", checkpoint)
        val state = JSONObject(checkpoint!!.text).getJSONObject("state")
        assertEquals(saved.getLong("createdAt"), state.getLong("createdAt"))
        assertEquals(saved.getInt("boots"), state.getJSONObject("upgrades").getInt("boots"))
      }
    }
  }

  @Test fun largerTextKeepsTheRealPurchaseInsideTheViewport() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      awaitReady(scenario)
      try {
        scenario.onActivity { activity -> findWebView(activity.window.decorView)!!.settings.textZoom = 130 }
        val enlarged = awaitReady(scenario)
        assertTrue("Text scaling must not grow the document horizontally", enlarged.getBoolean("noHorizontalOverflow"))
        assertTrue("Text scaling must keep the upgrade action tappable", enlarged.getBoolean("rendered"))
      } finally {
        scenario.onActivity { activity -> findWebView(activity.window.decorView)!!.settings.textZoom = 100 }
      }
    }
  }

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
          var action = document.querySelector('[data-perform="main:boots"]');
          var goal = document.querySelector('[data-goal]');
          var dock = document.querySelector('.wg-controls-column');
          var nav = document.querySelector('[data-nav]');
          var gameBox = game && game.getBoundingClientRect(), buttonBox = action && action.getBoundingClientRect();
          var sceneBox = scene && scene.getBoundingClientRect(), dockBox = dock && dock.getBoundingClientRect();
          var goalBox = goal && goal.getBoundingClientRect(), navBox = nav && nav.getBoundingClientRect();
          var hit = buttonBox && document.elementFromPoint(buttonBox.x + buttonBox.width / 2, buttonBox.y + buttonBox.height / 2);
          var noHorizontalOverflow = document.documentElement.scrollWidth <= innerWidth + 1 && game && game.scrollWidth <= game.clientWidth + 1;
          return {url: location.href, scene: scene && scene.dataset.sceneStatus, saved: !!state,
            createdAt: state && state.createdAt, lastUpdate: state && state.lastUpdate,
            boots: state && state.upgrades.boots, quiet: localStorage.getItem('wayfarers-guild-quiet'),
            nativeConfirmed: !!(window.WayfarersCheckpoint && window.WayfarersCheckpoint.confirmed()),
            canBuyBoots: action && !action.disabled,
            sceneTop: sceneBox && sceneBox.top, sceneHeight: sceneBox && sceneBox.height,
            dockTop: dockBox && dockBox.top, dockHeight: dockBox && dockBox.height,
            goalHeight: goalBox && goalBox.height, navTop: navBox && navBox.top,
            noHorizontalOverflow: !!noHorizontalOverflow,
            viewportWidth: innerWidth, viewportHeight: innerHeight, gameHeight: gameBox && gameBox.height,
            rendered: !!(gameBox && buttonBox && gameBox.height >= innerHeight * .8 &&
              noHorizontalOverflow && buttonBox.width >= 47 && buttonBox.height >= 47 &&
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

  private fun assertAnchored(before: JSONObject, after: JSONObject) {
    for (key in listOf("sceneTop", "sceneHeight", "dockTop", "dockHeight", "goalHeight", "navTop")) {
      assertEquals("Upgrade readiness must not move $key", before.getDouble(key), after.getDouble(key), 1.5)
    }
    assertTrue("The normal upgrade action stays visible", after.getBoolean("rendered"))
  }

  private fun evaluate(scenario: ActivityScenario<MainActivity>, script: String) {
    val latch = CountDownLatch(1)
    scenario.onActivity { activity ->
      findWebView(activity.window.decorView)!!.evaluateJavascript(script) { latch.countDown() }
    }
    assertTrue("Game action callback timed out", latch.await(5, TimeUnit.SECONDS))
  }

  private fun findWebView(view: View): WebView? {
    if (view is WebView) return view
    if (view is ViewGroup) for (index in 0 until view.childCount) findWebView(view.getChildAt(index))?.let { return it }
    return null
  }
}
