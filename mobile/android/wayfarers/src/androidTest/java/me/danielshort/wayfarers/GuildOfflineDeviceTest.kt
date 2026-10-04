package me.danielshort.wayfarers

import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.content.pm.ActivityInfo
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
  @Test fun persistentAreasAndGlobalCatalogUseTheRetainedGuild() {
    // This mutation test is opt-in and is run only against the disposable
    // emulator-5564 fixture. Ordinary connected tests never spend a real save.
    assumeTrue("Requires the explicitly opted-in disposable mature fixture",
      InstrumentationRegistry.getArguments().getString("guildMatureQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario)
      assertTrue("At least the first three permanent areas must exist in this QA fixture", before.getInt("areaCount") >= 3)
      evaluate(scenario, "var batch = document.querySelector('[data-wx-batch]'); if (batch && !batch.hidden) { batch.click(); document.querySelector('[data-wx-do=\"batch:1\"]').click(); } true")
      val ranks = before.getString("areaRanks")
      for (id in listOf("greenway", "watchtower", "quarry")) {
        selectArea(scenario, id)
        val selected = awaitSnapshot(scenario, "Selecting $id should only navigate") {
          it.optString("selectedArea") == id && it.optString("sceneKind") == id && it.optBoolean("rendered")
        }
        assertEquals("Navigation must retain every area's upgrade ranks", ranks, selected.getString("areaRanks"))
      }
      val start = readSnapshot(scenario)
      evaluate(scenario, "document.querySelector('[data-wx-nav=\"upgrades\"]').click(); true")
      val catalog = awaitSnapshot(scenario, "The canonical catalog must render") { it.optInt("catalogRows") > 0 }
      assertTrue("Global menu must fit the viewport", catalog.getBoolean("noHorizontalOverflow"))
      Thread.sleep(2200)
      evaluate(scenario, "window.WayfarersAndroidUI.flush(); true")
      val working = awaitSnapshot(scenario, "Hidden production should reach the normal checkpoint") {
        it.optLong("lastUpdate") > start.getLong("lastUpdate")
      }
      for (id in listOf("greenway", "quarry", "watchtower")) {
        assertTrue("Hidden $id must keep producing in the global menu",
          working.getJSONObject("areaElapsed").getDouble(id) > start.getJSONObject("areaElapsed").getDouble(id))
      }
      evaluate(scenario, """
        document.querySelector('[data-wx-upgrade="area:greenway:boots"] .wx-research-info').click(); true
      """.trimIndent())
      assertEquals("upgrade", readSnapshot(scenario).getString("sheetKind"))
      evaluate(scenario, """
        var offer = document.querySelector('.wx-sheet[open] .wx-confirm');
        if (!offer || offer.disabled) throw new Error('Mature QA fixture needs one affordable Trail boot rank');
        offer.click(); true
      """.trimIndent())
      val purchased = awaitSnapshot(scenario, "The targeted catalog purchase must be durable") {
        it.optInt("boots") == before.getInt("boots") + 1 && it.optBoolean("nativeConfirmed")
      }
      assertEquals("Buying Trail boots must not change the selected Quarry", "quarry", purchased.getString("selectedArea"))
      for (id in listOf("quarry", "watchtower")) {
        assertEquals("A Trail purchase cannot change $id ranks",
          JSONObject(ranks).getJSONObject(id).toString(),
          JSONObject(purchased.getString("areaRanks")).getJSONObject(id).toString())
      }
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      awaitSheet(scenario, "")
      selectArea(scenario, "quarry")
      awaitReady(scenario)
      scenario.recreate()
      val restored = awaitReady(scenario)
      assertEquals(purchased.getLong("createdAt"), restored.getLong("createdAt"))
      assertEquals(purchased.getString("areaRanks"), restored.getString("areaRanks"))
      assertEquals("quarry", restored.getString("selectedArea"))
    }
  }

  private fun selectArea(scenario: ActivityScenario<MainActivity>, id: String) {
    evaluate(scenario, "document.querySelector('[data-wx-nav=\"expedition\"]').click(); document.querySelector('[data-wx-objective]').click(); true")
    awaitSheet(scenario, "areas")
    evaluate(scenario, "document.querySelector('[data-wx-area=\"$id\"]').click(); true")
    awaitReady(scenario)
  }

  @Test fun sixAreaPickerExactBatchAndPlansRemainUsableWithLargeText() {
    assumeTrue("Requires the explicitly opted-in disposable six-area fixture",
      InstrumentationRegistry.getArguments().getString("guildProgressionQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario)
      assertEquals(6, before.getInt("areaCount"))
      val ranks = before.getString("areaRanks")
      try {
        scenario.onActivity { activity -> findWebView(activity.window.decorView)!!.settings.textZoom = 130 }
        for (id in listOf("greenway", "quarry", "watchtower", "workshop", "ruins", "harbor")) {
          selectArea(scenario, id)
          val selected = awaitSnapshot(scenario, "Six-area navigation must show $id") {
            it.optString("selectedArea") == id && it.optString("sceneKind") == id && it.optBoolean("rendered")
          }
          assertEquals("Navigation retains all ordinary ranks", ranks, selected.getString("areaRanks"))
          assertTrue("Large text keeps every currency line inside its button", selected.getBoolean("priceLabelsFit"))
          assertEquals("All six learned tracks remain available", 6, selected.getInt("trackCount"))
          evaluate(scenario, "document.querySelector('[data-wx-do=\"world-choice\"]').click(); true")
          awaitSheet(scenario, "choice")
          scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
          awaitSheet(scenario, "")
        }
        selectArea(scenario, "greenway")
        evaluate(scenario, "document.querySelector('[data-wx-batch]').click(); document.querySelector('[data-wx-do=\"batch:5\"]').click(); true")
        // Complete the real first-use lesson before measuring an ordinary batch.
        awaitReady(scenario)
        val purchasedBefore = readSnapshot(scenario)
        evaluate(scenario, "document.querySelector('[data-wx-buy=\"boots\"]').click(); true")
        val purchased = awaitSnapshot(scenario, "Exact five-rank purchase must be durably saved") {
          it.optInt("boots") == purchasedBefore.getInt("boots") + 5 && it.optBoolean("nativeConfirmed")
        }
        assertEquals(5, purchased.getInt("batch"))
        scenario.recreate()
        val restored = awaitReady(scenario)
        assertEquals(purchased.getInt("boots"), restored.getInt("boots"))
        assertEquals(5, restored.getInt("batch"))
      } finally {
        scenario.onActivity { activity -> findWebView(activity.window.decorView)!!.settings.textZoom = 100 }
      }
    }
  }

  @Test fun firstBootAvailabilityAndPurchaseKeepThePlayfieldAnchored() {
    assumeTrue("Requires explicit permission to spend on the disposable opening guild",
      InstrumentationRegistry.getArguments().getString("guildGuideAcknowledgementQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val opening = awaitReady(scenario)
      // The separate onboarding suite proves the supplied practice rank is exactly
      // zero to one. This case measures the next ordinary paid rank instead.
      assumeTrue("Opening transition requires a naturally fresh guild after its real lesson", opening.getInt("stageIndex") == 0 && opening.getInt("guildBoots") == 0 && opening.optInt("trailPracticeProgress") == 3)
      val initialRank = opening.getInt("boots")
      val initialSupplies = opening.getInt("trailPracticeSupplies")
      assertTrue("The completed lesson must leave its actual rank", initialRank >= 1)
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      var ready = opening
      while (!ready.optBoolean("canBuyBoots") && System.nanoTime() < deadline) {
        Thread.sleep(200)
        ready = readSnapshot(scenario)
        assertAnchored(opening, ready)
      }
      assertTrue("The next ordinary upgrade must arrive within 20 seconds on the device", ready.optBoolean("canBuyBoots"))
      assertAnchored(opening, ready)
      evaluate(scenario, "document.querySelector('[data-upgrade=\"boots\"] .wx-upgrade-info').click(); window.WayfarersAndroidUI.flush(); true")
      awaitSnapshot(scenario, "The exact ordinary price must have a durable starting checkpoint") {
        it.optBoolean("nativeConfirmed") && it.optString("sheetKind") == "local"
      }
      evaluate(scenario, """
        (function(){
          var audit = window.__guildPaidPurchaseQa = {ok:false};
          try {
            var read = function(){return JSON.parse(localStorage.getItem('wayfarers-guild-save-v1')).state;};
            var before = read(), core = window.WayfarersCore, n = core.Numbers;
            var item = core.getView(before).expedition.cards.find(function(row){return row.action.id === 'boots';});
            var summary = document.querySelector('.wx-sheet[open] .wx-purchase-summary');
            var button = document.querySelector('.wx-sheet[open] .wx-confirm');
            if(!item || item.quantity !== 1 || !button || button.disabled || button.hasAttribute('data-wx-practice')) throw new Error('Ordinary single-rank offer missing');
            audit.displayedCost = summary && summary.textContent;
            audit.priceMatches = !!summary && item.cost.every(function(cost){return summary.textContent.includes(cost.text);});
            if(!audit.priceMatches) throw new Error('Displayed exact cost differs from the canonical offer');
            button.click();
            var after = read(), accrued = JSON.parse(JSON.stringify(before));
            // Simulate only a copy to account for normal production between the
            // checkpoint and click; the live clock and resources are untouched.
            core.advanceTo(accrued,after.lastUpdate);
            audit.debits = item.cost.map(function(cost){
              var actual = n.toNumber(n.sub(accrued.resources[cost.resource],after.resources[cost.resource]));
              var expected = n.toNumber(cost.amount);
              return {resource:cost.resource,actual:actual,expected:expected,ok:Math.abs(actual-expected)<=Math.max(1e-7,Math.abs(expected)*1e-7)};
            });
            audit.rankDelta = after.expedition.areas.greenway.ranks.boots-before.expedition.areas.greenway.ranks.boots;
            audit.ok = audit.rankDelta === 1 && audit.debits.length > 0 && audit.debits.every(function(cost){return cost.ok;});
          } catch(error) { audit.error = String(error); }
          return true;
        }())
      """.trimIndent())
      val purchased = awaitSnapshot(scenario, "The normal paid rank must persist its exact improvement") {
        it.optInt("boots") == initialRank + 1 && it.optBoolean("nativeConfirmed")
      }
      val audit = purchased.getJSONObject("paidPurchaseAudit")
      assertTrue("The normal purchase must spend its displayed exact price: $audit", audit.optBoolean("ok"))
      assertEquals("An ordinary purchase must not consume a tutorial supply", initialSupplies, purchased.getInt("trailPracticeSupplies"))
      assertEquals("Normal purchase must persist its exact improvement", initialRank + 1, purchased.getInt("boots"))
      assertTrue("Native storage acknowledges the durable purchase", purchased.getBoolean("nativeConfirmed"))
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      val closed = awaitReady(scenario)
      assertAnchored(opening, closed)
      scenario.recreate()
      val restored = awaitReady(scenario)
      assertEquals(purchased.getLong("createdAt"), restored.getLong("createdAt"))
      assertEquals("The exact paid rank survives a cold WebView", initialRank + 1, restored.getInt("boots"))
      assertEquals(initialSupplies, restored.getInt("trailPracticeSupplies"))
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
        val expedition = state.getJSONObject("expedition")
        val trail = expedition.optJSONObject("areas")?.optJSONObject("greenway") ?: expedition
        assertEquals(saved.getInt("boots"), trail.getJSONObject("ranks").getInt("boots"))
        assertEquals(saved.getInt("guildBoots"), state.getJSONObject("upgrades").getInt("boots"))
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

  @Test fun nativeBackClosesContextSheetsBeforeLeavingTheGame() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      awaitReady(scenario)
      evaluate(scenario, "document.querySelector('[data-wx-options]').click(); true")
      assertEquals("options", readSnapshot(scenario).optString("sheetKind"))
      evaluate(scenario, "Array.from(document.querySelectorAll('.wx-sheet button')).find(b => b.textContent.includes('How this expedition works')).click(); true")
      assertEquals("objective", readSnapshot(scenario).optString("sheetKind"))
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      awaitSheet(scenario, "options")
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      awaitSheet(scenario, "")
      assertEquals(GuildContentPolicy.GAME_URL, awaitReady(scenario).getString("url"))
    }
  }

  private fun awaitSheet(scenario: ActivityScenario<MainActivity>, expected: String) {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
    while (System.nanoTime() < deadline) {
      if (readSnapshot(scenario).optString("sheetKind") == expected) return
      Thread.sleep(100)
    }
    assertEquals("Native Back preserves the game's sheet hierarchy", expected, readSnapshot(scenario).optString("sheetKind"))
  }

  private fun awaitSnapshot(scenario: ActivityScenario<MainActivity>, label: String, matches: (JSONObject) -> Boolean): JSONObject {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8)
    var snapshot = JSONObject()
    while (System.nanoTime() < deadline) {
      snapshot = readSnapshot(scenario)
      if (matches(snapshot)) return snapshot
      Thread.sleep(100)
    }
    throw AssertionError("$label: $snapshot")
  }

  private fun awaitReady(scenario: ActivityScenario<MainActivity>, landscape: Boolean? = null): JSONObject {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(35)
    var snapshot = JSONObject()
    var practiced = false
    while (System.nanoTime() < deadline) {
      snapshot = readSnapshot(scenario)
      if (snapshot.optString("sheetKind") in listOf("return", "finale")) {
        evaluate(scenario, "document.querySelector('.wx-sheet [data-wx-close]').click(); true")
        continue
      }
      if (snapshot.optString("sheetKind") in listOf("onboarding-notice", "onboarding-inbox")) {
        evaluate(scenario, "document.querySelector('.wx-sheet [data-wx-do=\"onboarding-later\"]').click(); true")
        continue
      }
      // Opted-in legacy cases use the actual required game controls. They never
      // skip a lesson or edit its proof; the guide suite owns resume assertions.
      if (snapshot.optBoolean("guideOpen") && InstrumentationRegistry.getArguments()
          .getString("guildGuideAcknowledgementQa") == "true") {
        practiced = true
        evaluate(scenario, "var guide=document.querySelector('.wx-guide[open]'); var target=guide && guide.dataset.step.startsWith('currency:') ? guide.querySelector('[data-guide-next]:not([hidden])') : Array.from(document.querySelectorAll('[aria-describedby~=\"wx-guide-body\"]')).find(function(node){return !node.classList.contains('wx-guide');}); if(target && !target.disabled && !target.closest('[inert]')) target.click(); true")
        Thread.sleep(150)
        continue
      }
      if (practiced && !snapshot.optBoolean("guideOpen") && snapshot.optString("sheetKind").isNotEmpty()) {
        scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
        Thread.sleep(150)
        continue
      }
      val orientationMatches = landscape == null || (snapshot.optInt("viewportWidth") > snapshot.optInt("viewportHeight")) == landscape
      if (snapshot.optString("scene") == "ready" && snapshot.optBoolean("saved") && snapshot.optBoolean("nativeConfirmed") && !snapshot.optBoolean("guideOpen") && snapshot.optBoolean("rendered") && orientationMatches) return snapshot
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
          var expedition = state && state.expedition;
          var trail = expedition && (expedition.areas ? expedition.areas.greenway : expedition);
          var areas = expedition && expedition.areas || {};
          var areaRanks = {}, areaElapsed = {};
          Object.keys(areas).forEach(function (id) { areaRanks[id] = areas[id].ranks; areaElapsed[id] = areas[id].elapsed; });
          var scene = document.querySelector('[data-wx-canvas]');
          var game = document.querySelector('.wx-game');
          var action = document.querySelector('[data-wx-buy="boots"]') || document.querySelector('[data-wx-buy]');
          var goal = document.querySelector('.wx-objective');
          var dock = document.querySelector('.wx-tray');
          var nav = document.querySelector('.wx-nav');
          var sheet = document.querySelector('.wx-sheet[open]');
          var gameBox = game && game.getBoundingClientRect(), buttonBox = action && action.getBoundingClientRect();
          var sceneBox = scene && scene.getBoundingClientRect(), dockBox = dock && dock.getBoundingClientRect();
          var goalBox = goal && goal.getBoundingClientRect(), navBox = nav && nav.getBoundingClientRect();
          var hit = buttonBox && document.elementFromPoint(buttonBox.x + buttonBox.width / 2, buttonBox.y + buttonBox.height / 2);
          var noHorizontalOverflow = document.documentElement.scrollWidth <= innerWidth + 1 && game && game.scrollWidth <= game.clientWidth + 1;
          return {url: location.href, scene: scene && scene.dataset.sceneStatus, saved: !!state,
            createdAt: state && state.createdAt, lastUpdate: state && state.lastUpdate,
            boots: trail && trail.ranks.boots,
            trailPracticeProgress: state && state.onboarding.practice.progress.greenway,
            trailPracticeSupplies: state ? state.onboarding.practice.supplies.filter(function(id){return id==='greenway:upgrade';}).length : 0,
            paidPurchaseAudit: window.__guildPaidPurchaseQa || {},
            areaCount: Object.keys(areas).length, selectedArea: expedition && expedition.selectedArea,
            batch: expedition && expedition.batch || 1, trackCount: document.querySelectorAll('[data-wx-buy]').length,
            priceLabelsFit: Array.from(document.querySelectorAll('.wx-dock .wx-price>span')).every(function (label) {
              return label.getBoundingClientRect().bottom <= label.parentElement.getBoundingClientRect().bottom + 1;
            }),
            areaRanks: JSON.stringify(areaRanks), areaElapsed: areaElapsed,
            sceneKind: scene && scene.dataset.sceneKind,
            catalogRows: document.querySelectorAll('[data-wx-upgrade]').length,
            guildBoots: state && state.upgrades.boots,
            stageIndex: state && state.expedition && state.expedition.index,
            sheetKind: sheet ? sheet.dataset.kind : '', quiet: localStorage.getItem('wayfarers-guild-quiet'),
            guideOpen: !!document.querySelector('.wx-guide[open]'),
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
