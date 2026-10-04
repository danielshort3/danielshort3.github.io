package me.danielshort.wayfarers

import android.content.pm.ActivityInfo
import android.os.SystemClock
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
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

/** Explicit disposable-device opt-in; no save replacement or progression injection. */
@RunWith(AndroidJUnit4::class)
class GuildOnboardingDeviceTest {
  @Test fun pendingUpgradeRemainsOperableInCompactLandscape() {
    assumeTrue("Requires a disposable guild paused at its first upgrade",
      InstrumentationRegistry.getArguments().getString("guildCompactGuideQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      awaitGuide(scenario, "greenway", "upgrade")
      try {
        scenario.onActivity { activity ->
          findWebView(activity.window.decorView)!!.settings.textZoom = 130
          activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
        val landscape = awaitSnapshot(scenario, "Compact landscape must expose the required real upgrade") {
          it.optInt("width") > it.optInt("height") && it.optBoolean("spotlightVisible") && it.optBoolean("targetReceivesHit")
        }
        assertCoach(landscape)
        saveScreenshot("compact-landscape")
        performHighlightedStep(scenario, "upgrade", nativeTouch = true)
        val bought = awaitGuide(scenario, "greenway", "operate")
        assertEquals(1, bought.getInt("boots"))
        assertEquals(1, bought.getInt("supplyCount"))
      } finally {
        scenario.onActivity { activity ->
          findWebView(activity.window.decorView)!!.settings.textZoom = 100
          activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
      }
    }
  }

  @Test fun retainedE2PlanUsesVisibleRealControlsAndPersistsItsChoice() {
    assumeTrue("Requires an explicitly imported disposable retained E2 save",
      InstrumentationRegistry.getArguments().getString("guildRetainedE2Qa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val opening = awaitSnapshot(scenario, "The retained E2 Trail lesson must be active") {
        it.optInt("expeditionVersion") == 2 && it.optString("guide") == "greenway" && it.optBoolean("nativeConfirmed")
      }
      assertTrue(opening.getInt("progress") in 0..2)
      val originalClaims = opening.getInt("claimCount")
      acknowledgeCurrencies(scenario)
      if (opening.getInt("progress") == 0) performHighlightedStep(scenario, "inspect")
      acknowledgeCurrencies(scenario)
      if (snapshot(scenario).getInt("progress") == 1) performHighlightedStep(scenario, "upgrade")
      val beforePlan = awaitGuide(scenario, "greenway", "operate")
      assertEquals(1, beforePlan.getInt("boots"))
      assertEquals(1, beforePlan.getInt("supplyCount"))
      assertEquals("short", beforePlan.getString("routeChoice"))
      assertEquals(2, beforePlan.getInt("proofCount"))
      scenario.recreate()
      awaitGuide(scenario, "greenway", "operate")
      performHighlightedStep(scenario, "operate", nativeTouch = true)
      val complete = awaitSnapshot(scenario, "The real legacy plan choice must persist without a repeat reward") {
        it.optInt("progress") == 3 && it.optString("routeChoice") == "supply" && it.optBoolean("nativeConfirmed")
      }
      assertEquals(originalClaims, complete.getInt("claimCount"))
      assertEquals(1, complete.getInt("boots"))
      assertEquals(1, complete.getInt("supplyCount"))
      assertEquals(3, complete.getInt("proofCount"))
      scenario.recreate()
      val restored = awaitSnapshot(scenario, "The retained legacy plan and receipts must survive recreation") {
        it.optInt("progress") == 3 && it.optString("routeChoice") == "supply" && it.optBoolean("nativeConfirmed")
      }
      assertEquals(opening.getLong("createdAt"), restored.getLong("createdAt"))
      assertEquals(originalClaims, restored.getInt("claimCount"))
      assertEquals(1, restored.getInt("boots"))
      assertEquals(1, restored.getInt("supplyCount"))
      assertEquals(3, restored.getInt("proofCount"))
    }
  }

  @Test fun freshTrailPracticeResumesAndItsSuppliedRankCannotRepeat() {
    assumeTrue("Requires an explicitly reset disposable guild",
      InstrumentationRegistry.getArguments().getString("guildOnboardingQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val currency = awaitGuide(scenario, "greenway", "currency:coins")
      assertEquals(0, currency.getInt("progress"))
      assertEquals(0, currency.getInt("proofCount"))
      assertEquals(0, currency.getJSONArray("currencyRead").length())
      assertCurrencyCoach(currency)
      scenario.recreate()
      val currencyResumed = awaitGuide(scenario, "greenway", "currency:coins")
      assertEquals(currency.getLong("createdAt"), currencyResumed.getLong("createdAt"))
      assertEquals(0, currencyResumed.getJSONArray("currencyRead").length())
      acknowledgeCurrencies(scenario)
      val opening = awaitGuide(scenario, "greenway", "inspect")
      assertEquals("Fixture must have its real untouched first guide", 0, opening.getInt("progress"))
      assertEquals(0, opening.getInt("claimCount"))
      assertEquals(0, opening.getInt("boots"))
      assertEquals("[\"coins\"]", opening.getJSONArray("currencyRead").toString())
      assertCoach(opening)
      performHighlightedStep(scenario, "inspect")
      val second = awaitGuide(scenario, "greenway", "upgrade")
      assertCoach(second)
      assertTrue("The normal first quote remains visible", second.getString("quoteText").contains("6 Coins", ignoreCase = true))
      assertEquals(1, second.getInt("progress"))
      assertEquals("Inspection must not buy the required practice rank", 0, second.getInt("boots"))
      assertEquals(1, second.getInt("proofCount"))
      assertEquals(0, second.getInt("supplyCount"))
      assertTrue(second.getBoolean("nativeConfirmed"))

      scenario.recreate()
      val resumed = awaitGuide(scenario, "greenway", "upgrade")
      assertEquals(opening.getLong("createdAt"), resumed.getLong("createdAt"))
      assertEquals(opening.getJSONArray("currencyRead").toString(), resumed.getJSONArray("currencyRead").toString())
      repeat(3) {
        scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
        Thread.sleep(250)
        val retained = awaitGuide(scenario, "greenway", "upgrade")
        assertEquals("Back must preserve the unfinished action", 1, retained.getInt("progress"))
        assertEquals(0, retained.getInt("boots"))
        assertFalse("Mandatory Back cannot expose an area-navigation bypass", retained.getString("sheetKind") in listOf("areas", "options"))
        assertCoach(retained)
      }
      evaluate(scenario, "document.querySelector('.wx-guide[open] [data-guide-settings]').click(); true")
      val recovery = awaitSnapshot(scenario, "The actual recovery control must open restricted Options") {
        it.optString("sheetKind") == "options" && it.optBoolean("recoveryOptions") && !it.optBoolean("guideOpen")
      }
      assertEquals(1, recovery.getInt("progress"))
      assertEquals(0, recovery.getInt("boots"))
      assertEquals(0, recovery.getInt("supplyCount"))
      assertEquals(1, recovery.getInt("proofCount"))
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      awaitGuide(scenario, "greenway", "upgrade")

      try {
        scenario.onActivity { activity ->
          findWebView(activity.window.decorView)!!.settings.textZoom = 130
          activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
        val enlarged = awaitSnapshot(scenario, "Landscape guide must retain its visible action") {
          it.optString("step") == "upgrade" && it.optInt("width") > it.optInt("height") && it.optBoolean("coachFits") && it.optBoolean("targetReceivesHit")
        }
        assertCoach(enlarged)
        assertEquals(1, enlarged.getInt("progress"))
      } finally {
        scenario.onActivity { activity ->
          findWebView(activity.window.decorView)!!.settings.textZoom = 100
          activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
      }
      var stablePortraitSamples = 0
      var previousDimensions = ""
      awaitSnapshot(scenario, "Portrait restoration must finish before the real purchase") {
        val dimensions = "${it.optInt("width")}:${it.optInt("height")}"
        val ready = it.optString("step") == "upgrade" && it.optInt("height") > it.optInt("width") && it.optBoolean("coachFits") && it.optBoolean("targetReceivesHit")
        stablePortraitSamples = if (ready && dimensions == previousDimensions) stablePortraitSamples + 1 else 0
        previousDimensions = dimensions
        stablePortraitSamples >= 3
      }
      performHighlightedStep(scenario, "upgrade")
      val purchased = awaitGuide(scenario, "greenway", "operate")
      assertCoach(purchased)
      assertEquals("The highlighted real control must purchase exactly one rank", 1, purchased.getInt("boots"))
      assertEquals(2, purchased.getInt("proofCount"))
      assertEquals("The guild practice supply is consumed once", 1, purchased.getInt("supplyCount"))
      assertTrue("The rank must improve actual Trail capacity", purchased.getDouble("travelCapacity") > opening.getDouble("travelCapacity"))
      scenario.recreate()
      val afterPurchase = awaitGuide(scenario, "greenway", "operate")
      assertEquals(1, afterPurchase.getInt("boots"))
      assertEquals(1, afterPurchase.getInt("supplyCount"))
      performHighlightedStep(scenario, "operate")
      val complete = awaitSnapshot(scenario, "The final actual interaction must save one learning reward") {
        it.optInt("progress") == 3 && it.optInt("claimCount") == 1 && it.optBoolean("nativeConfirmed")
      }
      assertEquals(1, complete.getInt("boots"))
      assertEquals(3, complete.getInt("proofCount"))
      assertEquals(1, complete.getInt("supplyCount"))
      scenario.recreate()
      awaitSnapshot(scenario, "Completion and reward survive recreation") {
        it.optInt("progress") == 3 && it.optInt("claimCount") == 1 && !it.optBoolean("guideOpen")
      }
      dismissNotice(scenario)
      evaluate(scenario, "var close=document.querySelector('.wx-sheet[open] [data-wx-close]'); if(close) close.click(); true")
      var closedSamples = 0
      awaitSnapshot(scenario, "The previous sheet must finish closing before Options opens") {
        closedSamples = if (it.optString("sheetKind").isEmpty()) closedSamples + 1 else 0
        closedSamples >= 2
      }
      evaluate(scenario, "document.querySelector('[data-wx-options]').click(); true")
      awaitSnapshot(scenario, "Options must render its actual Trail replay control") {
        it.optString("sheetKind") == "options" && it.optBoolean("replayAvailable")
      }
      evaluate(scenario, "document.querySelector('.wx-sheet[open] [data-wx-do=\"guide-replay:greenway\"]').click(); true")
      val replay = awaitSnapshot(scenario, "Replay must open the actual read-only lesson reference") {
        it.optString("sheetKind") == "lesson-help" && it.optInt("referenceSteps") == 3
      }
      assertFalse("A completed reference must not restart an interactive lesson", replay.getBoolean("guideOpen"))
      assertTrue("A completed reference must offer no practice purchase", replay.getBoolean("referenceReadOnly"))
      assertEquals(1, replay.getInt("claimCount"))
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      val afterReplay = awaitSnapshot(scenario, "Closing replay must not repeat practice or its reward") { it.optString("sheetKind") != "lesson-help" && !it.optBoolean("guideOpen") }
      assertEquals(1, afterReplay.getInt("claimCount"))
      assertEquals(1, afterReplay.getInt("supplyCount"))
      assertEquals(3, afterReplay.getInt("proofCount"))
      assertEquals(1, afterReplay.getInt("boots"))
      assertEquals(opening.getJSONArray("currencyRead").toString(), afterReplay.getJSONArray("currencyRead").toString())
      assertEquals(complete.getLong("createdAt"), afterReplay.getLong("createdAt"))
    }
  }

  private fun assertCoach(snapshot: JSONObject) {
    assertGuideChrome(snapshot)
    assertTrue("The highlighted real control must be visible", snapshot.getBoolean("spotlightVisible"))
    assertTrue("The real highlighted control must be operable and at least 48dp", snapshot.getBoolean("targetOperable"))
    assertTrue("The coach must not intercept its required game control: $snapshot", snapshot.getBoolean("targetReceivesHit"))
    if (snapshot.optString("step") == "upgrade") {
      assertTrue("The canonical price and coverage must be unoccluded", snapshot.getBoolean("quoteVisible"))
      assertTrue("The real control must describe its canonical quote", snapshot.getBoolean("quoteDescribed"))
      assertTrue("The changed-step announcement includes the quote", snapshot.getBoolean("quoteAnnounced"))
      assertTrue(snapshot.getString("quoteText").contains("×1"))
      assertTrue(snapshot.getString("quoteText").contains("Guild supplies"))
      assertTrue(snapshot.getString("quoteText").contains("wallet unchanged"))
    }
  }

  private fun assertCurrencyCoach(snapshot: JSONObject) {
    assertGuideChrome(snapshot)
    assertTrue("Currency teaching has an explicit operable 48dp Next", snapshot.getBoolean("currencyNextOperable"))
    assertEquals("Next", snapshot.getString("currencyNextLabel"))
    assertTrue("The explained currency must be highlighted", snapshot.getBoolean("spotlightVisible"))
  }

  private fun assertGuideChrome(snapshot: JSONObject) {
    assertTrue("The coach must fit", snapshot.getBoolean("coachFits"))
    assertFalse("A mandatory lesson must not show Skip or Leave", snapshot.getBoolean("leaveVisible"))
    assertTrue("The 48dp Settings recovery control must remain reachable", snapshot.getBoolean("recoveryOperable"))
    assertTrue("Unrelated controls must not receive click-through", snapshot.getBoolean("backgroundBlocked"))
    assertTrue("Keyboard focus stays in the allowed target, coach or sheet exit: $snapshot", snapshot.getBoolean("focusAllowed"))
    assertTrue("The guide has an accessible name and description", snapshot.getBoolean("accessible"))
    assertTrue("Each step exposes one atomic polite announcement", snapshot.getBoolean("stepAnnouncement"))
    assertTrue("The game must not acquire horizontal overflow", snapshot.getBoolean("noOverflow"))
    assertTrue("Supplied purchases use normal labels without Free", snapshot.getBoolean("normalPurchaseCopy"))
  }

  private fun acknowledgeCurrencies(scenario: ActivityScenario<MainActivity>) {
    repeat(16) {
      val before = snapshot(scenario)
      val step = before.optString("step")
      if (!before.optBoolean("guideOpen") || !step.startsWith("currency:")) return
      assertCurrencyCoach(before)
      evaluate(scenario, "document.querySelector('.wx-guide[open] [data-guide-next]:not([hidden])').click(); true")
      val after = awaitSnapshot(scenario, "Next must save the currency receipt and advance only the explanation") {
        it.optString("step") != step && it.optBoolean("nativeConfirmed")
      }
      for (key in listOf("progress", "proofCount", "supplyCount", "claimCount", "boots")) {
        assertEquals("Currency Next cannot change $key", before.getInt(key), after.getInt(key))
      }
      val previous = before.getJSONArray("currencyRead")
      val learned = after.getJSONArray("currencyRead")
      assertEquals(previous.length() + 1, learned.length())
      assertEquals(step.removePrefix("currency:"), learned.getString(learned.length() - 1))
      for (index in 0 until previous.length()) assertEquals(previous.getString(index), learned.getString(index))
    }
    fail("Currency teaching did not reach its real required action")
  }

  private fun performHighlightedStep(scenario: ActivityScenario<MainActivity>, step: String, nativeTouch: Boolean = false) {
    repeat(8) {
      val before = snapshot(scenario)
      if (!before.optBoolean("guideOpen") || before.optString("step") != step) return
      if (before.optInt("expeditionVersion") == 2) println("RETAINED_E2_RENDER $before")
      assertCoach(before)
      if (nativeTouch) touchHighlightedControl(scenario, before)
      else evaluate(scenario, "var target=Array.from(document.querySelectorAll('[aria-describedby~=\"wx-guide-body\"]')).find(function(node){return !node.classList.contains('wx-guide');}); if(!target || target.disabled || target.closest('[inert]')) throw new Error('The actual lesson target is not operable'); target.click(); true")
      awaitSnapshot(scenario, "The actual control must navigate or complete $step") {
        !it.optBoolean("guideOpen") || it.optString("step") != step || it.optString("targetKey") != before.optString("targetKey") || it.optString("sheetKind") != before.optString("sheetKind")
      }
    }
    val after = snapshot(scenario)
    if (!after.optBoolean("guideOpen") || after.optString("step") != step) return
    fail("The lesson did not complete $step after its real controls were used: $after")
  }

  private fun touchHighlightedControl(scenario: ActivityScenario<MainActivity>, snapshot: JSONObject) {
    val rect = snapshot.getJSONObject("renderDiagnostics").getJSONObject("target").getJSONObject("rect")
    var x = 0f
    var y = 0f
    scenario.onActivity { activity ->
      val webView = findWebView(activity.window.decorView)!!
      val location = IntArray(2)
      webView.getLocationOnScreen(location)
      val scale = webView.width / snapshot.getDouble("viewportExactWidth")
      x = (location[0] + (rect.getDouble("x") + rect.getDouble("width") / 2) * scale).toFloat()
      y = (location[1] + (rect.getDouble("y") + rect.getDouble("height") / 2) * scale).toFloat()
    }
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val downAt = SystemClock.uptimeMillis()
    val down = MotionEvent.obtain(downAt, downAt, MotionEvent.ACTION_DOWN, x, y, 0)
    try {
      instrumentation.sendPointerSync(down)
      SystemClock.sleep(40)
      val up = MotionEvent.obtain(downAt, SystemClock.uptimeMillis(), MotionEvent.ACTION_UP, x, y, 0)
      try { instrumentation.sendPointerSync(up) } finally { up.recycle() }
      instrumentation.waitForIdleSync()
    } finally {
      down.recycle()
    }
  }

  private fun dismissNotice(scenario: ActivityScenario<MainActivity>) {
    evaluate(scenario, "var later=document.querySelector('.wx-sheet[open] [data-wx-do=\"onboarding-later\"]'); if(later) later.click(); true")
  }

  private fun awaitGuide(scenario: ActivityScenario<MainActivity>, id: String, step: String): JSONObject =
    awaitSnapshot(scenario, "$id guide must show $step with a saved step") {
      it.optBoolean("guideOpen") && it.optString("guide") == id && it.optString("step") == step && it.optBoolean("nativeConfirmed")
    }

  private fun awaitSnapshot(scenario: ActivityScenario<MainActivity>, label: String, matches: (JSONObject) -> Boolean): JSONObject {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(35)
    var current = JSONObject()
    while (System.nanoTime() < deadline) {
      current = snapshot(scenario)
      if (current.optString("sheetKind") == "return") evaluate(scenario, "document.querySelector('.wx-sheet [data-wx-close]').click(); true")
      if (matches(current)) return current
      Thread.sleep(150)
    }
    saveScreenshot("guide-failure")
    throw AssertionError("$label: $current")
  }

  private fun saveScreenshot(name: String) {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val output = java.io.File(instrumentation.targetContext.getExternalFilesDir(null), "release-qa").apply { mkdirs() }
    instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
      java.io.FileOutputStream(java.io.File(output, "$name.png")).use { bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
      bitmap.recycle()
    }
  }

  private fun snapshot(scenario: ActivityScenario<MainActivity>): JSONObject {
    var result = JSONObject()
    evaluate(scenario, """
      JSON.stringify((function(){
        try {
        var envelope=JSON.parse(localStorage.getItem('wayfarers-guild-save-v1') || 'null');
        var state=envelope && envelope.state, o=state && state.onboarding, practice=o && o.practice;
        var guide=document.querySelector('.wx-guide[open]'), sheet=document.querySelector('.wx-sheet[open]');
        var card=guide && guide.querySelector('.wx-guide-card'), ring=guide && guide.querySelector('.wx-guide-ring');
        var leave=guide && guide.querySelector('[data-guide-leave]');
        var recovery=guide && guide.querySelector('[data-guide-settings]');
        var currencyNext=guide && guide.querySelector('[data-guide-next]');
        var quote=guide && guide.querySelector('[data-guide-quote]');
        var target=Array.from(document.querySelectorAll('[aria-describedby~="wx-guide-body"]')).find(function(node){return !node.classList.contains('wx-guide');});
        function box(el){return el && el.getBoundingClientRect();}
        function fits(el,action){var r=box(el);return !!r && r.width >= (action?47:1) && r.height >= (action?47:1) && r.left>=-1 && r.top>=-1 && r.right<=innerWidth+1 && r.bottom<=innerHeight+1;}
        function receivesHit(el){var r=box(el),hit=r && document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return !!el && (el===hit || el.contains(hit));}
        function inside(el,parent){var a=box(el),b=box(parent);return !!a && !!b && a.left>=b.left-1 && a.right<=b.right+1 && a.top>=b.top-1 && a.bottom<=b.bottom+1;}
        function renderInfo(el){if(!el)return null;var r=box(el),c=getComputedStyle(el),pop=false;try{pop=el.matches(':popover-open');}catch(error){}return {tag:el.tagName,classes:el.className,parent:el.parentElement?.className,popover:el.getAttribute('popover'),popoverOpen:pop,transform:c.transform,overflow:c.overflow,position:c.position,rect:{x:r.x,y:r.y,width:r.width,height:r.height}};}
        var focused=document.activeElement, unrelated=document.querySelector('[data-wx-nav="guild"]');
        var exits=sheet ? Array.from(sheet.querySelectorAll('[data-wx-close],[data-wx-back]')) : [];
        var capacity=state && state.expedition.version===3 && window.WayfarersProgression ? window.WayfarersProgression.rawRates(state).areas.greenway.travel : 0;
        return {createdAt:state && state.createdAt,areaCount:state ? Object.keys(state.expedition.areas).length : 0,progress:practice && practice.progress.greenway,claimCount:o ? o.rewardClaims.filter(function(id){return id==='greenway';}).length : 0,
          proofCount:practice ? practice.proofs.filter(function(id){return id.startsWith('greenway:');}).length : 0,
          supplyCount:practice ? practice.supplies.filter(function(id){return id==='greenway:upgrade';}).length : 0,travelCapacity:capacity,
          currencyRead:practice?.currencyRead || [],
          boots:state && state.expedition.areas.greenway.ranks.boots,guideOpen:!!guide,guide:guide && guide.dataset.guide,step:guide && guide.dataset.step,replay:!!guide && guide.dataset.replay==='true',
          expeditionVersion:state && state.expedition.version,routeChoice:state && (state.expedition.areas.greenway.choices?.route || state.expedition.areas.greenway.choice),
          renderDiagnostics:{popoverSupported:typeof HTMLElement.prototype.showPopover==='function',guide:renderInfo(guide),parent:renderInfo(guide?.parentElement),sheet:renderInfo(sheet),target:renderInfo(target),card:renderInfo(card),ring:renderInfo(ring),recovery:guide?.querySelector('[data-guide-recovery]')?.textContent},
          sheetKind:sheet ? sheet.dataset.kind:'',width:innerWidth,height:innerHeight,
          viewportExactWidth:visualViewport ? visualViewport.width : innerWidth,
          replayAvailable:!!sheet && !!sheet.querySelector('[data-wx-do="guide-replay:greenway"]'),
          recoveryOptions:!!sheet && !!sheet.querySelector('[data-wx-do="settings"]') && !sheet.querySelector('[data-wx-do^="guide-replay:"],[data-wx-do="caravan-options"],[data-wx-do="atlas"],[data-wx-do^="lesson-start:"]'),
          referenceSteps:sheet ? sheet.querySelectorAll('.wx-lesson-reference').length:0,
          referenceReadOnly:!!sheet && !sheet.querySelector('[data-wx-practice],[data-wx-do^="lesson-start:"]'),
          coachFits:fits(card,false),leaveVisible:!!leave && !leave.hidden && !!leave.getClientRects().length,
          recoveryOperable:fits(recovery,true) && !recovery.hidden && !recovery.disabled && !recovery.closest('[inert]') && receivesHit(recovery),
          currencyNextOperable:fits(currencyNext,true) && !currencyNext.hidden && !currencyNext.disabled && receivesHit(currencyNext),
          currencyNextLabel:currencyNext?.textContent.trim() || '',
          quoteText:quote?.textContent || '',
          quoteVisible:fits(quote,false) && !quote.hidden && inside(quote,card) && quote.scrollWidth<=quote.clientWidth+1 && receivesHit(quote),
          quoteDescribed:!!quote && !!target && (target.getAttribute('aria-describedby') || '').split(' ').includes(quote.id),
          quoteAnnounced:!!quote && !!quote.textContent && (guide.querySelector('[data-guide-announcement]')?.textContent || '').includes(quote.textContent),
          normalPurchaseCopy:!Array.from(document.querySelectorAll('[data-wx-practice],.wx-guide[open],.wx-sheet[open]')).some(function(node){return /\bfree\b/i.test(node.textContent);}),
          spotlightVisible:fits(ring,false) && !ring.hidden,
          targetOperable:fits(target,true) && !target.disabled && !target.closest('[inert]'),targetReceivesHit:receivesHit(target),
          targetKey:target ? [target.tagName,target.getAttribute('data-wx-do'),target.getAttribute('data-upgrade'),target.getAttribute('data-wx-practice'),target.textContent].join('|') : '',
          backgroundBlocked:!!unrelated && (!!unrelated.closest('[inert]') || !receivesHit(unrelated)),
          focusAllowed:!!guide && (guide.contains(focused) || !!target && (target===focused || target.contains(focused)) || exits.includes(focused)),
          activeElement:document.activeElement && document.activeElement.outerHTML.slice(0,250),
          accessible:!!guide && !!document.getElementById(guide.getAttribute('aria-labelledby'))?.textContent && !!document.getElementById(guide.getAttribute('aria-describedby'))?.textContent,
          stepAnnouncement:!!guide && !!guide.querySelector('[data-guide-announcement][aria-live="polite"][aria-atomic="true"]')?.textContent,
          noOverflow:document.documentElement.scrollWidth<=innerWidth+1,
          nativeConfirmed:!!(window.WayfarersCheckpoint && window.WayfarersCheckpoint.confirmed())};
        } catch(error) { return {snapshotError:String(error.stack || error)}; }
      }()))
    """.trimIndent()) { raw -> result = runCatching { JSONObject(JSONArray("[$raw]").getString(0)) }.getOrDefault(JSONObject()) }
    return result
  }

  private fun evaluate(scenario: ActivityScenario<MainActivity>, script: String, callback: (String) -> Unit = {}) {
    val latch=CountDownLatch(1)
    scenario.onActivity { activity ->
      findWebView(activity.window.decorView)!!.evaluateJavascript(script) { raw -> callback(raw); latch.countDown() }
    }
    assertTrue("WebView callback timed out", latch.await(5, TimeUnit.SECONDS))
  }

  private fun findWebView(view: View): WebView? {
    if (view is WebView) return view
    if (view is ViewGroup) for (index in 0 until view.childCount) findWebView(view.getChildAt(index))?.let { return it }
    return null
  }
}
