package me.danielshort.wayfarers

import android.content.pm.ActivityInfo
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
  @Test fun freshTrailPracticeResumesAndItsFreeRankCannotRepeat() {
    assumeTrue("Requires an explicitly reset disposable guild",
      InstrumentationRegistry.getArguments().getString("guildOnboardingQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val opening = awaitGuide(scenario, "greenway", "inspect")
      assertEquals("Fixture must have its real untouched first guide", 0, opening.getInt("progress"))
      assertEquals(0, opening.getInt("claimCount"))
      assertEquals(0, opening.getInt("boots"))
      assertCoach(opening)
      performHighlightedStep(scenario, "inspect")
      val second = awaitGuide(scenario, "greenway", "upgrade")
      assertEquals(1, second.getInt("progress"))
      assertEquals("Inspection must not buy the required practice rank", 0, second.getInt("boots"))
      assertEquals(1, second.getInt("proofCount"))
      assertEquals(0, second.getInt("supplyCount"))
      assertTrue(second.getBoolean("nativeConfirmed"))

      scenario.recreate()
      val resumed = awaitGuide(scenario, "greenway", "upgrade")
      assertEquals(opening.getLong("createdAt"), resumed.getLong("createdAt"))
      var paused = resumed
      repeat(4) {
        if (paused.optBoolean("guideOpen")) {
          val previousSheet = paused.optString("sheetKind")
          scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
          paused = awaitSnapshot(scenario, "Back must close a context sheet or pause the lesson") {
            !it.optBoolean("guideOpen") || it.optString("sheetKind") != previousSheet
          }
          assertEquals("Back must preserve the unfinished action", 1, paused.getInt("progress"))
          assertEquals(0, paused.getInt("boots"))
        }
      }
      assertFalse("Back must provide a safe exit from the required lesson", paused.getBoolean("guideOpen"))
      assertEquals(if (paused.getInt("areaCount") > 1) "areas" else "options", paused.getString("sheetKind"))
      assertEquals(1, paused.getInt("progress"))
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
      assertEquals("The free practice supply is consumed once", 1, purchased.getInt("supplyCount"))
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
      assertEquals(complete.getLong("createdAt"), afterReplay.getLong("createdAt"))
    }
  }

  private fun assertCoach(snapshot: JSONObject) {
    assertTrue("The coach and its visible 48dp exit must fit", snapshot.getBoolean("coachFits"))
    assertTrue("The highlighted real control must be visible", snapshot.getBoolean("spotlightVisible"))
    assertTrue("The real highlighted control must be operable and at least 48dp", snapshot.getBoolean("targetOperable"))
    assertTrue("The coach must not intercept its required game control", snapshot.getBoolean("targetReceivesHit"))
    assertTrue("Unrelated controls must not receive click-through", snapshot.getBoolean("backgroundBlocked"))
    assertTrue("Keyboard focus stays in the allowed target, coach or sheet exit: $snapshot", snapshot.getBoolean("focusAllowed"))
    assertTrue("The guide has an accessible name and description", snapshot.getBoolean("accessible"))
    assertTrue("Each step exposes one atomic polite announcement", snapshot.getBoolean("stepAnnouncement"))
    assertTrue("The game must not acquire horizontal overflow", snapshot.getBoolean("noOverflow"))
  }

  private fun performHighlightedStep(scenario: ActivityScenario<MainActivity>, step: String) {
    repeat(8) {
      val before = snapshot(scenario)
      if (!before.optBoolean("guideOpen") || before.optString("step") != step) return
      assertCoach(before)
      evaluate(scenario, "var target=Array.from(document.querySelectorAll('[aria-describedby~=\"wx-guide-body\"]')).find(function(node){return !node.classList.contains('wx-guide');}); if(!target || target.disabled || target.closest('[inert]')) throw new Error('The actual lesson target is not operable'); target.click(); true")
      awaitSnapshot(scenario, "The actual control must navigate or complete $step") {
        !it.optBoolean("guideOpen") || it.optString("step") != step || it.optString("targetKey") != before.optString("targetKey") || it.optString("sheetKind") != before.optString("sheetKind")
      }
    }
    val after = snapshot(scenario)
    if (!after.optBoolean("guideOpen") || after.optString("step") != step) return
    fail("The lesson did not complete $step after its real controls were used: $after")
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
    throw AssertionError("$label: $current")
  }

  private fun snapshot(scenario: ActivityScenario<MainActivity>): JSONObject {
    var result = JSONObject()
    evaluate(scenario, """
      JSON.stringify((function(){
        var envelope=JSON.parse(localStorage.getItem('wayfarers-guild-save-v1') || 'null');
        var state=envelope && envelope.state, o=state && state.onboarding, practice=o && o.practice;
        var guide=document.querySelector('.wx-guide[open]'), sheet=document.querySelector('.wx-sheet[open]');
        var card=guide && guide.querySelector('.wx-guide-card'), ring=guide && guide.querySelector('.wx-guide-ring');
        var leave=guide && guide.querySelector('[data-guide-leave]');
        var target=Array.from(document.querySelectorAll('[aria-describedby~="wx-guide-body"]')).find(function(node){return !node.classList.contains('wx-guide');});
        function box(el){return el && el.getBoundingClientRect();}
        function fits(el,action){var r=box(el);return !!r && r.width >= (action?47:1) && r.height >= (action?47:1) && r.left>=-1 && r.top>=-1 && r.right<=innerWidth+1 && r.bottom<=innerHeight+1;}
        function receivesHit(el){var r=box(el),hit=r && document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return !!el && (el===hit || el.contains(hit));}
        var focused=document.activeElement, wallet=document.querySelector('[data-wx-wallet]');
        var exits=sheet ? Array.from(sheet.querySelectorAll('[data-wx-close],[data-wx-back]')) : [];
        var capacity=state && window.WayfarersProgression ? window.WayfarersProgression.rawRates(state).areas.greenway.travel : 0;
        return {createdAt:state && state.createdAt,areaCount:state ? Object.keys(state.expedition.areas).length : 0,progress:practice && practice.progress.greenway,claimCount:o ? o.rewardClaims.filter(function(id){return id==='greenway';}).length : 0,
          proofCount:practice ? practice.proofs.filter(function(id){return id.startsWith('greenway:');}).length : 0,
          supplyCount:practice ? practice.supplies.filter(function(id){return id==='greenway:upgrade';}).length : 0,travelCapacity:capacity,
          boots:state && state.expedition.areas.greenway.ranks.boots,guideOpen:!!guide,guide:guide && guide.dataset.guide,step:guide && guide.dataset.step,replay:!!guide && guide.dataset.replay==='true',
          sheetKind:sheet ? sheet.dataset.kind:'',width:innerWidth,height:innerHeight,
          replayAvailable:!!sheet && !!sheet.querySelector('[data-wx-do="guide-replay:greenway"]'),
          referenceSteps:sheet ? sheet.querySelectorAll('.wx-lesson-reference').length:0,
          referenceReadOnly:!!sheet && !sheet.querySelector('[data-wx-practice],[data-wx-do^="lesson-start:"]'),
          coachFits:fits(card,false) && fits(leave,true),spotlightVisible:fits(ring,false) && !ring.hidden,
          targetOperable:fits(target,true) && !target.disabled && !target.closest('[inert]'),targetReceivesHit:receivesHit(target),
          targetKey:target ? [target.tagName,target.getAttribute('data-wx-do'),target.getAttribute('data-upgrade'),target.getAttribute('data-wx-practice'),target.textContent].join('|') : '',
          backgroundBlocked:!wallet || !!wallet.closest('[inert]') || !receivesHit(wallet),
          focusAllowed:!!guide && (guide.contains(focused) || !!target && (target===focused || target.contains(focused)) || exits.includes(focused)),
          activeElement:document.activeElement && document.activeElement.outerHTML.slice(0,250),
          accessible:!!guide && !!document.getElementById(guide.getAttribute('aria-labelledby'))?.textContent && !!document.getElementById(guide.getAttribute('aria-describedby'))?.textContent,
          stepAnnouncement:!!guide && !!guide.querySelector('[data-guide-announcement][aria-live="polite"][aria-atomic="true"]')?.textContent,
          noOverflow:document.documentElement.scrollWidth<=innerWidth+1,
          nativeConfirmed:!!(window.WayfarersCheckpoint && window.WayfarersCheckpoint.confirmed())};
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
