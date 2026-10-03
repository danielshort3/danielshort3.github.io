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
  @Test fun freshTrailGuideResumesAndCompletionCannotPayTwice() {
    assumeTrue("Requires an explicitly reset disposable guild",
      InstrumentationRegistry.getArguments().getString("guildOnboardingQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val opening = awaitGuide(scenario, "greenway", "purpose")
      assertEquals("Fixture must have its real untouched first guide", 0, opening.getInt("progress"))
      assertEquals(0, opening.getInt("claimCount"))
      assertEquals(0, opening.getInt("boots"))
      assertCoach(opening)
      acknowledge(scenario)
      val second = awaitGuide(scenario, "greenway", "operation")
      assertEquals(1, second.getInt("progress"))
      assertEquals("Instructions never spend or buy a rank", 0, second.getInt("boots"))
      assertTrue(second.getBoolean("nativeConfirmed"))

      scenario.recreate()
      val resumed = awaitGuide(scenario, "greenway", "operation")
      assertEquals(opening.getLong("createdAt"), resumed.getLong("createdAt"))
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      val paused = awaitSnapshot(scenario, "Back must offer Game options without losing step two") {
        !it.optBoolean("guideOpen") && it.optString("sheetKind") == "options"
      }
      assertEquals(1, paused.getInt("progress"))
      scenario.onActivity { it.onBackPressedDispatcher.onBackPressed() }
      awaitGuide(scenario, "greenway", "operation")

      try {
        scenario.onActivity { activity ->
          findWebView(activity.window.decorView)!!.settings.textZoom = 130
          activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        }
        val enlarged = awaitSnapshot(scenario, "Landscape guide must retain its visible action") {
          it.optString("step") == "operation" && it.optInt("width") > it.optInt("height") && it.optBoolean("coachFits")
        }
        assertCoach(enlarged)
        assertEquals(1, enlarged.getInt("progress"))
      } finally {
        scenario.onActivity { activity ->
          findWebView(activity.window.decorView)!!.settings.textZoom = 100
          activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
      }
      awaitGuide(scenario, "greenway", "operation")
      acknowledge(scenario)
      assertCoach(awaitGuide(scenario, "greenway", "next-step"))
      acknowledge(scenario)
      val complete = awaitSnapshot(scenario, "The final acknowledgement must save one learning reward") {
        it.optInt("progress") == 3 && it.optInt("claimCount") == 1 && it.optBoolean("nativeConfirmed")
      }
      assertEquals(0, complete.getInt("boots"))
      scenario.recreate()
      awaitSnapshot(scenario, "Completion and reward survive recreation") {
        it.optInt("progress") == 3 && it.optInt("claimCount") == 1 && !it.optBoolean("guideOpen")
      }
      dismissNotice(scenario)
      evaluate(scenario, "document.querySelector('[data-wx-options]').click(); document.querySelector('[data-wx-do=\"guide-replay:greenway\"]').click(); true")
      for (step in listOf("purpose", "operation", "next-step")) {
        val replay = awaitGuide(scenario, "greenway", step)
        assertTrue("Replay is explicitly labelled", replay.getBoolean("replay"))
        assertEquals(1, replay.getInt("claimCount"))
        acknowledge(scenario)
      }
      val afterReplay = awaitSnapshot(scenario, "Replay must end without granting another reward") { !it.optBoolean("guideOpen") }
      assertEquals(1, afterReplay.getInt("claimCount"))
      assertEquals(complete.getLong("createdAt"), afterReplay.getLong("createdAt"))
    }
  }

  private fun assertCoach(snapshot: JSONObject) {
    assertTrue("The coach card and two 48dp actions must fit", snapshot.getBoolean("coachFits"))
    assertTrue("The highlighted real control must be visible", snapshot.getBoolean("spotlightVisible"))
    assertTrue("Only the modal action receives pointer hits", snapshot.getBoolean("nextReceivesHit"))
    assertTrue("Initial keyboard focus stays inside the guide", snapshot.getBoolean("focusInside"))
    assertTrue("The guide has an accessible name and description", snapshot.getBoolean("accessible"))
    assertTrue("Each step exposes one atomic polite announcement", snapshot.getBoolean("stepAnnouncement"))
    assertTrue("The game must not acquire horizontal overflow", snapshot.getBoolean("noOverflow"))
  }

  private fun acknowledge(scenario: ActivityScenario<MainActivity>) {
    evaluate(scenario, "document.querySelector('.wx-guide[open] [data-guide-next]').click(); true")
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
        var state=envelope && envelope.state, o=state && state.onboarding;
        var guide=document.querySelector('.wx-guide[open]'), sheet=document.querySelector('.wx-sheet[open]');
        var card=guide && guide.querySelector('.wx-guide-card'), ring=guide && guide.querySelector('.wx-guide-ring');
        var next=guide && guide.querySelector('[data-guide-next]'), leave=guide && guide.querySelector('[data-guide-leave]');
        function box(el){return el && el.getBoundingClientRect();}
        function fits(el,action){var r=box(el);return !!r && r.width >= (action?47:1) && r.height >= (action?47:1) && r.left>=-1 && r.top>=-1 && r.right<=innerWidth+1 && r.bottom<=innerHeight+1;}
        var n=box(next), hit=n && document.elementFromPoint(n.x+n.width/2,n.y+n.height/2);
        return {createdAt:state && state.createdAt,progress:o && o.progress.greenway,claimCount:o ? o.rewardClaims.filter(function(id){return id==='greenway';}).length : 0,
          boots:state && state.expedition.areas.greenway.ranks.boots,guideOpen:!!guide,guide:guide && guide.dataset.guide,step:guide && guide.dataset.step,replay:!!guide && guide.dataset.replay==='true',
          sheetKind:sheet ? sheet.dataset.kind:'',width:innerWidth,height:innerHeight,
          coachFits:fits(card,false) && fits(next,true) && fits(leave,true),spotlightVisible:fits(ring,false) && !ring.hidden,
          focusInside:!!guide && guide.contains(document.activeElement),nextReceivesHit:!!next && (next===hit || next.contains(hit)),
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
