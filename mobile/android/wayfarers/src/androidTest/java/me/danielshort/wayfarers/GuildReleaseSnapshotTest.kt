package me.danielshort.wayfarers

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
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Release snapshots and an explicitly opted-in, backed-up disposable-guild reset. */
@RunWith(AndroidJUnit4::class)
class GuildReleaseSnapshotTest {
  @Test fun resetOnlyAnExplicitDisposableGuildThroughTheSettingsControls() {
    assumeTrue("Requires an explicitly backed-up disposable guild",
      InstrumentationRegistry.getArguments().getString("guildResetQa") == "true")
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val backup = File(context.getExternalFilesDir(null), "release-qa/guild.json")
    assertTrue("Export the native checkpoint before a destructive QA reset", backup.isFile && backup.length() > 100)
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      var deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(35)
      var before = JSONObject()
      while (System.nanoTime() < deadline) {
        before = read(scenario)
        if (before.optBoolean("confirmed")) break
        Thread.sleep(150)
      }
      assertTrue(before.optBoolean("confirmed"))
      assertEquals("The exported backup must belong to this disposable guild",
        before.getLong("createdAt"), JSONObject(backup.readText()).getJSONObject("state").getLong("createdAt"))
      evaluate(scenario, """
        var later=document.querySelector('.wx-sheet[open] [data-wx-do="onboarding-later"]'); if(later) later.click();
        var close=document.querySelector('.wx-sheet[open] [data-wx-close]'); if(close) close.click();
        (document.querySelector('.wx-guide[open] [data-guide-settings]') || document.querySelector('[data-wx-options]')).click();
        document.querySelector('[data-wx-do="settings"]').click(); true;
      """.trimIndent())
      Thread.sleep(400)
      evaluate(scenario, "document.querySelector('[data-testing]').open=true; document.querySelector('[data-open=\"testing-reset\"]').click(); true;")
      Thread.sleep(400)
      evaluate(scenario, "var input=document.querySelector('#wg-reset-confirm'); input.value='RESET'; input.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('[data-confirm-testing-reset]').click(); true;")
      deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(35)
      var after = JSONObject()
      while (System.nanoTime() < deadline) {
        after = read(scenario)
        if (after.optBoolean("confirmed") && after.optLong("createdAt") != before.optLong("createdAt") && after.optString("step") == "currency:coins") break
        Thread.sleep(150)
      }
      assertTrue("Confirmed reset must persist a distinct new guild: $after", after.optBoolean("confirmed") && after.optLong("createdAt") != before.optLong("createdAt"))
      assertEquals("currency:coins", after.optString("step"))
    }
  }

  @Test fun exportTheRealDurableGuildAndSettings() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      var snapshot = JSONObject()
      while (System.nanoTime() < deadline) {
        snapshot = read(scenario)
        if (snapshot.optBoolean("confirmed") && snapshot.optBoolean("valid")) break
        Thread.sleep(150)
      }
      assertTrue("Canonical save must remain valid: $snapshot", snapshot.optBoolean("valid"))
      assertTrue("Native checkpoint must acknowledge the canonical save", snapshot.optBoolean("confirmed"))
      val context = InstrumentationRegistry.getInstrumentation().targetContext
      val checkpoint = GuildCheckpointStore(File(context.filesDir, "guild-checkpoint.json")).read()
      assertNotNull(checkpoint)
      val state = JSONObject(checkpoint!!.text).getJSONObject("state")
      assertEquals(snapshot.getLong("createdAt"), state.getLong("createdAt"))
      val output = File(context.getExternalFilesDir(null), "release-qa").apply { mkdirs() }
      File(output, "guild.json").writeText(checkpoint.text)
      File(output, "snapshot.json").writeText(snapshot.toString(2))
      val prefs = File(context.applicationInfo.dataDir, "shared_prefs")
      prefs.listFiles()?.filter { it.isFile && it.extension == "xml" }?.forEach {
        File(output, it.name).writeBytes(it.readBytes())
      }
      println("GUILD_RELEASE_SNAPSHOT $snapshot")
    }
  }

  private fun read(scenario: ActivityScenario<MainActivity>): JSONObject {
    val latch = CountDownLatch(1)
    var result = JSONObject()
    scenario.onActivity { activity ->
      findWebView(activity.window.decorView)!!.evaluateJavascript("""
        JSON.stringify((function(){
          var raw=localStorage.getItem('wayfarers-guild-save-v1'), envelope=raw&&JSON.parse(raw), state=envelope&&envelope.state;
          var guide=document.querySelector('.wx-guide[open]'), canvas=document.querySelector('[data-wx-canvas]');
          return {version:state&&state.schemaVersion,createdAt:state&&state.createdAt,lastUpdate:state&&state.lastUpdate,
            valid:!!(state&&WayfarersCore.validateState(state).valid),confirmed:!!(window.WayfarersCheckpoint&&WayfarersCheckpoint.confirmed()),
            url:location.href,width:innerWidth,height:innerHeight,noOverflow:document.documentElement.scrollWidth<=innerWidth+1,
            scene:canvas&&canvas.dataset.sceneStatus,area:state&&state.expedition.selectedArea,expeditionVersion:state&&state.expedition.version,
            ranks:state&&state.expedition.areas,resources:state&&state.resources,quiet:localStorage.getItem('wayfarers-guild-quiet'),
            guide:guide&&guide.dataset.guide,step:guide&&guide.dataset.step,
            foundations:document.querySelectorAll('[data-wx-foundation]').length,
            processing:!!document.querySelector('[data-wx-plans]')};
        }()))
      """.trimIndent()) { raw ->
        result = runCatching { JSONObject(JSONArray("[$raw]").getString(0)) }.getOrDefault(JSONObject())
        latch.countDown()
      }
    }
    assertTrue(latch.await(5, TimeUnit.SECONDS))
    return result
  }

  private fun findWebView(view: View): WebView? {
    if (view is WebView) return view
    if (view is ViewGroup) for (index in 0 until view.childCount) findWebView(view.getChildAt(index))?.let { return it }
    return null
  }

  private fun evaluate(scenario: ActivityScenario<MainActivity>, script: String) {
    val latch = CountDownLatch(1)
    scenario.onActivity { activity -> findWebView(activity.window.decorView)!!.evaluateJavascript(script) { latch.countDown() } }
    assertTrue(latch.await(5, TimeUnit.SECONDS))
  }
}
