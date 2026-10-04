package me.danielshort.wayfarers

import android.content.pm.ActivityInfo
import android.os.Process
import android.view.View
import android.view.ViewGroup
import android.view.accessibility.AccessibilityNodeInfo
import android.webkit.WebView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.lifecycle.Lifecycle
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.util.Base64
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import me.danielshort.wayfarers.content.GuildContentLimits
import me.danielshort.wayfarers.content.GuildContentManifest
import me.danielshort.wayfarers.content.GuildContentStore
import me.danielshort.wayfarers.content.GuildContentVerifier
import me.danielshort.wayfarers.content.GuildContentUpdateManager
import me.danielshort.wayfarers.content.GuildContentUpdateState
import me.danielshort.wayfarers.content.GuildContentHttpsTransport
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel

/** Actual signed-feed UI flow; opt-in because applying a published content version is permanent. */
@RunWith(AndroidJUnit4::class)
class GuildContentUpdateDeviceTest {
  private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
  private val arguments get() = InstrumentationRegistry.getArguments()

  @Test fun captureActualApkSeventeenBeforeInlineUpgrade() {
    assumeTrue(arguments.getString("guildInlineBaselineQa") == "true")
    val context = instrumentation.targetContext
    assertEquals(17L, context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode)
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario, 4)
      val output = File(context.getExternalFilesDir(null), "inline-qa").apply { mkdirs() }
      File(output, "before-apk18.json").writeText(before.toString(2))
      File(output, "native-settings-before-apk18.xml").writeText(nativeSettings())
      exportEvidence("inline-before-apk18", before)
    }
  }

  @Test fun installedApkEighteenKeepsDisabledGuildAndSelectsItsNewBaseline() {
    assumeTrue(arguments.getString("guildInlineUpgradeQa") == "true")
    val context = instrumentation.targetContext
    assertEquals(18L, context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode)
    val before = JSONObject(File(context.getExternalFilesDir(null), "inline-qa/before-apk18.json").readText())
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val after = awaitReady(scenario, 5)
      assertGuildRetained(before.getJSONObject("state"), after.getJSONObject("state"))
      assertEquals(before.getString("resetGeneration"), after.getString("resetGeneration"))
      assertEquals(File(context.getExternalFilesDir(null), "inline-qa/native-settings-before-apk18.xml").readText(), nativeSettings())
      val policy = awaitSnapshot(scenario, "Native committed identity must reach the policy") {
        it.optJSONObject("debugPolicy")?.optJSONObject("state")?.optJSONObject("seen")?.optInt("contentVersion") == 5
      }.getJSONObject("debugPolicy")
      assertFalse(policy.getBoolean("enabled"))
      assertEquals(18, policy.getJSONObject("state").getJSONObject("seen").getInt("apkVersion"))
      val journal = JSONObject(File(context.filesDir, "guild-content/journal.json").readText())
      assertTrue(journal.getString("appUpgradeBackup").startsWith("v4-"))
      assertCheckpoint(after)
      scenario.recreate()
      val restored = awaitReady(scenario, 5)
      assertGuildRetained(after.getJSONObject("state"), restored.getJSONObject("state"))
      exportEvidence("inline-after-apk18", restored)
    }
  }

  @Test fun signedFullRuntimeUpdateHonorsTheActualDebugOptionAndNeverRepeatsReset() {
    assumeTrue("A backed-up disposable APK18 installation is required", arguments.getString("guildDebugUpdateQa") == "true")
    val context = instrumentation.targetContext
    assertEquals(18L, context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode)
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val current = awaitReady(scenario, 5)
      File(context.getExternalFilesDir(null), "inline-qa").mkdirs()
      File(context.getExternalFilesDir(null), "inline-qa/before-debug.json").writeText(current.getString("checkpoint"))
    }
    withIsolatedApplicationStore(baseline = 5) { fixture ->
      val candidate = fixture.releaseCanonical(6)
      fixture.store.stage(candidate.manifest, candidate.archive)
      ActivityScenario.launch(MainActivity::class.java).use { scenario ->
        val before = awaitReady(scenario, 5)
        val enabled = arguments.getString("guildDebugUpdateEnabled") == "true"
        setDebugOption(scenario, enabled)
        val armed = awaitReady(scenario, 5)
        assertGuildRetained(before.getJSONObject("state"), armed.getJSONObject("state"))
        val process = Process.myPid()
        var activity: MainActivity? = null
        scenario.onActivity { activity = it }
        exportEvidence(if (enabled) "debug-enabled-before" else "debug-disabled-before", armed)
        openUpdates(scenario)
        clickNativeButton("Apply game update")
        val after = awaitSnapshot(scenario, "The committed full-runtime update must honor the opted-in policy", 65) {
          it.optLong("contentVersion") == 6L && it.optBoolean("valid") && it.optBoolean("confirmed") &&
            (!enabled || it.getJSONObject("state").getLong("createdAt") != armed.getJSONObject("state").getLong("createdAt"))
        }
        assertEquals(process, Process.myPid())
        scenario.onActivity { assertSame(activity, it) }
        assertEquals(6L, fixture.store.currentVersion())
        assertCheckpoint(after)
        if (!enabled) {
          assertGuildRetained(armed.getJSONObject("state"), after.getJSONObject("state"))
          assertEquals(armed.getString("resetGeneration"), after.getString("resetGeneration"))
        } else {
          assertNotEquals(armed.getString("resetGeneration"), after.getString("resetGeneration"))
          val checkpoint = GuildCheckpointStore(File(context.filesDir, "guild-checkpoint.json")).read()!!
          assertEquals("debug-update-apk18-content6", checkpoint.updateId)
          val nativeBackup = GuildCheckpointStore(File(context.filesDir, "guild-checkpoint.json.before-update-reset")).read()!!
          assertEquals(armed.getJSONObject("state").getLong("createdAt"), nativeBackup.createdAt.toLong())
          val text = read(scenario).getString("updateResetBackup")
          assertEquals(nativeBackup.text, text)
          File(context.getExternalFilesDir(null), "inline-qa/pre-update-backup.json").writeText(text)
          assertEquals(armed.getJSONObject("preferences").toString(), after.getJSONObject("preferences").toString())
          assertJsonEqual("Local keepsakes are retained", armed.getJSONObject("state").getJSONObject("premium").get("owned"), after.getJSONObject("state").getJSONObject("premium").get("owned"))
          assertJsonEqual("Daily reward allowance is retained", armed.getJSONObject("state").getJSONObject("caravan").getJSONArray("completed"), after.getJSONObject("state").getJSONObject("caravan").getJSONArray("completed"))
          completeFirstLesson(scenario)
        }
        val settled = awaitReady(scenario, 6)
        val identity = settled.getJSONObject("state").getLong("createdAt")
        scenario.recreate()
        val recreated = awaitReady(scenario, 6)
        assertEquals(identity, recreated.getJSONObject("state").getLong("createdAt"))
        assertEquals(settled.getString("resetGeneration"), recreated.getString("resetGeneration"))
        val policy = awaitSnapshot(scenario, "The selected committed identity must remain handled") {
          it.optJSONObject("debugPolicy")?.optJSONObject("state")?.optJSONObject("seen")?.optInt("contentVersion") == 6
        }.getJSONObject("debugPolicy")
        assertEquals(enabled, policy.getBoolean("enabled"))
        assertTrue(policy.getJSONObject("state").isNull("pending"))
        exportEvidence(if (enabled) "debug-enabled-after" else "debug-disabled-after", recreated.put("sameActivity", true).put("processIdBefore", process))
      }
    }
  }

  private fun setDebugOption(scenario: ActivityScenario<MainActivity>, enabled: Boolean) {
    evaluate(scenario, "var close=document.querySelector('.wx-sheet[open] [data-wx-close]');if(close)close.click();document.querySelector('[data-wx-options]').click(); document.querySelector('[data-wx-do=\"settings\"]').click(); document.querySelector('[data-testing] summary').click(); true")
    awaitCondition("The actual toggle must wait for native commit") {
      var ready = false
      evaluate(scenario, "!!document.querySelector('[data-reset-after-update]:not(:disabled)')") { ready = it == "true" }
      ready
    }
    evaluate(scenario, "(function(){var input=document.querySelector('[data-reset-after-update]');if(!input||input.disabled)throw new Error('Debug option must be operable after native commit');if(input.checked!==$enabled)input.click();return true;}())")
    if (enabled) evaluate(scenario, "var confirm=document.querySelector('[data-confirm-update-reset]');if(confirm)confirm.click();true")
    evaluate(scenario, "var close=document.querySelector('[data-close-dialog]');if(close)close.click();true")
  }

  private fun completeFirstLesson(scenario: ActivityScenario<MainActivity>) {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(40)
    while (System.nanoTime() < deadline) {
      val snapshot = read(scenario)
      val state = snapshot.optJSONObject("state")
      if (state?.getJSONObject("stations")?.getJSONObject("ranks")?.optInt("station:greenway:path:pathfinding") == 1 &&
        state.getJSONObject("onboarding").getJSONObject("practice").getJSONObject("progress").optInt("greenway") == 3) return
      evaluate(scenario, "var guide=document.querySelector('.wx-guide[open]');var target=guide&&guide.dataset.step.startsWith('currency:')?guide.querySelector('[data-guide-next]:not([hidden])'):Array.from(document.querySelectorAll('[aria-describedby~=\"wx-guide-body\"]')).find(function(n){return !n.classList.contains('wx-guide');});if(target&&!target.disabled&&!target.closest('[inert]'))target.click();true")
      Thread.sleep(180)
    }
    fail("The reset must lead to the actual funded first purchase")
  }

  @Test fun debugBackupDownloadOpensTheRealAndroidDocumentPicker() {
    assumeTrue(arguments.getString("guildDebugBackupQa") == "true")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      awaitReady(scenario, 6)
      evaluate(scenario, "var close=document.querySelector('.wx-sheet[open] [data-wx-close]');if(close)close.click();true")
      completeFirstLesson(scenario)
      setDebugOption(scenario, true)
      val before = awaitReady(scenario, 6)
      evaluate(scenario, "document.querySelector('[data-wx-options]').click();document.querySelector('[data-wx-do=\"settings\"]').click();document.querySelector('[data-testing] summary').click();document.querySelector('[data-update-reset-backup]').click();true")
      awaitCondition("The exact pre-update backup must open the native CreateDocument picker") {
        findAccessible(instrumentation.uiAutomation.rootInActiveWindow) { it.packageName?.toString() == "com.google.android.documentsui" && it.isVisibleToUser } != null
      }
      val pending = MainActivity::class.java.getDeclaredField("pendingExport").apply { isAccessible = true }
      scenario.onActivity { assertEquals(before.getString("updateResetBackup"), pending.get(it)) }
      clickNativeButton("Save")
      awaitCondition("The native export callback must consume the exact backup") {
        var done = false
        scenario.onActivity { done = pending.get(it) == null }
        done
      }
      val after = awaitReady(scenario, 6)
      assertEquals(before.getJSONObject("state").getLong("createdAt"), after.getJSONObject("state").getLong("createdAt"))
      exportEvidence("debug-backup-exported", after)
    }
  }

  @Test fun enabledDebugOptionDoesNotResetAFailedOrInterruptedContentCandidate() {
    assumeTrue(arguments.getString("guildDebugRecoveryQa") == "true")
    captureCurrentDebugBackup()
    withIsolatedApplicationStore(baseline = 6) { fixture ->
      ActivityScenario.launch(MainActivity::class.java).use { scenario ->
        val before = awaitReady(scenario, 6)
        setDebugOption(scenario, true)
        val nativeBackup = File(instrumentation.targetContext.filesDir, "guild-checkpoint.json.before-update-reset").readBytes()
        for (version in listOf(7L, 8L)) {
          val candidate = fixture.release(version)
          fixture.store.stage(candidate.manifest, candidate.archive)
          (instrumentation.targetContext.applicationContext as WayfarersApplication).contentUpdates.refresh()
          awaitCondition("The directly staged fixture must publish its real Ready state") {
            (instrumentation.targetContext.applicationContext as WayfarersApplication).contentUpdates.state.value is GuildContentUpdateState.Ready
          }
          openUpdates(scenario)
          clickNativeButton("Apply game update")
          awaitCondition("The failing candidate must actually enter activation") { fixture.store.currentVersion() == version }
          if (version == 8L) scenario.recreate()
          val restored = awaitReady(scenario, 6, 65)
          awaitCondition("The rollback receipt must be durably consumed before staging another candidate") {
            runCatching {
              val journal = JSONObject(File(fixture.root, "store/journal.json").readText())
              !journal.has("pending") && !journal.has("recovery")
            }.getOrDefault(false)
          }
          assertGuildRetained(before.getJSONObject("state"), restored.getJSONObject("state"))
          assertEquals(before.getString("resetGeneration"), restored.getString("resetGeneration"))
          assertArrayEquals("Failure cannot create a new reset backup", nativeBackup,
            File(instrumentation.targetContext.filesDir, "guild-checkpoint.json.before-update-reset").readBytes())
          assertCheckpoint(restored)
          exportEvidence("debug-failed-candidate-$version", restored)
        }
      }
    }
  }

  @Test fun interruptedAutoResetResumesTheSameSeedAfterNativeFirstLostAcknowledgment() {
    assumeTrue(arguments.getString("guildDebugRecoveryQa") == "true")
    captureCurrentDebugBackup()
    withIsolatedApplicationStore(baseline = 6) { fixture ->
      val candidate = fixture.releaseCanonical(7)
      fixture.store.stage(candidate.manifest, candidate.archive)
      val checkpointFile = File(instrumentation.targetContext.filesDir, "guild-checkpoint.json")
      val checkpoints = GuildCheckpointStore(checkpointFile)
      val blocked = File(checkpointFile.parentFile, checkpointFile.name + ".before-update-reset.pending")
      assertFalse("Never replace an existing player's file for failure injection", blocked.exists())
      check(blocked.mkdir())
      try {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
          val before = awaitReady(scenario, 6)
          setDebugOption(scenario, true)
          openUpdates(scenario)
          clickNativeButton("Apply game update")
          var reset = JSONObject()
          awaitCondition("A committed candidate must leave exactly one recoverable auto-reset seed when native save fails") {
            evaluate(scenario, "localStorage.getItem(WayfarersStorage.RESET_KEY)") { raw ->
              reset = runCatching { JSONObject(JSONArray("[$raw]").getString(0)) }.getOrDefault(JSONObject())
            }
            reset.optString("updateId") == "debug-update-apk18-content7" && reset.optString("text").isNotEmpty()
          }
          assertEquals(before.getJSONObject("state").getLong("createdAt"), checkpoints.read()!!.createdAt.toLong())
          val seed = JSONObject(reset.getString("text")).getJSONObject("state").getLong("createdAt")
          exportEvidence("debug-interrupted-reset", read(scenario).put("nativeGuildBefore", checkpoints.read()!!.createdAt))
          check(blocked.delete())
          // A durable native write with a lost reply must not reroll the seed or
          // replay commerce on the next document. The JS journal is still pending.
          assertTrue(checkpoints.reset(reset.getString("text"), reset.getString("previousId"), reset.getString("id"), reset.getString("updateId")))
          scenario.recreate()
          val resumed = awaitSnapshot(scenario, "The matching authorized seed must resume before readiness", 65) {
            it.optLong("contentVersion") == 7L && it.optBoolean("confirmed") && it.optBoolean("valid") &&
              it.optJSONObject("state")?.optLong("createdAt") == seed
          }
          assertEquals(reset.getString("id"), resumed.getString("resetGeneration"))
          assertEquals("debug-update-apk18-content7", checkpoints.read()!!.updateId)
          completeFirstLesson(scenario)
          scenario.recreate()
          val settled = awaitReady(scenario, 7)
          assertEquals(seed, settled.getJSONObject("state").getLong("createdAt"))
          assertEquals(reset.getString("id"), settled.getString("resetGeneration"))
          assertCheckpoint(settled)
          exportEvidence("debug-interrupted-reset-resumed", settled)
        }
      } finally {
        if (blocked.isDirectory) check(blocked.delete())
      }
    }
  }

  private fun captureCurrentDebugBackup() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val current = awaitReady(scenario, 6)
      File(instrumentation.targetContext.getExternalFilesDir(null), "inline-qa/before-debug.json").writeText(current.getString("checkpoint"))
    }
  }

  @Test fun publishedSchemaEightContentCannotReplaceTheRealInstalledApkSixteenGuild() {
    assumeTrue("Requires the actual published schema-8 feed and a backed-up APK-16 guild",
      arguments.getString("guildOldApkIncompatibleQa") == "true")
    val context = instrumentation.targetContext
    val installedCode = arguments.getString("guildRejectedApkCode")?.toLong() ?: 16L
    val savedSchema = arguments.getString("guildRejectedSaveSchema")?.toInt() ?: 7
    val baselineVersion = arguments.getString("guildRejectedContentBefore")?.toLong() ?: 2L
    val publishedVersion = arguments.getString("guildRejectedContentAfter")?.toLong() ?: 4L
    val minimumCode = arguments.getString("guildRejectedMinimumApk")?.toInt() ?: 17
    assertEquals(installedCode, context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode)
    val exported = File(context.getExternalFilesDir(null), if (installedCode == 17L) "inline-qa/before-apk18.json" else "release-qa/guild.json")
    assertTrue("Export the real older guild before checking the incompatible feed", exported.isFile)
    val backup = JSONObject(exported.readText()).let { if (it.has("checkpoint")) JSONObject(it.getString("checkpoint")) else it }
    assertEquals(savedSchema, backup.getInt("version"))
    val envelope = GuildContentHttpsTransport().manifest(
      "https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-updates/latest-content.json")
    val published = GuildContentVerifier(me.danielshort.app.BuildConfig.CONTENT_PUBLIC_KEY,
      "me.danielshort.wayfarers", 1, 18, 8).verify(envelope)
    val payload = JSONObject(String(Base64.getDecoder().decode(JSONObject(envelope).getString("payload")), Charsets.UTF_8))
    assertEquals("This gate must read the actual new published signed content", publishedVersion, published.contentVersion)
    assertEquals(8, payload.getInt("saveSchema"))
    assertEquals(minimumCode, published.minAppVersionCode)
    val app = context.applicationContext as WayfarersApplication
    // APK 16's released verifier reports its signature/compatibility rejection
    // with this exact message. A transport error must not pass this gate.
    val compatibilityMessage = if (installedCode <= 16L) "This game update could not be verified or is incompatible with this app."
      else "This game update requires a compatible app version. Check for an app update first."
    val rejection = assertThrows(Exception::class.java) { app.contentStore.verifyManifest(envelope) }
    assertEquals(compatibilityMessage, rejection.message)
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario, baselineVersion)
      assertEquals(backup.getJSONObject("state").getLong("createdAt"),
        before.getJSONObject("state").getLong("createdAt"))
      val settings = nativeSettings()
      val pid = Process.myPid()
      var activity: MainActivity? = null
      scenario.onActivity { activity = it }
      exportEvidence("before-public-incompatible-check", before)
      openUpdates(scenario)
      clickNativeButton("Check game updates")
      awaitCondition("The real APK-16 update manager must reject the incompatible public feed") {
        app.contentUpdates.state.value is GuildContentUpdateState.Error
      }
      val failure = app.contentUpdates.state.value as GuildContentUpdateState.Error
      assertEquals("Reject the authenticated incompatible feed, not a transient transport failure", compatibilityMessage, failure.message)
      awaitCondition("The exact compatibility error must paint in the real update screen") {
        findAccessible(instrumentation.uiAutomation.rootInActiveWindow) {
          it.text?.toString() == failure.message && it.isVisibleToUser
        } != null
      }
      assertNull(findAccessible(instrumentation.uiAutomation.rootInActiveWindow) {
        it.text?.toString() in listOf("Apply game update", "Download game update") && it.isVisibleToUser && it.isEnabled
      })
      assertEquals(baselineVersion, app.contentStore.currentVersion())
      assertNull(app.contentStore.stagedManifest())
      val output = File(context.getExternalFilesDir(null), "content-update-qa").apply { mkdirs() }
      File(output, "public-incompatible-envelope.json").writeText(envelope)
      File(output, "public-incompatible-result.json").writeText(JSONObject().put("installedApk", installedCode)
        .put("publishedContent", published.contentVersion).put("saveSchema", payload.getInt("saveSchema"))
        .put("minimumApk", published.minAppVersionCode).put("error", failure.message).toString(2))
      File(output, "public-incompatible-ui.png").outputStream().use { stream ->
        check(instrumentation.uiAutomation.takeScreenshot().compress(android.graphics.Bitmap.CompressFormat.PNG, 100, stream))
      }
      clickNativeButton("Back to guild")
      val after = awaitReady(scenario, baselineVersion)
      assertEquals(pid, Process.myPid())
      scenario.onActivity { assertSame(activity, it) }
      assertGuildRetained(before.getJSONObject("state"), after.getJSONObject("state"))
      assertEquals(settings, nativeSettings())
      val checkpoint = GuildCheckpointStore(File(context.filesDir, "guild-checkpoint.json")).read()
      assertNotNull(checkpoint)
      assertEquals(savedSchema, JSONObject(checkpoint!!.text).getInt("version"))
      assertGuildRetained(after.getJSONObject("state"), JSONObject(checkpoint.text).getJSONObject("state"))
      exportEvidence("after-public-incompatible-check", after)
      if (installedCode == 17L) {
        val inline = File(context.getExternalFilesDir(null), "inline-qa").apply { mkdirs() }
        File(inline, "before-apk18.json").writeText(after.toString(2))
        File(inline, "native-settings-before-apk18.xml").writeText(nativeSettings())
      }
    }
  }

  @Test fun installedApkUpgradePreservesTheRealRetainedGuildAndSettings() {
    assumeTrue("Requires the actual installed APK-16 to APK-17 upgrade and an exported baseline",
      arguments.getString("guildAppUpgradeQa") == "true")
    val context = instrumentation.targetContext
    assertEquals("Check the installed package, not the test APK's build constants", 17L,
      context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode)
    val exported = File(context.getExternalFilesDir(null), "release-qa/guild.json")
    val exportedSettings = File(context.getExternalFilesDir(null), "release-qa/native-settings.xml")
    assertTrue("Export the real prior guild and settings before installing", exported.isFile && exportedSettings.isFile)
    val baseline = JSONObject(exported.readText())
    assertEquals("This gate must start from the published schema-7 app", 7, baseline.getInt("version"))
    val before = if (arguments.getString("guildUpgradeUsePublicCheckBaseline") == "true") {
      // The live guild may legitimately advance after an earlier export. Use the
      // actual durable APK-16 snapshot captured immediately before its public upgrade.
      val checked = JSONObject(File(context.getExternalFilesDir(null),
        "content-update-qa/after-public-incompatible-check.json").readText())
      assertEquals(2L, checked.getLong("contentVersion"))
      assertTrue(checked.getBoolean("confirmed") && checked.getBoolean("valid"))
      assertEquals(7, checked.getJSONObject("state").getInt("schemaVersion"))
      assertEquals(baseline.getJSONObject("state").getLong("createdAt"),
        checked.getJSONObject("state").getLong("createdAt"))
      checked.getJSONObject("state")
    } else baseline.getJSONObject("state")
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val after = awaitReady(scenario, 3)
      val state = after.getJSONObject("state")
      assertEquals(8, state.getInt("schemaVersion"))
      assertEquals("The old run keeps its actual economy until a confirmed reset",
        before.getJSONObject("expedition").getInt("version"), state.getJSONObject("expedition").getInt("version"))
      assertGuildRetained(before, state)
      val stations = state.getJSONObject("stations")
      assertEquals("A retained run must not fabricate built stations", 0, stations.getJSONArray("built").length())
      assertEquals(0, stations.getJSONArray("unlocked").length())
      for (id in stations.getJSONObject("ranks").keys()) assertEquals(0, stations.getJSONObject("ranks").getInt(id))
      assertEquals("The native update mode survives APK replacement", exportedSettings.readText(), nativeSettings())
      assertCheckpoint(after)
      val migrationFile = File(context.filesDir, "guild-checkpoint.json.before-schema-8")
      val recovery = GuildCheckpointStore(migrationFile).read()
      assertNotNull("Keep the exact prior checkpoint for recovery", recovery)
      val old = JSONObject(recovery!!.text)
      assertEquals(7, old.getInt("version"))
      assertGuildRetained(before, old.getJSONObject("state"))
      val backupBytes = migrationFile.readBytes()
      val contentRoot = File(context.filesDir, "guild-content")
      val journal = JSONObject(File(contentRoot, "journal.json").readText())
      val retainedContent = journal.getString("appUpgradeBackup")
      assertTrue("Keep the incompatible old signed snapshot without executing it", retainedContent.startsWith("v2-"))
      assertTrue(File(contentRoot, "bundles/$retainedContent/manifest-envelope.json").isFile)
      assertFalse("Select the compatible APK baseline", journal.has("active"))
      assertTrue(journal.getLong("highest") >= 3)
      val output = File(context.getExternalFilesDir(null), "content-update-qa").apply { mkdirs() }
      File(output, "after-app-upgrade-journal.json").writeText(journal.toString(2))
      File(output, "after-app-upgrade-schema7-backup.json").writeText(recovery.text)
      exportEvidence("after-app-upgrade", after)
      scenario.recreate()
      val restored = awaitReady(scenario, 3)
      assertGuildRetained(state, restored.getJSONObject("state"))
      assertArrayEquals("The first migration recovery backup remains immutable", backupBytes, migrationFile.readBytes())
      assertEquals(exportedSettings.readText(), nativeSettings())
      exportEvidence("after-app-upgrade-recreation", restored)
    }
  }

  @Test fun configureBackedUpDisposableBaselineThroughItsActualUpdateModeControl() {
    assumeTrue("Requires an explicitly backed-up disposable baseline",
      arguments.getString("guildPrepareBaselineQa") == "true")
    val context = instrumentation.targetContext
    val exported = File(context.getExternalFilesDir(null), "release-qa/guild.json")
    assertTrue("Export this guild before preparing the baseline", exported.isFile)
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitSnapshot(scenario, "The baseline must have a confirmed guild") { it.optBoolean("confirmed") && it.optBoolean("valid") }
      assertEquals(JSONObject(exported.readText()).getJSONObject("state").getLong("createdAt"), before.getJSONObject("state").getLong("createdAt"))
      openUpdates(scenario)
      clickNativeButton("Manual")
      awaitCondition("The actual update preference must finish its asynchronous disk write") {
        nativeSettings().contains("check-app-updates-on-launch")
      }
      clickNativeButton("Back to guild")
      val after = awaitSnapshot(scenario, "The actual mode control must retain the guild") { it.optBoolean("confirmed") && it.optBoolean("valid") }
      assertGuildRetained(before.getJSONObject("state"), after.getJSONObject("state"))
      assertTrue("The baseline must now have explicit persisted native preferences", nativeSettings().contains("check-app-updates-on-launch"))
      exportEvidence("prepared-baseline", after)
    }
  }

  @Test fun failedReadyRollsBackThroughTheActualActivityWithoutLosingTheGuild() {
    assumeTrue("Requires an explicitly backed-up disposable device",
      arguments.getString("guildContentFailureQa") == "true")
    withIsolatedApplicationStore { fixture ->
      val second = fixture.release(2)
      fixture.store.stage(second.manifest, second.archive)
      ActivityScenario.launch(MainActivity::class.java).use { scenario ->
        val before = awaitReady(scenario, 1)
        val process = Process.myPid()
        var activity: MainActivity? = null
        scenario.onActivity { activity = it }
        exportEvidence("before-failed-ready", before)
        openUpdates(scenario)
        clickNativeButton("Apply game update")
        awaitCondition("The staged signed candidate must actually enter activation") { fixture.store.currentVersion() == 2L }
        val after = awaitReady(scenario, 1, timeoutSeconds = 65)
        assertEquals("Failure recovery cannot restart the Android process", process, Process.myPid())
        scenario.onActivity { assertSame(activity, it) }
        assertGuildRetained(before.getJSONObject("state"), after.getJSONObject("state"))
        assertEquals(before.getJSONObject("preferences").toString(), after.getJSONObject("preferences").toString())
        assertCheckpoint(after)
        assertEquals(1L, fixture.store.currentVersion())
        assertNull("The failed candidate must not remain offered for Apply", fixture.store.stagedManifest())
        assertThrows(Exception::class.java) { fixture.store.stage(second.manifest, second.archive) }
        exportEvidence("after-failed-ready-rollback", after)
      }
    }
  }

  @Test fun backgroundAndRecreationDuringActivationRecoverThePreviousGuild() {
    assumeTrue("Requires an explicitly backed-up disposable device",
      arguments.getString("guildContentFailureQa") == "true")
    withIsolatedApplicationStore { fixture ->
      val second = fixture.release(2)
      fixture.store.stage(second.manifest, second.archive)
      ActivityScenario.launch(MainActivity::class.java).use { scenario ->
        val before = awaitReady(scenario, 1)
        val process = Process.myPid()
        exportEvidence("before-interrupted-activation", before)
        openUpdates(scenario)
        clickNativeButton("Apply game update")
        awaitCondition("The candidate must be pending before lifecycle interruption") { fixture.store.currentVersion() == 2L }
        scenario.moveToState(Lifecycle.State.CREATED)
        scenario.moveToState(Lifecycle.State.RESUMED)
        scenario.recreate()
        val after = awaitReady(scenario, 1, timeoutSeconds = 65)
        assertEquals(process, Process.myPid())
        assertGuildRetained(before.getJSONObject("state"), after.getJSONObject("state"))
        assertEquals(before.getJSONObject("preferences").toString(), after.getJSONObject("preferences").toString())
        assertCheckpoint(after)
        assertEquals(1L, fixture.store.currentVersion())
        assertNull(fixture.store.startupRecovery())
        exportEvidence("after-interrupted-activation", after)
      }
    }
  }

  @Test fun badSignatureAndTamperedArchiveCannotReplaceContentOnDevice() {
    withIsolatedStore { fixture ->
      val signed = fixture.release(2)
      val envelope = JSONObject(signed.manifest.envelope)
      val signature = Base64.getDecoder().decode(envelope.getString("signature"))
      signature[signature.lastIndex] = (signature.last().toInt() xor 1).toByte()
      envelope.put("signature", Base64.getEncoder().encodeToString(signature))
      assertThrows(Exception::class.java) { fixture.verifier.verify(envelope.toString()) }
      assertEquals(1L, fixture.store.currentVersion())
      assertNull(fixture.store.stagedManifest())

      val tampered = File(fixture.root, "tampered.zip")
      val bytes = signed.archive.readBytes()
      bytes[bytes.size / 2] = (bytes[bytes.size / 2].toInt() xor 1).toByte()
      tampered.writeBytes(bytes)
      assertThrows(Exception::class.java) { fixture.store.stage(signed.manifest, tampered) }
      assertEquals("Unverified bytes cannot change the active version", 1L, fixture.store.currentVersion())
      assertNull(fixture.store.stagedManifest())

      fixture.store.stage(signed.manifest, signed.archive)
      assertEquals("A verified download alone cannot activate content", 1L, fixture.store.currentVersion())
      assertEquals(2L, fixture.store.stagedManifest()!!.contentVersion)
    }
  }

  @Test fun interruptedApplyRestoresThePairedCheckpointAndRejectsLateCommit() {
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val live = awaitSnapshot(scenario, "A valid real guild is needed for the isolated recovery copy") {
        it.optBoolean("confirmed") && it.optBoolean("valid")
      }
      val rawCheckpoint = File(instrumentation.targetContext.filesDir, "guild-checkpoint.json").readBytes()
      val storage = JSONObject().put("wayfarers-guild-save-v1", live.getString("checkpoint"))
      live.getJSONObject("preferences").let { preferences ->
        for (key in preferences.keys()) if (!preferences.isNull(key)) storage.put("wayfarers-guild-$key", preferences.getString(key))
      }
      withIsolatedStore { fixture ->
        val second = fixture.release(2)
        fixture.store.stage(second.manifest, second.archive)
        val verified = fixture.store.beginApply(rawCheckpoint, storage.toString())
        assertTrue(fixture.store.commitApply(verified.id))
        val oldPage = fixture.store.session()
        val third = fixture.release(3)
        fixture.store.stage(third.manifest, third.archive)
        val pending = fixture.store.beginApply(rawCheckpoint, storage.toString())
        assertEquals(3L, pending.contentVersion)
        assertEquals("An existing page keeps its immutable old assets", "<!doctype html><title>content 2</title>", oldPage.open("wayfarers/index.html")!!.bufferedReader().use { it.readText() })
        assertEquals("The new page exclusively reads new assets", "<!doctype html><title>content 3</title>", pending.open("wayfarers/index.html")!!.bufferedReader().use { it.readText() })

        // A new instance exercises persisted recovery, as after process death;
        // the actual player's store and checkpoint are never modified here.
        val restarted = GuildContentStore(File(fixture.root, "store"), fixture.verifier)
        val recovery = restarted.startupRecovery()
        assertNotNull("Unacknowledged content cannot survive as active", recovery)
        assertEquals(2L, restarted.currentVersion())
        assertArrayEquals(rawCheckpoint, recovery!!.checkpointBytes)
        assertEquals(storage.toString(), recovery.localStorageJson)
        assertFalse("A late ready message cannot revive a rolled-back page", restarted.commitApply(pending.id))
        assertThrows(Exception::class.java) { restarted.stage(third.manifest, third.archive) }

        // Simulate a failed new document saving later data, then restore the
        // old native checkpoint through the explicit content rollback API.
        val nativeFile = File(fixture.root, "isolated-checkpoint.json")
        nativeFile.writeBytes(rawCheckpoint)
        val nativeStore = GuildCheckpointStore(nativeFile)
        val original = nativeStore.read()!!
        val changed = JSONObject(original.text)
        changed.put("savedAt", changed.getDouble("savedAt") + 1000)
        changed.getJSONObject("state").put("lastUpdate", changed.getJSONObject("state").getDouble("lastUpdate") + 1000)
        assertTrue(nativeStore.write(changed.toString(), generation = original.generation))
        assertFalse("Ordinary writes correctly refuse a timestamp rewind", nativeStore.write(original.text, generation = original.generation))
        assertTrue("Guarded content rollback must restore the paired earlier checkpoint", nativeStore.restoreForContentUpdate(recovery.checkpointBytes))
        assertArrayEquals(rawCheckpoint, nativeFile.readBytes())
        assertTrue(restarted.consumeRecovery(recovery.token))
        assertFalse("Recovery acknowledgment is once-only", restarted.consumeRecovery(recovery.token))
        assertNull(restarted.startupRecovery())
        assertEquals(2L, restarted.currentVersion())
      }
      assertGuildRetained(live.getJSONObject("state"), read(scenario).getJSONObject("state"))
    }
  }

  @Test fun publishedContentAppliesWithoutReplacingTheActivityOrGuild() {
    assumeTrue("Requires an explicitly backed-up device and a published signed content update",
      arguments.getString("guildContentUpdateQa") == "true")
    val expectedBefore = arguments.getString("guildContentBefore")?.toLong() ?: 3L
    val expectedAfter = arguments.getString("guildContentAfter")?.toLong() ?: 4L
    assertTrue(expectedAfter > expectedBefore)
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario, expectedBefore)
      val originalPid = Process.myPid()
      var originalActivity: MainActivity? = null
      scenario.onActivity { originalActivity = it }
      val originalSettings = nativeSettings()
      exportEvidence("before-live-apply", before)
      openUpdates(scenario)
      clickNativeButton("Check game updates")
      clickNativeButton("Download game update", timeoutSeconds = 40)
      clickNativeButton("Apply game update", timeoutSeconds = 180)
      val after = awaitReady(scenario, expectedAfter, timeoutSeconds = 55)
      assertEquals("Content activation must keep the Android process alive", originalPid, Process.myPid())
      scenario.onActivity { assertSame("Content activation must keep the current Activity", originalActivity, it) }
      assertEquals("The stable appassets origin owns the existing guild", before.getString("url"), after.getString("url"))
      assertNotEquals("The new content must load in a new document", before.getDouble("documentTimeOrigin"), after.getDouble("documentTimeOrigin"))
      assertGuildRetained(before.getJSONObject("state"), after.getJSONObject("state"))
      assertEquals(before.getJSONObject("preferences").toString(), after.getJSONObject("preferences").toString())
      assertEquals("Native preferences cannot change while applying game content", originalSettings, nativeSettings())
      assertCheckpoint(after)
      exportEvidence("after-live-apply", after.put("processIdBefore", originalPid).put("processIdAfter", Process.myPid()).put("sameActivity", true))

      // Normal activity destruction is distinct from live Apply, and must continue
      // loading the committed bundle without changing its guild or current version.
      scenario.recreate()
      val recreated = awaitReady(scenario, expectedAfter)
      assertGuildRetained(after.getJSONObject("state"), recreated.getJSONObject("state"))
      assertCheckpoint(recreated)
      exportEvidence("after-recreation", recreated)
    }
  }

  @Test fun committedContentAndGuildSurviveOfflineColdLaunchAndRotation() {
    assumeTrue("Requires externally disabled networking and an applied signed content version",
      arguments.getString("guildContentOfflineQa") == "true")
    val expectedVersion = arguments.getString("guildContentAfter")?.toLong() ?: 4L
    ActivityScenario.launch(MainActivity::class.java).use { scenario ->
      val before = awaitReady(scenario, expectedVersion)
      assertCheckpoint(before)
      try {
        scenario.onActivity { it.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE }
        val landscape = awaitSnapshot(scenario, "Committed offline content must render in landscape") {
          it.optLong("contentVersion") == expectedVersion && it.optBoolean("confirmed") &&
            it.optBoolean("valid") && it.optInt("width") > it.optInt("height") && it.optBoolean("noOverflow")
        }
        assertGuildRetained(before.getJSONObject("state"), landscape.getJSONObject("state"))
        scenario.onActivity { it.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT }
        val portrait = awaitSnapshot(scenario, "Committed offline content must restore portrait") {
          it.optLong("contentVersion") == expectedVersion && it.optBoolean("confirmed") &&
            it.optBoolean("valid") && it.optInt("height") > it.optInt("width") && it.optBoolean("noOverflow")
        }
        assertGuildRetained(before.getJSONObject("state"), portrait.getJSONObject("state"))
        assertCheckpoint(portrait)
        exportEvidence("offline-cold-launch-and-rotation", portrait)
      } finally {
        scenario.onActivity { it.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED }
      }
    }
  }

  private fun openUpdates(scenario: ActivityScenario<MainActivity>) {
    evaluate(scenario, """
      (function(){
        var later=document.querySelector('.wx-sheet[open] [data-wx-do="onboarding-later"]'); if(later) later.click();
        var close=document.querySelector('.wx-sheet[open] [data-wx-close]'); if(close) close.click();
        var button=document.querySelector('.wx-guide[open] [data-guide-settings]') || document.querySelector('[data-wx-options]') || document.querySelector('.wg-exit');
        if(!button || button.disabled) throw new Error('Actual game settings control missing');
        button.click();
        if(button.classList.contains('wg-exit'))return true;
        var updates=document.querySelector('[data-wx-do="native-options"]');
        if(!updates || updates.disabled) throw new Error('Actual native updates control missing');
        updates.click(); return true;
      }())
    """.trimIndent())
  }

  private fun clickNativeButton(label: String, timeoutSeconds: Long = 30) {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSeconds)
    while (System.nanoTime() < deadline) {
      val root = instrumentation.uiAutomation.rootInActiveWindow
      val match = findAccessible(root) { it.text?.toString()?.equals(label, ignoreCase = true) == true || it.contentDescription?.toString()?.equals(label, ignoreCase = true) == true }
      var target = match
      while (target != null && !target.isClickable) target = target.parent
      if (target != null && target.isEnabled && target.isVisibleToUser && target.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return
      if (match == null) findAccessible(root) { it.isScrollable && it.isVisibleToUser }
        ?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
      Thread.sleep(200)
    }
    val texts = mutableListOf<String>()
    findAccessible(instrumentation.uiAutomation.rootInActiveWindow) { node -> node.text?.toString()?.let(texts::add); false }
    fail("The actual native button '$label' did not become operable. Visible UI: $texts")
  }

  private fun findAccessible(node: AccessibilityNodeInfo?, matches: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
    if (node == null) return null
    if (matches(node)) return node
    for (index in 0 until node.childCount) findAccessible(node.getChild(index), matches)?.let { return it }
    return null
  }

  private fun awaitCondition(label: String, matches: () -> Boolean) {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
    while (System.nanoTime() < deadline) { if (matches()) return; Thread.sleep(100) }
    fail(label)
  }

  private fun awaitReady(scenario: ActivityScenario<MainActivity>, version: Long, timeoutSeconds: Long = 45): JSONObject =
    awaitSnapshot(scenario, "Signed content $version must load a valid guild with a durable native checkpoint", timeoutSeconds) {
      it.optLong("contentVersion") == version && it.optBoolean("confirmed") && it.optBoolean("valid") &&
        it.optString("scene") == "ready" && it.optBoolean("noOverflow")
    }

  private fun awaitSnapshot(scenario: ActivityScenario<MainActivity>, label: String, timeoutSeconds: Long = 45, matches: (JSONObject) -> Boolean): JSONObject {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSeconds)
    var last = JSONObject()
    while (System.nanoTime() < deadline) {
      last = read(scenario)
      if (matches(last)) return last
      Thread.sleep(150)
    }
    exportEvidence("failed-snapshot", last)
    throw AssertionError("$label. Latest snapshot: ${JSONObject(last.toString()).apply { remove("state"); remove("checkpoint") }}")
  }

  private fun read(scenario: ActivityScenario<MainActivity>): JSONObject {
    var snapshot = JSONObject()
    evaluate(scenario, """
      JSON.stringify((function(){
        try {
          var text=window.WayfarersCheckpoint?.snapshot() || localStorage.getItem('wayfarers-guild-save-v1');
          var envelope=text && JSON.parse(text), state=envelope?.state;
          var scene=document.querySelector('[data-wx-canvas]');
          var world=document.querySelector('[data-wx-station-world]'), clip=world?.getBoundingClientRect();
          var visible=world ? Array.from(world.querySelectorAll('canvas')).filter(function(canvas){
            var rect=canvas.getBoundingClientRect();return rect.width>0&&rect.height>0&&rect.right>Math.max(0,clip.left)&&
              rect.left<Math.min(innerWidth,clip.right)&&rect.bottom>Math.max(0,clip.top)&&rect.top<Math.min(innerHeight,clip.bottom);
          }) : [];
          var ready=world ? visible.length>0&&visible.every(function(canvas){return canvas.dataset.sceneStatus==='ready';}) : scene?.dataset.sceneStatus==='ready';
          return {contentVersion:window.WayfarersContent?.version || 0,
            documentTimeOrigin:performance.timeOrigin,url:location.href,width:innerWidth,height:innerHeight,
            state:state,checkpoint:text,valid:!!(state && WayfarersCore.validateState(state).valid),
            confirmed:!!window.WayfarersCheckpoint?.confirmed(),scene:ready?'ready':'loading',
            visibleStationScenes:visible.map(function(canvas){return {id:canvas.dataset.stationId,status:canvas.dataset.sceneStatus};}),
            noOverflow:document.documentElement.scrollWidth<=innerWidth+1,
            preferences:{quiet:localStorage.getItem('wayfarers-guild-quiet'),sound:localStorage.getItem('wayfarers-guild-sound')},
            debugPolicy:window.WayfarersDebugUpdates?.createPolicy().read(),
            updateResetBackup:localStorage.getItem(window.WayfarersStorage?.UPDATE_RESET_BACKUP_KEY || 'missing-key'),
            resetGeneration:window.WayfarersCheckpoint?.generation() || ''};
        } catch(error) { return {error:String(error.stack || error)}; }
      }()))
    """.trimIndent()) { raw -> snapshot = runCatching { JSONObject(JSONArray("[$raw]").getString(0)) }.getOrDefault(JSONObject()) }
    return snapshot
  }

  private fun assertGuildRetained(before: JSONObject, after: JSONObject) {
    assertEquals("Content updates retain guild identity", before.getLong("createdAt"), after.getLong("createdAt"))
    assertTrue("Content activation cannot rewind saved production", after.getLong("lastUpdate") >= before.getLong("lastUpdate"))
    for (key in listOf("upgrades", "refitUpgrades", "legacy")) {
      if (before.has(key)) assertJsonEqual("$key stays unchanged", before.get(key), after.get(key))
    }
    for ((group, keys) in listOf("premium" to listOf("owned", "equipped"), "luck" to listOf("owned", "active"))) {
      if (before.has(group)) for (key in keys) {
        assertJsonEqual("$group $key survives", before.getJSONObject(group).get(key), after.getJSONObject(group).get(key))
      }
    }
    val previousAreas = before.getJSONObject("expedition").getJSONObject("areas")
    val nextAreas = after.getJSONObject("expedition").getJSONObject("areas")
    for (id in previousAreas.keys()) {
      assertTrue("Previously unlocked $id remains available", nextAreas.has(id))
      for (key in listOf("ranks", "highRanks", "choice", "choices", "specialization", "plans")) {
        val area = previousAreas.getJSONObject(id)
        if (area.has(key)) assertJsonEqual("$id $key must survive", area.get(key), nextAreas.getJSONObject(id).get(key))
      }
    }
    if (before.has("stations")) {
      val previous = before.getJSONObject("stations")
      val next = after.getJSONObject("stations")
      assertEquals("Station ledger version survives", previous.getInt("version"), next.getInt("version"))
      assertTrue("Station revision cannot rewind", next.getLong("revision") >= previous.getLong("revision"))
      for (key in listOf("built", "unlocked", "areaUnlocked", "introduced")) {
        val retained = next.getJSONArray(key).let { array -> (0 until array.length()).map { array.getString(it) }.toSet() }
        val owned = previous.getJSONArray(key)
        for (index in 0 until owned.length()) assertTrue("Station $key ${owned.getString(index)} survives", owned.getString(index) in retained)
      }
      for (key in listOf("ranks", "highRanks", "areaRanks", "areaHighRanks", "selected")) {
        val owned = previous.getJSONObject(key)
        val retained = next.getJSONObject(key)
        for (id in owned.keys()) assertJsonEqual("Station $key.$id survives", owned.get(id), retained.get(id))
      }
      val oldMastery = previous.getJSONObject("mastery")
      for (id in oldMastery.keys()) assertTrue("Station mastery $id cannot rewind", next.getJSONObject("mastery").getDouble(id) >= oldMastery.getDouble(id))
      val oldOutput = previous.getJSONObject("output")
      for (id in oldOutput.keys()) {
        val old = oldOutput.getJSONObject(id)
        val earned = next.getJSONObject("output").getJSONObject(id)
        assertTrue("Lifetime station output $id cannot rewind", old.getDouble("m") == 0.0 ||
          earned.getDouble("m") > 0 && (earned.getDouble("e") > old.getDouble("e") ||
            earned.getDouble("e") == old.getDouble("e") && earned.getDouble("m") >= old.getDouble("m")))
      }
      val oldEncounter = previous.getJSONObject("encounter")
      val nextEncounter = next.getJSONObject("encounter")
      assertTrue("Encounter award sequence cannot rewind", nextEncounter.getLong("sequence") >= oldEncounter.getLong("sequence"))
      // Production phases and boost timers advance normally during downloads.
      if (nextEncounter.getLong("sequence") == oldEncounter.getLong("sequence")) {
        assertJsonEqual("Encounter RNG and last reward survive", oldEncounter.get("rng"), nextEncounter.get("rng"))
        assertJsonEqual("Encounter last reward survives", oldEncounter.get("lastReward"), nextEncounter.get("lastReward"))
      }
    }
    val previousCollection = before.getJSONObject("collection")
    val nextCollection = after.getJSONObject("collection")
    for (key in listOf("gear", "decks", "activeDeck", "equipped")) {
      assertJsonEqual("Collection $key survives the content update", previousCollection.get(key), nextCollection.get(key))
    }
    val previousCards = previousCollection.getJSONObject("cards")
    val nextCards = nextCollection.getJSONObject("cards")
    for (id in previousCards.keys()) {
      assertTrue("Owned card $id is retained", nextCards.has(id))
      // Ordinary timed drops may add duplicates while the update downloads.
      val old = previousCards.getJSONObject(id)
      val next = nextCards.getJSONObject(id)
      for (key in old.keys()) {
        if (key in listOf("copies", "count", "owned")) assertTrue("$id $key cannot be lost", next.getDouble(key) >= old.getDouble(key))
        else assertJsonEqual("Card $id $key survives", old.get(key), next.get(key))
      }
    }
  }

  private fun assertJsonEqual(label: String, before: Any, after: Any) {
    when {
      before is JSONObject && after is JSONObject -> {
        assertEquals(label, before.keys().asSequence().toSet(), after.keys().asSequence().toSet())
        for (key in before.keys()) assertJsonEqual("$label.$key", before.get(key), after.get(key))
      }
      before is JSONArray && after is JSONArray -> {
        assertEquals(label, before.length(), after.length())
        for (index in 0 until before.length()) assertJsonEqual("$label[$index]", before.get(index), after.get(index))
      }
      else -> assertEquals(label, before.toString(), after.toString())
    }
  }

  private fun assertCheckpoint(snapshot: JSONObject) {
    val checkpoint = GuildCheckpointStore(File(instrumentation.targetContext.filesDir, "guild-checkpoint.json")).read()
    assertNotNull("The canonical save has a valid fsynced native checkpoint", checkpoint)
    assertEquals(snapshot.getJSONObject("state").getLong("createdAt"), JSONObject(checkpoint!!.text).getJSONObject("state").getLong("createdAt"))
    assertEquals(snapshot.getString("resetGeneration"), checkpoint.generation)
    assertEquals(arguments.getString("guildExpectedSaveSchema")?.toInt() ?: 8, JSONObject(checkpoint.text).getInt("version"))
    assertGuildRetained(snapshot.getJSONObject("state"), JSONObject(checkpoint.text).getJSONObject("state"))
  }

  private fun nativeSettings(): String = File(instrumentation.targetContext.applicationInfo.dataDir, "shared_prefs/native-settings.xml")
    .let { if (it.exists()) it.readText() else "" }

  private fun exportEvidence(name: String, snapshot: JSONObject) {
    val output = File(instrumentation.targetContext.getExternalFilesDir(null), "content-update-qa").apply { mkdirs() }
    File(output, "$name.json").writeText(JSONObject(snapshot.toString()).put("processId", Process.myPid()).toString(2))
  }

  private fun evaluate(scenario: ActivityScenario<MainActivity>, script: String, callback: (String) -> Unit = {}) {
    val latch = CountDownLatch(1)
    scenario.onActivity { activity ->
      val view = findWebView(activity.window.decorView)
      if (view == null) { latch.countDown(); return@onActivity }
      view.evaluateJavascript(script) { value -> callback(value); latch.countDown() }
    }
    assertTrue("WebView callback must complete", latch.await(5, TimeUnit.SECONDS))
  }

  private fun findWebView(view: View): WebView? {
    if (view is WebView) return view
    if (view is ViewGroup) for (index in 0 until view.childCount) findWebView(view.getChildAt(index))?.let { return it }
    return null
  }

  private data class SignedFixture(val manifest: GuildContentManifest, val archive: File)

  private class StoreFixture(val root: File, baseline: Long = 1, selectedStore: GuildContentStore? = null) {
    private val keys: KeyPair = KeyPairGenerator.getInstance("EC").apply { initialize(ECGenParameterSpec("secp256r1")) }.generateKeyPair()
    val verifier = GuildContentVerifier(Base64.getEncoder().encodeToString(keys.public.encoded), "me.danielshort.wayfarers", 1, 18, 8)
    val store = selectedStore ?: GuildContentStore(File(root, "store"), verifier, baseline)

    fun releaseCanonical(version: Long): SignedFixture {
      val published = File(root, "latest-content.json")
      if (published.isFile) {
        val manifest = store.verifyManifest(published.readText())
        check(manifest.contentVersion == version)
        val archive = File(root, java.net.URI(manifest.archive.url).path.substringAfterLast('/'))
        check(archive.isFile && archive.length() == manifest.archive.size)
        check(GuildContentLimits.digest(archive.readBytes()) == manifest.archive.sha256)
        return SignedFixture(manifest, archive)
      }
      val assets = InstrumentationRegistry.getInstrumentation().targetContext.assets
      val files = linkedMapOf<String, ByteArray>()
      fun collect(path: String) {
        val children = assets.list(path)!!
        if (children.isEmpty()) files[path] = assets.open(path).use { it.readBytes() }
        else children.forEach { collect("$path/$it") }
      }
      collect("wayfarers"); collect("img/wayfarers-guild")
      listOf("native-checkpoint.js", "checkpoint.js", "android.js", "bundle-manifest.json").forEach { files.remove("wayfarers/$it") }
      val archive = File(root, "canonical-$version.zip")
      ZipOutputStream(archive.outputStream()).use { zip -> files.forEach { (path, bytes) -> zip.putNextEntry(ZipEntry(path)); zip.write(bytes); zip.closeEntry() } }
      val records = JSONArray()
      files.forEach { (path, bytes) -> records.put(JSONObject().put("path", path).put("sha256", GuildContentLimits.digest(bytes)).put("size", bytes.size)) }
      val payload = JSONObject().put("schemaVersion", 1).put("packageName", "me.danielshort.wayfarers")
        .put("nativeApi", 1).put("saveSchema", 8).put("minAppVersionCode", 18).put("contentVersion", version).put("label", "Local canonical QA $version")
        .put("archive", JSONObject().put("url", "https://github.com/danielshort3/danielshort3.github.io/releases/download/qa-content/canonical-$version.zip")
          .put("sha256", GuildContentLimits.digest(archive.readBytes())).put("size", archive.length())).put("records", records).toString().toByteArray()
      val signature = Signature.getInstance("SHA256withECDSA").apply { initSign(keys.private); update(payload) }.sign()
      val envelope = JSONObject().put("payload", Base64.getEncoder().encodeToString(payload)).put("signature", Base64.getEncoder().encodeToString(signature)).toString()
      return SignedFixture(verifier.verify(envelope), archive)
    }

    fun release(version: Long): SignedFixture {
      val files = linkedMapOf("wayfarers/index.html" to "<!doctype html><title>content $version</title>", "wayfarers/game.css" to "body{color:#ffffff;background:#112233}")
      val archive = File(root, "fixture-$version.zip")
      ZipOutputStream(archive.outputStream()).use { zip ->
        for ((path, text) in files) { zip.putNextEntry(ZipEntry(path)); zip.write(text.toByteArray()); zip.closeEntry() }
      }
      val records = JSONArray()
      for ((path, text) in files) {
        val bytes = text.toByteArray(Charsets.UTF_8)
        records.put(JSONObject().put("path", path).put("sha256", GuildContentLimits.digest(bytes)).put("size", bytes.size))
      }
      val payload = JSONObject().put("schemaVersion", 1).put("packageName", "me.danielshort.wayfarers")
        .put("nativeApi", 1).put("saveSchema", 8).put("minAppVersionCode", 17).put("contentVersion", version).put("label", "Native test $version")
        .put("archive", JSONObject().put("url", "https://github.com/danielshort3/danielshort3.github.io/releases/download/qa-content/fixture-$version.zip")
          .put("sha256", GuildContentLimits.digest(archive.readBytes())).put("size", archive.length())).put("records", records)
        .toString().toByteArray(Charsets.UTF_8)
      val signature = Signature.getInstance("SHA256withECDSA").apply { initSign(keys.private); update(payload) }.sign()
      val envelope = JSONObject().put("payload", Base64.getEncoder().encodeToString(payload)).put("signature", Base64.getEncoder().encodeToString(signature)).toString()
      return SignedFixture(verifier.verify(envelope), archive)
    }
  }

  private fun withIsolatedStore(baseline: Long = 1, block: (StoreFixture) -> Unit) {
    val parent = File(instrumentation.targetContext.cacheDir, "content-device-tests").apply { mkdirs() }
    val directory = File(parent, UUID.randomUUID().toString()).apply { mkdirs() }
    try { block(StoreFixture(directory, baseline)) } finally {
      check(directory.canonicalFile.parentFile == parent.canonicalFile)
      directory.deleteRecursively()
    }
  }

  private fun withIsolatedApplicationStore(baseline: Long = 1, block: (StoreFixture) -> Unit) {
    val app = instrumentation.targetContext.applicationContext as WayfarersApplication
    val checkpoint = GuildCheckpointStore(File(app.filesDir, "guild-checkpoint.json")).read()
    assertNotNull("Back up a real guild before an opted-in device failure case", checkpoint)
    val exported = File(app.getExternalFilesDir(null), if (arguments.getString("guildDebugUpdateQa") == "true" || arguments.getString("guildDebugRecoveryQa") == "true") "inline-qa/before-debug.json" else "release-qa/guild.json")
    assertTrue(exported.isFile)
    assertEquals("External backup must identify this same disposable guild", checkpoint!!.createdAt,
      JSONObject(exported.readText()).getJSONObject("state").getDouble("createdAt"), 0.0)
    if (arguments.getString("guildSignedCandidateQa") == "true") {
      // Use the exact staged release and installed production verifier/store.
      // This is local candidate evidence, not a publicly downloaded update.
      block(StoreFixture(File(app.getExternalFilesDir(null), "inline-qa/content6"), baseline, app.contentStore))
      return
    }
    val originalStore = app.contentStore
    val originalManager = app.contentUpdates
    val originalSettings = nativeSettings()
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    withIsolatedStore(baseline) { fixture ->
      val manager = GuildContentUpdateManager(fixture.store,
        "https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-content-updates/latest-content.json", scope = scope)
      val storeField = WayfarersApplication::class.java.getDeclaredField("contentStore").apply { isAccessible = true }
      val managerField = WayfarersApplication::class.java.getDeclaredField("contentUpdates").apply { isAccessible = true }
      instrumentation.runOnMainSync { storeField.set(app, fixture.store); managerField.set(app, manager) }
      try { block(fixture) } finally {
        scope.cancel()
        instrumentation.runOnMainSync { storeField.set(app, originalStore); managerField.set(app, originalManager) }
        assertEquals("Failure QA cannot alter native preferences", originalSettings, nativeSettings())
      }
    }
  }
}
