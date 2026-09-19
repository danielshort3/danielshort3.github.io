package me.danielshort.app.nativefeatures.games

import android.content.Context
import android.util.Log
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.espresso.IdlingPolicies
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import me.danielshort.app.nativefeatures.demos.NativeProjectDemoScreen
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Before
import org.junit.After
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.TimeUnit

/** Manual frame time bounds continuous native loops; no network is needed for these flows. */
@RunWith(AndroidJUnit4::class)
class NativeGamesFlowTest {
  @get:Rule val compose = createComposeRule()
  private val showing = mutableStateOf(true)
  private lateinit var gameLifecycle: ControlledGameLifecycle
  private val preferences get() = InstrumentationRegistry.getInstrumentation().targetContext
    .getSharedPreferences("native_games_v1", Context.MODE_PRIVATE)

  @Before fun freshLocalGames() {
    preferences.edit().clear().commit()
    compose.mainClock.autoAdvance = true
    IdlingPolicies.setMasterPolicyTimeout(15, TimeUnit.SECONDS)
    IdlingPolicies.setIdlingResourceTimeout(10, TimeUnit.SECONDS)
    compose.runOnUiThread { gameLifecycle = ControlledGameLifecycle() }
  }

  @After fun restoreSynchronizationLimits() {
    IdlingPolicies.setMasterPolicyTimeout(60, TimeUnit.SECONDS)
    IdlingPolicies.setIdlingResourceTimeout(26, TimeUnit.SECONDS)
  }

  private fun settle(milliseconds: Long = 100) {
    compose.mainClock.advanceTimeBy(milliseconds)
    compose.waitForIdle()
  }

  private fun open(id: String, reduceMotion: Boolean = false) {
    Log.i("NativeGameFlow", "Opening $id with stopped simulation")
    compose.setContent {
      MaterialTheme {
        CompositionLocalProvider(LocalGameLoopLifecycleOwner provides gameLifecycle) {
          if (showing.value) NativeGamesScreen(id, { showing.value = false }, reduceMotion)
          else Text("Returned to games")
        }
      }
    }
    settle()
    Log.i("NativeGameFlow", "Opened $id")
  }

  /** No implicit idling call runs while a continuous simulation is STARTED. */
  private fun advanceGame(milliseconds: Long) {
    Log.i("NativeGameFlow", "Advancing simulation $milliseconds ms")
    compose.mainClock.autoAdvance = false
    compose.runOnUiThread { gameLifecycle.registry.currentState = Lifecycle.State.STARTED }
    try { compose.mainClock.advanceTimeBy(milliseconds) }
    finally {
      compose.runOnUiThread { gameLifecycle.registry.currentState = Lifecycle.State.CREATED }
      compose.mainClock.autoAdvance = true
    }
    settle()
  }

  private fun click(text: String) {
    Log.i("NativeGameFlow", "Click $text")
    compose.onNodeWithText(text, substring = false).performScrollTo()
    settle()
    compose.onNodeWithText(text, substring = false).performClick()
    settle()
  }

  private fun leave() {
    click("‹  Games")
    compose.onNodeWithText("Returned to games").assertExists()
  }

  private fun reopen() {
    compose.runOnIdle { showing.value = true }
    settle()
  }

  private fun checkpoint(id: String): JSONObject = JSONObject(requireNotNull(preferences.getString(id, null)))

  @Test(timeout = 45_000) fun stellarResumeOverdrivePauseAndCheckpoint() {
    open("stellar-dogfight")
    compose.onNodeWithText("Overdrive").assertIsNotEnabled()
    click("Resume")
    click("Overdrive")
    val activeCooldown = checkpoint("stellar-dogfight").getDouble("abilityCooldown")
    assertTrue(activeCooldown > 0)
    advanceGame(500)
    assertTrue(checkpoint("stellar-dogfight").getDouble("abilityCooldown") < activeCooldown)
    click("Pause")
    val pausedCooldown = checkpoint("stellar-dogfight").getDouble("abilityCooldown")
    advanceGame(2_000)
    leave()
    assertEquals(pausedCooldown, checkpoint("stellar-dogfight").getDouble("abilityCooldown"), .0001)
    reopen()
    compose.onNodeWithText("Resume").assertExists()
    compose.onNodeWithText("Overdrive", substring = true).assertIsNotEnabled()
    leave()
  }

  @Test(timeout = 45_000) fun oceanPresetsResumeAndLocalSettingsSurviveReopening() {
    open("ocean-wave-simulation", reduceMotion = true)
    compose.onNodeWithText("Resume").assertExists()
    click("Storm swell")
    compose.onNodeWithText("Wind · 18").assertExists()
    compose.onNodeWithText("Wave height · 2.5").assertExists()
    assertEquals(18.0, checkpoint("ocean-wave-simulation").getDouble("wind"), 0.0)
    click("Resume")
    advanceGame(400)
    click("Pause")
    leave()
    reopen()
    compose.onNodeWithText("Wind · 18").assertExists()
    click("Quiet cove")
    compose.onNodeWithText("Wind · 3").assertExists()
    assertEquals(.5, checkpoint("ocean-wave-simulation").getDouble("amplitude"), .0001)
    leave()
  }

  @Test(timeout = 45_000) fun starfallClassCombatCheckpointAndReset() {
    open("project-starfall")
    click("Mage")
    compose.onNodeWithText("Mage · Training weapon · 20 power").assertExists()
    compose.onNodeWithText("Attack").assertIsNotEnabled()
    click("Resume")
    click("Attack")
    val wounded = checkpoint("project-starfall").getJSONArray("enemies").getJSONObject(0)
    assertTrue(wounded.getDouble("hp") < wounded.getDouble("maxHp"))
    advanceGame(1_000)
    click("Power strike")
    click("Pause")
    val saved = checkpoint("project-starfall")
    assertEquals(1, saved.getInt("class"))
    assertTrue(saved.getInt("kills") >= 1)
    assertTrue(saved.getInt("gold") > 0)
    leave()
    reopen()
    compose.onNodeWithText("Mage · Training weapon · 20 power").assertExists()
    click("Reset Starfall character")
    compose.onNodeWithText("Reset character", substring = false).performClick()
    settle()
    assertEquals(0, checkpoint("project-starfall").getInt("class"))
    assertEquals(0, checkpoint("project-starfall").getInt("kills"))
    compose.onNodeWithText("Attack").assertIsNotEnabled()
    leave()
  }

  @Test(timeout = 45_000) fun probabilitySpinCooldownRestoredProgressAndReset() {
    open("probability-engine")
    click("Spin · 5 credits")
    assertEquals(1, checkpoint("probability-engine").getInt("spins"))
    assertTrue(checkpoint("probability-engine").getDouble("reward") > 0)
    compose.onNodeWithText("Spin · 5 credits").assertIsNotEnabled()
    advanceGame(1_500)
    assertEquals("Spin cooldown must advance in the real game loop", 0.0, checkpoint("probability-engine").getDouble("cooldown"), .0001)
    compose.onNodeWithText("Spin · 5 credits").assertIsEnabled()
    leave()
    reopen()
    click("Spin · 5 credits")
    assertEquals(2, checkpoint("probability-engine").getInt("spins"))
    click("Reset Probability Engine")
    compose.onNodeWithText("Reset progress", substring = false).performClick()
    settle()
    assertEquals(0, checkpoint("probability-engine").getInt("spins"))
    assertEquals(120.0, checkpoint("probability-engine").getDouble("credits"), 0.0)
    compose.onNodeWithText("Spin · 5 credits").assertIsEnabled()
    leave()
  }

  @Test(timeout = 45_000) fun stormbreakManualStrikeAbilityPauseAndReset() {
    open("stormbreak")
    compose.onNodeWithContentDescription("Auto attack").performScrollTo()
    settle()
    compose.onNodeWithContentDescription("Auto attack").performClick()
    settle()
    val before = checkpoint("stormbreak").getDouble("enemyHp")
    compose.onNodeWithContentDescription("Battle against", substring = true).performScrollTo()
    settle()
    compose.onNodeWithContentDescription("Battle against", substring = true).performClick()
    settle()
    assertTrue(checkpoint("stormbreak").getDouble("enemyHp") < before)
    click("Bolt")
    assertTrue(checkpoint("stormbreak").getInt("kills") >= 1)
    compose.onNodeWithContentDescription("Battle against Cyclopean Mauler. Tap to strike.").assertExists()
    click("Pause")
    compose.onNodeWithText("Storm", substring = false).assertIsNotEnabled()
    val paused = checkpoint("stormbreak")
    assertFalse(paused.getBoolean("offlineEligible"))
    advanceGame(1_000)
    leave()
    assertEquals(paused.getDouble("gold"), checkpoint("stormbreak").getDouble("gold"), 0.0)
    reopen()
    assertEquals(paused.getDouble("gold"), checkpoint("stormbreak").getDouble("gold"), 0.0)
    click("Reset Stormbreak progress")
    compose.onNodeWithText("Reset progress", substring = false).performClick()
    settle()
    assertEquals(0, checkpoint("stormbreak").getInt("kills"))
    assertEquals(0.0, checkpoint("stormbreak").getDouble("gold"), 0.0)
    leave()
  }

  @Test(timeout = 45_000) fun pizzaEstimatorAcceptsScenarioCalculatesAndRejectsInvalidInput() {
    compose.setContent { MaterialTheme { NativeProjectDemoScreen("pizza", {}) } }
    settle()
    compose.onNode(hasSetTextAction() and hasText("Order cost ($)")).performTextReplacement("85.5")
    settle()
    click("City: Frisco")
    compose.onNode(hasText("Plano") and hasClickAction()).performClick()
    settle()
    click("Housing: Residential")
    compose.onNode(hasText("Apartment") and hasClickAction()).performClick()
    settle()
    click("Update estimate")
    compose.onNodeWithText("$9.04", substring = false).assertExists()
    compose.onNodeWithText("Estimated tip", substring = false).assertExists()
    compose.onNode(hasSetTextAction() and hasText("Order cost ($)")).performScrollTo()
    settle()
    compose.onNode(hasSetTextAction() and hasText("Order cost ($)")).performTextReplacement("abc")
    settle()
    click("Update estimate")
    compose.onNodeWithText("Enter an order cost between $0.01 and $1,000.").assertExists()
    compose.onNodeWithText("Estimated tip", substring = false).assertDoesNotExist()
  }
}

private class ControlledGameLifecycle : LifecycleOwner {
  val registry = LifecycleRegistry(this).apply { currentState = Lifecycle.State.CREATED }
  override val lifecycle: Lifecycle get() = registry
}
