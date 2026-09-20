package me.danielshort.app.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

@OptIn(ExperimentalTestApi::class)
class AdaptiveSiteLayoutFlowTest {
  @get:Rule val compose = createComposeRule()
  private val window = mutableStateOf(DpSize(1280.dp, 800.dp))
  private val section = mutableStateOf(Section.ABOUT)
  private lateinit var listState: LazyListState
  private lateinit var scope: CoroutineScope
  private lateinit var contentIdentity: Any

  private fun openFixture(width: Int = 1280, reduceMotion: Boolean = false) {
    window.value = DpSize(width.dp, 800.dp)
    compose.setContent {
      DeviceConfigurationOverride(DeviceConfigurationOverride.ForcedSize(window.value)) {
        MaterialTheme {
          CompositionLocalProvider(LocalNativeReduceMotion provides reduceMotion) {
            AdaptiveSiteLayout(selected = section.value, onSection = { section.value = it }) { wide ->
              val identity = remember { Any() }
              SideEffect { contentIdentity = identity }
              var query by rememberSaveable { mutableStateOf("") }
              listState = rememberLazyListState()
              scope = rememberCoroutineScope()
              ScrollChromeLayout(
                screenKey = section.value.name,
                topBar = {
                  Surface {
                    Box(Modifier.fillMaxWidth().height(64.dp), contentAlignment = Alignment.Center) {
                      Text("Site header")
                    }
                  }
                },
                bottomBar = if (wide) null else {
                  {
                    Surface {
                      Row(Modifier.fillMaxWidth().height(72.dp)) {
                        Section.entries.forEach { tab ->
                          TextButton(onClick = { section.value = tab }, modifier = Modifier.weight(1f)) { Text(tab.label) }
                        }
                      }
                    }
                  }
                }
              ) { padding ->
                LazyColumn(Modifier.fillMaxSize().testTag("adaptive-content"), state = listState, contentPadding = padding) {
                  item(key = "search") {
                    OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth().testTag("retained-search"), label = { Text("Search") })
                  }
                  items(50, key = { it }) { index ->
                    Text("Reading row $index", Modifier.fillMaxWidth().height(84.dp).padding(16.dp))
                  }
                }
              }
            }
          }
        }
      }
    }
    compose.waitForIdle()
  }

  @Test fun allFiveTabsBracketTheActivePanelAndExposeSelection() {
    openFixture(width = 840)
    compose.onNodeWithTag("full-site-navigation").assertDoesNotExist()
    Section.entries.forEach { selected ->
      compose.onNodeWithTag("site-rail-${selected.name.lowercase()}").performClick().assertIsSelected()
      val panelNode = compose.onNodeWithTag("site-content-panel")
      panelNode.assert(SemanticsMatcher.expectValue(SemanticsProperties.IsTraversalGroup, true))
        .assert(SemanticsMatcher.expectValue(SemanticsProperties.TraversalIndex, (selected.ordinal + 1).toFloat()))
      val panel = panelNode.getUnclippedBoundsInRoot()
      assertTrue(panel.right - panel.left >= 540.dp)
      Section.entries.forEach { tab ->
        val node = compose.onNodeWithTag("site-rail-${tab.name.lowercase()}")
        node.assertIsDisplayed().assertHasClickAction().assertWidthIsAtLeast(48.dp)
        if (tab != selected) node.assertIsNotSelected()
        val rail = node.getUnclippedBoundsInRoot()
        if (tab.ordinal <= selected.ordinal) assertTrue(rail.right <= panel.left)
        else assertTrue(rail.left >= panel.right)
      }
    }
  }

  @Test fun resizeKeepsTheContentCompositionAndSearchAcrossTheBreakpoint() {
    openFixture()
    compose.onNodeWithTag("site-rail-projects").performClick()
    compose.onNodeWithTag("retained-search").performTextInput("ocean")
    val original = compose.runOnIdle { contentIdentity }
    compose.runOnIdle { window.value = DpSize(600.dp, 800.dp) }
    compose.onNodeWithTag("compact-site-frame").assertExists()
    compose.onNodeWithTag("full-site-navigation").assertIsDisplayed()
    compose.onNodeWithTag("retained-search").assertTextContains("ocean")
    compose.runOnIdle { assertSame(original, contentIdentity); assertEquals(Section.PROJECTS, section.value) }
    compose.runOnIdle { window.value = DpSize(1280.dp, 800.dp) }
    compose.onNodeWithTag("site-rail-projects").assertIsSelected()
    compose.onNodeWithTag("full-site-navigation").assertDoesNotExist()
    compose.onNodeWithTag("retained-search").assertTextContains("ocean")
    compose.runOnIdle { assertSame(original, contentIdentity) }
  }

  @Test fun scrollPositionSurvivesWindowResizeAndContentCanReachTheEnd() {
    openFixture()
    compose.runOnIdle { scope.launch { listState.scrollToItem(18, 12) } }
    compose.waitForIdle()
    compose.runOnIdle { window.value = DpSize(600.dp, 900.dp) }
    compose.waitForIdle()
    compose.runOnIdle { assertEquals(18, listState.firstVisibleItemIndex); assertEquals(12, listState.firstVisibleItemScrollOffset) }
    compose.runOnIdle { window.value = DpSize(960.dp, 600.dp) }
    compose.onNodeWithTag("adaptive-content").performScrollToIndex(50)
    compose.onNodeWithText("Reading row 49").assertIsDisplayed()
    Section.entries.forEach { compose.onNodeWithTag("site-rail-${it.name.lowercase()}").assertIsDisplayed() }
  }

  @Test fun wideTabsRemainAvailableWhenReadingHidesTheHeaderWithReducedMotion() {
    openFixture(reduceMotion = true)
    compose.onNodeWithTag("adaptive-content").performTouchInput {
      swipeUp(startY = height * .75f, endY = height * .35f, durationMillis = 400)
    }
    compose.onNodeWithText("Site header").assertDoesNotExist()
    Section.entries.forEach { compose.onNodeWithTag("site-rail-${it.name.lowercase()}").assertIsDisplayed() }
    compose.onNodeWithTag("site-rail-tools").performClick().assertIsSelected()
    compose.onNodeWithText("Site header").assertIsDisplayed()
  }
}
