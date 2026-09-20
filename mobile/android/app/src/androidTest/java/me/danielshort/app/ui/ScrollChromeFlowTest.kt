package me.danielshort.app.ui

import androidx.compose.foundation.gestures.FlingBehavior
import androidx.compose.foundation.gestures.ScrollScope
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.swipeDown
import androidx.compose.ui.test.swipeUp
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class ScrollChromeFlowTest {
  @get:Rule val compose = createComposeRule()
  private val screenKey = mutableStateOf("first-screen")
  private lateinit var listState: LazyListState
  private lateinit var scope: CoroutineScope
  private lateinit var inputFocus: (Boolean) -> Unit

  private fun openFixture(reduceMotion: Boolean = false) {
    compose.setContent {
      MaterialTheme {
        CompositionLocalProvider(LocalNativeReduceMotion provides reduceMotion) {
          ScrollChromeLayout(
            screenKey = screenKey.value,
            topBar = {
              Surface {
                Box(Modifier.fillMaxWidth().height(64.dp), contentAlignment = Alignment.Center) {
                  Button(onClick = {}) { Text("Site header") }
                }
              }
            },
            bottomBar = {
              Surface {
                Box(Modifier.fillMaxWidth().height(72.dp), contentAlignment = Alignment.Center) {
                  Button(onClick = {}) { Text("Section navigation") }
                }
              }
            }
          ) { padding ->
            listState = rememberLazyListState()
            scope = rememberCoroutineScope()
            inputFocus = LocalChromeInputFocus.current
            // Eliminate inertia so a changing list anchor identifies layout movement,
            // rather than normal scrolling after the finger leaves the screen.
            val noFling = remember {
              object : FlingBehavior {
                override suspend fun ScrollScope.performFling(initialVelocity: Float) = 0f
              }
            }
            LazyColumn(
              modifier = Modifier.fillMaxSize().testTag("chrome-test-content"),
              state = listState,
              contentPadding = padding,
              flingBehavior = noFling
            ) {
              items(count = 40, key = { it }) { index ->
                Text("Content row $index", Modifier.fillMaxWidth().height(84.dp).padding(16.dp))
              }
            }
          }
        }
      }
    }
    compose.waitForIdle()
    assertChromeShown()
  }

  private fun assertChromeShown() {
    compose.onNodeWithText("Site header").assertIsDisplayed()
    compose.onNodeWithText("Section navigation").assertIsDisplayed()
  }

  private fun assertChromeHidden() {
    // Hidden bars also remove their descendants from the accessibility tree.
    compose.onNodeWithText("Site header").assertDoesNotExist()
    compose.onNodeWithText("Section navigation").assertDoesNotExist()
  }

  private fun readFurtherDown(durationMillis: Long = 400) {
    compose.onNodeWithTag("chrome-test-content").performTouchInput {
      swipeUp(startY = height * 0.7f, endY = height * 0.3f, durationMillis = durationMillis)
    }
  }

  private fun readBackUp() {
    compose.onNodeWithTag("chrome-test-content").performTouchInput {
      swipeDown(startY = height * 0.3f, endY = height * 0.7f, durationMillis = 400)
    }
  }

  private fun anchor(): Pair<Int, Int> = compose.runOnIdle {
    listState.firstVisibleItemIndex to listState.firstVisibleItemScrollOffset
  }

  @Test fun scrollingFurtherDownHidesBothBarsAndScrollingUpRestoresThem() {
    openFixture()
    readFurtherDown()
    assertChromeHidden()
    readBackUp()
    assertChromeShown()
  }

  @Test fun programmaticListPositioningDoesNotHideChrome() {
    openFixture()
    compose.runOnIdle { scope.launch { listState.scrollToItem(12, 19) } }
    compose.waitForIdle()
    assertEquals(12 to 19, anchor())
    assertChromeShown()
    compose.runOnIdle { scope.launch { listState.animateScrollToItem(20, 11) } }
    compose.waitForIdle()
    assertEquals(20 to 11, anchor())
    assertChromeShown()
    compose.runOnIdle { scope.launch { listState.scrollToItem(2, 7) } }
    compose.waitForIdle()
    assertEquals(2 to 7, anchor())
    assertChromeShown()
  }

  @Test fun changingScreensRevealsChromeWithoutMovingTheList() {
    openFixture()
    readFurtherDown()
    assertChromeHidden()
    val previousAnchor = anchor()
    compose.mainClock.autoAdvance = false
    try {
      compose.runOnIdle { screenKey.value = "second-screen" }
      compose.mainClock.advanceTimeBy(48)
      assertEquals(previousAnchor, anchor())
      compose.mainClock.advanceTimeBy(300)
      compose.waitForIdle()
      assertEquals(previousAnchor, anchor())
      assertChromeShown()
    } finally {
      compose.mainClock.autoAdvance = true
    }
  }

  @Test fun focusedInputRevealsAndPinsChromeUntilFocusIsReleased() {
    openFixture()
    readFurtherDown()
    assertChromeHidden()
    compose.runOnIdle { inputFocus(true) }
    assertChromeShown()
    readFurtherDown()
    assertChromeShown()
    compose.runOnIdle { inputFocus(false) }
    compose.waitForIdle()
    readFurtherDown()
    assertChromeHidden()
  }

  @Test fun hidingAnimationDoesNotShiftTheListAfterTheGestureEnds() {
    openFixture()
    compose.mainClock.autoAdvance = false
    try {
      readFurtherDown(durationMillis = 100)
      compose.mainClock.advanceTimeByFrame()
      val gestureEndAnchor = anchor()
      compose.mainClock.advanceTimeBy(64)
      assertEquals(gestureEndAnchor, anchor())
      compose.mainClock.advanceTimeBy(300)
      compose.waitForIdle()
      assertEquals(gestureEndAnchor, anchor())
      assertChromeHidden()
    } finally {
      compose.mainClock.autoAdvance = true
    }
  }

  @Test fun reducedMotionStillHidesAndRestoresTheFullNavigation() {
    openFixture(reduceMotion = true)
    readFurtherDown()
    assertChromeHidden()
    readBackUp()
    assertChromeShown()
  }
}
