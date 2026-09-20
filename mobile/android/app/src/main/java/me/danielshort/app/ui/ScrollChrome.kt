package me.danielshort.app.ui

import android.content.Context
import android.view.accessibility.AccessibilityManager
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.foundation.focusGroup
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.nestedscroll.NestedScrollConnection
import androidx.compose.ui.input.nestedscroll.NestedScrollSource
import androidx.compose.ui.input.nestedscroll.nestedScroll
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.unit.dp
import kotlin.math.abs

/** Negative consumed movement reads further down; reversing direction starts a fresh threshold. */
internal class ScrollChromePolicy(private val hideThreshold: Float, private val showThreshold: Float) {
  var visible = true
    private set
  private var distance = 0f

  fun reveal() {
    visible = true
    distance = 0f
  }

  fun onScroll(deltaY: Float, allowed: Boolean = true): Boolean {
    if (!allowed) {
      reveal()
      return visible
    }
    if (!deltaY.isFinite() || deltaY == 0f) return visible
    if (distance != 0f && (distance > 0f) != (deltaY > 0f)) distance = 0f
    distance += deltaY
    if (visible && distance <= -hideThreshold) {
      visible = false
      distance = 0f
    } else if (!visible && distance >= showThreshold) {
      visible = true
      distance = 0f
    }
    // Movement continuing in an already-applied direction cannot accumulate without bound.
    distance = distance.coerceIn(-hideThreshold, showThreshold)
    return visible
  }
}

internal val LocalChromeInputFocus = staticCompositionLocalOf<(Boolean) -> Unit> { {} }
internal val LocalNativeReduceMotion = staticCompositionLocalOf { false }

/** Bars overlay a stable scroll viewport. Only the list's end padding reserves space for them. */
@Composable
internal fun ScrollChromeLayout(
  screenKey: String,
  modifier: Modifier = Modifier,
  topBar: @Composable () -> Unit,
  bottomBar: (@Composable () -> Unit)? = null,
  snackbar: @Composable () -> Unit = {},
  content: @Composable (PaddingValues) -> Unit
) {
  val density = LocalDensity.current
  val reduceMotion = LocalNativeReduceMotion.current
  val touchExploration = rememberTouchExploration()
  var headerFocused by remember { mutableStateOf(false) }
  var navigationFocused by remember { mutableStateOf(false) }
  var inputFocused by remember(screenKey) { mutableStateOf(false) }
  val keyboardVisible = WindowInsets.ime.getBottom(density) > 0
  val allowed = !touchExploration && !headerFocused && !navigationFocused && !inputFocused && !keyboardVisible
  val allowedNow by rememberUpdatedState(allowed)
  val policy = remember(density.density) {
    ScrollChromePolicy(with(density) { 48.dp.toPx() }, with(density) { 18.dp.toPx() })
  }
  var visible by remember { mutableStateOf(true) }
  LaunchedEffect(screenKey, allowed) {
    policy.reveal()
    visible = true
  }
  val nestedScroll = remember(policy) {
    object : NestedScrollConnection {
      override fun onPostScroll(consumed: Offset, available: Offset, source: NestedScrollSource): Offset {
        if (source != NestedScrollSource.UserInput) return Offset.Zero
        if (available.y > 0f && consumed.y >= 0f) {
          policy.reveal()
          visible = true
        } else if (abs(consumed.y) > 0f) visible = policy.onScroll(consumed.y, allowedNow)
        return Offset.Zero
      }
    }
  }
  val animatedProgress by animateFloatAsState(
    targetValue = if (visible) 0f else 1f,
    animationSpec = if (reduceMotion) snap() else tween(190),
    label = "Full navigation visibility"
  )
  // Keyboard and accessibility focus restore controls immediately, before an action can run.
  val progress = if (allowed) animatedProgress else 0f
  // The wide frame owns the system-bar gutters. Raw inset access does not subtract consumed
  // insets, so avoid painting a second status-bar strip over the frame's native toolbar.
  val frameOwnsInsets = LocalSiteFrameInsetsHandled.current
  val statusInset = if (frameOwnsInsets) 0.dp else WindowInsets.statusBars.asPaddingValues().calculateTopPadding()
  val navigationInset = if (frameOwnsInsets) 0.dp else WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding()
  var topHeight by remember(density.density, statusInset) { mutableStateOf(66.dp + statusInset) }
  var bottomHeight by remember(density.density, navigationInset, bottomBar != null) {
    mutableStateOf(if (bottomBar == null) navigationInset else 81.dp + navigationInset)
  }
  val onInputFocus = remember(screenKey) { { focused: Boolean -> inputFocused = focused } }
  CompositionLocalProvider(LocalChromeInputFocus provides onInputFocus) {
    Box(modifier.fillMaxSize().clipToBounds().imePadding()
      .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Horizontal))
      .nestedScroll(nestedScroll)) {
      content(PaddingValues(top = topHeight, bottom = bottomHeight))
      Box(Modifier.align(Alignment.TopCenter).fillMaxWidth()
        .onSizeChanged { topHeight = with(density) { it.height.toDp() } }
        .graphicsLayer { translationY = -size.height * progress }
        .onFocusChanged { headerFocused = it.hasFocus }.focusGroup()
        .testTag("full-site-header")
        .then(if (!visible) Modifier.clearAndSetSemantics {} else Modifier)) { topBar() }
      if (bottomBar != null) {
        Box(Modifier.align(Alignment.BottomCenter).fillMaxWidth()
          .onSizeChanged { bottomHeight = with(density) { it.height.toDp() } }
          .graphicsLayer { translationY = size.height * progress }
          .onFocusChanged { navigationFocused = it.hasFocus }.focusGroup()
          .testTag("full-site-navigation")
          .then(if (!visible) Modifier.clearAndSetSemantics {} else Modifier)) { bottomBar() }
      }
      // App chrome can disappear; Android's own clock and gesture area retain a readable surface.
      Box(Modifier.align(Alignment.TopCenter).fillMaxWidth().height(statusInset).background(MaterialTheme.colorScheme.surface))
      Box(Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(if (keyboardVisible) 0.dp else navigationInset)
        .background(MaterialTheme.colorScheme.surface))
      Box(Modifier.align(Alignment.BottomCenter).padding(bottom = if (visible) bottomHeight else navigationInset)) { snackbar() }
    }
  }
}

@Composable
private fun rememberTouchExploration(): Boolean {
  val context = LocalContext.current
  val manager = remember(context) { context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager }
  var enabled by remember(manager) { mutableStateOf(manager.isTouchExplorationEnabled) }
  DisposableEffect(manager) {
    val listener = AccessibilityManager.TouchExplorationStateChangeListener { enabled = it }
    manager.addTouchExplorationStateChangeListener(listener)
    onDispose { manager.removeTouchExplorationStateChangeListener(listener) }
  }
  return enabled
}
