package me.danielshort.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.isTraversalGroup
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

internal const val WIDE_SITE_MIN_WIDTH_DP = 840
internal const val SITE_RAIL_WIDTH_DP = 52
internal val LocalSiteFrameInsetsHandled = staticCompositionLocalOf { false }

internal fun usesWideSiteLayout(availableWidthPx: Int, density: Float): Boolean =
  availableWidthPx >= (WIDE_SITE_MIN_WIDTH_DP * density).roundToInt()

/**
 * The website's accordion geometry, expressed as native, selectable tabs. The panel has one
 * composition position in both modes: only its measured width and placement change on resize.
 */
@Composable
internal fun AdaptiveSiteLayout(
  selected: Section,
  onSection: (Section) -> Unit,
  modifier: Modifier = Modifier,
  content: @Composable (wide: Boolean) -> Unit
) {
  BoxWithConstraints(modifier.fillMaxSize()) {
    // Compare pixels using the same rounding as Compose measurement. Converting an exact
    // 840dp allocation back from integer pixels can otherwise report 839.8dp at some densities.
    val wide = usesWideSiteLayout(constraints.maxWidth, LocalDensity.current.density)
    val outer = if (wide) Modifier.windowInsetsPadding(WindowInsets.safeDrawing).padding(12.dp) else Modifier
    Box(Modifier.fillMaxSize().then(outer), contentAlignment = Alignment.TopCenter) {
      val frame = if (wide) Modifier.widthIn(max = 1600.dp)
        .clip(RoundedCornerShape(16.dp))
        .border(1.dp, selected.color.copy(alpha = .24f), RoundedCornerShape(16.dp)) else Modifier
      Layout(
        modifier = frame.fillMaxSize().testTag(if (wide) "wide-site-frame" else "compact-site-frame")
          .then(if (wide) Modifier.selectableGroup().semantics { isTraversalGroup = true } else Modifier),
        content = {
          // Always first, including in compact mode; do not branch or key the screen by width.
          Box(Modifier.fillMaxSize().testTag("site-content-panel")
            .semantics {
              isTraversalGroup = true
              traversalIndex = (selected.ordinal + 1).toFloat()
            }
            .then(if (wide) Modifier.border(3.dp, selected.color).padding(3.dp) else Modifier)) {
            CompositionLocalProvider(LocalSiteFrameInsetsHandled provides wide) { content(wide) }
          }
          if (wide) Section.entries.forEach { section ->
            SiteRail(section, selected == section, onClick = { onSection(section) })
          }
        }
      ) { measurables, constraints ->
        val railWidth = if (wide) SITE_RAIL_WIDTH_DP.dp.roundToPx() else 0
        val panelWidth = (constraints.maxWidth - railWidth * Section.entries.size).coerceAtLeast(0)
        val panel = measurables.first().measure(Constraints.fixed(panelWidth, constraints.maxHeight))
        val rails = measurables.drop(1).map { it.measure(Constraints.fixed(railWidth, constraints.maxHeight)) }
        layout(constraints.maxWidth, constraints.maxHeight) {
          panel.placeRelative(if (wide) railWidth * (selected.ordinal + 1) else 0, 0)
          rails.forEachIndexed { index, rail ->
            rail.placeRelative(index * railWidth + if (index > selected.ordinal) panelWidth else 0, 0)
          }
        }
      }
    }
  }
}

@Composable
private fun SiteRail(section: Section, selected: Boolean, onClick: () -> Unit) {
  var focused by remember { mutableStateOf(false) }
  Box(Modifier.fillMaxHeight().width(SITE_RAIL_WIDTH_DP.dp)
    .background(section.color)
    .onFocusChanged { focused = it.hasFocus }
    .selectable(selected = selected, role = Role.Tab, onClick = onClick)
    .semantics {
      contentDescription = section.label
      // Place the active panel immediately after its tab in accessibility traversal too.
      traversalIndex = section.ordinal.toFloat() + .5f
    }
    .testTag("site-rail-${section.name.lowercase()}")
    .drawBehind {
      if (selected) {
        val center = size.height / 2
        val triangle = Path().apply {
          moveTo(size.width, center - 10.dp.toPx())
          lineTo(size.width + 8.dp.toPx(), center)
          lineTo(size.width, center + 10.dp.toPx())
          close()
        }
        drawPath(triangle, section.color)
      }
      if (focused) {
        val inset = 5.dp.toPx()
        drawRect(Color.White, topLeft = Offset(inset, inset),
          size = androidx.compose.ui.geometry.Size(size.width - inset * 2, size.height - inset * 2),
          style = androidx.compose.ui.graphics.drawscope.Stroke(2.dp.toPx()))
      }
    }, contentAlignment = Alignment.Center) {
    Icon(section.icon, contentDescription = null, tint = Color.White,
      modifier = Modifier.align(Alignment.TopCenter).padding(top = 26.dp).size(24.dp))
    Box(Modifier.fillMaxSize().padding(top = 64.dp, bottom = 24.dp), contentAlignment = Alignment.Center) {
      RotatedRailLabel(section.label, selected)
    }
  }
}

@Composable
private fun RotatedRailLabel(label: String, selected: Boolean) {
  // Swap measured axes before rotating so long labels and larger accessibility fonts are not
  // constrained to the rail's 52dp width. The tab itself supplies the unrotated accessible name.
  Layout(content = {
    Text(label.uppercase(), color = Color.White, fontSize = 13.sp,
      fontWeight = if (selected) FontWeight.ExtraBold else FontWeight.SemiBold,
      letterSpacing = .8.sp, maxLines = 1,
      modifier = Modifier.clearAndSetSemantics {}.graphicsLayer { rotationZ = -90f })
  }) { measurables, constraints ->
    val text = measurables.single().measure(Constraints(maxWidth = constraints.maxHeight))
    layout(text.height, text.width) {
      text.placeRelative((text.height - text.width) / 2, (text.width - text.height) / 2)
    }
  }
}

/** Wide library cards use more columns; larger fonts keep enough width for their text. */
@Composable
internal fun adaptiveCardMinWidth() = with(LocalDensity.current) { (340 * fontScale.coerceAtLeast(1f)).dp }
