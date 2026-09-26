@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package me.danielshort.app.nativefeatures.games

import android.graphics.BitmapFactory
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.*

val NATIVE_GAME_IDS = setOf("stellar-dogfight", "ocean-wave-simulation", "project-starfall", "probability-engine", "stormbreak")

/** Tests may bound simulation time without stopping the activity or dialog recomposers. */
internal val LocalGameLoopLifecycleOwner = staticCompositionLocalOf<LifecycleOwner?> { null }

@Composable
fun NativeGamesScreen(gameId: String, onBack: () -> Unit, reduceMotion: Boolean = false, onOpenWebsite: (() -> Unit)? = null) {
  when (gameId) {
    "stellar-dogfight" -> StellarScreen(onBack, reduceMotion)
    "ocean-wave-simulation" -> OceanScreen(onBack, reduceMotion)
    "project-starfall" -> StarfallScreen(onBack, reduceMotion, onOpenWebsite)
    "probability-engine" -> ProbabilityScreen(onBack)
    "stormbreak" -> StormbreakScreen(onBack, reduceMotion)
  }
}

private val StageColor = Color(0xFF06162A)
private val Gold = Color(0xFFFFD577)

@Composable
private fun GamePage(title: String, subtitle: String, onBack: () -> Unit, onOpenWebsite: (() -> Unit)? = null, content: @Composable ColumnScope.() -> Unit) {
  Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).safeDrawingPadding()
    .verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 8.dp),
    verticalArrangement = Arrangement.spacedBy(16.dp)) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
      TextButton(onClick = onBack) { Text("‹  Games") }
      if (onOpenWebsite != null) TextButton(onClick = onOpenWebsite) { Text("Website game →") }
    }
    Text(title, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, modifier = Modifier.semantics { heading() })
    Text(subtitle, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    HorizontalDivider(thickness = 2.dp, color = MaterialTheme.colorScheme.primary)
    content()
    Spacer(Modifier.height(12.dp))
  }
}

@Composable
private fun Stats(vararg values: Pair<String, String>) {
  Surface(shape = RoundedCornerShape(14.dp), color = MaterialTheme.colorScheme.surfaceContainerLow) {
    Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
      values.forEach { (label, value) -> Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.weight(1f)) {
        Text(value, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
        Text(label, style = MaterialTheme.typography.labelSmall)
      } }
    }
  }
}

@Composable
private fun GameBitmap(name: String): ImageBitmap? {
  val context = LocalContext.current
  return produceState<ImageBitmap?>(null, name) {
    value = withContext(Dispatchers.IO) { runCatching {
      context.assets.open("native-games/$name").use { BitmapFactory.decodeStream(it)?.asImageBitmap() }
    }.getOrNull() }
  }.value
}

private fun DrawScope.sprite(bitmap: ImageBitmap?, x: Float, y: Float, width: Float, height: Float,
  column: Int = 0, row: Int = 0, columns: Int = 1, rows: Int = 1) {
  if (bitmap == null) return
  val cell = IntSize(bitmap.width / columns, bitmap.height / rows)
  drawImage(bitmap, srcOffset = IntOffset(column * cell.width, row * cell.height), srcSize = cell,
    dstOffset = IntOffset(x.toInt(), y.toInt()), dstSize = IntSize(width.roundToInt().coerceAtLeast(1), height.roundToInt().coerceAtLeast(1)), filterQuality = FilterQuality.Low)
}

/** Work runs only while STARTED, stops immediately on navigation, and bounds each simulation step. */
@Composable
private fun GameLoop(paused: Boolean, onStep: (Float) -> Unit, onSave: () -> Unit) {
  val owner = LocalGameLoopLifecycleOwner.current ?: LocalLifecycleOwner.current
  val step by rememberUpdatedState(onStep)
  val save by rememberUpdatedState(onSave)
  val isPaused by rememberUpdatedState(paused)
  DisposableEffect(owner) {
    val observer = LifecycleEventObserver { _, event -> if (event == Lifecycle.Event.ON_STOP) save() }
    owner.lifecycle.addObserver(observer)
    onDispose { owner.lifecycle.removeObserver(observer); save() }
  }
  LaunchedEffect(owner) {
    owner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
      var previous = 0L
      var saved = 0f
      while (isActive) {
        if (isPaused) { previous = 0L; delay(80); continue }
        withFrameNanos { frame ->
          if (previous != 0L) {
            val dt = ((frame - previous) / 1_000_000_000f).coerceIn(0f, .05f)
            step(dt); saved += dt
            if (saved >= 5f) { save(); saved = 0f }
          }
          previous = frame
        }
      }
    }
  }
}

@Composable
private fun PauseControl(paused: Boolean, onToggle: () -> Unit) {
  OutlinedButton(onClick = onToggle) { Text(if (paused) "Resume" else "Pause") }
}

@Composable
private fun StellarScreen(onBack: () -> Unit, reduceMotion: Boolean) {
  val context = LocalContext.current
  val store = remember { GameStorage(context) }
  val game = remember { StellarModel().apply { restore(store.read("stellar-dogfight")) } }
  var tick by remember { mutableIntStateOf(0) }
  var paused by rememberSaveable { mutableStateOf(true) }
  fun save() = store.save("stellar-dogfight", game.snapshot())
  fun changed() { tick++; save() }
  val ended = remember(tick) { game.ended }
  GameLoop(paused || ended, { game.step(it); tick++ }, ::save)
  val background = GameBitmap("background-nebula.png")
  val ship = GameBitmap("ship-player-scout.png")
  val enemy = GameBitmap("enemy-interceptor.png")
  GamePage("Stellar Dogfight", "Drag to steer. Weapons target the nearest enemy.", onBack) {
    tick
    val remainingEnemies = game.enemies.size
    Stats("Wave" to "${game.wave}/10", "Hull" to "${game.hp.toInt()}/${game.maxHp.toInt()}", "Credits" to game.credits.toString())
    Canvas(Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(16.dp)).background(StageColor)
      .semantics { contentDescription = "Space combat field. Drag to steer your ship. $remainingEnemies enemies remain." }
      .pointerInput(game, paused) { detectDragGestures(onDragStart = {
        if (!paused) { game.targetX = it.x / size.width * 1000; game.targetY = it.y / size.height * 1000 }
      }) { change, _ -> change.consume(); if (!paused) { game.targetX = change.position.x / size.width * 1000; game.targetY = change.position.y / size.height * 1000 } } }) {
      sprite(background, 0f, 0f, size.width, size.height)
      val scale = size.width / 1000
      withTransform({ scale(scale, scale, Offset.Zero) }) {
        game.shots.forEach { drawCircle(if (it.friendly) Color.Cyan else Color(0xFFFF8763), if (it.friendly) 4f else 7f, Offset(it.x, it.y)) }
        game.enemies.forEach { sprite(enemy, it.x - 30, it.y - 30, 60f, 60f) }
        rotate(game.angle, Offset(game.x, game.y)) { sprite(ship, game.x - 34, game.y - 34, 68f, 68f) }
        if (game.overdrive > 0 && !reduceMotion) drawCircle(Gold.copy(alpha = .4f), 42f, Offset(game.x, game.y), style = Stroke(3f))
      }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { game.activate(); changed() }, enabled = !paused && !game.ended && game.abilityCooldown <= 0) { Text(if (game.abilityCooldown > 0) "Overdrive ${ceil(game.abilityCooldown).toInt()}s" else "Overdrive") }
      if (!game.ended) PauseControl(paused) { paused = !paused; save() }
      else Button(onClick = { game.restart(); paused = false; changed() }) { Text("New flight") }
    }
    if (game.ended) Text(if (game.completed) "Sector cleared. All ten waves defeated." else "Ship lost. Your credits and hangar upgrades are saved.", modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
    if (paused || game.ended) {
      Text("Hangar", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
      OutlinedButton(onClick = { game.upgrade(true); changed() }, enabled = game.weapon < 15 && game.credits >= game.upgradeCost(game.weapon), modifier = Modifier.fillMaxWidth()) {
        Text("Weapons ${game.weapon} · Upgrade ${game.upgradeCost(game.weapon)} credits")
      }
      OutlinedButton(onClick = { game.upgrade(false); changed() }, enabled = game.armor < 15 && game.credits >= game.upgradeCost(game.armor), modifier = Modifier.fillMaxWidth()) {
        Text("Hull ${game.armor} · Upgrade ${game.upgradeCost(game.armor)} credits")
      }
    }
  }
}

@Composable
private fun OceanScreen(onBack: () -> Unit, reduceMotion: Boolean) {
  val context = LocalContext.current
  val store = remember { GameStorage(context) }
  val saved = remember { store.read("ocean-wave-simulation") }
  var wind by rememberSaveable { mutableFloatStateOf((saved?.optDouble("wind", 8.0) ?: 8.0).toFloat().coerceIn(0f, 20f)) }
  var amplitude by rememberSaveable { mutableFloatStateOf((saved?.optDouble("amplitude", 1.0) ?: 1.0).toFloat().coerceIn(0f, 3f)) }
  var sunlight by rememberSaveable { mutableFloatStateOf((saved?.optDouble("sunlight", .65) ?: .65).toFloat().coerceIn(0f, 1f)) }
  var yaw by rememberSaveable { mutableFloatStateOf(0f) }
  var time by remember { mutableFloatStateOf(0f) }
  var paused by rememberSaveable { mutableStateOf(reduceMotion) }
  fun save() = store.save("ocean-wave-simulation", org.json.JSONObject().put("wind", wind).put("amplitude", amplitude).put("sunlight", sunlight))
  GameLoop(paused, { time = (time + it) % 10000 }, ::save)
  GamePage("Ocean Wave Simulation", "Shape the water, wind, and light. Drag to turn the view.", onBack) {
    Canvas(Modifier.fillMaxWidth().aspectRatio(1.15f).clip(RoundedCornerShape(16.dp)).background(StageColor)
      .semantics { contentDescription = "Animated ocean surface. Wind ${wind.toInt()}, wave height ${"%.1f".format(amplitude)}." }
      .pointerInput(Unit) { detectDragGestures { change, drag -> change.consume(); yaw = (yaw - drag.x * .03f).coerceIn(-35f, 35f) } }) {
      val w = size.width; val h = size.height
      val sky = lerp(Color(0xFF142847), Color(0xFF90CDE8), sunlight)
      drawRect(Brush.verticalGradient(listOf(sky, Color(0xFFF1D3B1)), endY = h * .47f))
      drawCircle(Color(0xFFFFE5A6), w * .055f, Offset(w * (.7f + yaw / 160), h * (.3f - sunlight * .15f)))
      drawRect(Brush.verticalGradient(listOf(Color(0xFF228D9E), Color(0xFF04263F)), startY = h * .43f), topLeft = Offset(0f, h * .43f))
      val rows = if (reduceMotion) 24 else 38
      for (r in 0 until rows) {
        val perspective = (r + 1f) / rows
        val y = h * .43f + perspective.pow(1.65f) * h * .6f
        val z = 80f * (1 - perspective)
        val path = Path()
        for (c in 0..50) {
          val x = c / 50f * w
          val wave = OceanModel.height(c * 1.7f + yaw, z, time, wind, amplitude)
          val wy = y + wave * h * .023f * perspective
          if (c == 0) path.moveTo(x, wy) else path.lineTo(x, wy)
        }
        path.lineTo(w, h + 60); path.lineTo(0f, h + 60); path.close()
        drawPath(path, lerp(Color(0xFF248E9A), Color(0xFF032B48), perspective).copy(alpha = .85f))
        drawPath(path, Color(0xFFB2ECE9).copy(alpha = .10f + .22f * sunlight), style = Stroke(.5f + perspective * 2.5f))
      }
    }
    PauseControl(paused) { paused = !paused }
    Text("Wind · ${wind.toInt()}", fontWeight = FontWeight.Medium)
    Slider(value = wind, onValueChange = { wind = it }, onValueChangeFinished = ::save, valueRange = 0f..20f, modifier = Modifier.semantics { contentDescription = "Wind" })
    Text("Wave height · ${"%.1f".format(amplitude)}", fontWeight = FontWeight.Medium)
    Slider(value = amplitude, onValueChange = { amplitude = it }, onValueChangeFinished = ::save, valueRange = 0f..3f, modifier = Modifier.semantics { contentDescription = "Wave height" })
    Text("Sunlight", fontWeight = FontWeight.Medium)
    Slider(value = sunlight, onValueChange = { sunlight = it }, onValueChangeFinished = ::save, modifier = Modifier.semantics { contentDescription = "Sunlight" })
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      OutlinedButton(onClick = { wind = 3f; amplitude = .5f; sunlight = .65f; save() }) { Text("Quiet cove") }
      OutlinedButton(onClick = { wind = 18f; amplitude = 2.5f; sunlight = .15f; save() }) { Text("Storm swell") }
    }
  }
}

@Composable
private fun StormbreakScreen(onBack: () -> Unit, reduceMotion: Boolean) {
  val context = LocalContext.current
  val store = remember { GameStorage(context) }
  val game = remember { StormbreakModel().apply { restore(store.read("stormbreak")) } }
  var tick by remember { mutableIntStateOf(0) }
  var paused by rememberSaveable { mutableStateOf(false) }
  var showReset by remember { mutableStateOf(false) }
  fun save() = store.save("stormbreak", game.snapshot().put("offlineEligible", !paused && !showReset))
  fun changed() { tick++; save() }
  GameLoop(paused || showReset, { game.step(it.toDouble()); tick++ }, ::save)
  val background = GameBitmap("temple-of-ash.webp")
  val zeus = GameBitmap("zeus-sprite-strip.webp")
  val atlas = GameBitmap("mythic-atlas.webp")
  GamePage("Stormbreak: Idle Olympus", "Strike, upgrade, and lead Zeus through mythic hordes.", onBack) {
    tick
    val currentEnemyName = game.enemy.name
    Stats("Level" to game.level.toString(), "Gold" to compact(game.gold), "Ambrosia" to compact(game.ambrosia))
    Text("${StormbreakModel.ZONES[game.zone].name} · Wave ${game.wave}", fontWeight = FontWeight.SemiBold)
    Canvas(Modifier.fillMaxWidth().aspectRatio(1.45f).clip(RoundedCornerShape(16.dp)).background(StageColor)
      .semantics {
        contentDescription = "Battle against $currentEnemyName. Tap to strike."
        onClick("Strike enemy") { if (!paused) { game.tap(); changed() }; true }
      }
      .pointerInput(paused) { detectTapGestures { if (!paused) { game.tap(); changed() } } }) {
      sprite(background, 0f, 0f, size.width, size.height)
      drawRect(Color.Black.copy(alpha = .14f))
      val heroFrame = if (game.flash > 0 && !reduceMotion) 1 else if (game.shield > 0) 3 else 0
      sprite(zeus, size.width * .04f, size.height * .23f, size.width * .38f, size.height * .7f, column = heroFrame, columns = 4)
      sprite(atlas, size.width * .55f, size.height * .29f, size.width * .36f, size.height * .6f, column = game.enemy.sprite, columns = 4, rows = 2)
      if (game.flash > 0) {
        val path = Path().apply {
          moveTo(size.width * .32f, size.height * .44f)
          lineTo(size.width * .46f, size.height * .35f)
          lineTo(size.width * .5f, size.height * .51f)
          lineTo(size.width * .73f, size.height * .46f)
        }
        drawPath(path, Gold.copy(alpha = if (reduceMotion) .4f else .9f), style = Stroke(if (reduceMotion) 3f else 5f))
      }
      if (game.shield > 0) drawOval(Color.Cyan.copy(alpha = .6f), Offset(size.width * .06f, size.height * .19f), Size(size.width * .35f, size.height * .68f), style = Stroke(3f))
    }
    Text("${game.enemy.name} · ${ceil(game.enemy.hp).toInt()} HP", style = MaterialTheme.typography.bodyMedium)
    LinearProgressIndicator(progress = { (game.enemy.hp / game.enemy.maxHp).toFloat().coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.primary)
    Text("Zeus · ${ceil(game.hp).toInt()} / ${game.maxHp.toInt()} HP", style = MaterialTheme.typography.bodySmall)
    LinearProgressIndicator(progress = { (game.hp / game.maxHp).toFloat().coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth(), color = Color(0xFF16845D))
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      listOf("Bolt", "Storm", "Aegis").forEachIndexed { i, label ->
        Button(onClick = { game.ability(i); changed() }, enabled = !paused && game.cooldowns[i] <= 0, modifier = Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 5.dp, vertical = 10.dp)) {
          Text(if (game.cooldowns[i] > 0) "$label ${ceil(game.cooldowns[i]).toInt()}s" else label, maxLines = 1)
        }
      }
    }
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
      PauseControl(paused) { paused = !paused; save() }
      Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Auto attack", style = MaterialTheme.typography.labelLarge)
        Switch(checked = game.autoAttack, onCheckedChange = { game.autoAttack = it; changed() }, modifier = Modifier.semantics { contentDescription = "Auto attack" })
      }
    }
    if (game.offlineReward > 0) Text("While away: +${compact(game.offlineReward)} gold", style = MaterialTheme.typography.bodySmall)
    Text("Upgrades", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
    listOf("Lightning", "Harvest", "Aegis").forEachIndexed { index, label ->
      OutlinedButton(onClick = { game.upgrade(index); changed() }, enabled = !paused && game.gold >= game.cost(index), modifier = Modifier.fillMaxWidth()) {
        Text("$label ${game.upgrades[index]} · ${if (game.upgrades[index] >= 50) "Max level" else "${compact(game.cost(index))} gold"}")
      }
    }
    Text("Zones", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
    StormbreakModel.ZONES.forEachIndexed { i, z ->
      FilterChip(selected = game.zone == i, onClick = { game.selectZone(i); changed() }, enabled = game.level >= z.level,
        label = { Text(z.name + if (game.level < z.level) " · Level ${z.level}" else "") }, modifier = Modifier.fillMaxWidth())
    }
    TextButton(onClick = { showReset = true }) { Text("Reset Stormbreak progress") }
  }
  if (showReset) AlertDialog(onDismissRequest = { showReset = false }, title = { Text("Reset Stormbreak?") },
    text = { Text("This removes your saved gold, upgrades, and zone progress on this device.") },
    confirmButton = { TextButton(onClick = {
      val fresh = StormbreakModel(); game.restore(fresh.snapshot().put("savedAt", System.currentTimeMillis())); game.offlineReward = 0.0
      showReset = false; changed()
    }) { Text("Reset progress") } }, dismissButton = { TextButton(onClick = { showReset = false }) { Text("Cancel") } })
}

@Composable
private fun ProbabilityScreen(onBack: () -> Unit) {
  val context = LocalContext.current
  val store = remember { GameStorage(context) }
  val game = remember { ProbabilityModel().apply { restore(store.read("probability-engine")) } }
  var tick by remember { mutableIntStateOf(0) }
  var deckOpen by rememberSaveable { mutableStateOf(false) }
  var packMessage by rememberSaveable { mutableStateOf("") }
  var showReset by remember { mutableStateOf(false) }
  fun save() = store.save("probability-engine", game.snapshot())
  fun changed() { tick++; save() }
  // The model is plain Kotlin; observe its revision here as well as in rendered content.
  // Otherwise the initial idle value survives a spin and its cooldown never advances.
  val idle = remember(tick) { !game.auto && game.cooldown <= 0 }
  GameLoop(showReset || idle, { game.step(it.toDouble()); tick++ }, ::save)
  GamePage("Probability Engine", "Build your reel deck and discover neighboring symbol synergies.", onBack) {
    tick
    Stats("Credits" to compact(game.credits), "Spins" to game.spins.toString(), "Last return" to compact(game.result.reward))
    Surface(color = StageColor, shape = RoundedCornerShape(16.dp)) {
      Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        game.grid.chunked(3).forEachIndexed { row, cells ->
          Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            cells.forEachIndexed { column, id ->
              val symbol = ProbabilityModel.SYMBOLS.getValue(id)
              val rarityColor = when (symbol.rarity) { 1 -> Color(0xFF82D0F4); 2 -> Color(0xFFC5A0F8); 3 -> Gold; else -> Color(0xFFCFDCE9) }
              Surface(Modifier.weight(1f).aspectRatio(1f), color = Color(0xFF142B43), shape = RoundedCornerShape(10.dp)) {
                Column(Modifier.fillMaxSize().padding(5.dp).semantics { contentDescription = "${symbol.name}, value ${symbol.value.toInt()}" },
                  horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                  Text(symbol.glyph, color = rarityColor, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                  Text(symbol.name, color = Color.White, style = MaterialTheme.typography.labelSmall, maxLines = 1)
                  if ((row * 3 + column) in game.result.removed) Text("Consumed", color = Gold, style = MaterialTheme.typography.labelSmall)
                }
              }
            }
          }
        }
      }
    }
    if (game.jackpotReady) Text("Jackpot ready for the next spin", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
      Button(onClick = { game.spin(); changed() }, enabled = game.cooldown <= 0 && game.credits >= 5) { Text("Spin · 5 credits") }
      Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Auto", style = MaterialTheme.typography.labelLarge)
        Switch(checked = game.auto, onCheckedChange = { game.auto = it; changed() }, modifier = Modifier.semantics { contentDescription = "Automatic spins" })
      }
    }
    if (game.spins > 0) Text(if (game.result.synergies.isEmpty()) "No synergies this spin." else game.result.synergies.distinct().take(4).joinToString(" · "),
      style = MaterialTheme.typography.bodySmall, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
    Text("Upgrades", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
    OutlinedButton(onClick = { game.buyUpgrade(true); changed() }, enabled = game.luck < 20 && game.credits >= game.luckCost, modifier = Modifier.fillMaxWidth()) {
      Text("Luck injector ${game.luck} · ${compact(game.luckCost)} credits")
    }
    OutlinedButton(onClick = { game.buyUpgrade(false); changed() }, enabled = game.motor < 20 && game.credits >= game.motorCost, modifier = Modifier.fillMaxWidth()) {
      Text("Spin motor ${game.motor} · ${compact(game.motorCost)} credits")
    }
    OutlinedButton(onClick = { packMessage = game.buyPack().joinToString(", ") { ProbabilityModel.SYMBOLS.getValue(it).name }; changed() }, enabled = game.credits >= 40, modifier = Modifier.fillMaxWidth()) {
      Text("Buy 3 symbols · 40 credits")
    }
    if (packMessage.isNotBlank()) Text("Added to inventory: $packMessage", style = MaterialTheme.typography.bodySmall)
    TextButton(onClick = { deckOpen = !deckOpen }) { Text(if (deckOpen) "Close reel deck" else "Edit reel deck · ${game.deck.size}/32") }
    if (deckOpen) {
      Text("Keep 10–32 symbols. Add copies to increase how often a symbol appears.", style = MaterialTheme.typography.bodySmall)
      ProbabilityModel.SYMBOLS.values.forEach { symbol ->
        val count = game.deck.count { it == symbol.id }
        val owned = game.inventory[symbol.id] ?: 0
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
          Column(Modifier.weight(1f)) {
            Text(symbol.name, fontWeight = FontWeight.Medium)
            Text("$count in deck · $owned spare", style = MaterialTheme.typography.labelSmall)
          }
          TextButton(onClick = { game.removeSymbol(symbol.id); changed() }, enabled = count > 0 && game.deck.size > 10) { Text("−", modifier = Modifier.semantics { contentDescription = "Remove ${symbol.name} from deck" }) }
          TextButton(onClick = { game.addSymbol(symbol.id); changed() }, enabled = owned > 0 && game.deck.size < 32) { Text("+", modifier = Modifier.semantics { contentDescription = "Add ${symbol.name} to deck" }) }
        }
      }
    }
    TextButton(onClick = { game.auto = false; showReset = true }) { Text("Reset Probability Engine") }
  }
  if (showReset) AlertDialog(onDismissRequest = { showReset = false }, title = { Text("Reset Probability Engine?") }, text = { Text("This removes the credits, deck, and upgrades saved on this device.") },
    confirmButton = { TextButton(onClick = { game.restore(ProbabilityModel().snapshot()); showReset = false; changed() }) { Text("Reset progress") } },
    dismissButton = { TextButton(onClick = { showReset = false }) { Text("Cancel") } })
}

@Composable
private fun StarfallScreen(onBack: () -> Unit, reduceMotion: Boolean, onOpenWebsite: (() -> Unit)?) {
  val context = LocalContext.current
  val store = remember { GameStorage(context) }
  val game = remember { StarfallModel().apply { restore(store.read("project-starfall")) } }
  var tick by remember { mutableIntStateOf(0) }
  var paused by rememberSaveable { mutableStateOf(true) }
  var inventoryOpen by rememberSaveable { mutableStateOf(false) }
  var showReset by remember { mutableStateOf(false) }
  fun save() = store.save("project-starfall", game.snapshot())
  fun changed() { tick++; save() }
  val dead = remember(tick) { game.dead }
  GameLoop(paused || dead || showReset || inventoryOpen, { game.step(it); tick++ }, ::save)
  val background = GameBitmap("greenroot-meadow.webp")
  val hero = GameBitmap(game.hero.name.lowercase() + ".png")
  val enemy = GameBitmap("shardling.png")
  GamePage("Project Starfall: Offline Expedition", "A compact mission with progress saved on this device.", onBack,
    onOpenWebsite = onOpenWebsite?.let { openWebsite -> {
      game.move = 0f
      paused = true
      save()
      openWebsite()
    } }) {
    tick
    Stats("Level" to game.level.toString(), "HP" to "${game.hp.toInt()}/${game.maxHp.toInt()}", "Gold" to game.gold.toString())
    if (game.kills == 0) FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
      StarfallModel.CLASSES.forEachIndexed { i, c -> FilterChip(selected = game.classIndex == i, onClick = { game.chooseClass(i); changed() }, label = { Text(c.name) }) }
    }
    Canvas(Modifier.fillMaxWidth().aspectRatio(1.4f).clip(RoundedCornerShape(16.dp)).background(StageColor)
      .semantics { contentDescription = "Starfall Verge battlefield. Use movement, jump, and attack controls below." }) {
      sprite(background, 0f, 0f, size.width, size.height)
      drawRect(Color(0xFF062132).copy(alpha = .16f))
      val sx = size.width / 1000; val sy = size.height / 650
      val camera = (game.x - 450).coerceIn(0f, 3200f)
      withTransform({ scale(sx, sy, Offset.Zero) }) {
        StarfallModel.PLATFORMS.forEach { platform ->
          drawRoundRect(Color(0xFF284941), topLeft = Offset(platform.x - camera, platform.y), size = Size(platform.width, 22f), cornerRadius = androidx.compose.ui.geometry.CornerRadius(5f, 5f))
          drawLine(Color(0xFFBCE498), Offset(platform.x - camera, platform.y), Offset(platform.x + platform.width - camera, platform.y), 3f)
        }
        game.enemies.forEach { e ->
          if (e.x - camera in -100f..1100f) {
            sprite(enemy, e.x - camera - 37, e.y - 66, 74f, 70f)
            drawRect(Color(0xFF6B3540), Offset(e.x - camera - 28, e.y - 80), Size(56f, 5f))
            drawRect(Color(0xFFE8AB80), Offset(e.x - camera - 28, e.y - 80), Size(56f * (e.hp / e.maxHp).coerceIn(0f, 1f), 5f))
          }
        }
        sprite(hero, game.x - camera - 39, game.y - 110, 78f, 114f)
        if (game.attackFlash > 0) drawLine(Gold.copy(alpha = if (reduceMotion) .4f else .9f), Offset(game.x - camera, game.y - 60),
          Offset(game.x - camera + game.facing * game.hero.range, game.y - 55), if (reduceMotion) 2f else 4f)
      }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
      HoldControl("Left", !paused && !game.dead && !inventoryOpen, Modifier.weight(1f), { game.move = -1f }, { game.move = 0f }, { game.x = (game.x - 45).coerceAtLeast(32f); game.facing = -1f; tick++ })
      HoldControl("Right", !paused && !game.dead && !inventoryOpen, Modifier.weight(1f), { game.move = 1f }, { game.move = 0f }, { game.x = (game.x + 45).coerceAtMost(4168f); game.facing = 1f; tick++ })
      Button(onClick = { game.jump(); changed() }, enabled = !paused && !game.dead && game.grounded && !inventoryOpen, modifier = Modifier.weight(1f), contentPadding = PaddingValues(5.dp)) { Text("Jump") }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = { game.strike(); changed() }, enabled = !paused && !game.dead && !inventoryOpen && game.attackCooldown <= 0, modifier = Modifier.weight(1f)) { Text("Attack") }
      OutlinedButton(onClick = { game.strike(true); changed() }, enabled = !paused && !game.dead && !inventoryOpen && game.skillCooldown <= 0 && game.attackCooldown <= 0, modifier = Modifier.weight(1f)) {
        Text(if (game.skillCooldown > 0) "Skill ${ceil(game.skillCooldown).toInt()}s" else "Power strike")
      }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (game.dead) Button(onClick = { game.revive(); paused = false; changed() }) { Text("Revive · 10% gold") }
      else PauseControl(paused) { game.move = 0f; paused = !paused; save() }
      TextButton(onClick = { game.move = 0f; inventoryOpen = !inventoryOpen }) { Text(if (inventoryOpen) "Close inventory" else "Inventory") }
    }
    if (game.enemies.isEmpty()) Button(onClick = { game.nextWave(); paused = false; changed() }) { Text("Next expedition") }
    Text("${game.kills.coerceAtMost(18)}/18 defeated · ${if (game.questClaimed) "Verge secured" else "Secure Starfall Verge"}", style = MaterialTheme.typography.bodyMedium)
    Text(game.message, style = MaterialTheme.typography.bodySmall)
    Text("${game.hero.name} · ${game.gear.name} · ${game.power.toInt()} power", fontWeight = FontWeight.Medium)
    if (inventoryOpen) {
      OutlinedButton(onClick = { game.rest(); changed() }, enabled = game.gold >= 20 && game.hp < game.maxHp) { Text("Recover health · 20 gold") }
      if (game.inventory.isEmpty()) Text("Defeat enemies to collect gear.", style = MaterialTheme.typography.bodySmall)
      game.inventory.forEachIndexed { index, loot -> OutlinedButton(onClick = { game.equip(index); changed() }, modifier = Modifier.fillMaxWidth()) {
        Text("Equip ${loot.name} · +${loot.bonus} power · ${loot.rarity}")
      } }
    }
    TextButton(onClick = { game.move = 0f; showReset = true }) { Text("Reset Starfall character") }
  }
  if (showReset) AlertDialog(onDismissRequest = { showReset = false }, title = { Text("Reset Starfall character?") },
    text = { Text("This removes your character, equipment, and expedition progress on this device.") },
    confirmButton = { TextButton(onClick = { game.restore(StarfallModel().snapshot()); paused = true; inventoryOpen = false; showReset = false; changed() }) { Text("Reset character") } },
    dismissButton = { TextButton(onClick = { showReset = false }) { Text("Cancel") } })
}

@Composable
private fun HoldControl(label: String, enabled: Boolean, modifier: Modifier, onPress: () -> Unit, onRelease: () -> Unit, onAccessibleClick: () -> Unit) {
  val press by rememberUpdatedState(onPress)
  val release by rememberUpdatedState(onRelease)
  DisposableEffect(enabled) { onDispose { release() } }
  Surface(modifier.heightIn(min = 48.dp).semantics {
    role = Role.Button; contentDescription = "Move $label"
    if (!enabled) disabled()
    onClick { if (enabled) onAccessibleClick(); enabled }
  }.pointerInput(enabled) { detectTapGestures(onPress = { if (enabled) { press(); try { tryAwaitRelease() } finally { release() } } }) },
    color = if (enabled) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainer,
    shape = RoundedCornerShape(24.dp)) {
    Box(contentAlignment = Alignment.Center) { Text(if (label == "Left") "‹ Left" else "Right ›", style = MaterialTheme.typography.labelLarge) }
  }
}

private fun compact(value: Double): String = when {
  !value.isFinite() -> "Max"
  value >= 1e12 -> "%.1fT".format(value / 1e12)
  value >= 1e9 -> "%.1fB".format(value / 1e9)
  value >= 1e6 -> "%.1fM".format(value / 1e6)
  value >= 1000 -> "%.1fK".format(value / 1000)
  else -> value.toInt().toString()
}
