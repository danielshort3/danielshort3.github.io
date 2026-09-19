package me.danielshort.app.nativefeatures

import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.Saver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

private val gameSaver = Saver<RouletteGame, ArrayList<Int>>(
  save = { game ->
    arrayListOf(game.chips, game.spins, game.wins).apply {
      game.recent.forEach { add(it.pocket.number); add(it.bet.ordinal) }
    }
  },
  restore = { saved ->
    RouletteGame(saved[0], saved[1], saved[2], saved.drop(3).chunked(2).map {
      RouletteRound(RoulettePocket(it[0]), RouletteBet.entries[it[1]])
    })
  }
)

/** Native practice game for the public roulette catalog entry. No money or account is used. */
@Composable
fun NativeGameScreen(onBack: () -> Unit) {
  var game by rememberSaveable(stateSaver = gameSaver) { mutableStateOf(RouletteGame()) }
  var betName by rememberSaveable { mutableStateOf(RouletteBet.Red.name) }
  val bet = RouletteBet.valueOf(betName)
  val latest = game.recent.firstOrNull()
  Column(
    Modifier.fillMaxSize()
      .background(MaterialTheme.colorScheme.background)
      .safeDrawingPadding()
      .imePadding()
      .verticalScroll(rememberScrollState())
      .padding(horizontal = 20.dp),
    verticalArrangement = Arrangement.spacedBy(16.dp)
  ) {
    NativeFeatureHeader("Double-Zero Roulette", "A probability game with practice chips.", onBack)
    Surface(shape = MaterialTheme.shapes.medium, color = MaterialTheme.colorScheme.surfaceContainerLow,
      modifier = Modifier.fillMaxWidth()) {
      Row(Modifier.padding(16.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
        RouletteStat("Chips", game.chips.toString())
        RouletteStat("Spins", game.spins.toString())
        RouletteStat("Wins", game.wins.toString())
      }
    }
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally,
      verticalArrangement = Arrangement.spacedBy(12.dp)) {
      AnimatedContent(targetState = game.spins to latest?.pocket, label = "Spin result") { (_, pocket) ->
        Box(
          Modifier.size(112.dp).background(pocketColor(pocket?.color), CircleShape).semantics {
            contentDescription = if (pocket == null) "Ready to spin" else "${pocket.label}, ${pocket.color.name.lowercase()}"
          },
          contentAlignment = Alignment.Center
        ) {
          Text(pocket?.label ?: "?", color = Color.White, style = MaterialTheme.typography.displayMedium,
            fontWeight = FontWeight.Bold)
        }
      }
      Text(
        when {
          game.chips == 0 -> "No chips left. Start a new game."
          latest == null -> "Choose a bet, then spin."
          latest.won -> "${latest.bet.label} wins · +1 chip"
          else -> "${latest.bet.label} loses · −1 chip"
        },
        textAlign = TextAlign.Center,
        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
        style = MaterialTheme.typography.titleSmall
      )
      Column {
        RouletteBet.entries.chunked(2).forEach { bets ->
          Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            bets.forEach { option ->
              FilterChip(
                selected = bet == option,
                onClick = { betName = option.name },
                label = { Text(option.label, modifier = Modifier.fillMaxWidth(), textAlign = TextAlign.Center) },
                enabled = game.chips > 0,
                modifier = Modifier.weight(1f).height(48.dp)
              )
            }
          }
        }
      }
      Button(onClick = { game = game.play(bet) }, enabled = game.chips > 0,
        modifier = Modifier.fillMaxWidth().height(52.dp)) { Text("Spin · 1 chip") }
      Text("0 and 00 lose. Winning bets pay 1:1.", style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    if (game.recent.isNotEmpty()) {
      Text("Recent spins", style = MaterialTheme.typography.titleSmall, modifier = Modifier.semantics { heading() })
      Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        game.recent.forEach { round ->
          Box(Modifier.weight(1f).height(40.dp).background(pocketColor(round.pocket.color), MaterialTheme.shapes.small)
            .semantics { contentDescription = "${round.pocket.label}, ${round.pocket.color.name.lowercase()}" },
            contentAlignment = Alignment.Center) {
            Text(round.pocket.label, color = Color.White, style = MaterialTheme.typography.labelMedium)
          }
        }
        repeat(8 - game.recent.size) { Spacer(Modifier.weight(1f)) }
      }
    }
    TextButton(onClick = { game = RouletteGame() }, modifier = Modifier.align(Alignment.CenterHorizontally)) {
      Text("New game")
    }
    Spacer(Modifier.height(16.dp))
  }
}

@Composable
private fun RouletteStat(label: String, value: String) {
  Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Text(value, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
    Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
  }
}

private fun pocketColor(color: RouletteColor?): Color = when (color) {
  RouletteColor.Red -> Color(0xFFB73C37)
  RouletteColor.Green -> Color(0xFF06756D)
  else -> Color(0xFF0A2342)
}
