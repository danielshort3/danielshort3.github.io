package me.danielshort.app.nativefeatures.demos

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlin.math.expm1

internal data class TipEstimate(val tip: Double, val tipLow: Double, val tipHigh: Double,
  val rate: Double, val rateLow: Double, val rateHigh: Double, val outsideTrainingRange: Boolean)

/** Version 4 coefficients from js/demos/pizza-tips-model.js (2025-12-22).
 * Native city selection supplies the same city category as the web polygon lookup. */
internal object PizzaTipsModel {
  val cities = listOf("Frisco", "Plano", "The Colony", "Lewisville", "Carrollton", "McKinney", "Allen")
  val housingTypes = listOf("Residential", "Apartment", "Hotel", "Business")
  private val zScores = mapOf(80 to 1.282, 85 to 1.44, 90 to 1.645, 95 to 1.96)

  fun estimate(cost: Double, hour: Int, city: String, housing: String, confidence: Int): TipEstimate {
    require(cost.isFinite() && cost in .01..1000.0 && hour in 0..23 && city in cities && housing in housingTypes && confidence in zScores)
    val logTip = 1.5823081640335854 + .01022700228091093 * cost +
      (if (city == "Plano") -.07521221370771652 else 0.0) +
      when (housing) { "Apartment" -> -.07491206308191074; "Business" -> -.22453944133434225; else -> 0.0 }
    val logRate = .1303849400342169 - .0009515125883025411 * cost + .004577273574251591 * hour +
      if (city == "Plano") -.014459822678887454 else 0.0
    val z = zScores.getValue(confidence)
    fun inverse(value: Double) = expm1(value).coerceAtLeast(0.0)
    return TipEstimate(inverse(logTip), inverse(logTip - z * .40663783618126387), inverse(logTip + z * .40663783618126387),
      inverse(logRate), inverse(logRate - z * .09039342874618946), inverse(logRate + z * .09039342874618946),
      cost !in 5.41..243.02 || hour !in 11..21)
  }
}

@Composable
internal fun PizzaTipsDemo(onBack: () -> Unit) {
  var cost by rememberSaveable { mutableStateOf("41.77") }
  var hour by rememberSaveable { mutableStateOf("18") }
  var city by rememberSaveable { mutableStateOf("Frisco") }
  var housing by rememberSaveable { mutableStateOf("Residential") }
  var confidence by rememberSaveable { mutableStateOf("80") }
  var showEstimate by rememberSaveable { mutableStateOf(true) }
  var error by rememberSaveable { mutableStateOf("") }
  val result = remember(cost, hour, city, housing, confidence, showEstimate) {
    if (showEstimate) cost.toDoubleOrNull()?.let { PizzaTipsModel.estimate(it, hour.toInt(), city, housing, confidence.toInt()) } else null
  }
  DemoPage("Pizza Tip Estimator", "Explore the website’s saved regression models with a delivery scenario.", onBack) {
    OutlinedTextField(cost, { cost = it.take(12); showEstimate = false; error = "" }, label = { Text("Order cost ($)") }, singleLine = true,
      keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal), modifier = Modifier.fillMaxWidth())
    DemoSelect("City", PizzaTipsModel.cities, city, { city = it; showEstimate = false })
    DemoSelect("Housing", PizzaTipsModel.housingTypes, housing, { housing = it; showEstimate = false })
    DemoSelect("Order hour (24h)", (0..23).map(Int::toString), hour, { hour = it; showEstimate = false })
    DemoSelect("Interval level", listOf("80%", "85%", "90%", "95%"), "$confidence%", { confidence = it.removeSuffix("%"); showEstimate = false })
    Button(onClick = {
      val parsed = cost.toDoubleOrNull()
      if (parsed == null || !parsed.isFinite() || parsed !in .01..1000.0) error = "Enter an order cost between $0.01 and $1,000."
      else { showEstimate = true; error = "" }
    }) { Text("Update estimate") }
    if (error.isNotBlank()) Text(error, color = MaterialTheme.colorScheme.error)
    result?.let { estimate ->
      Surface(shape = RoundedCornerShape(16.dp), color = MaterialTheme.colorScheme.surfaceContainerLow) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
          Text("Estimated tip", style = MaterialTheme.typography.labelLarge)
          Text(money(estimate.tip), style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
          Text("$confidence% interval: ${money(estimate.tipLow)} – ${money(estimate.tipHigh)}")
          HorizontalDivider()
          Text("Separate tip-rate model: ${percent(estimate.rate)}")
          Text("$confidence% interval: ${percent(estimate.rateLow)} – ${percent(estimate.rateHigh)}", style = MaterialTheme.typography.bodySmall)
        }
      }
      if (estimate.outsideTrainingRange) Text("This scenario is outside the training range: $5.41–$243.02 and 11:00–21:59. Interpret the estimate cautiously.", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
    }
    Text("Calculated on your device from 1,251 historical deliveries. Intervals use the saved model’s error estimate; individual tips vary. The amount and percentage come from separate models.", style = MaterialTheme.typography.bodySmall)
  }
}
