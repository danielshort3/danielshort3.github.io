package me.danielshort.app.nativefeatures.demos

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import org.json.JSONArray
import org.json.JSONObject

@Composable
internal fun TableauNativeView(id: String, data: JSONObject) {
  if (id == "pizzaDashboard") PizzaView(data) else UfoView(data)
}

@Composable
private fun PizzaView(data: JSONObject) {
  val all = remember(data) { data.getJSONArray("rows").arrayRows() }
  var city by rememberSaveable { mutableStateOf("All") }
  var housing by rememberSaveable { mutableStateOf("All") }
  var month by rememberSaveable { mutableStateOf("All") }
  var metric by rememberSaveable { mutableStateOf("Average tip") }
  DemoSelect("City", listOf("All") + all.map { it.getString(1) }.distinct().sorted(), city, { city = it })
  DemoSelect("Housing", listOf("All") + all.map { it.getString(2) }.distinct().sorted(), housing, { housing = it })
  DemoSelect("Month", listOf("All") + all.map { it.getString(0).take(7) }.distinct().sorted(), month, { month = it })
  val filtered = remember(all, city, housing, month) { filterPizza(all, city, housing, month) }
  Text("${filtered.size} deliveries", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
  if (filtered.isEmpty()) Text("No deliveries match these filters.")
  else {
    Text("Average tip: ${money(filtered.map { it.getDouble(4) }.average())}")
    Text("Average order: ${money(filtered.map { it.getDouble(3) }.average())}")
    Text("Average delivery: ${number(filtered.map { it.getDouble(5) }.average())} minutes")
    DemoSelect("Metric", listOf("Average tip", "Deliveries", "Average delivery minutes"), metric, { metric = it })
    val groups = filtered.groupBy { it.getString(0).take(7) }.toSortedMap()
    val points = groups.map { (_, rows) -> pizzaMetric(rows, metric) }
    Text("By month", fontWeight = FontWeight.SemiBold)
    LinePlot(points, "$metric by month")
    Text("${groups.firstKey()} – ${groups.lastKey()}", style = MaterialTheme.typography.bodySmall)
    Text("By housing type", fontWeight = FontWeight.SemiBold)
    Bars(filtered.groupBy { it.getString(2) }.map { (label, rows) -> label to pizzaMetric(rows, metric) }.sortedByDescending { it.second }, metric == "Average tip")
    Text("By city", fontWeight = FontWeight.SemiBold)
    Bars(filtered.groupBy { it.getString(1) }.map { (label, rows) -> label to pizzaMetric(rows, metric) }.sortedByDescending { it.second }, metric == "Average tip")
  }
  Text(data.optString("scope", "Historical delivery records from the published Tableau workbook."), style = MaterialTheme.typography.bodySmall)
  Text("Filters recalculate these charts on your device. This is descriptive analysis; tip prediction remains disabled.", style = MaterialTheme.typography.bodySmall)
}

internal fun filterPizza(rows: List<JSONArray>, city: String, housing: String, month: String) = rows.filter {
  (city == "All" || it.getString(1) == city) && (housing == "All" || it.getString(2) == housing) && (month == "All" || it.getString(0).startsWith(month))
}

private fun pizzaMetric(rows: List<JSONArray>, metric: String): Double = when (metric) {
  "Deliveries" -> rows.size.toDouble()
  "Average delivery minutes" -> rows.map { it.getDouble(5) }.average()
  else -> rows.map { it.getDouble(4) }.average()
}

@Composable
private fun UfoView(data: JSONObject) {
  val all = remember(data) { data.getJSONArray("rows").arrayRows() }
  var year by rememberSaveable { mutableStateOf("2013") }
  var state by rememberSaveable { mutableStateOf("All") }
  var shape by rememberSaveable { mutableStateOf("All") }
  var chart by rememberSaveable { mutableStateOf("Year") }
  val years = remember(all) { listOf("All") + all.map { it.getInt(0).toString() }.distinct().sortedDescending() }
  val states = remember(all) { listOf("All") + all.map { it.getString(3).uppercase() }.distinct().sorted() }
  val shapes = remember(all) { listOf("All") + all.map { it.getString(4) }.distinct().sorted() }
  DemoSelect("Year", years, year, { year = it })
  DemoSelect("State", states, state, { state = it })
  DemoSelect("Reported shape", shapes, shape, { shape = it })
  val rows = remember(all, year, state, shape) { filterUfo(all, year, state, shape) }
  Text("${number(rows.sumOf { it.getDouble(5) })} reported sightings", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
  DemoSelect("Chart by", listOf("Year", "Month", "Hour", "State", "Shape"), chart, { chart = it })
  val index = when (chart) { "Year" -> 0; "Month" -> 1; "Hour" -> 2; "State" -> 3; else -> 4 }
  val groups = remember(rows, chart) { rows.groupBy { it.get(index).toString() }.map { (label, grouped) -> label to grouped.sumOf { it.getDouble(5) } } }
  if (index <= 2) {
    val ordered = completeReportPeriods(groups, index)
    LinePlot(ordered.map { it.second }, "Reported sighting counts by ${chart.lowercase()}")
    if (ordered.isNotEmpty()) Text("${ordered.first().first} – ${ordered.last().first}", style = MaterialTheme.typography.bodySmall)
    Bars(ordered.sortedByDescending { it.second }.take(12))
  } else Bars(groups.sortedByDescending { it.second }.take(20))
  Text(data.optString("scope", "Historical reported sightings in the contiguous United States."), style = MaterialTheme.typography.bodySmall)
  Text("These are reported observations, not verified extraterrestrial events. Counts are not adjusted for population or reporting differences.", style = MaterialTheme.typography.bodySmall)
}

internal fun JSONArray.arrayRows() = (0 until length()).map { getJSONArray(it) }
internal fun filterUfo(rows: List<JSONArray>, year: String, state: String, shape: String) = rows.filter {
  (year == "All" || it.getInt(0).toString() == year) && (state == "All" || it.getString(3).equals(state, true)) && (shape == "All" || it.getString(4) == shape)
}

/** Missing reporting periods are zero counts, not adjacent time points. */
internal fun completeReportPeriods(groups: List<Pair<String, Double>>, index: Int): List<Pair<String, Double>> {
  if (groups.isEmpty()) return emptyList()
  val counts = groups.associate { it.first.toInt() to it.second }
  val range = when (index) { 1 -> 1..12; 2 -> 0..23; else -> counts.keys.min()..counts.keys.max() }
  return range.map { it.toString() to (counts[it] ?: 0.0) }
}
