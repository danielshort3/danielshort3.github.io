@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package me.danielshort.app.nativefeatures.demos

import android.content.Context
import android.util.AtomicFile
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.NumberFormat
import java.util.Locale

private val dashboardSources = mapOf(
  "babynames" to "baby-names.json",
  "covidAnalysis" to "covid-outbreak/meta.json",
  "retailStore" to "retail-loss-sales/data.json",
  "targetEmptyPackage" to "target-empty-package/data.json"
)

/** Cache only published, historical datasets; no personal input is sent. */
private suspend fun loadDataset(context: Context, path: String): Pair<JSONObject, Boolean> = withContext(Dispatchers.IO) {
  require(path.matches(Regex("[A-Za-z0-9/._-]+")) && !path.contains(".."))
  val file = File(context.cacheDir, "native-datasets/${path.replace('/', '_')}")
  try {
    val data = DemoApi.json("$SITE/demos/data/$path")
    file.parentFile?.mkdirs()
    val cache = AtomicFile(file)
    val output = cache.startWrite()
    try { output.write(data.toString().toByteArray(Charsets.UTF_8)); cache.finishWrite(output) }
    catch (e: Exception) { cache.failWrite(output); throw e }
    data to false
  } catch (e: CancellationException) { throw e }
  catch (e: Exception) {
    if (file.isFile) JSONObject(AtomicFile(file).openRead().bufferedReader().use { it.readText() }) to true else throw e
  }
}

@Composable
internal fun NativeDashboardDemo(projectId: String, onBack: () -> Unit) {
  val context = LocalContext.current
  var data by remember(projectId) { mutableStateOf<JSONObject?>(null) }
  var status by remember(projectId) { mutableStateOf("Loading published data…") }
  var attempt by remember { mutableStateOf(0) }
  var loading by remember { mutableStateOf(true) }
  LaunchedEffect(projectId, attempt) {
    loading = true
    try {
      if (projectId == "pizzaDashboard" || projectId == "ufoDashboard") {
        data = withContext(Dispatchers.IO) { context.assets.open("native-demos/tableau-${if (projectId == "pizzaDashboard") "pizza" else "ufo"}.json").bufferedReader().use { JSONObject(it.readText()) } }
        status = "Published Tableau data · bundled snapshot"
      } else {
        val (loaded, cached) = loadDataset(context, dashboardSources.getValue(projectId))
        data = loaded; status = if (cached) "Offline · saved historical data" else "Published historical data · up to date"
      }
    } catch (e: CancellationException) { throw e }
    catch (_: Exception) { status = "Data could not be loaded. Please check your connection." }
    finally { loading = false }
  }
  val title = when (projectId) {
    "babynames" -> "Baby Name Explorer"
    "covidAnalysis" -> "COVID-19 Outbreak Drivers"
    "retailStore" -> "Retail Sales & Loss"
    "targetEmptyPackage" -> "Empty-Package Dashboard"
    "pizzaDashboard" -> "Pizza Delivery Dashboard"
    else -> "UFO Sightings Dashboard"
  }
  val description = when (projectId) {
    "babynames" -> "Explore personal ratings and model recommendations."
    "covidAnalysis" -> "Explore historical model output from July 2022 to June 2023."
    "retailStore", "targetEmptyPackage" -> "Filter the website’s anonymized historical records."
    else -> "Explore the same historical data with native filters and charts."
  }
  DemoPage(title, description, onBack) {
    DemoStatus(loading, status, if (!loading) ({ attempt++ }) else null)
    data?.let { source ->
      when (projectId) {
        "babynames" -> BabyNamesView(source)
        "covidAnalysis" -> CovidView(source)
        "retailStore" -> RetailView(source)
        "targetEmptyPackage" -> EmptyPackageView(source)
        else -> TableauNativeView(projectId, source)
      }
    }
  }
}

@Composable
private fun BabyNamesView(data: JSONObject) {
  var gender by rememberSaveable { mutableStateOf("Girl") }
  var mode by rememberSaveable { mutableStateOf("Recommendations") }
  var query by rememberSaveable { mutableStateOf("") }
  var minimum by rememberSaveable { mutableStateOf(0f) }
  val rows = remember(data, gender, mode, query, minimum) {
    val raw = data.getJSONObject(if (mode == "Ratings") "ratings" else "recommendations").getJSONArray(if (gender == "Girl") "F" else "M")
    filterNames(raw, query, minimum.toDouble(), mode == "Ratings")
  }
  FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    listOf("Girl", "Boy").forEach { FilterChip(gender == it, { gender = it }, label = { Text(it) }) }
    listOf("Recommendations", "Ratings").forEach { FilterChip(mode == it, { mode = it }, label = { Text(it) }) }
  }
  OutlinedTextField(query, { query = it.take(80) }, label = { Text("Find a name") }, singleLine = true, modifier = Modifier.fillMaxWidth().protectChromeWhileEditing())
  Text("Minimum ${if (mode == "Ratings") "rating" else "predicted preference"}: ${minimum.toInt()} / 10")
  Slider(minimum, { minimum = it }, valueRange = 0f..10f, steps = 9)
  Text("${rows.size} matching names", fontWeight = FontWeight.SemiBold)
  rows.take(80).forEach { item -> ScoreRow(item.getString("name"), item.getDouble(if (mode == "Ratings") "rating" else "predicted") / 10, String.format(Locale.US, "%.1f / 10", item.getDouble(if (mode == "Ratings") "rating" else "predicted"))) }
  if (rows.isEmpty()) Text("No names match these filters.")
  if (rows.size > 80) Text("Showing the first 80. Refine your search for more.", style = MaterialTheme.typography.bodySmall)
  Text("Ratings are personal scores from 1 to 10. Repeated names remain separate rating entries. Recommendations estimate personal preference from name similarity; they are not population forecasts.", style = MaterialTheme.typography.bodySmall)
}

internal fun filterNames(array: JSONArray, query: String, minimum: Double, ratings: Boolean): List<JSONObject> =
  array.objects().filter { it.optString("name").contains(query.trim(), true) && it.optDouble(if (ratings) "rating" else "predicted", -1.0) >= minimum }
    .sortedWith(compareByDescending<JSONObject> { it.getDouble(if (ratings) "rating" else "predicted") }.thenBy { it.getString("name") })

@Composable
private fun CovidView(meta: JSONObject) {
  val context = LocalContext.current
  val dates = remember(meta) { meta.getJSONArray("dates").strings() }
  var date by rememberSaveable { mutableStateOf(meta.getString("latest")) }
  var query by rememberSaveable { mutableStateOf("") }
  var selected by rememberSaveable { mutableStateOf("") }
  var snapshot by remember { mutableStateOf<JSONObject?>(null) }
  var history by remember { mutableStateOf<JSONObject?>(null) }
  var error by remember { mutableStateOf("") }
  var loading by remember { mutableStateOf(false) }
  LaunchedEffect(date) {
    loading = true; error = ""; snapshot = null
    try { snapshot = loadDataset(context, "covid-outbreak/by-date/$date.json").first }
    catch (e: CancellationException) { throw e }
    catch (_: Exception) { error = "This date could not load. Choose another date or retry." }
    finally { loading = false }
  }
  LaunchedEffect(selected) {
    history = null
    if (selected.isNotBlank()) {
      try { history = loadDataset(context, "covid-outbreak/state/$selected.json").first }
      catch (e: CancellationException) { throw e }
      catch (_: Exception) { error = "State history could not load." }
    }
  }
  DemoSelect("Date", dates.reversed(), date, { date = it })
  OutlinedTextField(query, { query = it.take(60) }, label = { Text("Find a state") }, singleLine = true, modifier = Modifier.fillMaxWidth().protectChromeWhileEditing())
  if (loading) LinearProgressIndicator(Modifier.fillMaxWidth())
  if (error.isNotBlank()) Text(error, color = MaterialTheme.colorScheme.error)
  val rows = snapshot?.getJSONArray("states")?.objects()?.filter { it.getString("name").contains(query, true) } ?: emptyList()
  if (selected.isNotBlank()) {
    rows.find { it.getString("id") == selected }?.let { state ->
      Text(state.getString("name"), style = MaterialTheme.typography.titleLarge)
      ScoreRow("Model risk", state.getDouble("prob"))
      ScoreRow("ICU utilization", state.getDouble("icuUtilization"))
      state.optJSONArray("drivers")?.objects()?.forEach { ScoreRow(it.getString("label"), it.getDouble("value")) }
      history?.let { historical ->
        Text("Model risk over time", fontWeight = FontWeight.SemiBold)
        val entries = historical.getJSONArray("history").objects()
        LinePlot(entries.map { it.getDouble("prob") }, "Historical model risk for ${state.getString("name")}")
        Text("${entries.first().getString("date")} – ${entries.last().getString("date")}", style = MaterialTheme.typography.bodySmall)
      }
    }
    TextButton(onClick = { selected = "" }) { Text("All states") }
  }
  rows.take(60).forEach { state ->
    TextButton(onClick = { selected = state.getString("id") }, modifier = Modifier.fillMaxWidth()) {
      Text(state.getString("name"), Modifier.weight(1f)); Text(percent(state.getDouble("prob")))
    }
  }
  Text("Historical research output, not current surveillance or medical guidance. Risk is the model’s probability for its trained outcome.", style = MaterialTheme.typography.bodySmall)
}

@Composable
private fun EmptyPackageView(data: JSONObject) {
  val all = remember(data) { data.getJSONArray("rows").objects() }
  var year by rememberSaveable { mutableStateOf("All") }
  var location by rememberSaveable { mutableStateOf("All") }
  var group by rememberSaveable { mutableStateOf("Department") }
  var metric by rememberSaveable { mutableStateOf("Estimated value") }
  val years = remember(all) { listOf("All") + all.map { it.getString("datetime").take(4) }.distinct().sortedDescending() }
  val locations = remember(all) { listOf("All") + all.map { it.getString("location") }.distinct().sorted() }
  DemoSelect("Year", years, year, { year = it })
  DemoSelect("Location", locations, location, { location = it })
  val rows = remember(all, year, location) { all.filter { (year == "All" || it.getString("datetime").startsWith(year)) && (location == "All" || it.getString("location") == location) } }
  Text("${rows.size} packages · ${money(rows.sumOf { it.getDouble("value") })} estimated value", fontWeight = FontWeight.SemiBold)
  DemoSelect("Group by", listOf("Department", "Location", "Condition", "Month"), group, { group = it })
  FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    listOf("Estimated value", "Packages").forEach { FilterChip(metric == it, { metric = it }, label = { Text(it) }) }
  }
  val groups = remember(rows, group, metric) {
    aggregateRows(rows, when (group) { "Department" -> "department"; "Location" -> "location"; "Condition" -> "condition"; else -> "datetime" }, if (metric == "Packages") null else "value", group == "Month")
  }
  Bars(groups.take(15), moneyValues = metric != "Packages")
  Text("${data.getJSONObject("meta").getString("startDate").take(10)} – ${data.getJSONObject("meta").getString("endDate").take(10)} · Anonymized records. Estimated package value is not proven theft.", style = MaterialTheme.typography.bodySmall)
}

internal fun aggregateRows(rows: List<JSONObject>, key: String, valueKey: String?, month: Boolean = false): List<Pair<String, Double>> = rows
  .groupBy { it.optString(key, "Unknown").let { label -> if (month) label.take(7) else label }.ifBlank { "Unknown" } }
  .map { (label, group) -> label to if (valueKey == null) group.size.toDouble() else group.sumOf { it.optDouble(valueKey, 0.0) } }
  .sortedByDescending { it.second }

@Composable
private fun RetailView(data: JSONObject) {
  var mode by rememberSaveable { mutableStateOf("Sales") }
  var year by rememberSaveable { mutableStateOf("All") }
  var metric by rememberSaveable { mutableStateOf("sales") }
  FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    listOf("Sales", "Incidents", "Inventory", "Packages").forEach { FilterChip(mode == it, { mode = it; metric = if (it == "Sales") "sales" else if (it == "Incidents") "incidents" else "estimatedValue"; year = "All" }, label = { Text(it) }) }
  }
  when (mode) {
    "Sales", "Incidents" -> {
      val sales = mode == "Sales"
      val rows = data.getJSONObject(if (sales) "sales" else "incidents").getJSONArray(if (sales) "weekly" else "monthly").objects()
      val key = if (sales) "week" else "month"
      val years = listOf("All") + rows.map { it.getString(key).take(4) }.distinct().sortedDescending()
      DemoSelect("Year", years, year, { year = it })
      val metricLabels = if (sales) linkedMapOf("sales" to "Sales", "online" to "Online", "driveUp" to "Drive up", "returns" to "Returns", "units" to "Units") else linkedMapOf("incidents" to "Incidents", "proven" to "Proven value")
      DemoSelect("Metric", metricLabels.values.toList(), metricLabels[metric] ?: metric, { label -> metric = metricLabels.entries.first { it.value == label }.key })
      val filtered = rows.filter { year == "All" || it.getString(key).startsWith(year) }
      val values = filtered.map { it.optDouble(metric, 0.0) }
      val currency = metric !in setOf("units", "incidents")
      Text("Total: ${if (currency) money(values.sum()) else number(values.sum())}", style = MaterialTheme.typography.titleLarge)
      LinePlot(values, "${metricLabels[metric]} across ${filtered.size} periods")
      Text("${filtered.firstOrNull()?.getString(key).orEmpty()} – ${filtered.lastOrNull()?.getString(key).orEmpty()}", style = MaterialTheme.typography.bodySmall)
      if (sales) Text("Sales series: ${data.getJSONObject("sales").getString("store")}", style = MaterialTheme.typography.bodySmall)
      else if (year == "All") {
        Text("Regions · all published years", fontWeight = FontWeight.SemiBold)
        Bars(data.getJSONObject("incidents").getJSONArray("regions").objects().map { it.getString("region") to it.getDouble(if (metric == "incidents") "incidents" else "proven") }, currency)
      }
    }
    "Inventory" -> {
      Text("Average shortage rate by year", fontWeight = FontWeight.SemiBold)
      data.getJSONObject("inventory").getJSONArray("years").objects().forEach { ScoreRow(it.getInt("year").toString(), it.getDouble("avgShortagePercent")) }
      Text("Highest published store shortage rates (${data.getJSONObject("inventory").getInt("year")})", fontWeight = FontWeight.SemiBold)
      data.getJSONObject("inventory").getJSONArray("stores").objects().take(10).forEach { ScoreRow(it.getString("store"), it.getDouble("shortagePercent")) }
    }
    else -> {
      var breakdown by rememberSaveable { mutableStateOf("Areas") }
      DemoSelect("Group by", listOf("Areas", "Employees", "Conditions"), breakdown, { breakdown = it })
      val key = breakdown.lowercase(Locale.US)
      val labelKey = when (breakdown) { "Areas" -> "area"; "Employees" -> "employee"; else -> "condition" }
      Bars(data.getJSONObject("emptyPackages").getJSONArray(key).objects().map { it.getString(labelKey) to it.getDouble("estimatedValue") }, true)
    }
  }
  Text("Historical, anonymized records. Sales cover one selected store; incident and inventory summaries cover the published store network. These are different scopes.", style = MaterialTheme.typography.bodySmall)
}

@Composable
internal fun LinePlot(values: List<Double>, description: String) {
  val primary = MaterialTheme.colorScheme.primary
  val gridColor = MaterialTheme.colorScheme.outlineVariant
  val finite = values.filter(Double::isFinite)
  val max = finite.maxOrNull()?.coerceAtLeast(1.0) ?: 1.0
  val min = finite.minOrNull()?.coerceAtMost(0.0) ?: 0.0
  if (finite.isEmpty()) { Text("No values to chart."); return }
  Row(Modifier.fillMaxWidth().height(170.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    Column(Modifier.fillMaxHeight(), verticalArrangement = Arrangement.SpaceBetween) {
      Text(number(max), style = MaterialTheme.typography.labelSmall)
      Text(number(min), style = MaterialTheme.typography.labelSmall)
    }
    Canvas(Modifier.weight(1f).fillMaxHeight().semantics { contentDescription = "$description. Observed minimum ${number(finite.min())}, maximum ${number(finite.max())}." }) {
      drawLine(gridColor, Offset(0f, 6f), Offset(size.width, 6f))
      drawLine(gridColor, Offset(0f, size.height - 6f), Offset(size.width, size.height - 6f))
      val path = Path()
      var connected = false
      values.forEachIndexed { index, value ->
        if (!value.isFinite()) { connected = false; return@forEachIndexed }
        val x = if (values.size == 1) size.width / 2 else index.toFloat() / values.lastIndex * size.width
        val y = (size.height - 6f - ((value - min) / (max - min) * (size.height - 12)).toFloat()).coerceIn(0f, size.height)
        if (!connected) path.moveTo(x, y) else path.lineTo(x, y)
        connected = true
        if (values.size == 1) drawCircle(primary, 4.dp.toPx(), Offset(x, y))
      }
      drawPath(path, primary, style = Stroke(3.dp.toPx()))
    }
  }
}

@Composable
internal fun Bars(groups: List<Pair<String, Double>>, moneyValues: Boolean = false) {
  val maximum = groups.maxOfOrNull { it.second }?.coerceAtLeast(1.0) ?: 1.0
  if (groups.isEmpty()) Text("No records match these filters.")
  groups.forEach { (label, value) -> ScoreRow(label, value / maximum, if (moneyValues) money(value) else number(value)) }
}

internal fun money(value: Double): String = NumberFormat.getCurrencyInstance(Locale.US).format(value)
internal fun number(value: Double): String = NumberFormat.getNumberInstance(Locale.US).apply { maximumFractionDigits = 1 }.format(value)
