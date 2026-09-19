package me.danielshort.app.nativefeatures.demos

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class NativeDemoContractsTest {
  @Test fun generatorUsesProductionSchemaAndGridSize() {
    val body = digitRequest(4, 6, 42, 19, 5f)
    assertEquals("cluster", body.getString("mode"))
    assertEquals(4, body.getInt("cluster_digit"))
    assertEquals(6, body.getInt("rows"))
    assertEquals(6, body.getInt("cols"))
    assertEquals(42L, body.getLong("seed"))
    assertEquals(5.0, body.getDouble("value"), .0)
    val random = digitRequest(null, 6, 42, 0, 5f)
    assertEquals("random", random.getString("mode"))
    assertTrue(random.isNull("cluster_digit"))
  }

  @Test fun rejectsOversizedAndOutOfRangeGeneration() {
    listOf<() -> Unit>(
      { digitRequest(10, 6, 42, 0, 5f) }, { digitRequest(4, 9, 42, 0, 5f) },
      { digitRequest(4, 6, -1, 0, 5f) }, { digitRequest(4, 6, 42, 20, 5f) },
      { digitRequest(4, 6, 42, 0, -1f) }, { digitRequest(4, 6, 42, 0, Float.NaN) }
    ).forEach { call -> assertThrows(IllegalArgumentException::class.java) { call() } }
  }

  @Test fun parsesAllDigitConfidencesWithoutInventingMissingScores() {
    val scores = JSONObject((0..9).associate { it.toString() to if (it == 4) .91 else .01 })
    val parsed = parsePredictions(JSONObject().put("digit_confidences", scores), false)
    assertEquals(10, parsed.size)
    assertEquals("4", parsed.first().label)
    assertEquals(.91, parsed.first().confidence, .0001)
    scores.remove("3")
    assertThrows(IllegalArgumentException::class.java) { parsePredictions(JSONObject().put("digit_confidences", scores), false) }
  }

  @Test fun normalizesPercentageScoresAndShapeLabels() {
    val parsed = parsePredictions(JSONObject("""{"shape_confidences":{"circle":60,"triangle":20,"square":10,"hexagon":5,"octagon":5}}"""), true)
    assertEquals("circle", parsed.first().label)
    assertEquals(.6, parsed.first().confidence, .0001)
    assertThrows(IllegalArgumentException::class.java) { parsePredictions(JSONObject("""{"class":"unknown","confidence":0.9}"""), true) }
  }

  @Test fun nameFiltersRetainRepeatedRatings() {
    val rows = JSONArray("""[{"name":"Anna","rating":8},{"name":"Anna","rating":9},{"name":"Annie","rating":5}]""")
    val result = filterNames(rows, "ann", 7.0, true)
    assertEquals(2, result.size)
    assertEquals(9.0, result.first().getDouble("rating"), .0)
  }

  @Test fun packageAggregationUsesFilteredRecordsAndExactValues() {
    val rows = JSONArray("""[{"datetime":"2023-02-01","department":"1","value":4.5},{"datetime":"2023-02-02","department":"1","value":2.5},{"datetime":"2023-03-01","department":"2","value":3}]""").objects()
    assertEquals(listOf("1" to 7.0, "2" to 3.0), aggregateRows(rows, "department", "value"))
    assertEquals(listOf("2023-02" to 2.0, "2023-03" to 1.0), aggregateRows(rows, "datetime", null, true))
  }

  @Test fun pizzaFiltersCombineDimensions() {
    val rows = JSONArray("""[["2017-07-05","Frisco","Residential",42.83,10,24],["2017-08-05","Frisco","Apartment",20,1,10],["2017-07-06","Lewisville","Residential",10,5,20]]""").arrayRows()
    assertEquals(1, filterPizza(rows, "Frisco", "Residential", "2017-07").size)
    assertEquals(3, filterPizza(rows, "All", "All", "All").size)
    assertEquals(0, filterPizza(rows, "Missing", "All", "All").size)
  }

  @Test fun ufoAggregationCountsReportsInsteadOfGroupedRows() {
    val rows = JSONArray("""[[2013,1,20,"co","light",5],[2013,1,21,"co","light",7],[2014,1,20,"co","circle",3]]""").arrayRows()
    val matched = filterUfo(rows, "2013", "CO", "light")
    assertEquals(2, matched.size)
    assertEquals(12.0, matched.sumOf { it.getDouble(5) }, .0)
  }

  @Test fun onlyProjectsWithPublishedDemosAreNativeEntryPoints() {
    assertFalse("deliveryTip" in NATIVE_DEMO_IDS)
    assertTrue("pizza" in NATIVE_DEMO_IDS)
    assertFalse("minesweeper" in NATIVE_DEMO_IDS)
    assertTrue("nonogram" in NATIVE_DEMO_IDS)
    assertEquals(13, NATIVE_DEMO_IDS.size)
  }

  @Test fun nonogramAccuracyUsesPredictionAndAnswerRatherThanContradictoryServiceFlag() {
    assertFalse(nonogramPredictionMatches(JSONObject("""{"predicted":1,"actual":0,"correct":true}""")))
    assertTrue(nonogramPredictionMatches(JSONObject("""{"predicted":1,"actual":1,"correct":false}""")))
    assertFalse(nonogramPredictionMatches(JSONObject("""{"correct":true}""")))
  }

  @Test fun nativeTipEstimateMatchesPublishedJavascriptModel() {
    val estimate = PizzaTipsModel.estimate(41.77, 18, "Frisco", "Residential", 80)
    assertEquals(6.459512393719423, estimate.tip, 1e-10)
    assertEquals(3.4290288700070573, estimate.tipLow, 1e-10)
    assertEquals(11.563549885364605, estimate.tipHigh, 1e-10)
    assertEquals(.18890317873644086, estimate.rate, 1e-10)
    assertFalse(estimate.outsideTrainingRange)
    val plano = PizzaTipsModel.estimate(85.5, 20, "Plano", "Apartment", 95)
    assertEquals(9.040155299903615, plano.tip, 1e-10)
    assertEquals(3.5248409318268528, plano.tipLow, 1e-10)
    assertEquals(21.278068989595155, plano.tipHigh, 1e-10)
    assertEquals(.13441405047587746, plano.rate, 1e-10)
    assertEquals(0.0, plano.rateLow, 0.0)
  }

  @Test fun tipModelValidatesInputAndFlagsExtrapolation() {
    assertThrows(IllegalArgumentException::class.java) { PizzaTipsModel.estimate(Double.NaN, 18, "Frisco", "Residential", 80) }
    assertThrows(IllegalArgumentException::class.java) { PizzaTipsModel.estimate(30.0, 24, "Frisco", "Residential", 80) }
    assertThrows(IllegalArgumentException::class.java) { PizzaTipsModel.estimate(30.0, 18, "Unknown", "Residential", 80) }
    assertTrue(PizzaTipsModel.estimate(300.0, 18, "Frisco", "Residential", 80).outsideTrainingRange)
  }

  @Test fun reportSeriesRetainsMissingMonthsAndHoursAsZero() {
    val months = completeReportPeriods(listOf("1" to 7.0, "3" to 5.0), 1)
    assertEquals(12, months.size)
    assertEquals("2" to 0.0, months[1])
    assertEquals(12.0, months.sumOf { it.second }, 0.0)
    assertEquals(24, completeReportPeriods(listOf("20" to 5.0), 2).size)
  }
}
