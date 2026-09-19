package me.danielshort.app.nativefeatures.tools

import java.net.URI
import java.net.URLDecoder
import java.net.URLEncoder
import java.text.Normalizer
import java.util.Locale

const val MAX_TOOL_TEXT = 100_000
private val hardSpaces = setOf('\u00a0', '\u202f', '\u2007', '\u2009')
private val wordPattern = Regex("[\\p{L}\\p{N}]+(?:['’\\-][\\p{L}\\p{N}]+)*")
private val pronounPattern = Regex("[\\p{L}\\p{N}]+(?:['’][\\p{L}\\p{N}]+)*")

data class CleanedText(val text: String, val replacedSpaces: Int, val removedCharacters: Int)

fun cleanSpaces(text: String, replaceSpaces: Boolean = true, removeNonAscii: Boolean = false): CleanedText {
  require(text.length <= MAX_TOOL_TEXT)
  var replaced = 0
  var removed = 0
  val cleaned = buildString {
    text.codePoints().forEach { point ->
      when {
        replaceSpaces && point.toChar() in hardSpaces && point <= Char.MAX_VALUE.code -> { append(' '); replaced++ }
        removeNonAscii && point > 127 -> removed++
        else -> appendCodePoint(point)
      }
    }
  }
  return CleanedText(cleaned, replaced, removed)
}

private val stopWords = ("a about above after again against all am an and any are aren't as at be because been before being below between both but by " +
  "can cannot can't could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having " +
  "he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's me more most " +
  "mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such " +
  "than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very " +
  "was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't " +
  "you you'd you'll you're you've your yours yourself yourselves").split(' ').toSet()

data class WordCount(val word: String, val count: Int)
data class FrequencyResult(val totalWords: Int, val countedWords: Int, val words: List<WordCount>)

fun wordFrequency(text: String, skipCommon: Boolean = true, foldAccents: Boolean = false, phraseLength: Int = 1): FrequencyResult {
  require(text.length <= MAX_TOOL_TEXT)
  require(phraseLength in 1..3)
  val tokens = wordPattern.findAll(text).map { it.value.replace('’', '\'').lowercase(Locale.ROOT) }.toList()
  val normalized = tokens.map { token ->
    if (foldAccents) Normalizer.normalize(token, Normalizer.Form.NFD).replace(Regex("\\p{M}+"), "") else token
  }
  val counts = linkedMapOf<String, Int>()
  normalized.windowed(phraseLength).forEach { words ->
    if (words.any { word -> word.none(Char::isLetter) || word.length < 2 || (skipCommon && word in stopWords) }) return@forEach
    val phrase = words.joinToString(" ")
    counts[phrase] = (counts[phrase] ?: 0) + 1
  }
  return FrequencyResult(tokens.size, counts.values.sum(), counts.map { WordCount(it.key, it.value) }
    .sortedWith(compareByDescending<WordCount> { it.count }.thenBy { it.word }))
}

private val firstPerson = "i me my mine myself we us our ours ourselves i'm i'd i'll i've we're we'd we'll we've".split(' ').toSet()
private val secondPerson = "you your yours yourself yourselves you're you'd you'll you've y'all".split(' ').toSet()
private val thirdPerson = "he him his himself he's he'd he'll she her hers herself she's she'd she'll they them their theirs themself themselves they're they'd they'll they've".split(' ').toSet()
private val neutralPerson = setOf("it", "its", "itself", "it's")
data class PronounMatch(val start: Int, val end: Int, val person: Int, val word: String)

fun pointOfView(text: String, includeNeutral: Boolean = false): List<PronounMatch> {
  require(text.length <= MAX_TOOL_TEXT)
  return pronounPattern.findAll(text).mapNotNull { match ->
    val token = match.value.replace('’', '\'').lowercase(Locale.ROOT)
    val person = when {
      token in firstPerson -> 1
      token in secondPerson -> 2
      token in thirdPerson || (includeNeutral && token in neutralPerson) -> 3
      else -> return@mapNotNull null
    }
    PronounMatch(match.range.first, match.range.last + 1, person, match.value)
  }.toList()
}

data class OxfordCandidate(val text: String, val present: Boolean)

/** Same conservative list-candidate heuristic as the web tool, not a grammar verdict. */
fun oxfordCandidates(text: String): List<OxfordCandidate> {
  require(text.length <= MAX_TOOL_TEXT)
  val clauses = Regex("[^.!?\\n;:]+")
  val conjunction = Regex("\\b(and|or|nor)\\b", RegexOption.IGNORE_CASE)
  return clauses.findAll(text).mapNotNull { clause ->
    val last = conjunction.findAll(clause.value).lastOrNull() ?: return@mapNotNull null
    val before = clause.value.substring(0, last.range.first).trimEnd()
    val after = clause.value.substring(last.range.last + 1).trim()
    if (',' !in before || after.isEmpty()) return@mapNotNull null
    OxfordCandidate(clause.value.trim(), before.endsWith(','))
  }.toList()
}

fun buildUtmLinks(
  landingPages: String, sources: String, mediums: String, campaigns: String,
  content: String = "", term: String = "", cartesian: Boolean = true,
  normalize: Boolean = true, overrideExisting: Boolean = true
): List<String> {
  require(listOf(landingPages, sources, mediums, campaigns, content, term).all { it.length <= 10_000 }) {
    "Use at most 10,000 characters per field."
  }
  fun lines(value: String, optional: Boolean = false): List<String> {
    if (value.isBlank()) return if (optional) listOf("") else emptyList()
    val values = value.trimEnd('\r', '\n').lines().map(String::trim)
    // Row mode must retain duplicates and optional empty cells to preserve alignment.
    if (!cartesian) {
      require(optional || values.none(String::isEmpty)) { "Remove blank rows from the required fields." }
      return values
    }
    return values.filter(String::isNotEmpty).distinct()
  }
  val pages = lines(landingPages)
  val inputs = listOf(lines(sources), lines(mediums), lines(campaigns), lines(content, true), lines(term, true))
  require(pages.isNotEmpty() && inputs.take(3).all { it.isNotEmpty() }) { "Add a landing page, source, medium, and campaign." }
  require((listOf(pages) + inputs).all { it.size <= 200 }) { "Use at most 200 values per field." }
  val rows = mutableListOf<List<String>>()
  val all = listOf(pages) + inputs
  if (cartesian) {
    require(all.fold(1L) { total, values -> total * values.size } <= 1000) { "This produces over 1,000 links. Reduce the lists or use rows." }
    fun add(prefix: List<String>, index: Int) {
      if (index == all.size) rows.add(prefix) else all[index].forEach { add(prefix + it, index + 1) }
    }
    add(emptyList(), 0)
  } else {
    val size = all.maxOf { it.size }
    require(all.all { it.size == 1 || it.size == size }) { "Use one value per field, or the same number of rows in each list." }
    repeat(size) { row -> rows.add(all.map { if (it.size == 1) it.first() else it[row] }) }
  }
  val keys = listOf("utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term")
  val results = rows.map { row ->
    val uri = try { URI(row.first()) } catch (_: Exception) { throw IllegalArgumentException("Use valid http:// or https:// landing pages.") }
    require(uri.scheme?.lowercase(Locale.ROOT) in setOf("http", "https") && !uri.host.isNullOrBlank() && uri.rawUserInfo == null) { "Use valid http:// or https:// landing pages without login details." }
    val entries = uri.rawQuery?.split('&')?.filter(String::isNotEmpty)?.toMutableList() ?: mutableListOf()
    row.drop(1).forEachIndexed { index, raw ->
      if (raw.isEmpty()) return@forEachIndexed
      val value = if (normalize) raw.lowercase(Locale.ROOT).replace(Regex("\\s+"), "_") else raw
      val key = keys[index]
      fun matches(entry: String): Boolean = runCatching { URLDecoder.decode(entry.substringBefore('='), "UTF-8") == key }.getOrDefault(false)
      if (overrideExisting) entries.removeAll(::matches)
      if (entries.none(::matches)) entries.add("$key=${URLEncoder.encode(value, "UTF-8")}")
    }
    val path = uri.rawPath.ifEmpty { "/" }
    "${uri.scheme}://${uri.rawAuthority}$path" + (if (entries.isEmpty()) "" else "?${entries.joinToString("&")}") + (uri.rawFragment?.let { "#$it" } ?: "")
  }.distinct()
  require(results.sumOf { it.length } <= 250_000) { "These links are too large for one batch. Use fewer or shorter landing pages." }
  return results
}

fun wifiQrPayload(ssid: String, password: String, open: Boolean = false, hidden: Boolean = false): String {
  require(ssid.isNotBlank()) { "Enter a network name." }
  require(open || password.isNotEmpty()) { "Enter a password or choose an open network." }
  fun escape(value: String) = value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace(":", "\\:").replace("\"", "\\\"")
  return "WIFI:T:${if (open) "nopass" else "WPA"};S:${escape(ssid)};" +
    (if (open || password.isEmpty()) "" else "P:${escape(password)};") + (if (hidden) "H:true;" else "") + ";"
}

fun scaledImageSize(width: Int, height: Int, maxSide: Int): Pair<Int, Int> {
  require(width > 0 && height > 0 && maxSide in 1..4096)
  val scale = minOf(1.0, maxSide.toDouble() / maxOf(width, height))
  return maxOf(1, (width * scale).toInt()) to maxOf(1, (height * scale).toInt())
}
