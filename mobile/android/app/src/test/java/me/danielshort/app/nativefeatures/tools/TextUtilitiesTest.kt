package me.danielshort.app.nativefeatures.tools

import com.google.zxing.BinaryBitmap
import com.google.zxing.RGBLuminanceSource
import com.google.zxing.common.HybridBinarizer
import com.google.zxing.qrcode.QRCodeReader
import org.junit.Assert.*
import org.junit.Test

class TextUtilitiesTest {
  @Test fun cleanerPreservesAccentsEmojiAndParagraphsByDefault() {
    val result = cleanSpaces("Café\u00a0and\u202f👪\n\u2007music\u2009")
    assertEquals("Café and 👪\n music ", result.text)
    assertEquals(4, result.replacedSpaces)
    assertEquals(0, result.removedCharacters)
  }

  @Test fun cleanerNonAsciiRemovalCountsUnicodeCharactersNotSurrogates() {
    val result = cleanSpaces("A\u00a0é👪B", removeNonAscii = true)
    assertEquals("A B", result.text)
    assertEquals(1, result.replacedSpaces)
    assertEquals(2, result.removedCharacters)
  }

  @Test fun cleanerCanInspectWithoutReplacingAnything() {
    assertEquals("a\u00a0b", cleanSpaces("a\u00a0b", replaceSpaces = false).text)
  }

  @Test fun wordFrequencyMergesCaseAndCurlyApostrophes() {
    val result = wordFrequency("Café café CAFE don’t don't 2025", skipCommon = false, foldAccents = true)
    assertEquals(6, result.totalWords)
    assertEquals(listOf(WordCount("cafe", 3), WordCount("don't", 2)), result.words)
  }

  @Test fun phrasesRemainContiguousWhenCommonWordsAreExcluded() {
    val result = wordFrequency("customer success and customer success", phraseLength = 2)
    assertEquals(WordCount("customer success", 2), result.words.first())
    assertFalse(result.words.any { it.word == "success customer" })
  }

  @Test fun pronounMatchesIncludeContractionsAndRespectBoundaries() {
    val text = "I’m ready. You will tell them. This item is useful."
    val matches = pointOfView(text)
    assertEquals(listOf(1, 2, 3), matches.map { it.person })
    assertEquals(listOf("I’m", "You", "them"), matches.map { text.substring(it.start, it.end) })
  }

  @Test fun neutralPronounsAreOptIn() {
    assertTrue(pointOfView("It is in its place.").isEmpty())
    assertEquals(2, pointOfView("It is in its place.", true).size)
  }

  @Test fun oxfordMatchesWebsiteExampleWithoutClaimingGrammarCorrection() {
    val candidates = oxfordCandidates("We packed apples, bananas and cherries. Pricing, positioning, and messaging. Plain text.")
    assertEquals(listOf(false, true), candidates.map { it.present })
  }

  @Test fun utmBatchesPreserveExistingQueryEncodingAndFragments() {
    val result = buildUtmLinks("https://example.com/page?x=a%2Bb&utm_source=old#details", "Email\nSocial", "Paid Campaign", "Launch")
    assertEquals(2, result.size)
    assertEquals("https://example.com/page?x=a%2Bb&utm_source=email&utm_medium=paid_campaign&utm_campaign=launch#details", result.first())
    assertFalse(result.first().contains("old"))
  }

  @Test fun utmCanPreserveExistingParameters() {
    val result = buildUtmLinks("https://example.com/?utm_source=existing", "new", "email", "Launch", overrideExisting = false).single()
    assertTrue(result.contains("utm_source=existing"))
    assertFalse(result.contains("utm_source=new"))
  }

  @Test fun utmRowsRepeatSingleFieldsAndEncodeUnicode() {
    val results = buildUtmLinks("https://example.com", "email\nsocial", "cpc", "été\nwinter", cartesian = false, normalize = false)
    assertEquals(2, results.size)
    assertTrue(results.first().contains("utm_campaign=%C3%A9t%C3%A9"))
    assertTrue(results.last().contains("utm_source=social"))
  }

  @Test fun utmRejectsMismatchedRowsAndOversizedProducts() {
    assertThrows(IllegalArgumentException::class.java) { buildUtmLinks("https://example.com", "a\nb", "a\nb\nc", "c", cartesian = false) }
    val values = (1..11).joinToString("\n")
    assertThrows(IllegalArgumentException::class.java) { buildUtmLinks("https://example.com", values, values, values) }
  }

  @Test fun utmRowsKeepDuplicateValuesAndOptionalEmptyCellsAligned() {
    val results = buildUtmLinks("https://example.com", "email\nemail\nsocial", "newsletter", "first\nsecond\nthird",
      content = "hero\n\nfooter", cartesian = false)
    assertEquals(3, results.size)
    assertTrue(results[0].contains("utm_source=email") && results[0].contains("utm_campaign=first") && results[0].contains("utm_content=hero"))
    assertTrue(results[1].contains("utm_source=email") && results[1].contains("utm_campaign=second"))
    assertFalse(results[1].contains("utm_content="))
    assertTrue(results[2].contains("utm_source=social") && results[2].contains("utm_campaign=third") && results[2].contains("utm_content=footer"))
  }

  @Test fun utmRowsRejectEmptyRequiredCellsInsteadOfShiftingOtherColumns() {
    assertThrows(IllegalArgumentException::class.java) {
      buildUtmLinks("https://example.com", "email\n\nsocial", "newsletter", "first\nsecond\nthird", cartesian = false)
    }
  }

  @Test fun utmRejectsExecutableAndCredentialUrls() {
    for (url in listOf("javascript:alert(1)", "file:///secret", "https://user:secret@example.com/", "not-a-url")) {
      assertThrows(IllegalArgumentException::class.java) { buildUtmLinks(url, "email", "email", "launch") }
    }
  }

  @Test fun wifiCodesEscapeReservedCharactersAndOmitOpenNetworkPasswords() {
    assertEquals("WIFI:T:WPA;S:House\\;One;P:pass\\:word;H:true;;", wifiQrPayload("House;One", "pass:word", hidden = true))
    assertEquals("WIFI:T:nopass;S:Guest;;", wifiQrPayload("Guest", "unused", open = true))
    assertThrows(IllegalArgumentException::class.java) { wifiQrPayload("Private", "") }
  }

  @Test fun imageSizingKeepsAspectRatioAndNeverUpscales() {
    assertEquals(1600 to 900, scaledImageSize(3840, 2160, 1600))
    assertEquals(300 to 600, scaledImageSize(300, 600, 1600))
    assertEquals(1 to 2048, scaledImageSize(1, 10000, 2048))
  }

  @Test fun qrCodeRoundTripsUnicodeAndWifiPayloads() {
    for (payload in listOf("https://www.danielshort.me/?q=café", wifiQrPayload("Home;WiFi", "correct:horse"))) {
      val matrix = encodeQrMatrix(payload)
      val pixels = IntArray(matrix.width * matrix.height) { index -> if (matrix[index % matrix.width, index / matrix.width]) 0xff000000.toInt() else 0xffffffff.toInt() }
      val bitmap = BinaryBitmap(HybridBinarizer(RGBLuminanceSource(matrix.width, matrix.height, pixels)))
      assertEquals(payload, QRCodeReader().decode(bitmap).text)
    }
  }

  @Test fun algorithmsRejectUnboundedInputs() {
    assertThrows(IllegalArgumentException::class.java) { wordFrequency("a".repeat(MAX_TOOL_TEXT + 1)) }
    assertThrows(IllegalArgumentException::class.java) { encodeQrMatrix("a".repeat(1501)) }
    assertThrows(IllegalArgumentException::class.java) { buildUtmLinks("https://example.com", "a".repeat(10_001), "email", "campaign") }
  }
}
