package me.danielshort.app.nativefeatures

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class TextDiffTest {
  @Test fun exampleOnlyAppliesWhenBothInputsAreEmpty() {
    assertEquals(EXAMPLE_BEFORE to EXAMPLE_AFTER, comparisonInputs("", ""))
    assertEquals("typed" to "", comparisonInputs("typed", ""))
    assertEquals("" to "typed", comparisonInputs("", "typed"))
    assertEquals(" " to "", comparisonInputs(" ", ""))
  }

  @Test fun unchangedTextHasNoChanges() {
    val diff = compareText("same\ntext", "same\ntext")
    assertFalse(diff.hasChanges)
    assertEquals(listOf(DiffSpan("same\ntext", DiffKind.Unchanged)), diff.spans)
  }

  @Test fun diffPreservesBothDocumentsExactly() {
    listOf(
      "Hello, world!" to "Hello brave world!",
      "" to "added text",
      "removed text" to "",
      "space  and\nlines" to "space and\r\nlines",
      "Café 🧡 日本語" to "Café 💙 English",
      "one two one three" to "one three two one",
      "fun x() = 1;" to "fun x() = 2;"
    ).forEach { (before, after) ->
      val diff = compareText(before, after)
      assertEquals(before, diff.spans.filter { it.kind != DiffKind.Added }.joinToString("") { it.text })
      assertEquals(after, diff.spans.filter { it.kind != DiffKind.Removed }.joinToString("") { it.text })
      assertTrue(diff.hasChanges)
    }
  }

  @Test fun largeEditsUseBoundedFallbackWithoutLosingText() {
    val before = "same " + "a ".repeat(4_000) + " end"
    val after = "same " + "b ".repeat(4_000) + " end"
    val diff = compareText(before, after)
    assertTrue(diff.simplified)
    assertEquals(before, diff.spans.filter { it.kind != DiffKind.Added }.joinToString("") { it.text })
    assertEquals(after, diff.spans.filter { it.kind != DiffKind.Removed }.joinToString("") { it.text })
    assertEquals(DiffKind.Unchanged, diff.spans.first().kind)
    assertEquals(DiffKind.Unchanged, diff.spans.last().kind)
  }

  @Test fun randomizedRepeatedTokensRetainExactContentAndCoalesceChanges() {
    val random = Random(2026)
    val tokens = listOf("alpha", "beta", "gamma", " ", "  ", "\n", "\r\n", ",", ".", "💙", "é", "日本語")
    repeat(400) {
      fun document() = List(random.nextInt(50)) { tokens.random(random) }.joinToString("")
      val before = document()
      val after = document()
      val diff = compareText(before, after)
      assertEquals(before, diff.spans.filter { it.kind != DiffKind.Added }.joinToString("") { it.text })
      assertEquals(after, diff.spans.filter { it.kind != DiffKind.Removed }.joinToString("") { it.text })
      assertEquals(before != after, diff.hasChanges)
      assertTrue(diff.spans.zipWithNext().all { (left, right) -> left.kind != right.kind })
    }
  }

  @Test fun maximumSizedDocumentsRemainExactForSparseAndBroadEdits() {
    val before = "alpha ".repeat(8_332).padEnd(MAX_COMPARE_LENGTH, '!')
    listOf(before.replaceFirst("alpha", "ALPHA"), "beta ".repeat(10_000)).forEach { after ->
      val diff = compareText(before, after)
      assertEquals(before, diff.spans.filter { it.kind != DiffKind.Added }.joinToString("") { it.text })
      assertEquals(after, diff.spans.filter { it.kind != DiffKind.Removed }.joinToString("") { it.text })
    }
  }
}
