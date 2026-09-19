package me.danielshort.app.nativefeatures

enum class DiffKind { Unchanged, Added, Removed }

data class DiffSpan(val text: String, val kind: DiffKind)

data class TextDiff(val spans: List<DiffSpan>, val simplified: Boolean = false) {
  val hasChanges: Boolean get() = spans.any { it.kind != DiffKind.Unchanged }
}

const val EXAMPLE_BEFORE = "Small improvements make everyday work easier."
const val EXAMPLE_AFTER = "Thoughtful improvements make everyday work simpler."
const val MAX_COMPARE_LENGTH = 50_000

/** Exact whitespace is preserved so either input can be reconstructed from the diff. */
fun compareText(before: String, after: String): TextDiff {
  if (before == after) return TextDiff(listOf(DiffSpan(before, DiffKind.Unchanged)))
  val tokenPattern = Regex("\\s+|[\\p{L}\\p{N}_]+|[^\\s\\p{L}\\p{N}_]")
  val left = tokenPattern.findAll(before).map { it.value }.toList()
  val right = tokenPattern.findAll(after).map { it.value }.toList()
  var prefix = 0
  while (prefix < minOf(left.size, right.size) && left[prefix] == right[prefix]) prefix++
  var suffix = 0
  while (suffix < minOf(left.size, right.size) - prefix &&
    left[left.lastIndex - suffix] == right[right.lastIndex - suffix]) suffix++
  val leftEnd = left.size - suffix
  val rightEnd = right.size - suffix
  val n = leftEnd - prefix
  val m = rightEnd - prefix
  val spans = mutableListOf<DiffSpan>()
  fun append(text: String, kind: DiffKind) {
    if (text.isEmpty()) return
    val previous = spans.lastOrNull()
    if (previous?.kind == kind) spans[spans.lastIndex] = previous.copy(text = previous.text + text)
    else spans.add(DiffSpan(text, kind))
  }
  append(left.take(prefix).joinToString(""), DiffKind.Unchanged)
  // Bound both memory and work. Large edits remain an exact before/after comparison,
  // while common leading and trailing text stays unmarked.
  val simplified = n.toLong() * m > 900_000 || maxOf(n, m) > 1_600
  if (simplified || n == 0 || m == 0) {
    append(left.subList(prefix, leftEnd).joinToString(""), DiffKind.Removed)
    append(right.subList(prefix, rightEnd).joinToString(""), DiffKind.Added)
  } else {
    val width = m + 1
    val lengths = IntArray((n + 1) * width)
    for (i in n - 1 downTo 0) for (j in m - 1 downTo 0) {
      lengths[i * width + j] = if (left[prefix + i] == right[prefix + j]) {
        1 + lengths[(i + 1) * width + j + 1]
      } else maxOf(lengths[(i + 1) * width + j], lengths[i * width + j + 1])
    }
    var i = 0
    var j = 0
    while (i < n || j < m) {
      when {
        i < n && j < m && left[prefix + i] == right[prefix + j] -> {
          append(left[prefix + i], DiffKind.Unchanged)
          i++
          j++
        }
        i < n && (j == m || lengths[(i + 1) * width + j] >= lengths[i * width + j + 1]) -> {
          append(left[prefix + i], DiffKind.Removed)
          i++
        }
        else -> {
          append(right[prefix + j], DiffKind.Added)
          j++
        }
      }
    }
  }
  append(left.takeLast(suffix).joinToString(""), DiffKind.Unchanged)
  return TextDiff(spans, simplified)
}

fun comparisonInputs(before: String, after: String): Pair<String, String> =
  if (before.isEmpty() && after.isEmpty()) EXAMPLE_BEFORE to EXAMPLE_AFTER else before to after
