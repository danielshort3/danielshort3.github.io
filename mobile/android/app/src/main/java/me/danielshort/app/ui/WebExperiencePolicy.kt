package me.danielshort.app.ui

import java.net.URI

internal enum class WebExperienceKind { GAME, DEMO, PROJECT }

internal data class WebExperience(
  val title: String,
  val canonicalPath: String,
  val kind: WebExperienceKind
) {
  companion object {
    val STARFALL = WebExperience("Project Starfall", "/games/project-starfall", WebExperienceKind.GAME)
    val SMART_SENTENCE = demo("Sentence Search", "/sentence-demo.html")
    val CHATBOT = demo("Travel Chat", "/chatbot-demo.html")
    val NONOGRAM = demo("Nonogram Solver", "/nonogram-demo.html")
    val PIZZA_TIPS = demo("Pizza Tips", "/pizza-tips-demo.html")
    val SHAPE = demo("Shape Classifier", "/shape-demo.html")
    val HANDWRITING = demo("Handwriting Rating", "/handwriting-rating-demo.html")
    val DIGIT = demo("Digit Generator", "/digit-generator-demo.html")
    val BABY_NAMES = demo("Baby Name Explorer", "/baby-names-demo.html")
    val COVID = demo("COVID-19 Outbreak Drivers", "/covid-outbreak-demo.html")
    val RETAIL = demo("Retail Sales & Loss", "/retail-loss-sales-demo.html")
    val EMPTY_PACKAGE = demo("Empty-Package Dashboard", "/target-empty-package-demo.html")

    fun game(id: String, title: String): WebExperience? =
      id.takeIf(::validCatalogId)?.let { WebExperience(title, "/games/$it", WebExperienceKind.GAME) }

    fun project(id: String, title: String): WebExperience? =
      id.takeIf(::validCatalogId)?.let { WebExperience(title, "/portfolio/$it", WebExperienceKind.PROJECT) }

    private fun demo(title: String, path: String) = WebExperience(title, path, WebExperienceKind.DEMO)
  }
}

private fun validCatalogId(id: String) = Regex("[A-Za-z0-9][A-Za-z0-9_-]{0,80}").matches(id)

internal val WEB_DEMO_EXPERIENCES = mapOf(
  "smartSentence" to WebExperience.SMART_SENTENCE,
  "chatbotLora" to WebExperience.CHATBOT,
  "nonogram" to WebExperience.NONOGRAM,
  "pizza" to WebExperience.PIZZA_TIPS,
  "shapeClassifier" to WebExperience.SHAPE,
  "handwritingRating" to WebExperience.HANDWRITING,
  "digitGenerator" to WebExperience.DIGIT,
  "babynames" to WebExperience.BABY_NAMES,
  "covidAnalysis" to WebExperience.COVID,
  "retailStore" to WebExperience.RETAIL,
  "targetEmptyPackage" to WebExperience.EMPTY_PACKAGE
)

/** Keep top-level WebView navigation on the selected first-party catalog route. */
internal fun trustedWebExperienceUrl(experience: WebExperience, candidate: String): String? {
  val uri = runCatching { URI(candidate) }.getOrNull() ?: return null
  val permittedPaths = if (experience.canonicalPath.endsWith(".html"))
    setOf(experience.canonicalPath, experience.canonicalPath.removeSuffix(".html"))
  else setOf(experience.canonicalPath, "${experience.canonicalPath}.html")
  if (uri.scheme != "https" || uri.host != "www.danielshort.me" || uri.port != -1 ||
    uri.rawUserInfo != null || uri.rawPath !in permittedPaths) return null
  return uri.toASCIIString()
}

internal fun canonicalWebExperienceUrl(experience: WebExperience): String =
  "https://www.danielshort.me${experience.canonicalPath}"

/** A case-study pager may open another published project without leaving the app. */
internal fun trustedLinkedProjectId(candidate: String, publishedIds: Set<String>): String? {
  val uri = runCatching { URI(candidate) }.getOrNull() ?: return null
  if (uri.scheme != "https" || uri.host != "www.danielshort.me" || uri.port != -1 ||
    uri.rawUserInfo != null) return null
  val path = uri.rawPath ?: return null
  val id = path.removePrefix("/portfolio/").removeSuffix(".html")
  if (!validCatalogId(id) || id !in publishedIds ||
    path !in setOf("/portfolio/$id", "/portfolio/$id.html")) return null
  return id
}

/** Demo wrappers load the interactive experience in one known same-origin iframe. */
internal fun isPrimaryDemoFrameUrl(experience: WebExperience, candidate: String): Boolean {
  if (experience.kind != WebExperienceKind.DEMO) return false
  val uri = runCatching { URI(candidate) }.getOrNull() ?: return false
  return uri.scheme == "https" && uri.host == "www.danielshort.me" && uri.port == -1 &&
    uri.rawUserInfo == null && uri.rawPath == "/demos${experience.canonicalPath}"
}
