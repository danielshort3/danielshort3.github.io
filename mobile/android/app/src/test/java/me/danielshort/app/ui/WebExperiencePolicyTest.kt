package me.danielshort.app.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WebExperiencePolicyTest {
  @Test fun acceptsOnlyThePublishedFeatureRoutes() {
    assertEquals("https://www.danielshort.me/games/project-starfall",
      trustedWebExperienceUrl(WebExperience.STARFALL, "https://www.danielshort.me/games/project-starfall"))
    assertEquals("https://www.danielshort.me/sentence-demo.html",
      trustedWebExperienceUrl(WebExperience.SMART_SENTENCE, "https://www.danielshort.me/sentence-demo.html"))
    assertEquals("https://www.danielshort.me/sentence-demo",
      trustedWebExperienceUrl(WebExperience.SMART_SENTENCE, "https://www.danielshort.me/sentence-demo"))
    mapOf(
      "chatbotLora" to "/chatbot-demo.html",
      "nonogram" to "/nonogram-demo.html",
      "pizza" to "/pizza-tips-demo.html",
      "shapeClassifier" to "/shape-demo.html",
      "handwritingRating" to "/handwriting-rating-demo.html",
      "digitGenerator" to "/digit-generator-demo.html",
      "babynames" to "/baby-names-demo.html",
      "covidAnalysis" to "/covid-outbreak-demo.html",
      "retailStore" to "/retail-loss-sales-demo.html",
      "targetEmptyPackage" to "/target-empty-package-demo.html"
    ).forEach { (projectId, path) ->
      val experience = WEB_DEMO_EXPERIENCES.getValue(projectId)
      assertEquals("https://www.danielshort.me$path",
        trustedWebExperienceUrl(experience, "https://www.danielshort.me$path"))
      assertEquals("https://www.danielshort.me${path.removeSuffix(".html")}",
        trustedWebExperienceUrl(experience, "https://www.danielshort.me${path.removeSuffix(".html")}"))
    }
  }

  @Test fun catalogPagesStayOnTheirOwnPublishedRoute() {
    val game = requireNotNull(WebExperience.game("stellar-dogfight", "Stellar Dogfight"))
    val project = requireNotNull(WebExperience.project("sheetMusicUpscale", "Sheet Music"))
    assertEquals("https://www.danielshort.me/games/stellar-dogfight",
      trustedWebExperienceUrl(game, canonicalWebExperienceUrl(game)))
    assertEquals("https://www.danielshort.me/portfolio/sheetMusicUpscale",
      trustedWebExperienceUrl(project, canonicalWebExperienceUrl(project)))
    assertNull(trustedWebExperienceUrl(game, canonicalWebExperienceUrl(project)))
    assertNull(trustedWebExperienceUrl(project, "https://www.danielshort.me/portfolio/other"))
    assertNull(WebExperience.game("../../tools/admin", "Bad route"))
    assertNull(WebExperience.project("other/path", "Bad route"))
    assertFalse(isPrimaryDemoFrameUrl(project,
      "https://www.danielshort.me/demos/shape-demo.html"))
  }

  @Test fun nextProjectLinksStayWithinPublishedCaseStudies() {
    val published = setOf("smartSentence", "chatbotLora")
    assertEquals("chatbotLora", trustedLinkedProjectId(
      "https://www.danielshort.me/portfolio/chatbotLora", published))
    assertEquals("chatbotLora", trustedLinkedProjectId(
      "https://www.danielshort.me/portfolio/chatbotLora.html#main", published))
    assertNull(trustedLinkedProjectId("https://www.danielshort.me/portfolio/private", published))
    assertNull(trustedLinkedProjectId("https://evil.example/portfolio/chatbotLora", published))
    assertNull(trustedLinkedProjectId("https://www.danielshort.me/portfolio/../admin", published))
  }

  @Test fun rejectsFeedUrlsOutsideTheExpectedFirstPartyPath() {
    listOf(
      "http://www.danielshort.me/sentence-demo.html",
      "https://danielshort.me/sentence-demo.html",
      "https://www.danielshort.me.evil.example/sentence-demo.html",
      "https://www.danielshort.me:444/sentence-demo.html",
      "https://attacker@www.danielshort.me/sentence-demo.html",
      "https://www.danielshort.me/other-page",
      "javascript:alert(1)"
    ).forEach { assertNull(it, trustedWebExperienceUrl(WebExperience.SMART_SENTENCE, it)) }
    assertNull(trustedWebExperienceUrl(WebExperience.STARFALL, "https://www.danielshort.me/sentence-demo.html"))
    assertNull(trustedWebExperienceUrl(WebExperience.CHATBOT, "https://www.danielshort.me/nonogram-demo.html"))
  }

  @Test fun recognizesOnlyTheDemoInsideItsExpectedWrapper() {
    assertTrue(isPrimaryDemoFrameUrl(WebExperience.NONOGRAM,
      "https://www.danielshort.me/demos/nonogram-demo.html"))
    assertFalse(isPrimaryDemoFrameUrl(WebExperience.NONOGRAM,
      "https://www.danielshort.me/demos/chatbot-demo.html"))
    assertFalse(isPrimaryDemoFrameUrl(WebExperience.NONOGRAM,
      "https://evil.example/demos/nonogram-demo.html"))
    assertFalse(isPrimaryDemoFrameUrl(WebExperience.STARFALL,
      "https://www.danielshort.me/demos/nonogram-demo.html"))
  }
}
