package me.danielshort.app.nativefeatures.tools

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Canvas
import android.graphics.Paint
import android.net.Uri
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.junit4.StateRestorationTester
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTextReplacement
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.exifinterface.media.ExifInterface
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test

class NativeUtilityFlowTest {
  @get:Rule val compose = createComposeRule()

  private fun open(id: String) { compose.setContent { MaterialTheme { NativeToolsScreen(id, {}) } } }

  private fun fill(label: String, value: String) {
    compose.onNode(hasSetTextAction() and hasText(label)).performScrollTo().performTextInput(value)
  }

  private fun waitFor(text: String) {
    compose.waitUntil(5_000) { compose.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty() }
  }

  @Test fun cleanerRunsNativelyAndInvalidatesResultWhenEdited() {
    open("nbsp-cleaner")
    compose.onNodeWithText("Clean text").assertIsNotEnabled()
    fill("Text", "one\u00a0two")
    compose.onNodeWithText("Clean text").performScrollTo().performClick()
    waitFor("1 spaces replaced")
    compose.onNodeWithText("one two", substring = false).assertExists()
    compose.onNode(hasSetTextAction() and hasText("Text")).performScrollTo().performTextReplacement("changed")
    compose.onNodeWithText("1 spaces replaced", substring = true).assertDoesNotExist()
  }

  @Test fun wordFrequencyAnalyzesAndChangesOptions() {
    open("word-frequency")
    fill("Text", "Model model model data data the")
    compose.onNodeWithText("Analyze text").performScrollTo().performClick()
    waitFor("6 words · 2 unique results")
    compose.onNodeWithText("3  model\n2  data").assertExists()
    compose.onNodeWithText("2-word phrases").performScrollTo().performClick()
    compose.onNodeWithText("6 words · 2 unique results").assertDoesNotExist()
  }

  @Test fun utmBuildsUsableLinksInNativeEditors() {
    open("utm-batch-builder")
    fill("Landing pages", "https://example.com")
    fill("Source", "email")
    fill("Medium", "newsletter")
    fill("Campaign", "Fall Launch")
    compose.onNodeWithText("Build links").performScrollTo().performClick()
    waitFor("1 links")
    compose.onNodeWithText("https://example.com/?utm_source=email&utm_medium=newsletter&utm_campaign=fall_launch").assertExists()
    compose.onNodeWithText("Copy all links").performScrollTo().assertExists()
  }

  @Test fun qrCodeGeneratesANativeBitmapAndSaveAction() {
    open("qr-code-generator")
    fill("Text or link", "https://www.danielshort.me")
    compose.onNodeWithText("Create QR code").performScrollTo().performClick()
    waitFor("Save PNG")
    compose.onNodeWithContentDescription("Generated QR code").assertExists()
    compose.onNodeWithText("Save PNG").performScrollTo().assertExists()
  }

  @Test fun wifiPasswordIsNotWrittenToSavedInstanceState() {
    val restoration = StateRestorationTester(compose)
    restoration.setContent { MaterialTheme { NativeToolsScreen("qr-code-generator", {}) } }
    compose.onNodeWithText("Wi-Fi").performClick()
    fill("Network name", "Test network")
    fill("Password", "test-only-passphrase")
    restoration.emulateSavedInstanceStateRestore()
    assertEquals("Test network", compose.onNode(hasSetTextAction() and hasText("Network name"))
      .fetchSemanticsNode().config[SemanticsProperties.EditableText].text)
    assertEquals("", compose.onNode(hasSetTextAction() and hasText("Password"))
      .fetchSemanticsNode().config[SemanticsProperties.EditableText].text)
  }

  @Test fun bitmapExportsKeepPngTransparencyAndUseWhiteJpegMatte() {
    val input = Bitmap.createBitmap(300, 150, Bitmap.Config.ARGB_8888)
    input.eraseColor(Color.TRANSPARENT)
    val png = optimizeNativeImage(input, 200, 85, "PNG")
    assertEquals(200, png.bitmap.width)
    assertEquals(100, png.bitmap.height)
    assertEquals(0, Color.alpha(png.bitmap.getPixel(20, 20)))
    assertNotNull(BitmapFactory.decodeByteArray(png.bytes, 0, png.bytes.size))
    val jpeg = optimizeNativeImage(input, 200, 90, "JPEG")
    assertEquals(255, Color.alpha(jpeg.bitmap.getPixel(20, 20)))
    assertTrue(Color.red(jpeg.bitmap.getPixel(20, 20)) >= 250)
    val webp = optimizeNativeImage(input, 200, 85, "WebP")
    assertEquals("image/webp", webp.mimeType)
    assertNotNull(BitmapFactory.decodeByteArray(webp.bytes, 0, webp.bytes.size))
    assertEquals(0, Color.alpha(webp.bitmap.getPixel(20, 20)))
    assertFalse(input.isRecycled)
  }

  @Test fun imagePreviewRespectsHeightLimitAndDoesNotOverlapCaption() {
    val bitmap = Bitmap.createBitmap(384, 384, Bitmap.Config.ARGB_8888)
    compose.setContent {
      MaterialTheme {
        Column(Modifier.width(390.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
          ImagePreview(bitmap, "Bounded preview", checkerboard = true)
          Text("Image dimensions")
        }
      }
    }
    val imageBounds = compose.onNodeWithContentDescription("Bounded preview").fetchSemanticsNode().boundsInRoot
    val captionBounds = compose.onNodeWithText("Image dimensions").fetchSemanticsNode().boundsInRoot
    assertTrue(imageBounds.height <= with(compose.density) { 340.dp.toPx() } + 1f)
    assertTrue("Preview must leave the intended space before its caption", captionBounds.top - imageBounds.bottom >= with(compose.density) { 13.dp.toPx() })
  }

  @Test fun imageImportPreservesRequestedResolutionAndTransparency() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val file = File.createTempFile("native-image-size-", ".png", context.cacheDir)
    val input = Bitmap.createBitmap(3000, 1500, Bitmap.Config.ARGB_8888)
    try {
      file.outputStream().use { input.compress(Bitmap.CompressFormat.PNG, 100, it) }
      val loaded = loadNativeImage(context, Uri.fromFile(file))
      assertEquals(3000, loaded.originalWidth)
      assertEquals(2048, loaded.bitmap.width)
      assertEquals(1024, loaded.bitmap.height)
      assertEquals(0, Color.alpha(loaded.bitmap.getPixel(10, 10)))
      loaded.bitmap.recycle()
    } finally { input.recycle(); file.delete() }
  }

  @Test fun imageImportCorrectsEveryExifOrientation() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val file = File.createTempFile("native-image-exif-", ".jpg", context.cacheDir)
    val input = Bitmap.createBitmap(120, 80, Bitmap.Config.ARGB_8888)
    val colors = listOf(Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW)
    Canvas(input).apply {
      colors.forEachIndexed { i, color ->
        drawRect((i % 2) * 60f, (i / 2) * 40f, (i % 2 + 1) * 60f, (i / 2 + 1) * 40f, Paint().apply { this.color = color })
      }
    }
    val expectedCorners = listOf(
      listOf(0, 1, 2, 3), listOf(1, 0, 3, 2), listOf(3, 2, 1, 0), listOf(2, 3, 0, 1),
      listOf(0, 2, 1, 3), listOf(2, 0, 3, 1), listOf(3, 1, 2, 0), listOf(1, 3, 0, 2)
    )
    try {
      for (orientation in 1..8) {
        file.outputStream().use { input.compress(Bitmap.CompressFormat.JPEG, 100, it) }
        ExifInterface(file).apply { setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString()); saveAttributes() }
        val loaded = loadNativeImage(context, Uri.fromFile(file))
        assertEquals(if (orientation >= 5) 80 else 120, loaded.bitmap.width)
        assertEquals(if (orientation >= 5) 120 else 80, loaded.bitmap.height)
        assertEquals(loaded.originalWidth, loaded.bitmap.width)
        assertEquals(loaded.originalHeight, loaded.bitmap.height)
        expectedCorners[orientation - 1].forEachIndexed { corner, index ->
          val pixel = loaded.bitmap.getPixel(if (corner % 2 == 0) 10 else loaded.bitmap.width - 10,
            if (corner / 2 == 0) 10 else loaded.bitmap.height - 10)
          val expected = colors[index]
          assertTrue("EXIF $orientation corner $corner", kotlin.math.abs(Color.red(pixel) - Color.red(expected)) < 12 &&
            kotlin.math.abs(Color.green(pixel) - Color.green(expected)) < 12 && kotlin.math.abs(Color.blue(pixel) - Color.blue(expected)) < 12)
        }
        loaded.bitmap.recycle()
      }
    } finally { input.recycle(); file.delete() }
  }
}
