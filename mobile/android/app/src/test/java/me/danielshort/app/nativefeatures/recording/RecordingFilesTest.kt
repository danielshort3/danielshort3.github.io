package me.danielshort.app.nativefeatures.recording

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class RecordingFilesTest {
  @get:Rule val temporary = TemporaryFolder()

  @Test fun libraryRecoversOnlyCompletedNonemptyFilesNewestFirst() {
    val directory = temporary.newFolder("recordings")
    val old = File(directory, "old.mp4").apply { writeText("completed"); setLastModified(1_000L) }
    val latest = File(directory, "latest.mp4").apply { writeText("completed"); setLastModified(2_000L) }
    File(directory, "interrupted.part").writeText("unfinished")
    File(directory, "empty.mp4").createNewFile()
    File(directory, "directory.mp4").mkdirs()
    assertEquals(listOf(latest, old), RecordingFiles.completed(directory))
    assertEquals(listOf(latest, old), RecordingFiles.completed(File(directory.absolutePath)))
  }

  @Test fun stoppedClipBecomesRecoverableOnlyAfterSuccessfulFinalization() {
    val directory = temporary.newFolder("recordings")
    val partial = File(directory, "screen-42.part").apply { writeText("completed recording bytes") }
    assertTrue(RecordingFiles.completed(directory).isEmpty())
    assertNull(RecordingFiles.finalize(partial, stoppedSuccessfully = false))
    assertTrue(partial.exists())
    val clip = RecordingFiles.finalize(partial, stoppedSuccessfully = true)
    assertEquals(File(directory, "screen-42.mp4"), clip)
    assertFalse(partial.exists())
    assertEquals(listOf(clip), RecordingFiles.completed(directory))
  }

  @Test fun finalizationNeverOverwritesAnExistingClipOrPromotesAnEmptyFile() {
    val directory = temporary.newFolder("recordings")
    val existing = File(directory, "screen.mp4").apply { writeText("keep existing") }
    val collision = File(directory, "screen.part").apply { writeText("new bytes") }
    assertNull(RecordingFiles.finalize(collision, true))
    assertEquals("keep existing", existing.readText())
    val empty = File(directory, "empty.part").apply { createNewFile() }
    assertNull(RecordingFiles.finalize(empty, true))
    assertNull(RecordingFiles.finalize(existing, true))
  }
}
