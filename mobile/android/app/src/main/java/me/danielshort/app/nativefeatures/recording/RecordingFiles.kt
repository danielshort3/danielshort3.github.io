package me.danielshort.app.nativefeatures.recording

import java.io.File

/** Only successfully stopped, finalized clips belong in the recoverable library. */
internal object RecordingFiles {
  fun completed(directory: File): List<File> = directory.listFiles()
    ?.filter { it.isFile && it.extension == "mp4" && it.length() > 0 }
    ?.sortedWith(compareByDescending<File> { it.lastModified() }.thenByDescending { it.name })
    .orEmpty()

  fun finalize(partial: File, stoppedSuccessfully: Boolean): File? {
    if (!stoppedSuccessfully || !partial.isFile || partial.length() == 0L || partial.extension != "part") return null
    val clip = File(partial.parentFile, "${partial.nameWithoutExtension}.mp4")
    return clip.takeIf { !it.exists() && partial.renameTo(it) }
  }
}
