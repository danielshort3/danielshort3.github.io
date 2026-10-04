package me.danielshort.wayfarers.content

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Central-directory validation rejects links and ambiguous ZIPs before any extraction. */
object GuildContentArchiveReader {
  fun extract(archive: File, destination: File, manifest: GuildContentManifest, cancel: () -> Unit = {}) {
    require(!destination.exists()) { "Extraction needs a new private directory" }
    verifyArchive(archive, manifest.archive, cancel)
    validateDirectory(archive, manifest)
    require(destination.mkdirs())
    val expected = manifest.records.associateBy { it.path }
    val seen = mutableSetOf<String>()
    var extracted = 0L
    try {
      ZipInputStream(FileInputStream(archive)).use { zip ->
        while (true) {
          cancel()
          val entry = zip.nextEntry ?: break
          val record = expected[entry.name] ?: throw GuildContentFailure("The game update contains an unexpected file.")
          require(!entry.isDirectory && seen.add(entry.name))
          val target = safeFile(destination, record.path)
          val parent = requireNotNull(target.parentFile)
          require(parent.mkdirs() || parent.isDirectory)
          val digest = MessageDigest.getInstance("SHA-256")
          var count = 0L
          FileOutputStream(target).use { output ->
            val buffer = ByteArray(32 * 1024)
            while (true) {
              cancel()
              val read = zip.read(buffer)
              if (read < 0) break
              count += read
              extracted += read
              require(count <= record.size && count <= GuildContentLimits.MAX_FILE && extracted <= GuildContentLimits.MAX_EXTRACTED)
              digest.update(buffer, 0, read)
              output.write(buffer, 0, read)
            }
            require(count == record.size && hex(digest.digest()) == record.sha256)
            output.flush()
            output.fd.sync()
          }
          zip.closeEntry()
        }
      }
      require(seen == expected.keys) { "The game update is incomplete" }
    } catch (failure: Throwable) {
      destination.deleteRecursively()
      throw failure
    }
  }

  fun verifyDirectory(directory: File, manifest: GuildContentManifest, cancel: () -> Unit = {}) {
    val expected = manifest.records.associateBy { it.path }
    require(directory.isDirectory && !java.nio.file.Files.isSymbolicLink(directory.toPath()))
    val actual = directory.walkTopDown().onEnter { folder -> require(!java.nio.file.Files.isSymbolicLink(folder.toPath())); true }
      .filter { it.isFile }.map { it.relativeTo(directory).invariantSeparatorsPath }
      .filter { it != "manifest-envelope.json" }.toSet()
    require(actual == expected.keys) { "Installed game content is incomplete" }
    for (record in manifest.records) {
      cancel()
      val file = safeFile(directory, record.path)
      require(file.isFile && file.length() == record.size && !java.nio.file.Files.isSymbolicLink(file.toPath()))
      require(fileDigest(file, cancel) == record.sha256) { "Installed game content was changed" }
    }
  }

  fun verifyArchive(file: File, archive: GuildContentArchive, cancel: () -> Unit = {}) {
    require(file.isFile && file.length() == archive.size && file.length() <= GuildContentLimits.MAX_ARCHIVE)
    require(fileDigest(file, cancel) == archive.sha256) { "Downloaded content checksum mismatch" }
  }

  private fun validateDirectory(archive: File, manifest: GuildContentManifest) {
    RandomAccessFile(archive, "r").use { file ->
      val tailLength = minOf(file.length(), 65557L).toInt()
      val tail = ByteArray(tailLength)
      file.seek(file.length() - tailLength)
      file.readFully(tail)
      val end = (tail.size - 22 downTo 0).firstOrNull { u32(tail, it) == 0x06054b50L && it + 22 + u16(tail, it + 20) == tail.size }
        ?: throw GuildContentFailure("The game update archive is invalid.")
      require(u16(tail, end + 4) == 0 && u16(tail, end + 6) == 0)
      val count = u16(tail, end + 10)
      require(count in 2..GuildContentLimits.MAX_FILES && u16(tail, end + 8) == count)
      val directorySize = u32(tail, end + 12)
      val directoryOffset = u32(tail, end + 16)
      require(directoryOffset + directorySize == file.length() - tailLength + end)
      val expected = manifest.records.associateBy { it.path }
      require(count == expected.size)
      val seen = mutableSetOf<String>()
      file.seek(directoryOffset)
      repeat(count) {
        val header = ByteArray(46)
        file.readFully(header)
        require(u32(header, 0) == 0x02014b50L)
        val flags = u16(header, 8)
        require(flags and 1 == 0 && flags and 0x40 == 0 && flags and 0x2000 == 0)
        require(u16(header, 10) in setOf(0, 8))
        val nameLength = u16(header, 28)
        require(nameLength in 1..200)
        val nameBytes = ByteArray(nameLength)
        file.readFully(nameBytes)
        val path = GuildContentLimits.utf8(nameBytes)
        val record = expected[path] ?: throw GuildContentFailure("The game update contains an unexpected path.")
        require(seen.add(path) && GuildContentLimits.validPath(path))
        require(u32(header, 24) == record.size && u32(header, 20) <= archive.length())
        require(u16(header, 34) == 0)
        // UNIX external mode bits distinguish ordinary files from symlinks, devices and directories.
        val mode = (u32(header, 38) ushr 16).toInt() and 0xf000
        require(mode == 0 || mode == 0x8000) { "Archive links and special files are forbidden" }
        require(u32(header, 38) and 0x10L == 0L)
        require(u32(header, 42) < directoryOffset)
        file.seek(file.filePointer + u16(header, 30) + u16(header, 32))
        require(file.filePointer <= directoryOffset + directorySize)
      }
      require(file.filePointer == directoryOffset + directorySize && seen == expected.keys)
    }
  }

  private fun safeFile(root: File, path: String): File {
    require(GuildContentLimits.validPath(path))
    return File(root, path).also { require(it.canonicalPath.startsWith(root.canonicalPath + File.separator)) }
  }
  private fun fileDigest(file: File, cancel: () -> Unit): String {
    val digest = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).use { stream ->
      val buffer = ByteArray(32 * 1024)
      while (true) {
        cancel()
        val read = stream.read(buffer)
        if (read < 0) break
        digest.update(buffer, 0, read)
      }
    }
    return hex(digest.digest())
  }
  private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it.toInt() and 255) }
  private fun u16(bytes: ByteArray, offset: Int) = (bytes[offset].toInt() and 255) or ((bytes[offset + 1].toInt() and 255) shl 8)
  private fun u32(bytes: ByteArray, offset: Int) = u16(bytes, offset).toLong() or (u16(bytes, offset + 2).toLong() shl 16)
}
