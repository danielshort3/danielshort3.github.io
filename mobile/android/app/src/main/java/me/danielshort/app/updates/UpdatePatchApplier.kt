package me.danielshort.app.updates

import java.io.DataInputStream
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream
import java.io.RandomAccessFile
import java.io.SequenceInputStream
import java.security.MessageDigest
import java.util.zip.CRC32
import java.util.zip.Inflater
import java.util.zip.InflaterInputStream

internal fun ByteArray.hex(): String = joinToString("") { "%02x".format(it.toInt() and 255) }

fun sha256(file: File, checkCancelled: () -> Unit = {}): String {
  val digest = MessageDigest.getInstance("SHA-256")
  file.inputStream().buffered().use { input ->
    val buffer = ByteArray(64 * 1024)
    while (true) {
      checkCancelled()
      val count = input.read(buffer)
      if (count < 0) break
      digest.update(buffer, 0, count)
    }
  }
  return digest.digest().hex()
}

object UpdatePatchApplier {
  const val MAX_OPERATIONS = 100_000

  /** Output is untrusted until its exact length and SHA-256 match the signed target's manifest. */
  fun apply(
    base: File,
    patch: File,
    output: File,
    expectedBaseHash: String,
    expectedTargetHash: String,
    expectedTargetSize: Long,
    checkCancelled: () -> Unit = {},
  ) {
    require(base.canonicalFile != output.canonicalFile && patch.canonicalFile != output.canonicalFile)
    try {
      require(expectedTargetSize in 1..MAX_UPDATE_BYTES)
      require(base.length() in 1..MAX_UPDATE_BYTES && patch.length() in 1..MAX_UPDATE_BYTES)
      require(sha256(base, checkCancelled) == expectedBaseHash) { "The installed app changed before patching" }
      var written = 0L
      DataInputStream(StrictPatchGzip(patch.inputStream().buffered())).use { input ->
        val magic = ByteArray(8).also(input::readFully)
        require(magic.contentEquals("DSUPD001".toByteArray(Charsets.US_ASCII)))
        val baseHash = ByteArray(32).also(input::readFully).hex()
        val targetHash = ByteArray(32).also(input::readFully).hex()
        val baseSize = input.readLong()
        val targetSize = input.readLong()
        require(baseHash == expectedBaseHash && targetHash == expectedTargetHash && baseSize == base.length() && targetSize == expectedTargetSize)
        RandomAccessFile(base, "r").use { source ->
          output.outputStream().buffered(64 * 1024).use { destination ->
            val buffer = ByteArray(64 * 1024)
            var operations = 0
            while (true) {
              checkCancelled()
              val opcode = input.readUnsignedByte()
              if (opcode == 255) break
              require(++operations <= MAX_OPERATIONS)
              val offset = if (opcode == 0) input.readLong() else 0L
              require(opcode == 0 || opcode == 1) { "Unknown patch operation" }
              val length = input.readInt()
              require(length > 0 && length.toLong() <= targetSize - written) { "Patch output exceeds its declared size" }
              if (opcode == 0) {
                require(offset >= 0 && offset <= baseSize && length.toLong() <= baseSize - offset) { "Patch copy exceeds the installed app" }
                source.seek(offset)
              }
              var remaining = length
              while (remaining > 0) {
                checkCancelled()
                val count = minOf(remaining, buffer.size)
                if (opcode == 0) source.readFully(buffer, 0, count) else input.readFully(buffer, 0, count)
                destination.write(buffer, 0, count)
                remaining -= count
              }
              written += length
            }
            require(written == targetSize && input.read() == -1) { "Incomplete patch or trailing patch data" }
          }
        }
      }
      require(output.length() == expectedTargetSize && sha256(output, checkCancelled) == expectedTargetHash) { "The patched app did not match the published release" }
    } catch (failure: Throwable) {
      output.delete()
      throw failure
    }
  }
}

/** A single standard gzip member, with a checked CRC/size and no concatenated or ignored trailing bytes. */
private class StrictPatchGzip(source: InputStream) : InflaterInputStream(source, Inflater(true), 64 * 1024) {
  private val checksum = CRC32()
  private var count = 0L
  private var finished = false

  init {
    try {
      val header = ByteArray(10)
      DataInputStream(source).readFully(header)
      require((header[0].toInt() and 255) == 31 && (header[1].toInt() and 255) == 139 && header[2].toInt() == 8)
      // The version-one publisher emits the canonical ten-byte gzip header, with no optional fields.
      require(header[3].toInt() == 0) { "Unsupported patch compression header" }
    } catch (failure: Throwable) {
      source.close()
      inf.end()
      throw failure
    }
  }

  override fun read(bytes: ByteArray, offset: Int, length: Int): Int {
    val result = super.read(bytes, offset, length)
    if (result > 0) {
      checksum.update(bytes, offset, result)
      count += result
      require(count <= MAX_UPDATE_BYTES + 13L * UpdatePatchApplier.MAX_OPERATIONS + 89) { "Expanded patch is too large" }
    } else if (result == -1 && !finished) {
      finished = true
      val tail = SequenceInputStream(ByteArrayInputStream(buf, len - inf.remaining, inf.remaining), `in`)
      fun littleEndianInt(): Long {
        var value = 0L
        repeat(4) { shift ->
          val byte = tail.read()
          require(byte >= 0) { "Truncated patch compression trailer" }
          value = value or (byte.toLong() shl (shift * 8))
        }
        return value
      }
      require(littleEndianInt() == checksum.value && littleEndianInt() == (count and 0xffffffffL)) { "Corrupt patch compression trailer" }
      require(tail.read() == -1) { "Trailing compressed patch data" }
    }
    return result
  }

  override fun close() {
    try { super.close() } finally { inf.end() }
  }
}
