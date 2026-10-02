package me.danielshort.wayfarers

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GuildCheckpointStoreTest {
  @get:Rule val temporary = TemporaryFolder()
  private fun envelope(createdAt: Long = 1000, savedAt: Long = 2000, boots: Int = 1): String = JSONObject()
    .put("format", "wayfarers-guild-save").put("version", 4).put("savedAt", savedAt)
    .put("state", JSONObject().put("schemaVersion", 4).put("createdAt", createdAt).put("lastUpdate", savedAt)
      .put("resources", JSONObject()).put("upgrades", JSONObject().put("boots", boots)).put("rooms", JSONObject())).toString()

  @Test fun exactEnvelopeSurvivesASeparateStoreInstance() {
    val file = temporary.newFile("guild.json")
    val text = envelope()
    assertTrue(GuildCheckpointStore(file).write(text))
    assertEquals(text, GuildCheckpointStore(file).read()!!.text)
    assertFalse(file.resolveSibling("guild.json.pending").exists())
  }

  @Test fun corruptOrTruncatedCheckpointIsNeverRestored() {
    val file = temporary.newFile("guild.json")
    val store = GuildCheckpointStore(file)
    assertTrue(store.write(envelope()))
    val record = JSONObject(file.readText()).put("text", envelope(boots = 99))
    file.writeText(record.toString())
    assertNull(store.read())
    file.writeText("{\"version\":")
    assertNull(store.read())
  }

  @Test fun invalidOrOlderSaveCannotReplaceTheLastVerifiedCheckpoint() {
    val store = GuildCheckpointStore(temporary.newFile("guild.json"))
    val original = envelope()
    assertTrue(store.write(original))
    assertFalse(store.write(envelope(savedAt = 1999)))
    assertFalse(store.write("{}"))
    assertFalse(store.write(envelope().replace("\"version\":4", "\"version\":5")))
    assertEquals(original, store.read()!!.text)
  }

  @Test fun differentGuildNeedsReviewedReplacementIdentity() {
    val store = GuildCheckpointStore(temporary.newFile("guild.json"))
    assertTrue(store.write(envelope()))
    assertFalse(store.write(envelope(createdAt = 3000, savedAt = 4000)))
    assertFalse(store.write(envelope(createdAt = 3000, savedAt = 4000), 999.0))
    assertTrue(store.write(envelope(createdAt = 3000, savedAt = 4000), 1000.0))
    assertEquals(1000.0, store.read()!!.replacesCreatedAt!!, 0.0)
    assertTrue(store.write(envelope(createdAt = 3000, savedAt = 4100)))
    assertEquals(1000.0, store.read()!!.replacesCreatedAt!!, 0.0)
  }

  @Test fun equalTimestampPurchaseIsStillRecorded() {
    val store = GuildCheckpointStore(temporary.newFile("guild.json"))
    assertTrue(store.write(envelope(boots = 1)))
    assertTrue(store.write(envelope(boots = 2)))
    assertEquals(2, JSONObject(store.read()!!.text).getJSONObject("state").getJSONObject("upgrades").getInt("boots"))
  }

  @Test fun bootstrapEscapesTextAndDoesNotExposeUnverifiedBytes() {
    val file = temporary.newFile("guild.json")
    val store = GuildCheckpointStore(file)
    assertEquals("window.WayfarersNativeCheckpoint=null;", store.bootstrapScript())
    assertTrue(store.write(envelope()))
    val json = store.bootstrapScript().removePrefix("window.WayfarersNativeCheckpoint=").removeSuffix(";")
    assertEquals(envelope(), JSONObject(json).getString("text"))
    file.writeText("not json")
    assertEquals("window.WayfarersNativeCheckpoint=null;", store.bootstrapScript())
  }
}
