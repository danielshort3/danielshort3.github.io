import java.util.zip.ZipFile

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
  id("org.jetbrains.kotlin.plugin.compose")
}

val productionCatalogUrl = "https://www.danielshort.me/app-content/v1/catalog.json"
val previewCatalogUrl = providers.gradleProperty("catalogUrl").orElse(productionCatalogUrl)
val reviewAppUpdateUrl = "https://www.danielshort.me/app-updates/review/latest.json"
val websiteRoot = rootProject.projectDir.resolve("../..").canonicalFile
val generatedCatalogAssets = layout.buildDirectory.dir("generated/catalogAssets")

// Upload-key material is supplied outside Git. A missing configuration leaves a
// locally built release bundle unsigned for inspection, never signed with a debug key.
val playSigningValues = mapOf(
  "storeFile" to providers.environmentVariable("ANDROID_PLAY_UPLOAD_STORE_FILE").orNull,
  "storePassword" to providers.environmentVariable("ANDROID_PLAY_UPLOAD_STORE_PASSWORD").orNull,
  "keyAlias" to providers.environmentVariable("ANDROID_PLAY_UPLOAD_KEY_ALIAS").orNull,
  "keyPassword" to providers.environmentVariable("ANDROID_PLAY_UPLOAD_KEY_PASSWORD").orNull
)
val playSigningConfigured = playSigningValues.values.all { !it.isNullOrBlank() }
require(playSigningValues.values.none { !it.isNullOrBlank() } || playSigningConfigured) {
  "Set all four ANDROID_PLAY_UPLOAD_* signing variables, or leave all unset for an unsigned release bundle."
}

val generateCatalog by tasks.registering(Exec::class) {
  group = "content"
  description = "Generate the bundled offline catalog from the website's authoritative content."
  workingDir = websiteRoot
  commandLine(providers.gradleProperty("nodeExecutable").orElse("node").get(), "build/generate-mobile-content.js")
  inputs.dir(websiteRoot.resolve("content"))
  inputs.file(websiteRoot.resolve("build/generate-mobile-content.js"))
  outputs.file(websiteRoot.resolve("dist/app-content/v1/catalog.json"))
  // Image hashes and content-loader behavior also affect the offline snapshot.
  outputs.upToDateWhen { false }
}

val bundleCatalog by tasks.registering(Sync::class) {
  dependsOn(generateCatalog)
  from(websiteRoot.resolve("dist/app-content/v1/catalog.json"))
  into(generatedCatalogAssets)
}

android {
  namespace = "me.danielshort.app"
  compileSdk = 36
  compileSdkMinor = 1
  buildToolsVersion = "36.1.0"

  defaultConfig {
    applicationId = "me.danielshort.app"
    minSdk = 26
    targetSdk = 36
    versionCode = 12
    versionName = "0.5.5"
    buildConfigField("String", "CONTENT_URL", "\"$productionCatalogUrl\"")
    buildConfigField("String", "APP_UPDATE_URL", "\"\"")
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  buildTypes {
    debug {
      applicationIdSuffix = ".debug"
      versionNameSuffix = "-debug"
      buildConfigField("boolean", "ENABLE_SIDELOAD_UPDATES", "true")
      buildConfigField("String", "APP_UPDATE_URL", "\"$reviewAppUpdateUrl\"")
      buildConfigField("String", "CONTENT_URL", "\"${previewCatalogUrl.get().replace("\\", "\\\\").replace("\"", "\\\"")}\"")
    }
    release {
      buildConfigField("boolean", "ENABLE_SIDELOAD_UPDATES", "false")
      isMinifyEnabled = true
      isShrinkResources = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
    }
  }

  if (playSigningConfigured) {
    signingConfigs.create("playUpload") {
      storeFile = file(playSigningValues.getValue("storeFile")!!)
      storePassword = playSigningValues.getValue("storePassword")!!
      keyAlias = playSigningValues.getValue("keyAlias")!!
      keyPassword = playSigningValues.getValue("keyPassword")!!
    }
    buildTypes.getByName("release").signingConfig = signingConfigs.getByName("playUpload")
  }

  buildFeatures {
    compose = true
    buildConfig = true
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  sourceSets["main"].assets.srcDir(generatedCatalogAssets)
  packaging.resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
}

kotlin {
  compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

tasks.named("preBuild") { dependsOn(bundleCatalog) }

tasks.register("verifyPlayRelease") {
  group = "verification"
  description = "Check that the Play bundle excludes sideload permissions, entry points, and update transport."
  dependsOn("bundleRelease", "processDebugMainManifest")
  doLast {
    val releaseManifest = layout.buildDirectory.file("intermediates/merged_manifest/release/processReleaseMainManifest/AndroidManifest.xml").get().asFile.readText()
    val debugManifest = layout.buildDirectory.file("intermediates/merged_manifest/debug/processDebugMainManifest/AndroidManifest.xml").get().asFile.readText()
    val sideloadEntries = listOf("REQUEST_INSTALL_PACKAGES", "UPDATE_PACKAGES_WITHOUT_USER_ACTION", "AutomaticInstallReceiver")
    sideloadEntries.forEach { entry ->
      check(!releaseManifest.contains(entry)) { "Play release manifest still contains $entry" }
      check(debugManifest.contains(entry)) { "Review manifest lost $entry" }
    }
    val releaseConfig = layout.buildDirectory.file("generated/source/buildConfig/release/me/danielshort/app/BuildConfig.java").get().asFile.readText()
    val debugConfig = layout.buildDirectory.file("generated/source/buildConfig/debug/me/danielshort/app/BuildConfig.java").get().asFile.readText()
    check(releaseConfig.contains("ENABLE_SIDELOAD_UPDATES = false")) { "Play release enabled sideload updates" }
    check(releaseConfig.contains("APP_UPDATE_URL = \"\"")) { "Play release has an app-update feed" }
    check(debugConfig.contains("ENABLE_SIDELOAD_UPDATES = true")) { "Review updater is disabled" }
    val bundle = layout.buildDirectory.file("outputs/bundle/release/app-release.aab").get().asFile
    ZipFile(bundle).use { archive ->
      val dexEntries = archive.entries().asSequence().filter { it.name.matches(Regex("base/dex/classes.*\\.dex")) }.toList()
      check(dexEntries.isNotEmpty()) { "Play bundle has no DEX to inspect" }
      val prohibited = listOf("DanielShort-Android-Updater/1", "app-updates/review/latest.json",
        "app-updates/stable/latest.json", "REQUEST_INSTALL_PACKAGES", "UPDATE_PACKAGES_WITHOUT_USER_ACTION",
        "MANAGE_UNKNOWN_APP_SOURCES", "AutomaticInstallReceiver", "Allow app installation", "Install update")
      dexEntries.forEach { entry ->
        val text = archive.getInputStream(entry).use { it.readBytes() }.toString(Charsets.ISO_8859_1)
        prohibited.forEach { marker ->
          check(!text.contains(marker)) { "Play bundle still contains sideload marker $marker" }
        }
      }
    }
  }
}

dependencies {
  val composeBom = platform("androidx.compose:compose-bom:2025.08.01")
  implementation(composeBom)
  implementation("androidx.activity:activity-compose:1.10.1")
  implementation("androidx.compose.ui:ui")
  implementation("androidx.compose.ui:ui-tooling-preview")
  implementation("androidx.compose.foundation:foundation")
  implementation("androidx.compose.material3:material3")
  implementation("androidx.compose.material:material-icons-extended")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.2")
  implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.2")
  implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.2")
  implementation("androidx.work:work-runtime-ktx:2.10.3")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
  implementation("io.coil-kt:coil-compose:2.7.0")
  implementation("io.coil-kt:coil-svg:2.7.0")
  implementation("com.google.zxing:core:3.5.3")
  implementation("com.google.android.gms:play-services-mlkit-subject-segmentation:16.0.0-beta1")
  implementation("androidx.exifinterface:exifinterface:1.4.1")
  implementation("com.android.tools.build:apksig:8.13.2")
  testImplementation("junit:junit:4.13.2")
  testImplementation("org.json:json:20250517")
  androidTestImplementation(composeBom)
  androidTestImplementation("androidx.test.ext:junit:1.3.0")
  androidTestImplementation("androidx.test:runner:1.7.0")
  // 3.7 removes InputManager.getInstance reflection, which fails on Android 16.
  androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
  androidTestImplementation("androidx.compose.ui:ui-test-junit4")
  debugImplementation("androidx.compose.ui:ui-tooling")
  debugImplementation("androidx.compose.ui:ui-test-manifest")
}
