import java.io.File

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
  id("org.jetbrains.kotlin.plugin.compose")
}

val websiteRoot = rootProject.projectDir.resolve("../..").canonicalFile
val gameAssets = layout.buildDirectory.dir("generated/gameAssets")
val updaterSources = layout.buildDirectory.dir("generated/sharedUpdater/main")
val updaterTests = layout.buildDirectory.dir("generated/sharedUpdater/test")
val versionCodeProperty = providers.gradleProperty("wayfarersVersionCode").orElse("9").get()
val versionNameProperty = providers.gradleProperty("wayfarersVersionName").orElse("0.7.0").get()
val gameVersionCode = versionCodeProperty.toIntOrNull()
  ?: error("wayfarersVersionCode must be a positive Android version code.")
require(gameVersionCode > 0) { "wayfarersVersionCode must be positive." }
require(Regex("[0-9A-Za-z][0-9A-Za-z._+-]{0,63}").matches(versionNameProperty)) {
  "wayfarersVersionName must be a short version identifier."
}
val updateFeed = "https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-updates/latest.json"
val persistentSigningKey = File(System.getProperty("user.home"), ".android/debug.keystore")

// Compile the canonical implementation verbatim. The only application-specific
// updater class is supplied by this module: its private installer receiver.
val shareUpdaterSources by tasks.registering(Sync::class) {
  from(rootProject.file("app/src/main/java")) {
    include("me/danielshort/app/updates/*.kt", "me/danielshort/app/data/AppSettings.kt")
    exclude("me/danielshort/app/updates/AutomaticInstallReceiver.kt")
  }
  into(updaterSources)
}
val shareUpdaterTests by tasks.registering(Sync::class) {
  from(rootProject.file("app/src/test/java")) {
    include("me/danielshort/app/updates/*.kt")
  }
  into(updaterTests)
}
val bundleWayfarers by tasks.registering(Exec::class) {
  group = "content"
  description = "Bundle canonical Wayfarers game sources for the stable offline appassets origin."
  workingDir = websiteRoot
  commandLine(providers.gradleProperty("nodeExecutable").orElse("node").get(),
    "build/bundle-wayfarers-android.cjs", "--output", gameAssets.get().asFile.absolutePath)
  inputs.file(websiteRoot.resolve("build/bundle-wayfarers-android.cjs"))
  inputs.file(websiteRoot.resolve("pages/games/wayfarers-guild.html"))
  inputs.file(websiteRoot.resolve("css/games/wayfarers-guild.css"))
  inputs.dir(websiteRoot.resolve("js/games/wayfarers-guild"))
  inputs.dir(websiteRoot.resolve("img/wayfarers-guild"))
  inputs.dir(projectDir.resolve("web"))
  outputs.dir(gameAssets)
}
val verifyPersistentSigningIdentity by tasks.registering {
  group = "verification"
  description = "Require the retained workstation signing key; never generate a replacement identity."
  doLast {
    check(persistentSigningKey.isFile) {
      "The persistent workstation signing key is missing. Restore the retained key outside source control; do not generate a replacement or publish a CI-generated identity."
    }
  }
}

android {
  // BuildConfig remains at the canonical updater's import path. Android app/data
  // identity is the separate applicationId, not this compile-time namespace.
  namespace = "me.danielshort.app"
  compileSdk = 36
  compileSdkMinor = 1
  buildToolsVersion = "36.1.0"
  defaultConfig {
    applicationId = "me.danielshort.wayfarers"
    minSdk = 26
    targetSdk = 36
    versionCode = gameVersionCode
    versionName = versionNameProperty
    buildConfigField("String", "APP_UPDATE_URL", "\"$updateFeed\"")
    buildConfigField("String", "GAME_URL", "\"https://appassets.androidplatform.net/assets/wayfarers/index.html\"")
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }
  signingConfigs.getByName("debug") {
    storeFile = persistentSigningKey
  }
  buildTypes {
    debug {
      // No suffix: baseline and candidate must update the same standalone app.
      signingConfig = signingConfigs.getByName("debug")
    }
    release {
      // This is a sideload preview channel, not a Google Play production build.
      signingConfig = signingConfigs.getByName("debug")
      isMinifyEnabled = false
      isShrinkResources = false
    }
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  sourceSets["main"].java.srcDir(updaterSources)
  sourceSets["main"].assets.srcDir(gameAssets)
  sourceSets["test"].java.srcDir(updaterTests)
  packaging.resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
}
kotlin {
  compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}
tasks.named("preBuild") {
  dependsOn(shareUpdaterSources, shareUpdaterTests, bundleWayfarers, verifyPersistentSigningIdentity)
}

dependencies {
  val composeBom = platform("androidx.compose:compose-bom:2025.08.01")
  implementation(composeBom)
  implementation("androidx.activity:activity-compose:1.10.1")
  implementation("androidx.webkit:webkit:1.16.0")
  implementation("androidx.compose.ui:ui")
  implementation("androidx.compose.ui:ui-tooling-preview")
  implementation("androidx.compose.foundation:foundation")
  implementation("androidx.compose.material3:material3")
  implementation("androidx.compose.material:material-icons-extended")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.2")
  implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.2")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
  implementation("com.android.tools.build:apksig:8.13.2")
  testImplementation("junit:junit:4.13.2")
  testImplementation("org.json:json:20250517")
  androidTestImplementation(composeBom)
  androidTestImplementation("androidx.test.ext:junit:1.3.0")
  androidTestImplementation("androidx.test:runner:1.7.0")
  androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
  androidTestImplementation("androidx.compose.ui:ui-test-junit4")
  debugImplementation("androidx.compose.ui:ui-tooling")
  debugImplementation("androidx.compose.ui:ui-test-manifest")
}
