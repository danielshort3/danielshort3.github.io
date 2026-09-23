plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
  id("org.jetbrains.kotlin.plugin.compose")
}

val productionCatalogUrl = "https://www.danielshort.me/app-content/v1/catalog.json"
val previewCatalogUrl = providers.gradleProperty("catalogUrl").orElse(productionCatalogUrl)
val stableAppUpdateUrl = "https://www.danielshort.me/app-updates/stable/latest.json"
val reviewAppUpdateUrl = "https://www.danielshort.me/app-updates/review/latest.json"
val websiteRoot = rootProject.projectDir.resolve("../..").canonicalFile
val generatedCatalogAssets = layout.buildDirectory.dir("generated/catalogAssets")

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
    versionCode = 7
    versionName = "0.5.0"
    buildConfigField("String", "CONTENT_URL", "\"$productionCatalogUrl\"")
    buildConfigField("String", "APP_UPDATE_URL", "\"$stableAppUpdateUrl\"")
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  buildTypes {
    debug {
      applicationIdSuffix = ".debug"
      versionNameSuffix = "-debug"
      buildConfigField("String", "APP_UPDATE_URL", "\"$reviewAppUpdateUrl\"")
      buildConfigField("String", "CONTENT_URL", "\"${previewCatalogUrl.get().replace("\\", "\\\\").replace("\"", "\\\"")}\"")
    }
    release {
      isMinifyEnabled = true
      isShrinkResources = true
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
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
  sourceSets["main"].assets.srcDir(generatedCatalogAssets)
  packaging.resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
}

kotlin {
  compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

tasks.named("preBuild") { dependsOn(bundleCatalog) }

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
