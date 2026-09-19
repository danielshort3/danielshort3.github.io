package me.danielshort.app.nativefeatures.games

import kotlin.math.*

/** Native finite-wave surface using the website's deep-water dispersion relation.
 * This is a mobile Canvas renderer, not the website's WebGL/FFT Ultra renderer. */
object OceanModel {
  fun height(x: Float, z: Float, time: Float, wind: Float, amplitude: Float): Float {
    if (listOf(x, z, time, wind, amplitude).any { !it.isFinite() }) return 0f
    val strength = amplitude.coerceIn(0f, 3f)
    return (0..5).sumOf { index ->
      val k = 0.12 + index * 0.091
      val direction = 1.12 + sin(index * 2.4) * .8
      val phase = k * (x * cos(direction) + z * sin(direction)) - sqrt(9.81 * k) * time * (.5 + wind.coerceIn(0f, 20f) / 15)
      sin(phase) * strength / (index + 1).toDouble().pow(1.3)
    }.toFloat()
  }
}
