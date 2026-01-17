// Example sound shader - A simple 440Hz sine wave tone
// To use: rename this file to shader.glsl or load it via the file picker

vec2 mainSound(float time) {
    // Generate a 440Hz sine wave (A4 note)
    float frequency = 440.0;
    float wave = sin(time * frequency * 2.0 * 3.14159);

    // Apply a simple envelope to prevent clicks
    float envelope = smoothstep(0.0, 0.01, time) * smoothstep(0.0, 0.01, 1.0 - mod(time, 1.0));
    wave *= envelope * 0.3;  // Reduce volume to 30%

    // Return stereo output (left, right)
    return vec2(wave, wave);
}
