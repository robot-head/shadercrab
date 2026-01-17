/// GLSL sound shader interpreter
/// Parses and executes mainSound() GLSL code on CPU for real-time audio generation
use crate::error::{Result, ShaderCrabError};
use crate::glsl_interpreter::GlslInterpreter;
use naga::front::glsl::{Frontend, Options};
use naga::Module;

/// Represents a compiled sound shader that can generate audio samples
pub struct SoundShader {
    /// The parsed Naga module
    module: Module,
}

impl SoundShader {
    /// Parse and compile a GLSL mainSound shader
    pub fn from_glsl(source: &str) -> Result<Self> {
        // Wrap the user code with the necessary boilerplate
        let wrapped = wrap_sound_shader(source);

        // Parse with Naga
        let options = Options::from(naga::ShaderStage::Fragment);
        let mut frontend = Frontend::default();

        let module = frontend.parse(&options, &wrapped).map_err(|errors| {
            let error_msg = errors
                .errors
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join("\n");
            ShaderCrabError::GlslTranspileError(format!("GLSL parse error: {}", error_msg))
        })?;

        // Verify mainSound function exists
        let has_main_sound = module
            .functions
            .iter()
            .any(|(_, f)| f.name.as_deref() == Some("mainSound"));

        if !has_main_sound {
            return Err(ShaderCrabError::GlslTranspileError(
                "mainSound function not found in shader".to_string(),
            ));
        }

        Ok(Self { module })
    }

    /// Generate audio samples for the given time
    pub fn generate(&self, time: f32) -> [f32; 2] {
        let mut interpreter = GlslInterpreter::new(&self.module);
        interpreter.call_main_sound(time)
    }
}

/// Wrap user sound shader code with GLSL boilerplate
fn wrap_sound_shader(user_code: &str) -> String {
    format!(
        r#"#version 450

// Shadertoy-compatible uniforms (available but may not be used)
const float PI = 3.14159265359;

layout(location = 0) out vec2 outAudio;

{}

void main() {{
    // Dummy entry point - we don't actually use this
    // The interpreter directly calls mainSound
    outAudio = vec2(0.0, 0.0);
}}
"#,
        user_code
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_sine_wave() {
        let glsl = r#"
vec2 mainSound(float time) {
    float frequency = 440.0;
    float wave = sin(time * frequency * 2.0 * 3.14159);
    return vec2(wave, wave);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();

        // Test at time 0 - sin(0) = 0
        let samples = shader.generate(0.0);
        assert!(samples[0].abs() < 0.001, "Expected ~0, got {}", samples[0]);

        // Test at time where we expect a peak
        // sin(2*pi*440*t) = 1 when t = 1/(4*440) = 0.000568...
        let t = 1.0 / (4.0 * 440.0);
        let samples = shader.generate(t);
        assert!(
            (samples[0] - 1.0).abs() < 0.01,
            "Expected ~1, got {}",
            samples[0]
        );
    }

    #[test]
    fn test_stereo_output() {
        let glsl = r#"
vec2 mainSound(float time) {
    return vec2(0.5, -0.5);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();
        let samples = shader.generate(0.0);
        assert_eq!(samples[0], 0.5);
        assert_eq!(samples[1], -0.5);
    }

    #[test]
    fn test_with_smoothstep() {
        let glsl = r#"
vec2 mainSound(float time) {
    float envelope = smoothstep(0.0, 0.1, time);
    float wave = sin(time * 440.0 * 2.0 * 3.14159) * envelope;
    return vec2(wave, wave);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();

        // At time 0, envelope should be 0
        let samples = shader.generate(0.0);
        assert!(samples[0].abs() < 0.001);

        // At time 0.1, envelope should be 1
        let samples = shader.generate(0.1);
        // The wave value depends on the phase, just verify it's non-zero scaled
        assert!(samples[0].abs() <= 1.0);
    }

    #[test]
    fn test_with_mod() {
        let glsl = r#"
vec2 mainSound(float time) {
    float t = mod(time, 1.0);
    return vec2(t, t);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();

        let samples = shader.generate(0.5);
        assert!((samples[0] - 0.5).abs() < 0.001);

        let samples = shader.generate(1.5);
        assert!((samples[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_with_mix() {
        let glsl = r#"
vec2 mainSound(float time) {
    float a = 0.0;
    float b = 1.0;
    float result = mix(a, b, 0.5);
    return vec2(result, result);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();
        let samples = shader.generate(0.0);
        assert!((samples[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_with_clamp() {
        let glsl = r#"
vec2 mainSound(float time) {
    float value = clamp(time * 10.0, 0.0, 1.0);
    return vec2(value, value);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();

        let samples = shader.generate(0.0);
        assert_eq!(samples[0], 0.0);

        let samples = shader.generate(0.05);
        assert!((samples[0] - 0.5).abs() < 0.001);

        let samples = shader.generate(1.0);
        assert_eq!(samples[0], 1.0);
    }

    #[test]
    fn test_with_conditionals() {
        let glsl = r#"
vec2 mainSound(float time) {
    float wave;
    if (time < 0.5) {
        wave = 0.25;
    } else {
        wave = 0.75;
    }
    return vec2(wave, wave);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();

        let samples = shader.generate(0.25);
        assert!((samples[0] - 0.25).abs() < 0.001);

        let samples = shader.generate(0.75);
        assert!((samples[0] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_output_range() {
        let glsl = r#"
vec2 mainSound(float time) {
    float frequency = 440.0;
    float wave = sin(time * frequency * 2.0 * 3.14159) * 0.3;
    return vec2(wave, wave);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();

        // Test multiple time points
        for i in 0..100 {
            let time = i as f32 * 0.001;
            let samples = shader.generate(time);

            // Check that output is in valid range
            assert!(samples[0] >= -1.0 && samples[0] <= 1.0);
            assert!(samples[1] >= -1.0 && samples[1] <= 1.0);
        }
    }

    #[test]
    fn test_multiple_oscillators() {
        let glsl = r#"
vec2 mainSound(float time) {
    float freq1 = 440.0;
    float freq2 = 880.0;
    float wave1 = sin(time * freq1 * 6.28318);
    float wave2 = sin(time * freq2 * 6.28318);
    float mixed = (wave1 + wave2) * 0.25;
    return vec2(mixed, mixed);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();
        let samples = shader.generate(0.001);
        // Just verify it runs and produces reasonable output
        assert!(samples[0].abs() <= 1.0);
    }
}
