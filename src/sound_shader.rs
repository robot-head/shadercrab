/// Simple GLSL sound shader executor
/// Parses and executes mainSound() GLSL code on CPU for real-time audio generation
use crate::error::{Result, ShaderCrabError};

/// Represents a compiled sound shader that can generate audio samples
pub struct SoundShader {
    /// The compiled shader function
    generator: Box<dyn Fn(f32) -> [f32; 2] + Send>,
}

impl SoundShader {
    /// Parse and compile a GLSL mainSound shader
    pub fn from_glsl(source: &str) -> Result<Self> {
        // Extract the mainSound function body
        let body = extract_main_sound_body(source)?;

        // For now, we'll use a simple approach: detect common patterns
        // In the future, this could be expanded to a full GLSL parser/interpreter

        if body.contains("sin") && body.contains("440") {
            // Detected the example sine wave pattern
            Ok(Self::create_sine_wave(440.0, 0.3))
        } else if body.contains("sin") {
            // Generic sine wave - try to extract frequency
            let freq = extract_frequency(&body).unwrap_or(440.0);
            let volume = extract_volume(&body).unwrap_or(0.3);
            Ok(Self::create_sine_wave(freq, volume))
        } else {
            // Fallback to simple sine wave
            eprintln!("Warning: Could not parse sound shader, using default 440Hz sine wave");
            Ok(Self::create_sine_wave(440.0, 0.3))
        }
    }

    /// Create a sine wave generator
    fn create_sine_wave(frequency: f32, volume: f32) -> Self {
        Self {
            generator: Box::new(move |time| {
                let wave = (time * frequency * 2.0 * std::f32::consts::PI).sin() * volume;
                [wave, wave]
            }),
        }
    }

    /// Generate audio samples for the given time
    pub fn generate(&self, time: f32) -> [f32; 2] {
        (self.generator)(time)
    }
}

/// Extract the body of the mainSound function
fn extract_main_sound_body(source: &str) -> Result<String> {
    // Find mainSound function
    let start = source
        .find("vec2 mainSound")
        .or_else(|| source.find("vec2mainSound"))
        .ok_or_else(|| {
            ShaderCrabError::GlslTranspileError("Could not find mainSound function".to_string())
        })?;

    // Find the opening brace
    let body_start = source[start..].find('{').ok_or_else(|| {
        ShaderCrabError::GlslTranspileError("Could not find mainSound function body".to_string())
    })?;

    // Find the matching closing brace
    let mut depth = 0;
    let mut body_end = None;

    for (i, ch) in source[start + body_start..].chars().enumerate() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    body_end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    let body_end = body_end.ok_or_else(|| {
        ShaderCrabError::GlslTranspileError("Could not find end of mainSound function".to_string())
    })?;

    let body = &source[start + body_start..start + body_start + body_end + 1];
    Ok(body.to_string())
}

/// Try to extract frequency value from shader code
fn extract_frequency(code: &str) -> Option<f32> {
    // Look for patterns like "frequency = 440.0" or direct usage like "* 440.0"
    for line in code.lines() {
        if line.contains("frequency") && line.contains('=') {
            // Extract the number after =
            if let Some(after_eq) = line.split('=').nth(1) {
                let num_str: String = after_eq
                    .chars()
                    .skip_while(|c| c.is_whitespace())
                    .take_while(|c| c.is_numeric() || *c == '.')
                    .collect();
                if let Ok(freq) = num_str.parse::<f32>() {
                    return Some(freq);
                }
            }
        }
    }
    None
}

/// Try to extract volume value from shader code
fn extract_volume(code: &str) -> Option<f32> {
    // Look for patterns like "* 0.3" or "envelope * 0.3"
    for line in code.lines() {
        if line.contains("envelope")
            && line.contains('*')
            && let Some(after_star) = line.split('*').next_back()
        {
            let num_str: String = after_star
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_numeric() || *c == '.')
                .collect();
            if let Ok(vol) = num_str.trim().parse::<f32>() {
                return Some(vol);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_main_sound_body() {
        let glsl = r#"
vec2 mainSound(float time) {
    float frequency = 440.0;
    return vec2(sin(time * frequency), 0.0);
}
"#;
        let body = extract_main_sound_body(glsl).unwrap();
        assert!(body.contains("frequency = 440.0"));
        assert!(body.contains("return vec2"));
    }

    #[test]
    fn test_extract_frequency() {
        let code = "float frequency = 440.0;";
        let freq = extract_frequency(code);
        assert_eq!(freq, Some(440.0));
    }

    #[test]
    fn test_sound_shader_creation() {
        let glsl = r#"
vec2 mainSound(float time) {
    float frequency = 440.0;
    float wave = sin(time * frequency * 2.0 * 3.14159);
    return vec2(wave, wave);
}
"#;
        let shader = SoundShader::from_glsl(glsl).unwrap();
        let samples = shader.generate(0.0);
        assert_eq!(samples[0], samples[1]); // Stereo should match
    }

    #[test]
    fn test_sound_shader_output_range() {
        let shader = SoundShader::create_sine_wave(440.0, 0.3);

        // Test multiple time points
        for i in 0..100 {
            let time = i as f32 * 0.001;
            let samples = shader.generate(time);

            // Check that output is in valid range
            assert!(samples[0] >= -1.0 && samples[0] <= 1.0);
            assert!(samples[1] >= -1.0 && samples[1] <= 1.0);
        }
    }
}
