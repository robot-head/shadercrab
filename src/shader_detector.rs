use crate::error::{Result, ShaderCrabError};
use crate::glsl_transpiler::ShaderType;

pub fn detect_shader_type(source: &str) -> Result<ShaderType> {
    let has_main_image = source.contains("mainImage");
    let has_main_sound = source.contains("mainSound");

    match (has_main_image, has_main_sound) {
        (true, false) => Ok(ShaderType::Image),
        (false, true) => Ok(ShaderType::Sound),
        (true, true) => Err(ShaderCrabError::InvalidShaderType),
        (false, false) => Err(ShaderCrabError::InvalidShaderType),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_image_shader() {
        let shader = "void mainImage(out vec4 fragColor, in vec2 fragCoord) {}";
        assert!(matches!(
            detect_shader_type(shader).unwrap(),
            ShaderType::Image
        ));
    }

    #[test]
    fn test_detect_sound_shader() {
        let shader = "vec2 mainSound(float time) { return vec2(0.0); }";
        assert!(matches!(
            detect_shader_type(shader).unwrap(),
            ShaderType::Sound
        ));
    }

    #[test]
    fn test_detect_both_fails() {
        let shader = "void mainImage() {} vec2 mainSound() {}";
        assert!(detect_shader_type(shader).is_err());
    }

    #[test]
    fn test_detect_neither_fails() {
        let shader = "void someOtherFunction() {}";
        assert!(detect_shader_type(shader).is_err());
    }
}
