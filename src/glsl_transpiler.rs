use crate::error::{Result, ShaderCrabError};
use naga::front::glsl::{Frontend, Options};
use naga::valid::{Capabilities, ValidationFlags, Validator};

pub enum ShaderType {
    Image,
    Sound,
}

pub fn transpile_glsl_to_wgsl(glsl_source: &str, shader_type: ShaderType) -> Result<String> {
    let shader_stage = match shader_type {
        ShaderType::Image => naga::ShaderStage::Fragment,
        ShaderType::Sound => naga::ShaderStage::Fragment,
    };

    let options = Options::from(shader_stage);
    let mut frontend = Frontend::default();

    let module = frontend
        .parse(&options, glsl_source)
        .map_err(|errors| {
            let error_msg = errors
                .errors
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<_>>()
                .join("\n");
            ShaderCrabError::GlslTranspileError(error_msg)
        })?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let module_info = validator.validate(&module).map_err(|e| {
        ShaderCrabError::GlslTranspileError(format!("Validation error: {}", e))
    })?;

    let wgsl = naga::back::wgsl::write_string(
        &module,
        &module_info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|e| ShaderCrabError::GlslTranspileError(format!("WGSL generation error: {}", e)))?;

    Ok(wgsl)
}

pub fn wrap_glsl_shader(user_code: &str, shader_type: ShaderType) -> String {
    match shader_type {
        ShaderType::Image => format!(
            r#"#version 450

layout(set = 0, binding = 0) uniform Uniforms {{
    vec3 iResolution;
    float iTime;
    float iTimeDelta;
    float iFrame;
    vec4 iMouse;
    vec4 iDate;
    float iSampleRate;
    vec3 iChannelResolution[4];
}};

layout(set = 0, binding = 1) uniform sampler iChannelSampler;
layout(set = 0, binding = 2) uniform texture2D iChannel0Tex;
layout(set = 0, binding = 3) uniform texture2D iChannel1Tex;
layout(set = 0, binding = 4) uniform texture2D iChannel2Tex;
layout(set = 0, binding = 5) uniform texture2D iChannel3Tex;

#define iChannel0 sampler2D(iChannel0Tex, iChannelSampler)
#define iChannel1 sampler2D(iChannel1Tex, iChannelSampler)
#define iChannel2 sampler2D(iChannel2Tex, iChannelSampler)
#define iChannel3 sampler2D(iChannel3Tex, iChannelSampler)

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 outColor;

{}

void main() {{
    mainImage(outColor, fragCoord);
}}
"#,
            user_code
        ),
        ShaderType::Sound => format!(
            r#"#version 450

layout(set = 0, binding = 0) uniform Uniforms {{
    vec3 iResolution;
    float iTime;
    float iTimeDelta;
    float iFrame;
    vec4 iMouse;
    vec4 iDate;
    float iSampleRate;
    vec3 iChannelResolution[4];
}};

layout(location = 0) out vec2 outAudio;

{}

void main() {{
    outAudio = mainSound(iTime);
}}
"#,
            user_code
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpile_simple_shader() {
        let glsl = r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv, 0.0, 1.0);
}
"#;
        let wrapped = wrap_glsl_shader(glsl, ShaderType::Image);
        let result = transpile_glsl_to_wgsl(&wrapped, ShaderType::Image);
        assert!(result.is_ok(), "Failed to transpile shader: {:?}", result);
    }
}
