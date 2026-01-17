use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShaderCrabError {
    #[error("Failed to load shader file: {0}")]
    ShaderFileLoadError(#[from] std::io::Error),

    #[error("Failed to transpile GLSL shader: {0}")]
    GlslTranspileError(String),

    #[error("Failed to create shader module: {0}")]
    ShaderModuleError(String),

    #[error("Failed to load texture from file '{path}': {source}")]
    TextureLoadError {
        path: String,
        source: image::ImageError,
    },

    #[error("Audio device error: {0}")]
    AudioDeviceError(String),

    #[error("Audio stream error: {0}")]
    AudioStreamError(String),

    #[error("Surface error: {0}")]
    SurfaceError(#[from] wgpu::SurfaceError),

    #[error("Failed to create surface: {0}")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),

    #[error("Invalid shader type, expected mainImage or mainSound function")]
    InvalidShaderType,

    #[error("Failed to create WGPU adapter")]
    NoAdapter,

    #[error("Failed to request WGPU device: {0}")]
    DeviceRequestError(String),
}

pub type Result<T> = std::result::Result<T, ShaderCrabError>;
