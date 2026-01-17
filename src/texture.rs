use crate::error::{Result, ShaderCrabError};
use image::GenericImageView;
use std::path::Path;

pub struct TextureData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl TextureData {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let img = image::open(path_ref).map_err(|e| ShaderCrabError::TextureLoadError {
            path: path_ref.display().to_string(),
            source: e,
        })?;

        let rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        Ok(Self {
            data: rgba.into_raw(),
            width,
            height,
        })
    }

    pub fn create_default(width: u32, height: u32) -> Self {
        let data = vec![0u8; (width * height * 4) as usize];
        Self {
            data,
            width,
            height,
        }
    }
}

pub fn create_wgpu_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_data: &TextureData,
    label: Option<&str>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: texture_data.width,
        height: texture_data.height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texture_data.data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * texture_data.width),
            rows_per_image: Some(texture_data.height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Audio texture width (number of samples in waveform)
pub const AUDIO_TEXTURE_WIDTH: u32 = 512;
/// Audio texture height (row 0 = waveform, row 1 = FFT)
pub const AUDIO_TEXTURE_HEIGHT: u32 = 2;

/// Create a texture for audio data (512x2, waveform + FFT)
pub fn create_audio_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: Option<&str>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: AUDIO_TEXTURE_WIDTH,
        height: AUDIO_TEXTURE_HEIGHT,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, // Not sRGB for audio data
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Initialize with zeros
    let data = vec![0u8; (AUDIO_TEXTURE_WIDTH * AUDIO_TEXTURE_HEIGHT * 4) as usize];
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * AUDIO_TEXTURE_WIDTH),
            rows_per_image: Some(AUDIO_TEXTURE_HEIGHT),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Update audio texture with waveform data from microphone
pub fn update_audio_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    audio_samples: &[f32],
) {
    // Create RGBA data for the waveform (row 0)
    // Each sample becomes an RGBA pixel where R = waveform value
    let mut data = vec![0u8; (AUDIO_TEXTURE_WIDTH * AUDIO_TEXTURE_HEIGHT * 4) as usize];
    
    // Fill row 0 with waveform data
    let samples_to_use = audio_samples.len().min(AUDIO_TEXTURE_WIDTH as usize);
    
    // Downsample or use directly based on available samples
    if audio_samples.len() >= AUDIO_TEXTURE_WIDTH as usize {
        // Downsample if we have more samples than texture width
        let step = audio_samples.len() as f32 / AUDIO_TEXTURE_WIDTH as f32;
        for i in 0..AUDIO_TEXTURE_WIDTH as usize {
            let sample_idx = (i as f32 * step) as usize;
            let sample = audio_samples.get(sample_idx).copied().unwrap_or(0.0);
            // Convert from [-1, 1] to [0, 255]
            let normalized = ((sample + 1.0) * 0.5).clamp(0.0, 1.0);
            let byte_val = (normalized * 255.0) as u8;
            let pixel_idx = i * 4;
            data[pixel_idx] = byte_val;     // R
            data[pixel_idx + 1] = byte_val; // G
            data[pixel_idx + 2] = byte_val; // B
            data[pixel_idx + 3] = 255;      // A
        }
    } else {
        // Use samples directly, pad with zeros
        for i in 0..samples_to_use {
            let sample = audio_samples[i];
            let normalized = ((sample + 1.0) * 0.5).clamp(0.0, 1.0);
            let byte_val = (normalized * 255.0) as u8;
            let pixel_idx = i * 4;
            data[pixel_idx] = byte_val;
            data[pixel_idx + 1] = byte_val;
            data[pixel_idx + 2] = byte_val;
            data[pixel_idx + 3] = 255;
        }
    }
    
    // Row 1: Simple FFT approximation (magnitude of frequency bins)
    // For a proper implementation, you'd use a real FFT library
    // Here we do a simple approximation based on sample differences
    let row1_offset = (AUDIO_TEXTURE_WIDTH * 4) as usize;
    for i in 0..AUDIO_TEXTURE_WIDTH as usize {
        // Simple "frequency content" approximation
        let freq_bin = if i < samples_to_use.saturating_sub(1) {
            let diff = (audio_samples.get(i + 1).copied().unwrap_or(0.0) 
                       - audio_samples.get(i).copied().unwrap_or(0.0)).abs();
            (diff * 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let byte_val = (freq_bin * 255.0) as u8;
        let pixel_idx = row1_offset + i * 4;
        data[pixel_idx] = byte_val;
        data[pixel_idx + 1] = byte_val;
        data[pixel_idx + 2] = byte_val;
        data[pixel_idx + 3] = 255;
    }
    
    let size = wgpu::Extent3d {
        width: AUDIO_TEXTURE_WIDTH,
        height: AUDIO_TEXTURE_HEIGHT,
        depth_or_array_layers: 1,
    };
    
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * AUDIO_TEXTURE_WIDTH),
            rows_per_image: Some(AUDIO_TEXTURE_HEIGHT),
        },
        size,
    );
}
