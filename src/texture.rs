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
