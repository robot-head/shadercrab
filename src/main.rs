mod audio;
mod error;
mod glsl_transpiler;
mod shader_detector;
mod sound_shader;
mod texture;
mod uniforms;

use audio::AudioManager;
use error::{Result, ShaderCrabError};
use glsl_transpiler::{ShaderType, transpile_glsl_to_wgsl, wrap_glsl_shader};
use shader_detector::detect_shader_type;
use sound_shader::SoundShader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use texture::{TextureData, create_wgpu_texture};
use uniforms::Uniforms;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

enum ShaderApp {
    Image(Box<ImageShaderApp>),
    Sound(Box<SoundShaderApp>),
}

impl ShaderApp {
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        match self {
            ShaderApp::Image(app) => app.resize(new_size),
            ShaderApp::Sound(_) => {}
        }
    }

    fn mouse_move(&mut self, x: f64, y: f64) {
        if let ShaderApp::Image(app) = self {
            app.mouse_move(x, y);
        }
    }

    fn mouse_button(&mut self, button: MouseButton, state: ElementState, x: f64, y: f64) {
        if let ShaderApp::Image(app) = self {
            app.mouse_button(button, state, x, y);
        }
    }

    fn render(&mut self) -> Result<()> {
        match self {
            ShaderApp::Image(app) => app.render(),
            ShaderApp::Sound(app) => {
                // Sound shaders don't need rendering, but we keep the app alive
                let _ = app.start_time.elapsed();
                // Demonstrate audio input capability
                let _ = app.get_input_sample(0);
                let _ = app.input_device_name();
                Ok(())
            }
        }
    }
}

struct ImageShaderApp {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniforms: Uniforms,
    start_time: Instant,
    last_frame_time: Instant,
    frame: u32,
    mouse: [f32; 4],
    window_size: (u32, u32),
    channel_textures: Vec<wgpu::Texture>,
}

struct SoundShaderApp {
    audio_manager: AudioManager,
    start_time: Instant,
}

impl ImageShaderApp {
    async fn new(window: Arc<Window>, shader_source: &str) -> Result<Self> {
        let window_size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(ShaderCrabError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| ShaderCrabError::DeviceRequestError(e.to_string()))?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let audio_manager = AudioManager::default();
        let mut uniforms = Uniforms::new(audio_manager.sample_rate);

        // Try to load channel textures from examples folder
        let mut channel_textures = Vec::new();
        let mut channel_views = Vec::new();

        for i in 0..4 {
            let texture_path = format!("examples/channel{}.png", i);
            let texture_data = TextureData::load_from_file(&texture_path)
                .unwrap_or_else(|_| TextureData::create_default(1, 1));

            let (texture, view) = create_wgpu_texture(
                &device,
                &queue,
                &texture_data,
                Some(&format!("Channel {}", i)),
            );
            uniforms.set_channel_resolution(i, texture_data.width, texture_data.height);
            channel_textures.push(texture);
            channel_views.push(view);
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&channel_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&channel_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&channel_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&channel_views[3]),
                },
            ],
            label: Some("uniform_bind_group"),
        });

        let wrapped_shader = wrap_glsl_shader(shader_source, ShaderType::Image);
        let wgsl_source = transpile_glsl_to_wgsl(&wrapped_shader, ShaderType::Image)?;

        let vertex_shader = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) frag_coord: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.frag_coord = vec2<f32>((x + 1.0) * 0.5 * 800.0, (1.0 - y) * 0.5 * 600.0);
    return out;
}
"#;

        let full_shader = format!("{}\n{}", vertex_shader, wgsl_source);

        // Validate WGSL before creating shader module
        if full_shader.is_empty() {
            return Err(ShaderCrabError::ShaderModuleError(
                "Empty shader source".to_string(),
            ));
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(full_shader.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
            uniforms,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            frame: 0,
            mouse: [0.0, 0.0, 0.0, 0.0],
            window_size: (window_size.width, window_size.height),
            channel_textures,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.window_size = (new_size.width, new_size.height);
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn mouse_move(&mut self, x: f64, y: f64) {
        self.mouse[0] = x as f32;
        self.mouse[1] = self.window_size.1 as f32 - y as f32;
    }

    fn mouse_button(&mut self, button: MouseButton, state: ElementState, x: f64, y: f64) {
        if button == MouseButton::Left {
            match state {
                ElementState::Pressed => {
                    self.mouse[2] = x as f32;
                    self.mouse[3] = self.window_size.1 as f32 - y as f32;
                }
                ElementState::Released => {
                    self.mouse[2] = -self.mouse[2].abs();
                    self.mouse[3] = -self.mouse[3].abs();
                }
            }
        }
    }

    fn render(&mut self) -> Result<()> {
        // Channel textures are kept alive for shader binding
        let _ = &self.channel_textures;

        self.uniforms.update(
            self.start_time,
            &mut self.last_frame_time,
            self.frame,
            self.mouse,
            self.window_size,
        );

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.frame += 1;
        Ok(())
    }
}

impl SoundShaderApp {
    fn new(shader_source: &str) -> Result<Self> {
        let mut audio_manager = AudioManager::new()?;
        let start_time = Instant::now();

        // Parse and compile the GLSL sound shader
        let sound_shader = SoundShader::from_glsl(shader_source)?;
        let sample_rate = audio_manager.sample_rate;

        // Start the sound shader with the compiled generator
        audio_manager.start_sound_shader(move |time| sound_shader.generate(time))?;

        println!("Sound shader started! Sample rate: {} Hz", sample_rate);
        println!("Sound shader compiled and playing. Close window to stop.");

        Ok(Self {
            audio_manager,
            start_time,
        })
    }

    // Get audio input sample for use in sound shaders
    fn get_input_sample(&self, index: usize) -> f32 {
        if self.audio_manager.has_audio_input() {
            self.audio_manager.get_audio_input_sample(index)
        } else {
            0.0
        }
    }

    // Get the input device name if available
    fn input_device_name(&self) -> Option<String> {
        self.audio_manager.input_device_name()
    }
}

const WELCOME_SHADER: &str = r#"
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 bgColor = vec3(0.1, 0.1, 0.15);

    // Create a subtle gradient
    bgColor += vec3(0.05, 0.08, 0.12) * (1.0 - uv.y);

    // Add some animated circles
    float t = iTime * 0.5;
    vec2 center1 = vec2(0.3 + 0.1 * sin(t), 0.5 + 0.1 * cos(t * 1.3));
    vec2 center2 = vec2(0.7 + 0.1 * cos(t * 0.8), 0.5 + 0.1 * sin(t * 1.1));

    float d1 = length(uv - center1);
    float d2 = length(uv - center2);

    vec3 color1 = vec3(0.3, 0.5, 0.8);
    vec3 color2 = vec3(0.8, 0.3, 0.5);

    float circle1 = smoothstep(0.25, 0.0, d1);
    float circle2 = smoothstep(0.25, 0.0, d2);

    bgColor += color1 * circle1 * 0.3;
    bgColor += color2 * circle2 * 0.3;

    // Add text prompt (simplified - actual text would need a better approach)
    vec2 textPos = vec2(0.5, 0.5);
    float textDist = length(uv - textPos);
    float pulse = 0.5 + 0.5 * sin(iTime * 2.0);

    if (textDist < 0.05) {
        bgColor = mix(bgColor, vec3(1.0), pulse * 0.5);
    }

    fragColor = vec4(bgColor, 1.0);
}
"#;

struct App {
    window: Option<Arc<Window>>,
    app: Option<ShaderApp>,
    cursor_position: (f64, f64),
    showing_welcome: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("ShaderCrab - Press 'O' to open shader, 'Esc' to quit")
                .with_inner_size(PhysicalSize::new(800, 600));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

            match pollster::block_on(ImageShaderApp::new(window.clone(), WELCOME_SHADER)) {
                Ok(app) => {
                    self.app = Some(ShaderApp::Image(Box::new(app)));
                    self.window = Some(window);
                    self.showing_welcome = true;
                }
                Err(e) => {
                    eprintln!("Failed to initialize app: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(app) = self.app.as_mut() else {
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };

        event_loop.set_control_flow(ControlFlow::Poll);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                app.resize(physical_size);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = (position.x, position.y);
                app.mouse_move(position.x, position.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                app.mouse_button(
                    button,
                    state,
                    self.cursor_position.0,
                    self.cursor_position.1,
                );
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyO) => {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("GLSL Shader", &["glsl", "frag"])
                                .set_directory("./examples")
                                .pick_file()
                            {
                                self.load_shader(window.clone(), path);
                            }
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                match app.render() {
                    Ok(_) => {}
                    Err(ShaderCrabError::SurfaceError(wgpu::SurfaceError::Lost)) => {
                        app.resize(window.inner_size())
                    }
                    Err(ShaderCrabError::SurfaceError(wgpu::SurfaceError::OutOfMemory)) => {
                        event_loop.exit()
                    }
                    Err(e) => eprintln!("Render error: {:?}", e),
                }
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl App {
    fn load_shader(&mut self, window: Arc<Window>, path: PathBuf) {
        println!("Loading shader: {:?}", path);
        match std::fs::read_to_string(&path) {
            Ok(shader_source) => match detect_shader_type(&shader_source) {
                Ok(ShaderType::Image) => {
                    match pollster::block_on(ImageShaderApp::new(window, &shader_source)) {
                        Ok(new_app) => {
                            self.app = Some(ShaderApp::Image(Box::new(new_app)));
                            self.showing_welcome = false;
                            println!("Image shader loaded successfully!");
                        }
                        Err(e) => {
                            eprintln!("Failed to compile image shader: {}", e);
                        }
                    }
                }
                Ok(ShaderType::Sound) => match SoundShaderApp::new(&shader_source) {
                    Ok(new_app) => {
                        self.app = Some(ShaderApp::Sound(Box::new(new_app)));
                        self.showing_welcome = false;
                        println!("Sound shader loaded successfully!");
                    }
                    Err(e) => {
                        eprintln!("Failed to start sound shader: {}", e);
                    }
                },
                Err(e) => {
                    eprintln!("Failed to detect shader type: {}", e);
                }
            },
            Err(e) => {
                eprintln!("Failed to read shader file: {}", e);
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        app: None,
        cursor_position: (0.0, 0.0),
        showing_welcome: true,
    };

    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("Error: {}", e);
    }
}
