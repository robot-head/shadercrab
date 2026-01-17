use crate::error::{Result, ShaderCrabError};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::{Arc, Mutex};

pub struct AudioManager {
    output_device: Device,
    output_stream: Option<Stream>,
    input_device: Option<Device>,
    input_stream: Option<Stream>,
    pub sample_rate: f32,
    pub audio_buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioManager {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();

        let output_device = host
            .default_output_device()
            .ok_or_else(|| ShaderCrabError::AudioDeviceError("No output device found".into()))?;

        let config = output_device
            .default_output_config()
            .map_err(|e| ShaderCrabError::AudioDeviceError(format!("Config error: {}", e)))?;

        let sample_rate = config.sample_rate().0 as f32;
        let audio_buffer = Arc::new(Mutex::new(Vec::new()));

        let input_device = host.default_input_device();
        let input_stream = if let Some(ref device) = input_device {
            let input_config = device.default_input_config().ok();
            input_config.and_then(|cfg| {
                let buffer_clone = audio_buffer.clone();
                let stream = device
                    .build_input_stream(
                        &cfg.into(),
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            if let Ok(mut buf) = buffer_clone.lock() {
                                buf.clear();
                                buf.extend_from_slice(data);
                            }
                        },
                        |err| eprintln!("Audio input error: {}", err),
                        None,
                    )
                    .ok();

                if let Some(ref s) = stream {
                    let _ = s.play();
                }
                stream
            })
        } else {
            None
        };

        Ok(Self {
            output_device,
            output_stream: None,
            input_device,
            input_stream,
            sample_rate,
            audio_buffer,
        })
    }

    pub fn start_sound_shader<F>(&mut self, mut audio_callback: F) -> Result<()>
    where
        F: FnMut(f32) -> [f32; 2] + Send + 'static,
    {
        let config: StreamConfig = self
            .output_device
            .default_output_config()
            .map_err(|e| ShaderCrabError::AudioDeviceError(format!("Config error: {}", e)))?
            .into();

        let sample_rate = config.sample_rate.0 as f32;
        let mut time = 0.0;

        let stream = self
            .output_device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    for frame in data.chunks_mut(2) {
                        let output = audio_callback(time);
                        frame[0] = output[0].clamp(-1.0, 1.0);
                        if frame.len() > 1 {
                            frame[1] = output[1].clamp(-1.0, 1.0);
                        }
                        time += 1.0 / sample_rate;
                    }
                },
                |err| eprintln!("Audio output error: {}", err),
                None,
            )
            .map_err(|e| ShaderCrabError::AudioStreamError(format!("Stream error: {}", e)))?;

        stream
            .play()
            .map_err(|e| ShaderCrabError::AudioStreamError(format!("Play error: {}", e)))?;

        self.output_stream = Some(stream);
        Ok(())
    }

    pub fn get_audio_input_sample(&self, index: usize) -> f32 {
        if let Ok(buffer) = self.audio_buffer.lock() {
            buffer.get(index).copied().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn has_audio_input(&self) -> bool {
        self.input_device.is_some() && self.input_stream.is_some()
    }

    pub fn input_device_name(&self) -> Option<String> {
        self.input_device.as_ref().and_then(|d| d.name().ok())
    }
}

impl Default for AudioManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            eprintln!("Warning: Failed to initialize audio, using dummy audio manager");
            Self {
                output_device: cpal::default_host()
                    .default_output_device()
                    .expect("No audio device"),
                output_stream: None,
                input_device: None,
                input_stream: None,
                sample_rate: 44100.0,
                audio_buffer: Arc::new(Mutex::new(Vec::new())),
            }
        })
    }
}
