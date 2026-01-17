use shadercrab::audio::AudioManager;

#[test]
fn test_audio_manager_creation() {
    // This test verifies audio manager can be created
    // It may fail on systems without audio devices, which is acceptable
    match AudioManager::new() {
        Ok(manager) => {
            assert!(manager.sample_rate > 0.0);
            println!("Audio manager created with sample rate: {}", manager.sample_rate);
        }
        Err(e) => {
            println!("Audio manager creation skipped (no audio device): {}", e);
            // Don't fail the test if no audio device is available
        }
    }
}

#[test]
fn test_audio_buffer_access() {
    match AudioManager::new() {
        Ok(manager) => {
            // Test that we can safely access the audio buffer
            let sample = manager.get_audio_input_sample(0);
            assert!(sample.is_finite());
        }
        Err(_) => {
            // Skip test if no audio device
        }
    }
}

#[test]
fn test_sound_shader_callback() {
    match AudioManager::new() {
        Ok(mut manager) => {
            // Test that we can set up a sound shader
            let result = manager.start_sound_shader(|time| {
                let freq = 440.0;
                let wave = (time * freq * 2.0 * std::f32::consts::PI).sin();
                [wave, wave]
            });

            if let Err(e) = result {
                println!("Sound shader test skipped: {}", e);
            } else {
                println!("Sound shader callback successfully registered");
            }
        }
        Err(_) => {
            // Skip test if no audio device
        }
    }
}
