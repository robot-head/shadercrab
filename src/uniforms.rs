use chrono::{Datelike, Timelike};
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub resolution: [f32; 3],
    pub time: f32,
    pub time_delta: f32,
    pub frame: f32,
    padding1: [f32; 2],
    pub mouse: [f32; 4],
    pub date: [f32; 4],
    pub sample_rate: f32,
    padding2: [f32; 3],
    pub channel_resolution: [[f32; 3]; 4],
    padding3: [f32; 4],
}

impl Uniforms {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            resolution: [800.0, 600.0, 1.0],
            time: 0.0,
            time_delta: 0.0,
            frame: 0.0,
            padding1: [0.0; 2],
            mouse: [0.0, 0.0, 0.0, 0.0],
            date: [0.0, 0.0, 0.0, 0.0],
            sample_rate,
            padding2: [0.0; 3],
            channel_resolution: [[0.0; 3]; 4],
            padding3: [0.0; 4],
        }
    }

    pub fn update(
        &mut self,
        start_time: Instant,
        last_frame_time: &mut Instant,
        frame: u32,
        mouse: [f32; 4],
        window_size: (u32, u32),
    ) {
        let elapsed = start_time.elapsed().as_secs_f32();
        let delta = last_frame_time.elapsed().as_secs_f32();
        *last_frame_time = Instant::now();

        let now = chrono::Local::now();
        let year = now.year() as f32;
        let month = now.month() as f32 - 1.0;
        let day = now.day() as f32 - 1.0;
        let seconds = now.num_seconds_from_midnight() as f32;

        self.resolution = [window_size.0 as f32, window_size.1 as f32, 1.0];
        self.time = elapsed;
        self.time_delta = delta;
        self.frame = frame as f32;
        self.mouse = mouse;
        self.date = [year, month, day, seconds];
    }

    pub fn set_channel_resolution(&mut self, channel: usize, width: u32, height: u32) {
        if channel < 4 {
            self.channel_resolution[channel] = [width as f32, height as f32, 1.0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_uniforms_new() {
        let uniforms = Uniforms::new(44100.0);
        assert_eq!(uniforms.resolution, [800.0, 600.0, 1.0]);
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.time_delta, 0.0);
        assert_eq!(uniforms.frame, 0.0);
        assert_eq!(uniforms.sample_rate, 44100.0);
        assert_eq!(uniforms.mouse, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_uniforms_update_time() {
        let mut uniforms = Uniforms::new(44100.0);
        let start_time = Instant::now();
        let mut last_frame_time = Instant::now();

        sleep(Duration::from_millis(10));

        uniforms.update(
            start_time,
            &mut last_frame_time,
            0,
            [0.0, 0.0, 0.0, 0.0],
            (800, 600),
        );

        assert!(uniforms.time > 0.0, "Time should have increased");
        assert!(uniforms.time_delta > 0.0, "Time delta should be positive");
    }

    #[test]
    fn test_uniforms_update_frame() {
        let mut uniforms = Uniforms::new(44100.0);
        let start_time = Instant::now();
        let mut last_frame_time = Instant::now();

        uniforms.update(
            start_time,
            &mut last_frame_time,
            42,
            [0.0, 0.0, 0.0, 0.0],
            (800, 600),
        );

        assert_eq!(uniforms.frame, 42.0);
    }

    #[test]
    fn test_uniforms_update_mouse() {
        let mut uniforms = Uniforms::new(44100.0);
        let start_time = Instant::now();
        let mut last_frame_time = Instant::now();

        let mouse_state = [100.0, 200.0, 50.0, 75.0];
        uniforms.update(start_time, &mut last_frame_time, 0, mouse_state, (800, 600));

        assert_eq!(uniforms.mouse, mouse_state);
    }

    #[test]
    fn test_uniforms_update_resolution() {
        let mut uniforms = Uniforms::new(44100.0);
        let start_time = Instant::now();
        let mut last_frame_time = Instant::now();

        uniforms.update(
            start_time,
            &mut last_frame_time,
            0,
            [0.0, 0.0, 0.0, 0.0],
            (1920, 1080),
        );

        assert_eq!(uniforms.resolution, [1920.0, 1080.0, 1.0]);
    }

    #[test]
    fn test_uniforms_update_date() {
        let mut uniforms = Uniforms::new(44100.0);
        let start_time = Instant::now();
        let mut last_frame_time = Instant::now();

        uniforms.update(
            start_time,
            &mut last_frame_time,
            0,
            [0.0, 0.0, 0.0, 0.0],
            (800, 600),
        );

        let now = chrono::Local::now();
        let expected_year = now.year() as f32;
        let expected_month = now.month() as f32 - 1.0;

        assert_eq!(uniforms.date[0], expected_year);
        assert_eq!(uniforms.date[1], expected_month);
    }

    #[test]
    fn test_set_channel_resolution() {
        let mut uniforms = Uniforms::new(44100.0);

        uniforms.set_channel_resolution(0, 512, 512);
        assert_eq!(uniforms.channel_resolution[0], [512.0, 512.0, 1.0]);

        uniforms.set_channel_resolution(3, 1024, 768);
        assert_eq!(uniforms.channel_resolution[3], [1024.0, 768.0, 1.0]);
    }

    #[test]
    fn test_set_channel_resolution_bounds() {
        let mut uniforms = Uniforms::new(44100.0);

        uniforms.set_channel_resolution(4, 512, 512);
        assert_eq!(uniforms.channel_resolution[3], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_uniforms_memory_layout() {
        let size = std::mem::size_of::<Uniforms>();
        assert!(
            size.is_multiple_of(16),
            "Uniforms size should be 16-byte aligned"
        );
    }
}
