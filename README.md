# ShaderCrab

A safe, modern desktop application for running Shadertoy-compatible shaders, written in Rust.

## Features

- **100% Safe Rust** - Completely safe code with no unsafe blocks
- **GLSL Support** - Write shaders using familiar GLSL syntax (automatically transpiled to WGSL)
- **Sound Shader Support** - Parse and execute GLSL mainSound() for real-time audio generation
- **File Picker** - Press 'O' to open and load shader files interactively
- **All Shadertoy Uniforms** - Full support for the Shadertoy specification
- **Texture Channels** - Support for iChannel0-3 texture inputs
- **Error Handling** - Comprehensive error messages using thiserror
- **Unit Tests** - 39 passing tests with comprehensive coverage
- **Cross-Platform** - Works on Windows, macOS, and Linux using wgpu

## Supported Uniforms

All standard Shadertoy uniforms are supported:

- `vec3 iResolution` - Viewport resolution (width, height, aspect ratio)
- `float iTime` - Shader playback time in seconds
- `float iTimeDelta` - Render time for last frame
- `float iFrame` - Current frame number
- `vec4 iMouse` - Mouse position (xy) and click position (zw)
- `vec4 iDate` - Current date (year, month, day, time in seconds)
- `float iSampleRate` - Audio sample rate (44100 Hz)
- `vec3 iChannelResolution[4]` - Resolution of each texture channel
- `sampler2D iChannel0-3` - Input texture channels

## Usage

### Running the Application

```bash
cargo run
```

### Controls

- **Mouse** - Move and click to interact with shaders
- **O Key** - Open file picker to load a GLSL shader (opens to examples/ folder)
- **Esc Key** - Exit application
- **Close Window** - Exit application

### Writing Shaders

Create a shader file with the `.glsl` extension implementing the `mainImage` function:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    fragColor = vec4(col, 1.0);
}
```

Place your shader files in the `examples/` folder or load them from anywhere using the file picker.

The application will automatically:

1. Load your GLSL code
2. Wrap it with Shadertoy-compatible uniforms
3. Transpile it to WGSL using naga
4. Compile and run it with wgpu

### Sound Shaders

Sound shaders generate audio output by implementing the `mainSound` function, which returns stereo audio samples:

```glsl
vec2 mainSound(float time) {
    float wave = sin(time * 440.0 * 2.0 * 3.14159);
    return vec2(wave, wave);  // stereo output (left, right)
}
```

The function receives:

- `time` - Current time in seconds

And returns:

- `vec2` - Stereo audio samples in range [-1.0, 1.0] (left, right channels)

**Note**: Sound shaders can also access audio input from the microphone via texture channels (iChannel0-3) when those are configured as audio inputs, though the current implementation focuses on audio output generation.

## Architecture

### Modules

- **error.rs** - Type-safe error handling with thiserror
- **uniforms.rs** - Shadertoy uniform structure with unit tests
- **glsl_transpiler.rs** - GLSL to WGSL transpilation using naga
- **sound_shader.rs** - GLSL mainSound() parser and executor for audio generation
- **shader_detector.rs** - Automatic detection of shader type (Image vs Sound)
- **texture.rs** - Texture loading and management
- **audio.rs** - Audio input/output using cpal
- **main.rs** - Application entry point and event loop

### Safety

The codebase is 100% safe Rust:

- Zero unsafe blocks
- No raw pointer manipulation
- No manual memory management
- Type-safe shader uniform binding
- Automatic resource cleanup via RAII
- Uses `Arc<Window>` for safe window reference sharing

### Graphics Backend

Uses wgpu for cross-platform graphics:
- Vulkan on Linux
- Metal on macOS
- DirectX 12 on Windows
- WebGPU when compiled to WASM

## Running Tests

```bash
cargo test
```

Tests cover:
- Uniform initialization
- Uniform updates (time, frame, mouse, resolution, date)
- Channel resolution setting
- Memory alignment
- GLSL transpilation

## Example Shaders

### Animated Pattern
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);

    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    float pulse = 0.5 + 0.5 * sin(iTime * 2.0 - dist * 10.0);
    col *= pulse;

    fragColor = vec4(col, 1.0);
}
```

### Mouse Interaction
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    if (iMouse.z > 0.0) {
        vec2 mouseUV = iMouse.xy / iResolution.xy;
        float dist = distance(uv, mouseUV);
        if (dist < 0.1) {
            fragColor = vec4(1.0);
            return;
        }
    }

    fragColor = vec4(uv, 0.5, 1.0);
}
```

## Dependencies

- **wgpu** - Modern, safe graphics API
- **winit** - Cross-platform window management
- **naga** - Shader transpilation (GLSL → WGSL)
- **thiserror** - Ergonomic error handling
- **rfd** - Native file dialogs
- **cpal** - Cross-platform audio I/O
- **image** - Image loading for texture channels
- **chrono** - Date and time for iDate uniform
- **bytemuck** - Safe type casting for GPU buffers

## License

See LICENSE file for details.
