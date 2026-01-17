fn mainImage(frag_coord: vec2<f32>) -> vec4<f32> {
    let uv = frag_coord / uniforms.resolution.xy;

    let center = vec2<f32>(0.5, 0.5);
    let dist = distance(uv, center);

    var col = 0.5 + 0.5 * cos(uniforms.time + vec3<f32>(uv.x, uv.y, uv.x) + vec3<f32>(0.0, 2.0, 4.0));

    let pulse = 0.5 + 0.5 * sin(uniforms.time * 2.0 - dist * 10.0);
    col = col * pulse;

    if uniforms.mouse.z > 0.0 {
        let mouse_uv = uniforms.mouse.xy / uniforms.resolution.xy;
        let mouse_dist = distance(uv, mouse_uv);
        if mouse_dist < 0.1 {
            col = mix(col, vec3<f32>(1.0, 1.0, 1.0), 1.0 - mouse_dist / 0.1);
        }
    }

    return vec4<f32>(col, 1.0);
}
