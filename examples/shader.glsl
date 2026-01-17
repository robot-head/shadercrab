void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);

    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));

    float pulse = 0.5 + 0.5 * sin(iTime * 2.0 - dist * 10.0);
    col *= pulse;

    if (iMouse.z > 0.0) {
        vec2 mouseUV = iMouse.xy / iResolution.xy;
        float mouseDist = distance(uv, mouseUV);
        if (mouseDist < 0.1) {
            col = mix(col, vec3(1.0, 1.0, 1.0), 1.0 - mouseDist / 0.1);
        }
    }

    fragColor = vec4(col, 1.0);
}
