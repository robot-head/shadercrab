void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  
    vec2 uv = fragCoord/iResolution.xy;

    vec3 bgColor = texture(iChannel1, uv).rgb;
    bgColor *= 0.4;

    float rawAudio = texture(iChannel0, vec2(uv.x, uv.y)).x;
    
    if (rawAudio < 0.2) {
        rawAudio = 0.0;
    }
    // ----------------------------------------------------

    vec3 fgColor = vec3(0.0);
    
    if (rawAudio > uv.y + 0.5) {
        fgColor = vec3(1.0);
    }

    if (mod(uv.x * 31., 1.) < .1) {
        fgColor = vec3(0.);
    }

    vec3 timeVary = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));
    if (fgColor.x > 0.5) { 
        fgColor *= timeVary * vec3(0.557,0.582,0.518);
    }

    vec3 finalColor = vec3(1.0) - (vec3(1.0) - bgColor) * (vec3(1.0) - fgColor);

    fragColor = vec4(finalColor, 1.0);
}