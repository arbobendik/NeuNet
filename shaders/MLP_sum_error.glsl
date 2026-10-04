#version 300 es
precision highp float;
precision highp int;

uniform sampler2D error_sum_tex;

out vec4 error_out;

// Convert 4 bytes, texture channels to usable float.
float to_float(vec4 bytes) {
    ivec4 intBytes = ivec4(bytes * 255.0);
    uint intValue = uint(intBytes.x) | (uint(intBytes.y) << 8) | (uint(intBytes.z) << 16) | (uint(intBytes.w) << 24);
    return uintBitsToFloat(intValue);
}

// Split float into 4 8-bit texture channels.
vec4 to_bytes(float num) {
    uint intValue = floatBitsToUint(num);
    uint byteMask = uint(255);
    vec4 bytes;
    bytes.x = float(intValue & byteMask);
    bytes.y = float((intValue >> 8) & byteMask);
    bytes.z = float((intValue >> 16) & byteMask);
    bytes.w = float((intValue>> 24) & byteMask);
    return round(bytes) / 255.0;
}

// Sum all errors of one.
void main() {
    // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
    int row = int(gl_FragCoord.y);
    int column = int(gl_FragCoord.x);

    float sum = 0.0;

    for (int i = 0; i < textureSize(error_sum_tex, 0).y; i++) {
    vec4 error_texel = texelFetch(error_sum_tex, ivec2(column + 1, i), 0);
    // Sum up all values.
    sum += to_float(error_texel);
    }

    error_out = to_bytes(sum);
}