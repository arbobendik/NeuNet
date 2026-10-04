#version 300 es
precision highp float;
precision highp int;

uniform sampler2D neuron_tex;
uniform sampler2D data_tex;

uniform int is_sigmoid;
uniform int is_tanh;
uniform int is_leaky_relu;
uniform int is_linear;

out vec4 activity_out;

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
    bytes.w = float((intValue >> 24) & byteMask);
    return round(bytes) / 255.0;
}

// Shader to calculate activity of a single neuron.
void main() {
    // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
    // Initialize z with bias.
    float z = to_float(texelFetch(neuron_tex, ivec2(0, gl_FragCoord.y), 0));
    // Get width of neuron_tex to get number of weights + 1
    int neuron_tex_width = textureSize(neuron_tex, 0).x;
    // Normalize inputs for linear and leaky relu activation functions
    // Iterate over inputs and respective weights for this neuron.
    for (int i = 1; i < neuron_tex_width; i++) {
        float weight = to_float(texelFetch(neuron_tex, ivec2(i, gl_FragCoord.y), 0));
        float data = to_float(texelFetch(data_tex, ivec2(0, i - 1), 0));
        // Add weight[i] * input[i] to z.
        z += weight * data;
    }
    // Calculate activity.
    // Split activity into four bytes.
    if (is_tanh == 1) activity_out = to_bytes(tanh(z));
    if (is_sigmoid == 1) activity_out = to_bytes(1.0 / (1.0 + pow(2.718281828459045, - z)));
    if (is_leaky_relu == 1) activity_out = to_bytes(max(0.01 * z, z));
    if (is_linear == 1) activity_out = to_bytes(z);
}