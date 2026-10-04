#version 300 es
precision highp float;
precision highp int;

uniform sampler2D neuron_tex;
uniform sampler2D data_tex;
uniform sampler2D activity_tex;
uniform sampler2D error_tex;

uniform float learning_rate;

uniform int y_instead_of_error;

uniform int is_sigmoid;
uniform int is_tanh;
uniform int is_leaky_relu;
uniform int is_linear;

layout(location = 0) out vec4 neuron_out;
layout(location = 1) out vec4 error_out;

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

// Shader to calculate activity of a single neuron.
void main() {
    // Width is always one, so row gl_FragCoord.y in neuronTex is the line of neuron[gl_FragCoord.y].
    int row = int(gl_FragCoord.y);
    int column = int(gl_FragCoord.x);

    float activity = to_float(texelFetch(activity_tex, ivec2(0, row), 0));

    float error = to_float(texelFetch(error_tex, ivec2(row, 0), 0));
    if (y_instead_of_error == 0) error = activity - error;

    float dc_dz = 2.0 * error;
    if (is_leaky_relu == 1) dc_dz *= sign(activity) * 0.495 + 0.505;
    if (is_tanh == 1) dc_dz *= 1.0 - tanh(activity) * tanh(activity);
    if (is_sigmoid == 1)  {
    float sigmoid_activity = 1.0 / (1.0 + pow(2.718281828459045, - activity));
    dc_dz *= sigmoid_activity * (1.0 - sigmoid_activity);
    }

    float weight = to_float(texelFetch(neuron_tex, ivec2(column, row), 0));

    float modifier = dc_dz * learning_rate;
    float dx = weight * dc_dz;

    if (column != 0) {
    float data = to_float(texelFetch(data_tex, ivec2(0, column - 1), 0));
    modifier *= data;
    }

    neuron_out = to_bytes(weight - modifier);
    error_out = to_bytes(dx);
}