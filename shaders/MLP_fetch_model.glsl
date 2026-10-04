#version 300 es
precision highp float;
precision highp int;

uniform sampler2D tex;
out vec4 outTex;
void main() {
    outTex = texelFetch(tex, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0);
}