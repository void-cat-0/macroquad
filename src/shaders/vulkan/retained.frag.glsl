#version 100
precision mediump float;

varying lowp vec2 v_uv;
varying lowp vec4 v_color;

uniform sampler2D Texture;

void main() {
    gl_FragColor = v_color * texture2D(Texture, v_uv);
}
