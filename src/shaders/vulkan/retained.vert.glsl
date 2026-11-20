#version 100
attribute vec2 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying lowp vec2 v_uv;
varying lowp vec4 v_color;
varying highp vec2 v_pos;

uniform mat4 Model;
uniform mat4 Projection;
uniform float Alpha;

void main() {
    vec4 world = Model * vec4(position, 0.0, 1.0);
    gl_Position = Projection * world;
    v_pos = world.xy;
    v_uv = texcoord;
    v_color = vec4(color0.rgb, color0.a * Alpha);
}
