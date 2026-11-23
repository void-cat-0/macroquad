//! Run with `cargo run --example material_batching_regression` on OpenGL.
//! Exits after checking the rendered pixels and texture binding, or panics on regression.
use macroquad::prelude::*;
use miniquad::gl;

const RED_PIXEL: [u8; 4] = [255, 0, 0, 255];
const GREEN_PIXEL: [u8; 4] = [0, 255, 0, 255];
const BLUE_PIXEL: [u8; 4] = [0, 0, 255, 255];
const BLACK_PIXEL: [u8; 4] = [0, 0, 0, 255];

const VERTEX: &str = r#"#version 100
attribute vec3 position;
uniform mat4 Model;
uniform mat4 Projection;
void main() { gl_Position = Projection * Model * vec4(position, 1.0); }
"#;

fn tint_material() -> Material {
    load_material(
        ShaderSource::Glsl {
            vertex: VERTEX,
            fragment: r#"#version 100
precision lowp float;
uniform vec4 tint;
void main() { gl_FragColor = tint; }
"#,
        },
        MaterialParams {
            uniforms: vec![UniformDesc::new("tint", UniformType::Float4)],
            ..Default::default()
        },
    )
    .unwrap()
}

fn texture_material() -> Material {
    load_material(
        ShaderSource::Glsl {
            vertex: VERTEX,
            fragment: r#"#version 100
precision lowp float;
uniform sampler2D sample_tex;
void main() { gl_FragColor = texture2D(sample_tex, vec2(0.5)); }
"#,
        },
        MaterialParams {
            textures: vec!["sample_tex".to_owned()],
            ..Default::default()
        },
    )
    .unwrap()
}

fn draw_column(index: u32, color: Color) {
    let w = screen_width() / 3.;
    draw_rectangle(index as f32 * w, 0., w, screen_height(), color);
}

fn expect_pixel(index: u32, expected: [u8; 4], reason: &str) {
    let (w, h) = miniquad::window::screen_size();
    let mut pixel = [0u8; 4];
    unsafe {
        gl::glReadPixels(
            ((index as f32 + 0.5) * w / 3.) as i32,
            (h / 2.) as i32,
            1,
            1,
            gl::GL_RGBA,
            gl::GL_UNSIGNED_BYTE,
            pixel.as_mut_ptr().cast(),
        );
    }
    assert_eq!(pixel, expected, "{reason}");
}

fn gl_integer(param: gl::GLenum) -> gl::GLint {
    let mut value = 0;
    unsafe { gl::glGetIntegerv(param, &mut value) };
    value
}

fn flush() {
    unsafe { get_internal_gl().flush() };
}

#[macroquad::main("Material batching regression")]
async fn main() {
    if unsafe { get_internal_gl().quad_context.info().backend } != miniquad::Backend::OpenGl {
        warn!("Material batching regression requires OpenGL");
        return;
    }

    clear_background(BLACK);
    let tint = tint_material();
    gl_use_material(&tint);
    tint.set_uniform("tint", [1f32, 0., 0., 1.]);
    draw_column(0, WHITE);
    tint.set_uniform("tint", [0f32, 1., 0., 1.]);
    flush();
    draw_column(1, WHITE);
    flush();
    expect_pixel(0, RED_PIXEL, "queued uniform");
    expect_pixel(1, GREEN_PIXEL, "live uniform overwritten by previous flush");
    drop(tint);

    clear_background(BLACK);
    let material = texture_material();
    let red = Texture2D::from_rgba8(1, 1, &RED_PIXEL);
    let green = Texture2D::from_rgba8(1, 1, &GREEN_PIXEL);
    let red_weak = red.weak_clone();
    let green_weak = green.weak_clone();
    let green_id = green.raw_miniquad_id();
    let white_id = Texture2D::empty().raw_miniquad_id();
    let capture = Texture2D::from_rgba8(1, 1, &BLACK_PIXEL);

    gl_use_material(&material);
    material.set_texture("sample_tex", red);
    draw_column(0, WHITE);
    material.set_texture("sample_tex", green);
    expect_pixel(
        0,
        BLACK_PIXEL,
        "set_texture unexpectedly flushed the draw queue",
    );
    draw_column(1, WHITE);
    flush();
    expect_pixel(0, RED_PIXEL, "queued texture was replaced");
    expect_pixel(1, GREEN_PIXEL, "latest texture binding was not used");

    const GL_ACTIVE_TEXTURE: gl::GLenum = 0x84E0;
    const GL_TEXTURE_BINDING_2D: gl::GLenum = 0x8069;
    let active = gl_integer(GL_ACTIVE_TEXTURE);
    let binding = gl_integer(GL_TEXTURE_BINDING_2D);
    assert_eq!(
        active as u32,
        gl::GL_TEXTURE0 + 2,
        "test must exercise a nonzero unit"
    );
    assert_ne!(binding, 0, "test must exercise a nonempty binding");
    capture.grab_screen();
    assert_eq!(
        gl_integer(GL_ACTIVE_TEXTURE),
        active,
        "screen capture changed active unit"
    );
    assert_eq!(
        gl_integer(GL_TEXTURE_BINDING_2D),
        binding,
        "screen capture changed the active unit's binding"
    );

    next_frame().await;
    assert_eq!(
        red_weak.raw_miniquad_id(),
        white_id,
        "flushed batch leaked its binding"
    );
    assert_eq!(
        green_weak.raw_miniquad_id(),
        green_id,
        "material did not retain its binding"
    );

    clear_background(BLACK);
    gl_use_material(&material);
    draw_column(0, WHITE);
    drop(material);
    draw_column(1, color_u8!(0, 0, 255, 255));
    let replacement = tint_material();
    replacement.set_uniform("tint", [1f32, 0., 0., 1.]);
    gl_use_material(&replacement);
    draw_column(2, WHITE);
    flush();
    expect_pixel(
        0,
        GREEN_PIXEL,
        "pipeline was removed or reused before queued draw",
    );
    expect_pixel(1, BLUE_PIXEL, "dropped material remained selected");
    expect_pixel(2, RED_PIXEL, "replacement material failed");
    drop(replacement);

    next_frame().await;
    assert_eq!(
        green_weak.raw_miniquad_id(),
        white_id,
        "dropped material leaked its binding"
    );
    info!("Material batching regression passed");
}
