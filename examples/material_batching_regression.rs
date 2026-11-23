//! Pixel regressions for material batching, texture ownership and readback.
//! Run with `cargo run --example material_batching_regression` on OpenGL,
//! or add `--features vulkan` for Vulkan on desktop Linux and Windows.

use macroquad::prelude::*;
use miniquad::gl;

const WIDTH: f32 = 32.;
const HEIGHT: f32 = 8.;
const COLUMN: f32 = WIDTH / 4.;
const RED_PIXEL: [u8; 4] = [255, 0, 0, 255];
const GREEN_PIXEL: [u8; 4] = [0, 255, 0, 255];
const BLUE_PIXEL: [u8; 4] = [0, 0, 255, 255];
const WHITE_PIXEL: [u8; 4] = [255; 4];
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
    draw_rectangle(index as f32 * COLUMN, 0., COLUMN, HEIGHT, color);
}

fn expect_pixel(image: &Image, index: u32, expected: [u8; 4], reason: &str) {
    let actual: [u8; 4] = image
        .get_pixel(index * COLUMN as u32 + COLUMN as u32 / 2, HEIGHT as u32 / 2)
        .into();
    assert_eq!(actual, expected, "{reason}");
}

fn flush() {
    unsafe { get_internal_gl().flush() };
}

fn gl_integer(param: gl::GLenum) -> gl::GLint {
    let mut value = 0;
    unsafe { gl::glGetIntegerv(param, &mut value) };
    value
}

fn check_screen_capture(material: &Material) {
    if unsafe { get_internal_gl() }.quad_context.info().backend != miniquad::Backend::OpenGl {
        return;
    }

    let capture = Texture2D::from_rgba8(1, 1, &BLACK_PIXEL);
    gl_use_material(material);
    draw_column(0, WHITE);
    flush();

    const GL_ACTIVE_TEXTURE: gl::GLenum = 0x84E0;
    let active = gl_integer(GL_ACTIVE_TEXTURE);
    let binding = gl_integer(gl::GL_TEXTURE_BINDING_2D);
    assert_eq!(active as u32, gl::GL_TEXTURE0 + 2, "nonzero texture unit");
    assert_ne!(binding, 0, "nonempty texture binding");
    capture.grab_screen();
    assert_eq!(gl_integer(GL_ACTIVE_TEXTURE), active, "active texture unit");
    assert_eq!(
        gl_integer(gl::GL_TEXTURE_BINDING_2D),
        binding,
        "binding on the active texture unit"
    );
    gl_use_default_material();
}

fn check_direct_texture_resize() {
    let texture = Texture2D::from_rgba8(1, 1, &BLACK_PIXEL);
    let id = texture.raw_miniquad_id();
    let pixels = [RED_PIXEL, GREEN_PIXEL].concat();
    {
        let ctx = unsafe { get_internal_gl() }.quad_context;
        assert!(ctx.try_texture_resize(id, 2, 1, Some(&pixels)));
        assert!(
            !ctx.try_texture_resize(id, u16::MAX as u32 + 1, 1, None),
            "resize exceeds Image width limit"
        );
    }

    let image = texture.get_texture_data();
    assert_eq!((image.width, image.height), (2, 1));
    assert_eq!(image.bytes, pixels);
}

const USE_VULKAN: bool = cfg!(all(
    feature = "vulkan",
    any(windows, all(target_os = "linux", not(target_env = "ohos")))
));

fn config() -> Conf {
    Conf {
        window_title: "Material batching regression".to_owned(),
        platform: miniquad::conf::Platform {
            rendering_backend: if USE_VULKAN {
                miniquad::conf::RenderingBackend::Vulkan
            } else {
                miniquad::conf::RenderingBackend::OpenGl
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

#[macroquad::main(config)]
async fn main() {
    let backend = unsafe { get_internal_gl() }.quad_context.info().backend;
    if USE_VULKAN {
        assert_ne!(backend, miniquad::Backend::OpenGl, "Vulkan requested");
    } else {
        assert_eq!(backend, miniquad::Backend::OpenGl, "OpenGL requested");
        check_direct_texture_resize();
    }

    let target = render_target(WIDTH as u32, HEIGHT as u32);
    let camera = Camera2D {
        render_target: Some(target.clone()),
        ..Camera2D::from_display_rect(Rect::new(0., 0., WIDTH, HEIGHT))
    };
    set_camera(&camera);
    clear_background(BLACK);

    let tint = tint_material();
    gl_use_material(&tint);
    tint.set_uniform("tint", [1f32, 0., 0., 1.]);
    draw_column(0, WHITE);
    tint.set_uniform("tint", [0f32, 1., 0., 1.]);
    flush();
    draw_column(1, WHITE);
    flush();
    let pixels = target.texture.get_texture_data();
    expect_pixel(&pixels, 0, RED_PIXEL, "queued uniform");
    expect_pixel(&pixels, 1, GREEN_PIXEL, "live uniform after flush");
    drop(tint);

    clear_background(BLACK);
    let material = texture_material();
    let unset = texture_material();
    let inactive = texture_material();
    let red = Texture2D::from_rgba8(1, 1, &RED_PIXEL);
    let green = Texture2D::from_rgba8(1, 1, &GREEN_PIXEL);
    let (red_weak, green_weak) = (red.weak_clone(), green.weak_clone());
    let green_id = green.raw_miniquad_id();
    let white_id = Texture2D::empty().raw_miniquad_id();

    gl_use_material(&material);
    material.set_texture("sample_tex", red);
    draw_column(0, WHITE);
    inactive.set_texture("sample_tex", green.clone());
    material.set_texture("sample_tex", green_weak.clone());
    material.set_texture("unknown", Texture2D::empty());
    drop((green, inactive));
    expect_pixel(
        &target.texture.get_texture_data(),
        0,
        BLACK_PIXEL,
        "set_texture flushed the draw queue",
    );
    draw_column(1, WHITE);
    flush();

    // Exercise the same GPU binding storage in a second flush, after the
    // red draw dropped the last owner but before frame-end texture collection.
    gl_use_material(&unset);
    draw_column(2, WHITE);
    unset.set_texture("sample_tex", red_weak.clone());
    draw_column(3, WHITE);
    flush();
    let pixels = target.texture.get_texture_data();
    expect_pixel(&pixels, 0, RED_PIXEL, "queued texture was replaced");
    expect_pixel(&pixels, 1, GREEN_PIXEL, "latest texture binding");
    expect_pixel(&pixels, 2, WHITE_PIXEL, "unset slot reused an old texture");
    expect_pixel(
        &pixels,
        3,
        WHITE_PIXEL,
        "expired weak texture must bind white",
    );

    set_default_camera();
    check_screen_capture(&material);
    next_frame().await;
    assert_eq!(
        red_weak.raw_miniquad_id(),
        white_id,
        "flushed batch leaked its binding"
    );
    assert_eq!(
        green_weak.raw_miniquad_id(),
        green_id,
        "material did not retain its weak binding"
    );

    set_camera(&camera);
    clear_background(BLACK);
    gl_use_material(&material);
    draw_column(0, WHITE);
    drop(material);
    draw_column(1, Color::from_rgba(0, 0, 255, 255));
    let replacement = tint_material();
    replacement.set_uniform("tint", [1f32, 0., 0., 1.]);
    gl_use_material(&replacement);
    draw_column(2, WHITE);
    set_default_camera();
    let pixels = target.texture.get_texture_data();
    expect_pixel(
        &pixels,
        0,
        GREEN_PIXEL,
        "pipeline removed or reused before its queued draw",
    );
    expect_pixel(&pixels, 1, BLUE_PIXEL, "dropped material remained selected");
    expect_pixel(&pixels, 2, RED_PIXEL, "replacement material failed");
    drop(replacement);

    next_frame().await;
    assert_eq!(
        green_weak.raw_miniquad_id(),
        white_id,
        "dropped material leaked its binding"
    );
    info!("Material batching regression passed");
}
