//! Retained GPU geometry for large, static content.
//!
//! Unlike the normal batched path, a [`RetainedMesh`] uploads its vertices and
//! indices once and only submits draw calls on later frames. This is useful for
//! long text blocks or complex static shapes whose per-frame upload dominates
//! rendering, especially on translation layers such as WSLg's D3D12 path.
//!
//! The implementation uses miniquad's public `RenderingBackend` API only.

use std::{
    cell::RefCell, collections::HashSet, marker::PhantomData, mem::size_of, ops::Range, rc::Rc,
    sync::OnceLock,
};

use miniquad::{
    Backend, Bindings, BlendFactor, BlendState, BlendValue, BufferLayout, BufferSource, BufferType,
    BufferUsage, Equation, PassAction, Pipeline, PipelineParams, RenderPass, ShaderId, ShaderMeta,
    ShaderSource, TextureId, UniformBlockLayout, UniformDesc, UniformType, VertexAttribute,
    VertexFormat,
};

use crate::math::{Mat4, Vec2, Vec4};

/// One vertex in the retained pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RetainedVertex {
    pub position: [f32; 2],
    pub texcoord: [f32; 2],
    pub color: [f32; 4],
}

#[repr(C)]
struct Uniforms {
    model: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    alpha: f32,
}

const VERTEX: &str = r#"#version 100
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
"#;

const FRAGMENT: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;
varying lowp vec4 v_color;

uniform sampler2D Texture;

void main() {
    gl_FragColor = v_color * texture2D(Texture, v_uv);
}
"#;

#[derive(Clone, Copy)]
struct ActivePass {
    pass: Option<RenderPass>,
}

thread_local! {
    static PIPELINE: RefCell<Option<Pipeline>> = const { RefCell::new(None) };
    static WHITE: RefCell<Option<TextureId>> = const { RefCell::new(None) };
    static ACTIVE_PASS: RefCell<Option<ActivePass>> = const { RefCell::new(None) };
    static REMOVED_MATERIALS: RefCell<Vec<(Pipeline, ShaderId)>> = const { RefCell::new(Vec::new()) };
}

fn backend_supports_retained(ctx: &dyn miniquad::RenderingBackend) -> bool {
    let info = ctx.info();
    #[cfg(all(feature = "vulkan", any(windows, all(target_os = "linux", not(target_env = "ohos")))))]
    if matches!(info.backend, Backend::Vulkan) { return true; }
    // Native GLES2 has no guaranteed u32 index support. Do not guess extensions.
    matches!(info.backend, Backend::OpenGl)
        && (!info.gl_version_string.starts_with("OpenGL ES") || info.glsl_support.v300es)
}

/// Whether the current backend can use retained geometry.
///
/// The retained pipeline is written in GLSL; miniquad's Metal backend requires
/// an MSL shader, so Metal currently falls back to the immediate path. Call
/// this after macroquad has initialized its context.
pub fn is_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| backend_supports_retained(crate::get_quad_context()))
}

fn ensure_pipeline(ctx: &mut dyn miniquad::RenderingBackend) -> Pipeline {
    PIPELINE.with(|slot| {
        if let Some(pipeline) = *slot.borrow() {
            return pipeline;
        }
        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: VERTEX,
                    fragment: FRAGMENT,
                },
                miniquad::ShaderMeta {
                    uniforms: UniformBlockLayout {
                        uniforms: vec![
                            UniformDesc::new("Model", UniformType::Mat4),
                            UniformDesc::new("Projection", UniformType::Mat4),
                            UniformDesc::new("Alpha", UniformType::Float1),
                        ],
                    },
                    images: vec!["Texture".to_string()],
                },
            )
            .expect("retained geometry shader");
        let pipeline = ctx.new_pipeline(
            &[BufferLayout {
                stride: size_of::<RetainedVertex>() as i32,
                ..Default::default()
            }],
            &[
                VertexAttribute::new("position", VertexFormat::Float2),
                VertexAttribute::new("texcoord", VertexFormat::Float2),
                VertexAttribute::new("color0", VertexFormat::Float4),
            ],
            shader,
            PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );
        *slot.borrow_mut() = Some(pipeline);
        pipeline
    })
}

/// A 1x1 white texture for solid overlays such as underlines.
pub fn white_texture() -> TextureId {
    let ctx = crate::get_quad_context();
    WHITE.with(|slot| {
        if let Some(texture) = *slot.borrow() {
            return texture;
        }
        let texture = ctx.new_texture_from_rgba8(1, 1, &[255, 255, 255, 255]);
        *slot.borrow_mut() = Some(texture);
        texture
    })
}

/// Immutable vertex/index buffers plus their texture binding.
///
/// Create, draw and drop on the rendering thread while its context is alive.
/// The caller must keep the bound texture alive for the mesh lifetime.
pub struct RetainedMesh {
    bindings: Bindings,
    index_count: i32,
    _render_thread: PhantomData<Rc<()>>,
}

impl RetainedMesh {
    /// Upload geometry once. Returns `None` for invalid indices, oversized index
    /// buffers, empty geometry, or backends without the GL path.
    pub fn new(vertices: &[RetainedVertex], indices: &[u32], texture: TextureId) -> Option<Self> {
        if vertices.is_empty()
            || indices.is_empty()
            || indices.len() > i32::MAX as usize / size_of::<u32>()
            || indices
                .iter()
                .any(|&index| index as usize >= vertices.len())
        {
            return None;
        }
        let ctx = crate::get_quad_context();
        if !is_supported() {
            return None;
        }
        let vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(vertices),
        );
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(indices),
        );
        Some(Self {
            bindings: Bindings {
                vertex_buffers: vec![vertex_buffer],
                index_buffer,
                images: vec![texture],
            },
            index_count: indices.len() as i32,
            _render_thread: PhantomData,
        })
    }
}

impl Drop for RetainedMesh {
    fn drop(&mut self) {
        let ctx = crate::get_quad_context();
        // A shared retained pass may still reference these buffers.
        end_pass_if_active(ctx);
        ctx.delete_buffer(self.bindings.index_buffer);
        for buffer in &self.bindings.vertex_buffers {
            ctx.delete_buffer(*buffer);
        }
    }
}

/// A retained pipeline with a custom fragment shader and uniform block.
///
/// The vertex layout and vertex shader are fixed. Fragment shaders can use
/// v_pos, v_uv, v_color, and the mesh texture. Cloning shares the pipeline and
/// deep-copies the uniform buffer, so nested scopes can keep separate values.
/// The last clone schedules GPU resource deletion at the next flush/frame boundary.
/// Use on the rendering thread, and keep bound textures alive while drawing.
pub struct RetainedMaterial {
    gpu: Rc<MaterialGpu>,
    uniform_data: RefCell<Vec<f32>>,
    offsets: Vec<(String, usize, UniformType)>,
    textures: Vec<TextureId>,
}

struct MaterialGpu {
    pipeline: Pipeline,
    shader: ShaderId,
}

impl Drop for MaterialGpu {
    fn drop(&mut self) {
        // Reclaim at the next flush/frame boundary, with a live rendering context.
        // Thread-local teardown must not call the GPU; context teardown owns that case.
        let _ = REMOVED_MATERIALS.try_with(|removed| {
            removed.borrow_mut().push((self.pipeline, self.shader));
        });
    }
}

impl Clone for RetainedMaterial {
    fn clone(&self) -> Self {
        Self {
            gpu: Rc::clone(&self.gpu),
            uniform_data: RefCell::new(self.uniform_data.borrow().clone()),
            offsets: self.offsets.clone(),
            textures: self.textures.clone(),
        }
    }
}

impl RetainedMaterial {
    /// Build a material for the fixed retained vertex layout.
    ///
    /// Uniforms are appended after Model, Projection, and Alpha. Only non-array
    /// Float1, Float2, Float4 and Mat4 uniforms are accepted. Texture and uniform
    /// names must be unique and may not shadow the fixed names. Textures are
    /// bound after the mesh texture. Returns None for an invalid layout, shader
    /// compilation failure or an unsupported backend.
    pub fn new(
        fragment: &str,
        uniforms: &[UniformDesc],
        textures: &[(String, TextureId)],
    ) -> Option<Self> {
        let mut names = HashSet::from(["Model", "Projection", "Alpha", "Texture"]);
        for uniform in uniforms {
            if uniform.array_count != 1
                || !matches!(
                    uniform.uniform_type,
                    UniformType::Float1
                        | UniformType::Float2
                        | UniformType::Float4
                        | UniformType::Mat4
                )
                || !names.insert(uniform.name.as_str())
            {
                return None;
            }
        }
        if textures
            .iter()
            .any(|(name, _)| !names.insert(name.as_str()))
        {
            return None;
        }
        if !is_supported() {
            return None;
        }
        let ctx = crate::get_quad_context();
        let mut all_uniforms = vec![
            UniformDesc::new("Model", UniformType::Mat4),
            UniformDesc::new("Projection", UniformType::Mat4),
            UniformDesc::new("Alpha", UniformType::Float1),
        ];
        all_uniforms.extend(uniforms.iter().cloned());

        let mut offsets = Vec::with_capacity(all_uniforms.len());
        let mut size = 0usize;
        for uniform in &all_uniforms {
            offsets.push((uniform.name.clone(), size, uniform.uniform_type));
            size = size.checked_add(uniform.uniform_type.size() / size_of::<f32>())?;
        }
        if size.checked_mul(size_of::<f32>())? > i32::MAX as usize {
            return None;
        }
        let uniform_data = RefCell::new(vec![0.; size]);
        let mut images = vec!["Texture".to_string()];
        images.extend(textures.iter().map(|(name, _)| name.clone()));

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: VERTEX,
                    fragment,
                },
                ShaderMeta {
                    uniforms: UniformBlockLayout {
                        uniforms: all_uniforms.clone(),
                    },
                    images,
                },
            )
            .ok()?;

        let pipeline = ctx.new_pipeline(
            &[BufferLayout {
                stride: size_of::<RetainedVertex>() as i32,
                ..Default::default()
            }],
            &[
                VertexAttribute::new("position", VertexFormat::Float2),
                VertexAttribute::new("texcoord", VertexFormat::Float2),
                VertexAttribute::new("color0", VertexFormat::Float4),
            ],
            shader,
            PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );

        Some(Self {
            gpu: Rc::new(MaterialGpu { pipeline, shader }),
            uniform_data,
            offsets,
            textures: textures.iter().map(|(_, texture)| *texture).collect(),
        })
    }

    /// Set a Float1 uniform. Unknown names and mismatched types are ignored.
    pub fn set_f32(&self, name: &str, value: f32) {
        self.write(name, UniformType::Float1, &[value]);
    }

    /// Set a Float2 uniform. Unknown names and mismatched types are ignored.
    pub fn set_vec2(&self, name: &str, value: Vec2) {
        self.write(name, UniformType::Float2, &value.to_array());
    }

    /// Set a Float4 uniform. Unknown names and mismatched types are ignored.
    pub fn set_vec4(&self, name: &str, value: Vec4) {
        self.write(name, UniformType::Float4, &value.to_array());
    }

    /// Set a Mat4 uniform. Unknown names and mismatched types are ignored.
    pub fn set_mat4(&self, name: &str, value: Mat4) {
        self.write(name, UniformType::Mat4, &value.to_cols_array());
    }

    fn set_fixed(&self, model: Mat4, projection: Mat4, alpha: f32) {
        let mut data = self.uniform_data.borrow_mut();
        data[..16].copy_from_slice(&model.to_cols_array());
        data[16..32].copy_from_slice(&projection.to_cols_array());
        data[32] = alpha;
    }

    fn write(&self, name: &str, value_type: UniformType, values: &[f32]) {
        let Some((_, offset, uniform_type)) = self
            .offsets
            .iter()
            .find(|(uniform_name, _, _)| uniform_name == name)
        else {
            return;
        };
        if std::mem::discriminant(uniform_type) != std::mem::discriminant(&value_type)
            || uniform_type.size() != std::mem::size_of_val(values)
        {
            return;
        }
        let mut data = self.uniform_data.borrow_mut();
        if let Some(destination) = data.get_mut(*offset..*offset + values.len()) {
            destination.copy_from_slice(values);
        }
    }
}

fn pass_size(ctx: &dyn miniquad::RenderingBackend, pass: Option<RenderPass>) -> (i32, i32) {
    let (width, height) = if let Some(pass) = pass {
        ctx.render_pass_color_attachments(pass)
            .first()
            .copied()
            .map(|texture| {
                let (width, height) = ctx.texture_size(texture);
                (width as f32, height as f32)
            })
            .unwrap_or_else(miniquad::window::screen_size)
    } else {
        miniquad::window::screen_size()
    };
    (width as i32, height as i32)
}

fn begin_pass(ctx: &mut dyn miniquad::RenderingBackend, pass: Option<RenderPass>) -> (i32, i32) {
    let size = pass_size(ctx, pass);
    if let Some(pass) = pass {
        ctx.begin_pass(Some(pass), PassAction::Nothing);
    } else {
        ctx.begin_default_pass(PassAction::Nothing);
    }
    size
}

/// End a retained pass if one is still open.
///
/// Called by macroquad before it submits its own batched geometry, so retained
/// and immediate draws keep their original order.
pub(crate) fn end_pass_if_active(ctx: &mut dyn miniquad::RenderingBackend) {
    if ACTIVE_PASS.with(|slot| slot.borrow_mut().take()).is_some() {
        ctx.end_render_pass();
    }
    let removed = REMOVED_MATERIALS.with(|queue| std::mem::take(&mut *queue.borrow_mut()));
    for (pipeline, shader) in removed {
        ctx.delete_pipeline(pipeline);
        ctx.delete_shader(shader);
    }
}

/// Draw retained meshes, optionally restricting each mesh to an index range.
///
/// `ranges` is parallel to `meshes`: `Some(range)` draws only that slice of
/// the mesh, `None` skips the mesh. An empty slice draws every mesh in full.
/// This lets callers cull off-screen lines without splitting a mesh into many
/// draw calls.
pub fn draw_meshes_ranges(
    meshes: &[RetainedMesh],
    ranges: &[Option<Range<u32>>],
    model: Mat4,
    alpha: f32,
) {
    draw_meshes_ranges_inner(meshes, ranges, model, alpha, None);
}

/// Draw retained meshes with a custom retained material.
pub fn draw_meshes_ranges_with_material(
    meshes: &[RetainedMesh],
    ranges: &[Option<Range<u32>>],
    model: Mat4,
    alpha: f32,
    material: &RetainedMaterial,
) {
    draw_meshes_ranges_inner(meshes, ranges, model, alpha, Some(material));
}

fn draw_meshes_ranges_inner(
    meshes: &[RetainedMesh],
    ranges: &[Option<Range<u32>>],
    model: Mat4,
    alpha: f32,
    material: Option<&RetainedMaterial>,
) {
    if meshes.is_empty() {
        return;
    }

    let mut gl = unsafe { crate::window::get_internal_gl() };
    let pass = gl.quad_gl.get_active_render_pass();
    let scissor = gl.quad_gl.get_scissor();
    let viewport = gl.quad_gl.get_viewport_raw();
    let projection = gl.quad_gl.get_projection_matrix();

    // Submit pending immediate geometry first so retained geometry keeps its
    // place in the draw order. This also closes any previous retained pass.
    if gl.quad_gl.has_pending_draw_calls() {
        gl.flush();
    }

    let ctx = gl.quad_context;
    let active = ACTIVE_PASS.with(|slot| slot.borrow().map(|state| state.pass));
    let reuse = active == Some(pass);
    let (width, height) = if reuse {
        pass_size(ctx, pass)
    } else {
        end_pass_if_active(ctx);
        let size = begin_pass(ctx, pass);
        ACTIVE_PASS.with(|slot| *slot.borrow_mut() = Some(ActivePass { pass }));
        size
    };

    let (x, y, w, h) = viewport.unwrap_or((0, 0, width, height));
    ctx.apply_viewport(x, y, w, h);
    if let Some((x, y, w, h)) = scissor {
        ctx.apply_scissor_rect(x, height - (y + h), w, h);
    } else {
        ctx.apply_scissor_rect(0, 0, width, height);
    }

    if let Some(material) = material {
        ctx.apply_pipeline(&material.gpu.pipeline);
        material.set_fixed(model, projection, alpha);
        let data = material.uniform_data.borrow();
        ctx.apply_uniforms_from_bytes(data.as_ptr().cast(), data.len() * size_of::<f32>());
    } else {
        // Pass reuse does not imply that the selected shader is still bound.
        let pipeline = ensure_pipeline(ctx);
        ctx.apply_pipeline(&pipeline);
        let uniforms = Uniforms {
            model: model.to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            alpha,
        };
        ctx.apply_uniforms(miniquad::UniformsSource::table(&uniforms));
    }
    for (index, mesh) in meshes.iter().enumerate() {
        let (first, count) = if ranges.is_empty() {
            (0, mesh.index_count)
        } else {
            let Some(range) = ranges.get(index).and_then(|range| range.as_ref()) else {
                continue;
            };
            let start = range.start.min(mesh.index_count as u32);
            let end = range.end.min(mesh.index_count as u32);
            if end <= start {
                continue;
            }
            (start as i32, (end - start) as i32)
        };
        if let Some(material) = material {
            let mut images = Vec::with_capacity(1 + material.textures.len());
            images.push(mesh.bindings.images[0]);
            images.extend_from_slice(&material.textures);
            ctx.apply_bindings(&Bindings {
                vertex_buffers: mesh.bindings.vertex_buffers.clone(),
                index_buffer: mesh.bindings.index_buffer,
                images,
            });
        } else {
            ctx.apply_bindings(&mesh.bindings);
        }
        ctx.draw(first, count, 1);
    }
}
