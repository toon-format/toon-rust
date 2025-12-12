/* src/scene.rs */
//!▫~•◦-------------------------------‣
//! # 3D scene rendering and interaction system for E8 geometry visualization.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gate to provide comprehensive
//! 3D rendering, camera controls, and vertex interaction for E8-Life structures.
//!
//! ### Key Capabilities
//! - **3D Scene Management:** Sets up cameras, lighting, and renders E8 vertices as 3D spheres.
//! - **Interactive Controls:** Provides orbit camera controls with mouse and keyboard input.
//! - **Vertex Interaction:** Implements vertex picking, hovering, and selection with visual feedback.
//! - **8D to 3D Projection:** Projects 8-dimensional E8 coordinates into 3D space for visualization.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `bridge` and `ui`.
//! It uses Bevy's ECS architecture and integrates with the backend for vertex data.
//!
//! ### Example
//! ```rust
//! use crate::rune_gate::ScenePlugin;
//!
//! // Add the scene plugin to your Bevy app
//! app.add_plugins(ScenePlugin);
//!
//! // Access projection functionality
//! let projection = E8Projection::default();
//! let vec3d = projection.project(coord8d);
//! ```

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::primitives::Sphere;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
use bevy::render::alpha::AlphaMode;
use bevy::window::PrimaryWindow;

use crate::bridge::{BackendHandle, SelectedVertex, VertexDetail};

/// Plugin for the 3D scene (camera, lighting, nodes).
pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_camera, setup_light, spawn_nodes))
            .add_systems(
                Update,
                (
                    orbit_camera_controls,
                    pick_vertices,
                    hover_vertices,
                    update_node_visuals,
                    focus_on_selection,
                ),
            );
    }
}

#[derive(Component)]
struct MainCamera;

fn setup_camera(mut commands: Commands) {
    commands.insert_resource(CameraRig::default());
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0, 18.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));
}

fn setup_light(mut commands: Commands) {
    commands.spawn((
        PointLight {
            intensity: 2200.0,
            range: 120.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(14.0, 22.0, 12.0),
    ));
}

/// Projection resource: simple linear map from 8D to 3D.
#[derive(Resource)]
pub struct E8Projection {
    pub mat_8x3: [[f32; 3]; 8],
}

fn palette_colors(v: &VertexDetail) -> (Color, Color) {
    if let (Some(p), Some(inv)) = (&v.positive_color, &v.inverted_color) {
        let base = hex_to_color(p).unwrap_or(Color::srgb(0.5, 0.7, 1.0));
        let halo = hex_to_color(inv).unwrap_or(base);
        return (base, halo);
    }
    match v.domain.as_str() {
        "Relationships" => (Color::srgb(0.25, 0.6, 0.9), Color::srgb(0.9, 0.4, 0.9)),
        "Psychology" => (Color::srgb(0.8, 0.6, 0.2), Color::srgb(0.2, 0.6, 0.9)),
        "E8 Root" => (Color::srgb(0.35, 0.8, 1.0), Color::srgb(0.2, 0.4, 1.0)),
        "E8 Spinor" => (Color::srgb(1.0, 0.45, 0.7), Color::srgb(0.8, 0.2, 0.6)),
        _ => (Color::srgb(0.75, 0.8, 0.85), Color::srgb(0.4, 0.5, 0.7)),
    }
}

fn hex_to_color(hex: &str) -> Option<Color> {
    let trimmed = hex.trim_start_matches('#');
    if trimmed.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&trimmed[0..2], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&trimmed[2..4], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&trimmed[4..6], 16).ok()? as f32 / 255.0;
    Some(Color::srgb(r, g, b))
}

impl Default for E8Projection {
    fn default() -> Self {
        // Simple hand-picked projection
        Self {
            mat_8x3: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.2, 0.1],
                [0.1, 0.5, 0.2],
                [0.2, 0.1, 0.5],
                [0.3, 0.3, 0.1],
                [0.1, 0.3, 0.3],
            ],
        }
    }
}

impl E8Projection {
    pub fn project(&self, coord: [f32; 8]) -> Vec3 {
        let mut out = [0.0; 3];
        for i in 0..8 {
            out[0] += coord[i] * self.mat_8x3[i][0];
            out[1] += coord[i] * self.mat_8x3[i][1];
            out[2] += coord[i] * self.mat_8x3[i][2];
        }
        Vec3::new(out[0], out[1], out[2])
    }
}

#[derive(Component)]
pub struct SelectableVertex;

#[derive(Component, Clone)]
pub struct VertexData(pub VertexDetail);

#[derive(Component)]
pub struct VertexId(pub u32);

#[derive(Component)]
pub struct Halo(pub Handle<StandardMaterial>);

#[derive(Resource, Default, Clone)]
pub struct HoveredVertex(pub Option<VertexDetail>);

/// Simple orbital camera rig.
#[derive(Resource)]
struct CameraRig {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 18.0,
            yaw: 0.0,
            pitch: -0.2,
        }
    }
}

fn spawn_nodes(
    mut commands: Commands,
    backend: Res<BackendHandle>,
    projection: Res<E8Projection>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let verts: Vec<VertexDetail> = backend.0.list_vertices();
    for v in verts {
        let pos = projection.project(v.coord8d);
        let (color, halo_color) = palette_colors(&v);
        let scale = if v.kind.contains("II") { 0.35 } else { 0.28 };

        let mesh = meshes.add(Sphere::new(scale).mesh().ico(3).unwrap());
        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive: color.into(),
            metallic: 0.08,
            perceptual_roughness: 0.55,
            ..default()
        });
        let halo_material = materials.add(StandardMaterial {
            base_color: halo_color.with_alpha(0.2).into(),
            emissive: halo_color.into(),
            alpha_mode: AlphaMode::Add,
            unlit: true,
            ..default()
        });

        commands
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material),
                Transform::from_translation(pos),
                SelectableVertex,
                VertexData(v.clone()),
                VertexId(v.id),
            ))
            .with_children(|child| {
                child.spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(halo_material.clone()),
                    Transform::from_scale(Vec3::splat(1.1)),
                    Halo(halo_material.clone()),
                ));
            });
    }
}

fn orbit_camera_controls(
    time: Res<Time>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut rig: ResMut<CameraRig>,
    mut q_cam: Query<&mut Transform, With<MainCamera>>,
) {
    let mut look_delta = Vec2::ZERO;
    let mut pan_delta = Vec2::ZERO;

    for ev in mouse_motion.read() {
        if buttons.pressed(MouseButton::Right) {
            look_delta += ev.delta;
        } else if buttons.pressed(MouseButton::Middle) {
            pan_delta += ev.delta;
        }
    }

    // Orbit
    if look_delta.length_squared() > 0.0 {
        let sensitivity = 0.01;
        rig.yaw -= look_delta.x * sensitivity;
        rig.pitch = (rig.pitch - look_delta.y * sensitivity).clamp(-1.5, 1.5);
    }

    // Pan
    if pan_delta.length_squared() > 0.0 {
        if let Ok(transform) = q_cam.single() {
            let right = transform.right();
            let up = transform.up();
            let pan_speed = 0.002 * rig.distance.max(1.0);
            rig.target += (-pan_delta.x * pan_speed) * right + (pan_delta.y * pan_speed) * up;
        }
    }

    // Zoom
    for ev in mouse_wheel.read() {
        let zoom_speed = 6.0;
        rig.distance = (rig.distance - ev.y * zoom_speed * time.delta_secs()).clamp(3.0, 80.0);
    }

    // Apply to camera transform
    if let Ok(mut transform) = q_cam.single_mut() {
        let yaw_rot = Quat::from_rotation_y(rig.yaw);
        let pitch_rot = Quat::from_rotation_x(rig.pitch);
        let rotation = yaw_rot * pitch_rot;
        let offset = rotation * Vec3::new(0.0, 0.0, rig.distance);
        let eye = rig.target + offset;
        *transform = Transform::from_translation(eye).looking_at(rig.target, Vec3::Y);
    }
}

fn pick_vertices(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    verts: Query<(&GlobalTransform, &VertexData), With<SelectableVertex>>,
    mut selected: ResMut<SelectedVertex>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let window = if let Ok(w) = windows.single() {
        w
    } else {
        return;
    };
    let cursor = if let Some(pos) = window.cursor_position() {
        pos
    } else {
        return;
    };
    let (camera, cam_tf) = if let Ok(c) = camera_q.single() {
        c
    } else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        return;
    };

    let mut best: Option<(f32, VertexDetail)> = None;
    for (transform, data) in verts.iter() {
        let pos = transform.translation();
        let to_point = pos - ray.origin;
        let dir: Vec3 = ray.direction.into();
        let t = to_point.dot(dir);
        if t < 0.0 {
            continue;
        }
        let closest_point = ray.origin + dir * t;
        let dist_sq = pos.distance_squared(closest_point);
        let hit_radius_sq = 0.35f32 * 0.35;
        if dist_sq <= hit_radius_sq {
            if best.as_ref().map(|(d, _)| dist_sq < *d).unwrap_or(true) {
                best = Some((dist_sq, data.0.clone()));
            }
        }
    }

    if let Some((_, v)) = best {
        selected.0 = Some(v);
    }
}

fn hover_vertices(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    verts: Query<(&GlobalTransform, &VertexData), With<SelectableVertex>>,
    mut hovered: ResMut<HoveredVertex>,
) {
    let window = if let Ok(w) = windows.single() {
        w
    } else {
        return;
    };
    let cursor = if let Some(pos) = window.cursor_position() {
        pos
    } else {
        hovered.0 = None;
        return;
    };
    let (camera, cam_tf) = if let Ok(c) = camera_q.single() {
        c
    } else {
        hovered.0 = None;
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        hovered.0 = None;
        return;
    };

    let mut best: Option<(f32, VertexDetail)> = None;
    for (transform, data) in verts.iter() {
        let pos = transform.translation();
        let dir: Vec3 = ray.direction.into();
        let to_point = pos - ray.origin;
        let t = to_point.dot(dir);
        if t < 0.0 {
            continue;
        }
        let closest_point = ray.origin + dir * t;
        let dist_sq = pos.distance_squared(closest_point);
        let hit_radius_sq = 0.4f32 * 0.4;
        if dist_sq <= hit_radius_sq {
            if best.as_ref().map(|(d, _)| dist_sq < *d).unwrap_or(true) {
                best = Some((dist_sq, data.0.clone()));
            }
        }
    }
    hovered.0 = best.map(|(_, v)| v);
}

fn update_node_visuals(
    time: Res<Time>,
    selected: Res<SelectedVertex>,
    hovered: Res<HoveredVertex>,
    mut transforms: Query<(&mut Transform, Option<&Children>, &VertexData), With<SelectableVertex>>,
    mut halo_transforms: Query<&mut Transform, With<Halo>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    halos: Query<&Halo>,
) {
    let hover_id = hovered.0.as_ref().map(|v| v.id);
    let selected_id = selected.0.as_ref().map(|v| v.id);
    let pulse = (time.elapsed_secs() * 2.0).sin() * 0.5 + 0.5;

    for (mut transform, children, data) in transforms.iter_mut() {
        let mut scale = 1.0;
        if Some(data.0.id) == hover_id {
            scale *= 1.12;
        }
        if Some(data.0.id) == selected_id {
            scale *= 1.18;
        }
        transform.scale = Vec3::splat(scale);

        if let Some(children) = children {
            for child_entity in children.iter() {
                if let Ok(mut halo_tf) = halo_transforms.get_mut(child_entity) {
                    let mut halo_scale = 1.1;
                    let mut alpha = 0.2;
                    if Some(data.0.id) == hover_id {
                        halo_scale = 1.18;
                        alpha = 0.35;
                    }
                    if Some(data.0.id) == selected_id {
                        halo_scale = 1.2 + 0.08 * pulse;
                        alpha = 0.4 + 0.3 * pulse;
                    }
                    halo_tf.scale = Vec3::splat(halo_scale);
                    if let Ok(Halo(handle)) = halos.get(child_entity) {
                        if let Some(mat) = materials.get_mut(handle) {
                            mat.base_color = mat.base_color.with_alpha(alpha);
                            mat.emissive = mat.emissive.with_alpha(alpha);
                        }
                    }
                }
            }
        }
    }
}

fn focus_on_selection(
    projection: Res<E8Projection>,
    selected: Res<SelectedVertex>,
    mut rig: ResMut<CameraRig>,
) {
    if !selected.is_changed() {
        return;
    }
    if let Some(v) = &selected.0 {
        rig.target = projection.project(v.coord8d);
    }
}
