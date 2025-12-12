/* src/ui.rs */
//!▫~•◦-------------------------------‣
//! # User interface system providing interactive panels and controls for the E8 viewer.
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module is designed for integration into rune-gate to provide a comprehensive
//! UI system with panel layouts, vertex details, and interactive controls.
//!
//! ### Key Capabilities
//! - **Panel System:** Configurable left/right panel layouts with toggle functionality.
//! - **Vertex Information:** Displays detailed information about selected E8 vertices and domains.
//! - **Interactive Controls:** Provides UI controls for layout toggling and vertex inspection.
//! - **Hover Tooltips:** Shows contextual information when hovering over 3D vertices.
//!
//! ### Architectural Notes
//! This module is designed to work with modules such as `bridge` and `scene`.
//! It uses Bevy's UI system and integrates with backend data for domain and vertex information.
//!
//! ### Example
//! ```rust
//! use crate::rune_gate::{UiPlugin, ViewerLayout, PanelSide};
//!
//! // Add the UI plugin to your Bevy app
//! app.add_plugins(UiPlugin);
//!
//! // Configure the layout
//! app.insert_resource(ViewerLayout {
//!     panel_side: PanelSide::Right,
//! });
//! ```

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::bridge::{BackendHandle, DomainSummary, SelectedVertex};
use crate::scene::HoveredVertex;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PanelSide {
    Left,
    Right,
}

impl Default for PanelSide {
    fn default() -> Self {
        PanelSide::Left
    }
}

#[derive(Resource, Default)]
pub struct ViewerLayout {
    pub panel_side: PanelSide,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewerLayout>()
            .add_systems(Startup, setup_ui)
            .init_resource::<SelectedVertex>()
            .add_systems(
                Update,
                (
                    update_details_panel,
                    handle_layout_toggle,
                    update_hover_tooltip,
                ),
            );
    }
}

#[derive(Component)]
struct LayoutToggle;

#[derive(Component)]
struct LayoutToggleLabel;

#[derive(Component)]
struct RootUi;

fn setup_ui(mut commands: Commands, backend: Res<BackendHandle>, layout: Res<ViewerLayout>) {
    commands
        .spawn((
            Node {
                width: percent(100.0),
                height: percent(100.0),
                position_type: PositionType::Absolute,
                flex_direction: match layout.panel_side {
                    PanelSide::Left => FlexDirection::Row,
                    PanelSide::Right => FlexDirection::RowReverse,
                },
                ..default()
            },
            BackgroundColor(Color::NONE),
            RootUi,
        ))
        .with_children(|parent| {
            // Panel container
            parent
                .spawn((
                    Node {
                        width: percent(32.0),
                        height: percent(100.0),
                        flex_direction: FlexDirection::Row,
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.04, 0.05, 0.08, 0.95)),
                ))
                .with_children(|col| {
                    // Rail
                    col.spawn((
                        Node {
                            width: px(48.0),
                            height: percent(100.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.95)),
                    ));

                    // Panel body
                    col.spawn((
                        Node {
                            width: percent(100.0),
                            height: percent(100.0),
                            flex_direction: FlexDirection::Column,
                            padding: UiRect::all(px(16.0)),
                            row_gap: px(12.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.06, 0.07, 0.1, 0.9)),
                    ))
                    .with_children(|panel| {
                        // Layout toggle
                        panel
                            .spawn((
                                Button,
                                Node {
                                    width: px(140.0),
                                    height: px(32.0),
                                    justify_content: JustifyContent::Center,
                                    align_items: AlignItems::Center,
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.08, 0.1, 0.16, 0.95)),
                                BorderRadius::all(px(4.0)),
                                LayoutToggle,
                            ))
                            .with_children(|btn| {
                                btn.spawn((
                                    Text::new(match layout.panel_side {
                                        PanelSide::Left => "Panel: Left (press L / click)",
                                        PanelSide::Right => "Panel: Right (press L / click)",
                                    }),
                                    TextFont::default(),
                                    TextColor(Color::srgb(0.75, 0.85, 1.0)),
                                    LayoutToggleLabel,
                                ));
                            });

                        // Graph panel header
                        panel.spawn((
                            Text::new("Graph"),
                            TextFont {
                                font_size: 20.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.75, 0.85, 1.0)),
                        ));

                        // Domains list
                        let domains: Vec<DomainSummary> = backend.0.list_domains();
                        for d in domains {
                            panel.spawn((
                                Text::new(format!("{}  ({})", d.name, d.count)),
                                TextFont {
                                    font_size: 14.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.7, 0.8, 1.0)),
                            ));
                        }

                        // Query panel header
                        panel.spawn((
                            Text::new("Query"),
                            TextFont {
                                font_size: 20.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.75, 0.85, 1.0)),
                        ));
                        // Stub query buttons (visual only)
                        panel.spawn((
                            Text::new("Nearest to: courage (stub)"),
                            TextFont {
                                font_size: 14.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.6, 0.95, 0.9)),
                        ));
                        panel.spawn((
                            Text::new("Path: fear → resilience (stub)"),
                            TextFont {
                                font_size: 14.0,
                                ..default()
                            },
                            TextColor(Color::srgb(1.0, 0.6, 0.8)),
                        ));

                        // Details header
                        panel.spawn((
                            Text::new("Details"),
                            TextFont {
                                font_size: 20.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.75, 0.85, 1.0)),
                            DetailsHeader,
                        ));

                        // Details body
                        panel.spawn((
                            Text::new("Select a node…"),
                            TextFont {
                                font_size: 14.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.7, 0.8, 1.0)),
                            DetailsBody,
                        ));
                    });
                });

            // Scene placeholder (keeps flex layout balanced)
            parent.spawn((
                Node {
                    width: percent(68.0),
                    height: percent(100.0),
                    ..default()
                },
                BackgroundColor(Color::NONE),
            ));

            // Hover tooltip (initially hidden, absolute positioned)
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    display: Display::None,
                    padding: UiRect::all(px(6.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.05, 0.06, 0.08, 0.9)),
                BorderRadius::all(px(6.0)),
                HoverTooltip,
                children![(
                    Text::new(""),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.85, 0.9, 1.0)),
                )],
            ));
        });
}

#[derive(Component)]
struct DetailsHeader;

#[derive(Component)]
struct DetailsBody;

#[derive(Component)]
struct HoverTooltip;

fn update_details_panel(
    selection: Option<Res<SelectedVertex>>,
    mut query: Query<(&mut Text, Option<&DetailsHeader>, Option<&DetailsBody>)>,
) {
    if selection.is_none() {
        return;
    }
    let sel = selection.unwrap();
    for (mut text, is_header, is_body) in query.iter_mut() {
        if is_header.is_some() {
            **text = "Details".into();
        } else if is_body.is_some() {
            **text = if let Some(v) = &sel.0 {
                format!(
                    "{}\nDomain: {}\nKind: {}\nID: {}\nBlurb: {}",
                    v.label, v.domain, v.kind, v.id, v.blurb
                )
            } else {
                "Select a node…".into()
            };
        }
    }
}

fn handle_layout_toggle(
    mut interactions: Query<&Interaction, (Changed<Interaction>, With<LayoutToggle>)>,
    mut layout: ResMut<ViewerLayout>,
    keys: Res<ButtonInput<KeyCode>>,
    mut roots: Query<&mut Node, With<RootUi>>,
    mut labels: Query<&mut Text, With<LayoutToggleLabel>>,
) {
    let mut flipped = false;
    for interaction in interactions.iter_mut() {
        if *interaction == Interaction::Pressed {
            flipped = true;
        }
    }
    if keys.just_pressed(KeyCode::KeyL) {
        flipped = true;
    }
    if !flipped {
        return;
    }
    layout.panel_side = match layout.panel_side {
        PanelSide::Left => PanelSide::Right,
        PanelSide::Right => PanelSide::Left,
    };

    for mut style in roots.iter_mut() {
        style.flex_direction = match layout.panel_side {
            PanelSide::Left => FlexDirection::Row,
            PanelSide::Right => FlexDirection::RowReverse,
        };
    }
    for mut text in labels.iter_mut() {
        **text = match layout.panel_side {
            PanelSide::Left => "Panel: Left (press L / click)".into(),
            PanelSide::Right => "Panel: Right (press L / click)".into(),
        };
    }
}

fn update_hover_tooltip(
    hovered: Option<Res<HoveredVertex>>,
    window_q: Query<&Window, With<PrimaryWindow>>,
    mut tooltip_q: Query<(&mut Node, &mut Text), With<HoverTooltip>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
) {
    let Some(hovered) = hovered else { return };
    let Ok(window) = window_q.single() else {
        return;
    };
    let Ok((mut node, mut text)) = tooltip_q.single_mut() else {
        return;
    };

    if let Some(cursor) = window.cursor_position() {
        if let Some(v) = &hovered.0 {
            node.display = Display::Flex;
            node.left = px(cursor.x + 14.0);
            node.top = px(cursor.y + 14.0);
            let opp = v
                .opposite
                .as_ref()
                .map(|o| format!("Opposite: {}", o))
                .unwrap_or_else(|| "".into());
            **text = format!("{}\n{} • {}\n{}", v.label, v.domain, v.kind, opp)
                .trim()
                .to_string();
        } else {
            node.display = Display::None;
        }
    } else {
        node.display = Display::None;
    }

    // Hide tooltip while dragging/orbiting with right/middle buttons
    if mouse_buttons.pressed(MouseButton::Right) || mouse_buttons.pressed(MouseButton::Middle) {
        node.display = Display::None;
    }
}
