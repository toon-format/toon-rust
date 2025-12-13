/* src/ui.rs */
//!▫~•◦-------------------------------‣
//! # Rune-Gate Chat Interface
//!▫~•◦-------------------------------------------------------------------‣
//! Provides a Bevy-based UI for interacting with the Xypher Codex.

use bevy::prelude::*;
use bevy::input::keyboard::{KeyboardInput, Key};
use bevy::input::ButtonState;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ChatState::default())
           .add_systems(Startup, setup_chat_ui)
           .add_systems(Update, chat_input_system);
    }
}

#[derive(Component)]
pub struct ChatInput;

#[derive(Component)]
pub struct ChatOutput;

#[derive(Resource, Default)]
pub struct ChatState {
    pub messages: Vec<String>,
    pub input_buffer: String,
}

pub fn setup_chat_ui(mut commands: Commands, _asset_server: Res<AssetServer>) {
    // Root node
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::FlexEnd,
            ..default()
        })
        .with_children(|parent| {
            // Chat Output Area
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(80.0),
                        padding: UiRect::all(Val::Px(10.0)),
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::FlexStart,
                        overflow: bevy::ui::Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(Color::BLACK.with_alpha(0.8)),
                    ChatOutput,
                ));

            // Input Area
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(50.0),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    padding: UiRect::all(Val::Px(5.0)),
                    ..default()
                })
                .with_children(|input_row| {
                    // Text Input
                    input_row
                        .spawn((
                            Node {
                                width: Val::Percent(90.0),
                                height: Val::Percent(100.0),
                                border: UiRect::all(Val::Px(2.0)),
                                ..default()
                            },
                            BorderColor(Color::WHITE),
                            BackgroundColor(Color::srgb(0.1, 0.1, 0.1)),
                            ChatInput,
                        ))
                        .with_children(|text_node| {
                            text_node.spawn((
                                Text::new("Type command..."),
                                TextFont {
                                    font_size: 20.0,
                                    ..default()
                                },
                                TextColor(Color::WHITE),
                            ));
                        });
                });
        });
}

pub fn chat_input_system(
    mut events: EventReader<KeyboardInput>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut chat_state: ResMut<ChatState>,
    mut query: Query<&mut Text, With<ChatInput>>,
) {
    if !query.is_empty() {
        let mut text = query.single_mut();

        for event in events.read() {
            if event.state == ButtonState::Pressed {
                if let Key::Character(ref c) = event.logical_key {
                    // Filter control characters if needed, but Key::Character usually handles printables
                    // Except backspace/enter might be handled by KeyCode check below or logical key
                    if !c.chars().any(|ch| ch.is_control()) {
                         chat_state.input_buffer.push_str(c.as_str());
                    }
                }
            }
        }

        if keyboard_input.just_pressed(KeyCode::Backspace) {
            chat_state.input_buffer.pop();
        }

        if keyboard_input.just_pressed(KeyCode::Enter) {
            if !chat_state.input_buffer.is_empty() {
                let msg = chat_state.input_buffer.clone();
                chat_state.messages.push(format!("> {}", msg));
                chat_state.input_buffer.clear();

                info!("Command received: {}", msg);
                // Future: dispatch to rune-xyco
            }
        }

        // Update display
        **text = chat_state.input_buffer.clone();
    }
}
