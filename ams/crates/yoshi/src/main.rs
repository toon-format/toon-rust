/* crates/yoshi/src/main.rs */
//! # Yoshi Interactive Shell & Runtime Host
//!
//! This binary provides a feature-gated, interactive command-line interface (CLI) and
//! Terminal User Interface (TUI) for demonstrating and managing the Yoshi ecosystem's
//! runtime capabilities.
//!
//! # ArcMoon Studios – Yoshi Shell Module
//!▫~•◦-------------------------------------‣
//!
//! This module is designed to act as a live host process for `yoshi-std`'s powerful
//! runtime systems, such as the supervisor, ML recovery engine, and circuit breakers.
//!
//! ### Key Capabilities
//! - **CLI Interaction:** Provides direct command-line access to control and query Yoshi systems.
//! - **TUI Dashboard:** A real-time `ratatui` dashboard for visualizing system health, live error feeds, and worker status.
//! - **Feature Gating:** All CLI and TUI dependencies are gated behind the `cli` and `yoshell` features, ensuring zero overhead for library consumers.
//!
//! ### Architectural Notes
//! This binary is the designated runtime for demonstrating long-running, stateful Yoshi
//! components. It is compiled only when the corresponding features are enabled, preserving
//! the `yoshi` crate's integrity as a lean library facade.
//!
//! ### Example
//! ```bash
//! # Launch the TUI dashboard (requires the 'yoshell' feature)
//! cargo run --package yoshi --bin YoshiShell --features yoshell -- tui
//!
//! # Run a CLI-only command (requires only the 'cli' feature)
//! cargo run --package yoshi --bin YoshiShell --features cli -- supervisor status
//!
//! # Run the release binary directly after building it
//! .\target\release\YoshiShell.exe tui
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// This entire file is gated by the `cli` feature flag.
#![cfg(feature = "cli")]

use clap::{Parser, Subcommand, ValueEnum};
use std::env;
use std::sync::Arc;
use yoshi::prelude::*;

// TUI-specific imports are gated by the `yoshell` feature.
#[cfg(feature = "yoshell")]
use {
    chrono::Local,
    crossterm::{
        event::{
            self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
            MouseButton, MouseEvent, MouseEventKind,
        },
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    rand::random,
    ratatui::{
        backend::CrosstermBackend,
        prelude::*,
        text::Line,
        widgets::{
            Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph,
            ScrollbarState, Wrap,
        },
    },
    ratatui_image::{Resize, StatefulImage, picker::Picker, protocol::StatefulProtocol},
    std::{
        io,
        sync::Mutex,
        time::{Duration, Instant},
    },
    yoshi_std::{
        CircuitState, PerformanceMetrics, SystemHealth, performance_metrics, system_health,
    },
};

#[cfg(feature = "yoshell")]
/// Embedded background image for the terminal backdrop.
const TUI_BG_IMAGE: &[u8] = include_bytes!("../assets/TUI-Bg.png");

#[cfg(feature = "yoshell")]
/// Load the background image once and darken it so it doesn't fight the UI.
fn load_background_image() -> Option<image::DynamicImage> {
    let img = image::load_from_memory(TUI_BG_IMAGE).ok()?;
    // Darken only (no blur) so widgets stay readable.
    Some(img.brighten(-120))
}

/// Yoshi Interactive Shell & Runtime Host
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Select shell mode. OgShell is view-only, NeuShell enables autonomous apply.
    #[arg(long, value_enum, default_value_t = ShellMode::OgShell, env = "YOSHELL_MODE")]
    mode: ShellMode,
    /// Override NATS connection URL(s); comma-separated. Defaults to localhost when `nats` feature is on.
    #[cfg(feature = "nats")]
    #[arg(long, env = "NATS_URL")]
    nats_url: Option<String>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the real-time TUI dashboard.
    #[cfg(feature = "yoshell")]
    Tui,
    /// Interact with the Supervisor system.
    Supervisor(SupervisorArgs),
    /// Interact with the Circuit Breaker system.
    Circuit(CircuitArgs),
    /// Migrate codebase from anyhow/thiserror to Yoshi stack.
    Migrate(MigrateArgs),
}

#[derive(Parser)]
struct SupervisorArgs {
    #[command(subcommand)]
    action: SupervisorAction,
}

#[derive(Subcommand)]
enum SupervisorAction {
    /// Display the status of supervised workers.
    Status,
}

#[derive(Parser)]
struct CircuitArgs {
    #[command(subcommand)]
    action: CircuitAction,
}

#[derive(Subcommand)]
enum CircuitAction {
    /// Force the named circuit breaker open.
    ForceOpen,
    /// Reset the named circuit breaker to the closed state.
    Reset,
}

/// Migration command arguments and configuration
#[derive(Parser)]
struct MigrateArgs {
    /// Apply changes to files (default is dry-run)
    #[arg(long)]
    apply: bool,
    /// Disable backup creation before applying changes
    #[arg(long)]
    no_backup: bool,
    /// Skip running cargo fmt after migration
    #[arg(long)]
    no_fmt: bool,
    /// Skip running cargo clippy after migration
    #[arg(long)]
    no_clippy: bool,
    /// Skip running cargo check after migration
    #[arg(long)]
    no_check: bool,
    /// Disable post-migration cleanup of old dependencies
    #[arg(long)]
    no_cleanup: bool,
    /// Set logging level (quiet, info, verbose, debug)
    #[arg(long, value_enum, default_value_t = LogLevelCli::Info)]
    log_level: LogLevelCli,
    /// Target directory to migrate (defaults to current directory)
    #[arg(long)]
    target: Option<std::path::PathBuf>,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum LogLevelCli {
    Quiet,
    Info,
    Verbose,
    Debug,
}

impl From<LogLevelCli> for LogLevel {
    fn from(level: LogLevelCli) -> Self {
        match level {
            LogLevelCli::Quiet => LogLevel::Quiet,
            LogLevelCli::Info => LogLevel::Info,
            LogLevelCli::Verbose => LogLevel::Verbose,
            LogLevelCli::Debug => LogLevel::Debug,
        }
    }
}

/// Shell persona: OgShell (safe/view) vs NeuShell (auto-apply/auto-heal).
#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum ShellMode {
    OgShell,
    NeuShell,
}

impl ShellMode {
    fn autonomous_apply(self) -> bool {
        matches!(self, ShellMode::NeuShell)
    }
}

/// Configuration passed to the TUI runtime.
#[derive(Clone)]
struct TuiConfig {
    shell_mode: ShellMode,
    #[cfg(feature = "nats")]
    nats_status: NatsStatus,
}

#[cfg(feature = "nats")]
#[derive(Clone, Debug)]
pub struct NatsStatus {
    connected: bool,
    url: Option<String>,
}

#[cfg(feature = "nats")]
impl NatsStatus {
    fn describe(&self) -> String {
        match (self.connected, self.url.as_deref()) {
            (true, Some(url)) => format!("connected ({url})"),
            (true, None) => "connected (url unknown)".to_string(),
            (false, Some(url)) => format!("unreachable ({url})"),
            (false, None) => "disabled (no NATS_URL set)".to_string(),
        }
    }
}

#[cfg(feature = "yoshell")]
/// TUI Application state
pub struct App {
    /// Current selected menu item
    pub selected_menu: usize,
    /// Current screen/page
    pub current_screen: Screen,
    /// Should the app quit
    pub should_quit: bool,
    /// Current color scheme
    pub color_scheme: ColorScheme,
    /// List state for scrollable menus
    pub list_state: ListState,
    /// The core of the runtime: A fully integrated YoshiSystem.
    pub yoshi_system: Arc<YoshiSystem>,
    /// System health metrics
    pub system_health: SystemHealth,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Last update time
    pub last_update: Instant,
    /// Demo error log
    pub error_log: Vec<String>,
    /// Scrollbar state for error log
    pub scrollbar_state: ScrollbarState,
    /// Current scroll position
    pub scroll_position: usize,
    /// Background image original (for ratatui-image protocol)
    pub bg_image: Option<image::DynamicImage>,
    /// Ratatui-image protocol for rendering the background
    pub bg_protocol: Option<StatefulProtocol>,
    /// Cached size used for potential future tuning (currently unused)
    pub bg_last_size: Option<(u16, u16)>,
    /// Whether autonomous code apply is active (NeuShell)
    pub autonomous_apply: bool,
    /// NATS connectivity status (only with `nats` feature)
    #[cfg(feature = "nats")]
    pub nats_status: NatsStatus,
}

#[cfg(feature = "yoshell")]
/// Application screens/pages
#[derive(Debug, Clone, PartialEq)]
pub enum Screen {
    /// Main menu
    MainMenu,
    /// Autonomous recovery demo
    RecoveryDemo,
    /// Circuit breaker management
    CircuitBreakerManagement,
    /// Supervision tree control
    SupervisionControl,
    /// System monitoring
    SystemMonitoring,
    /// Error pattern analysis
    ErrorAnalysis,
    /// Settings and configuration
    Settings,
    /// Help/About screen
    Help,
    /// Exit confirmation
    ExitConfirmation,
}

#[cfg(feature = "yoshell")]
/// Main menu items: (icon, label)
const MAIN_MENU_ITEMS: &[(&str, &str)] = &[
    ("🌐", "NATS Distributed System Monitoring"),
    ("🤖", "Autonomous Recovery Assistance"),
    ("⚡", "Circuit Breaker Management"),
    ("🏗️", "Supervision Tree Control"),
    ("🧠", "Error Pattern Analysis"),
    ("⚙️", "Settings"),
    ("❓", "Help"),
    ("🚪", "Exit"),
];

#[cfg(feature = "yoshell")]
/// Theme variants for user selection
#[derive(Debug, Clone, PartialEq)]
pub enum ThemeVariant {
    /// OgShell retro gaming theme
    OgShell,
    /// NeuShell enhanced theme
    NeuShell,
}

#[cfg(feature = "yoshell")]
/// Color scheme for retro gaming feel
#[derive(Clone)]
pub struct ColorScheme {
    /// Primary theme color
    pub primary: Color,
    /// Secondary theme color
    pub secondary: Color,
    /// Accent color for highlights
    pub accent: Color,
    /// Background color
    pub background: Color,
    /// Text color
    pub text: Color,
    /// Success state color
    pub success: Color,
    /// Warning state color
    pub warning: Color,
    /// Error state color
    pub error: Color,
    /// Theme variant
    pub variant: ThemeVariant,
}

#[cfg(feature = "yoshell")]
impl Default for ColorScheme {
    fn default() -> Self {
        Self::ogshell()
    }
}

#[cfg(feature = "yoshell")]
impl ColorScheme {
    /// OgShell retro gaming theme
    pub fn ogshell() -> Self {
        Self {
            primary: Color::Rgb(0, 255, 127),    // Bright green
            secondary: Color::Rgb(255, 215, 0),  // Gold
            accent: Color::Rgb(255, 105, 180),   // Hot pink
            background: Color::Rgb(25, 25, 112), // Midnight blue
            text: Color::Rgb(245, 245, 245),     // White smoke
            success: Color::Rgb(50, 205, 50),    // Lime green
            warning: Color::Rgb(255, 165, 0),    // Orange
            error: Color::Rgb(255, 99, 71),      // Tomato
            variant: ThemeVariant::OgShell,
        }
    }

    /// NeuShell enhanced theme
    pub fn neushell() -> Self {
        Self {
            primary: Color::Rgb(0, 255, 191),   // Nova Cyan
            secondary: Color::Rgb(191, 0, 255), // Mythic Purple
            accent: Color::Rgb(255, 191, 0),    // Neon Gold
            background: Color::Rgb(13, 13, 32), // Deep Void
            text: Color::Rgb(235, 235, 255),    // Ethereal White
            success: Color::Rgb(0, 255, 127),   // Quantum Green
            warning: Color::Rgb(255, 127, 0),   // Plasma Orange
            error: Color::Rgb(255, 63, 127),    // Crimson Flux
            variant: ThemeVariant::NeuShell,
        }
    }

    /// Toggle between themes
    pub fn toggle_theme(&mut self) {
        *self = match self.variant {
            ThemeVariant::OgShell => Self::neushell(),
            ThemeVariant::NeuShell => Self::ogshell(),
        };
    }
}

#[cfg(feature = "nats")]
fn configure_nats(cli_url: &Option<String>) {
    if let Some(url) = cli_url {
        unsafe {
            env::set_var("NATS_URL", url);
        }
    } else if env::var("NATS_URL").is_err() {
        unsafe {
            env::set_var("NATS_URL", "nats://127.0.0.1:4222");
        }
    }
}

#[cfg(feature = "nats")]
fn detect_nats_status(system: &YoshiSystem) -> NatsStatus {
    NatsStatus {
        connected: system.nats.is_some(),
        url: env::var("NATS_URL").ok(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    #[cfg(feature = "nats")]
    configure_nats(&cli.nats_url);

    let shell_mode = cli.mode;
    set_autonomous_apply(shell_mode.autonomous_apply());

    let yoshi_system = Arc::new(YoshiSystem::with_workers(2, 100).await?);
    #[cfg(feature = "nats")]
    let nats_status = detect_nats_status(&yoshi_system);

    yoshi_system.start().await?;
    println!(
        "Yoshi systems initialized. Mode: {:?} (autonomous apply: {})",
        shell_mode,
        shell_mode.autonomous_apply()
    );
    println!("Starting runtime host...");
    #[cfg(feature = "nats")]
    println!("NATS status: {}", nats_status.describe());

    match cli.command {
        #[cfg(feature = "yoshell")]
        Commands::Tui => {
            let config = TuiConfig {
                shell_mode,
                #[cfg(feature = "nats")]
                nats_status,
            };
            run_tui(yoshi_system.clone(), config).await?
        }
        Commands::Supervisor(SupervisorArgs { action }) => match action {
            SupervisorAction::Status => {
                if let Some(supervisor) = &yoshi_system.supervisor {
                    println!("{:#?}", supervisor.status());
                } else {
                    println!("Supervisor not configured.");
                }
            }
        },
        Commands::Circuit(CircuitArgs { action }) => match action {
            CircuitAction::ForceOpen => {
                println!("Circuit breaker: ForceOpen placeholder - COMING SOON");
            }
            CircuitAction::Reset => {
                println!("Circuit breaker: Reset placeholder - COMING SOON");
            }
        },
        Commands::Migrate(args) => {
            handle_migration_command(args).await?
        },
    }

    yoshi_system.stop().await.ok();
    Ok(())
}

/// Handle migration commands
async fn handle_migration_command(args: MigrateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let target_dir = args.target.unwrap_or_else(|| std::env::current_dir().unwrap());
    
    println!("🔄 Starting migration process...");
    println!("📁 Target directory: {:?}", target_dir);
    
    if !args.apply {
        println!("⚠️  DRY-RUN mode: No files will be modified");
    } else {
        println!("✅ APPLY mode: Files will be modified");
    }
    
    if !args.no_backup && args.apply {
        println!("💾 Backup creation enabled");
    } else if args.no_backup {
        println!("⚠️  Backup creation disabled");
    }

    // Build migration configuration from CLI arguments
    let mut config = MigrationConfig::default();
    config = config.with_apply_mode(args.apply);
    config = config.with_backup_enabled(!args.no_backup);
    config.log_level = args.log_level.into();
    
    if args.no_fmt || args.no_clippy || args.no_check {
        config.run_fmt = !args.no_fmt;
        config.run_clippy = !args.no_clippy;
        config.run_check = !args.no_check;
        println!("🔧 Quality gates: fmt={}, clippy={}, check={}", 
                 config.run_fmt, config.run_clippy, config.run_check);
    }
    
    if args.no_cleanup {
        config.strip_after_success = false;
        println!("🧹 Post-migration cleanup disabled");
    }

    // Create and run migrator
    let migrator = Migrator::with_config(target_dir, config);
    
    println!("🚀 Executing migration...");
    let report = migrator.migrate()?;
    
    // Display migration summary
    let summary = report.summary();
    println!("\n📊 Migration Summary:");
    println!("   Files processed: {}", summary.total_files);
    println!("   Files with changes: {}", summary.files_with_changes);
    println!("   Changes applied: {}", summary.changes_applied);
    println!("   Quality gates: {}", if summary.quality_gates_passed { "PASSED" } else { "FAILED" });
    println!("   Migration successful: {}", if summary.migration_successful { "YES" } else { "NO" });
    
    if !report.errors.is_empty() {
        println!("\n❌ Errors encountered:");
        for error in &report.errors {
            println!("   • {}", error);
        }
    }
    
    if !report.warnings.is_empty() {
        println!("\n⚠️  Warnings:");
        for warning in &report.warnings {
            println!("   • {}", warning);
        }
    }
    
    if summary.migration_successful {
        println!("\n🎉 Migration completed successfully!");
        if args.apply {
            println!("💡 Your codebase has been migrated from anyhow/thiserror to Yoshi stack");
        }
    } else {
        println!("\n💥 Migration failed!");
        if args.apply && !args.no_backup {
            println!("🔄 Backups have been restored automatically");
        }
        return Err("Migration failed".into());
    }
    
    Ok(())
}

#[cfg(feature = "yoshell")]
async fn run_tui(yoshi_system: Arc<YoshiSystem>, config: TuiConfig) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut list_state = ListState::default();
    list_state.select(Some(0));

    // Load and prepare background image + ratatui-image protocol
    let bg_image = load_background_image();
    let bg_protocol: Option<StatefulProtocol> = bg_image.as_ref().map(|img| {
        // Use a fixed font size; ratatui-image will handle protocol details.
        let picker = Picker::from_fontsize((8, 16));
        picker.new_resize_protocol(img.clone())
    });
    let color_scheme = match config.shell_mode {
        ShellMode::OgShell => ColorScheme::default(),
        ShellMode::NeuShell => ColorScheme::neushell(),
    };

    let app = Arc::new(Mutex::new(App {
        selected_menu: 0,
        current_screen: Screen::MainMenu,
        should_quit: false,
        color_scheme,
        autonomous_apply: config.shell_mode.autonomous_apply(),
        list_state,
        yoshi_system: yoshi_system.clone(),
        system_health: system_health(),
        performance_metrics: performance_metrics(),
        last_update: Instant::now(),
        error_log: vec![
            "🚀 Yoshi TUI initialized successfully".to_string(),
            "🔧 Recovery engine started".to_string(),
            "📡 System monitoring active".to_string(),
        ],
        scrollbar_state: ScrollbarState::default(),
        scroll_position: 0,
        bg_image,
        bg_protocol,
        bg_last_size: None,
        #[cfg(feature = "nats")]
        nats_status: config.nats_status,
    }));

    {
        // Ensure autonomous apply flag matches the initial theme.
        let mut guard = app.lock().unwrap();
        #[cfg(feature = "nats")]
        {
            let nats_desc = guard.nats_status.describe();
            guard.error_log.push(format!("📡 NATS {}", nats_desc));
        }
        guard.apply_theme_side_effects();
    }

    // Spawn a background task to simulate a workload and update the app state
    let app_clone = app.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let should_stop = {
                let mut app_guard = app_clone.lock().unwrap();
                if app_guard.should_quit {
                    true
                } else {
                    app_guard.update();
                    false
                }
            };
            if should_stop {
                break;
            }

            // Simulate a workload that generates errors
            if random::<f32>() > 0.6 {
                let error_kind = match random::<u8>() % 3 {
                    0 => AppErrorKind::Timeout {
                        message: "API call timed out".to_string(),
                        context: yoshi_std::TimeoutContext {
                            operation: "api_call".into(),
                            timeout_duration_ms: 5000,
                            elapsed_time_ms: 6000,
                            bottleneck_analysis: None,
                            optimization_hints: vec![],
                        },
                    },
                    1 => AppErrorKind::Validation {
                        message: "Invalid user input".to_string(),
                        context: yoshi_std::ValidationInfo::new()
                            .with_parameter("input")
                            .with_expected("valid string")
                            .with_actual("invalid input")
                            .with_rule("validation"),
                    },
                    _ => AppErrorKind::Internal("Internal processing error".to_string()),
                };
                let app_err = app_error!(error_kind);

                // This is the magic: we feed the error to the live recovery system.
                let _recovered: String = Err::<String, YoError>(app_err)
                    .into_yoshi()
                    .auto_recover_with_context("live_tui_workload")
                    .await;

                let mut app_guard = app_clone.lock().unwrap();
                app_guard.error_log.insert(
                    0,
                    format!(
                        "{} | {} | ✅ Auto-Recovered",
                        Local::now().format("%H:%M:%S"),
                        "live_tui_workload"
                    ),
                );
                app_guard.error_log.truncate(100);
            }
        }
    });

    let res = run_app_loop(&mut terminal, app.clone()).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("TUI error: {err:?}");
    }

    yoshi_system.stop().await.ok();
    Ok(())
}

#[cfg(feature = "yoshell")]
async fn run_app_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    app: Arc<Mutex<App>>,
) -> io::Result<()> {
    loop {
        {
            let mut app = app.lock().unwrap();
            terminal.draw(|f| ui(f, &mut app))?;
            if app.should_quit {
                return Ok(());
            }
        }

        if event::poll(Duration::from_millis(100))? {
            let event = event::read()?;
            match event {
                Event::Key(key) => {
                    app.lock().unwrap().handle_key_event(key);
                }
                Event::Mouse(mouse) => {
                    app.lock().unwrap().handle_mouse_event(mouse);
                }
                _ => {}
            }
        }
    }
}

#[cfg(feature = "yoshell")]
fn ui(f: &mut Frame, app: &mut App) {
    let colors = app.color_scheme.clone();

    match app.current_screen {
        Screen::MainMenu => tui_render::render_main_menu(f, app, &colors),
        Screen::RecoveryDemo => tui_render::render_recovery_demo(f, app, &colors),
        Screen::CircuitBreakerManagement => {
            tui_render::render_circuit_breaker_management(f, app, &colors)
        }
        Screen::SupervisionControl => tui_render::render_supervision_control(f, app, &colors),
        Screen::SystemMonitoring => tui_render::render_system_monitoring(f, app, &colors),
        Screen::ErrorAnalysis => tui_render::render_error_analysis(f, app, &colors),
        Screen::Settings => tui_render::render_settings(f, app, &colors),
        Screen::Help => tui_render::render_help(f, app, &colors),
        Screen::ExitConfirmation => tui_render::render_exit_confirmation(f, app, &colors),
    }
}

#[cfg(feature = "yoshell")]
impl App {
    /// Apply side effects when switching themes (NeuShell = autonomous apply on).
    fn apply_theme_side_effects(&mut self) {
        let neu_enabled = matches!(self.color_scheme.variant, ThemeVariant::NeuShell);
        self.autonomous_apply = neu_enabled;
        set_autonomous_apply(neu_enabled);

        if neu_enabled {
            self.error_log.push(
                "⚠️ NeuShell: Autonomous ML + Corrector auto-apply ENABLED. Review diffs.".into(),
            );
        } else {
            self.error_log
                .push("🛡️ OgShell: Autonomous apply disabled; suggestions only.".into());
        }

        self.error_log.push(format!(
            "🎨 Theme switched to: {:?}",
            self.color_scheme.variant
        ));
    }

    /// Handle keyboard events
    pub fn handle_key_event(&mut self, key: KeyEvent) {
        if key.kind == KeyEventKind::Press {
            match self.current_screen {
                Screen::MainMenu => self.handle_main_menu_key(key),
                Screen::ExitConfirmation => self.handle_exit_confirmation_key(key),
                _ => self.handle_general_key(key),
            }
        }
    }

    /// Handle mouse events
    pub fn handle_mouse_event(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.handle_mouse_click(mouse.column, mouse.row);
            }
            MouseEventKind::ScrollUp => {
                self.scroll_up();
            }
            MouseEventKind::ScrollDown => {
                self.scroll_down();
            }
            _ => {}
        }
    }

    /// Handle mouse clicks
    fn handle_mouse_click(&mut self, x: u16, y: u16) {
        match self.current_screen {
            Screen::MainMenu => {
                // Approximate menu area
                if x > 5 && y > 5 && y < 5 + MAIN_MENU_ITEMS.len() as u16 {
                    let menu_index = (y - 5) as usize;
                    if menu_index < MAIN_MENU_ITEMS.len() {
                        self.selected_menu = menu_index;
                        self.list_state.select(Some(menu_index));
                        self.activate_selected_menu();
                    }
                }
            }
            _ => {
                // Back button area
                if x > 0 && x < 10 && y > 0 && y < 3 {
                    self.current_screen = Screen::MainMenu;
                }
            }
        }
    }

    /// Handle main menu keyboard navigation
    fn handle_main_menu_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected_menu > 0 {
                    self.selected_menu -= 1;
                    self.list_state.select(Some(self.selected_menu));
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected_menu < MAIN_MENU_ITEMS.len() - 1 {
                    self.selected_menu += 1;
                    self.list_state.select(Some(self.selected_menu));
                }
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.activate_selected_menu();
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                self.current_screen = Screen::ExitConfirmation;
            }
            KeyCode::Char('1') => self.quick_select(0),
            KeyCode::Char('2') => self.quick_select(1),
            KeyCode::Char('3') => self.quick_select(2),
            KeyCode::Char('4') => self.quick_select(3),
            KeyCode::Char('5') => self.quick_select(4),
            KeyCode::Char('6') => self.quick_select(5),
            KeyCode::Char('7') => self.quick_select(6),
            KeyCode::Char('8') => self.quick_select(7),
            KeyCode::Char('t') => {
                self.color_scheme.toggle_theme();
                self.apply_theme_side_effects();
            }
            _ => {}
        }
    }

    /// Handle exit confirmation
    fn handle_exit_confirmation_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                self.should_quit = true;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.current_screen = Screen::MainMenu;
            }
            _ => {}
        }
    }

    /// Handle general navigation keys
    fn handle_general_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Char('b') => {
                self.current_screen = Screen::MainMenu;
            }
            KeyCode::Char('q') => {
                self.current_screen = Screen::ExitConfirmation;
            }
            KeyCode::Char('r') => {
                self.refresh_data();
            }
            KeyCode::Char('t') => {
                self.color_scheme.toggle_theme();
                self.apply_theme_side_effects();
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                if self.current_screen == Screen::Settings {
                    self.color_scheme.toggle_theme();
                    self.apply_theme_side_effects();
                }
            }
            KeyCode::Up => self.scroll_up(),
            KeyCode::Down => self.scroll_down(),
            _ => {}
        }
    }

    /// Quick select menu item by number
    fn quick_select(&mut self, index: usize) {
        if index < MAIN_MENU_ITEMS.len() {
            self.selected_menu = index;
            self.list_state.select(Some(index));
            self.activate_selected_menu();
        }
    }

    /// Activate the currently selected menu item
    fn activate_selected_menu(&mut self) {
        self.current_screen = match self.selected_menu {
            0 => {
                self.run_recovery_demo();
                Screen::RecoveryDemo
            }
            1 => Screen::CircuitBreakerManagement,
            2 => Screen::SupervisionControl,
            3 => Screen::SystemMonitoring,
            4 => Screen::ErrorAnalysis,
            5 => Screen::Settings,
            6 => Screen::Help,
            7 => Screen::ExitConfirmation,
            _ => Screen::MainMenu,
        };
    }

    /// Scroll up in current view
    fn scroll_up(&mut self) {
        self.scroll_position = self.scroll_position.saturating_sub(1);
        self.scrollbar_state = self.scrollbar_state.position(self.scroll_position);
    }

    /// Scroll down in current view
    fn scroll_down(&mut self) {
        self.scroll_position = self.scroll_position.saturating_add(1);
        self.scrollbar_state = self.scrollbar_state.position(self.scroll_position);
    }

    /// Refresh system data
    fn refresh_data(&mut self) {
        self.system_health = system_health();
        self.performance_metrics = performance_metrics();
        self.last_update = Instant::now();
        self.error_log.push("🔄 Data refreshed".to_string());
    }

    /// Run autonomous recovery demonstration
    #[allow(dead_code)]
    fn run_recovery_demo(&mut self) {
        self.error_log
            .push("🤖 Starting autonomous recovery demo...".to_string());
        // This is now handled by the background task, just log the initiation.
    }

    /// Update function called periodically
    pub fn update(&mut self) {
        if self.last_update.elapsed() >= Duration::from_secs(1) {
            self.refresh_data();
        }
    }
}

// --- TUI Rendering Functions ---
#[cfg(feature = "yoshell")]
mod tui_render {
    use super::*;

    /// Render the background image using ratatui-image.
    /// The protocol itself adapts to the current frame area, so we just hand it `f.area()`.
    pub fn render_background(f: &mut Frame, app: &mut App) {
        if let Some(ref mut protocol) = app.bg_protocol {
            let area = f.area();
            // .resize(Resize::Scale(None)) forces the image to stretch to fill the area,
            // ignoring aspect ratio to cover the whole screen.
            let image_widget = StatefulImage::new().resize(Resize::Scale(None));
            f.render_stateful_widget(image_widget, area, protocol);
        }
    }

    pub fn render_main_menu(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        // Render background image first (if available)
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6),
                Constraint::Min(10),
                Constraint::Length(3),
            ])
            .split(f.area());

        let banner = create_theme_aware_banner(colors);
        f.render_widget(banner, chunks[0]);

        let menu_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        let items: Vec<ListItem> = MAIN_MENU_ITEMS
            .iter()
            .enumerate()
            .map(|(i, (icon, label))| {
                let style = if i == app.selected_menu {
                    Style::default()
                        .fg(colors.accent)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors.text)
                };

                // Fixed-width numeric column + label text, emoji at the end.
                let line = Line::from(vec![
                    Span::styled(format!("{:>2}. ", i + 1), style),
                    Span::styled(*label, style),
                    Span::raw("  "),
                    Span::styled(*icon, style),
                ]);

                ListItem::new(line)
            })
            .collect();

        let menu_list = List::new(items)
            .block(
                Block::default()
                    .title("🎯 Main Menu")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .highlight_style(
                Style::default()
                    .bg(colors.accent)
                    .fg(colors.background)
                    .add_modifier(Modifier::BOLD),
            );

        f.render_stateful_widget(menu_list, menu_chunks[0], &mut app.list_state);

        let system_status = create_system_status_widget(app, colors);
        f.render_widget(system_status, menu_chunks[1]);

        let theme_name = match colors.variant {
            ThemeVariant::OgShell => "OgShell",
            ThemeVariant::NeuShell => "NeuShell",
        };
        let instructions = Paragraph::new(format!("🎮 ↑↓/j,k move • Enter/Space select • 1-8 quick • t toggle theme ({}) • q quit • Mouse click", theme_name)).style(Style::default().fg(colors.text)).alignment(Alignment::Center).block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(colors.secondary)));
        f.render_widget(instructions, chunks[2]);
    }

    pub fn create_system_status_widget<'a>(app: &'a App, colors: &'a ColorScheme) -> Paragraph<'a> {
        let health = &app.system_health;
        let perf = &app.performance_metrics;

        let status_text = vec![
            Line::from(Span::styled(
                "📊 System Health",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("Errors: ", Style::default().fg(colors.text)),
                Span::styled(
                    format!("{}", health.error_count),
                    Style::default().fg(colors.accent),
                ),
            ]),
            Line::from(vec![
                Span::styled("Recovery: ", Style::default().fg(colors.text)),
                Span::styled(
                    format!("{:.1}%", health.recovery_success_rate * 100.0),
                    Style::default().fg(colors.success),
                ),
            ]),
            Line::from(vec![
                Span::styled("Detection: ", Style::default().fg(colors.text)),
                Span::styled(
                    format!("{:?}", perf.error_detection_latency),
                    Style::default().fg(colors.accent),
                ),
            ]),
        ];
        Paragraph::new(status_text)
            .block(
                Block::default()
                    .title("🚀 Live Status")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .wrap(Wrap { trim: true })
    }

    pub fn render_recovery_demo(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        // Render background first
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());
        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    🤖 Autonomous Recovery Demo",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        // Show ML recovery status
        let is_enabled = app.yoshi_system.recovery.is_enabled();

        let mut text = vec![
            Line::from(vec![Span::styled(
                "🧠 ML Recovery Engine",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::raw("Status: "),
                Span::styled(
                    if is_enabled { "ENABLED" } else { "DISABLED" },
                    Style::default()
                        .fg(if is_enabled {
                            colors.success
                        } else {
                            colors.error
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Recovery System Info:",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from("  • Pattern-Based Recovery"),
            Line::from("  • Learning-Based Recovery"),
            Line::from("  • Default Fallback Strategies"),
            Line::from(""),
            Line::from("Recovery learns from past attempts to improve"),
            Line::from("future error handling autonomously."),
        ];

        // Also show error log
        if !app.error_log.is_empty() {
            text.push(Line::from(""));
            text.push(Line::from(vec![Span::styled(
                "📝 Recent Events:",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )]));
            for log in app.error_log.iter().take(5) {
                text.push(Line::from(format!("  {}", log)));
            }
        }

        let paragraph = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .wrap(Wrap { trim: false });
        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_circuit_breaker_management(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());
        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    ⚡ Circuit Breaker Management",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        // Get circuit breaker state and metrics from yoshi_system
        let state = app.yoshi_system.circuit_breaker.state();
        let metrics = app.yoshi_system.circuit_breaker.metrics();

        let (state_str, state_color) = match state {
            CircuitState::Closed => ("CLOSED", colors.success),
            CircuitState::Open => ("OPEN", colors.error),
            CircuitState::HalfOpen => ("HALF-OPEN", colors.warning),
            CircuitState::ForcedOpen => ("FORCED OPEN", Color::Magenta),
        };

        let success_pct = if metrics.total_requests > 0 {
            (metrics.successful_requests * 100) / metrics.total_requests
        } else {
            0
        };

        let text = vec![
            Line::from(vec![
                Span::styled("State: ", Style::default().fg(colors.text)),
                Span::styled(
                    state_str,
                    Style::default()
                        .fg(state_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "📊 Metrics:",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(format!("  Total Requests: {}", metrics.total_requests)),
            Line::from(vec![
                Span::raw(format!("  Successful: {} (", metrics.successful_requests)),
                Span::styled(
                    format!("{}%", success_pct),
                    Style::default().fg(if success_pct >= 80 {
                        colors.success
                    } else {
                        colors.warning
                    }),
                ),
                Span::raw(")"),
            ]),
            Line::from(format!("  Failed: {}", metrics.failed_requests)),
            Line::from(format!("  Rejected: {}", metrics.rejected_requests)),
            Line::from(format!(
                "  Avg Response: {:?}",
                metrics.average_response_time
            )),
        ];

        let paragraph = Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_supervision_control(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());
        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    🏗️ Supervision Tree Control",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        let mut text = vec![
            Line::from(vec![Span::styled(
                "Supervisor Status",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
        ];

        // Get supervisor status if available
        if let Some(supervisor) = &app.yoshi_system.supervisor {
            let status = supervisor.status();

            let (status_str, status_color) = if status.is_running {
                ("RUNNING", colors.success)
            } else {
                ("STOPPED", colors.error)
            };

            text.push(Line::from(vec![
                Span::raw("State: "),
                Span::styled(
                    status_str,
                    Style::default()
                        .fg(status_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            text.push(Line::from(format!(
                "Total Workers: {}",
                status.worker_count
            )));
            text.push(Line::from(vec![
                Span::raw("Healthy Workers: "),
                Span::styled(
                    format!("{}/{}", status.healthy_workers, status.worker_count),
                    Style::default()
                        .fg(if status.healthy_workers == status.worker_count {
                            colors.success
                        } else {
                            colors.warning
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            text.push(Line::from(""));
            text.push(Line::from("Supervision Tree:"));
            text.push(Line::from("  • Automatic restart on failure"));
            text.push(Line::from("  • Health monitoring"));
            text.push(Line::from("  • Exponential backoff"));
        } else {
            text.push(Line::from(vec![Span::styled(
                "⚠ Supervisor not initialized",
                Style::default().fg(colors.warning),
            )]));
            text.push(Line::from(""));
            text.push(Line::from("Use SupervisorSystem::builder() to create"));
            text.push(Line::from("a supervisor tree with workers."));
        }

        let paragraph = Paragraph::new(text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_system_monitoring(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        // Render background first (requires mutable access to update protocol state)
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());

        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    📊 System Monitoring",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        // Real data from yoshi-std structs
        let health = &app.system_health;
        let perf = &app.performance_metrics;

        let mut lines = vec![
            Line::from(Span::styled(
                "📡 Live Telemetry Snapshot",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("Total Errors: ", Style::default().fg(colors.text)),
                Span::styled(
                    format!("{}", health.error_count),
                    Style::default().fg(colors.error),
                ),
            ]),
            Line::from(vec![
                Span::styled("Recovery Success: ", Style::default().fg(colors.text)),
                Span::styled(
                    format!("{:.1}%", health.recovery_success_rate * 100.0),
                    Style::default().fg(colors.success),
                ),
            ]),
            Line::from(vec![
                Span::styled("Detection Latency: ", Style::default().fg(colors.text)),
                Span::styled(
                    format!("{:?}", perf.error_detection_latency),
                    Style::default().fg(colors.accent),
                ),
            ]),
        ];

        #[cfg(feature = "nats")]
        {
            lines.push(Line::from(vec![
                Span::styled("NATS: ", Style::default().fg(colors.text)),
                Span::styled(
                    app.nats_status.describe(),
                    Style::default()
                        .fg(if app.nats_status.connected {
                            colors.success
                        } else {
                            colors.warning
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Press 'r' to refresh metrics. Circuit breaker and supervisor details live on their own screens.",
            Style::default().fg(colors.text),
        )));

        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .title("Runtime Metrics")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_error_analysis(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());

        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    🧠 Error Pattern Analysis",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        let mut lines = vec![
            Line::from(Span::styled(
                "Live Error Stream",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        if app.error_log.is_empty() {
            lines.push(Line::from(
                "No errors recorded yet. Yoshi is quietly keeping everything stable.",
            ));
        } else {
            // Simple windowed view of logs
            let view_height = chunks[1].height.saturating_sub(2) as usize;
            let start = app
                .scroll_position
                .min(app.error_log.len().saturating_sub(1));
            let end = (start + view_height).min(app.error_log.len());

            for (idx, entry) in app.error_log[start..end].iter().enumerate() {
                lines.push(Line::from(format!("{:>2}. {}", start + idx + 1, entry)));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Use ↑/↓ to scroll history.",
                Style::default().fg(colors.text),
            )));
        }

        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .title("Error Timeline")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_settings(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());
        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    ⚙️ Settings",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        let current_theme = match colors.variant {
            ThemeVariant::OgShell => "OgShell",
            ThemeVariant::NeuShell => "NeuShell",
        };

        let text = vec![
            Line::from(vec![Span::styled(
                "Theme",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::raw("Current: "),
                Span::styled(
                    current_theme,
                    Style::default()
                        .fg(colors.primary)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from("Press 't' or Enter/Space to toggle theme."),
            Line::from(vec![
                Span::styled(
                    "⚠ NeuShell",
                    Style::default()
                        .fg(colors.warning)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" enables ML + Corrector autonomous code apply."),
            ]),
            Line::from(vec![Span::styled(
                "Review diffs after NeuShell runs.",
                Style::default().fg(colors.warning),
            )]),
            Line::from("Use 'b' or Esc to go back."),
        ];

        let paragraph = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .wrap(Wrap { trim: false });
        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_help(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        render_background(f, app);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(f.area());

        let title_text = vec![Line::from(vec![
            Span::styled(
                "← Back (b)",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "    ❓ Help",
                Style::default()
                    .fg(colors.primary)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];
        let title_widget = Paragraph::new(title_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors.secondary)),
        );
        f.render_widget(title_widget, chunks[0]);

        let lines = vec![
            Line::from(Span::styled(
                "Keyboard",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from("  ↑ / ↓ or j / k   – Move selection / scroll"),
            Line::from("  1–8              – Jump directly to a main menu item"),
            Line::from("  Enter / Space    – Activate selection / toggle setting"),
            Line::from("  b / Esc          – Back to main menu"),
            Line::from("  q                – Open exit confirmation"),
            Line::from("  r                – Refresh live metrics"),
            Line::from(
                "  t                – Toggle OgShell / NeuShell theme (NeuShell = auto-apply)",
            ),
            Line::from(""),
            Line::from(Span::styled(
                "Mouse",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from("  Left click       – Activate menu entries / back button"),
            Line::from("  Scroll wheel     – Scroll error log and long views"),
            Line::from(""),
            Line::from(Span::styled(
                "Yoshi Runtime Shell",
                Style::default()
                    .fg(colors.accent)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from("  • Autonomous Recovery Demo shows live ML-based recovery events."),
            Line::from("  • Circuit Breaker Management reflects real circuit state + metrics."),
            Line::from("  • Supervision Tree Control mirrors supervisor + workers status."),
            Line::from("  • System Monitoring aggregates high-level runtime health signals."),
            Line::from("  • Error Pattern Analysis lets you inspect the live error stream."),
            Line::from(
                "  • NeuShell will apply ML/Corrector fixes automatically; OgShell is view-only.",
            ),
        ];

        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .title("Yoshi Shell – Help")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors.secondary)),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, chunks[1]);
    }

    pub fn render_exit_confirmation(f: &mut Frame, app: &mut App, colors: &ColorScheme) {
        // Render background first
        render_background(f, app);

        let area = centered_rect(50, 20, f.area());
        f.render_widget(Clear, area);
        let confirmation_text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "Are you sure you want to exit?",
                Style::default()
                    .fg(colors.text)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    "Y",
                    Style::default()
                        .fg(colors.success)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("es / ", Style::default().fg(colors.text)),
                Span::styled(
                    "N",
                    Style::default()
                        .fg(colors.error)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("o", Style::default().fg(colors.text)),
            ]),
        ];
        let confirmation = Paragraph::new(confirmation_text)
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .title("🚪 Exit Yoshi")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Thick)
                    .border_style(Style::default().fg(colors.error)),
            )
            .style(Style::default().bg(colors.background));
        f.render_widget(confirmation, area);
    }

    fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
        let popup_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ])
            .split(r);
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ])
            .split(popup_layout[1])[1]
    }

    fn create_theme_aware_banner(colors: &ColorScheme) -> Paragraph<'_> {
        match colors.variant {
            ThemeVariant::OgShell => create_ogshell_banner(colors),
            ThemeVariant::NeuShell => create_neushell_banner(colors),
        }
    }

    fn create_ogshell_banner(colors: &ColorScheme) -> Paragraph<'_> {
        Paragraph::new(vec![
            Line::from(Span::styled(
                "╔══════════════════════════════════════════════════╗",
                Style::default().fg(colors.accent),
            )),
            Line::from(vec![
                Span::styled("║  ", Style::default().fg(colors.accent)),
                Span::styled(
                    "🌙 YOSHI - Autonomous Error Handling Framework",
                    Style::default()
                        .fg(colors.primary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ║", Style::default().fg(colors.accent)),
            ]),
            Line::from(Span::styled(
                "╚══════════════════════════════════════════════════╝",
                Style::default().fg(colors.accent),
            )),
        ])
        .alignment(Alignment::Center)
    }

    fn create_neushell_banner(colors: &ColorScheme) -> Paragraph<'_> {
        Paragraph::new(vec![
            Line::from(Span::styled(
                "╔═════════════════════════════════════════════════════════════════╗",
                Style::default().fg(colors.accent),
            )),
            Line::from(vec![
                Span::styled("║  ", Style::default().fg(colors.accent)),
                Span::styled("✨", Style::default().fg(colors.secondary)),
                Span::styled(
                    " YOSHI ",
                    Style::default()
                        .fg(colors.primary)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("◈", Style::default().fg(colors.accent)),
                Span::styled(
                    " NeuShell ",
                    Style::default()
                        .fg(colors.secondary)
                        .add_modifier(Modifier::ITALIC),
                ),
                Span::styled("◈", Style::default().fg(colors.accent)),
                Span::styled(
                    " Autonomous Error Framework ",
                    Style::default().fg(colors.primary),
                ),
                Span::styled("✨", Style::default().fg(colors.secondary)),
                Span::styled("  ║", Style::default().fg(colors.accent)),
            ]),
            Line::from(Span::styled(
                "╚═════════════════════════════════════════════════════════════════╝",
                Style::default().fg(colors.accent),
            )),
        ])
        .alignment(Alignment::Center)
    }
}
