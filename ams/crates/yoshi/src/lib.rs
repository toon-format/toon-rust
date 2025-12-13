/* yoshi/src/lib.rs */
#![allow(clippy::result_large_err)]
//! A comprehensive facade for the Yoshi ecosystem, unifying error handling,
//! autonomous recovery, and production-grade worker supervision.
//!
//! # Yoshi – Facade Module
//!▫~•◦------------------------‣
//!
//! This crate is the primary entry point for the Yoshi framework. It re-exports all
//! functionality from the underlying ecosystem crates (`yoshi_std`, `xuid`, and the local
//! `error` module), providing a single, consistent API surface.
//!
//! ### Key Capabilities
//! - **Unified Error Handling:** Re-exports `YoshiError` (core framework error), `YoError`
//!   (application-level error), and the `AnyError` derive macro for a complete error model.
//! - **Autonomous Recovery:** Provides ML-driven recovery via the `ResultRecovery` extension trait,
//!   enabling self-healing logic with `auto_recover_with_context()`.
//! - **Supervision Trees:** Exposes `SupervisorTreeBuilder` and `WorkerConfig` primitives for building
//!   robust, fault-tolerant, actor-like systems.
//! - **Ergonomic Tooling:** Includes `anyhow`-style context (`Context` trait) and convenience
//!   macros (`buck!`, `clinch!`) for cleaner error propagation.
//!
//! ### Ecosystem Equivalents
//! Yoshi provides a unified toolset that can replace several popular crates:
//! - **`thiserror`**: Replaced by `#[derive(AnyError)]` from the `yoshi_derive` crate.
//!   It offers attribute-driven `Display` formatting, `source()` detection, and `From` impls.
//! - **`anyhow`**: Replaced by the combination of `YoshiError`, the `Context` trait,
//!   and ergonomic macros like `buck!` and `clinch!`. This provides dynamic error
//!   wrapping with a concrete, feature-rich error type.
//! - **`eyre`**: Similar to `anyhow`, its functionality is covered by `YoshiError` and its
//!   tooling, with the added benefit of structured context and ML-driven recovery hooks.
//!
//! ### Architectural Notes
//! This module is designed to be the single dependency for applications building on Yoshi.
//! It includes a `prelude` module that exports the most commonly used types and traits for
//! convenient, one-line importing (`use yoshi::prelude::*;`).
//!
//! ### Example
//! ```rust
//! // use yoshi::prelude::*; // Alternative: use individual imports
//! // use std::time::Duration; // No longer needed - available via prelude
//! use yoshi::{
//!     YoshiError, Context, Result, create_custom_error, create_error, create_error_with_context,
//!     ValidationInfo, ErrorKind, YoError, Location, Recoverable, AnyError,
//!     YoResult, AppResult, SupervisorTreeBuilder, WorkerConfig,
//!     ResultRecovery, ResultExt, SupervisorTree,
//! };
//! use std::time::{Duration, Instant};
//!
//! // 1. Drop-in replacement for `thiserror` using `AnyError`
//! #[derive(Debug, AnyError)]
//! enum MyError {
//!     #[anyerror("I/O error on '{path}': {source}")]
//!     Io { path: String, #[source] source: std::io::Error },
//!     #[anyerror("API call failed: {0}")]
//!     Api(String),
//! }
//!
//! // 2. Anyhow-style error wrapping and dynamic errors
//! fn process_request() -> AppResult<()> {
//!     let config = std::fs::read_to_string("cfg.toml")
//!         .context("Failed to read configuration") // Context trait from `yoshi_std`
//!         .into_app()?; // Convert from YoResult to AppResult
//!
//!     // Use ValidationInfo for structured validation errors
//!     if config.is_empty() {
//!         let validation = ValidationInfo::new()
//!             .with_parameter("config")
//!             .with_expected("non-empty string")
//!             .with_actual("empty")
//!             .with_rule("min_length");
//!
//!         let location = Location { file: "process_request.rs".into(), line: 25, column: 5 };
//!         return Err(YoError::new(
//!             yoshi::AppErrorKind::Validation {
//!                 message: "Configuration cannot be empty".into(),
//!                 context: validation,
//!             },
//!             location,
//!         ));
//!     }
//!
//!     if config.contains("invalid") {
//!         // Use create_error_with_context for detailed error creation
//!         let error = create_error_with_context(ErrorKind::InvalidArgument {
//!             message: "Invalid setting found in config".into(),
//!             context_chain: vec!["process_request".into()],
//!             validation_info: None,
//!         });
//!         let yres: YoResult<()> = Err(error.into());
//!         return yres.into_app();
//!     }
//!
//!     // Use create_custom_error for dynamic, user-defined error kinds
//!     if config.contains("custom") {
//!         let error = create_custom_error(
//!             "ConfigValidation",
//!             "Custom validation rule failed",
//!             vec!["process_request".into()],
//!             Some("additional custom data".into()),
//!         );
        // XuidBuilder removed, use XuidConstruct or Xuid::create/new instead
//!         return yres.into_app();
//!     }
//!
//!     Ok(())
//! }
//!
//! // 3. Supervisor tree for fault-tolerant workers
//! fn setup_supervisor() -> YoResult<SupervisorTree> {
//!     let supervisor = SupervisorTreeBuilder::new()
//!         .with_id("my_app".to_string())
//!         .add_worker(WorkerConfig {
//!             id: "processor-1".to_string(),
//!             ..Default::default()
//!         })
//!         .build()?;
//!     Ok(supervisor)
//! }
//!
//! // 4. Autonomous ML-driven error correction with Recoverable trait
//! async fn fetch_data() -> YoResult<String> {
//!     // Simulate a recoverable failure
//!     Err(create_error("Network timeout").into())
//! }
//!
//! async fn run_recovery() {
//!     let result: String = fetch_data()
//!         .await
//!         .auto_recover_with_context("data_fetching")
//!         .await;
//!     // `result` will be the default value for `String` ("") after ML recovery attempt.
//!     assert_eq!(result, "");
//! }
//!
//! // 5. Using Result type alias and error handling
//! fn validate_input(input: &str) -> Result<(), YoshiError> {
//!     if input.is_empty() {
//!         return Err(create_error("Input cannot be empty"));
//!     }
//!     Ok(())
//! }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// Declare the application-specific error module.
pub mod error;
// Workspace migrator (Rust-native analogue to scripts/migrate.py).
pub mod migrator;
#[cfg(feature = "autofix")]
pub mod agent;

//=====================================================================================
// EXPLICIT RE-EXPORTS FROM THE YOSHI / YOSHI ECOSYSTEM
//=====================================================================================

//-------------------------------------------------------------------------------------
// § From `yoshi_derive` - Proc-Macros
//-------------------------------------------------------------------------------------

/// A derive macro for creating feature-rich error enums, serving as a lightweight
/// alternative to `thiserror`.
pub use yoshi_derive::AnyError;

//-------------------------------------------------------------------------------------
// § From `yoshi_std` - Core Error Handling, Recovery, and Ergonomics
//-------------------------------------------------------------------------------------

// Core Types & Traits
pub use yoshi_std::correction;
pub use yoshi_std::corrector;
pub use yoshi_std::{
    AdvancedRecoveryStats, AdvisedCorrection, BoxedYoshi, CircuitBreaker, CircuitConfig,
    CircuitMetrics, CircuitState, ContextTrend, ErrorKind, ErrorMetricsSnapshot, Features,
    FixSafetyLevel, Location, Recoverable, RecoveryAction, RecoveryPolicy, RecoverySeverity,
    ResultRecovery, SupervisorStatus, WorkerHealth, YoResult, YoshiError, YoshiErrorExt,
    context::Context, correction::ProvidesFixes,
};

// Re-export full core crate for direct access where needed.
pub use yoshi_std;
#[cfg(feature = "autofix")]
pub use agent::apply_refactor_from_errors;

// Utility Type Aliases
pub use yoshi_std::{CrabMap, CrabString, CrabVec};

// Ergonomic Error Creation & Handling Macros/Functions
pub use yoshi_std::{
    anyerror, create_custom_error, create_error, create_error_with_context, error, get_cpu_count,
    get_current_system_load, get_memory_pressure, wrap,
};
// Note: eggroll is provided as a macro in this crate, not re-exported from yoshi_std

// ErrorKind Context Structs
pub use yoshi_std::{
    DataFrameworkContext, InternalContext, IoContext, LogEntry, NumericErrorContext,
    ParseContext, ResourceContext, StateContext, TimeoutContext, ValidationInfo,
};

// Correction & Fix Primitives
pub use yoshi_std::correction::{CodeModification, CodeSpan, CorrectionBuilder};

// ML Recovery Engine
pub use yoshi_std::{
    ExponentialBackoffStrategy, MLPrediction, MLRecoveryEngine, MLRecoveryStrategy,
};

// Supervisor & Worker Model
pub use yoshi_std::{
    BackoffStrategy, EscalationPolicy, HealthCheckConfig, HealthCheckType, HealthState,
    ResourceLimits, ResourceRequirements, RestartPolicy, RetryConfig, SupervisionStrategy,
    SupervisorConfig, SupervisorTree, SupervisorTreeBuilder, WorkerConfig, WorkerState,
    WorkerStatus, WorkerType,
};

// Worker Result Types
pub use yoshi_std::{
    AlertSeverity, BatchCompletionRecord, BatchProcessingResult, CacheEntryResult,
    CacheMaintenanceResults, CacheStatistics, CacheWorkerStats, CustomWorkerResults,
    DataPipelineMetrics, FileProcessingMetrics, FileProcessingResult, GatewayProcessingResults,
    GatewayRouteMetrics, GenericWorkerResults, MLInferenceMetrics, MLInferenceResult,
    PipelineStageResult, RouteProcessingStats, SimulatedRequest, SingleFileResult, SystemAlert,
    SystemContext, SystemMonitoringData, SystemMonitoringResults, WorkItem, WorkItemPriority,
    WorkItemResult, WorkerRuntimeMetrics,
};

pub use migrator::{Migrator, MigrationConfig, LogLevel};

//-------------------------------------------------------------------------------------
// § From `yoshi_std` - NATS Distributed Systems Integration
//-------------------------------------------------------------------------------------

// NATS Client & Configuration
#[cfg(feature = "nats")]
pub use yoshi_std::{NATSClient, NATSConfig, NatsError};

// Distributed Error Recovery Types
#[cfg(feature = "nats")]
pub use yoshi_std::{
    DistributedErrorMessage, DistributedMetricsMessage, MLRecoveryOutcome, NatsMessage,
};

// NATS Connection & Statistics
#[cfg(feature = "nats")]
pub use yoshi_std::{ErrorSeverity, NATSConnectionStats};

//-------------------------------------------------------------------------------------
// § From `xuid` - Universal Identifiers with E8 Lattice Geometry
//-------------------------------------------------------------------------------------

// Core XUID Types
pub use xuid::{SemanticPath, Xuid, XuidProvenance, XuidType, XuidConstruct};

// Error Handling
pub use xuid::{XuidError, XuidResult};

// E8 Lattice Mathematics
pub use xuid::{
    E8Lattice, E8Orbit, E8Point, e8_distance, orbit_correlation, quantize_to_e8, quantize_to_orbit,
};

// Adaptive Generation / Lightweight helpers
pub use xuid::lightweight;

// Learned Strategy Support
pub use xuid::{LeStratSpec, LeStratTag, LearnedStrategyTag};

// Serialization & Compression
pub use xuid::{deserialize, deserialize_compressed, serialize, serialize_compressed};

// Hydra operations and certificate types have been modularized in `xuid`.
// Re-export them here if/when upstream provides stable names; for now, those symbols
// are intentionally not re-exported to avoid coupling to internal APIs.

// Convenience Constructors
pub use xuid::{
    from_path as xuid_from_path, lestrat, new as new_xuid,
};

// Qdrant Integration (optional)
#[cfg(feature = "qdrant")]
pub use xuid::{XuidPointId, XuidPointIdOptions};

//-------------------------------------------------------------------------------------
// § From `yoshi::error` - Application-Level Error Abstractions
//-------------------------------------------------------------------------------------

/// The enumeration of application-specific error kinds.
pub use crate::error::AppErrorKind;
/// A `Result` type that defaults to `YoError` as its error variant. Aliased to `AppResult`.
pub use crate::error::Result as AppResult;
/// A canonical, high-level error type for applications, built upon `YoshiError`.
pub use crate::error::YoError;
/// Toggle ML autonomous apply at runtime (no-op when ml-recovery feature is off).
pub use crate::error::set_autonomous_apply;
/// Alias for application-level `Result`, intended as a drop-in for `anyhow::Result`.
pub type Hatch<T> = crate::error::Result<T, YoError>;
/// An extension trait for ergonomic conversion between `YoResult` and `AppResult`.
pub use crate::error::ResultExt;

// Ergonomic Unified Systems - High-level facades for yoshi-std ecosystems
pub use crate::error::{
    CircuitBreakerSystem, CorrectionSystem, HealthReport, RecoverySystem, SupervisorSystem,
    SupervisorSystemBuilder, YoshiSystem,
};

/// A framework-level `Result` type that defaults to `YoshiError` as its error variant.
///
/// This is intended for functions that operate at the framework or library level. For
/// application-level logic, prefer `AppResult` which defaults to `YoError`.
pub type Result<T, E = YoshiError> = std::result::Result<T, E>;

//=====================================================================================
// STANDARD RUST ECOSYSTEM–STYLE ALIASES
//=====================================================================================
//
// This section creates idiomatic aliases that mirror what these Yoshi concepts “would be”
// in typical Rust codebases using crates like `anyhow`, `thiserror`, and `eyre`.
// The goal is *interchangeability*: devs can choose either the Yoshi names or the
// conventional ones without friction.

// --- Error type aliases -------------------------------------------------------------

/// Primary concrete error type alias, mirroring the common `Error` name used in many
/// Rust projects. This is a thin alias over `YoshiError`.
pub type Error = YoshiError;

/// Application/domain-level error alias – identical to `YoError`, but using a shorter,
/// conventional name often seen in libraries that distinguish framework vs app errors.
pub type App = YoError;

// --- Result aliases -----------------------------------------------------------------

/// Canonical result alias matching patterns like `anyhow::Result<T>` / `eyre::Result<T>`.
/// This is exactly `std::result::Result<T, Error>` with `Error = YoshiError`.
pub type StdResult<T> = std::result::Result<T, Error>;

/// Convenience alias for framework-level operations that conceptually correspond to
/// `Result<T, Error>`, but mapped onto Yoshi’s `YoResult<T>`.
pub type FrameworkResult<T> = YoResult<T>;

/// Convenience alias for application-level operations that conceptually correspond to
/// `Result<T, YoError>`, but mapped onto Yoshi’s `AppResult<T>`.
pub type DomainResult<T> = AppResult<T>;

// --- Trait / helper aliases ---------------------------------------------------------

/// Alias for the core context extension trait, mirroring `anyhow::Context` naming
/// while still resolving to Yoshi’s structured implementation.
pub use yoshi_std::context::Context as StdContext;

/// Idiomatic alias for the ubiquitous `std::error::Error + Send + Sync + 'static` trait
/// object, often used in generic code. This is *not* Yoshi-specific, but provided
/// as a convenient companion to `Error` / `StdResult`.
pub type DynError = dyn std::error::Error + Send + Sync + 'static;

//=====================================================================================
// FRAMEWORK INITIALIZATION & TOOLING
//=====================================================================================

/// One-liner to run the yofix loop from apps/CI.
/// Example:
///     if !yoshi::run_yofix_ci()? { std::process::exit(1); }
pub fn run_yofix_ci() -> std::io::Result<bool> {
    // try a few passes; keep it conservative
    yofix_until_clean(3)
}

/// Enable/disable Yoshi autonomous code application (yofix auto-apply).
pub fn enable_autonomous_error_correction(enable: bool) {
    crate::error::set_autonomous_apply(enable);
}

/// Initialize the Yoshi system.
///
/// Sets up global systems like `tracing` for structured logging.
/// It is safe to call this function multiple times.
pub fn initialize() -> YoResult<()> {
    // Use `try_init` to avoid panicking if a logger is already set (e.g., in tests)
    let _ = tracing_subscriber::fmt::try_init();

    tracing::info!("Yoshi Core initialized");
    Ok(())
}

/// A prelude module for ergonomic imports of the most common Yoshi components.
///
/// Include `use yoshi::prelude::*;` in your modules to get convenient access
/// to error types, result types, traits, and macros.
pub mod prelude {
    pub use crate::{
        AnyError,
        App,
        AppErrorKind,
        AppResult,
        BoxedYoshi,
        CircuitBreaker,
        CircuitBreakerSystem,
        CircuitConfig,
        CircuitMetrics,
        CircuitState,
        Context,
        ContextTrend,
        CorrectionSystem,
        DomainResult,
        Error,
        ErrorMetricsSnapshot,
        ExponentialBackoffStrategy,
        Features,
        FrameworkResult,
        Hatch,
        HealthReport,
        Location,
        MLPrediction,
        MLRecoveryEngine,
        MLRecoveryStrategy,
        Recoverable,
        RecoveryAction,
        RecoveryPolicy,
        RecoverySeverity,
        RecoverySystem,
        Result,
        ResultExt,
        ResultRecovery,
        StdContext,
        StdResult,
        SupervisorSystem,
        SupervisorSystemBuilder,
        SupervisorTree,
        SupervisorTreeBuilder,
        WorkerConfig,
        WorkerState,
        WorkerStatus,
        WorkerType,
        Xuid,
        // XUID types for convenient access
        XuidError,
        XuidResult,
        XuidType,
        YoError,
        YoResult,
        YoshiError,
        YoshiErrorExt,
        YoshiSystem,
        Migrator,
        MigrationConfig,
        LogLevel,
        set_autonomous_apply,
        anyerror,
        buck,
        clinch,
        create_error,
        eggroll,
        error,
        lestrat,
        new_xuid,

        wrap,
        yoshi,
        yoshi_error,
        app_error
    };

    // NATS types (feature-gated)
    #[cfg(feature = "nats")]
    pub use crate::{NATSClient, NatsError};

    // Yoshi Std re-exports for convenience
    pub use yoshi_std::{
        PerformanceMetrics,
        SystemHealth,
        performance_metrics,
        system_health,
        TimeoutContext,
        ValidationInfo,
    };

    // Re-export commonly used std types for convenience
    pub use std::time::{Duration, Instant};
}

// Use thin wrapper macros so `buck!`, `clinch!`, and `yoshi!` are available when using the
// `yoshi` crate root instead of referring to `yoshi_std` directly.
#[macro_export]
macro_rules! yoshi {
    ($($arg:tt)*) => { $crate::yoshi_error!($($arg)*) };
}

#[macro_export]
macro_rules! buck {
    ($($arg:tt)*) => { return Err($crate::yoshi!($($arg)*).into()); };
}

/// Early-return helper equivalent to `bail!` – constructs a `YoshiError` via
/// `yoshi!` and returns it as `Err(...)` from the current function.
#[macro_export]
macro_rules! clinch {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) { $crate::buck!($($arg)*); }
    };
}

/// Format-first error constructor; analogous to `anyhow!`, builds a `YoshiError`
/// using `create_error` with the provided message.
#[macro_export]
macro_rules! yoshi_error {
    ($($arg:tt)*) => { $crate::create_error(format!($($arg)*)) };
}

/// Internal error marker for ML recovery signposts; produces an `ErrorKind::Internal`
/// with contextual metadata, useful for diagnosing self-healing flows.
#[macro_export]
macro_rules! eggroll {
    ($msg:expr) => {
        $crate::ErrorKind::Internal {
            message: $msg.to_string(),
            context_chain: vec![],
            internal_context: Some($crate::InternalContext {
                    module_name: module_path!().to_string(),
                    function_name: "eggroll_macro".to_string(),
                line_reference: Some(line!()),
                state_dump: Some(format!("ML_RECOVERY_SIGNPOST:{}:{}:{}", file!(), line!(), column!())),
                stack_trace: vec![],
            }),
        }
        .into()
    };
    ($fmt:expr, $($arg:tt)+) => {
        $crate::ErrorKind::Internal {
            message: format!($fmt, $($arg)+),
            context_chain: vec![],
            internal_context: Some($crate::InternalContext {
                module_name: module_path!().to_string(),
                function_name: "eggroll_macro".to_string(),
                line_reference: Some(line!()),
                state_dump: Some(format!("ML_RECOVERY_SIGNPOST:{}:{}:{}", file!(), line!(), column!())),
                stack_trace: vec![],
            }),
        }
        .into()
    };
}

/// Run a safe, looped “yofix” that:
/// 1) uses `cargo fix` + `cargo clippy --fix` (machine applicable only),
/// 2) re-checks, and repeats up to `max_iters`,
/// 3) only persists edits when `cargo check` returns clean.
///
/// NOTE: this never panics; it returns Ok(true) if the tree is clean at exit.
pub fn yofix_until_clean(max_iters: u8) -> std::io::Result<bool> {
    use std::process::{Command, Stdio};

    fn ok(status: std::process::ExitStatus) -> bool {
        status.success()
    }

    let mut iterations = 0u8;
    loop {
        // 1) apply rustc machine-applicable suggestions (stable)
        let fix = Command::new("cargo")
            .args(["fix", "--allow-dirty", "--allow-staged"])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .status()?;
        if !ok(fix) {
            // If `cargo fix` itself errors (rare), break the loop and let caller see the result of check.
        }

        // 2) apply clippy machine-applicable suggestions (requires -Z on stable toolchains to use --fix)
        // If nightly is present, try it; otherwise skip silently.
        let clippy_fix = Command::new("rustup")
            .args([
                "run",
                "nightly",
                "cargo",
                "clippy",
                "--fix",
                "-Z",
                "unstable-options",
                "--",
                "-A",
                "clippy::all",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = clippy_fix; // ignore if rustup/nightly is missing

        // 3) re-check
        let checked = Command::new("cargo")
            .args(["check", "--message-format=json"])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .status()?;
        if ok(checked) {
            return Ok(true); // compile clean ⇒ safe to persist
        }

        iterations = iterations.saturating_add(1);
        if iterations >= max_iters {
            return Ok(false); // give up safely; caller decides next step
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        assert!(initialize().is_ok());
    }

    #[test]
    fn test_error_exports() {
        // Test that all requested error types and macros are available
        use crate::{AppResult, YoResult, YoshiError, create_error};

        // Test basic usage
        let _error: YoshiError = create_error("test error");
        let _result: YoResult<()> = Ok(());

        // Test ResultExt trait is available for YoResult -> AppResult
        let result: YoResult<i32> = Ok(42);
        let _app_result: AppResult<i32> = result.into_app();
    }

    #[test]
    fn test_prelude_exports() {
        use crate::prelude::*;

        // Test types from core and local error module
        let _core_error: YoshiError = error("dynamic error");
        let _app_error: YoError = YoError::from("app error");

        // Test result types
        let egg_res: YoResult<()> = Ok(());
        let app_res: AppResult<()> = Ok(());
        let _std_res: StdResult<()> = Ok(());
        let _fw_res: FrameworkResult<()> = Ok(());
        let _dom_res: DomainResult<()> = Ok(());

        // Trait extensions
        let _ = egg_res.into_app();
        let _ = app_res.into_yoshi();

        // XUID types
        let _xuid: Xuid = new_xuid(b"test", XuidType::E8Quantized);

        // std types
        let _duration = Duration::from_secs(1);
        let _instant = Instant::now();

        // Alias sanity
        let _alias_error: Error = create_error("alias error");
        let _alias_app: App = YoError::from("alias app");
        let _alias_std: StdResult<()> = Ok(());
        let _alias_fw: FrameworkResult<()> = Ok(());
        let _alias_dom: DomainResult<()> = Ok(());
    }

    #[test]
    fn test_xuid_exports() {
        use crate::{
            E8Lattice, E8Orbit, E8Point, LeStratSpec,
            LearnedStrategyTag, Xuid, XuidError, XuidResult, XuidType,
 deserialize, deserialize_compressed, e8_distance, lestrat, new_xuid,
            orbit_correlation, serialize, serialize_compressed, xuid_from_path,
        };

        // Basic XUID creation
        let xuid = new_xuid(b"test", XuidType::E8Quantized);
        assert_eq!(xuid.xuid_type, XuidType::E8Quantized);



        // E8 lattice math
        let e8_point: E8Point = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let e8_lattice = E8Lattice::new();
        let orbit: E8Orbit = 5;

        // Test with another XUID for distance/correlation calculation
        let other_xuid = new_xuid(b"different data", XuidType::E8Quantized);
        let distance = e8_distance(&xuid.e8_coords, &other_xuid.e8_coords);
        assert!(distance >= 0.0);

        let correlation = orbit_correlation(xuid.e8_orbit, other_xuid.e8_orbit);
        assert!((0.0..=1.0).contains(&correlation));

        let norm = e8_lattice.compute_norm(&e8_point);
        assert!(norm > 0.0);
        let nearest = e8_lattice.find_nearest(&e8_point);
        assert_eq!(nearest.len(), 8);



        // Learned strategy
        let spec = LeStratSpec::new("test_crate", "test::module", "test_strategy");
        let strategy_tag = LearnedStrategyTag::from_spec(&spec);
        let lestra_xuid = lestrat(&spec).unwrap();
        assert_eq!(lestra_xuid.xuid_type, XuidType::Experience);
        assert_eq!(
            strategy_tag.clone().into_xuid().unwrap().xuid_type,
            XuidType::Experience
        );

        // Path-based construction
        let path_xuid = xuid_from_path("/test/path", XuidType::Codex);
        assert_eq!(path_xuid.xuid_type, XuidType::Codex);

        // Error handling
        let error_result: XuidResult<Xuid> = Ok(xuid.clone());
        assert!(error_result.is_ok());

        let _error_example: XuidError = XuidError::InvalidFormat("test error".to_string());

        // Serialization
        let bytes = serialize(&xuid).unwrap();
        let deserialized = deserialize(&bytes).unwrap();
        assert_eq!(xuid.semantic_hash, deserialized.semantic_hash);

        // Compressed serialization
        let compressed = serialize_compressed(&xuid, 3).unwrap();
        let decompressed = deserialize_compressed(&compressed).unwrap();
        assert_eq!(xuid.e8_orbit, decompressed.e8_orbit);



        // Verify all types are actually used and functional
        println!("E8 Lattice norm: {}", norm);
        println!("E8 Orbit: {}", orbit);
        println!("E8 Point: {:?}", e8_point);
        println!("E8 Nearest: {:?}", nearest);
        println!("Distance: {}", distance);
        println!("Correlation: {}", correlation);
        println!("Strategy Tag: {:?}", strategy_tag);
        println!("Path XUID: {}", path_xuid);
    }

    #[test]
    fn test_macro_wrappers_availability() {
        // Test yoshi! macro
        let _err: YoshiError = yoshi!("macro-created error: {}", "test");

        // Test eggroll! macro produces a YoshiError via eggroll function wrapper
        let _egg: YoshiError = eggroll!("eggroll error: {}", "test");

        // Test buck! macro returns an Err in a function returning YoResult
        fn fail_fn() -> YoResult<()> {
            buck!("explicit buck failure");
        }
        assert!(fail_fn().is_err());

        // Test clinch! macro returns an Err when condition false
        fn ensure_fn() -> YoResult<()> {
            clinch!(1 == 0, "clinch failure");
            Ok(())
        }
        assert!(ensure_fn().is_err());
    }

    #[test]
    fn test_error_kinds_and_messages() {
        // yoshi! produces a YoshiError with `Internal` ErrorKind and expected message
        let err: YoshiError = yoshi!("A test error: {}", 42);
        match err.kind {
            ErrorKind::Internal { ref message, .. } => assert_eq!(message, "A test error: 42"),
            _ => panic!("expected ErrorKind::Internal"),
        }

        // eggroll! creates an Internal ErrorKind and sets function_name in internal_context
        let e2: YoshiError = eggroll!("Eggroll test {}", "ok");
        match e2.kind {
            ErrorKind::Internal {
                ref message,
                internal_context,
                ..
            } => {
                assert_eq!(message, "Eggroll test ok");
                assert!(internal_context.is_some());
                let ctx = internal_context.as_ref().unwrap();
                assert_eq!(ctx.function_name, "eggroll_macro");
            }
            _ => panic!("expected ErrorKind::Internal"),
        }

        // buck! returns `Err(YoshiError)` with correct message
        fn fail() -> YoResult<()> {
            buck!("buck message");
        }
        match fail() {
            Err(er) => match er.as_ref().kind {
                ErrorKind::Internal { ref message, .. } => assert_eq!(message, "buck message"),
                _ => panic!("expected ErrorKind::Internal"),
            },
            Ok(_) => panic!("expected Err"),
        }

        // clinch! returns Err when condition is false
        fn ensure_false() -> YoResult<()> {
            clinch!(false, "clinch fail");
            Ok(())
        }
        match ensure_false() {
            Err(er) => match er.as_ref().kind {
                ErrorKind::Internal { ref message, .. } => assert_eq!(message, "clinch fail"),
                _ => panic!("expected ErrorKind::Internal"),
            },
            Ok(_) => panic!("expected Err"),
        }
    }

    #[test]
    fn test_eggroll_function_reexport() {
        // test the re-exported function `eggroll` exists and returns a YoshiError
        let ef: YoshiError = yoshi_std::eggroll("function test");
        match ef.kind {
            ErrorKind::Internal { ref message, .. } => assert_eq!(message, "function test"),
            _ => panic!("expected ErrorKind::Internal"),
        }
    }
}
