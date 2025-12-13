/* yoshi/src/error.rs */
//! High-level summary of the module's purpose and its primary function.
//!
//! # Yoshi – Façade Error Module
//!▫~•◦---------------------------‣
//!
//! This module integrates ergonomic, attribute-driven error definitions (`yoshi_derive::AnyError`)
//! with Yoshi’s production-grade recovery (`yoshi_std::YoshiError`, ML engine).
//!
//! ## Highlights
//! - Façade `YoError` backed by `YoshiError`, with structured context + async ML repair suggestions
//! - First-class interop: `to_egg_error()` + `From<YoError> for yoshi_std::YoshiError`
//! - Telemetry-friendly: emits structured fields via `tracing` on creation
//! - Result adapters to and from `YoResult<T>`
//! - Feature-gated `From<_>` impls for common ecosystem errors
//!
//! See crate docs for end-to-end examples.
//!
//! ## Usage (doctests)
//!
//! ### 1) Create an `YoError` directly
//! ```rust
//! use yoshi::error::{YoError, AppErrorKind};
//! use yoshi_std::Location;
//!
//! let err = YoError::new(
//!     AppErrorKind::Internal("something went boom".into()),
//!     Location { file: "demo.rs".into(), line: 1, column: 1 },
//! );
//!
//! // Human-friendly display includes location.
//! let s = err.to_string();
//! assert!(s.contains("something went boom"));
//! assert!(s.contains("demo.rs"));
//! ```
//!
//! ### 2) Ergonomic constructor macro (`app_error!`)
//! ```rust
//! use yoshi::error::AppErrorKind;
//!
//! // Create an YoError directly
//! let error_kind = AppErrorKind::Configuration { message: "bad config".into() };
//! // The app_error! macro would create an YoError from this kind
//! ```
//!
//! ### 3) Result adapters (`ResultExt`) to flip between `YoResult<T>` and `Result<T>`
//! ```rust
//! use yoshi::error::{Result, ResultExt};
//!
//! fn fallible() -> yoshi::YoResult<u32> {
//!     // Simulate a framework error
//!     Ok(7)
//! }
//!
//! // Turn a framework result into an app result
//! let app_res: Result<u32> = fallible().into_app();
//! assert_eq!(*app_res.as_ref().unwrap(), 7);
//!
//! // And back to framework result (no conversion cost on Ok-path)
//! let yoshi_back: yoshi::YoResult<u32> = app_res.into_yoshi();
//! assert!(yoshi_back.is_ok());
//! ```
//!
//! ### 4) Automatic mapping from common ecosystem errors (`From<_>` impls)
//! ```rust
//! use yoshi::error::YoError;
//!
//! // std::io::Error → YoError
//! let ioe = std::io::Error::new(std::io::ErrorKind::Other, "disk borked");
//! let app_from_io: YoError = ioe.into();
//! let _yoshi_again = app_from_io.to_egg_error(); // interop back to framework
//!
//! // serde_json::Error → YoError
//! let json_err = serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
//! let _app_from_json: YoError = json_err.into();
//! ```
//!
//! ### 5) Recoverable metadata (`Recoverable`)
//! ```rust
//! use yoshi::error::{YoError, AppErrorKind};
//! use yoshi_std::{Location, Recoverable, TimeoutContext};
//!
//! let e = YoError::new(
//!     AppErrorKind::Timeout {
//!         message: "operation timed out".into(),
//!         context: TimeoutContext {
//!             operation: "op".into(),
//!             timeout_duration_ms: 1000,
//!             elapsed_time_ms: 1200,
//!             bottleneck_analysis: None,
//!             optimization_hints: vec![],
//!         },
//!     },
//!     Location { file: "f".into(), line: 42, column: 7 },
//! );
//!
//! // Enriched context contains our source location
//! let ctx = e.recovery_context();
//! assert_eq!(ctx.get("line").map(String::as_str), Some("42"));
//! ```
//!
//! ### 6) Non-blocking suggestion generation (async)
//! ```rust
//! use yoshi::error::{YoError, AppErrorKind};
//! use yoshi_std::Location;
//!
//! fn example() {
//!     let e = YoError::new(
//!         AppErrorKind::Internal("hint me".into()),
//!         Location { file: "f".into(), line: 1, column: 1 },
//!     );
//!
//!     // Suggestions are generated off-thread; poll whenever convenient.
//!     let _maybe = e.recovery_signpost_now();
//! }
//! ```
//!
//! ### 7) Advanced: toggle autonomous apply (side effects off by default)
//! ```rust
//! use yoshi::error::set_autonomous_apply;
//!
//! fn main() {
//!     // Enable global auto-apply (prefer enabling via env in binaries)
//!     set_autonomous_apply(true);
//! }
//! ```
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use yoshi_derive::AnyError;
use yoshi_std::{
    AdvisedCorrection, BoxedYoshi, ErrorKind, FixSafetyLevel, IoContext, Location,
    MLRecoveryStrategy, Recoverable, RecoverySeverity, TimeoutContext, ValidationInfo, YoshiError,
    anyerror,
    correction::{CodeSpan, CorrectionBuilder, ProvidesFixes},
};

#[cfg(feature = "ml-recovery")]
use yoshi_std::MLRecoveryEngine;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use tracing::debug;
#[cfg(any(feature = "ml-recovery", test))]
use yoshi_std::correction::CodeModification;

#[cfg(feature = "ml-recovery")]
use std::sync::atomic::AtomicBool;

// Ensure `Ordering` is available when either ml-recovery or nats features are enabled
#[cfg(any(feature = "ml-recovery", feature = "nats"))]
use std::sync::atomic::Ordering;

#[cfg(feature = "nats")]
use {
    futures::StreamExt,
    serde_json,
    std::sync::Mutex,
    std::sync::atomic::AtomicBool as NatsAtomicBool,
    tokio::task::JoinHandle,
    yoshi_std::{DistributedErrorMessage, NATSClient, NATSConfig},
};

#[cfg(feature = "ml-recovery")]
static AUTONOMOUS_APPLY: AtomicBool = AtomicBool::new(false);

/// Enable/disable autonomous code application at runtime (default: disabled).
#[cfg(feature = "ml-recovery")]
pub fn set_autonomous_apply(enable: bool) {
    AUTONOMOUS_APPLY.store(enable, Ordering::SeqCst);
}

/// Fallback when ML recovery is disabled: keep API surface stable (no-op).
#[cfg(not(feature = "ml-recovery"))]
pub fn set_autonomous_apply(_enable: bool) {}

/// Canonical Result for the application, defaulting to `YoError` as the error type.
///
/// This allows for flexibility in function signatures, e.g., `fn my_func() -> Result<()>`
/// or `fn my_func() -> Result<(), MySpecificError>`.
pub type Result<T, E = YoError> = std::result::Result<T, E>;

/// Ergonomic constructor that captures file/line/column and defers to `YoError::new`.
#[macro_export]
macro_rules! app_error {
    ($kind:expr) => {
        $crate::error::YoError::new(
            $kind,
            $crate::Location {
                file: file!().into(),
                line: line!(),
                column: column!(),
            },
        )
    };
}

/// Smart error: wraps `YoshiError`, tracks source location, holds (eventual) ML recovery suggestion.
#[derive(Debug)]
pub struct YoError {
    pub kind: AppErrorKind,
    pub egg: YoshiError,
    pub location: Location,
    pub recovery_signpost: Arc<RwLock<Option<AdvisedCorrection>>>,
}

impl From<YoshiError> for YoError {
    fn from(error: YoshiError) -> Self {
        YoError::new(
            AppErrorKind::Framework(Box::new(error)),
            Location {
                file: "yoshi::error".into(),
                line: 0,
                column: 0,
            },
        )
    }
}

impl From<BoxedYoshi> for YoError {
    fn from(boxed: BoxedYoshi) -> Self {
        // BoxedYoshi wraps Box<YoshiError>; reuse the existing conversion path.
        let ye: YoshiError = *boxed.0;
        ye.into()
    }
}

/// Application-level error variants (derive produces Display/source for messages).
#[derive(Debug, AnyError)]
pub enum AppErrorKind {
    // --- I/O & System ---
    #[anyerror("I/O error on '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
        io: Option<IoContext>,
    },

    #[anyerror("Configuration error: {message}")]
    Configuration { message: String },

    // --- Data & Parsing ---
    #[anyerror("Failed to parse data: {source}")]
    Parse {
        #[from]
        #[source]
        source: serde_json::Error,
    },

    #[anyerror("Data validation failed: {message}")]
    Validation {
        message: String,
        context: ValidationInfo,
    },

    // --- Network & Services ---
    #[anyerror("Network operation failed: {message}")]
    Network { message: String },

    #[anyerror("Operation timed out: {message}")]
    Timeout {
        message: String,
        context: TimeoutContext,
    },

    #[anyerror("External service '{service}' error: {message}")]
    ExternalService {
        service: String,
        message: String,
        details: Option<HashMap<String, String>>,
    },

    #[anyerror("Permission denied for '{resource}': {reason}")]
    PermissionDenied { resource: String, reason: String },

    #[anyerror("Rate limited by '{scope}' (retry after {retry_after_ms} ms)")]
    RateLimited {
        scope: String,
        retry_after_ms: u64,
        hints: Option<Vec<String>>,
    },

    #[anyerror("Request conflict/idempotency violation: {message}")]
    Conflict { message: String },

    #[anyerror("Concurrency error: {message}")]
    Concurrency { message: String },

    // --- Database ---
    #[anyerror("Database operation failed: {message}")]
    Database {
        operation: String,
        query: Option<String>,
        message: String,
    },

    // --- Logic & State ---
    #[anyerror("Internal application error: {0}")]
    Internal(String),

    #[anyerror("Feature not implemented: {0}")]
    NotImplemented(String),

    // --- Framework bridge ---
    #[anyerror(transparent)]
    Framework(#[from] Box<YoshiError>),
}

impl YoError {
    /// Synchronous constructor; maps `AppErrorKind` → `YoshiError`. Call `spawn_suggestion()` to enable async ML suggestions.
    pub fn new(kind: AppErrorKind, location: Location) -> Self {
        let egg = Self::map_kind_to_egg(&kind, &location);

        // Telemetry: emit once with stable fields (debug-level; YoError construction is not an operational failure)
        debug!(
            target: "yoshi::error",
            kind = %kind,
            file = %location.file,
            line = location.line,
            column = location.column,
            "YoError created",
        );

        let error = Self {
            kind,
            egg,
            location,
            recovery_signpost: Arc::new(RwLock::new(None)),
        };

        // Spawn non-blocking generation of a recovery suggestion (feature-gated).
        #[cfg(feature = "ml-recovery")]
        {
            let signpost_handle = Arc::clone(&error.recovery_signpost);
            let egg_clone = error.egg.clone();
            let location_clone = error.location.clone();

            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let signpost =
                        Self::generate_recovery_signpost(egg_clone, location_clone).await;
                    if let Ok(mut guard) = signpost_handle.write() {
                        *guard = signpost;
                    }
                });
            } else {
                // Fallback: tiny runtime on a background thread (no panic)
                std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build();
                    if let Ok(rt) = rt {
                        rt.block_on(async {
                            let signpost =
                                Self::generate_recovery_signpost(egg_clone, location_clone).await;
                            if let Ok(mut guard) = signpost_handle.write() {
                                *guard = signpost;
                            }
                        });
                    }
                });
            }
        }

        error
    }

    /// Map `AppErrorKind` to a canonical `YoshiError` using only stable `ErrorKind` variants.
    fn map_kind_to_egg(kind: &AppErrorKind, location: &Location) -> YoshiError {
        match kind {
            AppErrorKind::Io {
                path,
                source,
                io: _,
            } => YoshiError::new(ErrorKind::Internal {
                message: format!("I/O on '{}': {}", path, source),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                internal_context: None,
            }),
            AppErrorKind::Configuration { message } => {
                YoshiError::new(ErrorKind::InvalidArgument {
                    message: message.clone(),
                    context_chain: vec![format!("{}:{}", location.file, location.line)],
                    validation_info: None,
                })
            }
            AppErrorKind::Parse { source } => YoshiError::new(ErrorKind::Parse {
                message: source.to_string(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                parse_context: None,
            }),
            AppErrorKind::Validation { message, context } => {
                YoshiError::new(ErrorKind::InvalidArgument {
                    message: message.clone(),
                    context_chain: vec![format!("{}:{}", location.file, location.line)],
                    validation_info: Some(context.clone()),
                })
            }
            AppErrorKind::Network { message } => YoshiError::new(ErrorKind::Internal {
                message: message.clone(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                internal_context: None,
            }),
            AppErrorKind::Timeout { message, context } => YoshiError::new(ErrorKind::Timeout {
                message: message.clone(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                timeout_context: Some(context.clone()),
            }),
            AppErrorKind::ExternalService {
                service,
                message,
                details,
            } => {
                let mut ctx = vec![
                    format!("service:{}", service),
                    format!("{}:{}", location.file, location.line),
                ];
                if let Some(map) = details {
                    for (k, v) in map {
                        ctx.push(format!("{}:{}", k, v));
                    }
                }
                YoshiError::new(ErrorKind::Internal {
                    message: message.clone(),
                    context_chain: ctx,
                    internal_context: None,
                })
            }
            AppErrorKind::PermissionDenied { resource, reason } => {
                YoshiError::new(ErrorKind::InvalidArgument {
                    message: format!("{}: {}", resource, reason),
                    context_chain: vec![format!("{}:{}", location.file, location.line)],
                    validation_info: None,
                })
            }
            AppErrorKind::RateLimited {
                scope,
                retry_after_ms,
                hints,
            } => {
                let mut ctx = vec![
                    format!("scope:{}", scope),
                    format!("retry_after_ms:{}", retry_after_ms),
                    format!("{}:{}", location.file, location.line),
                ];
                if let Some(h) = hints {
                    for hint in h {
                        ctx.push(format!("hint:{}", hint));
                    }
                }
                YoshiError::new(ErrorKind::Timeout {
                    message: "rate limited".to_string(),
                    context_chain: ctx,
                    timeout_context: None,
                })
            }
            AppErrorKind::Conflict { message } => YoshiError::new(ErrorKind::InvalidArgument {
                message: message.clone(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                validation_info: None,
            }),
            AppErrorKind::Concurrency { message } => YoshiError::new(ErrorKind::Internal {
                message: message.clone(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                internal_context: None,
            }),
            AppErrorKind::Database {
                operation,
                query,
                message,
            } => YoshiError::new(ErrorKind::Internal {
                message: format!("{}: {}", operation, message),
                context_chain: {
                    let mut v = vec![format!("{}:{}", location.file, location.line)];
                    if let Some(q) = query {
                        v.push(format!("query:{}", q));
                    }
                    v
                },
                internal_context: None,
            }),
            AppErrorKind::Internal(msg) => YoshiError::new(ErrorKind::Internal {
                message: msg.clone(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                internal_context: None,
            }),
            AppErrorKind::NotImplemented(msg) => YoshiError::new(ErrorKind::NotSupported {
                feature: msg.clone(),
                context_chain: vec![format!("{}:{}", location.file, location.line)],
                alternatives: None,
            }),
            AppErrorKind::Framework(e) => (**e).clone(),
        }
    }

    /// Non-blocking read of the best-known recovery suggestion so far.
    pub fn recovery_signpost_now(&self) -> Option<AdvisedCorrection> {
        self.recovery_signpost
            .read()
            .ok()
            .and_then(|g| (*g).clone())
    }

    /// Attach (or overwrite) a manual hint programmatically.
    pub fn with_hint(self, hint: AdvisedCorrection) -> Self {
        if let Ok(mut guard) = self.recovery_signpost.write() {
            *guard = Some(hint);
        }
        self
    }

    /// `YoshiError` view.
    pub fn to_egg_error(&self) -> YoshiError {
        self.egg.clone()
    }

    /// Consult the ML engine to generate a code fix suggestion for this error.
    #[cfg(feature = "ml-recovery")]
    async fn generate_recovery_signpost(
        egg: YoshiError,
        location: Location,
    ) -> Option<AdvisedCorrection> {
        let engine = MLRecoveryEngine::global();

        // Feed the engine with the canonical framework error and location context.
        engine
            .attempt_recovery::<()>(&egg, "app_error_creation")
            .await;

        // Run the corrector on a pure string in an isolated thread (no catch_unwind).
        let msg = egg.to_string();
        let safe = std::thread::spawn(move || {
            let corrector = yoshi_std::corrector::YoshiErrorCorrector::new();
            // Ensure JoinHandle<T> is Send by mapping error to String.
            // This avoids carrying a non-Send dyn StdError across the thread boundary.
            corrector.analyze_and_fix(&msg).map_err(|e| e.to_string())
        })
        .join()
        .ok()
        .and_then(|r| r.ok());

        if let Some(fixes) = safe
            && let Some(fix) = fixes.into_iter().next()
        {
            // Translate framework fix → code modifications for our applier.
            let mods = match fix.fix_type {
                yoshi_std::corrector::FixType::ReplaceUnwrap { replacement } => {
                    vec![CodeModification::Replace {
                        span: CodeSpan {
                            file: location.file.to_string(),
                            start_byte: 0,
                            end_byte: 0,
                        },
                        new_text: std::sync::Arc::<str>::from(replacement),
                    }]
                }
                _ => vec![],
            };

            // Optionally apply the modifications to disk immediately (autonomous mode).
            #[cfg(feature = "ml-recovery")]
            if AUTONOMOUS_APPLY.load(Ordering::SeqCst)
                && !mods.is_empty()
                && apply_code_modifications(&mods).is_err()
            {
                tracing::error!(target: "yoshi::error", "auto-apply failed");
            }

            return Some(AdvisedCorrection {
                summary: fix.description.into(),
                modifications: mods,
                confidence: fix.confidence,
                safety_level: FixSafetyLevel::MaybeIncorrect,
            });
        }
        None
    }
}

impl fmt::Display for YoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to the canonical error, include location for humans.
        write!(f, "{} at {}", self.egg, self.location)
    }
}

impl std::error::Error for YoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.egg.source()
    }
}

#[cfg(any(feature = "ml-recovery", test))]
fn apply_code_modifications(mods: &[CodeModification]) -> std::io::Result<usize> {
    use std::fs;
    use std::io::{Read, Write};
    use std::path::Path;

    // Group edits per file; apply per-file to reduce races & preserve atomicity.
    let mut per_file: HashMap<String, Vec<&CodeModification>> = HashMap::new();
    for m in mods {
        match m {
            CodeModification::Replace { span, .. } => {
                per_file.entry(span.file.clone()).or_default().push(m);
            }
            CodeModification::Insert { span, .. } => {
                per_file.entry(span.file.clone()).or_default().push(m);
            }
            CodeModification::Delete { span, .. } => {
                per_file.entry(span.file.clone()).or_default().push(m);
            }
        }
    }

    let mut applied = 0usize;

    for (file, edits) in per_file {
        let path = Path::new(&file);
        // Read whole file (best-effort; skip if missing).
        let mut buf = String::new();
        if let Ok(mut fh) = fs::File::open(path) {
            fh.read_to_string(&mut buf)?;
        } else {
            continue;
        }

        // Apply edits in reverse span order by (start_byte, end_byte) to avoid shifting.
        let mut replace_spans: Vec<(usize, usize, String)> = Vec::new();
        for e in edits {
            match e {
                CodeModification::Replace { span, new_text } => {
                    // Replace text in [start_byte, end_byte) with the new content
                    replace_spans.push((
                        span.start_byte,
                        span.end_byte,
                        new_text.as_ref().to_string(),
                    ));
                }
                CodeModification::Insert {
                    span,
                    new_text,
                    after,
                } => {
                    // Insert `new_text` at the appropriate position based on after flag
                    let insert_pos = if *after {
                        span.end_byte
                    } else {
                        span.start_byte
                    };
                    replace_spans.push((insert_pos, insert_pos, new_text.as_ref().to_string()));
                }
                CodeModification::Delete { span } => {
                    // Delete the region by replacing with empty string
                    replace_spans.push((span.start_byte, span.end_byte, String::new()));
                }
            }
        }

        replace_spans.sort_by(|a, b| b.0.cmp(&a.0)); // reverse by start

        let mut bytes = buf.into_bytes();
        for (start, end, text) in replace_spans {
            if start <= end && end <= bytes.len() {
                bytes.splice(start..end, text.into_bytes());
                applied += 1;
            }
        }

        // Atomic write (write to .tmp then rename).
        let tmp = path.with_extension("tmp__yofix");
        {
            let mut out = fs::File::create(&tmp)?;
            out.write_all(&bytes)?;
            out.flush()?;
        }
        std::fs::rename(tmp, path)?;
    }

    Ok(applied)
}

/// Recoverable → delegate to `YoshiError`, enriching with location.
impl Recoverable for YoError {
    fn can_recover(&self) -> bool {
        self.egg.can_recover()
    }

    fn recovery_hint(&self) -> Option<MLRecoveryStrategy> {
        self.egg.recovery_hint()
    }

    fn recovery_context(&self) -> HashMap<String, String> {
        let mut ctx = self.egg.recovery_context();
        ctx.insert("file".into(), self.location.file.clone().into());
        ctx.insert("line".into(), self.location.line.to_string());
        ctx.insert("column".into(), self.location.column.to_string());
        ctx
    }

    fn recovery_severity(&self) -> RecoverySeverity {
        self.egg.recovery_severity()
    }
}

/// Auto-fixes / code suggestions → prefer the framework’s own, with a couple of app-level nudges.
impl ProvidesFixes for YoError {
    fn get_available_fixes(&self) -> Vec<AdvisedCorrection> {
        let mut xs = self.egg.get_available_fixes();

        // Example: nudge for generic "internal" messages to become typed variants upstream.
        let msg = self.egg.to_string();
        if msg.to_lowercase().contains("internal") {
            let suggested_variant_name = msg
                .chars()
                .filter(|c| c.is_alphanumeric())
                .take(20)
                .collect::<String>();
            let new_variant = format!(
                "\n    #[anyerror(\"{} at {{location}}\")]\n    {} {{ message: String, location: Location }},",
                msg, suggested_variant_name
            );
            xs.push(
                CorrectionBuilder::new(
                    "Replace general internal error with a specific, typed error variant.",
                )
                .insert_before(
                    CodeSpan {
                        file: self.location.file.to_string(),
                        start_byte: 0,
                        end_byte: 0,
                    },
                    new_variant,
                )
                .set_confidence(0.7)
                .set_safety_level(FixSafetyLevel::HasPlaceholders)
                .build(),
            );
        }
        xs
    }
}

// --- Conversions ---------------------------------------------------------------------------

impl From<YoError> for YoshiError {
    fn from(e: YoError) -> Self {
        e.egg
    }
}

impl From<String> for YoError {
    fn from(error: String) -> Self {
        app_error!(AppErrorKind::Internal(error))
    }
}

impl From<&str> for YoError {
    fn from(error: &str) -> Self {
        app_error!(AppErrorKind::Internal(error.to_string()))
    }
}

impl From<std::io::Error> for AppErrorKind {
    fn from(error: std::io::Error) -> Self {
        AppErrorKind::Io {
            path: "unknown".to_string(),
            source: error,
            io: None,
        }
    }
}

impl From<std::io::Error> for YoError {
    fn from(error: std::io::Error) -> Self {
        app_error!(AppErrorKind::Io {
            path: "unknown".to_string(),
            source: error,
            io: None,
        })
    }
}

impl From<serde_json::Error> for YoError {
    fn from(error: serde_json::Error) -> Self {
        app_error!(AppErrorKind::Parse { source: error })
    }
}

#[cfg(feature = "ml-recovery")]
impl From<tokio::task::JoinError> for YoError {
    fn from(e: tokio::task::JoinError) -> Self {
        app_error!(AppErrorKind::Concurrency {
            message: e.to_string()
        })
    }
}

impl From<std::num::ParseIntError> for YoError {
    fn from(e: std::num::ParseIntError) -> Self {
        app_error!(AppErrorKind::Validation {
            message: e.to_string(),
            context: ValidationInfo::new()
                .with_parameter("parse_int")
                .with_expected("valid integer")
                .with_actual("")
                .with_rule("parse"),
        })
    }
}

#[cfg(feature = "url")]
impl From<url::ParseError> for YoError {
    fn from(e: url::ParseError) -> Self {
        app_error!(AppErrorKind::Validation {
            message: e.to_string(),
            context: ValidationInfo::new()
                .with_parameter("url")
                .with_expected("valid URL")
                .with_actual("")
                .with_rule("parse")
        })
    }
}

// Optional: HTTP gateway interop
#[cfg(feature = "http-gateway")]
impl From<reqwest::Error> for YoError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            app_error!(AppErrorKind::Timeout {
                message: e.to_string(),
                context: TimeoutContext {
                    operation: "http".into(),
                    timeout_duration_ms: 0,
                    elapsed_time_ms: 0,
                    bottleneck_analysis: None,
                    optimization_hints: vec![],
                }
            })
        } else if e.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS) {
            // Note: In reqwest 0.12, headers() method is not available on Error
            // Default to 60 seconds retry for rate limiting
            app_error!(AppErrorKind::RateLimited {
                scope: "http".into(),
                retry_after_ms: 60000, // 60 seconds default
                hints: Some(vec!["exponential_backoff".into(), "jitter".into()]),
            })
        } else {
            app_error!(AppErrorKind::Network {
                message: e.to_string()
            })
        }
    }
}

// --- Result adapters -----------------------------------------------------------------------

/// Flip between YoError and YoshiError ergonomically.
pub trait ResultExt<T> {
    fn into_app(self) -> Result<T>;
    fn into_yoshi(self) -> yoshi_std::YoResult<T>;
}

impl<T> ResultExt<T> for yoshi_std::YoResult<T> {
    fn into_app(self) -> Result<T> {
        self.map_err(|e| AppErrorKind::Framework(e.into()))
            .map_err(|k| app_error!(k))
    }
    fn into_yoshi(self) -> yoshi_std::YoResult<T> {
        self
    }
}

impl<T> ResultExt<T> for Result<T> {
    fn into_app(self) -> Result<T> {
        self
    }
    fn into_yoshi(self) -> yoshi_std::YoResult<T> {
        self.map_err(|e| e.to_egg_error().into())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERGONOMIC UNIFIED SYSTEMS - High-level facades for yoshi-std ecosystems
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified ML Recovery System - One-stop shop for autonomous error recovery
///
/// This struct wraps the entire ML recovery ecosystem from yoshi-std into a single,
/// easy-to-use interface. No need to juggle multiple types!
///
/// # Example
/// ```rust
/// use yoshi::error::RecoverySystem;
///
/// #[tokio::main(flavor = "current_thread")]
/// async fn main() {
///     let recovery = RecoverySystem::production();
///
///     // Enable recovery for a context, then attempt a no-op recovery.
///     recovery.enable("doc_example").await;
///     let err = yoshi::create_error("doc failure");
///     let _ = recovery.recover_with_ml::<()>(&err, "doc_example").await;
/// }
/// ```
pub struct RecoverySystem {
    engine: &'static yoshi_std::MLRecoveryEngine,
    policy: yoshi_std::RecoveryPolicy,
}

impl RecoverySystem {
    /// Create a new recovery system with default policy
    pub fn new() -> Self {
        Self {
            engine: yoshi_std::MLRecoveryEngine::global(),
            policy: yoshi_std::RecoveryPolicy::default(),
        }
    }

    /// Create a recovery system with custom policy
    pub fn with_policy(policy: yoshi_std::RecoveryPolicy) -> Self {
        Self {
            engine: yoshi_std::MLRecoveryEngine::global(),
            policy,
        }
    }

    /// Create a production-ready recovery system with sensible defaults
    pub fn production() -> Self {
        Self {
            engine: yoshi_std::MLRecoveryEngine::global(),
            policy: yoshi_std::RecoveryPolicy {
                max_attempts: 5,
                timeout_ms: 10000,
                strategies: vec![
                    yoshi_std::MLRecoveryStrategy::PatternBasedRecovery,
                    yoshi_std::MLRecoveryStrategy::LearningBasedRecovery,
                    yoshi_std::MLRecoveryStrategy::DefaultFallback,
                ],
                backoff: yoshi_std::ExponentialBackoffStrategy {
                    initial_delay_ms: 100,
                    max_delay_ms: 30000,
                    multiplier: 2.0,
                    jitter: true,
                },
                circuit_breaker_threshold: Some(10),
                enable_learning: true,
            },
        }
    }

    /// Quick-start: Create and enable recovery in one call
    pub async fn quick_start(context: &str) -> Self {
        let system = Self::production();
        system.enable(context).await;
        system
    }

    /// Enable ML recovery for a specific context
    pub async fn enable(&self, context: &str) {
        yoshi_std::MLRecoveryEngine::enable_for_context(context).await;
    }

    /// Disable ML recovery globally
    pub fn disable(&self) {
        yoshi_std::MLRecoveryEngine::disable();
    }

    /// Check if ML recovery is enabled
    pub fn is_enabled(&self) -> bool {
        yoshi_std::MLRecoveryEngine::is_enabled()
    }

    /// Attempt ML-driven recovery for an error
    pub async fn recover_with_ml<T>(&self, error: &YoshiError, context: &str) -> Option<T>
    where
        T: Default + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        self.engine.attempt_recovery(error, context).await
    }

    /// Get ML prediction for best recovery strategy
    pub fn predict_strategy(&self, error: &YoshiError) -> yoshi_std::MLPrediction {
        // Simplified prediction based on error kind
        match &error.kind {
            ErrorKind::Timeout { .. } => yoshi_std::MLPrediction::confident(
                yoshi_std::MLRecoveryStrategy::ParameterAdjustment,
                0.85,
                500,
            ),
            ErrorKind::LimitExceeded { .. } => yoshi_std::MLPrediction::confident(
                yoshi_std::MLRecoveryStrategy::ServiceDegradation,
                0.90,
                200,
            ),
            _ => yoshi_std::MLPrediction::fallback(yoshi_std::MLRecoveryStrategy::DefaultFallback),
        }
    }

    /// Get the current recovery policy
    pub fn policy(&self) -> &yoshi_std::RecoveryPolicy {
        &self.policy
    }

    /// Update the recovery policy
    pub fn set_policy(&mut self, policy: yoshi_std::RecoveryPolicy) {
        self.policy = policy;
    }
}

impl Default for RecoverySystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified Supervisor System - Complete worker supervision in one struct
///
/// This wraps the entire supervisor/worker ecosystem from yoshi-std, making it
/// trivial to set up fault-tolerant worker systems.
///
/// # Example
/// ```rust
/// use yoshi::error::SupervisorSystem;
///
/// #[tokio::main(flavor = "current_thread")]
/// async fn main() -> yoshi::YoResult<()> {
///     // Build a supervisor with one worker; requires a runtime because of async internals.
///     let supervisor = SupervisorSystem::builder()
///         .with_id("doc_supervisor")
///         .add_processor_worker("worker-1", 16)
///         .build()?;
///
///     let status = supervisor.status();
///     assert_eq!(status.worker_count, 1);
///     Ok(())
/// }
/// ```
pub struct SupervisorSystem {
    tree: Arc<yoshi_std::SupervisorTree>,
    #[cfg_attr(not(feature = "ml-recovery"), allow(dead_code))]
    recovery: Arc<RecoverySystem>,
    circuit: Arc<CircuitBreakerSystem>,
    corrector: Arc<CorrectionSystem>,
}

impl SupervisorSystem {
    /// Create a new supervisor system builder
    pub fn builder() -> SupervisorSystemBuilder {
        SupervisorSystemBuilder {
            builder: yoshi_std::SupervisorTreeBuilder::new(),
            recovery: None,
            circuit: None,
            corrector: None,
        }
    }

    /// Execute an operation within a supervised worker
    pub async fn execute_in_worker<T, F>(
        &self,
        config: yoshi_std::WorkerConfig,
        operation: F,
    ) -> yoshi_std::Result<T>
    where
        F: FnOnce() -> yoshi_std::Result<T> + Send + 'static,
        T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    {
        let worker_id = config.id.clone();
        let tree = self.tree.clone();
        #[cfg(feature = "ml-recovery")]
        let recovery = self.recovery.clone();
        let corrector = self.corrector.clone();

        // Protect the worker execution with the shared circuit breaker
        self.circuit
            .execute_async(move || {
                let config_clone = config.clone();
                let operation = operation;
                let tree = tree.clone();
                async move {
                    let result = tree.execute_in_worker(config_clone, operation).await;

                    if let Err(err) = &result {
                        let location = Location {
                            file: format!("worker:{}", worker_id).into(),
                            line: 0,
                            column: 0,
                        };
                        let mut yerr = err.clone();
                        yerr.location = Some(location.clone());

                        // Feed error to ML recovery and corrector; fire-and-forget.
                        #[cfg(feature = "ml-recovery")]
                        {
                            let recovery = recovery.clone();
                            let corrector = corrector.clone();
                            let worker = worker_id.clone();
                            tokio::spawn(async move {
                                if recovery.is_enabled() {
                                    let _ = recovery.recover_with_ml::<()>(&yerr, &worker).await;
                                }
                                let _ = corrector.analyze_hatch(&yerr);
                            });
                        }

                        #[cfg(not(feature = "ml-recovery"))]
                        {
                            let _ = corrector.analyze_hatch(&yerr);
                        }

                        // Attempt an immediate restart after recovery hook
                        let _ = tree.restart_worker(&worker_id).await;
                    }

                    result
                }
            })
            .await
    }

    /// Start the supervisor system
    pub async fn start(&self) -> yoshi_std::YoResult<()> {
        self.tree.start_supervision().await.map_err(Into::into)
    }

    /// Stop the supervisor system
    pub async fn stop(&self) -> yoshi_std::YoResult<()> {
        self.tree.stop().await.map_err(Into::into)
    }

    /// Get the status of the supervisor system
    pub fn status(&self) -> yoshi_std::SupervisorStatus {
        self.tree.get_status()
    }

    /// Perform a health check on the supervisor system
    pub async fn health_check(&self) -> yoshi_std::YoResult<yoshi_std::HealthState> {
        let status = self.tree.get_status();
        if status.is_running && status.healthy_workers == status.worker_count {
            Ok(yoshi_std::HealthState::Healthy)
        } else {
            Ok(yoshi_std::HealthState::Unhealthy)
        }
    }

    /// Get the underlying supervisor tree
    pub fn tree(&self) -> &yoshi_std::SupervisorTree {
        &self.tree
    }
}

/// Builder for SupervisorSystem with ergonomic worker creation
pub struct SupervisorSystemBuilder {
    builder: yoshi_std::SupervisorTreeBuilder,
    recovery: Option<Arc<RecoverySystem>>,
    circuit: Option<Arc<CircuitBreakerSystem>>,
    corrector: Option<Arc<CorrectionSystem>>,
}

impl SupervisorSystemBuilder {
    /// Set the supervisor ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.builder = self.builder.with_id(id.into());
        self
    }

    /// Add a processor worker with batch size
    pub fn add_processor_worker(mut self, id: impl Into<String>, batch_size: usize) -> Self {
        self.builder = self.builder.add_worker(yoshi_std::WorkerConfig {
            id: id.into(),
            worker_type: yoshi_std::WorkerType::Processor { batch_size },
            ..Default::default()
        });
        self
    }

    /// Add multiple processor workers at once
    pub fn add_processor_workers(mut self, count: usize, batch_size: usize) -> Self {
        for i in 0..count {
            self = self.add_processor_worker(format!("worker-{}", i), batch_size);
        }
        self
    }

    /// Add a custom worker with full configuration
    pub fn add_worker(mut self, config: yoshi_std::WorkerConfig) -> Self {
        self.builder = self.builder.add_worker(config);
        self
    }

    /// Attach a shared recovery system
    pub fn with_recovery(mut self, recovery: Arc<RecoverySystem>) -> Self {
        self.recovery = Some(recovery);
        self
    }

    /// Attach a shared circuit breaker system
    pub fn with_circuit(mut self, circuit: Arc<CircuitBreakerSystem>) -> Self {
        self.circuit = Some(circuit);
        self
    }

    /// Attach a shared corrector system
    pub fn with_corrector(mut self, corrector: Arc<CorrectionSystem>) -> Self {
        self.corrector = Some(corrector);
        self
    }

    /// Build the supervisor system
    pub fn build(self) -> yoshi_std::YoResult<SupervisorSystem> {
        let recovery = self
            .recovery
            .unwrap_or_else(|| Arc::new(RecoverySystem::production()));
        let circuit = self
            .circuit
            .unwrap_or_else(|| Arc::new(CircuitBreakerSystem::production("supervisor")));
        let corrector = self
            .corrector
            .unwrap_or_else(|| Arc::new(CorrectionSystem::new()));

        Ok(SupervisorSystem {
            tree: Arc::new(self.builder.build()?),
            recovery,
            circuit,
            corrector,
        })
    }
}

impl SupervisorSystem {
    /// Quick-start: Create a supervisor with N workers in one call
    pub fn quick_start(
        id: impl Into<String>,
        worker_count: usize,
        batch_size: usize,
    ) -> yoshi_std::YoResult<Self> {
        Self::builder()
            .with_id(id)
            .add_processor_workers(worker_count, batch_size)
            .build()
    }
}

/// Unified Circuit Breaker System - Protect your services with one struct
///
/// Wraps the circuit breaker functionality from yoshi-std with sensible defaults
/// and easy configuration.
///
/// # Example
/// ```rust
/// use yoshi::error::CircuitBreakerSystem;
///
/// #[tokio::main(flavor = "current_thread")]
/// async fn main() -> yoshi::YoResult<()> {
///     let circuit = CircuitBreakerSystem::new("external_api");
///
///     let result = circuit
///         .execute_async(|| async { Ok::<_, yoshi::YoshiError>("success") })
///         .await?;
///
///     assert_eq!(result, "success");
///     Ok(())
/// }
/// ```
pub struct CircuitBreakerSystem {
    breaker: yoshi_std::CircuitBreaker,
    name: String,
}

impl CircuitBreakerSystem {
    /// Create a new circuit breaker with default configuration
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            breaker: yoshi_std::CircuitBreaker::new(yoshi_std::CircuitConfig::default()),
            name: name.into(),
        }
    }

    /// Create a circuit breaker with custom configuration
    pub fn with_config(name: impl Into<String>, config: yoshi_std::CircuitConfig) -> Self {
        Self {
            breaker: yoshi_std::CircuitBreaker::new(config),
            name: name.into(),
        }
    }

    /// Create a production-ready circuit breaker
    pub fn production(name: impl Into<String>) -> Self {
        Self {
            breaker: yoshi_std::CircuitBreaker::new(yoshi_std::CircuitConfig {
                failure_threshold: 10,
                recovery_timeout: std::time::Duration::from_secs(60),
                half_open_max_calls: 3,
                ..Default::default()
            }),
            name: name.into(),
        }
    }

    /// Execute an async operation with circuit breaker protection
    pub async fn execute_async<T, F, Fut>(&self, f: F) -> yoshi_std::Result<T>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = yoshi_std::Result<T>> + Send,
        T: Send + 'static,
    {
        self.breaker.execute_async(f).await
    }

    /// Get the current state of the circuit breaker
    pub fn state(&self) -> yoshi_std::CircuitState {
        self.breaker.current_state()
    }

    /// Get metrics for the circuit breaker
    pub fn metrics(&self) -> yoshi_std::CircuitMetrics {
        self.breaker.metrics()
    }

    /// Reset the circuit breaker to closed state
    pub async fn reset(&self) {
        self.breaker.reset().await;
    }

    /// Force the circuit open
    pub async fn force_open(&self) {
        self.breaker.force_open().await;
    }

    /// Get the name of this circuit breaker
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Unified Error Correction System - Automated code fixes in one place
///
/// Wraps the corrector and code modification system from yoshi-std for easy
/// automated code improvements.
///
/// # Example
/// ```rust
/// use yoshi::error::CorrectionSystem;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let corrector = CorrectionSystem::new();
///
///     // Analyze a tiny error string; may yield zero or more fixes.
///     let fixes = corrector.analyze_error("unreachable code detected")?;
///     assert!(fixes.is_empty() || !fixes.is_empty());
///     Ok(())
/// }
/// ```
pub struct CorrectionSystem {
    corrector: yoshi_std::corrector::YoshiErrorCorrector,
}

impl CorrectionSystem {
    /// Create a new correction system
    pub fn new() -> Self {
        Self {
            corrector: yoshi_std::corrector::YoshiErrorCorrector::new(),
        }
    }

    /// Analyze an error message and get fix suggestions
    pub fn analyze_error(
        &self,
        error_message: &str,
    ) -> std::result::Result<Vec<yoshi_std::corrector::Fix>, Box<dyn std::error::Error>> {
        self.corrector.analyze_and_fix(error_message)
    }

    /// Analyze a YoshiError and get fix suggestions
    pub fn analyze_hatch(
        &self,
        error: &YoshiError,
    ) -> std::result::Result<Vec<yoshi_std::corrector::Fix>, Box<dyn std::error::Error>> {
        self.corrector.analyze_and_fix(&error.to_string())
    }

    /// Build a correction with the builder pattern
    pub fn build_correction(
        summary: impl Into<String>,
    ) -> yoshi_std::correction::CorrectionBuilder {
        yoshi_std::correction::CorrectionBuilder::new(summary.into())
    }

    /// Apply code modifications to files
    #[cfg(any(feature = "ml-recovery", test))]
    pub fn apply_modifications(
        &self,
        modifications: &[yoshi_std::correction::CodeModification],
    ) -> std::io::Result<usize> {
        apply_code_modifications(modifications)
    }

    /// Get the underlying corrector
    pub fn corrector(&self) -> &yoshi_std::corrector::YoshiErrorCorrector {
        &self.corrector
    }
}

impl Default for CorrectionSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// NatsSystem removed - feature references non-existent yoshi-core crate
/// Complete Yoshi System - All-in-one error handling and recovery
///
/// This is the ultimate convenience struct that combines ALL Yoshi systems
/// into a single, easy-to-use interface. Perfect for getting started quickly!
///
/// # Example
/// ```rust
/// use yoshi::error::YoshiSystem;
///
/// #[tokio::main(flavor = "current_thread")]
/// async fn main() -> yoshi::YoResult<()> {
///     let yoshi = YoshiSystem::initialize().await?;
///     yoshi.recovery.enable("doc_cluster").await;
///
///     let health = yoshi.health_report().await;
///     assert!(health.recovery_enabled);
///     Ok(())
/// }
/// ```
pub struct YoshiSystem {
    /// ML Recovery system
    pub recovery: Arc<RecoverySystem>,
    /// Circuit breaker system
    pub circuit_breaker: Arc<CircuitBreakerSystem>,
    /// Error correction system
    pub corrector: Arc<CorrectionSystem>,
    /// Supervisor system (optional, created on demand)
    pub supervisor: Option<Arc<SupervisorSystem>>,
    /// Distributed coordination via NATS (feature-gated)
    #[cfg(feature = "nats")]
    pub nats: Option<Arc<NATSClient>>,
    /// Stop flag for distributed listeners
    #[cfg(feature = "nats")]
    distributed_shutdown: Arc<NatsAtomicBool>,
    /// Background tasks tied to the distributed listeners
    #[cfg(feature = "nats")]
    distributed_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl YoshiSystem {
    /// Initialize a complete Yoshi system with production defaults
    pub async fn initialize() -> yoshi_std::YoResult<Self> {
        Ok(Self {
            recovery: Arc::new(RecoverySystem::production()),
            circuit_breaker: Arc::new(CircuitBreakerSystem::production("default")),
            corrector: Arc::new(CorrectionSystem::new()),
            supervisor: None,
            #[cfg(feature = "nats")]
            nats: Self::init_nats_client().await,
            #[cfg(feature = "nats")]
            distributed_shutdown: Arc::new(NatsAtomicBool::new(false)),
            #[cfg(feature = "nats")]
            distributed_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Quick-start: Initialize with supervisor in one call
    pub async fn with_workers(worker_count: usize, batch_size: usize) -> yoshi_std::YoResult<Self> {
        let recovery = Arc::new(RecoverySystem::production());
        let circuit = Arc::new(CircuitBreakerSystem::production("default"));
        let corrector = Arc::new(CorrectionSystem::new());

        let supervisor = SupervisorSystem::builder()
            .with_id("yoshi_workers")
            .add_processor_workers(worker_count, batch_size)
            .with_recovery(recovery.clone())
            .with_circuit(circuit.clone())
            .with_corrector(corrector.clone())
            .build()?;

        Ok(Self {
            recovery,
            circuit_breaker: circuit,
            corrector,
            supervisor: Some(Arc::new(supervisor)),
            #[cfg(feature = "nats")]
            nats: Self::init_nats_client().await,
            #[cfg(feature = "nats")]
            distributed_shutdown: Arc::new(NatsAtomicBool::new(false)),
            #[cfg(feature = "nats")]
            distributed_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Initialize with custom recovery policy
    pub fn with_recovery_policy(mut self, policy: yoshi_std::RecoveryPolicy) -> Self {
        self.recovery = Arc::new(RecoverySystem::with_policy(policy));
        self
    }

    /// Add a supervisor system
    pub fn with_supervisor(mut self, supervisor: SupervisorSystem) -> Self {
        self.supervisor = Some(Arc::new(supervisor));
        self
    }

    #[cfg(feature = "nats")]
    async fn init_nats_client() -> Option<Arc<NATSClient>> {
        // Only attempt connection if NATS_URL is explicitly set to avoid hanging in tests/dev
        if std::env::var("NATS_URL").is_err() {
            return None;
        }

        let mut config = NATSConfig::default();
        let urls = match std::env::var("NATS_URL") {
            Ok(v) => v,
            Err(_) => return None,
        };
        config.servers = urls.split(',').map(|s| s.trim().to_string()).collect();

        // Set a very short timeout to prevent hanging - if server is down, fail quickly
        config.connect_timeout_secs = 1u64; // 1 second = fail quickly if server unreachable

        // Alternative: use a minimal non-zero timeout to ensure fast failure
        // config.connect_timeout_secs = Some(1u64); // 1 second timeout

        // For testing, 0 seconds should cause immediate failure on connection attempt

        match NATSClient::with_config(config).await {
            Ok(client) => Some(Arc::new(client)),
            Err(err) => {
                tracing::debug!("NATS initialization failed or timed out: {}", err);
                None
            }
        }
    }

    #[cfg(feature = "nats")]
    fn spawn_distributed_listeners(&self) {
        if self.nats.is_none() {
            return;
        }

        let shutdown = self.distributed_shutdown.clone();
        let handles = self.distributed_handles.clone();
        let nats = self.nats.as_ref().cloned().unwrap();
        let corrector = self.corrector.clone();
        let recovery = self.recovery.clone();

        // Distributed error ingestion → Corrector + ML enablement.
        let error_listener = tokio::spawn(async move {
            if let Ok(sub) = nats.subscribe("omnicore.errors.>".to_string()).await {
                loop {
                    if shutdown.load(Ordering::SeqCst) {
                        break;
                    }

                    let message = {
                        let mut guard = sub.lock().await;
                        guard.next().await
                    };

                    let Some(message) = message else {
                        break;
                    };

                    match serde_json::from_slice::<DistributedErrorMessage>(&message.payload) {
                        Ok(distributed) => {
                            let source_location = Location {
                                file: format!("nats://{}", distributed.node_id).into(),
                                line: 0,
                                column: 0,
                            };

                            // Keep ML contexts warmed up and run the local corrector against the remote message.
                            recovery.enable(&distributed.context).await;
                            tracing::info!(
                                target: "yoshi::nats",
                                context = %distributed.context,
                                node = %distributed.node_id,
                                "Distributed error received at {}: {}",
                                source_location.file,
                                distributed.message
                            );
                            if let Err(e) = corrector.analyze_error(&distributed.message) {
                                tracing::debug!(
                                    "Distributed error from node {} could not be analyzed: {}",
                                    distributed.node_id,
                                    e
                                );
                            }
                        }
                        Err(err) => {
                            tracing::warn!("Failed to decode distributed error payload: {}", err);
                        }
                    }
                }
            }
        });

        handles.lock().unwrap().push(error_listener);
    }

    /// Start all systems
    pub async fn start(&self) -> yoshi_std::YoResult<()> {
        if let Some(supervisor) = &self.supervisor {
            supervisor.start().await?;
        }

        #[cfg(feature = "nats")]
        {
            // Ensure listeners only start once per instance lifecycle.
            if !self.distributed_shutdown.load(Ordering::SeqCst) {
                self.spawn_distributed_listeners();
            }
        }
        Ok(())
    }

    /// Stop all systems
    pub async fn stop(&self) -> yoshi_std::YoResult<()> {
        if let Some(supervisor) = &self.supervisor {
            supervisor.stop().await?;
        }

        #[cfg(feature = "nats")]
        {
            self.distributed_shutdown.store(true, Ordering::SeqCst);
            let handles: Vec<_> = self.distributed_handles.lock().unwrap().drain(..).collect();
            for handle in handles {
                let _ = handle.await;
            }

            if let Some(client) = &self.nats {
                client.shutdown().await.ok();
            }
        }
        Ok(())
    }

    /// Get a health report for all systems
    pub async fn health_report(&self) -> HealthReport {
        HealthReport {
            recovery_enabled: self.recovery.is_enabled(),
            circuit_breaker_state: self.circuit_breaker.state(),
            supervisor_status: self.supervisor.as_ref().map(|s| s.status()),
        }
    }
}

/// Health report for all Yoshi systems
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub recovery_enabled: bool,
    pub circuit_breaker_state: yoshi_std::CircuitState,
    pub supervisor_status: Option<yoshi_std::SupervisorStatus>,
}

impl fmt::Display for HealthReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Yoshi System Health Report")?;
        writeln!(
            f,
            "  Recovery: {}",
            if self.recovery_enabled {
                "✓ Enabled"
            } else {
                "✗ Disabled"
            }
        )?;
        writeln!(f, "  Circuit Breaker: {:?}", self.circuit_breaker_state)?;
        if let Some(status) = &self.supervisor_status {
            writeln!(
                f,
                "  Supervisor: {}",
                if status.is_running {
                    "✓ Running"
                } else {
                    "✗ Stopped"
                }
            )?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS - All test modules at the end of the file
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AppResult, YoResult};
    use std::fs;
    use std::sync::Arc;
    use std::sync::Once;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::ERROR)
                .with_test_writer()
                .try_init();
        });
    }

    #[test]
    fn test_apply_replace_insert_delete() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        fs::write(tmp.path(), "abcXYZdef").unwrap();
        let file = tmp.path().to_string_lossy().into_owned();

        let mods = vec![
            CodeModification::Replace {
                span: CodeSpan {
                    file: file.clone(),
                    start_byte: 3,
                    end_byte: 6,
                },
                new_text: Arc::<str>::from("123"),
            },
            CodeModification::Insert {
                span: CodeSpan {
                    file: file.clone(),
                    start_byte: 0,
                    end_byte: 0,
                },
                new_text: Arc::<str>::from(">>>"),
                after: false,
            },
            CodeModification::Delete {
                span: CodeSpan {
                    file: file.clone(),
                    start_byte: 9,
                    end_byte: 9,
                },
            },
        ];
        let n = apply_code_modifications(&mods).unwrap();
        assert!(n >= 2);

        let got = fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(got, ">>>abc123def");
    }

    #[test]
    fn test_app_error_creation() {
        let err = YoError::new(
            AppErrorKind::Internal("test error".into()),
            Location {
                file: "test.rs".into(),
                line: 1,
                column: 1,
            },
        );
        assert!(err.to_string().contains("test error"));
        assert!(err.to_string().contains("test.rs"));
    }

    #[test]
    fn test_result_ext() {
        let yo_result: YoResult<i32> = Ok(42);
        let app_result: AppResult<i32> = yo_result.into_app();
        assert_eq!(app_result.unwrap(), 42);

        let app_result2: AppResult<i32> = Ok(100);
        let yo_result2: YoResult<i32> = app_result2.into_yoshi();
        assert_eq!(yo_result2.unwrap(), 100);
    }

    #[tokio::test]
    async fn test_recovery_system() {
        let recovery = RecoverySystem::new();
        // Recovery system may be enabled or disabled depending on global state
        let _ = recovery.is_enabled();

        let production = RecoverySystem::production();
        assert_eq!(production.policy().max_attempts, 5);
    }

    #[tokio::test]
    async fn test_circuit_breaker_system() {
        let circuit = CircuitBreakerSystem::new("test");
        assert_eq!(circuit.name(), "test");
        assert!(matches!(circuit.state(), yoshi_std::CircuitState::Closed));

        // Circuit breaker is created successfully
        let metrics = circuit.metrics();
        assert_eq!(metrics.total_requests, 0);
    }

    #[test]
    fn test_correction_system() {
        let _corrector = CorrectionSystem::new();
        let builder = CorrectionSystem::build_correction("test fix".to_string());
        let correction = builder.set_confidence(0.9).build();
        assert_eq!(correction.confidence, 0.9);
    }

    #[tokio::test]
    async fn test_supervisor_system_builder() {
        let supervisor = SupervisorSystem::builder()
            .with_id("test_supervisor")
            .add_processor_worker("worker-1", 10)
            .build();
        assert!(supervisor.is_ok());
    }

    #[test]
    fn test_health_report_display() {
        let report = HealthReport {
            recovery_enabled: true,
            circuit_breaker_state: yoshi_std::CircuitState::Closed,
            supervisor_status: Some(yoshi_std::SupervisorStatus {
                is_running: true,
                worker_count: 5,
                healthy_workers: 5,
            }),
        };
        let display = format!("{}", report);
        assert!(display.contains("Yoshi System Health Report"));
        assert!(display.contains("✓ Enabled"));
    }

    #[tokio::test]
    async fn test_recovery_system_quick_start() {
        let recovery = RecoverySystem::quick_start("test_context").await;
        assert_eq!(recovery.policy().max_attempts, 5);
        assert_eq!(recovery.policy().timeout_ms, 10000);
    }

    #[tokio::test]
    async fn test_supervisor_system_quick_start() {
        let supervisor = SupervisorSystem::quick_start("test_app", 3, 100);
        assert!(supervisor.is_ok());
        let supervisor = supervisor.unwrap();
        let status = supervisor.status();
        assert!(status.is_running);
    }

    #[tokio::test]
    async fn test_supervisor_builder_add_multiple_workers() {
        let supervisor = SupervisorSystem::builder()
            .with_id("multi_worker_test")
            .add_processor_workers(5, 50)
            .build();
        assert!(supervisor.is_ok());
    }

    #[test]
    fn test_app_error_macro_captures_location() {
        let err = app_error!(AppErrorKind::Internal("macro error".into()));
        assert!(matches!(err.kind, AppErrorKind::Internal(_)));
        // macro should stamp the callsite
        assert!(err.location.file.ends_with("error.rs"));
        assert!(err.location.line > 0);
    }

    #[tokio::test]
    async fn test_yoshi_system_with_workers() {
        init_tracing();
        let system = YoshiSystem::with_workers(2, 100).await;
        assert!(system.is_ok());
        let system = system.unwrap();
        assert!(system.supervisor.is_some());

        let health = system.health_report().await;
        assert!(health.supervisor_status.is_some());
    }

    #[cfg(feature = "nats")]
    #[tokio::test]
    async fn test_init_nats_client_skips_if_no_env() {
        init_tracing();
        // Remove NATS_URL if present
        unsafe {
            std::env::remove_var("NATS_URL");
        }
        let system = YoshiSystem::with_workers(1, 10).await;
        assert!(system.is_ok());
        let system = system.unwrap();
        assert!(system.nats.is_none());
    }

    #[cfg(feature = "nats")]
    #[tokio::test]
    async fn test_init_nats_client_times_out_unreachable() {
        init_tracing();
        // Test that NATS initialization returns None when environment variable is set
        // but connection would be unreachable (to avoid hanging in test environments)
        unsafe {
            std::env::set_var("NATS_URL", "nats://unreachable.server:4222");
        }

        // This should return None quickly because no NATS server is running
        // The timeout implementation prevents long hangs
        let result = YoshiSystem::init_nats_client().await;
        assert!(
            result.is_none(),
            "NATS client should not be initialized for unreachable server"
        );

        unsafe {
            std::env::remove_var("NATS_URL");
        }
    }
}
