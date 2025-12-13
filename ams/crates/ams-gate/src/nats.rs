/* crates/ams-gate/src/nats.rs */
//! Local NATS launcher for integration tests and development.
//!
//! # AMS Gate – NATS Module
//!▫~•◦-------------------------------------‣
//!
//! Spawns a local `nats-server` process, waits for readiness, and tears it down
//! gracefully. Designed to keep Yoshi/NATS workflows runnable without manual setup.
//!
//! ### Key Capabilities
//! - **Spawn:** Launches a child `nats-server` with configurable host/port/binary.
//! - **Probe:** Waits for TCP readiness with bounded retries.
//! - **Shutdown:** Requests termination and waits for exit; best-effort kill if needed.
//!
//! ### Example
//! ```no_run
//! use ams_gate::nats::{NatsConfig, NatsServer};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), ams_gate::nats::NatsGateError> {
//! let server = NatsServer::start(NatsConfig::default()).await?;
//! // ... run clients/tests ...
//! server.stop().await?;
//! # Ok(())
//! # }
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::{
    net::{SocketAddr, ToSocketAddrs},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use flate2::read::GzDecoder;
use reqwest::Client;
use tokio::{
    fs,
    net::TcpStream,
    process::{Child, Command},
    time::{sleep, timeout},
};
use tracing::{debug, info};
use which::which;
use yoshi_derive::AnyError;

/// Configuration for launching a local NATS server.
#[derive(Debug, Clone)]
pub struct NatsConfig {
    /// Host to bind.
    pub host: String,
    /// Port to bind.
    pub port: u16,
    /// Path to the `nats-server` binary (default: `nats-server` in PATH).
    pub binary: String,
    /// Max time to wait for readiness.
    pub ready_timeout: Duration,
    /// Silence child output when false.
    pub log_output: bool,
}

impl Default for NatsConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 4222,
            binary: "nats-server".into(),
            ready_timeout: Duration::from_secs(5),
            log_output: false,
        }
    }
}

/// Errors produced by the NATS gateway.
#[derive(Debug, AnyError)]
pub enum NatsGateError {
    #[anyerror("failed to spawn nats-server: {source}")]
    Spawn {
        #[from]
        source: std::io::Error
    },
    #[anyerror("nats-server not ready on {addr} after {timeout_ms} ms")]
    ReadyTimeout { addr: String, timeout_ms: u64 },
    #[anyerror("failed to stop nats-server: {source}")]
    Stop { source: std::io::Error },
    #[anyerror("nats-server exited unexpectedly: code={code:?}")]
    EarlyExit { code: Option<i32> },
    #[anyerror("unsupported platform for nats-server prefetch: {platform}")]
    UnsupportedPlatform { platform: String },
    #[anyerror("failed to download nats-server: {source}")]
    Download { source: reqwest::Error },
    #[anyerror("failed to persist nats-server: {source}")]
    Persist { source: std::io::Error },
    #[anyerror("failed to extract nats-server archive: {source}")]
    Extract { source: std::io::Error },
}

/// Handle to a running local NATS server process.
pub struct NatsServer {
    child: Child,
    address: String,
}

impl NatsServer {
    /// Start a local NATS server and wait until it is reachable.
    pub async fn start(config: NatsConfig) -> Result<Self, NatsGateError> {
        let addr = format!("{}:{}", config.host, config.port);
        let binary_path = ensure_binary(&config.binary).await?;
        let mut cmd = Command::new(&binary_path);
        cmd.arg("--addr")
            .arg(&config.host)
            .arg("--port")
            .arg(config.port.to_string())
            .kill_on_drop(false);

        if !config.log_output {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }

        let mut child = cmd
            .spawn()
            .map_err(|source| NatsGateError::Spawn { source })?;
        debug!(
            "Spawned nats-server pid={:?} on {} (bin={})",
            child.id(),
            addr,
            binary_path
        );

        // Wait for readiness with bounded timeout.
        let ready = timeout(
            config.ready_timeout,
            wait_for_ready(&addr, config.ready_timeout),
        )
        .await;
        match ready {
            Ok(Ok(())) => {
                info!("nats-server ready at {}", addr);
                Ok(Self {
                    child,
                    address: addr,
                })
            }
            Ok(Err(e)) => {
                // Child may have exited; try to observe status.
                if let Some(status) = child.try_wait().ok().flatten() {
                    let _ = child.kill().await;
                    return Err(NatsGateError::EarlyExit {
                        code: status.code(),
                    });
                }
                let _ = child.kill().await;
                Err(e)
            }
            Err(_) => {
                let _ = child.kill().await;
                Err(NatsGateError::ReadyTimeout {
                    addr,
                    timeout_ms: config.ready_timeout.as_millis() as u64,
                })
            }
        }
    }

    /// NATS URL suitable for clients (`nats://host:port`).
    pub fn address(&self) -> String {
        format!("nats://{}", self.address)
    }

    /// Stop the server, allowing it to exit cleanly.
    pub async fn stop(mut self) -> Result<(), NatsGateError> {
        // Attempt graceful shutdown first.
        if let Err(e) = self.child.start_kill() {
            return Err(NatsGateError::Stop { source: e });
        }
        self.child
            .wait()
            .await
            .map(|_| ())
            .map_err(|source| NatsGateError::Stop { source })
    }
}

async fn wait_for_ready(addr: &str, deadline: Duration) -> Result<(), NatsGateError> {
    // Resolve early to catch obvious mistakes.
    let addrs: Vec<SocketAddr> = addr
        .to_socket_addrs()
        .map_err(|source| NatsGateError::Spawn { source })?
        .collect();

    let start = tokio::time::Instant::now();
    while start.elapsed() < deadline {
        for a in &addrs {
            if TcpStream::connect(a).await.is_ok() {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(50)).await;
    }

    Err(NatsGateError::ReadyTimeout {
        addr: addr.to_string(),
        timeout_ms: deadline.as_millis() as u64,
    })
}

async fn ensure_binary(configured: &str) -> Result<String, NatsGateError> {
    // If the configured path exists, use it.
    let configured_path = Path::new(configured);
    if configured_path.exists() {
        return Ok(configured.to_string());
    }
    // Try PATH.
    if let Ok(found) = which(configured) {
        return found
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|source| NatsGateError::Persist { source });
    }

    let (url, filename) = download_url()?;
    let target_dir = PathBuf::from("target").join("tools");
    fs::create_dir_all(&target_dir)
        .await
        .map_err(|source| NatsGateError::Persist { source })?;
    let dest_bin = target_dir.join(filename);
    if dest_bin.exists() {
        return dest_bin
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|source| NatsGateError::Persist { source });
    }

    let archive_path = target_dir.join(format!("{}.download", filename));
    let client = Client::new();
    let bytes = client
        .get(url)
        .send()
        .await
        .map_err(|source| NatsGateError::Download { source })?
        .bytes()
        .await
        .map_err(|source| NatsGateError::Download { source })?;

    fs::write(&archive_path, &bytes)
        .await
        .map_err(|source| NatsGateError::Persist { source })?;

    extract_archive(&archive_path, &dest_bin).await?;
    fs::remove_file(&archive_path).await.ok();

    dest_bin
        .canonicalize()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|source| NatsGateError::Persist { source })
}

fn download_url() -> Result<(&'static str, &'static str), NatsGateError> {
    let os = std::env::consts::OS;
    match os {
        "windows" => Ok((
            "https://github.com/nats-io/nats-server/releases/latest/download/nats-server-windows-amd64.zip",
            "nats-server.exe",
        )),
        "linux" => Ok((
            "https://github.com/nats-io/nats-server/releases/latest/download/nats-server-linux-amd64.tar.gz",
            "nats-server",
        )),
        "macos" => Ok((
            "https://github.com/nats-io/nats-server/releases/latest/download/nats-server-darwin-amd64.tar.gz",
            "nats-server",
        )),
        other => Err(NatsGateError::UnsupportedPlatform {
            platform: other.to_string(),
        }),
    }
}

async fn extract_archive(archive_path: &Path, dest_bin: &Path) -> Result<(), NatsGateError> {
    let path_str = archive_path.to_string_lossy().to_ascii_lowercase();
    if path_str.ends_with(".zip") {
        extract_zip(archive_path, dest_bin).await
    } else if path_str.ends_with(".tar.gz") {
        extract_targz(archive_path, dest_bin).await
    } else {
        Err(NatsGateError::Extract {
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "unsupported archive format",
            ),
        })
    }
}

async fn extract_zip(archive_path: &Path, dest_bin: &Path) -> Result<(), NatsGateError> {
    let data = fs::read(archive_path)
        .await
        .map_err(|source| NatsGateError::Extract { source })?;
    let dest = dest_bin.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<(), NatsGateError> {
        let reader = std::io::Cursor::new(data);
        let mut zip = zip::ZipArchive::new(reader).map_err(|source| NatsGateError::Extract {
            source: io_error(source),
        })?;
        for i in 0..zip.len() {
            let mut file = zip.by_index(i).map_err(|source| NatsGateError::Extract {
                source: io_error(source),
            })?;
            let name = file.name().to_string();
            if name.ends_with("nats-server.exe") || name.ends_with("nats-server") {
                let mut out = std::fs::File::create(&dest)
                    .map_err(|source| NatsGateError::Extract { source })?;
                std::io::copy(&mut file, &mut out)
                    .map_err(|source| NatsGateError::Extract { source })?;
                return Ok(());
            }
        }
        Err(NatsGateError::Extract {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "nats-server binary not found in zip",
            ),
        })
    })
    .await
    .map_err(|e| NatsGateError::Extract {
        source: io_error(e),
    })?
}

async fn extract_targz(archive_path: &Path, dest_bin: &Path) -> Result<(), NatsGateError> {
    let data = fs::read(archive_path)
        .await
        .map_err(|source| NatsGateError::Extract { source })?;
    let dest = dest_bin.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<(), NatsGateError> {
        let tar = GzDecoder::new(&data[..]);
        let mut archive = tar::Archive::new(tar);
        for entry in archive.entries().map_err(|source| NatsGateError::Extract {
            source: io_error(source),
        })? {
            let mut entry = entry.map_err(|source| NatsGateError::Extract {
                source: io_error(source),
            })?;
            if let Ok(path) = entry.path() {
                let name = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
                if name == "nats-server" {
                    let mut out = std::fs::File::create(&dest)
                        .map_err(|source| NatsGateError::Extract { source })?;
                    std::io::copy(&mut entry, &mut out)
                        .map_err(|source| NatsGateError::Extract { source })?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mut perms = out
                            .metadata()
                            .map_err(|source| NatsGateError::Extract { source })?
                            .permissions();
                        perms.set_mode(0o755);
                        out.set_permissions(perms)
                            .map_err(|source| NatsGateError::Extract { source })?;
                    }
                    return Ok(());
                }
            }
        }
        Err(NatsGateError::Extract {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "nats-server binary not found in tar.gz",
            ),
        })
    })
    .await
    .map_err(|e| NatsGateError::Extract {
        source: io_error(e),
    })?
}

fn io_error<E: std::error::Error>(err: E) -> std::io::Error {
    std::io::Error::other(err.to_string())
}
