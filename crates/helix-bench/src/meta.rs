//! Environment capture for benchmark provenance (methodology rec. 12).
//!
//! Every campaign writes a [`SystemMeta`] block next to its results so a
//! number can always be traced back to the machine, compiler and runtime
//! configuration that produced it. Everything here is best-effort: any probe
//! that fails degrades to `None`/`"unknown"` instead of failing the run —
//! a missing git rev must never cost us a measurement session.
//!
//! Probes used (all at runtime, no build-script coupling):
//!
//! * CPU brand / core counts / RAM / OS — `sysinfo` 0.37
//!   (`System::cpus()[0].brand()`, `System::physical_core_count()`,
//!   `System::long_os_version()`; note 0.37 exposes the OS getters as
//!   *associated* functions on `System`).
//! * rustc version — `rustc --version` via [`std::process::Command`]. The
//!   classic `env!("RUSTC_VERSION")` build-script trick is deliberately NOT
//!   used so the bench binary records the toolchain that is actually
//!   installed on the reporting machine, not whatever baked the artifact.
//! * git revision — `git rev-parse --short HEAD` run in `repo_root`
//!   (falls back to the process CWD), `None` outside a repo or when git is
//!   unavailable.
//! * thread-count knobs — the same `HELIX_NTHREADS` /
//!   `HELIX_RUNTIME=scope|pool` environment overrides helix-runtime honours.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use sysinfo::System;

/// Static description of the machine + toolchain + runtime configuration one
/// campaign ran under. Serialized into every results file (`"meta"` key) and
/// once per directory as `meta.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemMeta {
    /// Marketing string of the first logical CPU, e.g. `"AMD Ryzen 9 7950X"`.
    pub cpu_brand: String,
    /// Logical processors visible to the process (SMT siblings included).
    pub logical_cores: usize,
    /// Physical cores when the OS reports them; `None` otherwise.
    pub physical_cores: Option<usize>,
    /// Total DRAM in bytes (`sysinfo` reports bytes, not MiB).
    pub total_ram_bytes: u64,
    /// Long OS description, e.g. `"Windows 11 Home"`.
    pub os: String,
    /// First line of `rustc --version`, e.g. `"rustc 1.93.0 (...)"`
    /// — captured by spawning the toolchain, see module docs.
    pub rustc_version: String,
    /// Short git hash of the source tree (`None` when not a git checkout).
    pub git_rev: Option<String>,
    /// ISO-8601 UTC timestamp of the campaign start.
    pub timestamp_utc: String,
    /// Effective participant cap from `HELIX_NTHREADS` (`None` = uncapped).
    pub helix_nthreads_env: Option<u64>,
    /// Stage override from `HELIX_RUNTIME`: `"scope"` | `"pool"` | `null`.
    pub helix_runtime_env: Option<String>,
    /// Whether a working JIT backend is linked into this binary
    /// ([`crate::native_availability`]); campaigns degrade to interpreter-only.
    pub jit_available: bool,
}

impl SystemMeta {
    /// Captures everything about the current machine/process.
    ///
    /// `repo_root` is searched first for the git rev (the bench harness may
    /// run from any working directory); pass `None` to probe only the
    /// inherited CWD.
    #[must_use]
    pub fn capture(repo_root: Option<&Path>) -> Self {
        let mut sys = System::new();
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string());

        Self {
            cpu_brand,
            logical_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            physical_cores: System::physical_core_count(),
            total_ram_bytes: sys.total_memory(),
            os: System::long_os_version().unwrap_or_else(|| std::env::consts::OS.to_string()),
            rustc_version: probe_rustc_version().unwrap_or_else(|| "unknown".to_string()),
            git_rev: probe_git_rev(repo_root),
            timestamp_utc: timestamp_utc(),
            helix_nthreads_env: env_u64("HELIX_NTHREADS"),
            helix_runtime_env: std::env::var("HELIX_RUNTIME")
                .ok()
                .filter(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "scope" | "pool")),
            jit_available: crate::native_availability() == crate::NativeAvailability::Ready,
        }
    }

    /// One-line human summary for console banners.
    #[must_use]
    pub fn banner(&self) -> String {
        format!(
            "{} | {}L cores{} | {:.1} GiB | {} | {} | rev {}",
            self.cpu_brand,
            self.logical_cores,
            match self.physical_cores {
                Some(p) => format!(" ({p}P)"),
                None => String::new(),
            },
            self.total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.os,
            self.rustc_version,
            self.git_rev.as_deref().unwrap_or("?"),
        )
    }
}

/// Runs `rustc --version`; returns `None` if the toolchain is not on PATH.
fn probe_rustc_version() -> Option<String> {
    let out = Command::new("rustc").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(text.lines().next()?.trim().to_string())
}

/// Runs `git rev-parse --short HEAD` in `root` (or the CWD); `None` when the
/// directory is not a repository or git is unavailable.
fn probe_git_rev(root: Option<&Path>) -> Option<String> {
    let mut cmd = Command::new("git");
    cmd.arg("rev-parse").arg("--short").arg("HEAD");
    if let Some(dir) = root {
        cmd.current_dir(dir);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let rev = text.trim();
    (!rev.is_empty()).then(|| rev.to_string())
}

/// Parses an unsigned environment variable, ignoring malformed values.
fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
}

/// Current UTC time in ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`), computed from the
/// Unix epoch — no chrono dependency needed for second precision.
#[must_use]
pub fn timestamp_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 -> (y, m, d).
/// Public domain algorithm, widely used for epoch-to-civil conversion.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1); // [1, 31]
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1); // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_to_civil_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2026-08-24 is day 20689 after the epoch.
        assert_eq!(civil_from_days(20_689), (2026, 8, 24));
        // Leap-year boundary: 2000-02-29.
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
    }

    #[test]
    fn timestamp_has_iso_shape() {
        let ts = timestamp_utc();
        assert_eq!(ts.len(), 20, "{ts}");
        assert!(ts.ends_with('Z'), "{ts}");
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[10..11], "T");
    }

    #[test]
    fn capture_produces_sane_meta_inside_this_repo() {
        let meta = SystemMeta::capture(Some(Path::new(env!("CARGO_MANIFEST_DIR"))));
        // The workspace is a git checkout, so a rev should resolve.
        assert!(
            meta.git_rev.as_ref().is_some_and(|r| !r.is_empty()),
            "git rev unresolved: {meta:?}"
        );
        assert!(
            meta.rustc_version.starts_with("rustc "),
            "{}",
            meta.rustc_version
        );
        assert!(meta.logical_cores >= 1);
        assert!(!meta.cpu_brand.is_empty());
    }
}
