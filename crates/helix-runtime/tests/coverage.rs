//! Integration tests: chunk-coverage invariants and cross-stage correctness,
//! exercised through helix-runtime's public API only (as the M10 backend will
//! use it).
//!
//! Chunk coverage contract: for every schedule, the union of executed chunks
//! equals the region `[start, end)` exactly — no iteration lost, none run
//! twice. We verify this by recording per-iteration hit counts in a shared
//! table through a registered body.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use helix_runtime::{GRAIN, RuntimeStage, helix_parallel_for, register_body, set_stage};

/// Serializes env mutation across this binary's tests.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn set_env(key: &str, value: &str) {
    // SAFETY: callers hold ENV_LOCK.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    // SAFETY: callers hold ENV_LOCK.
    unsafe {
        std::env::remove_var(key);
    }
}

// --- Per-iteration hit table shared with a non-capturing body --------------

const TABLE_N: usize = 1 << 16;
static TABLE: [AtomicU64; TABLE_N] = {
    // const-friendly init via repeated AtomicU64::new(0)
    let _ = 0;
    [const { AtomicU64::new(0) }; TABLE_N]
};

extern "C" fn record_hit(i: i64, _ctx: *mut u8) {
    debug_assert!((0..TABLE_N as i64).contains(&i));
    if (0..TABLE_N as i64).contains(&i) {
        TABLE[i as usize].fetch_add(1, Ordering::Relaxed);
    }
}

fn reset_table(n: usize) {
    for slot in &TABLE[..n] {
        slot.store(0, Ordering::Relaxed);
    }
}

fn table_ok(n: usize) -> bool {
    TABLE[..n].iter().all(|s| s.load(Ordering::Relaxed) == 1)
}

#[test]
fn chunk_coverage_union_equals_range_no_overlap() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_body(900_001, record_hit);
    set_stage(RuntimeStage::Pool);

    for sched in ["static", "dynamic", "guided"] {
        set_env("HELIX_SCHEDULE", sched);
        for &(n, p) in &[
            (GRAIN as usize * 2, 2),
            (GRAIN as usize * 5, 5),
            (20_000, 9),
            (TABLE_N - 1, 11),
            (TABLE_N, 13),
        ] {
            reset_table(n);
            helix_parallel_for(0, n as i64, 900_001, p as i64);
            assert!(
                table_ok(n),
                "{sched} n={n} p={p}: some iterations ran 0 or 2+ times"
            );
        }
    }
    remove_env("HELIX_SCHEDULE");
}

#[test]
fn offset_regions_preserve_iteration_values() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_body(900_002, record_hit);
    set_stage(RuntimeStage::ScopeThreads);

    for sched in ["static", "guided"] {
        set_env("HELIX_SCHEDULE", sched);
        // Region does not start at zero; recorded indices must be the actual
        // iteration values (start + offset), proving value (not rank) passing.
        let (start, end) = (GRAIN * 4, GRAIN * 4 + 8_192);
        reset_table(end as usize);
        helix_parallel_for(start, end, 900_002, 6);
        assert!(
            TABLE[start as usize..end as usize]
                .iter()
                .all(|s| s.load(Ordering::Relaxed) == 1),
            "{sched}: some iteration in [{start},{end}) ran 0 or 2+ times"
        );
        assert!(
            TABLE[..start as usize]
                .iter()
                .all(|s| s.load(Ordering::Relaxed) == 0),
            "{sched}: iterations below the region start were executed"
        );
    }
    remove_env("HELIX_SCHEDULE");
}

#[test]
fn stage_switch_mid_program_is_supported() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_body(900_003, record_hit);
    set_env("HELIX_SCHEDULE", "dynamic");

    let n = GRAIN * 4;
    for stage in [RuntimeStage::ScopeThreads, RuntimeStage::Pool] {
        set_stage(stage);
        reset_table(n as usize);
        helix_parallel_for(0, n, 900_003, 4);
        assert!(table_ok(n as usize), "stage {stage:?} lost work");
    }
    set_stage(RuntimeStage::Pool);
    remove_env("HELIX_SCHEDULE");
}
