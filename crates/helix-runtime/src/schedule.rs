//! Chunk planning for HELIX parallel regions.
//!
//! This module owns the *math* of work distribution; [`crate::pool`] owns the
//! threads. Three schedules are provided, matching the OpenMP/libgomp models
//! studied in `docs/research/parallel-runtime.md`:
//!
//! * [`SchedKind::Static`] — equal contiguous chunks via the GCC libgomp
//!   formula (`q = n / P`, first `n % P` threads get one extra iteration),
//!   with interior boundaries snapped up to 64-byte element multiples so
//!   adjacent workers never stride the same cache line of the output array.
//! * [`SchedKind::Guided`] — chunk = `max(min_chunk, remaining / P)` claimed
//!   with a single relaxed `fetch_add` on one shared next-index counter
//!   (libgomp `gomp_loop_guided_next`). Chunk size decays as the region drains.
//! * [`SchedKind::Dynamic`] — fixed-size claims (`chunk = min_chunk`) from the
//!   same counter. Balances irregular work at the cost of cacheline ping-pong.

use std::sync::atomic::{AtomicU64, Ordering};

/// Elements per 64-byte cache line for the narrowest HELIX element type
/// (`i32`/`f32`, 4 bytes). Static chunk boundaries are snapped to multiples of
/// this step (anchored at the region start); an `f64` array (8 B/elem) then
/// lands on even multiples of its own line, so both element widths stay
/// line-aligned. This is "alignment where cheap": snapping is skipped outright
/// for small regions (see [`static_boundaries`]).
pub const CACHE_LINE_ELEMENTS: i64 = 16;

/// Lower bound on the guided/dynamic chunk size (elements). Research digest:
/// chunk sizes below 8 destroy throughput twice over — counter cacheline
/// ping-pong plus broken per-thread prefetch streams.
pub const MIN_CHUNK_DEFAULT: u64 = 8;

/// Padding width for shared hot counters/flags. 128 bytes, not 64: Intel
/// spatial prefetchers pull line pairs, so neighbouring 64-byte lines still
/// interfere (crossbeam `CachePadded` convention).
pub const PAD: usize = 128;

/// Wraps a value in its own 128-byte-aligned cache region so hot atomics never
/// share a line with neighbouring fields. Deref makes the inner atomic's API
/// available directly (`pad.load(...)`, `pad.fetch_add(...)`).
#[derive(Debug, Default)]
#[repr(align(128))]
pub(crate) struct Pad<T>(pub(crate) T);

impl<T> std::ops::Deref for Pad<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// One half-open chunk of iterations `[start, end)` handed to a worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// First iteration index (inclusive).
    pub start: i64,
    /// End iteration index (exclusive).
    pub end: i64,
}

impl Chunk {
    /// Number of iterations in this chunk (`end - start`).
    pub fn len(&self) -> i64 {
        self.end.saturating_sub(self.start)
    }

    /// True when the chunk contains no iterations.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Work-distribution strategy for one parallel region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedKind {
    /// Equal contiguous chunks, one per participant (libgomp static formula).
    Static,
    /// Fixed-size claims of `min_chunk` iterations from a shared counter.
    Dynamic,
    /// Decaying claims of `max(min_chunk, remaining / P)` (OpenMP `guided`).
    Guided,
}

impl SchedKind {
    /// Parses a schedule name; accepts the `HELIX_SCHEDULE` spellings
    /// `static`, `dynamic` and `guided` (case-insensitive). Returns `None` for
    /// anything else — callers treat that as "no override", never as an error.
    pub fn parse(s: &str) -> Option<SchedKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "static" => Some(SchedKind::Static),
            "dynamic" => Some(SchedKind::Dynamic),
            "guided" => Some(SchedKind::Guided),
            _ => None,
        }
    }

    /// Canonical lowercase name (used in dumps and verbose decisions).
    pub fn name(self) -> &'static str {
        match self {
            SchedKind::Static => "static",
            SchedKind::Dynamic => "dynamic",
            SchedKind::Guided => "guided",
        }
    }
}

/// Rounds `v` up to the nearest multiple of `grid` (grid > 0).
fn round_up_to_grid(v: i64, grid: i64) -> i64 {
    ((v + grid - 1) / grid) * grid
}

/// Computes the interior chunk boundaries (offsets from `start`) of a static
/// decomposition, using the GCC libgomp formula, optionally snapped to
/// [`CACHE_LINE_ELEMENTS`] multiples.
///
/// Returns `nthreads + 1` strictly monotonic offsets beginning at `0` and
/// ending at `n`, so consecutive pairs always tile `[0, n)` exactly — the
/// union of chunks equals the region and chunks never overlap, even after
/// snapping. Snapping only runs "where cheap": for uniform-sized regions of at
/// least `8 * CACHE_LINE_ELEMENTS` iterations per thread, and only when the
/// snapped boundary stays strictly between its neighbours (no chunk is ever
/// emptied or reordered).
pub fn static_boundaries(n: i64, nthreads: usize) -> Vec<i64> {
    let p = nthreads.max(1) as i64;
    let n = n.max(0);
    let q = n / p;
    let t = n % p;

    // Raw libgomp boundaries: prefix sums of q plus one extra for the first
    // t threads. Offsets, anchored at the region start.
    let mut bounds = Vec::with_capacity(p as usize + 1);
    bounds.push(0_i64);
    let mut acc = 0_i64;
    for i in 0..p {
        acc += q + if i < t { 1 } else { 0 };
        bounds.push(acc);
    }

    // Snap interior boundaries up to cache-line element multiples so adjacent
    // workers never share a line of the output array. Skipped when chunks are
    // too small for alignment to be cheap.
    if p > 1 && q >= CACHE_LINE_ELEMENTS * 8 {
        let step = CACHE_LINE_ELEMENTS;
        let mut prev = bounds[0];
        for i in 1..p as usize {
            let raw = bounds[i];
            let cand = round_up_to_grid(raw, step);
            // Keep the boundary strictly between its neighbours: chunk i-1
            // stays non-empty (`cand > prev`) and chunk i stays non-empty
            // (`cand < bounds[i + 1]`), so tiling is preserved.
            if cand > prev && cand < bounds[i + 1] {
                bounds[i] = cand;
                prev = cand;
            } else {
                prev = raw;
            }
        }
    }

    bounds
}

/// Full static plan for a region: exactly `nthreads` non-empty chunks (when
/// `n >= nthreads`) whose union is `[start, end)` with no overlap.
pub fn static_plan(start: i64, end: i64, nthreads: usize) -> Vec<Chunk> {
    let n = end.saturating_sub(start).max(0);
    let bounds = static_boundaries(n, nthreads);
    bounds
        .windows(2)
        .map(|w| Chunk {
            start: start + w[0],
            end: start + w[1],
        })
        .collect()
}

/// The static chunk executed by participant `worker` of `nthreads`.
///
/// O(1)-ish (one boundary vector per region, computed per worker without
/// shared state — workers never contend on the static path).
pub fn static_chunk_for(start: i64, n: i64, nthreads: usize, worker: usize) -> Chunk {
    let bounds = static_boundaries(n, nthreads);
    let idx = (worker.min(nthreads.max(1) - 1)) + 1;
    Chunk {
        start: start + bounds[idx - 1],
        end: start + bounds[idx],
    }
}

/// Shared next-index counter for dynamic/guided claiming.
///
/// The counter sits alone on its own 128-byte region ([`PAD`]); every claim is
/// a relaxed `fetch_add`, mirroring libgomp's `__sync_fetch_and_add(&ws->next,
/// chunk)`.
#[derive(Debug, Default)]
pub struct ClaimCounter {
    next: Pad<AtomicU64>,
}

impl ClaimCounter {
    /// Creates a counter positioned at iteration offset `start_offset`.
    pub fn new(start_offset: u64) -> ClaimCounter {
        ClaimCounter {
            next: Pad(AtomicU64::new(start_offset)),
        }
    }

    /// Resets the counter for a fresh region covering `0..total` (offsets).
    pub fn reset(&self, start_offset: u64) {
        self.next.store(start_offset, Ordering::Relaxed);
    }

    /// Current offset (diagnostics only — racy under contention by design).
    pub fn position(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    /// Claims the next chunk of iteration offsets, or `None` when drained.
    ///
    /// `total` is the region size (offsets run `0..total`); `participants` is
    /// the number of claiming workers `P`. For [`SchedKind::Guided`] the chunk
    /// is `max(min_chunk, remaining / P)` (decaying); for
    /// [`SchedKind::Dynamic`] it is flat `min_chunk`. `min_chunk` is clamped
    /// up to 1 so every claim makes progress.
    ///
    /// Ordering: `Relaxed` suffices — the atomicity of `fetch_add` alone
    /// assigns disjoint chunks, and no other data is published through this
    /// counter (the region payload is published via the pool's release/acquire
    /// generation flip, see [`crate::pool`]). Concurrent racing claims may push
    /// the counter past `total` by at most `P * chunk`; such overshoot simply
    /// yields `None` to late claimers and cannot wrap `u64` for any real
    /// region size.
    pub fn claim(
        &self,
        total: u64,
        kind: SchedKind,
        min_chunk: u64,
        participants: u64,
    ) -> Option<(u64, u64)> {
        debug_assert!(matches!(kind, SchedKind::Dynamic | SchedKind::Guided));
        let cur = self.next.load(Ordering::Relaxed);
        if cur >= total {
            return None;
        }
        let remaining = total - cur;
        let floor = min_chunk.max(1);
        let want = match kind {
            SchedKind::Guided => (remaining / participants.max(1)).max(floor),
            _ => floor,
        };
        let got = self.next.fetch_add(want, Ordering::Relaxed);
        if got >= total {
            return None;
        }
        let hi = got.saturating_add(want).min(total);
        if hi > got {
            return Some((got, hi));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Deterministic tiny LCG so coverage fuzzing never needs a rand dep.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
    }

    fn assert_tiling(chunks: &[Chunk], start: i64, end: i64) {
        let mut cursor = start;
        for c in chunks {
            assert_eq!(c.start, cursor, "gap/overlap at {}", cursor);
            assert!(c.end >= c.start);
            cursor = c.end;
        }
        assert_eq!(cursor, end, "union of chunks must equal the range");
    }

    #[test]
    fn static_formula_matches_libgomp() {
        // q = n/P, first n%P threads get q+1 (checked pre-snapping: small n).
        let (start, end, p) = (10, 10 + 103, 4);
        let plan = static_plan(start, end, p);
        let lens: Vec<i64> = plan.iter().map(|c| c.len()).collect();
        assert_eq!(lens, vec![26, 26, 26, 25]); // n=103: t=3 threads get q+1=26
        assert_tiling(&plan, start, end);
    }

    #[test]
    fn static_covers_all_thread_counts_and_ranges() {
        let mut rng = Lcg(0x5EED);
        for &(s, e) in &[
            (0i64, 0i64),
            (5, 5),
            (7, 8),
            (-50, 50),
            (i64::MAX - 100, i64::MAX),
        ] {
            for p in 1..=17usize {
                let plan = static_plan(s, e, p);
                assert_eq!(plan.len(), p);
                assert_tiling(&plan, s, e);
                let n = e.saturating_sub(s);
                if n >= p as i64 {
                    assert!(
                        plan.iter().all(|c| !c.is_empty()),
                        "empty chunk n={n} p={p}"
                    );
                }
            }
        }
        // Fuzzed ranges.
        for _ in 0..500 {
            let s = rng.next() as i64 % 10_000 - 5_000;
            let e = s + rng.next() as i64 % 50_000;
            let p = 1 + (rng.next() as usize % 64);
            assert_tiling(&static_plan(s, e, p), s, e);
        }
    }

    #[test]
    fn static_snaps_large_uniform_regions_to_line_multiples() {
        // 16 threads x 16384 iterations: plenty of slack, boundaries must land
        // on 16-element multiples anchored at start.
        let (start, n, p) = (0, 16 * 16384, 16);
        let plan = static_plan(start, start + n, p);
        for c in &plan[1..] {
            assert_eq!(
                c.start % CACHE_LINE_ELEMENTS,
                0,
                "boundary {} misaligned",
                c.start
            );
        }
        assert_tiling(&plan, start, start + n);
        // Small regions bypass snapping entirely ("where cheap").
        let small = static_plan(0, 33, 4);
        assert_tiling(&small, 0, 33);
    }

    #[test]
    fn static_chunk_for_agrees_with_full_plan() {
        let (s, e, p) = (-7, 9_999, 5);
        let plan = static_plan(s, e, p);
        for (i, expect) in plan.iter().enumerate() {
            assert_eq!(static_chunk_for(s, e - s, p, i), *expect);
        }
        // Out-of-range worker index clamps to the last chunk (never panics).
        assert_eq!(static_chunk_for(s, e - s, p, 999), plan[p - 1]);
    }

    #[test]
    fn schedule_parse_roundtrip() {
        assert_eq!(SchedKind::parse("static"), Some(SchedKind::Static));
        assert_eq!(SchedKind::parse(" GUIDED "), Some(SchedKind::Guided));
        assert_eq!(SchedKind::parse("Dynamic"), Some(SchedKind::Dynamic));
        assert_eq!(SchedKind::parse("work-stealing"), None);
        assert_eq!(SchedKind::Static.name(), "static");
    }

    #[test]
    fn claims_tile_the_region_single_threaded() {
        for kind in [SchedKind::Dynamic, SchedKind::Guided] {
            for &(total, mc) in &[(0u64, 1u64), (1, 8), (100, 8), (10_003, 8), (65_536, 64)] {
                let ctr = ClaimCounter::new(0);
                let mut covered = BTreeSet::new();
                while let Some((lo, hi)) = ctr.claim(total, kind, mc, 4) {
                    assert!(lo < hi && hi <= total);
                    for i in lo..hi {
                        assert!(covered.insert(i), "double claim of {i}");
                    }
                }
                assert_eq!(covered.len(), total as usize, "{kind:?} total={total}");
            }
        }
    }

    #[test]
    fn guided_chunks_decay_but_never_below_min() {
        let ctr = ClaimCounter::new(0);
        let total = 10_000u64;
        let mut first = None;
        let mut sizes = Vec::new();
        while let Some((lo, hi)) = ctr.claim(total, SchedKind::Guided, 8, 8) {
            sizes.push(hi - lo);
            if first.is_none() {
                first = Some(lo);
            }
        }
        // First guided chunk = remaining/P = 1250; interior chunks stay >= the
        // min clamp; the FINAL chunk is whatever remains (< min possible).
        assert_eq!(first, Some(0));
        assert_eq!(sizes[0], 1250);
        assert!(sizes[..sizes.len() - 1].iter().all(|&s| s >= 8));
        assert!(sizes[sizes.len() - 1] <= 8);
        assert_eq!(sizes.iter().sum::<u64>(), total);
    }

    #[test]
    fn concurrent_claims_are_disjoint_and_complete() {
        // Real contention: several OS threads racing the counter must tile the
        // region exactly, with no overlap and no lost iterations.
        for kind in [SchedKind::Dynamic, SchedKind::Guided] {
            let total = 50_000u64;
            let ctr = std::sync::Arc::new(ClaimCounter::new(0));
            let log = std::sync::Mutex::new(Vec::<(u64, u64)>::new());
            std::thread::scope(|s| {
                for _ in 0..8 {
                    let ctr = std::sync::Arc::clone(&ctr);
                    let log = &log;
                    s.spawn(move || {
                        let mut local = Vec::new();
                        while let Some(c) = ctr.claim(total, kind, 7, 8) {
                            local.push(c);
                        }
                        log.lock().unwrap_or_else(|e| e.into_inner()).extend(local);
                    });
                }
            });
            let mut got: Vec<(u64, u64)> = log.into_inner().unwrap_or_else(|e| e.into_inner());
            got.sort_unstable();
            let mut cursor = 0u64;
            for (lo, hi) in &got {
                assert_eq!(*lo, cursor, "overlap or gap in {kind:?}");
                cursor = *hi;
            }
            assert_eq!(cursor, total);
        }
    }

    #[test]
    fn padded_counter_occupies_its_own_cache_region() {
        assert_eq!(std::mem::size_of::<ClaimCounter>(), PAD);
        assert_eq!(std::mem::align_of::<ClaimCounter>(), PAD);
    }
}
