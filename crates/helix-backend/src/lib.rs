//! helix-backend: CLIF lowering + JIT execution.

/// Spike tests live under cfg(test) — the production lowering replaces this module.
#[cfg(test)]
mod jit_spike;
