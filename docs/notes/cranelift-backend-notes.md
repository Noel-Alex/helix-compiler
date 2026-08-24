# The Cranelift Backend (course notes)

*How HELIX IR becomes x86-64 machine code — without writing an assembler.*

## Why Cranelift

Writing a native code generator means instruction selection, register allocation,
prologue/epilogue generation, and relocation handling — a project by itself. Cranelift
(the WebAssembly runtime's compiler) provides all of it behind a small typed IR (CLIF),
so HELIX's backend is a *lowering*: one pass from our SSA to theirs.

The pedagogical payoff: HELIX's φ-functions lower **1:1 onto Cranelift block parameters**
(`jump bb3(x1)` carries edge values; every predecessor supplies its own). No out-of-SSA
translation needed — lost-copy and swap-cycle problems become regalloc2's job.

## Verified API facts (0.135, Aug 2026 — most tutorials are stale)

- Flow: `settings::builder()` → `cranelift_native::builder().finish(flags)` →
  `JITBuilder::with_isa(isa, default_libcall_names())` → declare/define →
  `finalize_definitions() -> ModuleResult<()>` → `get_finalized_function(FuncId)`.
- **WindowsFastcall everywhere** — it equals Rust's `extern "C"` on windows-msvc.
  SystemV signatures corrupt silently past 4 arguments.
- Terminator args are `BlockArg::Value(v)` (new sum type in 0.135).
- Memory flags rebuilt as builders: `MemFlagsData::trusted().with_notrap().with_aligned()`.
- Address operands must be pointer-width: indices get `sextend.i64` before address math.
- Host functions via `JITBuilder::symbol(name, ptr)` before module creation; imports
  declared with `Linkage::Import`.
- Keep `is_pic=false`, `use_colocated_libcalls=false`. Wrap JIT calls in host-side
  `catch_unwind`; keep the JITModule alive while any thread runs its code.

## Lowering rules per construct

| HELIX | CLIF |
|---|---|
| `Const/Bin/Unary/Cast` | `iconst/fconst/iadd/fmul/sdiv/fcvt_to_int_sat/...` |
| int `/`,`%` | guard (`icmp imm 0` + MIN/-1 pair) → trap block, else `sdiv/srem` |
| comparisons | `icmp(IntCC)` / `fcmp(FloatCC)` |
| `a[i]` load/store | `sextend idx`, `imul stride`, `iadd base`, guarded vs len, `load/store` |
| φ | block param + per-pred terminator args |
| user call | `declare_func_in_func` + `call` (arrays = fat ptr+len pair) |
| builtins | print→host import; zeros→host alloc; len→fat len; abs/sqrt→`fabs/sqrt`; min/max→`fmin/fmax`(floats)/select-cmp(ints) |

## Checked semantics as generated branches (not OS faults)

Div-by-zero under a JIT would surface as an OS exception needing SEH/vectored handlers —
fragile and Windows-specific. Instead every division gets an explicit compare-and-branch
to a panic stub calling imported `helix_panic(code,line)`, which prints
`runtime error: ... at line N` and exits — byte-identical messages to the interpreter.
Bounds checks work the same way; `--unchecked` strips them (div guards stay).

## Parallel regions

Approved loops become extracted body functions `extern "C" fn(i64 iter, *const Ctx)`;
main calls imported `helix_parallel_for(start,end,body_id,nthreads)`. The body_id indirection
lets the HOST registry bind real code pointers only after `finalize_definitions()`
(embedding not-yet-known pointers in JIT code is impossible). The ctx struct packs array
fat pointers + captured scalars at fixed offsets.

## Compile-time cost

JIT compilation of a kernel takes single-digit milliseconds — reported separately in the
benchmark tables so steady-state execution numbers stay honest.
