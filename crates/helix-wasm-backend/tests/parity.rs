//! End-to-end parity: every example must print the interpreter's lines and
//! fold to the interpreter's checksum inside the emitted module.
//!
//! The module runs on the dev-only `wasmi` runtime under the same host
//! contract the browser implements: print hooks collect lines, the harness
//! folds each line plus its newline through `hx_fold`, then reads
//! `hx_checksum`. A failure here is a compiler bug, so every assertion names
//! the program it came from.

use std::path::{Path, PathBuf};

use helix_ir::FuncIr;
use helix_sema::{fmt_bool, fmt_f32, fmt_f64, fmt_i64};
use helix_syntax::parse_str;
use helix_wasm_backend::{EmitError, emit_wasm};
use wasmi::{Caller, Engine, Linker, Module, Store};

/// What the host records: print lines and the trap codes it was handed.
#[derive(Default)]
struct Host {
    printed: Vec<String>,
    traps: Vec<i32>,
}

/// A host error carrying the trap payload, so the test can assert on it.
#[derive(Debug)]
struct Trap {
    code: i32,
}

impl wasmi::errors::HostError for Trap {}

impl std::fmt::Display for Trap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hx_trap(code={})", self.code)
    }
}

/// Compiles `source` the way the pipeline does: front end, IR, SSA, passes.
fn lower(source: &str) -> Result<Vec<FuncIr>, String> {
    let program = parse_str(source).map_err(|error| format!("parse: {error}"))?;
    let typed = helix_sema::check(&program).map_err(|diags| format!("check: {diags:#?}"))?;
    let mut funcs = helix_ir::build(&typed);
    for function in &mut funcs {
        helix_ir::to_ssa(function);
        helix_ir::run_optimization_pipeline(function);
    }
    Ok(funcs)
}

/// Links the six host hooks every instantiation needs.
fn link(engine: &Engine) -> Result<Linker<Host>, String> {
    let mut linker = Linker::new(engine);
    linker
        .func_wrap(
            "env",
            "hx_print_i32",
            |mut caller: Caller<'_, Host>, value: i32| {
                caller.data_mut().printed.push(fmt_i64(i64::from(value)));
                Ok(())
            },
        )
        .map_err(|error| format!("link print_i32: {error}"))?;
    linker
        .func_wrap(
            "env",
            "hx_print_i64",
            |mut caller: Caller<'_, Host>, value: i64| {
                caller.data_mut().printed.push(fmt_i64(value));
                Ok(())
            },
        )
        .map_err(|error| format!("link print_i64: {error}"))?;
    linker
        .func_wrap(
            "env",
            "hx_print_f32",
            |mut caller: Caller<'_, Host>, value: f32| {
                caller.data_mut().printed.push(fmt_f32(value));
                Ok(())
            },
        )
        .map_err(|error| format!("link print_f32: {error}"))?;
    linker
        .func_wrap(
            "env",
            "hx_print_f64",
            |mut caller: Caller<'_, Host>, value: f64| {
                caller.data_mut().printed.push(fmt_f64(value));
                Ok(())
            },
        )
        .map_err(|error| format!("link print_f64: {error}"))?;
    linker
        .func_wrap(
            "env",
            "hx_print_bool",
            |mut caller: Caller<'_, Host>, value: i32| {
                caller.data_mut().printed.push(fmt_bool(value != 0));
                Ok(())
            },
        )
        .map_err(|error| format!("link print_bool: {error}"))?;
    linker
        .func_wrap(
            "env",
            "hx_trap",
            |mut caller: Caller<'_, Host>, code: i32, _: i64, _: i64| {
                caller.data_mut().traps.push(code);
                Err::<(), _>(wasmi::Error::host(Trap { code }))
            },
        )
        .map_err(|error| format!("link trap: {error}"))?;
    Ok(linker)
}

/// One instantiation of the module, wrapped in the ABI the contract names.
struct Run {
    store: Store<Host>,
    memory: wasmi::Memory,
    run: wasmi::TypedFunc<(), ()>,
    fold: wasmi::TypedFunc<(i32, i32), ()>,
    checksum: wasmi::TypedFunc<(), i64>,
    heap: wasmi::TypedFunc<(), i32>,
}

impl Run {
    /// Instantiates the module with the browser's host contract.
    fn new(bytes: &[u8]) -> Result<Self, String> {
        let engine = Engine::default();
        let module = Module::new(&engine, bytes).map_err(|error| format!("module: {error}"))?;
        let linker = link(&engine)?;
        let mut store = Store::new(&engine, Host::default());
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|error| format!("instantiate: {error}"))?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or("module does not export memory")?;
        let run = instance
            .get_typed_func::<(), ()>(&store, "hx_run")
            .map_err(|error| format!("no hx_run export: {error}"))?;
        let fold = instance
            .get_typed_func::<(i32, i32), ()>(&store, "hx_fold")
            .map_err(|error| format!("no hx_fold export: {error}"))?;
        let checksum = instance
            .get_typed_func::<(), i64>(&store, "hx_checksum")
            .map_err(|error| format!("no hx_checksum export: {error}"))?;
        let heap = instance
            .get_typed_func::<(), i32>(&store, "hx_heap_bytes")
            .map_err(|error| format!("no hx_heap_bytes export: {error}"))?;
        Ok(Self {
            store,
            memory,
            run,
            fold,
            checksum,
            heap,
        })
    }

    /// Runs, folds the printed lines, and returns
    /// `(printed, checksum, heap_bytes)`.
    fn measure(&mut self) -> Result<(Vec<String>, u64, i32), String> {
        self.run
            .call(&mut self.store, ())
            .map_err(|error| format!("hx_run trapped: {error}"))?;
        let printed = self.store.data().printed.clone();
        for line in &printed {
            let mut bytes = line.clone().into_bytes();
            bytes.push(b'\n');
            self.memory
                .write(&mut self.store, 0, &bytes)
                .map_err(|error| format!("print scratch too small: {error}"))?;
            self.fold
                .call(&mut self.store, (0, bytes.len() as i32))
                .map_err(|error| format!("hx_fold trapped: {error}"))?;
        }
        // The checksum is an unsigned FNV value carried in an i64 register.
        let folded = self
            .checksum
            .call(&mut self.store, ())
            .map_err(|error| format!("hx_checksum: {error}"))? as u64;
        let allocated = self
            .heap
            .call(&mut self.store, ())
            .map_err(|error| format!("hx_heap_bytes: {error}"))?;
        Ok((printed, folded, allocated))
    }
}

/// Runs the module under the browser's host contract.
fn run_module(bytes: &[u8]) -> Result<(Vec<String>, u64, i32), String> {
    Run::new(bytes)?.measure()
}

/// The interpreter's answer for the same source — the oracle.
fn interpret(source: &str) -> (Vec<String>, u64) {
    let program = parse_str(source).expect("source parses");
    let typed = helix_sema::check(&program).expect("source checks");
    let output = helix_engine::run_with_source(source, &typed).expect("source runs");
    (output.printed, output.checksum)
}

/// Every `.hx` example that sema accepts, the interpreter completes, and that
/// is small enough to interpret in a test.
///
/// The benchmark-sized kernels (saxpy allocates 2 × 32 Mi `f64`) are excluded
/// with the project's own work estimate, exactly like the Observatory's
/// interactive execution gate — interpreting them here would take minutes and
/// prove nothing the teaching examples do not.
#[test]
fn examples_match_the_interpreter() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0;
    for path in examples(root) {
        let source = std::fs::read_to_string(&path).expect("readable example");
        if helix_artifact::estimated_work(&source) > 2_000_000 {
            continue;
        }
        let funcs = match lower(&source) {
            Ok(funcs) => funcs,
            Err(_) => continue, // negative fixtures are the front end's job
        };
        let (want_lines, want_checksum) = interpret(&source);
        let module = emit_wasm(&funcs).unwrap_or_else(|error| {
            panic!("{}: emit failed: {error}", path.display());
        });
        let (got_lines, got_checksum, heap) = run_module(&module.bytes)
            .unwrap_or_else(|error| panic!("{}: run failed: {error}", path.display()));
        assert_eq!(
            got_lines,
            want_lines,
            "{}: printed lines differ",
            path.display()
        );
        assert_eq!(
            got_checksum,
            want_checksum,
            "{}: checksum differs (interpreter 0x{want_checksum:016x}, module 0x{got_checksum:016x})",
            path.display()
        );
        assert!(
            heap >= 0,
            "{}: heap byte count must not be negative",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked >= 8,
        "expected the repository examples to run, saw {checked}"
    );
}

/// Inline programs that exercise shapes the examples do not all cover.
#[test]
fn inline_programs_match_the_interpreter() {
    let programs = [
        // A saxpy-shaped loop with prints, array writes and reads.
        "fn main() {\n  let n = 8;\n  let x: [f64] = zeros(n);\n  let y: [f64] = zeros(n);\n  for i in 0..n {\n    x[i] = i as f64;\n    y[i] = 0.5;\n  }\n  for i in 0..n {\n    y[i] = 2.0 * x[i] + y[i];\n    print(y[i]);\n  }\n}\n",
        // Recursion, and a call that writes the caller's array.
        "fn poke(a: [i64], v: i64) {\n  a[0] = v;\n  a[1] = v + 1;\n}\nfn fact(n: i64) -> i64 {\n  if n <= 1 { return 1; }\n  return n * fact(n - 1);\n}\nfn main() {\n  let a: [i64] = zeros(2);\n  poke(a, 7);\n  print(a[0]);\n  print(a[1]);\n  print(fact(10));\n}\n",
        // Casts, bools, short-circuit logic, min/max/sqrt/abs.
        "fn main() {\n  print(300.7 as i32);\n  print(3.9 as i64);\n  print(true);\n  print(false);\n  print(sqrt(2.0));\n  print(abs(-12));\n  print(min(3, 9));\n  print(max(3, 9));\n  let a: [bool] = zeros(2);\n  a[0] = true;\n  print(a[0]);\n  print(1 < 2 && 2 < 3);\n  print(1 > 2 || 2 > 3);\n}\n",
        // i32 storage, 32-bit arithmetic, mixed-width float printing.
        "fn main() {\n  let small: i32 = 21;\n  print(small * 2);\n  let f: f32 = 0.1;\n  print(f);\n  print(f as f64);\n  print((1 as f32) * (3 as f32));\n}\n",
        // A diamond with different arm assignments, plus nested loops.
        "fn main() {\n  let total = 0;\n  for i in 0..3 {\n    for j in 0..2 {\n      total = total + i * j;\n    }\n  }\n  print(total);\n  if total > 4 { print(1); } else { print(2); }\n  let a: [i32] = zeros(4);\n  for i in 0..4 { a[i] = (i * i) as i32; }\n  for i in 0..4 { print(a[i]); }\n}\n",
        // Deep recursion: exercises the dispatch loop's backward edges.
        "fn fib(n: i64) -> i64 {\n  if n < 2 { return n; }\n  return fib(n - 1) + fib(n - 2);\n}\nfn main() {\n  print(fib(18));\n}\n",
        // A function called before it is defined, and an early return.
        "fn early(n: i64) -> i64 {\n  if n < 0 { return 0; }\n  return n + 1;\n}\nfn main() {\n  print(early(-5));\n  print(early(4));\n  let a: [i64] = zeros(1);\n  a[0] = early(9);\n  print(a[0]);\n}\n",
    ];
    for program in programs {
        let funcs = lower(program).expect("inline program compiles");
        let (want_lines, want_checksum) = interpret(program);
        let module = emit_wasm(&funcs).expect("emit succeeds");
        let (got_lines, got_checksum, _) = run_module(&module.bytes).expect("module runs");
        assert_eq!(
            got_lines, want_lines,
            "printed lines differ for:\n{program}"
        );
        assert_eq!(
            got_checksum, want_checksum,
            "checksum differs for:\n{program}"
        );
    }
}

/// A trap must abort the compiled run and keep the lines printed before it.
#[test]
fn traps_abort_the_compiled_run() {
    let source =
        "fn main() {\n  print(1);\n  print(2);\n  let a: [i64] = zeros(2);\n  print(a[7]);\n}\n";
    let funcs = lower(source).expect("compiles");
    let module = emit_wasm(&funcs).expect("emit succeeds");
    let mut run = Run::new(&module.bytes).expect("instantiates");
    let error = run
        .run
        .call(&mut run.store, ())
        .expect_err("bounds violation must trap");
    assert!(
        error.to_string().contains("hx_trap(code=2)"),
        "unexpected trap error: {error}"
    );
    assert_eq!(
        run.store.data().printed,
        vec!["1".to_string(), "2".to_string()]
    );
    assert_eq!(run.store.data().traps, vec![2], "bounds trap code");
}

/// Division by zero and the `MIN / -1` edge trap with their own codes.
#[test]
fn division_traps_carry_their_codes() {
    let sources = [
        ("fn main() {\n  print(1);\n  let q = 10 / 0;\n}\n", 0),
        (
            "fn main() {\n  let half = 4611686018427387904;\n  let m = 0 - half - half;\n  let q = m / -1;\n}\n",
            1,
        ),
        (
            "fn main() {\n  let n = -3;\n  let a: [i64] = zeros(n);\n}\n",
            3,
        ),
    ];
    for (source, code) in sources {
        let funcs = lower(source).expect("compiles");
        let module = emit_wasm(&funcs).expect("emit succeeds");
        let mut run = Run::new(&module.bytes).expect("instantiates");
        run.run.call(&mut run.store, ()).expect_err("must trap");
        assert_eq!(
            run.store.data().traps,
            vec![code],
            "wrong code for {source}"
        );
    }
}

/// The module refuses to exist without `fn main()`.
#[test]
fn a_module_without_main_is_refused() {
    let mut funcs = lower("fn helper() -> i64 { return 1; }\nfn main() { print(helper()); }\n")
        .expect("compiles");
    funcs.retain(|function| function.name != "main");
    assert_eq!(emit_wasm(&funcs), Err(EmitError::NoMain));
}

/// Duplicate function names would make call sites ambiguous.
#[test]
fn duplicate_names_are_refused() {
    let source = "fn main() {\n  print(1);\n}\n";
    let mut funcs = lower(source).expect("compiles");
    funcs.push(funcs[0].clone());
    match emit_wasm(&funcs) {
        Err(EmitError::DuplicateName(name)) => assert_eq!(name, "main"),
        other => panic!("expected a duplicate-name error, got {other:?}"),
    }
}

/// Emitted modules must validate as real wasm, and repeat runs must agree.
#[test]
fn emitted_modules_validate_and_repeat_identically() {
    let source = "fn main() {\n  let a: [i64] = zeros(3);\n  a[0] = 4;\n  print(a[0]);\n  print(len(a));\n}\n";
    let funcs = lower(source).expect("compiles");
    let module = emit_wasm(&funcs).expect("emit succeeds");
    wasmparser::Validator::new()
        .validate_all(&module.bytes)
        .expect("module validates");

    let mut run = Run::new(&module.bytes).expect("instantiates");
    let (first_lines, first_checksum, first_heap) = run.measure().expect("first run");
    // A second run of the same instance must recycle the heap region and
    // produce identical output — that is what `hx_run`'s reset means.
    run.store.data_mut().printed.clear();
    let (second_lines, second_checksum, second_heap) = run.measure().expect("second run");
    assert_eq!(first_lines, second_lines);
    assert_eq!(first_checksum, second_checksum);
    assert_eq!(first_heap, second_heap);
}

/// The repository's examples, sorted so failures are reproducible.
fn examples(root: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root.join("../../examples"))
        .expect("examples directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "hx"))
        .collect();
    paths.sort();
    paths
}
