# Cross-crate interface contracts (v1 — the parallel-development treaty)

Each crate implements exactly these types. Agents building crate X read this + lang-spec.md.
If a contract must change, the changer updates THIS file and posts a note in devlog.

## helix-syntax (lexer, parser, AST)

```rust
pub struct Span { pub start: u32, pub end: u32 }            // byte offsets
pub struct LexError { pub span: Span, pub msg: String }
pub enum TokKind { /* Ident, Int(i64), Float(f64), Kw(Kw), plus punctuation variants */ }
pub struct Token { pub kind: TokKind, pub span: Span }      // serde::Serialize+Deserialize

pub struct ParseError { pub span: Span, pub msg: String }

// AST — every node carries Span; whole tree serde Serialize+Deserialize (Observatory needs it).
pub struct Program { pub items: Vec<Item> }                 // Item::{Fn(FnDef), Const(ConstDef)}
pub struct FnDef { pub name: Ident, pub params: Vec<Param>, pub ret: Option<Type>,
                   pub body: Block, pub span: Span }
pub struct Param { pub name: Ident, pub ty: Type }
pub struct ConstDef { pub name: Ident, pub ty: Type, pub value: Literal, pub span: Span }
pub struct Ident { pub name: String, pub span: Span }
pub struct Block { pub stmts: Vec<Stmt>, pub span: Span }
pub enum Stmt {
    Let { name: Ident, ty: Option<Type>, init: Expr, span: Span },
    Assign { target: LValue, value: Expr, span: Span },
    If { cond: Expr, then_blk: Block, else_part: Option<Box<ElsePart>>, span: Span },
    For { iv: Ident, start: Expr, end: Box<Expr>, body: Block, span: Span },
    Return { value: Option<Expr>, span: Span },
    Expr(Expr), Empty,
    Block(Block),
}
pub enum ElsePart { If(Box<Stmt>), Block(Block) }           // Stmt::If for else-if chains
pub struct LValue { pub base: Ident, pub index: Option<Expr>, pub span: Span }
pub enum Expr { IntLit(i64,Span), FloatLit(f64,Span), Bool(bool,Span),
    Var(Ident), Unary(UnOp,Box<Expr>,Span), Bin(BinOp,Box<Expr>,Box<Expr>,Span),
    Index(Ident,Box<Expr>,Span), Call { callee: Ident, args: Vec<Expr>, span: Span },
    Cast(Box<Expr>,Type,Span) }
pub enum UnOp { Neg, Not }
pub enum BinOp { Add,Sub,Mul,Div,Rem, Lt,Gt,Le,Ge, Eq,Ne, And,Or }
```
APIs: `lex(src:&str)->Result<Vec<Token>,LexError>` · `parse(tokens)->Result<Program,ParseError>`
· `Program::print_tree(&self)->String` (indented tree for dumps/Observatory).

## helix-sema (types + checks)

```rust
pub enum Ty { I32,I64,F32,F64,Bool,Array(ElemTy),Unit }     // ElemTy scalar only
pub struct TypeError { pub span: Span, pub msg: String }     // reuse syntax::Span
pub struct TypedProgram { /* mirrors AST with Ty on every expr/lvalue; Serialize */ }
```
APIs: `check(program:&Program)->Result<TypedProgram,Vec<SemDiag>>` where `SemDiag{span,msg,kind}`.
Enforces EVERY static rule in lang-spec.md (incl. f(a,a), loop-var assign, definite assignment).

## helix-ir (CFG + SSA + passes)

```rust
pub struct FuncIr { pub name: String, pub blocks: Vec<BlockData>,
                    pub entry: BlockId, pub n_locals: usize,
                    pub next_value: u32, /* + typed side tables (types) */ }
pub struct BlockId(pub u32);                                 // dense 0..n
pub struct BlockData { pub phis: Vec<Phi>, pub insts: Vec<Inst>, pub term: Term }
pub struct Phi   { pub dst: ValueId, pub var: LocalId, pub args: Vec<(BlockId,ValueId)> } // arg per pred, aligned
pub struct ValueId(pub u32);  // SSA values; also covers constants via Inst::Const defs
pub struct LocalId(pub u32);  // source-level variables
pub enum Inst { Const{dst:ValueId,c:Constant}, Bin{op:BinOp,dst:ValueId,a:ValueId,b:ValueId},
    Unary{op:UnOp,dst:ValueId,a:ValueId}, Cast{dst:ValueId,val:ValueId,to:Ty},
    Load{dst:ValueId,arr:LocalId,idx:ValueId}, Store{arr:LocalId,idx:ValueId,val:ValueId},
    Call{dst:Option<ValueId>,callee:String,args:Vec<ValueId>} }   // builtins or user fns
pub enum Term { Jump(BlockId,Vec<ValueId>), Branch{cond:ValueId,t:BlockId,f:BlockId},
                Return(Option<ValueId>) }                  // Branch args implicit (no block params in HELIX IR)
pub enum Constant { I64(i64),I32(i32),F32(f32),F64(f64),Bool(bool) }
```
Design notes: HELIX IR is NOT SSA initially (locals may reassign; loads/stores to locals are
lowered as SSA-friendly straight-line code — sema guarantees single static assignment per path).
`to_ssa(&mut FuncIr)` runs CHK-dominators → semi-pruned φ placement → renaming. `from_ssa` not
needed (backend consumes φ directly). APIs:
`build(program:&TypedProgram)->Vec<FuncIr>` · `to_ssa(&mut FuncIr)` · `verify(&FuncIr)->Result<(),String>`
(dominance, φ arity/pred-match, reaching-def) · `print_ir(&FuncIr, ssa:bool)->String` (bb0-style text)
· passes module: `const_fold`, `const_prop`, `copy_prop`, `dce`, `cse`, `licm`, each `fn(&mut FuncIr)->ChangeFlag`.
Pass driver snapshots IR text between passes (Observatory OPT view).

## helix-analysis (loops + dependence)

```rust
pub struct LoopInfo { pub loops: Vec<LoopNest> }
pub struct Loop { pub id: usize, pub header: BlockId, pub blocks: Vec<BlockId>,
                  pub depth: u32, pub parent: Option<usize>,
                  pub iv: LocalId, pub bounds: (Bound,Bound) }   // Bound = Const(i64)|Sym(ValueId)
pub struct LoopReport { pub loop_id: usize, pub depth: u32,
    pub accesses: Vec<AccessLine>,                 // pretty lines for the Observatory card
    pub raw_deps: Vec<DepEdge>, pub war_deps: Vec<DepEdge>, pub waw_deps: Vec<DepEdge>,
    pub reduction: Option<Reduction>,              // recognized ⇒ exempt from its own RAW
    pub verdict: Verdict }
pub struct DepEdge { pub array: String, pub kind_label: String,   // "RAW a[i] <- a[i-1]"
    pub distance: Option<i64>, pub level: u32, pub direction: DirVec, pub explain: String }
pub enum Verdict { SafeParallel, ReductionParallel(ReductionOp), Sequential(String /*reason*/) }
pub enum ReductionOp { Add, Mul, Min, Max }                // '-' folds into Add(negate) at lowering
```
APIs: `find_loops(&FuncIr)->LoopInfo` · `analyze(func:&FuncIr, loops:&LoopInfo)->Vec<LoopReport>`.
Battery order per dim: ZIV→StrongSIV→WeakZeroSIV→WeakCrossingSIV→gcd+bounded-box(Diophantine); anything unproven → '*' (conservative).
ALL arithmetic i128. RAR never a dependence. Report strings polished (this is the demo's star).

## helix-backend (JIT)

```rust
pub struct JitEngine { /* owns JITModule; keep alive while fn ptrs used! */ }
impl JitEngine {
    pub fn compile(program:&[FuncIr], plan:&ParallelPlan, unchecked:bool)->Result<JitEngine,String>;
    pub fn run_main(&self)->Result<(),String>;             // calls JITed main
}
pub struct ParallelPlan { pub regions: Vec<RegionDesc> }    // one per approved/reduction loop
pub struct RegionDesc { pub func:usize, pub header:BlockId, pub kind:RegionKind,
                        pub reduction: Option<ReductionOp>, pub body_fn_name: String }
pub enum RegionKind { DoAll, Reduction(ReductionOp) }
```
Lowering: per HELIX fn one CLIF fn (`WindowsFastcall`, sig from Ty). Approved loop bodies become
`extern "C" fn(i64 iter, *const Ctx)` helper fns; main calls imported symbol
`helix_parallel_for(start:i64,end:i64,body_id:i64,nthreads:i64)`; host registry maps
body_id→ptr AFTER finalize (never embed unknown pointers). Bounds/div guards = compare+branch to
imported `helix_panic(msg_ptr:i64,line:i64)` which prints and exits. print/zeros/len/abs/sqrt/min/max =
imports. φ → block params (every pred terminator passes args; use BlockArg::Value).

## helix-runtime

```rust
pub extern "C" fn helix_parallel_for(start:i64,end:i64,body_id:i64,nthreads:i64);
// Stage A: thread::scope + static chunks. Stage B: pool + spin/park.
// Cost gate: serial if (end-start) < max(1024, GRAIN*nthreads). Env: HELIX_NTHREADS overrides hint.
// Registry: register_body(id, fn_ptr); thread-local ctx passing via *const Ctx captured at call.
// Reductions: per-thread accumulators in #[repr(align(128))] cells; combine after join.
pub fn set_stage(stage: RuntimeStage); pub enum RuntimeStage { ScopeThreads, Pool }
pub fn pool_stats()->PoolStats;                            // for the overhead microbench graph
```

## helix-engine (interpreter)

```rust
pub struct Interp { /* env, arrays as Rc<RefCell<Vec<Elem>>> so writes escape */ }
pub fn run(program:&TypedProgram)->Result<RunOutput,String>;
pub struct RunOutput { pub printed: Vec<String>, pub checksum: u64 }  // FNV over final array bits + prints
```
Semantics identical to spec incl. checked ops & saturating casts.

## helix-bench / helix-observe / helix-cli

- bench: `run_kernel_suite(config)->CampaignJson` (hyperfine-like schema, raw samples, meta JSON);
  kernels registered with (a) HELIX source, (b) Rust twin closure, (c) sizes, (d) expected verdict.
- observe: `artifact(program_src)->CompileArtifact` (serde JSON of every stage dump incl. layouts);
  axum server on 127.0.0.1; routes GET `/api/artifact?example=...`, POST `/api/run` {source};
  standalone HTML export route. Assets embedded via include_bytes! from web/.
- cli: subcommands run/check/dump/bench/observe/selftest as planned; `dump <stage> file.hx`.

## Shared conventions

- Errors: each crate returns Result<T, Vec<Diag>>-style with spans; CLI formats carets.
- All public data types serde Serialize+Deserialize (artifacts flow to web UI as JSON).
- No panics across FFI/JIT boundaries: host wraps JIT calls in catch_unwind.
- Naming: snake_case modules, CamelCase types, SCREAMING_SNAKE consts.
- Every crate: `#![forbid(unsafe_code)]` EXCEPT backend+runtime (document each unsafe block).

## Addendum (2026-08-24): helix-ir helpers required by helix-analysis

helix-analysis (already drafted) consumes these FuncIr/Inst conveniences beyond the base
contract. helix-ir must provide:

```rust
impl FuncIr {
    pub fn def_block(&self, v: ValueId) -> BlockId;          // block defining value
    pub fn const_of(&self, v: ValueId) -> Option<i64>;       // i64 view of Const defs
    pub fn inst_defining(&self, v: ValueId) -> Option<&Inst>;
    pub fn local_of_value(&self, v: ValueId) -> Option<LocalId>; // value that IS a local slot
    pub fn local_name(&self, l: LocalId) -> &str;            // from sema symbol arena
    pub fn loop_has_print(&self, lp: &LoopLike) -> bool;     // any print call inside blocks
}
impl Inst {
    pub fn local_reads(&self) -> Vec<LocalId>;
    pub fn local_write(&self) -> Option<LocalId>;
}
pub mod testutil { pub fn counting_loop() -> FuncIr; pub fn nested_loops() -> FuncIr; }
pub mod dom { pub struct Dominators { pub reachable: Vec<bool>, /* + idoms */ }
              impl Dominators { pub fn compute(f:&FuncIr)->Self; pub fn dominates(&self,a:BlockId,b:BlockId)->bool; } }
// Inst::StoreScalar{dst: LocalId, val: ValueId} — scalar variable store (pre-SSA model keeps
// locals as memory-like slots until SSA renames them).
```

## Addendum 2 (2026-08-24): ParallelPlan ownership + sema CallTarget change

1. `ParallelPlan` lives in **helix-analysis** (module `plan`):
```rust
pub struct ParallelPlan { pub regions: Vec<RegionDesc> }
pub struct RegionDesc {
    pub func_idx: usize,          // index into Vec<FuncIr>
    pub header: BlockId,
    pub kind: RegionKind,         // DoAll | Reduction(ReductionOp)
    pub body_fn_name: String,     // e.g. "main.loop0.body"
    pub start_val: Option<u32>, pub end_val: Option<u32>, // SSA ids of bounds (None=const in ctx)
}
pub fn build_plan(funcs:&[FuncIr], loops:&[LoopInfo], reports:&[Vec<LoopReport>]) -> ParallelPlan;
```
2. **sema change (breaking, done)**: `CallTarget::Builtin{which,args}` and
   `CallTarget::User{fn_idx,name,args}` now carry typed argument subtrees.
   - helix-ir build.rs must lower Inst::Call operands from these args (zeros(n) →
     Call{callee:"zeros", args:[n], arr_refs:[dst-slot]}).
   - helix-engine adapter should consume typed args directly instead of re-parsing source;
     keep run_with_source signature (LineMap still needs text).
