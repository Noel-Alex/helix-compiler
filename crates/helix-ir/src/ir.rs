//! Core CFG intermediate representation for HELIX.
//!
//! This module defines the data types fixed by the cross-crate contract
//! (`docs/notes/interface-contracts.md`, section *helix-ir*): a function is a
//! dense-indexed vector of basic blocks, each holding φ-nodes, effect-free
//! scalar instructions and exactly one terminator. Everything derives
//! `Serialize + Deserialize` because the Observatory ships whole IR snapshots
//! to the browser as JSON.
//!
//! ## Value model — why there are two id spaces
//!
//! [`ValueId`] and [`LocalId`] deliberately model different things:
//!
//! * A [`LocalId`] is a *source-level variable slot*. Indices `0..
//!   n_source_locals` mirror `helix_sema`'s `SymId` arena one-to-one (top-level
//!   consts first, then parameters, then `let` bindings in declaration order).
//!   Indices at and above [`FuncIr::n_source_locals`] are compiler temporaries
//!   (short-circuit results, the function return accumulator) that the builder
//!   appends after the arena.
//! * A [`ValueId`] is a *definition site*. Before SSA construction the IR is
//!   intentionally **not** in SSA form: every read or write of source variable
//!   `x` spells the same `ValueId(x_slot)` — the variable's *cell* — so a
//!   re-assignment redefines that id in place and a use may be reached by one
//!   of several defs. `crate::ssa::to_ssa` performs the classic Cytron-style
//!   renaming that turns each cell into a family of unique definitions joined
//!   by φ-nodes. Compiler temporaries are born single-assignment.
//!
//! ## Arrays stay out of SSA
//!
//! Following the LLVM/GCC precedent (see `docs/research/ssa-design.md`), only
//! scalars live in SSA values. Arrays are addressed through the [`LocalId`] of
//! their binding: [`Inst::Load`] / [`Inst::Store`] name the array directly, and
//! calls receive arrays through [`Inst::Call::arr_refs`] rather than value
//! operands — arrays are never copied, so they must never masquerade as
//! register values.

use serde::{Deserialize, Serialize};

use helix_sema::Ty;
use helix_syntax::ast::{BinOp, UnOp};

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Dense basic-block index. Blocks live in `FuncIr::blocks` at position
/// `self.0`; the set of live ids is always `0..blocks.len()` right after
/// [`FuncIr::compact`] (passes may transiently leave holes until they
/// renumber).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// Identifier of a single definition (an SSA *name* once `to_ssa` has run).
/// Ids below `FuncIr::n_source_locals` double as variable cells before SSA;
/// the builder allocates fresh temporaries from `n_source_locals` upwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Source-level variable slot (see the module docs for the id-space layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LocalId(pub u32);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// A compile-time scalar. Literal widths mirror the source types: unannotated
/// integer literals are `I64`, floats default to `F64`, and sema has already
/// range-checked every narrowed literal.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Constant {
    I64(i64),
    I32(i32),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl Constant {
    /// The [`Ty`] this constant inhabits.
    #[must_use]
    pub fn ty(&self) -> Ty {
        match self {
            Constant::I64(_) => Ty::I64,
            Constant::I32(_) => Ty::I32,
            Constant::F32(_) => Ty::F32,
            Constant::F64(_) => Ty::F64,
            Constant::Bool(_) => Ty::Bool,
        }
    }

    /// Numeric view used by constant folding. The payload carries its own
    /// width so the folder dispatches to native per-width ops — folding an
    /// `F64` constant through `f32` arithmetic silently rounds it to single
    /// precision. Returns `None` for bools.
    #[must_use]
    pub(crate) fn as_num(&self) -> Option<Num> {
        Some(match self {
            Constant::I64(v) => Num::I(*v),
            Constant::I32(v) => Num::I(*v as i64),
            Constant::F32(v) => Num::F32(*v),
            Constant::F64(v) => Num::F64(*v),
            Constant::Bool(_) => return None,
        })
    }
}

/// Width-tagged numeric payload shared by the folder.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Num {
    I(i64),
    F32(f32),
    F64(f64),
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

/// A φ-node: `dst` receives the value of the argument belonging to the edge
/// actually taken. `args` is aligned 1:1 with `BlockData::preds` of the owning
/// block (sorted ascending by `BlockId`), and each predecessor appears exactly
/// once. Entry-block φ-nodes with zero arguments represent *function
/// parameters* — they mirror Cranelift entry block params one-to-one.
///
/// `var` records the source variable the φ merges; arrays never get φ-nodes
/// (memory deliberately stays outside SSA).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Phi {
    /// Definition produced by this φ.
    pub dst: ValueId,
    /// Source variable being merged (used by renaming and pretty-printing).
    pub var: LocalId,
    /// One `(predecessor, value)` pair per predecessor, sorted by pred id.
    pub args: Vec<(BlockId, ValueId)>,
}

/// One scalar instruction. Every variant except [`Inst::Store`] and unit
/// [`Inst::Call`] defines `dst`. Operations are pure *except* `Store`, `Load`
/// (may trap on out-of-bounds indices) and `Call` (arbitrary effects) — this
/// triage drives DCE, CSE and LICM.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Inst {
    /// Materialise a constant. Literal operands of the source are folded into
    /// these defs at build time; `const_fold` grows the family.
    Const {
        /// Definition.
        dst: ValueId,
        /// The constant value.
        c: Constant,
    },
    /// Binary arithmetic / comparison / logical (non-short-circuit) op.
    Bin {
        /// Operator.
        op: BinOp,
        /// Definition.
        dst: ValueId,
        /// Left operand.
        a: ValueId,
        /// Right operand.
        b: ValueId,
    },
    /// Unary negation or logical not.
    Unary {
        /// Operator.
        op: UnOp,
        /// Definition.
        dst: ValueId,
        /// Operand.
        a: ValueId,
    },
    /// Numeric conversion with the frozen semantics: float→int saturates
    /// (NaN→0), int→int truncates, int↔float rounds toward zero.
    Cast {
        /// Definition.
        dst: ValueId,
        /// Value being converted.
        val: ValueId,
        /// Target type.
        to: Ty,
    },
    /// Read an array element. May trap when `idx` is out of bounds (checked
    /// mode), which is why loads are *not* treated as dead-removable or
    /// hoistable.
    Load(Load),
    /// Write an array element. A side effect: callee writes escape, so stores
    /// are always live roots for DCE.
    Store {
        /// Array being written.
        arr: LocalId,
        /// Element index.
        idx: ValueId,
        /// Value stored.
        val: ValueId,
    },
    /// A call to a builtin or a user function.
    Call(Call),
}

/// Payload of [`Inst::Call`] (boxed through the enum to keep it small).
///
/// Scalar arguments travel in `args`; array arguments travel in `arr_refs`.
/// Arrays are passed by reference and never copied, so they must not flow as
/// SSA values — keeping them in a separate list lets the backend lower them to
/// raw fat pointers and lets dependence analysis identify callee-visible
/// memory at a glance. If the callee *returns* an array (e.g. `zeros`), the
/// destination local is appended to `arr_refs` last and `dst` is `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Call {
    /// Definition, `None` for unit/array-returning callees.
    pub dst: Option<ValueId>,
    /// Builtin name or source name of the user function.
    pub callee: String,
    /// Scalar arguments in source order.
    pub args: Vec<ValueId>,
    /// Array (reference) arguments in source order, plus the output
    /// destination last when the callee returns an array.
    pub arr_refs: Vec<LocalId>,
}

/// Payload of [`Inst::Load`] (boxed through the enum to keep it small).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Load {
    /// Definition.
    pub dst: ValueId,
    /// Array being read (its binding's local slot).
    pub arr: LocalId,
    /// Element index (i64 after the spec's index widening).
    pub idx: ValueId,
}

/// Payload of [`Inst::Load`]-style struct variants kept out-of-line for size:
/// an element read `dst = a[idx]`.
impl Load {
    /// Convenience constructor.
    #[must_use]
    pub fn new(dst: ValueId, arr: LocalId, idx: ValueId) -> Self {
        Self { dst, arr, idx }
    }
}

impl Inst {
    /// The value defined by this instruction, if any.
    #[must_use]
    pub fn dst(&self) -> Option<ValueId> {
        match self {
            Inst::Const { dst, .. }
            | Inst::Bin { dst, .. }
            | Inst::Unary { dst, .. }
            | Inst::Cast { dst, .. } => Some(*dst),
            Inst::Load(l) => Some(l.dst),
            Inst::Call(c) => c.dst,
            Inst::Store { .. } => None,
        }
    }

    /// Operands read by this instruction, in a fixed order.
    #[must_use]
    pub fn uses(&self) -> Vec<ValueId> {
        match self {
            Inst::Const { .. } => Vec::new(),
            Inst::Bin { a, b, .. } => vec![*a, *b],
            Inst::Unary { a, .. } | Inst::Cast { val: a, .. } => vec![*a],
            Inst::Load(l) => vec![l.idx],
            Inst::Store { idx, val, .. } => vec![*idx, *val],
            Inst::Call(c) => c.args.clone(),
        }
    }

    /// Rewrite every operand through `map` (used by renaming and propagation).
    pub fn rewrite_uses(&mut self, map: &mut impl FnMut(ValueId) -> ValueId) {
        match self {
            Inst::Const { .. } => {}
            Inst::Bin { a, b, .. } => {
                *a = map(*a);
                *b = map(*b);
            }
            Inst::Unary { a, .. } | Inst::Cast { val: a, .. } => *a = map(*a),
            Inst::Load(l) => l.idx = map(l.idx),
            Inst::Store { idx, val, .. } => {
                *idx = map(*idx);
                *val = map(*val);
            }
            Inst::Call(c) => {
                for a in c.args.iter_mut() {
                    *a = map(*a);
                }
            }
        }
    }

    /// Is this instruction free of side effects and traps (given checked-mode
    /// semantics)? Exactly the candidates for CSE and LICM. Loads are excluded
    /// because an out-of-bounds load must still trap.
    #[must_use]
    pub fn is_pure(&self) -> bool {
        match self {
            Inst::Const { .. } | Inst::Bin { .. } | Inst::Unary { .. } | Inst::Cast { .. } => true,
            Inst::Load(_) | Inst::Store { .. } | Inst::Call(_) => false,
        }
    }

    /// May evaluating this instruction trap at runtime? Integer div/rem trap
    /// on a zero divisor (and `MIN / -1`); loads may trap on bounds checks;
    /// calls are treated conservatively. Such instructions must survive DCE
    /// even when unused and are never speculated out of loops. Casts never
    /// trap: float→int saturates, int→int truncates (`lower.rs`).
    #[must_use]
    pub fn may_trap(&self) -> bool {
        matches!(
            self,
            Inst::Bin {
                op: BinOp::Div | BinOp::Rem,
                ..
            }
        ) || matches!(self, Inst::Load(_) | Inst::Call(_))
    }
}

impl Call {
    /// Convenience constructor for a call that defines `dst`.
    #[must_use]
    pub fn new(
        dst: Option<ValueId>,
        callee: &str,
        args: Vec<ValueId>,
        arr_refs: Vec<LocalId>,
    ) -> Self {
        Self {
            dst,
            callee: callee.to_string(),
            args,
            arr_refs,
        }
    }
}

// ---------------------------------------------------------------------------
// Terminators
// ---------------------------------------------------------------------------

/// The single control transfer ending every block.
///
/// `Jump` carries one value per φ of its *target*, positionally aligned with
/// `target.phis` (Cranelift block-parameter style). `Branch` carries no
/// argument lists by contract — edge values for branch targets live in the
/// target's φ `args`, keyed by predecessor. `Return` closes the function.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Term {
    /// Unconditional jump; `args[i]` feeds `target.phis[i]`.
    Jump(BlockId, Vec<ValueId>),
    /// Conditional branch on a bool value.
    Branch {
        /// Condition (must be bool-typed).
        cond: ValueId,
        /// Successor when true.
        t: BlockId,
        /// Successor when false.
        f: BlockId,
    },
    /// Leave the function, yielding `Some(v)` for value-returning functions.
    Return(Option<ValueId>),
}

impl Term {
    /// Successor blocks, in a stable order (t before f for branches).
    #[must_use]
    pub fn succs(&self) -> Vec<BlockId> {
        match self {
            Term::Jump(t, _) => vec![*t],
            Term::Branch { t, f, .. } => vec![*t, *f],
            Term::Return(_) => Vec::new(),
        }
    }

    /// Values this terminator forwards to successor φ-nodes (empty except for
    /// jumps).
    #[must_use]
    pub fn forwarded_args(&self) -> &[ValueId] {
        match self {
            Term::Jump(_, args) => args,
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// Blocks and functions
// ---------------------------------------------------------------------------

/// One basic block: φ-nodes, instructions, a terminator, and the structural
/// edge lists. `preds` is kept sorted ascending and mirrors `succs` of the
/// predecessors symmetrically; φ argument lists are aligned with `preds`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockData {
    /// φ-nodes, listed before any instruction (they conceptually execute on
    /// every edge into the block simultaneously).
    pub phis: Vec<Phi>,
    /// Straight-line scalar instructions.
    pub insts: Vec<Inst>,
    /// The terminating control transfer (always present by construction).
    pub term: Term,
    /// Predecessors, sorted ascending.
    pub preds: Vec<BlockId>,
    /// Successors, deduplicated in terminator order.
    pub succs: Vec<BlockId>,
}

impl BlockData {
    /// An empty block with a placeholder `Return(None)` terminator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            phis: Vec::new(),
            insts: Vec::new(),
            term: Term::Return(None),
            preds: Vec::new(),
            succs: Vec::new(),
        }
    }

    /// Operand uses of every instruction plus every φ argument.
    #[must_use]
    pub fn uses(&self) -> Vec<ValueId> {
        let mut out = Vec::new();
        for p in &self.phis {
            out.extend(p.args.iter().map(|(_, v)| *v));
        }
        for i in &self.insts {
            out.extend(i.uses());
        }
        out.extend(self.term.forwarded_args().iter().copied());
        if let Term::Branch { cond, .. } = &self.term {
            out.push(*cond);
        }
        if let Term::Return(Some(v)) = &self.term {
            out.push(*v);
        }
        out
    }
}

impl Default for BlockData {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-function auxiliary information that passes and the verifier need but
/// that is not part of the graph itself. Built alongside the IR; passes must
/// preserve it (none of them mint values of a different type than the ones
/// they delete).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SideTable {
    /// Type of every `ValueId`, indexed by `ValueId::0`.
    pub val_tys: Vec<Ty>,
    /// Type of every `LocalId`, indexed by `LocalId::0`.
    pub local_tys: Vec<Ty>,
    /// Source names of locals (pretty-printing and Observatory cards).
    pub local_names: Vec<String>,
    /// Function return type (`Unit` for procedures).
    pub ret: Ty,
}

impl Default for SideTable {
    fn default() -> Self {
        Self {
            val_tys: Vec::new(),
            local_tys: Vec::new(),
            local_names: Vec::new(),
            ret: helix_sema::Ty::Unit,
        }
    }
}

impl SideTable {
    /// Type of a value, if known.
    #[must_use]
    pub fn val_ty(&self, v: ValueId) -> Option<Ty> {
        self.val_tys.get(v.0 as usize).copied()
    }

    /// Type of a local slot, if known.
    #[must_use]
    pub fn local_ty(&self, l: LocalId) -> Option<Ty> {
        self.local_tys.get(l.0 as usize).copied()
    }

    /// Source name of a local slot, if known.
    #[must_use]
    pub fn local_name(&self, l: LocalId) -> Option<&str> {
        self.local_names.get(l.0 as usize).map(String::as_str)
    }

    /// Element type of an array local, if it is indeed an array.
    #[must_use]
    pub fn elem_ty(&self, l: LocalId) -> Option<helix_sema::ElemTy> {
        self.local_ty(l)?.elem()
    }
}

/// A whole function in CFG form.
///
/// Field-note versus the contract sketch: `blocks` is a plain `Vec<BlockData>`
/// indexed densely by `BlockId` (the workspace forbids adding an `IndexMap`
/// dependency, and dense indexing is all the contract needs). `types` and
/// `next_value` carry the side table and the fresh-value cursor so the struct
/// stays self-contained for the pass driver.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FuncIr {
    /// Source name of the function.
    pub name: String,
    /// Basic blocks; live ids are `0..blocks.len()` after compaction.
    pub blocks: Vec<BlockData>,
    /// Entry block (always `BlockId(0)` as built).
    pub entry: BlockId,
    /// Total number of local slots: source arena plus compiler temporaries.
    pub n_locals: usize,
    /// Number of leading slots that mirror the sema `SymId` arena; ids at and
    /// above this are compiler temporaries.
    pub n_source_locals: usize,
    /// Types/names side table (see [`SideTable`]).
    pub types: SideTable,
    /// Cursor for allocating fresh [`ValueId`]s; always greater than every id
    /// in use.
    pub next_value: u32,
}

impl FuncIr {
    /// Start a fresh function with a single (so far empty) entry block.
    #[must_use]
    pub fn new(name: &str, ret: Ty, n_source_locals: usize) -> Self {
        let mut ir = Self {
            name: name.to_string(),
            blocks: Vec::new(),
            entry: BlockId(0),
            n_locals: n_source_locals,
            n_source_locals,
            types: SideTable {
                val_tys: Vec::new(),
                local_tys: Vec::new(),
                local_names: Vec::new(),
                ret,
            },
            next_value: n_source_locals as u32,
        };
        let e = ir.new_block();
        debug_assert_eq!(e, ir.entry);
        ir
    }

    // -- construction -------------------------------------------------------

    /// Append a fresh block; the caller must install a real terminator before
    /// verification.
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BlockData::new());
        id
    }

    /// Allocate a fresh SSA value id and (optionally) record its type.
    pub fn new_value(&mut self, ty: Ty) -> ValueId {
        let v = ValueId(self.next_value);
        self.next_value = self.next_value.checked_add(1).expect("value id overflow");
        if self.types.val_tys.len() <= v.0 as usize {
            self.types.val_tys.resize(v.0 as usize + 1, ty);
        }
        self.types.val_tys[v.0 as usize] = ty;
        v
    }

    /// Reserve `k` additional compiler-temporary local slots named `$tmp`.
    pub fn add_temp_locals(&mut self, k: usize, ty: Ty) -> LocalId {
        let first = LocalId(self.n_locals as u32);
        for _ in 0..k {
            self.types.local_tys.push(ty);
            self.types
                .local_names
                .push(format!("$tmp{}", self.n_locals));
            self.n_locals += 1;
        }
        first
    }

    /// Register a source local slot's type and name (builder-time).
    pub fn declare_local(&mut self, l: LocalId, ty: Ty, name: &str) {
        let i = l.0 as usize;
        if self.types.local_tys.len() <= i {
            self.types.local_tys.resize(i + 1, ty);
            self.types.local_names.resize(i + 1, String::new());
        }
        self.types.local_tys[i] = ty;
        self.types.local_names[i] = name.to_string();
        self.n_locals = self.n_locals.max(i + 1);
    }

    // -- access -------------------------------------------------------------

    /// Immutable view of a block.
    #[must_use]
    pub fn block(&self, b: BlockId) -> &BlockData {
        &self.blocks[b.0 as usize]
    }

    /// Mutable view of a block.
    pub fn block_mut(&mut self, b: BlockId) -> &mut BlockData {
        &mut self.blocks[b.0 as usize]
    }

    /// Successors of a block (maintained structurally).
    #[must_use]
    pub fn succs(&self, b: BlockId) -> &[BlockId] {
        &self.block(b).succs
    }

    /// Predecessors of a block, sorted ascending (maintained structurally).
    #[must_use]
    pub fn preds(&self, b: BlockId) -> &[BlockId] {
        &self.block(b).preds
    }

    /// Terminator of a block.
    #[must_use]
    pub fn term(&self, b: BlockId) -> &Term {
        &self.block(b).term
    }

    /// Mutable terminator of a block; call [`FuncIr::set_term`] instead when
    /// the *target* changes so predecessor lists stay symmetric.
    pub fn term_mut(&mut self, b: BlockId) -> &mut Term {
        &mut self.blocks[b.0 as usize].term
    }

    /// Type of a value from the side table.
    #[must_use]
    pub fn val_ty(&self, v: ValueId) -> Option<Ty> {
        self.types.val_ty(v)
    }

    /// True when `v` denotes a source-variable cell (pre-SSA spelling).
    #[must_use]
    pub fn is_slot_value(&self, v: ValueId) -> bool {
        (v.0 as usize) < self.n_source_locals
    }

    // -- mutation -----------------------------------------------------------

    /// Install a terminator and repair the affected edge lists atomically:
    /// the block's `succs` are rebuilt and predecessor lists of the old and
    /// new targets are updated. This is the only sanctioned way to change
    /// control flow — one-sided edge edits are the classic corruption bug.
    pub fn set_term(&mut self, b: BlockId, term: Term) {
        let old_succs = self.block(b).term.succs();
        let new_succs = term.succs();
        for s in old_succs {
            self.blocks[s.0 as usize].preds.retain(|p| *p != b);
        }
        self.blocks[b.0 as usize].term = term;
        for s in new_succs {
            let preds = &mut self.blocks[s.0 as usize].preds;
            if !preds.contains(&b) {
                preds.push(b);
                preds.sort_unstable();
            }
        }
        let mut succs = self.block(b).term.succs();
        succs.sort_unstable();
        succs.dedup();
        self.blocks[b.0 as usize].succs = succs;
    }

    /// Recompute every `preds`/`succs` pair from the terminators. Passes that
    /// edit many terminators may call this once afterwards instead of using
    /// [`FuncIr::set_term`] per edit.
    pub fn recompute_edges(&mut self) {
        for b in &mut self.blocks {
            b.preds.clear();
            b.succs.clear();
        }
        for i in 0..self.blocks.len() {
            let id = BlockId(i as u32);
            let succs = self.blocks[i].term.succs();
            for s in succs {
                self.blocks[s.0 as usize].preds.push(id);
                if !self.blocks[i].succs.contains(&s) {
                    self.blocks[i].succs.push(s);
                }
            }
            self.blocks[i].succs.sort_unstable();
            self.blocks[i].preds.sort_unstable();
        }
    }

    /// Sort every φ argument list to match the (ascending) predecessor order,
    /// dropping duplicates. Call after any edge surgery.
    pub fn normalize_phis(&mut self) {
        for b in &mut self.blocks {
            let preds = b.preds.clone();
            for p in &mut b.phis {
                p.args.retain(|(from, _)| preds.contains(from));
                p.args.sort_unstable_by_key(|(from, _)| *from);
                p.args.dedup_by_key(|(from, _)| *from);
            }
        }
    }

    /// Replace every occurrence of `from` (instruction operands, φ arguments,
    /// terminator arguments and conditions) with `to`.
    pub fn replace_all_uses(&mut self, from: ValueId, to: ValueId) {
        for b in &mut self.blocks {
            for p in &mut b.phis {
                for a in &mut p.args {
                    if a.1 == from {
                        a.1 = to;
                    }
                }
            }
            for i in &mut b.insts {
                i.rewrite_uses(&mut |v| if v == from { to } else { v });
            }
            match &mut b.term {
                Term::Jump(_, args) => {
                    for a in args.iter_mut() {
                        if *a == from {
                            *a = to;
                        }
                    }
                }
                Term::Branch { cond, .. } => {
                    if *cond == from {
                        *cond = to;
                    }
                }
                Term::Return(v) => {
                    if *v == Some(from) {
                        *v = Some(to);
                    }
                }
            }
        }
    }

    /// Drop tombstoned/unreachable blocks and renumber the rest densely,
    /// rewriting every `BlockId` reference (terminators and φ arguments).
    /// `keep` is consulted with the *old* ids; the surviving order preserves
    /// ascending id order so entry stays first.
    ///
    /// Returns the mapping old → new for blocks that survived.
    pub fn compact(&mut self, keep: &[bool]) -> Vec<Option<BlockId>> {
        let n = self.blocks.len();
        debug_assert_eq!(keep.len(), n);
        let mut map: Vec<Option<BlockId>> = Vec::with_capacity(n);
        let mut next = 0u32;
        for alive in keep {
            if *alive {
                map.push(Some(BlockId(next)));
                next += 1;
            } else {
                map.push(None);
            }
        }
        let remap = |x: BlockId| -> BlockId {
            match map.get(x.0 as usize).copied().flatten() {
                Some(m) => m,
                None => panic!("compact: terminator references dropped block bb{}", x.0),
            }
        };
        let mut old = std::mem::take(&mut self.blocks);
        for (i, mut b) in old.drain(..).enumerate() {
            if !keep[i] {
                continue;
            }
            match &mut b.term {
                Term::Jump(t, _) => *t = remap(*t),
                Term::Branch { t, f, .. } => {
                    *t = remap(*t);
                    *f = remap(*f);
                }
                Term::Return(_) => {}
            }
            // Phi arguments from deleted predecessors are stale edges; drop
            // them here and let normalize_phis/recompute_edges realign.
            // Surviving arguments must be RENUMBERED like every other block
            // reference — the terminator remap above shifts all ids, so a φ
            // arg left un-remapped names a *different* block afterwards and
            // the 1:1 phi-args/preds alignment breaks silently.
            let preds_before: Vec<BlockId> = b.preds.clone();
            for p in &mut b.phis {
                p.args.retain(|(from, _)| {
                    let fi = from.0 as usize;
                    keep[fi] && preds_before.contains(from)
                });
                for (from, _) in p.args.iter_mut() {
                    *from = remap(*from);
                }
                p.args.sort_unstable_by_key(|(f, _)| f.0);
            }
            self.blocks.push(b);
        }
        self.recompute_edges();
        map
    }

    /// Block containing the (first) definition of `v`, or the entry block
    /// when `v` is a cell spelling with no explicit def site. Dependence
    /// analysis uses this for loop-invariance queries: a value whose def
    /// block sits outside the loop body is invariant there.
    #[must_use]
    pub fn def_block(&self, v: ValueId) -> BlockId {
        for (bi, b) in self.blocks.iter().enumerate() {
            if b.phis.iter().any(|p| p.dst == v) || b.insts.iter().any(|i| i.dst() == Some(v)) {
                return BlockId(bi as u32);
            }
        }
        self.entry
    }

    /// Constant payload of `v` if its unique definition is an `Inst::Const`
    /// carrying an integer; `None` otherwise (non-const defs, floats, bools).
    #[must_use]
    pub fn const_of(&self, v: ValueId) -> Option<i64> {
        for b in &self.blocks {
            for inst in &b.insts {
                if let Inst::Const { dst, c } = inst
                    && *dst == v
                    && let Constant::I64(x) = c
                {
                    return Some(*x);
                }
            }
        }
        None
    }

    /// The instruction defining `v`, if it is defined by an instruction
    /// (rather than a φ). First match wins; unique post-SSA.
    #[must_use]
    pub fn inst_defining(&self, v: ValueId) -> Option<&Inst> {
        for b in &self.blocks {
            if b.phis.iter().any(|p| p.dst == v) {
                return None; // phi-defined: not a plain instruction
            }
            if let Some(i) = b.insts.iter().find(|i| i.dst() == Some(v)) {
                return Some(i);
            }
        }
        None
    }

    /// Highest `ValueId` mentioned anywhere (defs and uses), for allocating a
    /// collision-free renaming base.
    #[must_use]
    pub fn max_value_id(&self) -> u32 {
        let mut mx = self.next_value.saturating_sub(1);
        for b in &self.blocks {
            for p in &b.phis {
                mx = mx.max(p.dst.0);
                for (_, v) in &p.args {
                    mx = mx.max(v.0);
                }
            }
            for i in &b.insts {
                if let Some(d) = i.dst() {
                    mx = mx.max(d.0);
                }
                for u in i.uses() {
                    mx = mx.max(u.0);
                }
            }
            match &b.term {
                Term::Jump(_, args) => {
                    for a in args {
                        mx = mx.max(a.0);
                    }
                }
                Term::Branch { cond, .. } => mx = mx.max(cond.0),
                Term::Return(Some(v)) => mx = mx.max(v.0),
                Term::Return(None) => {}
            }
        }
        mx
    }
}
