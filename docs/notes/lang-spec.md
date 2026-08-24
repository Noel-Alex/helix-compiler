# HELIX Language Specification (FROZEN v1.0 — 2026-08-23)

All crates implement THIS document. Changes require updating this file + devlog entry.

## Lexical grammar

```
comment      ::= "//" {!NEWLINE} | "/*" {!'*/'} '*/'        (non-nested)
INT_LIT      ::= DIGIT+                                      (fits i64; also used for i32 elems)
FLOAT_LIT    ::= DIGIT+ '.' DIGIT+ [EXP] | DIGIT+ EXP        EXP ::= ('e'|'E') ['+'|'-'] DIGIT+
IDENT        ::= (ALPHA|'_') (ALPHA|DIGIT|'_)*               (not a keyword)
keywords     ::= fn let const if else for return true false as in
reserved     ::= while break continue mut by and or not struct import
punct        ::= ( ) { } [ ] , ; : :: .. -> + - * / % < > <= >= == != && || ! =
```

`..` appears ONLY in for headers. `_` allowed as identifier char, not as discard.

## Phrase grammar (EBNF)

```
program    ::= { fn_def | const_def }
fn_def     ::= "fn" IDENT "(" [param {"," param}] ")" ["->" type] block
const_def  ::= "const" IDENT ":" type "=" literal ";"
param      ::= IDENT ":" type
type       ::= "i32" | "i64" | "f32" | "f64" | "bool" | "[" scalar_type "]" | "()"
scalar_type::= "i32" | "i64" | "f32" | "f64" | "bool"
block      ::= "{" {stmt} "}"
stmt       ::= let_stmt | assign_stmt | if_stmt | for_stmt | return_stmt | expr ";" | ";" | block
let_stmt   ::= "let" IDENT [":" type] "=" expr ";"
assign_stmt::= lvalue "=" expr ";"
lvalue     ::= IDENT ["[" expr "]"]
if_stmt    ::= "if" expr block {"else" (if_stmt | block)}
for_stmt   ::= "for" IDENT "in" expr ".." expr block          // half-open [start,end)
return_stmt::= "return" [expr] ";"
expr       ::= or_expr
or_expr    ::= and_expr {"||" and_expr}
and_expr   ::= eq_expr {"&&" eq_expr}
eq_expr    ::= rel_expr {("=="|"!=") rel_expr}
rel_expr   ::= add_expr {("<"|">"|"<="|">=") add_expr}
add_expr   ::= mul_expr {("+"|"-" ) mul_expr}
mul_expr   ::= unary {("*"|"/"|"%") unary}
unary      ::= ("-"|"!") unary | cast
cast       ::= postfix {"as" scalar_type}                     // binds tighter than unary -
postfix    ::= primary ["[" expr "]"] | primary "(" [expr {"," expr}] ")"
primary    ::= INT_LIT | FLOAT_LIT | "true" | "false" | IDENT | "(" expr ")"
```

Precedence (low→high): `||` < `&&` < `== !=` < `< > <= >=` < `+ -` < `* / %` < unary < `as` <
postfix < primary. All binary ops LEFT-associative. Assignment is a STATEMENT (no `a=b=c`,
no `if x=y`). Braces mandatory on if/for bodies (kills dangling-else). `else if` chains nest.
Unary minus binds tighter than binary `-`: `-a*b == (-a)*b`; `-x as i32 == -(x as i32)`.

## Types & coercions

- Arithmetic integer type = **i64** (unannotated INT_LIT infers i64). i32/f32 exist as ARRAY
  ELEMENT ("storage") types and via explicit annotations/casts; mixing i32 and i64 VALUES in an
  operator requires explicit `as`. Literals adapt: an INT_LIT directly annotating/assigning into
  i32 context is i32 (still compile-time range-checked); otherwise i64.
- ZERO implicit coercions. Single exception: array INDEX position accepts i32, implicitly widened
  to i64 (`a[i]` with `i: i32` is legal; `a[i] + 1` with `a:[i32]` still needs `(a[i]) as i64`).
- bool has NO truthiness; conditions must be bool; no int<->bool coercion.
- Arrays `[T]`, T scalar ONLY (no nesting). Fat pointer (ptr,len); assignment/passing is BY
  REFERENCE; callee writes ESCAPE. Arrays are never copied and not comparable with ==.
- Unit `()` for procedures; calling a unit fn inside an expression is a type error;
  `return;` only in unit fns, `return e;` only in value fns (all paths must return — enforced).

## Builtins (exactly 7)

| Signature | Notes |
|---|---|
| `print(e) -> ()` | any scalar; newline appended. In a loop ⇒ that loop is NEVER parallelized |
| `zeros(n: i64) -> [T]` | element type from annotation: `let a: [f64] = zeros(n);` |
| `len(a: [T]) -> i64` | |
| `abs(x) -> same` | numeric |
| `sqrt(x) -> same` | f32→f32, f64→f64; negative ⇒ NaN (IEEE) |
| `min(a,b) -> same` | numeric, IEEE minNum: non-NaN operand wins |
| `max(a,b) -> same` | numeric, IEEE maxNum |

## Operator semantics

- `/` `%` integers: TRAP (runtime error, source location printed) on divisor==0 and
  i64::MIN/-1. `%` = truncated remainder, SIGN OF DIVIDEND: `-7 % 2 == -1`, `7 % -2 == 1`.
- `/` floats: IEEE, div-by-zero ⇒ Inf/NaN, never traps.
- Integer overflow elsewhere: two's-complement WRAPPING (documented).
- `as`: numeric→numeric only. float→int = SATURATING (NaN→0, clamp to MIN/MAX) — identical in
  interpreter and JIT. int→int = truncating reinterpretation of two's complement. int↔float
  rounds toward zero. bool/array excluded from casts.
- `&& ||` SHORT-CIRCUIT (right side may not evaluate).

## Program structure & rules (statically enforced)

- Exactly one `fn main()` with zero params returning unit; missing/duplicate = error.
- Top level = fn defs + scalar consts only. No nested functions. Recursion ALLOWED.
- Definite assignment: every variable initialized before use on ALL paths (dataflow-checked).
  Arrays come fully initialized from zeros.
- Loop variable reassignment inside its loop = error (guarantees affine iteration space).
- Passing the SAME array variable twice in one call `f(a, a)` = error (protects dependence
  analysis from invisible aliasing).
- Duplicate parameter/local names in one scope = error; inner scopes shadow outer.

## Runtime errors (checked mode, default)

Bounds violation and div/rem-by-zero print `runtime error: <message> at line N` and exit(1).
`--unchecked` strips bounds checks (div checks stay); out-of-bounds under --unchecked is UB.

## Parallelization-relevant sentences (normative)

- `for` bodies must not contain `print` (side effect ⇒ never parallelized; compiler says why).
- Reduction recognition: `x = x OP t` or `x = t OP x`, OP ∈ {+,-,*,min,max}, x written exactly
  once per iteration, x referenced nowhere else in the body. FP `+/*` reductions: combination
  order unspecified (documented nondeterminism, OpenMP-style); integer/min/max exact.
